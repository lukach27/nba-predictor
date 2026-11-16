"""Training pipeline: Elo tuning, feature selection, hyperparameter search, calibration,
evaluation and artefact export.

Protocol (what protects the test set)
-------------------------------------
1. **Development seasons** = everything before ``config.TEST_SEASONS``. Every decision is made
   here with *walk-forward* validation (validate on season ``s``, train on seasons ``< s``):
   Elo parameters, which feature groups to keep (ablation), hyperparameters, the ensemble
   weight and the Platt calibrators (fitted on out-of-fold scores).
2. **Test seasons** are scored exactly once, by a model frozen after step 1. Baselines (home
   rate, pure Elo, the original project's Elo) are scored on the same games, and every
   comparison comes with a paired-bootstrap confidence interval.
3. Only then is the **production model** refitted on all seasons with the frozen settings; it
   is the one saved for live predictions.

Run with ``python -m nba_predictor train``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor

from . import config
from .config import FeatureConfig, TrainingConfig
from .data_collection import load_clean_games, load_schedule, upcoming_fixtures
from .elo import EloParams, home_win_probability, run_elo, tune_elo
from .explain import global_importance
from .features import FeatureSet, available_feature_columns, build_features, save_processed
from .metrics import (bootstrap_ci, calibration_table, evaluate_probabilities, log_loss_per_game, paired_bootstrap,
                      sigmoid)
from .modeling import (BaseModels, ModelBundle, PlattCalibrator, best_ensemble_weight, blend, fit_base_models,
                       fit_calibrators, out_of_fold_logits, runtime_versions)

logger = logging.getLogger(__name__)

#: The original project's Elo, kept as a baseline: K=20, no home court, no margin, no regression.
ORIGINAL_ELO = EloParams(k=20.0, home_advantage=0.0, season_carry=1.0, use_mov=False)


@dataclass
class TrainingResult:
    """Everything produced by :func:`train`."""

    bundle: ModelBundle
    metrics: dict
    backtest: pd.DataFrame
    features: FeatureSet


# ------------------------------------------------------------------------ helpers
def _jsonable(obj):
    """Recursively convert numpy / pandas scalars so ``json.dump`` accepts the structure."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def _flatten(groups: dict[str, list[str]]) -> list[str]:
    return config.all_feature_columns(groups)


def _mean_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    """Bootstrap interval for mean(a - b) over paired per-game losses."""
    delta = a - b
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), size=(n_boot, len(delta)))].mean(axis=1)
    low, high = float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
    return {"delta_log_loss": float(delta.mean()), "low": low, "high": high, "significant": bool(low > 0 or high < 0)}


def _oof_losses(data: pd.DataFrame, columns: list[str], validation_seasons: list[int], lr_c: float,
                xgb_params: dict) -> np.ndarray:
    """Per-game log loss of the equal-weight LR+XGBoost blend on walk-forward validation rows."""
    y = data["home_win"].to_numpy(dtype=float)
    oof = out_of_fold_logits(data[columns].astype(float), y, data["season"].to_numpy(), validation_seasons, lr_c, xgb_params)
    scored = ~np.isnan(oof["lr"])
    return log_loss_per_game(y[scored], sigmoid(0.5 * (oof["lr"][scored] + oof["xgb"][scored])))


# -------------------------------------------------------------- feature selection
def select_feature_groups(data: pd.DataFrame, groups: dict[str, list[str]], validation_seasons: list[int],
                          tcfg: TrainingConfig, lr_c: float = 0.1) -> dict:
    """Ablation study and greedy backward elimination of feature groups.

    Uses walk-forward log loss on development seasons only. A group is dropped when removing
    it does not make the forecast worse (``delta <= tcfg.ablation_tolerance``); the loop stops
    when every remaining group earns its place. Also reports an "add one group to Elo" table
    so each group's stand-alone value is visible.
    """
    params = tcfg.xgb_default_params

    def evaluate(sets: dict[str, dict[str, list[str]]]) -> dict[str, np.ndarray]:
        names = list(sets)
        losses = Parallel(n_jobs=tcfg.n_jobs)(
            delayed(_oof_losses)(data, _flatten(sets[name]), validation_seasons, lr_c, params) for name in names
        )
        return dict(zip(names, losses))

    full = evaluate({"full": groups})["full"]
    report: dict = {"full_log_loss": float(full.mean()), "steps": [], "add_one_to_elo": [], "drop_one_final": []}

    if "elo" in groups and len(groups) > 1:
        sets = {"elo": {"elo": groups["elo"]}}
        sets.update({name: {"elo": groups["elo"], name: cols} for name, cols in groups.items() if name != "elo"})
        scores = evaluate(sets)
        for name in sets:
            if name == "elo":
                continue
            row = _mean_diff_ci(scores[name], scores["elo"], tcfg.n_bootstrap_ci, config.RANDOM_SEED)
            report["add_one_to_elo"].append({"group": name, "log_loss": float(scores[name].mean()), **row})
        report["elo_only_log_loss"] = float(scores["elo"].mean())

    current, current_losses = dict(groups), full
    while len(current) > 1:
        candidates = {name: {k: v for k, v in current.items() if k != name} for name in current}
        scores = evaluate(candidates)
        deltas = {name: float(scores[name].mean() - current_losses.mean()) for name in candidates}
        weakest = min(deltas, key=deltas.get)
        step = {"evaluated": deltas, "log_loss_before": float(current_losses.mean())}
        if deltas[weakest] <= tcfg.ablation_tolerance:
            step["dropped"] = weakest
            report["steps"].append(step)
            logger.info("ablation: dropping '%s' (removing it changes log loss by %+.5f)", weakest, deltas[weakest])
            current, current_losses = candidates[weakest], scores[weakest]
        else:
            step["dropped"] = None
            report["steps"].append(step)
            for name in candidates:
                row = _mean_diff_ci(scores[name], current_losses, tcfg.n_bootstrap_ci, config.RANDOM_SEED)
                report["drop_one_final"].append({"group": name, **row})
            break

    report["selected_groups"] = list(current)
    report["dropped_groups"] = [g for g in groups if g not in current]
    report["selected_log_loss"] = float(current_losses.mean())
    return report


# ---------------------------------------------------------- hyperparameter search
def _sample_xgb_params(rng: np.random.Generator) -> dict:
    """One random XGBoost configuration (shallow, regularised trees: the signal is weak)."""
    return {
        "n_estimators": int(rng.choice([150, 250, 400, 600, 800])),
        "max_depth": int(rng.choice([2, 3, 4])),
        "learning_rate": float(rng.choice([0.01, 0.02, 0.03, 0.05, 0.08])),
        "subsample": float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
        "colsample_bytree": float(rng.choice([0.5, 0.65, 0.8, 1.0])),
        "min_child_weight": float(rng.choice([1, 5, 10, 20, 40])),
        "reg_lambda": float(rng.choice([1, 2, 5, 10, 20])),
        "gamma": float(rng.choice([0, 0.5, 1, 2])),
        "monotone": bool(rng.choice([True, False])),
    }


def _single_model_loss(x: pd.DataFrame, y: np.ndarray, seasons: np.ndarray, validation_seasons: list[int],
                       model: str, lr_c: float, xgb_params: dict) -> float:
    oof = out_of_fold_logits(x, y, seasons, validation_seasons, lr_c, xgb_params, models=(model,))[model]
    scored = ~np.isnan(oof)
    return float(log_loss_per_game(y[scored], sigmoid(oof[scored])).mean())


def tune_hyperparameters(x: pd.DataFrame, y: np.ndarray, seasons: np.ndarray, validation_seasons: list[int],
                         tcfg: TrainingConfig) -> dict:
    """Pick the LR regularisation strength and XGBoost parameters by walk-forward log loss."""
    lr_losses = Parallel(n_jobs=tcfg.n_jobs)(
        delayed(_single_model_loss)(x, y, seasons, validation_seasons, "lr", c, {}) for c in tcfg.lr_c_grid
    )
    lr_table = sorted(zip(tcfg.lr_c_grid, lr_losses), key=lambda t: t[1])

    rng = np.random.default_rng(config.RANDOM_SEED)
    candidates = [dict(tcfg.xgb_default_params, monotone=False)]
    candidates += [_sample_xgb_params(rng) for _ in range(max(0, tcfg.xgb_search_iterations - 1))]
    xgb_losses = Parallel(n_jobs=tcfg.n_jobs)(
        delayed(_single_model_loss)(x, y, seasons, validation_seasons, "xgb", 1.0, params) for params in candidates
    )
    ranked = sorted(zip(candidates, xgb_losses), key=lambda t: t[1])
    logger.info("tuning: best LR C=%s (%.5f); best XGB %.5f %s", lr_table[0][0], lr_table[0][1], ranked[0][1], ranked[0][0])
    return {
        "lr_c": float(lr_table[0][0]),
        "xgb_params": ranked[0][0],
        "lr_search": [{"C": float(c), "log_loss": float(l)} for c, l in lr_table],
        "xgb_search": [{**p, "log_loss": float(l)} for p, l in ranked[:10]],
        "xgb_default_log_loss": float(xgb_losses[0]),
    }


# -------------------------------------------------------------------- calibration
def walk_forward_calibrated(z_lr: np.ndarray, z_xgb: np.ndarray, y: np.ndarray, seasons: np.ndarray,
                            validation_seasons: list[int], min_rows: int = 500) -> np.ndarray:
    """Honest backtest probabilities: season ``s`` is blended and calibrated with a weight and
    Platt parameters fitted only on out-of-fold scores from seasons before ``s``."""
    out = np.full(len(y), np.nan)
    has_score = ~np.isnan(z_lr) & ~np.isnan(z_xgb)
    for season in sorted(validation_seasons):
        now = has_score & (seasons == season)
        before = has_score & (seasons < season)
        if not now.any():
            continue
        if before.sum() >= min_rows:
            weight = best_ensemble_weight(z_lr[before], z_xgb[before], y[before])
            calibrator = PlattCalibrator.fit(blend(z_lr[before], z_xgb[before], weight), y[before])
        else:
            weight, calibrator = 0.5, PlattCalibrator()
        out[now] = calibrator.predict(blend(z_lr[now], z_xgb[now], weight))
    return out


def _fit_bundle(data: pd.DataFrame, columns: list[str], validation_seasons: list[int], lr_c: float, xgb_params: dict,
                elo_params: EloParams, fcfg: FeatureConfig, tcfg: TrainingConfig, replicas: int = 0) -> tuple[ModelBundle, dict]:
    """Fit base models on ``data`` and calibrate them on walk-forward out-of-fold scores."""
    x = data[columns].astype(float)
    y = data["home_win"].to_numpy(dtype=float)
    seasons = data["season"].to_numpy()
    oof = out_of_fold_logits(x, y, seasons, validation_seasons, lr_c, xgb_params, n_jobs=tcfg.n_jobs)
    scored = ~np.isnan(oof["lr"])
    weight = best_ensemble_weight(oof["lr"][scored], oof["xgb"][scored], y[scored])
    calibrators = fit_calibrators(oof["lr"][scored], oof["xgb"][scored], y[scored], weight)
    models = fit_base_models(x, y, lr_c, xgb_params, n_jobs=tcfg.n_jobs)

    bagged: list[BaseModels] = []
    rng = np.random.default_rng(config.RANDOM_SEED)
    for i in range(replicas):  # bootstrap replicas -> spread of the forecast = model uncertainty
        idx = rng.integers(0, len(x), size=len(x))
        bagged.append(fit_base_models(x.iloc[idx], y[idx], lr_c, xgb_params, seed=config.RANDOM_SEED + 1 + i, n_jobs=tcfg.n_jobs))

    bundle = ModelBundle(feature_columns=columns, models=models, ensemble_weight=weight, calibrators=calibrators,
                         elo_params=elo_params, feature_config=fcfg, replicas=bagged)
    return bundle, {"oof": oof, "scored": scored}


# --------------------------------------------------------------------- evaluation
def _baseline_probabilities(matchups: pd.DataFrame, dev: pd.Series, target: pd.Series, elo_params: EloParams) -> dict[str, np.ndarray]:
    """Baseline forecasts for the ``target`` rows, fitted (where needed) on ``dev`` rows only."""
    y_dev = matchups.loc[dev, "home_win"].to_numpy(dtype=float)
    out = {"home_rate": np.full(int(target.sum()), y_dev.mean())}
    out["elo"] = home_win_probability(matchups.loc[target, "home_elo"], matchups.loc[target, "away_elo"],
                                      elo_params.home_advantage)
    original = run_elo(matchups, ORIGINAL_ELO)
    diff = (original.home_pre - original.away_pre).reshape(-1, 1)
    fitted = LogisticRegression(C=1e6, max_iter=1000).fit(diff[dev.to_numpy()], y_dev)
    out["original_elo"] = fitted.predict_proba(diff[target.to_numpy()])[:, 1]
    return out


def _score_block(y: np.ndarray, forecasts: dict[str, np.ndarray], reference: str, tcfg: TrainingConfig) -> dict:
    """Metrics, bootstrap intervals, calibration tables and paired comparisons for a set of forecasts."""
    block: dict = {"models": {}, "calibration": {}, "vs_reference": {}, "reference": reference}
    for name, p in forecasts.items():
        block["models"][name] = {**evaluate_probabilities(y, p, tcfg.calibration_bins),
                                 "ci": bootstrap_ci(y, p, tcfg.n_bootstrap_ci, config.RANDOM_SEED)}
        block["calibration"][name] = calibration_table(y, p, tcfg.calibration_bins)
        if name != reference:
            block["vs_reference"][name] = paired_bootstrap(y, p, forecasts[reference], tcfg.n_bootstrap_ci, config.RANDOM_SEED)
    return block


def _per_season(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    rows = []
    for season, part in frame.groupby("season"):
        y = part["home_win"].to_numpy(dtype=float)
        if len(np.unique(y)) < 2:
            continue
        for col in columns:
            m = evaluate_probabilities(y, part[col].to_numpy())
            rows.append({"season": int(season), "model": col.removeprefix("p_"), "n": m["n"], "accuracy": m["accuracy"],
                         "log_loss": m["log_loss"], "brier": m["brier"], "roc_auc": m["roc_auc"], "ece": m["ece"]})
    return rows


def _margin_model(dev: pd.DataFrame, test: pd.DataFrame, columns: list[str], xgb_params: dict, tcfg: TrainingConfig) -> tuple[dict, dict]:
    """Optional point-spread model: XGBoost regression on the home margin, same features."""
    params = {k: v for k, v in xgb_params.items() if k != "monotone"}
    reg = XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=config.RANDOM_SEED,
                       n_jobs=tcfg.n_jobs, **params)
    reg.fit(dev[columns].astype(float), dev["home_margin"].to_numpy(dtype=float))
    report: dict = {}
    if len(test):
        actual = test["home_margin"].to_numpy(dtype=float)
        predicted = reg.predict(test[columns].astype(float))
        linear = LinearRegression().fit(dev[["elo_diff"]], dev["home_margin"])
        baseline = linear.predict(test[["elo_diff"]])
        report = {
            "n": int(len(test)),
            "mae": float(np.mean(np.abs(actual - predicted))),
            "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
            "mae_elo_linear_baseline": float(np.mean(np.abs(actual - baseline))),
            "mae_always_dev_mean": float(np.mean(np.abs(actual - dev["home_margin"].mean()))),
            "winner_accuracy": float(np.mean((predicted > 0) == (actual > 0))),
            "residual_std": float(np.std(actual - predicted)),
        }
    return params, report


# --------------------------------------------------------------------------- train
def train(games: pd.DataFrame | None = None, *, tcfg: TrainingConfig | None = None, fcfg: FeatureConfig | None = None,
          test_seasons: tuple[int, ...] = config.TEST_SEASONS,
          cv_first_validation_season: int = config.CV_FIRST_VALIDATION_SEASON,
          min_train_season: int | None = None, fixtures: pd.DataFrame | None = None,
          team_extras: pd.DataFrame | None = None, save: bool = True, output_dir: Path | None = None) -> TrainingResult:
    """Run the full protocol described in the module docstring and (optionally) save artefacts."""
    tcfg = tcfg or TrainingConfig()
    fcfg = fcfg or FeatureConfig()
    from_disk = games is None
    if from_disk:
        games = load_clean_games()
        fixtures = upcoming_fixtures(load_schedule(), games) if fixtures is None else fixtures
    min_train_season = int(games["season"].min()) + config.BURN_IN_SEASONS if min_train_season is None else min_train_season
    first_test = min(test_seasons)

    # 1. Elo parameters - tuned on development regular-season games only.
    base = build_features(games, None, elo_params=EloParams(), cfg=fcfg, team_extras=team_extras).matchups
    dev_rows = (base["is_played"] & (base["game_type"] == "regular") & ~base["is_neutral"]
                & (base["season"] >= min_train_season) & (base["season"] < first_test))
    elo_params, elo_table = tune_elo(base, tcfg.elo_grid, dev_rows.to_numpy())
    logger.info("Elo tuned: %s (log loss %.5f)", elo_params.to_dict(), elo_table.iloc[0]["log_loss"])

    # 2. Features with the tuned Elo (fixtures included so upcoming games are scored with the same code).
    features = build_features(games, fixtures, elo_params=elo_params, cfg=fcfg, team_extras=team_extras)
    matchups = features.matchups
    usable = matchups["is_played"] & (matchups["game_type"] == "regular") & ~matchups["is_neutral"] & (matchups["season"] >= min_train_season)
    dev_mask = usable & (matchups["season"] < first_test)
    test_mask = usable & matchups["season"].isin(test_seasons)
    dev, test = matchups[dev_mask].reset_index(drop=True), matchups[test_mask].reset_index(drop=True)
    if dev.empty:
        raise ValueError("No development rows available for training")
    cv_seasons = [int(s) for s in sorted(dev["season"].unique()) if s >= cv_first_validation_season]
    if len(cv_seasons) < 2:
        raise ValueError("Need at least two validation seasons for walk-forward CV")
    logger.info("rows: %d development, %d test; validation seasons %s", len(dev), len(test), cv_seasons)

    # 3. Feature-group ablation and selection (development seasons only).
    groups = available_feature_columns(matchups)
    selection = select_feature_groups(dev, groups, cv_seasons, tcfg)
    selected = {name: groups[name] for name in selection["selected_groups"]}
    columns = _flatten(selected)
    logger.info("selected groups: %s (%d features)", list(selected), len(columns))

    # 4. Hyperparameters, ensemble weight and calibrators (development seasons only).
    y_dev, seasons_dev = dev["home_win"].to_numpy(dtype=float), dev["season"].to_numpy()
    tuning = tune_hyperparameters(dev[columns].astype(float), y_dev, seasons_dev, cv_seasons, tcfg)
    lr_c, xgb_params = tuning["lr_c"], tuning["xgb_params"]
    dev_bundle, dev_cv = _fit_bundle(dev, columns, cv_seasons, lr_c, xgb_params, elo_params, fcfg, tcfg)
    oof, scored = dev_cv["oof"], dev_cv["scored"]

    cv_frame = dev.loc[scored, ["game_id", "game_date", "season", "home_team_id", "away_team_id", "home_win"]].copy()
    cv_frame["split"] = "cv"
    cv_frame["p_lr"] = sigmoid(oof["lr"][scored])
    cv_frame["p_xgb"] = sigmoid(oof["xgb"][scored])
    cv_frame["p_ensemble"] = walk_forward_calibrated(oof["lr"], oof["xgb"], y_dev, seasons_dev, cv_seasons)[scored]
    cv_frame["p_ensemble_raw"] = sigmoid(0.5 * (oof["lr"][scored] + oof["xgb"][scored]))
    cv_base = _baseline_probabilities(matchups, dev_mask & (matchups["season"] < cv_seasons[0]), dev_mask & matchups["season"].isin(cv_seasons), elo_params)
    for name, p in cv_base.items():
        cv_frame[f"p_{name}"] = p
    cv_block = _score_block(cv_frame["home_win"].to_numpy(dtype=float),
                            {c.removeprefix("p_"): cv_frame[c].to_numpy() for c in cv_frame.columns if c.startswith("p_")},
                            "elo", tcfg)

    # 5. One-shot evaluation of the frozen model on the test seasons.
    test_block: dict = {}
    test_frame = pd.DataFrame()
    if len(test):
        pred = dev_bundle.predict(test)
        raw = dev_bundle.models.logits(test[columns].astype(float))
        test_frame = test[["game_id", "game_date", "season", "home_team_id", "away_team_id", "home_win"]].copy()
        test_frame["split"] = "test"
        test_frame["p_lr"], test_frame["p_xgb"], test_frame["p_ensemble"] = pred["p_lr"].to_numpy(), pred["p_xgb"].to_numpy(), pred["p_ensemble"].to_numpy()
        test_frame["p_ensemble_raw"] = sigmoid(blend(raw["lr"], raw["xgb"], dev_bundle.ensemble_weight))
        for name, p in _baseline_probabilities(matchups, dev_mask, test_mask, elo_params).items():
            test_frame[f"p_{name}"] = p
        test_block = _score_block(test_frame["home_win"].to_numpy(dtype=float),
                                  {c.removeprefix("p_"): test_frame[c].to_numpy() for c in test_frame.columns if c.startswith("p_")},
                                  "elo", tcfg)
        test_block["per_season"] = _per_season(test_frame, ["p_ensemble", "p_lr", "p_xgb", "p_elo", "p_original_elo", "p_home_rate"])
    else:
        logger.warning("No test-season rows found: skipping the hold-out evaluation")
    margin_params, margin_report = _margin_model(dev, test, columns, xgb_params, tcfg)

    # 6. Production model: all seasons, frozen settings, calibrated on walk-forward scores.
    everything = matchups[usable].reset_index(drop=True)
    prod_cv_seasons = [int(s) for s in sorted(everything["season"].unique()) if s >= cv_first_validation_season]
    bundle, _ = _fit_bundle(everything, columns, prod_cv_seasons, lr_c, xgb_params, elo_params, fcfg, tcfg,
                            replicas=tcfg.bootstrap_replicas)
    margin_reg = XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=config.RANDOM_SEED,
                              n_jobs=tcfg.n_jobs, **margin_params)
    margin_reg.fit(everything[columns].astype(float), everything["home_margin"].to_numpy(dtype=float))
    bundle.margin_model = margin_reg

    importance = global_importance(dev_bundle, test if len(test) else dev)
    backtest = pd.concat([cv_frame, test_frame], ignore_index=True)
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": runtime_versions(),
        "data": {
            "games": int(matchups["is_played"].sum()), "first_date": matchups["game_date"].min(),
            "last_played_date": matchups.loc[matchups["is_played"], "game_date"].max(),
            "development_rows": int(len(dev)), "test_rows": int(len(test)), "production_rows": int(len(everything)),
            "development_seasons": sorted(int(s) for s in dev["season"].unique()),
            "validation_seasons": cv_seasons, "test_seasons": sorted(int(s) for s in test["season"].unique()) if len(test) else [],
        },
        "elo": {"params": elo_params.to_dict(), "grid_top": elo_table.head(8).to_dict("records"),
                "mean_mov_multiplier": features.elo.mean_multiplier},
        "feature_config": asdict(fcfg),
        "feature_selection": selection,
        "features": {"groups": selected, "columns": columns},
        "hyperparameters": {**tuning, "ensemble_weight_xgb_dev": dev_bundle.ensemble_weight,
                            "ensemble_weight_xgb_production": bundle.ensemble_weight},
        "calibration": {
            "development": {k: asdict(v) for k, v in dev_bundle.calibrators.items()},
            "production": {k: asdict(v) for k, v in bundle.calibrators.items()},
        },
        "cv": {**cv_block, "per_season": _per_season(cv_frame, ["p_ensemble", "p_lr", "p_xgb", "p_elo", "p_original_elo", "p_home_rate"])},
        "test": test_block,
        "margin_model": margin_report,
        "feature_importance": importance.to_dict("records"),
    }
    bundle.metadata.update(
        {
            "trained_at": metrics["generated_at"], "feature_groups": selected, "lr_c": lr_c, "xgb_params": xgb_params,
            "training_rows": int(len(everything)), "last_training_date": str(everything["game_date"].max().date()),
            "mean_mov_multiplier": features.elo.mean_multiplier,
            "margin_residual_std": margin_report.get("residual_std"),
            "uses_team_extras": team_extras is not None,
        }
    )
    metrics = _jsonable(metrics)

    if save:
        out_dir = config.MODELS_DIR if output_dir is None else Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle.save(out_dir / config.BUNDLE_PATH.name)
        (out_dir / config.METRICS_PATH.name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        backtest.to_csv(out_dir / config.BACKTEST_PATH.name, index=False)
        if from_disk and output_dir is None:
            save_processed(features)
        logger.info("Saved model bundle, metrics and backtest to %s", out_dir)
    return TrainingResult(bundle=bundle, metrics=metrics, backtest=backtest, features=features)


def summarize(metrics: dict) -> str:
    """Human-readable summary of the hold-out evaluation for the CLI."""
    lines = [f"Selected feature groups: {', '.join(metrics['feature_selection']['selected_groups'])}",
             f"Dropped by ablation:     {', '.join(metrics['feature_selection']['dropped_groups']) or 'none'}"]
    for split in ("cv", "test"):
        block = metrics.get(split) or {}
        if not block.get("models"):
            continue
        lines.append(f"\n{split.upper()} ({next(iter(block['models'].values()))['n']} games)")
        lines.append(f"{'model':<16}{'acc':>8}{'logloss':>10}{'brier':>9}{'auc':>8}{'ece':>8}{'slope':>8}")
        for name, m in sorted(block["models"].items(), key=lambda kv: kv[1]["log_loss"]):
            lines.append(f"{name:<16}{m['accuracy']:>8.4f}{m['log_loss']:>10.4f}{m['brier']:>9.4f}"
                         f"{(m['roc_auc'] or float('nan')):>8.4f}{m['ece']:>8.4f}{(m['calibration_slope'] or float('nan')):>8.3f}")
        cmp = block["vs_reference"].get("ensemble")
        if cmp:
            ll = cmp["log_loss"]
            lines.append(f"ensemble vs pure Elo: log loss {ll['difference']:+.4f} [{ll['low']:+.4f}, {ll['high']:+.4f}]"
                         f"{' (significant)' if ll['significant'] else ' (not significant)'}")
    return "\n".join(lines)
