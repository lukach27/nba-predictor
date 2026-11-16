"""Metrics, calibration, walk-forward splits, the training protocol, serialisation and SHAP."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nba_predictor import config
from nba_predictor.explain import ensemble_contributions, explain_row, global_importance
from nba_predictor.features import OUTCOME_COLUMNS
from nba_predictor.metrics import (bootstrap_ci, calibration_table, evaluate_probabilities, expected_calibration_error,
                                   log_loss_per_game, logit, paired_bootstrap, sigmoid)
from nba_predictor.modeling import (ModelBundle, ModelBundleError, PlattCalibrator, best_ensemble_weight,
                                    walk_forward_folds)
from nba_predictor.train_model import walk_forward_calibrated


# ------------------------------------------------------------------------ metrics
def test_metrics_on_a_hand_checked_example():
    y = np.array([1, 0, 1, 1])
    p = np.array([0.8, 0.3, 0.6, 0.4])
    m = evaluate_probabilities(y, p)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["brier"] == pytest.approx(np.mean([0.04, 0.09, 0.16, 0.36]))
    assert m["log_loss"] == pytest.approx(-np.mean(np.log([0.8, 0.7, 0.6, 0.4])))
    assert m["roc_auc"] == pytest.approx(1.0)


def test_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError):
        evaluate_probabilities([1, 0], [1.2, 0.3])
    with pytest.raises(ValueError):
        evaluate_probabilities([1, 0], [np.nan, 0.3])


def test_logit_sigmoid_roundtrip_and_extremes_stay_finite():
    p = np.array([0.0, 0.01, 0.5, 0.99, 1.0])
    assert np.isfinite(logit(p)).all()
    np.testing.assert_allclose(sigmoid(logit(p[1:-1])), p[1:-1])
    assert np.isfinite(log_loss_per_game([1, 0], [0.0, 1.0])).all()


def test_calibration_error_is_small_for_calibrated_and_large_for_overconfident():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20_000)
    y = (rng.random(20_000) < p).astype(float)
    assert expected_calibration_error(y, p) < 0.02
    overconfident = sigmoid(3 * logit(p))
    assert expected_calibration_error(y, overconfident) > 0.08
    table = calibration_table(y, p, bins=10)
    assert sum(row["count"] for row in table) == 20_000
    assert all(abs(row["mean_predicted"] - row["observed_rate"]) < 0.05 for row in table)


def test_paired_bootstrap_detects_a_better_forecast():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.1, 0.9, 3000)
    y = (rng.random(3000) < truth).astype(float)
    result = paired_bootstrap(y, truth, np.full(3000, 0.5), n_boot=300, seed=0)
    assert result["log_loss"]["difference"] < 0 and result["log_loss"]["significant"]
    same = paired_bootstrap(y, truth, truth, n_boot=100, seed=0)
    assert same["brier"]["difference"] == 0 and not same["brier"]["significant"]
    ci = bootstrap_ci(y, truth, n_boot=200, seed=0)
    assert ci["accuracy"]["low"] <= np.mean((truth > 0.5) == (y == 1)) <= ci["accuracy"]["high"]


# -------------------------------------------------------------------- calibration
def test_platt_calibration_fixes_overconfidence():
    rng = np.random.default_rng(2)
    true_logit = rng.normal(0, 1.2, 8000)
    y = (rng.random(8000) < sigmoid(true_logit)).astype(float)
    overconfident = 2.5 * true_logit  # a model that is far too sure of itself
    fit, test = slice(0, 5000), slice(5000, None)
    cal = PlattCalibrator.fit(overconfident[fit], y[fit])
    assert cal.slope == pytest.approx(1 / 2.5, rel=0.15)
    calibrated = cal.predict(overconfident[test])
    assert ((calibrated > 0) & (calibrated < 1)).all()
    assert log_loss_per_game(y[test], calibrated).mean() < log_loss_per_game(y[test], sigmoid(overconfident[test])).mean()
    assert expected_calibration_error(y[test], calibrated) < expected_calibration_error(y[test], sigmoid(overconfident[test]))
    order = np.argsort(overconfident[test])
    assert np.all(np.diff(calibrated[order]) >= 0), "calibration must preserve the ranking of games"


def test_platt_falls_back_to_identity_on_degenerate_input():
    assert PlattCalibrator.fit(np.zeros(10), np.ones(10)) == PlattCalibrator()
    np.testing.assert_allclose(PlattCalibrator().predict([0.0, 1.0]), sigmoid([0.0, 1.0]))


def test_ensemble_weight_picks_the_informative_model():
    rng = np.random.default_rng(3)
    z = rng.normal(0, 1, 4000)
    y = (rng.random(4000) < sigmoid(z)).astype(float)
    noise = rng.normal(0, 1, 4000)
    assert best_ensemble_weight(z_lr=noise, z_xgb=z, y=y) >= 0.9
    assert best_ensemble_weight(z_lr=z, z_xgb=noise, y=y) <= 0.1


# ------------------------------------------------------------- chronological splits
def test_walk_forward_folds_never_train_on_the_future():
    seasons = np.repeat([2018, 2019, 2020, 2021], 5)
    folds = walk_forward_folds(seasons, [2019, 2020, 2021])
    assert [f[0] for f in folds] == [2019, 2020, 2021]
    for season, train_idx, val_idx in folds:
        assert seasons[train_idx].max() < season == seasons[val_idx].min()
        assert not set(train_idx) & set(val_idx)
    assert walk_forward_folds(seasons, [2018]) == []  # nothing earlier to train on


def test_walk_forward_calibration_only_uses_earlier_seasons():
    rng = np.random.default_rng(4)
    seasons = np.repeat([2019, 2020, 2021], 700)
    z = rng.normal(0, 1, len(seasons))
    y = (rng.random(len(seasons)) < sigmoid(z)).astype(float)
    base = walk_forward_calibrated(2 * z, 2 * z, y, seasons, [2019, 2020, 2021])
    y_future_changed = y.copy()
    y_future_changed[seasons == 2021] = 1 - y_future_changed[seasons == 2021]
    again = walk_forward_calibrated(2 * z, 2 * z, y_future_changed, seasons, [2019, 2020, 2021])
    np.testing.assert_allclose(base[seasons <= 2020], again[seasons <= 2020])
    np.testing.assert_allclose(base[seasons == 2019], sigmoid(2 * z[seasons == 2019]))  # no history: identity


# ------------------------------------------------------------- training protocol
def test_training_respects_the_chronological_protocol(training):
    data = training.metrics["data"]
    assert max(data["development_seasons"]) < min(data["test_seasons"])
    assert max(data["validation_seasons"]) < min(data["test_seasons"])
    backtest = training.backtest
    cv, test = backtest[backtest["split"] == "cv"], backtest[backtest["split"] == "test"]
    assert pd.to_datetime(cv["game_date"]).max() < pd.to_datetime(test["game_date"]).min()
    assert set(test["season"]) == {2022} and len(test) == data["test_rows"]


def test_training_outputs_valid_probabilities_and_metrics(training):
    backtest = training.backtest
    for col in ("p_ensemble", "p_lr", "p_xgb", "p_elo", "p_original_elo", "p_home_rate"):
        assert backtest[col].between(0, 1).all() and backtest[col].notna().all(), col
    test = training.metrics["test"]
    for name in ("ensemble", "lr", "xgb", "elo", "original_elo", "home_rate"):
        m = test["models"][name]
        assert 0 <= m["accuracy"] <= 1 and m["brier"] >= 0 and m["log_loss"] > 0
        assert m["ci"]["accuracy"]["low"] <= m["accuracy"] <= m["ci"]["accuracy"]["high"]
    assert test["models"]["ensemble"]["log_loss"] < test["models"]["home_rate"]["log_loss"]
    assert "ensemble" in test["vs_reference"] and test["reference"] == "elo"
    assert test["calibration"]["ensemble"]


def test_selected_features_are_configured_pre_game_columns(training):
    allowed = set(config.all_feature_columns({**config.FEATURE_GROUPS, **config.OPTIONAL_FEATURE_GROUPS}))
    selected = training.bundle.feature_columns
    assert selected and set(selected) <= allowed
    assert not set(selected) & set(OUTCOME_COLUMNS)
    selection = training.metrics["feature_selection"]
    assert set(selection["selected_groups"]) | set(selection["dropped_groups"]) >= set(config.FEATURE_GROUPS)
    assert selection["selected_log_loss"] <= selection["full_log_loss"] + 1e-12


def test_artifacts_are_written_and_metrics_are_valid_json(training):
    out = training.output_dir
    for name in (config.BUNDLE_PATH.name, config.METRICS_PATH.name, config.BACKTEST_PATH.name):
        assert (out / name).exists(), name
    loaded = json.loads((out / config.METRICS_PATH.name).read_text(encoding="utf-8"))
    assert loaded["hyperparameters"]["lr_c"] in (0.1, 1.0)
    assert loaded["elo"]["params"]["k"] == 20.0


def test_bundle_roundtrip_reproduces_predictions_exactly(training):
    frame = training.features.matchups
    frame = frame[frame["season"] == 2022]
    loaded = ModelBundle.load(training.output_dir / config.BUNDLE_PATH.name)
    original, restored = training.bundle.predict(frame), loaded.predict(frame)
    pd.testing.assert_frame_equal(original, restored)
    assert loaded.feature_columns == training.bundle.feature_columns
    assert loaded.elo_params == training.bundle.elo_params and loaded.feature_config == training.bundle.feature_config
    assert restored["p_ensemble"].between(0, 1).all() and (restored["p_low"] <= restored["p_high"]).all()
    np.testing.assert_allclose(training.bundle.predict_margin(frame), loaded.predict_margin(frame))


def test_loading_a_missing_or_wrong_bundle_fails_loudly(tmp_path):
    with pytest.raises(ModelBundleError, match="not found"):
        ModelBundle.load(tmp_path / "nope.joblib")
    import joblib

    joblib.dump({"not": "a bundle"}, tmp_path / "bad.joblib")
    with pytest.raises(ModelBundleError, match="does not contain"):
        ModelBundle.load(tmp_path / "bad.joblib")


def test_bundle_rejects_frames_without_its_features(training):
    with pytest.raises(ModelBundleError, match="lacks model features"):
        training.bundle.predict(pd.DataFrame({"elo_diff": [0.0]})[[]].assign(dummy=1.0))


# --------------------------------------------------------------------------- SHAP
def test_shap_contributions_add_up_to_the_forecast(training):
    frame = training.features.matchups
    frame = frame[frame["season"] == 2022].head(50)
    contributions, base = ensemble_contributions(training.bundle, frame)
    predicted = training.bundle.predict(frame)
    np.testing.assert_allclose(base + contributions.sum(axis=1).to_numpy(), predicted["logit_ensemble"].to_numpy(), atol=1e-4)
    np.testing.assert_allclose(sigmoid(base + contributions.sum(axis=1).to_numpy()), predicted["p_ensemble"].to_numpy(), atol=1e-5)


def test_explanations_are_ranked_and_complete(training):
    frame = training.features.matchups
    row = frame[frame["season"] == 2022].head(1)
    table = explain_row(training.bundle, row)
    assert list(table.columns) == ["feature", "value", "contribution"]
    assert set(table["feature"]) == set(training.bundle.feature_columns)
    assert table["contribution"].abs().is_monotonic_decreasing
    importance = global_importance(training.bundle, frame[frame["season"] == 2022])
    assert (importance[["mean_abs_lr", "mean_abs_xgb", "mean_abs_ensemble"]] >= 0).all().all()
    assert importance["mean_abs_ensemble"].is_monotonic_decreasing
