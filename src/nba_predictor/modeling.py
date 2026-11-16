"""Model definitions, calibration and the serialisable :class:`ModelBundle`.

The deployed forecast is a calibrated blend of two models in log-odds space::

    z_ensemble = w * z_xgboost + (1 - w) * z_logistic
    p          = sigmoid(a * z_ensemble + b)            # Platt calibration

Blending log-odds (not probabilities) keeps the final forecast an exactly additive function
of each model's feature contributions, which is what makes the SHAP explanation in
``explain.py`` exact rather than approximate.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import __version__, config
from .config import FeatureConfig
from .elo import EloParams
from .metrics import log_loss_per_game, sigmoid

logger = logging.getLogger(__name__)
MODEL_NAMES: tuple[str, ...] = ("lr", "xgb", "ensemble")


class ModelBundleError(RuntimeError):
    """Raised when a saved model bundle is missing or inconsistent."""


# ------------------------------------------------------------------- calibration
@dataclass
class PlattCalibrator:
    """Two-parameter logistic recalibration of a log-odds score: ``p = sigmoid(a*z + b)``."""

    slope: float = 1.0
    intercept: float = 0.0

    @classmethod
    def fit(cls, logits: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        """Fit on out-of-fold scores. Falls back to the identity if the fit is degenerate."""
        logits = np.asarray(logits, dtype=float)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2 or len(y) < 50:
            logger.warning("PlattCalibrator.fit: too little data (%d rows) - using identity", len(y))
            return cls()
        model = LogisticRegression(C=1e6, max_iter=1000).fit(logits.reshape(-1, 1), y)
        slope, intercept = float(model.coef_[0, 0]), float(model.intercept_[0])
        if not np.isfinite(slope) or slope <= 0:
            logger.warning("PlattCalibrator.fit: non-positive slope %.3f - using identity", slope)
            return cls()
        return cls(slope=slope, intercept=intercept)

    def transform(self, logits) -> np.ndarray:
        """Calibrated log-odds."""
        return self.slope * np.asarray(logits, dtype=float) + self.intercept

    def predict(self, logits) -> np.ndarray:
        """Calibrated probability."""
        return sigmoid(self.transform(logits))


# ------------------------------------------------------------------- base models
def make_lr(c: float) -> Pipeline:
    """Median imputation -> standardisation -> L2 logistic regression."""
    return Pipeline(
        [
            # keep_empty_features: a column with no data yet stays in place (as zeros) so the
            # feature order - and therefore every explanation - always matches the bundle.
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=c, max_iter=2000)),
        ]
    )


def make_xgb(params: dict, feature_columns: list[str], seed: int = config.RANDOM_SEED, n_jobs: int = 1) -> XGBClassifier:
    """Gradient-boosted trees; ``params['monotone']`` enables the sign constraints in config."""
    params = dict(params)
    monotone = bool(params.pop("monotone", False))
    constraints = None
    if monotone:
        constraints = "(" + ",".join("1" if c in config.MONOTONE_INCREASING else "0" for c in feature_columns) + ")"
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=seed,
        n_jobs=n_jobs, monotone_constraints=constraints, **params,
    )


@dataclass
class BaseModels:
    """A fitted logistic regression and XGBoost pair."""

    lr: Pipeline
    xgb: XGBClassifier

    def logits(self, x: pd.DataFrame) -> dict[str, np.ndarray]:
        """Uncalibrated log-odds of a home win from each model."""
        return {
            "lr": self.lr.decision_function(x),
            "xgb": self.xgb.predict(x, output_margin=True).astype(float),
        }


def fit_base_models(x: pd.DataFrame, y: np.ndarray, lr_c: float, xgb_params: dict,
                    seed: int = config.RANDOM_SEED, n_jobs: int = 1) -> BaseModels:
    """Fit both base models on the same rows."""
    lr = make_lr(lr_c).fit(x, y)
    xgb = make_xgb(xgb_params, list(x.columns), seed=seed, n_jobs=n_jobs).fit(x, y)
    return BaseModels(lr=lr, xgb=xgb)


def walk_forward_folds(seasons: np.ndarray, validation_seasons: list[int]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Expanding-window folds: validate on season ``s``, train on every earlier season.

    Chronological by construction - no validation game is ever older than a training game.
    """
    seasons = np.asarray(seasons)
    folds = []
    for season in validation_seasons:
        train_idx = np.flatnonzero(seasons < season)
        val_idx = np.flatnonzero(seasons == season)
        if len(train_idx) and len(val_idx):
            folds.append((int(season), train_idx, val_idx))
    return folds


def out_of_fold_logits(x: pd.DataFrame, y: np.ndarray, seasons: np.ndarray, validation_seasons: list[int],
                       lr_c: float, xgb_params: dict, seed: int = config.RANDOM_SEED, n_jobs: int = 1,
                       models: tuple[str, ...] = ("lr", "xgb")) -> dict[str, np.ndarray]:
    """Walk-forward out-of-fold log-odds (NaN for rows that are in no validation season)."""
    out = {name: np.full(len(x), np.nan) for name in models}
    for _, train_idx, val_idx in walk_forward_folds(seasons, validation_seasons):
        x_train, y_train, x_val = x.iloc[train_idx], y[train_idx], x.iloc[val_idx]
        if "lr" in models:
            out["lr"][val_idx] = make_lr(lr_c).fit(x_train, y_train).decision_function(x_val)
        if "xgb" in models:
            model = make_xgb(xgb_params, list(x.columns), seed=seed, n_jobs=n_jobs).fit(x_train, y_train)
            out["xgb"][val_idx] = model.predict(x_val, output_margin=True)
    return out


def best_ensemble_weight(z_lr: np.ndarray, z_xgb: np.ndarray, y: np.ndarray) -> float:
    """Weight on the XGBoost log-odds that minimises out-of-fold log loss (grid of 0.05)."""
    grid = np.linspace(0.0, 1.0, 21)
    losses = [log_loss_per_game(y, sigmoid(w * z_xgb + (1.0 - w) * z_lr)).mean() for w in grid]
    return float(grid[int(np.argmin(losses))])


def blend(z_lr: np.ndarray, z_xgb: np.ndarray, weight: float) -> np.ndarray:
    """Ensemble log-odds."""
    return weight * np.asarray(z_xgb, dtype=float) + (1.0 - weight) * np.asarray(z_lr, dtype=float)


def fit_calibrators(z_lr: np.ndarray, z_xgb: np.ndarray, y: np.ndarray, weight: float) -> dict[str, PlattCalibrator]:
    """Platt calibrators for each base model and for the ensemble, from out-of-fold scores."""
    return {
        "lr": PlattCalibrator.fit(z_lr, y),
        "xgb": PlattCalibrator.fit(z_xgb, y),
        "ensemble": PlattCalibrator.fit(blend(z_lr, z_xgb, weight), y),
    }


# ------------------------------------------------------------------------ bundle
@dataclass
class ModelBundle:
    """Everything inference needs, saved as one file so training and serving cannot diverge."""

    feature_columns: list[str]
    models: BaseModels
    ensemble_weight: float
    calibrators: dict[str, PlattCalibrator]
    elo_params: EloParams
    feature_config: FeatureConfig
    replicas: list[BaseModels] = field(default_factory=list)
    margin_model: object | None = None  # optional point-spread regressor (see train_model)
    metadata: dict = field(default_factory=dict)

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise ModelBundleError(f"input frame lacks model features {missing}")
        return frame[self.feature_columns].astype(float)

    def logits(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Calibrated log-odds of a home win for ``lr``, ``xgb`` and ``ensemble``."""
        raw = self.models.logits(self._matrix(frame))
        raw["ensemble"] = blend(raw["lr"], raw["xgb"], self.ensemble_weight)
        return {name: self.calibrators[name].transform(raw[name]) for name in MODEL_NAMES}

    def replica_logits(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated ensemble log-odds from each bootstrap replica: shape (replicas, rows)."""
        if not self.replicas:
            return np.empty((0, len(frame)))
        x = self._matrix(frame)
        rows = []
        for replica in self.replicas:
            raw = replica.logits(x)
            rows.append(self.calibrators["ensemble"].transform(blend(raw["lr"], raw["xgb"], self.ensemble_weight)))
        return np.vstack(rows)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Probabilities for every model plus the bootstrap uncertainty band of the ensemble."""
        z = self.logits(frame)
        out = pd.DataFrame({f"p_{name}": sigmoid(z[name]) for name in MODEL_NAMES}, index=frame.index)
        out["logit_ensemble"] = z["ensemble"]
        replica_p = sigmoid(self.replica_logits(frame))
        if len(replica_p):
            out["p_std"] = replica_p.std(axis=0)
            out["p_low"] = np.quantile(replica_p, 0.1, axis=0)
            out["p_high"] = np.quantile(replica_p, 0.9, axis=0)
        else:
            out["p_std"], out["p_low"], out["p_high"] = 0.0, out["p_ensemble"], out["p_ensemble"]
        return out

    def predict_margin(self, frame: pd.DataFrame) -> np.ndarray | None:
        """Expected home point margin, when the optional margin model was trained."""
        if self.margin_model is None:
            return None
        return np.asarray(self.margin_model.predict(self._matrix(frame)), dtype=float)

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path | None = None) -> Path:
        """Serialise the bundle (models, preprocessing, calibration, feature definitions)."""
        path = config.BUNDLE_PATH if path is None else Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata.setdefault("saved_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.metadata.setdefault("versions", runtime_versions())
        joblib.dump(self, path, compress=3)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ModelBundle":
        """Load a bundle and warn when library versions differ from those used in training."""
        path = config.BUNDLE_PATH if path is None else Path(path)
        if not path.exists():
            raise ModelBundleError(f"{path} not found - run `python -m nba_predictor train` first")
        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise ModelBundleError(f"{path} does not contain a ModelBundle")
        saved = bundle.metadata.get("versions", {})
        current = runtime_versions()
        for lib in ("xgboost", "scikit-learn"):
            if saved.get(lib) and saved[lib] != current[lib]:
                logger.warning("Model was trained with %s %s but %s is installed", lib, saved[lib], current[lib])
        return bundle


def runtime_versions() -> dict[str, str]:
    """Library versions recorded with every artefact for reproducibility."""
    return {
        "nba_predictor": __version__, "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
    }
