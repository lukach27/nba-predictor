"""Exact additive explanations (SHAP values) of the calibrated ensemble forecast.

* XGBoost: TreeSHAP values come from XGBoost itself (``pred_contribs=True``), the same
  algorithm the ``shap`` package wraps, so no extra dependency is needed.
* Logistic regression: for a linear model the SHAP value of a feature is
  ``coefficient * (standardised value)`` with the training mean as the baseline.
* Ensemble: the forecast is linear in the two models' log-odds
  (``a * (w*z_xgb + (1-w)*z_lr) + b``), so its SHAP values are the same linear combination.

Contributions are in log-odds of a home win and satisfy
``base_value + sum(contributions) == logit(p_ensemble)`` exactly (tested).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .modeling import ModelBundle


def model_contributions(bundle: ModelBundle, frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-model ``(contributions, base_value)`` in calibrated log-odds.

    Returns a dict with keys ``lr``, ``xgb`` and ``ensemble``; contributions have shape
    (rows, features) and base values shape (rows,).
    """
    x = frame[bundle.feature_columns].astype(float)
    booster = bundle.models.xgb.get_booster()
    tree = booster.predict(xgb.DMatrix(x, feature_names=bundle.feature_columns), pred_contribs=True)
    xgb_contrib, xgb_base = tree[:, :-1].astype(float), tree[:, -1].astype(float)

    pipeline = bundle.models.lr
    scaled = pipeline[:-1].transform(x)
    clf = pipeline.named_steps["clf"]
    lr_contrib = scaled * clf.coef_[0]
    lr_base = np.full(len(x), float(clf.intercept_[0]))

    w = bundle.ensemble_weight
    cal = bundle.calibrators
    out = {
        "lr": (cal["lr"].slope * lr_contrib, cal["lr"].slope * lr_base + cal["lr"].intercept),
        "xgb": (cal["xgb"].slope * xgb_contrib, cal["xgb"].slope * xgb_base + cal["xgb"].intercept),
    }
    ens_contrib = cal["ensemble"].slope * (w * xgb_contrib + (1.0 - w) * lr_contrib)
    ens_base = cal["ensemble"].slope * (w * xgb_base + (1.0 - w) * lr_base) + cal["ensemble"].intercept
    out["ensemble"] = (ens_contrib, ens_base)
    return out


def ensemble_contributions(bundle: ModelBundle, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """SHAP values of the deployed forecast as a DataFrame (columns = features) plus base values."""
    contrib, base = model_contributions(bundle, frame)["ensemble"]
    return pd.DataFrame(contrib, index=frame.index, columns=bundle.feature_columns), base


def global_importance(bundle: ModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per feature for each model, sorted by the ensemble's."""
    parts = model_contributions(bundle, frame)
    table = pd.DataFrame({"feature": bundle.feature_columns})
    for name, (contrib, _) in parts.items():
        table[f"mean_abs_{name}"] = np.abs(contrib).mean(axis=0)
    return table.sort_values("mean_abs_ensemble", ascending=False).reset_index(drop=True)


def explain_row(bundle: ModelBundle, row: pd.DataFrame, top: int | None = None) -> pd.DataFrame:
    """Feature contributions for a single game, largest absolute effect first.

    Columns: ``feature, value, contribution`` (log-odds; positive favours the home team).
    """
    if len(row) != 1:
        raise ValueError("explain_row expects exactly one row")
    contrib, _ = ensemble_contributions(bundle, row)
    table = pd.DataFrame(
        {
            "feature": bundle.feature_columns,
            "value": row[bundle.feature_columns].astype(float).iloc[0].to_numpy(),
            "contribution": contrib.iloc[0].to_numpy(),
        }
    )
    table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return table if top is None else table.head(top)
