"""Probability-forecast metrics, calibration diagnostics and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EPS: float = 1e-7


def clip_probs(p) -> np.ndarray:
    """Clip probabilities away from 0 and 1 so that logs and logits stay finite."""
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def logit(p) -> np.ndarray:
    """Log-odds of a probability."""
    p = clip_probs(p)
    return np.log(p / (1.0 - p))


def sigmoid(z) -> np.ndarray:
    """Inverse of :func:`logit` (numerically stable)."""
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def log_loss_per_game(y, p) -> np.ndarray:
    """Negative log-likelihood of each outcome."""
    y = np.asarray(y, dtype=float)
    p = clip_probs(p)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def brier_per_game(y, p) -> np.ndarray:
    """Squared error of each probability."""
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def calibration_table(y, p, bins: int = 10) -> list[dict]:
    """Reliability-diagram data: equal-width probability bins with predicted vs observed rates."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table = []
    for b in range(bins):
        in_bin = index == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        table.append(
            {
                "bin_low": float(edges[b]), "bin_high": float(edges[b + 1]), "count": count,
                "mean_predicted": float(p[in_bin].mean()), "observed_rate": float(y[in_bin].mean()),
            }
        )
    return table


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Count-weighted mean absolute gap between predicted and observed rates across bins."""
    table = calibration_table(y, p, bins)
    total = sum(row["count"] for row in table)
    return float(sum(row["count"] * abs(row["mean_predicted"] - row["observed_rate"]) for row in table) / total)


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Fit ``logit(P(y)) = intercept + slope * logit(p)``.

    A perfectly calibrated forecast has slope 1 and intercept 0; slope < 1 means the
    probabilities are too extreme (over-confident), slope > 1 too timid.
    """
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    model = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def evaluate_probabilities(y, p, bins: int = 10) -> dict:
    """All headline metrics for one set of probability forecasts."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        raise ValueError("evaluate_probabilities: no rows to score")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("evaluate_probabilities: probabilities must be finite and within [0, 1]")
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean((p > 0.5) == (y == 1))),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "log_loss": float(log_loss_per_game(y, p).mean()),
        "brier": float(brier_per_game(y, p).mean()),
        "ece": expected_calibration_error(y, p, bins),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
    }


def _metric_arrays(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "accuracy": ((p > 0.5) == (y == 1)).astype(float),
        "log_loss": log_loss_per_game(y, p),
        "brier": brier_per_game(y, p),
    }


def bootstrap_ci(y, p, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Percentile bootstrap intervals (resampling games) for accuracy, log loss and Brier."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(y), size=(n_boot, len(y)))
    alpha = (1.0 - level) / 2.0
    out = {}
    for name, values in _metric_arrays(y, p).items():
        samples = values[index].mean(axis=1)
        out[name] = {"low": float(np.quantile(samples, alpha)), "high": float(np.quantile(samples, 1 - alpha))}
    return out


def paired_bootstrap(y, p_model, p_reference, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Bootstrap the *difference* (model minus reference) on the same resampled games.

    Pairing removes the game-to-game noise shared by both forecasts, so this is the right
    test for "is model A really better than model B on this sample".
    """
    y = np.asarray(y, dtype=float)
    a = _metric_arrays(y, np.asarray(p_model, dtype=float))
    b = _metric_arrays(y, np.asarray(p_reference, dtype=float))
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(y), size=(n_boot, len(y)))
    alpha = (1.0 - level) / 2.0
    out = {}
    for name in a:
        delta = a[name] - b[name]
        samples = delta[index].mean(axis=1)
        low, high = float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))
        out[name] = {"difference": float(delta.mean()), "low": low, "high": high,
                     "significant": bool(low > 0 or high < 0)}
    return out
