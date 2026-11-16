"""Confidence and volatility indicators derived from the statistical models.

These are deterministic functions of model outputs and pre-game facts. The LLM analyst
receives them as inputs and may explain them, but it can never change them.

* **Confidence** answers "how firmly does the model favour one side?" - the size of the
  edge, discounted when the two models disagree, when bootstrap replicas disagree, or when
  the season is too young for the statistics to be reliable.
* **Volatility risk** (0-100) answers "how exposed is this forecast to an upset or a wild
  game?". It is a transparent weighted index, not a probability; the weights below are
  judgement calls and are shown to the user alongside the score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Component weights of the volatility index (sum to 1).
VOLATILITY_WEIGHTS: dict[str, float] = {
    "close_matchup": 0.45,
    "model_disagreement": 0.15,
    "model_uncertainty": 0.10,
    "small_sample": 0.10,
    "three_point_reliance": 0.10,
    "schedule_stress": 0.10,
}
DISAGREEMENT_SCALE: float = 0.10  # |p_xgb - p_lr| treated as "maximal" disagreement
UNCERTAINTY_SCALE: float = 0.04  # bootstrap std treated as "maximal" model uncertainty
EARLY_SEASON_GAMES: int = 10
THREE_POINT_RANGE: float = 0.06  # 3PA-rate above the league average treated as "maximal" reliance

VOLATILITY_LABELS: dict[str, str] = {
    "close_matchup": "Teams are closely matched",
    "model_disagreement": "Logistic regression and XGBoost disagree",
    "model_uncertainty": "Bootstrap replicas of the model disagree",
    "small_sample": "Few games played this season",
    "three_point_reliance": "Three-point volume above the league average (high-variance scoring)",
    "schedule_stress": "Back-to-back or third game in four nights",
}


@dataclass(frozen=True)
class UncertaintyInputs:
    """Numbers the indicators are computed from."""

    p_home: float
    p_lr: float
    p_xgb: float
    p_std: float
    min_games_played: float
    mean_fg3a_rate: float
    league_fg3a_rate: float
    schedule_stress: bool


def confidence(inputs: UncertaintyInputs) -> dict:
    """Confidence score (0-100) and level for the favoured side."""
    edge = abs(2.0 * inputs.p_home - 1.0)
    disagreement = abs(inputs.p_xgb - inputs.p_lr)
    early = 0.15 if inputs.min_games_played < EARLY_SEASON_GAMES else 0.0
    penalty = float(min(0.5, 2.0 * disagreement + 4.0 * inputs.p_std + early))
    score = 100.0 * edge * (1.0 - penalty)
    level = "high" if score >= 40.0 else "medium" if score >= 15.0 else "low"
    return {
        "level": level, "score": round(score, 1), "edge": round(edge, 4), "penalty": round(penalty, 4),
        "model_disagreement": round(disagreement, 4), "bootstrap_std": round(inputs.p_std, 4),
    }


def volatility(inputs: UncertaintyInputs) -> dict:
    """Volatility risk score (0-100) with its drivers, largest contribution first."""
    components = {
        "close_matchup": 1.0 - abs(2.0 * inputs.p_home - 1.0),
        "model_disagreement": min(1.0, abs(inputs.p_xgb - inputs.p_lr) / DISAGREEMENT_SCALE),
        "model_uncertainty": min(1.0, inputs.p_std / UNCERTAINTY_SCALE),
        "small_sample": 1.0 if inputs.min_games_played < EARLY_SEASON_GAMES
        else 0.5 if inputs.min_games_played < 2 * EARLY_SEASON_GAMES else 0.0,
        "three_point_reliance": float(np.clip((inputs.mean_fg3a_rate - inputs.league_fg3a_rate) / THREE_POINT_RANGE, 0.0, 1.0))
        if np.isfinite(inputs.mean_fg3a_rate) and np.isfinite(inputs.league_fg3a_rate) else 0.0,
        "schedule_stress": 1.0 if inputs.schedule_stress else 0.0,
    }
    drivers = [
        {"factor": name, "label": VOLATILITY_LABELS[name], "level": round(float(value), 3),
         "weight": VOLATILITY_WEIGHTS[name], "points": round(100.0 * VOLATILITY_WEIGHTS[name] * float(value), 1)}
        for name, value in components.items()
    ]
    drivers.sort(key=lambda d: d["points"], reverse=True)
    score = float(sum(d["points"] for d in drivers))
    level = "high" if score >= 60.0 else "medium" if score >= 40.0 else "low"
    return {"score": round(score, 1), "level": level, "drivers": drivers}
