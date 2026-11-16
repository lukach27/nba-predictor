"""Single-game basketball metrics derived from the team box score.

Two kinds of quantities are produced and kept clearly apart:

* **Observed** ratios computed exactly from box-score counts: the Four Factors
  (eFG%, TOV%, ORB%, FT rate), their defensive mirrors and the three-point attempt rate.
* **Estimated** quantities that rest on the possession estimate
  ``FGA + 0.44*FTA - OREB + TOV`` (the 0.44 free-throw coefficient is an empirical league
  average, not an observation): possessions, pace and offensive/defensive/net rating.

These are *post-game* numbers. They must only reach a model after being lagged by
``features.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FT_POSSESSION_WEIGHT: float = 0.44
REGULATION_MINUTES: float = 48.0

#: Metrics observed directly from the box score.
OBSERVED_METRICS: list[str] = [
    "efg", "tov_pct", "orb_pct", "ft_rate", "opp_efg", "opp_tov_pct", "drb_pct", "opp_ft_rate", "fg3a_rate",
]
#: Metrics that depend on the possession estimate.
ESTIMATED_METRICS: list[str] = ["poss", "pace", "off_rtg", "def_rtg", "net_rtg"]
GAME_METRICS: list[str] = OBSERVED_METRICS + ESTIMATED_METRICS


def _safe_div(num, den):
    """Element-wise division returning NaN (not inf) when the denominator is zero or missing."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=(den != 0) & ~np.isnan(den))
    return out


def effective_fg_pct(fgm, fg3m, fga):
    """eFG% = (FGM + 0.5 * 3PM) / FGA - field-goal percentage that credits threes with 1.5x."""
    return _safe_div(np.asarray(fgm, dtype=float) + 0.5 * np.asarray(fg3m, dtype=float), fga)


def turnover_pct(tov, fga, fta):
    """TOV% = TOV / (FGA + 0.44 * FTA + TOV) - share of plays ending in a turnover."""
    tov = np.asarray(tov, dtype=float)
    return _safe_div(tov, np.asarray(fga, dtype=float) + FT_POSSESSION_WEIGHT * np.asarray(fta, dtype=float) + tov)


def rebound_pct(own_reb, opp_reb):
    """Share of available rebounds secured: ORB% = OREB / (OREB + opp DREB); DRB% is the mirror."""
    own_reb = np.asarray(own_reb, dtype=float)
    return _safe_div(own_reb, own_reb + np.asarray(opp_reb, dtype=float))


def free_throw_rate(ftm, fga):
    """FT rate = FTM / FGA (Dean Oliver's definition: how often a team gets to the line and converts)."""
    return _safe_div(ftm, fga)


def estimate_possessions(fga, fta, oreb, tov):
    """One team's possession estimate: FGA + 0.44 * FTA - OREB + TOV."""
    return (np.asarray(fga, dtype=float) + FT_POSSESSION_WEIGHT * np.asarray(fta, dtype=float)
            - np.asarray(oreb, dtype=float) + np.asarray(tov, dtype=float))


def game_minutes(team_minutes, opp_minutes=None):
    """Length of the game in minutes (48 plus 5 per overtime), robust to +-1 rounding in ``MIN``."""
    minutes = np.asarray(team_minutes, dtype=float) / 5.0
    if opp_minutes is not None:
        other = np.asarray(opp_minutes, dtype=float) / 5.0
        both = ~np.isnan(minutes) & ~np.isnan(other)
        minutes = np.where(both, 0.5 * (minutes + other), np.where(np.isnan(minutes), other, minutes))
    overtime_periods = np.round(np.clip(minutes - REGULATION_MINUTES, 0, None) / 5.0)
    return REGULATION_MINUTES + 5.0 * overtime_periods


def add_game_metrics(games: pd.DataFrame) -> pd.DataFrame:
    """Return ``games`` with every metric in :data:`GAME_METRICS` appended.

    Requires the team box columns and the opponent's (``OPP_`` prefix). Rows with missing box
    scores (for example unplayed fixtures) get NaN metrics.
    """
    df = games.copy()
    df["efg"] = effective_fg_pct(df["FGM"], df["FG3M"], df["FGA"])
    df["tov_pct"] = turnover_pct(df["TOV"], df["FGA"], df["FTA"])
    df["orb_pct"] = rebound_pct(df["OREB"], df["OPP_DREB"])
    df["ft_rate"] = free_throw_rate(df["FTM"], df["FGA"])
    df["opp_efg"] = effective_fg_pct(df["OPP_FGM"], df["OPP_FG3M"], df["OPP_FGA"])
    df["opp_tov_pct"] = turnover_pct(df["OPP_TOV"], df["OPP_FGA"], df["OPP_FTA"])
    df["drb_pct"] = rebound_pct(df["DREB"], df["OPP_OREB"])
    df["opp_ft_rate"] = free_throw_rate(df["OPP_FTM"], df["OPP_FGA"])
    df["fg3a_rate"] = _safe_div(df["FG3A"], df["FGA"])

    own = estimate_possessions(df["FGA"], df["FTA"], df["OREB"], df["TOV"])
    opp = estimate_possessions(df["OPP_FGA"], df["OPP_FTA"], df["OPP_OREB"], df["OPP_TOV"])
    df["poss"] = 0.5 * (own + opp)  # both teams share (almost) the same number of possessions
    df["pace"] = _safe_div(REGULATION_MINUTES * df["poss"], game_minutes(df["MIN"], df["OPP_MIN"]))
    df["off_rtg"] = _safe_div(100.0 * df["PTS"], df["poss"])
    df["def_rtg"] = _safe_div(100.0 * df["OPP_PTS"], df["poss"])
    df["net_rtg"] = df["off_rtg"] - df["def_rtg"]
    return df
