"""Elo ratings with home-court advantage, margin-of-victory scaling and season regression.

Compared with the original project's Elo (K=20, nothing else) this version:

* adds a home-court term to the expected score (and removes it at neutral sites),
* scales each update by the margin of victory with the FiveThirtyEight multiplier, which
  also damps the autocorrelation of favourites winning big,
* regresses every rating toward the mean between seasons,
* uses playoff and play-in games, not only the regular season.

The engine is a pure function of the game list: the rating attached to a game is the rating
*before* tip-off and never includes that game's result.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

ELO_SCALE: float = 400.0
#: Natural-log odds per Elo point; converts rating differences into logit shifts.
LOGIT_PER_ELO: float = math.log(10.0) / ELO_SCALE


@dataclass(frozen=True)
class EloParams:
    """Parameters of the Elo engine."""

    k: float = 20.0
    home_advantage: float = 70.0
    season_carry: float = 0.75  # share of the distance from the mean kept between seasons
    use_mov: bool = True
    mean_rating: float = 1500.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EloResult:
    """Output of :func:`run_elo`. Arrays are aligned with the input frame's row order."""

    home_pre: np.ndarray
    away_pre: np.ndarray
    ratings: dict[int, float]  # ratings after the last played game
    mean_multiplier: float  # average margin-of-victory multiplier over played games


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B under the Elo logistic curve."""
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / ELO_SCALE))


def mov_multiplier(margin: float, winner_edge: float) -> float:
    """FiveThirtyEight margin-of-victory multiplier.

    ``winner_edge`` is the winner's pre-game rating edge (home advantage included). Bigger wins
    move ratings more, but less so when the winner was already the favourite.
    """
    return ((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * winner_edge)


def home_win_probability(elo_home, elo_away, home_advantage: float, is_neutral=False):
    """Vectorised pure-Elo home win probability."""
    edge = np.asarray(elo_home, dtype=float) - np.asarray(elo_away, dtype=float)
    edge = edge + np.where(np.asarray(is_neutral, dtype=bool), 0.0, home_advantage)
    return 1.0 / (1.0 + 10.0 ** (-edge / ELO_SCALE))


def run_elo(games: pd.DataFrame, params: EloParams) -> EloResult:
    """Compute pre-game ratings for every row of ``games``.

    ``games`` needs: ``game_id, game_date, season, home_team_id, away_team_id, home_margin,
    is_neutral``. Rows whose ``home_margin`` is NaN are unplayed fixtures: they receive the
    current ratings and update nothing. Played games are processed chronologically first.
    """
    n = len(games)
    played = games["home_margin"].notna().to_numpy()
    order = np.lexsort((games["game_id"].to_numpy().astype(str), games["game_date"].to_numpy(), ~played))
    home = games["home_team_id"].to_numpy()
    away = games["away_team_id"].to_numpy()
    margin = games["home_margin"].to_numpy(dtype=float)
    season = games["season"].to_numpy()
    neutral = games["is_neutral"].to_numpy(dtype=bool)

    ratings: dict[int, float] = {}
    home_pre = np.empty(n)
    away_pre = np.empty(n)
    current_season: int | None = None
    mult_sum, mult_n = 0.0, 0
    mean, carry = params.mean_rating, params.season_carry

    for i in order:
        if current_season is None:
            current_season = int(season[i])
        elif season[i] > current_season:  # new season: pull everyone toward the mean
            ratings = {t: mean + carry * (r - mean) for t, r in ratings.items()}
            current_season = int(season[i])
        h, a = int(home[i]), int(away[i])
        rh, ra = ratings.get(h, mean), ratings.get(a, mean)
        home_pre[i], away_pre[i] = rh, ra
        if not played[i]:
            continue
        edge = rh - ra + (0.0 if neutral[i] else params.home_advantage)
        expected_home = 1.0 / (1.0 + 10.0 ** (-edge / ELO_SCALE))
        home_won = margin[i] > 0
        mult = mov_multiplier(margin[i], edge if home_won else -edge) if params.use_mov else 1.0
        mult_sum += mult
        mult_n += 1
        delta = params.k * mult * ((1.0 if home_won else 0.0) - expected_home)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

    return EloResult(home_pre, away_pre, ratings, mult_sum / mult_n if mult_n else 1.0)


def tune_elo(games: pd.DataFrame, grid: dict, eval_mask: np.ndarray) -> tuple[EloParams, pd.DataFrame]:
    """Grid-search Elo parameters by the log loss of the pure Elo forecast.

    ``eval_mask`` selects the rows scored (development-season regular-season games); the
    ratings themselves are still built from every played game. Returns the best parameters
    and the full result table.
    """
    played = games["home_margin"].notna().to_numpy()
    score_rows = eval_mask & played
    if not score_rows.any():
        raise ValueError("tune_elo: no played games selected for evaluation")
    y = (games["home_margin"].to_numpy(dtype=float)[score_rows] > 0).astype(float)
    neutral = games["is_neutral"].to_numpy(dtype=bool)[score_rows]
    rows = []
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = EloParams(**dict(zip(keys, values)))
        result = run_elo(games, params)
        p = home_win_probability(result.home_pre[score_rows], result.away_pre[score_rows], params.home_advantage, neutral)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        rows.append({**params.to_dict(), "log_loss": log_loss, "accuracy": float(np.mean((p > 0.5) == (y == 1)))})
    table = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    best = table.iloc[0]
    best_params = EloParams(k=float(best["k"]), home_advantage=float(best["home_advantage"]),
                            season_carry=float(best["season_carry"]), use_mov=bool(best["use_mov"]),
                            mean_rating=float(best["mean_rating"]))
    return best_params, table
