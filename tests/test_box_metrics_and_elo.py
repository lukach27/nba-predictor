"""Four Factors, possession-based ratings and the Elo engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_predictor import box_metrics as bm
from nba_predictor.elo import EloParams, expected_score, home_win_probability, mov_multiplier, run_elo, tune_elo

# ------------------------------------------------------------------ Four Factors
def test_effective_fg_pct_credits_threes():
    assert bm.effective_fg_pct(40, 10, 80)[()] == pytest.approx((40 + 0.5 * 10) / 80)


def test_turnover_pct_formula():
    assert bm.turnover_pct(12, 85, 20)[()] == pytest.approx(12 / (85 + 0.44 * 20 + 12))


def test_rebound_pct_and_free_throw_rate():
    assert bm.rebound_pct(10, 30)[()] == pytest.approx(0.25)
    assert bm.free_throw_rate(18, 90)[()] == pytest.approx(0.2)


def test_zero_denominator_gives_nan_not_inf():
    assert np.isnan(bm.effective_fg_pct(0, 0, 0)[()])
    assert np.isnan(bm.rebound_pct(0, 0)[()])


def test_possessions_and_ratings_on_a_known_box_score():
    row = {"MIN": 240, "PTS": 110, "FGM": 40, "FGA": 85, "FG3M": 12, "FG3A": 30, "FTM": 18, "FTA": 22, "OREB": 10,
           "DREB": 33, "REB": 43, "AST": 24, "STL": 7, "BLK": 4, "TOV": 13, "PF": 20}
    opp = {"MIN": 240, "PTS": 100, "FGM": 37, "FGA": 88, "FG3M": 9, "FG3A": 31, "FTM": 17, "FTA": 20, "OREB": 9,
           "DREB": 31, "REB": 40, "AST": 20, "STL": 6, "BLK": 3, "TOV": 15, "PF": 21}
    frame = pd.DataFrame([{**row, **{f"OPP_{k}": v for k, v in opp.items()}}])
    out = bm.add_game_metrics(frame).iloc[0]
    own = 85 + 0.44 * 22 - 10 + 13
    other = 88 + 0.44 * 20 - 9 + 15
    poss = 0.5 * (own + other)
    assert out["poss"] == pytest.approx(poss)
    assert out["off_rtg"] == pytest.approx(100 * 110 / poss)
    assert out["def_rtg"] == pytest.approx(100 * 100 / poss)
    assert out["net_rtg"] == pytest.approx(out["off_rtg"] - out["def_rtg"])
    assert out["pace"] == pytest.approx(poss)  # regulation: 48 minutes
    assert out["orb_pct"] == pytest.approx(10 / (10 + 31))
    assert out["drb_pct"] == pytest.approx(33 / (33 + 9))
    assert out["opp_efg"] == pytest.approx((37 + 4.5) / 88)


def test_pace_accounts_for_overtime():
    assert bm.game_minutes(265, 265)[()] == 53.0  # one overtime
    assert bm.game_minutes(241, 239)[()] == 48.0  # rounding noise in MIN is ignored


def test_metrics_are_nan_for_rows_without_a_box_score():
    cols = ["MIN", "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF"]
    frame = pd.DataFrame([{**{c: np.nan for c in cols}, **{f"OPP_{c}": np.nan for c in cols}}])
    out = bm.add_game_metrics(frame)
    assert out[bm.GAME_METRICS].isna().all().all()


# --------------------------------------------------------------------------- Elo
def _games(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["game_id", "game_date", "season", "home_team_id", "away_team_id", "home_margin", "is_neutral"]).assign(
        game_date=lambda d: pd.to_datetime(d["game_date"]))


def test_expected_score_values():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)
    assert expected_score(1600, 1500) + expected_score(1500, 1600) == pytest.approx(1.0)


def test_single_update_by_hand_without_extras():
    params = EloParams(k=20, home_advantage=0, season_carry=1.0, use_mov=False)
    result = run_elo(_games([("g1", "2024-01-01", 2023, 1, 2, 5.0, False)]), params)
    assert result.home_pre[0] == 1500 and result.away_pre[0] == 1500  # pre-game: result not included
    assert result.ratings[1] == pytest.approx(1510) and result.ratings[2] == pytest.approx(1490)


def test_updates_are_zero_sum_and_pre_game_ratings_lag_results():
    games = _games([("g1", "2024-01-01", 2023, 1, 2, 8.0, False), ("g2", "2024-01-03", 2023, 2, 1, -3.0, False)])
    result = run_elo(games, EloParams())
    assert sum(result.ratings.values()) == pytest.approx(3000)
    assert result.away_pre[1] > 1500  # team 1 won game 1; that shows only from game 2 onwards
    assert result.home_pre[0] == 1500


def test_home_advantage_makes_a_home_win_less_surprising():
    home_win = run_elo(_games([("g", "2024-01-01", 2023, 1, 2, 4.0, False)]), EloParams(use_mov=False))
    away_win = run_elo(_games([("g", "2024-01-01", 2023, 1, 2, -4.0, False)]), EloParams(use_mov=False))
    assert (home_win.ratings[1] - 1500) < (away_win.ratings[2] - 1500)


def test_neutral_site_removes_home_advantage():
    params = EloParams(use_mov=False)
    neutral = run_elo(_games([("g", "2024-01-01", 2023, 1, 2, 4.0, True)]), params)
    assert neutral.ratings[1] - 1500 == pytest.approx(params.k * 0.5)
    assert home_win_probability(1500, 1500, 60.0, True) == pytest.approx(0.5)
    assert home_win_probability(1500, 1500, 60.0, False) > 0.5


def test_margin_of_victory_scales_the_update():
    close = run_elo(_games([("g", "2024-01-01", 2023, 1, 2, 1.0, False)]), EloParams())
    blowout = run_elo(_games([("g", "2024-01-01", 2023, 1, 2, 30.0, False)]), EloParams())
    assert blowout.ratings[1] > close.ratings[1]
    assert mov_multiplier(10, 0) > mov_multiplier(10, 200)  # favourites get less credit


def test_season_regression_pulls_ratings_toward_the_mean():
    games = _games([("g1", "2024-01-01", 2023, 1, 2, 10.0, False), ("g2", "2024-11-01", 2024, 1, 3, 2.0, False)])
    params = EloParams(season_carry=0.5)
    result = run_elo(games, params)
    end_of_season = run_elo(games.iloc[:1], params).ratings[1]
    assert result.home_pre[1] == pytest.approx(1500 + 0.5 * (end_of_season - 1500))


def test_unplayed_fixture_gets_current_ratings_and_changes_nothing():
    played = _games([("g1", "2024-01-01", 2023, 1, 2, 10.0, False)])
    with_fixture = pd.concat([played, _games([("f1", "2024-01-05", 2023, 2, 1, np.nan, False)])], ignore_index=True)
    a, b = run_elo(played, EloParams()), run_elo(with_fixture, EloParams())
    assert b.ratings == a.ratings
    assert b.home_pre[1] == pytest.approx(a.ratings[2]) and b.away_pre[1] == pytest.approx(a.ratings[1])


def test_tune_elo_returns_the_lowest_log_loss(games):
    from nba_predictor.features import games_wide

    wide = games_wide(games)
    grid = {"k": (10.0, 20.0), "home_advantage": (0.0, 60.0), "season_carry": (0.75,), "use_mov": (True,)}
    best, table = tune_elo(wide, grid, np.ones(len(wide), dtype=bool))
    assert len(table) == 4 and table["log_loss"].is_monotonic_increasing
    assert best.k == table.iloc[0]["k"] and best.home_advantage == table.iloc[0]["home_advantage"]
    assert best.home_advantage == 60.0  # the synthetic league has a real home edge
