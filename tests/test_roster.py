"""Roster-availability proxy: formula, temporal shift, leakage and fixture parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_predictor.config import FeatureConfig
from nba_predictor.features import build_features, hypothetical_fixtures
from nba_predictor.roster import _team_season_availability, compute_roster_availability

CFG = FeatureConfig(roster_top_n=2, roster_recent_games=1, roster_halflife=1.0)  # alpha = 0.5


def test_availability_by_hand():
    out = _team_season_availability(np.array([[30.0, 30.0], [30.0, 0.0]]), CFG)
    assert len(out) == 3
    assert np.isnan(out[0])                 # nothing is known before the first game
    assert out[1] == pytest.approx(1.0)     # both rotation players appeared in game 0
    # after game 1 the weights are A = 22.5, B = 7.5 and only A played the most recent game
    assert out[2] == pytest.approx(22.5 / (22.5 + 7.5))


def test_rotation_follows_recent_minutes():
    # A and B start three games, then C takes over B's minutes for good (rotation size 2)
    slow = FeatureConfig(roster_top_n=2, roster_recent_games=1, roster_halflife=5.0)
    minutes = np.array([[30.0, 30.0, 0.0]] * 3 + [[30.0, 0.0, 30.0]] * 4)
    out = _team_season_availability(minutes, slow)
    assert out[3] == pytest.approx(1.0)     # before B's first absence the rotation is complete
    assert out[4] < 0.7                     # right after it, B is still "rotation" and did not play
    assert out[-1] == pytest.approx(1.0)    # a few games later C has replaced B in the rotation


def _player_logs(games: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Synthetic logs: eight players per team, each missing roughly one game in six."""
    rng = np.random.default_rng(seed)
    rows = []
    for row in games.itertuples():
        for player in range(8):
            played = rng.random() > 0.17
            rows.append({"season": row.season, "PLAYER_ID": int(row.team_id) * 10 + player, "TEAM_ID": int(row.team_id),
                         "GAME_ID": row.game_id, "GAME_DATE": row.game_date.strftime("%Y-%m-%d"),
                         "MIN": float(36 - 3 * player) if played else 0.0})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def logs(games):
    return _player_logs(games)


def test_values_are_bounded_and_season_openers_are_unknown(games, logs):
    extras = compute_roster_availability(logs, games)
    played = extras[~extras["is_next"]]
    assert len(played) == len(games)
    assert played["pre_roster_avail"].dropna().between(0, 1).all()
    openers = played.sort_values("game_date").groupby(["team_id", "season"]).head(1)
    assert openers["pre_roster_avail"].isna().all()   # shift: a season opener has no earlier game
    assert extras["is_next"].sum() == games["team_id"].nunique()
    assert (extras.loc[extras["is_next"], "season"] == games["season"].max()).all()


def test_later_games_do_not_change_earlier_values(games, logs):
    """Leakage check: wiping every minute of the last season leaves earlier seasons untouched."""
    last = games["season"].max()
    altered_logs = logs.copy()
    altered_logs.loc[altered_logs["season"] == last, "MIN"] = 0.0
    base = compute_roster_availability(logs, games)
    altered = compute_roster_availability(altered_logs, games)
    merged = base[~base["is_next"]].merge(altered[~altered["is_next"]], on=["team_id", "game_date"], suffixes=("", "_alt"))
    earlier = merged["season"] < last
    pd.testing.assert_series_equal(merged.loc[earlier, "pre_roster_avail"], merged.loc[earlier, "pre_roster_avail_alt"],
                                   check_names=False)
    assert not merged.loc[~earlier, "pre_roster_avail"].equals(merged.loc[~earlier, "pre_roster_avail_alt"])


def test_next_game_state_equals_the_value_the_game_later_receives(games, logs):
    """Parity: the forecast-time value (from truncated history) is the training-time value."""
    full = compute_roster_availability(logs, games)
    game = games[games["is_home"]].sort_values(["game_date", "game_id"]).iloc[-20]
    history = games[games["game_date"] < game["game_date"]]
    truncated = compute_roster_availability(logs, history)
    for team in (game["team_id"], game["opp_team_id"]):
        expected = full[(full["team_id"] == team) & (full["game_date"] == game["game_date"])]["pre_roster_avail"].iloc[0]
        actual = truncated[(truncated["team_id"] == team) & truncated["is_next"]]["pre_roster_avail"].iloc[0]
        assert actual == pytest.approx(expected)


def test_feature_pipeline_uses_roster_values_for_played_games_and_fixtures(games, logs):
    extras = compute_roster_availability(logs, games)
    ids = sorted(games["team_id"].unique())
    fixtures = hypothetical_fixtures([(ids[0], ids[1])], games["game_date"].max() + pd.Timedelta(days=1), int(games["season"].max()))
    matchups = build_features(games, fixtures, team_extras=extras).matchups
    upcoming = extras[extras["is_next"]].set_index("team_id")["pre_roster_avail"]
    pending = matchups[~matchups["is_played"]].iloc[0]
    assert pending["roster_avail_home"] == pytest.approx(upcoming[ids[0]])
    assert pending["roster_avail_away"] == pytest.approx(upcoming[ids[1]])
    assert pending["roster_avail_diff"] == pytest.approx(upcoming[ids[0]] - upcoming[ids[1]])
    played = matchups[matchups["is_played"] & (matchups["season"] == games["season"].max())].iloc[-1]
    recorded = extras[(extras["team_id"] == played["home_team_id"]) & (extras["game_date"] == played["game_date"])]
    assert played["roster_avail_home"] == pytest.approx(recorded["pre_roster_avail"].iloc[0])
    assert "roster_avail_diff" not in build_features(games).matchups.columns  # absent without player logs
