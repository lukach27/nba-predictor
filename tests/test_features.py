"""Feature pipeline: temporal shifts, leakage, training/inference parity, missing data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_predictor import config
from nba_predictor.config import FeatureConfig
from nba_predictor.data_collection import clean_games
from nba_predictor.features import (OUTCOME_COLUMNS, build_features, games_wide, hypothetical_fixtures, swap_sides,
                                    training_mask)

FEATURES = config.all_feature_columns()
#: league_home_win_pct is deliberately NaN until 200 league games exist in the trailing year,
#: which the small synthetic league never reaches; it has its own test below.
DENSE = [c for c in FEATURES if c != "league_home_win_pct"]


def test_every_configured_feature_exists_and_no_outcome_is_a_feature(feature_set):
    missing = [c for c in FEATURES if c not in feature_set.matchups.columns]
    assert not missing
    assert not set(FEATURES) & set(OUTCOME_COLUMNS)
    assert not [c for c in FEATURES if c.startswith(("home_pts", "away_pts")) or "margin" in c and c != "h2h_margin"]


def test_first_game_has_no_in_season_information(feature_set):
    """shift(1): the very first game in the data has no history at all; the second sees exactly one game."""
    rows = feature_set.team_rows
    team = rows[rows["team_id"] == rows["team_id"].iloc[0]].sort_values("game_date")
    first, second = team.iloc[0], team.iloc[1]
    assert np.isnan(first["pre_net_rtg"]) and np.isnan(first["pre_efg"])
    assert second["pre_net_rtg"] == pytest.approx(first["net_rtg"])
    assert second["pre_efg"] == pytest.approx(first["efg"])
    assert first["pre_games_played"] == 0 and second["pre_games_played"] == 1


def test_new_season_starts_from_the_previous_season_prior_only(feature_set):
    """Game 1 of a later season must equal the regressed previous-season value (no current-season data)."""
    rows = feature_set.team_rows
    cfg = FeatureConfig()
    team_id = rows["team_id"].iloc[0]
    last = rows[(rows["season"] == 2019) & (rows["game_type"] == "regular")]
    league_mean = last.groupby("team_id")["net_rtg"].mean().mean()
    team_mean = last[last["team_id"] == team_id]["net_rtg"].mean()
    opener = rows[(rows["team_id"] == team_id) & (rows["season"] == 2020)].sort_values("game_date").iloc[0]
    assert opener["pre_net_rtg"] == pytest.approx(league_mean + cfg.season_carry * (team_mean - league_mean))


def test_no_future_leakage_under_perturbation(games, feature_set):
    """Changing one game's result must not move the features of that game or of any earlier game."""
    wide = games_wide(games).sort_values(["game_date", "game_id"])
    target = wide.iloc[len(wide) // 2]
    altered = games.copy()
    hit = altered["game_id"] == target["game_id"]
    for col in ("PTS", "FGM", "FG3M", "FTM", "OREB", "TOV"):
        altered.loc[hit, col] = altered.loc[hit, col] + np.where(altered.loc[hit, "is_home"], 25, 0)
        altered.loc[hit, f"OPP_{col}"] = altered.loc[hit, f"OPP_{col}"] + np.where(altered.loc[hit, "is_home"], 0, 25)
    altered.loc[hit, "win"] = altered.loc[hit, "is_home"].astype(float)

    before = feature_set.matchups.set_index("game_id")
    after = build_features(altered).matchups.set_index("game_id").loc[before.index]
    on_or_before = before["game_date"] <= target["game_date"]
    pd.testing.assert_frame_equal(before.loc[on_or_before, FEATURES], after.loc[on_or_before, FEATURES])
    later_changed = (before.loc[~on_or_before, FEATURES] - after.loc[~on_or_before, FEATURES]).abs().sum().sum()
    assert later_changed > 0, "the perturbation should reach later games (otherwise the test proves nothing)"


def test_training_inference_parity(games, feature_set):
    """A played game's training features == features computed for it as an unplayed fixture."""
    wide = feature_set.matchups[feature_set.matchups["is_played"]].sort_values(["game_date", "game_id"])
    for position in (len(wide) - 1, len(wide) - 40, len(wide) // 2):
        game = wide.iloc[position]
        history = games[games["game_date"] < game["game_date"]]
        fixture = pd.DataFrame([{"game_id": game["game_id"], "game_date": game["game_date"], "season": game["season"],
                                 "game_type": "regular", "home_team_id": game["home_team_id"],
                                 "away_team_id": game["away_team_id"], "is_neutral": False}])
        as_fixture = build_features(history, fixture).matchups.set_index("game_id").loc[game["game_id"]]
        assert not as_fixture["is_played"]
        np.testing.assert_allclose(as_fixture[FEATURES].to_numpy(dtype=float), game[FEATURES].to_numpy(dtype=float),
                                   rtol=1e-9, atol=1e-9)


def test_fixtures_do_not_change_history(games, feature_set):
    ids = sorted(games["team_id"].unique())
    fixtures = hypothetical_fixtures([(ids[0], ids[1]), (ids[2], ids[3])], games["game_date"].max() + pd.Timedelta(days=2), 2022)
    with_fixtures = build_features(games, fixtures).matchups
    played = with_fixtures[with_fixtures["is_played"]].set_index("game_id")
    base = feature_set.matchups.set_index("game_id")
    pd.testing.assert_frame_equal(played.loc[base.index, FEATURES], base[FEATURES])
    pending = with_fixtures[~with_fixtures["is_played"]]
    assert len(pending) == 2 and pending[DENSE].notna().all().all()
    assert pending[OUTCOME_COLUMNS].isna().all().all()


def _mini_league() -> pd.DataFrame:
    """Two teams, hand-written results, to check rest days and head-to-head by hand."""
    from nba_predictor.data_collection import RAW_COLUMNS

    a, b = 1610612737, 1610612738
    schedule = [("0022300001", "2023-11-01", a, b, 110, 100), ("0022300002", "2023-11-02", b, a, 95, 105),
                ("0022300003", "2023-11-06", a, b, 99, 104)]
    rows = []
    for gid, date, home, away, hp, ap in schedule:
        for team, pts, opp_pts, is_home in ((home, hp, ap, True), (away, ap, hp, False)):
            me, other = ("ATL", "BOS") if team == a else ("BOS", "ATL")
            rows.append({"SEASON_ID": 22023, "TEAM_ID": team, "TEAM_ABBREVIATION": me, "TEAM_NAME": me, "GAME_ID": gid,
                         "GAME_DATE": date, "MATCHUP": f"{me} vs. {other}" if is_home else f"{me} @ {other}",
                         "WL": "W" if pts > opp_pts else "L", "MIN": 240, "PTS": pts, "FGM": 40, "FGA": 85, "FG_PCT": 0.47,
                         "FG3M": 10, "FG3A": 30, "FG3_PCT": 0.33, "FTM": pts - 90, "FTA": 30, "FT_PCT": 0.7, "OREB": 10,
                         "DREB": 30, "REB": 40, "AST": 20, "STL": 7, "BLK": 4, "TOV": 12, "PF": 20,
                         "PLUS_MINUS": float(pts - opp_pts)})
    return pd.DataFrame(rows)[RAW_COLUMNS]


def test_rest_days_and_back_to_back_by_hand():
    matchups = build_features(clean_games(_mini_league())).matchups.sort_values("game_date").reset_index(drop=True)
    cap = FeatureConfig().rest_cap_days
    assert matchups.loc[0, ["rest_days_home", "rest_days_away"]].tolist() == [cap, cap]  # season opener
    assert matchups.loc[1, ["rest_days_home", "rest_days_away", "b2b_home", "b2b_away"]].tolist() == [1, 1, 1, 1]
    assert matchups.loc[2, ["rest_days_home", "rest_days_away", "b2b_home", "rest_diff"]].tolist() == [4, 4, 0, 0]
    assert matchups.loc[1, "away_road_streak"] == 0 and matchups.loc[2, "away_road_streak"] == 0


def test_head_to_head_uses_past_meetings_from_the_home_teams_view():
    cfg = FeatureConfig()
    m = build_features(clean_games(_mini_league())).matchups.sort_values("game_date").reset_index(drop=True)
    assert m.loc[0, ["h2h_margin", "h2h_games"]].tolist() == [0, 0]  # no earlier meeting
    # game 2: BOS at home; ATL won game 1 by 10 at home -> from BOS's view the margin is -10
    assert m.loc[1, "h2h_margin"] == pytest.approx(-10 / (1 + cfg.h2h_shrink_games))
    # game 3: ATL at home; earlier margins from ATL's view are +10 and +10
    assert m.loc[2, "h2h_margin"] == pytest.approx(20 / (2 + cfg.h2h_shrink_games))
    assert m.loc[2, "h2h_games"] == 2


def test_missing_box_score_values_do_not_break_the_pipeline(games):
    damaged = games.copy()
    victims = damaged.index[damaged["season"] == 2021][:6]
    damaged.loc[victims, ["FTA", "OREB", "OPP_DREB"]] = np.nan
    matchups = build_features(damaged).matchups
    usable = matchups[matchups["season"] >= 2019]
    assert np.isfinite(usable[DENSE].to_numpy(dtype=float)).all()


def test_swap_sides_mirrors_the_matchup(feature_set):
    sample = feature_set.matchups[feature_set.matchups["season"] == 2022].head(20)
    mirrored = swap_sides(sample)
    np.testing.assert_allclose(mirrored["elo_diff"], -sample["elo_diff"])
    np.testing.assert_allclose(mirrored["net_rtg_diff"], -sample["net_rtg_diff"])
    np.testing.assert_allclose(mirrored["h2h_margin"], -sample["h2h_margin"])
    assert (mirrored["home_team_id"].to_numpy() == sample["away_team_id"].to_numpy()).all()
    pd.testing.assert_frame_equal(swap_sides(mirrored)[FEATURES], sample[FEATURES])


def test_training_mask_excludes_burn_in_neutral_and_unplayed(games):
    ids = sorted(games["team_id"].unique())
    fixtures = hypothetical_fixtures([(ids[0], ids[1])], games["game_date"].max() + pd.Timedelta(days=1), 2022)
    matchups = build_features(games, fixtures).matchups
    matchups.loc[matchups.index[5], "is_neutral"] = True
    mask = training_mask(matchups)
    assert not mask[~matchups["is_played"]].any()
    assert not mask[matchups["is_neutral"]].any()
    assert not mask[matchups["season"] < config.FIRST_SEASON + config.BURN_IN_SEASONS].any()


def test_invalid_fixtures_are_rejected(games):
    ids = sorted(games["team_id"].unique())
    late = games["game_date"].max() + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="same home and away"):
        build_features(games, hypothetical_fixtures([(ids[0], ids[0])], late, 2022))
    with pytest.raises(ValueError, match="before the last played game"):
        build_features(games, hypothetical_fixtures([(ids[0], ids[1])], games["game_date"].min(), 2022))


def test_league_home_win_pct_uses_only_earlier_days(games):
    cfg = FeatureConfig(league_hca_min_games=30)
    matchups = build_features(games, cfg=cfg).matchups.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    assert matchups["league_home_win_pct"].iloc[:30].isna().all()  # not enough history yet
    row = matchups.iloc[400]
    window = matchups[(matchups["game_date"] < row["game_date"])
                      & (matchups["game_date"] >= row["game_date"] - pd.Timedelta(days=cfg.league_hca_window_days))]
    assert row["league_home_win_pct"] == pytest.approx(window["home_win"].mean())
