"""Shared fixtures: a small synthetic league so tests are fast, offline and deterministic.

The synthetic box scores are generated for tests only - they are never used by the product.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_predictor import config
from nba_predictor.config import TrainingConfig
from nba_predictor.data_collection import RAW_COLUMNS, SCHEDULE_COLUMNS, clean_games
from nba_predictor.features import build_features

TEAMS: list[tuple[int, str]] = [
    (1610612737, "ATL"), (1610612738, "BOS"), (1610612751, "BKN"), (1610612766, "CHA"), (1610612741, "CHI"),
    (1610612739, "CLE"), (1610612742, "DAL"), (1610612743, "DEN"), (1610612744, "GSW"), (1610612745, "HOU"),
    (1610612746, "LAC"), (1610612747, "LAL"),
]
SEASONS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022)


def _box(rng: np.random.Generator, edge: float) -> dict:
    """One team's box score; ``edge`` is its quality advantage in this game."""
    fga = int(rng.normal(88, 4))
    fg3a = int(np.clip(rng.normal(34, 4), 15, fga - 20))
    fgm = int(np.clip(fga * (0.46 + 0.012 * edge + rng.normal(0, 0.035)), 20, fga))
    fg3m = int(np.clip(fg3a * (0.36 + rng.normal(0, 0.05)), 0, min(fgm, fg3a)))
    fta = int(np.clip(rng.normal(22, 5), 5, 45))
    ftm = int(round(fta * 0.78))
    oreb, dreb = int(np.clip(rng.normal(10, 3), 2, 20)), int(np.clip(rng.normal(34, 4), 20, 50))
    return {
        "MIN": 240, "FGM": fgm, "FGA": fga, "FG3M": fg3m, "FG3A": fg3a, "FTM": ftm, "FTA": fta,
        "OREB": oreb, "DREB": dreb, "REB": oreb + dreb, "AST": int(rng.normal(25, 4)), "STL": int(rng.normal(7, 2)),
        "BLK": int(rng.normal(5, 2)), "TOV": int(np.clip(rng.normal(13, 3), 4, 25)), "PF": int(rng.normal(20, 3)),
        "PTS": 2 * fgm + fg3m + ftm,
    }


def make_raw_games(seasons: tuple[int, ...] = SEASONS, seed: int = 0) -> pd.DataFrame:
    """Synthetic raw game log in the exact schema of ``data/raw/games_raw.csv``."""
    rng = np.random.default_rng(seed)
    strength = {tid: s for (tid, _), s in zip(TEAMS, rng.normal(0, 1.0, len(TEAMS)))}
    abbr = dict(TEAMS)
    rows = []
    for season in seasons:
        pairs = [(h, a) for h, _ in TEAMS for a, _ in TEAMS if h != a]
        rng.shuffle(pairs)
        day, game_no = pd.Timestamp(year=season, month=10, day=20), 0
        while pairs:
            busy: set[int] = set()
            today = []
            for pair in list(pairs):
                if pair[0] not in busy and pair[1] not in busy and len(today) < 4:
                    today.append(pair)
                    busy.update(pair)
                    pairs.remove(pair)
            for home, away in today:
                game_no += 1
                edge = strength[home] - strength[away]
                hb, ab = _box(rng, edge + 0.6), _box(rng, -edge)
                if hb["PTS"] == ab["PTS"]:
                    hb["FTM"] += 1
                    hb["PTS"] += 1
                game_id = f"002{season % 100:02d}{game_no:05d}"
                for team, opp, box, other, is_home in ((home, away, hb, ab, True), (away, home, ab, hb, False)):
                    rows.append(
                        {
                            "SEASON_ID": 20000 + season, "TEAM_ID": team, "TEAM_ABBREVIATION": abbr[team],
                            "TEAM_NAME": f"Team {abbr[team]}", "GAME_ID": game_id, "GAME_DATE": day.strftime("%Y-%m-%d"),
                            "MATCHUP": f"{abbr[team]} vs. {abbr[opp]}" if is_home else f"{abbr[team]} @ {abbr[opp]}",
                            "WL": "W" if box["PTS"] > other["PTS"] else "L", **box,
                            "FG_PCT": box["FGM"] / box["FGA"], "FG3_PCT": box["FG3M"] / max(box["FG3A"], 1),
                            "FT_PCT": box["FTM"] / max(box["FTA"], 1), "PLUS_MINUS": float(box["PTS"] - other["PTS"]),
                        }
                    )
            day += pd.Timedelta(days=int(rng.integers(1, 3)))
    return pd.DataFrame(rows)[RAW_COLUMNS]


def make_schedule(games: pd.DataFrame, pairs: list[tuple[int, int]], days_ahead: int = 1) -> pd.DataFrame:
    """A schedule frame with the given unplayed (home, away) pairs on consecutive days."""
    start = games["game_date"].max() + pd.Timedelta(days=days_ahead)
    season = int(games["season"].max())
    rows = []
    for i, (home, away) in enumerate(pairs):
        date = start + pd.Timedelta(days=i)
        rows.append(
            {
                "game_id": f"002{season % 100:02d}9{i:04d}", "game_date": date, "season": season, "game_type": "regular",
                "game_status": 1, "home_team_id": home, "away_team_id": away, "home_score": 0.0, "away_score": 0.0,
                "is_neutral": False, "label": "", "subtype": "", "arena_city": "",
                "game_time_utc": (date + pd.Timedelta(hours=23)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
    return pd.DataFrame(rows)[SCHEDULE_COLUMNS]


TINY_TRAINING = TrainingConfig(
    lr_c_grid=(0.1, 1.0),
    xgb_search_iterations=2,
    xgb_default_params={"n_estimators": 25, "max_depth": 2, "learning_rate": 0.1, "subsample": 1.0,
                        "colsample_bytree": 1.0, "min_child_weight": 1.0, "reg_lambda": 1.0, "gamma": 0.0},
    elo_grid={"k": (20.0,), "home_advantage": (60.0,), "season_carry": (0.75,), "use_mov": (True,)},
    bootstrap_replicas=2, n_bootstrap_ci=50, n_jobs=1,
)


@pytest.fixture(scope="session")
def raw_games() -> pd.DataFrame:
    return make_raw_games()


@pytest.fixture(scope="session")
def games(raw_games: pd.DataFrame) -> pd.DataFrame:
    return clean_games(raw_games)


@pytest.fixture(scope="session")
def feature_set(games: pd.DataFrame):
    return build_features(games)


@pytest.fixture(scope="session")
def training(games: pd.DataFrame, tmp_path_factory):
    """Full training protocol on the synthetic league with a tiny search budget."""
    from nba_predictor.train_model import train

    out = tmp_path_factory.mktemp("models")
    result = train(games, tcfg=TINY_TRAINING, test_seasons=(2022,), cv_first_validation_season=2020,
                   save=True, output_dir=out)
    result.output_dir = out  # type: ignore[attr-defined]
    return result


@pytest.fixture(scope="session")
def predictor(training, games: pd.DataFrame):
    from nba_predictor.predict import Predictor

    ids = [t for t, _ in TEAMS]
    schedule = make_schedule(games, [(ids[0], ids[1]), (ids[2], ids[3]), (ids[1], ids[6]), (ids[7], ids[0])])
    return Predictor(training.bundle, games, schedule)


@pytest.fixture(scope="session")
def prediction(predictor):
    return predictor.predict_matchup("BOS", "LAL")


def model_feature_columns() -> list[str]:
    return config.all_feature_columns()
