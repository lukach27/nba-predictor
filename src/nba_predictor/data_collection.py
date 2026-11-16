"""Data access: download game logs and the schedule from the NBA stats API and clean them.

Two files are maintained under ``data/raw``:

* ``games_raw.csv``  - one row per team per game (``LeagueGameFinder``), same schema as the
  original project file.
* ``schedule.csv``   - the current season schedule (``ScheduleLeagueV2``), used for upcoming
  games, neutral-site flags and the season simulator.

Nothing here engineers features; :func:`clean_games` only validates, types and pairs rows.
"""

from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

RAW_COLUMNS: list[str] = [
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
    "MIN", "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
]
BOX_COLUMNS: list[str] = [
    "MIN", "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
]
SCHEDULE_COLUMNS: list[str] = [
    "game_id", "game_date", "season", "game_type", "game_status", "home_team_id", "away_team_id",
    "home_score", "away_score", "is_neutral", "label", "subtype", "arena_city", "game_time_utc",
]
FIXTURE_COLUMNS: list[str] = ["game_id", "game_date", "season", "game_type", "home_team_id", "away_team_id", "is_neutral"]


class DataCollectionError(RuntimeError):
    """Raised when the NBA API cannot be reached or returns unusable data."""


class DataValidationError(ValueError):
    """Raised when a data file does not have the expected schema."""


# ----------------------------------------------------------------------- utilities
def current_season(today: date | None = None) -> int:
    """Start year of the season in progress (new seasons are assumed to begin in September)."""
    today = today or date.today()
    return today.year if today.month >= 9 else today.year - 1


def _with_retries(label: str, func, retries: int = 4, pause: float = 2.0):
    """Call ``func`` with exponential backoff; the NBA API drops connections regularly."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as exc:  # network / JSON errors from nba_api are heterogeneous
            last_error = exc
            wait = pause * 2 ** (attempt - 1)
            logger.warning("%s failed (attempt %d/%d): %s - retrying in %.0fs", label, attempt, retries, exc, wait)
            time.sleep(wait)
    raise DataCollectionError(f"{label} failed after {retries} attempts: {last_error}") from last_error


# ------------------------------------------------------------------------ download
def fetch_season_games(season: int, timeout: int = 60) -> pd.DataFrame:
    """Download every NBA team-game row of one season (all game types)."""
    from nba_api.stats.endpoints import leaguegamefinder

    def _call() -> pd.DataFrame:
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=config.season_label(season), league_id_nullable="00", timeout=timeout
        )
        return finder.get_data_frames()[0]

    frame = _with_retries(f"LeagueGameFinder {config.season_label(season)}", _call)
    missing = [c for c in RAW_COLUMNS if c not in frame.columns]
    if missing:
        raise DataCollectionError(f"LeagueGameFinder response lacks columns {missing}")
    return frame[RAW_COLUMNS].copy()


def fetch_schedule(season: int, timeout: int = 60) -> pd.DataFrame:
    """Download and normalise the league schedule of one season."""
    from nba_api.stats.endpoints import scheduleleaguev2

    def _call() -> pd.DataFrame:
        return scheduleleaguev2.ScheduleLeagueV2(
            season=config.season_label(season), league_id="00", timeout=timeout
        ).get_data_frames()[0]

    raw = _with_retries(f"ScheduleLeagueV2 {config.season_label(season)}", _call)
    game_id = raw["gameId"].astype(str).str.zfill(10)
    schedule = pd.DataFrame(
        {
            "game_id": game_id,
            "game_date": pd.to_datetime(raw["gameDateEst"].astype(str).str[:10]),
            "season": season,
            "game_type": game_id.str[:3].map(config.GAME_TYPE_BY_PREFIX),
            "game_status": pd.to_numeric(raw["gameStatus"], errors="coerce").fillna(1).astype(int),
            "home_team_id": pd.to_numeric(raw["homeTeam_teamId"], errors="coerce").fillna(0).astype("int64"),
            "away_team_id": pd.to_numeric(raw["awayTeam_teamId"], errors="coerce").fillna(0).astype("int64"),
            "home_score": pd.to_numeric(raw["homeTeam_score"], errors="coerce"),
            "away_score": pd.to_numeric(raw["awayTeam_score"], errors="coerce"),
            "is_neutral": raw["isNeutral"].fillna(False).astype(bool),
            "label": raw["gameLabel"].fillna("").astype(str),
            "subtype": raw["gameSubtype"].fillna("").astype(str),
            "arena_city": raw["arenaCity"].fillna("").astype(str),
            "game_time_utc": raw["gameDateTimeUTC"].fillna("").astype(str),
        }
    )
    return schedule[SCHEDULE_COLUMNS].sort_values(["game_date", "game_id"]).reset_index(drop=True)


def refresh_raw_games(first_season: int = config.FIRST_SEASON, last_season: int | None = None,
                      full: bool = False, pause: float = 1.0) -> pd.DataFrame:
    """Update ``games_raw.csv``.

    Seasons already on disk are kept; missing seasons and the two most recent ones (which can
    still change) are downloaded again. ``full=True`` re-downloads everything.
    """
    last_season = current_season() if last_season is None else last_season
    existing = None if full or not config.RAW_GAMES_PATH.exists() else load_raw_games()
    wanted = list(range(first_season, last_season + 1))
    have = set() if existing is None else set(_season_of(existing["SEASON_ID"]).unique())
    to_fetch = [s for s in wanted if s not in have or s >= last_season - 1]

    frames: list[pd.DataFrame] = []
    if existing is not None:
        keep = _season_of(existing["SEASON_ID"]).isin([s for s in wanted if s not in to_fetch])
        frames.append(existing[keep])
    for season in to_fetch:
        logger.info("Downloading season %s", config.season_label(season))
        frames.append(fetch_season_games(season))
        time.sleep(pause)

    combined = pd.concat(frames, ignore_index=True)
    combined["GAME_ID"] = combined["GAME_ID"].astype(str).str.zfill(10)
    combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    combined = combined.drop_duplicates(["GAME_ID", "TEAM_ID"]).sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"])
    config.RAW_GAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(config.RAW_GAMES_PATH, index=False)
    logger.info("Saved %d rows to %s", len(combined), config.RAW_GAMES_PATH)
    return combined


def refresh_schedule(season: int | None = None) -> pd.DataFrame:
    """Download the schedule of ``season`` (default: the current one) and save it."""
    schedule = fetch_schedule(current_season() if season is None else season)
    config.SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(config.SCHEDULE_PATH, index=False)
    logger.info("Saved %d schedule rows to %s", len(schedule), config.SCHEDULE_PATH)
    return schedule


# --------------------------------------------------------------------------- load
def _season_of(season_id: pd.Series) -> pd.Series:
    """``22019`` -> ``2019`` (the last four digits of SEASON_ID are the start year)."""
    return pd.to_numeric(season_id, errors="coerce").astype("int64") % 10000


def load_raw_games(path=None) -> pd.DataFrame:
    """Read ``games_raw.csv`` with stable dtypes and validate its schema."""
    path = config.RAW_GAMES_PATH if path is None else path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m nba_predictor fetch` first")
    raw = pd.read_csv(path, dtype={"GAME_ID": str})
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise DataValidationError(f"{path} lacks columns {missing}")
    raw["GAME_ID"] = raw["GAME_ID"].str.zfill(10)
    return raw


def load_schedule(path=None) -> pd.DataFrame | None:
    """Read ``schedule.csv``; returns ``None`` when it has not been downloaded yet."""
    path = config.SCHEDULE_PATH if path is None else path
    if not path.exists():
        return None
    schedule = pd.read_csv(path, dtype={"game_id": str, "label": str, "subtype": str, "arena_city": str,
                                        "game_time_utc": str})
    missing = [c for c in SCHEDULE_COLUMNS if c not in schedule.columns]
    if missing:
        raise DataValidationError(f"{path} lacks columns {missing}")
    schedule["game_id"] = schedule["game_id"].str.zfill(10)
    schedule["game_date"] = pd.to_datetime(schedule["game_date"])
    schedule["is_neutral"] = schedule["is_neutral"].astype(bool)
    for col in ("label", "subtype", "arena_city", "game_time_utc"):
        schedule[col] = schedule[col].fillna("")
    return schedule


# -------------------------------------------------------------------------- clean
def clean_games(raw: pd.DataFrame, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    """Validate raw rows and return one typed row per team per completed competitive game.

    Each output row carries the team's box score, the opponent's box score (``OPP_`` prefix),
    the venue flag and a neutral-site flag. Rows are dropped (and counted in the log) when the
    game is not competitive, involves a non-NBA team, is unfinished, or is not a clean
    home/away pair.
    """
    df = raw.copy()
    df["game_id"] = df["GAME_ID"].astype(str).str.zfill(10)
    df["game_date"] = pd.to_datetime(df["GAME_DATE"])
    df["season"] = _season_of(df["SEASON_ID"])
    df["game_type"] = df["game_id"].str[:3].map(config.GAME_TYPE_BY_PREFIX)
    lo, hi = config.NBA_TEAM_ID_RANGE
    n_start = len(df)
    df = df[df["game_type"].isin(config.COMPETITIVE_GAME_TYPES) & df["TEAM_ID"].between(lo, hi) & df["WL"].isin(["W", "L"])]
    df = df.drop_duplicates(["game_id", "TEAM_ID"])
    matchup = df["MATCHUP"].astype(str)
    df["is_home"] = matchup.str.contains("vs.", regex=False)
    # Since 2024-25 the API labels neutral-site games "A @ B" on BOTH rows (NBA Cup in Las
    # Vegas, Mexico City, Paris). The team named second is the designated home team.
    both_away = df.groupby("game_id")["is_home"].transform("sum") == 0
    designated_home = matchup.str.split("@").str[-1].str.strip()
    df.loc[both_away, "is_home"] = (df["TEAM_ABBREVIATION"] == designated_home)[both_away]
    api_neutral_ids = set(df.loc[both_away, "game_id"])

    per_game = df.groupby("game_id")["is_home"].agg(["size", "sum"])
    valid_ids = per_game.index[(per_game["size"] == 2) & (per_game["sum"] == 1)]
    df = df[df["game_id"].isin(valid_ids)].copy()
    logger.info("clean_games: kept %d of %d raw rows (%d games)", len(df), n_start, len(valid_ids))

    for col in BOX_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"TEAM_ID": "team_id", "TEAM_ABBREVIATION": "team_abbr", "TEAM_NAME": "team_name"})
    df["win"] = (df["WL"] == "W").astype(float)

    opp = df[["game_id", "team_id", "team_abbr"] + BOX_COLUMNS].rename(
        columns={"team_id": "opp_team_id", "team_abbr": "opp_abbr", **{c: f"OPP_{c}" for c in BOX_COLUMNS}}
    )
    df = df.merge(opp, on="game_id")
    df = df[df["team_id"] != df["opp_team_id"]]

    bubble = df["game_date"].between(*map(pd.Timestamp, config.BUBBLE_DATES))
    neutral_ids = set() if schedule is None else set(schedule.loc[schedule["is_neutral"], "game_id"])
    df["is_neutral"] = bubble | df["game_id"].isin(neutral_ids | api_neutral_ids)

    keep = ["game_id", "game_date", "season", "game_type", "team_id", "team_abbr", "team_name",
            "opp_team_id", "opp_abbr", "is_home", "is_neutral", "win"]
    out = df[keep + BOX_COLUMNS + [f"OPP_{c}" for c in BOX_COLUMNS]]
    return out.sort_values(["game_date", "game_id", "is_home"]).reset_index(drop=True)


def load_clean_games() -> pd.DataFrame:
    """Convenience: load raw games and the schedule from disk and clean them."""
    return clean_games(load_raw_games(), load_schedule())


def upcoming_fixtures(schedule: pd.DataFrame | None, games: pd.DataFrame,
                      game_types: tuple[str, ...] = ("regular",)) -> pd.DataFrame:
    """Scheduled games that have not been played yet and whose two teams are known."""
    if schedule is None or schedule.empty:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    lo, hi = config.NBA_TEAM_ID_RANGE
    played_ids = set(games["game_id"])
    mask = (
        schedule["game_type"].isin(game_types)
        & (schedule["game_status"] != 3)
        & ~schedule["game_id"].isin(played_ids)
        & schedule["home_team_id"].between(lo, hi)
        & schedule["away_team_id"].between(lo, hi)
    )
    last_played = games["game_date"].max() if len(games) else pd.Timestamp.min
    fixtures = schedule.loc[mask & (schedule["game_date"] >= last_played), FIXTURE_COLUMNS]
    return fixtures.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def team_directory(games: pd.DataFrame) -> pd.DataFrame:
    """Current abbreviation, name and conference of every team (most recent row wins)."""
    latest = games.sort_values("game_date").groupby("team_id").tail(1)
    teams = latest[["team_id", "team_abbr", "team_name"]].copy()
    teams["conference"] = teams["team_abbr"].map(config.CONFERENCE_BY_ABBR)
    unknown = teams.loc[teams["conference"].isna(), "team_abbr"].tolist()
    if unknown:
        raise DataValidationError(f"No conference configured for teams {unknown}")
    return teams.sort_values("team_abbr").reset_index(drop=True)


def data_status(games: pd.DataFrame, schedule: pd.DataFrame | None) -> dict:
    """Small summary used by the CLI and the dashboard to show data freshness."""
    regular = games[games["game_type"] == "regular"]
    status = {
        "games": int(games["game_id"].nunique()),
        "regular_season_games": int(regular["game_id"].nunique()),
        "first_date": games["game_date"].min().date().isoformat(),
        "last_date": games["game_date"].max().date().isoformat(),
        "seasons": sorted(int(s) for s in games["season"].unique()),
        "schedule_rows": 0 if schedule is None else int(len(schedule)),
        "upcoming_fixtures": int(len(upcoming_fixtures(schedule, games))),
    }
    return status


def update_data(full: bool = False) -> dict:
    """Refresh games and schedule from the NBA API and return a status summary."""
    refresh_raw_games(full=full)
    refresh_schedule()
    games = load_clean_games()
    return data_status(games, load_schedule())


__all__ = [
    "BOX_COLUMNS", "DataCollectionError", "DataValidationError", "RAW_COLUMNS", "clean_games",
    "current_season", "data_status", "fetch_schedule", "fetch_season_games", "load_clean_games",
    "load_raw_games", "load_schedule", "refresh_raw_games", "refresh_schedule", "team_directory",
    "update_data", "upcoming_fixtures",
]
