"""Roster-availability proxy (optional feature group ``roster``).

The models have no injury report. What *is* observable before a game is who actually played
in the team's recent games. This module turns player game logs into one pre-game number per
team:

    roster_avail = share of the team's current rotation (top-N players by recent minutes,
                   minutes-weighted) that appeared in its last few games

A value near 1 means the usual rotation has been playing; a low value means important players
have recently been missing. It is a *proxy*: it cannot know that a player returns tonight or
was ruled out this morning, and it never names a player.

Leakage: the value attached to a game uses only that team's earlier games of the same season.
The feature is enabled with ``python -m nba_predictor fetch --players`` followed by
``python -m nba_predictor train --roster``; the ablation in ``train_model`` then decides
whether the group is kept.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from . import config
from .config import FeatureConfig
from .data_collection import _with_retries, current_season

logger = logging.getLogger(__name__)

PLAYER_LOG_COLUMNS: list[str] = ["season", "PLAYER_ID", "TEAM_ID", "GAME_ID", "GAME_DATE", "MIN"]


def fetch_player_logs(season: int, timeout: int = 90) -> pd.DataFrame:
    """Download every player's regular-season game log for one season (minutes only are kept)."""
    from nba_api.stats.endpoints import playergamelogs

    def _call() -> pd.DataFrame:
        return playergamelogs.PlayerGameLogs(
            season_nullable=config.season_label(season), season_type_nullable="Regular Season",
            league_id_nullable="00", timeout=timeout,
        ).get_data_frames()[0]

    raw = _with_retries(f"PlayerGameLogs {config.season_label(season)}", _call)
    frame = pd.DataFrame(
        {
            "season": season,
            "PLAYER_ID": raw["PLAYER_ID"].astype("int64"),
            "TEAM_ID": raw["TEAM_ID"].astype("int64"),
            "GAME_ID": raw["GAME_ID"].astype(str).str.zfill(10),
            "GAME_DATE": raw["GAME_DATE"].astype(str).str[:10],
            "MIN": pd.to_numeric(raw["MIN"], errors="coerce").fillna(0.0).round(2),
        }
    )
    return frame[PLAYER_LOG_COLUMNS]


def refresh_player_logs(first_season: int = config.FIRST_SEASON, last_season: int | None = None,
                        full: bool = False, pause: float = 1.0) -> pd.DataFrame:
    """Update ``player_logs.csv``: missing seasons plus the two most recent ones are downloaded."""
    last_season = current_season() if last_season is None else last_season
    existing = None if full else load_player_logs()
    have = set() if existing is None else set(existing["season"].unique())
    to_fetch = [s for s in range(first_season, last_season + 1) if s not in have or s >= last_season - 1]
    frames = [] if existing is None else [existing[~existing["season"].isin(to_fetch)]]
    for season in to_fetch:
        logger.info("Downloading player logs %s", config.season_label(season))
        frames.append(fetch_player_logs(season))
        time.sleep(pause)
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(["GAME_ID", "PLAYER_ID"])
    combined = combined.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID", "PLAYER_ID"])
    config.PLAYER_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(config.PLAYER_LOGS_PATH, index=False)
    logger.info("Saved %d player-game rows to %s", len(combined), config.PLAYER_LOGS_PATH)
    return combined


def load_player_logs(path=None) -> pd.DataFrame | None:
    """Read ``player_logs.csv`` (``None`` when it has not been downloaded)."""
    path = config.PLAYER_LOGS_PATH if path is None else path
    if not path.exists():
        return None
    logs = pd.read_csv(path, dtype={"GAME_ID": str})
    missing = [c for c in PLAYER_LOG_COLUMNS if c not in logs.columns]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    logs["GAME_ID"] = logs["GAME_ID"].str.zfill(10)
    return logs


def _team_season_availability(minutes: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Availability before each game and for the next unplayed game.

    ``minutes`` has shape (games, players) in chronological order. Returns ``games + 1`` values:
    entry ``g`` uses only rows ``< g``; the last entry is the state after every played game.
    """
    n_games, n_players = minutes.shape
    alpha = 1.0 - 0.5 ** (1.0 / cfg.roster_halflife)
    weights = np.zeros(n_players)  # EWMA of minutes per team game = each player's rotation weight
    played = minutes > 0
    out = np.full(n_games + 1, np.nan)
    for g in range(n_games + 1):
        if g > 0:
            top = np.argsort(-weights)[: cfg.roster_top_n]
            total = weights[top].sum()
            if total > 0:
                recent = played[max(0, g - cfg.roster_recent_games):g].mean(axis=0)
                out[g] = float((weights[top] * recent[top]).sum() / total)
        if g < n_games:
            weights = (1.0 - alpha) * weights + alpha * minutes[g]
    return out


def compute_roster_availability(player_logs: pd.DataFrame, games: pd.DataFrame,
                                cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Pre-game ``pre_roster_avail`` for every regular-season team game in ``games``.

    Returns columns ``team_id, game_date, season, pre_roster_avail, is_next``. Rows with
    ``is_next`` (one per team, ``game_date`` NaT) hold the value for that team's next unplayed
    game in its latest season. Only games present in ``games`` are used, so passing a truncated
    history yields values free of later information.
    """
    cfg = cfg or FeatureConfig()
    regular = games[games["game_type"] == "regular"][["game_id", "game_date", "season", "team_id"]]
    logs = player_logs.drop(columns="season", errors="ignore").merge(
        regular, left_on=["GAME_ID", "TEAM_ID"], right_on=["game_id", "team_id"], how="inner")
    latest_season = regular.groupby("team_id")["season"].max()
    rows: list[dict] = []
    for (team_id, season), part in logs.groupby(["team_id", "season"]):
        order = part[["game_id", "game_date"]].drop_duplicates().sort_values(["game_date", "game_id"])
        matrix = part.pivot_table(index="game_id", columns="PLAYER_ID", values="MIN", aggfunc="sum", fill_value=0.0)
        matrix = matrix.reindex(order["game_id"])
        values = _team_season_availability(matrix.to_numpy(dtype=float), cfg)
        for date, value in zip(order["game_date"], values[:-1]):
            rows.append({"team_id": int(team_id), "game_date": date, "season": int(season), "pre_roster_avail": value, "is_next": False})
        if int(season) == int(latest_season[team_id]):
            rows.append({"team_id": int(team_id), "game_date": pd.NaT, "season": int(season), "pre_roster_avail": values[-1], "is_next": True})
    out = pd.DataFrame(rows, columns=["team_id", "game_date", "season", "pre_roster_avail", "is_next"])
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["is_next"] = out["is_next"].astype(bool)
    return out


def load_team_extras(games: pd.DataFrame | None = None, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Roster availability from the files on disk, ready for ``build_features(team_extras=...)``."""
    from .data_collection import load_clean_games

    logs = load_player_logs()
    if logs is None:
        raise FileNotFoundError(f"{config.PLAYER_LOGS_PATH} not found - run `python -m nba_predictor fetch --players` first")
    games = load_clean_games() if games is None else games
    last_log, last_game = logs["GAME_DATE"].max(), str(games.loc[games["game_type"] == "regular", "game_date"].max().date())
    if last_log < last_game:
        logger.warning("Player logs end %s but games run to %s: roster availability is stale. "
                       "Run `python -m nba_predictor fetch --players`.", last_log, last_game)
    return compute_roster_availability(logs, games, cfg)
