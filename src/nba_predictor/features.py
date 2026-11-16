"""Leak-free, stateless feature engineering shared by training and inference.

The whole pipeline is one pure function, :func:`build_features`. It takes the cleaned game
log plus (optionally) a list of unplayed *fixtures* and returns pre-game features for every
game. Training rows and prediction rows therefore go through exactly the same code - there
is no second "inference" implementation that could drift.

How leakage is prevented
------------------------
* Every team-level statistic is lagged with ``shift(1)`` inside its (team, season) group
  before any averaging, so a game never contributes to its own features.
* Early-season values are blended with a prior built from the *previous* season only.
* Elo ratings and head-to-head history are read before the game updates them.
* The league-wide home-court trend uses a window that closes on the previous day.
* Fixtures carry no box score, so there is nothing post-game to leak.

``tests/test_features.py`` verifies this by perturbation: changing a game's result must not
change the features of that game or of any earlier one.

Naming
------
Team-level pre-game columns start with ``pre_``. In the matchup frame they appear as
``home_<name>`` / ``away_<name>``; model inputs are the columns listed in
``config.FEATURE_GROUPS``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .box_metrics import GAME_METRICS, add_game_metrics
from .config import FeatureConfig
from .data_collection import BOX_COLUMNS, FIXTURE_COLUMNS
from .elo import EloParams, EloResult, run_elo

logger = logging.getLogger(__name__)

#: Per-game metrics averaged into season-level team quality (``pre_<stat>``).
LONG_STATS: list[str] = [
    "off_rtg", "def_rtg", "net_rtg", "pace", "efg", "tov_pct", "orb_pct", "ft_rate",
    "opp_efg", "opp_tov_pct", "drb_pct", "opp_ft_rate", "fg3a_rate", "win",
]
SHORT_STATS: list[str] = ["net_rtg", "win"]
VENUE_STATS: list[str] = ["net_rtg", "win"]
#: Statistics differenced as home minus away to form model inputs.
DIFF_STATS: list[str] = [
    "net_rtg", "off_rtg", "win_pct", "efg", "tov_pct", "orb_pct", "ft_rate", "opp_efg", "opp_tov_pct",
    "drb_pct", "opp_ft_rate", "net_rtg_short", "win_pct_short", "elo_trend", "games_last_7d",
]
OUTCOME_COLUMNS: list[str] = ["home_pts", "away_pts", "home_margin", "home_win"]
ID_COLUMNS: list[str] = [
    "game_id", "game_date", "season", "game_type", "is_neutral", "is_played", "home_team_id", "away_team_id",
]


@dataclass
class FeatureSet:
    """Result of :func:`build_features`."""

    matchups: pd.DataFrame  # one row per game/fixture: ids, outcomes, home_/away_ values, model inputs
    team_rows: pd.DataFrame  # one row per team per game: observed metrics and ``pre_`` features
    elo: EloResult


# --------------------------------------------------------------------- frame setup
def games_wide(games: pd.DataFrame) -> pd.DataFrame:
    """One row per played game with home/away ids and the outcome."""
    home = games[games["is_home"]]
    wide = pd.DataFrame(
        {
            "game_id": home["game_id"].to_numpy(),
            "game_date": home["game_date"].to_numpy(),
            "season": home["season"].to_numpy(),
            "game_type": home["game_type"].to_numpy(),
            "is_neutral": home["is_neutral"].to_numpy(dtype=bool),
            "is_played": True,
            "home_team_id": home["team_id"].to_numpy(),
            "away_team_id": home["opp_team_id"].to_numpy(),
            "home_pts": home["PTS"].to_numpy(dtype=float),
            "away_pts": home["OPP_PTS"].to_numpy(dtype=float),
        }
    )
    wide["home_margin"] = wide["home_pts"] - wide["away_pts"]
    wide["home_win"] = (wide["home_margin"] > 0).astype(float)
    return wide


def _normalise_fixtures(fixtures: pd.DataFrame | None, games: pd.DataFrame) -> pd.DataFrame:
    """Validate fixtures and give them the same columns as :func:`games_wide`."""
    if fixtures is None or len(fixtures) == 0:
        return pd.DataFrame(columns=ID_COLUMNS + OUTCOME_COLUMNS)
    missing = [c for c in FIXTURE_COLUMNS if c not in fixtures.columns]
    if missing:
        raise ValueError(f"fixtures lack columns {missing}")
    fx = fixtures[FIXTURE_COLUMNS].copy()
    fx["game_id"] = fx["game_id"].astype(str)
    fx["game_date"] = pd.to_datetime(fx["game_date"])
    fx["is_neutral"] = fx["is_neutral"].astype(bool)
    if fx["game_id"].duplicated().any():
        raise ValueError("fixtures contain duplicate game_id values")
    if (fx["home_team_id"] == fx["away_team_id"]).any():
        raise ValueError("a fixture cannot have the same home and away team")
    if len(games) and (fx["game_date"] < games["game_date"].max()).any():
        raise ValueError("fixtures must not be dated before the last played game")
    fx["is_played"] = False
    for col in OUTCOME_COLUMNS:
        fx[col] = np.nan
    return fx[ID_COLUMNS + OUTCOME_COLUMNS]


def _team_rows(games: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per team per played game, plus one row per team per fixture date."""
    box = BOX_COLUMNS + [f"OPP_{c}" for c in BOX_COLUMNS]
    played = games[["game_id", "game_date", "season", "game_type", "team_id", "opp_team_id", "win"] + box].copy()
    played["home_flag"] = games["is_home"].astype(float).to_numpy()
    played["is_played"] = True
    rows = played
    if len(fixtures):
        parts = []
        for side, other, flag in (("home_team_id", "away_team_id", 1.0), ("away_team_id", "home_team_id", 0.0)):
            part = fixtures[["game_id", "game_date", "season", "game_type", side, other]].rename(
                columns={side: "team_id", other: "opp_team_id"}
            )
            part["home_flag"] = flag
            parts.append(part)
        fx = pd.concat(parts, ignore_index=True)
        # A team listed in several hypothetical fixtures on one date gets a single row whose
        # venue is unknown (used for round-robin "what if" matchups).
        multi = fx.groupby(["team_id", "game_date"])["home_flag"].transform("size") > 1
        fx.loc[multi, ["home_flag", "opp_team_id"]] = np.nan
        fx.loc[multi, "game_id"] = None
        fx = fx.drop_duplicates(["team_id", "game_date"])
        already = pd.MultiIndex.from_frame(fx[["team_id", "game_date"]]).isin(
            pd.MultiIndex.from_frame(played[["team_id", "game_date"]])
        )
        fx = fx[~already].copy()
        fx["is_played"] = False
        rows = pd.concat([played, fx], ignore_index=True)
    rows["season"] = rows["season"].astype("int64")
    rows["team_id"] = rows["team_id"].astype("int64")
    return rows.sort_values(["team_id", "game_date"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------- team-level pre-game stats
def _keep_rows(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Set every column to NaN on the rows where ``mask`` is False."""
    return frame.where(np.broadcast_to(np.asarray(mask, dtype=bool)[:, None], frame.shape))


def _pregame_blend(rows: pd.DataFrame, stats: list[str], halflife: float, prior_games: float,
                   carry: float, mask: np.ndarray | None = None) -> pd.DataFrame:
    """Pre-game estimate of each stat: in-season EWMA of *earlier* games blended with a prior.

    ``estimate = (n * ewma + k * prior) / (n + k)`` where ``n`` is the number of earlier
    in-season games, ``k = prior_games`` and the prior is the team's previous regular season
    regressed toward the league mean by ``carry``. With ``mask`` only the selected rows (e.g.
    home games) count as observations, which yields venue-specific splits.
    """
    values = rows[stats] if mask is None else _keep_rows(rows[stats], mask)
    keys = [rows["team_id"], rows["season"]]
    shifted = values.groupby(keys).shift(1)  # <- the temporal shift: exclude the current game
    ewma = shifted.groupby(keys).ewm(halflife=halflife, ignore_na=True).mean()
    ewma = ewma.droplevel([0, 1]).reindex(rows.index)
    n_prior = shifted.notna().groupby(keys).cumsum()

    regular = _keep_rows(values, (rows["game_type"] == "regular").to_numpy())
    season_mean = regular.groupby(keys).mean().reset_index()
    league_mean = season_mean.groupby("season")[stats].transform("mean")
    prior_table = season_mean[["team_id", "season"]].copy()
    prior_table[stats] = league_mean + carry * (season_mean[stats] - league_mean)
    prior_table["season"] += 1  # last season's value is the prior of the next one
    prior = rows[["team_id", "season"]].merge(prior_table, on=["team_id", "season"], how="left")[stats]

    n = n_prior.to_numpy(dtype=float)
    e = ewma.to_numpy(dtype=float)
    p = prior.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        blended = (n * np.nan_to_num(e) + prior_games * np.nan_to_num(p)) / (n + prior_games)
    blended = np.where(np.isnan(p), e, blended)  # no prior (first season in the data): in-season only
    blended = np.where(n == 0, p, blended)  # no in-season game yet: prior only
    return pd.DataFrame(blended, index=rows.index, columns=stats)


def _prior_streak(flag: pd.Series, keys: list[pd.Series]) -> pd.Series:
    """Length of the run of consecutive ``flag == 1`` games immediately *before* each row."""
    run_id = (flag != flag.groupby(keys).shift()).cumsum()
    streak = flag.groupby([keys[0], keys[1], run_id]).cumsum()
    return streak.groupby(keys).shift(1).fillna(0.0)


def _add_schedule_features(rows: pd.DataFrame, cfg: FeatureConfig) -> None:
    """Rest, back-to-backs, schedule density and home/road streaks (all known before tip-off)."""
    keys = [rows["team_id"], rows["season"]]
    previous_date = rows.groupby(keys)["game_date"].shift(1)
    rest = (rows["game_date"] - previous_date).dt.days.clip(lower=0, upper=cfg.rest_cap_days)
    rows["pre_rest_days"] = rest.fillna(cfg.rest_cap_days).astype(float)
    rows["pre_b2b"] = (rows["pre_rest_days"] <= 1).astype(float)

    days = rows["game_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    last7 = np.zeros(len(rows))
    last3 = np.zeros(len(rows))
    for idx in rows.groupby("team_id").indices.values():  # rows are sorted by team, then date
        d = days[idx]
        position = np.arange(len(d))
        last7[idx] = position - np.searchsorted(d, d - 7, side="left")
        last3[idx] = position - np.searchsorted(d, d - 3, side="left")
    rows["pre_games_last_7d"] = last7
    rows["pre_three_in_four"] = (last3 >= 2).astype(float)  # third game in four nights

    rows["pre_away_streak"] = _prior_streak((rows["home_flag"] == 0).astype(float), keys)
    rows["pre_home_streak"] = _prior_streak((rows["home_flag"] == 1).astype(float), keys)
    played = rows["is_played"].astype(float)
    rows["pre_games_played"] = played.groupby(keys).cumsum() - played


def _add_team_pregame(rows: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Append every ``pre_`` column that depends only on the team's own history."""
    long = _pregame_blend(rows, LONG_STATS, cfg.long_halflife, cfg.long_prior_games, cfg.season_carry)
    for stat in LONG_STATS:
        rows[f"pre_{'win_pct' if stat == 'win' else stat}"] = long[stat]
    short = _pregame_blend(rows, SHORT_STATS, cfg.short_halflife, cfg.short_prior_games, cfg.season_carry)
    rows["pre_net_rtg_short"] = short["net_rtg"]
    rows["pre_win_pct_short"] = short["win"]
    for label, flag in (("home", 1.0), ("away", 0.0)):
        split = _pregame_blend(rows, VENUE_STATS, cfg.venue_halflife, cfg.venue_prior_games, cfg.season_carry,
                               mask=(rows["home_flag"] == flag).to_numpy())
        rows[f"pre_net_rtg_{label}_split"] = split["net_rtg"]
        rows[f"pre_win_pct_{label}_split"] = split["win"]
    _add_schedule_features(rows, cfg)
    return rows


# -------------------------------------------------------------- matchup-level history
def _h2h_features(wide: pd.DataFrame, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    """Shrunk average margin of recent meetings, from the current home team's point of view.

    The sign of each past margin is flipped when home court was reversed in that meeting (the
    perspective fix found in the original project). ``sum / (n + shrink)`` pulls small samples
    toward zero instead of trusting two or three games.
    """
    n = len(wide)
    played = wide["is_played"].to_numpy(dtype=bool)
    game_ids = wide["game_id"].to_numpy().astype(str)
    days = wide["game_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    order = np.lexsort((game_ids, days, ~played))
    home = wide["home_team_id"].to_numpy()
    away = wide["away_team_id"].to_numpy()
    margin = wide["home_margin"].to_numpy(dtype=float)
    out_margin = np.zeros(n)
    out_count = np.zeros(n)
    history: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for i in order:
        h, a = int(home[i]), int(away[i])
        key = (h, a) if h < a else (a, h)
        past = history.get(key)
        if past:
            recent = [(hid, m) for day, hid, m in past[-cfg.h2h_max_games:] if days[i] - day <= cfg.h2h_window_days]
            if recent:
                total = sum(m if hid == h else -m for hid, m in recent)
                out_margin[i] = total / (len(recent) + cfg.h2h_shrink_games)
                out_count[i] = len(recent)
        if played[i]:
            history.setdefault(key, []).append((int(days[i]), h, float(margin[i])))
    return out_margin, out_count


def _league_home_win_pct(wide: pd.DataFrame, cfg: FeatureConfig) -> pd.Series:
    """League-wide home win rate over the trailing window, using only earlier days."""
    valid = wide[wide["is_played"] & ~wide["is_neutral"]]
    daily = valid.groupby("game_date")["home_win"].agg(["sum", "count"])
    all_dates = pd.DatetimeIndex(np.sort(wide["game_date"].unique()))
    daily = daily.reindex(all_dates, fill_value=0.0)
    trailing = daily.rolling(f"{cfg.league_hca_window_days}D", closed="left").sum()
    pct = (trailing["sum"] / trailing["count"]).where(trailing["count"] >= cfg.league_hca_min_games)
    return wide["game_date"].map(pct)


def compute_matchup_features(m: pd.DataFrame) -> pd.DataFrame:
    """Derive the model inputs from the ``home_*`` / ``away_*`` columns of a matchup frame."""
    m = m.copy()
    m["elo_diff"] = m["home_elo"] - m["away_elo"]
    for stat in DIFF_STATS:
        m[f"{stat}_diff"] = m[f"home_{stat}"] - m[f"away_{stat}"]
    m["venue_net_rtg_diff"] = m["home_net_rtg_home_split"] - m["away_net_rtg_away_split"]
    m["venue_win_pct_diff"] = m["home_win_pct_home_split"] - m["away_win_pct_away_split"]
    m["rest_days_home"] = m["home_rest_days"]
    m["rest_days_away"] = m["away_rest_days"]
    m["rest_diff"] = m["home_rest_days"] - m["away_rest_days"]
    m["b2b_home"] = m["home_b2b"]
    m["b2b_away"] = m["away_b2b"]
    m["three_in_four_home"] = m["home_three_in_four"]
    m["three_in_four_away"] = m["away_three_in_four"]
    m["away_road_streak"] = m["away_away_streak"]
    games_played = np.minimum(m["home_games_played"], m["away_games_played"])
    m["season_progress"] = np.clip(games_played / config.GAMES_PER_SEASON, 0.0, 1.0)
    m["expected_pace"] = 0.5 * (m["home_pace"] + m["away_pace"])
    return m


def swap_sides(matchups: pd.DataFrame) -> pd.DataFrame:
    """Return the same games with home and away exchanged (features recomputed).

    Used to remove home-court advantage for neutral-site games: the neutral forecast is the
    average of "A hosts B" and the mirror of "B hosts A".
    """
    m = matchups.copy()
    home_cols = [c for c in m.columns if c.startswith("home_") and f"away_{c[5:]}" in m.columns]
    for col in home_cols:
        other = f"away_{col[5:]}"
        m[col], m[other] = matchups[other].to_numpy(), matchups[col].to_numpy()
    m["h2h_margin"] = -matchups["h2h_margin"]
    m["home_margin"] = -matchups["home_margin"]
    m["home_win"] = 1.0 - matchups["home_win"]
    return compute_matchup_features(m)


# ------------------------------------------------------------------------ pipeline
def build_features(games: pd.DataFrame, fixtures: pd.DataFrame | None = None, *,
                   elo_params: EloParams | None = None, cfg: FeatureConfig | None = None,
                   team_extras: pd.DataFrame | None = None) -> FeatureSet:
    """Build pre-game features for every played game and every fixture.

    Parameters
    ----------
    games:
        Output of :func:`data_collection.clean_games` (played games only).
    fixtures:
        Unplayed games (``FIXTURE_COLUMNS``). They receive features computed from the history
        available today; because they have no box score they cannot influence anything.
    elo_params, cfg:
        Parameters of the Elo engine and of the rolling statistics. Inference must pass the
        values stored in the trained model bundle.
    team_extras:
        Optional extra pre-game columns keyed by ``(team_id, game_date)`` (``pre_`` prefixed),
        e.g. the roster-availability proxy.
    """
    elo_params = elo_params or EloParams()
    cfg = cfg or FeatureConfig()
    if games.empty:
        raise ValueError("build_features needs at least one played game")

    fx = _normalise_fixtures(fixtures, games)
    played_wide = games_wide(games)
    wide = played_wide if fx.empty else pd.concat([played_wide, fx.astype(played_wide.dtypes.to_dict())], ignore_index=True)

    elo = run_elo(wide, elo_params)
    wide["home_elo"] = elo.home_pre
    wide["away_elo"] = elo.away_pre

    rows = add_game_metrics(_team_rows(games, fx))
    rows = _add_team_pregame(rows, cfg)
    elo_long = pd.concat(
        [
            wide[["home_team_id", "game_date", "home_elo"]].set_axis(["team_id", "game_date", "pre_elo"], axis=1),
            wide[["away_team_id", "game_date", "away_elo"]].set_axis(["team_id", "game_date", "pre_elo"], axis=1),
        ],
        ignore_index=True,
    ).drop_duplicates(["team_id", "game_date"])
    rows = rows.merge(elo_long, on=["team_id", "game_date"], how="left", validate="one_to_one")
    if rows["pre_elo"].isna().any():
        raise RuntimeError("internal error: a team row has no Elo rating")
    keys = [rows["team_id"], rows["season"]]
    elo_then = rows.groupby(keys)["pre_elo"].shift(cfg.elo_trend_games)
    rows["pre_elo_trend"] = rows["pre_elo"] - elo_then.fillna(rows.groupby(keys)["pre_elo"].transform("first"))
    if team_extras is not None:
        extra_cols = [c for c in team_extras.columns if c.startswith("pre_")]
        rows = rows.merge(team_extras[["team_id", "game_date"] + extra_cols], on=["team_id", "game_date"],
                          how="left", validate="one_to_one")

    pre_cols = [c for c in rows.columns if c.startswith("pre_") and c != "pre_elo"]
    team_pre = rows[["team_id", "game_date"] + pre_cols]
    for side in ("home", "away"):
        renamed = team_pre.rename(columns={"team_id": f"{side}_team_id", **{c: f"{side}_{c[4:]}" for c in pre_cols}})
        wide = wide.merge(renamed, on=[f"{side}_team_id", "game_date"], how="left", validate="many_to_one")

    wide["h2h_margin"], wide["h2h_games"] = _h2h_features(wide, cfg)
    wide["league_home_win_pct"] = _league_home_win_pct(wide, cfg).to_numpy()
    matchups = compute_matchup_features(wide)
    if "home_roster_avail" in matchups.columns:
        matchups["roster_avail_home"] = matchups["home_roster_avail"]
        matchups["roster_avail_away"] = matchups["away_roster_avail"]
        matchups["roster_avail_diff"] = matchups["home_roster_avail"] - matchups["away_roster_avail"]
    matchups = matchups.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)
    return FeatureSet(matchups=matchups, team_rows=rows, elo=elo)


def training_mask(matchups: pd.DataFrame) -> pd.Series:
    """Rows a model may be trained or scored on: completed regular-season games at a real home
    court, after the burn-in season(s)."""
    return (
        matchups["is_played"].astype(bool)
        & (matchups["game_type"] == "regular")
        & ~matchups["is_neutral"].astype(bool)
        & (matchups["season"] >= config.FIRST_SEASON + config.BURN_IN_SEASONS)
    )


def hypothetical_fixtures(pairs: list[tuple[int, int]], game_date: pd.Timestamp, season: int,
                          is_neutral: bool = False) -> pd.DataFrame:
    """Fixtures for arbitrary (home, away) pairs on one date, e.g. a "what if" matchup or the
    full round-robin used by the playoff simulator."""
    return pd.DataFrame(
        {
            "game_id": [f"HYP-{h}-{a}" for h, a in pairs],
            "game_date": pd.Timestamp(game_date),
            "season": season,
            "game_type": "regular",
            "home_team_id": [h for h, _ in pairs],
            "away_team_id": [a for _, a in pairs],
            "is_neutral": is_neutral,
        }
    )


def available_feature_columns(matchups: pd.DataFrame) -> dict[str, list[str]]:
    """Feature groups whose columns all exist in ``matchups`` (optional groups included)."""
    groups = {**config.FEATURE_GROUPS, **config.OPTIONAL_FEATURE_GROUPS}
    return {name: cols for name, cols in groups.items() if all(c in matchups.columns for c in cols)}


# -------------------------------------------------------------------- persistence
def save_processed(features: FeatureSet) -> None:
    """Write the matchup frame and the team rows to ``data/processed``."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.matchups.to_csv(config.MATCHUPS_PATH, index=False)
    keep = ["game_id", "game_date", "season", "game_type", "team_id", "opp_team_id", "home_flag", "is_played",
            "win", "PTS", "OPP_PTS"] + GAME_METRICS + [c for c in features.team_rows.columns if c.startswith("pre_")]
    features.team_rows[keep].to_csv(config.TEAM_GAMES_PATH, index=False)
    logger.info("Saved %d matchups and %d team rows to %s", len(features.matchups), len(features.team_rows),
                config.PROCESSED_DIR)


def load_matchups(path=None) -> pd.DataFrame:
    """Read the processed matchup frame with stable dtypes."""
    path = config.MATCHUPS_PATH if path is None else path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m nba_predictor features` first")
    frame = pd.read_csv(path, dtype={"game_id": str}, parse_dates=["game_date"])
    frame["is_played"] = frame["is_played"].astype(bool)
    frame["is_neutral"] = frame["is_neutral"].astype(bool)
    return frame
