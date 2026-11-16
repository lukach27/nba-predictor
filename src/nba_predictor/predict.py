"""Inference: turn a trained :class:`ModelBundle` plus the game log into predictions.

There is deliberately no inference-only feature code here. Every prediction - a scheduled
game, an arbitrary "what if" matchup, or the round-robin used by the season simulator - is
made by appending *fixtures* to the history and calling the same
:func:`features.build_features` that produced the training set, with the Elo and feature
parameters stored in the bundle.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from . import config
from .data_collection import (FIXTURE_COLUMNS, load_clean_games, load_schedule, team_directory,
                              upcoming_fixtures)
from .elo import LOGIT_PER_ELO, home_win_probability
from .explain import explain_row
from .features import FeatureSet, build_features, hypothetical_fixtures, swap_sides
from .metrics import sigmoid
from .modeling import MODEL_NAMES, ModelBundle
from .uncertainty import UncertaintyInputs, confidence, volatility

logger = logging.getLogger(__name__)

#: Team-level pre-game values exposed to the dashboard and to the LLM analyst.
FACT_STATS: list[str] = [
    "elo", "elo_trend", "win_pct", "net_rtg", "off_rtg", "def_rtg", "pace", "efg", "tov_pct", "orb_pct", "ft_rate",
    "opp_efg", "opp_tov_pct", "drb_pct", "opp_ft_rate", "fg3a_rate", "net_rtg_short", "win_pct_short",
    "net_rtg_home_split", "net_rtg_away_split", "win_pct_home_split", "win_pct_away_split", "rest_days", "b2b",
    "three_in_four", "games_last_7d", "games_played", "away_streak", "home_streak", "roster_avail",
]


class PredictionError(ValueError):
    """Raised for invalid prediction requests (unknown team, same team twice, bad date)."""


@dataclass
class MatchupPrediction:
    """A single game forecast with everything needed to display and explain it."""

    game_date: str
    home_team_id: int
    away_team_id: int
    home_team: str
    away_team: str
    is_neutral: bool
    p_home: float
    p_lr: float
    p_xgb: float
    p_elo: float
    p_low: float
    p_high: float
    p_std: float
    expected_margin: float | None
    favorite: str
    favorite_probability: float
    confidence: dict
    volatility: dict
    home_facts: dict
    away_facts: dict
    matchup_facts: dict
    features: dict
    contributions: list[dict]
    elo_adjustment: dict = field(default_factory=dict)
    p_home_unadjusted: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Predictor:
    """Stateless-by-construction prediction service around a model bundle and a game log."""

    def __init__(self, bundle: ModelBundle, games: pd.DataFrame, schedule: pd.DataFrame | None = None,
                 team_extras: pd.DataFrame | None = None) -> None:
        if games.empty:
            raise PredictionError("Predictor needs a non-empty game log")
        self.bundle = bundle
        self.games = games
        self.schedule = schedule
        self.team_extras = team_extras
        self.teams = team_directory(games)
        self.current_season = int(games["season"].max())
        self.last_played_date = pd.Timestamp(games["game_date"].max())
        self._abbr_by_id = dict(zip(self.teams["team_id"], self.teams["team_abbr"]))
        self._id_by_abbr = {a: i for i, a in self._abbr_by_id.items()}

    @classmethod
    def from_disk(cls) -> "Predictor":
        """Load the saved bundle, the raw game log and the schedule."""
        bundle = ModelBundle.load()
        extras = None
        if bundle.metadata.get("uses_team_extras"):
            from .roster import load_team_extras

            extras = load_team_extras()
        return cls(bundle, load_clean_games(), load_schedule(), team_extras=extras)

    # ------------------------------------------------------------------ helpers
    def resolve_team(self, team: int | str) -> int:
        """Accept a team id, an abbreviation (``'BOS'``) or a full name and return the id."""
        if isinstance(team, (int, np.integer)) and int(team) in self._abbr_by_id:
            return int(team)
        text = str(team).strip()
        if text.upper() in self._id_by_abbr:
            return self._id_by_abbr[text.upper()]
        match = self.teams[self.teams["team_name"].str.lower() == text.lower()]
        if len(match) == 1:
            return int(match["team_id"].iloc[0])
        raise PredictionError(f"Unknown team '{team}'. Use one of: {', '.join(sorted(self._id_by_abbr))}")

    def abbr(self, team_id: int) -> str:
        return self._abbr_by_id[int(team_id)]

    def _build(self, fixtures: pd.DataFrame | None) -> FeatureSet:
        return build_features(self.games, fixtures, elo_params=self.bundle.elo_params,
                              cfg=self.bundle.feature_config, team_extras=self.team_extras)

    @cached_property
    def league_fg3a_rate(self) -> float:
        """Share of field-goal attempts taken from three across the current season (observed)."""
        season = self.games[self.games["season"] == self.current_season]
        return float(season["FG3A"].sum() / season["FGA"].sum())

    @cached_property
    def league_pace(self) -> float:
        """Average estimated pace (possessions per 48 minutes) in the current season."""
        from .box_metrics import add_game_metrics

        season = self.games[(self.games["season"] == self.current_season) & self.games["is_home"]]
        return float(add_game_metrics(season)["pace"].mean())

    @cached_property
    def fixtures(self) -> pd.DataFrame:
        """Scheduled regular-season games that have not been played."""
        return upcoming_fixtures(self.schedule, self.games)

    @cached_property
    def features(self) -> FeatureSet:
        """History plus scheduled fixtures, through the shared feature pipeline."""
        return self._build(self.fixtures if len(self.fixtures) else None)

    # ------------------------------------------------------------------ scoring
    def score(self, matchups: pd.DataFrame) -> pd.DataFrame:
        """Probabilities for rows of a matchup frame.

        Neutral-site rows are symmetrised in log-odds - the average of "A hosts B" and the
        mirror of "B hosts A" - which removes home-court advantage without a second model.
        """
        z = self.bundle.logits(matchups)
        replicas = self.bundle.replica_logits(matchups)
        margin = self.bundle.predict_margin(matchups)
        neutral = matchups["is_neutral"].to_numpy(dtype=bool)
        if neutral.any():
            mirrored = swap_sides(matchups[neutral])
            z_mirror = self.bundle.logits(mirrored)
            for name in MODEL_NAMES:
                z[name] = z[name].copy()
                z[name][neutral] = 0.5 * (z[name][neutral] - z_mirror[name])
            if len(replicas):
                replicas[:, neutral] = 0.5 * (replicas[:, neutral] - self.bundle.replica_logits(mirrored))
            if margin is not None:
                margin[neutral] = 0.5 * (margin[neutral] - self.bundle.predict_margin(mirrored))

        out = matchups[["game_id", "game_date", "season", "home_team_id", "away_team_id", "is_neutral"]].copy()
        out["home_team"] = out["home_team_id"].map(self._abbr_by_id)
        out["away_team"] = out["away_team_id"].map(self._abbr_by_id)
        for name in MODEL_NAMES:
            out[f"p_{name}"] = sigmoid(z[name])
        out["logit_ensemble"] = z["ensemble"]
        out["p_elo"] = home_win_probability(matchups["home_elo"], matchups["away_elo"],
                                            self.bundle.elo_params.home_advantage, neutral)
        if len(replicas):
            p_rep = sigmoid(replicas)
            out["p_std"] = p_rep.std(axis=0)
            out["p_low"], out["p_high"] = np.quantile(p_rep, 0.1, axis=0), np.quantile(p_rep, 0.9, axis=0)
        else:
            out["p_std"], out["p_low"], out["p_high"] = 0.0, out["p_ensemble"], out["p_ensemble"]
        out["expected_margin"] = np.nan if margin is None else margin
        out["elo_diff"] = matchups["elo_diff"].to_numpy()
        return out

    def upcoming(self, days: int | None = 7) -> pd.DataFrame:
        """Predictions for scheduled games in the next ``days`` days (``None`` = all remaining)."""
        matchups = self.features.matchups
        rows = matchups[~matchups["is_played"]]
        if days is not None and len(rows):
            start = rows["game_date"].min()
            rows = rows[rows["game_date"] <= start + pd.Timedelta(days=days - 1)]
        if rows.empty:
            return pd.DataFrame()
        scored = self.score(rows)
        levels = []
        for (_, s), (_, row) in zip(scored.iterrows(), rows.iterrows()):
            inputs = UncertaintyInputs(
                p_home=float(s["p_ensemble"]), p_lr=float(s["p_lr"]), p_xgb=float(s["p_xgb"]), p_std=float(s["p_std"]),
                min_games_played=float(min(row["home_games_played"], row["away_games_played"])),
                mean_fg3a_rate=float(np.nanmean([row["home_fg3a_rate"], row["away_fg3a_rate"]])),
                league_fg3a_rate=self.league_fg3a_rate,
                schedule_stress=bool(row["home_b2b"] or row["away_b2b"] or row["home_three_in_four"] or row["away_three_in_four"]),
            )
            conf, vol = confidence(inputs), volatility(inputs)
            levels.append((conf["level"], conf["score"], vol["level"], vol["score"]))
        scored[["confidence_level", "confidence_score", "volatility_level", "volatility_score"]] = levels
        tipoff = {} if self.schedule is None else dict(zip(self.schedule["game_id"], self.schedule["game_time_utc"]))
        scored["tipoff_utc"] = scored["game_id"].map(tipoff)
        return scored.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    # --------------------------------------------------------------- one matchup
    def _facts(self, row: pd.Series, side: str) -> dict:
        facts = {stat: _clean(row.get(f"{side}_{stat}")) for stat in FACT_STATS}
        team_id = int(row[f"{side}_team_id"])
        facts["team"] = self.abbr(team_id)
        facts["team_name"] = self.teams.set_index("team_id").loc[team_id, "team_name"]
        facts.update(self.team_record(team_id))
        return facts

    def team_record(self, team_id: int, last_n: int = 10) -> dict:
        """Current-season regular-season record and the last ``last_n`` results (observed)."""
        g = self.games
        season = g[(g["team_id"] == team_id) & (g["season"] == self.current_season) & (g["game_type"] == "regular")]
        season = season.sort_values("game_date")
        home, away = season[season["is_home"]], season[~season["is_home"]]
        recent = season.tail(last_n)
        return {
            "wins": int(season["win"].sum()), "losses": int(len(season) - season["win"].sum()),
            "home_record": f"{int(home['win'].sum())}-{int(len(home) - home['win'].sum())}",
            "away_record": f"{int(away['win'].sum())}-{int(len(away) - away['win'].sum())}",
            "last_10_record": f"{int(recent['win'].sum())}-{int(len(recent) - recent['win'].sum())}",
            "last_5_results": "".join("W" if w else "L" for w in season.tail(5)["win"]),
        }

    def predict_matchup(self, home: int | str, away: int | str, game_date: str | pd.Timestamp | None = None,
                        is_neutral: bool = False, elo_adjust: dict[int | str, float] | None = None,
                        fixture_row: pd.Series | None = None) -> MatchupPrediction:
        """Forecast one game.

        ``elo_adjust`` maps a team to a manual strength adjustment in Elo points (for example
        ``{'BOS': -60}`` for a missing star). It is applied to the final log-odds on the Elo
        scale and reported separately; it is a user assumption, not something the model knows.
        Pass ``fixture_row`` (a row of ``self.features.matchups``) to explain a scheduled game.
        """
        home_id, away_id = self.resolve_team(home), self.resolve_team(away)
        if home_id == away_id:
            raise PredictionError("Home and away team must differ")
        if fixture_row is not None:
            matchup = fixture_row.to_frame().T.infer_objects()
            matchup["game_date"] = pd.to_datetime(matchup["game_date"])
        else:
            date = self.last_played_date + pd.Timedelta(days=1) if game_date is None else pd.Timestamp(game_date)
            if date < self.last_played_date:
                raise PredictionError(
                    f"game_date {date.date()} is before the last game in the data ({self.last_played_date.date()})")
            fx = hypothetical_fixtures([(home_id, away_id)], date, self.current_season, is_neutral)
            built = self._build(fx).matchups
            matchup = built[built["game_id"] == fx["game_id"].iloc[0]]
        row = matchup.iloc[0]
        scored = self.score(matchup).iloc[0]

        adjustments = {self.abbr(self.resolve_team(k)): float(v) for k, v in (elo_adjust or {}).items() if float(v) != 0.0}
        shift = LOGIT_PER_ELO * (adjustments.get(self.abbr(home_id), 0.0) - adjustments.get(self.abbr(away_id), 0.0))
        p_base = float(scored["p_ensemble"])
        p_home = float(sigmoid(scored["logit_ensemble"] + shift))
        favorite_is_home = p_home >= 0.5
        inputs = UncertaintyInputs(
            p_home=p_home, p_lr=float(scored["p_lr"]), p_xgb=float(scored["p_xgb"]), p_std=float(scored["p_std"]),
            min_games_played=float(min(row["home_games_played"], row["away_games_played"])),
            mean_fg3a_rate=float(np.nanmean([row["home_fg3a_rate"], row["away_fg3a_rate"]])),
            league_fg3a_rate=self.league_fg3a_rate,
            schedule_stress=bool(row["home_b2b"] or row["away_b2b"] or row["home_three_in_four"] or row["away_three_in_four"]),
        )
        contributions = explain_row(self.bundle, matchup)
        margin = None if pd.isna(scored["expected_margin"]) else float(scored["expected_margin"])
        return MatchupPrediction(
            game_date=str(pd.Timestamp(row["game_date"]).date()), home_team_id=home_id, away_team_id=away_id,
            home_team=self.abbr(home_id), away_team=self.abbr(away_id), is_neutral=bool(row["is_neutral"]),
            p_home=p_home, p_lr=float(scored["p_lr"]), p_xgb=float(scored["p_xgb"]), p_elo=float(scored["p_elo"]),
            p_low=float(scored["p_low"]), p_high=float(scored["p_high"]), p_std=float(scored["p_std"]),
            expected_margin=margin,
            favorite=self.abbr(home_id if favorite_is_home else away_id),
            favorite_probability=p_home if favorite_is_home else 1.0 - p_home,
            confidence=confidence(inputs), volatility=volatility(inputs),
            home_facts=self._facts(row, "home"), away_facts=self._facts(row, "away"),
            matchup_facts={
                "elo_diff": _clean(row["elo_diff"]), "home_court_elo": self.bundle.elo_params.home_advantage,
                "h2h_margin": _clean(row["h2h_margin"]), "h2h_games": _clean(row["h2h_games"]),
                "expected_pace": _clean(row["expected_pace"]), "league_home_win_pct": _clean(row["league_home_win_pct"]),
                "rest_diff": _clean(row["rest_diff"]), "league_pace": self.league_pace,
            },
            features={c: _clean(row[c]) for c in self.bundle.feature_columns},
            contributions=contributions.to_dict("records"),
            elo_adjustment=adjustments, p_home_unadjusted=p_base if adjustments else None,
        )

    # -------------------------------------------------------- simulator support
    def pairwise_logits(self) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble log-odds ``Z[h, a]`` of ``h`` beating ``a`` at home, for every ordered pair.

        Built from a round-robin of hypothetical fixtures dated far enough ahead that every
        team is rested, so the matrix reflects team strength rather than today's schedule.
        """
        ids = self.teams["team_id"].to_numpy()
        pairs = [(int(h), int(a)) for h in ids for a in ids if h != a]
        date = self.last_played_date + pd.Timedelta(days=self.bundle.feature_config.rest_cap_days + 3)
        fx = hypothetical_fixtures(pairs, date, self.current_season)
        built = self._build(fx).matchups
        rows = built[~built["is_played"]]
        z = self.bundle.logits(rows)["ensemble"]
        index = {int(t): i for i, t in enumerate(ids)}
        matrix = np.zeros((len(ids), len(ids)))
        matrix[rows["home_team_id"].map(index).to_numpy(), rows["away_team_id"].map(index).to_numpy()] = z
        return ids, matrix

    def team_snapshot(self) -> pd.DataFrame:
        """Each team's current pre-game values (as they would enter its next game)."""
        ids = self.teams["team_id"].to_numpy()
        pairs = [(int(ids[i]), int(ids[(i + 1) % len(ids)])) for i in range(len(ids))]
        date = self.last_played_date + pd.Timedelta(days=1)
        rows = self._build(hypothetical_fixtures(pairs, date, self.current_season)).matchups
        rows = rows[~rows["is_played"]]
        snap = pd.DataFrame({"team_id": rows["home_team_id"].to_numpy()})
        for stat in FACT_STATS:  # optional groups (e.g. roster) are absent without their data
            snap[stat] = rows[f"home_{stat}"].to_numpy() if f"home_{stat}" in rows.columns else np.nan
        snap = snap.merge(self.teams, on="team_id")
        records = pd.DataFrame([{"team_id": int(t), **self.team_record(int(t))} for t in snap["team_id"]])
        return snap.merge(records, on="team_id").sort_values("elo", ascending=False).reset_index(drop=True)


def _clean(value):
    """Convert numpy scalars to plain Python and NaN to ``None`` (JSON friendly)."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


__all__ = ["FACT_STATS", "FIXTURE_COLUMNS", "MatchupPrediction", "PredictionError", "Predictor", "config"]
