"""Data-driven NBA tactical analyst (LLM layer on top of the statistical models).

Design
------
The statistical models decide; the analyst explains. The division of labour is enforced in
code, not just requested in a prompt:

1. :func:`build_fact_sheet` turns a :class:`~nba_predictor.predict.MatchupPrediction` into a
   flat sheet of pre-rounded numbers, each with an id (``home.efg``, ``diff.net_rtg`` ...).
   This is the *only* information the LLM receives.
2. The LLM returns an :class:`AnalystNarrative` - text and references to metric ids. It has
   no field for a probability, a confidence score or a volatility score.
3. :func:`validate_narrative` rejects the answer if it names a different favourite, cites a
   metric id that does not exist, contains a number that is not on the fact sheet, or talks
   about injuries / lineups (no such data is supplied).
4. :func:`assemble_report` attaches the prediction, confidence and volatility **from the
   statistical models** and validates the final JSON against :class:`TacticalAnalysis`.

If the LLM is disabled, unreachable, refuses, returns malformed output or fails validation
twice, a deterministic rule-based analyst fills the same schema and the report says so
(``meta.source == "rule_based"`` with the reason). Nothing fails silently.

The Anthropic SDK is imported lazily, so the rest of the project works without it.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import LLMConfig

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
#: Numbers that may appear in ordinary phrasing without being statistics
#: ("per 100 possessions", "48 minutes", "last 10", "top 5", "3-point").
CONVENTIONAL_NUMBERS: frozenset[float] = frozenset({1, 2, 3, 4, 5, 7, 10, 48, 50, 100})
#: The analyst is given no injury or lineup data, so it must not discuss them.
FORBIDDEN_TERMS: tuple[str, ...] = (
    "injur", "sidelined", "questionable", "doubtful", "day-to-day", "ruled out", "suspended", "suspension",
    "load management", "minutes restriction", "starting lineup", "starting five", "returning from",
)
DATA_LIMITATIONS: list[str] = [
    "No injury, lineup, trade or minutes-restriction information is available to the models or to this analysis.",
    "Team statistics are rolling averages of past games; ratings and pace rely on an estimated possession count.",
    "Win probabilities come from the calibrated statistical ensemble; the analyst explains them and cannot change them.",
]

Category = Literal[
    "overall_strength", "efficiency", "shooting", "ball_security", "offensive_rebounding", "free_throws",
    "shot_defense", "forcing_turnovers", "defensive_rebounding", "foul_discipline", "recent_form", "rest",
    "venue", "pace", "head_to_head",
]

# stat, label, source, scale, decimals, higher_is_better, typical spread between teams (display units)
METRIC_SPECS: list[tuple[str, str, str, float, int, bool | None, float]] = [
    ("elo", "Elo rating", "model", 1, 0, True, 107.0),
    ("elo_trend", "Elo change over the last 10 games", "model", 1, 0, True, 35.0),
    ("win_pct", "Win percentage (season form blended with last season's prior)", "estimated", 100, 1, True, 13.0),
    ("net_rtg", "Net rating (points per 100 possessions, estimated)", "estimated", 1, 1, True, 4.3),
    ("off_rtg", "Offensive rating (points scored per 100 possessions, estimated)", "estimated", 1, 1, True, 2.9),
    ("def_rtg", "Defensive rating (points allowed per 100 possessions, estimated)", "estimated", 1, 1, False, 2.9),
    ("pace", "Pace (possessions per 48 minutes, estimated)", "estimated", 1, 1, None, 1.8),
    ("efg", "Effective field-goal %", "observed", 100, 1, True, 1.6),
    ("tov_pct", "Turnover % (share of plays ending in a turnover)", "observed", 100, 1, False, 0.8),
    ("orb_pct", "Offensive rebound %", "observed", 100, 1, True, 2.0),
    ("ft_rate", "Free-throw rate (FTM per FGA, x100)", "observed", 100, 1, True, 1.7),
    ("opp_efg", "Opponent effective field-goal % allowed", "observed", 100, 1, False, 1.3),
    ("opp_tov_pct", "Opponent turnover % forced", "observed", 100, 1, True, 0.9),
    ("drb_pct", "Defensive rebound %", "observed", 100, 1, True, 1.4),
    ("opp_ft_rate", "Opponent free-throw rate allowed (x100)", "observed", 100, 1, False, 1.7),
    ("fg3a_rate", "Share of shots taken from three (%)", "observed", 100, 1, None, 3.0),
    ("net_rtg_short", "Net rating, recent form (5-game half-life)", "estimated", 1, 1, True, 5.6),
    ("win_pct_short", "Win percentage, recent form", "estimated", 100, 1, True, 17.0),
    ("rest_days", "Days since the previous game (capped at 7)", "observed", 1, 0, True, 1.0),
    ("games_last_7d", "Games played in the previous 7 days", "observed", 1, 0, False, 0.6),
    ("games_played", "Games played this season", "observed", 1, 0, None, 1.0),
]
_SPEC = {s[0]: s for s in METRIC_SPECS}
#: (category, stat, noun phrase) used by the rule-based analyst to rank edges.
ADVANTAGE_STATS: list[tuple[str, str, str]] = [
    ("overall_strength", "elo", "Elo rating"),
    ("efficiency", "net_rtg", "net rating"),
    ("shooting", "efg", "effective field-goal percentage"),
    ("ball_security", "tov_pct", "turnover rate"),
    ("offensive_rebounding", "orb_pct", "offensive rebound rate"),
    ("free_throws", "ft_rate", "free-throw rate (makes per 100 shot attempts)"),
    ("shot_defense", "opp_efg", "opponent effective field-goal percentage"),
    ("forcing_turnovers", "opp_tov_pct", "forced turnover rate"),
    ("defensive_rebounding", "drb_pct", "defensive rebound rate"),
    ("foul_discipline", "opp_ft_rate", "opponent free-throw rate allowed (makes per 100 shot attempts)"),
    ("recent_form", "net_rtg_short", "recent-form net rating"),
]
_INTERPRETATION: dict[str, str] = {
    "overall_strength": "The rating gap reflects sustained results over many games, the single strongest signal in the model.",
    "efficiency": "Outscoring opponents per possession is the most stable indicator of team quality.",
    "shooting": "Shot quality and shot-making is the most important of the Four Factors.",
    "ball_security": "Fewer empty possessions means more shots at the rim and fewer transition chances conceded.",
    "offensive_rebounding": "Second-chance possessions can offset a cold shooting night.",
    "free_throws": "Getting to the line produces efficient points and puts the opponent in foul trouble.",
    "shot_defense": "Holding opponents to poor shooting is the defensive side of the most important factor.",
    "forcing_turnovers": "Forced turnovers end possessions without a shot and feed the transition game.",
    "defensive_rebounding": "Finishing defensive possessions with the rebound denies second chances.",
    "foul_discipline": "Keeping the opponent off the line removes their most efficient scoring source.",
    "recent_form": "Recent results suggest a team playing above or below its season baseline.",
}


# ---------------------------------------------------------------------- exceptions
class LLMError(RuntimeError):
    """Base class: the LLM analyst could not produce a usable answer."""


class LLMUnavailableError(LLMError):
    """The LLM is disabled, not installed, not authenticated or unreachable."""


class LLMResponseError(LLMError):
    """The LLM answered, but the answer was refused, truncated or not valid JSON for the schema."""


class LLMGroundingError(LLMError):
    """The LLM answer contradicted the model or contained statistics that were not supplied."""


# -------------------------------------------------------------------------- schema
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StatAdvantage(_Strict):
    """One statistical edge. ``statement`` is data; ``interpretation`` is the analyst's reading."""

    team: str = Field(description="Abbreviation of the team that holds the edge.")
    category: Category
    metric_ids: list[str] = Field(description="Ids from the fact sheet that back the statement.")
    statement: str = Field(description="What the data says, using only fact-sheet numbers.")
    interpretation: str = Field(description="Analytical reading of that data; no new numbers.")


class Vulnerability(_Strict):
    """A weakness one team could be exposed on in this matchup."""

    team: str
    metric_ids: list[str]
    statement: str
    interpretation: str


class SideAnalysis(_Strict):
    """Two-sided analysis: one paragraph per team."""

    home: str
    away: str


class AnalystNarrative(_Strict):
    """Everything the analyst (LLM or rule-based) writes. It carries no probabilities or scores."""

    favored_team: str = Field(description="Abbreviation of the team the statistical model favours (echo it).")
    matchup_overview: str
    key_statistical_advantages: list[StatAdvantage]
    offensive_matchup: SideAnalysis = Field(description="Each team's offence against the other's defence.")
    defensive_matchup: SideAnalysis = Field(description="What each defence can take away.")
    pace_analysis: str
    potential_vulnerabilities: list[Vulnerability]
    confidence_rationale: str
    volatility_rationale: str
    narrative_preview: str


class PredictionBlock(_Strict):
    """Forecast copied verbatim from the statistical models."""

    favorite: str
    home_win_probability: float = Field(ge=0, le=1)
    away_win_probability: float = Field(ge=0, le=1)
    logistic_regression_probability: float = Field(ge=0, le=1)
    xgboost_probability: float = Field(ge=0, le=1)
    elo_only_probability: float = Field(ge=0, le=1)
    probability_band: tuple[float, float]
    expected_margin: float | None
    manual_elo_adjustment: dict[str, float] = Field(default_factory=dict)


class PaceBlock(_Strict):
    estimate: float | None
    league_average: float | None
    label: Literal["slow", "average", "fast", "unknown"]
    analysis: str


class ConfidenceBlock(_Strict):
    level: Literal["low", "medium", "high"]
    score: float = Field(ge=0, le=100)
    rationale: str


class VolatilityBlock(_Strict):
    score: float = Field(ge=0, le=100)
    level: Literal["low", "medium", "high"]
    drivers: list[dict[str, Any]]
    rationale: str


class ReportMeta(_Strict):
    schema_version: str
    source: Literal["llm", "rule_based"]
    model: str | None
    generated_at: str
    fallback_reason: str | None = None


class TacticalAnalysis(_Strict):
    """The structured report returned to the dashboard / CLI (validated before it is returned)."""

    meta: ReportMeta
    matchup: dict[str, Any]
    prediction: PredictionBlock
    matchup_overview: str
    key_statistical_advantages: list[StatAdvantage]
    offensive_matchup: SideAnalysis
    defensive_matchup: SideAnalysis
    expected_pace: PaceBlock
    potential_vulnerabilities: list[Vulnerability]
    prediction_confidence: ConfidenceBlock
    volatility_risk: VolatilityBlock
    narrative_preview: str
    data_limitations: list[str]
    fact_sheet: dict[str, Any]


# ---------------------------------------------------------------------- fact sheet
def _display(value: Any, scale: float, decimals: int) -> float | None:
    if value is None:
        return None
    out = round(float(value) * scale, decimals)
    return int(out) if decimals == 0 else out


def _pct(p: float) -> float:
    return round(100.0 * float(p), 1)


def build_fact_sheet(prediction: Any) -> dict[str, Any]:
    """Flatten a prediction into the pre-rounded facts the analyst may use.

    ``prediction`` is a :class:`~nba_predictor.predict.MatchupPrediction` (or its ``to_dict()``).
    Percentages are expressed in percent with one decimal, exactly as they should be quoted.
    """
    p = prediction if isinstance(prediction, dict) else prediction.to_dict()
    home, away = p["home_facts"], p["away_facts"]
    metrics: dict[str, Any] = {}
    definitions: dict[str, dict[str, str]] = {}
    for stat, label, source, scale, decimals, _, _ in METRIC_SPECS:
        h, a = _display(home.get(stat), scale, decimals), _display(away.get(stat), scale, decimals)
        metrics[f"home.{stat}"], metrics[f"away.{stat}"] = h, a
        metrics[f"diff.{stat}"] = None if h is None or a is None else round(h - a, decimals) if decimals else int(h - a)
        definitions[stat] = {"label": label, "source": source}
    # venue-specific form: the home team's home split against the away team's road split
    metrics["home.venue_net_rtg"] = _display(home.get("net_rtg_home_split"), 1, 1)
    metrics["away.venue_net_rtg"] = _display(away.get("net_rtg_away_split"), 1, 1)
    metrics["home.venue_win_pct"] = _display(home.get("win_pct_home_split"), 100, 1)
    metrics["away.venue_win_pct"] = _display(away.get("win_pct_away_split"), 100, 1)
    definitions["venue_net_rtg"] = {"label": "Net rating at this venue type (home team at home, away team on the road)", "source": "estimated"}
    definitions["venue_win_pct"] = {"label": "Win percentage at this venue type", "source": "estimated"}
    for side, facts in (("home", home), ("away", away)):
        for key in ("wins", "losses", "home_record", "away_record", "last_10_record", "last_5_results"):
            metrics[f"{side}.{key}"] = facts.get(key)
        metrics[f"{side}.back_to_back"] = bool(facts.get("b2b"))
        metrics[f"{side}.third_game_in_four_nights"] = bool(facts.get("three_in_four"))
        metrics[f"{side}.consecutive_road_games_before"] = _display(facts.get("away_streak"), 1, 0)

    mf = p["matchup_facts"]
    pace, league_pace = _display(mf.get("expected_pace"), 1, 1), _display(mf.get("league_pace"), 1, 1)
    pace_label = "unknown"
    if pace is not None and league_pace is not None:
        pace_label = "fast" if pace - league_pace > 1.5 else "slow" if pace - league_pace < -1.5 else "average"
    metrics.update(
        {
            "matchup.elo_diff": _display(mf.get("elo_diff"), 1, 0),
            "matchup.home_court_elo": _display(mf.get("home_court_elo"), 1, 0),
            "matchup.h2h_games": _display(mf.get("h2h_games"), 1, 0),
            "matchup.h2h_margin_shrunk": _display(mf.get("h2h_margin"), 1, 1),
            "matchup.expected_pace": pace,
            "matchup.league_pace": league_pace,
            "matchup.league_home_win_pct": _display(mf.get("league_home_win_pct"), 100, 1),
        }
    )
    favorite_is_home = p["favorite"] == p["home_team"]
    return {
        "matchup": {
            "home_team": p["home_team"], "away_team": p["away_team"],
            "home_team_name": home.get("team_name"), "away_team_name": away.get("team_name"),
            "game_date": p["game_date"], "neutral_site": bool(p["is_neutral"]),
        },
        "model": {
            "favorite": p["favorite"],
            "underdog": p["away_team"] if favorite_is_home else p["home_team"],
            "favorite_win_probability_pct": _pct(p["favorite_probability"]),
            "home_win_probability_pct": _pct(p["p_home"]),
            "away_win_probability_pct": _pct(1 - p["p_home"]),
            "logistic_regression_home_pct": _pct(p["p_lr"]),
            "xgboost_home_pct": _pct(p["p_xgb"]),
            "elo_only_home_pct": _pct(p["p_elo"]),
            "band_low_home_pct": _pct(p["p_low"]), "band_high_home_pct": _pct(p["p_high"]),
            "expected_home_margin": None if p.get("expected_margin") is None else round(p["expected_margin"], 1),
            "manual_elo_adjustment": p.get("elo_adjustment") or {},
        },
        "uncertainty": {
            "confidence_level": p["confidence"]["level"], "confidence_score": p["confidence"]["score"],
            "model_disagreement_pct": round(100 * p["confidence"]["model_disagreement"], 1),
            "volatility_score": p["volatility"]["score"], "volatility_level": p["volatility"]["level"],
            "volatility_drivers": [
                {"label": d["label"], "points": d["points"]} for d in p["volatility"]["drivers"] if d["points"] > 0
            ],
        },
        "pace_label": pace_label,
        "metrics": metrics,
        "metric_definitions": definitions,
        "top_model_factors": [
            {"feature": c["feature"], "direction": "favours home" if c["contribution"] > 0 else "favours away",
             "log_odds": round(c["contribution"], 2)}
            for c in p["contributions"][:6]
        ],
        "_raw": {"prediction": {k: p[k] for k in ("p_home", "p_lr", "p_xgb", "p_elo", "p_low", "p_high", "expected_margin")},
                 "confidence": p["confidence"], "volatility": p["volatility"]},
    }


# ---------------------------------------------------------------------- validation
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _numbers_in(value: Any) -> list[float]:
    """Every numeric value found anywhere in a JSON-like structure (numbers and digits in strings)."""
    found: list[float] = []
    if isinstance(value, bool) or value is None:
        return found
    if isinstance(value, (int, float)):
        found.append(abs(float(value)))
    elif isinstance(value, str):
        found.extend(float(tok) for tok in _NUMBER.findall(value))
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_numbers_in(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_numbers_in(item))
    return found


def allowed_numbers(facts: dict[str, Any]) -> list[float]:
    """Numbers the analyst may quote: everything on the fact sheet (not the raw block)."""
    public = {k: v for k, v in facts.items() if k != "_raw"}
    return sorted(set(_numbers_in(public)))


def ungrounded_numbers(text: str, allowed: list[float]) -> list[str]:
    """Numeric tokens in ``text`` that are neither on the fact sheet nor conventional.

    A token counts as grounded when it is a correct rounding of a fact-sheet number at the
    precision it was written with (``61.6`` may be quoted as ``62``; ``45`` may not).
    """
    missing = []
    for token in _NUMBER.findall(text):
        value = float(token)
        if value in CONVENTIONAL_NUMBERS:
            continue
        decimals = len(token.split(".")[1]) if "." in token else 0
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(value - number) <= tolerance for number in allowed):
            missing.append(token)
    return missing


def _texts(narrative: AnalystNarrative) -> list[tuple[str, str]]:
    """(field name, text) pairs for every free-text field of a narrative."""
    out = [
        ("matchup_overview", narrative.matchup_overview), ("offensive_matchup.home", narrative.offensive_matchup.home),
        ("offensive_matchup.away", narrative.offensive_matchup.away), ("defensive_matchup.home", narrative.defensive_matchup.home),
        ("defensive_matchup.away", narrative.defensive_matchup.away), ("pace_analysis", narrative.pace_analysis),
        ("confidence_rationale", narrative.confidence_rationale), ("volatility_rationale", narrative.volatility_rationale),
        ("narrative_preview", narrative.narrative_preview),
    ]
    for i, item in enumerate(narrative.key_statistical_advantages):
        out += [(f"key_statistical_advantages[{i}].statement", item.statement),
                (f"key_statistical_advantages[{i}].interpretation", item.interpretation)]
    for i, item in enumerate(narrative.potential_vulnerabilities):
        out += [(f"potential_vulnerabilities[{i}].statement", item.statement),
                (f"potential_vulnerabilities[{i}].interpretation", item.interpretation)]
    return out


def validate_narrative(narrative: AnalystNarrative, facts: dict[str, Any]) -> list[str]:
    """Return every grounding violation (an empty list means the narrative is acceptable)."""
    problems: list[str] = []
    teams = {facts["matchup"]["home_team"], facts["matchup"]["away_team"]}
    if narrative.favored_team != facts["model"]["favorite"]:
        problems.append(f"favored_team is '{narrative.favored_team}' but the statistical model favours "
                        f"'{facts['model']['favorite']}'; the analyst may not change the prediction")
    known_ids = set(facts["metrics"])
    for kind, items in (("key_statistical_advantages", narrative.key_statistical_advantages),
                        ("potential_vulnerabilities", narrative.potential_vulnerabilities)):
        for i, item in enumerate(items):
            if item.team not in teams:
                problems.append(f"{kind}[{i}].team '{item.team}' is not one of {sorted(teams)}")
            if not item.metric_ids:
                problems.append(f"{kind}[{i}] cites no metric ids")
            unknown = [m for m in item.metric_ids if m not in known_ids]
            if unknown:
                problems.append(f"{kind}[{i}] cites metric ids that are not on the fact sheet: {unknown}")
    if not narrative.key_statistical_advantages:
        problems.append("key_statistical_advantages is empty")
    allowed = allowed_numbers(facts)
    for name, text in _texts(narrative):
        if not text.strip():
            problems.append(f"{name} is empty")
        missing = ungrounded_numbers(text, allowed)
        if missing:
            problems.append(f"{name} contains numbers that are not on the fact sheet: {missing}")
        lowered = text.lower()
        banned = [term for term in FORBIDDEN_TERMS if term in lowered]
        if banned:
            problems.append(f"{name} discusses topics with no supplied data: {banned}")
    return problems


def parse_narrative_json(text: str) -> AnalystNarrative:
    """Parse and schema-validate a JSON string produced by the LLM."""
    try:
        return AnalystNarrative.model_validate_json(text)
    except ValidationError as exc:
        raise LLMResponseError(f"LLM output does not match the AnalystNarrative schema: {exc.error_count()} error(s); "
                               f"first: {exc.errors()[0]['msg']} at {exc.errors()[0]['loc']}") from exc


# ---------------------------------------------------------------- rule-based analyst
def _edge(facts: dict[str, Any], stat: str) -> float | None:
    """Standardised home-minus-away edge on ``stat`` (positive favours the home team)."""
    m = facts["metrics"]
    h, a = m.get(f"home.{stat}"), m.get(f"away.{stat}")
    _, _, _, _, _, higher_is_better, spread = _SPEC[stat]
    if h is None or a is None or higher_is_better is None:
        return None
    return (h - a) / spread * (1.0 if higher_is_better else -1.0)


def _unit(stat: str) -> str:
    """Display unit: percentages get a % sign; free-throw rates are a ratio (makes per 100 attempts)."""
    return "%" if _SPEC[stat][3] == 100 and stat not in ("ft_rate", "opp_ft_rate") else ""


def rule_based_narrative(facts: dict[str, Any]) -> AnalystNarrative:
    """Deterministic analyst: same schema as the LLM, built from templates over the fact sheet."""
    mu, model, m, unc = facts["matchup"], facts["model"], facts["metrics"], facts["uncertainty"]
    home, away = mu["home_team"], mu["away_team"]
    fav, dog = model["favorite"], model["underdog"]
    venue = "on a neutral floor" if mu["neutral_site"] else f"at {home}"

    edges = []
    for category, stat, noun in ADVANTAGE_STATS:
        z = _edge(facts, stat)
        if z is not None:
            edges.append((abs(z), z, category, stat, noun))
    edges.sort(reverse=True)

    def describe(z: float, stat: str, noun: str) -> tuple[str, str]:
        leader, trailer = (home, away) if z > 0 else (away, home)
        side_l, side_t = ("home", "away") if z > 0 else ("away", "home")
        u = _unit(stat)
        return leader, (f"{leader} hold the edge in {noun}: {m[f'{side_l}.{stat}']}{u} against "
                        f"{trailer}'s {m[f'{side_t}.{stat}']}{u}.")

    advantages = []
    for _, z, category, stat, noun in [e for e in edges if e[0] >= 0.5][:4] or edges[:2]:
        team, statement = describe(z, stat, noun)
        advantages.append(StatAdvantage(team=team, category=category, metric_ids=[f"home.{stat}", f"away.{stat}"],
                                        statement=statement, interpretation=_INTERPRETATION[category]))

    def offence(att: str, dfn: str, att_team: str, dfn_team: str) -> str:
        return (f"{att_team} score an estimated {m[f'{att}.off_rtg']} points per 100 possessions and {dfn_team} allow "
                f"{m[f'{dfn}.def_rtg']}. {att_team} shoot {m[f'{att}.efg']}% effective from the field while {dfn_team} "
                f"concede {m[f'{dfn}.opp_efg']}%, and {att_team} turn the ball over on {m[f'{att}.tov_pct']}% of plays "
                f"against a defence forcing turnovers on {m[f'{dfn}.opp_tov_pct']}%.")

    def defence(dfn: str, att: str, dfn_team: str, att_team: str) -> str:
        return (f"{dfn_team} secure {m[f'{dfn}.drb_pct']}% of available defensive rebounds against a {att_team} side "
                f"that recovers {m[f'{att}.orb_pct']}% of its own misses, and allow a free-throw rate of "
                f"{m[f'{dfn}.opp_ft_rate']} to an offence that draws {m[f'{att}.ft_rate']}.")

    pace_label = facts["pace_label"]
    if pace_label == "unknown":
        pace_text = "Pace data is not available for this matchup."
    else:
        relation = {"fast": "quicker than", "slow": "slower than", "average": "close to"}[pace_label]
        pace_text = (f"{home} average an estimated {m['home.pace']} possessions per 48 minutes and {away} {m['away.pace']}, "
                     f"which points to a game of about {m['matchup.expected_pace']} possessions, {relation} the league "
                     f"average of {m['matchup.league_pace']}.")

    vulnerabilities = []
    for team, side, opp_side in ((home, "home", "away"), (away, "away", "home")):
        sign = 1.0 if side == "home" else -1.0
        worst = [e for e in sorted(edges, key=lambda e: sign * e[1]) if sign * e[1] < -0.5 and e[3] != "elo"][:1]
        for _, z, category, stat, noun in worst:
            u = _unit(stat)
            vulnerabilities.append(Vulnerability(
                team=team, metric_ids=[f"{side}.{stat}", f"{opp_side}.{stat}"],
                statement=f"{team} trail in {noun}: {m[f'{side}.{stat}']}{u} against {m[f'{opp_side}.{stat}']}{u}.",
                interpretation="This is the area where the opponent's profile matches up best against them."))
        if m[f"{side}.back_to_back"] or m[f"{side}.third_game_in_four_nights"]:
            kind = "the second night of a back-to-back" if m[f"{side}.back_to_back"] else "a third game in four nights"
            vulnerabilities.append(Vulnerability(
                team=team, metric_ids=[f"{side}.rest_days", f"{side}.games_last_7d"],
                statement=f"{team} play on {kind}, with {m[f'{side}.games_last_7d']} games in the previous 7 days.",
                interpretation="Compressed schedules historically cost teams efficiency, especially late in games."))

    drivers = unc["volatility_drivers"]
    driver_text = "; ".join(d["label"].lower() for d in drivers[:3]) or "no single factor stands out"
    lead = advantages[0].statement if advantages else ""
    opening = f"{away} and {home} meet {venue}" if mu["neutral_site"] else f"{away} visit {home}"
    overview = (f"{opening} on {mu['game_date']}. The calibrated ensemble makes {fav} the favourite at "
                f"{model['favorite_win_probability_pct']}%. {home} are {m['home.wins']}-{m['home.losses']} with an Elo "
                f"rating of {m['home.elo']}; {away} are {m['away.wins']}-{m['away.losses']} at {m['away.elo']}.")
    confidence = (f"Confidence is {unc['confidence_level']} ({unc['confidence_score']} out of 100): the favourite's win "
                  f"probability is {model['favorite_win_probability_pct']}% and the two models differ by "
                  f"{unc['model_disagreement_pct']} percentage points.")
    volatility = (f"Volatility risk is {unc['volatility_level']} ({unc['volatility_score']} out of 100). Main drivers: "
                  f"{driver_text}.")
    preview = (f"{fav} enter as the model's pick over {dog} at {model['favorite_win_probability_pct']}%. {lead} "
               f"{pace_text} {volatility}")
    return AnalystNarrative(
        favored_team=fav, matchup_overview=overview, key_statistical_advantages=advantages,
        offensive_matchup=SideAnalysis(home=offence("home", "away", home, away), away=offence("away", "home", away, home)),
        defensive_matchup=SideAnalysis(home=defence("home", "away", home, away), away=defence("away", "home", away, home)),
        pace_analysis=pace_text, potential_vulnerabilities=vulnerabilities,
        confidence_rationale=confidence, volatility_rationale=volatility, narrative_preview=preview.strip(),
    )


# ---------------------------------------------------------------------- LLM analyst
SYSTEM_PROMPT = """You are an experienced NBA analyst writing a pre-game tactical breakdown for a \
sports-analytics product. A statistical model has already produced the forecast; your job is to \
explain it in basketball terms.

You receive a fact sheet as JSON. It is the only information you have about these teams.

How to use it:
- Every number you write must appear on the fact sheet, written at the same or coarser \
precision. Do not compute new numbers (no sums, differences or averages of your own); the \
"diff.*" metrics already give home-minus-away differences.
- For each statistical advantage and vulnerability, list the metric ids that back it in \
"metric_ids", put what the data says in "statement", and put your basketball reading of it \
in "interpretation". Keep those two separate: statements are data, interpretations are judgement.
- The forecast is fixed. Set "favored_team" to the model's favourite and write the analysis \
so it is consistent with the model's probabilities, confidence and volatility. You may \
explain why the game could go the other way; you may not pick a different winner.
- Do not mention players, injuries, lineups, trades, coaching decisions or anything else \
that is not on the fact sheet. That information was not provided, and a reader could mistake \
a guess for a fact.
- Metric definitions say whether a metric is observed, estimated or a model output. Ratings \
and pace are estimates; say "estimated" when you first use them.

Style: write like a sharp television analyst - concrete, specific to this matchup, no hype \
and no filler. Three to five key advantages, one or two vulnerabilities per team, and a \
narrative preview of four to six sentences. Use team abbreviations exactly as given."""


def _fact_payload(facts: dict[str, Any]) -> str:
    """The JSON shown to the LLM (the raw, unrounded block is withheld)."""
    return json.dumps({k: v for k, v in facts.items() if k != "_raw"}, indent=1, sort_keys=True)


def llm_narrative(facts: dict[str, Any], cfg: LLMConfig | None = None, client: Any | None = None,
                  feedback: list[str] | None = None) -> AnalystNarrative:
    """Ask Claude for a narrative. Raises :class:`LLMError` subclasses on any failure.

    ``client`` may be injected (tests use a stub); otherwise an ``anthropic.Anthropic`` client is
    created, which resolves credentials from the environment.
    """
    cfg = cfg or LLMConfig()
    if not cfg.enabled:
        raise LLMUnavailableError("LLM analyst is disabled (NBA_LLM_ENABLED=0)")
    try:
        import anthropic
    except ImportError as exc:
        raise LLMUnavailableError("the 'anthropic' package is not installed (pip install anthropic)") from exc

    user = "Fact sheet:\n" + _fact_payload(facts)
    if feedback:
        user += ("\n\nYour previous answer was rejected by the automatic fact check for these reasons:\n- "
                 + "\n- ".join(feedback) + "\nWrite it again and fix every point.")
    try:
        if client is None:
            client = anthropic.Anthropic(timeout=cfg.timeout_seconds)
        response = client.beta.messages.parse(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # a safety-classifier decline is re-run on a fallback model
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": cfg.effort},
            output_format=AnalystNarrative,
        )
    except anthropic.AuthenticationError as exc:
        raise LLMUnavailableError("the Anthropic API rejected the credentials (check ANTHROPIC_API_KEY)") from exc
    except anthropic.PermissionDeniedError as exc:
        raise LLMUnavailableError(f"the API key lacks permission for this request: {exc.message}") from exc
    except anthropic.NotFoundError as exc:
        raise LLMUnavailableError(f"model '{cfg.model}' was not found: {exc.message}") from exc
    except anthropic.RateLimitError as exc:
        raise LLMUnavailableError("the Anthropic API rate limit was hit; try again later") from exc
    except anthropic.APITimeoutError as exc:
        raise LLMUnavailableError(f"the Anthropic API timed out after {cfg.timeout_seconds:.0f}s") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMUnavailableError("could not reach the Anthropic API (network error)") from exc
    except anthropic.APIStatusError as exc:
        raise LLMUnavailableError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except ValidationError as exc:
        raise LLMResponseError(f"LLM output failed schema validation ({exc.error_count()} error(s))") from exc
    except anthropic.AnthropicError as exc:
        raise LLMUnavailableError(f"Anthropic client error: {exc}") from exc
    except TypeError as exc:  # the SDK raises TypeError when no credentials can be resolved
        if "authentication" in str(exc).lower():
            raise LLMUnavailableError("no Anthropic credentials found - set ANTHROPIC_API_KEY in .env") from exc
        raise

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        raise LLMResponseError("the model declined to answer (stop_reason=refusal)")
    if stop_reason == "max_tokens":
        raise LLMResponseError("the answer was cut off at max_tokens; raise NBA_LLM_MAX_TOKENS")
    parsed = getattr(response, "parsed_output", None)
    if isinstance(parsed, AnalystNarrative):
        return parsed
    text = next((b.text for b in getattr(response, "content", []) if getattr(b, "type", "") == "text"), None)
    if not text:
        raise LLMResponseError("the response contained no text block")
    return parse_narrative_json(text)


# ------------------------------------------------------------------------ assembly
def assemble_report(narrative: AnalystNarrative, facts: dict[str, Any], source: str, model: str | None = None,
                    fallback_reason: str | None = None) -> TacticalAnalysis:
    """Combine the analyst's text with the statistical model's numbers and validate the result.

    Probabilities, confidence and volatility are read from the fact sheet's raw block, i.e.
    from the statistical models - never from the narrative.
    """
    raw, model_facts, m = facts["_raw"], facts["model"], facts["metrics"]
    pred, conf, vol = raw["prediction"], raw["confidence"], raw["volatility"]
    public_facts = {k: v for k, v in facts.items() if k != "_raw"}
    return TacticalAnalysis(
        meta=ReportMeta(schema_version=SCHEMA_VERSION, source=source, model=model,
                        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        fallback_reason=fallback_reason),
        matchup=facts["matchup"],
        prediction=PredictionBlock(
            favorite=model_facts["favorite"], home_win_probability=pred["p_home"], away_win_probability=1.0 - pred["p_home"],
            logistic_regression_probability=pred["p_lr"], xgboost_probability=pred["p_xgb"],
            elo_only_probability=pred["p_elo"], probability_band=(pred["p_low"], pred["p_high"]),
            expected_margin=pred["expected_margin"], manual_elo_adjustment=model_facts["manual_elo_adjustment"]),
        matchup_overview=narrative.matchup_overview,
        key_statistical_advantages=narrative.key_statistical_advantages,
        offensive_matchup=narrative.offensive_matchup, defensive_matchup=narrative.defensive_matchup,
        expected_pace=PaceBlock(estimate=m.get("matchup.expected_pace"), league_average=m.get("matchup.league_pace"),
                                label=facts["pace_label"], analysis=narrative.pace_analysis),
        potential_vulnerabilities=narrative.potential_vulnerabilities,
        prediction_confidence=ConfidenceBlock(level=conf["level"], score=conf["score"], rationale=narrative.confidence_rationale),
        volatility_risk=VolatilityBlock(score=vol["score"], level=vol["level"], drivers=vol["drivers"],
                                        rationale=narrative.volatility_rationale),
        narrative_preview=narrative.narrative_preview,
        data_limitations=DATA_LIMITATIONS,
        fact_sheet=public_facts,
    )


def analyze_matchup(prediction: Any, use_llm: bool = True, cfg: LLMConfig | None = None,
                    client: Any | None = None) -> TacticalAnalysis:
    """Produce the structured tactical report for one prediction.

    Tries the LLM (with one retry when the fact check rejects the first answer) and otherwise
    returns the rule-based analysis with ``meta.fallback_reason`` explaining why.
    """
    cfg = cfg or LLMConfig()
    facts = build_fact_sheet(prediction)
    reason: str | None = None
    if use_llm and cfg.enabled:
        try:
            feedback: list[str] | None = None
            for attempt in range(1, cfg.max_attempts + 1):
                narrative = llm_narrative(facts, cfg, client, feedback)
                feedback = validate_narrative(narrative, facts)
                if not feedback:
                    return assemble_report(narrative, facts, source="llm", model=cfg.model)
                logger.warning("LLM analysis rejected by the fact check (attempt %d): %s", attempt, feedback)
            raise LLMGroundingError(f"LLM answer failed the fact check {cfg.max_attempts} times: {feedback[0]}")
        except LLMError as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("LLM analyst unavailable, using the rule-based analyst (%s)", reason)
    else:
        reason = "LLM analyst disabled" if not cfg.enabled else "LLM analyst not requested"

    narrative = rule_based_narrative(facts)
    problems = validate_narrative(narrative, facts)
    if problems:  # would be a bug in the templates, never a data condition to hide
        raise RuntimeError(f"rule-based analysis failed its own fact check: {problems}")
    return assemble_report(narrative, facts, source="rule_based", fallback_reason=reason)
