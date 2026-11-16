"""LLM analyst (schema, grounding, failure handling), prediction ledger and data cleaning."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from nba_predictor import config
from nba_predictor.config import LLMConfig
from nba_predictor.data_collection import clean_games, current_season, data_status, team_directory, upcoming_fixtures
from nba_predictor.ledger import LEDGER_COLUMNS, grade_ledger, ledger_metrics, load_ledger, record_predictions
from nba_predictor.llm_model import (AnalystNarrative, LLMResponseError, TacticalAnalysis, allowed_numbers,
                                     analyze_matchup, build_fact_sheet, parse_narrative_json, rule_based_narrative,
                                     ungrounded_numbers, validate_narrative)

ENABLED = LLMConfig(enabled=True)


# ------------------------------------------------------------------- stub client
class StubClient:
    """Stands in for ``anthropic.Anthropic``: returns queued responses or raises queued errors."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def reply(narrative: AnalystNarrative | None = None, stop_reason: str = "end_turn", text: str | None = None):
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    return SimpleNamespace(stop_reason=stop_reason, parsed_output=narrative, content=content)


@pytest.fixture()
def facts(prediction):
    return build_fact_sheet(prediction)


@pytest.fixture()
def good(facts) -> AnalystNarrative:
    return rule_based_narrative(facts)


# ------------------------------------------------------------------------ schema
def test_rule_based_report_is_valid_and_grounded(prediction, facts, good):
    assert validate_narrative(good, facts) == []
    report = analyze_matchup(prediction, use_llm=False)
    assert report.meta.source == "rule_based" and report.meta.fallback_reason == "LLM analyst not requested"
    roundtrip = TacticalAnalysis.model_validate(json.loads(report.model_dump_json()))
    assert roundtrip == report
    assert report.key_statistical_advantages and report.narrative_preview and report.data_limitations
    assert 0 <= report.volatility_risk.score <= 100 and report.prediction_confidence.level in ("low", "medium", "high")


def test_schema_rejects_unknown_fields_and_bad_values(prediction):
    payload = analyze_matchup(prediction, use_llm=False).model_dump()
    with pytest.raises(ValidationError):
        TacticalAnalysis.model_validate({**payload, "invented_field": 1})
    broken = json.loads(json.dumps(payload))
    broken["volatility_risk"]["score"] = 140
    with pytest.raises(ValidationError):
        TacticalAnalysis.model_validate(broken)
    broken = json.loads(json.dumps(payload))
    broken["prediction"]["home_win_probability"] = 1.4
    with pytest.raises(ValidationError):
        TacticalAnalysis.model_validate(broken)


def test_malformed_llm_json_raises_a_response_error(good):
    with pytest.raises(LLMResponseError):
        parse_narrative_json("this is not json")
    with pytest.raises(LLMResponseError):
        parse_narrative_json(json.dumps({"favored_team": "BOS"}))  # required fields missing
    with pytest.raises(LLMResponseError):
        parse_narrative_json(json.dumps({**good.model_dump(), "win_probability": 0.99}))  # extra field
    assert parse_narrative_json(good.model_dump_json()) == good


# --------------------------------------------------------------------- grounding
def test_numbers_must_come_from_the_fact_sheet(facts):
    allowed = allowed_numbers(facts)
    pct = facts["model"]["favorite_win_probability_pct"]
    assert ungrounded_numbers(f"The favourite wins {pct}% of the time.", allowed) == []
    assert ungrounded_numbers(f"Roughly {round(pct)}% per 100 possessions over 48 minutes.", allowed) == []
    assert ungrounded_numbers("They shoot 87.3% and score 163.7 points.", allowed) == ["87.3", "163.7"]


def test_validator_catches_invented_statistics(facts, good):
    bad = good.model_copy(update={"narrative_preview": good.narrative_preview + " They have won 937 straight."})
    problems = validate_narrative(bad, facts)
    assert any("not on the fact sheet" in p and "937" in p for p in problems)


def test_validator_blocks_a_changed_prediction(facts, good):
    flipped = good.model_copy(update={"favored_team": facts["model"]["underdog"]})
    assert any("may not change the prediction" in p for p in validate_narrative(flipped, facts))


def test_validator_blocks_unknown_metrics_teams_and_injury_talk(facts, good):
    advantage = good.key_statistical_advantages[0]
    cases = {
        "not on the fact sheet": good.model_copy(update={"key_statistical_advantages": [advantage.model_copy(update={"metric_ids": ["home.vibes"]})]}),
        "is not one of": good.model_copy(update={"key_statistical_advantages": [advantage.model_copy(update={"team": "NYK"})]}),
        "cites no metric ids": good.model_copy(update={"key_statistical_advantages": [advantage.model_copy(update={"metric_ids": []})]}),
        "no supplied data": good.model_copy(update={"matchup_overview": "Their star is questionable with an ankle injury."}),
        "is empty": good.model_copy(update={"pace_analysis": "  "}),
    }
    for expected, narrative in cases.items():
        assert any(expected in p for p in validate_narrative(narrative, facts)), expected


# ------------------------------------------------------------- LLM call handling
def test_valid_llm_answer_is_used_and_cannot_touch_the_numbers(prediction, good):
    client = StubClient(reply(good))
    report = analyze_matchup(prediction, cfg=ENABLED, client=client)
    assert report.meta.source == "llm" and report.meta.model == ENABLED.model and report.meta.fallback_reason is None
    assert report.prediction.home_win_probability == pytest.approx(prediction.p_home)
    assert report.prediction.favorite == prediction.favorite
    assert report.volatility_risk.score == prediction.volatility["score"]
    assert report.prediction_confidence.score == prediction.confidence["score"]
    call = client.calls[0]
    assert call["output_format"] is AnalystNarrative and call["model"] == ENABLED.model
    sent = call["messages"][0]["content"]
    assert "_raw" not in sent and str(build_fact_sheet(prediction)["model"]["favorite_win_probability_pct"]) in sent


def test_rejected_answer_is_retried_with_feedback(prediction, good):
    bad = good.model_copy(update={"narrative_preview": "They score 163.7 points a night."})
    client = StubClient(reply(bad), reply(good))
    report = analyze_matchup(prediction, cfg=ENABLED, client=client)
    assert report.meta.source == "llm" and len(client.calls) == 2
    assert "rejected by the automatic fact check" in client.calls[1]["messages"][0]["content"]
    assert "163.7" in client.calls[1]["messages"][0]["content"]


def test_persistently_ungrounded_answer_falls_back(prediction, good, facts):
    bad = good.model_copy(update={"favored_team": facts["model"]["underdog"]})
    client = StubClient(reply(bad), reply(bad))
    report = analyze_matchup(prediction, cfg=ENABLED, client=client)
    assert report.meta.source == "rule_based" and "LLMGroundingError" in report.meta.fallback_reason
    assert report.prediction.favorite == prediction.favorite


def test_api_errors_refusals_and_truncation_fall_back_with_a_reason(prediction, good):
    import anthropic
    import httpx2 as httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    outcomes = {
        "network error": anthropic.APIConnectionError(request=request),
        "timed out": anthropic.APITimeoutError(request=request),
        "no Anthropic credentials": TypeError("Could not resolve authentication method."),
        "declined to answer": reply(None, stop_reason="refusal"),
        "cut off at max_tokens": reply(None, stop_reason="max_tokens"),
        "no text block": reply(None),
        "does not match the AnalystNarrative schema": reply(None, text='{"favored_team": 5}'),
    }
    for expected, outcome in outcomes.items():
        report = analyze_matchup(prediction, cfg=ENABLED, client=StubClient(outcome))
        assert report.meta.source == "rule_based", expected
        assert expected in report.meta.fallback_reason, (expected, report.meta.fallback_reason)
    # a text block holding valid JSON is accepted even when the SDK did not pre-parse it
    assert analyze_matchup(prediction, cfg=ENABLED, client=StubClient(reply(None, text=good.model_dump_json()))).meta.source == "llm"


def test_unexpected_programming_errors_are_not_swallowed(prediction):
    with pytest.raises(TypeError, match="unrelated"):
        analyze_matchup(prediction, cfg=ENABLED, client=StubClient(TypeError("unrelated bug")))


def test_llm_can_be_disabled_independently(prediction, good):
    client = StubClient(reply(good))
    report = analyze_matchup(prediction, cfg=LLMConfig(enabled=False), client=client)
    assert report.meta.source == "rule_based" and report.meta.fallback_reason == "LLM analyst disabled"
    assert client.calls == []


# ------------------------------------------------------------------------ ledger
def test_ledger_records_only_future_games_once_and_grades_them(predictor, games, tmp_path):
    path = tmp_path / "ledger.csv"
    upcoming = predictor.upcoming(days=None)
    first_tip = pd.Timestamp(predictor.schedule["game_time_utc"].iloc[0])
    now = (first_tip - pd.Timedelta(hours=2)).to_pydatetime()
    added = record_predictions(upcoming, predictor.schedule, "model-v1", now=now, horizon_days=1.5, path=path)
    assert len(added) == 2 and list(added.columns) == LEDGER_COLUMNS  # games 3 and 4 are beyond the horizon
    assert record_predictions(upcoming, predictor.schedule, "model-v1", now=now, horizon_days=1.5, path=path).empty
    after_tip = (first_tip + pd.Timedelta(hours=1)).to_pydatetime()
    later = record_predictions(upcoming, predictor.schedule, "model-v2", now=after_tip, horizon_days=10, path=path)
    assert set(later["game_id"]) == set(upcoming["game_id"].iloc[2:])  # started game is never back-filled
    ledger = load_ledger(path)
    assert len(ledger) == 4 and ledger["game_id"].is_unique
    assert (pd.to_datetime(ledger["predicted_at_utc"]) < pd.to_datetime(ledger["tipoff_utc"])).all()

    graded = grade_ledger(ledger, games)
    assert not graded["graded"].any() and ledger_metrics(graded) is None
    result = games[games["is_home"]].iloc[[0]].copy()
    result["game_id"] = ledger["game_id"].iloc[0]
    graded = grade_ledger(ledger, pd.concat([games, result]))
    assert graded["graded"].sum() == 1
    row = graded[graded["graded"]].iloc[0]
    assert row["correct"] == ((row["p_home"] > 0.5) == (row["home_win"] == 1))
    summary = ledger_metrics(graded)
    assert summary["n"] == 1 and summary["pending"] == 3 and summary["ensemble"]["accuracy"] in (0.0, 1.0)


def test_empty_ledger_has_the_right_columns(tmp_path):
    assert list(load_ledger(tmp_path / "missing.csv").columns) == LEDGER_COLUMNS


# --------------------------------------------------------------------- data layer
def test_clean_games_pairs_rows_and_drops_invalid_ones(raw_games):
    raw = raw_games.copy()
    clean = clean_games(raw)
    assert len(clean) == len(raw) and (clean.groupby("game_id")["is_home"].sum() == 1).all()
    assert (clean["PTS"] == clean.merge(clean, left_on=["game_id", "opp_team_id"], right_on=["game_id", "team_id"])["OPP_PTS_y"].to_numpy()).all()
    first, second = raw["GAME_ID"].unique()[:2]
    raw.loc[raw["GAME_ID"] == first, "WL"] = None  # unfinished game
    raw.loc[(raw["GAME_ID"] == second) & raw["MATCHUP"].str.contains("@"), "TEAM_ID"] = 12345  # non-NBA opponent
    preseason = raw[raw["GAME_ID"] == raw["GAME_ID"].unique()[2]].assign(GAME_ID="0011800001")
    cleaned = clean_games(pd.concat([raw, preseason]))
    assert first not in set(cleaned["game_id"]) and second not in set(cleaned["game_id"])
    assert "0011800001" not in set(cleaned["game_id"])
    assert cleaned["win"].isin([0.0, 1.0]).all() and cleaned["game_type"].eq("regular").all()


def test_neutral_site_games_are_detected(raw_games):
    raw = raw_games.copy()
    gid = raw["GAME_ID"].unique()[10]
    rows = raw["GAME_ID"] == gid
    home_abbr = raw.loc[rows & raw["MATCHUP"].str.contains("vs."), "TEAM_ABBREVIATION"].iloc[0]
    away_abbr = raw.loc[rows & raw["MATCHUP"].str.contains("@"), "TEAM_ABBREVIATION"].iloc[0]
    raw.loc[rows, "MATCHUP"] = f"{away_abbr} @ {home_abbr}"  # the API's neutral-site convention since 2024-25
    cleaned = clean_games(raw)
    game = cleaned[cleaned["game_id"] == gid]
    assert game["is_neutral"].all() and game["is_home"].sum() == 1
    assert game.loc[game["is_home"], "team_abbr"].iloc[0] == home_abbr
    bubble = raw_games.copy()
    bubble["GAME_DATE"] = "2020-08-05"
    assert clean_games(bubble)["is_neutral"].all()


def test_fixtures_directory_and_status_helpers(games, predictor):
    fixtures = upcoming_fixtures(predictor.schedule, games)
    assert len(fixtures) == 4 and (fixtures["game_date"] > games["game_date"].max()).all()
    finished = predictor.schedule.assign(game_status=3)
    assert upcoming_fixtures(finished, games).empty and upcoming_fixtures(None, games).empty
    directory = team_directory(games)
    assert len(directory) == 12 and set(directory["conference"]) == {"East", "West"}
    status = data_status(games, predictor.schedule)
    assert status["games"] == 660 and status["upcoming_fixtures"] == 4
    assert config.season_label(2024) == "2024-25" and config.season_label(1999) == "1999-00"
    assert current_season(datetime(2026, 11, 16, tzinfo=timezone.utc).date()) == 2026
    assert current_season(datetime(2027, 3, 1, tzinfo=timezone.utc).date()) == 2026
    assert np.isfinite(predictor.league_pace)
