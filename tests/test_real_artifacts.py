"""Integration checks against the real data and the trained model on disk.

Skipped automatically when ``data/raw/games_raw.csv`` or ``models/model_bundle.joblib`` is
missing (for example on a fresh clone before ``fetch`` and ``train`` have been run).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nba_predictor import config

pytestmark = pytest.mark.skipif(
    not (config.RAW_GAMES_PATH.exists() and config.BUNDLE_PATH.exists() and config.METRICS_PATH.exists()),
    reason="real data / trained model not present",
)


@pytest.fixture(scope="module")
def real_predictor():
    from nba_predictor.predict import Predictor

    return Predictor.from_disk()


def test_saved_model_loads_and_scores_real_games(real_predictor):
    matchups = real_predictor.features.matchups
    recent = matchups[matchups["is_played"] & (matchups["game_type"] == "regular")].tail(500)
    scored = real_predictor.score(recent)
    for column in ("p_ensemble", "p_lr", "p_xgb", "p_elo"):
        assert scored[column].between(0.01, 0.99).all(), column
    assert recent[real_predictor.bundle.feature_columns].notna().all().all()


def test_training_inference_parity_on_real_data(real_predictor):
    """A real played game, rebuilt as an unplayed fixture from earlier history only, gets identical inputs."""
    from nba_predictor.features import build_features

    bundle, games = real_predictor.bundle, real_predictor.games
    matchups = real_predictor.features.matchups
    played = matchups[matchups["is_played"] & (matchups["game_type"] == "regular") & ~matchups["is_neutral"]]
    for game in (played.iloc[-1], played.iloc[-200]):
        history = games[games["game_date"] < game["game_date"]]
        fixture = pd.DataFrame([{"game_id": game["game_id"], "game_date": game["game_date"], "season": game["season"],
                                 "game_type": "regular", "home_team_id": game["home_team_id"],
                                 "away_team_id": game["away_team_id"], "is_neutral": False}])
        rebuilt = build_features(history, fixture, elo_params=bundle.elo_params, cfg=bundle.feature_config).matchups
        row = rebuilt[rebuilt["game_id"] == game["game_id"]].iloc[0]
        np.testing.assert_allclose(row[bundle.feature_columns].to_numpy(dtype=float),
                                   game[bundle.feature_columns].to_numpy(dtype=float), rtol=1e-9, atol=1e-9)


def test_metrics_file_documents_a_clean_hold_out():
    metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    data = metrics["data"]
    assert max(data["development_seasons"]) < min(data["test_seasons"])
    assert set(data["test_seasons"]) <= set(config.TEST_SEASONS)
    test = metrics["test"]["models"]
    assert test["ensemble"]["log_loss"] < test["home_rate"]["log_loss"]
    assert test["ensemble"]["log_loss"] <= test["original_elo"]["log_loss"]
    assert 0.5 < test["ensemble"]["accuracy"] < 0.8, "an accuracy outside this range would suggest leakage or a bug"


def test_real_matchup_prediction_and_analysis(real_predictor):
    from nba_predictor.llm_model import TacticalAnalysis, analyze_matchup

    teams = real_predictor.teams["team_abbr"].tolist()
    prediction = real_predictor.predict_matchup(teams[0], teams[1])
    assert 0 < prediction.p_home < 1 and prediction.p_low <= prediction.p_high
    report = analyze_matchup(prediction, use_llm=False)
    assert TacticalAnalysis.model_validate(report.model_dump()) == report
    assert report.prediction.favorite == prediction.favorite


def test_real_season_simulation_is_valid_and_reproducible(real_predictor):
    from nba_predictor.simulation import build_simulation_inputs, simulate_season

    inputs, k_factor = build_simulation_inputs(real_predictor)
    a = simulate_season(inputs, 300, seed=5, k_factor=k_factor)
    b = simulate_season(inputs, 300, seed=5, k_factor=k_factor)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert a.table["p_champion"].sum() == pytest.approx(1.0)
    assert (a.table["games_total"] == config.GAMES_PER_SEASON).all()
    assert a.table.groupby("conference")["p_playoffs"].sum().to_numpy() == pytest.approx([8.0, 8.0])
