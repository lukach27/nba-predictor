"""Inference service, uncertainty indicators and the Monte Carlo simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_predictor.elo import LOGIT_PER_ELO
from nba_predictor.features import build_features, hypothetical_fixtures
from nba_predictor.metrics import logit
from nba_predictor.predict import PredictionError
from nba_predictor.simulation import (SimulationError, SimulationInputs, build_simulation_inputs, game_leverage,
                                      simulate_season)
from nba_predictor.uncertainty import VOLATILITY_WEIGHTS, UncertaintyInputs, confidence, volatility


# -------------------------------------------------------------------- predictions
def test_matchup_prediction_is_a_valid_consistent_forecast(prediction):
    for name in ("p_home", "p_lr", "p_xgb", "p_elo", "p_low", "p_high"):
        assert 0 < getattr(prediction, name) < 1, name
    assert prediction.p_low <= prediction.p_high
    assert prediction.favorite in (prediction.home_team, prediction.away_team)
    assert prediction.favorite_probability == pytest.approx(max(prediction.p_home, 1 - prediction.p_home))
    assert prediction.favorite_probability >= 0.5
    assert {c["feature"] for c in prediction.contributions} == set(prediction.features)
    assert prediction.home_facts["team"] == "BOS" and prediction.away_facts["team"] == "LAL"


def test_prediction_uses_the_same_features_as_the_shared_pipeline(predictor, games):
    """Inference parity: the Predictor's inputs equal build_features() called directly."""
    date = predictor.last_played_date + pd.Timedelta(days=1)
    result = predictor.predict_matchup("BOS", "LAL", game_date=date)
    home, away = predictor.resolve_team("BOS"), predictor.resolve_team("LAL")
    direct = build_features(games, hypothetical_fixtures([(home, away)], date, predictor.current_season),
                            elo_params=predictor.bundle.elo_params, cfg=predictor.bundle.feature_config).matchups
    row = direct[~direct["is_played"]].iloc[0]
    for feature, value in result.features.items():
        if value is None:  # a feature with no history yet is missing in both paths
            assert np.isnan(row[feature]), feature
        else:
            assert value == pytest.approx(float(row[feature])), feature


def test_neutral_site_forecast_is_symmetric(predictor):
    ab = predictor.predict_matchup("BOS", "LAL", is_neutral=True)
    ba = predictor.predict_matchup("LAL", "BOS", is_neutral=True)
    assert ab.p_home + ba.p_home == pytest.approx(1.0, abs=1e-9)
    assert ab.p_lr + ba.p_lr == pytest.approx(1.0, abs=1e-9)
    home_court = predictor.predict_matchup("BOS", "LAL")
    assert home_court.p_home > ab.p_home, "removing home court must lower the home team's chances"


def test_manual_elo_adjustment_shifts_log_odds_exactly(predictor):
    base = predictor.predict_matchup("BOS", "LAL")
    adjusted = predictor.predict_matchup("BOS", "LAL", elo_adjust={"BOS": -80})
    assert adjusted.p_home < base.p_home
    assert logit(adjusted.p_home) - logit(base.p_home) == pytest.approx(-80 * LOGIT_PER_ELO, abs=1e-6)
    assert adjusted.p_home_unadjusted == pytest.approx(base.p_home)
    assert adjusted.elo_adjustment == {"BOS": -80.0} and base.elo_adjustment == {}


def test_bad_requests_raise_clear_errors(predictor):
    with pytest.raises(PredictionError, match="Unknown team"):
        predictor.predict_matchup("XXX", "LAL")
    with pytest.raises(PredictionError, match="must differ"):
        predictor.predict_matchup("BOS", "BOS")
    with pytest.raises(PredictionError, match="before the last game"):
        predictor.predict_matchup("BOS", "LAL", game_date="2018-01-01")
    assert predictor.resolve_team("bos") == predictor.resolve_team("Team BOS") == 1610612738


def test_upcoming_scores_every_scheduled_fixture(predictor):
    upcoming = predictor.upcoming(days=None)
    assert len(upcoming) == 4 and upcoming["game_date"].is_monotonic_increasing
    assert upcoming[["p_ensemble", "p_lr", "p_xgb", "p_elo"]].apply(lambda c: c.between(0, 1)).all().all()
    assert len(predictor.upcoming(days=1)) == 1
    matchups = predictor.features.matchups
    fixture_row = matchups[matchups["game_id"] == upcoming.loc[0, "game_id"]].iloc[0]
    explained = predictor.predict_matchup(upcoming.loc[0, "home_team"], upcoming.loc[0, "away_team"], fixture_row=fixture_row)
    assert explained.p_home == pytest.approx(upcoming.loc[0, "p_ensemble"])


def test_team_snapshot_and_pairwise_matrix(predictor):
    snapshot = predictor.team_snapshot()
    assert len(snapshot) == len(predictor.teams) and snapshot["elo"].is_monotonic_decreasing
    ids, matrix = predictor.pairwise_logits()
    assert matrix.shape == (len(ids), len(ids)) and np.isfinite(matrix).all()
    assert np.allclose(np.diag(matrix), 0.0)


# -------------------------------------------------------------------- uncertainty
def test_confidence_and_volatility_behave_sensibly():
    coin_flip = UncertaintyInputs(0.5, 0.5, 0.5, 0.0, 40, 0.40, 0.40, False)
    lock = UncertaintyInputs(0.9, 0.9, 0.9, 0.0, 40, 0.40, 0.40, False)
    shaky = UncertaintyInputs(0.9, 0.95, 0.80, 0.05, 3, 0.48, 0.40, True)
    assert confidence(coin_flip)["level"] == "low" and confidence(lock)["level"] == "high"
    assert confidence(shaky)["score"] < confidence(lock)["score"]
    assert volatility(coin_flip)["score"] > volatility(lock)["score"]
    assert volatility(shaky)["score"] > volatility(lock)["score"]
    for case in (coin_flip, lock, shaky):
        v = volatility(case)
        assert 0 <= v["score"] <= 100 and v["score"] == pytest.approx(sum(d["points"] for d in v["drivers"]), abs=0.2)
    assert sum(VOLATILITY_WEIGHTS.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------- simulation
def make_inputs(n_fixtures: int = 300, logit_scale: float = 0.8, seed: int = 0, unscheduled: int = 0) -> SimulationInputs:
    rng = np.random.default_rng(seed)
    n = 30
    conference = np.array(["East"] * 15 + ["West"] * 15)
    home = rng.integers(0, n, n_fixtures)
    away = (home + rng.integers(1, n, n_fixtures)) % n
    strength = rng.normal(0, 0.5, n)
    pair = strength[:, None] - strength[None, :] + 0.25
    np.fill_diagonal(pair, 0.0)
    return SimulationInputs(
        team_ids=np.arange(n) + 1000, team_abbr=[f"T{i:02d}" for i in range(n)], conference=conference,
        wins=rng.integers(0, 10, n), losses=rng.integers(0, 10, n), conf_wins=np.zeros(n, int),
        fixture_home=home, fixture_away=away, fixture_logit=rng.normal(0.2, logit_scale, n_fixtures),
        fixture_ids=[f"g{i}" for i in range(n_fixtures)], fixture_dates=["2026-12-01"] * n_fixtures,
        unscheduled=np.full(n, unscheduled), pair_logit=pair,
    )


def test_simulation_is_reproducible_for_a_seed_and_varies_across_seeds():
    inputs = make_inputs()
    a, b, c = (simulate_season(inputs, 500, seed) for seed in (7, 7, 8))
    assert np.array_equal(a.wins, b.wins) and np.array_equal(a.champion, b.champion)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert not np.array_equal(a.wins, c.wins)


@pytest.mark.parametrize("hot", [True, False])
def test_simulation_conserves_games_and_probabilities(hot):
    inputs = make_inputs(unscheduled=2)
    result = simulate_season(inputs, 400, seed=1, hot=hot)
    # every simulated fixture produces exactly one win
    fixture_wins = result.wins.sum(axis=1) - inputs.wins.sum()
    assert (fixture_wins >= len(inputs.fixture_home)).all() and (fixture_wins <= len(inputs.fixture_home) + 2 * 30).all()
    table = result.table
    assert (table["mean_wins"] + table["mean_losses"]).to_numpy() == pytest.approx(table["games_total"].to_numpy())
    assert table["p_champion"].sum() == pytest.approx(1.0)
    assert table.groupby("conference")["p_playoffs"].sum().to_numpy() == pytest.approx([8.0, 8.0])
    assert table.groupby("conference")["p_finals"].sum().to_numpy() == pytest.approx([1.0, 1.0])
    assert table.groupby("conference")["p_top6"].sum().to_numpy() == pytest.approx([6.0, 6.0])
    assert (result.made_playoffs.sum(axis=1) == 16).all()
    for col in ("p_playoffs", "p_second_round", "p_conf_finals", "p_finals", "p_champion"):
        assert table[col].between(0, 1).all()
    assert (table["p_champion"] <= table["p_finals"] + 1e-12).all() and (table["p_finals"] <= table["p_playoffs"] + 1e-12).all()
    for conf in ("East", "West"):
        ranks = result.rank[:, inputs.conference == conf]
        assert (np.sort(ranks, axis=1) == np.arange(1, 16)).all()  # ranks are a permutation of 1..15
    assert result.seed_distribution().sum(axis=1).to_numpy() == pytest.approx(np.ones(30))


def test_simulation_is_exact_when_outcomes_are_certain():
    inputs = make_inputs(n_fixtures=200)
    inputs.fixture_logit = np.full(200, 60.0)  # the home team always wins
    result = simulate_season(inputs, 50, seed=3, hot=False, simulate_playoffs=False)
    expected = inputs.wins + np.bincount(inputs.fixture_home, minlength=30)
    assert (result.wins == expected).all()
    assert result.fixture_results.all()
    assert (result.champion == -1).all() and result.table["p_champion"].sum() == 0


def test_simulated_win_totals_match_the_input_probabilities():
    inputs = make_inputs(n_fixtures=400, seed=5)
    result = simulate_season(inputs, 4000, seed=11, hot=False)
    p = 1 / (1 + np.exp(-inputs.fixture_logit))
    expected = inputs.wins + np.bincount(inputs.fixture_home, weights=p, minlength=30) + np.bincount(inputs.fixture_away, weights=1 - p, minlength=30)
    assert np.abs(result.wins.mean(axis=0) - expected).max() < 0.2


def test_hot_simulation_widens_the_distribution_of_records():
    inputs = make_inputs(n_fixtures=900, seed=2)
    hot = simulate_season(inputs, 2000, seed=4, hot=True, k_factor=20.0)
    static = simulate_season(inputs, 2000, seed=4, hot=False)
    assert hot.wins.std(axis=0).mean() > static.wins.std(axis=0).mean()


def test_leverage_reports_playoff_swings():
    inputs = make_inputs()
    result = simulate_season(inputs, 2000, seed=9)
    table = game_leverage(result, inputs, max_games=25)
    assert len(table) == 25 and table["leverage"].is_monotonic_decreasing
    assert (table["leverage"] >= 0).all() and table["p_home"].between(0, 1).all()
    assert (table["swing_home"] > -0.05).mean() > 0.9, "winning a game should not hurt a team's playoff odds"


def test_invalid_simulation_inputs_are_rejected():
    inputs = make_inputs()
    inputs.fixture_away = inputs.fixture_home.copy()
    with pytest.raises(SimulationError, match="itself"):
        simulate_season(inputs, 10)
    inputs = make_inputs()
    inputs.fixture_logit[0] = np.nan
    with pytest.raises(SimulationError, match="finite"):
        simulate_season(inputs, 10)
    with pytest.raises(SimulationError, match="n_sims"):
        simulate_season(make_inputs(), 0)


def test_simulation_inputs_from_the_predictor(predictor):
    inputs, k_factor = build_simulation_inputs(predictor)
    assert k_factor > 0 and len(inputs.fixture_home) == 4
    played = inputs.wins + inputs.losses
    assert played.sum() == 2 * 132  # each of the 132 games of the current season counts for two teams
    assert (played + np.bincount(np.concatenate([inputs.fixture_home, inputs.fixture_away]), minlength=12) + inputs.unscheduled == 82).all()
    result = simulate_season(inputs, 200, seed=1, k_factor=k_factor, simulate_playoffs=False)  # 6 teams per conference
    assert result.table["mean_wins"].between(0, 82).all()


def test_upcoming_carries_card_information(predictor):
    upcoming = predictor.upcoming(days=None)
    assert set(upcoming["confidence_level"]) <= {"low", "medium", "high"}
    assert set(upcoming["volatility_level"]) <= {"low", "medium", "high"}
    assert upcoming["confidence_score"].between(0, 100).all() and upcoming["volatility_score"].between(0, 100).all()
    assert upcoming["tipoff_utc"].notna().all()
    first = upcoming.iloc[0]
    single = predictor.predict_matchup(first["home_team"], first["away_team"],
                                       fixture_row=predictor.features.matchups.set_index("game_id", drop=False).loc[first["game_id"]])
    assert single.confidence["score"] == first["confidence_score"]
    assert single.volatility["score"] == first["volatility_score"]
