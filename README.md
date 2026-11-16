# NBA Match Predictor

Calibrated pre-game win probabilities for NBA games, built from public team box scores
(`nba_api`). The project contains a leak-free feature pipeline, an improved Elo system, a
logistic regression + XGBoost ensemble with probability calibration, exact SHAP explanations,
a Monte Carlo season simulator, a fact-checked tactical analyst (LLM or rule-based) and a
Streamlit dashboard.

The original version of this project (a learning log in Croatian) is kept in
[docs/original_project_log_hr.md](docs/original_project_log_hr.md) and in the first git commit.

## Results

All modelling decisions were made on seasons 2015-16 to 2024-25 with walk-forward validation.
The model was then frozen and scored **once** on seasons it had never seen: 2025-26 and
2026-27 to date (1,421 regular-season games).

| Model | Accuracy | Log loss | Brier | ROC-AUC | Calibration error |
|---|---|---|---|---|---|
| **Ensemble (calibrated)** | 68.5% | **0.5940** | **0.2038** | 0.743 | 3.2 pts |
| XGBoost alone | 68.7% | 0.5968 | 0.2050 | 0.741 | 4.0 pts |
| Elo only (improved, tuned) | 68.6% | 0.5975 | 0.2053 | 0.742 | 3.1 pts |
| Original project's Elo | 67.6% | 0.6104 | 0.2109 | 0.722 | 2.5 pts |
| Always the home-win rate | 55.0% | 0.6887 | 0.2478 | 0.500 | 1.6 pts |

Accuracy has a 95% interval of roughly ±2.4 points on this sample (ensemble: 66.1-70.9%), so
the accuracy column cannot rank these models. Log loss and Brier score can, a little.

What these numbers do and do not show:

- **The improved Elo is the real gain.** Against the original Elo (K=20, nothing else) it
  lowers log loss by 0.013, and the 95% paired-bootstrap interval (0.007 to 0.018) excludes zero.
- **The extra features add a small edge on top of Elo.** In walk-forward validation
  (6,966 games) the ensemble beats pure Elo by 0.0069 log loss, interval 0.0039 to 0.0097.
  On the hold-out the difference is 0.0035 with an interval of -0.0033 to +0.0103: the same
  direction, but not statistically distinguishable at this sample size.
- **The roster-availability proxy is the most useful addition.** It measures how much of a
  team's recent rotation actually played in its last three games (from player game logs).
  Removing it makes validation log loss worse by 0.0018 (interval 0.0001 to 0.0036), and the
  hold-out log loss is 0.5940 with it against 0.5959 without.
- **Calibration error is about 3 points for every Elo-based model.** Calibration slopes on the
  hold-out are above 1 (ensemble 1.08), meaning forecasts were slightly too cautious in these
  two lopsided seasons.
- **XGBoost does not beat logistic regression here.** The ensemble weight chosen by validation
  is 0% XGBoost, so the deployed ensemble equals the calibrated logistic regression. Both models
  are still trained, compared and shown; their disagreement feeds the uncertainty indicators.
- **Test-season accuracy is higher than validation accuracy (68.6% vs 65.2%)** because the
  2025-26 season was more lopsided, not because the model improved. Compare models on the same
  games, not across seasons.
- **Point margin model** (XGBoost regression, same features): mean absolute error 11.4 points
  on the hold-out, practically the same as a one-variable Elo regression (11.45).

Without injury, lineup or betting-market data, team-level models top out around 65-69%.

## Quick start

Python 3.11+ (built and tested on 3.13, Windows).

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m nba_predictor fetch --players   # games, schedule, player logs: 2014-15 to today (about 2 minutes)
python -m nba_predictor train             # full training and evaluation protocol (about 3 minutes)
streamlit run app/main.py                 # dashboard at http://localhost:8501
python -m pytest                          # 94 tests
```

The repository already contains downloaded data (`data/raw/`) and a trained model (`models/`),
so the dashboard and the tests work right after installation.

### Commands

| Command | What it does |
|---|---|
| `python -m nba_predictor fetch [--full] [--players]` | Download game logs and the current schedule from the NBA stats API. `--players` also downloads player game logs; once that file exists it is refreshed automatically. |
| `python -m nba_predictor train [--no-roster]` | Tune Elo, select features, tune models, calibrate, evaluate, save artefacts. The roster group is offered to the ablation whenever player logs exist. |
| `python -m nba_predictor update` | Daily routine: fetch, rebuild features, log forecasts for the next two days, grade old ones |
| `python -m nba_predictor upcoming --days 3` | Forecasts for scheduled games |
| `python -m nba_predictor predict BOS LAL [--date YYYY-MM-DD] [--neutral] [--json]` | One matchup (home team first) |
| `python -m nba_predictor analyze BOS LAL [--no-llm]` | Structured tactical analysis as JSON |
| `python -m nba_predictor simulate --sims 10000 --seed 42 [--static]` | Season simulation |

Add `-v` before the command for progress logs.

### Optional: the LLM analyst

The tactical analysis works without any key (a deterministic rule-based analyst fills the same
JSON schema). To have Claude write it instead:

1. Copy `.env.example` to `.env`.
2. Put your Anthropic API key after `ANTHROPIC_API_KEY=`. API usage is billed separately from
   a Claude subscription.
3. In the dashboard, open a matchup, go to **Tactical analysis** and switch on
   **Write the analysis with Claude**.

Settings: `NBA_LLM_MODEL` (default `claude-opus-5-5`), `NBA_LLM_EFFORT` (default `medium`),
`NBA_LLM_ENABLED=0` to switch the LLM layer off entirely. `.env` is git-ignored.

## Architecture

```
NBA stats API ──> data/raw/games_raw.csv, schedule.csv          data_collection.py
                         │  clean_games(): one typed row per team per game
                         ▼
        box_metrics.py   Four Factors, possessions, off/def/net rating, pace (post-game)
        elo.py           Elo with home court, margin of victory, season regression
                         │
                         ▼
        features.py      build_features(games, fixtures) ──> pre-game features
                         │        (the same function for training rows and for predictions)
          ┌──────────────┴───────────────┐
          ▼                              ▼
   train_model.py                    predict.py ──> uncertainty.py, explain.py (SHAP)
   models/model_bundle.joblib            │
   models/metrics.json                   ├──> simulation.py   (Monte Carlo season + playoffs)
   models/backtest_predictions.csv       ├──> llm_model.py    (fact-checked analyst)
                                         ├──> ledger.py       (forecasts logged before tip-off)
                                         └──> app/            (Streamlit dashboard)
```

| Module (`src/nba_predictor/`) | Responsibility |
|---|---|
| `config.py` | Paths, season boundaries, feature groups, model and LLM settings. No secrets. |
| `data_collection.py` | Download with retries, schema validation, cleaning, neutral-site detection, fixtures |
| `box_metrics.py` | Single-game Four Factors and possession-based ratings |
| `elo.py` | Elo engine and parameter grid search |
| `features.py` | The one feature pipeline (team history, rest, head-to-head, matchup differences) |
| `roster.py` | Player game logs and the roster-availability proxy (optional feature group) |
| `metrics.py` | Accuracy, log loss, Brier, ROC-AUC, ECE, calibration tables, bootstrap intervals |
| `modeling.py` | Model definitions, Platt calibration, walk-forward folds, `ModelBundle` save/load |
| `train_model.py` | The training and evaluation protocol |
| `explain.py` | Exact SHAP values for XGBoost, logistic regression and the ensemble |
| `predict.py` | `Predictor`: scheduled games, arbitrary matchups, neutral sites, what-if adjustments |
| `uncertainty.py` | Confidence level and volatility risk index |
| `simulation.py` | Seeded Monte Carlo of the regular season, play-in and playoffs; game leverage |
| `ledger.py` | Append-only log of forecasts made before tip-off, graded later |
| `llm_model.py` | Fact sheet, JSON schema, grounding validator, Claude call, rule-based analyst |
| `cli.py` | Command-line interface |

`app/` holds the dashboard (`main.py`, `common.py` for theme and cached data access,
`charts.py` for Plotly figures, `views/` for the five pages). `tests/` holds the test suite.

## Data

| File | Content | Source |
|---|---|---|
| `data/raw/games_raw.csv` | One row per team per game, 2014-15 onward, all game types. Same 28 columns as the original project (`SEASON_ID, TEAM_ID, ..., PLUS_MINUS`). | `LeagueGameFinder` |
| `data/raw/schedule.csv` | Current season schedule: date, tip-off time (UTC), teams, status, neutral-site flag | `ScheduleLeagueV2` |
| `data/raw/player_logs.csv` | Minutes per player per regular-season game (season, player id, team id, game id, date, minutes). No names are stored. | `PlayerGameLogs` |
| `data/processed/*.csv` | Regenerated feature tables (git-ignored) | `features.py` |
| `data/predictions/ledger.csv` | Forecasts logged before tip-off | `ledger.py` |
| `models/` | Model bundle, metrics, backtest predictions | `train_model.py` |

Cleaning rules: NBA teams only; regular season, play-in, playoffs and Cup final (preseason and
All-Star games are dropped); unfinished games are dropped; every game must be one home row and
one away row. Neutral sites are the 2020 bubble (by date), games flagged neutral in the
schedule, and games the API lists as `A @ B` on both rows (its convention since 2024-25).
Earlier international games cannot be identified from the data and count as home games.

## Features

Every model input is a pre-game quantity. Team statistics are lagged with `shift(1)` inside
each (team, season) before any averaging, early-season values are blended with a prior built
from the previous season only, and Elo and head-to-head history are read before the game
updates them.

| Group | Inputs | Kept by ablation |
|---|---|---|
| `elo` | Elo rating gap | yes |
| `form` | Recent net rating and win % (5-game half-life), Elo change over 10 games | yes |
| `venue` | Home team's home-game form against the away team's road form | yes |
| `rest` | Rest days per team, rest gap, back-to-back flags | yes (the only group significant on its own) |
| `h2h` | Shrunk average margin of recent meetings (sign flips with home court) | yes (marginal) |
| `schedule` | Games in the last 7 days, third game in four nights, road-trip length | yes (marginal) |
| `roster` | Share of the recent rotation, minutes-weighted, that played in the last 3 games: per team and the gap | yes (significant) |
| `efficiency` | Net rating, offensive rating, blended win % | no |
| `four_factors` | eFG%, TOV%, ORB%, FT rate, and their defensive mirrors | no |
| `context` | Trailing league home win rate, season progress | no (it made validation worse) |

"No" means removing the group did not make walk-forward log loss worse, so it was dropped:
with a margin-of-victory Elo in the model, season-long efficiency and the Four Factors carry
no additional information about who wins. They are still computed, tested, shown in the
dashboard and given to the analyst.

**Observed vs estimated.** The Four Factors are exact ratios of box-score counts. Possessions
(`FGA + 0.44 x FTA - OREB + TOV`, averaged over both teams), pace and offensive/defensive/net
rating are estimates and are labelled as such everywhere.

**Roster availability** is a proxy, not an injury report: it only knows who played in earlier
games. It cannot know that a player returns tonight or was ruled out this morning.

**Tuned Elo:** K = 16, home court = 60 points, 60% of the distance from 1500 carried between
seasons, FiveThirtyEight margin-of-victory multiplier, playoffs included. The final model has
20 inputs; logistic regression uses C = 0.01.

## Training and evaluation protocol

1. **Development seasons** (2015-16 to 2024-25; 2014-15 only warms up Elo and priors). With
   walk-forward validation (validate on one season, train on all earlier ones) the pipeline
   tunes Elo, runs the group ablation and backward elimination, tunes logistic regression and
   XGBoost (40-configuration random search), picks the ensemble weight and fits Platt calibrators
   on out-of-fold scores.
2. **Hold-out seasons** (`config.TEST_SEASONS`) are scored once by the frozen model, next to
   three baselines, with bootstrap intervals and paired comparisons.
3. **Production model**: refitted on all seasons with the frozen settings, plus 15 bootstrap
   replicas whose spread is the model-uncertainty band.

The ensemble blends the two models' log-odds and applies one Platt calibration, which keeps
the final forecast exactly additive in feature contributions. That is why the SHAP values in
the dashboard add up to the forecast.

## Dashboard

| Page | Content |
|---|---|
| **Matchups** | Upcoming games, custom matchups (teams, date, neutral site), win probability, model comparison, Elo gap, confidence and volatility, SHAP explanation, team comparison, tactical analysis, what-if Elo adjustment |
| **Team analytics** | Elo trajectories, league Elo heat map (optional 3D surface), rolling offensive and defensive rating, Four Factors percentiles, recent games, home/road splits |
| **Explainability** | Global SHAP importance and summary, dependence plots, single-game explanations, ablation results |
| **Season simulator** | Projected standings, playoff / title odds, finishing-position heat map, one team's win distribution, highest-leverage games; configurable simulations and seed |
| **Model performance** | Hold-out metrics against baselines with intervals, calibration curve, rolling accuracy and Brier, per-season results, live ledger |

Streamlit was kept instead of moving to React + FastAPI: the dashboard calls the same Python
feature and model code directly, and a second language, a build chain and an API layer would
add surface without adding analysis.

## Season simulator

Each remaining fixture is drawn from its calibrated ensemble probability. With "hot"
simulation (default) a team's strength drifts inside each simulated season by the Elo update
rule, which widens the range of final records to something realistic. Simplifications: ties
are broken by conference record and then at random (not the official head-to-head rules);
playoff games reuse regular-season probabilities; games the NBA has not scheduled yet
(two per team until the Cup group stage ends) are played against a league-average opponent.

## The tactical analyst

`llm_model.py` builds a fact sheet of pre-rounded numbers from a prediction. The analyst (LLM
or rule-based) may only return text and references to fact ids. A validator rejects the answer
if it names a different favourite, cites an unknown metric, contains a number that is not on
the fact sheet, or mentions injuries or lineups. Probabilities, confidence and volatility in
the final JSON are copied from the statistical models. On any API error, refusal, malformed
answer or repeated validation failure the rule-based analyst is used and the report says why.

Known gap: the validator cannot recognise player names, so "do not name players" is enforced
by the prompt only.

## Tests

`python -m pytest` runs 94 tests in about 40 seconds, offline, on a synthetic league plus the
real artefacts when present. They cover: Four Factors and rating formulas; Elo (expected score,
zero-sum updates, home court, margin, season regression); temporal shifts; **no future
leakage** (changing a game's result must not change that game's or any earlier game's
features); **training/inference parity** (a played game rebuilt as an unplayed fixture gets
identical inputs, on synthetic and real data, roster values included); missing data; walk-forward splits; calibration;
model serialisation; SHAP additivity; simulator invariants and reproducibility; LLM schema,
grounding and failure handling; the ledger; data cleaning.

## Limitations

- No injury reports, lineups, trades, rest decisions or betting lines. The roster-availability
  proxy only sees who played in past games. This is the largest source of error and the reason
  the what-if adjustment exists.
- The hold-out is 1,421 games, so differences below roughly 0.007 log loss cannot be confirmed.
  The model's edge over a well-tuned Elo is real in validation but small.
- XGBoost adds nothing over logistic regression on these features.
- The margin model is no better than a linear Elo fit.
- Playoff simulation uses regular-season probabilities and simplified tie-breaks.
- Neutral-site games before 2024-25 (other than the 2020 bubble) are treated as home games.
- The live ledger only grows when `update` (or **Refresh data**) is run before games start.
- The LLM analyst needs a paid API key and was tested here against simulated responses only,
  not against the live API.
