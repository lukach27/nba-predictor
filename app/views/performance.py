"""Model performance: hold-out metrics vs baselines, calibration, accuracy over time, live ledger."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import common
from nba_predictor.ledger import grade_ledger, ledger_metrics, load_ledger

predictor = common.predictor()
version = common.data_version()
info = common.metrics(version)
backtest = common.backtest(version)
common.page_header("Model performance", "Scored on seasons the model never saw while it was being built, against honest baselines.")

NAMES = {"ensemble": "Ensemble (calibrated)", "lr": "Logistic regression", "xgb": "XGBoost", "ensemble_raw": "Ensemble before calibration",
         "elo": "Elo only (tuned)", "original_elo": "Original project Elo", "home_rate": "Always the home-win rate"}
season_label = lambda s: f"{s}-{(s + 1) % 100:02d}"  # noqa: E731
test = info.get("test") or {}
if not test.get("models"):
    st.warning("No hold-out seasons were available when the model was trained, so only cross-validation results exist.")
    test = info["cv"]

models = test["models"]
ens, elo = models["ensemble"], models["elo"]
data = info["data"]
st.subheader(f"Hold-out test: {', '.join(season_label(s) for s in data['test_seasons'])} ({ens['n']:,} games)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", common.pct(ens["accuracy"], 1), f"{100 * (ens['accuracy'] - elo['accuracy']):+.1f} pts vs Elo only")
m2.metric("Log loss", f"{ens['log_loss']:.4f}", f"{ens['log_loss'] - elo['log_loss']:+.4f} vs Elo only", delta_color="inverse")
m3.metric("Brier score", f"{ens['brier']:.4f}", f"{ens['brier'] - elo['brier']:+.4f} vs Elo only", delta_color="inverse")
m4.metric("Calibration error (ECE)", f"{100 * ens['ece']:.1f} pts", f"{100 * (ens['ece'] - elo['ece']):+.1f} pts vs Elo only", delta_color="inverse",
          help="Average gap between predicted probability and observed frequency.")

comparison = test["vs_reference"].get("ensemble")
if comparison:
    ll = comparison["log_loss"]
    verdict = "a statistically significant improvement" if ll["significant"] and ll["difference"] < 0 else "not statistically distinguishable"
    st.markdown(f"Against pure Elo on the same games, the ensemble changes log loss by **{ll['difference']:+.4f}** "
                f"(95% paired-bootstrap interval {ll['low']:+.4f} to {ll['high']:+.4f}): **{verdict}**.")
    original = test["vs_reference"].get("original_elo")
    if original:
        common.note(f"For reference, the original project's Elo is {original['log_loss']['difference']:+.4f} against the tuned Elo "
                    f"(interval {original['log_loss']['low']:+.4f} to {original['log_loss']['high']:+.4f}).")

rows = []
for key, name in NAMES.items():
    if key in models:
        m = models[key]
        rows.append({"Model": name, "Accuracy": 100 * m["accuracy"], "Accuracy 95% CI": f"{100 * m['ci']['accuracy']['low']:.1f} - {100 * m['ci']['accuracy']['high']:.1f}",
                     "Log loss": m["log_loss"], "Brier": m["brier"], "ROC-AUC": m["roc_auc"], "ECE (pts)": 100 * m["ece"],
                     "Calibration slope": m["calibration_slope"]})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
             column_config={"Accuracy": st.column_config.NumberColumn(format="%.1f%%"), "Log loss": st.column_config.NumberColumn(format="%.4f"),
                            "Brier": st.column_config.NumberColumn(format="%.4f"), "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
                            "ECE (pts)": st.column_config.NumberColumn(format="%.1f"),
                            "Calibration slope": st.column_config.NumberColumn(format="%.2f", help="1.0 is ideal. Below 1 = over-confident, above 1 = too cautious.")})
common.note("Lower is better for log loss, Brier and ECE. Accuracy depends on how lopsided a season is, so compare models on "
            "the same games rather than across seasons.")

st.subheader("Calibration")
left, right = st.columns([3, 2])
with left:
    shown = {k: NAMES[k] for k in ("ensemble", "elo", "original_elo") if k in test["calibration"]}
    common.show(charts.calibration_curves(test["calibration"], shown), key="calibration")
with right:
    st.markdown("**When the model says 70%, does the home team win 70% of the time?**")
    st.markdown("Points on the dotted diagonal are perfectly calibrated. Bins with fewer than 15 games are hidden.")
    scored = backtest[backtest["split"] == ("test" if (backtest["split"] == "test").any() else "cv")]
    common.show(charts.prediction_histogram(scored["p_ensemble"]), key="histogram")
    common.note("How often each probability is issued. Most games sit between 30% and 80%.")

st.subheader("Performance over time")
window = st.select_slider("Rolling window (games)", options=[100, 200, 300, 500], value=300)
series = {"ensemble": NAMES["ensemble"], "elo": NAMES["elo"], "original_elo": NAMES["original_elo"]}
rolling = pd.DataFrame({"game_date": backtest["game_date"]})
accuracy, brier = rolling.copy(), rolling.copy()
for key in series:
    p, y = backtest[f"p_{key}"], backtest["home_win"]
    accuracy[key] = ((p > 0.5) == (y == 1)).astype(float).rolling(window).mean()
    brier[key] = ((p - y) ** 2).rolling(window).mean()
accuracy, brier = accuracy.dropna(), brier.dropna()
split = backtest.loc[backtest["split"] == "test", "game_date"].min() if (backtest["split"] == "test").any() else None
t1, t2 = st.tabs(["Brier score", "Accuracy"])
with t1:
    common.show(charts.rolling_lines(brier, series, f"Brier score, trailing {window} games (lower is better)", False, split), key="roll_brier")
with t2:
    common.show(charts.rolling_lines(accuracy, series, f"Accuracy, trailing {window} games", True, split), key="roll_acc")
common.note("Before the dotted line: walk-forward validation (each season predicted by a model trained only on earlier seasons). "
            "After it: the frozen model on the hold-out seasons.")

st.subheader("By season")
per_season = pd.DataFrame(info["cv"]["per_season"] + (info.get("test") or {}).get("per_season", []))
metric = st.segmented_control("Metric", ["log_loss", "brier", "accuracy"], default="log_loss",
                              format_func={"log_loss": "Log loss", "brier": "Brier score", "accuracy": "Accuracy"}.get) or "log_loss"
common.show(charts.season_lines(per_season, metric, series, {"log_loss": "Log loss", "brier": "Brier score", "accuracy": "Accuracy"}[metric]), key="by_season")

st.subheader("Live track record")
ledger = load_ledger()
if ledger.empty:
    st.info("No forecasts have been logged yet. **Refresh data** in the sidebar (or `python -m nba_predictor update`) records a "
            "forecast for every game tipping off in the next two days, before it starts, and grades it afterwards.")
else:
    graded = grade_ledger(ledger, predictor.games)
    summary = ledger_metrics(graded)
    l1, l2, l3 = st.columns(3)
    l1.metric("Forecasts logged before tip-off", f"{len(graded):,}")
    l2.metric("Graded", f"{int(graded['graded'].sum()):,}")
    if summary:
        l3.metric("Live accuracy", common.pct(summary["ensemble"]["accuracy"], 1), f"Brier {summary['ensemble']['brier']:.3f}", delta_color="off")
    view = graded.sort_values("tipoff_utc", ascending=False).head(50)
    st.dataframe(pd.DataFrame({"Date": view["game_date"], "Game": view["away_team"] + " @ " + view["home_team"], "Home win": 100 * view["p_home"],
                               "Result": view["home_win"].map({1.0: "Home won", 0.0: "Away won"}).fillna("Pending"),
                               "Correct": view["correct"].map({1.0: "Yes", 0.0: "No", True: "Yes", False: "No"}).fillna(""),
                               "Logged (UTC)": view["predicted_at_utc"]}),
                 hide_index=True, width="stretch", column_config={"Home win": st.column_config.NumberColumn(format="%.0f%%")})
    common.note("A forecast is written only while tip-off is still in the future and is never overwritten.")

with st.expander("Evaluation protocol and remaining caveats", icon=":material/science:"):
    margin = info.get("margin_model") or {}
    hp = info["hyperparameters"]
    st.markdown(
        f"""
- **Development seasons** ({season_label(data['development_seasons'][0])} to {season_label(data['development_seasons'][-1])},
  {data['development_rows']:,} games): Elo parameters, feature-group selection, hyperparameters, the ensemble weight and the
  calibrators were all chosen here with walk-forward validation (validate on one season, train on all earlier ones).
- **Hold-out seasons** were scored once by the frozen model. The model deployed in this dashboard was then refitted on all
  {data['production_rows']:,} games with the same settings.
- **Tuned Elo:** K = {info['elo']['params']['k']:.0f}, home court = {info['elo']['params']['home_advantage']:.0f} points,
  {100 * info['elo']['params']['season_carry']:.0f}% carried between seasons, margin of victory
  {'on' if info['elo']['params']['use_mov'] else 'off'}.
- **Ensemble weight on XGBoost:** {100 * hp['ensemble_weight_xgb_dev']:.0f}% (the rest is logistic regression, C = {hp['lr_c']}).
- **Point-margin model:** mean absolute error {margin.get('mae', float('nan')):.2f} points on the hold-out
  (Elo-only linear baseline {margin.get('mae_elo_linear_baseline', float('nan')):.2f}).
- **Caveat:** no injury, lineup or betting-market information is used. Team-level models of this kind top out around 65-69% accuracy.
"""
    )
