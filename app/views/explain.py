"""Model explainability: global SHAP importance, value effects, single-game explanations, ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import charts
import common
from nba_predictor.metrics import sigmoid

predictor = common.predictor()
version = common.data_version()
info = common.metrics(version)
common.page_header("Model explainability", "What the model looks at, how much each input matters, and why it made a given call.")

all_seasons = sorted(int(s) for s in predictor.games["season"].unique())[1:]
default = [s for s in all_seasons if s in info["data"]["test_seasons"]] or all_seasons[-1:]
label = lambda s: f"{s}-{(s + 1) % 100:02d}"  # noqa: E731
seasons = st.multiselect("Seasons to explain", all_seasons, default=default, format_func=label)
if not seasons:
    st.info("Select at least one season.")
    st.stop()
games, values, contribs = common.shap_frame(version, tuple(sorted(seasons)))
if games.empty:
    st.info("No completed games in the selected seasons.")
    st.stop()

importance = pd.DataFrame({"feature": contribs.columns, "mean_abs_ensemble": contribs.abs().mean().to_numpy()})
order = importance.sort_values("mean_abs_ensemble", ascending=False)["feature"].tolist()

st.subheader("What matters most")
left, right = st.columns(2)
with left:
    common.show(charts.importance_bars(importance), key="importance")
    common.note(f"Average absolute SHAP value over {len(games):,} games. Larger means the input moves forecasts more.")
with right:
    common.show(charts.shap_summary(values, contribs, order[:10]), key="summary")
    common.note("Each dot is one game. Position shows how far that input pushed the forecast toward the home team (right) "
                "or the away team (left); lighter dots are higher values of the input.")

st.subheader("How one input changes the forecast")
feature = st.selectbox("Input", order, format_func=common.feature_label)
common.show(charts.shap_dependence(values[feature], contribs[feature], feature), key="dependence")
weight = predictor.bundle.ensemble_weight
common.note(f"The ensemble puts {100 * weight:.0f}% of its weight on XGBoost and {100 * (1 - weight):.0f}% on logistic regression "
            "(chosen by walk-forward log loss). A straight line here means the effect is linear in log-odds.")

st.subheader("Explain a single game")
recent = games.tail(300).iloc[::-1].reset_index()
describe = lambda i: (f"{recent.loc[i, 'game_date']:%d %b %Y}  {recent.loc[i, 'away_team']} {recent.loc[i, 'away_pts']:.0f} @ "  # noqa: E731
                      f"{recent.loc[i, 'home_team']} {recent.loc[i, 'home_pts']:.0f}")
pick = st.selectbox("Game", recent.index, format_func=describe)
row = recent.loc[pick]
position = int(row["index"])
game_contribs = contribs.iloc[position]
probability = float(sigmoid(row["base"] + game_contribs.sum()))
home, away = row["home_team"], row["away_team"]
g1, g2, g3 = st.columns(3)
g1.metric(f"Pre-game forecast for {home}", common.pct(probability, 1))
g2.metric("Result", f"{home if row['home_win'] == 1 else away} won", f"{row['home_pts']:.0f}-{row['away_pts']:.0f}", delta_color="off")
g3.metric("Model baseline", common.pct(float(sigmoid(row["base"])), 1), help="The forecast when every input is at its average: mostly home-court advantage.")
records = [{"feature": f, "value": float(values.iloc[position][f]), "contribution": float(game_contribs[f])} for f in contribs.columns]
common.show(charts.contributions(records, home, away, top=10), key="single_game")
common.note("These are the inputs as they stood before tip-off. Baseline plus all contributions equals the forecast exactly "
            "(in log-odds).")

st.subheader("Which feature groups earn their place")
selection = info["feature_selection"]
st.markdown(f"**Kept:** {', '.join(selection['selected_groups'])}  \n**Dropped by ablation:** {', '.join(selection['dropped_groups']) or 'none'}")
a1, a2 = st.columns(2)
with a1:
    st.caption("Adding one group to an Elo-only model")
    if selection["add_one_to_elo"]:
        common.show(charts.ablation_bars(selection["add_one_to_elo"], "Change in log loss ×10⁻³ (left of zero = improves on Elo alone)"), key="add_one")
with a2:
    st.caption("Removing one group from the final model")
    if selection["drop_one_final"]:
        rows = [{**r, "delta_log_loss": r["delta_log_loss"], "low": r["low"], "high": r["high"]} for r in selection["drop_one_final"]]
        common.show(charts.ablation_bars(rows, "Change in log loss ×10⁻³ (right of zero = the model gets worse without it)"), key="drop_one")
common.note("Walk-forward validation on development seasons only; whiskers are 95% bootstrap intervals. A group is dropped when "
            "removing it does not make the forecast worse. Dropped groups are still computed and shown elsewhere in the dashboard - "
            "they simply carry no information the kept groups do not already have.")
with st.expander("Model inputs"):
    table = pd.DataFrame({"Input": [common.feature_label(f) for f in predictor.bundle.feature_columns], "Column": predictor.bundle.feature_columns,
                          "Mean |SHAP|": [float(np.abs(contribs[f]).mean()) for f in predictor.bundle.feature_columns]})
    st.dataframe(table.sort_values("Mean |SHAP|", ascending=False), hide_index=True, width="stretch",
                 column_config={"Mean |SHAP|": st.column_config.NumberColumn(format="%.3f")})
