"""Matchup predictions: upcoming games, custom matchups, model comparison, explanation, analysis."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import charts
import common
from nba_predictor.config import LLMConfig
from nba_predictor.llm_model import build_fact_sheet, standardised_edges

predictor = common.predictor()
version = common.data_version()
common.page_header("Matchup predictions", "Calibrated pre-game win probabilities, how the models compare, and why.")

# ------------------------------------------------------------------ upcoming games
upcoming = common.upcoming(version, 10)
selected_fixture = None
if upcoming.empty:
    st.info("No scheduled games were found. Use **Refresh data** in the sidebar, or analyse a custom matchup below.")
else:
    st.subheader("Upcoming games")
    days = sorted(upcoming["game_date"].dt.date.unique())
    day = st.pills("Game day", days, default=days[0], format_func=lambda d: f"{d:%a %d %b}", label_visibility="collapsed")
    day = day or days[0]
    slate = upcoming[upcoming["game_date"].dt.date == day].reset_index(drop=True)
    pick_home = slate["p_ensemble"] >= 0.5
    table = pd.DataFrame(
        {
            "Matchup": slate["away_team"] + " @ " + slate["home_team"],
            "Home win": 100 * slate["p_ensemble"],
            "Model pick": slate["home_team"].where(pick_home, slate["away_team"]),
            "Pick prob.": (100 * slate["p_ensemble"].where(pick_home, 1 - slate["p_ensemble"])).round(0),
            "Exp. margin (home)": slate["expected_margin"].round(1),
            "Elo gap": slate["elo_diff"].round(0),
            "Logistic": (100 * slate["p_lr"]).round(0),
            "XGBoost": (100 * slate["p_xgb"]).round(0),
            "Elo only": (100 * slate["p_elo"]).round(0),
        }
    )
    event = st.dataframe(
        table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key=f"slate_{day}",
        column_config={
            "Home win": st.column_config.ProgressColumn("Home win", format="%.0f%%", min_value=0, max_value=100),
            "Pick prob.": st.column_config.NumberColumn(format="%.0f%%"),
            "Exp. margin (home)": st.column_config.NumberColumn(format="%+.1f", help="Predicted home points minus away points."),
            "Elo gap": st.column_config.NumberColumn(format="%+.0f", help="Home Elo minus away Elo, before home-court advantage."),
            "Logistic": st.column_config.NumberColumn(format="%.0f%%", help="Home win probability, logistic regression."),
            "XGBoost": st.column_config.NumberColumn(format="%.0f%%", help="Home win probability, XGBoost."),
            "Elo only": st.column_config.NumberColumn(format="%.0f%%", help="Home win probability from Elo ratings alone."),
        },
    )
    common.note("Select a row to analyse that game. Probabilities refer to the home team.")
    chosen = event.selection.rows[0] if event.selection.rows else 0
    selected_fixture = slate.iloc[chosen]

# ------------------------------------------------------------------ choose matchup
st.subheader("Matchup analysis")
options = common.team_options(predictor)
abbrs = list(options)
modes = ["Scheduled game", "Custom matchup"] if selected_fixture is not None else ["Custom matchup"]
mode = st.segmented_control("Source", modes, default=modes[0], label_visibility="collapsed") or modes[0]

if mode == "Scheduled game":
    home, away = selected_fixture["home_team"], selected_fixture["away_team"]
    game_date, neutral, fixture_id = None, bool(selected_fixture["is_neutral"]), selected_fixture["game_id"]
else:
    c1, c2, c3, c4 = st.columns([3, 3, 2, 2], vertical_alignment="bottom")
    home = c1.selectbox("Home team", abbrs, index=abbrs.index("BOS") if "BOS" in abbrs else 0, format_func=options.get)
    away = c2.selectbox("Away team", abbrs, index=abbrs.index("LAL") if "LAL" in abbrs else 1, format_func=options.get)
    first_day = (predictor.last_played_date + pd.Timedelta(days=1)).date()
    game_date = str(c3.date_input("Game date", value=first_day, min_value=first_day,
                                  help="Rest days and back-to-backs are computed from each team's last game."))
    neutral = c4.toggle("Neutral site", value=False, help="Removes home-court advantage by averaging both venue assignments.")
    fixture_id = None

with st.expander("What-if: adjust team strength", icon=":material/tune:"):
    common.note("The model has no injury or lineup data. If you know something it does not, shift a team's strength in Elo "
                "points (about 25 points per point of margin). The shift is applied on top of the model and shown separately.")
    a1, a2 = st.columns(2)
    adj_home = a1.slider(f"{home} adjustment", -150, 150, 0, 5, key="adj_home")
    adj_away = a2.slider(f"{away} adjustment", -150, 150, 0, 5, key="adj_away")

if home == away:
    st.warning("Choose two different teams.")
    st.stop()

adjustments = tuple((team, float(value)) for team, value in ((home, adj_home), (away, adj_away)) if value)
try:
    pred = common.matchup(version, home, away, game_date, neutral, adjustments, fixture_id)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# ------------------------------------------------------------------------ headline
venue = "neutral site" if pred["is_neutral"] else f"at {home}"
st.markdown(f'<div class="nba-kicker">{pred["game_date"]} · {venue}</div>', unsafe_allow_html=True)
common.probability_bar(home, away, pred["p_home"])

margin = pred["expected_margin"]
leader = home if (margin or 0) >= 0 else away
m1, m2, m3, m4 = st.columns(4)
m1.metric("Model pick", f"{pred['favorite']} {common.pct(pred['favorite_probability'])}")
m2.metric("Expected margin", "n/a" if margin is None else f"{leader} by {abs(margin):.1f}",
          help="From a separate XGBoost regression on the point margin. Typical error is about 11 points.")
m3.metric("Elo gap (home - away)", common.signed(pred["matchup_facts"]["elo_diff"], 0),
          help=f"Before home-court advantage, which is worth {pred['matchup_facts']['home_court_elo']:.0f} Elo points.")
m4.metric("80% model band", f"{common.pct(pred['p_low'])} - {common.pct(pred['p_high'])}",
          help="Range of the home win probability across bootstrap re-fits of the model. It measures model uncertainty, "
               "not the randomness of the game itself.")
b1, b2, _ = st.columns([2, 2, 5])
with b1:
    common.level_badge("Confidence", pred["confidence"]["level"], pred["confidence"]["score"])
with b2:
    common.level_badge("Volatility risk", pred["volatility"]["level"], pred["volatility"]["score"], high_is_good=False)
if pred["elo_adjustment"]:
    shifts = ", ".join(f"{team} {value:+.0f}" for team, value in pred["elo_adjustment"].items())
    st.info(f"Manual adjustment applied ({shifts} Elo). The unadjusted model gives {home} {common.pct(pred['p_home_unadjusted'], 1)}.",
            icon=":material/tune:")

facts = build_fact_sheet(pred)
tab_models, tab_why, tab_teams, tab_analysis = st.tabs(["Model comparison", "Why this pick", "Team comparison", "Tactical analysis"])

with tab_models:
    common.show(charts.model_comparison(pred), key="models")
    common.note("The ensemble blends logistic regression and XGBoost in log-odds and is then calibrated. "
                "Elo only is the baseline every model has to beat. The whisker on the ensemble is its 80% bootstrap band.")
    with st.expander("What drives the confidence and volatility scores"):
        conf = pred["confidence"]
        st.markdown(f"**Confidence {conf['score']:.0f}/100** - edge {conf['edge']:.2f} (distance from a coin flip), "
                    f"discounted by {100 * conf['penalty']:.0f}% for model disagreement "
                    f"({100 * conf['model_disagreement']:.1f} points) and bootstrap spread ({100 * conf['bootstrap_std']:.1f} points).")
        drivers = pd.DataFrame(pred["volatility"]["drivers"])[["label", "level", "weight", "points"]]
        st.dataframe(drivers.rename(columns={"label": "Volatility driver", "level": "Level (0-1)", "weight": "Weight", "points": "Points"}),
                     hide_index=True, width="stretch")
        common.note("Volatility is a transparent weighted index, not a probability. The weights are judgement calls.")

with tab_why:
    common.show(charts.contributions(pred["contributions"], home, away), key="why")
    common.note("Exact SHAP values of the calibrated forecast, in log-odds. Bars to the right push the probability toward "
                f"{home}, bars to the left toward {away}. They add up to the forecast together with the model's baseline.")

with tab_teams:
    common.show(charts.statistical_edges(standardised_edges(facts), home, away), key="edges")
    metrics = facts["metrics"]
    rows = []
    for stat, definition in facts["metric_definitions"].items():
        if f"home.{stat}" in metrics and metrics.get(f"home.{stat}") is not None:
            rows.append({"Statistic": definition["label"], home: metrics[f"home.{stat}"], away: metrics[f"away.{stat}"],
                         "Type": definition["source"]})
    record = lambda side: f"{metrics[f'{side}.wins']}-{metrics[f'{side}.losses']}"  # noqa: E731
    r1, r2 = st.columns(2)
    for column, side, team in ((r1, "home", home), (r2, "away", away)):
        column.markdown(f"**{team}** · {record(side)} · last 10: {metrics[f'{side}.last_10_record']} · "
                        f"home {metrics[f'{side}.home_record']} · road {metrics[f'{side}.away_record']}")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=420)
    common.note("Observed = computed directly from box scores. Estimated = depends on the possession estimate "
                "(FGA + 0.44 x FTA - OREB + TOV). All values are rolling pre-game averages.")

with tab_analysis:
    hint = LLMConfig.credentials_hint()
    has_key = hint != "none in environment"
    use_llm = st.toggle("Write the analysis with Claude", value=False, disabled=not has_key,
                        help="Uses the Anthropic API (billed to your API key). Off: the built-in rule-based analyst.")
    if not has_key:
        common.note("No Anthropic API key found, so the built-in rule-based analyst is used. "
                    "Add ANTHROPIC_API_KEY to a .env file to enable the LLM analyst.")
    report = common.tactical_report(version, json.dumps(pred, sort_keys=True, default=str), use_llm)
    meta = report["meta"]
    if meta["source"] == "llm":
        st.badge(f"Written by {meta['model']}, fact-checked against the data", icon=":material/verified:", color="green")
    elif meta["fallback_reason"] and use_llm:
        st.warning(f"The LLM analysis was not used: {meta['fallback_reason']}. Showing the rule-based analysis.")
    else:
        st.badge("Rule-based analyst", icon=":material/rule:", color="gray")

    st.markdown(f"#### {report['matchup_overview']}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Key statistical advantages**")
        for item in report["key_statistical_advantages"]:
            st.markdown(f"**{item['team']}** · {item['category'].replace('_', ' ')}  \n{item['statement']}  \n"
                        f"<span class='nba-note'>Analyst's read: {item['interpretation']}</span>", unsafe_allow_html=True)
    with right:
        st.markdown("**Potential vulnerabilities**")
        if not report["potential_vulnerabilities"]:
            common.note("None stand out in the data.")
        for item in report["potential_vulnerabilities"]:
            st.markdown(f"**{item['team']}**  \n{item['statement']}  \n"
                        f"<span class='nba-note'>Analyst's read: {item['interpretation']}</span>", unsafe_allow_html=True)
    o1, o2 = st.columns(2)
    with o1:
        st.markdown("**Offensive matchup**")
        st.markdown(f"{report['offensive_matchup']['home']}\n\n{report['offensive_matchup']['away']}")
    with o2:
        st.markdown("**Defensive matchup**")
        st.markdown(f"{report['defensive_matchup']['home']}\n\n{report['defensive_matchup']['away']}")
    pace = report["expected_pace"]
    st.markdown(f"**Expected pace: {pace['label']}**  \n{pace['analysis']}")
    st.markdown(f"**Prediction confidence: {report['prediction_confidence']['level']}**  \n{report['prediction_confidence']['rationale']}")
    st.markdown(f"**Volatility risk: {report['volatility_risk']['score']:.0f}/100**  \n{report['volatility_risk']['rationale']}")
    with st.container(border=True):
        st.markdown("**Preview**")
        st.markdown(report["narrative_preview"])
    for line in report["data_limitations"]:
        common.note(f"- {line}")
    with st.expander("Structured JSON"):
        st.json({k: v for k, v in report.items() if k != "fact_sheet"}, expanded=1)
        st.download_button("Download full report (JSON)", json.dumps(report, indent=2), file_name=f"{away}_at_{home}_{pred['game_date']}.json",
                           mime="application/json")
