"""Team analytics: Elo trajectories, rolling ratings, Four Factors, recent form, home/away splits."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import common

predictor = common.predictor()
version = common.data_version()
common.page_header("Team analytics", "How each team's strength has moved, and what its profile looks like today.")

snapshot = common.snapshot(version)
options = common.team_options(predictor)
abbrs = list(options)
rows = predictor.features.team_rows
played = rows[rows["is_played"]].copy()
played["team"] = played["team_id"].map(predictor.abbr)
seasons = sorted(int(s) for s in played["season"].unique())
label = lambda s: f"{s}-{(s + 1) % 100:02d}"  # noqa: E731

# ------------------------------------------------------------------ Elo trajectories
st.subheader("Elo trajectories")
c1, c2 = st.columns([3, 2])
default_teams = snapshot["team_abbr"].head(3).tolist()
teams = c1.multiselect("Teams (up to 6)", abbrs, default=default_teams, max_selections=common.MAX_TEAM_SERIES, format_func=options.get)
first, last = c2.select_slider("Seasons", options=seasons, value=(max(seasons[0], seasons[-1] - 4), seasons[-1]), format_func=label)
if teams:
    colours = common.stable_colours(teams, "elo_colours")
    history = played[played["team"].isin(teams) & played["season"].between(first, last)][["game_date", "team", "pre_elo"]]
    common.show(charts.elo_lines(history.rename(columns={"pre_elo": "elo"}), {t: colours[t] for t in teams}), key="elo_lines")
    common.note("Ratings step toward 1500 at the start of each season (the tuned model keeps "
                f"{100 * predictor.bundle.elo_params.season_carry:.0f}% of the gap), then move with each result and its margin.")
else:
    st.info("Select at least one team.")

with st.expander("League-wide Elo map", icon=":material/grid_on:"):
    v1, v2 = st.columns([2, 3])
    span = v1.segmented_control("Period", ["Last 3 seasons", "All seasons"], default="Last 3 seasons") or "Last 3 seasons"
    as_surface = v2.toggle("Show as 3D surface", value=False, help="Same data as the heat map. The flat map is easier to read precisely.")
    window = played[played["season"] >= seasons[-1] - 2] if span == "Last 3 seasons" else played
    monthly = window.assign(month=window["game_date"].dt.strftime("%Y-%m")).groupby(["team", "month"])["pre_elo"].mean().unstack("month")
    monthly = monthly.reindex(snapshot["team_abbr"]).ffill(axis=1)
    common.show(charts.elo_surface(monthly) if as_surface else charts.elo_heatmap(monthly), key="elo_map")
    common.note("Teams are ordered by current Elo, strongest at the top. Monthly average of pre-game ratings.")

# ---------------------------------------------------------------------- team profile
st.subheader("Team profile")
focus = st.selectbox("Team", abbrs, index=abbrs.index(teams[0]) if teams else 0, format_func=options.get)
team = snapshot[snapshot["team_abbr"] == focus].iloc[0]
rank = int(snapshot.index[snapshot["team_abbr"] == focus][0]) + 1
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Record", f"{team['wins']}-{team['losses']}")
m2.metric("Elo", f"{team['elo']:.0f}", help=f"Rank {rank} of {len(snapshot)}")
m3.metric("Net rating", common.signed(team["net_rtg"]), help="Estimated points per 100 possessions, blended with last season early in the year.")
m4.metric("Last 10", team["last_10_record"])
m5.metric("Elo change, last 10", common.signed(team["elo_trend"], 0))

left, right = st.columns([3, 2])
with left:
    st.markdown("**Offensive and defensive rating**")
    recent = played[(played["team"] == focus) & (played["season"] >= seasons[-1] - 1)].sort_values("game_date")
    frame = recent[["game_date", "pre_off_rtg", "pre_def_rtg"]].rename(columns={"pre_off_rtg": "off_rtg", "pre_def_rtg": "def_rtg"}).dropna()
    if len(frame) >= 2:
        common.show(charts.rating_lines(frame, focus, float(snapshot["off_rtg"].mean())), key="ratings")
        common.note("Rolling pre-game values (30-game half-life). A good team has the blue line above the orange one.")
    else:
        st.info("Not enough games yet.")
with right:
    st.markdown("**Home and road form**")
    common.show(charts.split_bars(float(team["net_rtg_home_split"]), float(team["net_rtg_away_split"]), "Net rating (estimated)"), key="splits")
    common.note(f"Home record {team['home_record']} · road record {team['away_record']} this season.")

st.markdown("**Four Factors profile**")
FACTORS = [("efg", "Shooting: effective FG%", True, 100), ("tov_pct", "Ball security: turnover %", False, 100),
           ("orb_pct", "Offensive rebounding %", True, 100), ("ft_rate", "Free-throw rate (FTM per 100 FGA)", True, 100),
           ("opp_efg", "Shot defence: opponent eFG%", False, 100), ("opp_tov_pct", "Forcing turnovers %", True, 100),
           ("drb_pct", "Defensive rebounding %", True, 100), ("opp_ft_rate", "Keeping opponents off the line", False, 100)]
profile = []
for stat, name, higher_is_better, scale in FACTORS:
    order = snapshot[stat].rank(ascending=not higher_is_better, method="min")
    position = int(order[snapshot["team_abbr"] == focus].iloc[0])
    profile.append({"label": name, "rank": position, "percentile": round(100 * (len(snapshot) - position) / (len(snapshot) - 1)),
                    "display": f"{scale * team[stat]:.1f}"})
p1, p2 = st.columns(2)
with p1:
    st.caption("Offence")
    common.show(charts.percentile_bars(pd.DataFrame(profile[:4]), focus), key="ff_off")
with p2:
    st.caption("Defence")
    common.show(charts.percentile_bars(pd.DataFrame(profile[4:]), focus), key="ff_def")
common.note("Bar length is the league percentile (100 = best of 30 on that factor); the number at the end is the team's value. "
            "Dean Oliver's Four Factors: shooting, turnovers, rebounding and free throws, on both ends.")

st.markdown("**Recent games**")
log = played[played["team"] == focus].sort_values("game_date").tail(10).iloc[::-1]
table = pd.DataFrame(
    {
        "Date": log["game_date"].dt.strftime("%d %b %Y"),
        "Opponent": log["opp_team_id"].map(predictor.abbr),
        "Venue": log["home_flag"].map({1.0: "Home", 0.0: "Road"}),
        "Result": log["win"].map({1.0: "W", 0.0: "L"}),
        "Score": log["PTS"].astype(int).astype(str) + "-" + log["OPP_PTS"].astype(int).astype(str),
        "Net rating": log["net_rtg"].round(1),
        "eFG%": (100 * log["efg"]).round(1),
        "Pace": log["pace"].round(1),
        "Elo before": log["pre_elo"].round(0),
    }
)
st.dataframe(table, hide_index=True, width="stretch",
             column_config={"Net rating": st.column_config.NumberColumn(format="%+.1f"), "Elo before": st.column_config.NumberColumn(format="%.0f")})
