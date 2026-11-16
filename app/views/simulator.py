"""Monte Carlo season simulator: projected records, playoff and title odds, game leverage."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import common

predictor = common.predictor()
version = common.data_version()
common.page_header("Season simulator", "The rest of the season played thousands of times with the calibrated game probabilities.")

c1, c2, c3 = st.columns([2, 2, 3], vertical_alignment="bottom")
n_sims = c1.select_slider("Simulations", options=[1000, 2000, 5000, 10000, 20000], value=10000)
seed = int(c2.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1, help="Same seed, same result."))
hot = c3.toggle("Let team strength drift during the season", value=True,
                help="Each simulated win or loss nudges a team's strength (Elo update). Without it, today's ratings are "
                     "treated as exact for the rest of the year and the range of outcomes is too narrow.")

result, inputs, leverage = common.simulation(version, n_sims, seed, hot)
table = result.table
remaining = len(inputs.fixture_home)
unscheduled = int(inputs.unscheduled.sum()) // 2
m1, m2, m3, m4 = st.columns(4)
favourite = table.sort_values("p_champion", ascending=False).iloc[0]
m1.metric("Title favourite", f"{favourite['team']} {common.pct(favourite['p_champion'])}")
m2.metric("Scheduled games left", f"{remaining:,}")
m3.metric("Games not yet scheduled", f"{unscheduled}", help="The NBA sets two games per team after the Cup group stage. "
          "Until then they are simulated against a league-average opponent on a neutral floor.")
m4.metric("Simulations", f"{result.n_sims:,}", f"seed {result.seed}", delta_color="off")

st.subheader("Projected standings")
tabs = st.tabs(["East", "West"])
for tab, conference in zip(tabs, ("East", "West")):
    with tab:
        standings = result.standings(conference)
        view = pd.DataFrame(
            {
                "Team": standings["team"],
                "Now": standings["current_wins"].astype(str) + "-" + standings["current_losses"].astype(str),
                "Projected": standings["mean_wins"].round(1).astype(str) + "-" + standings["mean_losses"].round(1).astype(str),
                "Wins, 10th-90th pct": standings["wins_p10"].astype(int).astype(str) + " to " + standings["wins_p90"].astype(int).astype(str),
                "Playoffs": 100 * standings["p_playoffs"],
                "Top 6": 100 * standings["p_top6"],
                "Play-in": 100 * standings["p_playin"],
                "Conf. finals": 100 * standings["p_conf_finals"],
                "Finals": 100 * standings["p_finals"],
                "Title": 100 * standings["p_champion"],
            }
        )
        bar = lambda name: st.column_config.ProgressColumn(name, format="%.0f%%", min_value=0, max_value=100)  # noqa: E731
        num = lambda name: st.column_config.NumberColumn(name, format="%.1f%%")  # noqa: E731
        st.dataframe(view, hide_index=True, width="stretch", height=38 * (len(view) + 1),
                     column_config={"Playoffs": bar("Playoffs"), "Top 6": bar("Top 6"), "Play-in": num("Play-in"),
                                    "Conf. finals": num("Conf. finals"), "Finals": num("Finals"), "Title": num("Title")})
        seeds = result.seed_distribution().loc[standings["team"]]
        st.markdown("**Where each team finishes**")
        common.show(charts.seed_heatmap(seeds), key=f"seeds_{conference}")
        common.note("Each cell is the share of simulations in which the team ends the regular season in that position "
                    "(numbers are percentages; blank means under 1%).")

st.subheader("One team's range of outcomes")
options = common.team_options(predictor)
abbrs = list(options)
t1, t2 = st.columns([2, 3])
team = t1.selectbox("Team", abbrs, index=abbrs.index(favourite["team"]), format_func=options.get)
row = table[table["team"] == team].iloc[0]
with t1:
    st.metric("Projected record", f"{row['mean_wins']:.1f}-{row['mean_losses']:.1f}")
    st.metric("Makes the playoffs", common.pct(row["p_playoffs"], 1))
    st.metric("Wins the title", common.pct(row["p_champion"], 1))
with t2:
    common.show(charts.win_distribution(result.win_distribution(team), team, int(row["current_wins"]), int(row["current_losses"])), key="wins")

left, right = st.columns(2)
with left:
    st.subheader("Title odds")
    contenders = table[table["p_champion"] > 0].sort_values("p_champion", ascending=False).head(10)
    common.show(charts.odds_bars(contenders, "p_champion", "Share of simulations ending in a championship"), key="title")
with right:
    st.subheader("Highest-leverage upcoming games")
    if leverage.empty:
        st.info("No upcoming games to evaluate.")
    else:
        top = leverage.head(10)
        st.dataframe(
            pd.DataFrame({"Date": top["game_date"], "Game": top["away_team"] + " @ " + top["home_team"], "Home win": 100 * top["p_home"],
                          "Home swing": 100 * top["swing_home"], "Away swing": 100 * top["swing_away"], "Leverage": 100 * top["leverage"]}),
            hide_index=True, width="stretch",
            column_config={"Home win": st.column_config.NumberColumn(format="%.0f%%"),
                           "Home swing": st.column_config.NumberColumn(format="%+.1f pts", help="Change in the home team's playoff probability if it wins rather than loses."),
                           "Away swing": st.column_config.NumberColumn(format="%+.1f pts"),
                           "Leverage": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=float(100 * leverage["leverage"].max()))})
        common.note("Swing = playoff probability if the team wins this game minus if it loses, across the simulations. "
                    "Covers the next 60 scheduled games.")

with st.expander("How the simulation works, and what it leaves out", icon=":material/info:"):
    st.markdown(
        """
- Every remaining regular-season game is drawn from the calibrated ensemble probability for that exact fixture
  (home court, rest and schedule density included).
- Standings ties are broken by conference record and then at random. The NBA's head-to-head and division rules are not implemented.
- Seeds 7-10 play the play-in; the playoffs are best-of-seven with a 2-2-1-1-1 home pattern. Playoff games reuse regular-season
  win probabilities between rested teams, and strength does not drift during the playoffs.
- Injuries, trades and deliberate resting are unknown to the model, so real seasons vary more than any simulation of this kind.
"""
    )
