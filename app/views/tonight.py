"""Tonight: the one-screen summary - next slate of games, the headline matchups, the title race."""

from __future__ import annotations

import streamlit as st

import common
import components
from nba_predictor.ledger import grade_ledger, ledger_metrics, load_ledger

predictor = common.predictor()
version = common.data_version()
upcoming = common.upcoming(version, 10)

if upcoming.empty:
    common.page_header("Tonight in the NBA", "No scheduled games were found.")
    st.info("Use **Refresh data** in the sidebar to download the latest schedule.")
    st.stop()

day = upcoming["game_date"].dt.date.min()
slate = upcoming[upcoming["game_date"].dt.date == day].reset_index(drop=True)
common.page_header("Tonight in the NBA", f"{day:%A %d %B %Y} · {len(slate)} game{'s' if len(slate) != 1 else ''} · "
                                         f"forecasts from the calibrated ensemble")

snapshot = common.snapshot(version)
records = {r.team_abbr: f"{r.wins}-{r.losses}" for r in snapshot.itertuples()}
favourite_p = slate["p_ensemble"].where(slate["p_ensemble"] >= 0.5, 1 - slate["p_ensemble"])
favourite = slate["home_team"].where(slate["p_ensemble"] >= 0.5, slate["away_team"])
underdog = slate["away_team"].where(slate["p_ensemble"] >= 0.5, slate["home_team"])
label = lambda i: f"{slate.loc[i, 'away_team']} at {slate.loc[i, 'home_team']}"  # noqa: E731

closest, lock = favourite_p.idxmin(), favourite_p.idxmax()
contenders = slate[favourite_p >= 0.6]
upset = (contenders if len(contenders) else slate)["volatility_score"].idxmax()

graded = grade_ledger(load_ledger(), predictor.games)
summary = ledger_metrics(graded) if len(graded) else None

t1, t2, t3, t4 = st.columns(4)
t1.metric("Game of the night", label(closest), f"{favourite[closest]} {common.pct(favourite_p[closest])} · closest call", delta_color="off", delta_arrow="off")
t2.metric("Safest pick", f"{favourite[lock]} {common.pct(favourite_p[lock])}", label(lock), delta_color="off", delta_arrow="off")
t3.metric("Upset watch", f"{underdog[upset]} {common.pct(1 - favourite_p[upset])}", f"{label(upset)} · volatility {slate.loc[upset, 'volatility_score']:.0f}/100",
          delta_color="off", delta_arrow="off", help="The favoured game with the highest volatility score: the pick most exposed to a surprise.")
if summary:
    correct = round(summary["ensemble"]["accuracy"] * summary["n"])
    t4.metric("Live record", f"{correct}-{summary['n'] - correct}", f"{summary['pending']} forecasts pending", delta_color="off", delta_arrow="off",
              help="Forecasts logged before tip-off and graded afterwards.")
else:
    t4.metric("Live record", "0-0", f"{len(graded)} forecasts pending", delta_color="off", delta_arrow="off",
              help="Forecasts are logged before tip-off and graded after the next data refresh.")

st.subheader("The slate")
clicked = components.game_cards(slate, records, key="tonight", button="Open analysis")
if clicked:
    st.session_state["selected_game"] = clicked
    st.switch_page("views/matchups.py")
common.note("Blue is the home team, orange the away team. Tip-off times are shown in Central European time.")

result, inputs, leverage = common.simulation(version, 5000, 42, True)
table = result.table
left, right = st.columns(2)
with left:
    st.subheader("Title race")
    components.odds_rows(table.sort_values("p_champion", ascending=False).head(6), "p_champion")
    common.note("Share of 5,000 simulated seasons ending in a championship.")
with right:
    st.subheader("On the playoff bubble")
    bubble = table.assign(gap=(table["p_playoffs"] - 0.5).abs()).sort_values("gap").head(6).sort_values("p_playoffs", ascending=False)
    components.odds_rows(bubble, "p_playoffs", scale_to=1.0)
    common.note("The six teams whose playoff chances are closest to a coin flip.")

if not leverage.empty:
    stakes = leverage.iloc[0]
    st.subheader("Biggest game on the horizon")
    st.markdown(f"**{stakes['away_team']} at {stakes['home_team']}** on {stakes['game_date']}: a win instead of a loss moves "
                f"{stakes['home_team']}'s playoff chances by **{100 * stakes['swing_home']:+.1f} points** and "
                f"{stakes['away_team']}'s by **{100 * stakes['swing_away']:+.1f} points**.")

c1, c2, c3 = st.columns(3)
c1.page_link("views/matchups.py", label="All upcoming games and analysis", icon=":material/sports_basketball:")
c2.page_link("views/simulator.py", label="Playoff bracket and full standings", icon=":material/account_tree:")
c3.page_link("views/performance.py", label="How accurate is the model?", icon=":material/fact_check:")
