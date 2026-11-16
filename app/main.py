"""Dashboard entry point.  Run with:  streamlit run app/main.py"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

import common

st.set_page_config(page_title="NBA Match Predictor", page_icon=":basketball:", layout="wide", initial_sidebar_state="expanded")
common.apply_theme()

navigation = st.navigation(
    [
        st.Page("views/matchups.py", title="Matchups", icon=":material/sports_basketball:", default=True),
        st.Page("views/teams.py", title="Team analytics", icon=":material/monitoring:"),
        st.Page("views/explain.py", title="Explainability", icon=":material/account_tree:"),
        st.Page("views/simulator.py", title="Season simulator", icon=":material/casino:"),
        st.Page("views/performance.py", title="Model performance", icon=":material/fact_check:"),
    ]
)


def _refresh() -> None:
    """Download new results, rebuild features, log today's forecasts and grade old ones."""
    from nba_predictor.data_collection import DataCollectionError, update_data
    from nba_predictor.features import save_processed
    from nba_predictor.ledger import record_predictions

    try:
        with st.spinner("Downloading games and schedule from the NBA API..."):
            update_data()
        st.cache_data.clear()
        st.cache_resource.clear()
        predictor = common.predictor()
        save_processed(predictor.features)
        added = record_predictions(predictor.upcoming(days=3), predictor.schedule, predictor.bundle.metadata.get("trained_at", "unknown"))
        st.session_state["refresh_message"] = f"Data refreshed. {len(added)} new pre-game forecast(s) logged."
    except DataCollectionError as exc:
        st.session_state["refresh_error"] = f"The NBA API could not be reached: {exc}"
    except Exception as exc:  # shown to the user and logged, never swallowed
        logging.getLogger(__name__).exception("refresh failed")
        st.session_state["refresh_error"] = f"Refresh failed: {type(exc).__name__}: {exc}"


with st.sidebar:
    common.require_artifacts()
    predictor = common.predictor()
    info = common.metrics(common.data_version())
    st.markdown('<div class="nba-kicker">Data status</div>', unsafe_allow_html=True)
    last = predictor.last_played_date
    age_days = (pd.Timestamp.today().normalize() - last).days
    st.markdown(f"**Results through {last:%d %b %Y}**")
    st.caption(f"{predictor.games['game_id'].nunique():,} games since {predictor.games['game_date'].min():%Y} · "
               f"{len(predictor.fixtures):,} games still to play")
    if age_days > 2:
        st.badge(f"Data is {age_days} days old", icon=":material/schedule:", color="orange")
    st.caption(f"Model trained {info['generated_at'][:10]} on {info['data']['production_rows']:,} games")
    st.button("Refresh data", icon=":material/refresh:", on_click=_refresh, width="stretch",
              help="Downloads new results from the NBA stats API, then logs forecasts for games in the next two days.")
    if message := st.session_state.pop("refresh_message", None):
        st.success(message)
    if error := st.session_state.pop("refresh_error", None):
        st.error(error)
    st.divider()
    st.caption("Forecasts come from a calibrated ensemble trained on team results only. "
               "The model knows nothing about injuries or lineups.")

navigation.run()
