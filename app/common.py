"""Shared dashboard code: theme, cached access to the backend, formatting helpers.

Every page talks to the real backend (``nba_predictor``) through the cached functions in
this module; nothing in the dashboard is mocked or hard-coded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from nba_predictor import config

# ----------------------------------------------------------------------- palette
# Dark-surface steps of a colour-blind-validated categorical palette (fixed order, never cycled).
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
HOME, AWAY = SERIES[0], SERIES[1]  # the home team is always blue, the away team always orange
SURFACE, PAGE = "#1a1a19", "#0d0d0d"
INK, INK_2, MUTED = "#f0efec", "#c3c2b7", "#898781"
GRID, AXIS = "#2c2c2a", "#383835"
NEUTRAL = "#6f6e69"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQUENTIAL = [[0.0, "#15202e"], [0.35, "#184f95"], [0.7, "#3987e5"], [1.0, "#b7d3f6"]]  # one hue, dark -> light
DIVERGING = [[0.0, "#e66767"], [0.5, "#383835"], [1.0, "#3987e5"]]  # two poles, neutral midpoint
LEVEL_STATUS = {"high": "good", "medium": "warning", "low": "serious"}
LEVEL_ICON = {"high": ":material/verified:", "medium": ":material/adjust:", "low": ":material/help:"}
MAX_TEAM_SERIES = 6

FEATURE_LABELS: dict[str, str] = {
    "elo_diff": "Elo rating gap", "net_rtg_diff": "Net rating gap", "off_rtg_diff": "Offensive rating gap",
    "win_pct_diff": "Win % gap", "efg_diff": "Effective FG% gap", "tov_pct_diff": "Turnover % gap",
    "orb_pct_diff": "Offensive rebound % gap", "ft_rate_diff": "Free-throw rate gap",
    "opp_efg_diff": "Opponent eFG% allowed gap", "opp_tov_pct_diff": "Forced turnover % gap",
    "drb_pct_diff": "Defensive rebound % gap", "opp_ft_rate_diff": "Opponent FT rate allowed gap",
    "net_rtg_short_diff": "Recent-form net rating gap", "win_pct_short_diff": "Recent-form win % gap",
    "elo_trend_diff": "Elo momentum gap (last 10)", "venue_net_rtg_diff": "Home-at-home vs away-on-road net rating",
    "venue_win_pct_diff": "Home-at-home vs away-on-road win %", "rest_days_home": "Home rest days",
    "rest_days_away": "Away rest days", "rest_diff": "Rest-day gap", "b2b_home": "Home on a back-to-back",
    "b2b_away": "Away on a back-to-back", "games_last_7d_diff": "Games in last 7 days gap",
    "three_in_four_home": "Home: 3rd game in 4 nights", "three_in_four_away": "Away: 3rd game in 4 nights",
    "away_road_streak": "Away team's consecutive road games", "h2h_margin": "Head-to-head margin (shrunk)",
    "h2h_games": "Recent head-to-head games", "league_home_win_pct": "League home win rate (trailing year)",
    "season_progress": "Season progress", "roster_avail_diff": "Roster availability gap",
    "roster_avail_home": "Home roster availability", "roster_avail_away": "Away roster availability",
}


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " "))


# ------------------------------------------------------------------------- theme
def apply_theme() -> None:
    """Register the Plotly template and inject the small amount of custom CSS."""
    axis = dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED, size=12),
                title=dict(font=dict(color=INK_2, size=12)), ticks="", automargin=True)
    pio.templates["nba"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK_2, size=13),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=SERIES,
            xaxis=axis, yaxis=axis, margin=dict(l=8, r=8, t=36, b=8),
            title=dict(font=dict(color=INK, size=15), x=0, xanchor="left"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK_2), bgcolor="rgba(0,0,0,0)"),
            hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(color=INK, size=13)),
            barcornerradius=4, bargap=0.25, hovermode="closest",
        )
    )
    pio.templates.default = "nba"
    st.markdown(
        f"""
        <style>
        .block-container {{padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1280px;}}
        h1 {{font-size: 1.9rem; letter-spacing: -0.01em;}}
        h3 {{font-size: 1.05rem; color: {INK}; margin-top: 0.4rem;}}
        [data-testid="stMetric"] {{background: {SURFACE}; border: 1px solid {GRID}; border-radius: 10px; padding: 14px 16px;}}
        [data-testid="stMetricLabel"] p {{color: {MUTED}; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em;}}
        [data-testid="stMetricValue"] {{font-size: 1.7rem;}}
        .nba-kicker {{color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.1rem;}}
        .nba-note {{color: {MUTED}; font-size: 0.85rem;}}
        .nba-probbar {{display: flex; gap: 2px; height: 44px; border-radius: 8px; overflow: hidden; margin: 6px 0 2px 0;}}
        .nba-probbar div {{display: flex; align-items: center; padding: 0 14px; color: #fff; font-weight: 600; font-size: 1.05rem; white-space: nowrap;}}
        .nba-probbar .away {{justify-content: flex-end;}}
        .nba-legend {{display: flex; justify-content: space-between; color: {INK_2}; font-size: 0.85rem;}}
        .nba-dot {{display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, kicker: str = "NBA Match Predictor") -> None:
    st.markdown(f'<div class="nba-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def note(text: str) -> None:
    st.markdown(f'<div class="nba-note">{text}</div>', unsafe_allow_html=True)


def show(fig: go.Figure, key: str | None = None) -> None:
    """Render a Plotly figure full-width with a quiet mode bar."""
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def probability_bar(home: str, away: str, p_home: float) -> None:
    """The headline: one bar split between the two teams (home blue, away orange)."""
    h, a = 100 * p_home, 100 * (1 - p_home)
    st.markdown(
        f"""<div class="nba-probbar">
        <div style="width:{max(h, 12):.1f}%; background:{HOME};">{home} {h:.0f}%</div>
        <div class="away" style="width:{max(a, 12):.1f}%; background:{AWAY};">{a:.0f}% {away}</div></div>
        <div class="nba-legend"><span><span class="nba-dot" style="background:{HOME}"></span>{home} (home)</span>
        <span>{away} (away)<span class="nba-dot" style="background:{AWAY}; margin:0 0 0 6px"></span></span></div>""",
        unsafe_allow_html=True,
    )


def level_badge(label: str, level: str, score: float, high_is_good: bool = True) -> None:
    """Status is never colour alone: icon + word + number. For risk scores, low is the good end."""
    quality = level if high_is_good else {"high": "low", "medium": "medium", "low": "high"}[level]
    colour = {"good": "green", "warning": "orange", "serious": "red"}[LEVEL_STATUS[quality]]
    st.badge(f"{label}: {level} ({score:.0f}/100)", icon=LEVEL_ICON[quality], color=colour)


# ------------------------------------------------------------------ data access
def data_version() -> tuple:
    """Changes whenever data or model files change, which invalidates every cache below."""
    paths = (config.RAW_GAMES_PATH, config.SCHEDULE_PATH, config.BUNDLE_PATH, config.METRICS_PATH, config.LEDGER_PATH)
    return tuple(p.stat().st_mtime if p.exists() else 0.0 for p in paths)


def require_artifacts() -> None:
    """Stop with setup instructions when the data or the trained model is missing."""
    missing = [p for p in (config.RAW_GAMES_PATH, config.BUNDLE_PATH, config.METRICS_PATH) if not Path(p).exists()]
    if missing:
        st.error("The dashboard needs downloaded data and a trained model.")
        st.code("python -m nba_predictor fetch\npython -m nba_predictor train", language="bash")
        st.caption("Missing: " + ", ".join(str(p.relative_to(config.PROJECT_ROOT)) for p in missing))
        st.stop()


@st.cache_resource(show_spinner="Loading games and model...")
def _predictor(version: tuple):
    from nba_predictor.predict import Predictor

    predictor = Predictor.from_disk()
    predictor.features  # build once, share across pages
    return predictor


def predictor():
    require_artifacts()
    return _predictor(data_version())


@st.cache_data(show_spinner=False)
def metrics(version: tuple) -> dict:
    return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def backtest(version: tuple) -> pd.DataFrame:
    frame = pd.read_csv(config.BACKTEST_PATH, dtype={"game_id": str}, parse_dates=["game_date"])
    return frame.sort_values("game_date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def upcoming(version: tuple, days: int | None) -> pd.DataFrame:
    return predictor().upcoming(days=days)


@st.cache_data(show_spinner=False)
def snapshot(version: tuple) -> pd.DataFrame:
    return predictor().team_snapshot()


@st.cache_data(show_spinner="Computing forecast...")
def matchup(version: tuple, home: str, away: str, game_date: str | None, neutral: bool,
            adjustments: tuple[tuple[str, float], ...], fixture_id: str | None) -> dict:
    p = predictor()
    row = None
    if fixture_id is not None:
        m = p.features.matchups
        row = m[m["game_id"] == fixture_id].iloc[0]
    return p.predict_matchup(home, away, game_date=game_date, is_neutral=neutral, elo_adjust=dict(adjustments),
                             fixture_row=row).to_dict()


@st.cache_data(show_spinner="Writing the analysis...")
def tactical_report(version: tuple, prediction_json: str, use_llm: bool) -> dict:
    from nba_predictor.llm_model import analyze_matchup

    return analyze_matchup(json.loads(prediction_json), use_llm=use_llm).model_dump()


@st.cache_resource(show_spinner="Simulating the season...")
def simulation(version: tuple, n_sims: int, seed: int, hot: bool):
    from nba_predictor.simulation import build_simulation_inputs, game_leverage, simulate_season

    inputs, k_factor = build_simulation_inputs(predictor())
    result = simulate_season(inputs, n_sims=n_sims, seed=seed, hot=hot, k_factor=k_factor)
    return result, inputs, game_leverage(result, inputs, max_games=60)


@st.cache_data(show_spinner="Computing SHAP values...")
def shap_frame(version: tuple, seasons: tuple[int, ...], max_rows: int = 2500) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(game info, feature values, SHAP contributions) for played regular-season games."""
    from nba_predictor.explain import ensemble_contributions
    from nba_predictor.features import training_mask

    p = predictor()
    m = p.features.matchups
    rows = m[training_mask(m) & m["season"].isin(seasons)].sort_values("game_date").tail(max_rows)
    contributions, base = ensemble_contributions(p.bundle, rows)
    info = rows[["game_id", "game_date", "season", "home_team_id", "away_team_id", "home_win", "home_pts", "away_pts"]].copy()
    info["home_team"] = info["home_team_id"].map(p.abbr)
    info["away_team"] = info["away_team_id"].map(p.abbr)
    info["base"] = base
    return info.reset_index(drop=True), rows[p.bundle.feature_columns].reset_index(drop=True), contributions.reset_index(drop=True)


# -------------------------------------------------------------------- formatting
def pct(value: float | None, digits: int = 0) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{100 * value:.{digits}f}%"


def signed(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:+.{digits}f}"


def team_options(p) -> dict[str, str]:
    """{abbreviation: 'BOS - Boston Celtics'} sorted by abbreviation."""
    return {row.team_abbr: f"{row.team_abbr} - {row.team_name}" for row in p.teams.itertuples()}


def stable_colours(selected: list[str], state_key: str) -> dict[str, str]:
    """Colour follows the entity: a team keeps its colour while other teams are added or removed."""
    assigned: dict[str, int] = st.session_state.setdefault(state_key, {})
    for team in list(assigned):
        if team not in selected:
            del assigned[team]
    for team in selected:
        if team not in assigned:
            free = [i for i in range(len(SERIES)) if i not in assigned.values()]
            assigned[team] = free[0] if free else len(assigned) % len(SERIES)
    return {team: SERIES[slot] for team, slot in assigned.items()}
