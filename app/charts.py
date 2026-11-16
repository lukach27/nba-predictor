"""Plotly figure builders. Pure functions: data in, figure out (no Streamlit calls).

Conventions: one y-axis per chart, home = blue / away = orange everywhere, a single hue for
magnitude, a neutral-midpoint diverging scale for polarity, and hover details on every mark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from common import AWAY, AXIS, DIVERGING, GRID, HOME, INK, INK_2, MUTED, NEUTRAL, SEQUENTIAL, SERIES, feature_label


def _end_labels(fig: go.Figure, series: list[tuple[str, float, float, str]]) -> None:
    """Direct labels at the right end of up to four lines (legend stays as well)."""
    if len(series) > 4:
        return
    for name, x, y, colour in series:
        fig.add_annotation(x=x, y=y, text=name, showarrow=False, xanchor="left", xshift=6, font=dict(color=INK_2, size=12))
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", marker=dict(color=colour, size=8, line=dict(color="#0d0d0d", width=2)),
                                 hoverinfo="skip", showlegend=False))


# ------------------------------------------------------------------- matchup page
def model_comparison(pred: dict) -> go.Figure:
    """Home win probability according to each model; the ensemble carries its 80% bootstrap band."""
    home, away = pred["home_team"], pred["away_team"]
    rows = [("Ensemble (calibrated)", pred["p_home"]), ("Logistic regression", pred["p_lr"]),
            ("XGBoost", pred["p_xgb"]), ("Elo only", pred["p_elo"])]
    fig = go.Figure()
    fig.add_vline(x=50, line=dict(color=AXIS, width=1, dash="dot"))
    for label, p in rows:
        colour = HOME if p >= 0.5 else AWAY
        error = None
        if label.startswith("Ensemble") and not pred.get("elo_adjustment"):
            error = dict(type="data", symmetric=False, array=[100 * max(pred["p_high"] - p, 0)],
                         arrayminus=[100 * max(p - pred["p_low"], 0)], color=INK_2, thickness=2, width=6)
        fig.add_trace(go.Scatter(
            x=[100 * p], y=[label], mode="markers+text", marker=dict(color=colour, size=14, line=dict(color="#0d0d0d", width=2)),
            error_x=error, text=[f"{100 * p:.0f}%"], textposition="top center", textfont=dict(color=INK, size=13),
            hovertemplate=f"{label}<br>{home} {100 * p:.1f}% · {away} {100 * (1 - p):.1f}%<extra></extra>", showlegend=False))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title=f"Probability that {home} (home) win")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(height=280, margin=dict(t=24))
    return fig


def contributions(contribs: list[dict], home: str, away: str, top: int = 10) -> go.Figure:
    """SHAP contributions to the forecast (log-odds). Bars point toward the team they favour."""
    frame = pd.DataFrame(contribs)
    frame["abs"] = frame["contribution"].abs()
    frame = frame.sort_values("abs", ascending=False)
    head, rest = frame.head(top), frame.iloc[top:]
    if len(rest):
        head = pd.concat([head, pd.DataFrame([{"feature": "__other__", "value": np.nan, "contribution": rest["contribution"].sum()}])])
    head = head.iloc[::-1]
    labels = [f"All other features ({len(rest)})" if f == "__other__" else feature_label(f) for f in head["feature"]]
    fig = go.Figure()
    for name, colour, mask in ((f"Favours {home}", HOME, head["contribution"] >= 0), (f"Favours {away}", AWAY, head["contribution"] < 0)):
        part = head[mask]
        fig.add_trace(go.Bar(
            x=part["contribution"], y=[l for l, m in zip(labels, mask) if m], orientation="h", name=name, marker_color=colour,
            customdata=np.stack([part["value"].fillna(np.nan)], axis=-1),
            hovertemplate="%{y}<br>contribution %{x:+.3f} log-odds<br>feature value %{customdata[0]:.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_xaxes(title="Contribution to the home team's log-odds", zeroline=False)
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=labels)  # keep the magnitude ranking
    fig.update_layout(height=60 + 32 * len(head), barmode="relative", bargap=0.3)
    return fig


def statistical_edges(rows: list[dict], home: str, away: str) -> go.Figure:
    """Standardised edges per statistic: bar toward the team that is better on that metric."""
    frame = pd.DataFrame(rows).iloc[::-1]
    fig = go.Figure()
    for name, colour, mask in ((f"{home} better", HOME, frame["edge"] >= 0), (f"{away} better", AWAY, frame["edge"] < 0)):
        part = frame[mask]
        fig.add_trace(go.Bar(
            x=part["edge"], y=part["label"], orientation="h", name=name, marker_color=colour,
            customdata=np.stack([part["home"], part["away"]], axis=-1),
            hovertemplate=f"%{{y}}<br>{home} %{{customdata[0]}} · {away} %{{customdata[1]}}<br>edge %{{x:+.2f}} typical spreads<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_xaxes(title="Edge in units of the typical gap between two teams", zeroline=False)
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=frame["label"].tolist())
    fig.update_layout(height=60 + 30 * len(frame), barmode="relative", bargap=0.3)
    return fig


# ---------------------------------------------------------------------- team page
def elo_lines(history: pd.DataFrame, colours: dict[str, str]) -> go.Figure:
    """Pre-game Elo over time, one line per team."""
    fig = go.Figure()
    ends = []
    for team, colour in colours.items():
        part = history[history["team"] == team].sort_values("game_date")
        if part.empty:
            continue
        # break the line over the off-season instead of drawing a misleading diagonal
        gaps = part[part["game_date"].diff() > pd.Timedelta(days=45)]
        if len(gaps):
            blanks = gaps.assign(game_date=gaps["game_date"] - pd.Timedelta(days=1), elo=np.nan)
            part = pd.concat([part, blanks]).sort_values("game_date")
        fig.add_trace(go.Scatter(x=part["game_date"], y=part["elo"], mode="lines", name=team, line=dict(color=colour, width=2), connectgaps=False,
                                 hovertemplate=f"{team} %{{y:.0f}}<extra></extra>"))
        ends.append((team, part["game_date"].iloc[-1], float(part["elo"].iloc[-1]), colour))
    fig.add_hline(y=1500, line=dict(color=AXIS, width=1, dash="dot"), annotation_text="league average 1500",
                  annotation_font=dict(color=MUTED, size=11), annotation_position="bottom right")
    _end_labels(fig, ends)
    fig.update_yaxes(title="Elo rating (before each game)")
    fig.update_layout(height=400, hovermode="x unified")
    return fig


def elo_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """League-wide Elo by team and month: blue above average, red below, grey at 1500."""
    spread = float(np.nanmax(np.abs(matrix.to_numpy() - 1500)))
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=matrix.columns, y=matrix.index, colorscale=DIVERGING, zmid=1500, zmin=1500 - spread,
        zmax=1500 + spread, xgap=1, ygap=1, colorbar=dict(title="Elo", thickness=12, outlinewidth=0, tickfont=dict(color=MUTED)),
        hovertemplate="%{y} · %{x}<br>Elo %{z:.0f}<extra></extra>"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_layout(height=640)
    return fig


def elo_surface(matrix: pd.DataFrame) -> go.Figure:
    """The same matrix as a 3D surface (teams ordered by current rating)."""
    fig = go.Figure(go.Surface(
        z=matrix.to_numpy(), x=list(range(matrix.shape[1])), y=list(range(matrix.shape[0])), colorscale=DIVERGING, cmid=1500,
        colorbar=dict(title="Elo", thickness=12, outlinewidth=0), customdata=np.array([[c for c in matrix.columns]] * matrix.shape[0]),
        text=np.array([[t] * matrix.shape[1] for t in matrix.index]),
        hovertemplate="%{text} · %{customdata}<br>Elo %{z:.0f}<extra></extra>"))
    step = max(1, matrix.shape[1] // 8)
    fig.update_layout(height=640, scene=dict(
        xaxis=dict(title="", tickvals=list(range(0, matrix.shape[1], step)), ticktext=list(matrix.columns[::step]), backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID),
        yaxis=dict(title="", tickvals=list(range(matrix.shape[0])), ticktext=list(matrix.index), backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID, tickfont=dict(size=9)),
        zaxis=dict(title="Elo", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID),
        camera=dict(eye=dict(x=-1.6, y=-1.5, z=0.9))), margin=dict(l=0, r=0, t=10, b=0))
    return fig


def rating_lines(frame: pd.DataFrame, team: str, league_average: float) -> go.Figure:
    """Rolling offensive and defensive rating for one team (same unit, one axis)."""
    fig = go.Figure()
    ends = []
    for column, name, colour in (("off_rtg", "Offence (points scored)", SERIES[0]), ("def_rtg", "Defence (points allowed)", SERIES[1])):
        fig.add_trace(go.Scatter(x=frame["game_date"], y=frame[column], mode="lines", name=name, line=dict(color=colour, width=2),
                                 hovertemplate=f"{name} %{{y:.1f}}<extra></extra>"))
        ends.append((name.split(" ")[0], frame["game_date"].iloc[-1], float(frame[column].iloc[-1]), colour))
    fig.add_hline(y=league_average, line=dict(color=AXIS, width=1, dash="dot"), annotation_text=f"league average {league_average:.1f}",
                  annotation_font=dict(color=MUTED, size=11), annotation_position="bottom left")
    _end_labels(fig, ends)
    fig.update_yaxes(title="Points per 100 possessions (estimated)")
    fig.update_layout(height=360, hovermode="x unified")
    return fig


def percentile_bars(frame: pd.DataFrame, team: str) -> go.Figure:
    """League percentile of each factor (100 = best in the league on that factor)."""
    frame = frame.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=frame["percentile"], y=frame["label"], orientation="h", marker_color=SERIES[0],
        text=[f"{v}" for v in frame["display"]], textposition="outside", textfont=dict(color=INK_2), cliponaxis=False,
        customdata=np.stack([frame["rank"], frame["display"]], axis=-1),
        hovertemplate="%{y}<br>value %{customdata[1]} · rank %{customdata[0]} of 30<extra></extra>"))
    fig.add_vline(x=50, line=dict(color=AXIS, width=1, dash="dot"))
    fig.update_xaxes(range=[0, 112], title="League percentile (higher is better)", tickvals=[0, 25, 50, 75, 100])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(height=60 + 34 * len(frame), showlegend=False)
    return fig


def split_bars(home_value: float, away_value: float, label: str, suffix: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(x=["Home games", "Road games"], y=[home_value, away_value], marker_color=[SERIES[0], SERIES[1]], width=0.5,
                           text=[f"{home_value:+.1f}{suffix}" if suffix == "" else f"{home_value:.0f}{suffix}",
                                 f"{away_value:+.1f}{suffix}" if suffix == "" else f"{away_value:.0f}{suffix}"],
                           textposition="outside", textfont=dict(color=INK), cliponaxis=False,
                           hovertemplate="%{x}: %{y:.1f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.update_yaxes(title=label, zeroline=False)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(height=280, showlegend=False)
    return fig


# ------------------------------------------------------------------- explain page
def importance_bars(frame: pd.DataFrame, column: str = "mean_abs_ensemble") -> go.Figure:
    frame = frame.sort_values(column)
    fig = go.Figure(go.Bar(x=frame[column], y=[feature_label(f) for f in frame["feature"]], orientation="h", marker_color=SERIES[0],
                           hovertemplate="%{y}<br>mean |SHAP| %{x:.3f} log-odds<extra></extra>"))
    fig.update_xaxes(title="Average absolute contribution (log-odds)")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(height=60 + 30 * len(frame), showlegend=False)
    return fig


def shap_summary(values: pd.DataFrame, contribs: pd.DataFrame, order: list[str], seed: int = 0) -> go.Figure:
    """One dot per game per feature: position = contribution, shade = the feature's value (dark low, light high)."""
    rng = np.random.default_rng(seed)
    fig = go.Figure()
    for i, feature in enumerate(reversed(order)):
        v = values[feature].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
        shade = np.clip((v - lo) / (hi - lo), 0, 1) if hi > lo else np.full(len(v), 0.5)
        fig.add_trace(go.Scattergl(
            x=contribs[feature], y=i + rng.uniform(-0.3, 0.3, len(v)), mode="markers", showlegend=False,
            marker=dict(color=shade, colorscale=SEQUENTIAL, cmin=0, cmax=1, size=4, opacity=0.75,
                        showscale=i == 0, colorbar=dict(title="Feature value", tickvals=[0, 1], ticktext=["low", "high"], thickness=10, outlinewidth=0)),
            customdata=v, hovertemplate=f"{feature_label(feature)}<br>value %{{customdata:.3f}}<br>contribution %{{x:+.3f}}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=[feature_label(f) for f in reversed(order)], showgrid=False)
    fig.update_xaxes(title="Contribution to the home team's log-odds", zeroline=False)
    fig.update_layout(height=80 + 42 * len(order))
    return fig


def shap_dependence(x: pd.Series, y: pd.Series, feature: str) -> go.Figure:
    fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers", marker=dict(color=SERIES[0], size=5, opacity=0.5),
                                 hovertemplate=f"{feature_label(feature)} %{{x:.3f}}<br>contribution %{{y:+.3f}}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.update_xaxes(title=feature_label(feature))
    fig.update_yaxes(title="Contribution (log-odds)", zeroline=False)
    fig.update_layout(height=360, showlegend=False)
    return fig


def ablation_bars(rows: list[dict], x_title: str) -> go.Figure:
    """Change in walk-forward log loss with a 95% bootstrap interval; left of zero = the group helps."""
    frame = pd.DataFrame(rows).sort_values("delta_log_loss", ascending=False)
    colours = [SERIES[0] if d < 0 else NEUTRAL for d in frame["delta_log_loss"]]
    fig = go.Figure(go.Bar(
        x=1000 * frame["delta_log_loss"], y=frame["group"], orientation="h", marker_color=colours,
        error_x=dict(type="data", symmetric=False, array=1000 * (frame["high"] - frame["delta_log_loss"]),
                     arrayminus=1000 * (frame["delta_log_loss"] - frame["low"]), color=INK_2, thickness=1.5, width=4),
        customdata=np.stack([1000 * frame["low"], 1000 * frame["high"], frame["significant"]], axis=-1),
        hovertemplate="%{y}<br>Δ log loss %{x:+.2f} ×10⁻³<br>95% interval [%{customdata[0]:+.2f}, %{customdata[1]:+.2f}]<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_xaxes(title=x_title, zeroline=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(height=70 + 32 * len(frame), showlegend=False)
    return fig


# ----------------------------------------------------------------- simulator page
def win_distribution(dist: pd.Series, team: str, current_wins: int, current_losses: int) -> go.Figure:
    mean = float((dist.index.to_numpy() * dist.to_numpy()).sum())
    fig = go.Figure(go.Bar(x=dist.index, y=100 * dist.to_numpy(), marker_color=SERIES[0],
                           hovertemplate="%{x} wins: %{y:.1f}% of simulations<extra></extra>"))
    fig.add_vline(x=mean, line=dict(color=INK_2, width=1, dash="dot"), annotation_text=f"mean {mean:.1f}", annotation_font=dict(color=INK_2, size=12))
    fig.update_xaxes(title=f"{team} final wins (currently {current_wins}-{current_losses})", showgrid=False)
    fig.update_yaxes(title="Share of simulations", ticksuffix="%")
    fig.update_layout(height=320, showlegend=False, bargap=0.12)
    return fig


def seed_heatmap(frame: pd.DataFrame) -> go.Figure:
    """Probability of finishing in each conference position (rows sorted by expected finish)."""
    z = 100 * frame.to_numpy()
    text = np.where(z >= 1, np.round(z).astype(int).astype(str), "")
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in frame.columns], y=frame.index, colorscale=SEQUENTIAL, zmin=0, zmax=max(40.0, float(z.max())),
        xgap=2, ygap=2, text=text, texttemplate="%{text}", textfont=dict(size=11, color=INK), showscale=False,
        hovertemplate="%{y} finishes %{x}: %{z:.1f}%<extra></extra>"))
    fig.add_vline(x=5.5, line=dict(color=INK_2, width=1, dash="dot"))
    fig.add_vline(x=9.5, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_xaxes(title="Conference position after the regular season (1-6 playoffs, 7-10 play-in)", showgrid=False, side="bottom")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(height=60 + 30 * len(frame))
    return fig


def odds_bars(frame: pd.DataFrame, column: str, title: str) -> go.Figure:
    frame = frame.sort_values(column)
    fig = go.Figure(go.Bar(x=100 * frame[column], y=frame["team"], orientation="h", marker_color=SERIES[0],
                           text=[f"{100 * v:.1f}%" for v in frame[column]], textposition="outside", textfont=dict(color=INK_2), cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(title=title, ticksuffix="%", range=[0, max(5.0, 118 * float(frame[column].max()))])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(height=60 + 28 * len(frame), showlegend=False)
    return fig


# --------------------------------------------------------------- performance page
def calibration_curves(tables: dict[str, list[dict]], names: dict[str, str]) -> go.Figure:
    """Reliability diagram: on the diagonal, a forecast of x% comes true x% of the time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect calibration", line=dict(color=AXIS, width=1, dash="dot"), hoverinfo="skip"))
    for i, (key, label) in enumerate(names.items()):
        table = pd.DataFrame(tables[key])
        table = table[table["count"] >= 15]
        fig.add_trace(go.Scatter(
            x=100 * table["mean_predicted"], y=100 * table["observed_rate"], mode="lines+markers", name=label,
            line=dict(color=SERIES[i], width=2), marker=dict(size=8, line=dict(color="#0d0d0d", width=2)), customdata=table["count"],
            hovertemplate=f"{label}<br>predicted %{{x:.1f}}% · happened %{{y:.1f}}%<br>%{{customdata}} games<extra></extra>"))
    fig.update_xaxes(title="Predicted home win probability", ticksuffix="%", range=[0, 100])
    fig.update_yaxes(title="How often the home team actually won", ticksuffix="%", range=[0, 100])
    fig.update_layout(height=420)
    return fig


def prediction_histogram(p: pd.Series) -> go.Figure:
    fig = go.Figure(go.Histogram(x=100 * p, xbins=dict(start=0, end=100, size=5), marker_color=SERIES[0],
                                 hovertemplate="%{x}: %{y} games<extra></extra>"))
    fig.update_xaxes(title="Predicted home win probability", ticksuffix="%", range=[0, 100], showgrid=False)
    fig.update_yaxes(title="Games")
    fig.update_layout(height=220, showlegend=False, bargap=0.08)
    return fig


def rolling_lines(frame: pd.DataFrame, names: dict[str, str], y_title: str, percent: bool, split_date=None) -> go.Figure:
    fig = go.Figure()
    ends = []
    for i, (column, label) in enumerate(names.items()):
        y = 100 * frame[column] if percent else frame[column]
        fig.add_trace(go.Scatter(x=frame["game_date"], y=y, mode="lines", name=label, line=dict(color=SERIES[i], width=2),
                                 hovertemplate=f"{label} %{{y:.{1 if percent else 3}f}}{'%' if percent else ''}<extra></extra>"))
        ends.append((label, frame["game_date"].iloc[-1], float(y.iloc[-1]), SERIES[i]))
    if split_date is not None:
        fig.add_vline(x=split_date, line=dict(color=MUTED, width=1, dash="dot"))
        fig.add_annotation(x=split_date, y=1.0, yref="paper", text="hold-out test starts", showarrow=False, xanchor="left", xshift=6,
                           font=dict(color=MUTED, size=11))
    fig.update_yaxes(title=y_title, ticksuffix="%" if percent else "")
    fig.update_layout(height=340, hovermode="x unified")
    return fig


def season_lines(frame: pd.DataFrame, metric: str, names: dict[str, str], y_title: str) -> go.Figure:
    fig = go.Figure()
    for i, (model, label) in enumerate(names.items()):
        part = frame[frame["model"] == model].sort_values("season")
        fig.add_trace(go.Scatter(
            x=[f"{s}-{(s + 1) % 100:02d}" for s in part["season"]], y=part[metric], mode="lines+markers", name=label,
            line=dict(color=SERIES[i], width=2), marker=dict(size=8, line=dict(color="#0d0d0d", width=2)), customdata=part["n"],
            hovertemplate=f"{label} %{{y:.4f}} (%{{customdata}} games)<extra></extra>"))
    fig.update_yaxes(title=y_title)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(height=340, hovermode="x unified")
    return fig
