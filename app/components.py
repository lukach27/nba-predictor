"""Reusable visual components: game cards, odds rows and the playoff bracket.

Everything here is presentation only. Numbers come from the Predictor and the simulator.
Team logos are loaded by the viewer's browser from the NBA's public CDN.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

DISPLAY_TZ = "Europe/Zagreb"
LEVEL_COLOUR = {"high": "#0ca30c", "medium": "#fab219", "low": "#ec835a"}

CSS = """
<style>
[class*="st-key-gamecard_"], [class*="st-key-gamecardsel_"] {
  background: #1a1a19; border-radius: 14px; transition: border-color .15s ease, transform .15s ease;
}
[class*="st-key-gamecard_"]:hover { border-color: #6f6e69; transform: translateY(-2px); }
[class*="st-key-gamecardsel_"] { border-color: #3987e5 !important; box-shadow: 0 0 0 1px #3987e5; }
.gc-top { display: flex; justify-content: space-between; align-items: center; color: #898781; font-size: .78rem;
  text-transform: uppercase; letter-spacing: .05em; margin-bottom: 10px; }
.gc-chip { display: inline-flex; align-items: center; gap: 6px; color: #c3c2b7; text-transform: none; letter-spacing: 0;
  font-size: .78rem; border: 1px solid #2c2c2a; border-radius: 999px; padding: 2px 9px; }
.gc-chip i { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.gc-teams { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.gc-team { display: flex; align-items: center; gap: 10px; min-width: 0; }
.gc-team.right { flex-direction: row-reverse; text-align: right; }
.gc-team img { width: 46px; height: 46px; object-fit: contain; }
.gc-team b { display: block; color: #f0efec; font-size: 1.15rem; line-height: 1.1; }
.gc-team span { color: #898781; font-size: .8rem; }
.gc-vs { color: #6f6e69; font-size: .8rem; }
.gc-bar { display: flex; gap: 2px; height: 26px; border-radius: 6px; overflow: hidden; margin: 12px 0 8px 0; }
.gc-bar div { display: flex; align-items: center; padding: 0 8px; color: #fff; font-weight: 600; font-size: .85rem; }
.gc-bar .a { justify-content: flex-end; }
.gc-foot { display: flex; justify-content: space-between; color: #c3c2b7; font-size: .88rem; margin-bottom: 4px; }
.gc-foot b { color: #f0efec; }
.or-row { display: flex; align-items: center; gap: 10px; padding: 7px 0; border-bottom: 1px solid #2c2c2a; }
.or-row:last-child { border-bottom: 0; }
.or-row img { width: 28px; height: 28px; object-fit: contain; }
.or-row .t { width: 44px; color: #f0efec; font-weight: 600; }
.or-row .s { color: #898781; font-size: .8rem; width: 64px; }
.or-row .track { flex: 1; height: 8px; background: #2c2c2a; border-radius: 4px; overflow: hidden; }
.or-row .track div { height: 100%; background: #3987e5; border-radius: 4px; }
.or-row .v { width: 52px; text-align: right; color: #f0efec; font-variant-numeric: tabular-nums; }
.bk-wrap { overflow-x: auto; padding-bottom: 6px; }
.bk, .bk-heads { display: flex; gap: 8px; min-width: 840px; align-items: stretch; }
.bk-heads { margin-bottom: 8px; }
.bk-col, .bk-h { flex: 1 1 0; min-width: 0; }
.bk-col { display: flex; flex-direction: column; justify-content: space-around; gap: 10px; }
.bk-col.final, .bk-h.final { flex: 1.3 1 0; }
.bk-col.final { justify-content: center; }
.bk-h { color: #898781; font-size: .68rem; text-transform: uppercase; letter-spacing: .06em; text-align: center; }
.bk-h b { display: block; color: #c3c2b7; font-weight: 600; }
.bk-box { background: #1a1a19; border: 1px solid #2c2c2a; border-radius: 10px; overflow: hidden; }
.bk-box.champ { border-color: #3987e5; box-shadow: 0 0 0 1px #3987e5; }
.bk-row { display: flex; align-items: center; gap: 6px; padding: 7px 8px; color: #898781; cursor: default; }
.bk-row + .bk-row { border-top: 1px solid #2c2c2a; }
.bk-row img { width: 22px; height: 22px; object-fit: contain; opacity: .55; }
.bk-row .seed { width: 10px; font-size: .7rem; color: #6f6e69; }
.bk-row .t { flex: 1; font-weight: 600; }
.bk-row .p { font-variant-numeric: tabular-nums; font-size: .88rem; }
.bk-row.fav { color: #f0efec; background: rgba(57,135,229,.10); }
.bk-row.fav img { opacity: 1; }
.bk-row:hover { background: rgba(255,255,255,.05); }
.bk-title { text-align: center; color: #f0efec; font-weight: 600; margin: 6px 0; }
.bk-trophy { text-align: center; font-size: 1.6rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def logo(team_id: int) -> str:
    """URL of a team's logo on the NBA CDN."""
    return f"https://cdn.nba.com/logos/nba/{int(team_id)}/primary/L/logo.svg"


def tipoff_label(game) -> str:
    """Local tip-off time (DISPLAY_TZ), or the game date when the time is unknown."""
    stamp = pd.to_datetime(getattr(game, "tipoff_utc", None), errors="coerce", utc=True)
    if pd.isna(stamp):
        return f"{pd.Timestamp(game.game_date):%a %d %b}"
    return f"{stamp.tz_convert(DISPLAY_TZ):%a %d %b · %H:%M}"


def _card_html(game, records: dict[str, str]) -> str:
    p = float(game.p_ensemble)
    home, away = game.home_team, game.away_team
    pick, pick_p = (home, p) if p >= 0.5 else (away, 1 - p)
    margin = getattr(game, "expected_margin", None)
    by = "" if margin is None or pd.isna(margin) else f"by {abs(float(margin)):.1f}"
    level = getattr(game, "confidence_level", None)
    chip = "" if not level else (f'<span class="gc-chip"><i style="background:{LEVEL_COLOUR[level]}"></i>'
                                 f'{level.capitalize()} confidence</span>')
    venue = "Neutral site" if bool(game.is_neutral) else tipoff_label(game)
    h, a = 100 * p, 100 * (1 - p)
    return f"""
<div class="gc-top"><span>{html.escape(venue)}</span>{chip}</div>
<div class="gc-teams">
  <div class="gc-team"><img src="{logo(game.home_team_id)}" alt=""><div><b>{home}</b><span>{records.get(home, '')} · home</span></div></div>
  <div class="gc-vs">vs</div>
  <div class="gc-team right"><img src="{logo(game.away_team_id)}" alt=""><div><b>{away}</b><span>{records.get(away, '')} · away</span></div></div>
</div>
<div class="gc-bar"><div style="width:{max(h, 16):.1f}%;background:#3987e5">{h:.0f}%</div>
<div class="a" style="width:{max(a, 16):.1f}%;background:#d95926">{a:.0f}%</div></div>
<div class="gc-foot"><span>Model pick <b>{pick} {100 * pick_p:.0f}%</b></span><span>{by}</span></div>
"""


def game_cards(slate: pd.DataFrame, records: dict[str, str], key: str, selected: str | None = None,
               button: str = "Analyse", columns: int = 3) -> str | None:
    """Grid of game cards. Returns the game_id whose button was pressed in this run, if any."""
    clicked = None
    cols = st.columns(columns)
    for i, game in enumerate(slate.itertuples()):
        chosen = selected is not None and game.game_id == selected
        with cols[i % columns]:
            with st.container(border=True, key=f"{'gamecardsel' if chosen else 'gamecard'}_{key}_{game.game_id}"):
                st.markdown(_card_html(game, records), unsafe_allow_html=True)
                if st.button("Showing below" if chosen else button, key=f"btn_{key}_{game.game_id}", width="stretch",
                             disabled=chosen, type="primary" if chosen else "secondary"):
                    clicked = game.game_id
    return clicked


def odds_rows(frame: pd.DataFrame, column: str, sub: str | None = None, scale_to: float | None = None) -> None:
    """Compact ranked list: logo, team, optional sub-label, bar and percentage."""
    top = scale_to or max(float(frame[column].max()), 1e-9)
    rows = []
    for r in frame.itertuples():
        value = float(getattr(r, column))
        extra = f'<span class="s">{html.escape(str(getattr(r, sub)))}</span>' if sub else ""
        rows.append(f'<div class="or-row"><img src="{logo(r.team_id)}" alt=""><span class="t">{r.team}</span>{extra}'
                    f'<div class="track"><div style="width:{100 * value / top:.1f}%"></div></div>'
                    f'<span class="v">{100 * value:.1f}%</span></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)


# ------------------------------------------------------------------------ bracket
def projected_seeds(table: pd.DataFrame, conference: str) -> pd.DataFrame:
    """Seeds 1-8 implied by the simulation: top six by projected wins, then the two teams
    ranked 7-10 that survive the play-in most often."""
    conf = table[table["conference"] == conference].sort_values("mean_wins", ascending=False).reset_index(drop=True)
    direct = conf.iloc[:6]
    survivors = conf.iloc[6:10].sort_values("p_playoffs", ascending=False).head(2).sort_values("mean_wins", ascending=False)
    seeds = pd.concat([direct, survivors]).reset_index(drop=True)
    seeds["seed"] = range(1, len(seeds) + 1)
    return seeds


def _row(team, column: str, favourite: bool) -> str:
    tip = (f"{team.team}: projected {team.mean_wins:.0f} wins · playoffs {100 * team.p_playoffs:.0f}% · "
           f"conference finals {100 * team.p_conf_finals:.0f}% · NBA Finals {100 * team.p_finals:.0f}% · title {100 * team.p_champion:.1f}%")
    return (f'<div class="bk-row{" fav" if favourite else ""}" title="{html.escape(tip)}"><span class="seed">{team.seed}</span>'
            f'<img src="{logo(team.team_id)}" alt=""><span class="t">{team.team}</span>'
            f'<span class="p">{100 * float(getattr(team, column)):.0f}%</span></div>')


def _box(a, b, column: str, extra_class: str = "") -> tuple[str, object]:
    """One series box; returns its HTML and the team more likely to win it."""
    winner = a if float(getattr(a, column)) >= float(getattr(b, column)) else b
    body = _row(a, column, winner is a) + _row(b, column, winner is b)
    return f'<div class="bk-box {extra_class}">{body}</div>', winner


def _conference_columns(seeds: pd.DataFrame) -> tuple[list[str], object]:
    """HTML of the three rounds of one conference (first round, semifinals, final) and its favourite."""
    team = {int(r.seed): r for r in seeds.itertuples()}
    first, winners = [], []
    for hi, lo in ((1, 8), (4, 5), (3, 6), (2, 7)):
        box, w = _box(team[hi], team[lo], "p_second_round")
        first.append(box)
        winners.append(w)
    semis, semi_winners = [], []
    for i in (0, 2):
        box, w = _box(winners[i], winners[i + 1], "p_conf_finals")
        semis.append(box)
        semi_winners.append(w)
    final, champion = _box(semi_winners[0], semi_winners[1], "p_finals")
    return ["".join(first), "".join(semis), final], champion


def bracket(table: pd.DataFrame) -> None:
    """Projected playoff bracket: West on the left, East on the right, the Finals in the middle."""
    west, west_champ = _conference_columns(projected_seeds(table, "West"))
    east, east_champ = _conference_columns(projected_seeds(table, "East"))
    finals, champion = _box(west_champ, east_champ, "p_champion", "champ")
    names = ["First round", "Semifinals", "Conf. finals"]
    heads = [f'<div class="bk-h"><b>West</b>{n}</div>' for n in names]
    heads.append('<div class="bk-h final"><b>NBA Finals</b>Champion</div>')
    heads += [f'<div class="bk-h"><b>East</b>{n}</div>' for n in reversed(names)]
    cols = [f'<div class="bk-col">{c}</div>' for c in west]
    cols.append(f'<div class="bk-col final">{finals}<div class="bk-trophy">🏆</div>'
                f'<div class="bk-title">{champion.team} · {100 * champion.p_champion:.0f}% to win it all</div></div>')
    cols += [f'<div class="bk-col">{c}</div>' for c in reversed(east)]
    st.markdown(f'<div class="bk-wrap"><div class="bk-heads">{"".join(heads)}</div><div class="bk">{"".join(cols)}</div></div>',
                unsafe_allow_html=True)
