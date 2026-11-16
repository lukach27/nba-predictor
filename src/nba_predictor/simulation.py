"""Monte Carlo simulation of the rest of the season, the play-in and the playoffs.

Every remaining regular-season game is simulated from the calibrated ensemble probability.
Two refinements make the season-level distributions more honest:

* **Hot simulation** (default on): within each simulated season a team's strength drifts as
  it wins and loses, using the Elo update rule on the log-odds scale. A static simulation
  treats today's ratings as exact forever and therefore understates how wide the range of
  final records really is.
* **Unscheduled games**: the NBA fixes two games per team only after the Cup group stage.
  Until they appear on the schedule they are simulated against a league-average opponent
  at a neutral site.

Simplifications (documented, not hidden): standings ties are broken by conference record and
then at random (the official head-to-head and division rules are not implemented); playoff
games reuse regular-season win probabilities; strength does not drift during the playoffs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .elo import LOGIT_PER_ELO
from .metrics import sigmoid

PLAYOFF_SEEDS: int = 8
DIRECT_SEEDS: int = 6
PLAYIN_LAST_SEED: int = 10
#: Games hosted by the team with home-court advantage in a best-of-seven (2-2-1-1-1).
_HOME_GAMES = [0, 1, 4, 6]
_AWAY_GAMES = [2, 3, 5]


class SimulationError(ValueError):
    """Raised when simulation inputs are inconsistent."""


@dataclass
class SimulationInputs:
    """Everything the simulator needs, as plain arrays indexed by team position."""

    team_ids: np.ndarray
    team_abbr: list[str]
    conference: np.ndarray  # "East" / "West" per team
    wins: np.ndarray  # current regular-season wins
    losses: np.ndarray
    conf_wins: np.ndarray  # current wins against same-conference opponents
    fixture_home: np.ndarray  # team positions, chronological
    fixture_away: np.ndarray
    fixture_logit: np.ndarray  # calibrated log-odds of a home win
    fixture_ids: list[str] = field(default_factory=list)
    fixture_dates: list[str] = field(default_factory=list)
    unscheduled: np.ndarray | None = None  # games per team not yet on the schedule
    pair_logit: np.ndarray | None = None  # Z[h, a]: log-odds that h beats a at home

    def validate(self) -> None:
        n = len(self.team_ids)
        if len(self.team_abbr) != n or len(self.conference) != n or len(self.wins) != n or len(self.losses) != n:
            raise SimulationError("team arrays must all have the same length")
        if not (len(self.fixture_home) == len(self.fixture_away) == len(self.fixture_logit)):
            raise SimulationError("fixture arrays must all have the same length")
        if len(self.fixture_home) and (np.asarray(self.fixture_home) == np.asarray(self.fixture_away)).any():
            raise SimulationError("a fixture cannot pair a team with itself")
        if not np.isfinite(np.asarray(self.fixture_logit, dtype=float)).all():
            raise SimulationError("fixture log-odds must be finite")
        if self.pair_logit is not None and np.asarray(self.pair_logit).shape != (n, n):
            raise SimulationError("pair_logit must be a square matrix over teams")
        for conf in np.unique(self.conference):
            if (self.conference == conf).sum() < PLAYIN_LAST_SEED and self.pair_logit is not None:
                raise SimulationError(f"conference {conf} needs at least {PLAYIN_LAST_SEED} teams for the playoffs")


@dataclass
class SimulationResult:
    """Simulation output. Arrays have one row per simulated season."""

    table: pd.DataFrame  # one row per team with projections and probabilities
    wins: np.ndarray  # (n_sims, teams) final regular-season wins
    rank: np.ndarray  # (n_sims, teams) conference rank after the regular season (1 = best)
    made_playoffs: np.ndarray  # (n_sims, teams) bool
    champion: np.ndarray  # (n_sims,) team position, -1 when playoffs were not simulated
    fixture_results: np.ndarray  # (n_sims, fixtures) bool: home team won
    n_sims: int
    seed: int
    hot: bool

    def standings(self, conference: str | None = None) -> pd.DataFrame:
        """Projected standings (best first), optionally for one conference."""
        table = self.table if conference is None else self.table[self.table["conference"] == conference]
        return table.sort_values(["conference", "mean_wins"], ascending=[True, False]).reset_index(drop=True)

    def seed_distribution(self) -> pd.DataFrame:
        """P(conference rank = r) for every team (rows) and rank (columns)."""
        max_rank = int(self.rank.max())
        data = {r: (self.rank == r).mean(axis=0) for r in range(1, max_rank + 1)}
        return pd.DataFrame(data, index=self.table["team"].to_numpy())

    def win_distribution(self, team: str) -> pd.Series:
        """Probability of each final win total for one team."""
        position = int(np.flatnonzero(self.table["team"].to_numpy() == team)[0])
        counts = pd.Series(self.wins[:, position]).value_counts(normalize=True).sort_index()
        counts.index.name, counts.name = "wins", "probability"
        return counts


# ------------------------------------------------------------------ building blocks
def _series_winner(rng: np.random.Generator, hi: np.ndarray, lo: np.ndarray, pair_logit: np.ndarray,
                   shift: np.ndarray) -> np.ndarray:
    """Best-of-seven winner for each simulation; ``hi`` holds home court.

    All seven games are drawn and the team with four or more wins advances, which has the
    same distribution as stopping at the fourth win.
    """
    edge = shift[np.arange(len(hi)), hi] - shift[np.arange(len(hi)), lo]
    p_home = sigmoid(pair_logit[hi, lo] + edge)  # hi wins a game on its own floor
    p_away = 1.0 - sigmoid(pair_logit[lo, hi] - edge)  # hi wins a game on lo's floor
    draws = rng.random((len(hi), 7))
    hi_wins = (draws[:, _HOME_GAMES] < p_home[:, None]).sum(axis=1) + (draws[:, _AWAY_GAMES] < p_away[:, None]).sum(axis=1)
    return np.where(hi_wins >= 4, hi, lo)


def _game_winner(rng: np.random.Generator, home: np.ndarray, away: np.ndarray, pair_logit: np.ndarray,
                 shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-game (winner, loser) for each simulation."""
    rows = np.arange(len(home))
    p = sigmoid(pair_logit[home, away] + shift[rows, home] - shift[rows, away])
    home_won = rng.random(len(home)) < p
    return np.where(home_won, home, away), np.where(home_won, away, home)


def _order_by_strength(a: np.ndarray, b: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (home-court team, other team) per simulation, by regular-season standing score."""
    rows = np.arange(len(a))
    a_first = score[rows, a] >= score[rows, b]
    return np.where(a_first, a, b), np.where(a_first, b, a)


def simulate_season(inputs: SimulationInputs, n_sims: int = 10_000, seed: int = config.RANDOM_SEED,
                    hot: bool = True, k_factor: float = 20.0, simulate_playoffs: bool = True) -> SimulationResult:
    """Simulate the remaining season ``n_sims`` times. Deterministic for a given ``seed``."""
    inputs.validate()
    if n_sims < 1:
        raise SimulationError("n_sims must be at least 1")
    rng = np.random.default_rng(seed)
    n_teams, n_fixtures = len(inputs.team_ids), len(inputs.fixture_home)
    same_conf = inputs.conference[inputs.fixture_home] == inputs.conference[inputs.fixture_away] if n_fixtures else np.array([], bool)

    wins = np.tile(np.asarray(inputs.wins, dtype=np.int32), (n_sims, 1))
    conf_wins = np.tile(np.asarray(inputs.conf_wins, dtype=np.int32), (n_sims, 1))
    drift = np.zeros((n_sims, n_teams))  # simulated change in strength, in Elo points
    results = np.zeros((n_sims, n_fixtures), dtype=bool)

    for g in range(n_fixtures):
        h, a = int(inputs.fixture_home[g]), int(inputs.fixture_away[g])
        z = inputs.fixture_logit[g] + (LOGIT_PER_ELO * (drift[:, h] - drift[:, a]) if hot else 0.0)
        p = sigmoid(z)
        home_won = rng.random(n_sims) < p
        results[:, g] = home_won
        wins[:, h] += home_won
        wins[:, a] += ~home_won
        if same_conf[g]:
            conf_wins[:, h] += home_won
            conf_wins[:, a] += ~home_won
        if hot:
            delta = k_factor * (home_won - p)
            drift[:, h] += delta
            drift[:, a] -= delta

    unscheduled = np.zeros(n_teams, dtype=int) if inputs.unscheduled is None else np.asarray(inputs.unscheduled, dtype=int)
    if unscheduled.any():
        if inputs.pair_logit is None:
            raise SimulationError("pair_logit is required to simulate unscheduled games")
        z_pair = inputs.pair_logit
        off_diag = ~np.eye(n_teams, dtype=bool)
        neutral = 0.5 * (z_pair - z_pair.T)  # log-odds of row beating column on a neutral floor
        generic = np.array([neutral[t][off_diag[t]].mean() for t in range(n_teams)])
        for t in np.flatnonzero(unscheduled > 0):
            p = np.broadcast_to(sigmoid(generic[t] + (LOGIT_PER_ELO * drift[:, t] if hot else 0.0)), (n_sims,))
            wins[:, t] += (rng.random((n_sims, unscheduled[t])) < p[:, None]).sum(axis=1)

    scheduled_games = np.bincount(np.concatenate([inputs.fixture_home, inputs.fixture_away]).astype(int), minlength=n_teams) if n_fixtures else np.zeros(n_teams, int)
    total_games = np.asarray(inputs.wins) + np.asarray(inputs.losses) + scheduled_games + unscheduled

    # Standings: wins, then conference wins, then a random tie-break.
    score = wins * 1000.0 + conf_wins + rng.random((n_sims, n_teams))
    rank = np.zeros((n_sims, n_teams), dtype=np.int16)
    rows = np.arange(n_sims)
    conf_order: dict[str, np.ndarray] = {}
    for conf in np.unique(inputs.conference):
        members = np.flatnonzero(inputs.conference == conf)
        order = members[np.argsort(-score[:, members], axis=1)]  # (n_sims, members) best first
        conf_order[str(conf)] = order
        for position in range(order.shape[1]):
            rank[rows, order[:, position]] = position + 1

    made_playoffs = np.zeros((n_sims, n_teams), dtype=bool)
    rounds = {name: np.zeros((n_sims, n_teams), dtype=bool) for name in ("second_round", "conf_finals", "finals", "champion")}
    champion = np.full(n_sims, -1, dtype=int)
    if simulate_playoffs and inputs.pair_logit is not None:
        shift = LOGIT_PER_ELO * drift if hot else np.zeros_like(drift)
        z_pair = inputs.pair_logit
        finalists = []
        for conf, order in conf_order.items():
            seeds = [order[:, i] for i in range(PLAYIN_LAST_SEED)]
            seven, loser_78 = _game_winner(rng, seeds[6], seeds[7], z_pair, shift)  # 7 hosts 8
            winner_910, _ = _game_winner(rng, seeds[8], seeds[9], z_pair, shift)  # 9 hosts 10
            eight, _ = _game_winner(rng, loser_78, winner_910, z_pair, shift)
            field_ = seeds[:DIRECT_SEEDS] + [seven, eight]
            for team in field_:
                made_playoffs[rows, team] = True
            first_round = [(field_[0], field_[7]), (field_[3], field_[4]), (field_[1], field_[6]), (field_[2], field_[5])]
            quarter = [_series_winner(rng, hi, lo, z_pair, shift) for hi, lo in first_round]
            for team in quarter:
                rounds["second_round"][rows, team] = True
            semi = [_series_winner(rng, *_order_by_strength(quarter[0], quarter[1], score), z_pair, shift),
                    _series_winner(rng, *_order_by_strength(quarter[2], quarter[3], score), z_pair, shift)]
            for team in semi:
                rounds["conf_finals"][rows, team] = True
            conf_champion = _series_winner(rng, *_order_by_strength(semi[0], semi[1], score), z_pair, shift)
            rounds["finals"][rows, conf_champion] = True
            finalists.append(conf_champion)
        if len(finalists) == 2:
            champion = _series_winner(rng, *_order_by_strength(finalists[0], finalists[1], score), z_pair, shift)
            rounds["champion"][rows, champion] = True
    else:
        made_playoffs = rank <= PLAYOFF_SEEDS

    table = pd.DataFrame(
        {
            "team": inputs.team_abbr,
            "team_id": inputs.team_ids,
            "conference": inputs.conference,
            "current_wins": np.asarray(inputs.wins, dtype=int),
            "current_losses": np.asarray(inputs.losses, dtype=int),
            "games_total": total_games.astype(int),
            "mean_wins": wins.mean(axis=0),
            "mean_losses": total_games - wins.mean(axis=0),
            "wins_p10": np.quantile(wins, 0.10, axis=0),
            "wins_median": np.quantile(wins, 0.50, axis=0),
            "wins_p90": np.quantile(wins, 0.90, axis=0),
            "mean_rank": rank.mean(axis=0),
            "p_top6": (rank <= DIRECT_SEEDS).mean(axis=0),
            "p_playin": ((rank > DIRECT_SEEDS) & (rank <= PLAYIN_LAST_SEED)).mean(axis=0),
            "p_playoffs": made_playoffs.mean(axis=0),
            "p_second_round": rounds["second_round"].mean(axis=0),
            "p_conf_finals": rounds["conf_finals"].mean(axis=0),
            "p_finals": rounds["finals"].mean(axis=0),
            "p_champion": rounds["champion"].mean(axis=0),
        }
    )  # rows stay in input team order so they line up with the per-simulation arrays
    return SimulationResult(table=table, wins=wins, rank=rank, made_playoffs=made_playoffs, champion=champion,
                            fixture_results=results, n_sims=n_sims, seed=seed, hot=hot)


def game_leverage(result: SimulationResult, inputs: SimulationInputs, max_games: int = 40) -> pd.DataFrame:
    """How much each upcoming game moves the two teams' playoff odds.

    For a game ``g``: ``swing = P(playoffs | team wins g) - P(playoffs | team loses g)``,
    estimated from the simulated seasons. ``leverage`` is the sum of both teams' swings.
    """
    rows = []
    for g in range(min(max_games, result.fixture_results.shape[1])):
        home_won = result.fixture_results[:, g]
        if home_won.all() or not home_won.any():
            continue
        h, a = int(inputs.fixture_home[g]), int(inputs.fixture_away[g])
        swing_home = result.made_playoffs[home_won, h].mean() - result.made_playoffs[~home_won, h].mean()
        swing_away = result.made_playoffs[~home_won, a].mean() - result.made_playoffs[home_won, a].mean()
        rows.append(
            {
                "game_id": inputs.fixture_ids[g] if inputs.fixture_ids else str(g),
                "game_date": inputs.fixture_dates[g] if inputs.fixture_dates else "",
                "home_team": inputs.team_abbr[h], "away_team": inputs.team_abbr[a],
                "p_home": float(sigmoid(inputs.fixture_logit[g])),
                "swing_home": float(swing_home), "swing_away": float(swing_away),
                "leverage": float(abs(swing_home) + abs(swing_away)),
            }
        )
    out = pd.DataFrame(rows)
    return out if out.empty else out.sort_values("leverage", ascending=False).reset_index(drop=True)


def build_simulation_inputs(predictor) -> tuple[SimulationInputs, float]:
    """Assemble simulator inputs from a :class:`~nba_predictor.predict.Predictor`.

    Returns the inputs and the Elo K-factor to use for hot simulations (the tuned K times the
    historical average margin-of-victory multiplier).
    """
    teams = predictor.teams.reset_index(drop=True)
    position = {int(t): i for i, t in enumerate(teams["team_id"])}
    conference = teams["conference"].to_numpy()
    games = predictor.games
    current = games[(games["season"] == predictor.current_season) & (games["game_type"] == "regular")]
    wins, losses, conf_wins = np.zeros(len(teams), int), np.zeros(len(teams), int), np.zeros(len(teams), int)
    for team_id, part in current.groupby("team_id"):
        i = position[int(team_id)]
        wins[i] = int(part["win"].sum())
        losses[i] = int(len(part) - part["win"].sum())
        same = part["opp_team_id"].map(lambda t: conference[position[int(t)]]) == conference[i]
        conf_wins[i] = int(part.loc[same, "win"].sum())

    matchups = predictor.features.matchups
    pending = matchups[~matchups["is_played"] & (matchups["game_type"] == "regular")]
    if len(pending):
        scored = predictor.score(pending).sort_values(["game_date", "game_id"])
        home = scored["home_team_id"].map(position).to_numpy()
        away = scored["away_team_id"].map(position).to_numpy()
        logits = scored["logit_ensemble"].to_numpy(dtype=float)
        ids = scored["game_id"].tolist()
        dates = pd.to_datetime(scored["game_date"]).dt.strftime("%Y-%m-%d").tolist()
    else:
        home, away, logits, ids, dates = np.array([], int), np.array([], int), np.array([], float), [], []
    scheduled = np.bincount(np.concatenate([home, away]).astype(int), minlength=len(teams)) if len(home) else np.zeros(len(teams), int)
    unscheduled = np.clip(config.GAMES_PER_SEASON - wins - losses - scheduled, 0, None)
    _, pair_logit = predictor.pairwise_logits()

    inputs = SimulationInputs(
        team_ids=teams["team_id"].to_numpy(), team_abbr=teams["team_abbr"].tolist(), conference=conference,
        wins=wins, losses=losses, conf_wins=conf_wins, fixture_home=home, fixture_away=away, fixture_logit=logits,
        fixture_ids=ids, fixture_dates=dates, unscheduled=unscheduled, pair_logit=pair_logit,
    )
    k_factor = predictor.bundle.elo_params.k * float(predictor.bundle.metadata.get("mean_mov_multiplier") or 1.0)
    return inputs, k_factor
