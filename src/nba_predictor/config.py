"""Central configuration: paths, season boundaries, feature groups and model settings.

Everything tunable lives here so that application logic never hard-codes a path,
a season boundary or a hyperparameter grid. Secrets (API keys) are read from the
environment / a local ``.env`` file and are never stored in this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: load a local .env so ANTHROPIC_API_KEY etc. are picked up
    from dotenv import load_dotenv

    load_dotenv(Path(os.environ.get("NBA_PREDICTOR_ROOT", Path(__file__).resolve().parents[2])) / ".env")
except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
    pass

# --------------------------------------------------------------------------- paths
PROJECT_ROOT: Path = Path(os.environ.get("NBA_PREDICTOR_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_GAMES_PATH: Path = DATA_DIR / "raw" / "games_raw.csv"
SCHEDULE_PATH: Path = DATA_DIR / "raw" / "schedule.csv"
PLAYER_LOGS_PATH: Path = DATA_DIR / "raw" / "player_logs.csv"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MATCHUPS_PATH: Path = PROCESSED_DIR / "matchups.csv"
TEAM_GAMES_PATH: Path = PROCESSED_DIR / "team_games.csv"
LEDGER_PATH: Path = DATA_DIR / "predictions" / "ledger.csv"
MODELS_DIR: Path = PROJECT_ROOT / "models"
BUNDLE_PATH: Path = MODELS_DIR / "model_bundle.joblib"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"
BACKTEST_PATH: Path = MODELS_DIR / "backtest_predictions.csv"

# ------------------------------------------------------------------------- seasons
# Seasons are identified by their start year (2024 == the 2024-25 season).
FIRST_SEASON: int = 2014
#: The first season only warms up Elo and the previous-season priors; it is never trained on.
BURN_IN_SEASONS: int = 1
#: Final hold-out seasons. No tuning, feature selection or calibration ever sees them.
TEST_SEASONS: tuple[int, ...] = (2025, 2026)
#: Walk-forward validation starts here (each fold trains on all earlier seasons).
CV_FIRST_VALIDATION_SEASON: int = 2019
RANDOM_SEED: int = 42

GAME_TYPE_BY_PREFIX: dict[str, str] = {
    "001": "preseason",
    "002": "regular",
    "003": "allstar",
    "004": "playoffs",
    "005": "playin",
    "006": "cup_final",
}
#: Game types that update team strength (Elo, rolling form).
COMPETITIVE_GAME_TYPES: tuple[str, ...] = ("regular", "playoffs", "playin", "cup_final")
NBA_TEAM_ID_RANGE: tuple[int, int] = (1610612737, 1610612766)
#: 2020 restart at Walt Disney World: every game was played at a neutral site.
BUBBLE_DATES: tuple[str, str] = ("2020-07-30", "2020-10-11")
GAMES_PER_SEASON: int = 82

CONFERENCE_BY_ABBR: dict[str, str] = {
    **{a: "East" for a in ("ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS")},
    **{a: "West" for a in ("DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA")},
}


# ------------------------------------------------------------------------ features
@dataclass(frozen=True)
class FeatureConfig:
    """Hyperparameters of the feature pipeline (fixed a priori, not tuned on test data)."""

    long_halflife: float = 30.0  # games; season-level team quality
    long_prior_games: float = 12.0  # weight (in games) of the previous-season prior
    short_halflife: float = 5.0  # games; recent form
    short_prior_games: float = 4.0
    venue_halflife: float = 25.0  # home-only / away-only splits
    venue_prior_games: float = 10.0
    season_carry: float = 0.6  # share of last season's deviation from league mean kept as prior
    rest_cap_days: int = 7
    elo_trend_games: int = 10
    h2h_max_games: int = 6
    h2h_window_days: int = 1100
    h2h_shrink_games: float = 3.0
    league_hca_window_days: int = 365
    league_hca_min_games: int = 200
    roster_top_n: int = 8  # rotation size for the roster-availability proxy
    roster_recent_games: int = 3


#: Model inputs, grouped for ablation. Every name is a pre-game quantity built in ``features.py``.
FEATURE_GROUPS: dict[str, list[str]] = {
    "elo": ["elo_diff"],
    "efficiency": ["net_rtg_diff", "off_rtg_diff", "win_pct_diff"],
    "four_factors": [
        "efg_diff", "tov_pct_diff", "orb_pct_diff", "ft_rate_diff",
        "opp_efg_diff", "opp_tov_pct_diff", "drb_pct_diff", "opp_ft_rate_diff",
    ],
    "form": ["net_rtg_short_diff", "win_pct_short_diff", "elo_trend_diff"],
    "venue": ["venue_net_rtg_diff", "venue_win_pct_diff"],
    "rest": ["rest_days_home", "rest_days_away", "rest_diff", "b2b_home", "b2b_away"],
    "schedule": ["games_last_7d_diff", "three_in_four_home", "three_in_four_away", "away_road_streak"],
    "h2h": ["h2h_margin", "h2h_games"],
    "context": ["league_home_win_pct", "season_progress"],
}
#: Optional groups that are only available when extra data has been downloaded.
OPTIONAL_FEATURE_GROUPS: dict[str, list[str]] = {
    "roster": ["roster_avail_diff", "roster_avail_home", "roster_avail_away"],
}
#: Features whose effect on the home win probability is monotone by construction.
MONOTONE_INCREASING: tuple[str, ...] = ("elo_diff", "net_rtg_diff", "win_pct_diff", "venue_net_rtg_diff")


def all_feature_columns(groups: dict[str, list[str]] | None = None) -> list[str]:
    """Flatten feature groups into an ordered, duplicate-free column list."""
    groups = FEATURE_GROUPS if groups is None else groups
    seen: dict[str, None] = {}
    for cols in groups.values():
        for col in cols:
            seen.setdefault(col, None)
    return list(seen)


# ------------------------------------------------------------------------ training
@dataclass(frozen=True)
class TrainingConfig:
    """Settings of the model-selection procedure."""

    lr_c_grid: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    xgb_search_iterations: int = 40
    xgb_default_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 400, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 2.0, "gamma": 0.0,
        }
    )
    #: Elo grid: tuned on development seasons only (log loss of the pure Elo forecast).
    elo_grid: dict = field(
        default_factory=lambda: {
            "k": (12.0, 16.0, 20.0, 24.0),
            "home_advantage": (40.0, 60.0, 80.0, 100.0),
            "season_carry": (0.3, 0.45, 0.6, 0.75, 0.9),
            "use_mov": (True, False),
        }
    )
    ablation_tolerance: float = 0.0  # drop a group if removing it changes CV log loss by <= this
    bootstrap_replicas: int = 15  # bagged models used for the uncertainty band
    n_bootstrap_ci: int = 2000  # resamples for confidence intervals on metrics
    calibration_bins: int = 10
    n_jobs: int = max(1, min(8, (os.cpu_count() or 2) - 2))


# ----------------------------------------------------------------------------- LLM
@dataclass(frozen=True)
class LLMConfig:
    """Settings of the optional LLM analyst.

    Credentials are never stored here: the Anthropic SDK resolves them from the environment
    (``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN``, e.g. via a local ``.env`` file) or from
    an ``ant auth login`` profile.
    """

    enabled: bool = os.environ.get("NBA_LLM_ENABLED", "1").lower() not in ("0", "false", "no")
    model: str = os.environ.get("NBA_LLM_MODEL", "claude-opus-5-5")
    effort: str = os.environ.get("NBA_LLM_EFFORT", "medium")
    max_tokens: int = int(os.environ.get("NBA_LLM_MAX_TOKENS", "16000"))
    timeout_seconds: float = float(os.environ.get("NBA_LLM_TIMEOUT", "120"))
    max_attempts: int = 2  # one retry when the first answer fails the grounding checks

    @staticmethod
    def credentials_hint() -> str:
        """Which credential source is visible in the environment (never the value itself)."""
        if os.environ.get("ANTHROPIC_API_KEY", "").strip():
            return "ANTHROPIC_API_KEY"
        if os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip():
            return "ANTHROPIC_AUTH_TOKEN"
        return "none in environment"


def season_label(start_year: int) -> str:
    """``2024`` -> ``'2024-25'`` (the format used by the NBA stats API)."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"
