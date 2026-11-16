"""Prediction ledger: an append-only record of forecasts made *before* tip-off.

A backtest can always be suspected of hindsight. The ledger cannot: a row is written only
when the scheduled tip-off time is still in the future, it is never overwritten, and it is
graded later against the official result. Over time this becomes the model's live track
record.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .metrics import evaluate_probabilities

logger = logging.getLogger(__name__)

LEDGER_COLUMNS: list[str] = [
    "game_id", "game_date", "tipoff_utc", "home_team", "away_team", "home_team_id", "away_team_id",
    "p_home", "p_lr", "p_xgb", "p_elo", "expected_margin", "model_trained_at", "predicted_at_utc",
]


def load_ledger(path: Path | None = None) -> pd.DataFrame:
    """Read the ledger (an empty frame with the right columns if it does not exist yet)."""
    path = config.LEDGER_PATH if path is None else Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    ledger = pd.read_csv(path, dtype={"game_id": str})
    missing = [c for c in LEDGER_COLUMNS if c not in ledger.columns]
    if missing:
        raise ValueError(f"{path} lacks ledger columns {missing}")
    return ledger


def record_predictions(predictions: pd.DataFrame, schedule: pd.DataFrame, model_trained_at: str,
                       now: datetime | None = None, horizon_days: float = 2.0, path: Path | None = None) -> pd.DataFrame:
    """Append forecasts for games that tip off within ``horizon_days`` and have not started.

    ``predictions`` is the output of :meth:`Predictor.upcoming`. A game is skipped when it is
    already in the ledger (first forecast wins), when its tip-off time is unknown, or when
    tip-off is not strictly in the future. Returns the rows that were added.
    """
    path = config.LEDGER_PATH if path is None else Path(path)
    now = datetime.now(timezone.utc) if now is None else now.astimezone(timezone.utc)
    ledger = load_ledger(path)
    if predictions is None or predictions.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    tipoff = pd.to_datetime(schedule.set_index("game_id")["game_time_utc"], errors="coerce", utc=True)
    rows = predictions.copy()
    rows["tipoff"] = rows["game_id"].map(tipoff)
    eligible = (
        rows["tipoff"].notna()
        & (rows["tipoff"] > pd.Timestamp(now))
        & (rows["tipoff"] <= pd.Timestamp(now) + pd.Timedelta(days=horizon_days))
        & ~rows["game_id"].isin(set(ledger["game_id"]))
    )
    rows = rows[eligible]
    if rows.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    new = pd.DataFrame(
        {
            "game_id": rows["game_id"].astype(str),
            "game_date": pd.to_datetime(rows["game_date"]).dt.strftime("%Y-%m-%d"),
            "tipoff_utc": rows["tipoff"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_team": rows["home_team"], "away_team": rows["away_team"],
            "home_team_id": rows["home_team_id"].astype("int64"), "away_team_id": rows["away_team_id"].astype("int64"),
            "p_home": rows["p_ensemble"].round(6), "p_lr": rows["p_lr"].round(6), "p_xgb": rows["p_xgb"].round(6),
            "p_elo": rows["p_elo"].round(6), "expected_margin": rows["expected_margin"].round(3),
            "model_trained_at": model_trained_at,
            "predicted_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )[LEDGER_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = new if ledger.empty else pd.concat([ledger, new], ignore_index=True)
    combined.to_csv(path, index=False)
    logger.info("Ledger: recorded %d new pre-game predictions (%d total)", len(new), len(combined))
    return new


def grade_ledger(ledger: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach final results to ledger rows. Ungraded games keep ``home_win`` = NaN."""
    graded = ledger.copy()
    results = games[games["is_home"]].set_index("game_id")
    graded["home_win"] = graded["game_id"].map(results["win"])
    graded["home_pts"] = graded["game_id"].map(results["PTS"])
    graded["away_pts"] = graded["game_id"].map(results["OPP_PTS"])
    graded["graded"] = graded["home_win"].notna()
    graded["correct"] = np.where(graded["graded"], (graded["p_home"] > 0.5) == (graded["home_win"] == 1), np.nan)
    return graded


def ledger_metrics(graded: pd.DataFrame) -> dict | None:
    """Live performance on graded ledger rows (``None`` until at least one game is graded)."""
    done = graded[graded["graded"]] if "graded" in graded.columns else graded.iloc[0:0]
    if done.empty:
        return None
    y = done["home_win"].to_numpy(dtype=float)
    out = {"n": int(len(done)), "pending": int((~graded["graded"]).sum())}
    for name, col in (("ensemble", "p_home"), ("elo", "p_elo")):
        p = done[col].to_numpy(dtype=float)
        out[name] = {"accuracy": float(np.mean((p > 0.5) == (y == 1))), "brier": float(np.mean((p - y) ** 2))}
        if len(np.unique(y)) == 2 and len(y) >= 20:
            out[name] = evaluate_probabilities(y, p)
    return out
