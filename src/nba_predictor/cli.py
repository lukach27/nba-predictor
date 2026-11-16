"""Command-line interface.

Examples::

    python -m nba_predictor fetch              # download games + schedule
    python -m nba_predictor train              # full training protocol, saves models/
    python -m nba_predictor update             # fetch, rebuild features, log + grade predictions
    python -m nba_predictor upcoming --days 3  # forecasts for the next games
    python -m nba_predictor predict BOS LAL    # one matchup (home team first)
    python -m nba_predictor analyze BOS LAL    # structured tactical analysis (JSON)
    python -m nba_predictor simulate --sims 10000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from . import config


def _predictor():
    from .predict import Predictor

    return Predictor.from_disk()


def cmd_fetch(args: argparse.Namespace) -> int:
    from .data_collection import update_data

    print(json.dumps(update_data(full=args.full), indent=2))
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    from .features import save_processed

    predictor = _predictor()
    save_processed(predictor.features)
    matchups = predictor.features.matchups
    print(f"Saved {len(matchups)} matchup rows ({int((~matchups['is_played']).sum())} upcoming) to {config.PROCESSED_DIR}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train_model import summarize, train

    extras = None
    if args.roster:
        from .roster import load_team_extras

        extras = load_team_extras()
    result = train(team_extras=extras)
    print(summarize(result.metrics))
    return 0


def cmd_upcoming(args: argparse.Namespace) -> int:
    table = _predictor().upcoming(days=args.days)
    if table.empty:
        print("No upcoming games on the schedule.")
        return 0
    view = table[["game_date", "away_team", "home_team", "p_ensemble", "p_lr", "p_xgb", "p_elo", "expected_margin"]].copy()
    view["game_date"] = pd.to_datetime(view["game_date"]).dt.strftime("%Y-%m-%d")
    view = view.rename(columns={"p_ensemble": "p_home"})
    print(view.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    prediction = _predictor().predict_matchup(args.home, args.away, game_date=args.date, is_neutral=args.neutral)
    if args.json:
        print(json.dumps(prediction.to_dict(), indent=2))
        return 0
    venue = "neutral site" if prediction.is_neutral else f"at {prediction.home_team}"
    print(f"{prediction.away_team} @ {prediction.home_team} ({venue}) on {prediction.game_date}")
    print(f"  Home win probability : {prediction.p_home:.1%}  (80% band {prediction.p_low:.1%} - {prediction.p_high:.1%})")
    print(f"  Logistic regression  : {prediction.p_lr:.1%}   XGBoost: {prediction.p_xgb:.1%}   pure Elo: {prediction.p_elo:.1%}")
    if prediction.expected_margin is not None:
        print(f"  Expected margin      : {prediction.home_team} {prediction.expected_margin:+.1f}")
    print(f"  Confidence           : {prediction.confidence['level']} ({prediction.confidence['score']:.0f}/100)")
    print(f"  Volatility risk      : {prediction.volatility['level']} ({prediction.volatility['score']:.0f}/100)")
    print("  Top factors (log-odds, + favours home):")
    for row in prediction.contributions[:5]:
        print(f"    {row['feature']:<22} {row['contribution']:+.3f}   (value {row['value']:.3f})")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .llm_model import analyze_matchup

    prediction = _predictor().predict_matchup(args.home, args.away, game_date=args.date, is_neutral=args.neutral)
    report = analyze_matchup(prediction, use_llm=not args.no_llm)
    print(json.dumps(report.model_dump(), indent=2))
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from .simulation import build_simulation_inputs, simulate_season

    inputs, k_factor = build_simulation_inputs(_predictor())
    result = simulate_season(inputs, n_sims=args.sims, seed=args.seed, hot=not args.static, k_factor=k_factor)
    cols = ["team", "current_wins", "current_losses", "mean_wins", "p_top6", "p_playoffs", "p_finals", "p_champion"]
    for conference in ("East", "West"):
        print(f"\n{conference} ({args.sims} simulations, seed {args.seed}, {'hot' if result.hot else 'static'})")
        print(result.standings(conference)[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    """Daily routine: refresh data, rebuild features, log new pre-game forecasts, grade old ones."""
    from .data_collection import update_data
    from .features import save_processed
    from .ledger import grade_ledger, ledger_metrics, load_ledger, record_predictions

    status = update_data()
    predictor = _predictor()
    save_processed(predictor.features)
    upcoming = predictor.upcoming(days=3)
    added = record_predictions(upcoming, predictor.schedule, predictor.bundle.metadata.get("trained_at", "unknown"))
    graded = grade_ledger(load_ledger(), predictor.games)
    print(json.dumps({"data": status, "predictions_logged": int(len(added)), "ledger": ledger_metrics(graded)}, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nba_predictor", description="NBA match predictor")
    parser.add_argument("-v", "--verbose", action="store_true", help="show progress logs")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download games and schedule from the NBA API")
    p.add_argument("--full", action="store_true", help="re-download every season")
    p.set_defaults(func=cmd_fetch)

    sub.add_parser("features", help="rebuild data/processed with the trained model's settings").set_defaults(func=cmd_features)

    p = sub.add_parser("train", help="run the full training and evaluation protocol")
    p.add_argument("--roster", action="store_true", help="include the roster-availability features (needs player logs)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("upcoming", help="forecast scheduled games")
    p.add_argument("--days", type=int, default=7)
    p.set_defaults(func=cmd_upcoming)

    for name, func, text in (("predict", cmd_predict, "forecast one matchup"),
                             ("analyze", cmd_analyze, "tactical analysis (JSON) for one matchup")):
        p = sub.add_parser(name, help=text)
        p.add_argument("home", help="home team (abbreviation, e.g. BOS)")
        p.add_argument("away", help="away team")
        p.add_argument("--date", default=None, help="game date YYYY-MM-DD (default: day after the last game)")
        p.add_argument("--neutral", action="store_true", help="neutral-site game")
        if name == "predict":
            p.add_argument("--json", action="store_true", help="print the full prediction as JSON")
        else:
            p.add_argument("--no-llm", action="store_true", help="use the rule-based analyst only")
        p.set_defaults(func=func)

    p = sub.add_parser("simulate", help="Monte Carlo season simulation")
    p.add_argument("--sims", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--static", action="store_true", help="disable team-strength drift")
    p.set_defaults(func=cmd_simulate)

    sub.add_parser("update", help="daily routine: fetch, rebuild, log and grade predictions").set_defaults(func=cmd_update)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
