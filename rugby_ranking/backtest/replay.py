"""Re-score stored forecasts, optionally with a calibration layer.

    python -m rugby_ranking.backtest.replay forecasts.csv [more.csv ...] \\
        --calibrate [--forecasters bayes-prod,elo] [--reference elo]

The CSVs are what ``python -m rugby_ranking.backtest --out`` writes; several
are concatenated. Nothing is refit, so a correction can be tried in seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rugby_ranking.backtest.calibration import CalibrationConfig
from rugby_ranking.backtest.harness import BacktestResult
from rugby_ranking.backtest.report import print_report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", type=Path)
    p.add_argument("--calibrate", action="store_true", help="add <name>+cal versions")
    p.add_argument("--forecasters", default=None, help="comma-separated names to calibrate (default all)")
    p.add_argument("--reference", default=None, help="forecaster to compare the others against")
    p.add_argument("--only", default=None, help="comma-separated forecasters to keep in the report")
    p.add_argument("--no-level", action="store_true")
    p.add_argument("--no-margin", action="store_true")
    p.add_argument("--no-spread", action="store_true")
    p.add_argument("--shrinkage", type=float, default=CalibrationConfig.shrinkage)
    p.add_argument("--half-life", type=float, default=None, help="origins; down-weights older errors")
    p.add_argument("--min-history", type=int, default=CalibrationConfig.min_history)
    p.add_argument("--no-by-competition", action="store_true")
    args = p.parse_args(argv)

    res = BacktestResult.from_csv(*args.csv)
    if args.calibrate:
        cfg = CalibrationConfig(
            level=not args.no_level, margin=not args.no_margin, spread=not args.no_spread,
            shrinkage=args.shrinkage, half_life_origins=args.half_life, min_history=args.min_history,
        )
        res = res.with_calibration(cfg, args.forecasters.split(",") if args.forecasters else None)
    if args.only:
        keep = set(args.only.split(","))
        res = BacktestResult.from_frame(res.frame[res.frame["forecaster"].isin(keep)])
    print_report(res, args.reference, by_competition=not args.no_by_competition)


if __name__ == "__main__":
    main()
