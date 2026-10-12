"""python -m rugby_ranking.backtest --data-dir ../Rugby-Data --start 2022-09-01"""

from __future__ import annotations

import argparse
from pathlib import Path

from rugby_ranking.backtest import BacktestConfig, run_backtest
from rugby_ranking.backtest.report import print_report
from rugby_ranking.forecast import ConstantBaseline, EloForecaster, MatchStore


def _build(names: list[str], args) -> list:
    out = []
    for n in names:
        if n == "constant":
            out.append(ConstantBaseline())
        elif n == "elo":
            out.append(EloForecaster())
        elif n in ("bayes", "bayes-prod"):
            from rugby_ranking.forecast.bayes import BayesianForecaster, production_config
            out.append(BayesianForecaster(
                config=production_config() if n == "bayes-prod" else None,
                vi_iterations=args.vi_iterations, name=n))
        else:
            raise SystemExit(f"unknown forecaster {n!r} (choose from constant, elo, bayes, bayes-prod)")
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--start", required=True, help="first origin, e.g. 2022-09-01")
    p.add_argument("--end", default=None)
    p.add_argument("--step-days", type=int, default=28)
    p.add_argument("--forecasters", default="constant,elo")
    p.add_argument("--competitions", default=None, help="comma-separated eval competitions")
    p.add_argument("--vi-iterations", type=int, default=20000)
    p.add_argument("--out", type=Path, default=None, help="write forecasts as CSV here")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    store = MatchStore.from_data_dir(args.data_dir)
    comps = tuple(args.competitions.split(",")) if args.competitions else None
    cfg = BacktestConfig(args.start, args.end, args.step_days, eval_competitions=comps)
    res = run_backtest(store, _build(args.forecasters.split(","), args), cfg,
                       log=(lambda m: None) if args.quiet else print)

    print_report(res)
    if args.out:
        res.frame.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
