"""Console report shared by the backtest and replay entry points."""

from __future__ import annotations

import pandas as pd

from rugby_ranking.backtest.harness import BacktestResult


def print_report(res: BacktestResult, reference: str | None = None, by_competition: bool = True) -> None:
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)
    print(f"\n{res.n_fixtures} fixtures; coverage:\n{res.coverage().to_string()}")
    print("\nOverall (matches every forecaster predicted):\n", res.summary().to_string())
    if by_competition:
        print("\nBy competition:\n", res.summary(by=["competition"]).to_string())
    names = res.forecasters
    ref = reference or names[0]
    for a in names:
        if a != ref:
            print(f"\nPaired difference {a} - {ref} (negative favours {a}):\n", res.compare(a, ref).to_string())
    if res.failures:
        print(f"\n{len(res.failures)} failed fits, e.g. {res.failures[0]}")
