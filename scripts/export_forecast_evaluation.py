"""Summarise stored backtest forecasts as JSON for the blog's forecasting figures.

Input is the CSV written by ``python -m rugby_ranking.backtest ... --out``. Output
(one file) holds, for each forecaster and its calibrated ``+cal`` version:
overall and per-competition scores, a reliability table for the home-win
probability, per-origin mean log loss, and the cumulative log-loss advantage over
the constant baseline through time. Scores use only matches every forecaster
predicted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rugby_ranking.backtest import BacktestResult, metrics
from rugby_ranking.backtest.calibration import CalibrationConfig


def wilson(k: float, n: int, z: float = 1.645) -> tuple[float, float]:
    """90% Wilson interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("forecasts", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--baseline", default="constant")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    res = BacktestResult.from_csv(args.forecasts)
    res = res.with_calibration(CalibrationConfig(margin=False))
    df = metrics.add_losses(metrics.common_matches(res.frame))
    df["home_won"] = (df["outcome"] == 0).astype(float)
    names = sorted(df["forecaster"].unique(), key=lambda f: df[df.forecaster == f]["logloss"].mean())
    r = lambda v, k=4: None if pd.isna(v) else round(float(v), k)  # noqa: E731

    summary = []
    for f in names:
        g = df[df.forecaster == f]
        summary.append(dict(
            f=f, n=len(g), logloss=r(g["logloss"].mean()), brier=r(g["brier"].mean()), accuracy=r(g["correct"].mean()),
            margin_mae=r(g["abs_margin_err"].mean(), 2), total_mae=r(g["abs_total_err"].mean(), 2),
            ci90=r(pd.concat([g["home_in_ci"], g["away_in_ci"]]).mean(), 3),
            margin_z_sd=r(g["margin_z"].std(), 3)))

    by_comp = {c: {f: r(g[g.forecaster == f]["logloss"].mean()) for f in names}
               for c, g in df.groupby("competition")}
    counts = df[df.forecaster == names[0]].groupby("competition").size().to_dict()

    edges = np.linspace(0, 1, args.bins + 1)
    reliability = {}
    for f in names:
        g = df[df.forecaster == f].copy()
        g["bin"] = np.clip(np.digitize(g["home_win_prob"], edges) - 1, 0, args.bins - 1)
        rows = []
        for b, gb in g.groupby("bin"):
            lo, hi = wilson(gb["home_won"].sum(), len(gb))
            rows.append(dict(lo=r(edges[b], 2), hi=r(edges[b + 1], 2), n=len(gb), pred=r(gb["home_win_prob"].mean()),
                             obs=r(gb["home_won"].mean()), obs_lo=r(lo), obs_hi=r(hi)))
        reliability[f] = rows

    blocks = []
    for o, g in df.groupby("origin"):
        blocks.append(dict(origin=o.strftime("%Y-%m-%d"), n=int(g[g.forecaster == names[0]].shape[0]),
                           loss={f: r(g[g.forecaster == f]["logloss"].mean()) for f in names}))

    wide = df.pivot(index="match_id", columns="forecaster", values="logloss")
    date = df.drop_duplicates("match_id").set_index("match_id")["date"].reindex(wide.index)
    order = date.sort_values().index
    cum = (wide.loc[order].sub(wide.loc[order, args.baseline], axis=0)).cumsum()
    cum["date"] = date.loc[order].dt.strftime("%Y-%m-%d")
    keep = cum.iloc[np.linspace(0, len(cum) - 1, 120).astype(int)]
    cumulative = dict(date=keep["date"].tolist(), **{f: [r(v, 2) for v in keep[f]] for f in names})

    out = dict(
        start=df["origin"].min().strftime("%Y-%m-%d"), end=df["date"].max().strftime("%Y-%m-%d"),
        n_matches=int(df["match_id"].nunique()), n_origins=int(df["origin"].nunique()), baseline=args.baseline,
        forecasters=names, summary=summary, by_competition=by_comp, competition_n=counts,
        reliability=reliability, blocks=blocks, cumulative=cumulative)
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{out['n_matches']} matches, {out['n_origins']} origins; wrote {args.out}")
    for s in summary:
        print(f"  {s['f']:<14} logloss {s['logloss']}  acc {s['accuracy']}  margin_mae {s['margin_mae']}  ci90 {s['ci90']}")


if __name__ == "__main__":
    main()
