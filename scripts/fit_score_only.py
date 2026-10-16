"""Fit the score-only Bayesian team-strength model on every season in Rugby-Data.

Needs only final scores, so it covers seasons before the scoring-event data
begins (2011-12). Writes one row per team-season in the same shape as
``fit_strength_history.py`` plus a ``competition`` field, for
``export_strength_series.py --score-file``.

    python scripts/fit_score_only.py --data-dir ../Rugby-Data --out score_only_strengths.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rugby_ranking.forecast.store import MatchStore  # noqa: E402
from rugby_ranking.model import score_only as so  # noqa: E402

LEAGUES = ("celtic", "premiership", "top14", "pro-d2", "championship", "super-rugby")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, required=True, help="Rugby-Data checkout")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--competitions", default=",".join(LEAGUES))
    ap.add_argument("--vi-iterations", type=int, default=30000)
    ap.add_argument("--n-samples", type=int, default=500)
    args = ap.parse_args()

    store = MatchStore.from_data_dir(args.data_dir)
    m = store.played
    m = m[m["competition"].isin(args.competitions.split(","))]
    data = so.prepare(m)
    print(f"{len(m):,} matches, {len(data['ents'])} teams, seasons {data['years'][0]}-{data['years'][-1]}", flush=True)
    cfg = so.ScoreOnlyConfig(vi_iterations=args.vi_iterations)
    t = time.time()
    approx, post = so.fit(so.build(data, cfg), cfg, n_samples=args.n_samples)
    hist = approx.hist
    print(f"fit took {time.time() - t:.0f}s; loss {hist[-2000:-1000].mean():.5g} -> {hist[-1000:].mean():.5g}", flush=True)
    rows = so.summarise(data, post)
    args.out.write_text(json.dumps(rows.to_dict("records")))
    print(f"wrote {len(rows)} team-season rows to {args.out}")


if __name__ == "__main__":
    main()
