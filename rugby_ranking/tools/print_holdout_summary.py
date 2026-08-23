"""
Print a Markdown summary of a holdout_validation_report.json for GitHub
Actions' step summary (or any Markdown-consuming destination).

Kept as a standalone, testable script rather than inline shell in the
workflow YAML -- embedding Python inside a bash `run:` block inside YAML is
exactly the kind of fragile, hard-to-review, easy-to-silently-break pattern
that caused the original CI dependency bug this tooling was added to catch.

Usage:
    python -m rugby_ranking.tools.print_holdout_summary path/to/holdout_validation_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def format_summary(report: dict) -> str:
    match_test = report.get("match_level_test", {})
    model = match_test.get("model", {})
    baseline = match_test.get("baseline_always_predict_mode_outcome", {})
    n_matches = match_test.get("n_matches")
    as_of = str(report.get("as_of", "?"))[:10]

    def pct(d: dict, key: str) -> str:
        v = d.get(key)
        return f"{v:.1%}" if isinstance(v, (int, float)) else "n/a"

    def num(d: dict, key: str, fmt: str = ".4f") -> str:
        v = d.get(key)
        return format(v, fmt) if isinstance(v, (int, float)) else "n/a"

    lines = [
        f"Evaluated on {n_matches} held-out matches, as of {as_of}.",
        "",
        "| Metric | Model | Baseline (train-set rate/mean) |",
        "|---|---|---|",
        f"| Win accuracy | {pct(model, 'win_accuracy')} | {pct(baseline, 'win_accuracy')} |",
        f"| Brier score | {num(model, 'brier_score')} | {num(baseline, 'brier_score')} |",
        f"| Score RMSE | {num(model, 'score_rmse', '.2f')} | "
        f"{num(baseline, 'score_rmse_predicting_mean_score', '.2f')} |",
        "",
        "Not gated on thresholds yet -- see ROADMAP.md.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python -m rugby_ranking.tools.print_holdout_summary <report.json>", file=sys.stderr)
        sys.exit(1)
    report_path = Path(sys.argv[1])
    with open(report_path) as f:
        report = json.load(f)
    print(format_summary(report))
