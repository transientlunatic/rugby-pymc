#!/usr/bin/env python3
"""
Demonstrate the impact of minimum score thresholds on rankings.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from rugby_ranking.model.data import MatchDataset


def analyze_threshold_impact(data_dir: Path | None = None) -> None:
    """Show how many players are affected by different thresholds."""
    data_dir = data_dir or Path("../Rugby-Data")

    dataset = MatchDataset(data_dir, fuzzy_match_names=False)
    dataset.load_json_files()
    df = dataset.to_dataframe(played_only=True)
    df = df[df["position"].between(1, 23)].copy()

    print("=" * 70)
    print("IMPACT OF MINIMUM SCORE THRESHOLDS")
    print("=" * 70)

    for score_type in ["tries", "penalties", "conversions", "drop_goals"]:
        score_counts = df.groupby("player_name")[score_type].sum()

        print(f"\n{score_type.upper()}")
        print("-" * 70)
        print(f"Total players in dataset: {len(score_counts):,}")
        print(f"Players with at least 1 {score_type[:-1]}: {(score_counts >= 1).sum():,}")

        thresholds = [5, 10, 15, 20, 30, 50, 100]
        print("\nPlayers remaining at different thresholds:")
        for thresh in thresholds:
            count = (score_counts >= thresh).sum()
            pct = 100 * count / len(score_counts)
            bar = "█" * int(pct / 2)
            print(f"  >= {thresh:3d}: {count:4d} players ({pct:5.1f}%) {bar}")

        top_scorers = score_counts.nlargest(5)
        print(f"\nTop 5 {score_type} scorers:")
        for player, count in top_scorers.items():
            print(f"  {player:30s}: {int(count):3d} {score_type}")

    print("\n" + "=" * 70)
    print("BEST KICKERS ANALYSIS (Penalties + Conversions)")
    print("=" * 70)

    pen_counts = df.groupby("player_name")["penalties"].sum()
    con_counts = df.groupby("player_name")["conversions"].sum()

    both = pd.DataFrame({
        "penalties": pen_counts,
        "conversions": con_counts,
    }).fillna(0)

    print(f"\nTotal players: {len(both):,}")
    print(f"Players with any penalties: {(both['penalties'] > 0).sum():,}")
    print(f"Players with any conversions: {(both['conversions'] > 0).sum():,}")
    print(
        f"Players with BOTH penalties and conversions: {((both['penalties'] > 0) & (both['conversions'] > 0)).sum():,}"
    )

    print("\nWith minimum threshold of 20 for each:")
    qualified = both[(both["penalties"] >= 20) & (both["conversions"] >= 20)]
    print(f"  Qualified kickers: {len(qualified):,}")
    print(f"  Reduction: {100 * (1 - len(qualified) / len(both)):.1f}%")

    if len(qualified) > 0:
        qualified["total_kicks"] = qualified["penalties"] + qualified["conversions"]
        qualified = qualified.sort_values("total_kicks", ascending=False)

        print("\nTop 10 qualified all-round kickers:")
        for i, (player, row) in enumerate(qualified.head(10).iterrows(), 1):
            print(
                f"  {i:2d}. {player:30s}: {int(row['penalties']):3d} pen + {int(row['conversions']):3d} con = {int(row['total_kicks']):3d} total"
            )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    analyze_threshold_impact()
