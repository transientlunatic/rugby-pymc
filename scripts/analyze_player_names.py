#!/usr/bin/env python3
"""
Analyze player name matching and merging.

Usage:
    # Export merge report
    python analyze_player_names.py --data-dir /path/to/Rugby-Data --export merged_names.xlsx

    # Find potential duplicates
    python analyze_player_names.py --data-dir /path/to/Rugby-Data --find-duplicates

    # Interactive review
    python analyze_player_names.py --data-dir /path/to/Rugby-Data --interactive-review

    # Search for name variations
    python analyze_player_names.py --data-dir /path/to/Rugby-Data --search "Johnny Sexton"
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rugby_ranking.model import (
    MatchDataset,
    analyze_merged_names,
    find_potential_duplicates,
    review_merges,
    get_name_variations,
    export_merge_report,
    generate_correction_dict,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze player name matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to Rugby-Data directory",
    )

    parser.add_argument(
        "--export",
        type=str,
        help="Export merge report to file (.csv or .xlsx)",
    )

    parser.add_argument(
        "--find-duplicates",
        action="store_true",
        help="Find potential duplicates that weren't merged",
    )

    parser.add_argument(
        "--interactive-review",
        action="store_true",
        help="Interactively review merged names",
    )

    parser.add_argument(
        "--search",
        type=str,
        help="Search for variations of a player name",
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for fuzzy matching (default: 0.85)",
    )

    parser.add_argument(
        "--min-appearances",
        type=int,
        default=10,
        help="Minimum appearances for duplicate detection (default: 10)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("PLAYER NAME ANALYSIS")
    print("=" * 70)

    # Load data with fuzzy matching
    print(f"\nLoading data (similarity threshold: {args.similarity_threshold})...")
    dataset = MatchDataset(
        args.data_dir,
        fuzzy_match_names=True,
        name_similarity_threshold=args.similarity_threshold,
    )
    dataset.load_json_files(pattern="*.json")

    print(f"Loaded {len(dataset.observations):,} observations")
    print(f"Unique players: {len(set(obs.player_name for obs in dataset.observations)):,}")

    # Export report
    if args.export:
        print(f"\nExporting merge report to {args.export}...")
        export_merge_report(dataset, args.export)
        print("✓ Export complete")

    # Analyze merged names
    if not args.search and not args.find_duplicates:
        print("\nAnalyzing merged names...")
        merged_df = analyze_merged_names(dataset)

        if len(merged_df) == 0:
            print("  No names were merged")
        else:
            print(f"  {len(merged_df)} name variations merged")
            print(f"  Unique canonical names: {merged_df['canonical'].nunique()}")

            print("\n  Lowest similarity merges:")
            print(merged_df.head(10).to_string(index=False))

    # Find potential duplicates
    if args.find_duplicates:
        print("\nSearching for potential duplicates...")
        dupes_df = find_potential_duplicates(
            dataset,
            min_similarity=0.75,
            max_similarity=args.similarity_threshold,
            min_appearances=args.min_appearances,
        )

        if len(dupes_df) == 0:
            print("  No potential duplicates found")
        else:
            print(f"  Found {len(dupes_df)} potential duplicate pairs")
            print("\n  Top candidates (with team overlap):")
            print(
                dupes_df[dupes_df["team_overlap"]]
                .head(20)
                .to_string(index=False)
            )

            print("\n  Other candidates (no team overlap):")
            print(
                dupes_df[~dupes_df["team_overlap"]]
                .head(10)
                .to_string(index=False)
            )

    # Interactive review
    if args.interactive_review:
        print("\nStarting interactive review...")
        merged_df = analyze_merged_names(dataset)
        reviewed_df = review_merges(
            merged_df,
            min_similarity=args.similarity_threshold,
            interactive=True,
        )

        # Generate correction dictionary
        if "manual_review" in reviewed_df.columns:
            corrections = generate_correction_dict(reviewed_df)
            if corrections:
                print("\n" + "=" * 70)
                print("MANUAL CORRECTIONS DICTIONARY")
                print("=" * 70)
                print("\nAdd these to PlayerNameMatcher in data.py:\n")
                print("manual_corrections = {")
                for var, canon in corrections.items():
                    print(f'    "{var}": "{canon}",')
                print("}")

    # Search for name variations
    if args.search:
        print(f"\nSearching for variations of '{args.search}'...")
        variations = get_name_variations(dataset, args.search)

        if len(variations) == 0:
            print("  No similar names found")
        else:
            print(f"  Found {len(variations)} similar names:")
            print(variations.to_string(index=False))

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
