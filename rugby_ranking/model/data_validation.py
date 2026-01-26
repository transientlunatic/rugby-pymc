"""
Data validation and cleaning utilities for rugby data.

Detects and corrects common data quality issues:
- Kicking scores attributed to wrong players (positions that don't kick)
- Duplicate name issues in same match
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Literal


# Positions that typically take kicks
KICKING_POSITIONS = {9, 10, 12, 15}  # Scrum-half, fly-half, inside center, fullback

# Positions that almost never take kicks
NON_KICKING_POSITIONS = {1, 2, 3, 4, 5, 6, 7, 8}  # Props, hooker, locks, back row


def detect_kicking_anomalies(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Detect players in non-kicking positions with conversion/penalty scores.

    Args:
        df: DataFrame from MatchDataset.to_dataframe()
        verbose: Print summary of anomalies

    Returns:
        DataFrame of anomalous records with columns:
            - player_name, position, date, team, opponent
            - conversions, penalties
            - likely_reason (e.g., "surname_conflict")
    """
    # Find non-kickers with kicks
    anomalies = df[
        (df['position'].isin(NON_KICKING_POSITIONS)) &
        ((df['conversions'] > 0) | (df['penalties'] > 0))
    ].copy()

    if len(anomalies) == 0:
        if verbose:
            print("✓ No kicking anomalies detected")
        return pd.DataFrame()

    # Check for surname conflicts (multiple players with same surname in same match)
    anomalies['surname'] = anomalies['player_name'].str.split().str[-1]

    # For each anomaly, check if there's a kicker with the same surname in the same match
    anomalies['likely_reason'] = 'unknown'

    for idx, row in anomalies.iterrows():
        # Get all players from the same match
        same_match = df[
            (df['date'] == row['date']) &
            (df['team'] == row['team']) &
            (df['opponent'] == row['opponent'])
        ]

        # Check for kickers with same surname
        same_surname = same_match[
            same_match['player_name'].str.contains(row['surname'], na=False) &
            same_match['position'].isin(KICKING_POSITIONS) &
            ((same_match['conversions'] > 0) | (same_match['penalties'] > 0))
        ]

        if len(same_surname) > 0:
            anomalies.at[idx, 'likely_reason'] = 'surname_conflict'
        else:
            anomalies.at[idx, 'likely_reason'] = 'data_error'

    if verbose:
        print(f"⚠️  Found {len(anomalies)} kicking anomalies in {anomalies['player_name'].nunique()} players")
        print("\nTop 10 affected players:")
        player_summary = anomalies.groupby('player_name').agg({
            'conversions': 'sum',
            'penalties': 'sum',
            'position': lambda x: sorted(x.unique()),
            'likely_reason': 'first'
        }).sort_values('conversions', ascending=False).head(10)

        for player, stats in player_summary.iterrows():
            print(f"  {player}: {stats['conversions']} C, {stats['penalties']} P "
                  f"(positions {stats['position']}) - {stats['likely_reason']}")

    return anomalies[['player_name', 'position', 'date', 'team', 'opponent',
                     'conversions', 'penalties', 'likely_reason', 'surname']]


def clean_kicking_data(
    df: pd.DataFrame,
    strategy: Literal['remove', 'redistribute'] = 'remove',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Clean kicking data anomalies.

    Args:
        df: DataFrame from MatchDataset.to_dataframe()
        strategy: How to handle anomalies:
            - 'remove': Set conversions/penalties to 0 for non-kickers
            - 'redistribute': Try to redistribute to actual kicker (not implemented)
        verbose: Print summary of changes

    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()

    # Detect anomalies
    anomalies = detect_kicking_anomalies(df, verbose=False)

    if len(anomalies) == 0:
        if verbose:
            print("✓ No cleaning needed")
        return df_clean

    if strategy == 'remove':
        # Set kicks to 0 for non-kickers
        mask = (
            df_clean['position'].isin(NON_KICKING_POSITIONS) &
            ((df_clean['conversions'] > 0) | (df_clean['penalties'] > 0))
        )

        removed_convs = df_clean.loc[mask, 'conversions'].sum()
        removed_pens = df_clean.loc[mask, 'penalties'].sum()

        df_clean.loc[mask, 'conversions'] = 0
        df_clean.loc[mask, 'penalties'] = 0

        if verbose:
            print(f"✓ Removed {removed_convs} conversions and {removed_pens} penalties")
            print(f"  from {mask.sum()} player-match records in non-kicking positions")
            print(f"  Affected players: {df_clean.loc[mask, 'player_name'].nunique()}")

    elif strategy == 'redistribute':
        raise NotImplementedError("Redistribution strategy not yet implemented")

    return df_clean


def validate_position_scores(df: pd.DataFrame) -> dict:
    """
    Validate that scores are reasonable for each position.

    Returns dict with validation results:
        - total_anomalies: Total anomalous records
        - kicking_anomalies: Non-kickers with kicks
        - scoring_patterns: Expected vs actual scoring by position
    """
    results = {}

    # Kicking anomalies
    kicking_anomalies = detect_kicking_anomalies(df, verbose=False)
    results['total_anomalies'] = len(kicking_anomalies)
    results['kicking_anomalies'] = len(kicking_anomalies[kicking_anomalies['likely_reason'] == 'surname_conflict'])

    # Scoring patterns by position
    position_scoring = df.groupby('position').agg({
        'tries': ['sum', 'mean'],
        'conversions': ['sum', 'mean'],
        'penalties': ['sum', 'mean'],
    }).round(3)

    results['scoring_patterns'] = position_scoring

    return results


def print_validation_report(df: pd.DataFrame):
    """Print comprehensive validation report."""
    print("=" * 70)
    print("DATA VALIDATION REPORT")
    print("=" * 70)

    print(f"\nDataset size: {len(df):,} player-match records")
    print(f"Players: {df['player_name'].nunique():,}")
    print(f"Matches: {df.groupby(['date', 'team', 'opponent']).ngroups:,}")
    print(f"Seasons: {df['season'].nunique()}")

    # Detect anomalies
    print("\n" + "-" * 70)
    print("KICKING ANOMALIES")
    print("-" * 70)
    anomalies = detect_kicking_anomalies(df, verbose=True)

    if len(anomalies) > 0:
        surname_conflicts = (anomalies['likely_reason'] == 'surname_conflict').sum()
        print(f"\n  Likely surname conflicts: {surname_conflicts}/{len(anomalies)}")
        print(f"  Recommendation: Use clean_kicking_data() to remove anomalies")

    # Position-wise scoring
    print("\n" + "-" * 70)
    print("SCORING BY POSITION")
    print("-" * 70)

    scoring = df.groupby('position').agg({
        'tries': 'sum',
        'conversions': 'sum',
        'penalties': 'sum',
    })

    print("\n  Forward positions (should have minimal kicks):")
    for pos in [1, 2, 3, 4, 5, 6, 7, 8]:
        if pos in scoring.index:
            s = scoring.loc[pos]
            flag = "⚠️" if s['conversions'] > 10 or s['penalties'] > 10 else "✓"
            print(f"    {flag} Pos {pos}: {s['tries']} T, {s['conversions']} C, {s['penalties']} P")

    print("\n  Back positions (typical kickers):")
    for pos in [9, 10, 12, 15]:
        if pos in scoring.index:
            s = scoring.loc[pos]
            print(f"    Pos {pos}: {s['tries']} T, {s['conversions']} C, {s['penalties']} P")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from rugby_ranking.model.data import MatchDataset

    DATA_DIR = Path("../../../Rugby-Data")
    if DATA_DIR.exists():
        dataset = MatchDataset(DATA_DIR, fuzzy_match_names=False)
        dataset.load_json_files()
        df = dataset.to_dataframe(played_only=True)

        print_validation_report(df)

        print("\n\nCleaning data...")
        df_clean = clean_kicking_data(df, strategy='remove', verbose=True)

        print("\n\nValidation after cleaning:")
        detect_kicking_anomalies(df_clean, verbose=True)
