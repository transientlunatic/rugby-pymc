#!/usr/bin/env python
"""
# Data Quality & Validation

This notebook demonstrates data quality checks, anomaly detection, and data cleaning workflows.

**Topics**:
1. Data completeness and missing values
2. Kicking score anomalies
3. Name matching and player identification
4. Position consistency
5. Temporal continuity checks
6. Data cleaning report
"""

from rugby_ranking.notebook_utils import setup_notebook_environment
from rugby_ranking.model.data import normalize_player_name
from rugby_ranking.model.inference import MODEL_CONFIG
import pandas as pd
import numpy as np
from collections import Counter

# Setup
dataset, df, model_dir = setup_notebook_environment()

# %%
# ## 1. Data Overview
# 
# Basic statistics about the dataset.

print(f"Dataset Shape: {df.shape}")
print(f"\nDate Range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Unique Teams: {df['team'].nunique()}")
print(f"Unique Players: {df['player_name'].nunique()}")
print(f"Unique Competitions: {df.get('competition', pd.Series()).nunique()}")

print(f"\nScore Distribution:")
print(df['score'].value_counts().sort_index().head(10))

# %%
# ## 2. Missing Values
# 
# Check for missing data in key columns.

print("Missing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

for col in df.columns:
    if missing[col] > 0:
        print(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
    
if missing.sum() == 0:
    print("  ✓ No missing values")

# %%
# ## 3. Kicking Score Anomalies
# 
# Identify unusual kicking patterns (e.g., kicker with 100+ points in a season).

# Filter to kickers (forwards don't kick)
kicking_positions = MODEL_CONFIG['kicking_positions']
kickers = df[df['position'].isin(kicking_positions)].copy()

# Seasonal aggregation
seasonal_kicking = kickers.groupby(['season', 'player_name', 'team']).agg({
    'score': 'sum',
    'date': 'count'  # number of matches
}).rename(columns={'date': 'matches'})

# Find anomalies
anomalies = seasonal_kicking[seasonal_kicking['score'] > 100]

print(f"High kicking seasons (>100 points):")
if len(anomalies) > 0:
    print(anomalies.sort_values('score', ascending=False).head(10))
else:
    print("  None found")

# Average points per match
seasonal_kicking['points_per_match'] = seasonal_kicking['score'] / seasonal_kicking['matches']
high_avg = seasonal_kicking[seasonal_kicking['points_per_match'] > 5]

print(f"\nHigh average kicking (>5 pts/match):")
if len(high_avg) > 0:
    print(high_avg.sort_values('points_per_match', ascending=False).head(10))
else:
    print("  None found")

# %%
# ## 4. Name Normalization Issues
# 
# Check for name variations that might represent the same player.

# Find variations of player names (same team, similar names)
def find_name_variations(df, team, max_variations=5):
    team_players = df[df['team'] == team]['player_name'].unique()
    
    # Group by normalized name
    normalized = {}
    for name in team_players:
        norm = normalize_player_name(name)
        if norm not in normalized:
            normalized[norm] = []
        normalized[norm].append(name)
    
    # Find groups with multiple names
    variations = {k: v for k, v in normalized.items() if len(v) > 1}
    return variations

# Sample teams
teams_sample = df['team'].unique()[:5]
print(f"Checking {len(teams_sample)} teams for name variations:")

all_variations = {}
for team in teams_sample:
    vars = find_name_variations(df, team)
    if vars:
        all_variations[team] = vars

if all_variations:
    for team, vars in all_variations.items():
        print(f"\n{team}:")
        for norm_name, names in list(vars.items())[:3]:
            print(f"  {norm_name}: {names}")
else:
    print("\n✓ No name variations detected")

# %%
# ## 5. Position Consistency
# 
# Check whether players maintain consistent positions over time.

# For each player, check position changes
player_positions = df.groupby('player_name')['position'].nunique()

position_changers = player_positions[player_positions > 1]
print(f"Players with position changes: {len(position_changers)} / {len(player_positions)} ({len(position_changers)/len(player_positions)*100:.1f}%)")

if len(position_changers) > 0:
    print(f"\nTop position-changers:")
    for player in position_changers.nlargest(5).index:
        positions = df[df['player_name'] == player]['position'].unique()
        counts = df[df['player_name'] == player].groupby('position').size()
        print(f"  {player}: {dict(counts)}")

# %%
# ## 6. Temporal Continuity
# 
# Check for gaps in match records and unusual patterns.

# Match frequency by team
matches_by_team = df.groupby('team').groupby(['date', 'team']).size().reset_index(name='players').groupby('team').size()

print(f"Matches per team (min, max, mean):")
print(f"  Min: {matches_by_team.min()}")
print(f"  Max: {matches_by_team.max()}")
print(f"  Mean: {matches_by_team.mean():.1f}")

# Find teams with suspiciously few matches
low_match_teams = matches_by_team[matches_by_team < matches_by_team.quantile(0.25)]
if len(low_match_teams) > 0:
    print(f"\n⚠️  Teams with low match counts:")
    for team, count in low_match_teams.items():
        print(f"  {team}: {count} matches")
else:
    print(f"\n✓ Match distribution looks reasonable")

# %%
# ## 7. Data Cleaning Summary
# 
# Generate a report of data quality issues and recommended actions.

print("=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)

issues = []
recommendations = []

# Check 1: Missing values
missing_total = df.isnull().sum().sum()
if missing_total > 0:
    issues.append(f"{missing_total} missing values across all columns")
    recommendations.append("Investigate and impute missing values")
else:
    print("✓ No missing values\n")

# Check 2: Name variations
if len(all_variations) > 0:
    issues.append(f"Name variations found in {len(all_variations)} teams")
    recommendations.append("Standardize player names or merge duplicate records")
else:
    print("✓ Name variations minimal\n")

# Check 3: Position changes
if len(position_changers) > 0:
    issues.append(f"{len(position_changers)} players changed positions")
    recommendations.append("Review position changes; consider position as time-varying")
else:
    print("✓ Position consistency good\n")

# Print summary
if issues:
    print("Issues Identified:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\nRecommended Actions:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
else:
    print("✓ Dataset appears clean and consistent")

print("\n" + "=" * 60)
