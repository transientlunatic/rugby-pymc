"""Shared utilities for Jupyter notebooks in rugby-ranking project."""

from pathlib import Path
import sys
from typing import Tuple

import pandas as pd
import numpy as np


def setup_notebook_environment(data_dir_relative: str = "../Rugby-Data"):
    """
    Configure notebook environment with paths and imports.
    
    Call this as the first cell in your notebook:
    
        from rugby_ranking.notebook_utils import setup_notebook_environment
        dataset, df, model_path = setup_notebook_environment()
    
    Args:
        data_dir_relative: Relative path to Rugby-Data repo from notebook
        
    Returns:
        (dataset, dataframe, model_checkpoint_dir) tuple
    """
    # Silence warnings
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Configure plotting
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Get paths
    notebook_dir = Path(".").resolve()
    data_dir = (notebook_dir.parent / data_dir_relative).resolve()
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Rugby-Data not found at {data_dir}\n"
            f"Expected: {data_dir}\n"
            f"Adjust the 'data_dir_relative' parameter"
        )
    
    # Load data
    from rugby_ranking.model.data import MatchDataset
    from rugby_ranking.model.data_validation import detect_kicking_anomalies, clean_kicking_data
    
    print(f"Loading from: {data_dir}")
    dataset = MatchDataset(data_dir, fuzzy_match_names=False)
    dataset.load_json_files()
    df = dataset.to_dataframe(played_only=True)
    df = df[df['position'].between(1, 23)].copy()
    
    print(f"✓ Loaded {len(df):,} player-match observations")
    print(f"  Players: {df['player_name'].nunique():,}")
    print(f"  Teams: {df['team'].nunique()}")
    print(f"  Matches: {df.groupby(['date', 'team']).ngroups:,}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Check for data quality issues and clean
    print("\nData Quality Checks:")

    df = clean_kicking_data(df, strategy='remove', verbose=True)
    
    # Model checkpoint directory
    model_dir = notebook_dir.parent / "rugby_ranking" / "checkpoints"
    
    return dataset, df, model_dir


def configure_plot_style(figsize: Tuple[int, int] = (12, 6), font_scale: float = 1.0):
    """Configure matplotlib and seaborn for better-looking plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.rcParams['figure.figsize'] = figsize
    sns.set_context("notebook", font_scale=font_scale)
    plt.tight_layout()


def print_summary(df: pd.DataFrame, title: str = "Dataset Summary"):
    """Print a formatted summary of the dataset."""
    from rugby_ranking.utils.logging import print_section
    
    print_section(title)
    print(f"Records: {len(df):,}")
    print(f"Players: {df['player_name'].nunique():,}")
    print(f"Teams: {df['team'].nunique():,}")
    print(f"Seasons: {df['season'].nunique()}")
    print(f"Matches: {df.groupby(['date', 'team']).ngroups:,}")
    
    if 'date' in df.columns:
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")


def load_model_and_trace(checkpoint_name: str = "latest", verbose: bool = True):
    """Load a trained model checkpoint."""
    from rugby_ranking.utils.cli_helpers import load_checkpoint
    
    return load_checkpoint(checkpoint_name, verbose=verbose)


def get_top_players(
    trace,
    model,
    score_type: str = "tries",
    top: int = 20,
    sort_by: str = "median",
) -> pd.DataFrame:
    """Get top players for a given score type."""
    rankings = model.get_player_rankings(trace, score_type=score_type)
    
    if sort_by == "median":
        rankings = rankings.sort_values("median_effect", ascending=False)
    elif sort_by == "mean":
        rankings = rankings.sort_values("mean_effect", ascending=False)
    
    return rankings.head(top)


def compare_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """Compare stats across seasons."""
    by_season = df.groupby('season').agg({
        'player_name': 'nunique',
        'team': 'nunique',
        'tries': 'sum',
        'penalties': 'sum',
        'conversions': 'sum',
        'drop_goals': 'sum',
    }).rename(columns={
        'player_name': 'n_players',
        'team': 'n_teams',
    })
    
    return by_season


def create_position_ranking_matrix(df: pd.DataFrame, score_type: str = "tries") -> pd.DataFrame:
    """Create a matrix of top scorers by position."""
    position_ranks = []
    
    for pos in sorted(df['position'].unique()):
        pos_data = df[df['position'] == pos]
        top = pos_data.groupby('player_name')[score_type].sum().nlargest(3)
        
        for rank, (player, count) in enumerate(top.items(), 1):
            position_ranks.append({
                'position': pos,
                'rank': rank,
                'player': player,
                score_type: count,
            })
    
    return pd.DataFrame(position_ranks)
