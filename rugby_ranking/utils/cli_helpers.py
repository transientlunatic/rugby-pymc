"""CLI helper functions shared across commands."""

from pathlib import Path
from typing import Tuple, Optional

import arviz as az

from rugby_ranking.model.core import RugbyModel
from rugby_ranking.model.inference import ModelFitter
from rugby_ranking.model.data import MatchDataset
from rugby_ranking.utils.logging import print_section, print_success, print_error, print_info


def load_checkpoint(
    checkpoint_name: str, verbose: bool = True
) -> Tuple[RugbyModel, az.InferenceData]:
    """
    Load a trained model checkpoint.

    Args:
        checkpoint_name: Name of checkpoint (e.g., "joint_model_v2")
        verbose: Print loading status

    Returns:
        (model, trace) tuple

    Raises:
        ValueError: If checkpoint cannot be loaded

    Example:
        >>> model, trace = load_checkpoint("joint_model_v2")
        >>> rankings = model.get_player_rankings(trace, score_type='tries')
    """
    if verbose:
        print_info(f"Loading checkpoint: {checkpoint_name}")

    model = RugbyModel()

    try:
        fitter = ModelFitter.load(checkpoint_name, model)
        trace = fitter.trace

        if verbose:
            print_success("Loaded successfully")
            print_info(f"  Players: {len(model._player_ids):,}")
            print_info(f"  Team-seasons: {len(model._team_season_ids)}")

        return model, trace

    except Exception as e:
        if verbose:
            print_error(f"Failed to load checkpoint: {e}")
        raise


def setup_data(
    data_dir: Path,
    pattern: str = "*.json",
    fuzzy_match: bool = False,
    verbose: bool = True,
) -> Tuple[MatchDataset, "pd.DataFrame"]:
    """
    Load and prepare match dataset.

    Args:
        data_dir: Path to Rugby-Data repository
        pattern: File pattern to load
        fuzzy_match: Whether to use fuzzy player name matching
        verbose: Print status messages

    Returns:
        (dataset, dataframe) tuple
    """
    if verbose:
        print_section("LOADING DATA")

    dataset = MatchDataset(data_dir, fuzzy_match_names=fuzzy_match)
    dataset.load_json_files(pattern=pattern)

    df = dataset.to_dataframe(played_only=True)
    df = df[df['position'].between(1, 23)].copy()

    if verbose:
        print_success(f"Loaded: {len(df):,} observations")
        print_info(f"  Players: {df['player_name'].nunique():,}")
        print_info(f"  Teams: {df['team'].nunique()}")
        print_info(f"  Seasons: {df['season'].nunique()}")
        print_info(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    return dataset, df


def format_large_number(n: int, precision: int = 1) -> str:
    """Format large numbers with K, M, B suffixes."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.{precision}f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.{precision}f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.{precision}f}K"
    return str(n)
