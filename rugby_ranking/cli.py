"""
Command-line interface for rugby ranking model.

Supports weekly update workflow:
    rugby-ranking update --data-dir /path/to/Rugby-Data
    rugby-ranking rankings --type players --top 20
    rugby-ranking predict --home "Leinster" --away "Munster"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

from rugby_ranking.model.core import RugbyModel, ModelConfig
from rugby_ranking.model.inference import ModelFitter


def load_checkpoint(checkpoint_name: str, verbose: bool = True):
    """
    Load a trained model checkpoint.

    Args:
        checkpoint_name: Name of checkpoint (e.g., "joint_model_v2")
        verbose: Print loading status

    Returns:
        (model, trace) tuple

    Example:
        >>> model, trace = load_checkpoint("joint_model_v2")
        >>> rankings = model.get_player_rankings(trace, score_type='tries')
    """
    if verbose:
        print(f"Loading checkpoint: {checkpoint_name}")

    # Create a model instance to load into
    # The actual config and indices will be loaded from the checkpoint
    model = RugbyModel()

    try:
        fitter = ModelFitter.load(checkpoint_name, model)
        trace = fitter.trace

        if verbose:
            print(f"✓ Loaded successfully")
            print(f"  Players: {len(model._player_ids):,}")
            print(f"  Team-seasons: {len(model._team_season_ids)}")

        return model, trace

    except Exception as e:
        if verbose:
            print(f"✗ Failed to load checkpoint: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Rugby player and team ranking system"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Update command
    update_parser = subparsers.add_parser(
        "update",
        help="Update model with latest data"
    )
    update_parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to Rugby-Data repository"
    )
    update_parser.add_argument(
        "--method",
        choices=["vi", "mcmc"],
        default="vi",
        help="Inference method (default: vi for speed)"
    )
    update_parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint name to save/load"
    )

    # Rankings command
    rankings_parser = subparsers.add_parser(
        "rankings",
        help="Display current rankings"
    )
    rankings_parser.add_argument(
        "--type",
        choices=["players", "teams"],
        default="teams",
        help="What to rank"
    )
    rankings_parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Filter to season (teams only)"
    )
    rankings_parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of entries to show"
    )
    rankings_parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint to load"
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict match outcome"
    )
    predict_parser.add_argument(
        "--home",
        type=str,
        required=True,
        help="Home team name"
    )
    predict_parser.add_argument(
        "--away",
        type=str,
        required=True,
        help="Away team name"
    )
    predict_parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Season (defaults to current)"
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint to load"
    )

    args = parser.parse_args()

    if args.command == "update":
        run_update(args)
    elif args.command == "rankings":
        run_rankings(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        parser.print_help()


def run_update(args):
    """Run model update with latest data."""
    from rugby_ranking.model.data import MatchDataset
    from rugby_ranking.model.core import RugbyModel, ModelConfig
    from rugby_ranking.model.inference import ModelFitter, InferenceConfig

    print(f"Loading data from {args.data_dir}...")
    dataset = MatchDataset(args.data_dir)
    dataset.load_json_files()

    print("Preparing model data...")
    df = dataset.to_dataframe(played_only=True)
    print(f"  {len(df)} player-match observations")
    print(f"  {df['season'].nunique()} seasons")

    print("Building model...")
    config = ModelConfig()
    model = RugbyModel(config)
    model.build(df, score_type="tries")

    print(f"Fitting model using {args.method.upper()}...")
    fitter = ModelFitter(model, InferenceConfig())

    if args.method == "vi":
        trace = fitter.fit_vi()
    else:
        trace = fitter.fit_mcmc()

    # Diagnostics
    diag = fitter.diagnostics()
    print(f"  R-hat max: {diag['r_hat_max']:.3f}")
    print(f"  ESS min: {diag['ess_bulk_min']:.0f}")

    # Save checkpoint
    fitter.save(args.checkpoint)
    print(f"Saved checkpoint: {args.checkpoint}")


def run_rankings(args):
    """Display rankings from saved checkpoint."""
    from rugby_ranking.model.core import RugbyModel
    from rugby_ranking.model.inference import ModelFitter

    model = RugbyModel()
    fitter = ModelFitter.load(args.checkpoint, model)

    if args.type == "players":
        rankings = model.get_player_rankings(top_n=args.top)
        print(f"\nTop {args.top} Players:")
        print("=" * 60)
        for i, row in rankings.iterrows():
            print(
                f"{i+1:2d}. {row['player']:<30} "
                f"Effect: {row['effect_mean']:+.3f} "
                f"(±{row['effect_std']:.3f})"
            )
    else:
        rankings = model.get_team_rankings(season=args.season, top_n=args.top)
        print(f"\nTop {args.top} Teams (Season: {args.season or 'all'}):")
        print("=" * 60)
        for i, row in rankings.iterrows():
            print(
                f"{i+1:2d}. {row['team']:<20} ({row['season']}) "
                f"Effect: {row['effect_mean']:+.3f} "
                f"(±{row['effect_std']:.3f})"
            )


def run_predict(args):
    """Predict match outcome."""
    from rugby_ranking.model.core import RugbyModel
    from rugby_ranking.model.inference import ModelFitter
    from rugby_ranking.model.predictions import MatchPredictor

    model = RugbyModel()
    fitter = ModelFitter.load(args.checkpoint, model)

    # Determine season
    if args.season:
        season = args.season
    else:
        # Use most recent season in data
        seasons = sorted(model._season_ids.keys())
        season = seasons[-1] if seasons else "2025-2026"

    predictor = MatchPredictor(model)

    try:
        prediction = predictor.predict_teams_only(
            home_team=args.home,
            away_team=args.away,
            season=season,
        )
        print(f"\n{prediction.summary()}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Available teams:")
        for team, season in sorted(model._team_season_ids.keys()):
            print(f"  - {team} ({season})")


if __name__ == "__main__":
    main()
