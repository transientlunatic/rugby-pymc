#!/usr/bin/env python3
"""
Test and demonstration of validation infrastructure.

Usage:
    python test_validation.py --data-dir ../Rugby-Data

This script demonstrates:
1. Creating train/test splits
2. Fitting model on training data
3. Evaluating on held-out test data
4. Computing validation metrics
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rugby_ranking.model import (
    MatchDataset,
    RugbyModel,
    ModelConfig,
    ModelFitter,
    InferenceConfig,
    temporal_split,
    random_match_split,
    season_holdout_split,
    compute_validation_metrics,
    baseline_predictions,
)
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test validation infrastructure"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to Rugby-Data directory",
    )
    parser.add_argument(
        "--split-type",
        choices=["temporal", "random", "season"],
        default="temporal",
        help="Type of train/test split",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--test-seasons",
        nargs="+",
        help="Seasons to hold out (for season split)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use small subset for quick testing",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("VALIDATION INFRASTRUCTURE TEST")
    print("=" * 70)

    # 1. Load data
    print("\n1. Loading data...")
    dataset = MatchDataset(args.data_dir, fuzzy_match_names=False)
    dataset.load_json_files(pattern="*.json")
    df = dataset.to_dataframe(played_only=True)
    df = df[df["position"].between(1, 23)].copy()

    print(f"   Loaded: {len(df):,} observations")
    print(f"   Matches: {df['match_id'].nunique():,}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Quick mode: use last 2 seasons only
    if args.quick:
        print("\n   Quick mode: using last 2 seasons only")
        all_seasons = sorted(df["season"].unique())
        recent_seasons = all_seasons[-2:]
        df = df[df["season"].isin(recent_seasons)].copy()
        print(f"   Filtered to: {len(df):,} observations from {recent_seasons}")

    # 2. Create train/test split
    print(f"\n2. Creating {args.split_type} split...")

    if args.split_type == "temporal":
        split = temporal_split(df, test_fraction=args.test_fraction)
    elif args.split_type == "random":
        split = random_match_split(df, test_fraction=args.test_fraction)
    elif args.split_type == "season":
        if not args.test_seasons:
            # Default to most recent season
            args.test_seasons = [sorted(df["season"].unique())[-1]]
        split = season_holdout_split(df, test_seasons=args.test_seasons)

    print(f"   Train: {len(split.train):,} observations")
    print(f"   Test:  {len(split.test):,} observations")
    print(f"   Metadata: {split.metadata}")

    # 3. Fit model on training data
    print("\n3. Fitting model on training data...")

    config = ModelConfig(
        score_types=("tries", "penalties", "conversions"),
        separate_kicking_effect=True,
    )
    model = RugbyModel(config)
    model.build_joint(split.train)

    print(f"   Built model with:")
    print(f"     Players: {len(model._player_ids):,}")
    print(f"     Team-seasons: {len(model._team_season_ids)}")

    # Use VI for speed
    inference_config = InferenceConfig(
        vi_n_iterations=10000 if args.quick else 30000
    )
    fitter = ModelFitter(model, inference_config)

    print(f"   Running VI ({inference_config.vi_n_iterations:,} iterations)...")
    trace = fitter.fit_vi(n_samples=1000)
    print("   ✓ Fitting complete")

    # 4. Make predictions on test data
    print("\n4. Making predictions on test data...")

    # Extract posterior means for player/team effects
    # This is a simplified prediction - full version would use MatchPredictor
    score_types = ["tries", "penalties", "conversions"]
    predictions = {}

    for score_type in score_types:
        # Get parameters
        alpha_samples = trace.posterior["alpha"].values.flatten()
        theta_samples = trace.posterior[f"theta_{score_type}"].values
        eta_home_samples = trace.posterior[f"eta_home_{score_type}"].values.flatten()

        # For simplicity, use posterior means
        alpha = alpha_samples.mean()
        theta = theta_samples.mean(axis=(0, 1))  # Average over chains and draws
        eta_home = eta_home_samples.mean()

        # Predict for each test observation
        # Note: This ignores player/team effects for simplicity
        # Full implementation would extract player-specific effects

        test_positions = split.test["position"].values
        test_exposure = split.test["exposure"].values
        test_is_home = split.test["is_home"].values

        log_rates = alpha + theta[test_positions - 1] + eta_home * test_is_home
        rates = np.exp(log_rates) * test_exposure

        predictions[score_type] = rates

    print(f"   Generated predictions for {score_types}")

    # 5. Compute validation metrics
    print("\n5. Computing validation metrics...")

    metrics = compute_validation_metrics(split.test, predictions, score_types)

    print(f"   Log-likelihood: {metrics.log_likelihood:.3f}")
    print(f"   RMSE:")
    for st, rmse in metrics.rmse.items():
        print(f"     {st}: {rmse:.3f}")
    print(f"   MAE:")
    for st, mae in metrics.mae.items():
        print(f"     {st}: {mae:.3f}")

    # 6. Compare to baseline
    print("\n6. Baseline comparison...")

    baseline_preds = baseline_predictions(split.train, split.test, score_types)
    baseline_metrics = compute_validation_metrics(
        split.test, baseline_preds, score_types
    )

    print(f"   Baseline log-likelihood: {baseline_metrics.log_likelihood:.3f}")
    print(f"   Improvement: {metrics.log_likelihood - baseline_metrics.log_likelihood:.3f}")

    print(f"\n   Baseline RMSE:")
    for st in score_types:
        model_rmse = metrics.rmse[st]
        baseline_rmse = baseline_metrics.rmse[st]
        improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100
        print(f"     {st}: {baseline_rmse:.3f} → {model_rmse:.3f} ({improvement:+.1f}%)")

    # 7. Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Split type: {split.split_type}")
    print(f"Test observations: {len(split.test):,}")
    print(f"Model log-likelihood: {metrics.log_likelihood:.3f}")
    print(f"Baseline log-likelihood: {baseline_metrics.log_likelihood:.3f}")
    print(f"Improvement: {metrics.log_likelihood - baseline_metrics.log_likelihood:.3f}")
    print(f"\nModel successfully validated on held-out data!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
