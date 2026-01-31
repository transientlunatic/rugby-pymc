"""
Validation and testing infrastructure for rugby ranking models.

Includes:
- Train/test split strategies
- Held-out match prediction
- Calibration analysis
- Performance metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ValidationSplit:
    """
    Container for train/test split.

    Attributes:
        train: Training data
        test: Test data (held-out matches)
        split_type: Type of split used
        metadata: Additional split information
    """

    train: pd.DataFrame
    test: pd.DataFrame
    split_type: str
    metadata: dict


def temporal_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    by_match: bool = True,
) -> ValidationSplit:
    """
    Split data temporally - most recent matches for testing.

    This is the most realistic validation approach as it mimics
    the actual use case: predicting future matches based on past data.

    Args:
        df: Full match dataset
        test_fraction: Fraction of data to hold out (default: 0.2)
        by_match: If True, split by matches (recommended).
                  If False, split by player-match observations.

    Returns:
        ValidationSplit with train/test data
    """
    df = df.sort_values("date").copy()

    if by_match:
        # Split by unique matches
        matches = df[["match_id", "date"]].drop_duplicates().sort_values("date")
        n_test_matches = int(len(matches) * test_fraction)
        test_match_ids = matches.tail(n_test_matches)["match_id"].values

        train = df[~df["match_id"].isin(test_match_ids)].copy()
        test = df[df["match_id"].isin(test_match_ids)].copy()

        metadata = {
            "n_train_matches": len(matches) - n_test_matches,
            "n_test_matches": n_test_matches,
            "test_date_range": (
                test["date"].min().date(),
                test["date"].max().date(),
            ),
        }
    else:
        # Split by observations (not recommended - leaks match info)
        n_test = int(len(df) * test_fraction)
        train = df.iloc[:-n_test].copy()
        test = df.iloc[-n_test:].copy()

        metadata = {
            "n_train_obs": len(train),
            "n_test_obs": len(test),
        }

    return ValidationSplit(
        train=train,
        test=test,
        split_type="temporal",
        metadata=metadata,
    )


def random_match_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> ValidationSplit:
    """
    Randomly split matches into train/test.

    Less realistic than temporal split, but useful for checking
    if the model can interpolate across the season.

    Args:
        df: Full match dataset
        test_fraction: Fraction of matches to hold out
        random_seed: Random seed for reproducibility

    Returns:
        ValidationSplit with train/test data
    """
    np.random.seed(random_seed)

    # Get unique matches
    matches = df[["match_id"]].drop_duplicates()
    n_test = int(len(matches) * test_fraction)

    # Randomly select test matches
    test_match_ids = np.random.choice(
        matches["match_id"].values,
        size=n_test,
        replace=False,
    )

    train = df[~df["match_id"].isin(test_match_ids)].copy()
    test = df[df["match_id"].isin(test_match_ids)].copy()

    metadata = {
        "n_train_matches": len(matches) - n_test,
        "n_test_matches": n_test,
        "random_seed": random_seed,
    }

    return ValidationSplit(
        train=train,
        test=test,
        split_type="random_match",
        metadata=metadata,
    )


def season_holdout_split(
    df: pd.DataFrame,
    test_seasons: list[str],
) -> ValidationSplit:
    """
    Hold out entire seasons for testing.

    Useful for validating model generalization to new seasons.

    Args:
        df: Full match dataset
        test_seasons: List of seasons to hold out (e.g., ['2024-2025'])

    Returns:
        ValidationSplit with train/test data
    """
    test = df[df["season"].isin(test_seasons)].copy()
    train = df[~df["season"].isin(test_seasons)].copy()

    metadata = {
        "test_seasons": test_seasons,
        "n_train_seasons": train["season"].nunique(),
        "n_test_matches": test["match_id"].nunique(),
    }

    return ValidationSplit(
        train=train,
        test=test,
        split_type="season_holdout",
        metadata=metadata,
    )


@dataclass
class ValidationMetrics:
    """
    Container for validation results.

    Attributes:
        log_likelihood: Per-observation log-likelihood
        rmse: Root mean squared error for counts
        mae: Mean absolute error for counts
        calibration: Calibration analysis results
        metadata: Additional information
    """

    log_likelihood: float
    rmse: dict[str, float]
    mae: dict[str, float]
    calibration: dict
    metadata: dict


def compute_validation_metrics(
    test_data: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    score_types: list[str] = None,
) -> ValidationMetrics:
    """
    Compute validation metrics on held-out data.

    Args:
        test_data: Held-out test data
        predictions: Dictionary mapping score_type -> predicted rates per observation
                     Shape: (n_test_obs,) or (n_samples, n_test_obs)
        score_types: Score types to evaluate (default: all in predictions)

    Returns:
        ValidationMetrics with performance summary
    """
    if score_types is None:
        score_types = list(predictions.keys())

    # Compute log-likelihood
    total_ll = 0.0
    n_obs = 0

    rmse = {}
    mae = {}

    for score_type in score_types:
        if score_type not in predictions:
            continue

        # Get actual counts and predicted rates
        actual = test_data[score_type].values
        pred_rates = predictions[score_type]

        # Handle both single prediction and posterior samples
        if pred_rates.ndim == 1:
            # Single prediction per observation
            lambda_pred = pred_rates
        else:
            # Posterior samples - take mean
            lambda_pred = pred_rates.mean(axis=0)

        # Log-likelihood under Poisson
        ll = stats.poisson.logpmf(actual, lambda_pred).sum()
        total_ll += ll
        n_obs += len(actual)

        # RMSE and MAE
        rmse[score_type] = np.sqrt(np.mean((actual - lambda_pred) ** 2))
        mae[score_type] = np.mean(np.abs(actual - lambda_pred))

    # Average log-likelihood
    avg_ll = total_ll / n_obs if n_obs > 0 else float("nan")

    # Calibration (for now, placeholder)
    calibration = {}

    metadata = {
        "n_observations": n_obs,
        "score_types": score_types,
    }

    return ValidationMetrics(
        log_likelihood=avg_ll,
        rmse=rmse,
        mae=mae,
        calibration=calibration,
        metadata=metadata,
    )


def calibration_analysis(
    actual_scores: np.ndarray,
    predicted_scores: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Analyze calibration of probabilistic predictions.

    Checks if predicted probabilities match observed frequencies.

    Args:
        actual_scores: Actual scores (n_matches,)
        predicted_scores: Predicted score distributions (n_samples, n_matches)
        n_bins: Number of bins for calibration plot

    Returns:
        Dictionary with calibration metrics
    """
    # Convert predictions to probabilities for each outcome
    # This is a placeholder - full implementation would compute
    # P(home_score - away_score) and compare to actual margins

    return {
        "implemented": False,
        "note": "Calibration analysis requires match-level predictions",
    }


def cross_validation(
    df: pd.DataFrame,
    n_folds: int = 5,
    split_type: Literal["temporal", "random"] = "temporal",
) -> list[ValidationSplit]:
    """
    Create cross-validation folds.

    Args:
        df: Full dataset
        n_folds: Number of folds
        split_type: How to create folds

    Returns:
        List of ValidationSplit objects (one per fold)
    """
    folds = []

    if split_type == "temporal":
        # Sort by date
        df = df.sort_values("date").copy()
        matches = df[["match_id", "date"]].drop_duplicates().sort_values("date")
        match_ids = matches["match_id"].values

        # Split matches into n_folds
        fold_size = len(match_ids) // n_folds

        for i in range(n_folds):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_folds - 1 else len(match_ids)

            test_match_ids = match_ids[test_start:test_end]
            train_match_ids = np.concatenate(
                [match_ids[:test_start], match_ids[test_end:]]
            )

            train = df[df["match_id"].isin(train_match_ids)].copy()
            test = df[df["match_id"].isin(test_match_ids)].copy()

            folds.append(
                ValidationSplit(
                    train=train,
                    test=test,
                    split_type=f"temporal_cv_fold_{i}",
                    metadata={"fold": i, "n_folds": n_folds},
                )
            )

    elif split_type == "random":
        # Random k-fold
        np.random.seed(42)
        matches = df[["match_id"]].drop_duplicates()
        match_ids = matches["match_id"].values
        np.random.shuffle(match_ids)

        fold_size = len(match_ids) // n_folds

        for i in range(n_folds):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_folds - 1 else len(match_ids)

            test_match_ids = match_ids[test_start:test_end]
            train_match_ids = np.concatenate(
                [match_ids[:test_start], match_ids[test_end:]]
            )

            train = df[df["match_id"].isin(train_match_ids)].copy()
            test = df[df["match_id"].isin(test_match_ids)].copy()

            folds.append(
                ValidationSplit(
                    train=train,
                    test=test,
                    split_type=f"random_cv_fold_{i}",
                    metadata={"fold": i, "n_folds": n_folds},
                )
            )

    return folds


def baseline_predictions(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    score_types: list[str],
) -> dict[str, np.ndarray]:
    """
    Compute simple baseline predictions for comparison.

    Uses position-specific means from training data.

    Args:
        train_data: Training data
        test_data: Test data
        score_types: Score types to predict

    Returns:
        Dictionary mapping score_type -> predicted rates
    """
    predictions = {}

    for score_type in score_types:
        # Compute position-specific means from training data
        position_means = train_data.groupby("position")[score_type].mean()

        # Predict using position means (adjusted for exposure)
        test_positions = test_data["position"].values
        test_exposure = test_data["exposure"].values

        baseline_rates = np.array(
            [position_means.get(pos, 0.0) for pos in test_positions]
        )

        # Adjust for exposure (baseline is per-80-minute rate)
        predictions[score_type] = baseline_rates * test_exposure

    return predictions
