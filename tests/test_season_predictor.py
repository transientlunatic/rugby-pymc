#!/usr/bin/env python3
"""
Test the season prediction functionality.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from rugby_ranking.model.season_predictor import SeasonPredictor
from rugby_ranking.model.predictions import MatchPredictor, MatchPrediction, ScorePrediction
from rugby_ranking.model.league_table import BonusPointRules


class MockMatchPredictor:
    """Mock predictor for testing without loading a full model."""

    def predict_teams_only(self, home_team: str, away_team: str, season: str, n_samples: int = 1000):
        """Generate mock predictions with realistic variation."""
        # Simple home advantage model
        np.random.seed(hash(home_team + away_team) % 2**32)

        # Home team slightly favored
        home_mean = 24
        away_mean = 20

        # Generate samples
        home_samples = np.random.normal(home_mean, 8, n_samples).clip(0, 60).round().astype(int)
        away_samples = np.random.normal(away_mean, 8, n_samples).clip(0, 60).round().astype(int)

        home_wins = (home_samples > away_samples).mean()
        away_wins = (away_samples > home_samples).mean()
        draws = (home_samples == away_samples).mean()

        home_pred = ScorePrediction(
            team=home_team,
            mean=float(home_samples.mean()),
            std=float(home_samples.std()),
            median=float(np.median(home_samples)),
            ci_lower=float(np.percentile(home_samples, 5)),
            ci_upper=float(np.percentile(home_samples, 95)),
            samples=home_samples,
        )

        away_pred = ScorePrediction(
            team=away_team,
            mean=float(away_samples.mean()),
            std=float(away_samples.std()),
            median=float(np.median(away_samples)),
            ci_lower=float(np.percentile(away_samples, 5)),
            ci_upper=float(np.percentile(away_samples, 95)),
            samples=away_samples,
        )

        return MatchPrediction(
            home=home_pred,
            away=away_pred,
            home_win_prob=float(home_wins),
            away_win_prob=float(away_wins),
            draw_prob=float(draws),
            predicted_margin=float((home_samples - away_samples).mean()),
            margin_std=float((home_samples - away_samples).std()),
        )


def test_season_prediction():
    """Test end-to-end season prediction."""
    print("=" * 70)
    print("TEST: Season Prediction")
    print("=" * 70)

    # Create sample played matches (4 teams, 2 rounds completed)
    played_matches = pd.DataFrame([
        # Round 1
        {'team': 'Leinster', 'opponent': 'Munster', 'score': 28, 'opponent_score': 14, 'tries': 4, 'opponent_tries': 2, 'is_home': True},
        {'team': 'Munster', 'opponent': 'Leinster', 'score': 14, 'opponent_score': 28, 'tries': 2, 'opponent_tries': 4, 'is_home': False},
        {'team': 'Ulster', 'opponent': 'Connacht', 'score': 21, 'opponent_score': 20, 'tries': 3, 'opponent_tries': 3, 'is_home': True},
        {'team': 'Connacht', 'opponent': 'Ulster', 'score': 20, 'opponent_score': 21, 'tries': 3, 'opponent_tries': 3, 'is_home': False},

        # Round 2
        {'team': 'Leinster', 'opponent': 'Connacht', 'score': 35, 'opponent_score': 7, 'tries': 5, 'opponent_tries': 1, 'is_home': True},
        {'team': 'Connacht', 'opponent': 'Leinster', 'score': 7, 'opponent_score': 35, 'tries': 1, 'opponent_tries': 5, 'is_home': False},
        {'team': 'Munster', 'opponent': 'Ulster', 'score': 24, 'opponent_score': 24, 'tries': 4, 'opponent_tries': 3, 'is_home': True},
        {'team': 'Ulster', 'opponent': 'Munster', 'score': 24, 'opponent_score': 24, 'tries': 3, 'opponent_tries': 4, 'is_home': False},
    ])

    # Create remaining fixtures (2 more rounds)
    remaining_fixtures = pd.DataFrame([
        # Round 3
        {'home_team': 'Munster', 'away_team': 'Connacht'},
        {'home_team': 'Leinster', 'away_team': 'Ulster'},

        # Round 4
        {'home_team': 'Connacht', 'away_team': 'Munster'},
        {'home_team': 'Ulster', 'away_team': 'Leinster'},
    ])

    # Create mock predictor and season predictor
    mock_predictor = MockMatchPredictor()
    season_predictor = SeasonPredictor(
        match_predictor=mock_predictor,
        competition=BonusPointRules.URC,
        playoff_spots=2,  # Only top 2 for this small test
    )

    print("\n1. Predicting season with 4 teams, 2 rounds played, 2 rounds remaining...")

    # Run prediction
    season_pred = season_predictor.predict_season(
        played_matches=played_matches,
        remaining_fixtures=remaining_fixtures,
        season="2024-2025",
        n_simulations=100,  # Small number for fast test
    )

    # Display results
    print("\n2. Current Standings:")
    print("-" * 70)
    print(season_pred.current_standings[['position', 'team', 'played', 'won', 'drawn', 'lost', 'total_points']])

    print("\n3. Predicted Final Standings:")
    print("-" * 70)
    print(season_pred.predicted_standings[['predicted_position', 'team', 'expected_points', 'expected_diff']])

    print("\n4. Playoff Probabilities (Top 2):")
    print("-" * 70)
    print(season_pred.playoff_probabilities)

    print("\n5. Position Probabilities:")
    print("-" * 70)
    print(season_pred.position_probabilities[['most_likely_position', 'P(pos 1)', 'P(pos 2)', 'P(pos 3)', 'P(pos 4)']])

    # Verification
    print("\n\n6. Verification:")
    print("-" * 70)

    # Check that all 4 teams are present
    assert len(season_pred.current_standings) == 4, "Should have 4 teams"
    assert len(season_pred.predicted_standings) == 4, "Should predict 4 teams"

    # Check that Leinster is currently leading (2 wins with bonus)
    current_leader = season_pred.current_standings.iloc[0]['team']
    print(f"  Current leader: {current_leader} (expected: Leinster)")
    assert current_leader == 'Leinster', "Leinster should be leading after 2 rounds"

    # Check that probabilities sum to 1
    prob_cols = [col for col in season_pred.position_probabilities.columns if col.startswith('P(pos')]
    for team in season_pred.position_probabilities.index:
        prob_sum = season_pred.position_probabilities.loc[team, prob_cols].sum()
        assert abs(prob_sum - 1.0) < 0.01, f"Position probabilities should sum to 1, got {prob_sum}"

    print(f"  ✓ Position probabilities sum to 1.0 for all teams")

    # Check that playoff probabilities are between 0 and 1
    for _, row in season_pred.playoff_probabilities.iterrows():
        prob = row['playoff_probability']
        assert 0 <= prob <= 1, f"Playoff probability should be in [0, 1], got {prob}"

    print(f"  ✓ Playoff probabilities in valid range [0, 1]")

    # Check that remaining fixtures were predicted
    assert len(season_pred.remaining_fixtures) == 4, "Should have 4 remaining fixtures"
    print(f"  ✓ All 4 remaining fixtures predicted")

    print("\n✅ Season prediction test passed!")
    return True


def test_formatted_output():
    """Test formatted output generation."""
    print("\n" + "=" * 70)
    print("TEST: Formatted Output")
    print("=" * 70)

    # Create minimal test case
    played_matches = pd.DataFrame([
        {'team': 'Team A', 'opponent': 'Team B', 'score': 30, 'opponent_score': 20, 'tries': 4, 'opponent_tries': 3, 'is_home': True},
        {'team': 'Team B', 'opponent': 'Team A', 'score': 20, 'opponent_score': 30, 'tries': 3, 'opponent_tries': 4, 'is_home': False},
    ])

    remaining_fixtures = pd.DataFrame([
        {'home_team': 'Team A', 'away_team': 'Team B'},
    ])

    mock_predictor = MockMatchPredictor()
    season_predictor = SeasonPredictor(
        match_predictor=mock_predictor,
        competition=BonusPointRules.URC,
        playoff_spots=1,
    )

    season_pred = season_predictor.predict_season(
        played_matches=played_matches,
        remaining_fixtures=remaining_fixtures,
        season="2024-2025",
        n_simulations=50,
    )

    # Generate formatted output
    formatted = season_predictor.format_predictions(season_pred)

    print("\nFormatted Output:")
    print(formatted)

    # Verify formatting
    assert "SEASON PREDICTION" in formatted, "Should contain title"
    assert "CURRENT STANDINGS" in formatted, "Should contain current standings section"
    assert "PREDICTED FINAL STANDINGS" in formatted, "Should contain predicted standings"
    assert "PLAYOFF PROBABILITIES" in formatted, "Should contain playoff probabilities"

    print("\n✅ Formatted output test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("SEASON PREDICTOR TESTS")
    print("=" * 70 + "\n")

    try:
        test_season_prediction()
        test_formatted_output()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        return True
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
