#!/usr/bin/env python3
"""
Simple integration test for prediction archival system.
Tests the basic workflow: archive prediction → update with result → verify metrics.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from rugby_ranking.model.predictions import MatchPrediction, ScorePrediction
from rugby_ranking.model.prediction_archive import (
    PredictionArchiver,
    MatchMetadata,
    ActualResult
)


def test_basic_workflow():
    """Test basic prediction archival workflow."""
    print("Testing Prediction Archive System")
    print("=" * 70)

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_dir = Path(tmpdir) / "predictions"
        print(f"\n1. Creating archiver with temporary directory: {archive_dir}")
        archiver = PredictionArchiver(archive_dir=archive_dir)
        print(f"   ✓ Archive directory created")
        print(f"   ✓ Metadata file created")

        # Create a sample prediction
        print("\n2. Creating sample prediction...")
        prediction = MatchPrediction(
            home=ScorePrediction(
                team="France",
                mean=28.5,
                std=8.2,
                median=28.0,
                ci_lower=15.0,
                ci_upper=43.0
            ),
            away=ScorePrediction(
                team="Wales",
                mean=18.3,
                std=7.8,
                median=18.0,
                ci_lower=6.0,
                ci_upper=32.0
            ),
            home_win_prob=0.78,
            away_win_prob=0.19,
            draw_prob=0.03,
            predicted_margin=10.2,
            margin_std=11.3
        )
        print(f"   ✓ Prediction: France {prediction.home.mean:.0f} - {prediction.away.mean:.0f} Wales")
        print(f"   ✓ Win probabilities: Home {prediction.home_win_prob:.1%}, Away {prediction.away_win_prob:.1%}")

        # Create match metadata
        match_meta = MatchMetadata(
            match_id="six-nations_2026-02-14_france-vs-wales",
            competition="six-nations",
            season="2026-2027",
            date=datetime(2026, 2, 14, 21, 10, tzinfo=timezone.utc),
            home_team="France",
            away_team="Wales",
            stadium="Stade de France",
            round=2
        )

        # Archive the prediction
        print("\n3. Archiving prediction...")
        prediction_id = archiver.archive_prediction(
            prediction=prediction,
            match_metadata=match_meta,
            model_checkpoint="test-checkpoint",
            prediction_type="teams_only",
            model_inputs={"season": "2026-2027"}
        )
        print(f"   ✓ Archived with ID: {prediction_id}")

        # Verify it was saved
        print("\n4. Retrieving archived prediction...")
        predictions = archiver.get_predictions()
        assert len(predictions) == 1, f"Expected 1 prediction, found {len(predictions)}"
        print(f"   ✓ Found {len(predictions)} prediction(s)")

        retrieved = predictions[0]
        assert retrieved.match_metadata.home_team == "France"
        assert retrieved.match_metadata.away_team == "Wales"
        assert retrieved.actual_result is None
        print(f"   ✓ Match: {retrieved.match_metadata.home_team} vs {retrieved.match_metadata.away_team}")
        print(f"   ✓ Status: No result yet")

        # Create actual result
        print("\n5. Simulating match result...")
        actual = ActualResult(
            home_score=32,
            away_score=20,
            home_tries=4,
            away_tries=3,
            result_fetched_at=datetime.now(timezone.utc),
            result_source="test"
        )
        print(f"   ✓ Actual result: France {actual.home_score} - {actual.away_score} Wales")

        # Update with result
        print("\n6. Updating prediction with actual result...")
        success = archiver.update_with_result(
            match_id=match_meta.match_id,
            actual_result=actual
        )
        assert success, "Failed to update prediction with result"
        print(f"   ✓ Prediction updated successfully")

        # Retrieve and verify calibration metrics
        print("\n7. Verifying calibration metrics...")
        predictions = archiver.get_predictions(match_id=match_meta.match_id)
        updated = predictions[0]

        assert updated.actual_result is not None
        assert updated.actual_result.home_score == 32
        assert updated.calibration_metrics is not None

        metrics = updated.calibration_metrics
        print(f"   ✓ Home score error: {metrics.home_score_error:.1f} points")
        print(f"   ✓ Away score error: {metrics.away_score_error:.1f} points")
        print(f"   ✓ Margin error: {metrics.margin_error:.1f} points")
        print(f"   ✓ Outcome correct: {metrics.outcome_correct}")
        print(f"   ✓ Home in CI: {metrics.home_in_ci}")
        print(f"   ✓ Away in CI: {metrics.away_in_ci}")

        # Test filtering
        print("\n8. Testing prediction filtering...")
        six_nations = archiver.get_predictions(competition="six-nations")
        assert len(six_nations) == 1
        print(f"   ✓ Filter by competition: found {len(six_nations)} Six Nations prediction(s)")

        with_results = archiver.get_predictions(has_result=True)
        assert len(with_results) == 1
        print(f"   ✓ Filter by has_result=True: found {len(with_results)} prediction(s)")

        without_results = archiver.get_predictions(has_result=False)
        assert len(without_results) == 0
        print(f"   ✓ Filter by has_result=False: found {len(without_results)} prediction(s)")

        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)


if __name__ == "__main__":
    try:
        test_basic_workflow()
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
