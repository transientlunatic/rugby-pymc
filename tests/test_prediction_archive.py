"""
Unit tests for prediction archival system.
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import json

from rugby_ranking.model.predictions import MatchPrediction, ScorePrediction
from rugby_ranking.model.prediction_archive import (
    PredictionArchiver,
    PredictionMetadata,
    MatchMetadata,
    ActualResult,
    CalibrationMetrics,
    ArchivedPrediction
)


@pytest.fixture
def temp_archive_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_prediction():
    """Create a sample MatchPrediction for testing."""
    return MatchPrediction(
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


@pytest.fixture
def sample_match_metadata():
    """Create sample match metadata."""
    return MatchMetadata(
        match_id="six-nations_2026-02-14_france-vs-wales",
        competition="six-nations",
        season="2026-2027",
        date=datetime(2026, 2, 14, 21, 10, tzinfo=timezone.utc),
        home_team="France",
        away_team="Wales",
        stadium="Stade de France",
        round=2
    )


@pytest.fixture
def sample_actual_result():
    """Create sample actual result."""
    return ActualResult(
        home_score=36,
        away_score=17,
        home_tries=4,
        away_tries=2,
        result_fetched_at=datetime.now(timezone.utc),
        result_source="rugby-data-json"
    )


class TestPredictionArchiver:
    """Tests for PredictionArchiver class."""

    def test_init_creates_directory(self, temp_archive_dir):
        """Test that initialization creates archive directory."""
        archive_dir = temp_archive_dir / "predictions"
        archiver = PredictionArchiver(archive_dir=archive_dir)

        assert archive_dir.exists()
        assert archiver.archive_dir == archive_dir
        assert archiver.metadata_file.exists()

    def test_init_creates_metadata_file(self, temp_archive_dir):
        """Test that metadata.json is created on init."""
        archive_dir = temp_archive_dir / "predictions"
        archiver = PredictionArchiver(archive_dir=archive_dir)

        metadata_file = archive_dir / "metadata.json"
        assert metadata_file.exists()

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        assert metadata['archive_version'] == "1.0"
        assert metadata['total_predictions'] == 0
        assert metadata['predictions_with_results'] == 0

    def test_archive_prediction(
        self, temp_archive_dir, sample_prediction, sample_match_metadata
    ):
        """Test archiving a prediction."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        prediction_id = archiver.archive_prediction(
            prediction=sample_prediction,
            match_metadata=sample_match_metadata,
            model_checkpoint="test-checkpoint",
            prediction_type="teams_only",
            model_inputs={"season": "2026-2027"}
        )

        # Check prediction ID format
        assert "france-vs-wales" in prediction_id.lower()
        assert "2026-02-14" in prediction_id

        # Check that file was created
        date_str = sample_match_metadata.date.date().isoformat()
        archive_file = temp_archive_dir / f"{date_str}.json"
        assert archive_file.exists()

        # Check file contents
        with open(archive_file, 'r') as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]['prediction_id'] == prediction_id
        assert data[0]['match_metadata']['home_team'] == "France"
        assert data[0]['match_metadata']['away_team'] == "Wales"
        assert data[0]['prediction']['home']['mean'] == 28.5

    def test_archive_multiple_predictions_same_date(
        self, temp_archive_dir, sample_prediction, sample_match_metadata
    ):
        """Test archiving multiple predictions on the same date."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        # Archive first prediction
        id1 = archiver.archive_prediction(
            prediction=sample_prediction,
            match_metadata=sample_match_metadata,
            model_checkpoint="test-checkpoint",
            prediction_type="teams_only"
        )

        # Modify metadata for second match
        match_meta_2 = MatchMetadata(
            match_id="six-nations_2026-02-14_england-vs-ireland",
            competition="six-nations",
            season="2026-2027",
            date=datetime(2026, 2, 14, 16, 45, tzinfo=timezone.utc),
            home_team="England",
            away_team="Ireland"
        )

        # Archive second prediction
        id2 = archiver.archive_prediction(
            prediction=sample_prediction,
            match_metadata=match_meta_2,
            model_checkpoint="test-checkpoint",
            prediction_type="teams_only"
        )

        # Check that both are in the same file
        date_str = sample_match_metadata.date.date().isoformat()
        archive_file = temp_archive_dir / f"{date_str}.json"

        with open(archive_file, 'r') as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]['prediction_id'] == id1
        assert data[1]['prediction_id'] == id2

    def test_get_predictions_no_filters(
        self, temp_archive_dir, sample_prediction, sample_match_metadata
    ):
        """Test retrieving all predictions without filters."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        # Archive a prediction
        archiver.archive_prediction(
            prediction=sample_prediction,
            match_metadata=sample_match_metadata,
            model_checkpoint="test-checkpoint",
            prediction_type="teams_only"
        )

        # Retrieve all predictions
        predictions = archiver.get_predictions()

        assert len(predictions) == 1
        assert predictions[0].match_metadata.home_team == "France"
        assert predictions[0].match_metadata.away_team == "Wales"

    def test_get_predictions_filter_by_competition(
        self, temp_archive_dir, sample_prediction, sample_match_metadata
    ):
        """Test filtering predictions by competition."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        # Archive Six Nations prediction
        archiver.archive_prediction(
            prediction=sample_prediction,
            match_metadata=sample_match_metadata,
            model_checkpoint="test-checkpoint",
            prediction_type="teams_only"
        )

        # Archive Celtic prediction
        celtic_meta = MatchMetadata(
            match_id="celtic_2026-02-15_leinster-vs-munster",
            competition="celtic",
            season="2025-2026",
            date=datetime(2026, 2, 15, 19, 35, tzinfo=timezone.utc),
            home_team="Leinster",
            away_team="Munster"
        )
        archiver.archive_prediction(
            prediction=sample_prediction,
            match_metadata=celtic_meta,
            model_checkpoint="test-checkpoint",
            prediction_type="teams_only"
        )

        # Filter by Six Nations
        six_nations = archiver.get_predictions(competition="six-nations")
        assert len(six_nations) == 1
        assert six_nations[0].match_metadata.competition == "six-nations"

        # Filter by Celtic
        celtic = archiver.get_predictions(competition="celtic")
        assert len(celtic) == 1
        assert celtic[0].match_metadata.competition == "celtic"

    def test_get_predictions_filter_by_has_result(
        self, temp_archive_dir, sample_prediction, sample_match_metadata,
        sample_actual_result
    ):
        """Test filtering predictions by whether they have results."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        # Archive two predictions
        archiver.archive_prediction(
            prediction=sample_prediction,
            match_metadata=sample_match_metadata,
            model_checkpoint="test-checkpoint",
            prediction_type="teams_only"
        )

        match_meta_2 = MatchMetadata(
            match_id="six-nations_2026-02-15_england-vs-ireland",
            competition="six-nations",
            season="2026-2027",
            date=datetime(2026, 2, 15, 16, 45, tzinfo=timezone.utc),
            home_team="England",
            away_team="Ireland"
        )
        archiver.archive_prediction(
            prediction=sample_prediction,
            match_metadata=match_meta_2,
            model_checkpoint="test-checkpoint",
            prediction_type="teams_only"
        )

        # Update first prediction with result
        archiver.update_with_result(
            match_id=sample_match_metadata.match_id,
            actual_result=sample_actual_result
        )

        # Filter for predictions with results
        with_results = archiver.get_predictions(has_result=True)
        assert len(with_results) == 1
        assert with_results[0].actual_result is not None

        # Filter for predictions without results
        without_results = archiver.get_predictions(has_result=False)
        assert len(without_results) == 1
        assert without_results[0].actual_result is None

    def test_update_with_result(
        self, temp_archive_dir, sample_prediction, sample_match_metadata,
        sample_actual_result
    ):
        """Test updating a prediction with actual result."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        # Archive prediction
        archiver.archive_prediction(
            prediction=sample_prediction,
            match_metadata=sample_match_metadata,
            model_checkpoint="test-checkpoint",
            prediction_type="teams_only"
        )

        # Update with result
        success = archiver.update_with_result(
            match_id=sample_match_metadata.match_id,
            actual_result=sample_actual_result
        )

        assert success is True

        # Retrieve and check
        predictions = archiver.get_predictions(
            match_id=sample_match_metadata.match_id
        )

        assert len(predictions) == 1
        pred = predictions[0]
        assert pred.actual_result is not None
        assert pred.actual_result.home_score == 36
        assert pred.actual_result.away_score == 17
        assert pred.result_updated_at is not None
        assert pred.calibration_metrics is not None

    def test_update_with_result_not_found(
        self, temp_archive_dir, sample_actual_result
    ):
        """Test updating a non-existent prediction."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        success = archiver.update_with_result(
            match_id="nonexistent_match",
            actual_result=sample_actual_result
        )

        assert success is False

    def test_calculate_calibration_metrics_correct_winner(
        self, temp_archive_dir, sample_prediction
    ):
        """Test calibration metrics when winner is predicted correctly."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        # Actual result: home team won (as predicted)
        actual = ActualResult(
            home_score=30,
            away_score=20,
            home_tries=4,
            away_tries=3
        )

        metrics = archiver._calculate_calibration_metrics(sample_prediction, actual)

        assert metrics.outcome_correct is True
        assert metrics.home_score_error == pytest.approx(30 - 28.5)
        assert metrics.away_score_error == pytest.approx(20 - 18.3)
        assert metrics.margin_error == pytest.approx((30 - 20) - 10.2)
        assert metrics.home_in_ci is True  # 30 is within [15, 43]
        assert metrics.away_in_ci is True  # 20 is within [6, 32]

    def test_calculate_calibration_metrics_wrong_winner(
        self, temp_archive_dir, sample_prediction
    ):
        """Test calibration metrics when winner is predicted incorrectly."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        # Actual result: away team won (upset!)
        actual = ActualResult(
            home_score=15,
            away_score=25,
            home_tries=2,
            away_tries=4
        )

        metrics = archiver._calculate_calibration_metrics(sample_prediction, actual)

        assert metrics.outcome_correct is False
        assert metrics.home_in_ci is True  # 15 is within [15, 43]
        assert metrics.away_in_ci is True  # 25 is within [6, 32]

    def test_calculate_calibration_metrics_outside_ci(
        self, temp_archive_dir, sample_prediction
    ):
        """Test calibration metrics when scores are outside confidence intervals."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        # Actual result: home team scored way more than predicted
        actual = ActualResult(
            home_score=50,  # Above ci_upper of 43
            away_score=5,   # Below ci_lower of 6
            home_tries=7,
            away_tries=1
        )

        metrics = archiver._calculate_calibration_metrics(sample_prediction, actual)

        assert metrics.home_in_ci is False
        assert metrics.away_in_ci is False

    def test_metadata_index_updated(
        self, temp_archive_dir, sample_prediction, sample_match_metadata
    ):
        """Test that metadata index is updated when predictions are archived."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        # Archive a prediction
        archiver.archive_prediction(
            prediction=sample_prediction,
            match_metadata=sample_match_metadata,
            model_checkpoint="test-checkpoint",
            prediction_type="teams_only"
        )

        # Read metadata
        with open(archiver.metadata_file, 'r') as f:
            metadata = json.load(f)

        assert metadata['total_predictions'] == 1
        assert metadata['predictions_with_results'] == 0
        assert "six-nations" in metadata['competitions']
        assert "2026-2027" in metadata['seasons']

    def test_serialization_roundtrip(
        self, temp_archive_dir, sample_prediction, sample_match_metadata
    ):
        """Test that predictions can be serialized and deserialized correctly."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        # Archive prediction
        prediction_id = archiver.archive_prediction(
            prediction=sample_prediction,
            match_metadata=sample_match_metadata,
            model_checkpoint="test-checkpoint",
            prediction_type="teams_only",
            model_inputs={"season": "2026-2027", "test_param": 123}
        )

        # Retrieve prediction
        predictions = archiver.get_predictions(match_id=sample_match_metadata.match_id)

        assert len(predictions) == 1
        retrieved = predictions[0]

        # Check all fields preserved
        assert retrieved.prediction_id == prediction_id
        assert retrieved.match_metadata.home_team == "France"
        assert retrieved.match_metadata.away_team == "Wales"
        assert retrieved.prediction.home.mean == 28.5
        assert retrieved.prediction.away.mean == 18.3
        assert retrieved.prediction.home_win_prob == 0.78
        assert retrieved.model_inputs['season'] == "2026-2027"
        assert retrieved.model_inputs['test_param'] == 123


class TestCalibrationReport:
    """Tests for calibration_report method."""

    def test_empty_archive(self, temp_archive_dir):
        """Test report with no scored predictions."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)
        report = archiver.calibration_report()
        assert report == {"n": 0}

    def test_calibration_report_basic(
        self, temp_archive_dir, sample_prediction, sample_match_metadata, sample_actual_result
    ):
        """Test report with one scored prediction."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        archiver.archive_prediction(
            prediction=sample_prediction,
            match_metadata=sample_match_metadata,
            model_checkpoint="test-checkpoint",
            prediction_type="teams_only"
        )
        archiver.update_with_result(
            match_id=sample_match_metadata.match_id,
            actual_result=sample_actual_result
        )

        report = archiver.calibration_report()

        assert report['n'] == 1
        assert 'outcome_accuracy' in report
        assert 'brier_score' in report
        assert 'mae_home' in report
        assert 'mae_away' in report
        assert 'mae_margin' in report
        assert 'home_ci_coverage' in report
        assert 'away_ci_coverage' in report
        # France predicted to win (home_win_prob=0.78), actual France 36-17: correct
        assert report['outcome_accuracy'] == 1.0
        assert report['mae_home'] == pytest.approx(abs(36 - 28.5))
        assert report['mae_away'] == pytest.approx(abs(17 - 18.3))

    def test_calibration_report_filter_by_competition(
        self, temp_archive_dir, sample_prediction, sample_match_metadata, sample_actual_result
    ):
        """Test report filtered by competition."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        archiver.archive_prediction(
            prediction=sample_prediction,
            match_metadata=sample_match_metadata,
            model_checkpoint="test-checkpoint",
            prediction_type="teams_only"
        )
        archiver.update_with_result(
            match_id=sample_match_metadata.match_id,
            actual_result=sample_actual_result
        )

        report_sn = archiver.calibration_report(competition="six-nations")
        assert report_sn['n'] == 1

        report_other = archiver.calibration_report(competition="celtic")
        assert report_other == {"n": 0}


class TestPredictionID:
    """Tests for prediction ID generation."""

    def test_prediction_id_format(self, temp_archive_dir):
        """Test that prediction IDs have the correct format."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        match_meta = MatchMetadata(
            match_id="test_match",
            competition="six-nations",
            season="2026-2027",
            date=datetime(2026, 2, 14, 21, 10, tzinfo=timezone.utc),
            home_team="France",
            away_team="Wales"
        )

        timestamp = datetime(2026, 2, 10, 15, 30, 0, tzinfo=timezone.utc)

        pred_id = archiver._generate_prediction_id(match_meta, timestamp)

        # Should contain timestamp
        assert "2026-02-10" in pred_id
        assert "15:30:00" in pred_id

        # Should contain teams
        assert "france" in pred_id.lower()
        assert "wales" in pred_id.lower()

        # Should contain match date
        assert "2026-02-14" in pred_id

    def test_prediction_ids_unique(self, temp_archive_dir):
        """Test that prediction IDs are unique even for same match."""
        archiver = PredictionArchiver(archive_dir=temp_archive_dir)

        match_meta = MatchMetadata(
            match_id="test_match",
            competition="six-nations",
            season="2026-2027",
            date=datetime(2026, 2, 14, 21, 10, tzinfo=timezone.utc),
            home_team="France",
            away_team="Wales"
        )

        # Generate IDs at different times
        timestamp1 = datetime(2026, 2, 10, 15, 30, 0, tzinfo=timezone.utc)
        timestamp2 = datetime(2026, 2, 10, 15, 30, 1, tzinfo=timezone.utc)

        id1 = archiver._generate_prediction_id(match_meta, timestamp1)
        id2 = archiver._generate_prediction_id(match_meta, timestamp2)

        assert id1 != id2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
