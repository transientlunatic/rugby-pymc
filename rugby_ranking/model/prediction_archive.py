"""
Prediction archiving system for tracking and evaluating model performance.

Supports:
- Archiving predictions when made
- Updating with actual results
- Retrieving predictions with filtering
- Calibration analysis
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, date as date_type
from pathlib import Path
from typing import Optional, List, Dict, Any
import glob

from rugby_ranking.model.predictions import MatchPrediction, ScorePrediction
from rugby_ranking.model.data import MatchData, normalize_team_name


@dataclass
class PredictionMetadata:
    """Metadata about when and how a prediction was made."""

    timestamp: datetime
    model_checkpoint: str
    model_version: str  # From metadata.pkl
    prediction_type: str  # "teams_only" | "full_lineup"
    software_version: str


@dataclass
class MatchMetadata:
    """Key information identifying the match being predicted."""

    match_id: str
    competition: str
    season: str
    date: datetime
    home_team: str
    away_team: str
    stadium: Optional[str] = None
    round: Optional[int] = None


@dataclass
class ActualResult:
    """Actual match outcome (added after match is played)."""

    home_score: int
    away_score: int
    home_tries: Optional[int] = None
    away_tries: Optional[int] = None
    result_fetched_at: Optional[datetime] = None
    result_source: str = "rugby-data-json"


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating prediction quality."""

    home_score_error: float
    away_score_error: float
    margin_error: float
    outcome_correct: bool  # Did we predict the winner correctly?
    home_in_ci: bool  # Was actual score within 90% CI?
    away_in_ci: bool


@dataclass
class ArchivedPrediction:
    """Complete prediction record with metadata and optional actual result."""

    prediction_id: str
    prediction_metadata: PredictionMetadata
    match_metadata: MatchMetadata
    prediction: MatchPrediction
    model_inputs: Dict[str, Any]

    actual_result: Optional[ActualResult] = None
    result_updated_at: Optional[datetime] = None
    calibration_metrics: Optional[CalibrationMetrics] = None


class PredictionArchiver:
    """
    Manages prediction archival and retrieval.

    Usage:
        # Initialize
        archiver = PredictionArchiver()

        # Archive a prediction
        archiver.archive_prediction(
            prediction=match_pred,
            match_metadata=match_meta,
            model_checkpoint="international-mini5",
            prediction_type="teams_only"
        )

        # Update with actual result (run after match played)
        archiver.update_with_result(
            match_id="six-nations_2026-02-14_france-vs-wales",
            actual_result=result
        )

        # Retrieve predictions
        predictions = archiver.get_predictions(
            competition="six-nations",
            season="2026-2027",
            date_from="2026-02-01"
        )
    """

    def __init__(
        self,
        archive_dir: Optional[Path] = None,
        auto_update_metadata: bool = True
    ):
        """
        Initialize the archiver.

        Args:
            archive_dir: Directory for prediction archives
                        (default: ~/.cache/rugby_ranking/predictions/)
            auto_update_metadata: Automatically update metadata.json on each save
        """
        if archive_dir is None:
            archive_dir = Path("~/.cache/rugby_ranking/predictions").expanduser()

        self.archive_dir = archive_dir
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.archive_dir / "metadata.json"
        self.auto_update_metadata = auto_update_metadata

        # Initialize metadata if doesn't exist
        if not self.metadata_file.exists():
            self._init_metadata()

    def archive_prediction(
        self,
        prediction: MatchPrediction,
        match_metadata: MatchMetadata,
        model_checkpoint: str,
        prediction_type: str,
        model_inputs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Archive a prediction when it's made.

        Args:
            prediction: The match prediction to archive
            match_metadata: Metadata about the match
            model_checkpoint: Name of model checkpoint used
            prediction_type: "teams_only" or "full_lineup"
            model_inputs: Optional additional inputs (lineups, season, etc.)

        Returns:
            prediction_id for tracking
        """
        # Generate prediction ID
        timestamp = datetime.now(timezone.utc)
        prediction_id = self._generate_prediction_id(
            match_metadata, timestamp
        )

        # Get model version
        model_version = self._get_model_version(model_checkpoint)

        # Create archived prediction
        archived = ArchivedPrediction(
            prediction_id=prediction_id,
            prediction_metadata=PredictionMetadata(
                timestamp=timestamp,
                model_checkpoint=model_checkpoint,
                model_version=model_version,
                prediction_type=prediction_type,
                software_version=self._get_software_version()
            ),
            match_metadata=match_metadata,
            prediction=prediction,
            model_inputs=model_inputs or {}
        )

        # Save to date-based file
        self._save_to_archive(archived, timestamp.date())

        if self.auto_update_metadata:
            self._update_metadata_index()

        return prediction_id

    def update_with_result(
        self,
        match_id: str,
        actual_result: ActualResult,
        competition: Optional[str] = None,
        season: Optional[str] = None
    ) -> bool:
        """
        Update archived prediction with actual match result.

        Args:
            match_id: Match identifier
            actual_result: Actual match outcome
            competition: Optional filter to narrow search
            season: Optional filter to narrow search

        Returns:
            True if prediction found and updated, False otherwise
        """
        # Find the prediction(s) for this match
        predictions = self.get_predictions(
            match_id=match_id,
            competition=competition,
            season=season
        )

        if not predictions:
            return False

        # Update all predictions for this match
        # (there may be multiple if predicted at different times)
        for pred in predictions:
            # Calculate calibration metrics
            metrics = self._calculate_calibration_metrics(
                pred.prediction, actual_result
            )

            # Update prediction
            pred.actual_result = actual_result
            pred.result_updated_at = datetime.now(timezone.utc)
            pred.calibration_metrics = metrics

            # Save back to archive
            pred_date = pred.prediction_metadata.timestamp.date()
            self._update_in_archive(pred, pred_date)

        if self.auto_update_metadata:
            self._update_metadata_index()

        return True

    def get_predictions(
        self,
        match_id: Optional[str] = None,
        competition: Optional[str] = None,
        season: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        has_result: Optional[bool] = None,
        model_checkpoint: Optional[str] = None
    ) -> List[ArchivedPrediction]:
        """
        Retrieve predictions with optional filters.

        Args:
            match_id: Specific match ID
            competition: Filter by competition
            season: Filter by season
            date_from: ISO date string for start of range
            date_to: ISO date string for end of range
            has_result: Filter by whether actual result is available
            model_checkpoint: Filter by model checkpoint

        Returns:
            List of matching predictions
        """
        all_predictions = []

        # Load all prediction files
        pattern = str(self.archive_dir / "*.json")
        for filepath in glob.glob(pattern):
            # Skip metadata file
            if Path(filepath).name == "metadata.json":
                continue

            # Load predictions from this file
            with open(filepath, 'r') as f:
                data = json.load(f)
                for pred_dict in data:
                    pred = self._deserialize_prediction(pred_dict)
                    all_predictions.append(pred)

        # Apply filters
        filtered = all_predictions

        if match_id is not None:
            filtered = [p for p in filtered if p.match_metadata.match_id == match_id]

        if competition is not None:
            filtered = [p for p in filtered
                       if p.match_metadata.competition == competition]

        if season is not None:
            filtered = [p for p in filtered if p.match_metadata.season == season]

        if date_from is not None:
            date_from_dt = datetime.fromisoformat(date_from)
            filtered = [p for p in filtered
                       if p.match_metadata.date >= date_from_dt]

        if date_to is not None:
            date_to_dt = datetime.fromisoformat(date_to)
            filtered = [p for p in filtered
                       if p.match_metadata.date <= date_to_dt]

        if has_result is not None:
            if has_result:
                filtered = [p for p in filtered if p.actual_result is not None]
            else:
                filtered = [p for p in filtered if p.actual_result is None]

        if model_checkpoint is not None:
            filtered = [p for p in filtered
                       if p.prediction_metadata.model_checkpoint == model_checkpoint]

        return filtered

    def ingest_results_from_rugby_data(
        self,
        rugby_data_dir: Path,
        match_date_from: Optional[str] = None,
        match_date_to: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Batch update predictions with results from Rugby-Data JSON files.

        This is the main result ingestion pipeline:
        1. Load predictions that need results
        2. Load matches from Rugby-Data
        3. Match predictions to results using fuzzy team names + date
        4. Update predictions with actual outcomes

        Args:
            rugby_data_dir: Path to Rugby-Data json/ directory
            match_date_from: Only process matches after this date (ISO format)
            match_date_to: Only process matches before this date (ISO format)
            dry_run: If True, show what would be updated without making changes

        Returns:
            Stats dict with counts: matched, unmatched, updated, unmatched_list
        """
        from rugby_ranking.model.data import MatchDataset

        # Load predictions that need results
        predictions_without_results = self.get_predictions(has_result=False)

        if not predictions_without_results:
            return {
                "matched": 0,
                "updated": 0,
                "unmatched": 0,
                "unmatched_list": []
            }

        # Load Rugby-Data matches
        dataset = MatchDataset(rugby_data_dir)
        all_matches = dataset.matches

        # Filter by date range if specified
        if match_date_from:
            date_from_dt = datetime.fromisoformat(match_date_from)
            all_matches = [m for m in all_matches if m.date >= date_from_dt]

        if match_date_to:
            date_to_dt = datetime.fromisoformat(match_date_to)
            all_matches = [m for m in all_matches if m.date <= date_to_dt]

        # Only keep played matches
        played_matches = [m for m in all_matches if m.is_played]

        # Match predictions to results
        matched_count = 0
        updated_count = 0
        unmatched_list = []

        for pred in predictions_without_results:
            match_result = self._match_prediction_to_result(pred, played_matches)

            if match_result:
                # Extract actual result
                actual = ActualResult(
                    home_score=match_result.home_score,
                    away_score=match_result.away_score,
                    home_tries=self._count_tries(match_result.home_scores),
                    away_tries=self._count_tries(match_result.away_scores),
                    result_fetched_at=datetime.now(timezone.utc),
                    result_source="rugby-data-json"
                )

                if not dry_run:
                    # Update the prediction
                    self.update_with_result(
                        match_id=pred.match_metadata.match_id,
                        actual_result=actual,
                        competition=pred.match_metadata.competition,
                        season=pred.match_metadata.season
                    )
                    updated_count += 1

                matched_count += 1
            else:
                unmatched_list.append({
                    "prediction_id": pred.prediction_id,
                    "match_id": pred.match_metadata.match_id,
                    "date": pred.match_metadata.date.isoformat(),
                    "home_team": pred.match_metadata.home_team,
                    "away_team": pred.match_metadata.away_team,
                    "competition": pred.match_metadata.competition
                })

        return {
            "matched": matched_count,
            "updated": updated_count if not dry_run else 0,
            "unmatched": len(unmatched_list),
            "unmatched_list": unmatched_list,
            "dry_run": dry_run
        }

    # Private helper methods

    def _generate_prediction_id(
        self, match_meta: MatchMetadata, timestamp: datetime
    ) -> str:
        """Generate unique prediction ID."""
        # Format: timestamp_home-vs-away_match-date
        ts_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        match_date_str = match_meta.date.strftime("%Y-%m-%d")
        home = match_meta.home_team.lower().replace(" ", "-")
        away = match_meta.away_team.lower().replace(" ", "-")

        return f"{ts_str}_{home}-vs-{away}_{match_date_str}"

    def _get_model_version(self, checkpoint_name: str) -> str:
        """Extract model version from checkpoint metadata."""
        try:
            checkpoint_dir = Path("~/.cache/rugby_ranking").expanduser() / checkpoint_name
            metadata_file = checkpoint_dir / "metadata.pkl"

            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                    # Try to extract version info from config
                    if hasattr(metadata.get('config'), '__dict__'):
                        config_dict = metadata['config'].__dict__
                        # Build a simple version string from config
                        features = []
                        if config_dict.get('separate_kicking_effect'):
                            features.append("separate_kicking")
                        if config_dict.get('time_varying_effects'):
                            features.append("time_varying")
                        if features:
                            return "_".join(features)
                return "standard"
        except Exception:
            pass

        return "unknown"

    def _get_software_version(self) -> str:
        """Get rugby-ranking package version."""
        try:
            import rugby_ranking
            if hasattr(rugby_ranking, '__version__'):
                return f"rugby-ranking=={rugby_ranking.__version__}"
        except (ImportError, AttributeError):
            pass

        return "rugby-ranking==unknown"

    def _calculate_calibration_metrics(
        self, prediction: MatchPrediction, actual: ActualResult
    ) -> CalibrationMetrics:
        """Calculate how well prediction matched actual outcome."""
        # Score errors
        home_error = actual.home_score - prediction.home.mean
        away_error = actual.away_score - prediction.away.mean
        margin_error = (actual.home_score - actual.away_score) - prediction.predicted_margin

        # Outcome correctness
        actual_margin = actual.home_score - actual.away_score
        if actual_margin > 0:
            outcome_correct = prediction.home_win_prob > max(prediction.away_win_prob,
                                                              prediction.draw_prob)
        elif actual_margin < 0:
            outcome_correct = prediction.away_win_prob > max(prediction.home_win_prob,
                                                              prediction.draw_prob)
        else:
            outcome_correct = prediction.draw_prob > max(prediction.home_win_prob,
                                                          prediction.away_win_prob)

        # Check if within confidence intervals
        home_in_ci = (prediction.home.ci_lower <= actual.home_score <=
                     prediction.home.ci_upper)
        away_in_ci = (prediction.away.ci_lower <= actual.away_score <=
                     prediction.away.ci_upper)

        return CalibrationMetrics(
            home_score_error=home_error,
            away_score_error=away_error,
            margin_error=margin_error,
            outcome_correct=outcome_correct,
            home_in_ci=home_in_ci,
            away_in_ci=away_in_ci
        )

    def _match_prediction_to_result(
        self,
        pred: ArchivedPrediction,
        rugby_data_matches: List[MatchData]
    ) -> Optional[MatchData]:
        """
        Match prediction to actual result using fuzzy matching.

        Uses:
        - Normalized team names
        - Date matching (exact or within 24h for timezone issues)
        - Competition matching

        Returns:
            Matching MatchData or None if no match found
        """
        # Normalize team names for matching
        pred_home = normalize_team_name(pred.match_metadata.home_team)
        pred_away = normalize_team_name(pred.match_metadata.away_team)
        pred_date = pred.match_metadata.date.date()
        pred_comp = pred.match_metadata.competition.lower().replace("-", "").replace("_", "")

        candidates = []

        for match in rugby_data_matches:
            match_home = normalize_team_name(match.home_team)
            match_away = normalize_team_name(match.away_team)
            match_date = match.date.date()
            match_comp = match.competition.lower().replace("-", "").replace("_", "")

            # Check team match
            teams_match = (
                (match_home == pred_home and match_away == pred_away) or
                # Handle reversed home/away (shouldn't happen but be safe)
                (match_home == pred_away and match_away == pred_home)
            )

            if not teams_match:
                continue

            # Check date proximity (within 1 day for timezone issues)
            date_diff = abs((match_date - pred_date).days)
            if date_diff > 1:
                continue

            # Score the match
            score = 0
            if date_diff == 0:
                score += 10
            if match_comp == pred_comp:
                score += 5
            if match_date == pred_date:
                score += 3

            candidates.append((score, match))

        # Return best match
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None

    def _count_tries(self, scores: List[Dict[str, Any]]) -> int:
        """Count tries from scoring events."""
        if not scores:
            return 0

        try_count = 0
        for score_event in scores:
            if score_event.get('type', '').lower() == 'try':
                try_count += 1

        return try_count

    def _save_to_archive(self, pred: ArchivedPrediction, date: date_type) -> None:
        """Save prediction to date-based JSON file."""
        filepath = self.archive_dir / f"{date.isoformat()}.json"

        # Load existing predictions for this date
        if filepath.exists():
            with open(filepath, 'r') as f:
                predictions = json.load(f)
        else:
            predictions = []

        # Add new prediction
        predictions.append(self._serialize_prediction(pred))

        # Save back
        with open(filepath, 'w') as f:
            json.dump(predictions, f, indent=2)

    def _update_in_archive(self, pred: ArchivedPrediction, date: date_type) -> None:
        """Update existing prediction in archive."""
        filepath = self.archive_dir / f"{date.isoformat()}.json"

        if not filepath.exists():
            # If file doesn't exist, just save it
            self._save_to_archive(pred, date)
            return

        # Load predictions
        with open(filepath, 'r') as f:
            predictions = json.load(f)

        # Find and update the prediction
        updated = False
        for i, pred_dict in enumerate(predictions):
            if pred_dict['prediction_id'] == pred.prediction_id:
                predictions[i] = self._serialize_prediction(pred)
                updated = True
                break

        # If not found, append it
        if not updated:
            predictions.append(self._serialize_prediction(pred))

        # Save back
        with open(filepath, 'w') as f:
            json.dump(predictions, f, indent=2)

    def _init_metadata(self) -> None:
        """Initialize metadata.json file."""
        metadata = {
            "archive_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_predictions": 0,
            "predictions_with_results": 0,
            "competitions": [],
            "seasons": [],
            "date_range": {
                "first_prediction": None,
                "last_prediction": None
            },
            "files": []
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _update_metadata_index(self) -> None:
        """Rebuild metadata.json from archive files."""
        all_predictions = self.get_predictions()

        competitions = set()
        seasons = set()
        files_info = []

        # Scan all date files
        pattern = str(self.archive_dir / "*.json")
        for filepath in glob.glob(pattern):
            if Path(filepath).name == "metadata.json":
                continue

            with open(filepath, 'r') as f:
                preds = json.load(f)
                files_info.append({
                    "date": Path(filepath).stem,
                    "count": len(preds),
                    "file": Path(filepath).name
                })

        # Gather stats
        for pred in all_predictions:
            competitions.add(pred.match_metadata.competition)
            seasons.add(pred.match_metadata.season)

        dates = [p.prediction_metadata.timestamp for p in all_predictions]

        metadata = {
            "archive_version": "1.0",
            "created_at": self.metadata_file.stat().st_ctime
                          if self.metadata_file.exists()
                          else datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_predictions": len(all_predictions),
            "predictions_with_results": len([p for p in all_predictions
                                            if p.actual_result is not None]),
            "competitions": sorted(list(competitions)),
            "seasons": sorted(list(seasons)),
            "date_range": {
                "first_prediction": min(dates).isoformat() if dates else None,
                "last_prediction": max(dates).isoformat() if dates else None
            },
            "files": sorted(files_info, key=lambda x: x['date'])
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _serialize_prediction(self, pred: ArchivedPrediction) -> Dict[str, Any]:
        """Convert ArchivedPrediction to JSON-serializable dict."""
        return {
            "prediction_id": pred.prediction_id,
            "prediction_metadata": {
                "timestamp": pred.prediction_metadata.timestamp.isoformat(),
                "model_checkpoint": pred.prediction_metadata.model_checkpoint,
                "model_version": pred.prediction_metadata.model_version,
                "prediction_type": pred.prediction_metadata.prediction_type,
                "software_version": pred.prediction_metadata.software_version
            },
            "match_metadata": {
                "match_id": pred.match_metadata.match_id,
                "competition": pred.match_metadata.competition,
                "season": pred.match_metadata.season,
                "date": pred.match_metadata.date.isoformat(),
                "home_team": pred.match_metadata.home_team,
                "away_team": pred.match_metadata.away_team,
                "stadium": pred.match_metadata.stadium,
                "round": pred.match_metadata.round
            },
            "prediction": {
                "home": {
                    "team": pred.prediction.home.team,
                    "mean": pred.prediction.home.mean,
                    "std": pred.prediction.home.std,
                    "median": pred.prediction.home.median,
                    "ci_lower": pred.prediction.home.ci_lower,
                    "ci_upper": pred.prediction.home.ci_upper
                },
                "away": {
                    "team": pred.prediction.away.team,
                    "mean": pred.prediction.away.mean,
                    "std": pred.prediction.away.std,
                    "median": pred.prediction.away.median,
                    "ci_lower": pred.prediction.away.ci_lower,
                    "ci_upper": pred.prediction.away.ci_upper
                },
                "home_win_prob": pred.prediction.home_win_prob,
                "away_win_prob": pred.prediction.away_win_prob,
                "draw_prob": pred.prediction.draw_prob,
                "predicted_margin": pred.prediction.predicted_margin,
                "margin_std": pred.prediction.margin_std,
                "home_season_used": pred.prediction.home_season_used,
                "away_season_used": pred.prediction.away_season_used
            },
            "model_inputs": pred.model_inputs,
            "actual_result": {
                "home_score": pred.actual_result.home_score,
                "away_score": pred.actual_result.away_score,
                "home_tries": pred.actual_result.home_tries,
                "away_tries": pred.actual_result.away_tries,
                "result_fetched_at": pred.actual_result.result_fetched_at.isoformat()
                                     if pred.actual_result.result_fetched_at else None,
                "result_source": pred.actual_result.result_source
            } if pred.actual_result else None,
            "result_updated_at": pred.result_updated_at.isoformat()
                                if pred.result_updated_at else None,
            "calibration_metrics": {
                "home_score_error": pred.calibration_metrics.home_score_error,
                "away_score_error": pred.calibration_metrics.away_score_error,
                "margin_error": pred.calibration_metrics.margin_error,
                "outcome_correct": pred.calibration_metrics.outcome_correct,
                "home_in_ci": pred.calibration_metrics.home_in_ci,
                "away_in_ci": pred.calibration_metrics.away_in_ci
            } if pred.calibration_metrics else None
        }

    def _deserialize_prediction(self, data: Dict[str, Any]) -> ArchivedPrediction:
        """Convert JSON dict to ArchivedPrediction object."""
        return ArchivedPrediction(
            prediction_id=data['prediction_id'],
            prediction_metadata=PredictionMetadata(
                timestamp=datetime.fromisoformat(data['prediction_metadata']['timestamp']),
                model_checkpoint=data['prediction_metadata']['model_checkpoint'],
                model_version=data['prediction_metadata']['model_version'],
                prediction_type=data['prediction_metadata']['prediction_type'],
                software_version=data['prediction_metadata']['software_version']
            ),
            match_metadata=MatchMetadata(
                match_id=data['match_metadata']['match_id'],
                competition=data['match_metadata']['competition'],
                season=data['match_metadata']['season'],
                date=datetime.fromisoformat(data['match_metadata']['date']),
                home_team=data['match_metadata']['home_team'],
                away_team=data['match_metadata']['away_team'],
                stadium=data['match_metadata'].get('stadium'),
                round=data['match_metadata'].get('round')
            ),
            prediction=MatchPrediction(
                home=ScorePrediction(
                    team=data['prediction']['home']['team'],
                    mean=data['prediction']['home']['mean'],
                    std=data['prediction']['home']['std'],
                    median=data['prediction']['home']['median'],
                    ci_lower=data['prediction']['home']['ci_lower'],
                    ci_upper=data['prediction']['home']['ci_upper']
                ),
                away=ScorePrediction(
                    team=data['prediction']['away']['team'],
                    mean=data['prediction']['away']['mean'],
                    std=data['prediction']['away']['std'],
                    median=data['prediction']['away']['median'],
                    ci_lower=data['prediction']['away']['ci_lower'],
                    ci_upper=data['prediction']['away']['ci_upper']
                ),
                home_win_prob=data['prediction']['home_win_prob'],
                away_win_prob=data['prediction']['away_win_prob'],
                draw_prob=data['prediction']['draw_prob'],
                predicted_margin=data['prediction']['predicted_margin'],
                margin_std=data['prediction']['margin_std'],
                home_season_used=data['prediction'].get('home_season_used'),
                away_season_used=data['prediction'].get('away_season_used')
            ),
            model_inputs=data.get('model_inputs', {}),
            actual_result=ActualResult(
                home_score=data['actual_result']['home_score'],
                away_score=data['actual_result']['away_score'],
                home_tries=data['actual_result'].get('home_tries'),
                away_tries=data['actual_result'].get('away_tries'),
                result_fetched_at=datetime.fromisoformat(
                    data['actual_result']['result_fetched_at']
                ) if data['actual_result'].get('result_fetched_at') else None,
                result_source=data['actual_result'].get('result_source', 'rugby-data-json')
            ) if data.get('actual_result') else None,
            result_updated_at=datetime.fromisoformat(data['result_updated_at'])
                             if data.get('result_updated_at') else None,
            calibration_metrics=CalibrationMetrics(
                home_score_error=data['calibration_metrics']['home_score_error'],
                away_score_error=data['calibration_metrics']['away_score_error'],
                margin_error=data['calibration_metrics']['margin_error'],
                outcome_correct=data['calibration_metrics']['outcome_correct'],
                home_in_ci=data['calibration_metrics']['home_in_ci'],
                away_in_ci=data['calibration_metrics']['away_in_ci']
            ) if data.get('calibration_metrics') else None
        )
