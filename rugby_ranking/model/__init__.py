"""
Bayesian modelling components for rugby ranking.
"""

from rugby_ranking.model.data import MatchDataset, PlayerMatchObservation
from rugby_ranking.model.core import RugbyModel, ModelConfig
from rugby_ranking.model.inference import ModelFitter, InferenceConfig
from rugby_ranking.model.predictions import MatchPredictor, MatchPrediction
from rugby_ranking.model.validation import (
    ValidationSplit,
    ValidationMetrics,
    temporal_split,
    random_match_split,
    season_holdout_split,
    compute_validation_metrics,
    cross_validation,
    baseline_predictions,
)
from rugby_ranking.model.name_analysis import (
    analyze_merged_names,
    find_potential_duplicates,
    review_merges,
    generate_correction_dict,
    get_name_variations,
    export_merge_report,
)
from rugby_ranking.model import positions
from rugby_ranking.model.knockout_forecast import (
    TournamentTreeSimulator,
    TournamentForecast,
    BracketStructure,
    URCPlayoffBracket,
    WorldCupBracket,
    ChampionsCupBracket,
    KnockoutMatch,
    KnockoutStageResult,
    format_knockout_forecast,
)

__all__ = [
    "MatchDataset",
    "PlayerMatchObservation",
    "RugbyModel",
    "ModelConfig",
    "ModelFitter",
    "InferenceConfig",
    "MatchPredictor",
    "MatchPrediction",
    "ValidationSplit",
    "ValidationMetrics",
    "temporal_split",
    "random_match_split",
    "season_holdout_split",
    "compute_validation_metrics",
    "cross_validation",
    "baseline_predictions",
    "analyze_merged_names",
    "find_potential_duplicates",
    "review_merges",
    "generate_correction_dict",
    "get_name_variations",
    "export_merge_report",
    "positions",
    "TournamentTreeSimulator",
    "TournamentForecast",
    "BracketStructure",
    "URCPlayoffBracket",
    "WorldCupBracket",
    "ChampionsCupBracket",
    "KnockoutMatch",
    "KnockoutStageResult",
    "format_knockout_forecast",
]
