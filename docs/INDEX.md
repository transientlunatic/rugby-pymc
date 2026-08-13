# Documentation Index

> **Note**: This is the legacy flat documentation index. The primary documentation is now structured under `docs/getting_started/`, `docs/guides/`, and `docs/development/`. See `docs/index.md` for the Sphinx documentation root.

---

## Getting Started

- [docs/getting_started/installation.md](getting_started/installation.md) — Installation and setup
- [docs/getting_started/quickstart.md](getting_started/quickstart.md) — Quick start with code examples
- [docs/getting_started/architecture.md](getting_started/architecture.md) — System architecture

## User Guides

- [guides/model_fundamentals.md](guides/model_fundamentals.md) — Model architecture and design decisions
- [guides/weekly_workflow.md](guides/weekly_workflow.md) — **MCMC training, upload, and weekly update workflow**
- [guides/predictions.md](guides/predictions.md) — Match predictions
- [guides/squad_analysis.md](guides/squad_analysis.md) — Squad strength analysis
- [guides/paths_to_victory.md](guides/paths_to_victory.md) — Tournament path analysis

## Feature Documentation

- [HTCONDOR_TRAINING.md](HTCONDOR_TRAINING.md) — HTCondor cluster training with checkpointing
- [HTCONDOR_LOGGING.md](HTCONDOR_LOGGING.md) — HTCondor logging configuration
- [DASHBOARD.md](DASHBOARD.md) — Web dashboard setup and deployment
- [DASHBOARD_QUICKSTART.md](DASHBOARD_QUICKSTART.md) — Dashboard quick start
- [DATA_UTILS_GUIDE.md](DATA_UTILS_GUIDE.md) — Data utilities reference
- [LEAGUE_TABLE_AND_SEASON_PREDICTION.md](LEAGUE_TABLE_AND_SEASON_PREDICTION.md) — League table and season prediction
- [SQUAD_ANALYSIS_DESIGN.md](SQUAD_ANALYSIS_DESIGN.md) — Squad analysis design
- [SQUAD_ANALYSIS_QUICKSTART.md](SQUAD_ANALYSIS_QUICKSTART.md) — Squad analysis quick start
- [PATHS_TO_VICTORY_DESIGN.md](PATHS_TO_VICTORY_DESIGN.md) — Paths to victory design
- [PATHS_TO_VICTORY_SUMMARY.md](PATHS_TO_VICTORY_SUMMARY.md) — Paths to victory summary
- [BRACKET_PREDICTION_DESIGN.md](BRACKET_PREDICTION_DESIGN.md) — Knockout bracket prediction
- [BRACKET_PREDICTION_QUICKSTART.md](BRACKET_PREDICTION_QUICKSTART.md) — Bracket prediction quick start
- [BRACKET_PREDICTION_SUMMARY.md](BRACKET_PREDICTION_SUMMARY.md) — Bracket prediction summary
- [TIME_VARYING_IMPLEMENTATION.md](TIME_VARYING_IMPLEMENTATION.md) — Time-varying effects feature
- [DEFENSIVE_MINIBATCH_UPDATE.md](DEFENSIVE_MINIBATCH_UPDATE.md) — Defensive effects and minibatch VI
- [RANKING_IMPROVEMENTS.md](RANKING_IMPROVEMENTS.md) — Player ranking enhancements
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) — Phase 1-3 implementation overview

## Development

- [development/code_organization.md](development/code_organization.md) — Module structure and utilities
- [development/testing.md](development/testing.md) — Test suite and coverage
- [development/contributing.md](development/contributing.md) — Contributing guidelines

## Archive

Historical completion reports and fix summaries are in [archive/](archive/).

---

## Module Structure

```
rugby_ranking/
├── model/
│   ├── data.py              # Data loading: MatchDataset, PlayerMatchObservation
│   ├── core.py              # PyMC model: RugbyModel, ModelConfig
│   ├── inference.py         # Fitting: ModelFitter, InferenceConfig
│   ├── predictions.py       # Predictions: MatchPredictor, MatchPrediction
│   ├── validation.py        # Validation: temporal_split, cross_validation, etc.
│   ├── data_validation.py   # Data quality: detect_kicking_anomalies, clean_kicking_data
│   ├── data_utils.py        # Data utilities
│   ├── positions.py         # Position mappings and groupings
│   ├── name_analysis.py     # Player name fuzzy matching analysis
│   ├── season_predictor.py  # Monte Carlo season simulation
│   ├── league_table.py      # League standings computation
│   ├── knockout_forecast.py # Knockout bracket simulation
│   ├── bracket.py           # Bracket structure definitions
│   ├── bracket_predictor.py # Bracket predictions
│   ├── tournament_paths.py  # Tournament path analysis
│   ├── tbd_resolver.py      # TBD match resolution
│   ├── paths_to_victory.py  # Paths to victory analysis
│   ├── squad_analysis.py    # Squad analysis and comparison
│   └── prediction_archive.py # Prediction history
├── utils/
│   ├── logging.py           # Consistent output formatting
│   ├── cli_helpers.py       # Common CLI operations
│   └── constants.py         # Shared constants (positions, scoring)
├── cli.py                   # Command-line interface
└── notebook_utils.py        # Notebook boilerplate utilities
```
