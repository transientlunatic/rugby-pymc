# Rugby Ranking Documentation

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting_started/installation
getting_started/quickstart
getting_started/architecture
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guides/model_fundamentals
guides/model_validation
guides/weekly_workflow
guides/predictions
guides/squad_analysis
guides/paths_to_victory
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/model
api/data
api/inference
api/predictions
api/utils
```

```{toctree}
:maxdepth: 2
:caption: Development

development/architecture
development/testing
development/contributing
```

---

**Rugby Ranking** is a Bayesian hierarchical model for ranking rugby union players and teams, with support for match predictions and tournament analysis.

## Features

- **Hierarchical Player Rankings**: Accounts for player ability, team quality, and position effects
- **Match Predictions**: Team-only or full-lineup predictions with uncertainty quantification
- **Squad Analysis**: Pre-match squad strength assessment and injury impact analysis
- **Paths to Victory**: Tournament-wide analysis showing how teams can reach target finishing positions
- **Incremental Updates**: Weekly model updates via variational inference (~5 min) or thorough MCMC (~30-60 min)

## Quick Links

- [Installation](getting_started/installation)
- [Quick Start Guide](getting_started/quickstart)
- [Model Architecture](guides/model_fundamentals)
- [API Reference](api/model)
