# Model Fundamentals

## Overview

The rugby ranking model is a **Bayesian hierarchical Poisson regression** that estimates scoring rates for individual players. It separates intrinsic player ability from the team system they play in, allowing fair comparisons across teams and over time.

## Mathematical Structure

### Core Equation

For each player-match observation, the expected scoring rate is:

```
log(λ_score[i,m]) = α                           # global baseline rate
                  + β_player[i]                 # intrinsic player ability
                  + γ_team_season[j,s]          # team system effect (per season)
                  + θ_position[k]              # positional base rate
                  + η_home × is_home           # home advantage
                  + log(minutes / 80)          # exposure offset
```

where `i` is the player, `j` is the team, `s` is the season, `k` is the position, and `m` is the match.

### Separate Kicking and Try-Scoring Effects

By default (`ModelConfig.separate_kicking_effect=True`), the model uses two distinct player ability parameters:

- **β_player_try[i]**: Try-scoring ability (running, finishing, positioning)
  — used for *tries*
- **β_player_kick[i]**: Kicking ability (accuracy, technique)
  — used for *conversions, penalties, drop goals*

This reflects that try-scoring and kicking are different skills — a great finisher may be a poor kicker, and vice versa.

### Hierarchical Priors

All player and team effects have weakly informative hierarchical priors:

```
β_player_try[i] ~ Normal(0, σ_player_try)     # global hyper-prior
β_player_kick[i] ~ Normal(0, σ_player_kick)
γ_team_season[j,s] ~ Normal(0, σ_team)
θ_position[k] ~ Normal(0, σ_position)
η_home ~ Normal(0, 0.5)

σ_player_try ~ HalfNormal(player_try_effect_sd)
```

The hierarchical structure means that even players with few observations are regularised toward the population mean, preventing overfitting.

## Score Types

The joint model fits all four scoring types simultaneously, sharing player and team effects:

| Type | Points | Primary positions |
|------|--------|-------------------|
| Tries | 5 (+2 conversion) | All (wings/centres score most) |
| Conversions | 2 | 10 (fly-half), 15 (fullback) |
| Penalties | 3 | 10, 15 |
| Drop goals | 3 | 10 |

## Key Design Decisions

### Player Mobility

Players changing teams are handled naturally:
- `β_player` follows the player (it's intrinsic ability)
- `γ_team_season` captures the system (coaching, tactics, squad)

This lets the model answer: "How much of Player X's output is them vs. their team?"

### Exposure Normalisation

Playing time varies due to substitutions and red cards. The `log(minutes/80)` offset ensures a player's rate is estimated per 80 minutes, not per match, making comparisons fair.

### Season-Specific Team Effects

Teams change significantly between seasons (coaching changes, player transfers). `γ_team[j,s]` is indexed by both team and season, so the model doesn't assume Leicester 2023-2024 plays like Leicester 2025-2026.

## Configuration

```python
from rugby_ranking.model import ModelConfig, RugbyModel

config = ModelConfig(
    separate_kicking_effect=True,  # Default: separate try/kick player effects
    time_varying_effects=False,    # Within-season form trends (experimental)
    player_try_effect_sd=0.5,      # Prior scale for try-scoring ability
    player_kicking_effect_sd=0.5,  # Prior scale for kicking ability
    team_effect_sd=0.3,            # Prior scale for team system effect
    position_effect_sd=0.5,        # Prior scale for positional effects
)
model = RugbyModel(config=config)
```

## Model Variants

| Variant | When to use | Build method |
|---------|-------------|--------------|
| **Joint static** (default) | Normal use | `model.build_joint(df)` |
| **Joint time-varying** | Capturing form trends | `model.build_joint_time_varying(df)` |
| **Single score type** | Debugging, comparison | `model.build(df, score_type="tries")` |
| **Minibatch** | Large datasets, fast VI | `model.build_joint_minibatch(df)` |

## Inference Methods

| Method | Speed | When to use |
|--------|-------|-------------|
| **VI (ADVI)** | ~5 min | Weekly updates, iteration |
| **MCMC (NUTS)** | ~4-8 hours | Monthly validation, publication |

VI gives a Gaussian approximation to the posterior. MCMC gives exact samples. For weekly rankings, VI is sufficient. For verifying model behaviour or publishing results, use MCMC.

## Interpreting Rankings

Player rankings are the posterior mean of `β_player_try` (or `β_player_kick`) across all posterior samples, on the log-rate scale. Higher = better.

To convert to expected tries per 80 minutes at average team/position:
```python
import numpy as np
expected_tries_per_80 = np.exp(alpha_mean + beta_player_mean)
```

Team rankings are the posterior mean of `γ_team_season` for the current season.

## Further Reading

- [Quick Start](../getting_started/quickstart) — code examples
- [Weekly Workflow](weekly_workflow) — training and update process
- [Predictions Guide](predictions) — match predictions
- [MODEL_EXPLAINED.md](../../MODEL_EXPLAINED.md) — extended conceptual discussion
