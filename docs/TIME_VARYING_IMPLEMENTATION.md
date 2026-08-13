# Time-Varying Effects Implementation

## Overview

Implemented within-season trend effects to capture form changes over the course of a season. This allows the model to distinguish between:
- **Base ability**: A player's/team's underlying skill level for the season
- **Form trend**: Whether they're improving, declining, or maintaining form as the season progresses

## Model Structure

### Original Model (Static)
```
log(λ[i,m]) = α + β_player[i] + γ_team[j,season] + ...
```
Each player has one fixed effect per season.

### New Model (Time-Varying)
```
log(λ[i,m]) = α
            + (β_player_base[i,season] + β_player_trend[i,season] * t_m)
            + (γ_team_base[j,season] + γ_team_trend[j,season] * t_m)
            + ...
```

Where:
- `t_m` = normalized time within season (0 = season start, 1 = season end)
- `β_player_base` = player's base ability for this season
- `β_player_trend` = rate of improvement/decline within season
- `γ_team_base` = team's base strength for this season
- `γ_team_trend` = team's form trajectory within season

## Implementation Details

### 1. Configuration

Added to `ModelConfig`:
```python
time_varying_effects: bool = False  # Enable within-season trends
player_trend_sd: float = 0.1       # Prior SD for player trend slopes
team_trend_sd: float = 0.1          # Prior SD for team trend slopes
season_evolution_sd: float = 0.2    # Prior SD for season-to-season changes
```

### 2. Data Preparation

Modified `_prepare_data()` to compute `season_progress`:
- Gets season start/end dates for each season
- Computes `days_into_season` and `season_duration`
- Normalizes to `season_progress ∈ [0, 1]`
- Only computed when `time_varying_effects=True`

### 3. Model Method

Added `build_joint_time_varying()` method:
- Separate base and trend parameters for players and teams
- Compatible with separate kicking/try-scoring effects
- Player effects are 2D arrays: `(n_players, n_seasons)`
- Team effects are per team-season

### 4. Hierarchical Structure

**Player Effects** (per season):
```python
# Try-scoring
beta_player_try_base[i, s] ~ Normal(0, sigma_player_try_base)
beta_player_try_trend[i, s] ~ Normal(0, sigma_player_try_trend)

# Kicking
beta_player_kick_base[i, s] ~ Normal(0, sigma_player_kick_base)
beta_player_kick_trend[i, s] ~ Normal(0, sigma_player_kick_trend)
```

**Team Effects** (per team-season):
```python
gamma_team_base[j, s] ~ Normal(0, sigma_team_base)
gamma_team_trend[j, s] ~ Normal(0, sigma_team_trend)
```

**Loading Factors** (per score type):
```python
# Scale effects differently for tries vs penalties vs conversions
lambda_player_try[score_type] ~ HalfNormal(0.5)
lambda_player_kick[score_type] ~ HalfNormal(0.5)
lambda_team[score_type] ~ HalfNormal(0.5)
```

## Usage

### Basic Usage

```python
from rugby_ranking.model.core import RugbyModel, ModelConfig
from rugby_ranking.model.data import MatchDataset

# Load data
dataset = MatchDataset(data_dir)
dataset.load_json_files()
df = dataset.to_dataframe(played_only=True)

# Configure with time-varying effects
config = ModelConfig(
    score_types=("tries", "penalties", "conversions", "drop_goals"),
    separate_kicking_effect=True,
    time_varying_effects=True,  # Enable time-varying
    player_trend_sd=0.1,
    team_trend_sd=0.1,
)

# Build model
model = RugbyModel(config)
pymc_model = model.build_joint_time_varying(df)

# Fit (VI or MCMC)
fitter = ModelFitter(model, inference_config)
trace = fitter.fit_vi(n_samples=2000)
```

### Comparison with Static Model

```python
# Static model
config_static = ModelConfig(time_varying_effects=False)
model_static = RugbyModel(config_static)
pymc_static = model_static.build_joint(df)

# Time-varying model
config_tv = ModelConfig(time_varying_effects=True)
model_tv = RugbyModel(config_tv)
pymc_tv = model_tv.build_joint_time_varying(df)
```

## Model Complexity

**Test Results** (3 seasons, 2 score types):
- **Static model**: 27 random variables
- **Time-varying model**: 37 random variables
- **Additional**: +10 variables (~37% increase)

**Parameter Shapes**:
- `beta_player_try_base_raw`: (4657 players, 3 seasons)
- `beta_player_try_trend_raw`: (4657 players, 3 seasons)
- `gamma_team_base_raw`: (169 team-seasons,)
- `gamma_team_trend_raw`: (169 team-seasons,)

## Interpretation

### Positive Trend
- **Player**: Improving form as season progresses (fitness, confidence, learning system)
- **Team**: Building momentum (gel time, tactics refined, depth improves)

### Negative Trend
- **Player**: Declining form (fatigue, injury, aging)
- **Team**: Fading (fixture congestion, injuries accumulate, squad depth issues)

### Zero Trend
- Consistent performance throughout season
- Effect_mean at season start ≈ Effect_mean at season end

## Example Interpretations

**Player A:**
- `base = 0.5, trend = -0.2`
- Early season: Strong (0.5)
- Mid-season: Declining (0.4)
- Late season: Weaker (0.3)
- **Interpretation**: Started hot but faded (injury/fatigue?)

**Player B:**
- `base = 0.3, trend = +0.3`
- Early season: Weak (0.3)
- Mid-season: Improving (0.45)
- Late season: Strong (0.6)
- **Interpretation**: Took time to adapt or recovered from injury

**Team C:**
- `base = 0.8, trend = +0.2`
- **Interpretation**: Strong team that got even better (depth, momentum)

## Visualization Ideas

1. **Career Trajectories**: Plot player base + trend across multiple seasons
2. **Form Lines**: Show within-season trend for top players
3. **Team Momentum**: Compare early vs late season team strengths
4. **Prediction Weights**: Show how recent matches weigh more with trends

## Future Enhancements

### Already Implemented
- ✅ Within-season linear trends
- ✅ Separate try-scoring vs kicking effects
- ✅ Season progress computation
- ✅ Compatible with defensive effects

### Potential Extensions
- [ ] Non-linear trends (splines, quadratic)
- [ ] Match-by-match updates (like Elo)
- [ ] Career-long trajectories (random walk across seasons)
- [ ] Age-based effects (when DOB data available)
- [ ] Fixture-difficulty adjustments
- [ ] Rest period effects (days since last match)

## Testing

Run test suite:
```bash
python test_time_varying.py
```

**Tests**:
1. ✅ Static model builds (baseline)
2. ✅ Time-varying model builds
3. ✅ All expected variables present
4. ✅ Season progress data valid [0, 1]
5. ✅ Model complexity comparison

## Files Modified

- `rugby_ranking/model/core.py`:
  - Added `time_varying_effects` config options
  - Updated `_prepare_data()` to compute season progress
  - Added `build_joint_time_varying()` method (~250 lines)

## Files Created

- `test_time_varying.py`: Test suite for time-varying model
- `TIME_VARYING_IMPLEMENTATION.md`: This documentation

## Performance Considerations

**Computational Cost**:
- ~37% more parameters than static model
- Still amenable to VI (tested successfully)
- MCMC may take longer due to increased dimensionality

**Recommendations**:
- Use VI for weekly updates (faster)
- Use MCMC for validation (more accurate)
- Consider reducing to single score type for speed

**Data Requirements**:
- Needs match dates for season progress
- Works best with full season data
- Partial seasons supported (progress clipped to [0, 1])

## Notes

- Currently requires `separate_kicking_effect=True`
- Unified player effect version not yet implemented
- Trends are linear (future: could add splines/GP)
- Season-to-season evolution not yet implemented (future: random walk)
