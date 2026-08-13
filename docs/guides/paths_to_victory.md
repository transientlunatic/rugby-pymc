# Paths to Victory

## Overview

The `PathsAnalyzer` determines what results a team needs to finish in a target position in a tournament. It uses Monte Carlo simulations from the fitted model and extracts the key conditions for success.

## Basic Usage

```bash
# Via CLI (Rugby-Data)
rugby analysis paths-to-victory -t "Scotland" -p 2 -c six-nations

# With JSON export
rugby analysis paths-to-victory -t "Scotland" -p 2 -c six-nations --format json -o paths.json

# Critical games
rugby analysis critical-games -c six-nations -s 2025-2026
```

In Python:
```python
from rugby_ranking.model.paths_to_victory import PathsAnalyzer
from rugby_ranking.model.season_predictor import SeasonPredictor

# Run simulations
predictor = SeasonPredictor(model, trace, fixtures, bonus_rules)
analysis = predictor.simulate_season(n_simulations=5000, return_samples=True)

# Extract paths
analyzer = PathsAnalyzer(analysis)
paths = analyzer.analyze(team="Scotland", target_position=2)

# Generate narrative
print(paths.narrative(style="blog"))

# Export for web
paths.export_markdown("scotland_paths.md")
```

## Analysis Modes

The analyzer switches between two modes depending on how many matches remain:

| Remaining games | Mode | Method |
|-----------------|------|--------|
| > ~10 | **MCMC simulation** | Pattern mining via decision trees |
| ≤ ~10 | **Combinatorial** | Enumerate all outcome combinations |

## Output

The analysis provides:
- **Critical games**: which matches most determine the target outcome (by mutual information)
- **Key conditions**: what results the team needs (e.g. "win vs England AND France to lose")
- **Probability**: current probability of reaching the target position
- **Narratives**: human-readable descriptions in blog, detailed, or social-media styles

## See Also

- [PATHS_TO_VICTORY_DESIGN.md](../PATHS_TO_VICTORY_DESIGN.md)
- [PATHS_TO_VICTORY_SUMMARY.md](../PATHS_TO_VICTORY_SUMMARY.md)
- Notebook: `notebooks/07_paths_to_victory_demo.ipynb`
