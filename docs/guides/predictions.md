# Match Predictions

## Overview

The `MatchPredictor` generates probabilistic match predictions using the fitted posterior. It supports:
- **Team-only predictions**: when lineups are unknown (higher uncertainty)
- **Full-lineup predictions**: when team sheets are available (lower uncertainty)

Predictions return full posterior distributions, not just point estimates.

## Basic Usage

```python
from rugby_ranking.model import MatchPredictor

# After fitting the model
predictor = MatchPredictor(model, trace)

# Team-only prediction (season is required)
pred = predictor.predict_teams_only("Leinster", "Munster", season="2025-2026")
print(pred.summary())
```

## Team-Only Predictions

Use when team sheets are not available (e.g. predicting next week's fixtures):

```python
pred = predictor.predict_teams_only(
    home_team="Leinster",
    away_team="Munster",
    season="2025-2026",   # Required; if team/season unseen, falls back to team's most recent season
    n_samples=1000,       # Posterior samples to draw
)

# Win probabilities (direct attributes)
print(f"Home win probability: {pred.home_win_prob:.1%}")
print(f"Draw probability: {pred.draw_prob:.1%}")
print(f"Away win probability: {pred.away_win_prob:.1%}")

# Score predictions (nested ScorePrediction objects)
print(f"Expected home score: {pred.home.mean:.1f} ± {pred.home.std:.1f}")
print(f"Expected away score: {pred.away.mean:.1f} ± {pred.away.std:.1f}")
print(f"Home 90% CI: [{pred.home.ci_lower:.0f}, {pred.home.ci_upper:.0f}]")
```

## Full-Lineup Predictions

When team sheets are published, full-lineup predictions have lower uncertainty:

```python
home_lineup = {
    1: "Andrew Porter", 2: "Dan Sheehan", 3: "Tadhg Furlong",
    4: "Joe McCarthy", 5: "James Ryan", 6: "Caelan Doris",
    7: "Josh van der Flier", 8: "Jack Conan", 9: "Jamison Gibson-Park",
    10: "Ross Byrne", 11: "James Lowe", 12: "Robbie Henshaw",
    13: "Garry Ringrose", 14: "Jordan Larmour", 15: "Hugo Keenan",
}
away_lineup = {
    1: "Jeremy Loughman", 2: "Niall Scannell", 3: "Stephen Archer",
    # ... positions 4-15
}

pred = predictor.predict_full_lineup(
    "Leinster", "Munster",
    home_lineup, away_lineup,
    season="2025-2026",
)
print(pred.summary())
```

Missing players (positions not in the dict) fall back to team-average effects.

## Upcoming Matches

Predict all matches in the next N days:

```bash
# CLI: matches in next 7 days
rugby-ranking upcoming

# Specify days and competition
rugby-ranking upcoming --days 14 --competition urc --season 2025-2026
```

In Python:
```python
from rugby_ranking.model.data import MatchDataset

# Get upcoming fixtures
dataset = MatchDataset(Path("../Rugby-Data"))
dataset.load_json_files()
upcoming = dataset.get_upcoming_matches(days=7)

for match in upcoming:
    pred = predictor.predict_teams_only(
        match.home_team, match.away_team, season=match.season
    )
    print(f"{match.home_team} vs {match.away_team}: {pred.home_win_prob:.0%} / {pred.away_win_prob:.0%}")
```

## Prediction Output

`MatchPrediction` contains:

| Attribute | Type | Description |
|-----------|------|-------------|
| `home` | `ScorePrediction` | Home team score distribution |
| `away` | `ScorePrediction` | Away team score distribution |
| `home_win_prob` | `float` | P(home team wins) |
| `away_win_prob` | `float` | P(away team wins) |
| `draw_prob` | `float` | P(draw) |
| `predicted_margin` | `float` | Expected home − away margin |
| `margin_std` | `float` | Standard deviation of margin |

`ScorePrediction` (accessed as `pred.home` / `pred.away`) contains:

| Attribute | Description |
|-----------|-------------|
| `mean` | Expected score |
| `std` | Standard deviation |
| `median` | Median score |
| `ci_lower` | 5th percentile |
| `ci_upper` | 95th percentile |
| `samples` | Full posterior samples (if requested) |

## Fallback Behaviour

If a team or season has not been seen in training:
- **Unseen season**: falls back to the team's most recent season
- **Unseen team**: uses the model prior (average team)
- **Unseen player**: uses position average

This ensures predictions can always be generated, with wider uncertainty intervals when data is sparse.

## Prediction Archiving

Predictions are automatically archived to `~/.cache/rugby_ranking/predictions/` whenever `rugby-ranking predict` or `rugby-ranking upcoming` is run. Each archived prediction records:

- The full probability distribution (mean, std, CI, win probabilities)
- Which model checkpoint was used
- The match metadata (competition, season, date, teams)

To opt out for a one-off prediction:

```bash
rugby-ranking predict --home Leinster --away Munster --checkpoint mcmc-2026-03 --no-archive
rugby-ranking upcoming --no-archive
```

### Ingesting Results

After matches are played, update the archive with actual scores from Rugby-Data:

```bash
rugby-ranking ingest-results --data-dir ../Rugby-Data/json

# Limit to a date range
rugby-ranking ingest-results --data-dir ../Rugby-Data/json --date-from 2026-01-01 --date-to 2026-03-21

# Preview what would be updated (no changes written)
rugby-ranking ingest-results --data-dir ../Rugby-Data/json --dry-run
```

Matching uses normalised team names and date proximity (±1 day for timezone issues), with competition as a tiebreaker.

### Calibration Report

Once predictions have been matched with results:

```bash
# Overall report
rugby-ranking calibration

# Filter by competition or season
rugby-ranking calibration --competition six-nations
rugby-ranking calibration --competition urc --season 2025-2026
```

Output includes:

| Metric | Description |
|--------|-------------|
| Outcome accuracy | Fraction of correct winner predictions |
| Brier score | Probabilistic accuracy (lower = better; 0.333 = random) |
| Home / Away MAE | Mean absolute error on predicted scores |
| Score bias | Signed mean error (positive = under-predicting) |
| 90% CI coverage | Fraction of actual scores within the predicted interval (expect ~90%) |

### Python API

```python
from rugby_ranking.model.prediction_archive import PredictionArchiver, MatchMetadata, ActualResult

archiver = PredictionArchiver()

# Archive a prediction
prediction_id = archiver.archive_prediction(
    prediction=pred,
    match_metadata=MatchMetadata(
        match_id="urc_2026-03-21_leinster-vs-munster",
        competition="urc",
        season="2025-2026",
        date=match_date,
        home_team="Leinster",
        away_team="Munster",
    ),
    model_checkpoint="mcmc-2026-03",
    prediction_type="teams_only",
)

# Retrieve archived predictions
predictions = archiver.get_predictions(competition="urc", season="2025-2026")
scored = archiver.get_predictions(has_result=True)

# Aggregate calibration metrics
report = archiver.calibration_report(competition="urc")
print(f"Outcome accuracy: {report['outcome_accuracy']:.1%}")
print(f"Brier score: {report['brier_score']:.4f}")
```

## See Also

- [Squad Analysis](squad_analysis) — pre-match squad strength assessment
- [Quick Start](../getting_started/quickstart) — code examples
