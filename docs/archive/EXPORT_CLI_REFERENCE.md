# Dashboard Export via CLI

The dashboard export functionality is now integrated into the main CLI with full argument support.

## Usage

### Basic Export (uses default checkpoint)
```bash
rugby-ranking export --data-dir ../Rugby-Data
```

### Export with Specific Checkpoint
```bash
rugby-ranking export \
  --data-dir ../Rugby-Data \
  --checkpoint international-mini5
```

### Export with Custom Output Directory
```bash
rugby-ranking export \
  --data-dir ../Rugby-Data \
  --checkpoint international-mini5 \
  --output-dir /tmp/dashboard-data
```

### Export More Seasons
```bash
rugby-ranking export \
  --data-dir ../Rugby-Data \
  --checkpoint international-mini5 \
  --seasons 5
```

### Train New Model (no checkpoint specified)
```bash
rugby-ranking export \
  --data-dir ../Rugby-Data \
  --seasons 3
```

## Available Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--data-dir` | Path | ✅ Yes | - | Path to Rugby-Data repository |
| `--checkpoint` | String | ❌ No | None | Model checkpoint name (None = train new) |
| `--output-dir` | Path | ❌ No | `dashboard/data` | Output directory for JSON files |
| `--seasons` | Integer | ❌ No | 3 | Number of recent seasons to export |

## Examples

### Quick Export for Dashboard
```bash
rugby-ranking export --data-dir ../Rugby-Data --checkpoint international-mini5
```

### Full Export (5 seasons, custom output)
```bash
rugby-ranking export \
  --data-dir ../Rugby-Data \
  --checkpoint international-mini5 \
  --output-dir ./exports/dashboard-2025 \
  --seasons 5
```

### Train and Export Fresh Model
```bash
rugby-ranking export \
  --data-dir ../Rugby-Data \
  --seasons 3
```

## Output

All commands generate the same 11 JSON files:
```
dashboard/data/
├── team_offense.json
├── team_defense.json
├── player_rankings.json
├── match_stats.json
├── team_stats.json
├── summary.json
├── team_strength_series.json
├── team_finish_positions.json
├── upcoming_predictions.json
├── paths_to_victory.json
└── squad_depth.json
```

## Integration with Dashboard

After exporting, serve the dashboard:
```bash
# Run in one command
rugby-ranking export --data-dir ../Rugby-Data --checkpoint international-mini5 && \
python -m http.server 8000 --directory dashboard

# Or separately
rugby-ranking export --data-dir ../Rugby-Data --checkpoint international-mini5
python -m http.server 8000 --directory dashboard
```

Then open: **http://localhost:8000**
