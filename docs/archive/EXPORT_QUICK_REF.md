# 🎯 Dashboard Export - Quick Reference

## New CLI Command

```bash
rugby-ranking export --data-dir ../Rugby-Data [OPTIONS]
```

## Quick Examples

### 1️⃣ Export with Current Model
```bash
rugby-ranking export --data-dir ../Rugby-Data --checkpoint international-mini5
```

### 2️⃣ Export Latest 5 Seasons
```bash
rugby-ranking export --data-dir ../Rugby-Data --checkpoint international-mini5 --seasons 5
```

### 3️⃣ Train New Model and Export
```bash
rugby-ranking export --data-dir ../Rugby-Data --seasons 3
```

### 4️⃣ Custom Output Location
```bash
rugby-ranking export \
  --data-dir ../Rugby-Data \
  --checkpoint international-mini5 \
  --output-dir /tmp/dashboard-data
```

## Arguments

| Argument | Required | Default | Example |
|----------|----------|---------|---------|
| `--data-dir` | ✅ | - | `../Rugby-Data` |
| `--checkpoint` | ❌ | None | `international-mini5` |
| `--output-dir` | ❌ | `dashboard/data` | `./exports` |
| `--seasons` | ❌ | 3 | `5` |

## One-Liner Deploy

```bash
rugby-ranking export --data-dir ../Rugby-Data --checkpoint international-mini5 && \
python -m http.server 8000 --directory dashboard
```

Then open: **http://localhost:8000** 🚀

## Help

```bash
rugby-ranking export --help
```

## Checkpoint Options

- `international-mini5` - Your recent training
- `time_model_v1` - Existing checkpoint
- None (omit `--checkpoint`) - Trains new model
- Any checkpoint in `models/checkpoints/`

## Output Files (11 total)

```
dashboard/data/
├── team_offense.json
├── team_defense.json
├── player_rankings.json
├── match_stats.json
├── team_stats.json
├── summary.json
├── team_strength_series.json       ← NEW
├── team_finish_positions.json      ← NEW
├── upcoming_predictions.json       ← NEW
├── paths_to_victory.json           ← NEW
└── squad_depth.json                ← NEW
```

## Time Estimates

| Seasons | Time |
|---------|------|
| 1 | 2-3 min |
| 3 | 5-8 min |
| 5 | 10-15 min |

## Common Workflows

### Daily Export
```bash
# Add to crontab
0 2 * * * rugby-ranking export --data-dir ../Rugby-Data --checkpoint international-mini5
```

### Development (Quick)
```bash
rugby-ranking export --data-dir ../Rugby-Data --seasons 1
```

### Production (Full)
```bash
rugby-ranking export --data-dir ../Rugby-Data --checkpoint international-mini5 --seasons 5 --output-dir /var/www/dashboard/data
```

## Changes Made

✅ Dashboard export integrated into main CLI  
✅ Full argument support (checkpoint, seasons, output dir)  
✅ No more editing Python scripts  
✅ Backward compatible (original script still works)  
✅ Production ready  

---

**That's it!** Use `rugby-ranking export` to generate all dashboard data. 🎉
