# Weekly Training Workflow

This guide explains the complete lifecycle: from running MCMC training and uploading a checkpoint, through to weekly variational inference updates used in GitHub Actions.

## Overview

The model is trained in two phases:

1. **Monthly MCMC training** — thorough, slow (~4-8 hours), run locally or on HTCondor. Checkpoint uploaded to a GitHub Release.
2. **Weekly VI updates** — fast (~5 min), run automatically in GitHub Actions, warm-started from the latest MCMC checkpoint.

```
MCMC checkpoint (monthly)
        ↓ upload to GitHub Release
GitHub Actions (weekly)
        ↓ download checkpoint → warm-start VI → save new checkpoint
Dashboard updated
```

## Step 1: Run MCMC Training

### Locally (recommended for first run or validation)

```bash
cd /path/to/rugby-ranking

# Basic training (last 5 seasons, 1000 draws, 4 chains)
make mcmc

# With custom parameters
make mcmc MCMC_DRAWS=2000 MCMC_CHAINS=4 LAST_SEASONS=5

# Check what will be run without executing
make help
```

This calls `train_model.py` with `--method mcmc` and saves a checkpoint to `~/.cache/rugby_ranking/mcmc-YYYY-MM/`.

### On HTCondor (for large-scale training)

```bash
# Generate submit file and optionally submit
make mcmc-condor

# Or submit manually
condor_submit condor_mcmc_mcmc-YYYY-MM.sub

# Monitor progress
condor_q
tail -f condor_logs/mcmc_mcmc-YYYY-MM.log
```

See [HTCONDOR_TRAINING.md](../HTCONDOR_TRAINING.md) for cluster-specific setup.

### Directly via train_model.py

```bash
python train_model.py \
    --model static \
    --method mcmc \
    --data-dir ../Rugby-Data \
    --mcmc-draws 1000 \
    --mcmc-chains 4 \
    --save-as mcmc-2026-03 \
    --last-seasons 5 \
    --verbose
```

## Step 2: Upload Checkpoint to GitHub Release

```bash
# Upload (creates release if it doesn't exist)
make upload-release

# Or train and upload in one step
make mcmc-and-upload

# Check status
make status
```

This creates a GitHub Release tagged `v2026.03` (year.month) and uploads `mcmc-2026-03.tar.gz` containing `trace.nc` (ArviZ NetCDF) and `metadata.pkl`.

The GitHub Actions weekly workflow downloads this checkpoint by looking for the most recent release.

## Step 3: Weekly Automatic Updates (GitHub Actions)

Weekly updates run automatically via `.github/workflows/`. The workflow:

1. Downloads the latest MCMC checkpoint from GitHub Releases
2. Loads the Rugby-Data JSON files
3. Runs VI with warm-start from the MCMC posterior
4. Saves the updated checkpoint
5. Exports dashboard data and deploys

To trigger a manual update:
```bash
gh workflow run update-predictions.yml
```

## Step 4: Manual Weekly Update (Local)

If you need to run the weekly update manually:

```bash
python update_with_new_data.py \
    --data-dir ../Rugby-Data \
    --checkpoint mcmc-2026-03 \
    --verbose
```

Or via CLI:
```bash
rugby-ranking update --data-dir ../Rugby-Data --method vi
```

## Step 5: Ingest Results and Check Calibration

After matches from the previous week have been played, match archived predictions to actual results:

```bash
rugby-ranking ingest-results --data-dir ../Rugby-Data/json
```

This scans all archived predictions that lack a result, matches them to played matches in the Rugby-Data JSON files by team name and date, and records the actual score alongside calibration metrics (score error, outcome correctness, CI coverage).

To review model accuracy over time:

```bash
rugby-ranking calibration
rugby-ranking calibration --competition urc --season 2025-2026
```

**Recommended order each week:**
1. Pull latest Rugby-Data (`cd ../Rugby-Data && git pull`)
2. Run `rugby-ranking upcoming` to generate and archive predictions for the coming week
3. Run `rugby-ranking ingest-results` to score last week's predictions
4. Run the VI update (`make vi` or `rugby-ranking update`)
5. Deploy dashboard (`make deploy` or push to trigger GitHub Actions)

## Checkpoint Management

Checkpoints are stored in `~/.cache/rugby_ranking/`. Each checkpoint is a directory containing:

- `trace.nc` — ArviZ NetCDF with posterior samples
- `metadata.pkl` — Model configuration and data indices

```bash
# List cached checkpoints
make status

# Verify a checkpoint is valid
make verify CHECKPOINT_NAME=mcmc-2026-03

# Compress for manual upload
make compress CHECKPOINT_NAME=mcmc-2026-03
```

## Loading a Checkpoint in Python

```python
from rugby_ranking.model import RugbyModel, ModelFitter, MatchPredictor
from rugby_ranking.model.data import MatchDataset
from pathlib import Path

# Load data (needed to rebuild the model index)
dataset = MatchDataset(Path("../Rugby-Data"))
dataset.load_json_files()
df = dataset.to_dataframe(played_only=True)

# Rebuild model structure
model = RugbyModel()
model.build_joint(df)

# Load checkpoint
fitter = ModelFitter(model)
fitter.load("mcmc-2026-03")

# Now use normally
predictor = MatchPredictor(model, fitter.trace)
rankings = model.get_player_rankings(score_type='tries', top_n=20)
```

## Configuration Reference

| Makefile variable | Default | Description |
|-------------------|---------|-------------|
| `DATA_DIR` | `../Rugby-Data/json` | Path to JSON match files |
| `MCMC_DRAWS` | `1000` | Posterior samples per chain |
| `MCMC_CHAINS` | `4` | Number of parallel chains |
| `LAST_SEASONS` | `5` | Seasons of data to use |
| `MODEL_TYPE` | `static` | `static`, `time-varying`, `minibatch` |
| `CHECKPOINT_NAME` | `mcmc-YYYY-MM` | Name for saved checkpoint |
| `RELEASE_TAG` | `vYYYY.MM` | GitHub Release tag |

## Troubleshooting

**MCMC divergences**: If you see many divergences, reduce the step size or increase `target_accept`:
```bash
python train_model.py --method mcmc --mcmc-target-accept 0.9
```

**Out of memory**: Reduce chains or use minibatch model:
```bash
make mcmc MODEL_TYPE=minibatch MCMC_CHAINS=2
```

**Checkpoint not found**: Check `make status` and ensure the release exists:
```bash
gh release list --limit 5
```

**VI not converging**: Increase iterations or check data quality:
```bash
python train_model.py --method vi --vi-iterations 100000
```
