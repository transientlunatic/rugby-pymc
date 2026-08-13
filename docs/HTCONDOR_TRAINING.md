# Training on HTCondor with Automatic Checkpointing

The training script now supports periodic checkpointing and automatic resume, making it resilient to interruptions on HTCondor clusters.

## Quick Start

### 1. Basic Training with Checkpoints

```bash
# Train with checkpoints every 5000 iterations
python train_model.py \
    --model static \
    --data-dir /path/to/Rugby-Data \
    --method vi \
    --vi-iterations 100000 \
    --checkpoint-every 5000 \
    --auto-resume \
    --save-as my_training_run
```

### 2. Submit to HTCondor

Edit [submit_training.sub](submit_training.sub) to set your data path and parameters, then:

```bash
# Create logs directory
mkdir -p logs

# Submit job
condor_submit submit_training.sub
```

### 3. Monitor Progress

```bash
# Watch job status
condor_q

# Check output
tail -f logs/training_*.out

# View checkpoints
ls -lh ~/.cache/rugby_ranking/
```

## How It Works

### Periodic Checkpointing

- The `--checkpoint-every N` flag saves progress every N iterations
- Checkpoints include:
  - Current posterior approximation
  - Model indices (player/team mappings)
  - Training metadata
- Checkpoint names: `{save-as}_iter{iteration}`
- Example: `my_run_iter5000`, `my_run_iter10000`, etc.

### Auto-Resume

- The `--auto-resume` flag automatically finds the latest checkpoint
- If training is interrupted, simply resubmit the same command
- Training continues from the last saved checkpoint
- Particularly useful on HTCondor where jobs can be evicted

### Manual Resume

You can also manually specify which checkpoint to resume from:

```bash
python train_model.py \
    --resume my_training_run_iter15000 \
    --data-dir /path/to/Rugby-Data
```

## HTCondor Configuration

The [submit_training.sub](submit_training.sub) file includes:

1. **Graceful removal**: HTCondor sends SIGTERM before eviction, giving time to save
2. **Auto-retry**: Jobs automatically restart after eviction (up to 10 times)
3. **Resource requests**: Adjust CPU/memory based on your needs

### Important HTCondor Settings

```
+WantGracefulRemoval = True          # Request graceful eviction
on_exit_hold = (ExitCode =?= UNDEFINED)  # Hold on eviction
periodic_release = ...               # Auto-retry evicted jobs
```

## Best Practices

### Checkpoint Frequency

Choose `--checkpoint-every` based on:
- **Too frequent** (< 1000): Overhead from saving slows training
- **Too rare** (> 20000): More work lost if interrupted
- **Recommended**: 5000-10000 for VI, 100-200 for MCMC

### Storage Management

Checkpoints are saved to `~/.cache/rugby_ranking/`. Each checkpoint includes:
- `trace.nc`: Posterior samples (~50-200MB)
- `metadata.pkl`: Model indices (~1-10MB)

To clean up old checkpoints:

```bash
# Remove all checkpoints for a run
rm -rf ~/.cache/rugby_ranking/my_training_run_iter*

# Keep only the latest checkpoint
cd ~/.cache/rugby_ranking
ls -t my_training_run_iter* | tail -n +2 | xargs rm -rf
```

### HTCondor-Specific Tips

1. **Test locally first**: Run a short training locally to verify it works
2. **Use absolute paths**: Specify full paths for --data-dir
3. **Monitor disk usage**: Ensure checkpoint directory is not in /tmp
4. **Check logs**: Always review logs/training_*.err for issues

## Example: Long Training Run

**VI with checkpointing:**
```bash
python train_model.py \
    --model time-varying \
    --data-dir /home/user/Rugby-Data \
    --method vi \
    --vi-iterations 200000 \
    --checkpoint-every 10000 \
    --auto-resume \
    --save-as production_vi_$(date +%Y%m%d) \
    --verbose
```

**MCMC with checkpointing:**
```bash
python train_model.py \
    --model static \
    --data-dir /home/user/Rugby-Data \
    --method mcmc \
    --mcmc-draws 2000 \
    --mcmc-chains 4 \
    --checkpoint-every 500 \
    --auto-resume \
    --save-as production_mcmc_$(date +%Y%m%d) \
    --verbose
```

If a job gets interrupted:
1. HTCondor restarts the job
2. Script finds latest checkpoint (e.g., `_draw1000` or `_iter120000`)
3. Training resumes from that checkpoint
4. Minimal work lost!

## Troubleshooting

### Job keeps restarting from scratch

Check that:
- `--auto-resume` flag is present
- `--save-as` name is consistent across runs
- `~/.cache/rugby_ranking/` is accessible from compute nodes

### Checkpoints filling up disk

Clean up intermediate checkpoints after successful completion:

```bash
# Keep only final checkpoint
cd ~/.cache/rugby_ranking
ls -t my_run_iter* | tail -n +2 | xargs rm -rf

# Or keep only every 5th checkpoint
python -c "
import os
checkpoints = sorted([d for d in os.listdir('.') if d.startswith('my_run_iter')],
                     key=lambda x: int(x.split('_iter')[1]))
for i, cp in enumerate(checkpoints):
    if i % 5 != 0 and i != len(checkpoints) - 1:
        print(f'rm -rf {cp}')
" | bash
```

### VI approximation not improving after resume

This is expected - VI may converge differently when resumed. If concerned:
- Check the ELBO (loss) values in logs
- Compare final posterior to a fresh run
- Use `--warm-start` for VI continuation
