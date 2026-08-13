# HTCondor Training Progress Monitoring

Since HTCondor output is redirected to files, standard progress bars don't work well. We've added comprehensive logging that appears in your output files.

## Important Note: MCMC vs VI on HTCondor

**HTCondor is most beneficial for MCMC training**, which is:
- Computationally intensive (hours to days)
- CPU-bound (benefits from parallel chains)
- The gold standard for final model validation

**VI training** is faster and can often run locally, but HTCondor is still useful for:
- Very long runs (>100k iterations)
- When your local machine is busy
- Reproducible, uninterrupted training

## Quick Start

Add the `--verbose` flag to get detailed progress logs:

### For MCMC (recommended for HTCondor)
```bash
python train_model.py \
    --model static \
    --data-dir /path/to/Rugby-Data \
    --method mcmc \
    --mcmc-draws 2000 \
    --mcmc-chains 4 \
    --auto-resume \
    --save-as mcmc_run \
    --verbose
```

### For VI
```bash
python train_model.py \
    --model static \
    --data-dir /path/to/Rugby-Data \
    --method vi \
    --vi-iterations 100000 \
    --checkpoint-every 5000 \
    --auto-resume \
    --save-as vi_run \
    --verbose
```

## What You'll See in the Output Files

### VI with Checkpointing (--checkpoint-every 5000)

```
===============================================================================================
FITTING MODEL (VI)
===============================================================================================
Running VI for 100,000 iterations...
Saving checkpoint every 5,000 iterations
Verbose mode enabled - progress will be logged every 1000 iterations
Starting VI optimization (100,000 iterations)...

[10:23:15] Iteration 5,000/100,000 - ELBO: -125432.45
[10:23:15] Sampling 2000 draws from approximate posterior...
[10:23:17] Saving checkpoint...
[10:23:19] ✓ Checkpoint saved to /home/user/.cache/rugby_ranking/my_run_iter5000

[10:28:32] Iteration 10,000/100,000 - ELBO: -118234.12
[10:28:32] Sampling 2000 draws from approximate posterior...
[10:28:34] Saving checkpoint...
[10:28:36] ✓ Checkpoint saved to /home/user/.cache/rugby_ranking/my_run_iter10000
...
```

### MCMC with Checkpointing

MCMC uses PyMC's built-in progress display plus periodic checkpointing:

```
===============================================================================================
FITTING MODEL (MCMC)
===============================================================================================
Running MCMC: 2000 draws × 4 chains (+ 1000 tuning)...
Saving checkpoint every 500 draws
Verbose mode enabled - checkpoint progress will be logged
MCMC checkpointing enabled: saving every 500 draws per chain
Starting MCMC: 2000 draws × 4 chains (+ 1000 tuning)
Total samples per chain: 3000

Multiprocess sampling (4 chains in 4 jobs)
NUTS: [beta_player_try, beta_player_kick, gamma_team, ...]
Sampling 4 chains for 1_000 tune and 2_000 draw iterations...
Chain 0:  17%|████      | 500/3000 [02:34<12:49,  3.25it/s]
Chain 1:  17%|████      | 500/3000 [02:35<12:50,  3.24it/s]
Chain 2:  17%|████      | 500/3000 [02:36<12:51,  3.23it/s]
Chain 3:  17%|████      | 500/3000 [02:34<12:48,  3.24it/s]

[10:23:15] Draw 500/2000 (chain 0)
[10:23:15] Saving checkpoint...
[10:23:18] ✓ Checkpoint saved to ~/.cache/rugby_ranking/mcmc_run_draw500

Chain 0:  33%|████████  | 1000/3000 [05:08<10:15,  3.25it/s]
...

[10:28:32] Draw 1000/2000 (chain 0)
[10:28:32] Saving checkpoint...
[10:28:35] ✓ Checkpoint saved to ~/.cache/rugby_ranking/mcmc_run_draw1000

...

Chain 0: 100%|██████████| 3000/3000 [15:23<00:00,  3.25it/s]
Chain 1: 100%|██████████| 3000/3000 [15:25<00:00,  3.24it/s]
Chain 2: 100%|██████████| 3000/3000 [15:27<00:00,  3.23it/s]
Chain 3: 100%|██████████| 3000/3000 [15:24<00:00,  3.24it/s]
✓ MCMC sampling complete
```

**Checkpointing:** Saves trace after every N draws (on chain 0), allowing resume from interruptions.

## Understanding the Output

### Progress Indicators

- **Timestamp**: `[10:23:15]` - Time when this event happened
- **Iteration**: `5,000/100,000` - Current iteration out of total
- **ELBO** (VI only): `-125432.45` - Evidence Lower Bound
  - Should generally decrease (become less negative) over time
  - Convergence means ELBO stabilizes

### What to Look For (VI)

**Good signs:**
```
[10:23:15] Iteration 5,000/100,000 - ELBO: -125432.45
[10:28:32] Iteration 10,000/100,000 - ELBO: -118234.12  ← ELBO improved
[10:33:45] Iteration 15,000/100,000 - ELBO: -115678.89  ← Continuing to improve
```

**Warning signs:**
```
[10:23:15] Iteration 5,000/100,000 - ELBO: -125432.45
[10:28:32] Iteration 10,000/100,000 - ELBO: -125432.45  ← Not changing
```

### What to Look For (MCMC)

**Good signs:**
- All chains progress at similar rates
- Sampling rate steady (e.g., 3-4 iterations/sec)
- No divergences reported

**Warning signs:**
- One chain much slower than others (could indicate numerical issues)
- Many divergences (>100)
- Very slow sampling (<1 it/sec)

## Monitoring Your Job

### 1. Check if job is running
```bash
condor_q
```

### 2. Watch the output file in real-time
```bash
tail -f logs/training_<cluster_id>.out
```

### 3. Check for errors
```bash
tail -f logs/training_<cluster_id>.err
```

### 4. Check checkpoint directory
```bash
ls -lth ~/.cache/rugby_ranking/ | head -10
```

## Performance Expectations

### VI (100k iterations, full dataset)
- **Per 5k iterations**: 5-7 minutes
- **Per checkpoint**: 1-2 minutes extra
- **Total**: ~2-3 hours

### MCMC (2000 draws × 4 chains, full dataset)
- **Tuning**: ~20-30 minutes
- **Sampling**: ~1-2 hours
- **Total**: ~2-3 hours
- **Note**: No periodic checkpoints during sampling

### When to Use Each

**Use MCMC on HTCondor when:**
- Final model validation needed
- Publishing results (gold standard)
- Comparing to VI approximation
- Have time for 2-6 hour runs

**Use VI on HTCondor when:**
- Need very long runs (>100k iterations)
- Weekly updates with large dataset
- Exploring different model configurations
- Want faster iteration

**Run VI locally when:**
- Quick experiments (<50k iterations)
- Small datasets (--last-seasons 2)
- Rapid prototyping

## Best Practices

### 1. Choose the right method for HTCondor

**Recommended:** MCMC for thorough, uninterrupted sampling
```bash
--method mcmc --mcmc-draws 2000 --mcmc-chains 4
```

**Alternative:** Long VI runs
```bash
--method vi --vi-iterations 200000 --checkpoint-every 10000
```

### 2. Always use --verbose

So you can see progress in output files (not just progress bars that don't work with file redirection).

### 3. Test locally first

```bash
# Test with small dataset
python train_model.py \
    --model static \
    --data-dir /path/to/Rugby-Data \
    --last-seasons 2 \
    --method mcmc \
    --mcmc-draws 100 \
    --verbose
```

### 4. Monitor regularly

```bash
# Watch latest output
tail -f logs/training_*.out

# Or set up monitoring script
watch -n 10 'condor_q && echo && tail -5 logs/training_*.out'
```

## Troubleshooting

See main [HTCONDOR_TRAINING.md](HTCONDOR_TRAINING.md) guide for detailed troubleshooting.

### Quick Fixes

**No output appearing:**
- Check job is running: `condor_q`
- Check error file: `cat logs/training_*.err`
- Verify paths in .sub file

**VI not progressing:**
- Check ELBO is changing
- Wait longer between checks (5-10 minutes)
- Check error log for numerical issues

**MCMC very slow:**
- This is normal for large models
- Check all chains are progressing
- Look for "divergences" warnings
- Consider smaller dataset for testing

## Summary

**For HTCondor:**
- **Best:** MCMC with --verbose
- **Good:** Long VI runs with --checkpoint-every and --verbose
- **Always:** Test locally first with small data

**Monitoring:**
- Use `tail -f logs/training_*.out` for real-time progress
- With --verbose, see timestamped progress updates
- For VI: watch ELBO improve
- For MCMC: watch sampling progress bars

All logs go to output file, errors to error file, both saved by HTCondor.
