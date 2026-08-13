#!/usr/bin/env bash
# Run MCMC training on the wiay HTCondor cluster.
#
# This script is executed by HTCondor on a worker node.
# It assumes:
#   - rugby-ranking is installed (pip install -e .) in /data/wiay/conda_envs/pymc-sandbox
#   - Rugby-Data is cloned at $RUGBY_DATA_DIR (default below)
#   - The checkpoint will be written to ~/.cache/rugby_ranking/ (NFS-mounted home)
#
# Submit with: condor_submit scripts/condor/mcmc.sub

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
PYTHON=/data/wiay/conda_envs/pymc-sandbox/bin/python
RUGBY_DATA_DIR="${RUGBY_DATA_DIR:-/data/wiay/daniel/scratch/Rugby-Data/json}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-mcmc_corrected_model}"
N_DRAWS="${N_DRAWS:-2000}"
N_CHAINS="${N_CHAINS:-4}"
# Train on last N years of data (set to "" to use all seasons)
SEASONS="${SEASONS:-}"

# ─── Sanity checks ────────────────────────────────────────────────────────────
echo "=== Rugby Ranking MCMC Job ==="
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "Python:      $PYTHON"
echo "Data dir:    $RUGBY_DATA_DIR"
echo "Checkpoint:  $CHECKPOINT_NAME"
echo "Draws/chain: $N_DRAWS"
echo "Chains:      $N_CHAINS"
echo "NCPUS:       ${_CONDOR_NTHREADS:-$(nproc)}"
echo ""

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    exit 1
fi

if [[ ! -d "$RUGBY_DATA_DIR" ]]; then
    echo "ERROR: Data directory not found: $RUGBY_DATA_DIR"
    echo "Clone Rugby-Data first and set RUGBY_DATA_DIR or update the default above."
    exit 1
fi

# Limit PyTensor/numpy threads to what Condor allocated (avoids oversubscription)
NCPUS="${_CONDOR_NTHREADS:-4}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export PYTENSOR_FLAGS="blas__ldflags="

# ─── Build the command ────────────────────────────────────────────────────────
CMD=(
    "$PYTHON" -m rugby_ranking.cli update
    --method mcmc
    --data-dir "$RUGBY_DATA_DIR"
    --checkpoint "$CHECKPOINT_NAME"
    --n-draws "$N_DRAWS"
    --n-chains "$N_CHAINS"
)

if [[ -n "$SEASONS" ]]; then
    CMD+=(--seasons "$SEASONS")
fi

echo "Running: ${CMD[*]}"
echo ""

exec "${CMD[@]}"
