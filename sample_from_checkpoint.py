#!/usr/bin/env python3
"""
Draw additional samples from a saved VI approximation.

This script allows you to draw more posterior samples from a previously
trained VI model without re-running the optimization. Useful when you
want more samples for better uncertainty estimates.

LIMITATION: this only works within the same process that ran fit_vi() --
i.e. you didn't just run `ModelFitter.load(...)` in a fresh script/session.
A live PyMC Approximation holds pytensor graph nodes bound to a specific
model context and cannot be pickled/reloaded; a checkpoint loaded from disk
only carries the numeric parameter values (enough to warm-start a *new*
fit via `ModelFitter.fit_vi(warm_start=True)`, not enough to draw more
samples from the *existing* one). This script currently cannot do that --
see the checkpoint-load error below if you try.

Usage:
    # Draw 5000 samples in batches of 100
    python sample_from_checkpoint.py --checkpoint international-mini5 --samples 5000 --batch-size 100

    # Replace the stored trace with new samples
    python sample_from_checkpoint.py --checkpoint my_model --samples 10000 --batch-size 100 --replace

    # Save to a new checkpoint
    python sample_from_checkpoint.py --checkpoint my_model --samples 5000 --batch-size 100 --save-as my_model_5k_samples
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rugby_ranking.model.core import RugbyModel, ModelConfig
from rugby_ranking.model.inference import ModelFitter, _LoadedVIApprox


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Draw additional samples from a saved VI approximation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Name of the checkpoint to load')
    parser.add_argument('--samples', type=int, default=2000,
                       help='Number of samples to draw (default: 2000)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Sample batch size to avoid OOM (default: 100)')
    parser.add_argument('--replace', action='store_true',
                       help='Replace the existing trace in the checkpoint')
    parser.add_argument('--save-as', type=str,
                       help='Save to a new checkpoint with this name')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')

    return parser.parse_args()


def main():
    """Main sampling pipeline."""
    args = parse_args()

    try:
        if not args.quiet:
            print("=" * 70)
            print("LOADING CHECKPOINT")
            print("=" * 70)

        # Create a dummy model (will be populated by load)
        config = ModelConfig()  # Default config, will be overridden
        model = RugbyModel(config)

        # Load checkpoint
        fitter = ModelFitter.load(args.checkpoint, model)

        if not args.quiet:
            print(f"✓ Loaded checkpoint: {args.checkpoint}")
            if fitter.trace is not None:
                n_existing = len(fitter.trace.posterior.draw)
                print(f"  Existing samples: {n_existing}")

        if fitter._vi_approx is None:
            print("\n❌ ERROR: This checkpoint does not contain a VI approximation.", file=sys.stderr)
            print("Only checkpoints from VI training can draw additional samples.", file=sys.stderr)
            print(f"Checkpoint was fit using: {fitter._fit_method}", file=sys.stderr)
            return 1

        if isinstance(fitter._vi_approx, _LoadedVIApprox):
            print(
                "\n❌ ERROR: This checkpoint's VI approximation was loaded from disk, "
                "which only carries the fitted parameter *values* -- not a live "
                "PyMC Approximation bound to a model, which is what drawing more "
                "samples needs (and can't survive a pickle round-trip; see this "
                "script's module docstring).",
                file=sys.stderr,
            )
            print(
                "This checkpoint's saved trace.nc already has the samples drawn "
                "during the original fit. To get more, warm-start a new fit "
                "instead: ModelFitter.fit_vi(warm_start=True) after loading this "
                "checkpoint as the starting point (see update_with_new_data.py).",
                file=sys.stderr,
            )
            return 1

        if not args.quiet:
            print("\n" + "=" * 70)
            print("DRAWING SAMPLES")
            print("=" * 70)
            print(f"Samples: {args.samples:,}")
            print(f"Batch size: {args.batch_size}")

        # Draw new samples
        new_trace = fitter.sample_from_vi(
            n_samples=args.samples,
            sample_batch_size=args.batch_size,
            replace_trace=args.replace,
            verbose=not args.quiet,
        )

        if not args.quiet:
            print("\n" + "=" * 70)
            print("SAVING")
            print("=" * 70)

        # Save if requested
        if args.save_as:
            checkpoint_name = args.save_as
        elif args.replace:
            checkpoint_name = args.checkpoint
        else:
            # Don't save by default unless --replace or --save-as specified
            if not args.quiet:
                print("Samples drawn but not saved (use --replace or --save-as to save)")
            return 0

        path = fitter.save(checkpoint_name)

        if not args.quiet:
            print(f"✓ Saved to: {path}")
            print(f"\nLoad with: ModelFitter.load('{checkpoint_name}', model)")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
