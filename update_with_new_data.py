#!/usr/bin/env python3
"""
Update an existing model with new data using warm-start.

This script loads a trained model and continues training with new data,
using warm-start to preserve learned parameters for existing players/teams
while initializing new players/teams from the prior.

Usage:
    # Update weekly with new Six Nations results
    python update_with_new_data.py \
      --checkpoint my_model \
      --new-data-dir ../Rugby-Data \
      --pattern "six_nations_2025*.json" \
      --iterations 5000 \
      --save-as my_model_updated

    # Update monthly with all new internationals
    python update_with_new_data.py \
      --checkpoint international_model \
      --new-data-dir ../Rugby-Data \
      --iterations 10000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rugby_ranking.model.data import MatchDataset
from rugby_ranking.model.core import RugbyModel, ModelConfig
from rugby_ranking.model.inference import ModelFitter


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Update model with new data using warm-start',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Name of the checkpoint to update')
    parser.add_argument('--new-data-dir', type=Path, required=True,
                       help='Directory containing new match data')
    parser.add_argument('--pattern', type=str, default='*.json',
                       help='File pattern for new data (default: *.json)')
    parser.add_argument('--iterations', type=int, default=10000,
                       help='VI iterations for update (default: 10000)')
    parser.add_argument('--samples', type=int, default=500,
                       help='Samples to draw after update (default: 500)')
    parser.add_argument('--sample-batch-size', type=int, default=100,
                       help='Sample batch size (default: 100)')
    parser.add_argument('--save-as', type=str,
                       help='Save updated model with this name (default: same as checkpoint)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')

    return parser.parse_args()


def main():
    """Main update pipeline."""
    args = parse_args()

    try:
        if not args.quiet:
            print("=" * 70)
            print("LOADING CHECKPOINT")
            print("=" * 70)

        # Create dummy model (will be populated by load)
        config = ModelConfig()
        model = RugbyModel(config)

        # Load existing checkpoint
        fitter = ModelFitter.load(args.checkpoint, model)

        if not args.quiet:
            print(f"✓ Loaded checkpoint: {args.checkpoint}")
            print(f"  Existing players: {len(model._player_ids):,}")
            print(f"  Existing teams: {len(model._team_ids)}")

        if fitter._vi_approx is None:
            print("\n❌ ERROR: Checkpoint has no VI approximation.", file=sys.stderr)
            print("Only VI-trained models can be updated with warm-start.", file=sys.stderr)
            return 1

        # Load new data
        if not args.quiet:
            print("\n" + "=" * 70)
            print("LOADING NEW DATA")
            print("=" * 70)

        dataset = MatchDataset(args.new_data_dir, fuzzy_match_names=False)
        dataset.load_json_files(pattern=args.pattern)
        
        df = dataset.to_dataframe(played_only=True)
        df = df[df['position'].between(1, 23)].copy()

        if not args.quiet:
            print(f"New observations: {len(df):,}")
            print(f"New/existing players: {df['player_name'].nunique():,}")
            print(f"Teams: {df['team'].nunique()}")
            print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

        # Rebuild model with new data
        if not args.quiet:
            print("\n" + "=" * 70)
            print("REBUILDING MODEL WITH NEW DATA")
            print("=" * 70)

        # Use same config as original model
        config = fitter.config
        model = RugbyModel(config)
        
        if config.time_varying_effects:
            model.build_joint_time_varying(df)
        else:
            model.build_joint(df)

        # Count new players/teams
        old_player_count = len(fitter.rugby_model._player_ids)
        new_player_count = len(model._player_ids)
        old_team_count = len(fitter.rugby_model._team_ids)
        new_team_count = len(model._team_ids)

        if not args.quiet:
            print(f"Updated model built!")
            print(f"  Players: {new_player_count:,} ({new_player_count - old_player_count:+,} new)")
            print(f"  Teams: {new_team_count} ({new_team_count - old_team_count:+,} new)")

        # Update fitter with new model and keep old approximation for warm-start
        old_approx = fitter._vi_approx
        fitter.rugby_model = model
        fitter._vi_approx = old_approx  # Preserve for warm-start

        # Continue training with warm-start
        if not args.quiet:
            print("\n" + "=" * 70)
            print("UPDATING WITH WARM-START")
            print("=" * 70)
            print(f"Running {args.iterations:,} VI iterations...")
            print("Warm-start will initialize:")
            print("  - Existing players/teams: From previous posterior")
            print("  - New players/teams: From prior distribution")

        trace = fitter.fit_vi(
            warm_start=True,
            n_samples=args.samples,
            sample_batch_size=args.sample_batch_size,
            verbose=not args.quiet,
        )

        # Save updated model
        if not args.quiet:
            print("\n" + "=" * 70)
            print("SAVING UPDATED MODEL")
            print("=" * 70)

        checkpoint_name = args.save_as or args.checkpoint
        path = fitter.save(checkpoint_name)

        if not args.quiet:
            print(f"✓ Saved to: {path}")
            print(f"\nUpdated model ready!")
            print(f"  - Load with: ModelFitter.load('{checkpoint_name}', model)")
            print(f"  - Use for predictions immediately")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
