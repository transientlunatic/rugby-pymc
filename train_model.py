#!/usr/bin/env python3
"""
Unified training script for rugby ranking models.

Usage:
    # Train static model
    python train_model.py --model static --data-dir ../Rugby-Data

    # Train time-varying model
    python train_model.py --model time-varying --data-dir ../Rugby-Data

    # Resume from checkpoint
    python train_model.py --model static --resume joint_model_v2

    # Train on recent seasons only
    python train_model.py --model static --data-dir ../Rugby-Data --last-seasons 3
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rugby_ranking.model.data import MatchDataset
from rugby_ranking.model.core import RugbyModel, ModelConfig
from rugby_ranking.model.inference import ModelFitter, InferenceConfig
from rugby_ranking.model.data_validation import clean_kicking_data, detect_kicking_anomalies


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train rugby ranking model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train static model with VI
  %(prog)s --model static --data-dir ../Rugby-Data --save-as my_model

  # Train time-varying model with MCMC
  %(prog)s --model time-varying --data-dir ../Rugby-Data --method mcmc

  # Resume from checkpoint (load only)
  %(prog)s --resume joint_model_v2 --data-dir ../Rugby-Data

  # Continue training from checkpoint (warm-start)
  %(prog)s --resume my_model --continue-training --vi-iterations 10000 --data-dir ../Rugby-Data

  # Train on recent data only
  %(prog)s --model static --data-dir ../Rugby-Data --last-seasons 3
        """
    )

    # Model selection
    parser.add_argument('--model', choices=['static', 'time-varying', 'minibatch'],
                       default='static',
                       help='Model variant to train (default: static)')

    # Data
    parser.add_argument('--data-dir', type=Path, required=True,
                       help='Path to Rugby-Data directory')
    parser.add_argument('--last-seasons', type=int,
                       help='Only use last N seasons (for faster training)')
    parser.add_argument('--pattern', type=str, default='*.json',
                       help='File pattern for data loading (default: *.json)')

    # Inference
    parser.add_argument('--method', choices=['vi', 'mcmc'], default='vi',
                       help='Inference method (default: vi)')
    parser.add_argument('--vi-iterations', type=int, default=50000,
                       help='VI iterations (default: 50000)')
    parser.add_argument('--vi-samples', type=int, default=500,
                       help='Number of samples to draw from VI posterior (default: 500, use fewer for large models)')
    parser.add_argument('--vi-sample-batch-size', type=int, default=50,
                       help='Sample in batches of this size to avoid OOM (default: sample all at once)')
    parser.add_argument('--mcmc-draws', type=int, default=1000,
                       help='MCMC draws (default: 1000)')
    parser.add_argument('--mcmc-tune', type=int, default=500,
                       help='MCMC tuning steps (default: 500)')
    parser.add_argument('--mcmc-chains', type=int, default=4,
                       help='MCMC chains (default: 4)')

    # Checkpoints
    parser.add_argument('--resume', type=str,
                       help='Resume from this checkpoint name')
    parser.add_argument('--continue-training', action='store_true',
                       help='Continue VI training from checkpoint (warm-start). Requires --resume.')
    parser.add_argument('--auto-resume', action='store_true',
                       help='Automatically resume from latest checkpoint if available')
    parser.add_argument('--save-as', type=str,
                       help='Save checkpoint with this name (default: auto-generated)')
    parser.add_argument('--checkpoint-every', type=int,
                       help='Save checkpoint every N iterations (for HTCondor resilience)')

    # Model configuration
    parser.add_argument('--score-types', nargs='+',
                       default=['tries', 'penalties', 'conversions', 'drop_goals'],
                       help='Score types to model')
    parser.add_argument('--no-separate-kicking', action='store_true',
                       help='Disable separate kicking/try-scoring effects')
    parser.add_argument('--no-defense', action='store_true',
                       help='Disable defensive effects')
    parser.add_argument('--player-trend-sd', type=float, default=0.1,
                       help='Prior SD for player trends (time-varying only)')
    parser.add_argument('--team-trend-sd', type=float, default=0.1,
                       help='Prior SD for team trends (time-varying only)')

    # Data cleaning
    parser.add_argument('--no-clean-data', action='store_true',
                       help='Skip automatic data cleaning (not recommended)')

    # Output
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    parser.add_argument('--verbose', action='store_true',
                       help='Extra verbose output (useful for HTCondor logs)')

    return parser.parse_args()


def load_data(args):
    """Load and prepare data."""
    if not args.quiet:
        print("=" * 70)
        print("LOADING DATA")
        print("=" * 70)

    dataset = MatchDataset(args.data_dir, fuzzy_match_names=False)
    dataset.load_json_files(pattern=args.pattern)

    df = dataset.to_dataframe(played_only=True)
    df = df[df['position'].between(1, 23)].copy()

    if not args.quiet:
        print(f"Loaded: {len(df):,} observations")
        print(f"Players: {df['player_name'].nunique():,}")
        print(f"Teams: {df['team'].nunique()}")
        print(f"Seasons: {df['season'].nunique()}")
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Filter to recent seasons if requested
    if args.last_seasons:
        all_seasons = sorted(df['season'].unique())
        recent_seasons = all_seasons[-args.last_seasons:]
        df = df[df['season'].isin(recent_seasons)].copy()

        if not args.quiet:
            print(f"\nFiltered to last {args.last_seasons} seasons: {recent_seasons}")
            print(f"Observations: {len(df):,}")
            print(f"Players: {df['player_name'].nunique():,}")

    # Clean data (remove kicking anomalies) unless disabled
    if not args.no_clean_data:
        if not args.quiet:
            print("\n" + "=" * 70)
            print("DATA CLEANING")
            print("=" * 70)
            print("Detecting kicking anomalies (forwards with conversion/penalty scores)...")

        # Detect anomalies (verbose=False to avoid duplicate output)
        anomalies_before = detect_kicking_anomalies(df, verbose=False)

        if len(anomalies_before) > 0:
            if not args.quiet:
                print(f"Found {len(anomalies_before)} anomalies in {anomalies_before['player_name'].nunique()} players")
                print("\nTop 5 affected players:")
                top_affected = anomalies_before.groupby('player_name').agg({
                    'conversions': 'sum',
                    'penalties': 'sum',
                }).sort_values('conversions', ascending=False).head(5)
                for player, stats in top_affected.iterrows():
                    print(f"  {player}: {stats['conversions']} conversions, {stats['penalties']} penalties")

            # Clean the data
            df = clean_kicking_data(df, strategy='remove', verbose=False)

            if not args.quiet:
                print("\nData cleaned successfully!")
                removed_convs = anomalies_before['conversions'].sum()
                removed_pens = anomalies_before['penalties'].sum()
                print(f"  Removed: {removed_convs} conversions, {removed_pens} penalties")
        else:
            if not args.quiet:
                print("✓ No anomalies detected - data is clean")
    elif not args.quiet:
        print("\n⚠️  Data cleaning skipped (--no-clean-data flag)")

    return df


def create_config(args):
    """Create model configuration."""
    config = ModelConfig(
        score_types=tuple(args.score_types),
        separate_kicking_effect=not args.no_separate_kicking,
        include_defense=not args.no_defense,
        time_varying_effects=(args.model == 'time-varying'),
        player_trend_sd=args.player_trend_sd,
        team_trend_sd=args.team_trend_sd,
    )

    if not args.quiet:
        print("\n" + "=" * 70)
        print("MODEL CONFIGURATION")
        print("=" * 70)
        print(f"Variant: {args.model}")
        print(f"Score types: {config.score_types}")
        print(f"Separate kicking effect: {config.separate_kicking_effect}")
        print(f"Defensive effects: {config.include_defense}")
        print(f"Time-varying effects: {config.time_varying_effects}")
        if config.time_varying_effects:
            print(f"  Player trend SD: {config.player_trend_sd}")
            print(f"  Team trend SD: {config.team_trend_sd}")

    return config


def build_model(args, config, df):
    """Build PyMC model."""
    if not args.quiet:
        print("\n" + "=" * 70)
        print("BUILDING MODEL")
        print("=" * 70)

    model = RugbyModel(config)

    # Choose appropriate build method
    if args.model == 'static':
        pymc_model = model.build_joint(df)
    elif args.model == 'time-varying':
        pymc_model = model.build_joint_time_varying(df)
    elif args.model == 'minibatch':
        pymc_model = model.build_joint_minibatch(df)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    if not args.quiet:
        print(f"Built successfully!")
        print(f"  Players: {len(model._player_ids):,}")
        print(f"  Team-seasons: {len(model._team_season_ids)}")
        print(f"  Positions: {len(model._position_ids)}")
        if config.time_varying_effects:
            print(f"  Seasons: {len(model._season_ids)}")

    return model, pymc_model


def create_inference_config(args):
    """Create inference configuration."""
    if args.method == 'vi':
        config = InferenceConfig(
            vi_n_iterations=args.vi_iterations,
            vi_method='advi',
        )
    else:  # mcmc
        config = InferenceConfig(
            mcmc_draws=args.mcmc_draws,
            mcmc_tune=args.mcmc_tune,
            mcmc_chains=args.mcmc_chains,
            mcmc_cores=args.mcmc_chains,
        )

    return config


def find_latest_checkpoint(checkpoint_name_base: str, cache_dir: Path = None) -> str | None:
    """Find the latest checkpoint for a given base name (e.g., iter10000, iter20000)."""
    cache_dir = cache_dir or Path("~/.cache/rugby_ranking").expanduser()
    if not cache_dir.exists():
        return None

    # Look for checkpoints matching pattern: {base}_iter*
    matching_dirs = list(cache_dir.glob(f"{checkpoint_name_base}_iter*"))
    if not matching_dirs:
        return None

    # Extract iteration numbers and find the latest
    checkpoint_iters = []
    for dir_path in matching_dirs:
        try:
            # Extract iteration number from name like "checkpoint_iter10000"
            iter_str = dir_path.name.split("_iter")[-1]
            iter_num = int(iter_str)
            checkpoint_iters.append((iter_num, dir_path.name))
        except (ValueError, IndexError):
            continue

    if not checkpoint_iters:
        return None

    # Return the checkpoint with highest iteration
    latest_iter, latest_name = max(checkpoint_iters, key=lambda x: x[0])
    return latest_name


def fit_model(args, model, inference_config):
    """Fit model using specified inference method."""
    if not args.quiet:
        print("\n" + "=" * 70)
        print(f"FITTING MODEL ({args.method.upper()})")
        print("=" * 70)

    # Determine checkpoint name for periodic saves
    if args.save_as:
        checkpoint_base = args.save_as
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        checkpoint_base = f"{args.model}_{args.method}_{timestamp}"

    # Auto-resume: try to find latest checkpoint for this run
    resume_from = None
    if args.auto_resume and args.checkpoint_every:
        latest = find_latest_checkpoint(checkpoint_base)
        if latest:
            resume_from = latest
            if not args.quiet:
                print(f"Auto-resume: Found checkpoint '{latest}'")

    # Explicit resume takes precedence
    if args.resume:
        resume_from = args.resume

    # Try to resume from checkpoint
    if resume_from:
        try:
            fitter = ModelFitter.load(resume_from, model)
            trace = fitter.trace
            if not args.quiet:
                print(f"✓ Resumed from checkpoint: {resume_from}")
            
            # If --continue-training, warm-start VI from the loaded approximation
            if args.continue_training:
                if args.method != 'vi':
                    print("⚠ Warning: --continue-training only works with --method vi", file=sys.stderr)
                    return fitter, trace
                
                if fitter._vi_approx is None:
                    print("⚠ Warning: Checkpoint has no VI approximation, cannot continue training", file=sys.stderr)
                    return fitter, trace
                
                if not args.quiet:
                    print("\nContinuing VI training with warm-start...")
                    print(f"Running {args.vi_iterations:,} additional iterations")
                    if args.checkpoint_every:
                        print(f"Saving checkpoint every {args.checkpoint_every:,} iterations")
                    import sys
                    sys.stdout.flush()
                
                # Continue training with warm-start
                trace = fitter.fit_vi(
                    warm_start=True,  # Use loaded approximation
                    n_samples=args.vi_samples,
                    sample_batch_size=args.vi_sample_batch_size,
                    checkpoint_every=args.checkpoint_every,
                    checkpoint_name=checkpoint_base,
                    verbose=args.verbose or (not args.quiet),
                )
                
                return fitter, trace
            
            # Otherwise just return the loaded checkpoint
            return fitter, trace
        except (ValueError, FileNotFoundError) as e:
            if not args.quiet:
                print(f"Could not load checkpoint '{resume_from}': {e}")
                print("Starting fresh training...")

    fitter = ModelFitter(model, inference_config)

    # Fit with periodic checkpointing if requested
    if args.method == 'vi':
        if not args.quiet:
            print(f"Running VI for {args.vi_iterations:,} iterations...")
            if args.checkpoint_every:
                print(f"Saving checkpoint every {args.checkpoint_every:,} iterations")
            if args.verbose:
                print("Verbose mode enabled - progress will be logged every 1000 iterations")
            import sys
            sys.stdout.flush()

        trace = fitter.fit_vi(
            n_samples=args.vi_samples,
            sample_batch_size=args.vi_sample_batch_size,
            checkpoint_every=args.checkpoint_every,
            checkpoint_name=checkpoint_base,
            verbose=args.verbose or (not args.quiet),  # Verbose if --verbose or not --quiet
        )
    else:  # mcmc
        if not args.quiet:
            print(f"Running MCMC: {args.mcmc_draws} draws × {args.mcmc_chains} chains "
                  f"(+ {args.mcmc_tune} tuning)...")
            if args.checkpoint_every:
                print(f"Saving checkpoint every {args.checkpoint_every} draws")
            if args.verbose:
                print("Verbose mode enabled - checkpoint progress will be logged")
            import sys
            sys.stdout.flush()

        trace = fitter.fit_mcmc(
            checkpoint_every=args.checkpoint_every,
            checkpoint_name=checkpoint_base,
            verbose=args.verbose or (not args.quiet),
        )

    if not args.quiet:
        print("Fitting complete!")
        import sys
        sys.stdout.flush()

    return fitter, trace


def save_checkpoint(args, fitter):
    """Save model checkpoint."""
    # Generate name if not provided
    if args.save_as:
        name = args.save_as
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        name = f"{args.model}_{args.method}_{timestamp}"

    if not args.quiet:
        print("\n" + "=" * 70)
        print("SAVING CHECKPOINT")
        print("=" * 70)

    path = fitter.save(name)

    if not args.quiet:
        print(f"Saved to: {path}")
        print(f"Load with: ModelFitter.load('{name}', model)")

    return name


def print_diagnostics(args, model, trace):
    """Print model diagnostics."""
    if args.quiet:
        return

    print("\n" + "=" * 70)
    print("DIAGNOSTICS")
    print("=" * 70)

    # Check for time-varying parameters
    if 'beta_player_try_base_raw' in trace.posterior:
        print("Model type: Time-varying with separate kicking effects")
        base_shape = trace.posterior['beta_player_try_base_raw'].shape
        print(f"Player base effects shape: {base_shape}")
        trend_shape = trace.posterior['beta_player_try_trend_raw'].shape
        print(f"Player trend effects shape: {trend_shape}")
    elif 'beta_player_try_raw' in trace.posterior:
        print("Model type: Static with separate kicking effects")
        shape = trace.posterior['beta_player_try_raw'].shape
        print(f"Player effects shape: {shape}")
    elif 'beta_player_raw' in trace.posterior:
        print("Model type: Static with unified effects")
        shape = trace.posterior['beta_player_raw'].shape
        print(f"Player effects shape: {shape}")

    # Show some example posterior means
    if 'sigma_player_try' in trace.posterior:
        sigma_try = trace.posterior['sigma_player_try'].mean().item()
        print(f"σ_player_try: {sigma_try:.3f}")

    if 'sigma_player_kick' in trace.posterior:
        sigma_kick = trace.posterior['sigma_player_kick'].mean().item()
        print(f"σ_player_kick: {sigma_kick:.3f}")

    if 'sigma_team' in trace.posterior:
        sigma_team = trace.posterior['sigma_team'].mean().item()
        print(f"σ_team: {sigma_team:.3f}")
    elif 'sigma_team_base' in trace.posterior:
        sigma_team = trace.posterior['sigma_team_base'].mean().item()
        print(f"σ_team_base: {sigma_team:.3f}")


def main():
    """Main training pipeline."""
    args = parse_args()

    try:
        # 1. Load data
        df = load_data(args)

        # 2. Create configuration
        config = create_config(args)

        # 3. Build model
        model, pymc_model = build_model(args, config, df)

        # 4. Configure inference
        inference_config = create_inference_config(args)

        # 5. Fit model
        fitter, trace = fit_model(args, model, inference_config)

        # 6. Save checkpoint
        checkpoint_name = save_checkpoint(args, fitter)

        # 7. Print diagnostics
        print_diagnostics(args, model, trace)

        if not args.quiet:
            print("\n" + "=" * 70)
            print("✅ TRAINING COMPLETE")
            print("=" * 70)
            print(f"Checkpoint: {checkpoint_name}")
            print(f"\nNext steps:")
            print(f"  - Visualize: Load checkpoint in notebooks")
            print(f"  - Predict: Use MatchPredictor with this model")
            print(f"  - Rankings: Call model.get_player_rankings()")
            if args.method == 'vi':
                print(f"  - More samples: python sample_from_checkpoint.py --checkpoint {checkpoint_name} --samples 5000")
                print(f"  - Continue training: python train_model.py --resume {checkpoint_name} --continue-training")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
