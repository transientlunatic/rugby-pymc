"""
Command-line interface for rugby ranking model.

Supports weekly update workflow:
    rugby-ranking update --data-dir /path/to/Rugby-Data
    rugby-ranking rankings --type players --top 20
    rugby-ranking predict --home "Leinster" --away "Munster"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rugby_ranking.model.core import RugbyModel, ModelConfig
from rugby_ranking.model.inference import ModelFitter


def _resolve_data_dir(args_data_dir):
    """Resolve data directory from args, config, or environment."""
    if args_data_dir:
        return args_data_dir
    try:
        from rugby.config import get_data_dir
        data_dir = get_data_dir()
        if data_dir:
            return data_dir / "json"
    except ImportError:
        pass
    return None


def _resolve_squads_dirs():
    """Return a list of squad directories to search."""
    dirs = ["squads"]
    try:
        from rugby.config import get_squads_dir
        squads_dir = get_squads_dir()
        if squads_dir:
            dirs.insert(0, str(squads_dir))
    except ImportError:
        pass
    dirs.append("../Rugby-Data/squads")
    return dirs


def load_checkpoint(checkpoint_name: str, verbose: bool = True):
    """
    Load a trained model checkpoint.

    Args:
        checkpoint_name: Name of checkpoint (e.g., "joint_model_v2")
        verbose: Print loading status

    Returns:
        (model, trace) tuple

    Example:
        >>> model, trace = load_checkpoint("joint_model_v2")
        >>> rankings = model.get_player_rankings(trace, score_type='tries')
    """
    if verbose:
        print(f"Loading checkpoint: {checkpoint_name}")

    # Create a model instance to load into
    # The actual config and indices will be loaded from the checkpoint
    model = RugbyModel()

    try:
        fitter = ModelFitter.load(checkpoint_name, model)
        trace = fitter.trace

        if verbose:
            print(f"✓ Loaded successfully")
            print(f"  Players: {len(model._player_ids):,}")
            print(f"  Team-seasons: {len(model._team_season_ids)}")

        return model, trace

    except Exception as e:
        if verbose:
            print(f"✗ Failed to load checkpoint: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Rugby player and team ranking system"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Update command
    update_parser = subparsers.add_parser(
        "update",
        help="Update model with latest data"
    )
    update_parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to Rugby-Data json/ directory (auto-detected from rugby config if not specified)"
    )
    update_parser.add_argument(
        "--method",
        choices=["vi", "mcmc"],
        default="vi",
        help="Inference method (default: vi for speed)"
    )
    update_parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint name to save/load"
    )

    # Rankings command
    rankings_parser = subparsers.add_parser(
        "rankings",
        help="Display current rankings"
    )
    rankings_parser.add_argument(
        "--type",
        choices=["players", "teams"],
        default="teams",
        help="What to rank"
    )
    rankings_parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Filter to season (teams only)"
    )
    rankings_parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of entries to show"
    )
    rankings_parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint to load"
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict match outcome"
    )
    predict_parser.add_argument(
        "--home",
        type=str,
        required=True,
        help="Home team name"
    )
    predict_parser.add_argument(
        "--away",
        type=str,
        required=True,
        help="Away team name"
    )
    predict_parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Season (defaults to current)"
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint to load"
    )

    # Upcoming command
    upcoming_parser = subparsers.add_parser(
        "upcoming",
        help="Predict upcoming matches"
    )
    upcoming_parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to Rugby-Data json/ directory (auto-detected from rugby config if not specified)"
    )
    upcoming_parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days ahead to show (default: 7)"
    )
    upcoming_parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Season to filter (defaults to current)"
    )
    upcoming_parser.add_argument(
        "--competition",
        type=str,
        default=None,
        help="Filter by competition (e.g., 'premiership', 'celtic')"
    )
    upcoming_parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint to load"
    )

    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export dashboard data to JSON files"
    )
    export_parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to Rugby-Data json/ directory (auto-detected from rugby config if not specified)"
    )
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dashboard/data"),
        help="Output directory for JSON files (default: dashboard/data)"
    )
    export_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint to use (if None, trains new model)"
    )
    export_parser.add_argument(
        "--seasons",
        type=int,
        default=3,
        help="Number of recent seasons to export (default: 3)"
    )

    # Squad commands
    squad_parser = subparsers.add_parser(
        "squad",
        help="Squad analysis commands"
    )
    squad_subparsers = squad_parser.add_subparsers(dest="squad_command", help="Squad operations")

    # Squad input
    squad_input_parser = squad_subparsers.add_parser(
        "input",
        help="Input squad from text/clipboard"
    )
    squad_input_parser.add_argument(
        "--team",
        type=str,
        required=True,
        help="Team name"
    )
    squad_input_parser.add_argument(
        "--season",
        type=str,
        default="2024-2025",
        help="Season (default: 2024-2025)"
    )
    squad_input_parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Input file path (if not using interactive mode)"
    )
    squad_input_parser.add_argument(
        "--format",
        choices=["auto", "wikipedia", "simple", "csv"],
        default="auto",
        help="Input format (default: auto-detect)"
    )

    # Squad analyze
    squad_analyze_parser = squad_subparsers.add_parser(
        "analyze",
        help="Analyze squad strength and depth"
    )
    squad_analyze_parser.add_argument(
        "--team",
        type=str,
        required=True,
        help="Team name"
    )
    squad_analyze_parser.add_argument(
        "--season",
        type=str,
        default="2024-2025",
        help="Season (default: 2024-2025)"
    )
    squad_analyze_parser.add_argument(
        "--checkpoint",
        type=str,
        default="international-mini5",
        help="Model checkpoint to use (default: international-mini5)"
    )
    squad_analyze_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed breakdown"
    )

    # Squad compare
    squad_compare_parser = squad_subparsers.add_parser(
        "compare",
        help="Compare squads across teams"
    )
    squad_compare_parser.add_argument(
        "--tournament",
        type=str,
        default="six-nations",
        help="Tournament name (default: six-nations)"
    )
    squad_compare_parser.add_argument(
        "--season",
        type=str,
        default="2024-2025",
        help="Season (default: 2024-2025)"
    )
    squad_compare_parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Model checkpoint to use"
    )

    # Squad lineup
    squad_lineup_parser = squad_subparsers.add_parser(
        "lineup",
        help="Predict likely starting lineup"
    )
    squad_lineup_parser.add_argument(
        "--team",
        type=str,
        required=True,
        help="Team name"
    )
    squad_lineup_parser.add_argument(
        "--season",
        type=str,
        default="2024-2025",
        help="Season (default: 2024-2025)"
    )
    squad_lineup_parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Model checkpoint to use"
    )
    squad_lineup_parser.add_argument(
        "--unavailable",
        type=str,
        nargs="+",
        default=None,
        help="List of unavailable players (injuries)"
    )

    # Squad critical-players
    squad_critical_parser = squad_subparsers.add_parser(
        "critical-players",
        help="Identify most critical players"
    )
    squad_critical_parser.add_argument(
        "--team",
        type=str,
        required=True,
        help="Team name"
    )
    squad_critical_parser.add_argument(
        "--season",
        type=str,
        default="2024-2025",
        help="Season (default: 2024-2025)"
    )
    squad_critical_parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Model checkpoint to use"
    )
    squad_critical_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of critical players to show (default: 10)"
    )

    # Squad robustness
    squad_robustness_parser = squad_subparsers.add_parser(
        "robustness",
        help="Analyze squad robustness to injuries"
    )
    squad_robustness_parser.add_argument(
        "--team",
        type=str,
        required=True,
        help="Team name"
    )
    squad_robustness_parser.add_argument(
        "--season",
        type=str,
        default="2024-2025",
        help="Season (default: 2024-2025)"
    )
    squad_robustness_parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Model checkpoint to use"
    )
    squad_robustness_parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="Number of injury simulations (default: 100)"
    )
    squad_robustness_parser.add_argument(
        "--injury-prob",
        type=float,
        default=0.15,
        help="Injury probability per player (default: 0.15)"
    )

    args = parser.parse_args()

    if args.command == "update":
        run_update(args)
    elif args.command == "rankings":
        run_rankings(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "upcoming":
        run_upcoming(args)
    elif args.command == "export":
        run_export(args)
    elif args.command == "squad":
        if args.squad_command == "input":
            run_squad_input(args)
        elif args.squad_command == "analyze":
            run_squad_analyze(args)
        elif args.squad_command == "compare":
            run_squad_compare(args)
        elif args.squad_command == "lineup":
            run_squad_lineup(args)
        elif args.squad_command == "critical-players":
            run_squad_critical_players(args)
        elif args.squad_command == "robustness":
            run_squad_robustness(args)
        else:
            squad_parser.print_help()
    else:
        parser.print_help()


def run_update(args):
    """Run model update with latest data."""
    from rugby_ranking.model.data import MatchDataset
    from rugby_ranking.model.core import RugbyModel, ModelConfig
    from rugby_ranking.model.inference import ModelFitter, InferenceConfig

    args.data_dir = _resolve_data_dir(args.data_dir)
    if not args.data_dir:
        print("Error: --data-dir is required (or run 'rugby config init' to set it)")
        return
    print(f"Loading data from {args.data_dir}...")
    dataset = MatchDataset(args.data_dir)
    dataset.load_json_files()

    print("Preparing model data...")
    df = dataset.to_dataframe(played_only=True)
    print(f"  {len(df)} player-match observations")
    print(f"  {df['season'].nunique()} seasons")

    print("Building model...")
    config = ModelConfig()
    model = RugbyModel(config)
    model.build(df, score_type="tries")

    print(f"Fitting model using {args.method.upper()}...")
    fitter = ModelFitter(model, InferenceConfig())

    if args.method == "vi":
        trace = fitter.fit_vi()
    else:
        trace = fitter.fit_mcmc()

    # Diagnostics
    diag = fitter.diagnostics()
    print(f"  R-hat max: {diag['r_hat_max']:.3f}")
    print(f"  ESS min: {diag['ess_bulk_min']:.0f}")

    # Save checkpoint
    fitter.save(args.checkpoint)
    print(f"Saved checkpoint: {args.checkpoint}")


def run_rankings(args):
    """Display rankings from saved checkpoint."""
    from rugby_ranking.model.core import RugbyModel
    from rugby_ranking.model.inference import ModelFitter

    model = RugbyModel()
    fitter = ModelFitter.load(args.checkpoint, model)

    if args.type == "players":
        rankings = model.get_player_rankings(top_n=args.top)
        print(f"\nTop {args.top} Players:")
        print("=" * 60)
        for i, row in rankings.iterrows():
            print(
                f"{i+1:2d}. {row['player']:<30} "
                f"Effect: {row['effect_mean']:+.3f} "
                f"(±{row['effect_std']:.3f})"
            )
    else:
        rankings = model.get_team_rankings(season=args.season, top_n=args.top)
        print(f"\nTop {args.top} Teams (Season: {args.season or 'all'}):")
        print("=" * 60)
        for i, row in rankings.iterrows():
            print(
                f"{i+1:2d}. {row['team']:<20} ({row['season']}) "
                f"Effect: {row['effect_mean']:+.3f} "
                f"(±{row['effect_std']:.3f})"
            )


def run_predict(args):
    """Predict match outcome."""
    from rugby_ranking.model.core import RugbyModel
    from rugby_ranking.model.inference import ModelFitter
    from rugby_ranking.model.predictions import MatchPredictor

    model = RugbyModel()
    fitter = ModelFitter.load(args.checkpoint, model)

    # Determine season
    if args.season:
        season = args.season
    else:
        # Use most recent season in data
        seasons = sorted(model._season_ids.keys())
        season = seasons[-1] if seasons else "2025-2026"

    predictor = MatchPredictor(model)

    try:
        prediction = predictor.predict_teams_only(
            home_team=args.home,
            away_team=args.away,
            season=season,
        )
        print(f"\n{prediction.summary()}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Available teams:")
        for team, season in sorted(model._team_season_ids.keys()):
            print(f"  - {team} ({season})")


def run_upcoming(args):
    """Show predictions for upcoming matches."""
    from datetime import datetime, timezone, timedelta
    from rugby_ranking.model.core import RugbyModel
    from rugby_ranking.model.inference import ModelFitter
    from rugby_ranking.model.predictions import MatchPredictor
    from rugby_ranking.model.data import MatchDataset

    args.data_dir = _resolve_data_dir(args.data_dir)
    if not args.data_dir:
        print("Error: --data-dir is required (or run 'rugby config init' to set it)")
        return
    print(f"Loading data from {args.data_dir}...")
    dataset = MatchDataset(args.data_dir, fuzzy_match_names=False)
    dataset.load_json_files()

    # Load model
    print(f"Loading model checkpoint: {args.checkpoint}...")
    model = RugbyModel()
    fitter = ModelFitter.load(args.checkpoint, model)

    # Filter to unplayed matches
    today = datetime.now(timezone.utc)
    cutoff_date = today + timedelta(days=args.days)

    upcoming_matches = []
    for match in dataset.matches:
        # Skip played matches
        if match.is_played:
            continue

        # Filter by date range
        if match.date < today or match.date > cutoff_date:
            continue

        # Filter by season if specified
        if args.season and match.season != args.season:
            continue

        # Filter by competition if specified
        if args.competition and args.competition.lower() not in match.competition.lower():
            continue

        upcoming_matches.append(match)

    if not upcoming_matches:
        print(f"\nNo upcoming matches found in the next {args.days} days")
        if args.season:
            print(f"  Season filter: {args.season}")
        if args.competition:
            print(f"  Competition filter: {args.competition}")
        return

    # Sort by date
    upcoming_matches.sort(key=lambda m: m.date)

    # Determine display season (for header only)
    display_season = args.season if args.season else "multiple seasons"

    print(f"\n{'='*70}")
    print(f"UPCOMING MATCHES (Next {args.days} Days)")
    print(f"{'='*70}")
    print(f"Season: {display_season}")
    print(f"Total matches: {len(upcoming_matches)}")
    print(f"{'='*70}\n")

    # Generate predictions
    predictor = MatchPredictor(model, fitter.trace)

    current_date = None
    predictions_count = 0
    errors_count = 0

    for match in upcoming_matches:
        # Print date header if new date
        if match.date.date() != current_date:
            current_date = match.date.date()
            print(f"\n{match.date.strftime('%A, %B %d, %Y')}")
            print("-" * 70)

        prediction_notes = []
        try:
            # Check if we have lineup data
            has_lineups = bool(match.home_lineup and match.away_lineup)

            # Transform lineup format from {str: dict} to {int: str}
            home_lineup_simple = None
            away_lineup_simple = None
            if has_lineups:
                try:
                    home_lineup_simple = {
                        int(pos): player_data['name']
                        for pos, player_data in match.home_lineup.items()
                        if isinstance(player_data, dict) and 'name' in player_data
                    }
                    away_lineup_simple = {
                        int(pos): player_data['name']
                        for pos, player_data in match.away_lineup.items()
                        if isinstance(player_data, dict) and 'name' in player_data
                    }
                    # Check we got valid lineups
                    if not home_lineup_simple or not away_lineup_simple:
                        has_lineups = False
                except (ValueError, KeyError, AttributeError):
                    # Lineup format incompatible, fall back to team-only
                    has_lineups = False

            if has_lineups:
                # Use lineup-based prediction
                try:
                    prediction = predictor.predict_full_lineup(
                        home_team=match.home_team,
                        away_team=match.away_team,
                        season=match.season,
                        home_lineup=home_lineup_simple,
                        away_lineup=away_lineup_simple
                    )
                    prediction_notes.append("Method: Lineup-based")
                except (ValueError, AttributeError, Exception):
                    # Fallback to team-only if lineup prediction fails
                    # (model checkpoint may not support player-level predictions)
                    prediction = predictor.predict_teams_only(
                        home_team=match.home_team,
                        away_team=match.away_team,
                        season=match.season,
                    )
                    prediction_notes.append(
                        "Method: Team strength only (model doesn't support lineup predictions)"
                    )
            else:
                # Use team-only prediction
                prediction = predictor.predict_teams_only(
                    home_team=match.home_team,
                    away_team=match.away_team,
                    season=match.season,
                )
                prediction_notes.append("Method: Team strength only")

            # Check for season fallback
            if prediction.home_season_used or prediction.away_season_used:
                fallback_note = "Using"
                if prediction.home_season_used:
                    fallback_note += f" {match.home_team} data from {prediction.home_season_used}"
                if prediction.away_season_used:
                    if prediction.home_season_used:
                        fallback_note += " and"
                    fallback_note += f" {match.away_team} data from {prediction.away_season_used}"
                prediction_notes.append(fallback_note)

            # Format output
            print(f"\n  {match.home_team} vs {match.away_team}")
            print(f"  Competition: {match.competition} | Season: {match.season}")
            print(f"  Predicted: {prediction.home.mean:.0f} - {prediction.away.mean:.0f}")
            print(f"  Probabilities: Home {prediction.home_win_prob:.1%} | "
                  f"Draw {prediction.draw_prob:.1%} | Away {prediction.away_win_prob:.1%}")
            print(f"  90% CI: [{prediction.home.ci_lower:.0f}-{prediction.home.ci_upper:.0f}] vs "
                  f"[{prediction.away.ci_lower:.0f}-{prediction.away.ci_upper:.0f}]")
            if prediction_notes:
                print(f"  {' | '.join(prediction_notes)}")
            predictions_count += 1

        except ValueError as e:
            print(f"\n  {match.home_team} vs {match.away_team}")
            print(f"  Competition: {match.competition} | Season: {match.season}")
            print(f"  Prediction unavailable: {e}")
            errors_count += 1

    print(f"\n{'='*70}")
    print(f"Predictions generated: {predictions_count}")
    if errors_count > 0:
        print(f"Errors: {errors_count} (teams/seasons not in training data)")
    print(f"{'='*70}\n")


def run_export(args):
    """Export dashboard data to JSON files."""
    from rugby_ranking.tools.export_dashboard_data import export_dashboard_data

    args.data_dir = _resolve_data_dir(args.data_dir)
    if not args.data_dir:
        print("Error: --data-dir is required (or run 'rugby config init' to set it)")
        return
    export_dashboard_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint,
        recent_seasons_only=args.seasons
    )


def run_squad_input(args):
    """Input squad from text/clipboard."""
    from rugby_ranking.model.squad_analysis import SquadParser
    import os

    parser = SquadParser()

    # Read input
    if args.file:
        # Read from file
        with open(args.file, 'r') as f:
            text = f.read()
        print(f"Reading squad from {args.file}...")
    else:
        # Interactive input
        print(f"\nInput squad for {args.team} ({args.season})")
        print("=" * 60)
        print("Paste squad list (Wikipedia format recommended).")
        print("Press Ctrl+D (Unix) or Ctrl+Z (Windows) when done:")
        print()

        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass

        text = '\n'.join(lines)

    # Parse squad
    try:
        squad = parser.parse_text(
            text,
            team=args.team,
            season=args.season,
            format=args.format
        )

        print(f"\n✓ Parsed {len(squad)} players:")

        # Count by section
        if 'section' in squad.columns:
            forwards = len(squad[squad['section'] == 'forwards'])
            backs = len(squad[squad['section'] == 'backs'])
            print(f"  - {forwards} forwards")
            print(f"  - {backs} backs")

        # Show sample
        print("\nSample players:")
        print(squad[['player', 'primary_position', 'club']].head(10).to_string(index=False))

        # Save to file
        os.makedirs('squads', exist_ok=True)
        filename = f"squads/{args.team.lower().replace(' ', '_')}_{args.season}.csv"
        squad.to_csv(filename, index=False)
        print(f"\n✓ Squad saved to: {filename}")

    except Exception as e:
        print(f"\n✗ Error parsing squad: {e}")
        print("Please check the format and try again.")


def run_squad_analyze(args):
    """Analyze squad strength and depth."""
    from rugby_ranking.model.squad_analysis import SquadAnalyzer, format_squad_analysis
    from rugby_ranking.model.core import RugbyModel
    from rugby_ranking.model.inference import ModelFitter
    import os
    import json

    # Position to section mapping for JSON format conversion
    POSITION_SECTIONS = {
        'Hooker': 'forwards',
        'Prop': 'forwards',
        'Lock': 'forwards',
        'Back Row': 'mixed',
        'Scrum-half': 'backs',
        'Fly-half': 'backs',
        'Centre': 'backs',
        'Wing': 'backs',
        'Fullback': 'backs',
    }

    # Try to find squad file - check JSON first (single file with all teams), then CSV
    squad = None
    filename = None

    base_paths = _resolve_squads_dirs()

    # Try single JSON file with all teams (e.g., 2026_six_nations_championship_squads.json)
    json_candidates = []
    for base in base_paths:
        json_candidates.extend([
            f"{base}/{args.season.split('-')[0]}_six_nations_championship_squads.json",
            f"{base}/six_nations_{args.season}.json",
            f"{base}/six_nations_championship_{args.season}.json",
        ])

    for json_file in json_candidates:
        if os.path.exists(json_file):
            try:
                with open(json_file) as f:
                    squads_data = json.load(f)
                if args.team in squads_data:
                    squad_json = pd.DataFrame(squads_data[args.team]['players'])
                    # Convert JSON format to SquadAnalyzer format
                    squad = pd.DataFrame({
                        'player': squad_json['name'],
                        'position_text': squad_json['position'],
                        'club': squad_json.get('club', 'Unknown'),
                        'team': args.team,
                        'season': args.season,
                        'section': squad_json['position'].map(POSITION_SECTIONS).fillna('mixed'),
                        'primary_position': squad_json['position'],
                        'secondary_positions': '[]'
                    })
                    filename = json_file
                    break
            except Exception as e:
                print(f"Warning: Error reading {json_file}: {e}")

    # If not found in JSON, try CSV
    if squad is None:
        for base in base_paths:
            csv_filename = f"{base}/{args.team.lower().replace(' ', '_')}_{args.season}.csv"
            if os.path.exists(csv_filename):
                squad = pd.read_csv(csv_filename)
                filename = csv_filename
                break

    if squad is None:
        print(f"✗ Squad file not found for {args.team} ({args.season})")
        print("\nTried:")
        for jf in json_candidates:
            print(f"  - {jf}")
        print("\nFirst input the squad using:")
        print(f"  rugby-ranking squad input --team \"{args.team}\" --season {args.season}")
        return

    print(f"✓ Loaded squad from {filename} ({len(squad)} players)")

    # Load model
    print(f"Loading model checkpoint: {args.checkpoint}...")
    model = RugbyModel()
    fitter = ModelFitter.load(args.checkpoint, model)

    # Analyze squad
    analyzer = SquadAnalyzer(model, fitter.trace)
    analysis = analyzer.analyze_squad(squad, args.team, args.season)

    # Display results
    report = format_squad_analysis(analysis, detailed=args.detailed)
    print("\n" + report)

    # Save report
    os.makedirs('reports', exist_ok=True)
    report_filename = f"reports/{args.team.lower().replace(' ', '_')}_{args.season}_analysis.txt"
    with open(report_filename, 'w') as f:
        f.write(report)
    print(f"\n✓ Report saved to: {report_filename}")


def run_squad_compare(args):
    """Compare squads across tournament."""
    from rugby_ranking.model.squad_analysis import SquadAnalyzer, format_squad_analysis
    from rugby_ranking.model.core import RugbyModel
    from rugby_ranking.model.inference import ModelFitter
    import os
    import glob

    # Define tournament teams
    tournaments = {
        'six-nations': ['Scotland', 'England', 'Ireland', 'France', 'Wales', 'Italy'],
    }

    teams = tournaments.get(args.tournament.lower(), [])

    if not teams:
        print(f"✗ Unknown tournament: {args.tournament}")
        print(f"Available tournaments: {', '.join(tournaments.keys())}")
        return

    # Load model
    print(f"Loading model checkpoint: {args.checkpoint}...")
    model = RugbyModel()
    fitter = ModelFitter.load(args.checkpoint, model)

    # Analyze each squad
    analyzer = SquadAnalyzer(model, fitter.trace)
    analyses = {}

    # Position mapping for JSON conversion
    POSITION_SECTIONS = {
        'Hooker': 'forwards', 'Prop': 'forwards', 'Lock': 'forwards',
        'Back Row': 'mixed', 'Scrum-half': 'backs', 'Fly-half': 'backs',
        'Centre': 'backs', 'Wing': 'backs', 'Fullback': 'backs',
    }

    # Try to load from single JSON file first
    import json

    base_paths = _resolve_squads_dirs()
    json_candidates = []
    for base in base_paths:
        json_candidates.extend([
            f"{base}/{args.season.split('-')[0]}_six_nations_championship_squads.json",
            f"{base}/six_nations_{args.season}.json",
        ])

    squads_json = None
    squads_json_file = None
    for json_file in json_candidates:
        if os.path.exists(json_file):
            try:
                with open(json_file) as f:
                    squads_json = json.load(f)
                squads_json_file = json_file
                break
            except Exception:
                pass

    for team in teams:
        squad = None
        filename = None

        # Try JSON first
        if squads_json and team in squads_json:
            try:
                squad_data = pd.DataFrame(squads_json[team]['players'])
                squad = pd.DataFrame({
                    'player': squad_data['name'],
                    'position_text': squad_data['position'],
                    'club': squad_data.get('club', 'Unknown'),
                    'team': team,
                    'season': args.season,
                    'section': squad_data['position'].map(POSITION_SECTIONS).fillna('mixed'),
                    'primary_position': squad_data['position'],
                    'secondary_positions': '[]'
                })
                filename = squads_json_file
            except Exception as e:
                print(f"Warning: Error parsing {team} from JSON: {e}")

        # Fall back to CSV
        if squad is None:
            for base in base_paths:
                csv_filename = f"{base}/{team.lower().replace(' ', '_')}_{args.season}.csv"
                if os.path.exists(csv_filename):
                    squad = pd.read_csv(csv_filename)
                    filename = csv_filename
                    break

        if squad is None:
            print(f"⚠ Skipping {team}: squad file not found")
            continue

        print(f"\nAnalyzing {team}... ({len(squad)} players from {filename})")

        try:
            analysis = analyzer.analyze_squad(squad, team, args.season)
            analyses[team] = analysis
        except Exception as e:
            print(f"✗ Error analyzing {team}: {e}")

    if not analyses:
        print("\n✗ No squads found to compare")
        print(f"\nFirst input squads using:")
        print(f"  rugby-ranking squad input --team \"<Team>\" --season {args.season}")
        return

    # Generate comparison report
    print("\n" + "=" * 70)
    print(f"{args.tournament.upper()} SQUAD COMPARISON ({args.season})")
    print("=" * 70)
    print()

    # Overall rankings
    rankings = []
    for team, analysis in analyses.items():
        rankings.append({
            'team': team,
            'strength': analysis.overall_strength or 0,
            'depth': analysis.depth_score or 0,
        })

    rankings_df = pd.DataFrame(rankings).sort_values('strength', ascending=False)

    print("OVERALL RANKINGS")
    print("-" * 70)
    print(f"{'Rank':<6} {'Team':<20} {'Strength':<12} {'Depth':<12}")
    print("-" * 70)

    for i, row in rankings_df.iterrows():
        rank = i + 1
        print(f"{rank:<6} {row['team']:<20} {row['strength']*100:>6.0f}/100    {row['depth']*100:>6.0f}/100")

    print("\n" + "=" * 70)

    # Save comparison
    os.makedirs('reports', exist_ok=True)
    report_filename = f"reports/{args.tournament}_{args.season}_comparison.txt"

    with open(report_filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"{args.tournament.upper()} SQUAD COMPARISON ({args.season})\n")
        f.write("=" * 70 + "\n\n")
        f.write(rankings_df.to_string(index=False))

        f.write("\n\n" + "=" * 70 + "\n")
        f.write("INDIVIDUAL TEAM ANALYSES\n")
        f.write("=" * 70 + "\n\n")

        for team in rankings_df['team']:
            if team in analyses:
                f.write(format_squad_analysis(analyses[team], detailed=True))
                f.write("\n\n")

    print(f"✓ Comparison report saved to: {report_filename}")


def run_squad_lineup(args):
    """Predict likely starting lineup."""
    from rugby_ranking.model.squad_analysis import SquadAnalyzer, LineupPredictor
    from rugby_ranking.model.core import RugbyModel
    from rugby_ranking.model.inference import ModelFitter
    import os
    import json

    # Load squad (same logic as run_squad_analyze)
    squad = None
    filename = None
    base_paths = _resolve_squads_dirs()

    # Try JSON first
    json_candidates = []
    for base in base_paths:
        json_candidates.extend([
            f"{base}/{args.season.split('-')[0]}_six_nations_championship_squads.json",
            f"{base}/six_nations_{args.season}.json",
        ])

    POSITION_SECTIONS = {
        'Hooker': 'forwards', 'Prop': 'forwards', 'Lock': 'forwards',
        'Back Row': 'mixed', 'Scrum-half': 'backs', 'Fly-half': 'backs',
        'Centre': 'backs', 'Wing': 'backs', 'Fullback': 'backs',
    }

    for json_file in json_candidates:
        if os.path.exists(json_file):
            try:
                with open(json_file) as f:
                    squads_data = json.load(f)
                if args.team in squads_data:
                    squad_json = pd.DataFrame(squads_data[args.team]['players'])
                    squad = pd.DataFrame({
                        'player': squad_json['name'],
                        'position_text': squad_json['position'],
                        'club': squad_json.get('club', 'Unknown'),
                        'team': args.team,
                        'season': args.season,
                        'section': squad_json['position'].map(POSITION_SECTIONS).fillna('mixed'),
                        'primary_position': squad_json['position'],
                        'secondary_positions': '[]'
                    })
                    filename = json_file
                    break
            except Exception:
                pass

    # Fall back to CSV
    if squad is None:
        for base in base_paths:
            csv_filename = f"{base}/{args.team.lower().replace(' ', '_')}_{args.season}.csv"
            if os.path.exists(csv_filename):
                squad = pd.read_csv(csv_filename)
                filename = csv_filename
                break

    if squad is None:
        print(f"✗ Squad file not found for {args.team} ({args.season})")
        print(f"\nFirst input the squad using:")
        print(f"  rugby-ranking squad input --team \"{args.team}\" --season {args.season}")
        return

    print(f"✓ Loaded squad from {filename} ({len(squad)} players)")

    # Load model
    print(f"Loading model checkpoint: {args.checkpoint}...")
    model = RugbyModel()
    fitter = ModelFitter.load(args.checkpoint, model)

    # Analyze squad
    analyzer = SquadAnalyzer(model, fitter.trace)
    analysis = analyzer.analyze_squad(squad, args.team, args.season)

    # Predict lineup
    print("\nPredicting lineup...")
    predictor = LineupPredictor()

    try:
        lineup = predictor.predict_lineup(
            analysis,
            unavailable=args.unavailable
        )

        print("\n" + "=" * 70)
        print(f"PREDICTED LINEUP: {args.team}")
        print("=" * 70)
        print()

        # Starting XV
        print("STARTING XV")
        print("-" * 70)
        for pos_num in sorted(lineup['starting_xv'].keys()):
            position_name = LineupPredictor.STARTING_XV_POSITIONS[pos_num]
            player = lineup['starting_xv'][pos_num]
            print(f"{pos_num:2d}. {position_name:<15} {player}")

        # Bench
        print("\nBENCH")
        print("-" * 70)
        for i, player in enumerate(lineup['bench'][:8], 16):
            print(f"{i}. {player}")

        print()
        print(f"Total Rating: {lineup['total_rating']:.3f}")
        print(f"Coverage Valid: {'✓' if lineup['coverage_valid'] else '✗'}")

        if args.unavailable:
            print(f"\nUnavailable: {', '.join(args.unavailable)}")

        print("=" * 70)

    except ValueError as e:
        print(f"✗ Error predicting lineup: {e}")


def run_squad_critical_players(args):
    """Identify most critical players."""
    from rugby_ranking.model.squad_analysis import (
        SquadAnalyzer, LineupPredictor, InjuryImpactAnalyzer
    )
    from rugby_ranking.model.core import RugbyModel
    from rugby_ranking.model.inference import ModelFitter
    import os
    import json

    # Load squad (same logic as above)
    squad = None
    filename = None
    base_paths = _resolve_squads_dirs()

    json_candidates = []
    for base in base_paths:
        json_candidates.extend([
            f"{base}/{args.season.split('-')[0]}_six_nations_championship_squads.json",
            f"{base}/six_nations_{args.season}.json",
        ])

    POSITION_SECTIONS = {
        'Hooker': 'forwards', 'Prop': 'forwards', 'Lock': 'forwards',
        'Back Row': 'mixed', 'Scrum-half': 'backs', 'Fly-half': 'backs',
        'Centre': 'backs', 'Wing': 'backs', 'Fullback': 'backs',
    }

    for json_file in json_candidates:
        if os.path.exists(json_file):
            try:
                with open(json_file) as f:
                    squads_data = json.load(f)
                if args.team in squads_data:
                    squad_json = pd.DataFrame(squads_data[args.team]['players'])
                    squad = pd.DataFrame({
                        'player': squad_json['name'],
                        'position_text': squad_json['position'],
                        'club': squad_json.get('club', 'Unknown'),
                        'team': args.team,
                        'season': args.season,
                        'section': squad_json['position'].map(POSITION_SECTIONS).fillna('mixed'),
                        'primary_position': squad_json['position'],
                        'secondary_positions': '[]'
                    })
                    filename = json_file
                    break
            except Exception:
                pass

    if squad is None:
        for base in base_paths:
            csv_filename = f"{base}/{args.team.lower().replace(' ', '_')}_{args.season}.csv"
            if os.path.exists(csv_filename):
                squad = pd.read_csv(csv_filename)
                filename = csv_filename
                break

    if squad is None:
        print(f"✗ Squad file not found for {args.team} ({args.season})")
        return

    print(f"✓ Loaded squad from {filename} ({len(squad)} players)")

    # Load model
    print(f"Loading model checkpoint: {args.checkpoint}...")
    model = RugbyModel()
    fitter = ModelFitter.load(args.checkpoint, model)

    # Analyze squad
    analyzer = SquadAnalyzer(model, fitter.trace)
    analysis = analyzer.analyze_squad(squad, args.team, args.season)

    # Identify critical players
    print(f"\nAnalyzing player criticality...")
    predictor = LineupPredictor()
    impact_analyzer = InjuryImpactAnalyzer(predictor)

    critical = impact_analyzer.identify_critical_players(analysis, top_n=args.top)

    print("\n" + "=" * 70)
    print(f"MOST CRITICAL PLAYERS: {args.team}")
    print("=" * 70)
    print()
    print(f"{'Rank':<6} {'Player':<25} {'Position':<12} {'Criticality':<12} {'Replacement':<20}")
    print("-" * 70)

    for i, row in critical.iterrows():
        rank = i + 1
        player = row['player']
        position = row.get('position', '?')
        criticality = row.get('criticality_score', 0)
        replacement = row.get('replacement', 'None')

        print(f"{rank:<6} {player:<25} {position:<12} {criticality:>6.0%}      {replacement:<20}")

    print("=" * 70)
    print("\nCriticality: 100% = irreplaceable, 0% = easily replaceable")


def run_squad_robustness(args):
    """Analyze squad robustness to injuries."""
    from rugby_ranking.model.squad_analysis import (
        SquadAnalyzer, LineupPredictor, InjuryImpactAnalyzer
    )
    from rugby_ranking.model.core import RugbyModel
    from rugby_ranking.model.inference import ModelFitter
    import os
    import json

    # Load squad
    squad = None
    filename = None
    base_paths = _resolve_squads_dirs()

    json_candidates = []
    for base in base_paths:
        json_candidates.extend([
            f"{base}/{args.season.split('-')[0]}_six_nations_championship_squads.json",
            f"{base}/six_nations_{args.season}.json",
        ])

    POSITION_SECTIONS = {
        'Hooker': 'forwards', 'Prop': 'forwards', 'Lock': 'forwards',
        'Back Row': 'mixed', 'Scrum-half': 'backs', 'Fly-half': 'backs',
        'Centre': 'backs', 'Wing': 'backs', 'Fullback': 'backs',
    }

    for json_file in json_candidates:
        if os.path.exists(json_file):
            try:
                with open(json_file) as f:
                    squads_data = json.load(f)
                if args.team in squads_data:
                    squad_json = pd.DataFrame(squads_data[args.team]['players'])
                    squad = pd.DataFrame({
                        'player': squad_json['name'],
                        'position_text': squad_json['position'],
                        'club': squad_json.get('club', 'Unknown'),
                        'team': args.team,
                        'season': args.season,
                        'section': squad_json['position'].map(POSITION_SECTIONS).fillna('mixed'),
                        'primary_position': squad_json['position'],
                        'secondary_positions': '[]'
                    })
                    filename = json_file
                    break
            except Exception:
                pass

    if squad is None:
        for base in base_paths:
            csv_filename = f"{base}/{args.team.lower().replace(' ', '_')}_{args.season}.csv"
            if os.path.exists(csv_filename):
                squad = pd.read_csv(csv_filename)
                filename = csv_filename
                break

    if squad is None:
        print(f"✗ Squad file not found for {args.team} ({args.season})")
        return

    print(f"✓ Loaded squad from {filename} ({len(squad)} players)")

    # Load model
    print(f"Loading model checkpoint: {args.checkpoint}...")
    model = RugbyModel()
    fitter = ModelFitter.load(args.checkpoint, model)

    # Analyze squad
    analyzer = SquadAnalyzer(model, fitter.trace)
    analysis = analyzer.analyze_squad(squad, args.team, args.season)

    # Analyze robustness
    print(f"\nSimulating injuries ({args.simulations} scenarios)...")
    predictor = LineupPredictor()
    impact_analyzer = InjuryImpactAnalyzer(predictor)

    robustness = impact_analyzer.analyze_squad_robustness(
        analysis,
        n_simulations=args.simulations,
        injury_prob=args.injury_prob
    )

    print("\n" + "=" * 70)
    print(f"SQUAD ROBUSTNESS ANALYSIS: {args.team}")
    print("=" * 70)
    print()
    print(f"Robustness Score: {robustness['robustness_score']:.0%}")
    print(f"  (Higher is better: team maintains strength despite injuries)")
    print()
    print(f"Average Impact:   {robustness['mean_impact']:.3f} rating points")
    print(f"Std Deviation:    {robustness['std_impact']:.3f}")
    print(f"Best Case:        {robustness['best_case']:.3f} (no/minor injuries)")
    print(f"Worst Case:       {robustness['worst_case']:.3f} (severe injuries)")
    print()

    if robustness.get('vulnerable_positions'):
        print("POSITIONS MOST VULNERABLE TO INJURIES")
        print("-" * 70)
        for pos_impact in robustness['vulnerable_positions'][:5]:
            position = pos_impact['position']
            impact = pos_impact['average_impact']
            print(f"  {position:<20} Average Impact: {impact:.3f}")

    print("=" * 70)
    print(f"\nSimulation parameters: {args.simulations} scenarios, {args.injury_prob:.0%} injury probability per player")


if __name__ == "__main__":
    main()
