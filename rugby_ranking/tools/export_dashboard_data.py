"""Export model data to static JSON files for dashboard deployment."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from rugby_ranking.model.data import MatchDataset
from rugby_ranking.model.core import RugbyModel, ModelConfig
from rugby_ranking.model.inference import ModelFitter, InferenceConfig
from rugby_ranking.model.predictions import MatchPredictor
from rugby_ranking.model.season_predictor import SeasonPredictor
from rugby_ranking.model.paths_to_victory import PathsAnalyzer
from rugby_ranking.model.squad_analysis import SquadAnalyzer
from rugby_ranking.model.league_table import LeagueTable


def export_team_strength_series(
    model: RugbyModel,
    trace,
    df: pd.DataFrame,
    seasons: list,
    output_dir: Path,
) -> None:
    """Export team strength over time (per season/score type)."""
    print("  - Team strength series...")
    strength_series = []

    for season in seasons:
        for score_type in ["tries", "penalties", "conversions"]:
            offensive = model.get_team_rankings(
                trace=trace,
                season=season,
                score_type=score_type,
                top_n=50,
            )

            defensive = None
            if model.config.include_defense:
                defensive = model.get_defensive_rankings(
                    trace=trace,
                    season=season,
                    score_type=score_type,
                    top_n=50,
                )

            for _, row in offensive.iterrows():
                entry = {
                    "team": row["team"],
                    "season": season,
                    "score_type": score_type,
                    "offense_mean": float(row["effect_mean"]),
                    "offense_std": float(row["effect_std"]),
                }

                if defensive is not None:
                    defense_row = defensive[defensive["team"] == row["team"]]
                    if not defense_row.empty:
                        entry["defense_mean"] = float(defense_row.iloc[0]["defense_mean"])
                        entry["defense_std"] = float(defense_row.iloc[0]["defense_std"])

                strength_series.append(entry)

    with open(output_dir / "team_strength_series.json", "w") as f:
        json.dump(strength_series, f, indent=2)


def export_team_finish_positions(
    df: pd.DataFrame,
    seasons: list,
    output_dir: Path,
) -> None:
    """Export historical final positions by season and competition."""
    print("  - Team finish positions...")
    positions_data = []

    for season in seasons:
        season_df = df[df["season"] == season]
        competitions = season_df["competition"].unique()

        for competition in competitions:
            comp_df = season_df[season_df["competition"] == competition]
            
            # Compute team statistics from match data
            team_stats = []
            for team in comp_df["team"].unique():
                team_df = comp_df[comp_df["team"] == team]
                played = len(team_df["match_id"].unique())
                
                # Get match results (one row per match)
                match_results = team_df.groupby("match_id").agg({
                    "team_score": "first",
                    "opponent_score": "first"
                }).reset_index()
                
                won = len(match_results[match_results["team_score"] > match_results["opponent_score"]])
                drawn = len(match_results[match_results["team_score"] == match_results["opponent_score"]])
                lost = len(match_results[match_results["team_score"] < match_results["opponent_score"]])
                
                # Rugby points: 4 for win, 2 for draw, 0 for loss
                # Plus bonus points (try bonus, losing bonus typically, but we'll simplify)
                points = won * 4 + drawn * 2
                total_score = team_df.groupby("match_id")["team_score"].first().sum()
                
                team_stats.append({
                    "team": team,
                    "played": played,
                    "won": won,
                    "drawn": drawn,
                    "lost": lost,
                    "points": points,
                    "total_score": total_score
                })
            
            # Sort by points, then by total score
            team_stats_df = pd.DataFrame(team_stats).sort_values(
                ["points", "total_score"], ascending=[False, False]
            )
            team_stats_df["position"] = range(1, len(team_stats_df) + 1)
            
            for _, row in team_stats_df.iterrows():
                positions_data.append({
                    "team": row["team"],
                    "season": season,
                    "competition": competition,
                    "position": int(row["position"]),
                    "played": int(row["played"]),
                    "won": int(row["won"]),
                    "total_points": int(row["points"]),
                })

    with open(output_dir / "team_finish_positions.json", "w") as f:
        json.dump(positions_data, f, indent=2)


def export_upcoming_predictions(
    model: RugbyModel,
    trace,
    df: pd.DataFrame,
    season: str,
    output_dir: Path,
) -> None:
    """Export predictions for sample matches (both upcoming and recent historical)."""
    print("  - Sample match predictions...")

    match_predictor = MatchPredictor(model, trace)

    predictions_data = []

    # Get recent matches from the season, sampling diverse competitions
    season_df = df[df["season"] == season]
    
    # Sample up to 5 matches per competition for variety
    sampled_matches = []
    for competition in season_df["competition"].unique():
        comp_df = season_df[season_df["competition"] == competition]
        match_ids = comp_df["match_id"].unique()
        # Take up to 5 random matches from this competition
        sample_size = min(5, len(match_ids))
        if sample_size > 0:
            import random
            sampled = random.sample(list(match_ids), sample_size)
            sampled_matches.extend(sampled)
    
    # Limit total to 30 predictions to keep file size reasonable
    sampled_matches = sampled_matches[:30]

    for match_id in sampled_matches:
        match_df = season_df[season_df["match_id"] == match_id]

        if match_df.empty:
            continue

        first_row = match_df.iloc[0]

        try:
            pred = match_predictor.predict_teams_only(
                home_team=first_row["team"],
                away_team=first_row["opponent"],
                season=season,
                n_samples=500,
            )

            date_value = first_row.get("date", "")
            if hasattr(date_value, "isoformat"):
                date_value = date_value.isoformat()

            predictions_data.append({
                "date": str(date_value),
                "home_team": first_row["team"],
                "away_team": first_row["opponent"],
                "season": season,
                "competition": first_row.get("competition", ""),
                "home_score_pred": float(pred.home.mean),
                "away_score_pred": float(pred.away.mean),
                "home_win_prob": float(pred.home_win_prob),
                "away_win_prob": float(pred.away_win_prob),
                "draw_prob": float(pred.draw_prob),
            })
        except Exception as e:
            print(f"    Skipping prediction for {first_row['team']} vs {first_row['opponent']}: {e}")
            continue

    with open(output_dir / "upcoming_predictions.json", "w") as f:
        json.dump(predictions_data, f, indent=2)


def export_paths_to_victory(
    match_predictor: MatchPredictor,
    df: pd.DataFrame,
    season: str,
    output_dir: Path,
    target_teams: list | None = None,
) -> None:
    """
    Export paths to victory analysis for top teams.
    
    Note: This requires remaining/unplayed fixtures. For historical data
    with only completed matches, this will produce an empty result.
    """
    print("  - Paths to victory...")

    if target_teams is None:
        season_df = df[df["season"] == season]
        team_match_counts = season_df.groupby("team").size().sort_values(ascending=False)
        target_teams = team_match_counts.head(6).index.tolist()

    paths_data = []
    
    # Check if we have any remaining fixtures at all
    # For historical data export, this will typically be empty
    season_df = df[df["season"] == season]
    if len(season_df) == 0:
        print("    No data for season, skipping paths analysis")
        with open(output_dir / "paths_to_victory.json", "w") as f:
            json.dump([], f, indent=2)
        return

    for team in target_teams:
        try:
            team_season_df = season_df[
                (season_df["team"] == team) | (season_df["opponent"] == team)
            ]

            remaining_fixtures = pd.DataFrame({
                "home_team": [],
                "away_team": [],
            })

            if len(remaining_fixtures) == 0:
                # This is expected for historical data - no remaining fixtures
                continue

            season_predictor = SeasonPredictor(
                match_predictor=match_predictor,
                competition="URC",
            )

            season_pred = season_predictor.predict_season(
                played_matches=team_season_df,
                remaining_fixtures=remaining_fixtures,
                season=season,
                n_simulations=500,
                return_samples=True,
            )

            paths_analyzer = PathsAnalyzer(season_pred, match_predictor)
            paths_output = paths_analyzer.analyze_paths(team=team, target_position=2)

            critical_games = [
                {
                    "home_team": game[0],
                    "away_team": game[1],
                    "mutual_information": float(score),
                }
                for game, score in paths_output.critical_games[:10]
            ]

            paths_data.append({
                "team": team,
                "competition": "International",
                "target_position": paths_output.target_position,
                "probability": float(paths_output.probability),
                "narrative": paths_output.narrative,
                "critical_games": critical_games,
            })

        except Exception as e:
            print(f"    Error analyzing paths for {team}: {e}")
            continue

    with open(output_dir / "paths_to_victory.json", "w") as f:
        json.dump(paths_data, f, indent=2)


def export_squad_depth(
    model: RugbyModel,
    trace,
    season: str,
    output_dir: Path,
) -> None:
    """Export squad depth analysis for available squads."""
    print("  - Squad depth...")

    squad_data = []

    squads_dir = Path("squads")
    if not squads_dir.exists():
        print("    Skipping squad depth: squads/ directory not found")
        return

    analyzer = SquadAnalyzer(model, trace)

    for squad_file in squads_dir.glob(f"*_{season}.csv"):
        try:
            team = squad_file.stem.replace(f"_{season}", "").replace("_", " ").title()

            squad_df = pd.read_csv(squad_file)

            analysis = analyzer.analyze_squad(squad_df, team, season)

            positions_list = []
            if analysis.position_strength is not None:
                for _, row in analysis.position_strength.iterrows():
                    position_entry = {
                        "position": row["position"],
                        "expected_strength": float(row["expected_strength"]),
                        "depth_score": float(row.get("depth_score", 0)),
                    }

                    if analysis.depth_chart and row["position"] in analysis.depth_chart:
                        players = [
                            {"name": player, "rating": float(rating)}
                            for player, rating in analysis.depth_chart[row["position"]][:3]
                        ]
                        position_entry["top_players"] = players

                    positions_list.append(position_entry)

            squad_data.append({
                "team": team,
                "season": season,
                "overall_strength": float(analysis.overall_strength) if analysis.overall_strength else 0,
                "depth_score": float(analysis.depth_score) if analysis.depth_score else 0,
                "positions": positions_list,
            })

        except Exception as e:
            print(f"    Error processing squad for {squad_file.name}: {e}")
            continue

    with open(output_dir / "squad_depth.json", "w") as f:
        json.dump(squad_data, f, indent=2)


def export_dashboard_data(
    data_dir: Path,
    output_dir: Path,
    checkpoint_name: str | None = None,
    recent_seasons_only: int = 3,
) -> None:
    """
    Export model predictions and rankings to JSON for static dashboard.

    Args:
        data_dir: Path to Rugby-Data directory
        output_dir: Path to output dashboard data
        checkpoint_name: Name of saved model checkpoint (or None to fit new)
        recent_seasons_only: Only export last N seasons
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPORTING DASHBOARD DATA")
    print("=" * 70)

    print("\nLoading data...")
    dataset = MatchDataset(data_dir, fuzzy_match_names=False)
    dataset.load_json_files()

    df = dataset.to_dataframe(played_only=True)
    print(f"Loaded {len(df):,} observations")

    seasons = sorted(df["season"].unique())
    recent_seasons = seasons[-recent_seasons_only:]
    df_recent = df[df["season"].isin(recent_seasons)]
    print(f"Filtering to {recent_seasons_only} recent seasons: {recent_seasons}")
    print(f"Using {len(df_recent):,} observations")

    if checkpoint_name:
        print(f"\nLoading checkpoint: {checkpoint_name}")
        
        # First, load the trace to inspect the model structure
        cache_dir = Path("~/.cache/rugby_ranking").expanduser()
        checkpoint_dir = cache_dir / checkpoint_name
        trace_path = checkpoint_dir / "trace.nc"
        
        import arviz as az
        temp_trace = az.from_netcdf(trace_path)
        
        # Infer model configuration from trace variable names
        has_time_varying = "gamma_team_base_raw" in temp_trace.posterior.data_vars
        
        # Create model with appropriate configuration
        config = ModelConfig(
            include_defense=True,
            separate_kicking_effect=True,
            time_varying_effects=has_time_varying,
        )
        
        model = RugbyModel(config=config)
        fitter = ModelFitter.load(checkpoint_name, model)
        trace = fitter.trace
    else:
        print("\nFitting new model...")
        model.build_joint(df_recent)

        inference_config = InferenceConfig(
            vi_n_iterations=30000,
            vi_method="advi",
        )

        fitter = ModelFitter(model, config=inference_config)
        trace = fitter.fit_vi(progressbar=True, random_seed=42)

        checkpoint_path = fitter.save("dashboard_export")
        print(f"Saved checkpoint to: {checkpoint_path}")

    print("\nExporting rankings and statistics...")

    print("  - Team offensive rankings...")
    offensive_data = []
    for season in recent_seasons:
        for score_type in ["tries", "penalties", "conversions"]:
            rankings = model.get_team_rankings(
                trace=trace,
                season=season,
                score_type=score_type,
                top_n=50,
            )

            for _, row in rankings.iterrows():
                offensive_data.append({
                    "team": row["team"],
                    "season": season,
                    "score_type": score_type,
                    "offense_mean": float(row["effect_mean"]),
                    "offense_std": float(row["effect_std"]),
                    "offense_lower": float(row["effect_lower"]),
                    "offense_upper": float(row["effect_upper"]),
                })

    with open(output_dir / "team_offense.json", "w") as f:
        json.dump(offensive_data, f, indent=2)

    if config.include_defense:
        print("  - Team defensive rankings...")
        defensive_data = []
        for season in recent_seasons:
            for score_type in ["tries", "penalties", "conversions"]:
                rankings = model.get_defensive_rankings(
                    trace=trace,
                    season=season,
                    score_type=score_type,
                    top_n=50,
                )

                for _, row in rankings.iterrows():
                    defensive_data.append({
                        "team": row["team"],
                        "season": season,
                        "score_type": score_type,
                        "defense_mean": float(row["defense_mean"]),
                        "defense_std": float(row["defense_std"]),
                        "defense_lower": float(row["defense_lower"]),
                        "defense_upper": float(row["defense_upper"]),
                    })

        with open(output_dir / "team_defense.json", "w") as f:
            json.dump(defensive_data, f, indent=2)

    print("  - Player rankings...")
    player_data = []
    for score_type in ["tries", "penalties", "conversions"]:
        rankings = model.get_player_rankings(
            trace=trace,
            score_type=score_type,
            top_n=100,
        )

        for _, row in rankings.iterrows():
            player_data.append({
                "player": row["player"],
                "score_type": score_type,
                "effect_mean": float(row["effect_mean"]),
                "effect_std": float(row["effect_std"]),
                "effect_lower": float(row["effect_lower"]),
                "effect_upper": float(row["effect_upper"]),
            })

    with open(output_dir / "player_rankings.json", "w") as f:
        json.dump(player_data, f, indent=2)

    print("  - Match statistics...")
    match_stats = []
    for season in recent_seasons:
        season_df = df_recent[df_recent["season"] == season]

        match_agg = season_df.groupby(["match_id", "team", "opponent"]).agg({
            "team_score": "first",
            "opponent_score": "first",
            "tries": "sum",
            "penalties": "sum",
            "conversions": "sum",
            "date": "first",
            "competition": "first",
        }).reset_index()

        for _, row in match_agg.iterrows():
            match_stats.append({
                "match_id": row["match_id"],
                "season": season,
                "team": row["team"],
                "opponent": row["opponent"],
                "team_score": int(row["team_score"]),
                "opponent_score": int(row["opponent_score"]),
                "team_tries": int(row["tries"]),
                "team_penalties": int(row["penalties"]),
                "team_conversions": int(row["conversions"]),
                "date": row["date"].isoformat(),
                "competition": row["competition"],
            })

    with open(output_dir / "match_stats.json", "w") as f:
        json.dump(match_stats, f, indent=2)

    print("  - Summary statistics...")
    summary = {
        "generated_at": datetime.now().isoformat(),
        "seasons": recent_seasons,
        "total_matches": len(df_recent["match_id"].unique()),
        "total_teams": len(df_recent["team"].unique()),
        "total_players": len(df_recent["player_name"].unique()),
        "competitions": list(df_recent["competition"].unique()),
        "model_config": {
            "include_defense": config.include_defense,
            "separate_kicking_effect": config.separate_kicking_effect,
            "defense_effect_sd": config.defense_effect_sd,
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("  - Team aggregated statistics...")
    team_list = []
    for season in recent_seasons:
        season_df = df_recent[df_recent["season"] == season]

        for team in season_df["team"].unique():
            team_df = season_df[season_df["team"] == team]

            team_list.append({
                "team": team,
                "season": season,
                "matches": len(team_df["match_id"].unique()),
                "total_tries": int(team_df["tries"].sum()),
                "total_penalties": int(team_df["penalties"].sum()),
                "total_conversions": int(team_df["conversions"].sum()),
                "total_points": int(team_df["team_score"].sum()),
                "avg_points_per_match": float(team_df.groupby("match_id")["team_score"].first().mean()),
            })

    with open(output_dir / "team_stats.json", "w") as f:
        json.dump(team_list, f, indent=2)

    export_team_strength_series(model, trace, df_recent, recent_seasons, output_dir)
    export_team_finish_positions(df_recent, recent_seasons, output_dir)

    for season in recent_seasons:
        export_upcoming_predictions(model, trace, df_recent, season, output_dir)

    try:
        match_predictor = MatchPredictor(model, trace)
        for season in recent_seasons:
            export_paths_to_victory(match_predictor, df_recent, season, output_dir)
    except Exception as e:
        print(f"  - Paths to victory (skipped: {e})")

    for season in recent_seasons:
        export_squad_depth(model, trace, season, output_dir)

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nFiles written to: {output_dir}/")
    print("  - team_offense.json")
    print("  - team_defense.json (if defense enabled)")
    print("  - player_rankings.json")
    print("  - match_stats.json")
    print("  - team_stats.json")
    print("  - summary.json")
    print("  - team_strength_series.json")
    print("  - team_finish_positions.json")
    print("  - upcoming_predictions.json")
    print("  - paths_to_victory.json (empty for historical data)")
    print("  - squad_depth.json")
    print("\nNote: Paths to victory analysis requires unplayed fixtures.")
    print("For historical data, this file will be empty.")
    print("\nReady for dashboard deployment!")


if __name__ == "__main__":
    DATA_DIR = Path("../Rugby-Data")
    if not DATA_DIR.exists():
        DATA_DIR = Path("../../Rugby-Data")

    OUTPUT_DIR = Path("dashboard/data")

    export_dashboard_data(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        checkpoint_name="time_model_v1",
        recent_seasons_only=3,
    )
