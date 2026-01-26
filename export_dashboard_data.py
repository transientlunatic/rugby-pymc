"""Export model data to static JSON files for dashboard deployment."""

import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from rugby_ranking.model.data import MatchDataset
from rugby_ranking.model.core import RugbyModel, ModelConfig
from rugby_ranking.model.inference import ModelFitter, InferenceConfig


def export_dashboard_data(
    data_dir: Path,
    output_dir: Path,
    checkpoint_name: str | None = None,
    recent_seasons_only: int = 3,
):
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

    # Load data
    print("\nLoading data...")
    dataset = MatchDataset(data_dir, fuzzy_match_names=True)
    dataset.load_json_files()

    df = dataset.to_dataframe(played_only=True)
    print(f"Loaded {len(df):,} observations")

    # Filter to recent seasons
    seasons = sorted(df['season'].unique())
    recent_seasons = seasons[-recent_seasons_only:]
    df_recent = df[df['season'].isin(recent_seasons)]
    print(f"Filtering to {recent_seasons_only} recent seasons: {recent_seasons}")
    print(f"Using {len(df_recent):,} observations")

    # Load or fit model
    config = ModelConfig(
        include_defense=True,
        separate_kicking_effect=True
    )

    model = RugbyModel(config=config)

    if checkpoint_name:
        print(f"\nLoading checkpoint: {checkpoint_name}")
        fitter = ModelFitter.load(checkpoint_name, model)
        trace = fitter.trace
    else:
        print("\nFitting new model...")
        model.build_joint(df_recent)

        inference_config = InferenceConfig(
            vi_n_iterations=30000,
            vi_method="advi"
        )

        fitter = ModelFitter(model, config=inference_config)
        trace = fitter.fit_vi(progressbar=True, random_seed=42)

        # Save checkpoint
        checkpoint_path = fitter.save("dashboard_export")
        print(f"Saved checkpoint to: {checkpoint_path}")

    # Export data
    print("\nExporting rankings and statistics...")

    # 1. Team offensive rankings by season
    print("  - Team offensive rankings...")
    offensive_data = []
    for season in recent_seasons:
        for score_type in ["tries", "penalties", "conversions"]:
            rankings = model.get_team_rankings(
                trace=trace,
                season=season,
                score_type=score_type,
                top_n=50
            )

            for _, row in rankings.iterrows():
                offensive_data.append({
                    "team": row["team"],
                    "season": season,
                    "score_type": score_type,
                    "offense_mean": float(row["effect_mean"]),
                    "offense_std": float(row["effect_std"]),
                    "offense_lower": float(row["effect_lower"]),
                    "offense_upper": float(row["effect_upper"])
                })

    with open(output_dir / "team_offense.json", "w") as f:
        json.dump(offensive_data, f, indent=2)

    # 2. Team defensive rankings by season
    if config.include_defense:
        print("  - Team defensive rankings...")
        defensive_data = []
        for season in recent_seasons:
            for score_type in ["tries", "penalties", "conversions"]:
                rankings = model.get_defensive_rankings(
                    trace=trace,
                    season=season,
                    score_type=score_type,
                    top_n=50
                )

                for _, row in rankings.iterrows():
                    defensive_data.append({
                        "team": row["team"],
                        "season": season,
                        "score_type": score_type,
                        "defense_mean": float(row["defense_mean"]),
                        "defense_std": float(row["defense_std"]),
                        "defense_lower": float(row["defense_lower"]),
                        "defense_upper": float(row["defense_upper"])
                    })

        with open(output_dir / "team_defense.json", "w") as f:
            json.dump(defensive_data, f, indent=2)

    # 3. Player rankings
    print("  - Player rankings...")
    player_data = []
    for score_type in ["tries", "penalties", "conversions"]:
        rankings = model.get_player_rankings(
            trace=trace,
            score_type=score_type,
            top_n=100
        )

        for _, row in rankings.iterrows():
            player_data.append({
                "player": row["player"],
                "score_type": score_type,
                "effect_mean": float(row["effect_mean"]),
                "effect_std": float(row["effect_std"]),
                "effect_lower": float(row["effect_lower"]),
                "effect_upper": float(row["effect_upper"])
            })

    with open(output_dir / "player_rankings.json", "w") as f:
        json.dump(player_data, f, indent=2)

    # 4. Match statistics
    print("  - Match statistics...")
    match_stats = []
    for season in recent_seasons:
        season_df = df_recent[df_recent['season'] == season]

        # Aggregate by match
        match_agg = season_df.groupby(['match_id', 'team', 'opponent']).agg({
            'team_score': 'first',
            'opponent_score': 'first',
            'tries': 'sum',
            'penalties': 'sum',
            'conversions': 'sum',
            'date': 'first',
            'competition': 'first'
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
                "competition": row["competition"]
            })

    with open(output_dir / "match_stats.json", "w") as f:
        json.dump(match_stats, f, indent=2)

    # 5. Summary statistics
    print("  - Summary statistics...")
    summary = {
        "generated_at": datetime.now().isoformat(),
        "seasons": recent_seasons,
        "total_matches": len(df_recent['match_id'].unique()),
        "total_teams": len(df_recent['team'].unique()),
        "total_players": len(df_recent['player_name'].unique()),
        "competitions": list(df_recent['competition'].unique()),
        "model_config": {
            "include_defense": config.include_defense,
            "separate_kicking_effect": config.separate_kicking_effect,
            "defense_effect_sd": config.defense_effect_sd
        }
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 6. Team list with aggregated stats
    print("  - Team aggregated statistics...")
    team_list = []
    for season in recent_seasons:
        season_df = df_recent[df_recent['season'] == season]

        for team in season_df['team'].unique():
            team_df = season_df[season_df['team'] == team]

            team_list.append({
                "team": team,
                "season": season,
                "matches": len(team_df['match_id'].unique()),
                "total_tries": int(team_df['tries'].sum()),
                "total_penalties": int(team_df['penalties'].sum()),
                "total_conversions": int(team_df['conversions'].sum()),
                "total_points": int(team_df['team_score'].sum()),
                "avg_points_per_match": float(team_df.groupby('match_id')['team_score'].first().mean())
            })

    with open(output_dir / "team_stats.json", "w") as f:
        json.dump(team_list, f, indent=2)

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
    print("\nReady for dashboard deployment!")


if __name__ == "__main__":
    # Configuration
    DATA_DIR = Path("../Rugby-Data")
    if not DATA_DIR.exists():
        DATA_DIR = Path("../../Rugby-Data")

    OUTPUT_DIR = Path("dashboard/data")

    # Export data
    export_dashboard_data(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        checkpoint_name="time_model_v1",  # Set to checkpoint name to load existing model
        recent_seasons_only=3
    )
