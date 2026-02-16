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
from rugby_ranking.model.data_utils import quick_standings, prepare_season_data


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
    dataset,
    output_dir: Path,
) -> None:
    """Export predictions for actual upcoming/unplayed matches."""
    print("  - Upcoming match predictions...")

    match_predictor = MatchPredictor(model, trace)
    predictions_data = []

    # Get actual unplayed matches from the dataset
    unplayed_matches = dataset.get_unplayed_matches()
    
    if not unplayed_matches:
        print("    No upcoming matches found")
        with open(output_dir / "upcoming_predictions.json", "w") as f:
            json.dump([], f, indent=2)
        return
    
    print(f"    Found {len(unplayed_matches)} upcoming matches")
    
    # Limit to next 50 matches to keep file size reasonable
    for match in unplayed_matches[:50]:
        try:
            pred = match_predictor.predict_teams_only(
                home_team=match.home_team,
                away_team=match.away_team,
                season=match.season,
                n_samples=500,
            )

            predictions_data.append({
                "date": match.date.isoformat() if hasattr(match.date, "isoformat") else str(match.date),
                "home_team": match.home_team,
                "away_team": match.away_team,
                "season": match.season,
                "competition": match.competition,
                "home_score_pred": float(pred.home.mean),
                "away_score_pred": float(pred.away.mean),
                "home_win_prob": float(pred.home_win_prob),
                "away_win_prob": float(pred.away_win_prob),
                "draw_prob": float(pred.draw_prob),
            })
        except Exception as e:
            print(f"    Skipping prediction for {match.home_team} vs {match.away_team}: {e}")
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
                            {"player": player, "rating": float(rating)}
                            for player, rating in analysis.depth_chart[row["position"]][:3]
                        ]
                        position_entry["players"] = players

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


def export_league_table(
    dataset: MatchDataset,
    season: str,
    competition: str,
    output_dir: Path,
    top_n: int = 20,
) -> None:
    """Export league table/standings for a competition and season."""
    print(f"  - League table: {competition} {season}...")
    try:
        standings = quick_standings(dataset, season=season, competition=competition, top_n=top_n)
        standings_json = standings.to_dict(orient="records")
        fname = f"league_table_{competition.replace(' ', '_')}_{season}.json"
        with open(output_dir / fname, "w") as f:
            json.dump(standings_json, f, indent=2)
    except Exception as e:
        print(f"    [WARNING] Skipping league table export for {competition} {season}: {e}")


def export_season_prediction(
    model: RugbyModel,
    trace,
    dataset: MatchDataset,
    season: str,
    competition: str,
    output_dir: Path,
    n_simulations: int = 1000,
    playoff_spots: int = 8,
) -> None:
    """Export season prediction (position probabilities, playoff probabilities, etc)."""
    print(f"  - Season prediction: {competition} {season}...")
    try:
        played_matches, remaining_fixtures = prepare_season_data(
            dataset, season=season, competition=competition, include_tries=True
        )
        required_cols = ["team", "opponent", "score", "opponent_score", "tries"]
        if played_matches.empty or not all(col in played_matches.columns for col in required_cols):
            print(f"    [WARNING] Skipping season prediction export for {competition} {season}: missing or empty match data.")
            return
        
        match_predictor = MatchPredictor(model, trace)
        season_predictor = SeasonPredictor(
            match_predictor=match_predictor, competition=competition, playoff_spots=playoff_spots
        )
        season_pred = season_predictor.predict_season(
            played_matches=played_matches,
            remaining_fixtures=remaining_fixtures,
            season=season,
            n_simulations=n_simulations,
        )
        
        if season_pred.position_probabilities is not None:
            pos_probs = season_pred.position_probabilities.reset_index().rename(columns={"index": "team"})
            fname = f"season_position_probs_{competition.replace(' ', '_')}_{season}.json"
            with open(output_dir / fname, "w") as f:
                json.dump(pos_probs.to_dict(orient="records"), f, indent=2)
        
        if season_pred.playoff_probabilities is not None:
            fname = f"season_playoff_probs_{competition.replace(' ', '_')}_{season}.json"
            with open(output_dir / fname, "w") as f:
                json.dump(season_pred.playoff_probabilities.to_dict(orient="records"), f, indent=2)
        
        if season_pred.predicted_standings is not None:
            fname = f"season_predicted_standings_{competition.replace(' ', '_')}_{season}.json"
            with open(output_dir / fname, "w") as f:
                json.dump(season_pred.predicted_standings.to_dict(orient="records"), f, indent=2)
    except Exception as e:
        print(f"    [WARNING] Skipping season prediction export for {competition} {season}: {e}")


def export_team_heatmap(
    df: pd.DataFrame,
    season: str,
    competition: str,
    output_dir: Path,
) -> None:
    """Export team-vs-team heatmap for a competition and season."""
    print(f"  - Team heatmap: {competition} {season}...")
    season_df = df[(df["season"] == season) & (df["competition"] == competition)]
    teams = sorted(season_df["team"].unique())
    if not teams:
        print(f"    [WARNING] No teams found for {competition} {season}, skipping heatmap export.")
        return
    
    matrix = np.zeros((len(teams), len(teams)))
    for i, team_i in enumerate(teams):
        for j, team_j in enumerate(teams):
            if i == j:
                matrix[i, j] = 0
            else:
                matches = season_df[(season_df["team"] == team_i) & (season_df["opponent"] == team_j)]
                if not matches.empty:
                    avg_diff = (matches["team_score"] - matches["opponent_score"]).mean()
                    matrix[i, j] = avg_diff
                else:
                    matrix[i, j] = None
    
    matrix_json = [[(None if (v is None or np.isnan(v)) else float(v)) for v in row] for row in matrix]
    out = {"teams": teams, "matrix": matrix_json}
    fname = f"team_heatmap_{competition.replace(' ', '_')}_{season}.json"
    with open(output_dir / fname, "w") as f:
        json.dump(out, f, indent=2)


def export_matches_index(
    dataset: MatchDataset,
    recent_seasons: list,
    output_dir: Path,
) -> None:
    """Export a lightweight match index for the blog matches table."""
    print("  - Matches index (blog)...")
    matches_index = []

    for match in dataset.matches:
        if match.season not in recent_seasons:
            continue
        if not match.is_played:
            continue

        matches_index.append({
            "id": match.match_id,
            "date": match.date.strftime("%Y-%m-%d"),
            "home": match.home_team,
            "away": match.away_team,
            "home_score": match.home_score,
            "away_score": match.away_score,
            "competition": match.competition,
            "season": match.season,
            "stadium": match.stadium or None,
            "attendance": match.attendance,
        })

    matches_index.sort(key=lambda m: m["date"], reverse=True)

    with open(output_dir / "matches_index.json", "w") as f:
        json.dump(matches_index, f)
    print(f"    Wrote {len(matches_index)} matches")


def export_match_details(
    dataset: MatchDataset,
    recent_seasons: list,
    output_dir: Path,
) -> None:
    """Export per-competition-season match detail files with lineups and events."""
    print("  - Match details (blog)...")
    matches_dir = output_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    def _flatten_lineup(lineup: dict) -> list:
        result = []
        for pos_str, player in lineup.items():
            try:
                pos = int(pos_str)
            except (ValueError, TypeError):
                pos = 0
            result.append({
                "pos": pos,
                "name": player.get("name", player) if isinstance(player, dict) else str(player),
                "on": player.get("on", []) if isinstance(player, dict) else [],
                "off": player.get("off", []) if isinstance(player, dict) else [],
                "yellows": player.get("yellows", []) if isinstance(player, dict) else [],
                "reds": player.get("reds", []) if isinstance(player, dict) else [],
            })
        result.sort(key=lambda p: p["pos"])
        return result

    groups: dict[str, dict] = {}
    for match in dataset.matches:
        if match.season not in recent_seasons:
            continue
        if not match.is_played:
            continue

        key = f"{match.competition}_{match.season}"
        idx = match.match_id.rsplit("_", 1)[-1]

        detail = {
            "home": {
                "team": match.home_team,
                "score": match.home_score,
                "lineup": _flatten_lineup(match.home_lineup),
                "events": match.home_scores,
            },
            "away": {
                "team": match.away_team,
                "score": match.away_score,
                "lineup": _flatten_lineup(match.away_lineup),
                "events": match.away_scores,
            },
            "round": match.round,
            "round_type": match.round_type,
            "stadium": match.stadium or None,
            "date": match.date.isoformat(),
            "attendance": match.attendance,
        }

        if key not in groups:
            groups[key] = {}
        groups[key][idx] = detail

    for key, data in groups.items():
        with open(matches_dir / f"{key}.json", "w") as f:
            json.dump(data, f)

    print(f"    Wrote {len(groups)} detail files")


def export_player_profiles(
    dataset: MatchDataset,
    player_rankings_data: list[dict],
    recent_seasons: list,
    output_dir: Path,
) -> None:
    """Export enriched player profiles combining match stats with model rankings."""
    print("  - Player profiles (blog)...")
    from collections import defaultdict

    player_stats: dict[str, dict] = defaultdict(lambda: {
        "matches": 0,
        "seasons": set(),
        "teams_by_season": {},
        "last_date": "",
        "yellows": 0,
        "reds": 0,
        "tries_scored": 0,
        "conversions_scored": 0,
        "penalties_scored": 0,
    })

    for match in dataset.matches:
        if match.season not in recent_seasons:
            continue
        if not match.is_played:
            continue

        date_str = match.date.strftime("%Y-%m-%d")

        for side, lineup, scores in [
            ("home", match.home_lineup, match.home_scores),
            ("away", match.away_lineup, match.away_scores),
        ]:
            team = match.home_team if side == "home" else match.away_team

            for pos_str, player in lineup.items():
                if not isinstance(player, dict):
                    continue
                name = player.get("name", "")
                if not name:
                    continue

                ps = player_stats[name]
                ps["matches"] += 1
                ps["seasons"].add(match.season)
                current = ps["teams_by_season"].get(match.season)
                if current is None or date_str > ps.get(f"_last_{match.season}", ""):
                    ps["teams_by_season"][match.season] = team
                    ps[f"_last_{match.season}"] = date_str
                if date_str > ps["last_date"]:
                    ps["last_date"] = date_str
                ps["yellows"] += len(player.get("yellows", []))
                ps["reds"] += len(player.get("reds", []))

            for ev in scores:
                player_name = ev.get("player")
                if not player_name or player_name not in player_stats:
                    continue
                ev_type = ev.get("type", "")
                if ev_type == "Try":
                    player_stats[player_name]["tries_scored"] += 1
                elif ev_type == "Conversion":
                    player_stats[player_name]["conversions_scored"] += 1
                elif ev_type == "Penalty":
                    player_stats[player_name]["penalties_scored"] += 1

    rank_map: dict[str, dict] = defaultdict(dict)
    for r in player_rankings_data:
        rank_map[r["player"]][r["score_type"]] = r

    players = []
    for name, stats in player_stats.items():
        seasons = sorted(stats["seasons"])
        teams_by_season = {s: stats["teams_by_season"].get(s, "") for s in seasons}
        current_team = teams_by_season.get(seasons[-1], "") if seasons else ""

        row = {
            "name": name,
            "matches": stats["matches"],
            "seasons": seasons,
            "teams_by_season": teams_by_season,
            "current_team": current_team,
            "yellows": stats["yellows"],
            "reds": stats["reds"],
            "tries_scored": stats["tries_scored"],
        }

        if name in rank_map:
            t = rank_map[name].get("tries")
            row["attacking"] = round(t["effect_mean"], 3) if t else None
            c = rank_map[name].get("conversions")
            p = rank_map[name].get("penalties")
            kicking_vals = [x["effect_mean"] for x in [c, p] if x]
            row["kicking"] = round(sum(kicking_vals) / len(kicking_vals), 3) if kicking_vals else None
        else:
            row["attacking"] = None
            row["kicking"] = None

        players.append(row)

    players.sort(key=lambda p: p["attacking"] if p["attacking"] is not None else -999, reverse=True)

    with open(output_dir / "players.json", "w") as f:
        json.dump(players, f)
    print(f"    Wrote {len(players)} players")


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

    if len(df) == 0:
        print("\nERROR: No player-level observations found!")
        print("The data files do not contain player lineups with positions and statistics.")
        print("This model requires player-level data, not just match results.")
        print("\nData files like 'international.json' only have match results with empty lineups.")
        print("You need files like 'six_nations_2025_adapted.json' with complete player information.")
        return

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
        
        # Create model with default configuration
        config = ModelConfig(
            include_defense=True,
            separate_kicking_effect=True,
            time_varying_effects=False,
        )
        
        model = RugbyModel(config=config)
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

    # Export league tables and season predictions for each competition/season
    # Note: Season predictions only work for league competitions with bonus point rules
    # Skip for international tournaments (Six Nations, World Cup, etc.)
    LEAGUE_COMPETITIONS = {
        'urc', 'celtic', 'premiership', 'top14', 'pro-d2',
        'euro-champions', 'euro-challenge', 'championship'
    }

    competitions = df_recent["competition"].unique()
    for season in recent_seasons:
        for competition in competitions:
            comp_season_df = df_recent[
                (df_recent["season"] == season) &
                (df_recent["competition"] == competition)
            ]
            if len(comp_season_df) > 0:
                try:
                    export_league_table(dataset, season, competition, output_dir)

                    # Only export season predictions for league competitions
                    if competition.lower() in LEAGUE_COMPETITIONS:
                        export_season_prediction(
                            model, trace, dataset, season, competition, output_dir
                        )
                    else:
                        print(
                            f"  - Season prediction: {competition} {season}... "
                            f"(skipped - international tournament)"
                        )

                    export_team_heatmap(df_recent, season, competition, output_dir)
                except Exception as e:
                    print(f"    Error exporting {competition} {season}: {e}")

    # Export upcoming predictions using actual unplayed matches
    export_upcoming_predictions(model, trace, dataset, output_dir)

    try:
        match_predictor = MatchPredictor(model, trace)
        for season in recent_seasons:
            export_paths_to_victory(match_predictor, df_recent, season, output_dir)
    except Exception as e:
        print(f"  - Paths to victory (skipped: {e})")

    for season in recent_seasons:
        export_squad_depth(model, trace, season, output_dir)

    # Export blog-specific data files
    print("\nExporting blog data files...")
    export_matches_index(dataset, recent_seasons, output_dir)
    export_match_details(dataset, recent_seasons, output_dir)
    export_player_profiles(dataset, player_data, recent_seasons, output_dir)

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
    print("  - matches_index.json (blog)")
    print("  - matches/*.json (blog)")
    print("  - players.json (blog)")
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
