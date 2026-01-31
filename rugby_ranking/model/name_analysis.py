"""
Tools for analyzing and reviewing player name matching.

Helps identify:
- Names that were merged (and should they have been?)
- Potential duplicates that weren't merged
- Optimal similarity thresholds
- Manual corrections needed
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

import pandas as pd
from difflib import SequenceMatcher


@dataclass
class NameMerge:
    """Represents a merge of multiple name variations to a canonical name."""

    canonical: str
    variations: list[str]
    similarity_scores: list[float]
    teams: set[str]
    n_appearances: int
    should_merge: Optional[bool] = None  # For manual review


def analyze_merged_names(dataset) -> pd.DataFrame:
    """
    Analyze which names were merged by fuzzy matching.

    Args:
        dataset: MatchDataset with fuzzy matching enabled

    Returns:
        DataFrame with merged name analysis
    """
    if not hasattr(dataset, "_name_matcher") or dataset._name_matcher is None:
        raise ValueError("Dataset does not have fuzzy matching enabled")

    merged = dataset.get_merged_names()

    if not merged:
        return pd.DataFrame(columns=["canonical", "variation", "similarity", "count"])

    # Convert to structured format
    records = []
    for canonical, variations in merged.items():
        for var_name, similarity in variations:
            records.append(
                {
                    "canonical": canonical,
                    "variation": var_name,
                    "similarity": similarity,
                }
            )

    df = pd.DataFrame(records)

    # Add appearance counts
    if hasattr(dataset, "observations") and dataset.observations:
        name_counts = defaultdict(int)
        for obs in dataset.observations:
            name_counts[obs.player_name] += 1

        df["canonical_count"] = df["canonical"].map(name_counts)
        df["variation_count"] = df["variation"].map(name_counts)
    else:
        df["canonical_count"] = 0
        df["variation_count"] = 0

    return df.sort_values("similarity")


def find_potential_duplicates(
    dataset,
    min_similarity: float = 0.75,
    max_similarity: float = 0.85,
    min_appearances: int = 10,
) -> pd.DataFrame:
    """
    Find player names that are similar but were NOT merged.

    These might be genuine duplicates that need investigation.

    Args:
        dataset: MatchDataset
        min_similarity: Minimum similarity to consider
        max_similarity: Maximum similarity (below current threshold)
        min_appearances: Only consider players with this many appearances

    Returns:
        DataFrame of potential duplicates
    """
    # Get all unique player names
    if hasattr(dataset, "observations") and dataset.observations:
        name_counts = defaultdict(int)
        name_teams = defaultdict(set)

        for obs in dataset.observations:
            name_counts[obs.player_name] += 1
            name_teams[obs.player_name].add(obs.team)

        # Filter to players with sufficient appearances
        candidate_names = [
            name for name, count in name_counts.items() if count >= min_appearances
        ]
    else:
        # Fall back to dataframe if available
        df = dataset.to_dataframe()
        name_counts = df["player_name"].value_counts()
        name_teams = df.groupby("player_name")["team"].apply(set)

        candidate_names = name_counts[name_counts >= min_appearances].index.tolist()

    # Check all pairs for similarity
    potential_dupes = []

    for i, name1 in enumerate(candidate_names):
        for name2 in candidate_names[i + 1 :]:
            similarity = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()

            if min_similarity <= similarity <= max_similarity:
                # Check if they share teams (increases likelihood of being same person)
                shared_teams = name_teams[name1] & name_teams[name2]
                team_overlap = len(shared_teams) > 0

                potential_dupes.append(
                    {
                        "name1": name1,
                        "name2": name2,
                        "similarity": similarity,
                        "name1_appearances": name_counts[name1],
                        "name2_appearances": name_counts[name2],
                        "shared_teams": ", ".join(sorted(shared_teams))
                        if shared_teams
                        else "",
                        "team_overlap": team_overlap,
                        "name1_teams": ", ".join(sorted(name_teams[name1])),
                        "name2_teams": ", ".join(sorted(name_teams[name2])),
                    }
                )

    df = pd.DataFrame(potential_dupes)

    if len(df) > 0:
        # Sort by similarity (descending) and team overlap
        df = df.sort_values(["team_overlap", "similarity"], ascending=[False, False])

    return df


def review_merges(
    merged_df: pd.DataFrame,
    min_similarity: float = 0.85,
    interactive: bool = False,
) -> pd.DataFrame:
    """
    Review merged names and flag suspicious merges.

    Args:
        merged_df: Output from analyze_merged_names()
        min_similarity: Flag merges below this threshold
        interactive: If True, prompt for manual review (requires user input)

    Returns:
        DataFrame with review annotations
    """
    # Flag low-confidence merges
    merged_df["flagged"] = merged_df["similarity"] < min_similarity

    # Add suggested action
    def suggest_action(row):
        if row["similarity"] >= 0.95:
            return "auto_accept"
        elif row["similarity"] >= min_similarity:
            return "review"
        else:
            return "likely_incorrect"

    merged_df["suggestion"] = merged_df.apply(suggest_action, axis=1)

    if interactive:
        print("\n" + "=" * 70)
        print("INTERACTIVE MERGE REVIEW")
        print("=" * 70)
        print(
            "\nReview flagged merges. For each, respond with:"
            "\n  y/yes = correct merge"
            "\n  n/no = incorrect merge"
            "\n  s/skip = skip for now"
            "\n  q/quit = stop reviewing"
            "\n"
        )

        flagged = merged_df[merged_df["flagged"]].copy()

        for idx, row in flagged.iterrows():
            print(f"\n{'-' * 70}")
            print(f"Canonical: {row['canonical']}")
            print(f"Variation: {row['variation']}")
            print(f"Similarity: {row['similarity']:.3f}")
            print(f"Appearances: {row['canonical_count']} / {row['variation_count']}")

            while True:
                response = input("Accept this merge? (y/n/s/q): ").lower().strip()

                if response in ["y", "yes"]:
                    merged_df.at[idx, "manual_review"] = "accept"
                    break
                elif response in ["n", "no"]:
                    merged_df.at[idx, "manual_review"] = "reject"
                    break
                elif response in ["s", "skip"]:
                    merged_df.at[idx, "manual_review"] = "skip"
                    break
                elif response in ["q", "quit"]:
                    print("\nExiting review.")
                    return merged_df
                else:
                    print("Invalid response. Try again.")

    return merged_df


def generate_correction_dict(merged_df: pd.DataFrame) -> dict[str, str]:
    """
    Generate a manual correction dictionary from reviewed merges.

    Use this to create entries for PlayerNameMatcher's manual_corrections dict.

    Args:
        merged_df: DataFrame from review_merges() with manual_review column

    Returns:
        Dictionary of {variation: canonical} for accepted merges
    """
    if "manual_review" not in merged_df.columns:
        raise ValueError("DataFrame must have 'manual_review' column (run review_merges first)")

    accepted = merged_df[merged_df["manual_review"] == "accept"]

    corrections = {}
    for _, row in accepted.iterrows():
        corrections[row["variation"]] = row["canonical"]

    return corrections


def analyze_threshold_impact(
    dataset,
    thresholds: list[float] = None,
) -> pd.DataFrame:
    """
    Analyze how different similarity thresholds affect name merging.

    Args:
        dataset: MatchDataset
        thresholds: List of thresholds to test (default: 0.75 to 0.95)

    Returns:
        DataFrame showing merge statistics for each threshold
    """
    if thresholds is None:
        thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]

    # Get all player names
    if hasattr(dataset, "observations") and dataset.observations:
        all_names = [obs.player_name for obs in dataset.observations]
    else:
        df = dataset.to_dataframe()
        all_names = df["player_name"].tolist()

    results = []

    for threshold in thresholds:
        # Count how many names would be merged at this threshold
        # This is a simplified analysis - full version would re-run matching

        unique_names = set(all_names)
        n_original = len(unique_names)

        # Estimate merges (rough approximation)
        # Real implementation would instantiate PlayerNameMatcher with this threshold

        results.append(
            {
                "threshold": threshold,
                "original_names": n_original,
                "note": "Full implementation requires re-running matcher",
            }
        )

    return pd.DataFrame(results)


def get_name_variations(dataset, player_name: str) -> pd.DataFrame:
    """
    Get all variations of a specific player's name in the dataset.

    Args:
        dataset: MatchDataset
        player_name: Player name to search for

    Returns:
        DataFrame of name variations found
    """
    if hasattr(dataset, "observations") and dataset.observations:
        # Find similar names
        all_names = {obs.player_name for obs in dataset.observations}
    else:
        df = dataset.to_dataframe()
        all_names = set(df["player_name"].unique())

    # Find names similar to the query
    similar = []
    for name in all_names:
        similarity = SequenceMatcher(None, player_name.lower(), name.lower()).ratio()
        if similarity > 0.7:  # Generous threshold for exploration
            similar.append({"name": name, "similarity": similarity})

    df = pd.DataFrame(similar).sort_values("similarity", ascending=False)

    # Add appearance counts
    if hasattr(dataset, "observations") and dataset.observations:
        name_counts = defaultdict(int)
        for obs in dataset.observations:
            name_counts[obs.player_name] += 1
        df["appearances"] = df["name"].map(name_counts)
    else:
        dataset_df = dataset.to_dataframe()
        counts = dataset_df["player_name"].value_counts()
        df["appearances"] = df["name"].map(counts)

    return df


def export_merge_report(
    dataset,
    output_path: str,
    include_potential_duplicates: bool = True,
) -> None:
    """
    Export comprehensive merge analysis to CSV/Excel.

    Args:
        dataset: MatchDataset
        output_path: Path to save report (supports .csv, .xlsx)
        include_potential_duplicates: Include duplicate analysis
    """
    from pathlib import Path

    output_path = Path(output_path)

    # Get merged names
    merged_df = analyze_merged_names(dataset)

    if output_path.suffix == ".xlsx":
        # Excel with multiple sheets
        with pd.ExcelWriter(output_path) as writer:
            merged_df.to_excel(writer, sheet_name="Merged Names", index=False)

            if include_potential_duplicates:
                dupes_df = find_potential_duplicates(dataset)
                dupes_df.to_excel(writer, sheet_name="Potential Duplicates", index=False)

        print(f"Report exported to {output_path}")
    else:
        # CSV (merged names only)
        merged_df.to_csv(output_path, index=False)
        print(f"Merged names exported to {output_path}")

        if include_potential_duplicates:
            dupes_path = output_path.with_name(
                output_path.stem + "_duplicates" + output_path.suffix
            )
            dupes_df = find_potential_duplicates(dataset)
            dupes_df.to_csv(dupes_path, index=False)
            print(f"Potential duplicates exported to {dupes_path}")
