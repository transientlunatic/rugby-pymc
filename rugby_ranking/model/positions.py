"""
Rugby position groupings and analysis.

Standard rugby positions (1-15):
1. Loosehead Prop
2. Hooker
3. Tighthead Prop
4. Lock
5. Lock
6. Blindside Flanker
7. Openside Flanker
8. Number Eight
9. Scrum-half
10. Fly-half
11. Left Wing
12. Inside Center
13. Outside Center
14. Right Wing
15. Fullback

Positions 16-23 are substitutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import numpy as np


# Position group definitions
FORWARDS = list(range(1, 9))  # 1-8
BACKS = list(range(9, 16))  # 9-15

FRONT_ROW = [1, 2, 3]
SECOND_ROW = [4, 5]
BACK_ROW = [6, 7, 8]

HALF_BACKS = [9, 10]
CENTERS = [12, 13]
BACK_THREE = [11, 14, 15]

# Kickers (primary kicking positions)
PRIMARY_KICKERS = [10, 15]  # Fly-half, Fullback
SECONDARY_KICKERS = [9, 12]  # Scrum-half, Inside center

# Try-scoring positions
HIGH_TRY_SCORERS = [11, 14, 15, 7, 8]  # Wings, fullback, flankers, number 8

# Position names
POSITION_NAMES = {
    1: "Loosehead Prop",
    2: "Hooker",
    3: "Tighthead Prop",
    4: "Lock",
    5: "Lock",
    6: "Blindside Flanker",
    7: "Openside Flanker",
    8: "Number Eight",
    9: "Scrum-half",
    10: "Fly-half",
    11: "Left Wing",
    12: "Inside Center",
    13: "Outside Center",
    14: "Right Wing",
    15: "Fullback",
}

# Short names
POSITION_SHORT_NAMES = {
    1: "LHP",
    2: "HK",
    3: "THP",
    4: "LK",
    5: "LK",
    6: "FL",
    7: "FL",
    8: "N8",
    9: "SH",
    10: "FH",
    11: "W",
    12: "C",
    13: "C",
    14: "W",
    15: "FB",
}

# Position group mappings
POSITION_GROUPS = {
    "forwards": FORWARDS,
    "backs": BACKS,
    "front_row": FRONT_ROW,
    "second_row": SECOND_ROW,
    "back_row": BACK_ROW,
    "half_backs": HALF_BACKS,
    "centers": CENTERS,
    "back_three": BACK_THREE,
    "primary_kickers": PRIMARY_KICKERS,
    "secondary_kickers": SECONDARY_KICKERS,
    "high_try_scorers": HIGH_TRY_SCORERS,
}


@dataclass
class PositionGroup:
    """A grouping of rugby positions."""

    name: str
    positions: list[int]
    description: str


def get_position_group(
    group_name: str,
) -> PositionGroup:
    """
    Get a predefined position group.

    Args:
        group_name: One of 'forwards', 'backs', 'front_row', 'second_row',
                    'back_row', 'half_backs', 'centers', 'back_three',
                    'primary_kickers', 'secondary_kickers', 'high_try_scorers'

    Returns:
        PositionGroup object

    Raises:
        ValueError: If group_name is not recognized
    """
    if group_name not in POSITION_GROUPS:
        raise ValueError(
            f"Unknown position group: {group_name}. "
            f"Available groups: {list(POSITION_GROUPS.keys())}"
        )

    positions = POSITION_GROUPS[group_name]

    descriptions = {
        "forwards": "Forwards (1-8)",
        "backs": "Backs (9-15)",
        "front_row": "Front Row (1-3)",
        "second_row": "Second Row (4-5)",
        "back_row": "Back Row (6-8)",
        "half_backs": "Half-backs (9-10)",
        "centers": "Centers (12-13)",
        "back_three": "Back Three (11, 14, 15)",
        "primary_kickers": "Primary Kickers (10, 15)",
        "secondary_kickers": "Secondary Kickers (9, 12)",
        "high_try_scorers": "High Try Scorers (11, 14, 15, 7, 8)",
    }

    return PositionGroup(
        name=group_name,
        positions=positions,
        description=descriptions[group_name],
    )


def assign_position_group(
    position: int,
    grouping: Literal["forward_back", "detailed"] = "forward_back",
) -> str:
    """
    Assign a position to a group.

    Args:
        position: Position number (1-23)
        grouping: Level of grouping detail

    Returns:
        Group name
    """
    if grouping == "forward_back":
        if position in FORWARDS:
            return "forward"
        elif position in BACKS:
            return "back"
        else:
            return "substitute"

    elif grouping == "detailed":
        if position in FRONT_ROW:
            return "front_row"
        elif position in SECOND_ROW:
            return "second_row"
        elif position in BACK_ROW:
            return "back_row"
        elif position in HALF_BACKS:
            return "half_back"
        elif position in CENTERS:
            return "center"
        elif position in BACK_THREE:
            return "back_three"
        else:
            return "substitute"

    else:
        raise ValueError(f"Unknown grouping: {grouping}")


def add_position_groups(
    df: pd.DataFrame,
    grouping: Literal["forward_back", "detailed"] = "forward_back",
) -> pd.DataFrame:
    """
    Add position group column to dataframe.

    Args:
        df: DataFrame with 'position' column
        grouping: Level of grouping detail

    Returns:
        DataFrame with new 'position_group' column
    """
    df = df.copy()
    df["position_group"] = df["position"].apply(
        lambda p: assign_position_group(p, grouping)
    )
    return df


def aggregate_by_position_group(
    df: pd.DataFrame,
    score_columns: list[str],
    grouping: Literal["forward_back", "detailed"] = "forward_back",
) -> pd.DataFrame:
    """
    Aggregate scoring statistics by position group.

    Args:
        df: Player-match observations
        score_columns: Columns to aggregate (e.g., ['tries', 'penalties'])
        grouping: Level of grouping detail

    Returns:
        DataFrame with aggregated statistics per position group
    """
    df = add_position_groups(df, grouping)

    # Aggregate
    agg_dict = {col: "sum" for col in score_columns}
    agg_dict["exposure"] = "sum"  # Total playing time
    agg_dict["player_name"] = "count"  # Number of observations

    result = df.groupby("position_group").agg(agg_dict).reset_index()
    result.rename(columns={"player_name": "n_observations"}, inplace=True)

    # Compute rates per 80 minutes
    for col in score_columns:
        result[f"{col}_per_80"] = result[col] / (result["exposure"] / 80)

    return result


def filter_by_position_group(
    df: pd.DataFrame,
    group_name: str,
) -> pd.DataFrame:
    """
    Filter dataframe to players in a specific position group.

    Args:
        df: Player-match observations with 'position' column
        group_name: Position group to filter to

    Returns:
        Filtered DataFrame
    """
    group = get_position_group(group_name)
    return df[df["position"].isin(group.positions)].copy()


def get_position_rankings(
    model,
    position_group: str | None = None,
    score_type: str = "tries",
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Get player rankings filtered by position group.

    Args:
        model: Fitted RugbyModel
        position_group: Position group to filter (None = all players)
        score_type: Type of score to rank by
        top_n: Number of top players to return

    Returns:
        DataFrame of player rankings
    """
    # Get all rankings
    rankings = model.get_player_rankings(score_type=score_type, top_n=None)

    if position_group is not None:
        # Filter to positions in group
        group = get_position_group(position_group)

        # Get player positions (most common position)
        if hasattr(model, "_df") and model._df is not None:
            player_positions = (
                model._df.groupby("player_name")["position"]
                .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
                .to_dict()
            )

            # Add position to rankings
            rankings["position"] = rankings["player_name"].map(player_positions)

            # Filter to group
            rankings = rankings[rankings["position"].isin(group.positions)].copy()

    # Return top N
    return rankings.head(top_n).reset_index(drop=True)


def visualize_position_effects(
    model,
    score_type: str = "tries",
    grouping: Literal["individual", "grouped"] = "individual",
) -> tuple:
    """
    Visualize positional effects from the model.

    Args:
        model: Fitted RugbyModel
        score_type: Score type to visualize
        grouping: Show individual positions or grouped

    Returns:
        Tuple of (figure, axis) from matplotlib
    """
    import matplotlib.pyplot as plt

    if model.trace is None:
        raise ValueError("Model must be fitted first")

    # Extract position effects
    theta_var = f"theta_{score_type}"
    if theta_var not in model.trace.posterior:
        raise ValueError(f"Score type '{score_type}' not in model")

    theta = model.trace.posterior[theta_var].values
    theta_mean = theta.mean(axis=(0, 1))  # Average over chains and samples
    theta_std = theta.std(axis=(0, 1))

    if grouping == "individual":
        # Plot individual positions
        positions = list(range(1, 16))
        names = [POSITION_SHORT_NAMES[p] for p in positions]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(15), theta_mean[:15], yerr=theta_std[:15], capsize=5)
        ax.set_xticks(range(15))
        ax.set_xticklabels(names, rotation=45)
        ax.set_xlabel("Position")
        ax.set_ylabel(f"Position Effect (log rate) - {score_type}")
        ax.set_title(f"Positional Effects for {score_type.title()}")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.3)

    else:  # grouped
        # Aggregate by position groups
        groups = ["front_row", "second_row", "back_row", "half_backs", "centers", "back_three"]
        group_means = []
        group_stds = []
        group_labels = []

        for g in groups:
            positions = POSITION_GROUPS[g]
            # Average effects for positions in group
            group_theta = theta_mean[[p - 1 for p in positions]]
            group_means.append(group_theta.mean())
            group_stds.append(group_theta.std())
            group_labels.append(g.replace("_", " ").title())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(groups)), group_means, yerr=group_stds, capsize=5)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(group_labels, rotation=45, ha="right")
        ax.set_xlabel("Position Group")
        ax.set_ylabel(f"Position Effect (log rate) - {score_type}")
        ax.set_title(f"Grouped Positional Effects for {score_type.title()}")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def get_position_name(position: int, short: bool = False) -> str:
    """
    Get the name of a position.

    Args:
        position: Position number (1-23)
        short: Return short name (e.g., 'LHP') instead of full name

    Returns:
        Position name
    """
    if short:
        return POSITION_SHORT_NAMES.get(position, f"Sub{position}")
    else:
        return POSITION_NAMES.get(position, f"Substitute {position}")
