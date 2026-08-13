#!/usr/bin/env python3
"""
Create simplified plate notation diagram for the rugby ranking model.

This version is cleaner and more suitable for blog posts where you want
to show the key structure without all the details.

Requirements:
    pip install daft-pgm matplotlib

Usage:
    python create_simple_plate_diagram.py
"""

try:
    import daft
except ImportError:
    print("Error: daft-pgm not installed.")
    print("Please install it with: pip install daft-pgm")
    exit(1)

import matplotlib.pyplot as plt

# Create a simpler, more compact PGM
pgm = daft.PGM(shape=[6, 7], origin=[0, 0], grid_unit=3)

# ============================================================================
# HYPERPARAMETERS (Top level)
# ============================================================================
pgm.add_node("sigma", r"Hyperparameters", 3, 6.5, aspect=2.2,
             plot_params={"ec": "black", "lw": 1.5}, scale=1.1)

# ============================================================================
# LATENT EFFECTS
# ============================================================================
# Score-type level
pgm.add_node("score_params", r"Score-type params", 3, 5.2, aspect=2.5,
             plot_params={"ec": "black"})

# Player effects
pgm.add_node("player", r"Player ability", 1.5, 3.5, aspect=2,
             plot_params={"ec": "blue", "lw": 1.5})

# Team effects
pgm.add_node("team", r"Team strength", 4.5, 3.5, aspect=2,
             plot_params={"ec": "red", "lw": 1.5})

# ============================================================================
# OBSERVATION LEVEL
# ============================================================================
pgm.add_node("rate", r"Scoring rate", 3, 2, aspect=2,
             plot_params={"ec": "black"}, fixed=True)

pgm.add_node("y", r"Observed scores", 3, 0.5, aspect=2.2,
             observed=True, scale=1.1)

# ============================================================================
# EDGES
# ============================================================================
pgm.add_edge("sigma", "score_params")
pgm.add_edge("sigma", "player")
pgm.add_edge("sigma", "team")

pgm.add_edge("score_params", "rate")
pgm.add_edge("player", "rate")
pgm.add_edge("team", "rate")

pgm.add_edge("rate", "y")

# ============================================================================
# PLATES
# ============================================================================
# Score types
pgm.add_plate([2.3, 4.9, 1.4, 0.8],
              label=r"4 score types",
              label_offset=(5, 5))

# Players
pgm.add_plate([0.8, 3.1, 1.9, 1],
              label=r"$N$ players",
              label_offset=(5, 5))

# Teams
pgm.add_plate([3.8, 3.1, 1.9, 1],
              label=r"$N$ teams",
              label_offset=(5, 5))

# Observations
pgm.add_plate([2.2, 0, 1.6, 2.5],
              label=r"288k observations",
              label_offset=(5, 5))

# ============================================================================
# RENDER
# ============================================================================
pgm.render()

plt.suptitle("Rugby Model: Hierarchical Structure (Simplified)",
             fontsize=14, fontweight='bold')

# Add caption
plt.text(0.5, 0.02,
         "Player ability follows players across teams | Team strength is season-specific",
         transform=plt.gcf().transFigure,
         fontsize=10, ha='center', va='bottom', style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save
for fmt in ['pdf', 'png', 'svg']:
    filename = f"rugby_model_simple.{fmt}"
    params = {'dpi': 300, 'bbox_inches': 'tight'} if fmt == 'png' else {'bbox_inches': 'tight'}
    if fmt in ['png', 'svg']:
        params['facecolor'] = 'white'
    plt.savefig(filename, **params)
    print(f"✓ Saved {filename}")

plt.show()
