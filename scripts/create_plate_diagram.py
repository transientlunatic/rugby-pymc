#!/usr/bin/env python3
"""
Create plate notation diagram for the rugby ranking model using Daft.

This generates publication-quality figures showing the hierarchical structure
of the Bayesian model.

Requirements:
    pip install daft-pgm matplotlib

Usage:
    python create_plate_diagram.py

Outputs:
    - rugby_model_plate_diagram.pdf (high quality, for papers)
    - rugby_model_plate_diagram.png (for web/blog)
    - rugby_model_plate_diagram.svg (scalable, for web)
"""

try:
    import daft
except ImportError:
    print("Error: daft-pgm not installed.")
    print("Please install it with: pip install daft-pgm")
    exit(1)

import matplotlib.pyplot as plt
from matplotlib import rc

# Use LaTeX for better typography (optional - comment out if LaTeX not installed)
# rc("font", family="serif", size=12)
# rc("text", usetex=True)

# Create the PGM with appropriate dimensions
# Grid is [width, height], origin is bottom-left corner
pgm = daft.PGM(shape=[8, 10], origin=[0, 0], grid_unit=2.5)

# ============================================================================
# HYPERPARAMETERS (Top level - Global)
# ============================================================================
pgm.add_node("sigma_player_try", r"$\sigma_{\beta^{\mathrm{try}}}$", 1, 9,
             aspect=1.8, plot_params={"ec": "black"})
pgm.add_node("sigma_player_kick", r"$\sigma_{\beta^{\mathrm{kick}}}$", 2.5, 9,
             aspect=1.8, plot_params={"ec": "black"})
pgm.add_node("sigma_team", r"$\sigma_{\gamma}$", 4.5, 9,
             aspect=1.3, plot_params={"ec": "black"})
pgm.add_node("sigma_defense", r"$\sigma_{\delta}$", 6, 9,
             aspect=1.3, plot_params={"ec": "black"})

# ============================================================================
# SCORE-TYPE LEVEL PARAMETERS (s = 1..4)
# ============================================================================
# Intercepts and fixed effects
pgm.add_node("alpha", r"$\alpha_s$", 1, 7.5, aspect=1.2,
             plot_params={"ec": "black"})
pgm.add_node("eta", r"$\eta_s$", 2, 7.5, aspect=1.2,
             plot_params={"ec": "black"})

# Loading factors (how much each score type loads on latent factors)
pgm.add_node("lambda_try", r"$\lambda^{\mathrm{try}}_s$", 3.5, 7.5,
             aspect=1.6, plot_params={"ec": "black"})
pgm.add_node("lambda_kick", r"$\lambda^{\mathrm{kick}}_s$", 5, 7.5,
             aspect=1.6, plot_params={"ec": "black"})
pgm.add_node("lambda_team", r"$\lambda^{\gamma}_s$", 6.5, 7.5,
             aspect=1.4, plot_params={"ec": "black"})

# Position effects (nested within score type)
pgm.add_node("theta", r"$\theta_{s,k}$", 2.5, 6.2, aspect=1.3,
             plot_params={"ec": "black"})

# ============================================================================
# PLAYER LEVEL (i = 1..N_players)
# ============================================================================
pgm.add_node("beta_try", r"$\beta^{\mathrm{try}}_i$", 1.5, 4.5, aspect=1.5,
             plot_params={"ec": "black"})
pgm.add_node("beta_kick", r"$\beta^{\mathrm{kick}}_i$", 3, 4.5, aspect=1.5,
             plot_params={"ec": "black"})

# ============================================================================
# TEAM-SEASON LEVEL ((j,t) = 1..N_team_seasons)
# ============================================================================
pgm.add_node("gamma", r"$\gamma_{j,t}$", 5, 4.5, aspect=1.3,
             plot_params={"ec": "black"})
pgm.add_node("delta", r"$\delta_{j,t}$", 6.5, 4.5, aspect=1.3,
             plot_params={"ec": "black"})

# ============================================================================
# OBSERVATION LEVEL
# ============================================================================
# Linear predictor (deterministic node)
pgm.add_node("lambda", r"$\lambda_{i,m,s}$", 4, 2.5, aspect=1.5,
             plot_params={"ec": "black"}, fixed=True)

# Observed data
pgm.add_node("y", r"$y_{i,m,s}$", 4, 1, aspect=1.3, observed=True)

# ============================================================================
# EDGES (Dependencies)
# ============================================================================
# Hyperparameters to player effects
pgm.add_edge("sigma_player_try", "beta_try")
pgm.add_edge("sigma_player_kick", "beta_kick")

# Hyperparameters to team effects
pgm.add_edge("sigma_team", "gamma")
pgm.add_edge("sigma_defense", "delta")

# Score-type parameters to loading factors (implicit - shown via plates)
pgm.add_edge("lambda_try", "beta_try", linestyle="--")
pgm.add_edge("lambda_kick", "beta_kick", linestyle="--")
pgm.add_edge("lambda_team", "gamma", linestyle="--")

# All components to linear predictor
pgm.add_edge("alpha", "lambda")
pgm.add_edge("eta", "lambda")
pgm.add_edge("beta_try", "lambda")
pgm.add_edge("beta_kick", "lambda")
pgm.add_edge("gamma", "lambda")
pgm.add_edge("delta", "lambda")
pgm.add_edge("theta", "lambda")

# Linear predictor to observation
pgm.add_edge("lambda", "y")

# ============================================================================
# PLATES (Repeated structures)
# ============================================================================
# Score types plate (outermost for score-type specific params)
pgm.add_plate([0.3, 6.8, 7.4, 1.2],
              label=r"$s \in \{$tries, penalties, conversions, drop goals$\}$",
              label_offset=(250, 5), fontsize=11)

# Position plate (nested within score types)
pgm.add_plate([2, 5.9, 1.5, 0.8],
              label=r"$k = 1{:}23$ positions",
              label_offset=(5, 5), fontsize=10)

# Player plate
pgm.add_plate([0.8, 4.1, 3, 1],
              label=r"$i = 1{:}N_{\mathrm{players}}$",
              label_offset=(5, 5), fontsize=11)

# Team-season plate
pgm.add_plate([4.3, 4.1, 2.9, 1],
              label=r"$(j,t) = 1{:}N_{\mathrm{team\_seasons}}$",
              label_offset=(5, 5), fontsize=11)

# Observation plate (innermost - actual data)
pgm.add_plate([3, 0.5, 2.5, 2.5],
              label=r"$(i,m,s)$ observations",
              label_offset=(5, 5), fontsize=11)

# ============================================================================
# ANNOTATIONS (Optional - add key model info)
# ============================================================================
# Add text annotation for the linear predictor equation
plt.text(0.5, 0.3,
         r"$\log(\lambda_{i,m,s}) = \alpha_s + "
         r"\sigma_{\beta}\lambda^{\beta}_s\beta^{\mathrm{type}}_i + "
         r"\sigma_{\gamma}\lambda^{\gamma}_s\gamma_{j,t} + "
         r"\theta_{s,k} + \eta_s h_{i,m} - "
         r"\sigma_{\delta}\delta_{\mathrm{opp}} + "
         r"\log(e_{i,m})$",
         transform=plt.gcf().transFigure,
         fontsize=10, ha='left', va='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# ============================================================================
# RENDER AND SAVE
# ============================================================================
pgm.render()

# Add title
plt.suptitle("Rugby Ranking Model: Hierarchical Bayesian Structure",
             fontsize=14, fontweight='bold', y=0.98)

# Add subtitle with model description
plt.text(0.5, 0.95,
         "Joint model across scoring types with separate try-scoring and kicking effects",
         transform=plt.gcf().transFigure,
         fontsize=11, ha='center', va='top', style='italic')

# Adjust layout to prevent clipping
plt.tight_layout(rect=[0, 0.05, 1, 0.93])

# Save in multiple formats
output_formats = {
    'pdf': {'dpi': 300, 'bbox_inches': 'tight'},
    'png': {'dpi': 300, 'bbox_inches': 'tight', 'facecolor': 'white'},
    'svg': {'bbox_inches': 'tight', 'facecolor': 'white'}
}

for fmt, params in output_formats.items():
    filename = f"rugby_model_plate_diagram.{fmt}"
    plt.savefig(filename, **params)
    print(f"✓ Saved {filename}")

print("\nDone! Generated:")
print("  - rugby_model_plate_diagram.pdf (best for papers/printing)")
print("  - rugby_model_plate_diagram.png (best for web/blog)")
print("  - rugby_model_plate_diagram.svg (scalable vector graphics)")

# Show the plot
plt.show()
