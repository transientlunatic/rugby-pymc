"""
Simulation-Based Calibration (SBC) tests for the rugby ranking model.

These tests implement the P-P test framework used in gravitational wave
astronomy (Abbott et al. 2016, PRX 6 041015) to verify that the inference
engine correctly recovers posteriors. The idea is identical:

    1. Draw (θ_true, y_sim) from the joint prior: θ ~ p(θ), y ~ p(y | θ)
    2. Run inference on y_sim to obtain the posterior p(θ | y_sim)
    3. Compute the rank of θ_true within the posterior samples
    4. Repeat N times — ranks should be Uniform(0, 1) if inference is
       well-calibrated (Talts et al. 2018, arXiv:1804.06788)

Scalar parameters are tested (alpha, eta_home, sigma_player, sigma_team).
Vector parameters (player/team effects) are excluded because SBC for
hierarchical random effects requires label-consistent index mappings across
simulations, which is non-trivial in the hierarchical setting.

The key diagnostic question this answers (from ROADMAP):
    "How does VI posterior quality compare to MCMC? Are VI uncertainty
    estimates reliable?"

If VI is well-calibrated, the P-P plot lies on the diagonal.
If VI underestimates posterior width (common for mean-field ADVI), ranks
will cluster near 0 and 1 — the true value sits in the tails too often.

Running
-------
These tests are excluded from the default test run. To execute:

    pytest tests/test_statistical_calibration.py -m statistical -v -s

For a quicker smoke test (fewer simulations):

    pytest tests/test_statistical_calibration.py -m statistical -v -s \
        --sbc-sims 10

Outputs
-------
Each test run saves a P-P plot to tests/statistical_outputs/ so the visual
can be inspected even when the KS test passes.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytest
from scipy import stats

from rugby_ranking.model.core import ModelConfig, RugbyModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path(__file__).parent / "statistical_outputs"

# Default number of SBC simulations.  30 gives reasonable KS power while
# keeping runtime under ~15 minutes with VI.  Override with --sbc-sims N.
_DEFAULT_N_SIMS = 30

# VI convergence: 15k iterations is sufficient for the minimal dataset.
_VI_ITERATIONS = 15_000

# Posterior samples drawn from the VI approximation per simulation.
_N_POSTERIOR = 500

# KS test significance level.  0.01 is generous (multiple parameters tested)
# but appropriate for a test that is already slow and has low power at n=30.
_KS_ALPHA = 0.01


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _build_minimal_df(
    n_players: int = 10,
    n_teams: int = 3,
    n_matches: int = 15,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build the smallest structurally-valid player-match dataset for SBC.

    All outcome columns (tries, penalties, conversions, drop_goals) start at
    zero; the prior predictive will overwrite them with simulated counts.

    Design rationale
    ----------------
    - n_players=10, n_teams=3, n_matches=15 keeps VI fast (~30s per simulation).
    - All players play 80 minutes so _prepare_data's minutes_played>0 filter
      leaves every row intact — y_sim maps 1-to-1 onto the dataframe rows.
    - A single season avoids season-specific index complications.
    """
    rng = np.random.default_rng(seed)
    players = [f"P{i}" for i in range(n_players)]
    teams = [f"T{j}" for j in range(n_teams)]
    season = "2024-2025"

    records = []
    for mid in range(n_matches):
        pair = rng.choice(n_teams, size=2, replace=False)
        home, away = teams[pair[0]], teams[pair[1]]
        date = pd.Timestamp("2024-09-01") + pd.Timedelta(weeks=mid)

        for team, opp, is_home in [(home, away, 1), (away, home, 0)]:
            # Put all players on the field to keep the dataset dense
            for k, player in enumerate(players):
                records.append(
                    {
                        "match_id": f"m{mid}",
                        "date": date,
                        "player_name": player,
                        "team": team,
                        "opponent": opp,
                        "season": season,
                        "position": (k % 15) + 1,  # positions 1-15, cycling
                        "is_home": is_home,
                        "minutes_played": 80,
                        "tries": 0,
                        "penalties": 0,
                        "conversions": 0,
                        "drop_goals": 0,
                    }
                )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# SBC engine
# ---------------------------------------------------------------------------

def _run_sbc_vi(
    minimal_df: pd.DataFrame,
    param_names: list[str],
    n_sims: int = _DEFAULT_N_SIMS,
    vi_iterations: int = _VI_ITERATIONS,
    n_posterior: int = _N_POSTERIOR,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Run Simulation-Based Calibration using Variational Inference.

    Returns
    -------
    dict mapping param_name -> 1-D array of rank fractions, length n_sims.
    Rank fraction = P(θ_posterior < θ_true): should be Uniform(0,1).

    Implementation notes
    --------------------
    We use the single-score-type model (tries only, no kicking split, no
    defense) to keep parameter names simple and unambiguous. The model is
    built fresh for each simulation to avoid any stale state from pm.set_data.

    Only scalar parameters can be tested without solving the label-switching
    problem for hierarchical effects (see module docstring).
    """
    _OUTPUT_DIR.mkdir(exist_ok=True)
    ranks: dict[str, list[float]] = {p: [] for p in param_names}

    # ---- Step 1: draw (θ_true, y_sim) pairs from the joint prior ----------
    # Build a reference model on the zero-outcome dataframe just to obtain
    # the prior predictive distribution.
    ref_model = RugbyModel(
        ModelConfig(
            score_types=("tries",),
            separate_kicking_effect=False,
            include_defense=False,
        )
    )
    ref_model.build(minimal_df, score_type="tries")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ref_model.model:
            prior_ppc = pm.sample_prior_predictive(
                samples=n_sims,
                random_seed=seed,
            )

    # prior_ppc.prior[param]           shape: (1, n_sims)         for scalars
    # prior_ppc.prior_predictive["y"]  shape: (1, n_sims, n_obs)

    # ---- Step 2: for each simulation, infer and rank ----------------------
    for sim_idx in range(n_sims):
        # Simulated observations for this draw
        y_sim = prior_ppc.prior_predictive["y"].values[0, sim_idx].astype(int)

        # Replace tries column with simulated values
        df_sim = minimal_df.copy()
        df_sim["tries"] = y_sim

        # Build a fresh model on simulated data (same structure, new outcomes)
        sim_model = RugbyModel(
            ModelConfig(
                score_types=("tries",),
                separate_kicking_effect=False,
                include_defense=False,
            )
        )
        sim_model.build(df_sim, score_type="tries")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with sim_model.model:
                approx = pm.fit(
                    n=vi_iterations,
                    method="advi",
                    progressbar=False,
                    random_seed=seed + sim_idx,
                )
                posterior = approx.sample(n_posterior)

        # Rank each requested scalar parameter
        for param in param_names:
            if param not in prior_ppc.prior.data_vars:
                continue
            theta_true = float(prior_ppc.prior[param].values[0, sim_idx])
            theta_post = posterior.posterior[param].values.reshape(-1)
            ranks[param].append(float(np.mean(theta_post < theta_true)))

    return {p: np.array(v) for p, v in ranks.items()}


# ---------------------------------------------------------------------------
# P-P plot helper
# ---------------------------------------------------------------------------

def _save_pp_plot(ranks_dict: dict[str, np.ndarray], filename: str) -> None:
    """Save a P-P plot for each parameter to the statistical_outputs directory."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    params = list(ranks_dict.keys())
    n_cols = min(4, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flat

    for ax, (param, ranks) in zip(axes, ranks_dict.items()):
        n = len(ranks)
        empirical_cdf = np.sort(ranks)
        theoretical = np.linspace(0, 1, n)

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
        ax.plot(theoretical, empirical_cdf, "steelblue", lw=2, label="Empirical")

        # 95% simultaneous confidence band (DKW inequality)
        epsilon = np.sqrt(np.log(2.0 / 0.05) / (2 * n))
        ax.fill_between(
            theoretical,
            np.clip(theoretical - epsilon, 0, 1),
            np.clip(theoretical + epsilon, 0, 1),
            alpha=0.2,
            color="steelblue",
            label="95% band",
        )

        ks_stat, p_val = stats.kstest(ranks, "uniform")
        ax.set_title(f"{param}\nKS p={p_val:.3f}")
        ax.set_xlabel("Theoretical quantile")
        ax.set_ylabel("Empirical CDF")
        ax.legend(fontsize=7)

    for ax in list(axes)[len(params):]:
        ax.set_visible(False)

    fig.suptitle(f"SBC P-P plot — {filename}")
    plt.tight_layout()
    out_path = _OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nP-P plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Uniformity assertion
# ---------------------------------------------------------------------------

def _assert_uniform(
    ranks: np.ndarray,
    param_name: str,
    alpha: float = _KS_ALPHA,
) -> None:
    """
    KS test that rank fractions are Uniform(0, 1).

    A failure means the inference is miscalibrated for this parameter:
    - Ranks clustering near 0.5  → posterior too wide (over-dispersed)
    - Ranks clustering near 0 & 1 → posterior too narrow (typical of VI)
    - Systematic shift            → biased posterior mean
    """
    ks_stat, p_value = stats.kstest(ranks, "uniform")
    assert p_value > alpha, (
        f"SBC FAILED for '{param_name}': rank fractions are not uniform "
        f"(KS statistic={ks_stat:.4f}, p={p_value:.4f} < alpha={alpha}).\n"
        f"  ranks = {np.round(ranks, 3)}\n"
        f"  This suggests miscalibrated inference. If using VI, the posterior "
        f"may be too narrow (underestimated uncertainty). Consider comparing "
        f"against MCMC-based SBC (TestSBC_MCMC below)."
    )
    print(f"  {param_name}: KS p={p_value:.3f}  ✓")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def minimal_df():
    return _build_minimal_df(n_players=10, n_teams=3, n_matches=15, seed=0)


@pytest.fixture(scope="module")
def n_sims(request):
    try:
        return request.config.getoption("--sbc-sims")
    except ValueError:
        return _DEFAULT_N_SIMS


@pytest.fixture(scope="module")
def sbc_ranks_vi(minimal_df, n_sims):
    """
    Run all VI SBC simulations once and cache for the entire module.

    Shared across all TestSBC_VI test methods so simulations only run once.
    """
    _SCALAR_PARAMS = ["alpha", "eta_home", "sigma_player", "sigma_team"]
    print(f"\nRunning VI SBC with {n_sims} simulations...")
    ranks = _run_sbc_vi(
        minimal_df,
        param_names=_SCALAR_PARAMS,
        n_sims=n_sims,
        vi_iterations=_VI_ITERATIONS,
        n_posterior=_N_POSTERIOR,
        seed=42,
    )
    _save_pp_plot(ranks, "sbc_vi_pp_plot.png")
    return ranks


# ---------------------------------------------------------------------------
# SBC engine — production model config (joint likelihood, defense, kicking split)
# ---------------------------------------------------------------------------
#
# TestSBC_VI above validates the inference *machinery* on a deliberately
# simplified model (single score type, no defense, no kicking split) to keep
# scalar parameter names unambiguous. It says nothing about whether the
# structural additions actually used in production — defense, the
# try/kicking effect split, and the joint 4-likelihood fit — are themselves
# calibrated. This section runs the same SBC procedure through
# RugbyModel.build_joint() with the exact ModelConfig used by
# train-model-weekly.yml (include_defense=True, separate_kicking_effect=True).
#
# Vector parameters indexed by score type (alpha, eta_home, ...) are safe to
# rank at a fixed index here — score_types is a fixed tuple, not an
# exchangeable/label-switchable dimension like player or team-season effects.
# We test index 0 ("tries"), since that's the only score type with a defense
# term and therefore the most structurally novel relative to TestSBC_VI.

_PRODUCTION_CONFIG_KWARGS = dict(
    include_defense=True,
    separate_kicking_effect=True,
)

# Scalar hyperparameters new to the production config (not covered above).
_JOINT_SCALAR_PARAMS = [
    "sigma_defense",
    "sigma_player_try",
    "sigma_player_kick",
    "lambda_defense",
]

# (param name, index into the score_types axis, label for reporting/plots).
# Index 0 = "tries" — the only score type with a defense term.
_JOINT_INDEXED_PARAMS = [
    ("alpha", 0, "alpha_tries"),
    ("eta_home", 0, "eta_home_tries"),
]


def _run_sbc_vi_joint(
    minimal_df: pd.DataFrame,
    n_sims: int = _DEFAULT_N_SIMS,
    vi_iterations: int = _VI_ITERATIONS,
    n_posterior: int = _N_POSTERIOR,
    seed: int = 43,
) -> dict[str, np.ndarray]:
    """
    SBC for the production model config (build_joint, defense, kicking split).

    Same procedure as _run_sbc_vi, but simulates and refits all four score
    types jointly through build_joint() rather than the single-score build().
    """
    _OUTPUT_DIR.mkdir(exist_ok=True)
    all_param_names = _JOINT_SCALAR_PARAMS + [label for _, _, label in _JOINT_INDEXED_PARAMS]
    ranks: dict[str, list[float]] = {p: [] for p in all_param_names}

    score_types = ("tries", "penalties", "conversions", "drop_goals")

    ref_model = RugbyModel(ModelConfig(**_PRODUCTION_CONFIG_KWARGS))
    ref_model.build_joint(minimal_df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ref_model.model:
            prior_ppc = pm.sample_prior_predictive(samples=n_sims, random_seed=seed)

    for sim_idx in range(n_sims):
        df_sim = minimal_df.copy()
        for score_type in score_types:
            y_sim = prior_ppc.prior_predictive[f"y_{score_type}"].values[0, sim_idx].astype(int)
            df_sim[score_type] = y_sim

        sim_model = RugbyModel(ModelConfig(**_PRODUCTION_CONFIG_KWARGS))
        sim_model.build_joint(df_sim)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with sim_model.model:
                approx = pm.fit(
                    n=vi_iterations,
                    method="advi",
                    progressbar=False,
                    random_seed=seed + sim_idx,
                )
                posterior = approx.sample(n_posterior)

        for param in _JOINT_SCALAR_PARAMS:
            if param not in prior_ppc.prior.data_vars:
                continue
            theta_true = float(prior_ppc.prior[param].values[0, sim_idx])
            theta_post = posterior.posterior[param].values.reshape(-1)
            ranks[param].append(float(np.mean(theta_post < theta_true)))

        for param, idx, label in _JOINT_INDEXED_PARAMS:
            if param not in prior_ppc.prior.data_vars:
                continue
            theta_true = float(prior_ppc.prior[param].values[0, sim_idx, idx])
            theta_post = posterior.posterior[param].values[..., idx].reshape(-1)
            ranks[label].append(float(np.mean(theta_post < theta_true)))

    return {p: np.array(v) for p, v in ranks.items()}


@pytest.fixture(scope="module")
def sbc_ranks_vi_joint(minimal_df, n_sims):
    """Run production-config VI SBC once, shared across TestSBC_VI_ProductionConfig."""
    print(f"\nRunning production-config VI SBC with {n_sims} simulations...")
    ranks = _run_sbc_vi_joint(
        minimal_df,
        n_sims=n_sims,
        vi_iterations=_VI_ITERATIONS,
        n_posterior=_N_POSTERIOR,
        seed=43,
    )
    _save_pp_plot(ranks, "sbc_vi_joint_pp_plot.png")
    return ranks


@pytest.mark.statistical
class TestSBC_VI_ProductionConfig:
    """
    P-P tests for VI calibration on the actual production model config —
    build_joint() with include_defense=True, separate_kicking_effect=True,
    matching train-model-weekly.yml. Complements TestSBC_VI, which only
    exercises the simplified single-score-type model.

    PASS  This parameter's posterior is well-calibrated under VI for the
          model config actually trained weekly.
    FAIL  Check the defense/kicking-split additions specifically — these are
          exactly what TestSBC_VI cannot see.
    """

    def test_defense_scale_sigma_defense(self, sbc_ranks_vi_joint):
        _assert_uniform(sbc_ranks_vi_joint["sigma_defense"], "sigma_defense")

    def test_defense_loading_lambda_defense(self, sbc_ranks_vi_joint):
        _assert_uniform(sbc_ranks_vi_joint["lambda_defense"], "lambda_defense")

    def test_try_effect_scale_sigma_player_try(self, sbc_ranks_vi_joint):
        _assert_uniform(sbc_ranks_vi_joint["sigma_player_try"], "sigma_player_try")

    def test_kicking_effect_scale_sigma_player_kick(self, sbc_ranks_vi_joint):
        _assert_uniform(sbc_ranks_vi_joint["sigma_player_kick"], "sigma_player_kick")

    def test_tries_intercept_alpha(self, sbc_ranks_vi_joint):
        _assert_uniform(sbc_ranks_vi_joint["alpha_tries"], "alpha_tries")

    def test_tries_home_advantage_eta_home(self, sbc_ranks_vi_joint):
        _assert_uniform(sbc_ranks_vi_joint["eta_home_tries"], "eta_home_tries")


@pytest.mark.statistical
class TestSBC_VI:
    """
    P-P tests for VI (ADVI) calibration on the single-score rugby model.

    Each test checks one scalar parameter. All tests share the same set of
    SBC simulations (via the module-scoped ``sbc_ranks_vi`` fixture) so the
    expensive simulation step runs only once for the whole class.

    Interpretation guide
    --------------------
    PASS  The posterior for this parameter is well-calibrated under VI.
    FAIL  VI is systematically mis-estimating this parameter's uncertainty.
          Most commonly: ranks cluster near 0 & 1 → VI posterior too narrow.
          Check the P-P plot in tests/statistical_outputs/sbc_vi_pp_plot.png.
    """

    def test_intercept_alpha(self, sbc_ranks_vi):
        """alpha (per-score-type baseline rate) should be recovered correctly."""
        _assert_uniform(sbc_ranks_vi["alpha"], "alpha")

    def test_home_advantage_eta_home(self, sbc_ranks_vi):
        """eta_home (home advantage) should be recovered correctly."""
        _assert_uniform(sbc_ranks_vi["eta_home"], "eta_home")

    def test_player_effect_scale_sigma_player(self, sbc_ranks_vi):
        """sigma_player (player effect SD hyperparameter) recovery."""
        _assert_uniform(sbc_ranks_vi["sigma_player"], "sigma_player")

    def test_team_effect_scale_sigma_team(self, sbc_ranks_vi):
        """sigma_team (team-season effect SD hyperparameter) recovery."""
        _assert_uniform(sbc_ranks_vi["sigma_team"], "sigma_team")


@pytest.mark.statistical
@pytest.mark.slow
class TestSBC_MCMC:
    """
    MCMC reference P-P tests.

    These run full NUTS sampling for each simulation and are substantially
    slower than the VI version (~2-4 hours for n_sims=30). They serve as the
    ground-truth calibration check: if MCMC passes but VI fails, VI is
    the problem; if MCMC also fails, the model is mis-specified.

    Run with:
        pytest tests/test_statistical_calibration.py -m "statistical and slow" -v -s
    """

    _MCMC_DRAWS = 500
    _MCMC_TUNE = 500
    _MCMC_CHAINS = 2
    _SCALAR_PARAMS = ["alpha", "eta_home", "sigma_player", "sigma_team"]

    @pytest.fixture(scope="class")
    def sbc_ranks_mcmc(self, minimal_df, n_sims):
        """Run MCMC SBC simulations (slow — shared across all test methods)."""
        _OUTPUT_DIR.mkdir(exist_ok=True)
        ranks: dict[str, list[float]] = {p: [] for p in self._SCALAR_PARAMS}

        ref_model = RugbyModel(
            ModelConfig(
                score_types=("tries",),
                separate_kicking_effect=False,
                include_defense=False,
            )
        )
        ref_model.build(minimal_df, score_type="tries")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with ref_model.model:
                prior_ppc = pm.sample_prior_predictive(
                    samples=n_sims, random_seed=0
                )

        for sim_idx in range(n_sims):
            y_sim = prior_ppc.prior_predictive["y"].values[0, sim_idx].astype(int)
            df_sim = minimal_df.copy()
            df_sim["tries"] = y_sim

            sim_model = RugbyModel(
                ModelConfig(
                    score_types=("tries",),
                    separate_kicking_effect=False,
                    include_defense=False,
                )
            )
            sim_model.build(df_sim, score_type="tries")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with sim_model.model:
                    trace = pm.sample(
                        draws=self._MCMC_DRAWS,
                        tune=self._MCMC_TUNE,
                        chains=self._MCMC_CHAINS,
                        progressbar=False,
                        random_seed=sim_idx,
                    )

            for param in self._SCALAR_PARAMS:
                if param not in prior_ppc.prior.data_vars:
                    continue
                theta_true = float(prior_ppc.prior[param].values[0, sim_idx])
                theta_post = trace.posterior[param].values.reshape(-1)
                ranks[param].append(float(np.mean(theta_post < theta_true)))

        result = {p: np.array(v) for p, v in ranks.items()}
        _save_pp_plot(result, "sbc_mcmc_pp_plot.png")
        return result

    def test_intercept_alpha(self, sbc_ranks_mcmc):
        _assert_uniform(sbc_ranks_mcmc["alpha"], "alpha")

    def test_home_advantage_eta_home(self, sbc_ranks_mcmc):
        _assert_uniform(sbc_ranks_mcmc["eta_home"], "eta_home")

    def test_player_effect_scale_sigma_player(self, sbc_ranks_mcmc):
        _assert_uniform(sbc_ranks_mcmc["sigma_player"], "sigma_player")

    def test_team_effect_scale_sigma_team(self, sbc_ranks_mcmc):
        _assert_uniform(sbc_ranks_mcmc["sigma_team"], "sigma_team")


@pytest.mark.statistical
@pytest.mark.slow
class TestSBC_MCMC_ProductionConfig:
    """
    MCMC reference P-P tests for the production model config (build_joint,
    defense, kicking split) — the ground-truth counterpart to
    TestSBC_VI_ProductionConfig. If a parameter fails under VI but passes
    here, the posterior itself is fine and the problem is VI's approximation
    (typically an overly narrow mean-field posterior); if it fails here too,
    the model — not just the inference method — needs attention.

    Run with:
        pytest tests/test_statistical_calibration.py -m "statistical and slow" -v -s
    """

    _MCMC_DRAWS = 500
    _MCMC_TUNE = 500
    _MCMC_CHAINS = 2

    @pytest.fixture(scope="class")
    def sbc_ranks_mcmc_joint(self, minimal_df, n_sims):
        _OUTPUT_DIR.mkdir(exist_ok=True)
        all_param_names = _JOINT_SCALAR_PARAMS + [label for _, _, label in _JOINT_INDEXED_PARAMS]
        ranks: dict[str, list[float]] = {p: [] for p in all_param_names}
        score_types = ("tries", "penalties", "conversions", "drop_goals")

        ref_model = RugbyModel(ModelConfig(**_PRODUCTION_CONFIG_KWARGS))
        ref_model.build_joint(minimal_df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with ref_model.model:
                prior_ppc = pm.sample_prior_predictive(samples=n_sims, random_seed=44)

        for sim_idx in range(n_sims):
            df_sim = minimal_df.copy()
            for score_type in score_types:
                y_sim = prior_ppc.prior_predictive[f"y_{score_type}"].values[0, sim_idx].astype(int)
                df_sim[score_type] = y_sim

            sim_model = RugbyModel(ModelConfig(**_PRODUCTION_CONFIG_KWARGS))
            sim_model.build_joint(df_sim)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with sim_model.model:
                    trace = pm.sample(
                        draws=self._MCMC_DRAWS,
                        tune=self._MCMC_TUNE,
                        chains=self._MCMC_CHAINS,
                        progressbar=False,
                        random_seed=sim_idx,
                    )

            for param in _JOINT_SCALAR_PARAMS:
                if param not in prior_ppc.prior.data_vars:
                    continue
                theta_true = float(prior_ppc.prior[param].values[0, sim_idx])
                theta_post = trace.posterior[param].values.reshape(-1)
                ranks[param].append(float(np.mean(theta_post < theta_true)))

            for param, idx, label in _JOINT_INDEXED_PARAMS:
                if param not in prior_ppc.prior.data_vars:
                    continue
                theta_true = float(prior_ppc.prior[param].values[0, sim_idx, idx])
                theta_post = trace.posterior[param].values[..., idx].reshape(-1)
                ranks[label].append(float(np.mean(theta_post < theta_true)))

        result = {p: np.array(v) for p, v in ranks.items()}
        _save_pp_plot(result, "sbc_mcmc_joint_pp_plot.png")
        return result

    def test_defense_scale_sigma_defense(self, sbc_ranks_mcmc_joint):
        _assert_uniform(sbc_ranks_mcmc_joint["sigma_defense"], "sigma_defense")

    def test_defense_loading_lambda_defense(self, sbc_ranks_mcmc_joint):
        _assert_uniform(sbc_ranks_mcmc_joint["lambda_defense"], "lambda_defense")

    def test_try_effect_scale_sigma_player_try(self, sbc_ranks_mcmc_joint):
        _assert_uniform(sbc_ranks_mcmc_joint["sigma_player_try"], "sigma_player_try")

    def test_kicking_effect_scale_sigma_player_kick(self, sbc_ranks_mcmc_joint):
        _assert_uniform(sbc_ranks_mcmc_joint["sigma_player_kick"], "sigma_player_kick")

    def test_tries_intercept_alpha(self, sbc_ranks_mcmc_joint):
        _assert_uniform(sbc_ranks_mcmc_joint["alpha_tries"], "alpha_tries")

    def test_tries_home_advantage_eta_home(self, sbc_ranks_mcmc_joint):
        _assert_uniform(sbc_ranks_mcmc_joint["eta_home_tries"], "eta_home_tries")
