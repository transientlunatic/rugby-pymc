"""
Inference machinery for rugby ranking models.

Supports:
- Full MCMC sampling (for validation)
- Variational inference (for fast weekly updates)
- Warm-starting from previous posteriors
- Caching of fitted models
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Literal

import arviz as az
import numpy as np
import pymc as pm

from rugby_ranking.model.core import RugbyModel


@dataclass
class InferenceConfig:
    """Configuration for model inference."""

    # MCMC settings
    mcmc_draws: int = 2000
    mcmc_tune: int = 1000
    mcmc_chains: int = 4
    mcmc_cores: int = 4
    mcmc_target_accept: float = 0.9

    # VI settings
    vi_n_iterations: int = 50000
    vi_method: Literal["advi", "fullrank_advi"] = "advi"
    vi_use_minibatch: bool = False
    vi_minibatch_size: int = 1024

    # Caching
    cache_dir: Path = Path("~/.cache/rugby_ranking").expanduser()


class ModelFitter:
    """
    Handles model fitting with support for incremental updates.

    Usage:
        fitter = ModelFitter(model, config)

        # Full MCMC (monthly validation)
        trace = fitter.fit_mcmc()

        # Fast VI (weekly updates)
        trace = fitter.fit_vi()

        # Warm-started VI from previous run
        trace = fitter.fit_vi(warm_start=True)

        # Save/load for persistence
        fitter.save("model_checkpoint")
        fitter = ModelFitter.load("model_checkpoint")
    """

    def __init__(
        self,
        model: RugbyModel,
        config: InferenceConfig | None = None,
    ):
        self.rugby_model = model
        self.config = config or InferenceConfig()
        self.trace = None
        self._last_fit_time: datetime | None = None
        self._fit_method: str | None = None
        self._vi_approx = None  # Store VI approximation for warm starts

        # Ensure cache directory exists
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def fit_mcmc(
        self,
        random_seed: int | None = None,
        progressbar: bool = True,
        **kwargs,
    ) -> az.InferenceData:
        """
        Fit model using MCMC (NUTS sampler).

        This is the gold standard but slower. Use for:
        - Initial model validation
        - Monthly full refits
        - Comparing against VI approximation

        Args:
            random_seed: Random seed for reproducibility
            progressbar: Show sampling progress
            **kwargs: Additional arguments to pm.sample()

        Returns:
            ArviZ InferenceData with posterior samples
        """
        if self.rugby_model.model is None:
            raise ValueError("Model not built. Call model.build() first.")

        with self.rugby_model.model:
            trace = pm.sample(
                draws=self.config.mcmc_draws,
                tune=self.config.mcmc_tune,
                chains=self.config.mcmc_chains,
                cores=self.config.mcmc_cores,
                target_accept=self.config.mcmc_target_accept,
                random_seed=random_seed,
                progressbar=progressbar,
                **kwargs,
            )

        self.trace = trace
        self.rugby_model.trace = trace
        self._last_fit_time = datetime.now()
        self._fit_method = "mcmc"

        return trace

    def fit_vi(
        self,
        warm_start: bool = False,
        random_seed: int | None = None,
        progressbar: bool = True,
        n_samples: int = 2000,
        **kwargs,
    ) -> az.InferenceData:
        """
        Fit model using Variational Inference.

        Much faster than MCMC, suitable for weekly updates.
        May underestimate posterior uncertainty.

        Args:
            warm_start: Initialize from previous VI fit
            random_seed: Random seed for reproducibility
            progressbar: Show optimization progress
            n_samples: Number of samples to draw from approximate posterior
            **kwargs: Additional arguments to pm.fit()

        Returns:
            ArviZ InferenceData with approximate posterior samples
        """
        if self.rugby_model.model is None:
            raise ValueError("Model not built. Call model.build() first.")

        with self.rugby_model.model:
            # Set up VI
            if self.config.vi_method == "fullrank_advi":
                approx = pm.FullRankADVI()
            else:
                approx = pm.ADVI()

            # Warm start from previous approximation
            if warm_start and self._vi_approx is not None:
                # Transfer learned parameters
                try:
                    approx = self._transfer_vi_params(approx)
                except Exception as e:
                    print(f"Warm start failed, starting fresh: {e}")

            # Fit
            approx = pm.fit(
                n=self.config.vi_n_iterations,
                method=approx,
                random_seed=random_seed,
                progressbar=progressbar,
                **kwargs,
            )

            # Store for future warm starts
            self._vi_approx = approx

            # Sample from approximate posterior
            trace = approx.sample(n_samples)


        self.trace = trace
        self.rugby_model.trace = trace
        self._last_fit_time = datetime.now()
        self._fit_method = "vi"

        return trace

    def _transfer_vi_params(self, new_approx):
        """Transfer parameters from previous VI approximation."""
        # This is a simplified version - full implementation would
        # handle dimension changes when new players/teams are added
        if self._vi_approx is not None:
            # Copy mean and std parameters where dimensions match
            old_params = self._vi_approx.params
            new_params = new_approx.params

            for old_p, new_p in zip(old_params, new_params):
                if old_p.shape == new_p.shape:
                    new_p.set_value(old_p.get_value())

        return new_approx

    def save(self, name: str) -> Path:
        """
        Save fitted model to cache.

        Saves:
        - Trace (posterior samples)
        - Model indices (player/team mappings)
        - VI approximation (if available)
        - Metadata

        Args:
            name: Name for this checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.config.cache_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save trace
        if self.trace is not None:
            self.trace.to_netcdf(checkpoint_dir / "trace.nc")

        # Save model indices and metadata
        metadata = {
            "player_ids": self.rugby_model._player_ids,
            "team_ids": self.rugby_model._team_ids,
            "season_ids": self.rugby_model._season_ids,
            "team_season_ids": self.rugby_model._team_season_ids,
            "last_fit_time": self._last_fit_time,
            "fit_method": self._fit_method,
            "config": self.config,
        }

        with open(checkpoint_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        print(f"Saved checkpoint to {checkpoint_dir}")
        return checkpoint_dir

    @classmethod
    def load(
        cls,
        name: str,
        model: RugbyModel,
        cache_dir: Path | None = None,
    ) -> "ModelFitter":
        """
        Load a previously saved model checkpoint.

        Args:
            name: Checkpoint name
            model: RugbyModel instance to populate
            cache_dir: Override default cache directory

        Returns:
            ModelFitter with restored state
        """
        cache_dir = cache_dir or Path("~/.cache/rugby_ranking").expanduser()
        checkpoint_dir = cache_dir / name

        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_dir}")

        # Load metadata
        with open(checkpoint_dir / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        # Restore model indices
        model._player_ids = metadata["player_ids"]
        model._team_ids = metadata["team_ids"]
        model._season_ids = metadata["season_ids"]
        model._team_season_ids = metadata["team_season_ids"]

        # Create fitter
        fitter = cls(model, metadata.get("config"))
        fitter._last_fit_time = metadata.get("last_fit_time")
        fitter._fit_method = metadata.get("fit_method")

        # Load trace
        trace_path = checkpoint_dir / "trace.nc"
        if trace_path.exists():
            fitter.trace = az.from_netcdf(trace_path)
            model.trace = fitter.trace

        print(f"Loaded checkpoint from {checkpoint_dir}")
        return fitter

    def diagnostics(self) -> dict:
        """
        Compute convergence diagnostics for the fitted model.

        Returns:
            Dictionary with diagnostic summaries
        """
        if self.trace is None:
            raise ValueError("No trace available. Run inference first.")

        # R-hat and ESS
        summary = az.summary(self.trace)

        diagnostics = {
            "r_hat_max": summary["r_hat"].max(),
            "r_hat_mean": summary["r_hat"].mean(),
            "ess_bulk_min": summary["ess_bulk"].min(),
            "ess_tail_min": summary["ess_tail"].min(),
            "divergences": 0,  # Would need to check trace attrs for MCMC
            "fit_method": self._fit_method,
            "fit_time": self._last_fit_time,
        }

        # Check for problems
        if diagnostics["r_hat_max"] > 1.01:
            print(f"Warning: High R-hat detected ({diagnostics['r_hat_max']:.3f})")
        if diagnostics["ess_bulk_min"] < 400:
            print(f"Warning: Low ESS detected ({diagnostics['ess_bulk_min']:.0f})")

        return diagnostics


def compare_vi_to_mcmc(
    model: RugbyModel,
    mcmc_trace: az.InferenceData,
    vi_trace: az.InferenceData,
) -> dict:
    """
    Compare VI approximation to MCMC ground truth.

    Useful for validating that VI is giving reasonable results.

    Returns:
        Dictionary with comparison metrics
    """
    results = {}

    # Compare key parameters
    for var in ["beta_player", "gamma_team_season", "alpha"]:
        if var in mcmc_trace.posterior and var in vi_trace.posterior:
            mcmc_mean = mcmc_trace.posterior[var].mean(dim=["chain", "draw"]).values
            vi_mean = vi_trace.posterior[var].mean(dim=["chain", "draw"]).values

            mcmc_std = mcmc_trace.posterior[var].std(dim=["chain", "draw"]).values
            vi_std = vi_trace.posterior[var].std(dim=["chain", "draw"]).values

            # Correlation of means
            corr = np.corrcoef(mcmc_mean.flatten(), vi_mean.flatten())[0, 1]

            # Average uncertainty ratio
            std_ratio = (vi_std / mcmc_std).mean()

            results[var] = {
                "mean_correlation": corr,
                "std_ratio": std_ratio,  # < 1 means VI underestimates uncertainty
            }

    return results
