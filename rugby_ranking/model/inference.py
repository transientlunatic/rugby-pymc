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
        checkpoint_every: int | None = None,
        checkpoint_name: str = "training_checkpoint",
        verbose: bool = False,
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
            checkpoint_every: Save checkpoint every N draws per chain (None = no periodic saving)
            checkpoint_name: Name for periodic checkpoints
            verbose: Print detailed progress (useful for HTCondor)
            **kwargs: Additional arguments to pm.sample()

        Returns:
            ArviZ InferenceData with posterior samples
        """
        import sys

        if self.rugby_model.model is None:
            raise ValueError("Model not built. Call model.build() first.")

        # Set up periodic checkpointing via callback
        callback = kwargs.pop("callback", None)

        if checkpoint_every is not None:
            # Create MCMC checkpoint callback
            checkpoint_callback = self._create_mcmc_checkpoint_callback(
                checkpoint_name=checkpoint_name,
                checkpoint_every=checkpoint_every,
            )
            callback = checkpoint_callback

            if verbose:
                print(f"MCMC checkpointing enabled: saving every {checkpoint_every} draws per chain", flush=True)

        if verbose:
            print(f"Starting MCMC: {self.config.mcmc_draws} draws × {self.config.mcmc_chains} chains "
                  f"(+ {self.config.mcmc_tune} tuning)", flush=True)
            print(f"Total samples per chain: {self.config.mcmc_tune + self.config.mcmc_draws}", flush=True)
            sys.stdout.flush()

        with self.rugby_model.model:
            trace = pm.sample(
                draws=self.config.mcmc_draws,
                tune=self.config.mcmc_tune,
                chains=self.config.mcmc_chains,
                cores=self.config.mcmc_cores,
                target_accept=self.config.mcmc_target_accept,
                random_seed=random_seed,
                progressbar=progressbar,
                callback=callback,
                **kwargs,
            )

        if verbose:
            print(f"✓ MCMC sampling complete", flush=True)
            sys.stdout.flush()

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
        sample_batch_size: int | None = None,
        checkpoint_every: int | None = None,
        checkpoint_name: str = "training_checkpoint",
        verbose: bool = False,
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
            sample_batch_size: Draw samples in batches of this size to avoid OOM (None = draw all at once)
            checkpoint_every: Save checkpoint every N iterations (None = no periodic saving)
            checkpoint_name: Name for periodic checkpoints
            verbose: Print detailed progress (useful for HTCondor/file output)
            **kwargs: Additional arguments to pm.fit()

        Returns:
            ArviZ InferenceData with approximate posterior samples
        """
        import sys

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
                    if verbose:
                        print("Warm-started from previous approximation", flush=True)
                except Exception as e:
                    print(f"Warm start failed, starting fresh: {e}", flush=True)

            # Set up periodic checkpointing callback if requested
            callbacks = kwargs.pop("callbacks", [])
            if checkpoint_every is not None:
                checkpoint_callback = self._create_checkpoint_callback(
                    checkpoint_name=checkpoint_name,
                    checkpoint_every=checkpoint_every,
                    n_samples=n_samples,
                )
                callbacks.append(checkpoint_callback)

            # Add verbose progress callback if requested (even without checkpointing)
            elif verbose:
                # Create a progress-only callback
                def progress_callback(approx, losses, i):
                    # Print every 1000 iterations
                    if i > 0 and i % 1000 == 0:
                        loss_str = f"{losses[-1]:.2f}" if len(losses) > 0 else "N/A"
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp}] Iteration {i:,}/{self.config.vi_n_iterations:,} - ELBO: {loss_str}", flush=True)

                callbacks.append(progress_callback)

            if verbose:
                print(f"Starting VI optimization ({self.config.vi_n_iterations:,} iterations)...", flush=True)

            # Fit
            approx = pm.fit(
                n=self.config.vi_n_iterations,
                method=approx,
                random_seed=random_seed,
                progressbar=progressbar,
                callbacks=callbacks,
                **kwargs,
            )

            # Store for future warm starts
            self._vi_approx = approx

            if verbose:
                print("Sampling from approximate posterior...", flush=True)

            # Sample from approximate posterior (in batches if requested)
            if sample_batch_size is not None and sample_batch_size < n_samples:
                # Sample in batches to avoid OOM
                traces = []
                n_batches = (n_samples + sample_batch_size - 1) // sample_batch_size  # Ceiling division
                
                for batch_idx in range(n_batches):
                    # Calculate batch size (last batch may be smaller)
                    current_batch_size = min(sample_batch_size, n_samples - batch_idx * sample_batch_size)
                    
                    if verbose:
                        print(f"  Batch {batch_idx + 1}/{n_batches}: sampling {current_batch_size} samples...", flush=True)
                    
                    batch_trace = approx.sample(current_batch_size)
                    traces.append(batch_trace)
                
                # Concatenate all batches
                if verbose:
                    print("  Concatenating batches...", flush=True)
                trace = az.concat(traces, dim="draw")
            else:
                # Sample all at once (original behavior)
                trace = approx.sample(n_samples)

            if verbose:
                print(f"✓ VI complete. Drew {n_samples} samples from approximate posterior.", flush=True)


        self.trace = trace
        self.rugby_model.trace = trace
        self._last_fit_time = datetime.now()
        self._fit_method = "vi"

        return trace

    def _transfer_vi_params(self, new_approx):
        """
        Transfer parameters from previous VI approximation.
        
        Handles dimension changes when new players/teams are added:
        - Exact match: Copy parameters directly
        - New model larger: Copy old parameters, keep new ones at initialization
        - New model smaller: Subset the old parameters (for filtering old data)
        """
        if self._vi_approx is not None:
            old_params = self._vi_approx.params
            new_params = new_approx.params

            for old_p, new_p in zip(old_params, new_params):
                old_shape = old_p.shape
                new_shape = new_p.shape
                
                if old_shape == new_shape:
                    # Exact match - copy directly
                    new_p.set_value(old_p.get_value())
                elif len(old_shape) == len(new_shape):
                    # Same number of dimensions - try to copy what we can
                    old_val = old_p.get_value()
                    new_val = new_p.get_value()
                    
                    # Build slicing tuples for copying common dimensions
                    slices_old = []
                    slices_new = []
                    can_copy = True
                    
                    for old_dim, new_dim in zip(old_shape, new_shape):
                        min_dim = min(old_dim, new_dim)
                        if min_dim > 0:
                            slices_old.append(slice(0, min_dim))
                            slices_new.append(slice(0, min_dim))
                        else:
                            can_copy = False
                            break
                    
                    if can_copy:
                        # Copy the overlapping part
                        new_val[tuple(slices_new)] = old_val[tuple(slices_old)]
                        new_p.set_value(new_val)
                # If shapes are incompatible, keep initialization from new_approx

        return new_approx

    def sample_from_vi(
        self,
        n_samples: int = 2000,
        sample_batch_size: int | None = None,
        replace_trace: bool = False,
        verbose: bool = False,
    ) -> az.InferenceData:
        """
        Draw additional samples from a fitted VI approximation.

        Useful for drawing more samples after training without re-running optimization.
        Requires that the model was fitted with VI and saved/loaded with the approximation.

        Args:
            n_samples: Number of samples to draw
            sample_batch_size: Draw samples in batches of this size to avoid OOM (None = draw all at once)
            replace_trace: Replace the existing trace with new samples (default: False)
            verbose: Print progress messages

        Returns:
            ArviZ InferenceData with new samples

        Example:
            # Load a checkpoint
            fitter = ModelFitter.load("my_model", model)
            
            # Draw 5000 more samples in batches
            new_trace = fitter.sample_from_vi(
                n_samples=5000,
                sample_batch_size=100,
                verbose=True
            )
        """
        if self._vi_approx is None:
            raise ValueError(
                "No VI approximation available. "
                "Either fit with VI first or load a checkpoint that includes the approximation."
            )

        if verbose:
            print(f"Drawing {n_samples} samples from saved VI approximation...", flush=True)

        # Sample in batches if requested
        if sample_batch_size is not None and sample_batch_size < n_samples:
            traces = []
            n_batches = (n_samples + sample_batch_size - 1) // sample_batch_size
            
            for batch_idx in range(n_batches):
                current_batch_size = min(sample_batch_size, n_samples - batch_idx * sample_batch_size)
                
                if verbose:
                    print(f"  Batch {batch_idx + 1}/{n_batches}: sampling {current_batch_size} samples...", flush=True)
                
                batch_trace = self._vi_approx.sample(current_batch_size)
                traces.append(batch_trace)
            
            if verbose:
                print("  Concatenating batches...", flush=True)
            trace = az.concat(traces, dim="draw")
        else:
            trace = self._vi_approx.sample(n_samples)

        if verbose:
            print(f"✓ Drew {n_samples} samples successfully.", flush=True)

        # Optionally replace the stored trace
        if replace_trace:
            self.trace = trace
            self.rugby_model.trace = trace
            if verbose:
                print("Updated stored trace with new samples.", flush=True)

        return trace

    def _create_checkpoint_callback(
        self,
        checkpoint_name: str,
        checkpoint_every: int,
        n_samples: int,
    ):
        """
        Create a callback function for periodic checkpointing during VI.

        Args:
            checkpoint_name: Base name for checkpoints
            checkpoint_every: Save every N iterations
            n_samples: Number of samples to draw when saving

        Returns:
            Callback function compatible with pm.fit()
        """
        import sys

        iteration_counter = {"count": 0, "last_loss": None}

        def checkpoint_callback(approx, losses, i):
            """Callback invoked during VI optimization."""
            iteration_counter["count"] = i

            # Get current loss (ELBO)
            current_loss = losses[-1] if len(losses) > 0 else None
            iteration_counter["last_loss"] = current_loss

            if i > 0 and i % checkpoint_every == 0:
                try:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    loss_str = f"{current_loss:.2f}" if current_loss is not None else "N/A"

                    # Print progress (will appear in HTCondor output files)
                    print(f"\n[{timestamp}] Iteration {i:,}/{self.config.vi_n_iterations:,} - ELBO: {loss_str}", flush=True)
                    sys.stdout.flush()  # Force flush to file

                    # Store approximation
                    self._vi_approx = approx

                    # Sample and save
                    print(f"[{timestamp}] Sampling {n_samples} draws from approximate posterior...", flush=True)
                    trace = approx.sample(n_samples)
                    old_trace = self.trace
                    self.trace = trace
                    self.rugby_model.trace = trace

                    # Save checkpoint with iteration number
                    print(f"[{timestamp}] Saving checkpoint...", flush=True)
                    checkpoint_path = self.save(f"{checkpoint_name}_iter{i}")
                    print(f"[{timestamp}] ✓ Checkpoint saved to {checkpoint_path}", flush=True)
                    sys.stdout.flush()

                    # Restore original trace
                    self.trace = old_trace
                    self.rugby_model.trace = old_trace

                except Exception as e:
                    print(f"\n[{timestamp}] ⚠ Warning: Checkpoint save failed at iteration {i}: {e}", flush=True)
                    sys.stdout.flush()

        return checkpoint_callback

    def _create_mcmc_checkpoint_callback(
        self,
        checkpoint_name: str,
        checkpoint_every: int,
    ):
        """
        Create a callback function for periodic checkpointing during MCMC.

        PyMC calls the callback after each draw with (trace, draw_idx).

        Args:
            checkpoint_name: Base name for checkpoints
            checkpoint_every: Save every N draws (per chain)

        Returns:
            Callback function compatible with pm.sample()
        """
        import sys

        draw_counter = {"count": 0, "last_checkpoint": 0}

        def mcmc_checkpoint_callback(trace, draw):
            """
            Callback invoked after each MCMC draw.

            Args:
                trace: Current trace (InferenceData)
                draw: Dictionary with draw info (draw_idx, chain, etc.)
            """
            # draw is a dict like: {'chain': 0, 'draw_idx': 100, 'is_last': False}
            # We checkpoint based on total draws across all chains

            draw_counter["count"] += 1

            # Only checkpoint from chain 0 to avoid race conditions
            if draw.get("chain", 0) == 0:
                chain_draw_idx = draw.get("draw_idx", 0)

                # Check if it's time to checkpoint (every N draws)
                if chain_draw_idx > 0 and chain_draw_idx % checkpoint_every == 0:
                    if draw_counter["last_checkpoint"] != chain_draw_idx:
                        try:
                            timestamp = datetime.now().strftime("%H:%M:%S")

                            # Print progress
                            print(f"\n[{timestamp}] Draw {chain_draw_idx}/{self.config.mcmc_draws} (chain 0)", flush=True)
                            print(f"[{timestamp}] Saving checkpoint...", flush=True)
                            sys.stdout.flush()

                            # Save current trace
                            old_trace = self.trace
                            self.trace = trace
                            self.rugby_model.trace = trace

                            # Save checkpoint
                            checkpoint_path = self.save(f"{checkpoint_name}_draw{chain_draw_idx}")
                            print(f"[{timestamp}] ✓ Checkpoint saved to {checkpoint_path}", flush=True)
                            sys.stdout.flush()

                            # Restore
                            self.trace = old_trace
                            self.rugby_model.trace = old_trace

                            draw_counter["last_checkpoint"] = chain_draw_idx

                        except Exception as e:
                            print(f"\n[{timestamp}] ⚠ Warning: Checkpoint save failed at draw {chain_draw_idx}: {e}", flush=True)
                            sys.stdout.flush()

        return mcmc_checkpoint_callback

    def save(self, name: str) -> Path:
        """
        Save fitted model to cache.

        Saves:
        - Trace (posterior samples)
        - Model indices (player/team mappings)
        - VI approximation (if available, for drawing more samples)
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

        # Save VI approximation (enables drawing more samples later)
        # Note: PyMC VI approximations may contain unpicklable objects (e.g., functools.partial)
        # so we try to save it but continue if it fails
        if self._vi_approx is not None:
            try:
                with open(checkpoint_dir / "vi_approx.pkl", "wb") as f:
                    pickle.dump(self._vi_approx, f)
            except (AttributeError, TypeError, pickle.PicklingError) as e:
                print(f"Warning: Could not save VI approximation ({e})")
                print("The trace has been saved and can still be used for predictions.")

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

        # Load VI approximation (if available)
        vi_approx_path = checkpoint_dir / "vi_approx.pkl"
        if vi_approx_path.exists():
            try:
                with open(vi_approx_path, "rb") as f:
                    fitter._vi_approx = pickle.load(f)
                print(f"Loaded VI approximation (can draw more samples)")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Warning: Could not load VI approximation ({e})")
                print("Trace is available but cannot draw additional samples")
                fitter._vi_approx = None

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
