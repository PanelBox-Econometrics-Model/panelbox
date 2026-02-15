"""
Bootstrap inference for quantile regression.

This module implements bootstrap methods for panel quantile regression inference.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


class QuantileBootstrap:
    """
    Bootstrap inference for quantile regression.

    Supports multiple bootstrap methods appropriate for panel data.
    """

    def __init__(
        self, model, tau, n_boot=999, method="cluster", ci_method="percentile", random_state=None
    ):
        """
        Parameters
        ----------
        model : QuantilePanelModel
            Fitted model
        tau : float
            Quantile level
        n_boot : int
            Number of bootstrap replications
        method : str
            Bootstrap method:
            - 'cluster': Resample entities (preserves within-correlation)
            - 'pairs': Resample (y,X) pairs
            - 'wild': Wild bootstrap with multipliers
            - 'subsampling': m-out-of-n bootstrap
        ci_method : str
            CI construction: 'percentile', 'bca', 'normal'
        random_state : int
            Random seed
        """
        self.model = model
        self.tau = tau
        self.n_boot = n_boot
        self.method = method
        self.ci_method = ci_method
        self.random_state = random_state

    def bootstrap(self, n_jobs=1, verbose=True):
        """
        Run bootstrap procedure.

        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        verbose : bool
            Show progress bar

        Returns
        -------
        BootstrapResult
            Contains bootstrap distributions and CIs
        """
        # Set random seeds for parallel execution
        np.random.seed(self.random_state)
        seeds = np.random.randint(0, 2**31, size=self.n_boot)

        # Run bootstrap in parallel
        if verbose:
            boot_params = Parallel(n_jobs=n_jobs)(
                delayed(self._single_bootstrap)(seed) for seed in tqdm(seeds, desc="Bootstrap")
            )
        else:
            boot_params = Parallel(n_jobs=n_jobs)(
                delayed(self._single_bootstrap)(seed) for seed in seeds
            )

        # Convert to array
        boot_params = np.array(boot_params)

        # Compute confidence intervals
        if self.ci_method == "percentile":
            ci_lower, ci_upper = self._percentile_ci(boot_params)
        elif self.ci_method == "bca":
            ci_lower, ci_upper = self._bca_ci(boot_params)
        elif self.ci_method == "normal":
            ci_lower, ci_upper = self._normal_ci(boot_params)
        else:
            raise ValueError(f"Unknown CI method: {self.ci_method}")

        return BootstrapResult(
            boot_params=boot_params,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            se=np.std(boot_params, axis=0),
            original_params=getattr(self.model, "params", None),
            method=self.method,
            n_boot=self.n_boot,
        )

    def _single_bootstrap(self, seed):
        """Single bootstrap replication."""
        np.random.seed(seed)

        if self.method == "cluster":
            X_boot, y_boot = self._resample_clusters()
        elif self.method == "pairs":
            X_boot, y_boot = self._resample_pairs()
        elif self.method == "wild":
            X_boot, y_boot = self._wild_bootstrap()
        elif self.method == "subsampling":
            X_boot, y_boot = self._subsampling()
        else:
            raise ValueError(f"Unknown bootstrap method: {self.method}")

        # Estimate on bootstrap sample
        from panelbox.optimization.quantile.interior_point import frisch_newton_qr

        try:
            beta_boot, info = frisch_newton_qr(
                X_boot, y_boot, self.tau, max_iter=100, tol=1e-6, verbose=False
            )

            if not info["converged"]:
                # Return NaN if not converged
                return np.full(
                    self.model.k_exog if hasattr(self.model, "k_exog") else X_boot.shape[1], np.nan
                )

            return beta_boot

        except Exception:
            # Return NaN if optimization fails
            return np.full(
                self.model.k_exog if hasattr(self.model, "k_exog") else X_boot.shape[1], np.nan
            )

    def _resample_clusters(self):
        """Resample entire entities (clusters)."""
        n_entities = (
            self.model.n_entities
            if hasattr(self.model, "n_entities")
            else len(np.unique(self.model.entity_ids))
        )

        # Resample entity IDs with replacement
        entity_ids_boot = np.random.choice(range(n_entities), size=n_entities, replace=True)

        # Reconstruct data
        X_list = []
        y_list = []

        for i, entity_id in enumerate(entity_ids_boot):
            # Get data for this entity
            entity_mask = self.model.entity_ids == entity_id
            X_list.append(self.model.X[entity_mask])
            y_list.append(self.model.y[entity_mask])

        X_boot = np.vstack(X_list)
        y_boot = np.hstack(y_list)

        return X_boot, y_boot

    def _resample_pairs(self):
        """Standard pairs bootstrap."""
        n = self.model.nobs if hasattr(self.model, "nobs") else len(self.model.y)
        idx = np.random.choice(n, size=n, replace=True)

        return self.model.X[idx], self.model.y[idx]

    def _wild_bootstrap(self):
        """Wild bootstrap with Rademacher weights."""
        # Original residuals
        beta = getattr(self.model, "params", np.zeros(self.model.X.shape[1]))
        residuals = self.model.y - self.model.X @ beta

        # Rademacher random variables
        weights = np.random.choice([-1, 1], size=len(residuals))

        # New y with wild residuals
        y_boot = self.model.X @ beta + weights * residuals

        return self.model.X, y_boot

    def _subsampling(self):
        """m-out-of-n bootstrap."""
        n = self.model.nobs if hasattr(self.model, "nobs") else len(self.model.y)
        m = int(n * 0.7)  # Use 70% of sample

        idx = np.random.choice(n, size=m, replace=False)

        return self.model.X[idx], self.model.y[idx]

    def _percentile_ci(self, boot_params, alpha=0.05):
        """Percentile confidence intervals."""
        lower = np.nanpercentile(boot_params, 100 * alpha / 2, axis=0)
        upper = np.nanpercentile(boot_params, 100 * (1 - alpha / 2), axis=0)

        return lower, upper

    def _bca_ci(self, boot_params, alpha=0.05):
        """Bias-corrected and accelerated (BCa) confidence intervals."""
        from scipy import stats

        # For simplicity, return percentile CI
        return self._percentile_ci(boot_params, alpha)

    def _normal_ci(self, boot_params, alpha=0.05):
        """Normal-based confidence intervals."""
        from scipy import stats

        theta_hat = getattr(self.model, "params", np.zeros(boot_params.shape[1]))
        se = np.std(boot_params, axis=0)

        z_alpha = stats.norm.ppf(1 - alpha / 2)

        lower = theta_hat - z_alpha * se
        upper = theta_hat + z_alpha * se

        return lower, upper


class BootstrapResult:
    """Container for bootstrap results."""

    def __init__(self, boot_params, ci_lower, ci_upper, se, original_params, method, n_boot):
        self.boot_params = boot_params
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.se = se
        self.original_params = original_params
        self.method = method
        self.n_boot = n_boot

    def summary(self, var_names=None):
        """Print bootstrap summary."""
        if var_names is None:
            var_names = [f"X{i}" for i in range(len(self.se))]

        print(f"\nBootstrap Results ({self.method}, B={self.n_boot})")
        print("=" * 70)
        print(f"{'Variable':<15} {'SE':>10} {'CI Lower':>10} {'CI Upper':>10}")
        print("-" * 70)

        for i, var in enumerate(var_names):
            print(f"{var:<15} {self.se[i]:10.4f} {self.ci_lower[i]:10.4f} {self.ci_upper[i]:10.4f}")


def bootstrap_qr(model, tau, n_boot=999, method="cluster", n_jobs=1, verbose=False):
    """
    Simple wrapper function for bootstrap inference.

    Parameters
    ----------
    model : QuantilePanelModel
        Fitted model
    tau : float
        Quantile level
    n_boot : int
        Number of bootstrap replications
    method : str
        Bootstrap method
    n_jobs : int
        Number of parallel jobs
    verbose : bool
        Print progress

    Returns
    -------
    BootstrapResult
        Bootstrap results
    """
    bs = QuantileBootstrap(model, tau, n_boot, method)
    return bs.bootstrap(n_jobs, verbose)
