"""
Bootstrap inference for Stochastic Frontier Analysis.

This module implements parametric and pairs bootstrap methods for
constructing confidence intervals and performing inference in SFA models.

Bootstrap is particularly useful for:
1. Constructing CI for parameters when asymptotics are unreliable
2. Inference on efficiency scores (small-sample correction)
3. Hypothesis tests with non-standard distributions
4. Model selection and averaging

References:
    Simar, L., & Wilson, P. W. (1998). Sensitivity analysis of efficiency
        scores: How to bootstrap in nonparametric frontier models.
        Management Science, 44(1), 49-61.

    Simar, L., & Wilson, P. W. (2000). Statistical inference in nonparametric
        frontier models: The state of the art. Journal of Productivity Analysis,
        13(1), 49-78.

    Kumbhakar, S. C., & Lovell, C. K. (2000). Stochastic Frontier Analysis.
        Cambridge University Press. Chapter 7.

Author: PanelBox Development Team
Date: 2026-02-15
"""

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats


class SFABootstrap:
    """Bootstrap inference for SFA models.

    This class provides parametric and pairs bootstrap methods for
    constructing confidence intervals for parameters and efficiency scores.

    Attributes:
        result: SFResult object from fitted SFA model
        method: Bootstrap method ('parametric' or 'pairs')
        n_boot: Number of bootstrap replications
        ci_level: Confidence level (default: 0.95)
        seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all cores)

    Example:
        >>> result = sf.fit()
        >>> bootstrap = SFABootstrap(result, n_boot=999, method='parametric')
        >>> ci_params = bootstrap.bootstrap_parameters()
        >>> ci_efficiency = bootstrap.bootstrap_efficiency()
    """

    def __init__(
        self,
        result,  # SFResult
        method: Literal["parametric", "pairs"] = "parametric",
        n_boot: int = 999,
        ci_level: float = 0.95,
        seed: Optional[int] = None,
        n_jobs: int = -1,
    ):
        """
        Initialize bootstrap inference.

        Parameters:
            result: Fitted SFResult object
            method: Bootstrap method
                - 'parametric': Resample from estimated distributions
                - 'pairs': Resample (y, X) pairs with replacement
            n_boot: Number of bootstrap replications
            ci_level: Confidence level for intervals (0 < ci_level < 1)
            seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 uses all cores)
        """
        self.result = result
        self.method = method
        self.n_boot = n_boot
        self.ci_level = ci_level
        self.seed = seed
        self.n_jobs = n_jobs

        # Validate
        if not 0 < ci_level < 1:
            raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")

        if n_boot < 100:
            raise ValueError(f"n_boot should be >= 100, got {n_boot}")

        # Store original data
        self.y = result.model.y
        self.X = result.model.X
        self.n_obs = len(self.y)

        # Random number generator
        self.rng = np.random.default_rng(seed)

    def bootstrap_parameters(self) -> dict:
        """
        Bootstrap confidence intervals for model parameters.

        Returns dictionary with:
            - params_boot: (n_boot × n_params) array of bootstrap estimates
            - ci_lower: Lower CI bounds for each parameter
            - ci_upper: Upper CI bounds for each parameter
            - mean_boot: Mean of bootstrap estimates
            - std_boot: Standard deviation of bootstrap estimates
            - bias: Bootstrap bias estimate

        Returns:
            dict: Bootstrap results for parameters
        """
        print(f"Bootstrap: {self.n_boot} replications using {self.method} method...")

        # Run bootstrap replications in parallel
        if self.method == "parametric":
            boot_results = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed(self._bootstrap_rep_parametric)(b) for b in range(self.n_boot)
            )
        else:  # pairs
            boot_results = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed(self._bootstrap_rep_pairs)(b) for b in range(self.n_boot)
            )

        # Collect parameter estimates
        params_boot = []
        for res in boot_results:
            if res is not None and hasattr(res, "params"):
                params_boot.append(res.params.values)
            else:
                # Failed convergence: use NaN
                params_boot.append(np.full(len(self.result.params), np.nan))

        params_boot = np.array(params_boot)

        # Remove failed replications
        valid_mask = ~np.isnan(params_boot).any(axis=1)
        n_failed = (~valid_mask).sum()
        if n_failed > 0:
            print(f"Warning: {n_failed}/{self.n_boot} bootstrap reps failed to converge")

        params_boot_valid = params_boot[valid_mask]

        # Compute confidence intervals (percentile method)
        alpha = 1 - self.ci_level
        ci_lower = np.percentile(params_boot_valid, 100 * alpha / 2, axis=0)
        ci_upper = np.percentile(params_boot_valid, 100 * (1 - alpha / 2), axis=0)

        # Statistics
        mean_boot = np.mean(params_boot_valid, axis=0)
        std_boot = np.std(params_boot_valid, axis=0, ddof=1)

        # Bias: E[θ̂*] - θ̂
        bias = mean_boot - self.result.params.values

        # Create DataFrame for easy viewing
        results_df = pd.DataFrame(
            {
                "parameter": self.result.params.index,
                "estimate": self.result.params.values,
                "boot_mean": mean_boot,
                "boot_std": std_boot,
                "bias": bias,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

        return {
            "params_boot": params_boot_valid,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mean_boot": mean_boot,
            "std_boot": std_boot,
            "bias": bias,
            "results_df": results_df,
            "n_valid": valid_mask.sum(),
            "n_failed": n_failed,
        }

    def bootstrap_efficiency(self, estimator: str = "bc") -> pd.DataFrame:
        """
        Bootstrap confidence intervals for efficiency scores.

        This is useful when the Horrace-Schmidt approach gives imprecise
        intervals or when dealing with small samples.

        Parameters:
            estimator: Efficiency estimator ('bc' or 'jlms')

        Returns:
            DataFrame with columns:
                - entity: Entity identifier (if panel)
                - time: Time identifier (if panel)
                - efficiency: Point estimate
                - boot_mean: Mean of bootstrap estimates
                - boot_std: Standard deviation
                - ci_lower: Lower CI bound
                - ci_upper: Upper CI bound
        """
        print(f"Bootstrapping efficiency scores ({estimator} estimator)...")

        # Run bootstrap replications
        if self.method == "parametric":
            boot_results = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed(self._bootstrap_rep_parametric)(b) for b in range(self.n_boot)
            )
        else:
            boot_results = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed(self._bootstrap_rep_pairs)(b) for b in range(self.n_boot)
            )

        # Collect efficiency estimates
        eff_boot = []
        for res in boot_results:
            if res is not None and hasattr(res, "efficiency"):
                try:
                    eff = res.efficiency(estimator=estimator)
                    eff_boot.append(eff["te"].values)
                except Exception:
                    # Failed: use NaN
                    eff_boot.append(np.full(self.n_obs, np.nan))
            else:
                eff_boot.append(np.full(self.n_obs, np.nan))

        eff_boot = np.array(eff_boot)  # (n_boot × n_obs)

        # Remove failed replications
        valid_mask = ~np.isnan(eff_boot).any(axis=1)
        n_failed = (~valid_mask).sum()
        if n_failed > 0:
            print(f"Warning: {n_failed}/{self.n_boot} efficiency bootstrap reps failed")

        eff_boot_valid = eff_boot[valid_mask]

        # Compute intervals
        alpha = 1 - self.ci_level
        ci_lower = np.percentile(eff_boot_valid, 100 * alpha / 2, axis=0)
        ci_upper = np.percentile(eff_boot_valid, 100 * (1 - alpha / 2), axis=0)

        # Statistics
        mean_boot = np.mean(eff_boot_valid, axis=0)
        std_boot = np.std(eff_boot_valid, axis=0, ddof=1)

        # Original efficiency
        eff_original = self.result.efficiency(estimator=estimator)

        # Create results DataFrame
        results = eff_original.copy()
        results["boot_mean"] = mean_boot
        results["boot_std"] = std_boot
        results["ci_lower"] = ci_lower
        results["ci_upper"] = ci_upper

        # Clip to valid efficiency range [0, 1]
        results["ci_lower"] = np.clip(results["ci_lower"], 0, 1)
        results["ci_upper"] = np.clip(results["ci_upper"], 0, 1)

        return results

    def _bootstrap_rep_parametric(self, b: int):
        """
        Single parametric bootstrap replication.

        Steps:
        1. Generate v* ~ N(0, σ̂²_v)
        2. Generate u* from estimated distribution (e.g., N⁺(0, σ̂²_u))
        3. Construct y* = Xβ̂ + v* - sign*u* (sign depends on frontier type)
        4. Re-estimate model on (y*, X)

        Parameters:
            b: Bootstrap replication index (for seeding)

        Returns:
            SFResult object or None if failed
        """
        # Seed for this replication
        rng = np.random.default_rng(self.seed + b if self.seed is not None else None)

        # Step 1: Generate v* ~ N(0, σ̂²_v)
        v_star = rng.normal(0, self.result.sigma_v, self.n_obs)

        # Step 2: Generate u* from estimated distribution
        dist_type = self.result.model.dist

        if dist_type == "half_normal":
            # u* ~ |N(0, σ̂²_u)|
            u_star = np.abs(rng.normal(0, self.result.sigma_u, self.n_obs))

        elif dist_type == "exponential":
            # u* ~ Exp(λ = 1/σ_u)
            u_star = rng.exponential(self.result.sigma_u, self.n_obs)

        elif dist_type == "truncated_normal":
            # u* ~ N⁺(μ̂, σ̂²_u)
            mu = self.result.params.get("mu", 0.0)
            u_star = stats.truncnorm.rvs(
                a=-mu / self.result.sigma_u,  # lower bound (0)
                b=np.inf,
                loc=mu,
                scale=self.result.sigma_u,
                size=self.n_obs,
                random_state=rng,
            )

        else:
            raise NotImplementedError(f"Bootstrap for {dist_type} not implemented")

        # Step 3: Construct y* = Xβ̂ + v* - sign*u*
        # Sign depends on frontier type
        sign = -1 if self.result.model.frontier_type == "production" else 1

        y_frontier = self.X @ self.result.params[: self.X.shape[1]].values
        y_star = y_frontier + v_star + sign * u_star

        # Step 4: Re-estimate model
        try:
            # Create new model with same specification
            from panelbox.frontier import StochasticFrontier

            # Prepare data
            data_star = pd.DataFrame({"y_star": y_star})
            for j in range(self.X.shape[1] - 1):  # exclude const
                data_star[f"x{j}"] = self.X[:, j + 1]

            exog_vars = [f"x{j}" for j in range(self.X.shape[1] - 1)]

            # Create and fit model
            sf_star = StochasticFrontier(
                data=data_star,
                depvar="y_star",
                exog=exog_vars if len(exog_vars) > 0 else None,
                frontier=self.result.model.frontier_type,
                dist=self.result.model.dist,
            )

            # Fit WITHOUT starting values to avoid convergence to same point
            # Starting values can cause bootstrap to converge to identical values
            result_star = sf_star.fit(
                method="mle",
                maxiter=200,  # allow more iterations since no starting values
            )

            return result_star

        except Exception as e:
            # Convergence failure
            return None

    def _bootstrap_rep_pairs(self, b: int):
        """
        Single pairs bootstrap replication.

        Steps:
        1. Resample (y, X) with replacement
        2. Re-estimate model on resampled data

        Parameters:
            b: Bootstrap replication index

        Returns:
            SFResult object or None if failed
        """
        rng = np.random.default_rng(self.seed + b if self.seed is not None else None)

        # Resample indices with replacement
        indices = rng.choice(self.n_obs, size=self.n_obs, replace=True)

        # Resample data
        y_star = self.y[indices]
        X_star = self.X[indices]

        # Re-estimate
        try:
            from panelbox.frontier import StochasticFrontier

            # Prepare data
            data_star = pd.DataFrame({"y_star": y_star})
            for j in range(X_star.shape[1] - 1):
                data_star[f"x{j}"] = X_star[:, j + 1]

            exog_vars = [f"x{j}" for j in range(X_star.shape[1] - 1)]

            # Fit
            sf_star = StochasticFrontier(
                data=data_star,
                depvar="y_star",
                exog=exog_vars if len(exog_vars) > 0 else None,
                frontier=self.result.model.frontier_type,
                dist=self.result.model.dist,
            )

            # Fit WITHOUT starting values to avoid convergence to same point
            result_star = sf_star.fit(
                method="mle",
                maxiter=200,
            )

            return result_star

        except Exception:
            return None

    def bias_corrected_ci(self, param_name: str) -> tuple:
        """
        Bias-corrected accelerated (BCa) bootstrap confidence interval.

        This is more accurate than percentile method, especially when
        the bootstrap distribution is biased or skewed.

        Parameters:
            param_name: Name of parameter

        Returns:
            (lower, upper): BCa confidence interval
        """
        # Run bootstrap if not done yet
        if not hasattr(self, "_boot_results"):
            self._boot_results = self.bootstrap_parameters()

        params_boot = self._boot_results["params_boot"]
        param_idx = list(self.result.params.index).index(param_name)
        theta_hat = self.result.params[param_name]
        theta_boot = params_boot[:, param_idx]

        # Bias correction factor
        z0 = stats.norm.ppf(np.mean(theta_boot < theta_hat))

        # Acceleration factor (via jackknife)
        # TODO: Implement jackknife for acceleration
        # For now, use z0 only (BC instead of BCa)
        a = 0  # acceleration

        # Adjusted percentiles
        alpha = 1 - self.ci_level
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

        # BCa adjustments
        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

        # Compute percentiles
        ci_lower = np.percentile(theta_boot, 100 * p_lower)
        ci_upper = np.percentile(theta_boot, 100 * p_upper)

        return (ci_lower, ci_upper)


# ============================================================================
# Convenience functions
# ============================================================================


def bootstrap_sfa(
    result,
    n_boot: int = 999,
    method: Literal["parametric", "pairs"] = "parametric",
    ci_level: float = 0.95,
    seed: Optional[int] = None,
    n_jobs: int = -1,
) -> dict:
    """
    Convenience function for bootstrap inference on SFA parameters.

    Parameters:
        result: Fitted SFResult object
        n_boot: Number of bootstrap replications
        method: Bootstrap method ('parametric' or 'pairs')
        ci_level: Confidence level
        seed: Random seed
        n_jobs: Number of parallel jobs

    Returns:
        dict: Bootstrap results

    Example:
        >>> result = sf.fit()
        >>> boot_results = bootstrap_sfa(result, n_boot=999, method='parametric')
        >>> print(boot_results['results_df'])
    """
    boot = SFABootstrap(
        result=result,
        method=method,
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
        n_jobs=n_jobs,
    )

    return boot.bootstrap_parameters()


def bootstrap_efficiency(
    result,
    estimator: str = "bc",
    n_boot: int = 999,
    method: Literal["parametric", "pairs"] = "parametric",
    ci_level: float = 0.95,
    seed: Optional[int] = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Convenience function for bootstrap inference on efficiency scores.

    Parameters:
        result: Fitted SFResult object
        estimator: Efficiency estimator ('bc' or 'jlms')
        n_boot: Number of bootstrap replications
        method: Bootstrap method
        ci_level: Confidence level
        seed: Random seed
        n_jobs: Number of parallel jobs

    Returns:
        DataFrame with efficiency CIs

    Example:
        >>> eff_ci = bootstrap_efficiency(result, n_boot=999)
        >>> print(eff_ci[['te', 'ci_lower', 'ci_upper']])
    """
    boot = SFABootstrap(
        result=result,
        method=method,
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
        n_jobs=n_jobs,
    )

    return boot.bootstrap_efficiency(estimator=estimator)
