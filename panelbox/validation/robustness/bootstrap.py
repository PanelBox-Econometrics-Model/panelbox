"""
Bootstrap inference for panel data models.

This module implements various bootstrap methods for panel data, including:
- Pairs bootstrap (entity resampling)
- Wild bootstrap (heteroskedasticity-robust)
- Block bootstrap (time series dependence)
- Residual bootstrap (i.i.d. errors)

References
----------
Cameron, A. C., & Trivedi, P. K. (2005). Microeconometrics: Methods and Applications.
Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap.
"""

import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from panelbox.core.results import PanelResults


class PanelBootstrap:
    """
    Bootstrap inference for panel data models.

    This class implements various bootstrap methods adapted for panel data structure.
    Bootstrap resampling provides an alternative to asymptotic inference, particularly
    useful for small samples or complex dependence structures.

    Parameters
    ----------
    results : PanelResults
        Fitted model results to bootstrap
    n_bootstrap : int, default=1000
        Number of bootstrap replications
    method : {'pairs', 'wild', 'block', 'residual'}, default='pairs'
        Bootstrap method to use:

        - 'pairs': Resample entire entities (recommended for most cases)
        - 'wild': Keep X fixed, resample residuals with random weights
        - 'block': Resample blocks of time periods
        - 'residual': Resample residuals (assumes i.i.d. errors)
    block_size : int, optional
        Block size for block bootstrap. If None, uses rule-of-thumb: T^(1/3)
    random_state : int, optional
        Random seed for reproducibility
    show_progress : bool, default=True
        Show progress bar during bootstrap
    parallel : bool, default=False
        Use parallel processing (not yet implemented)

    Attributes
    ----------
    bootstrap_estimates_ : np.ndarray
        Bootstrap coefficient estimates (n_bootstrap x n_params)
    bootstrap_se_ : np.ndarray
        Bootstrap standard errors for each parameter
    bootstrap_t_stats_ : np.ndarray
        Bootstrap t-statistics (for studentized bootstrap)
    n_failed_ : int
        Number of failed bootstrap replications

    Examples
    --------
    >>> import panelbox as pb
    >>> import pandas as pd
    >>>
    >>> # Fit a fixed effects model
    >>> data = pd.read_csv('panel_data.csv')
    >>> fe = pb.FixedEffects("y ~ x1 + x2", data, "id", "time")
    >>> results = fe.fit()
    >>>
    >>> # Bootstrap with pairs method (recommended)
    >>> bootstrap = pb.PanelBootstrap(
    ...     results,
    ...     n_bootstrap=1000,
    ...     method='pairs',
    ...     random_state=42
    ... )
    >>> bootstrap.run()
    >>>
    >>> # Get bootstrap confidence intervals
    >>> ci = bootstrap.conf_int(alpha=0.05, method='percentile')
    >>> print(ci)
    >>>
    >>> # Compare with asymptotic CI
    >>> ci_asymp = results.conf_int(alpha=0.05)
    >>> print(ci_asymp)
    >>>
    >>> # Get bootstrap standard errors
    >>> se_boot = bootstrap.bootstrap_se_
    >>> se_asymp = results.std_errors
    >>> comparison = pd.DataFrame({
    ...     'Bootstrap SE': se_boot,
    ...     'Asymptotic SE': se_asymp
    ... }, index=results.params.index)
    >>> print(comparison)

    Notes
    -----
    **Choosing a Bootstrap Method:**

    1. **Pairs Bootstrap** (default, recommended):
       - Resamples entire entities (with all their time periods)
       - Preserves within-entity correlation structure
       - Robust to heteroskedasticity and serial correlation
       - Use when entities are independent

    2. **Wild Bootstrap**:
       - Keeps X fixed, resamples residuals with random weights
       - Specifically designed for heteroskedasticity
       - Does not account for serial correlation
       - Use when heteroskedasticity is primary concern

    3. **Block Bootstrap**:
       - Resamples blocks of consecutive time periods
       - Preserves temporal dependence within blocks
       - Use when time-series dependence is important

    4. **Residual Bootstrap**:
       - Resamples residuals assuming i.i.d. errors
       - Most restrictive assumptions
       - Use only when you're confident errors are i.i.d.

    **Number of Replications:**

    - For standard errors: n_bootstrap >= 500
    - For confidence intervals: n_bootstrap >= 1000
    - For hypothesis testing: n_bootstrap >= 2000

    References
    ----------
    Cameron, A. C., & Trivedi, P. K. (2005). "Microeconometrics: Methods and
    Applications." Cambridge University Press, Chapter 11.

    Efron, B., & Tibshirani, R. J. (1994). "An Introduction to the Bootstrap."
    Chapman and Hall/CRC.

    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). "Bootstrap-based
    improvements for inference with clustered errors." *The Review of Economics
    and Statistics*, 90(3), 414-427.
    """

    def __init__(
        self,
        results: PanelResults,
        n_bootstrap: int = 1000,
        method: Literal["pairs", "wild", "block", "residual"] = "pairs",
        block_size: Optional[int] = None,
        random_state: Optional[int] = None,
        show_progress: bool = True,
        parallel: bool = False,
    ):
        # Validation
        if not isinstance(results, PanelResults):
            raise TypeError(f"results must be PanelResults, got {type(results)}")

        if n_bootstrap < 100:
            warnings.warn(
                f"n_bootstrap={n_bootstrap} is quite small. "
                "Recommend at least 500 for standard errors, 1000 for confidence intervals.",
                UserWarning,
            )

        valid_methods = ["pairs", "wild", "block", "residual"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

        # Store inputs
        self.results = results
        self.n_bootstrap = n_bootstrap
        self.method = method
        self.block_size = block_size
        self.random_state = random_state
        self.show_progress = show_progress
        self.parallel = parallel

        # Check if model is available for refitting
        if results._model is None:
            raise ValueError(
                "Bootstrap requires access to the original model. "
                "Ensure the model stores a reference to itself in results._model"
            )

        self.model = results._model

        # Initialize random state
        self.rng = np.random.RandomState(random_state)

        # Check for parallel (not yet implemented)
        if parallel:
            warnings.warn(
                "Parallel processing not yet implemented. Running sequentially.", UserWarning
            )

        # Results storage
        self.bootstrap_estimates_: Optional[np.ndarray] = None
        self.bootstrap_se_: Optional[np.ndarray] = None
        self.bootstrap_t_stats_: Optional[np.ndarray] = None
        self.n_failed_: int = 0
        self._fitted = False

    def run(self) -> "PanelBootstrap":
        """
        Run bootstrap procedure.

        Performs bootstrap resampling according to the specified method
        and stores results.

        Returns
        -------
        self : PanelBootstrap
            Returns self for method chaining

        Raises
        ------
        ValueError
            If bootstrap method is not recognized
        RuntimeError
            If too many bootstrap replications fail
        """
        # Dispatch to appropriate method
        if self.method == "pairs":
            estimates = self._bootstrap_pairs()
        elif self.method == "wild":
            estimates = self._bootstrap_wild()
        elif self.method == "block":
            estimates = self._bootstrap_block()
        elif self.method == "residual":
            estimates = self._bootstrap_residual()
        else:
            raise ValueError(f"Unknown bootstrap method: {self.method}")

        # Store results
        self.bootstrap_estimates_ = estimates

        # Compute bootstrap standard errors
        self.bootstrap_se_ = np.std(estimates, axis=0, ddof=1)

        # Compute studentized statistics (for advanced CI methods)
        # t_b = (θ_b - θ_hat) / se(θ_b)
        theta_hat = self.results.params.values
        self.bootstrap_t_stats_ = (estimates - theta_hat) / self.bootstrap_se_

        self._fitted = True

        # Warn if many failures
        if self.n_failed_ > self.n_bootstrap * 0.1:
            warnings.warn(
                f"{self.n_failed_} out of {self.n_bootstrap} bootstrap replications failed "
                f"({self.n_failed_/self.n_bootstrap*100:.1f}%). "
                "Results may be unreliable. Consider using a different method or "
                "checking your model specification.",
                UserWarning,
            )

        return self

    def _bootstrap_pairs(self) -> np.ndarray:
        """
        Pairs (entity) bootstrap.

        Resamples entire entities with replacement. This preserves the
        within-entity correlation structure and is robust to both
        heteroskedasticity and serial correlation within entities.

        Returns
        -------
        estimates : np.ndarray
            Bootstrap coefficient estimates (n_bootstrap x n_params)
        """
        # Get entity IDs
        data_df = self.model.data.data
        entity_col = self.model.data.entity_col
        entities = data_df[entity_col].unique()
        n_entities = len(entities)

        # Storage for estimates
        n_params = len(self.results.params)
        estimates = np.zeros((self.n_bootstrap, n_params))

        # Bootstrap loop
        iterator = range(self.n_bootstrap)
        if self.show_progress:
            iterator = tqdm(iterator, desc=f"Bootstrap ({self.method})")

        for b in iterator:
            try:
                # Resample entities with replacement
                boot_entities = self.rng.choice(entities, size=n_entities, replace=True)

                # Create bootstrap sample by stacking selected entities
                boot_data_list = []
                for entity in boot_entities:
                    entity_data = data_df[data_df[entity_col] == entity].copy()
                    boot_data_list.append(entity_data)

                boot_data = pd.concat(boot_data_list, ignore_index=True)

                # Refit model on bootstrap sample
                # We need to create a new model instance with bootstrap data
                boot_model = self._create_bootstrap_model(boot_data)
                boot_results = boot_model.fit()

                # Store estimates
                estimates[b, :] = boot_results.params.values

            except Exception as e:
                # If estimation fails, use NaN
                estimates[b, :] = np.nan
                self.n_failed_ += 1

                if self.show_progress and self.n_failed_ <= 5:
                    # Print first few failures for debugging
                    print(f"\nBootstrap iteration {b} failed: {str(e)}")

        # Remove failed replications
        valid_mask = ~np.isnan(estimates).any(axis=1)
        estimates = estimates[valid_mask, :]

        if estimates.shape[0] < self.n_bootstrap * 0.5:
            raise RuntimeError(
                f"More than 50% of bootstrap replications failed. "
                f"Only {estimates.shape[0]} out of {self.n_bootstrap} succeeded. "
                "Check your model specification."
            )

        return estimates

    def _bootstrap_wild(self) -> np.ndarray:
        """
        Wild bootstrap.

        Keeps X fixed and resamples residuals with random weights.
        Designed for heteroskedasticity but does not account for
        serial correlation.

        Uses Rademacher distribution: w ∈ {-1, +1} with equal probability.
        Alternative: Mammen distribution (set via _wild_distribution attribute).

        Returns
        -------
        estimates : np.ndarray
            Bootstrap coefficient estimates (n_bootstrap x n_params)

        Notes
        -----
        Wild bootstrap is particularly useful for heteroskedastic errors.
        It maintains the X matrix fixed and only resamples the error structure.

        For panel data, this method resamples residuals for each observation
        independently, which may not preserve serial correlation within entities.
        Consider pairs bootstrap if serial correlation is a concern.

        References
        ----------
        Liu, R. Y. (1988). "Bootstrap procedures under some non-i.i.d. models."
        *The Annals of Statistics*, 16(4), 1696-1708.
        """
        # Get data
        data_df = self.model.data.data
        self.model.data.entity_col
        self.model.data.time_col

        # Get residuals and fitted values from original model
        residuals = self.results.resid
        fitted_values = self.results.fittedvalues

        # Get original data in same order
        # We need to reconstruct y from fitted + residuals
        fitted_values + residuals

        # Storage for estimates
        n_params = len(self.results.params)
        estimates = np.zeros((self.n_bootstrap, n_params))

        # Bootstrap loop
        iterator = range(self.n_bootstrap)
        if self.show_progress:
            iterator = tqdm(iterator, desc=f"Bootstrap ({self.method})")

        for b in iterator:
            try:
                # Generate wild bootstrap weights using Rademacher distribution
                # w ∈ {-1, +1} with probability 0.5 each
                weights = self.rng.choice([-1, 1], size=len(residuals))

                # Create bootstrap residuals: e* = w * e
                boot_residuals = weights * residuals

                # Reconstruct bootstrap outcome: y* = ŷ + e*
                y_boot = fitted_values + boot_residuals

                # Create bootstrap dataset with new y values
                boot_data = data_df.copy()

                # Get the dependent variable name from the formula
                dep_var = self.model.formula_parser.dependent
                boot_data[dep_var] = y_boot

                # Refit model on bootstrap sample
                boot_model = self._create_bootstrap_model(boot_data)
                boot_results = boot_model.fit()

                # Store estimates
                estimates[b, :] = boot_results.params.values

            except Exception as e:
                # If estimation fails, use NaN
                estimates[b, :] = np.nan
                self.n_failed_ += 1

                if self.show_progress and self.n_failed_ <= 5:
                    print(f"\nBootstrap iteration {b} failed: {str(e)}")

        # Remove failed replications
        valid_mask = ~np.isnan(estimates).any(axis=1)
        estimates = estimates[valid_mask, :]

        if estimates.shape[0] < self.n_bootstrap * 0.5:
            raise RuntimeError(
                f"More than 50% of bootstrap replications failed. "
                f"Only {estimates.shape[0]} out of {self.n_bootstrap} succeeded. "
                "Check your model specification."
            )

        return estimates

    def _bootstrap_block(self) -> np.ndarray:
        """
        Block bootstrap.

        Resamples blocks of consecutive time periods. Preserves temporal
        dependence within blocks.

        Uses moving block bootstrap where blocks can overlap. Block size
        is determined by the block_size parameter, or defaults to T^(1/3).

        Returns
        -------
        estimates : np.ndarray
            Bootstrap coefficient estimates (n_bootstrap x n_params)

        Notes
        -----
        Block bootstrap is useful when there is temporal dependence in the data.
        It preserves the correlation structure within blocks while breaking
        dependence between blocks.

        For panel data, blocks are time periods that apply to all entities.
        This maintains the cross-sectional structure while accounting for
        time-series dependence.

        References
        ----------
        Künsch, H. R. (1989). "The jackknife and the bootstrap for general
        stationary observations." *The Annals of Statistics*, 17(3), 1217-1241.
        """
        # Get data
        data_df = self.model.data.data
        self.model.data.entity_col
        time_col = self.model.data.time_col

        # Get unique time periods
        time_periods = sorted(data_df[time_col].unique())
        n_periods = len(time_periods)

        # Determine block size
        if self.block_size is None:
            # Rule of thumb: T^(1/3)
            block_size = max(1, int(np.ceil(n_periods ** (1 / 3))))
            if self.show_progress:
                print(f"\nUsing automatic block size: {block_size} (T^(1/3) where T={n_periods})")
        else:
            block_size = self.block_size

        if block_size > n_periods:
            warnings.warn(
                f"block_size={block_size} is larger than n_periods={n_periods}. "
                f"Setting block_size={n_periods}",
                UserWarning,
            )
            block_size = n_periods

        # Storage for estimates
        n_params = len(self.results.params)
        estimates = np.zeros((self.n_bootstrap, n_params))

        # Bootstrap loop
        iterator = range(self.n_bootstrap)
        if self.show_progress:
            iterator = tqdm(iterator, desc=f"Bootstrap ({self.method}, block_size={block_size})")

        for b in iterator:
            try:
                # Resample blocks to cover approximately n_periods
                boot_time_periods = []
                n_blocks_needed = int(np.ceil(n_periods / block_size))

                for _ in range(n_blocks_needed):
                    # Randomly select a starting point for the block
                    start_idx = self.rng.randint(0, n_periods - block_size + 1)

                    # Extract block of time periods
                    block = time_periods[start_idx : start_idx + block_size]
                    boot_time_periods.extend(block)

                # Trim to original length
                boot_time_periods = boot_time_periods[:n_periods]

                # Create bootstrap sample by selecting these time periods
                boot_data_list = []
                for t in boot_time_periods:
                    time_data = data_df[data_df[time_col] == t].copy()
                    boot_data_list.append(time_data)

                boot_data = pd.concat(boot_data_list, ignore_index=True)

                # Refit model on bootstrap sample
                boot_model = self._create_bootstrap_model(boot_data)
                boot_results = boot_model.fit()

                # Store estimates
                estimates[b, :] = boot_results.params.values

            except Exception as e:
                # If estimation fails, use NaN
                estimates[b, :] = np.nan
                self.n_failed_ += 1

                if self.show_progress and self.n_failed_ <= 5:
                    print(f"\nBootstrap iteration {b} failed: {str(e)}")

        # Remove failed replications
        valid_mask = ~np.isnan(estimates).any(axis=1)
        estimates = estimates[valid_mask, :]

        if estimates.shape[0] < self.n_bootstrap * 0.5:
            raise RuntimeError(
                f"More than 50% of bootstrap replications failed. "
                f"Only {estimates.shape[0]} out of {self.n_bootstrap} succeeded. "
                "Check your model specification."
            )

        return estimates

    def _bootstrap_residual(self) -> np.ndarray:
        """
        Residual bootstrap.

        Resamples residuals assuming they are i.i.d. Most restrictive
        assumptions.

        Returns
        -------
        estimates : np.ndarray
            Bootstrap coefficient estimates (n_bootstrap x n_params)

        Notes
        -----
        Residual bootstrap assumes residuals are independent and identically
        distributed (i.i.d.). This is the most restrictive assumption among
        bootstrap methods.

        **When to use**:
        - When you're confident errors are i.i.d.
        - After conditioning on X (design matrix fixed)
        - For computational efficiency

        **When NOT to use**:
        - With heteroskedasticity (use wild bootstrap instead)
        - With serial correlation (use block or pairs bootstrap)
        - With clustered errors (use pairs bootstrap)

        The algorithm:
        1. Center residuals: e_centered = e - mean(e)
        2. Resample centered residuals with replacement
        3. Reconstruct y* = ŷ + e*
        4. Refit model

        References
        ----------
        Efron, B., & Tibshirani, R. J. (1994). "An Introduction to the
        Bootstrap." CRC press, Chapter 6.
        """
        # Get data
        data_df = self.model.data.data

        # Get residuals and fitted values from original model
        residuals = self.results.resid
        fitted_values = self.results.fittedvalues

        # Center residuals (important for maintaining mean zero)
        centered_residuals = residuals - np.mean(residuals)

        # Storage for estimates
        n_params = len(self.results.params)
        estimates = np.zeros((self.n_bootstrap, n_params))

        # Bootstrap loop
        iterator = range(self.n_bootstrap)
        if self.show_progress:
            iterator = tqdm(iterator, desc=f"Bootstrap ({self.method})")

        for b in iterator:
            try:
                # Resample centered residuals with replacement
                boot_indices = self.rng.choice(len(residuals), size=len(residuals), replace=True)
                boot_residuals = centered_residuals[boot_indices]

                # Reconstruct bootstrap outcome: y* = ŷ + e*
                y_boot = fitted_values + boot_residuals

                # Create bootstrap dataset with new y values
                boot_data = data_df.copy()

                # Get the dependent variable name from the formula
                dep_var = self.model.formula_parser.dependent
                boot_data[dep_var] = y_boot

                # Refit model on bootstrap sample
                boot_model = self._create_bootstrap_model(boot_data)
                boot_results = boot_model.fit()

                # Store estimates
                estimates[b, :] = boot_results.params.values

            except Exception as e:
                # If estimation fails, use NaN
                estimates[b, :] = np.nan
                self.n_failed_ += 1

                if self.show_progress and self.n_failed_ <= 5:
                    print(f"\nBootstrap iteration {b} failed: {str(e)}")

        # Remove failed replications
        valid_mask = ~np.isnan(estimates).any(axis=1)
        estimates = estimates[valid_mask, :]

        if estimates.shape[0] < self.n_bootstrap * 0.5:
            raise RuntimeError(
                f"More than 50% of bootstrap replications failed. "
                f"Only {estimates.shape[0]} out of {self.n_bootstrap} succeeded. "
                "Check your model specification."
            )

        return estimates

    def _create_bootstrap_model(self, boot_data: pd.DataFrame):
        """
        Create a new model instance with bootstrap data.

        Parameters
        ----------
        boot_data : pd.DataFrame
            Bootstrap sample data

        Returns
        -------
        model
            New model instance
        """
        # Get model class and parameters
        model_class = type(self.model)

        # Common parameters for all models
        init_kwargs = {
            "formula": self.model.formula,
            "data": boot_data,
            "entity_col": self.model.data.entity_col,
            "time_col": self.model.data.time_col,
        }

        # Add model-specific parameters
        if hasattr(self.model, "entity_effects"):
            init_kwargs["entity_effects"] = self.model.entity_effects
        if hasattr(self.model, "time_effects"):
            init_kwargs["time_effects"] = self.model.time_effects
        if hasattr(self.model, "weights"):
            init_kwargs["weights"] = self.model.weights

        # Create model instance
        boot_model = model_class(**init_kwargs)

        return boot_model

    def conf_int(
        self,
        alpha: float = 0.05,
        method: Literal["percentile", "basic", "bca", "studentized"] = "percentile",
    ) -> pd.DataFrame:
        """
        Compute bootstrap confidence intervals.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level (e.g., 0.05 for 95% CI)
        method : {'percentile', 'basic', 'bca', 'studentized'}, default='percentile'
            Method for computing confidence intervals:

            - 'percentile': Percentile method (simplest, recommended)
            - 'basic': Basic bootstrap (reflection method)
            - 'bca': Bias-corrected and accelerated (most accurate but complex)
            - 'studentized': Studentized bootstrap (requires nested bootstrap)

        Returns
        -------
        conf_int : pd.DataFrame
            Confidence intervals with columns 'lower' and 'upper'

        Examples
        --------
        >>> # After running bootstrap
        >>> ci_perc = bootstrap.conf_int(alpha=0.05, method='percentile')
        >>> ci_basic = bootstrap.conf_int(alpha=0.05, method='basic')
        >>>
        >>> # Compare with asymptotic
        >>> ci_asymp = results.conf_int(alpha=0.05)
        """
        if not self._fitted:
            raise RuntimeError("Must call run() before conf_int()")

        if method == "percentile":
            ci = self._conf_int_percentile(alpha)
        elif method == "basic":
            ci = self._conf_int_basic(alpha)
        elif method == "bca":
            ci = self._conf_int_bca(alpha)
        elif method == "studentized":
            ci = self._conf_int_studentized(alpha)
        else:
            raise ValueError(
                f"method must be 'percentile', 'basic', 'bca', or 'studentized', " f"got '{method}'"
            )

        return ci

    def _conf_int_percentile(self, alpha: float) -> pd.DataFrame:
        """
        Percentile confidence interval.

        CI = [θ_α/2, θ_1-α/2] where θ_p is the p-th percentile of bootstrap estimates.
        """
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        lower = np.percentile(self.bootstrap_estimates_, lower_pct, axis=0)
        upper = np.percentile(self.bootstrap_estimates_, upper_pct, axis=0)

        ci = pd.DataFrame({"lower": lower, "upper": upper}, index=self.results.params.index)

        return ci

    def _conf_int_basic(self, alpha: float) -> pd.DataFrame:
        """
        Basic (reflection) confidence interval.

        CI = [2θ_hat - θ_1-α/2, 2θ_hat - θ_α/2]
        """
        theta_hat = self.results.params.values

        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        # Note the reversal for basic method
        lower = 2 * theta_hat - np.percentile(self.bootstrap_estimates_, upper_pct, axis=0)
        upper = 2 * theta_hat - np.percentile(self.bootstrap_estimates_, lower_pct, axis=0)

        ci = pd.DataFrame({"lower": lower, "upper": upper}, index=self.results.params.index)

        return ci

    def _conf_int_bca(self, alpha: float) -> pd.DataFrame:
        """
        Bias-corrected and accelerated (BCa) confidence interval.

        More accurate than percentile but requires estimating bias and acceleration.
        """
        warnings.warn(
            "BCa confidence intervals not yet fully implemented. "
            "Falling back to percentile method.",
            UserWarning,
        )
        return self._conf_int_percentile(alpha)

    def _conf_int_studentized(self, alpha: float) -> pd.DataFrame:
        """
        Studentized (bootstrap-t) confidence interval.

        Uses bootstrap distribution of t-statistics. Most accurate but computationally
        intensive (requires nested bootstrap).
        """
        warnings.warn(
            "Studentized confidence intervals not yet fully implemented. "
            "Falling back to percentile method.",
            UserWarning,
        )
        return self._conf_int_percentile(alpha)

    def summary(self) -> pd.DataFrame:
        """
        Generate bootstrap summary table.

        Returns
        -------
        summary : pd.DataFrame
            Summary table with original estimates, bootstrap SEs, and comparison

        Examples
        --------
        >>> summary = bootstrap.summary()
        >>> print(summary)
        """
        if not self._fitted:
            raise RuntimeError("Must call run() before summary()")

        summary = pd.DataFrame(
            {
                "Original": self.results.params,
                "Bootstrap Mean": self.bootstrap_estimates_.mean(axis=0),
                "Bootstrap Bias": self.bootstrap_estimates_.mean(axis=0)
                - self.results.params.values,
                "Original SE": self.results.std_errors,
                "Bootstrap SE": self.bootstrap_se_,
                "SE Ratio": self.bootstrap_se_ / self.results.std_errors.values,
            },
            index=self.results.params.index,
        )

        return summary

    def plot_distribution(self, param: Optional[str] = None):
        """
        Plot bootstrap distribution of coefficients.

        Parameters
        ----------
        param : str, optional
            Parameter name to plot. If None, plots all parameters.

        Raises
        ------
        ImportError
            If matplotlib is not installed
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. " "Install with: pip install matplotlib"
            )

        if not self._fitted:
            raise RuntimeError("Must call run() before plot_distribution()")

        if param is not None:
            # Plot single parameter
            if param not in self.results.params.index:
                raise ValueError(f"Parameter '{param}' not found in model")

            param_idx = self.results.params.index.get_loc(param)
            boot_values = self.bootstrap_estimates_[:, param_idx]
            original_value = self.results.params.iloc[param_idx]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(boot_values, bins=50, alpha=0.7, edgecolor="black")
            ax.axvline(
                original_value,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Original: {original_value:.4f}",
            )
            ax.set_xlabel("Coefficient Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Bootstrap Distribution: {param}")
            ax.legend()
            plt.tight_layout()
            plt.show()
        else:
            # Plot all parameters
            n_params = len(self.results.params)
            n_cols = min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_params == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            for i, param_name in enumerate(self.results.params.index):
                boot_values = self.bootstrap_estimates_[:, i]
                original_value = self.results.params.iloc[i]

                axes[i].hist(boot_values, bins=30, alpha=0.7, edgecolor="black")
                axes[i].axvline(
                    original_value,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Original: {original_value:.4f}",
                )
                axes[i].set_xlabel("Value")
                axes[i].set_ylabel("Frequency")
                axes[i].set_title(param_name)
                axes[i].legend(fontsize=8)

            # Hide unused subplots
            for i in range(n_params, len(axes)):
                axes[i].set_visible(False)

            plt.suptitle(
                f"Bootstrap Distributions ({self.method} method, n={self.n_bootstrap})",
                fontsize=14,
                y=1.00,
            )
            plt.tight_layout()
            plt.show()

    def __repr__(self) -> str:
        """String representation."""
        if self._fitted:
            status = f"fitted with {self.n_bootstrap - self.n_failed_} successful replications"
        else:
            status = "not fitted"

        return (
            f"PanelBootstrap("
            f"method='{self.method}', "
            f"n_bootstrap={self.n_bootstrap}, "
            f"{status})"
        )
