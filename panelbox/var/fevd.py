"""
Forecast Error Variance Decomposition (FEVD) for Panel VAR models.

This module implements:
- FEVD based on Cholesky decomposition
- Generalized FEVD (Pesaran-Shin)
- Bootstrap confidence intervals for FEVD

References
----------
.. [1] Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis.
       Springer-Verlag. Chapter 2.
.. [2] Pesaran, H. H., & Shin, Y. (1998). Generalized impulse response analysis
       in linear multivariate models. Economics letters, 58(1), 17-29.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


class FEVDResult:
    """
    Container for Forecast Error Variance Decomposition results.

    This class stores FEVD estimates and provides methods for accessing,
    transforming, and analyzing the results.

    Parameters
    ----------
    decomposition : np.ndarray
        FEVD array of shape (periods+1, K, K) where
        decomposition[h, i, j] = share of variance of variable i at horizon h
        explained by shocks to variable j
    var_names : List[str]
        Names of variables
    periods : int
        Number of periods (horizons)
    method : str
        Method used: 'cholesky' or 'generalized'
    ordering : List[str], optional
        Variable ordering for Cholesky decomposition

    Attributes
    ----------
    decomposition : np.ndarray
        FEVD array (periods+1, K, K)
    var_names : List[str]
        Variable names
    periods : int
        Number of periods
    K : int
        Number of variables
    method : str
        Method used
    ordering : List[str] or None
        Variable ordering
    ci_lower : np.ndarray or None
        Lower confidence interval (if computed)
    ci_upper : np.ndarray or None
        Upper confidence interval (if computed)
    ci_level : float or None
        Confidence level (if computed)
    """

    def __init__(
        self,
        decomposition: np.ndarray,
        var_names: List[str],
        periods: int,
        method: str,
        ordering: Optional[List[str]] = None,
    ):
        self.decomposition = decomposition
        self.var_names = var_names
        self.periods = periods
        self.K = len(var_names)
        self.method = method
        self.ordering = ordering

        # Confidence intervals (to be set if bootstrap is computed)
        self.ci_lower = None
        self.ci_upper = None
        self.ci_level = None

        # Validate that FEVD sums to 1
        self._validate_fevd()

    def _validate_fevd(self):
        """Validate that FEVD sums to 100% for each variable at each horizon."""
        for h in range(self.periods + 1):
            for i in range(self.K):
                row_sum = self.decomposition[h, i, :].sum()
                if not np.isclose(row_sum, 1.0, atol=1e-6):
                    import warnings

                    warnings.warn(
                        f"FEVD for variable {i} at horizon {h} sums to {row_sum:.6f}, not 1.0. "
                        "This may indicate numerical issues.",
                        UserWarning,
                    )

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Access FEVD for a specific variable.

        Parameters
        ----------
        key : str
            Variable name

        Returns
        -------
        np.ndarray
            FEVD array of shape (periods+1, K) showing decomposition of
            variance of `key` across all shocks

        Examples
        --------
        >>> fevd = result.fevd(periods=10)
        >>> fevd['gdp']  # FEVD of GDP across all shocks
        array([[1.0, 0.0, 0.0],
               [0.95, 0.03, 0.02],
               ...])
        """
        try:
            var_idx = self.var_names.index(key)
        except ValueError:
            raise KeyError(f"Variable '{key}' not found")

        return self.decomposition[:, var_idx, :]

    def to_dataframe(
        self, variable: Optional[str] = None, horizons: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Convert FEVD to DataFrame format.

        Parameters
        ----------
        variable : str, optional
            If specified, return FEVD only for this variable
        horizons : list of int, optional
            If specified, return FEVD only for these horizons

        Returns
        -------
        pd.DataFrame
            DataFrame with FEVD values

        Examples
        --------
        >>> # Get FEVD for GDP at all horizons
        >>> df = fevd.to_dataframe(variable='gdp')
        >>> # Get FEVD for all variables at selected horizons
        >>> df = fevd.to_dataframe(horizons=[0, 1, 5, 10])
        """
        if horizons is None:
            horizons = list(range(self.periods + 1))

        if variable is not None:
            # Return FEVD for single variable
            var_idx = self.var_names.index(variable)
            data = {"horizon": horizons}
            for j, shock_var in enumerate(self.var_names):
                data[shock_var] = [self.decomposition[h, var_idx, j] for h in horizons]
            return pd.DataFrame(data)
        else:
            # Return FEVD for all variables (long format)
            rows = []
            for h in horizons:
                for i, resp_var in enumerate(self.var_names):
                    row = {"horizon": h, "variable": resp_var}
                    for j, shock_var in enumerate(self.var_names):
                        row[f"shock_{shock_var}"] = self.decomposition[h, i, j]
                    rows.append(row)
            return pd.DataFrame(rows)

    def summary(self, horizons: Optional[List[int]] = None) -> str:
        """
        Generate summary table of FEVD at selected horizons.

        Parameters
        ----------
        horizons : list of int, optional
            Horizons to display (default: [1, 5, 10, periods])

        Returns
        -------
        str
            Formatted summary table
        """
        if horizons is None:
            horizons = [1, 5, 10, self.periods]
            horizons = [h for h in horizons if h <= self.periods]

        lines = []
        lines.append("=" * 80)
        lines.append(f"Forecast Error Variance Decomposition ({self.method.upper()})")
        lines.append("=" * 80)
        lines.append(f"Method: {self.method}")
        if self.ordering and self.method == "cholesky":
            lines.append(f"Ordering: {', '.join(self.ordering)}")
        lines.append("")

        for i, var in enumerate(self.var_names):
            lines.append(f"Variable: {var}")
            lines.append("-" * 80)

            # Build table
            header = ["Horizon"] + [f"Shock {v}" for v in self.var_names]
            lines.append("  ".join([f"{col:>12}" for col in header]))
            lines.append("-" * 80)

            for h in horizons:
                row = [f"h={h}"]
                for j in range(self.K):
                    value = self.decomposition[h, i, j]
                    row.append(f"{value * 100:>11.2f}%")
                lines.append("  ".join([f"{cell:>12}" for cell in row]))

            lines.append("")

        lines.append("=" * 80)
        lines.append("Note: Values are percentages of total variance")
        lines.append("=" * 80)

        return "\n".join(lines)

    def plot(
        self,
        kind: str = "area",
        variables: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
        backend: str = "matplotlib",
        figsize: Optional[tuple] = None,
        theme: str = "academic",
        show: bool = True,
    ):
        """
        Plot Forecast Error Variance Decomposition.

        Parameters
        ----------
        kind : str, default='area'
            Type of plot: 'area' (stacked area) or 'bar' (stacked bar)
        variables : list of str, optional
            If specified, plot only these variables
        horizons : list of int, optional
            For 'bar' plot: horizons to display
        backend : str, default='matplotlib'
            Plotting backend: 'matplotlib' or 'plotly'
        figsize : tuple, optional
            Figure size (width, height)
        theme : str, default='academic'
            Visual theme: 'academic', 'professional', or 'presentation'
        show : bool, default=True
            Whether to display the plot immediately

        Returns
        -------
        fig : Figure or None
            Figure object if show=False, otherwise None

        Examples
        --------
        >>> fevd = result.fevd(periods=20)
        >>> fevd.plot()
        >>> fevd.plot(kind='bar', horizons=[1, 5, 10, 20])
        >>> fevd.plot(backend='plotly', variables=['gdp', 'inflation'])
        """
        from panelbox.visualization.var_plots import plot_fevd

        return plot_fevd(
            fevd_result=self,
            kind=kind,
            variables=variables,
            horizons=horizons,
            backend=backend,
            figsize=figsize,
            theme=theme,
            show=show,
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"FEVDResult(K={self.K}, periods={self.periods}, method='{self.method}')"


def compute_fevd_cholesky(
    Phi: np.ndarray, P: np.ndarray, Sigma: np.ndarray, periods: int
) -> np.ndarray:
    """
    Compute FEVD based on Cholesky decomposition.

    Parameters
    ----------
    Phi : np.ndarray
        Orthogonalized IRF array (periods+1, K, K) from Cholesky
    P : np.ndarray
        Cholesky factor (K, K)
    Sigma : np.ndarray
        Residual covariance matrix (K, K)
    periods : int
        Number of periods

    Returns
    -------
    FEVD : np.ndarray
        FEVD array (periods+1, K, K)
        FEVD[h, i, j] = share of variance of variable i at horizon h
        explained by shocks to variable j

    Notes
    -----
    Formula:
        ω_ij(h) = [Σ_{s=0}^h (e_i' · Φ_s · P · e_j)²] / [Σ_{s=0}^h (e_i' · Φ_s · Σ · Φ_s' · e_i)]

    For Cholesky IRFs, the numerator simplifies to:
        Σ_{s=0}^h Φ_s[i, j]²

    The denominator is the total forecast error variance of variable i at horizon h.
    """
    K = P.shape[0]
    FEVD = np.zeros((periods + 1, K, K))

    for h in range(periods + 1):
        for i in range(K):
            # Compute total variance of variable i at horizon h
            # For orthogonalized IRF: total_var = Σ_{s=0}^h Σ_j Φ_s[i,j]²
            total_var = 0.0
            for s in range(h + 1):
                for j in range(K):
                    total_var += Phi[s][i, j] ** 2

            # Compute contribution of each shock j
            for j in range(K):
                contrib = 0.0
                for s in range(h + 1):
                    # Contribution of shock j: Φ_s[i, j]²
                    contrib += Phi[s][i, j] ** 2

                # Share of variance
                if total_var > 1e-12:
                    FEVD[h, i, j] = contrib / total_var
                else:
                    FEVD[h, i, j] = 0.0

    return FEVD


def compute_fevd_generalized(Phi: np.ndarray, Sigma: np.ndarray, periods: int) -> np.ndarray:
    """
    Compute Generalized FEVD (Pesaran-Shin).

    Parameters
    ----------
    Phi : np.ndarray
        Non-orthogonalized MA coefficients (periods+1, K, K)
    Sigma : np.ndarray
        Residual covariance matrix (K, K)
    periods : int
        Number of periods

    Returns
    -------
    FEVD : np.ndarray
        Generalized FEVD array (periods+1, K, K), normalized to sum to 1

    Notes
    -----
    Formula (before normalization):
        ω_ij(h) = σ_jj^{-1} · [Σ_{s=0}^h (e_i' · Φ_s · Σ · e_j)²] / [Σ_{s=0}^h (e_i' · Φ_s · Σ · Φ_s' · e_i)]

    After computing, we normalize so that Σ_j ω_ij(h) = 1 for each (i, h).
    """
    K = Sigma.shape[0]
    sigma_diag = np.diag(Sigma)
    FEVD_raw = np.zeros((periods + 1, K, K))

    for h in range(periods + 1):
        for i in range(K):
            # Total variance
            total_var = 0.0
            for s in range(h + 1):
                total_var += Phi[s][i, :] @ Sigma @ Phi[s][i, :]

            # Contribution of each shock j
            for j in range(K):
                contrib = 0.0
                e_j = np.zeros(K)
                e_j[j] = 1.0

                for s in range(h + 1):
                    impulse = Phi[s][i, :] @ Sigma @ e_j
                    contrib += impulse**2

                if total_var > 1e-12 and sigma_diag[j] > 1e-12:
                    FEVD_raw[h, i, j] = contrib / sigma_diag[j] / total_var
                else:
                    FEVD_raw[h, i, j] = 0.0

        # Normalize to sum to 1 (GFEVD doesn't automatically sum to 1)
        for i in range(K):
            row_sum = FEVD_raw[h, i, :].sum()
            if row_sum > 1e-12:
                FEVD_raw[h, i, :] /= row_sum
            else:
                # Fallback: equal weights
                FEVD_raw[h, i, :] = 1.0 / K

    return FEVD_raw


def _bootstrap_fevd_iteration(
    A_matrices: List[np.ndarray],
    Sigma: np.ndarray,
    residuals: np.ndarray,
    periods: int,
    method: str,
    seed: int,
) -> np.ndarray:
    """
    Single bootstrap iteration for FEVD confidence intervals.

    Parameters
    ----------
    A_matrices : list of np.ndarray
        Coefficient matrices
    Sigma : np.ndarray
        Residual covariance matrix
    residuals : np.ndarray
        Original residuals (n_obs, K)
    periods : int
        Number of periods
    method : str
        'cholesky' or 'generalized'
    seed : int
        Random seed for this iteration

    Returns
    -------
    fevd_boot : np.ndarray
        Bootstrap FEVD (periods+1, K, K)
    """
    # Import here to avoid circular dependency
    from panelbox.var.irf import compute_irf_cholesky, compute_phi_non_orthogonalized

    np.random.seed(seed)

    K = Sigma.shape[0]
    p = len(A_matrices)
    n_obs = residuals.shape[0]

    # 1. Resample residuals (with replacement)
    indices = np.random.choice(n_obs, size=n_obs, replace=True)
    resampled_residuals = residuals[indices]

    # 2. Reconstruct data
    y_bootstrap = np.zeros((n_obs + p, K))

    # Use mean as initial values
    for t in range(p):
        y_bootstrap[t] = 0.0

    # Generate bootstrap sample
    for t in range(p, n_obs + p):
        # Lagged values
        y_lag = np.zeros(K)
        for lag in range(1, p + 1):
            y_lag += A_matrices[lag - 1] @ y_bootstrap[t - lag]

        # Add resampled residual
        y_bootstrap[t] = y_lag + resampled_residuals[t - p]

    # 3. Re-estimate VAR on bootstrap sample
    y_boot = y_bootstrap[p:]
    X_boot = np.zeros((n_obs, K * p))

    for t in range(n_obs):
        for lag in range(1, p + 1):
            X_boot[t, (lag - 1) * K : lag * K] = y_bootstrap[p + t - lag]

    # Estimate coefficients (OLS)
    A_boot_flat = np.linalg.lstsq(X_boot, y_boot, rcond=None)[0]

    # Reshape into A matrices
    A_boot = []
    for lag in range(p):
        A_l = A_boot_flat[lag * K : (lag + 1) * K, :].T
        A_boot.append(A_l)

    # Compute residuals and Sigma_boot
    resid_boot = y_boot - X_boot @ A_boot_flat
    Sigma_boot = (resid_boot.T @ resid_boot) / n_obs

    # 4. Compute FEVD from bootstrap estimates
    if method == "cholesky":
        # Compute Cholesky IRF
        Phi_boot = compute_irf_cholesky(A_boot, Sigma_boot, periods)
        try:
            P_boot = np.linalg.cholesky(Sigma_boot)
        except np.linalg.LinAlgError:
            # Add regularization if needed
            Sigma_boot_reg = Sigma_boot + 1e-8 * np.eye(K)
            P_boot = np.linalg.cholesky(Sigma_boot_reg)

        fevd_boot = compute_fevd_cholesky(Phi_boot, P_boot, Sigma_boot, periods)

    elif method == "generalized":
        # Compute Generalized FEVD
        Phi_boot = compute_phi_non_orthogonalized(A_boot, periods)
        fevd_boot = compute_fevd_generalized(Phi_boot, Sigma_boot, periods)

    else:
        raise ValueError(f"Unknown method: {method}")

    return fevd_boot


def bootstrap_fevd(
    A_matrices: List[np.ndarray],
    Sigma: np.ndarray,
    residuals: np.ndarray,
    periods: int,
    method: str = "cholesky",
    n_bootstrap: int = 500,
    ci_level: float = 0.95,
    n_jobs: int = -1,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bootstrap confidence intervals for FEVD.

    Parameters
    ----------
    A_matrices : list of np.ndarray
        Coefficient matrices [A_1, ..., A_p]
    Sigma : np.ndarray
        Residual covariance matrix (K, K)
    residuals : np.ndarray
        Residuals from VAR estimation (n_obs, K)
    periods : int
        Number of periods
    method : str, default='cholesky'
        'cholesky' or 'generalized'
    n_bootstrap : int, default=500
        Number of bootstrap iterations
    ci_level : float, default=0.95
        Confidence level
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores)
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=True
        Show progress bar

    Returns
    -------
    ci_lower : np.ndarray
        Lower confidence interval (periods+1, K, K)
    ci_upper : np.ndarray
        Upper confidence interval (periods+1, K, K)
    bootstrap_dist : np.ndarray
        Bootstrap distribution (n_bootstrap, periods+1, K, K)

    Notes
    -----
    This function reuses the IRF bootstrap approach to compute FEVD confidence
    intervals. For each bootstrap sample, we re-estimate the VAR, compute IRFs,
    and then compute FEVD from those IRFs.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate seeds for each bootstrap iteration
    seeds = np.random.randint(0, 2**31 - 1, size=n_bootstrap)

    # Parallel bootstrap
    if verbose:
        print(f"Running {n_bootstrap} bootstrap iterations for FEVD...")

    bootstrap_fevds = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_fevd_iteration)(A_matrices, Sigma, residuals, periods, method, seeds[i])
        for i in tqdm(range(n_bootstrap), desc="Bootstrap FEVD", disable=not verbose)
    )

    # Convert to array
    bootstrap_dist = np.array(bootstrap_fevds)  # (n_bootstrap, periods+1, K, K)

    # Compute confidence intervals (percentile method)
    alpha = 1 - ci_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    ci_lower = np.percentile(bootstrap_dist, lower_percentile, axis=0)
    ci_upper = np.percentile(bootstrap_dist, upper_percentile, axis=0)

    return ci_lower, ci_upper, bootstrap_dist
