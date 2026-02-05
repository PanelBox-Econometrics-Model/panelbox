"""
Driscoll-Kraay standard errors for panel data.

Driscoll-Kraay (1998) standard errors are robust to general forms of
spatial and temporal dependence when the number of time periods is large.
They are particularly useful for macro panel data with potential cross-
sectional correlation.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .utils import compute_bread, sandwich_covariance

KernelType = Literal["bartlett", "parzen", "quadratic_spectral"]


@dataclass
class DriscollKraayResult:
    """
    Result of Driscoll-Kraay covariance estimation.

    Attributes
    ----------
    cov_matrix : np.ndarray
        Driscoll-Kraay covariance matrix (k x k)
    std_errors : np.ndarray
        Driscoll-Kraay standard errors (k,)
    max_lags : int
        Maximum number of lags used
    kernel : str
        Kernel function used
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters
    n_periods : int
        Number of time periods
    bandwidth : Optional[float]
        Bandwidth parameter (for some kernels)
    """

    cov_matrix: np.ndarray
    std_errors: np.ndarray
    max_lags: int
    kernel: str
    n_obs: int
    n_params: int
    n_periods: int
    bandwidth: Optional[float] = None


class DriscollKraayStandardErrors:
    """
    Driscoll-Kraay (1998) standard errors for panel data.

    Robust to general forms of spatial and temporal dependence.
    Particularly useful for macro panels with cross-sectional correlation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags. If None, uses floor(4(T/100)^(2/9))
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function for weighting lags

    Attributes
    ----------
    X : np.ndarray
        Design matrix
    resid : np.ndarray
        Residuals
    time_ids : np.ndarray
        Time identifiers
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters
    n_periods : int
        Number of time periods

    Examples
    --------
    >>> # Panel data with T=20 periods
    >>> dk = DriscollKraayStandardErrors(X, resid, time_ids)
    >>> result = dk.compute()
    >>> print(result.std_errors)

    >>> # Custom lags
    >>> dk = DriscollKraayStandardErrors(X, resid, time_ids, max_lags=5)
    >>> result = dk.compute()

    References
    ----------
    Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix
        estimation with spatially dependent panel data. Review of Economics
        and Statistics, 80(4), 549-560.

    Hoechle, D. (2007). Robust standard errors for panel regressions with
        cross-sectional dependence. The Stata Journal, 7(3), 281-312.
    """

    def __init__(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        time_ids: np.ndarray,
        max_lags: Optional[int] = None,
        kernel: KernelType = "bartlett",
    ):
        self.X = X
        self.resid = resid
        self.time_ids = np.asarray(time_ids)
        self.kernel = kernel

        self.n_obs, self.n_params = X.shape

        # Validate dimensions
        if len(self.time_ids) != self.n_obs:
            raise ValueError(
                f"time_ids dimension mismatch: expected {self.n_obs}, " f"got {len(self.time_ids)}"
            )

        # Count time periods
        unique_periods = np.unique(self.time_ids)
        self.n_periods = len(unique_periods)

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_periods / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_periods:
            self.max_lags = self.n_periods - 1

        # Cache
        self._bread = None
        self._time_sorted = None

    @property
    def bread(self) -> np.ndarray:
        """Compute and cache bread matrix."""
        if self._bread is None:
            self._bread = compute_bread(self.X)
        return self._bread

    def _sort_by_time(self):
        """Sort data by time periods."""
        if self._time_sorted is None:
            # Get unique time periods in order
            unique_times = np.unique(self.time_ids)

            # Create mapping
            time_map = {t: i for i, t in enumerate(unique_times)}

            # Sort indices by time
            time_indices = np.array([time_map[t] for t in self.time_ids])
            sort_idx = np.argsort(time_indices)

            self._time_sorted = {
                "X": self.X[sort_idx],
                "resid": self.resid[sort_idx],
                "time_ids": self.time_ids[sort_idx],
                "sort_idx": sort_idx,
                "unique_times": unique_times,
            }

        return self._time_sorted

    def _kernel_weight(self, lag: int) -> float:
        """
        Compute kernel weight for given lag.

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        weight : float
            Kernel weight
        """
        if lag > self.max_lags:
            return 0.0

        if self.kernel == "bartlett":
            # Bartlett (triangular) kernel
            # w(l) = 1 - l/(max_lags + 1)
            return 1.0 - lag / (self.max_lags + 1)

        elif self.kernel == "parzen":
            # Parzen kernel
            z = lag / (self.max_lags + 1)
            if z <= 0.5:
                return 1 - 6 * z**2 + 6 * z**3
            else:
                return 2 * (1 - z) ** 3

        elif self.kernel == "quadratic_spectral":
            # Quadratic Spectral kernel
            if lag == 0:
                return 1.0
            z = 6 * np.pi * lag / (self.max_lags + 1) / 5
            return 3 / z**2 * (np.sin(z) / z - np.cos(z))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_gamma(self, lag: int) -> np.ndarray:
        """
        Compute autocovariance matrix for given lag.

        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        sorted_data = self._sort_by_time()
        unique_times = sorted_data["unique_times"]
        k = self.n_params

        gamma = np.zeros((k, k))

        # For each time period t
        for t_idx in range(lag, self.n_periods):
            t = unique_times[t_idx]
            t_lag = unique_times[t_idx - lag]

            # Get observations for time t
            mask_t = sorted_data["time_ids"] == t
            X_t = sorted_data["X"][mask_t]
            resid_t = sorted_data["resid"][mask_t]

            # Get observations for time t-lag
            mask_t_lag = sorted_data["time_ids"] == t_lag
            X_t_lag = sorted_data["X"][mask_t_lag]
            resid_t_lag = sorted_data["resid"][mask_t_lag]

            # Compute cross-product
            # For each pair of observations
            for i in range(len(X_t)):
                for j in range(len(X_t_lag)):
                    gamma += np.outer(X_t[i] * resid_t[i], X_t_lag[j] * resid_t_lag[j])

        return gamma

    def compute(self) -> DriscollKraayResult:
        """
        Compute Driscoll-Kraay covariance matrix.

        Returns
        -------
        result : DriscollKraayResult
            Driscoll-Kraay covariance and standard errors

        Notes
        -----
        The Driscoll-Kraay estimator is:

        V_DK = (X'X)^{-1} S_DK (X'X)^{-1}

        where:
        S_DK = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix:
        Γ_l = Σ_t X_t' ε̂_t ε̂_{t-l}' X_{t-l}

        The kernel weights w_l ensure positive semi-definiteness.
        """
        k = self.n_params

        # Start with lag-0 autocovariance
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            n_periods=self.n_periods,
        )

    def diagnostic_summary(self) -> str:
        """
        Generate diagnostic summary.

        Returns
        -------
        summary : str
            Diagnostic information
        """
        lines = []
        lines.append("Driscoll-Kraay Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of time periods: {self.n_periods}")
        lines.append(f"Avg obs per period: {self.n_obs / self.n_periods:.1f}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append("")

        # Recommendations
        if self.n_periods < 20:
            lines.append("⚠ WARNING: Few time periods (<20)")
            lines.append("  Driscoll-Kraay SEs may not perform well with T < 20")
            lines.append("  Consider alternative methods")
        if self.max_lags > self.n_periods / 4:
            lines.append("⚠ WARNING: Large max_lags relative to T")
            lines.append(f"  max_lags = {self.max_lags}, T = {self.n_periods}")

        return "\n".join(lines)


def driscoll_kraay(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: Optional[int] = None,
    kernel: KernelType = "bartlett",
) -> DriscollKraayResult:
    """
    Convenience function for Driscoll-Kraay standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time period identifiers (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function

    Returns
    -------
    result : DriscollKraayResult
        Driscoll-Kraay covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import driscoll_kraay
    >>> result = driscoll_kraay(X, resid, time_ids, max_lags=3)
    >>> print(result.std_errors)
    """
    dk = DriscollKraayStandardErrors(X, resid, time_ids, max_lags, kernel)
    return dk.compute()
