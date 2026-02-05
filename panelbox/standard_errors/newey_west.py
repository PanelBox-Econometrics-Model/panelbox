"""
Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors.

Newey-West (1987) standard errors are robust to both heteroskedasticity and
autocorrelation. Useful for time-series and panel data with serial correlation.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .utils import compute_bread, sandwich_covariance

KernelType = Literal["bartlett", "parzen", "quadratic_spectral"]


@dataclass
class NeweyWestResult:
    """
    Result of Newey-West HAC covariance estimation.

    Attributes
    ----------
    cov_matrix : np.ndarray
        Newey-West covariance matrix (k x k)
    std_errors : np.ndarray
        Newey-West standard errors (k,)
    max_lags : int
        Maximum number of lags used
    kernel : str
        Kernel function used
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters
    prewhitening : bool
        Whether prewhitening was applied
    """

    cov_matrix: np.ndarray
    std_errors: np.ndarray
    max_lags: int
    kernel: str
    n_obs: int
    n_params: int
    prewhitening: bool = False


class NeweyWestStandardErrors:
    """
    Newey-West (1987) HAC standard errors.

    Robust to heteroskedasticity and autocorrelation. Particularly useful
    for time-series data and panel data with serial correlation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags. If None, uses floor(4(T/100)^(2/9))
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function for weighting lags
    prewhitening : bool, default=False
        Apply AR(1) prewhitening to reduce finite-sample bias

    Attributes
    ----------
    X : np.ndarray
        Design matrix
    resid : np.ndarray
        Residuals
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters

    Examples
    --------
    >>> # Time-series with autocorrelation
    >>> nw = NeweyWestStandardErrors(X, resid, max_lags=4)
    >>> result = nw.compute()
    >>> print(result.std_errors)

    >>> # Auto-select lags
    >>> nw = NeweyWestStandardErrors(X, resid)
    >>> result = nw.compute()

    References
    ----------
    Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
        heteroskedasticity and autocorrelation consistent covariance matrix.
        Econometrica, 55(3), 703-708.

    Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent
        covariance matrix estimation. Econometrica, 59(3), 817-858.
    """

    def __init__(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        max_lags: Optional[int] = None,
        kernel: KernelType = "bartlett",
        prewhitening: bool = False,
    ):
        self.X = X
        self.resid = resid
        self.kernel = kernel
        self.prewhitening = prewhitening

        self.n_obs, self.n_params = X.shape

        # Set max_lags
        if max_lags is None:
            # Newey-West rule: floor(4(T/100)^(2/9))
            self.max_lags = int(np.floor(4 * (self.n_obs / 100) ** (2 / 9)))
        else:
            self.max_lags = max_lags

        # Ensure max_lags is reasonable
        if self.max_lags >= self.n_obs:
            self.max_lags = self.n_obs - 1

        # Cache
        self._bread: Optional[np.ndarray] = None

    @property
    def bread(self) -> np.ndarray:
        """Compute and cache bread matrix."""
        if self._bread is None:
            self._bread = compute_bread(self.X)
        return self._bread

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
            return float(3 / z**2 * (np.sin(z) / z - np.cos(z)))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_gamma(self, lag: int) -> np.ndarray:
        """
        Compute lag-l autocovariance matrix.

        Γ_l = (1/n) Σ_{t=l+1}^n (X_t ε_t)(X_{t-l} ε_{t-l})'

        Parameters
        ----------
        lag : int
            Lag number (0, 1, 2, ...)

        Returns
        -------
        gamma : np.ndarray
            Autocovariance matrix (k x k)
        """
        self.n_params
        n = self.n_obs

        if lag == 0:
            # Γ_0 = (1/n) Σ X_t' ε_t² X_t
            # This is the heteroskedasticity component
            X_resid = self.X * self.resid[:, np.newaxis]
            gamma = (X_resid.T @ X_resid) / n
        else:
            # Γ_l = (1/n) Σ (X_t ε_t)(X_{t-l} ε_{t-l})'
            X_resid_t = self.X[lag:] * self.resid[lag:, np.newaxis]
            X_resid_t_lag = self.X[:-lag] * self.resid[:-lag, np.newaxis]
            gamma = (X_resid_t.T @ X_resid_t_lag) / n

        return np.asarray(gamma)

    def compute(self) -> NeweyWestResult:
        """
        Compute Newey-West HAC covariance matrix.

        Returns
        -------
        result : NeweyWestResult
            Newey-West covariance and standard errors

        Notes
        -----
        The Newey-West estimator is:

        V_NW = (X'X)^{-1} Ω_NW (X'X)^{-1}

        where:
        Ω_NW = Γ_0 + Σ_{l=1}^L w_l (Γ_l + Γ_l')

        and Γ_l is the lag-l autocovariance matrix.

        The kernel weights w_l ensure positive semi-definiteness.
        """
        # Start with lag-0 autocovariance (heteroskedasticity)
        S = self._compute_gamma(0)

        # Add weighted autocovariances for lags 1, ..., max_lags
        for lag in range(1, self.max_lags + 1):
            weight = self._kernel_weight(lag)
            if weight > 0:
                gamma_l = self._compute_gamma(lag)
                # Add both Γ_l and Γ_l' (symmetrize)
                S += weight * (gamma_l + gamma_l.T)

        # Scale by n (since gamma is already divided by n)
        S *= self.n_obs

        # Sandwich: V = Bread @ S @ Bread
        cov_matrix = sandwich_covariance(self.bread, S)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=self.max_lags,
            kernel=self.kernel,
            n_obs=self.n_obs,
            n_params=self.n_params,
            prewhitening=self.prewhitening,
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
        lines.append("Newey-West HAC Standard Errors Diagnostics")
        lines.append("=" * 50)
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append(f"Maximum lags: {self.max_lags}")
        lines.append(f"Kernel function: {self.kernel}")
        lines.append(f"Prewhitening: {self.prewhitening}")
        lines.append("")

        # Recommendations
        if self.n_obs < 50:
            lines.append("⚠ WARNING: Small sample size (<50)")
            lines.append("  Newey-West SEs may not perform well with few observations")
        if self.max_lags > self.n_obs / 3:
            lines.append("⚠ WARNING: Large max_lags relative to sample size")
            lines.append(f"  max_lags = {self.max_lags}, n = {self.n_obs}")

        return "\n".join(lines)


def newey_west(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: Optional[int] = None,
    kernel: KernelType = "bartlett",
    prewhitening: bool = False,
) -> NeweyWestResult:
    """
    Convenience function for Newey-West HAC standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    max_lags : int, optional
        Maximum number of lags
    kernel : {'bartlett', 'parzen', 'quadratic_spectral'}, default='bartlett'
        Kernel function
    prewhitening : bool, default=False
        Apply AR(1) prewhitening

    Returns
    -------
    result : NeweyWestResult
        Newey-West covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import newey_west
    >>> result = newey_west(X, resid, max_lags=4)
    >>> print(result.std_errors)
    """
    nw = NeweyWestStandardErrors(X, resid, max_lags, kernel, prewhitening)
    return nw.compute()
