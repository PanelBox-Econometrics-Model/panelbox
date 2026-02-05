"""
Heteroskedasticity-robust standard errors (HC0, HC1, HC2, HC3).

This module implements White's heteroskedasticity-robust covariance
estimators and their finite-sample improvements.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .utils import (
    compute_bread,
    compute_leverage,
    compute_meat_hc,
    sandwich_covariance,
)

HC_TYPES = Literal["HC0", "HC1", "HC2", "HC3"]


@dataclass
class RobustCovarianceResult:
    """
    Result of robust covariance estimation.

    Attributes
    ----------
    cov_matrix : np.ndarray
        Robust covariance matrix (k x k)
    std_errors : np.ndarray
        Robust standard errors (k,)
    method : str
        Method used ('HC0', 'HC1', 'HC2', 'HC3')
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters
    leverage : np.ndarray, optional
        Leverage values (for HC2, HC3)
    """

    cov_matrix: np.ndarray
    std_errors: np.ndarray
    method: str
    n_obs: int
    n_params: int
    leverage: Optional[np.ndarray] = None


class RobustStandardErrors:
    """
    Heteroskedasticity-robust standard errors.

    Implements White (1980) and improved finite-sample variants.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)

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

    Methods
    -------
    hc0()
        White (1980) heteroskedasticity-robust SE
    hc1()
        Degrees of freedom corrected
    hc2()
        Leverage-adjusted
    hc3()
        MacKinnon-White (1985)

    Examples
    --------
    >>> from panelbox.standard_errors import RobustStandardErrors
    >>> robust = RobustStandardErrors(X, resid)
    >>> result_hc1 = robust.hc1()
    >>> print(result_hc1.std_errors)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.

    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """

    def __init__(self, X: np.ndarray, resid: np.ndarray):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = X.shape

        # Cache for efficiency
        self._leverage = None
        self._bread = None

    @property
    def leverage(self) -> np.ndarray:
        """Compute and cache leverage values."""
        if self._leverage is None:
            self._leverage = compute_leverage(self.X)
        return self._leverage

    @property
    def bread(self) -> np.ndarray:
        """Compute and cache bread matrix."""
        if self._bread is None:
            self._bread = compute_bread(self.X)
        return self._bread

    def hc0(self) -> RobustCovarianceResult:
        """
        HC0: White's heteroskedasticity-robust covariance.

        V_HC0 = (X'X)^{-1} X' Ω̂ X (X'X)^{-1}

        where Ω̂ = diag(ε̂²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC0 is the original White (1980) estimator. It can be biased
        downward in finite samples. Consider using HC1, HC2, or HC3
        for better finite-sample properties.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC0")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC0",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def hc1(self) -> RobustCovarianceResult:
        """
        HC1: Degrees of freedom corrected.

        V_HC1 = [n/(n-k)] × V_HC0

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC1 is the most commonly used robust SE in practice.
        It provides better finite-sample properties than HC0.

        This is the default in Stata's "robust" option.
        """
        meat = compute_meat_hc(self.X, self.resid, method="HC1")
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC1",
            n_obs=self.n_obs,
            n_params=self.n_params,
        )

    def hc2(self) -> RobustCovarianceResult:
        """
        HC2: Leverage-adjusted.

        V_HC2 = (X'X)^{-1} X' Ω̂_2 X (X'X)^{-1}

        where Ω̂_2 = diag(ε̂²/(1-h_i))

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC2 adjusts for leverage (hat values). Observations with
        high leverage receive more weight. Generally performs
        better than HC0 and HC1 in finite samples.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC2", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC2",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def hc3(self) -> RobustCovarianceResult:
        """
        HC3: MacKinnon-White leverage-adjusted.

        V_HC3 = (X'X)^{-1} X' Ω̂_3 X (X'X)^{-1}

        where Ω̂_3 = diag(ε̂²/(1-h_i)²)

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Notes
        -----
        HC3 provides the most aggressive leverage adjustment.
        Often recommended for small samples or when there are
        high leverage points.

        MacKinnon & White (1985) found HC3 to have good properties
        in simulation studies.
        """
        leverage = self.leverage
        meat = compute_meat_hc(self.X, self.resid, method="HC3", leverage=leverage)
        cov_matrix = sandwich_covariance(self.bread, meat)
        std_errors = np.sqrt(np.diag(cov_matrix))

        return RobustCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            method="HC3",
            n_obs=self.n_obs,
            n_params=self.n_params,
            leverage=leverage,
        )

    def compute(self, method: HC_TYPES = "HC1") -> RobustCovarianceResult:
        """
        Compute robust covariance with specified method.

        Parameters
        ----------
        method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
            Type of heteroskedasticity-robust covariance

        Returns
        -------
        result : RobustCovarianceResult
            Covariance matrix and standard errors

        Examples
        --------
        >>> robust = RobustStandardErrors(X, resid)
        >>> result = robust.compute('HC1')
        >>> print(result.std_errors)
        """
        method_upper = method.upper()

        if method_upper == "HC0":
            return self.hc0()
        elif method_upper == "HC1":
            return self.hc1()
        elif method_upper == "HC2":
            return self.hc2()
        elif method_upper == "HC3":
            return self.hc3()
        else:
            raise ValueError(
                f"Unknown HC method: {method}. " f"Must be one of: 'HC0', 'HC1', 'HC2', 'HC3'"
            )


def robust_covariance(
    X: np.ndarray, resid: np.ndarray, method: HC_TYPES = "HC1"
) -> RobustCovarianceResult:
    """
    Convenience function for computing robust covariance.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : {'HC0', 'HC1', 'HC2', 'HC3'}, default='HC1'
        Type of heteroskedasticity-robust covariance

    Returns
    -------
    result : RobustCovarianceResult
        Covariance matrix and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import robust_covariance
    >>> result = robust_covariance(X, resid, method='HC1')
    >>> print(result.std_errors)
    """
    robust = RobustStandardErrors(X, resid)
    return robust.compute(method)
