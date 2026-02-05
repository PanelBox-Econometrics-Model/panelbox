"""
Statistical functions for panel econometrics.

This module provides statistical functions for hypothesis testing
and inference in panel models.
"""

from typing import Optional, Tuple

import numpy as np
from scipy import stats


def compute_tstat(coef: float, se: float) -> float:
    """
    Compute t-statistic.

    Parameters
    ----------
    coef : float
        Coefficient estimate
    se : float
        Standard error

    Returns
    -------
    float
        t-statistic
    """
    return coef / se if se > 0 else np.nan


def compute_pvalue(tstat: float, df: int, two_sided: bool = True) -> float:
    """
    Compute p-value for t-statistic.

    Parameters
    ----------
    tstat : float
        t-statistic
    df : int
        Degrees of freedom
    two_sided : bool, default=True
        Whether to compute two-sided p-value

    Returns
    -------
    float
        p-value
    """
    if two_sided:
        return float(2 * (1 - stats.t.cdf(np.abs(tstat), df)))
    else:
        return float(1 - stats.t.cdf(tstat, df))


def compute_fstat(
    rss_restricted: float, rss_unrestricted: float, df_diff: int, df_resid: int
) -> Tuple[float, float]:
    """
    Compute F-statistic for nested model comparison.

    Parameters
    ----------
    rss_restricted : float
        Residual sum of squares from restricted model
    rss_unrestricted : float
        Residual sum of squares from unrestricted model
    df_diff : int
        Difference in degrees of freedom (number of restrictions)
    df_resid : int
        Degrees of freedom for residuals (unrestricted model)

    Returns
    -------
    fstat : float
        F-statistic
    pvalue : float
        p-value
    """
    numerator = (rss_restricted - rss_unrestricted) / df_diff
    denominator = rss_unrestricted / df_resid

    fstat = numerator / denominator if denominator > 0 else np.nan
    pvalue = 1 - stats.f.cdf(fstat, df_diff, df_resid)

    return fstat, pvalue


def wald_test(
    restrictions: np.ndarray,
    params: np.ndarray,
    vcov: np.ndarray,
    q: Optional[np.ndarray] = None,
) -> Tuple[float, float, int]:
    """
    Compute Wald test for linear restrictions.

    Tests H0: R*beta = q

    Parameters
    ----------
    restrictions : np.ndarray
        Restriction matrix R (m x k)
    params : np.ndarray
        Parameter estimates (k x 1)
    vcov : np.ndarray
        Covariance matrix (k x k)
    q : np.ndarray, optional
        RHS of restrictions (m x 1). Default is zeros.

    Returns
    -------
    statistic : float
        Wald chi-squared statistic
    pvalue : float
        p-value
    df : int
        Degrees of freedom
    """
    if params.ndim == 1:
        params = params.reshape(-1, 1)

    if q is None:
        q = np.zeros((restrictions.shape[0], 1))
    elif q.ndim == 1:
        q = q.reshape(-1, 1)

    # Compute R*beta - q
    restriction_value = restrictions @ params - q

    # Compute (R * Vcov * R')^{-1}
    middle = restrictions @ vcov @ restrictions.T
    middle_inv = np.linalg.inv(middle)

    # Wald statistic: (R*beta - q)' (R * Vcov * R')^{-1} (R*beta - q)
    statistic = float(restriction_value.T @ middle_inv @ restriction_value)

    # Degrees of freedom
    df = restrictions.shape[0]

    # p-value from chi-squared distribution
    pvalue = 1 - stats.chi2.cdf(statistic, df)

    return statistic, pvalue, df


def compute_chi2_pvalue(statistic: float, df: int) -> float:
    """
    Compute p-value for chi-squared statistic.

    Parameters
    ----------
    statistic : float
        Chi-squared statistic
    df : int
        Degrees of freedom

    Returns
    -------
    float
        p-value
    """
    return float(1 - stats.chi2.cdf(statistic, df))
