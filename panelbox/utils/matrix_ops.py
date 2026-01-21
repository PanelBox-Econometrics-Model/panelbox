"""
Matrix operations for panel econometrics.

This module provides optimized matrix operations commonly used in
panel data estimation.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def add_intercept(X: np.ndarray) -> np.ndarray:
    """
    Add intercept column to design matrix.

    Parameters
    ----------
    X : np.ndarray
        Design matrix without intercept

    Returns
    -------
    np.ndarray
        Design matrix with intercept as first column
    """
    n = X.shape[0]
    return np.column_stack([np.ones(n), X])


def demean_matrix(
    X: np.ndarray,
    groups: np.ndarray
) -> np.ndarray:
    """
    Demean matrix by groups (within transformation).

    Parameters
    ----------
    X : np.ndarray
        Matrix to demean (n x k)
    groups : np.ndarray
        Group identifiers (length n)

    Returns
    -------
    np.ndarray
        Demeaned matrix
    """
    X_demeaned = X.copy()
    unique_groups = np.unique(groups)

    for group in unique_groups:
        mask = groups == group
        group_mean = X[mask].mean(axis=0)
        X_demeaned[mask] -= group_mean

    return X_demeaned


def compute_ols(
    y: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute OLS estimates.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable (n x 1 or length n)
    X : np.ndarray
        Design matrix (n x k)
    weights : np.ndarray, optional
        Observation weights

    Returns
    -------
    beta : np.ndarray
        Coefficient estimates (k x 1)
    resid : np.ndarray
        Residuals (n x 1)
    fitted : np.ndarray
        Fitted values (n x 1)
    """
    # Ensure y is column vector
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if weights is not None:
        # Weighted least squares
        W = np.diag(np.sqrt(weights))
        y_w = W @ y
        X_w = W @ X
        beta = np.linalg.lstsq(X_w, y_w, rcond=None)[0]
    else:
        # Ordinary least squares
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # Compute fitted values and residuals
    fitted = X @ beta
    resid = y - fitted

    return beta, resid.ravel(), fitted.ravel()


def compute_vcov_nonrobust(
    X: np.ndarray,
    resid: np.ndarray,
    df_resid: int
) -> np.ndarray:
    """
    Compute non-robust covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (length n)
    df_resid : int
        Degrees of freedom for residuals

    Returns
    -------
    np.ndarray
        Covariance matrix (k x k)
    """
    # Estimate of error variance
    s2 = np.sum(resid ** 2) / df_resid

    # Covariance matrix: s^2 (X'X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)
    vcov = s2 * XtX_inv

    return vcov


def compute_rsquared(
    y: np.ndarray,
    fitted: np.ndarray,
    resid: np.ndarray,
    has_intercept: bool = True
) -> Tuple[float, float]:
    """
    Compute R-squared and adjusted R-squared.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable
    fitted : np.ndarray
        Fitted values
    resid : np.ndarray
        Residuals
    has_intercept : bool, default=True
        Whether model includes intercept

    Returns
    -------
    rsquared : float
        R-squared
    rsquared_adj : float
        Adjusted R-squared
    """
    # Total sum of squares
    if has_intercept:
        tss = np.sum((y - y.mean()) ** 2)
    else:
        tss = np.sum(y ** 2)

    # Residual sum of squares
    rss = np.sum(resid ** 2)

    # R-squared
    rsquared = 1 - (rss / tss) if tss > 0 else 0.0

    return rsquared


def compute_panel_rsquared(
    y: np.ndarray,
    fitted: np.ndarray,
    resid: np.ndarray,
    groups: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute panel-specific R-squared measures.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable
    fitted : np.ndarray
        Fitted values
    resid : np.ndarray
        Residuals
    groups : np.ndarray
        Group identifiers

    Returns
    -------
    rsquared_within : float
        Within R-squared
    rsquared_between : float
        Between R-squared
    rsquared_overall : float
        Overall R-squared
    """
    # Overall R-squared
    y_mean = y.mean()
    tss_overall = np.sum((y - y_mean) ** 2)
    rss = np.sum(resid ** 2)
    rsquared_overall = 1 - (rss / tss_overall) if tss_overall > 0 else 0.0

    # Within R-squared (variation within groups)
    y_demeaned = demean_matrix(y.reshape(-1, 1), groups).ravel()
    fitted_demeaned = demean_matrix(fitted.reshape(-1, 1), groups).ravel()
    tss_within = np.sum(y_demeaned ** 2)
    ess_within = np.sum(fitted_demeaned ** 2)
    rsquared_within = ess_within / tss_within if tss_within > 0 else 0.0

    # Between R-squared (variation between group means)
    unique_groups = np.unique(groups)
    y_means = np.array([y[groups == g].mean() for g in unique_groups])
    fitted_means = np.array([fitted[groups == g].mean() for g in unique_groups])
    y_grand_mean = y.mean()
    tss_between = np.sum((y_means - y_grand_mean) ** 2)
    ess_between = np.sum((fitted_means - y_grand_mean) ** 2)
    rsquared_between = ess_between / tss_between if tss_between > 0 else 0.0

    return rsquared_within, rsquared_between, rsquared_overall
