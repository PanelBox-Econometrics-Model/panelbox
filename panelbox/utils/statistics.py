"""
Statistical utility functions for panel data models.

This module provides statistical utilities including robust covariance estimation,
hypothesis testing, and other statistical computations.
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy import linalg, stats


def compute_sandwich_covariance(
    hessian: np.ndarray, gradient_contributions: np.ndarray, entity_id: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute sandwich (robust) covariance matrix.

    The sandwich estimator is: (H^{-1}) * B * (H^{-1})
    where H is the Hessian and B is the outer product of gradients.

    Parameters
    ----------
    hessian : ndarray
        Hessian matrix (K x K)
    gradient_contributions : ndarray
        Individual gradient contributions (N x K)
    entity_id : ndarray, optional
        Entity identifiers for clustering

    Returns
    -------
    ndarray
        Robust covariance matrix (K x K)
    """
    # Invert Hessian
    try:
        hessian_inv = linalg.inv(hessian)
    except linalg.LinAlgError:
        # Use pseudo-inverse if singular
        hessian_inv = linalg.pinv(hessian)

    if entity_id is not None:
        # Cluster-robust covariance
        entities = np.unique(entity_id)
        n_entities = len(entities)

        # Sum gradients within entities
        clustered_grads = np.zeros((n_entities, gradient_contributions.shape[1]))
        for i, entity in enumerate(entities):
            mask = entity_id == entity
            clustered_grads[i] = gradient_contributions[mask].sum(axis=0)

        # Outer product of clustered gradients
        B = clustered_grads.T @ clustered_grads

        # Finite-sample correction
        n = len(entity_id)
        g = n_entities
        correction = g / (g - 1) * n / (n - 1)
        B *= correction
    else:
        # Regular sandwich covariance
        B = gradient_contributions.T @ gradient_contributions

    # Sandwich formula
    vcov = hessian_inv @ B @ hessian_inv

    return vcov


def compute_cluster_robust_covariance(
    residuals: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    vcov_base: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Parameters
    ----------
    residuals : ndarray
        Residuals from the model
    X : ndarray
        Design matrix
    entity_id : ndarray
        Entity identifiers for clustering
    vcov_base : ndarray, optional
        Base covariance matrix (if None, uses (X'X)^{-1})

    Returns
    -------
    ndarray
        Cluster-robust covariance matrix
    """
    n, k = X.shape
    entities = np.unique(entity_id)
    n_entities = len(entities)

    # Base covariance if not provided
    if vcov_base is None:
        XtX = X.T @ X
        try:
            vcov_base = linalg.inv(XtX)
        except linalg.LinAlgError:
            vcov_base = linalg.pinv(XtX)

    # Compute clustered outer product
    B = np.zeros((k, k))
    for entity in entities:
        mask = entity_id == entity
        X_e = X[mask]
        resid_e = residuals[mask]
        score_e = X_e.T @ resid_e
        B += np.outer(score_e, score_e)

    # Finite-sample correction
    correction = n_entities / (n_entities - 1) * n / (n - k)

    # Cluster-robust covariance
    vcov = correction * vcov_base @ B @ vcov_base

    return vcov


def likelihood_ratio_test(llf_unrestricted: float, llf_restricted: float, df: int) -> dict:
    """
    Perform likelihood ratio test.

    Parameters
    ----------
    llf_unrestricted : float
        Log-likelihood of unrestricted model
    llf_restricted : float
        Log-likelihood of restricted model
    df : int
        Degrees of freedom (number of restrictions)

    Returns
    -------
    dict
        Test results with statistic, p-value, and conclusion
    """
    lr_stat = 2 * (llf_unrestricted - llf_restricted)
    p_value = 1 - stats.chi2.cdf(lr_stat, df)

    conclusion = "Reject H0" if p_value < 0.05 else "Fail to reject H0"

    return {
        "statistic": lr_stat,
        "pvalue": p_value,
        "df": df,
        "conclusion": conclusion,
        "llf_unrestricted": llf_unrestricted,
        "llf_restricted": llf_restricted,
    }


def wald_test(
    params: np.ndarray,
    vcov: np.ndarray,
    restrictions: np.ndarray,
    values: Optional[np.ndarray] = None,
) -> dict:
    """
    Perform Wald test for linear restrictions.

    Tests H0: R * params = q

    Parameters
    ----------
    params : ndarray
        Parameter estimates
    vcov : ndarray
        Covariance matrix of parameters
    restrictions : ndarray
        Restriction matrix R (q x k)
    values : ndarray, optional
        Values q (default is zeros)

    Returns
    -------
    dict
        Test results
    """
    R = np.atleast_2d(restrictions)
    q = np.zeros(R.shape[0]) if values is None else values

    # Compute Wald statistic
    Rb_q = R @ params - q
    RVR = R @ vcov @ R.T

    try:
        RVR_inv = linalg.inv(RVR)
    except linalg.LinAlgError:
        RVR_inv = linalg.pinv(RVR)

    W = Rb_q.T @ RVR_inv @ Rb_q

    # Degrees of freedom
    df = R.shape[0]

    # P-value
    p_value = 1 - stats.chi2.cdf(W, df)

    return {
        "statistic": W,
        "pvalue": p_value,
        "df": df,
        "conclusion": "Reject H0" if p_value < 0.05 else "Fail to reject H0",
    }


def hausman_test(
    params_fe: np.ndarray, params_re: np.ndarray, vcov_fe: np.ndarray, vcov_re: np.ndarray
) -> dict:
    """
    Perform Hausman test for fixed vs random effects.

    Parameters
    ----------
    params_fe : ndarray
        Fixed effects parameter estimates
    params_re : ndarray
        Random effects parameter estimates
    vcov_fe : ndarray
        Fixed effects covariance matrix
    vcov_re : ndarray
        Random effects covariance matrix

    Returns
    -------
    dict
        Test results
    """
    # Difference in parameters
    b_diff = params_fe - params_re

    # Difference in covariance matrices
    V_diff = vcov_fe - vcov_re

    try:
        V_diff_inv = linalg.inv(V_diff)
    except linalg.LinAlgError:
        # Use pseudo-inverse if singular
        V_diff_inv = linalg.pinv(V_diff)

    # Hausman statistic
    H = b_diff.T @ V_diff_inv @ b_diff

    # Degrees of freedom
    df = len(params_fe)

    # P-value
    p_value = 1 - stats.chi2.cdf(H, df)

    conclusion = "Use Fixed Effects" if p_value < 0.05 else "Random Effects is consistent"

    return {"statistic": H, "pvalue": p_value, "df": df, "conclusion": conclusion}


def compute_standard_errors(vcov: np.ndarray) -> np.ndarray:
    """
    Extract standard errors from covariance matrix.

    Parameters
    ----------
    vcov : ndarray
        Covariance matrix

    Returns
    -------
    ndarray
        Standard errors
    """
    return np.sqrt(np.diag(vcov))


def compute_t_statistics(
    params: np.ndarray, standard_errors: np.ndarray, null_values: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute t-statistics for parameters.

    Parameters
    ----------
    params : ndarray
        Parameter estimates
    standard_errors : ndarray
        Standard errors
    null_values : ndarray, optional
        Null hypothesis values (default is zeros)

    Returns
    -------
    ndarray
        t-statistics
    """
    if null_values is None:
        null_values = np.zeros_like(params)

    return (params - null_values) / standard_errors


def compute_p_values(t_statistics: np.ndarray, df: Optional[int] = None) -> np.ndarray:
    """
    Compute two-sided p-values from t-statistics.

    Parameters
    ----------
    t_statistics : ndarray
        t-statistics
    df : int, optional
        Degrees of freedom (uses normal if None)

    Returns
    -------
    ndarray
        Two-sided p-values
    """
    if df is None:
        # Use normal distribution
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_statistics)))
    else:
        # Use t-distribution
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_statistics), df))

    return p_values


def compute_confidence_intervals(
    params: np.ndarray, standard_errors: np.ndarray, alpha: float = 0.05, df: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence intervals for parameters.

    Parameters
    ----------
    params : ndarray
        Parameter estimates
    standard_errors : ndarray
        Standard errors
    alpha : float
        Significance level (default 0.05 for 95% CI)
    df : int, optional
        Degrees of freedom (uses normal if None)

    Returns
    -------
    lower : ndarray
        Lower bounds of confidence intervals
    upper : ndarray
        Upper bounds of confidence intervals
    """
    if df is None:
        # Use normal distribution
        critical_value = stats.norm.ppf(1 - alpha / 2)
    else:
        # Use t-distribution
        critical_value = stats.t.ppf(1 - alpha / 2, df)

    margin = critical_value * standard_errors
    lower = params - margin
    upper = params + margin

    return lower, upper


def compute_aic(llf: float, n_params: int) -> float:
    """
    Compute Akaike Information Criterion.

    Parameters
    ----------
    llf : float
        Log-likelihood
    n_params : int
        Number of parameters

    Returns
    -------
    float
        AIC value
    """
    return 2 * n_params - 2 * llf


def compute_bic(llf: float, n_params: int, n_obs: int) -> float:
    """
    Compute Bayesian Information Criterion.

    Parameters
    ----------
    llf : float
        Log-likelihood
    n_params : int
        Number of parameters
    n_obs : int
        Number of observations

    Returns
    -------
    float
        BIC value
    """
    return n_params * np.log(n_obs) - 2 * llf
