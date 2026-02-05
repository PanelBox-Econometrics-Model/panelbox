"""
Utility functions for covariance matrix estimation.

This module provides common functions for computing sandwich covariance
matrices and their components (bread and meat).
"""

from typing import Optional

import numpy as np


def compute_leverage(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    n, k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 0, 1)

    return np.asarray(leverage)


def compute_bread(X: np.ndarray) -> np.ndarray:
    """
    Compute the "bread" of the sandwich covariance estimator.

    Bread = (X'X)^{-1}

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    bread : np.ndarray
        Bread matrix (k x k)

    Notes
    -----
    The sandwich covariance estimator is:
    V = Bread @ Meat @ Bread

    where Meat depends on the specific robust estimator (HC, cluster, etc.)
    """
    XTX = X.T @ X
    bread = np.linalg.inv(XTX)
    return bread


def compute_meat_hc(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def sandwich_covariance(bread: np.ndarray, meat: np.ndarray) -> np.ndarray:
    """
    Compute sandwich covariance matrix.

    V = Bread @ Meat @ Bread

    Parameters
    ----------
    bread : np.ndarray
        Bread matrix (k x k)
    meat : np.ndarray
        Meat matrix (k x k)

    Returns
    -------
    cov : np.ndarray
        Covariance matrix (k x k)
    """
    return np.asarray(bread @ meat @ bread)


def compute_clustered_meat(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def compute_twoway_clustered_meat(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def hc_covariance(X: np.ndarray, resid: np.ndarray, method: str = "HC1") -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_meat_hc(X, resid, method)
    return sandwich_covariance(bread, meat)


def clustered_covariance(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_clustered_meat(X, resid, clusters, df_correction)
    return sandwich_covariance(bread, meat)


def twoway_clustered_covariance(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_twoway_clustered_meat(X, resid, clusters1, clusters2, df_correction)
    return sandwich_covariance(bread, meat)
