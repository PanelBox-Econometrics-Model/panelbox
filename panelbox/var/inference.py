"""
Inference and standard errors for Panel VAR models.

This module provides standard error computation and hypothesis testing
for Panel VAR models.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from scipy import stats

from panelbox.standard_errors import cluster_by_entity, driscoll_kraay, robust_covariance


@dataclass
class WaldTestResult:
    """
    Result of a Wald test.

    Attributes
    ----------
    statistic : float
        Wald test statistic
    pvalue : float
        P-value for the test
    df : int
        Degrees of freedom
    hypothesis : str
        Description of the null hypothesis
    """

    statistic: float
    pvalue: float
    df: int
    hypothesis: str

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Wald Test\n"
            f"H0: {self.hypothesis}\n"
            f"Statistic: {self.statistic:.4f}\n"
            f"P-value: {self.pvalue:.4f}\n"
            f"df: {self.df}"
        )


def compute_ols_equation(
    y: np.ndarray, X: np.ndarray, weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute OLS for a single equation.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Independent variables (n, k)
    weights : np.ndarray, optional
        Observation weights

    Returns
    -------
    beta : np.ndarray
        Coefficient estimates (k,)
    resid : np.ndarray
        Residuals (n,)
    fitted : np.ndarray
        Fitted values (n,)
    """
    if weights is not None:
        # Weighted least squares
        W = np.sqrt(weights)
        y_w = y * W
        X_w = X * W[:, np.newaxis]
        beta = np.linalg.lstsq(X_w, y_w, rcond=None)[0]
    else:
        # Ordinary least squares
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

    fitted = X @ beta
    resid = y - fitted

    return beta, resid, fitted


def within_transformation(data: np.ndarray, entities: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Apply within transformation (entity demeaning).

    Parameters
    ----------
    data : np.ndarray
        Data to demean (n, k) or (n,)
    entities : np.ndarray
        Entity identifiers (n,)

    Returns
    -------
    demeaned : np.ndarray
        Demeaned data
    entity_means : dict
        Dictionary of entity means for later reconstruction
    """
    # Handle 1D and 2D arrays
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        squeeze_output = True
    else:
        squeeze_output = False

    n, k = data.shape
    demeaned = np.zeros_like(data)
    entity_means = {}

    unique_entities = np.unique(entities)

    for entity in unique_entities:
        mask = entities == entity
        entity_data = data[mask]
        entity_mean = entity_data.mean(axis=0)
        entity_means[entity] = entity_mean
        demeaned[mask] = entity_data - entity_mean

    if squeeze_output:
        demeaned = demeaned.ravel()

    return demeaned, entity_means


def compute_sur_covariance(
    X: np.ndarray,
    residuals_all: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Compute SUR (Seemingly Unrelated Regression) covariance matrix.

    This exploits the contemporaneous correlation between residuals
    of different equations in the system.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (demeaned, n x k) - same for all equations
    residuals_all : np.ndarray
        Matrix of residuals for all K equations (n x K)
    K : int
        Number of equations in the system

    Returns
    -------
    vcov_sur : np.ndarray
        SUR covariance matrix (K*k x K*k)

    Notes
    -----
    The SUR covariance matrix is:

    V_SUR = Σ̂ ⊗ (X'X)⁻¹

    where:
    - Σ̂ is the residual covariance matrix (K x K)
    - X is the design matrix (same for all equations)
    - ⊗ denotes the Kronecker product

    This is appropriate for Panel VAR because all equations have the
    same regressors (lags of all variables).

    References
    ----------
    Zellner, A. (1962). "An Efficient Method of Estimating Seemingly
    Unrelated Regressions and Tests for Aggregation Bias".
    Journal of the American Statistical Association.
    """
    n, k = X.shape

    # Compute residual covariance matrix Σ̂ (K x K)
    # Σ̂ = (1/n) * Σ_t ε_t ε_t'
    Sigma_hat = residuals_all.T @ residuals_all / n

    # Compute (X'X)⁻¹
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    # SUR covariance: Σ̂ ⊗ (X'X)⁻¹
    vcov_sur = np.kron(Sigma_hat, XtX_inv)

    return vcov_sur


def compute_covariance_matrix(
    X: np.ndarray,
    resid: np.ndarray,
    cov_type: str,
    entities: Optional[np.ndarray] = None,
    times: Optional[np.ndarray] = None,
    **cov_kwds,
) -> np.ndarray:
    """
    Compute covariance matrix for OLS estimates.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (demeaned, n x k)
    resid : np.ndarray
        Residuals (demeaned, n)
    cov_type : str
        Type of covariance estimator:
        - 'nonrobust': Classical (assumes homoskedasticity)
        - 'hc1': Heteroskedasticity-robust (HC1)
        - 'clustered': Cluster-robust by entity
        - 'driscoll_kraay': Driscoll-Kraay HAC
        - 'sur': SUR (requires residuals_all in cov_kwds)
    entities : np.ndarray, optional
        Entity identifiers (required for clustered)
    times : np.ndarray, optional
        Time identifiers (required for driscoll_kraay)
    **cov_kwds
        Additional arguments for covariance estimation

    Returns
    -------
    vcov : np.ndarray
        Covariance matrix (k x k)
    """
    n, k = X.shape

    if cov_type == "nonrobust":
        # Classical covariance: σ² (X'X)⁻¹
        sigma2 = np.sum(resid**2) / (n - k)
        XtX = X.T @ X
        # Use pseudoinverse in case X'X is singular (can happen after demeaning)
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)
        vcov = sigma2 * XtX_inv

    elif cov_type == "hc1":
        # HC1: heteroskedasticity-robust
        result = robust_covariance(X, resid, method="HC1")
        vcov = result.cov_matrix

    elif cov_type == "clustered":
        # Cluster-robust by entity
        if entities is None:
            raise ValueError("entities required for clustered covariance")
        result = cluster_by_entity(X, resid, entities, df_correction=True)
        vcov = result.cov_matrix

    elif cov_type == "driscoll_kraay":
        # Driscoll-Kraay HAC
        if times is None:
            raise ValueError("times required for driscoll_kraay covariance")
        max_lags = cov_kwds.get("max_lags", None)
        kernel = cov_kwds.get("kernel", "bartlett")
        result = driscoll_kraay(X, resid, times, max_lags=max_lags, kernel=kernel)
        vcov = result.cov_matrix

    elif cov_type == "sur":
        # SUR covariance - requires residuals from all equations
        residuals_all = cov_kwds.get("residuals_all")
        K = cov_kwds.get("K")
        if residuals_all is None or K is None:
            raise ValueError("SUR requires residuals_all and K in cov_kwds")
        vcov = compute_sur_covariance(X, residuals_all, K)

    else:
        raise ValueError(
            f"cov_type must be one of: 'nonrobust', 'hc1', 'clustered', 'driscoll_kraay', 'sur', got '{cov_type}'"
        )

    return vcov


def wald_test(
    params: np.ndarray,
    cov_params: np.ndarray,
    R: np.ndarray,
    r: Optional[np.ndarray] = None,
) -> WaldTestResult:
    """
    Perform Wald test for linear restrictions.

    Tests H0: R*β = r vs H1: R*β ≠ r

    Parameters
    ----------
    params : np.ndarray
        Parameter estimates (k,)
    cov_params : np.ndarray
        Covariance matrix of parameters (k x k)
    R : np.ndarray
        Restriction matrix (q x k)
    r : np.ndarray, optional
        Right-hand side vector (q,). Default is zeros.

    Returns
    -------
    WaldTestResult
        Wald test result

    Notes
    -----
    The Wald statistic is computed as:

    W = (R*β - r)' [R*V*R']⁻¹ (R*β - r) ~ χ²(q)

    where V is the covariance matrix of β and q is the number of restrictions.

    Examples
    --------
    >>> # Test if first two coefficients are jointly zero
    >>> R = np.array([[1, 0, 0], [0, 1, 0]])
    >>> result = wald_test(params, cov_params, R)
    """
    if r is None:
        r = np.zeros(R.shape[0])

    # Compute restriction
    restriction = R @ params - r

    # Compute variance of restriction
    var_restriction = R @ cov_params @ R.T

    # Wald statistic: W = restriction' * var^(-1) * restriction
    try:
        var_restriction_inv = np.linalg.inv(var_restriction)
    except np.linalg.LinAlgError:
        # Singular matrix, use pseudoinverse
        var_restriction_inv = np.linalg.pinv(var_restriction)

    W = restriction.T @ var_restriction_inv @ restriction

    # Degrees of freedom
    df = R.shape[0]

    # P-value from chi-squared distribution
    pvalue = 1 - stats.chi2.cdf(W, df)

    return WaldTestResult(statistic=float(W), pvalue=float(pvalue), df=df, hypothesis="R*β = r")


def f_test_exclusion(
    params: np.ndarray,
    cov_params: np.ndarray,
    indices: list,
) -> WaldTestResult:
    """
    Test exclusion of variables (joint significance).

    Tests H0: β_j = 0 for j in indices

    Parameters
    ----------
    params : np.ndarray
        Parameter estimates (k,)
    cov_params : np.ndarray
        Covariance matrix (k x k)
    indices : list
        Indices of parameters to test

    Returns
    -------
    WaldTestResult
        Wald test result
    """
    k = len(params)
    q = len(indices)

    # Build restriction matrix
    R = np.zeros((q, k))
    for i, idx in enumerate(indices):
        R[i, idx] = 1.0

    return wald_test(params, cov_params, R)
