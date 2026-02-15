"""
Standard errors for Maximum Likelihood Estimation in panel models.

This module provides sandwich (robust) and cluster-robust standard errors
specifically for MLE estimators, including:
- Sandwich estimator (Huber-White robust SEs)
- Cluster-robust sandwich estimator
- Bootstrap standard errors
- Delta method for transformed parameters

References
----------
.. [1] White, H. (1982). "Maximum Likelihood Estimation of Misspecified Models."
       Econometrica, 50(1), 1-25.
.. [2] Cameron, A. C., & Trivedi, P. K. (2005). Microeconometrics.
       Cambridge University Press.
"""

from typing import Callable, Literal, Optional, Union

import numpy as np
from scipy import stats

from panelbox.standard_errors.clustered import cluster_by_entity


class MLECovarianceResult:
    """
    Container for MLE covariance matrix estimation results.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix
    std_errors : np.ndarray
        Standard errors (sqrt of diagonal)
    method : str
        Method used ('nonrobust', 'robust', 'cluster', 'bootstrap')
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters

    Attributes
    ----------
    cov_matrix : np.ndarray
        Covariance matrix (k × k)
    std_errors : np.ndarray
        Standard errors (k,)
    method : str
        Estimation method
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters
    """

    def __init__(
        self,
        cov_matrix: np.ndarray,
        std_errors: np.ndarray,
        method: str,
        n_obs: int,
        n_params: int,
    ):
        self.cov_matrix = cov_matrix
        self.std_errors = std_errors
        self.method = method
        self.n_obs = n_obs
        self.n_params = n_params

    def __repr__(self) -> str:
        return (
            f"MLECovarianceResult("
            f"method='{self.method}', "
            f"n_obs={self.n_obs}, "
            f"n_params={self.n_params})"
        )


def sandwich_estimator(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: Literal["nonrobust", "robust"] = "robust",
) -> MLECovarianceResult:
    """
    Compute sandwich (Huber-White robust) covariance estimator for MLE.

    The sandwich estimator is:

        V̂ = H⁻¹ S H⁻¹

    where:
    - H = -Hessian of log-likelihood (information matrix)
    - S = outer product of scores (variance of score)

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix of log-likelihood at MLE (k × k)
        Should be negative definite for maximum
    scores : np.ndarray
        Score vectors (gradients) for each observation (n × k)
        Each row is ∂ℓᵢ/∂β
    method : {'nonrobust', 'robust'}, default='robust'
        - 'nonrobust': V̂ = -H⁻¹ (classical MLE)
        - 'robust': V̂ = H⁻¹ S H⁻¹ (sandwich, robust to misspecification)

    Returns
    -------
    MLECovarianceResult
        Covariance matrix and standard errors

    Notes
    -----
    **Non-robust (Classical MLE):**

    Under correct specification and regularity conditions:

        Var(β̂) = -H⁻¹ = [E(-∂²ℓ/∂β∂β')]⁻¹

    **Robust (Sandwich):**

    Robust to misspecification:

        Var(β̂) = H⁻¹ S H⁻¹

    where S = Σᵢ (∂ℓᵢ/∂β)(∂ℓᵢ/∂β)' is the empirical variance of scores.

    The sandwich estimator is also called:
    - Huber-White estimator
    - Robust covariance estimator
    - QMLE (Quasi-MLE) standard errors

    **Important:**

    - Hessian should be evaluated at MLE
    - Scores should be for individual observations (not summed)
    - Hessian is negated in the formula (-H⁻¹) because it's
      the negative second derivative of log-likelihood

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import sandwich_estimator
    >>>
    >>> # Example: Logit model with 100 obs, 3 parameters
    >>> n, k = 100, 3
    >>>
    >>> # Hessian at MLE (negative definite)
    >>> H = -np.eye(k) * 10  # Simplified example
    >>>
    >>> # Scores for each observation
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Robust covariance
    >>> result = sandwich_estimator(H, scores, method='robust')
    >>> print(result.std_errors)

    See Also
    --------
    cluster_robust_mle : Cluster-robust version
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    if method == "nonrobust":
        # Classical MLE: V = -H⁻¹
        vcov = H_inv

    elif method == "robust":
        # Meat: S = Σᵢ sᵢ sᵢ' (outer product of scores)
        S = scores.T @ scores

        # Sandwich: H⁻¹ S H⁻¹
        vcov = H_inv @ S @ H_inv

    else:
        raise ValueError(f"method must be 'nonrobust' or 'robust', got '{method}'")

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method=method,
        n_obs=n_obs,
        n_params=n_params,
    )


def cluster_robust_mle(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult:
    """
    Compute cluster-robust covariance estimator for MLE.

    The cluster-robust sandwich estimator is:

        V̂ = H⁻¹ [Σᵢ gᵢ gᵢ'] H⁻¹

    where gᵢ = Σₜ ∂ℓᵢₜ/∂β is the sum of scores within cluster i.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix at MLE (k × k)
    scores : np.ndarray
        Score vectors for each observation (n × k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply small-sample degrees of freedom correction:
        (G/(G-1)) × ((N-1)/(N-K))

    Returns
    -------
    MLECovarianceResult
        Cluster-robust covariance matrix and standard errors

    Notes
    -----
    **Cluster-Robust Variance:**

    Allows for arbitrary correlation within clusters but assumes
    independence across clusters.

    For panel data, typically cluster by entity (i) to allow
    correlation across time (t) within each entity.

    **Degrees of Freedom Correction:**

    The correction factor is:

        adj = (G/(G-1)) × ((N-1)/(N-K))

    where G = number of clusters, N = observations, K = parameters.

    This improves small-sample performance.

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import cluster_robust_mle
    >>>
    >>> # 100 observations, 3 parameters, 20 clusters
    >>> n, k = 100, 3
    >>> H = -np.eye(k) * 10
    >>> scores = np.random.randn(n, k) * 0.1
    >>>
    >>> # Cluster IDs (e.g., firm IDs)
    >>> cluster_ids = np.repeat(np.arange(20), 5)
    >>>
    >>> # Cluster-robust SEs
    >>> result = cluster_robust_mle(H, scores, cluster_ids)
    >>> print(result.std_errors)

    See Also
    --------
    sandwich_estimator : Non-clustered robust estimator
    """
    n_obs, n_params = scores.shape

    # Bread: -H⁻¹
    H_inv = -np.linalg.inv(hessian)

    # Meat: cluster by summing scores within each cluster
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    meat = np.zeros((n_params, n_params))

    for cluster in unique_clusters:
        mask = cluster_ids == cluster
        # Sum scores within cluster
        g_i = scores[mask].sum(axis=0)
        # Outer product
        meat += np.outer(g_i, g_i)

    # Degrees of freedom correction
    if df_correction:
        adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        meat *= adj

    # Sandwich
    vcov = H_inv @ meat @ H_inv

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="cluster",
        n_obs=n_obs,
        n_params=n_params,
    )


def delta_method(
    vcov: np.ndarray,
    transform_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Compute variance of transformed parameters using delta method.

    For a transformation g(β), the variance is approximately:

        Var[g(β̂)] ≈ [∇g(β̂)]' Var(β̂) [∇g(β̂)]

    Parameters
    ----------
    vcov : np.ndarray
        Covariance matrix of original parameters (k × k)
    transform_func : callable
        Transformation function g: R^k -> R^m
        Should accept np.ndarray of shape (k,) and return (m,)
    params : np.ndarray
        Parameter estimates at which to evaluate gradient (k,)
    epsilon : float, default=1e-7
        Step size for numerical gradient

    Returns
    -------
    np.ndarray
        Covariance matrix of transformed parameters (m × m)

    Notes
    -----
    **Delta Method:**

    First-order Taylor approximation:

        g(β̂) ≈ g(β) + ∇g(β)(β̂ - β)

    Therefore:

        Var[g(β̂)] ≈ [∇g(β)]' Var(β̂) [∇g(β)]

    where ∇g is the Jacobian matrix (m × k).

    **Use Cases:**

    - Marginal effects (transformations of coefficients)
    - Odds ratios: exp(β)
    - Elasticities
    - Any nonlinear function of parameters

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import delta_method
    >>>
    >>> # Original parameters and covariance
    >>> params = np.array([0.5, 1.0])
    >>> vcov = np.eye(2) * 0.01
    >>>
    >>> # Transformation: odds ratio = exp(β)
    >>> def odds_ratio(beta):
    ...     return np.exp(beta)
    >>>
    >>> # Variance of odds ratios
    >>> vcov_or = delta_method(vcov, odds_ratio, params)
    >>> se_or = np.sqrt(np.diag(vcov_or))
    >>> print(f"Odds ratios: {odds_ratio(params)}")
    >>> print(f"Standard errors: {se_or}")

    See Also
    --------
    numpy.gradient : Numerical gradient computation
    """
    k = len(params)

    # Evaluate transformation at params
    g_beta = transform_func(params)
    m = len(np.atleast_1d(g_beta))

    # Compute Jacobian (gradient) numerically
    # J[i, j] = ∂gᵢ/∂βⱼ
    jacobian = np.zeros((m, k))

    for j in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += epsilon
        params_minus[j] -= epsilon

        g_plus = transform_func(params_plus)
        g_minus = transform_func(params_minus)

        jacobian[:, j] = (g_plus - g_minus) / (2 * epsilon)

    # Delta method: J' V J
    vcov_transformed = jacobian @ vcov @ jacobian.T

    # Ensure symmetry
    vcov_transformed = (vcov_transformed + vcov_transformed.T) / 2

    return vcov_transformed


def compute_mle_standard_errors(
    model,
    params: np.ndarray,
    se_type: str = "cluster",
    entity_id: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute standard errors for MLE models.

    Parameters
    ----------
    model : object
        Model object with methods to compute Hessian and scores.
    params : np.ndarray
        Parameter estimates.
    se_type : str, default='cluster'
        Type of standard errors:
        - 'nonrobust': Classical MLE standard errors
        - 'robust': Huber-White sandwich estimator
        - 'cluster': Cluster-robust standard errors
    entity_id : np.ndarray, optional
        Entity IDs for clustering (required if se_type='cluster').

    Returns
    -------
    np.ndarray
        Covariance matrix of parameters.
    """
    # Compute Hessian at MLE
    hessian = model._hessian(params)

    if se_type == "nonrobust":
        # Classical MLE: Var(β) = -H^{-1}
        try:
            cov_matrix = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse
            cov_matrix = -np.linalg.pinv(hessian)

    elif se_type in ["robust", "cluster"]:
        # Compute scores for each observation
        scores = model._score_obs(params)

        if se_type == "robust":
            # Sandwich estimator
            result = sandwich_estimator(hessian, scores, method="robust")
            cov_matrix = result.cov_matrix

        else:  # se_type == "cluster"
            if entity_id is None:
                raise ValueError("entity_id required for cluster-robust SEs")

            # Cluster-robust sandwich estimator
            result = cluster_robust_mle(hessian, scores, entity_id)
            cov_matrix = result.cov_matrix

    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    return cov_matrix


def bootstrap_mle(
    estimate_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: Optional[np.ndarray] = None,
    seed: int = 42,
) -> MLECovarianceResult:
    """
    Compute bootstrap standard errors for MLE estimators.

    Parameters
    ----------
    estimate_func : callable
        Function that estimates parameters from (y, X)
        Should return parameter vector of shape (k,)
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Design matrix (n × k)
    n_bootstrap : int, default=999
        Number of bootstrap replications
    cluster_ids : np.ndarray, optional
        If provided, use cluster bootstrap (resample clusters)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    MLECovarianceResult
        Bootstrap covariance matrix and standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample observations (or clusters) with replacement
    2. Re-estimate model on bootstrap sample
    3. Repeat B times
    4. Compute sample covariance of bootstrap estimates

    **Cluster Bootstrap:**

    When cluster_ids is provided, resample entire clusters
    rather than individual observations. This preserves
    within-cluster dependence.

    **Advantages:**

    - No distributional assumptions
    - Works when analytical SEs are difficult
    - Robust to misspecification

    **Disadvantages:**

    - Computationally expensive
    - Requires many replications (B ≥ 999)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.standard_errors.mle import bootstrap_mle
    >>> from scipy.optimize import minimize
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = (X @ [0.5, 1.0] + np.random.randn(n) > 0).astype(int)
    >>>
    >>> # Logit estimation function
    >>> def estimate_logit(y, X):
    ...     def neg_ll(beta):
    ...         eta = X @ beta
    ...         return -np.sum(y * eta - np.log1p(np.exp(eta)))
    ...     result = minimize(neg_ll, np.zeros(X.shape[1]))
    ...     return result.x
    >>>
    >>> # Bootstrap SEs
    >>> boot_result = bootstrap_mle(estimate_logit, y, X, n_bootstrap=199)
    >>> print(boot_result.std_errors)

    See Also
    --------
    scipy.stats.bootstrap : General bootstrap function
    """
    np.random.seed(seed)

    n_obs = len(y)
    params_original = estimate_func(y, X)
    n_params = len(params_original)

    # Store bootstrap estimates
    bootstrap_estimates = np.zeros((n_bootstrap, n_params))

    if cluster_ids is None:
        # Standard bootstrap: resample observations
        for b in range(n_bootstrap):
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except:
                # If estimation fails, use NaN
                bootstrap_estimates[b] = np.nan

    else:
        # Cluster bootstrap: resample clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for b in range(n_bootstrap):
            # Resample clusters
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

            # Get all observations from sampled clusters
            indices = []
            for cluster in sampled_clusters:
                indices.extend(np.where(cluster_ids == cluster)[0])

            indices = np.array(indices)
            y_boot = y[indices]
            X_boot = X[indices]

            try:
                bootstrap_estimates[b] = estimate_func(y_boot, X_boot)
            except:
                bootstrap_estimates[b] = np.nan

    # Remove failed replications
    valid_mask = ~np.isnan(bootstrap_estimates).any(axis=1)
    bootstrap_estimates = bootstrap_estimates[valid_mask]

    if len(bootstrap_estimates) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            f"More than 50% of bootstrap replications failed. " f"Results may be unreliable.",
            UserWarning,
        )

    # Compute covariance from bootstrap estimates
    vcov = np.cov(bootstrap_estimates, rowvar=False)

    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))

    return MLECovarianceResult(
        cov_matrix=vcov,
        std_errors=std_errors,
        method="bootstrap",
        n_obs=n_obs,
        n_params=n_params,
    )
