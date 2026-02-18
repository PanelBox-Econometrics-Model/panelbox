"""
Marginal effects for count data models.

This module implements marginal effects computations for count data models
including Poisson and Negative Binomial models. Provides Average Marginal
Effects (AME) and Marginal Effects at Means (MEM).
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from panelbox.marginal_effects.delta_method import delta_method_se, numerical_gradient
from panelbox.marginal_effects.discrete_me import MarginalEffectsResult


def compute_poisson_ame(result, varlist: Optional[List[str]] = None) -> MarginalEffectsResult:
    """
    Compute Average Marginal Effects (AME) for Poisson models.

    For Poisson models: E[y|X] = exp(X'β)
    Marginal effect: ME_k = β_k * exp(X'β)
    AME_k = (1/N) Σᵢ β_k * exp(Xᵢ'β)

    The marginal effect measures the change in expected count for a one-unit
    change in the covariate.

    Parameters
    ----------
    result : PanelModelResults
        Fitted Poisson model result
    varlist : list of str, optional
        Variables to compute ME for. If None, compute for all.

    Returns
    -------
    MarginalEffectsResult
        Container with AME and standard errors

    Examples
    --------
    >>> model = PooledPoisson(y, X, entity_id=firms)
    >>> result = model.fit()
    >>> ame = compute_poisson_ame(result)
    >>> print(ame.summary())

    Notes
    -----
    Standard errors are computed using the delta method to account for
    estimation uncertainty in β.

    References
    ----------
    Cameron, A. C., & Trivedi, P. K. (2013). Regression analysis of count data.
    Cambridge University Press.
    """
    model = result.model

    # Get exogenous variables and names
    if hasattr(model, "exog_df"):
        X = model.exog_df.values
        exog_names = model.exog_df.columns.tolist()
    else:
        X = model.exog
        exog_names = (
            model.exog_names
            if hasattr(model, "exog_names")
            else [f"x{i}" for i in range(X.shape[1])]
        )

    params = result.params
    if isinstance(params, pd.Series):
        params = params.values

    # Handle NegBin case where params includes alpha
    if hasattr(model, "alpha") and model.alpha is not None:
        # For NegBin, exclude alpha parameter
        params_beta = params[:-1] if len(params) > X.shape[1] else params
    else:
        params_beta = params

    if varlist is None:
        varlist = exog_names

    # Compute linear predictions
    linear_pred = X @ params_beta
    lambda_hat = np.exp(linear_pred)

    # Compute AME for each variable
    ame = {}
    ame_gradients = {}

    for var in varlist:
        if var not in exog_names:
            continue

        var_idx = exog_names.index(var)
        beta_k = params_beta[var_idx]

        # AME = (1/N) Σᵢ β_k * exp(Xᵢ'β)
        ame[var] = np.mean(beta_k * lambda_hat)

        # Gradient for delta method
        ame_gradients[var] = _compute_ame_gradient_poisson(X, params_beta, var_idx, lambda_hat)

    # Compute standard errors via delta method
    std_errors = {}
    cov_matrix = result.cov_params if hasattr(result, "cov_params") else result.vcov

    # Adjust covariance matrix for NegBin (exclude alpha)
    if (
        hasattr(model, "alpha")
        and model.alpha is not None
        and cov_matrix.shape[0] > len(params_beta)
    ):
        cov_matrix = cov_matrix[: len(params_beta), : len(params_beta)]

    for var in ame.keys():
        gradient = ame_gradients[var]
        se_results = delta_method_se(gradient, cov_matrix)
        std_errors[var] = (
            se_results["std_error"] if "std_error" in se_results else se_results["std_errors"]
        )

    return MarginalEffectsResult(ame, std_errors, result, me_type="ame")


def _compute_ame_gradient_poisson(
    X: np.ndarray, params: np.ndarray, var_idx: int, lambda_hat: np.ndarray
) -> np.ndarray:
    """
    Compute gradient of Poisson AME with respect to parameters.

    ∂AME_k/∂β_j = (1/N) Σᵢ ∂(β_k * exp(Xᵢ'β))/∂β_j

    For j = k:
        ∂AME_k/∂β_k = (1/N) Σᵢ [exp(Xᵢ'β) + β_k * X_ik * exp(Xᵢ'β)]
                     = (1/N) Σᵢ exp(Xᵢ'β) * (1 + β_k * X_ik)

    For j ≠ k:
        ∂AME_k/∂β_j = (1/N) Σᵢ β_k * X_ij * exp(Xᵢ'β)

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix (N x K)
    params : np.ndarray
        Parameter estimates (K,)
    var_idx : int
        Index of variable for which AME is computed
    lambda_hat : np.ndarray
        Predicted means exp(X'β) (N,)

    Returns
    -------
    np.ndarray
        Gradient vector (K,)
    """
    n_obs, n_params = X.shape
    gradient = np.zeros(n_params)

    beta_k = params[var_idx]

    for j in range(n_params):
        if j == var_idx:
            # Diagonal element
            gradient[j] = np.mean(lambda_hat * (1 + beta_k * X[:, var_idx]))
        else:
            # Off-diagonal
            gradient[j] = np.mean(beta_k * X[:, j] * lambda_hat)

    return gradient


def compute_poisson_mem(result, varlist: Optional[List[str]] = None) -> MarginalEffectsResult:
    """
    Compute Marginal Effects at Means (MEM) for Poisson models.

    For Poisson models: E[y|X] = exp(X'β)
    MEM_k = β_k * exp(X̄'β)

    Evaluates the marginal effect at the mean of all covariates.

    Parameters
    ----------
    result : PanelModelResults
        Fitted Poisson model result
    varlist : list of str, optional
        Variables to compute ME for. If None, compute for all.

    Returns
    -------
    MarginalEffectsResult
        Container with MEM and standard errors

    Examples
    --------
    >>> model = PooledPoisson(y, X, entity_id=firms)
    >>> result = model.fit()
    >>> mem = compute_poisson_mem(result)
    >>> print(mem.summary())

    Notes
    -----
    MEM is computationally simpler than AME and provides a single
    representative marginal effect. However, it may not be representative
    if there is substantial heterogeneity in the covariates.

    References
    ----------
    Cameron, A. C., & Trivedi, P. K. (2013). Regression analysis of count data.
    Cambridge University Press.
    """
    model = result.model

    # Get exogenous variables and names
    if hasattr(model, "exog_df"):
        X = model.exog_df.values
        exog_names = model.exog_df.columns.tolist()
    else:
        X = model.exog
        exog_names = (
            model.exog_names
            if hasattr(model, "exog_names")
            else [f"x{i}" for i in range(X.shape[1])]
        )

    params = result.params
    if isinstance(params, pd.Series):
        params = params.values

    # Handle NegBin case where params includes alpha
    if hasattr(model, "alpha") and model.alpha is not None:
        params_beta = params[:-1] if len(params) > X.shape[1] else params
    else:
        params_beta = params

    if varlist is None:
        varlist = exog_names

    # Compute mean of X and predicted value at mean
    X_mean = X.mean(axis=0)
    linear_pred_mean = X_mean @ params_beta
    lambda_mean = np.exp(linear_pred_mean)

    # Compute MEM for each variable
    mem = {}
    mem_gradients = {}

    for var in varlist:
        if var not in exog_names:
            continue

        var_idx = exog_names.index(var)
        beta_k = params_beta[var_idx]

        # MEM = β_k * exp(X̄'β)
        mem[var] = beta_k * lambda_mean

        # Gradient for delta method
        mem_gradients[var] = _compute_mem_gradient_poisson(
            X_mean, params_beta, var_idx, lambda_mean
        )

    # Compute standard errors via delta method
    std_errors = {}
    cov_matrix = result.cov_params if hasattr(result, "cov_params") else result.vcov

    # Adjust covariance matrix for NegBin (exclude alpha)
    if (
        hasattr(model, "alpha")
        and model.alpha is not None
        and cov_matrix.shape[0] > len(params_beta)
    ):
        cov_matrix = cov_matrix[: len(params_beta), : len(params_beta)]

    for var in mem.keys():
        gradient = mem_gradients[var]
        se_results = delta_method_se(gradient, cov_matrix)
        std_errors[var] = (
            se_results["std_error"] if "std_error" in se_results else se_results["std_errors"]
        )

    # Store mean values for reference
    at_values = {name: X_mean[i] for i, name in enumerate(exog_names)}

    return MarginalEffectsResult(mem, std_errors, result, me_type="mem", at_values=at_values)


def _compute_mem_gradient_poisson(
    X_mean: np.ndarray, params: np.ndarray, var_idx: int, lambda_mean: float
) -> np.ndarray:
    """
    Compute gradient of Poisson MEM with respect to parameters.

    ∂MEM_k/∂β_j = ∂(β_k * exp(X̄'β))/∂β_j

    For j = k:
        ∂MEM_k/∂β_k = exp(X̄'β) + β_k * X̄_k * exp(X̄'β)
                     = exp(X̄'β) * (1 + β_k * X̄_k)

    For j ≠ k:
        ∂MEM_k/∂β_j = β_k * X̄_j * exp(X̄'β)

    Parameters
    ----------
    X_mean : np.ndarray
        Mean of covariates (K,)
    params : np.ndarray
        Parameter estimates (K,)
    var_idx : int
        Index of variable for which MEM is computed
    lambda_mean : float
        Predicted mean at X̄: exp(X̄'β)

    Returns
    -------
    np.ndarray
        Gradient vector (K,)
    """
    n_params = len(params)
    gradient = np.zeros(n_params)

    beta_k = params[var_idx]

    for j in range(n_params):
        if j == var_idx:
            # Diagonal element
            gradient[j] = lambda_mean * (1 + beta_k * X_mean[var_idx])
        else:
            # Off-diagonal
            gradient[j] = beta_k * X_mean[j] * lambda_mean

    return gradient


def compute_negbin_ame(result, varlist: Optional[List[str]] = None) -> MarginalEffectsResult:
    """
    Compute Average Marginal Effects (AME) for Negative Binomial models.

    For Negative Binomial models, the conditional mean is the same as Poisson:
    E[y|X] = exp(X'β)

    Therefore, marginal effects follow the same formula:
    ME_k = β_k * exp(X'β)
    AME_k = (1/N) Σᵢ β_k * exp(Xᵢ'β)

    The difference from Poisson is in the variance structure, which affects
    standard errors through the covariance matrix of β̂.

    Parameters
    ----------
    result : NegativeBinomialResults
        Fitted Negative Binomial model result
    varlist : list of str, optional
        Variables to compute ME for. If None, compute for all.

    Returns
    -------
    MarginalEffectsResult
        Container with AME and standard errors

    Examples
    --------
    >>> model = NegativeBinomial(y, X, entity_id=firms)
    >>> result = model.fit()
    >>> ame = compute_negbin_ame(result)
    >>> print(ame.summary())

    Notes
    -----
    While the functional form of marginal effects is identical to Poisson,
    the NB model accounts for overdispersion, which affects the precision
    of coefficient estimates and thus the standard errors of marginal effects.

    References
    ----------
    Cameron, A. C., & Trivedi, P. K. (2013). Regression analysis of count data.
    Cambridge University Press.
    """
    # NB has same conditional mean as Poisson, so use same function
    return compute_poisson_ame(result, varlist=varlist)


def compute_negbin_mem(result, varlist: Optional[List[str]] = None) -> MarginalEffectsResult:
    """
    Compute Marginal Effects at Means (MEM) for Negative Binomial models.

    For Negative Binomial models, the conditional mean is the same as Poisson:
    E[y|X] = exp(X'β)

    Therefore, MEM follows the same formula:
    MEM_k = β_k * exp(X̄'β)

    The difference from Poisson is in the variance structure, which affects
    standard errors through the covariance matrix of β̂.

    Parameters
    ----------
    result : NegativeBinomialResults
        Fitted Negative Binomial model result
    varlist : list of str, optional
        Variables to compute ME for. If None, compute for all.

    Returns
    -------
    MarginalEffectsResult
        Container with MEM and standard errors

    Examples
    --------
    >>> model = NegativeBinomial(y, X, entity_id=firms)
    >>> result = model.fit()
    >>> mem = compute_negbin_mem(result)
    >>> print(mem.summary())

    Notes
    -----
    While the functional form of marginal effects is identical to Poisson,
    the NB model accounts for overdispersion, which affects the precision
    of coefficient estimates and thus the standard errors of marginal effects.

    References
    ----------
    Cameron, A. C., & Trivedi, P. K. (2013). Regression analysis of count data.
    Cambridge University Press.
    """
    # NB has same conditional mean as Poisson, so use same function
    return compute_poisson_mem(result, varlist=varlist)
