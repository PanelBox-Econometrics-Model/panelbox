"""
Delta method for standard errors of marginal effects.

This module provides tools for computing standard errors of transformations
using the delta method.
"""

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd


def numerical_gradient(func: Callable, params: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute numerical gradient using central differences.

    Parameters
    ----------
    func : callable
        Function to differentiate. Should accept params and return scalar or array.
    params : np.ndarray
        Point at which to evaluate gradient
    eps : float, default=1e-6
        Step size for finite differences

    Returns
    -------
    gradient : np.ndarray
        Gradient vector or Jacobian matrix
    """
    n_params = len(params)

    # Evaluate function at baseline
    f0 = func(params)

    # Determine output dimension
    if np.isscalar(f0):
        gradient = np.zeros(n_params)

        for i in range(n_params):
            # Forward step
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = func(params_plus)

            # Backward step
            params_minus = params.copy()
            params_minus[i] -= eps
            f_minus = func(params_minus)

            # Central difference
            gradient[i] = (f_plus - f_minus) / (2 * eps)
    else:
        # Multiple outputs - compute Jacobian
        f0 = np.asarray(f0)
        n_outputs = len(f0)
        gradient = np.zeros((n_outputs, n_params))

        for i in range(n_params):
            # Forward step
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = func(params_plus)

            # Backward step
            params_minus = params.copy()
            params_minus[i] -= eps
            f_minus = func(params_minus)

            # Central difference for each output
            gradient[:, i] = (f_plus - f_minus) / (2 * eps)

    return gradient


def delta_method_se(gradient: np.ndarray, cov_matrix: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Compute standard errors using the delta method.

    For a transformation g(β), the variance is:
    Var[g(β̂)] ≈ ∇g(β̂)' * Var(β̂) * ∇g(β̂)

    Parameters
    ----------
    gradient : np.ndarray
        Gradient or Jacobian of the transformation
    cov_matrix : np.ndarray
        Covariance matrix of parameters
    alpha : float, default=0.05
        Significance level for confidence intervals

    Returns
    -------
    results : dict
        Dictionary containing:
        - std_errors: Standard errors
        - z_stats: Z-statistics
        - pvalues: P-values from two-sided test
        - ci_lower: Lower confidence interval
        - ci_upper: Upper confidence interval
    """
    from scipy import stats

    # Ensure arrays
    gradient = np.asarray(gradient)
    cov_matrix = np.asarray(cov_matrix)

    if gradient.ndim == 1:
        # Single output - scalar variance
        variance = gradient @ cov_matrix @ gradient
        se = np.sqrt(variance)

        # Z-statistic (need point estimate separately)
        z_crit = stats.norm.ppf(1 - alpha / 2)

        return {"std_error": se, "z_critical": z_crit}
    else:
        # Multiple outputs - variance for each
        n_outputs = gradient.shape[0]
        variances = np.zeros(n_outputs)

        for i in range(n_outputs):
            grad_i = gradient[i, :]
            variances[i] = grad_i @ cov_matrix @ grad_i

        std_errors = np.sqrt(variances)
        z_crit = stats.norm.ppf(1 - alpha / 2)

        return {"std_errors": std_errors, "z_critical": z_crit}


def compute_me_gradient(
    model, params: np.ndarray, var_idx: int, X: np.ndarray, me_type: str = "ame"
) -> np.ndarray:
    """
    Compute gradient of marginal effects with respect to parameters.

    Parameters
    ----------
    model : object
        Fitted model with predict method
    params : np.ndarray
        Current parameter estimates
    var_idx : int
        Index of variable for which to compute ME gradient
    X : np.ndarray
        Covariate matrix
    me_type : str
        Type of marginal effect ('ame', 'mem', 'mer')

    Returns
    -------
    gradient : np.ndarray
        Gradient of marginal effect w.r.t. parameters
    """
    from scipy.stats import norm

    n_params = len(params)

    if me_type == "ame":
        # Average marginal effect - need to average gradients
        n_obs = X.shape[0]
        gradients = np.zeros((n_obs, n_params))

        for i in range(n_obs):
            Xi = X[i, :]
            linear_pred = Xi @ params

            if hasattr(model, "family") and model.family == "probit":
                # For Probit: ∂ME/∂β = ∂[β_k * φ(X'β)]/∂β
                pdf = norm.pdf(linear_pred)

                # Gradient w.r.t. β_k
                grad = np.zeros(n_params)
                grad[var_idx] = pdf  # Direct effect

                # Chain rule for other parameters
                for j in range(n_params):
                    if j != var_idx:
                        grad[j] = -params[var_idx] * Xi[j] * linear_pred * pdf
                    else:
                        grad[j] += -params[var_idx] * Xi[j] * linear_pred * pdf

                gradients[i, :] = grad

            elif hasattr(model, "family") and model.family == "logit":
                # For Logit: ∂ME/∂β = ∂[β_k * Λ(1-Λ)]/∂β
                exp_pred = np.exp(linear_pred)
                logit_pdf = exp_pred / (1 + exp_pred) ** 2
                logit_cdf = exp_pred / (1 + exp_pred)

                grad = np.zeros(n_params)
                grad[var_idx] = logit_pdf  # Direct effect

                # Chain rule for other parameters
                deriv2 = logit_pdf * (1 - 2 * logit_cdf)
                for j in range(n_params):
                    if j != var_idx:
                        grad[j] = params[var_idx] * Xi[j] * deriv2
                    else:
                        grad[j] += params[var_idx] * Xi[j] * deriv2

                gradients[i, :] = grad

        # Average over observations
        return gradients.mean(axis=0)

    elif me_type == "mem":
        # Marginal effect at means
        X_mean = X.mean(axis=0)
        linear_pred = X_mean @ params

        if hasattr(model, "family") and model.family == "probit":
            pdf = norm.pdf(linear_pred)
            grad = np.zeros(n_params)
            grad[var_idx] = pdf

            # Chain rule for other parameters
            for j in range(n_params):
                if j != var_idx:
                    grad[j] = -params[var_idx] * X_mean[j] * linear_pred * pdf
                else:
                    grad[j] += -params[var_idx] * X_mean[j] * linear_pred * pdf

            return grad

        elif hasattr(model, "family") and model.family == "logit":
            exp_pred = np.exp(linear_pred)
            logit_pdf = exp_pred / (1 + exp_pred) ** 2
            logit_cdf = exp_pred / (1 + exp_pred)

            grad = np.zeros(n_params)
            grad[var_idx] = logit_pdf

            deriv2 = logit_pdf * (1 - 2 * logit_cdf)
            for j in range(n_params):
                if j != var_idx:
                    grad[j] = params[var_idx] * X_mean[j] * deriv2
                else:
                    grad[j] += params[var_idx] * X_mean[j] * deriv2

            return grad

    else:
        raise ValueError(f"Unknown marginal effect type: {me_type}")
