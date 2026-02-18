"""
Marginal effects for censored (Tobit) models.

This module implements marginal effects computations for Tobit models
including Average Marginal Effects (AME) and Marginal Effects at Means (MEM)
for different prediction types: conditional, unconditional, and probability.

References
----------
.. [1] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section
       and Panel Data (2nd ed.). MIT Press. Chapter 17.
.. [2] Greene, W. H. (2018). Econometric Analysis (8th ed.). Pearson.
       Chapter 19.
.. [3] Amemiya, T. (1984). Tobit models: A survey. Journal of Econometrics,
       24(1-2), 3-61.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from panelbox.marginal_effects.delta_method import delta_method_se, numerical_gradient
from panelbox.marginal_effects.discrete_me import MarginalEffectsResult


def _inverse_mills_ratio(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute inverse Mills ratio: λ(z) = φ(z)/Φ(z)

    The inverse Mills ratio is used extensively in Tobit models for
    computing marginal effects on the conditional mean.

    Parameters
    ----------
    z : array_like
        Input values (standardized)

    Returns
    -------
    array_like
        Inverse Mills ratio values

    Notes
    -----
    Uses a safe computation to avoid division by zero when Φ(z) is very small.
    For very negative z, we use the asymptotic expansion:
    λ(z) ≈ -z for z << 0
    """
    z = np.asarray(z)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)

    # Avoid division by zero: use asymptotic approximation for very negative z
    result = np.zeros_like(z, dtype=float)
    safe_mask = Phi > 1e-10

    result[safe_mask] = phi[safe_mask] / Phi[safe_mask]
    # Asymptotic approximation for very negative z
    result[~safe_mask] = -z[~safe_mask]

    return result


def _mills_ratio_derivative(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute derivative of inverse Mills ratio.

    d/dz[λ(z)] = -λ(z) * [z + λ(z)]

    This derivative is used in computing marginal effects on the
    conditional mean for Tobit models.

    Parameters
    ----------
    z : array_like
        Input values (standardized)

    Returns
    -------
    array_like
        Derivative values

    Notes
    -----
    The derivative is always negative, which reflects the fact that
    the inverse Mills ratio is a decreasing function.
    """
    lambda_z = _inverse_mills_ratio(z)
    return -lambda_z * (z + lambda_z)


def compute_tobit_ame(
    result, which: str = "conditional", varlist: Optional[List[str]] = None
) -> MarginalEffectsResult:
    """
    Compute Average Marginal Effects (AME) for Tobit models.

    AME averages the marginal effect across all observations in the sample.

    Parameters
    ----------
    result : FitResult
        Fitted Tobit model result (PooledTobit or RandomEffectsTobit)
    which : str, default='conditional'
        Type of marginal effect:
        - 'conditional': E[y|y>c, X] - Expected value given non-censoring
        - 'unconditional': E[y|X] - Unconditional expected value
        - 'probability': P(y>c|X) - Probability of non-censoring
    varlist : list of str, optional
        Variables to compute ME for. If None, compute for all.

    Returns
    -------
    MarginalEffectsResult
        AME with standard errors via delta method

    Notes
    -----
    For Tobit with left censoring at c:

    1. **Unconditional mean** (accounts for censoring):

       .. math::
           E[y|X] = Φ((X'β - c)/σ) · X'β + σ · φ((X'β - c)/σ)
                    + c · Φ((c - X'β)/σ)

       Marginal effect:

       .. math::
           ∂E[y|X]/∂x_k = β_k · Φ((X'β - c)/σ)

    2. **Conditional mean** (given y > c):

       .. math::
           E[y|y>c, X] = X'β + σ · λ((X'β - c)/σ)

       where λ(z) = φ(z)/Φ(z) is the inverse Mills ratio.

       Marginal effect:

       .. math::
           ∂E[y|y>c, X]/∂x_k = β_k · [1 - λ(z) · (z + λ(z))]

       where z = (X'β - c)/σ

    3. **Probability of non-censoring**:

       .. math::
           P(y>c|X) = Φ((X'β - c)/σ)

       Marginal effect:

       .. math::
           ∂P(y>c|X)/∂x_k = (β_k/σ) · φ((X'β - c)/σ)

    where:
    - Φ is the standard normal CDF
    - φ is the standard normal PDF
    - σ is the error standard deviation
    - β_k is the coefficient for variable k

    References
    ----------
    .. [1] Wooldridge (2010), Econometric Analysis, Chapter 17
    .. [2] McDonald & Moffitt (1980), "The Uses of Tobit Analysis",
           Review of Economics and Statistics, 62(2), 318-321.
    """
    if which not in ["conditional", "unconditional", "probability"]:
        raise ValueError(
            f"which must be 'conditional', 'unconditional', or 'probability', got '{which}'"
        )

    model = result.model if hasattr(result, "model") else result

    # Get model parameters
    if hasattr(model, "beta"):
        beta = model.beta
        sigma = model.sigma if hasattr(model, "sigma") else model.sigma_eps
    else:
        raise ValueError("Model must be fitted before computing marginal effects")

    # Get exogenous variables
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

    # Get censoring point
    c = model.censoring_point
    censoring_type = model.censoring_type

    if censoring_type != "left":
        raise NotImplementedError(
            f"Marginal effects currently only implemented for left censoring, got '{censoring_type}'"
        )

    if varlist is None:
        varlist = exog_names

    # Compute linear prediction
    linear_pred = X @ beta  # X'β

    # Standardized z-score: (X'β - c) / σ
    z = (linear_pred - c) / sigma

    # Compute marginal effects based on type
    ame_dict = {}
    ame_se_dict = {}

    for var in varlist:
        if var not in exog_names:
            continue

        var_idx = exog_names.index(var)
        beta_k = beta[var_idx]

        if which == "unconditional":
            # AME for unconditional mean: E[∂E[y|X]/∂x_k]
            # = β_k * mean(Φ(z))
            Phi_z = norm.cdf(z)
            me_i = beta_k * Phi_z
            ame_dict[var] = np.mean(me_i)

        elif which == "conditional":
            # AME for conditional mean: E[∂E[y|y>c,X]/∂x_k]
            # = β_k * mean(1 - λ(z) * (z + λ(z)))
            lambda_z = _inverse_mills_ratio(z)
            me_i = beta_k * (1 - lambda_z * (z + lambda_z))
            ame_dict[var] = np.mean(me_i)

        elif which == "probability":
            # AME for probability: E[∂P(y>c|X)/∂x_k]
            # = (β_k/σ) * mean(φ(z))
            phi_z = norm.pdf(z)
            me_i = (beta_k / sigma) * phi_z
            ame_dict[var] = np.mean(me_i)

        # Compute standard errors via delta method
        def me_func(params):
            """Compute marginal effect for given parameters."""
            beta_params = params[: len(beta)]
            sigma_param = params[len(beta)]
            beta_k_param = beta_params[var_idx]

            linear_pred_param = X @ beta_params
            z_param = (linear_pred_param - c) / sigma_param

            if which == "unconditional":
                Phi_z_param = norm.cdf(z_param)
                return np.mean(beta_k_param * Phi_z_param)
            elif which == "conditional":
                lambda_z_param = _inverse_mills_ratio(z_param)
                return np.mean(beta_k_param * (1 - lambda_z_param * (z_param + lambda_z_param)))
            elif which == "probability":
                phi_z_param = norm.pdf(z_param)
                return np.mean((beta_k_param / sigma_param) * phi_z_param)

        # Get parameter vector and covariance matrix
        params = np.concatenate([beta, [sigma]])

        if hasattr(model, "cov_params"):
            # Extract covariance for beta and sigma (log scale in model, but we use level here)
            # Need to be careful with the transformation
            cov_params = model.cov_params

            # Compute SE using delta method
            try:
                se = delta_method_se(me_func, params, cov_params)
                ame_se_dict[var] = se
            except Exception:
                # Fallback to numerical gradient if delta method fails
                ame_se_dict[var] = np.nan
        else:
            ame_se_dict[var] = np.nan

    return MarginalEffectsResult(
        marginal_effects=ame_dict,
        std_errors=ame_se_dict,
        parent_result=result,
        me_type=f"AME_{which}",
    )


def compute_tobit_mem(
    result, which: str = "conditional", varlist: Optional[List[str]] = None
) -> MarginalEffectsResult:
    """
    Compute Marginal Effects at Means (MEM) for Tobit models.

    MEM evaluates the marginal effect at the mean of all covariates.

    Parameters
    ----------
    result : FitResult
        Fitted Tobit model result (PooledTobit or RandomEffectsTobit)
    which : str, default='conditional'
        Type of marginal effect:
        - 'conditional': E[y|y>c, X] - Expected value given non-censoring
        - 'unconditional': E[y|X] - Unconditional expected value
        - 'probability': P(y>c|X) - Probability of non-censoring
    varlist : list of str, optional
        Variables to compute ME for. If None, compute for all.

    Returns
    -------
    MarginalEffectsResult
        MEM with standard errors evaluated at mean of X

    Notes
    -----
    MEM evaluates the marginal effect formulas at X̄ (mean of covariates):

    1. **Unconditional mean**:

       .. math::
           ∂E[y|X̄]/∂x_k = β_k · Φ((X̄'β - c)/σ)

    2. **Conditional mean**:

       .. math::
           ∂E[y|y>c, X̄]/∂x_k = β_k · [1 - λ(z̄) · (z̄ + λ(z̄))]

       where z̄ = (X̄'β - c)/σ

    3. **Probability**:

       .. math::
           ∂P(y>c|X̄)/∂x_k = (β_k/σ) · φ((X̄'β - c)/σ)

    The interpretation is: "For an individual with average characteristics,
    how does a one-unit change in x_k affect the outcome?"
    """
    if which not in ["conditional", "unconditional", "probability"]:
        raise ValueError(
            f"which must be 'conditional', 'unconditional', or 'probability', got '{which}'"
        )

    model = result.model if hasattr(result, "model") else result

    # Get model parameters
    if hasattr(model, "beta"):
        beta = model.beta
        sigma = model.sigma if hasattr(model, "sigma") else model.sigma_eps
    else:
        raise ValueError("Model must be fitted before computing marginal effects")

    # Get exogenous variables
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

    # Get censoring point
    c = model.censoring_point
    censoring_type = model.censoring_type

    if censoring_type != "left":
        raise NotImplementedError(
            f"Marginal effects currently only implemented for left censoring, got '{censoring_type}'"
        )

    if varlist is None:
        varlist = exog_names

    # Compute mean of X
    X_mean = np.mean(X, axis=0)

    # Linear prediction at mean
    linear_pred_mean = X_mean @ beta  # X̄'β

    # Standardized z-score at mean
    z_mean = (linear_pred_mean - c) / sigma

    # Compute marginal effects at means
    mem_dict = {}
    mem_se_dict = {}

    for var in varlist:
        if var not in exog_names:
            continue

        var_idx = exog_names.index(var)
        beta_k = beta[var_idx]

        if which == "unconditional":
            # MEM for unconditional mean
            Phi_z = norm.cdf(z_mean)
            mem_dict[var] = beta_k * Phi_z

        elif which == "conditional":
            # MEM for conditional mean
            lambda_z = _inverse_mills_ratio(z_mean)
            mem_dict[var] = beta_k * (1 - lambda_z * (z_mean + lambda_z))

        elif which == "probability":
            # MEM for probability
            phi_z = norm.pdf(z_mean)
            mem_dict[var] = (beta_k / sigma) * phi_z

        # Compute standard errors via delta method
        def me_func(params):
            """Compute marginal effect for given parameters."""
            beta_params = params[: len(beta)]
            sigma_param = params[len(beta)]
            beta_k_param = beta_params[var_idx]

            linear_pred_param = X_mean @ beta_params
            z_param = (linear_pred_param - c) / sigma_param

            if which == "unconditional":
                Phi_z_param = norm.cdf(z_param)
                return beta_k_param * Phi_z_param
            elif which == "conditional":
                lambda_z_param = _inverse_mills_ratio(z_param)
                return beta_k_param * (1 - lambda_z_param * (z_param + lambda_z_param))
            elif which == "probability":
                phi_z_param = norm.pdf(z_param)
                return (beta_k_param / sigma_param) * phi_z_param

        # Get parameter vector and covariance matrix
        params = np.concatenate([beta, [sigma]])

        if hasattr(model, "cov_params"):
            cov_params = model.cov_params

            # Compute SE using delta method
            try:
                se = delta_method_se(me_func, params, cov_params)
                mem_se_dict[var] = se
            except Exception:
                mem_se_dict[var] = np.nan
        else:
            mem_se_dict[var] = np.nan

    return MarginalEffectsResult(
        marginal_effects=mem_dict,
        std_errors=mem_se_dict,
        parent_result=result,
        me_type=f"MEM_{which}",
        at_values=dict(zip(exog_names, X_mean)),
    )
