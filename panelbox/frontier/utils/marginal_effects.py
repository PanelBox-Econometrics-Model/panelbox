"""
Marginal effects for SFA models with inefficiency determinants.

This module computes marginal effects of covariates on inefficiency
and efficiency, with standard errors via delta method.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def marginal_effects_wang_2002(
    result,
    method: str = "location",
    at_means: bool = True,
) -> pd.DataFrame:
    """Compute marginal effects for Wang (2002) model.

    This function calculates how covariates affect inefficiency through
    two channels:
    1. Location (μ): Affects the mean level of inefficiency
    2. Scale (σ_u): Affects the variance/dispersion of inefficiency

    Parameters:
        result: SFResult from Wang (2002) model
        method: Type of marginal effect
            'location' - ∂E[u_i] / ∂z_k (effect on mean inefficiency)
            'scale' - ∂σ_u,i / ∂w_k (effect on std dev of inefficiency)
            'efficiency' - ∂E[TE_i] / ∂z_k or ∂E[TE_i] / ∂w_k
        at_means: Evaluate at sample means (True) or compute average ME (False)

    Returns:
        DataFrame with marginal effects and standard errors

    References:
        Wang, H. J. (2002). "Heteroscedasticity and non-monotonic efficiency
            effects of a stochastic frontier model."
            Journal of Productivity Analysis, 18, 241-253.

    Example:
        >>> # Estimate Wang model
        >>> model = StochasticFrontier(
        ...     data=df,
        ...     depvar='log_output',
        ...     exog=['log_labor', 'log_capital'],
        ...     frontier='production',
        ...     dist='truncated_normal',
        ...     inefficiency_vars=['firm_age'],
        ...     het_vars=['firm_size']
        ... )
        >>> result = model.fit()
        >>>
        >>> # Marginal effects on location (mean inefficiency)
        >>> me_location = result.marginal_effects(method='location')
        >>> print(me_location)
        >>>
        >>> # Marginal effects on scale (variance of inefficiency)
        >>> me_scale = result.marginal_effects(method='scale')
        >>> print(me_scale)
    """
    model = result.model

    # Extract parameters
    params = result.params
    k = model.n_exog
    m = len(model.ineff_var_names)
    p = len(model.hetero_var_names)

    # Get parameter names
    param_names = result.params.index.tolist()

    # Find indices for delta and gamma
    # Structure: [β, σ²_v, delta_*, gamma_*]
    delta_names = [name for name in param_names if name.startswith("delta_")]
    gamma_names = [name for name in param_names if name.startswith("gamma_")]

    # Extract delta and gamma values
    delta = np.array([params[param_names.index(name)] for name in delta_names])
    gamma = np.array([params[param_names.index(name)] for name in gamma_names])

    # Get sigma_v_sq
    sigma_v_sq = params[param_names.index("sigma_v_sq")]

    Z = model.Z
    W = model.W

    if method == "location":
        # ∂E[u_i] / ∂z_k = δ_k
        # For truncated normal, E[u_i | μ_i, σ_u,i] ≈ μ_i for μ_i >> 0
        # The marginal effect of z_k on E[u_i] is approximately δ_k

        # Get standard errors from variance-covariance matrix
        if result.vcov is not None:
            delta_indices = [param_names.index(name) for name in delta_names]
            delta_se = np.sqrt(np.diag(result.vcov)[delta_indices])
        else:
            delta_se = np.full(len(delta), np.nan)

        me_df = pd.DataFrame(
            {
                "variable": model.ineff_var_names,
                "marginal_effect": delta,
                "std_error": delta_se,
            }
        )

        # Add t-statistic and p-value
        me_df["t_stat"] = me_df["marginal_effect"] / me_df["std_error"]
        from scipy.stats import t as t_dist

        me_df["p_value"] = 2 * (
            1 - t_dist.cdf(np.abs(me_df["t_stat"]), df=model.n_obs - len(params))
        )

        # Add confidence intervals
        t_critical = t_dist.ppf(0.975, df=model.n_obs - len(params))
        me_df["ci_lower"] = me_df["marginal_effect"] - t_critical * me_df["std_error"]
        me_df["ci_upper"] = me_df["marginal_effect"] + t_critical * me_df["std_error"]

        return me_df

    elif method == "scale":
        # ∂σ_u,i / ∂w_k = (σ_u,i / 2) · γ_k
        # Since ln(σ²_u,i) = w_i'γ, we have:
        # σ²_u,i = exp(w_i'γ)
        # σ_u,i = exp(0.5 * w_i'γ)
        # ∂σ_u,i / ∂w_k = 0.5 · exp(0.5 * w_i'γ) · γ_k = (σ_u,i / 2) · γ_k

        if at_means:
            # Evaluate at mean
            w_mean = W.mean(axis=0)
            ln_sigma_u_sq_mean = w_mean @ gamma
            sigma_u_mean = np.sqrt(np.exp(ln_sigma_u_sq_mean))

            # ME = (σ_u / 2) · γ
            me = (sigma_u_mean / 2) * gamma

        else:
            # Average ME across observations
            ln_sigma_u_sq_i = W @ gamma
            sigma_u_i = np.sqrt(np.exp(ln_sigma_u_sq_i))

            # ME_i = (σ_u,i / 2) · γ for each observation
            # Average over observations
            me = np.mean((sigma_u_i[:, np.newaxis] / 2) * gamma, axis=0)

        # Get standard errors using delta method
        if result.vcov is not None:
            gamma_indices = [param_names.index(name) for name in gamma_names]

            # For simplicity, use SE of gamma as approximation
            # Full delta method would require gradient of ME w.r.t. all parameters
            gamma_se = np.sqrt(np.diag(result.vcov)[gamma_indices])

            # ME = (σ_u / 2) · γ, so SE(ME) ≈ (σ_u / 2) · SE(γ)
            if at_means:
                me_se = (sigma_u_mean / 2) * gamma_se
            else:
                # Average sigma_u
                avg_sigma_u = np.mean(sigma_u_i)
                me_se = (avg_sigma_u / 2) * gamma_se
        else:
            me_se = np.full(len(gamma), np.nan)

        me_df = pd.DataFrame(
            {
                "variable": model.hetero_var_names,
                "marginal_effect": me,
                "std_error": me_se,
            }
        )

        # Add t-statistic and p-value
        me_df["t_stat"] = me_df["marginal_effect"] / me_df["std_error"]
        from scipy.stats import t as t_dist

        me_df["p_value"] = 2 * (
            1 - t_dist.cdf(np.abs(me_df["t_stat"]), df=model.n_obs - len(params))
        )

        # Add confidence intervals
        t_critical = t_dist.ppf(0.975, df=model.n_obs - len(params))
        me_df["ci_lower"] = me_df["marginal_effect"] - t_critical * me_df["std_error"]
        me_df["ci_upper"] = me_df["marginal_effect"] + t_critical * me_df["std_error"]

        return me_df

    elif method == "efficiency":
        # ∂E[TE_i] / ∂z_k or ∂E[TE_i] / ∂w_k
        # This is complex and requires numerical differentiation
        # For production frontier: TE = exp(-u)
        # E[TE | ε] = E[exp(-u) | ε]

        # For now, return message that this is not yet implemented
        raise NotImplementedError(
            "Marginal effects on efficiency require numerical integration. "
            "This feature will be implemented in a future version. "
            "For now, use method='location' or method='scale' to understand "
            "how covariates affect inefficiency."
        )

    else:
        raise ValueError(f"Unknown method: {method}. Use 'location', 'scale', or 'efficiency'.")


def compute_delta_method_se(
    marginal_effect: np.ndarray,
    vcov: np.ndarray,
    gradient: np.ndarray,
) -> np.ndarray:
    """Compute standard errors via delta method.

    The delta method approximates the variance of a function of random variables.
    For a function g(θ) where θ has variance-covariance matrix Var(θ), the
    variance of g(θ) is approximately:

        Var(g(θ)) ≈ ∇g(θ)' · Var(θ) · ∇g(θ)

    where ∇g(θ) is the gradient of g with respect to θ.

    Parameters:
        marginal_effect: Point estimate g(θ) (scalar or vector)
            Not used in calculation, included for consistency
        vcov: Variance-covariance matrix of θ (k x k)
        gradient: Gradient ∂g/∂θ (k,) or (m, k) for m effects

    Returns:
        Standard error(s) - scalar if gradient is 1D, array if 2D

    Example:
        >>> # SE for a linear combination c'β
        >>> c = np.array([1, 2, 3])
        >>> gradient = c  # ∂(c'β)/∂β = c
        >>> se = compute_delta_method_se(None, vcov, gradient)
    """
    if gradient.ndim == 1:
        # Single marginal effect
        # Var(g) = g' · Vcov · g
        se_sq = gradient @ vcov @ gradient.T
        se = np.sqrt(se_sq)
    else:
        # Multiple marginal effects (gradient is m x k matrix)
        # Each row is gradient for one effect
        # Var(g_i) = g_i' · Vcov · g_i
        se_sq = np.sum((gradient @ vcov) * gradient, axis=1)
        se = np.sqrt(se_sq)

    return se


def marginal_effects_bc95(
    result,
    method: str = "location",
    at_means: bool = True,
) -> pd.DataFrame:
    """Compute marginal effects for Battese-Coelli (1995) model.

    BC95 model only has location determinants (no heteroscedasticity in variance).

    Model: μ_i = z_i'δ, σ²_u constant across observations

    Parameters:
        result: SFResult from BC95 model
        method: Type of marginal effect
            'location' - ∂E[u_i] / ∂z_k
        at_means: Evaluate at sample means (not used for BC95)

    Returns:
        DataFrame with marginal effects and standard errors

    References:
        Battese, G. E., & Coelli, T. J. (1995).
            A model for technical inefficiency effects in a stochastic frontier
            production function for panel data.
            Empirical Economics, 20(2), 325-332.
    """
    model = result.model

    # Check that this is a BC95 model (has Z but not W)
    if model.W is not None:
        raise ValueError(
            "This appears to be a Wang (2002) model. Use marginal_effects_wang_2002 instead."
        )

    if model.Z is None:
        raise ValueError(
            "Model has no inefficiency determinants (Z). Cannot compute marginal effects."
        )

    # Extract parameters
    params = result.params
    param_names = result.params.index.tolist()

    # Find delta parameters
    delta_names = [name for name in param_names if name.startswith("delta_")]
    delta = np.array([params[param_names.index(name)] for name in delta_names])

    if method == "location":
        # ∂E[u_i] / ∂z_k = δ_k
        # Same as Wang (2002) location effect

        # Get standard errors
        if result.vcov is not None:
            delta_indices = [param_names.index(name) for name in delta_names]
            delta_se = np.sqrt(np.diag(result.vcov)[delta_indices])
        else:
            delta_se = np.full(len(delta), np.nan)

        me_df = pd.DataFrame(
            {
                "variable": model.ineff_var_names,
                "marginal_effect": delta,
                "std_error": delta_se,
            }
        )

        # Add t-statistic and p-value
        me_df["t_stat"] = me_df["marginal_effect"] / me_df["std_error"]
        from scipy.stats import t as t_dist

        me_df["p_value"] = 2 * (
            1 - t_dist.cdf(np.abs(me_df["t_stat"]), df=model.n_obs - len(params))
        )

        # Add confidence intervals
        t_critical = t_dist.ppf(0.975, df=model.n_obs - len(params))
        me_df["ci_lower"] = me_df["marginal_effect"] - t_critical * me_df["std_error"]
        me_df["ci_upper"] = me_df["marginal_effect"] + t_critical * me_df["std_error"]

        return me_df

    else:
        raise ValueError(f"Unknown method: {method}. For BC95, only 'location' is available.")
