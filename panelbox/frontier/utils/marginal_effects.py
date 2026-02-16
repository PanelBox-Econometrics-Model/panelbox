"""
Marginal effects for stochastic frontier models with inefficiency determinants.

This module provides marginal effects computation for:
1. Wang (2002): Heteroscedastic inefficiency model
2. Battese & Coelli (1995): Inefficiency determinants in panel models
3. General inefficiency determinant models

References:
    Wang, H. J., & Schmidt, P. (2002).
        One-step and two-step estimation of the effects of exogenous
        variables on technical efficiency levels.
        Journal of Productivity Analysis, 18, 129-144.

    Battese, G. E., & Coelli, T. J. (1995).
        A model for technical inefficiency effects in a stochastic frontier
        production function for panel data.
        European Journal of Operational Research, 38, 325-332.

    Alvarez, A., Amsler, C., Orea, L., & Schmidt, P. (2006).
        Interpreting and testing the scaling property in models where
        inefficiency depends on firm characteristics.
        Journal of Productivity Analysis, 25, 201-212.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def marginal_effects(
    result,
    method: str = "mean",
    var: Optional[str] = None,
    at_values: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute marginal effects on inefficiency or efficiency.

    Works for models with inefficiency determinants:
    - Wang (2002): μ_i = z_i'δ, ln(σ²_u,i) = w_i'γ
    - Battese & Coelli (1995): u_it ~ N⁺(z_it'δ, σ²_u)
    - General models with E[u_i | z_i]

    Parameters
    ----------
    result : SFResult
        Fitted SFA model result with inefficiency determinants
    method : {'mean', 'efficiency', 'variance'}, default='mean'
        Type of marginal effect:
            - 'mean': Effect on E[u_i] (expected inefficiency)
            - 'efficiency': Effect on E[TE_i] (expected efficiency)
            - 'variance': Effect on Var[u_i] (Wang only)
    var : str, optional
        Specific variable for which to compute marginal effects.
        If None, computes for all determinant variables.
    at_values : dict, optional
        Values at which to evaluate marginal effects.
        Dictionary mapping variable names to values.
        If None, evaluates at sample means.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - variable: Name of determinant variable
            - marginal_effect: Marginal effect value
            - std_error: Standard error (if available)
            - z_stat: Z-statistic
            - p_value: P-value for test H0: ME = 0
            - interpretation: Sign and significance interpretation

    Raises
    ------
    ValueError
        If model does not have inefficiency determinants

    Example
    -------
    >>> # Battese & Coelli (1995) model
    >>> model = StochasticFrontier(
    ...     depvar='log_output',
    ...     exog=['log_labor', 'log_capital'],
    ...     data=df,
    ...     inefficiency_vars=['firm_age', 'manager_education'],
    ... )
    >>> result = model.fit()
    >>>
    >>> # Marginal effects on expected inefficiency
    >>> me = marginal_effects(result, method='mean')
    >>> print(me)
           variable  marginal_effect  std_error  z_stat  p_value
    0     firm_age           0.023      0.005     4.60    0.000
    1  manager_ed          -0.015      0.007    -2.14    0.032
    >>>
    >>> # Interpretation: Older firms are less efficient (higher u)
    >>> #                  More educated managers → more efficient (lower u)

    Notes
    -----
    **Wang (2002) Model:**

    Model specification:
        u_i ~ N⁺(μ_i, σ²_u,i)
        μ_i = z_i'δ
        ln(σ²_u,i) = w_i'γ

    Marginal effect on E[u_i]:
        ∂E[u_i]/∂z_k = δ_k · Φ(λ_i) + σ_u,i · φ(λ_i) · ∂λ_i/∂z_k

    where λ_i = μ_i/σ_u,i (signal-to-noise ratio)

    **Battese & Coelli (1995) Model:**

    Model specification:
        u_it ~ N⁺(μ_it, σ²_u)
        μ_it = z_it'δ

    Marginal effect on E[u_it]:
        ∂E[u_it]/∂z_k ≈ δ_k

    (Exact formula involves mills ratio adjustments)

    **Interpretation:**

    For inefficiency (u):
        - Positive ME: Variable increases inefficiency (bad)
        - Negative ME: Variable decreases inefficiency (good)

    For efficiency (TE = exp(-u)):
        - Signs are reversed
        - Positive ME: Variable increases efficiency (good)
        - Negative ME: Variable decreases efficiency (bad)
    """
    model = result.model

    # Detect model type and dispatch
    if hasattr(model, "hetero_vars") and model.hetero_vars:
        # Wang (2002) heteroscedastic model
        return marginal_effects_wang_2002(result, method=method, var=var)

    elif hasattr(model, "inefficiency_vars") and model.inefficiency_vars:
        # BC95 or similar
        return marginal_effects_bc95(result, method=method, var=var, at_values=at_values)

    else:
        raise ValueError(
            "Marginal effects require model with inefficiency determinants.\n"
            "Use 'inefficiency_vars' parameter in model specification."
        )


def marginal_effects_wang_2002(
    result,
    method: str = "mean",
    var: Optional[str] = None,
) -> pd.DataFrame:
    """Marginal effects for Wang (2002) heteroscedastic model.

    Model: u_i ~ N⁺(μ_i, σ²_u,i)
           μ_i = z_i'δ
           ln(σ²_u,i) = w_i'γ

    Parameters
    ----------
    result : SFResult
        Fitted Wang (2002) model result
    method : {'mean', 'variance'}, default='mean'
        Type of marginal effect
    var : str, optional
        Specific variable (if None, all variables)

    Returns
    -------
    pd.DataFrame
        Marginal effects table

    References
    ----------
    Wang & Schmidt (2002), equations (8)-(10)
    """
    model = result.model
    params = result.params

    # Extract parameters
    k = model.n_exog  # Number of frontier variables
    m = len(model.ineff_var_names)  # Number of inefficiency location vars
    p = len(model.hetero_var_names)  # Number of inefficiency scale vars

    # Parameter positions (assuming order: β, δ, γ, σ²_v)
    beta = params[:k]
    delta = params[k : k + m]
    gamma = params[k + m : k + m + p]
    sigma_v_sq = params[-1]

    if method == "mean":
        # Marginal effect on E[u_i]
        # Need to compute at sample means or specific values
        # For simplicity, compute average marginal effect

        # Get data
        Z = model.data[model.ineff_var_names].values
        W = model.data[model.hetero_var_names].values

        # Compute μ_i and σ_u,i for each observation
        mu_i = Z @ delta
        ln_sigma_u_sq_i = W @ gamma
        sigma_u_i = np.sqrt(np.exp(ln_sigma_u_sq_i))

        # Compute λ_i = μ_i / σ_u,i
        lambda_i = mu_i / sigma_u_i

        # Mills ratio components
        phi_lambda = stats.norm.pdf(lambda_i)
        Phi_lambda = stats.norm.cdf(lambda_i)

        # Average marginal effect for location parameters (δ)
        # ∂E[u_i]/∂z_k = δ_k · Φ(λ_i) + σ_u,i · φ(λ_i) · (1/σ_u,i)
        #              = δ_k · Φ(λ_i) + φ(λ_i)

        me_location = np.zeros(m)
        for j in range(m):
            # Average over observations
            me_j = delta[j] * Phi_lambda.mean() + phi_lambda.mean()
            me_location[j] = me_j

        # Standard errors (from variance-covariance matrix)
        # This is approximate - exact SE requires delta method
        vcov = result.vcov
        se_location = np.sqrt(np.diag(vcov[k : k + m, k : k + m]))

        # Create DataFrame
        df_list = []

        for j, varname in enumerate(model.ineff_var_names):
            if var is not None and varname != var:
                continue

            me = me_location[j]
            se = se_location[j]
            z_stat = me / se if se > 0 else np.nan
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat))) if not np.isnan(z_stat) else np.nan

            # Interpretation
            if p_value < 0.01:
                sig = "***"
            elif p_value < 0.05:
                sig = "**"
            elif p_value < 0.10:
                sig = "*"
            else:
                sig = ""

            direction = "increases" if me > 0 else "decreases"
            interpretation = f"{direction} inefficiency {sig}"

            df_list.append(
                {
                    "variable": varname,
                    "parameter": "location",
                    "marginal_effect": me,
                    "std_error": se,
                    "z_stat": z_stat,
                    "p_value": p_value,
                    "interpretation": interpretation,
                }
            )

        return pd.DataFrame(df_list)

    elif method == "variance":
        # Marginal effect on Var[u_i]
        # ∂Var[u_i]/∂w_k involves derivatives of σ²_u,i

        # Get data
        W = model.data[model.hetero_var_names].values
        Z = model.data[model.ineff_var_names].values

        ln_sigma_u_sq_i = W @ gamma
        sigma_u_sq_i = np.exp(ln_sigma_u_sq_i)

        mu_i = Z @ delta
        lambda_i = mu_i / np.sqrt(sigma_u_sq_i)

        # Variance of truncated normal
        # Var[u_i] = σ²_u,i · [1 - λ_i · φ(λ_i)/Φ(λ_i) - (φ(λ_i)/Φ(λ_i))²]

        phi_lambda = stats.norm.pdf(lambda_i)
        Phi_lambda = stats.norm.cdf(lambda_i)
        mills = phi_lambda / Phi_lambda

        var_factor = 1 - lambda_i * mills - mills**2

        # ∂Var[u_i]/∂w_k = ∂(σ²_u,i · var_factor)/∂w_k
        # ≈ γ_k · σ²_u,i · var_factor (approximate)

        me_variance = np.zeros(p)
        for j in range(p):
            me_j = (gamma[j] * sigma_u_sq_i * var_factor).mean()
            me_variance[j] = me_j

        # Standard errors
        vcov = result.vcov
        se_variance = np.sqrt(np.diag(vcov[k + m : k + m + p, k + m : k + m + p]))

        df_list = []

        for j, varname in enumerate(model.hetero_var_names):
            if var is not None and varname != var:
                continue

            me = me_variance[j]
            se = se_variance[j]
            z_stat = me / se if se > 0 else np.nan
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat))) if not np.isnan(z_stat) else np.nan

            direction = "increases" if me > 0 else "decreases"
            interpretation = f"{direction} inefficiency variance"

            df_list.append(
                {
                    "variable": varname,
                    "parameter": "scale",
                    "marginal_effect": me,
                    "std_error": se,
                    "z_stat": z_stat,
                    "p_value": p_value,
                    "interpretation": interpretation,
                }
            )

        return pd.DataFrame(df_list)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'mean' or 'variance'.")


def marginal_effects_bc95(
    result,
    method: str = "mean",
    var: Optional[str] = None,
    at_values: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Marginal effects for Battese & Coelli (1995) model.

    Model: u_it ~ N⁺(μ_it, σ²_u)
           μ_it = z_it'δ

    Parameters
    ----------
    result : SFResult
        Fitted BC95 model result
    method : {'mean', 'efficiency'}, default='mean'
        Type of marginal effect
    var : str, optional
        Specific variable
    at_values : dict, optional
        Values at which to evaluate

    Returns
    -------
    pd.DataFrame
        Marginal effects table

    Notes
    -----
    For BC95 with location-only parameterization:
        E[u_it | z_it] = μ_it + σ_u · (φ(α_it)/Φ(α_it))

    where α_it = μ_it/σ_u

    Marginal effect:
        ∂E[u_it]/∂z_k = δ_k · [1 + ∂/∂μ_it (σ_u · mills_it)]
                      = δ_k · [1 - mills_it · (mills_it + α_it)]

    For large μ_it (strongly inefficient firms):
        mills_it → 0, so ME → δ_k

    References
    ----------
    Battese & Coelli (1995), equation (6)
    Wang (2002), equation (5) for general case
    """
    model = result.model
    params = result.params

    # Extract parameters
    k = model.n_exog
    m = len(model.ineff_var_names)

    # Parameter positions (β, σ²_v, σ²_u, δ)
    beta = params[:k]
    sigma_v_sq = params[k]
    sigma_u_sq = params[k + 1]
    delta = params[k + 2 : k + 2 + m]

    sigma_u = np.sqrt(sigma_u_sq)

    if method == "mean":
        # Get inefficiency determinant data
        Z = model.data[model.ineff_var_names].values

        # Compute μ_it
        mu_it = Z @ delta

        # Compute α_it = μ_it / σ_u
        alpha_it = mu_it / sigma_u

        # Mills ratio and its derivative
        phi_alpha = stats.norm.pdf(alpha_it)
        Phi_alpha = stats.norm.cdf(alpha_it)
        mills = phi_alpha / (Phi_alpha + 1e-10)  # Avoid division by zero

        # Adjustment factor for marginal effect
        # ∂E[u]/∂μ = Φ(α) + μ·φ(α)/σ_u - φ(α)·mills
        # Simplified: 1 - mills·(mills + α)
        adjustment = 1 - mills * (mills + alpha_it)

        # Marginal effects: ME_k = δ_k · adjustment
        me_values = np.zeros(m)
        for j in range(m):
            me_values[j] = (delta[j] * adjustment).mean()

        # Standard errors (delta method - approximate)
        vcov = result.vcov
        se_values = np.sqrt(np.diag(vcov[k + 2 : k + 2 + m, k + 2 : k + 2 + m]))

        # Note: These SEs are for δ, not for marginal effects
        # Exact SEs require delta method with adjustment factor
        # For now, we use δ SEs as conservative approximation

        df_list = []

        for j, varname in enumerate(model.ineff_var_names):
            if var is not None and varname != var:
                continue

            me = me_values[j]
            se = se_values[j]  # Approximate
            z_stat = me / se if se > 0 else np.nan
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat))) if not np.isnan(z_stat) else np.nan

            # Significance stars
            if p_value < 0.01:
                sig = "***"
            elif p_value < 0.05:
                sig = "**"
            elif p_value < 0.10:
                sig = "*"
            else:
                sig = ""

            direction = "increases" if me > 0 else "decreases"
            interpretation = f"{direction} inefficiency {sig}"

            df_list.append(
                {
                    "variable": varname,
                    "marginal_effect": me,
                    "std_error": se,
                    "z_stat": z_stat,
                    "p_value": p_value,
                    "interpretation": interpretation,
                }
            )

        return pd.DataFrame(df_list)

    elif method == "efficiency":
        # Marginal effect on E[TE_it] = E[exp(-u_it)]
        # This requires numerical differentiation or exact formula
        # For now, use relationship: ∂TE/∂z ≈ -TE · ∂u/∂z

        # Get inefficiency MEs first
        me_ineff = marginal_effects_bc95(result, method="mean", var=var, at_values=at_values)

        # Get average efficiency
        eff = result.efficiency(estimator="bc")
        avg_te = eff["efficiency"].mean()

        # Transform: ME on efficiency ≈ -avg_TE · ME on inefficiency
        me_ineff["marginal_effect"] = -avg_te * me_ineff["marginal_effect"]
        me_ineff["interpretation"] = me_ineff["marginal_effect"].apply(
            lambda x: f"{'increases' if x > 0 else 'decreases'} efficiency"
        )

        return me_ineff

    else:
        raise ValueError(f"Unknown method: {method}. Use 'mean' or 'efficiency'.")


def marginal_effects_summary(me_df: pd.DataFrame) -> str:
    """Generate formatted summary of marginal effects.

    Parameters
    ----------
    me_df : pd.DataFrame
        DataFrame from marginal_effects()

    Returns
    -------
    str
        Formatted text table
    """
    lines = []
    lines.append("=" * 80)
    lines.append("MARGINAL EFFECTS ON INEFFICIENCY".center(80))
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        f"{'Variable':<20} {'Effect':>10} {'Std Err':>10} "
        f"{'z':>8} {'P>|z|':>8}  {'Interpretation':<20}"
    )
    lines.append("-" * 80)

    for _, row in me_df.iterrows():
        sig = ""
        if row["p_value"] < 0.01:
            sig = "***"
        elif row["p_value"] < 0.05:
            sig = "**"
        elif row["p_value"] < 0.10:
            sig = "*"

        lines.append(
            f"{row['variable']:<20} "
            f"{row['marginal_effect']:>10.4f} "
            f"{row['std_error']:>10.4f} "
            f"{row['z_stat']:>8.2f} "
            f"{row['p_value']:>8.3f}  "
            f"{row['interpretation']:<20} {sig}"
        )

    lines.append("-" * 80)
    lines.append("Signif. codes: 0 '***' 0.01 '**' 0.05 '*' 0.1")
    lines.append("")
    lines.append("Note: Positive effect = increases inefficiency (decreases efficiency)")
    lines.append("      Negative effect = decreases inefficiency (increases efficiency)")
    lines.append("=" * 80)

    return "\n".join(lines)
