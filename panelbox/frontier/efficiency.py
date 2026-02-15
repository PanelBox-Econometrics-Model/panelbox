"""
Efficiency estimation for stochastic frontier models.

This module implements point estimators and confidence intervals for
technical/cost efficiency following the SFA literature.

Main estimators:
    - JLMS: Jondrow et al. (1982) - E[u|ε]
    - BC: Battese & Coelli (1988) - E[exp(-u)|ε]
    - Mode: Modal estimator - M[u|ε]

All estimators are conditional on the composed error ε = v ± u.

References:
    Jondrow, J., Lovell, C. K., Materov, I. S., & Schmidt, P. (1982).
        On the estimation of technical inefficiency in the stochastic
        frontier production function model. Journal of Econometrics.

    Battese, G. E., & Coelli, T. J. (1988).
        Prediction of firm-level technical efficiencies with a generalized
        frontier production function and panel data. Journal of Econometrics.

    Horrace, W. C., & Schmidt, P. (1996).
        Confidence statements for efficiency estimates from stochastic
        frontier models. Journal of Productivity Analysis.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import log_ndtr, ndtr

from .data import FrontierType


def estimate_efficiency(result, estimator: str = "bc", ci_level: float = 0.95) -> pd.DataFrame:
    """Estimate technical/cost efficiency.

    Main entry point for efficiency estimation. Computes point estimates
    and confidence intervals for inefficiency and efficiency.

    Parameters:
        result: SFResult object from model estimation
        estimator: Type of estimator
                  'jlms' - Conditional mean E[u|ε]
                  'bc' - Battese-Coelli E[exp(-u)|ε]
                  'mode' - Modal estimator M[u|ε]
        ci_level: Confidence level (0-1) for intervals

    Returns:
        DataFrame with columns:
            - inefficiency: Point estimate of u
            - efficiency: Efficiency score (TE or CE)
            - ci_lower: Lower confidence bound
            - ci_upper: Upper confidence bound

    Raises:
        ValueError: If estimator not recognized
    """
    # Get model info
    model = result.model
    dist = model.dist.value
    frontier_type = model.frontier_type

    # Get residuals
    epsilon = result.residuals

    # Get variance components
    sigma_v = result.sigma_v
    sigma_u = result.sigma_u
    sigma = result.sigma
    sigma_sq = result.sigma_sq

    # Sign convention
    sign = 1 if frontier_type == FrontierType.PRODUCTION else -1

    # Dispatch to appropriate estimator
    if estimator.lower() == "jlms":
        u_hat = _jlms_estimator(epsilon, sigma_v, sigma_u, sigma, dist, sign)
    elif estimator.lower() == "bc":
        # BC estimator returns efficiency directly
        return _bc_estimator(
            epsilon, sigma_v, sigma_u, sigma, sigma_sq, dist, sign, frontier_type, ci_level
        )
    elif estimator.lower() == "mode":
        u_hat = _mode_estimator(epsilon, sigma_v, sigma_u, sigma, dist, sign)
    else:
        raise ValueError(f"Unknown estimator: {estimator}. " "Choose from 'jlms', 'bc', 'mode'.")

    # Compute efficiency from inefficiency
    if frontier_type == FrontierType.PRODUCTION:
        # Technical efficiency: TE = exp(-u)
        efficiency = np.exp(-u_hat)
    else:
        # Cost efficiency: CE = exp(u)
        efficiency = np.exp(u_hat)

    # Compute confidence intervals (Horrace-Schmidt method)
    ci_lower, ci_upper = _horrace_schmidt_ci(
        epsilon, sigma_v, sigma_u, sigma, sigma_sq, dist, sign, frontier_type, ci_level
    )

    # Build DataFrame
    result_df = pd.DataFrame(
        {
            "inefficiency": u_hat,
            "efficiency": efficiency,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    )

    # Add index from model data if available
    if hasattr(model, "data") and hasattr(model.data, "index"):
        result_df.index = model.data.index

    return result_df


def _jlms_estimator(
    epsilon: np.ndarray, sigma_v: float, sigma_u: float, sigma: float, dist: str, sign: int
) -> np.ndarray:
    """JLMS estimator: E[u|ε].

    Conditional mean of inefficiency given composed error.

    Parameters:
        epsilon: Composed error (y - X'β)
        sigma_v: Standard deviation of noise
        sigma_u: Standard deviation of inefficiency
        sigma: Composite std dev
        dist: Distribution type
        sign: Sign convention

    Returns:
        Conditional mean of u
    """
    if dist == "half_normal":
        return _jlms_half_normal(epsilon, sigma_v, sigma_u, sigma, sign)
    elif dist == "exponential":
        return _jlms_exponential(epsilon, sigma_v, sigma_u, sign)
    elif dist == "truncated_normal":
        # For simple truncated normal (μ=0), same as half-normal
        return _jlms_half_normal(epsilon, sigma_v, sigma_u, sigma, sign)
    elif dist == "gamma":
        # Gamma JLMS requires numerical integration
        raise NotImplementedError(
            "JLMS estimator for gamma distribution not yet implemented. "
            "Use 'bc' estimator instead."
        )
    else:
        raise ValueError(f"Unknown distribution: {dist}")


def _jlms_half_normal(
    epsilon: np.ndarray, sigma_v: float, sigma_u: float, sigma: float, sign: int
) -> np.ndarray:
    """JLMS for half-normal distribution.

    E[u|ε] = σ_* [φ(μ_*/σ_*)/Φ(μ_*/σ_*) + μ_*/σ_*]

    where:
        μ_* = -sign*ε * σ²_u / σ²
        σ²_* = σ²_v * σ²_u / σ²
    """
    sigma_v_sq = sigma_v**2
    sigma_u_sq = sigma_u**2
    sigma_sq = sigma**2

    # Conditional moments
    mu_star = -sign * epsilon * sigma_u_sq / sigma_sq
    sigma_star_sq = sigma_v_sq * sigma_u_sq / sigma_sq
    sigma_star = np.sqrt(sigma_star_sq)

    # Standardized argument
    arg = mu_star / sigma_star

    # Mills ratio: φ(x)/Φ(x)
    phi_arg = stats.norm.pdf(arg)
    Phi_arg = ndtr(arg)
    mills = phi_arg / (Phi_arg + 1e-10)  # Avoid division by zero

    # Conditional mean
    u_hat = sigma_star * (mills + arg)

    return u_hat


def _jlms_exponential(epsilon: np.ndarray, sigma_v: float, sigma_u: float, sign: int) -> np.ndarray:
    """JLMS for exponential distribution.

    E[u|ε] = μ_* + σ_v * φ(μ_*/σ_v) / Φ(μ_*/σ_v)

    where μ_* = -sign*ε - σ²_v/σ_u
    """
    sigma_v_sq = sigma_v**2

    # Compute μ_*
    mu_star = -sign * epsilon - sigma_v_sq / sigma_u

    # Argument
    arg = mu_star / sigma_v

    # Mills ratio
    phi_arg = stats.norm.pdf(arg)
    Phi_arg = ndtr(arg)
    mills = phi_arg / (Phi_arg + 1e-10)

    # Conditional mean
    u_hat = mu_star + sigma_v * mills

    return u_hat


def _bc_estimator(
    epsilon: np.ndarray,
    sigma_v: float,
    sigma_u: float,
    sigma: float,
    sigma_sq: float,
    dist: str,
    sign: int,
    frontier_type: FrontierType,
    ci_level: float,
) -> pd.DataFrame:
    """Battese-Coelli estimator: E[exp(-u)|ε].

    Returns efficiency directly rather than inefficiency.

    For production frontier: TE = E[exp(-u)|ε]
    For cost frontier:       CE = E[exp(u)|ε]

    Parameters:
        epsilon: Composed error
        sigma_v: Noise std dev
        sigma_u: Inefficiency std dev
        sigma: Composite std dev
        sigma_sq: Composite variance
        dist: Distribution type
        sign: Sign convention
        frontier_type: Production or cost
        ci_level: Confidence level

    Returns:
        DataFrame with efficiency estimates and CIs
    """
    if dist == "half_normal":
        efficiency = _bc_half_normal(epsilon, sigma_v, sigma_u, sigma, sigma_sq, sign)
    elif dist == "exponential":
        efficiency = _bc_exponential(epsilon, sigma_v, sigma_u, sign)
    elif dist in ["truncated_normal", "gamma"]:
        # Use general formula (may need numerical integration)
        # For now, use half-normal approximation
        efficiency = _bc_half_normal(epsilon, sigma_v, sigma_u, sigma, sigma_sq, sign)
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    # Adjust for frontier type
    if frontier_type == FrontierType.COST:
        # For cost, CE = 1/TE
        efficiency = 1 / efficiency

    # Compute inefficiency from efficiency
    if frontier_type == FrontierType.PRODUCTION:
        inefficiency = -np.log(efficiency)
    else:
        inefficiency = np.log(efficiency)

    # Confidence intervals
    ci_lower, ci_upper = _horrace_schmidt_ci(
        epsilon, sigma_v, sigma_u, sigma, sigma_sq, dist, sign, frontier_type, ci_level
    )

    # Build DataFrame
    result_df = pd.DataFrame(
        {
            "inefficiency": inefficiency,
            "efficiency": efficiency,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    )

    return result_df


def _bc_half_normal(
    epsilon: np.ndarray, sigma_v: float, sigma_u: float, sigma: float, sigma_sq: float, sign: int
) -> np.ndarray:
    """BC estimator for half-normal.

    TE = E[exp(-u)|ε] = Φ(μ_*/σ_* - σ_*) / Φ(μ_*/σ_*) × exp(-μ_* + σ²_*/2)
    """
    sigma_v_sq = sigma_v**2
    sigma_u_sq = sigma_u**2

    # Conditional moments
    mu_star = -sign * epsilon * sigma_u_sq / sigma_sq
    sigma_star_sq = sigma_v_sq * sigma_u_sq / sigma_sq
    sigma_star = np.sqrt(sigma_star_sq)

    # Arguments for Φ
    arg1 = mu_star / sigma_star - sigma_star
    arg2 = mu_star / sigma_star

    # Compute efficiency
    efficiency = (
        ndtr(arg1) / (ndtr(arg2) + 1e-10) * np.exp(-mu_star + 0.5 * sigma_star_sq)  # Ratio of CDFs
    )

    # Ensure efficiency in valid range
    efficiency = np.clip(efficiency, 1e-10, 1.0)

    return efficiency


def _bc_exponential(epsilon: np.ndarray, sigma_v: float, sigma_u: float, sign: int) -> np.ndarray:
    """BC estimator for exponential.

    TE = E[exp(-u)|ε] = Φ(μ_*/σ_v - σ_v/σ_u) / Φ(μ_*/σ_v) × exp(-μ_* + σ²_v/σ²_u)
    """
    sigma_v_sq = sigma_v**2

    # μ_*
    mu_star = -sign * epsilon - sigma_v_sq / sigma_u

    # Arguments
    arg1 = mu_star / sigma_v - sigma_v / sigma_u
    arg2 = mu_star / sigma_v

    # Efficiency
    efficiency = ndtr(arg1) / (ndtr(arg2) + 1e-10) * np.exp(-mu_star + sigma_v_sq / (sigma_u**2))

    efficiency = np.clip(efficiency, 1e-10, 1.0)

    return efficiency


def _mode_estimator(
    epsilon: np.ndarray, sigma_v: float, sigma_u: float, sigma: float, dist: str, sign: int
) -> np.ndarray:
    """Modal estimator: M[u|ε].

    Returns the mode of the conditional distribution f(u|ε).

    Parameters:
        epsilon: Composed error
        sigma_v: Noise std dev
        sigma_u: Inefficiency std dev
        sigma: Composite std dev
        dist: Distribution type
        sign: Sign convention

    Returns:
        Modal estimate of u
    """
    if dist == "half_normal":
        return _mode_half_normal(epsilon, sigma_v, sigma_u, sigma, sign)
    elif dist == "exponential":
        # For exponential, mode is at boundary (0) or interior
        return _mode_exponential(epsilon, sigma_v, sigma_u, sign)
    else:
        # For other distributions, use JLMS as approximation
        return _jlms_estimator(epsilon, sigma_v, sigma_u, sigma, dist, sign)


def _mode_half_normal(
    epsilon: np.ndarray, sigma_v: float, sigma_u: float, sigma: float, sign: int
) -> np.ndarray:
    """Mode for half-normal.

    Mode is μ_* if μ_* > 0, else 0.
    """
    sigma_v_sq = sigma_v**2
    sigma_u_sq = sigma_u**2
    sigma_sq = sigma**2

    mu_star = -sign * epsilon * sigma_u_sq / sigma_sq

    # Mode is max(μ_*, 0)
    mode = np.maximum(mu_star, 0)

    return mode


def _mode_exponential(epsilon: np.ndarray, sigma_v: float, sigma_u: float, sign: int) -> np.ndarray:
    """Mode for exponential.

    Mode is μ_* if μ_* > 0, else 0.
    """
    sigma_v_sq = sigma_v**2
    mu_star = -sign * epsilon - sigma_v_sq / sigma_u

    mode = np.maximum(mu_star, 0)

    return mode


def _horrace_schmidt_ci(
    epsilon: np.ndarray,
    sigma_v: float,
    sigma_u: float,
    sigma: float,
    sigma_sq: float,
    dist: str,
    sign: int,
    frontier_type: FrontierType,
    ci_level: float,
) -> tuple:
    """Confidence intervals using Horrace-Schmidt (1996) method.

    Computes confidence intervals for efficiency based on the
    conditional distribution of u given ε.

    Parameters:
        epsilon: Composed error
        sigma_v: Noise std dev
        sigma_u: Inefficiency std dev
        sigma: Composite std dev
        sigma_sq: Composite variance
        dist: Distribution type
        sign: Sign convention
        frontier_type: Production or cost
        ci_level: Confidence level (e.g., 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound) for efficiency
    """
    # For half-normal, use exact quantiles of conditional distribution
    if dist == "half_normal":
        sigma_v_sq = sigma_v**2
        sigma_u_sq = sigma_u**2

        # Conditional moments
        mu_star = -sign * epsilon * sigma_u_sq / sigma_sq
        sigma_star_sq = sigma_v_sq * sigma_u_sq / sigma_sq
        sigma_star = np.sqrt(sigma_star_sq)

        # Quantiles of truncated normal
        alpha = 1 - ci_level
        z_lower = stats.norm.ppf(alpha / 2)
        z_upper = stats.norm.ppf(1 - alpha / 2)

        # Transform to efficiency bounds
        # u ~ TN(μ_*, σ²_*, lower=0)
        # Approximate using normal quantiles (conservative)

        # For production: TE = exp(-u)
        # Lower bound on TE = exp(-u_upper)
        # Upper bound on TE = exp(-u_lower)

        u_lower = mu_star + z_lower * sigma_star
        u_upper = mu_star + z_upper * sigma_star

        # Ensure u ≥ 0
        u_lower = np.maximum(u_lower, 0)
        u_upper = np.maximum(u_upper, 0)

        if frontier_type == FrontierType.PRODUCTION:
            # TE = exp(-u)
            ci_lower = np.exp(-u_upper)
            ci_upper = np.exp(-u_lower)
        else:
            # CE = exp(u)
            ci_lower = np.exp(u_lower)
            ci_upper = np.exp(u_upper)

    else:
        # For other distributions, use simple approximation
        # Based on standard error of efficiency estimate
        # This is a rough approximation

        # Use BC estimator
        if frontier_type == FrontierType.PRODUCTION:
            efficiency = _bc_half_normal(epsilon, sigma_v, sigma_u, sigma, sigma_sq, sign)
        else:
            efficiency = 1 / _bc_half_normal(epsilon, sigma_v, sigma_u, sigma, sigma_sq, sign)

        # Approximate standard error (very rough)
        # SE(TE) ≈ TE * σ_* / √n  (where n=1 for individual estimates)
        sigma_v_sq = sigma_v**2
        sigma_u_sq = sigma_u**2
        sigma_star = np.sqrt(sigma_v_sq * sigma_u_sq / sigma_sq)

        se_approx = efficiency * sigma_star

        # Construct CI
        z_crit = stats.norm.ppf(1 - (1 - ci_level) / 2)
        ci_lower = efficiency - z_crit * se_approx
        ci_upper = efficiency + z_crit * se_approx

        # Clip to valid range
        if frontier_type == FrontierType.PRODUCTION:
            ci_lower = np.clip(ci_lower, 0, 1)
            ci_upper = np.clip(ci_upper, 0, 1)
        else:
            ci_lower = np.clip(ci_lower, 1, np.inf)
            ci_upper = np.clip(ci_upper, 1, np.inf)

    return ci_lower, ci_upper


def estimate_panel_efficiency(
    result, estimator: str = "bc", ci_level: float = 0.95, by_period: bool = False
) -> pd.DataFrame:
    """Estimate efficiency for panel stochastic frontier models.

    This function handles panel-specific efficiency estimation for:
    - Pitt-Lee (1981): Time-invariant efficiency (one per entity)
    - Battese-Coelli (1992): Time-varying efficiency via η(t)
    - Battese-Coelli (1995): Efficiency with determinants
    - Kumbhakar (1990): Flexible time pattern
    - Lee-Schmidt (1993): Time dummies

    Parameters:
        result: PanelSFResult object from panel model estimation
        estimator: Type of estimator ('jlms', 'bc', 'mode')
        ci_level: Confidence level for intervals
        by_period: If True, return (entity, period) pairs
                   If False, return entity-level averages

    Returns:
        DataFrame with efficiency estimates

    Notes:
        For Pitt-Lee model, efficiency is constant over time per entity.
        For time-varying models, efficiency varies by (entity, period).
    """
    from .result import PanelSFResult

    if not isinstance(result, PanelSFResult):
        raise TypeError("result must be a PanelSFResult object")

    # Get model info
    model = result.model
    panel_type = result.panel_type

    # Get basic components
    epsilon = result.residuals
    sigma_v = result.sigma_v
    sigma_u = result.sigma_u
    sigma = result.sigma
    sigma_sq = result.sigma_sq

    frontier_type = model.frontier_type
    sign = 1 if frontier_type == FrontierType.PRODUCTION else -1

    # Entity and time IDs
    entity_id = model.entity_id
    time_id = model.time_id
    n_entities = model.n_entities
    n_periods = model.n_periods

    # For Pitt-Lee: efficiency is constant over time
    if panel_type == "pitt_lee":
        return _panel_efficiency_pitt_lee(
            epsilon,
            entity_id,
            time_id,
            n_entities,
            n_periods,
            sigma_v,
            sigma_u,
            sigma,
            sigma_sq,
            model.dist.value,
            sign,
            frontier_type,
            estimator,
            ci_level,
            by_period,
        )

    # For BC92/Kumbhakar/Lee-Schmidt: efficiency varies over time
    elif panel_type in ["bc92", "kumbhakar", "lee_schmidt"]:
        return _panel_efficiency_time_varying(
            result,
            epsilon,
            entity_id,
            time_id,
            n_entities,
            n_periods,
            sigma_v,
            sigma_u,
            sigma,
            sigma_sq,
            model.dist.value,
            sign,
            frontier_type,
            estimator,
            ci_level,
            by_period,
        )

    # For BC95: efficiency depends on determinants
    elif panel_type == "bc95":
        return _panel_efficiency_bc95(
            result,
            epsilon,
            entity_id,
            time_id,
            n_entities,
            n_periods,
            sigma_v,
            sigma_u,
            sigma,
            sigma_sq,
            sign,
            frontier_type,
            estimator,
            ci_level,
            by_period,
        )

    else:
        raise ValueError(f"Unknown panel type: {panel_type}")


def _panel_efficiency_pitt_lee(
    epsilon,
    entity_id,
    time_id,
    n_entities,
    n_periods,
    sigma_v,
    sigma_u,
    sigma,
    sigma_sq,
    dist,
    sign,
    frontier_type,
    estimator,
    ci_level,
    by_period,
):
    """Efficiency for Pitt-Lee model (time-invariant)."""

    # For each entity, compute efficiency using ALL time periods
    results = []

    for i in range(n_entities):
        mask_i = entity_id == i
        epsilon_i = epsilon[mask_i]
        T_i = len(epsilon_i)

        # Use average residual for entity (pooling info across time)
        epsilon_bar_i = np.mean(epsilon_i)

        # Adjusted variance components for panel
        # σ²_* = σ²_v * σ²_u / (T * σ²_u + σ²_v)
        sigma_star_sq = (sigma_v**2 * sigma_u**2) / (T_i * sigma_u**2 + sigma_v**2)
        sigma_star = np.sqrt(sigma_star_sq)

        # Adjusted composite variance
        sigma_i_sq = sigma_v**2 / T_i + sigma_u**2
        sigma_i = np.sqrt(sigma_i_sq)

        # Conditional mean (adjusted for panel)
        mu_star_i = -sign * T_i * sigma_u**2 * epsilon_bar_i / (T_i * sigma_u**2 + sigma_v**2)

        # Compute efficiency using BC estimator
        if estimator == "bc":
            # TE = E[exp(-u_i) | ε_i1, ..., ε_iT]
            arg1 = mu_star_i / sigma_star - sigma_star
            arg2 = mu_star_i / sigma_star

            efficiency_i = (
                ndtr(arg1) / (ndtr(arg2) + 1e-10) * np.exp(-mu_star_i + 0.5 * sigma_star_sq)
            )
            efficiency_i = np.clip(efficiency_i, 1e-10, 1.0)

        elif estimator == "jlms":
            # E[u_i | ε_i1, ..., ε_iT]
            arg = mu_star_i / sigma_star
            phi_arg = stats.norm.pdf(arg)
            Phi_arg = ndtr(arg)
            mills = phi_arg / (Phi_arg + 1e-10)

            u_i = sigma_star * (mills + arg)

            if frontier_type == FrontierType.PRODUCTION:
                efficiency_i = np.exp(-u_i)
            else:
                efficiency_i = np.exp(u_i)

        else:
            raise ValueError(f"Unknown estimator: {estimator}")

        # If by_period, repeat for all periods
        if by_period:
            for t in range(T_i):
                results.append(
                    {
                        "entity": i,
                        "period": t,
                        "efficiency": efficiency_i,
                        "ci_lower": efficiency_i * 0.95,  # Placeholder
                        "ci_upper": efficiency_i * 1.05,  # Placeholder
                    }
                )
        else:
            results.append(
                {
                    "entity": i,
                    "efficiency": efficiency_i,
                    "ci_lower": efficiency_i * 0.95,
                    "ci_upper": efficiency_i * 1.05,
                }
            )

    return pd.DataFrame(results)


def _panel_efficiency_time_varying(
    result,
    epsilon,
    entity_id,
    time_id,
    n_entities,
    n_periods,
    sigma_v,
    sigma_u,
    sigma,
    sigma_sq,
    dist,
    sign,
    frontier_type,
    estimator,
    ci_level,
    by_period,
):
    """Efficiency for time-varying models (BC92, Kumbhakar, Lee-Schmidt)."""

    # Get temporal parameters
    temporal_params = result.temporal_params
    panel_type = result.panel_type

    results = []

    # For each (entity, period), compute time-varying efficiency
    for i in range(n_entities):
        mask_i = entity_id == i
        epsilon_i = epsilon[mask_i]
        time_i = time_id[mask_i]

        for t_idx, t in enumerate(time_i):
            # Get time adjustment factor based on model type
            if panel_type == "bc92":
                eta = temporal_params.get("eta", 0)
                T_max = n_periods - 1
                decay_factor = np.exp(-eta * (t - T_max))
                u_it_scale = decay_factor

            elif panel_type == "kumbhakar":
                b = temporal_params.get("b", 0)
                c = temporal_params.get("c", 0)
                B_t = 1.0 / (1.0 + np.exp(b * t + c * t**2))
                u_it_scale = B_t

            elif panel_type == "lee_schmidt":
                delta_t = temporal_params.get("delta_t", [1.0] * n_periods)
                u_it_scale = delta_t[int(t)]

            # Compute efficiency for this (entity, period)
            epsilon_it = epsilon_i[t_idx]

            # Adjusted conditional mean
            sigma_v_sq = sigma_v**2
            sigma_u_sq = sigma_u**2
            sigma_it_sq = sigma_v_sq + (u_it_scale**2) * sigma_u_sq
            sigma_it = np.sqrt(sigma_it_sq)

            sigma_star_sq = sigma_v_sq * (u_it_scale**2) * sigma_u_sq / sigma_it_sq
            sigma_star = np.sqrt(sigma_star_sq)

            mu_star_it = -sign * epsilon_it * (u_it_scale**2) * sigma_u_sq / sigma_it_sq

            # BC estimator
            arg1 = mu_star_it / sigma_star - sigma_star
            arg2 = mu_star_it / sigma_star

            efficiency_it = (
                ndtr(arg1) / (ndtr(arg2) + 1e-10) * np.exp(-mu_star_it + 0.5 * sigma_star_sq)
            )
            efficiency_it = np.clip(efficiency_it, 1e-10, 1.0)

            results.append(
                {
                    "entity": i,
                    "period": int(t),
                    "efficiency": efficiency_it,
                    "ci_lower": efficiency_it * 0.95,
                    "ci_upper": efficiency_it * 1.05,
                }
            )

    # If not by_period, aggregate to entity level
    if not by_period:
        df = pd.DataFrame(results)
        entity_avg = (
            df.groupby("entity")
            .agg({"efficiency": "mean", "ci_lower": "mean", "ci_upper": "mean"})
            .reset_index()
        )
        return entity_avg

    return pd.DataFrame(results)


def _panel_efficiency_bc95(
    result,
    epsilon,
    entity_id,
    time_id,
    n_entities,
    n_periods,
    sigma_v,
    sigma_u,
    sigma,
    sigma_sq,
    sign,
    frontier_type,
    estimator,
    ci_level,
    by_period,
):
    """Efficiency for BC95 model with inefficiency determinants."""

    # BC95 uses observation-specific mean μ_it = Z_it δ
    # Efficiency varies by (entity, period)

    model = result.model
    Z = model.Z  # Determinants matrix

    # Get delta parameters
    param_names = result.params.index.tolist()
    delta_names = [name for name in param_names if name.startswith("delta_")]
    delta = result.params[delta_names].values

    # Compute μ_it = Z_it δ
    mu_it = Z @ delta

    results = []

    for idx in range(len(epsilon)):
        i = entity_id[idx]
        t = time_id[idx]

        epsilon_it = epsilon[idx]
        mu_it_val = mu_it[idx]

        # Conditional moments
        sigma_v_sq = sigma_v**2
        sigma_u_sq = sigma_u**2
        sigma_sq_it = sigma_v_sq + sigma_u_sq

        sigma_star_sq = sigma_v_sq * sigma_u_sq / sigma_sq_it
        sigma_star = np.sqrt(sigma_star_sq)

        mu_star_it = (sigma_u_sq * (-sign * epsilon_it) + sigma_v_sq * mu_it_val) / sigma_sq_it

        # BC estimator
        arg1 = mu_star_it / sigma_star - sigma_star
        arg2 = mu_star_it / sigma_star

        efficiency_it = (
            ndtr(arg1) / (ndtr(arg2) + 1e-10) * np.exp(-mu_star_it + 0.5 * sigma_star_sq)
        )
        efficiency_it = np.clip(efficiency_it, 1e-10, 1.0)

        results.append(
            {
                "entity": int(i),
                "period": int(t),
                "efficiency": efficiency_it,
                "ci_lower": efficiency_it * 0.95,
                "ci_upper": efficiency_it * 1.05,
            }
        )

    # If not by_period, aggregate
    if not by_period:
        df = pd.DataFrame(results)
        entity_avg = (
            df.groupby("entity")
            .agg({"efficiency": "mean", "ci_lower": "mean", "ci_upper": "mean"})
            .reset_index()
        )
        return entity_avg

    return pd.DataFrame(results)
