"""
Utility functions for panel stochastic frontier models.

This module implements:
1. Starting values for panel models (BC92, BC95, Kumbhakar, Lee-Schmidt)
2. Marginal effects for BC95 inefficiency determinants
3. Likelihood ratio tests for nested models
4. Efficiency calculations for panel models
"""

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.special import log_ndtr, ndtr


def get_panel_starting_values(
    model_type: str,
    y: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    Z: Optional[np.ndarray] = None,
    sign: int = 1,
) -> np.ndarray:
    """Get starting values for panel SFA models.

    Parameters:
        model_type: One of 'pitt_lee', 'bc92', 'bc95', 'kumbhakar', 'lee_schmidt'
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        entity_id: Entity identifiers (n,)
        time_id: Time identifiers (n,)
        Z: Inefficiency determinants for BC95 (n, m)
        sign: Sign convention (+1 for production, -1 for cost)

    Returns:
        Starting parameter vector θ₀
    """
    k = X.shape[1]

    # Step 1: OLS for β and variance components
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta_ols

    # Compute moments for variance
    m2 = np.mean(residuals**2)
    m3 = np.mean(residuals**3)

    # Use half-normal moments
    sqrt_2_pi = np.sqrt(2 / np.pi)
    factor = sqrt_2_pi * (1 - 4 / np.pi)

    if abs(m3) < 1e-10:
        sigma_u_sq = 1e-4
        sigma_v_sq = max(m2, 1e-4)
    else:
        sigma_u = abs(m3 / (-factor)) ** (1 / 3)
        sigma_u_sq = sigma_u**2
        sigma_v_sq = max(m2 - (1 - 2 / np.pi) * sigma_u_sq, 1e-4)

    # Correct intercept bias
    if np.allclose(X[:, 0], 1.0):
        bias = np.sqrt(sigma_u_sq) * sqrt_2_pi
        beta_ols = beta_ols.copy()
        beta_ols[0] += sign * bias

    # Base parameters
    theta = np.concatenate([beta_ols, [np.log(sigma_v_sq)], [np.log(sigma_u_sq)]])

    # Model-specific parameters
    if model_type == "pitt_lee":
        # Pitt-Lee with truncated normal: add μ = 0
        theta = np.concatenate([theta, [0.0]])

    elif model_type == "bc92":
        # Battese-Coelli 1992: add μ = 0, η = 0
        theta = np.concatenate([theta, [0.0, 0.0]])

    elif model_type == "bc95":
        # Battese-Coelli 1995: add δ parameters
        if Z is None:
            raise ValueError("Z must be provided for BC95 model")

        m = Z.shape[1]

        # Use two-step approach for initial δ
        # (even though we warn against using it for final estimation)
        # Step 1: Get residuals from frontier
        u_hat = -sign * residuals  # Rough approximation
        u_hat = np.maximum(u_hat, 1e-6)  # Ensure positive

        # Step 2: Regress log(u_hat) on Z
        log_u_hat = np.log(u_hat)

        try:
            delta_init = np.linalg.lstsq(Z, log_u_hat, rcond=None)[0]
        except np.linalg.LinAlgError:
            delta_init = np.zeros(m)

        theta = np.concatenate([theta, delta_init])

    elif model_type == "kumbhakar":
        # Kumbhakar 1990: add μ = 0, b = 0, c = 0
        theta = np.concatenate([theta, [0.0, 0.0, 0.0]])

    elif model_type == "lee_schmidt":
        # Lee-Schmidt 1993: add μ = 0, δ_t = 1 for t = 1, ..., T-1
        T = time_id.max() + 1
        delta_t = np.ones(T - 1)
        theta = np.concatenate([theta, [0.0], delta_t])

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return theta


def bc95_marginal_effects(
    delta: np.ndarray, Z: np.ndarray, sigma_u: float
) -> Dict[str, np.ndarray]:
    """Compute marginal effects for BC95 inefficiency determinants.

    Marginal effect of z_j on E[u | Z]:
        ∂E[u | Z]/∂z_j = δ_j × [φ(μ/σ_u) / Φ(μ/σ_u)]

    where μ = Z'δ and φ, Φ are standard normal pdf/cdf.

    Parameters:
        delta: Coefficients of inefficiency determinants (m,)
        Z: Inefficiency determinants (n, m)
        sigma_u: Standard deviation of inefficiency

    Returns:
        Dictionary with:
            'marginal_effects': Average marginal effects (m,)
            'marginal_effects_i': Individual marginal effects (n, m)
            'mills_ratio': Inverse mills ratio at each observation (n,)
    """
    n, m = Z.shape

    # Compute μ = Z'δ
    mu = Z @ delta

    # Standardized μ
    mu_std = mu / sigma_u

    # Mills ratio: λ(μ/σ_u) = φ(μ/σ_u) / Φ(μ/σ_u)
    phi = stats.norm.pdf(mu_std)
    Phi = ndtr(mu_std)

    # Avoid division by zero
    Phi = np.maximum(Phi, 1e-10)
    mills_ratio = phi / Phi

    # Marginal effects: ∂E[u]/∂z_j = δ_j × λ(μ/σ_u)
    # Shape: (n, m) = (n, 1) * (1, m)
    marginal_effects_i = mills_ratio[:, np.newaxis] * delta[np.newaxis, :]

    # Average marginal effects
    marginal_effects = np.mean(marginal_effects_i, axis=0)

    return {
        "marginal_effects": marginal_effects,
        "marginal_effects_i": marginal_effects_i,
        "mills_ratio": mills_ratio,
    }


def likelihood_ratio_test(
    loglik_restricted: float, loglik_unrestricted: float, df: int
) -> Dict[str, float]:
    """Likelihood ratio test for nested models.

    H0: Restrictions are valid (restricted model is true)
    H1: Unrestricted model is true

    LR = 2 * (logL_unrestricted - logL_restricted) ~ χ²(df)

    Parameters:
        loglik_restricted: Log-likelihood of restricted model
        loglik_unrestricted: Log-likelihood of unrestricted model
        df: Degrees of freedom (number of restrictions)

    Returns:
        Dictionary with LR statistic, p-value, and df
    """
    LR = 2 * (loglik_unrestricted - loglik_restricted)

    # Ensure non-negative
    LR = max(LR, 0.0)

    # P-value from chi-squared distribution
    p_value = 1 - stats.chi2.cdf(LR, df)

    return {"LR_stat": LR, "p_value": p_value, "df": df, "reject_H0": p_value < 0.05}


def lr_test_bc92_eta_constant(loglik_bc92: float, loglik_pitt_lee: float) -> Dict[str, float]:
    """LR test for H0: η = 0 (efficiency is constant over time).

    Parameters:
        loglik_bc92: Log-likelihood of BC92 model with η estimated
        loglik_pitt_lee: Log-likelihood of Pitt-Lee model (η = 0)

    Returns:
        LR test results
    """
    return likelihood_ratio_test(
        loglik_restricted=loglik_pitt_lee,
        loglik_unrestricted=loglik_bc92,
        df=1,  # Testing 1 restriction: η = 0
    )


def lr_test_kumbhakar_constant(loglik_kumbhakar: float, loglik_pitt_lee: float) -> Dict[str, float]:
    """LR test for H0: b = c = 0 (efficiency is constant over time).

    Parameters:
        loglik_kumbhakar: Log-likelihood of Kumbhakar model
        loglik_pitt_lee: Log-likelihood of Pitt-Lee model

    Returns:
        LR test results
    """
    return likelihood_ratio_test(
        loglik_restricted=loglik_pitt_lee,
        loglik_unrestricted=loglik_kumbhakar,
        df=2,  # Testing 2 restrictions: b = 0, c = 0
    )


def compute_panel_efficiency_bc92(
    epsilon: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    sigma_v: float,
    sigma_u: float,
    mu: float,
    eta: float,
    sign: int = 1,
) -> np.ndarray:
    """Compute time-varying efficiency for BC92 model.

    For each (i, t):
        TE_{it} = E[exp(-u_{it}) | ε_{it}]

    where u_{it} = exp[-η(t - T)] * u_i

    Parameters:
        epsilon: Composed error (n,)
        entity_id: Entity identifiers (n,)
        time_id: Time identifiers (n,)
        sigma_v: Standard deviation of noise
        sigma_u: Standard deviation of base inefficiency
        mu: Mean of base inefficiency (truncated normal)
        eta: Time decay parameter
        sign: Sign convention

    Returns:
        Technical efficiency (n,)
    """
    n = len(epsilon)
    T_max = time_id.max()

    # Decay factors
    decay = np.exp(-eta * (time_id - T_max))

    # For truncated normal with time-varying component
    # This is an approximation using Jondrow et al. formula
    sigma_sq = sigma_v**2 + (decay * sigma_u) ** 2
    sigma = np.sqrt(sigma_sq)

    sigma_star_sq = (sigma_v**2 * (decay * sigma_u) ** 2) / sigma_sq
    sigma_star = np.sqrt(sigma_star_sq)

    mu_star = ((decay * sigma_u) ** 2 * (-sign * epsilon) + sigma_v**2 * mu) / sigma_sq

    # E[exp(-u) | ε]
    # Using formula: exp(-μ* + 0.5*σ*²) * Φ(μ*/σ* - σ*) / Φ(μ*/σ*)
    z = mu_star / sigma_star

    TE = np.exp(-mu_star + 0.5 * sigma_star_sq) * ndtr(z - sigma_star) / ndtr(z)

    # Clip to [0, 1]
    TE = np.clip(TE, 0, 1)

    return TE


def compute_panel_efficiency_bc95(
    epsilon: np.ndarray,
    Z: np.ndarray,
    delta: np.ndarray,
    sigma_v: float,
    sigma_u: float,
    sign: int = 1,
) -> np.ndarray:
    """Compute efficiency for BC95 model with inefficiency determinants.

    For each observation:
        TE_i = E[exp(-u_i) | ε_i, Z_i]

    where u_i ~ N⁺(Z_i'δ, σ²_u)

    Parameters:
        epsilon: Composed error (n,)
        Z: Inefficiency determinants (n, m)
        delta: Coefficients (m,)
        sigma_v: Standard deviation of noise
        sigma_u: Standard deviation of inefficiency
        sign: Sign convention

    Returns:
        Technical efficiency (n,)
    """
    # Compute μ_i = Z_i'δ
    mu_i = Z @ delta

    # Conditional distribution parameters
    sigma_sq = sigma_v**2 + sigma_u**2
    sigma_star_sq = (sigma_v**2 * sigma_u**2) / sigma_sq
    sigma_star = np.sqrt(sigma_star_sq)

    mu_star = (sigma_u**2 * (-sign * epsilon) + sigma_v**2 * mu_i) / sigma_sq

    # E[exp(-u) | ε, Z]
    z = mu_star / sigma_star

    # Avoid numerical issues
    z = np.clip(z, -10, 10)

    TE = np.exp(-mu_star + 0.5 * sigma_star_sq) * ndtr(z - sigma_star) / ndtr(z)

    # Clip to [0, 1]
    TE = np.clip(TE, 0, 1)

    return TE


def validate_panel_structure(
    entity_id: np.ndarray, time_id: np.ndarray, min_T: int = 3
) -> Dict[str, Any]:
    """Validate panel structure and provide warnings.

    Parameters:
        entity_id: Entity identifiers (n,)
        time_id: Time identifiers (n,)
        min_T: Minimum recommended time periods

    Returns:
        Dictionary with panel characteristics and warnings
    """
    unique_entities = np.unique(entity_id)
    unique_times = np.unique(time_id)

    N = len(unique_entities)
    T = len(unique_times)
    n = len(entity_id)

    # Check balance
    is_balanced = n == N * T

    # Count observations per entity
    obs_per_entity = np.array([np.sum(entity_id == i) for i in unique_entities])

    min_obs = obs_per_entity.min()
    max_obs = obs_per_entity.max()

    # Warnings
    warnings_list = []

    if T < min_T:
        warnings_list.append(
            f"T = {T} is less than recommended minimum of {min_T}. "
            "Time-varying parameters may be poorly identified."
        )

    if not is_balanced:
        warnings_list.append(
            f"Panel is unbalanced. N = {N}, T = {T}, n = {n}. "
            "Min obs per entity: {min_obs}, Max: {max_obs}."
        )

    if N < 20:
        warnings_list.append(f"N = {N} is small. Panel estimators work best with N ≥ 20.")

    return {
        "N": N,
        "T": T,
        "n": n,
        "is_balanced": is_balanced,
        "min_obs_per_entity": min_obs,
        "max_obs_per_entity": max_obs,
        "warnings": warnings_list,
    }
