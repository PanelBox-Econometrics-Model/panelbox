"""
True Fixed Effects and True Random Effects models (Greene 2005).

This module implements Greene's (2005) "true" panel stochastic frontier models
that separate heterogeneity from inefficiency, addressing the confounding issue
in classical panel SFA models.

Key innovations:
1. True Fixed Effects (TFE): Separates firm-specific heterogeneity (α_i) from
   time-varying inefficiency (u_it)
2. True Random Effects (TRE): Three-component error structure with heterogeneity
   (w_i), inefficiency (u_it), and noise (v_it)

References:
    Greene, W. H. (2005).
        Reconsidering heterogeneity in panel data estimators of the stochastic
        frontier model. Journal of Econometrics, 126(2), 269-303.

    Greene, W. H. (2005).
        Fixed and random effects in stochastic frontier models.
        Journal of Productivity Analysis, 23(1), 7-32.
"""

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.special import log_ndtr, ndtr, roots_hermite
from scipy.stats.qmc import Halton

# ============================================================================
# True Fixed Effects (TFE) Model - Greene (2005)
# ============================================================================


def loglik_true_fixed_effects(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    sign: int = 1,
    return_alpha: bool = False,
) -> float:
    """Log-likelihood for True Fixed Effects (TFE) model.

    Model: y_{it} = α_i + X_{it}β + v_{it} - sign*u_{it}

    where:
        α_i = entity-specific fixed effect (heterogeneity)
        u_{it} ~ N⁺(0, σ²_u) is time-varying inefficiency
        v_{it} ~ N(0, σ²_v) is noise

    Key difference from classical models:
        - Classical: confounds α_i with u_i (time-invariant)
        - TFE: separates α_i (heterogeneity) from u_{it} (inefficiency)

    Parameters:
        theta: Parameter vector [β, ln(σ²_v), ln(σ²_u), α_1, ..., α_N]
               OR [β, ln(σ²_v), ln(σ²_u)] if using concentrated likelihood
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        entity_id: Entity identifiers (n,) - coded 0, 1, ..., N-1
        time_id: Time identifiers (n,)
        sign: +1 for production, -1 for cost
        return_alpha: If True, return dict with alpha_i estimates

    Returns:
        Log-likelihood value (or dict if return_alpha=True)

    References:
        Greene, W. H. (2005). Journal of Econometrics, 126(2), 269-303.

    Notes:
        - This implementation uses the concentrated likelihood approach
        - α_i are profiled out for each candidate (β, σ²_v, σ²_u)
        - Reduces computational burden from N+k to just k parameters
    """
    n, k = X.shape

    # Extract parameters
    beta = theta[:k]
    ln_sigma_v_sq = theta[k]
    ln_sigma_u_sq = theta[k + 1]

    # Transform to natural scale
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)
    sigma_v = np.sqrt(sigma_v_sq)
    sigma_u = np.sqrt(sigma_u_sq)

    # Compute composite error variance
    sigma_sq = sigma_v_sq + sigma_u_sq
    sigma = np.sqrt(sigma_sq)
    sigma_star_sq = (sigma_v_sq * sigma_u_sq) / sigma_sq
    sigma_star = np.sqrt(sigma_star_sq)

    # Residuals before removing α_i
    residuals = y - X @ beta

    # Get unique entities
    unique_entities = np.unique(entity_id)
    N = len(unique_entities)

    loglik = 0.0
    alpha_estimates = {}

    # Loop over entities
    for i in unique_entities:
        # Get observations for entity i
        mask_i = entity_id == i
        resid_i = residuals[mask_i]
        T_i = len(resid_i)  # Number of periods for entity i

        # Concentrated likelihood: maximize over α_i analytically
        # For given (β, σ²_v, σ²_u), the optimal α_i can be found

        # The likelihood for entity i given α_i is:
        # L_i(α_i) = ∏_t f(ε_{it} - α_i | θ)
        # where f is the SFA composed error density

        # For half-normal inefficiency, we need to maximize:
        # sum_t ln[2/σ * φ((ε_{it} - α_i)/σ) * Φ(-sign*λ*(ε_{it} - α_i)/σ)]

        # This doesn't have a closed form, so we use numerical optimization
        # However, for computational efficiency, we use an approximation:
        # The MLE of α_i is approximately the mean of residuals,
        # adjusted for the SFA structure

        # Simple approximation: α̂_i ≈ median(residuals_i)
        # (more robust than mean to inefficiency)
        alpha_i_hat = np.median(resid_i)

        # Alternative: use numerical optimization for α_i
        # (more accurate but slower)
        from scipy.optimize import minimize_scalar

        def neg_loglik_alpha(alpha_i):
            epsilon_it = resid_i - alpha_i
            ll = 0.0

            for eps in epsilon_it:
                # SFA likelihood for each observation
                # ln f = constant - ln(σ) - (ε)²/(2σ²)
                #        + ln Φ(μ*/σ*) - ln Φ(0)
                mu_star = -sign * sigma_u_sq * eps / sigma_sq

                ll_t = (
                    -np.log(sigma)
                    - 0.5 * np.log(2 * np.pi)
                    - 0.5 * eps**2 / sigma_sq
                    + log_ndtr(mu_star / sigma_star)
                    + np.log(2)  # From half-normal: ln Φ(0) = ln(0.5)
                )

                ll += ll_t

            return -ll

        # Optimize α_i (bounded search around median)
        result = minimize_scalar(
            neg_loglik_alpha,
            bounds=(alpha_i_hat - 2 * resid_i.std(), alpha_i_hat + 2 * resid_i.std()),
            method="bounded",
        )

        alpha_i_mle = result.x
        alpha_estimates[i] = alpha_i_mle

        # Compute likelihood at optimal α_i
        epsilon_it = resid_i - alpha_i_mle
        mu_star = -sign * sigma_u_sq * epsilon_it / sigma_sq

        ll_i = np.sum(
            -np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * epsilon_it**2 / sigma_sq
            + log_ndtr(mu_star / sigma_star)
            + np.log(2)
        )

        if not np.isfinite(ll_i):
            return -np.inf

        loglik += ll_i

    if return_alpha:
        return {"loglik": loglik, "alpha": alpha_estimates}

    return loglik


def bias_correct_tfe_analytical(
    alpha_hat: np.ndarray, T: np.ndarray, sigma_v_sq: float, sigma_u_sq: float
) -> np.ndarray:
    """Analytical bias correction for TFE α_i estimates (Hahn & Newey 2004).

    The incidental parameters problem causes bias of order O(1/T) in α_i.

    Parameters:
        alpha_hat: Uncorrected α_i estimates (N,)
        T: Number of time periods per entity (N,) or scalar
        sigma_v_sq: Variance of noise
        sigma_u_sq: Variance of inefficiency

    Returns:
        Bias-corrected α_i estimates (N,)

    References:
        Hahn, J., & Newey, W. (2004).
            Jacknife and analytical bias reduction for nonlinear panel models.
            Econometrica, 72(4), 1295-1319.
    """
    # Simplified analytical correction
    # Full implementation would require derivatives of the likelihood

    # Approximate bias: E[α̂_i - α_i] ≈ -(σ²_u)/(T * (σ²_v + σ²_u))
    # This is a rough approximation from the literature

    if np.isscalar(T):
        T = np.full(len(alpha_hat), T)

    bias = -(sigma_u_sq) / (T * (sigma_v_sq + sigma_u_sq))
    alpha_corrected = alpha_hat - bias

    return alpha_corrected


def bias_correct_tfe_jackknife(
    y: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    theta: np.ndarray,
    sign: int = 1,
) -> Dict[str, np.ndarray]:
    """Jackknife bias correction for TFE model.

    Estimates model T times, each time leaving out one period,
    then combines estimates to reduce bias.

    Parameters:
        y: Dependent variable
        X: Exogenous variables
        entity_id: Entity identifiers
        time_id: Time identifiers
        theta: Parameter estimates [β, ln(σ²_v), ln(σ²_u)]
        sign: +1 for production, -1 for cost

    Returns:
        Dict with bias-corrected estimates

    References:
        Dhaene, G., & Jochmans, K. (2015).
            Split-panel jackknife estimation of fixed-effect models.
            The Review of Economic Studies, 82(3), 991-1030.
    """
    unique_times = np.unique(time_id)
    T = len(unique_times)
    unique_entities = np.unique(entity_id)
    N = len(unique_entities)

    # Storage for jackknife estimates
    alpha_jackknife = np.zeros((T, N))

    # Leave-one-out estimation
    for t_idx, t_out in enumerate(unique_times):
        # Create mask for all observations except time t_out
        mask = time_id != t_out

        # Estimate α_i on reduced sample
        result = loglik_true_fixed_effects(
            theta, y[mask], X[mask], entity_id[mask], time_id[mask], sign=sign, return_alpha=True
        )

        # Store estimates
        for i in unique_entities:
            alpha_jackknife[t_idx, i] = result["alpha"].get(i, 0)

    # Jackknife bias correction formula
    # α̂_bc = T*α̂_full - (T-1)*mean(α̂_(-t))
    result_full = loglik_true_fixed_effects(
        theta, y, X, entity_id, time_id, sign=sign, return_alpha=True
    )

    alpha_full = np.array([result_full["alpha"].get(i, 0) for i in unique_entities])
    alpha_mean_jackknife = alpha_jackknife.mean(axis=0)

    alpha_bc = T * alpha_full - (T - 1) * alpha_mean_jackknife

    return {
        "alpha_corrected": alpha_bc,
        "alpha_uncorrected": alpha_full,
        "bias_estimate": alpha_full - alpha_bc,
    }


# ============================================================================
# True Random Effects (TRE) Model - Greene (2005)
# ============================================================================


def loglik_true_random_effects(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    sign: int = 1,
    n_quadrature: int = 32,
    method: str = "gauss-hermite",
) -> float:
    """Log-likelihood for True Random Effects (TRE) model.

    Model: y_{it} = X_{it}β + w_i + v_{it} - sign*u_{it}

    where:
        w_i ~ N(0, σ²_w) is random heterogeneity (time-invariant)
        u_{it} ~ N⁺(0, σ²_u) is time-varying inefficiency
        v_{it} ~ N(0, σ²_v) is noise

    Three-component error structure:
        - w_i: captures permanent differences in technology/management
        - u_{it}: captures time-varying inefficiency
        - v_{it}: random shocks

    Parameters:
        theta: Parameter vector [β, ln(σ²_v), ln(σ²_u), ln(σ²_w)]
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        entity_id: Entity identifiers (n,)
        time_id: Time identifiers (n,)
        sign: +1 for production, -1 for cost
        n_quadrature: Number of quadrature points (default 32)
        method: 'gauss-hermite' or 'simulated'

    Returns:
        Log-likelihood value

    References:
        Greene, W. H. (2005). Journal of Econometrics, 126(2), 269-303.

    Notes:
        - Integrates over w_i using numerical quadrature
        - Higher n_quadrature = more accurate but slower
        - Recommended: n_quadrature ≥ 20 for three-component models
    """
    n, k = X.shape

    # Extract parameters
    beta = theta[:k]
    ln_sigma_v_sq = theta[k]
    ln_sigma_u_sq = theta[k + 1]
    ln_sigma_w_sq = theta[k + 2]

    # Transform to natural scale
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)
    sigma_w_sq = np.exp(ln_sigma_w_sq)
    sigma_v = np.sqrt(sigma_v_sq)
    sigma_u = np.sqrt(sigma_u_sq)
    sigma_w = np.sqrt(sigma_w_sq)

    # Residuals
    epsilon = y - X @ beta

    # Get unique entities
    unique_entities = np.unique(entity_id)

    if method == "gauss-hermite":
        # Gauss-Hermite quadrature integration
        loglik = _tre_gauss_hermite_integration(
            epsilon,
            entity_id,
            time_id,
            sigma_v_sq,
            sigma_u_sq,
            sigma_w_sq,
            sign,
            n_quadrature,
            unique_entities,
        )
    elif method == "simulated":
        # Simulated MLE with Halton sequences
        loglik = _tre_simulated_mle(
            epsilon,
            entity_id,
            time_id,
            sigma_v_sq,
            sigma_u_sq,
            sigma_w_sq,
            sign,
            n_quadrature,
            unique_entities,  # n_quadrature used as n_simulations
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return loglik


def _tre_gauss_hermite_integration(
    epsilon: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    sigma_v_sq: float,
    sigma_u_sq: float,
    sigma_w_sq: float,
    sign: int,
    n_quad_points: int,
    unique_entities: np.ndarray,
) -> float:
    """Gauss-Hermite quadrature integration for TRE model."""
    # Get quadrature nodes and weights
    nodes, weights = roots_hermite(n_quad_points)
    weights = weights / np.sqrt(np.pi)  # Adjust for our parameterization

    sigma_v = np.sqrt(sigma_v_sq)
    sigma_u = np.sqrt(sigma_u_sq)
    sigma_w = np.sqrt(sigma_w_sq)

    # Composite error variance
    sigma_sq = sigma_v_sq + sigma_u_sq
    sigma = np.sqrt(sigma_sq)
    sigma_star_sq = (sigma_v_sq * sigma_u_sq) / sigma_sq
    sigma_star = np.sqrt(sigma_star_sq)

    loglik = 0.0

    # Loop over entities
    for i in unique_entities:
        # Get observations for entity i
        mask_i = entity_id == i
        epsilon_i = epsilon[mask_i]
        T_i = len(epsilon_i)

        # Integrate over w_i using Gauss-Hermite quadrature
        # w_i = σ_w * sqrt(2) * node
        integral = 0.0

        for node, weight in zip(nodes, weights):
            # Transform node to w_i space
            w_k = sigma_w * np.sqrt(2) * node

            # Conditional likelihood given w_k
            # Each observation: y_{it} - X_{it}β - w_k = v_{it} - u_{it}
            epsilon_it_conditional = epsilon_i - w_k

            # SFA likelihood for each time period
            mu_star = -sign * sigma_u_sq * epsilon_it_conditional / sigma_sq

            ll_conditional = np.sum(
                -np.log(sigma)
                - 0.5 * np.log(2 * np.pi)
                - 0.5 * epsilon_it_conditional**2 / sigma_sq
                + log_ndtr(mu_star / sigma_star)
                + np.log(2)  # Half-normal normalization
            )

            # Add weighted contribution
            integral += weight * np.exp(ll_conditional)

        # Convert to log
        if integral > 0:
            loglik_i = np.log(integral)
        else:
            return -np.inf

        if not np.isfinite(loglik_i):
            return -np.inf

        loglik += loglik_i

    return loglik


def _tre_simulated_mle(
    epsilon: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    sigma_v_sq: float,
    sigma_u_sq: float,
    sigma_w_sq: float,
    sign: int,
    n_simulations: int,
    unique_entities: np.ndarray,
) -> float:
    """Simulated MLE using Halton sequences for TRE model."""
    # Create Halton sequence generator
    halton = Halton(d=1, seed=42)
    draws = halton.random(n_simulations)

    sigma_v = np.sqrt(sigma_v_sq)
    sigma_u = np.sqrt(sigma_u_sq)
    sigma_w = np.sqrt(sigma_w_sq)

    # Composite error variance
    sigma_sq = sigma_v_sq + sigma_u_sq
    sigma = np.sqrt(sigma_sq)
    sigma_star_sq = (sigma_v_sq * sigma_u_sq) / sigma_sq
    sigma_star = np.sqrt(sigma_star_sq)

    loglik = 0.0

    # Loop over entities
    for i in unique_entities:
        # Get observations for entity i
        mask_i = entity_id == i
        epsilon_i = epsilon[mask_i]
        T_i = len(epsilon_i)

        # Simulate w_i using Halton draws
        simulated_likelihood = 0.0

        for draw in draws:
            # Transform uniform draw to normal
            w_sim = sigma_w * stats.norm.ppf(draw)

            # Conditional likelihood given w_sim
            epsilon_it_conditional = epsilon_i - w_sim

            # SFA likelihood for each time period
            mu_star = -sign * sigma_u_sq * epsilon_it_conditional / sigma_sq

            ll_conditional = np.sum(
                -np.log(sigma)
                - 0.5 * np.log(2 * np.pi)
                - 0.5 * epsilon_it_conditional**2 / sigma_sq
                + log_ndtr(mu_star / sigma_star)
                + np.log(2)
            )

            # Add contribution
            simulated_likelihood += np.exp(ll_conditional)

        # Average over simulations
        simulated_likelihood /= n_simulations

        # Convert to log
        if simulated_likelihood > 0:
            loglik_i = np.log(simulated_likelihood)
        else:
            return -np.inf

        if not np.isfinite(loglik_i):
            return -np.inf

        loglik += loglik_i

    return loglik


def variance_decomposition_tre(
    sigma_v_sq: float, sigma_u_sq: float, sigma_w_sq: float
) -> Dict[str, float]:
    """Decompose variance into components for TRE model.

    Parameters:
        sigma_v_sq: Variance of noise
        sigma_u_sq: Variance of inefficiency
        sigma_w_sq: Variance of heterogeneity

    Returns:
        Dict with variance shares:
            - gamma_v: Share of noise variance
            - gamma_u: Share of inefficiency variance
            - gamma_w: Share of heterogeneity variance
            - sigma_total_sq: Total variance
    """
    sigma_total_sq = sigma_v_sq + sigma_u_sq + sigma_w_sq

    return {
        "gamma_v": sigma_v_sq / sigma_total_sq,
        "gamma_u": sigma_u_sq / sigma_total_sq,
        "gamma_w": sigma_w_sq / sigma_total_sq,
        "sigma_total_sq": sigma_total_sq,
        "sigma_v_sq": sigma_v_sq,
        "sigma_u_sq": sigma_u_sq,
        "sigma_w_sq": sigma_w_sq,
    }


# ============================================================================
# True Models with BC95 Inefficiency Determinants
# ============================================================================


def loglik_tfe_bc95(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    sign: int = 1,
) -> float:
    """Log-likelihood for TFE model with BC95 inefficiency effects.

    Model: y_{it} = α_i + X_{it}β + v_{it} - sign*u_{it}

    where:
        α_i = fixed effect (heterogeneity)
        u_{it} ~ N⁺(Z_{it}δ, σ²_u) is inefficiency with determinants
        v_{it} ~ N(0, σ²_v) is noise

    Parameters:
        theta: [β, ln(σ²_v), ln(σ²_u), δ]
        y: Dependent variable
        X: Frontier variables
        Z: Inefficiency determinants
        entity_id: Entity identifiers
        time_id: Time identifiers
        sign: +1 for production, -1 for cost

    Returns:
        Log-likelihood value
    """
    n, k = X.shape
    m = Z.shape[1]

    # Extract parameters
    beta = theta[:k]
    ln_sigma_v_sq = theta[k]
    ln_sigma_u_sq = theta[k + 1]
    delta = theta[k + 2 : k + 2 + m]

    # Transform to natural scale
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)
    sigma_v = np.sqrt(sigma_v_sq)
    sigma_u = np.sqrt(sigma_u_sq)

    # Compute μ_{it} = Z_{it}δ
    mu_it = Z @ delta

    # Composite variance
    sigma_sq = sigma_v_sq + sigma_u_sq
    sigma = np.sqrt(sigma_sq)
    sigma_star_sq = (sigma_v_sq * sigma_u_sq) / sigma_sq
    sigma_star = np.sqrt(sigma_star_sq)

    # Residuals before removing α_i
    residuals = y - X @ beta

    # Get unique entities
    unique_entities = np.unique(entity_id)

    loglik = 0.0

    # Loop over entities
    for i in unique_entities:
        # Get observations for entity i
        mask_i = entity_id == i
        resid_i = residuals[mask_i]
        mu_it_i = mu_it[mask_i]
        T_i = len(resid_i)

        # Optimize α_i for entity i
        from scipy.optimize import minimize_scalar

        def neg_loglik_alpha(alpha_i):
            epsilon_it = resid_i - alpha_i
            mu_star = (sigma_u_sq * (-sign * epsilon_it) + sigma_v_sq * mu_it_i) / sigma_sq

            ll = np.sum(
                -np.log(sigma)
                - 0.5 * np.log(2 * np.pi)
                - 0.5 * (epsilon_it + sign * mu_it_i) ** 2 / sigma_sq
                + log_ndtr(mu_star / sigma_star)
                - log_ndtr(mu_it_i / sigma_u)
            )

            return -ll

        # Optimize α_i
        alpha_i_hat = np.median(resid_i)
        result = minimize_scalar(
            neg_loglik_alpha,
            bounds=(alpha_i_hat - 2 * resid_i.std(), alpha_i_hat + 2 * resid_i.std()),
            method="bounded",
        )

        alpha_i_mle = result.x

        # Compute likelihood at optimal α_i
        epsilon_it = resid_i - alpha_i_mle
        mu_star = (sigma_u_sq * (-sign * epsilon_it) + sigma_v_sq * mu_it_i) / sigma_sq

        ll_i = np.sum(
            -np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * (epsilon_it + sign * mu_it_i) ** 2 / sigma_sq
            + log_ndtr(mu_star / sigma_star)
            - log_ndtr(mu_it_i / sigma_u)
        )

        if not np.isfinite(ll_i):
            return -np.inf

        loglik += ll_i

    return loglik


def loglik_tre_bc95(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    sign: int = 1,
    n_quadrature: int = 32,
) -> float:
    """Log-likelihood for TRE model with BC95 inefficiency effects.

    Model: y_{it} = X_{it}β + w_i + v_{it} - sign*u_{it}

    where:
        w_i ~ N(0, σ²_w) is random heterogeneity
        u_{it} ~ N⁺(Z_{it}δ, σ²_u) is inefficiency with determinants
        v_{it} ~ N(0, σ²_v) is noise

    Parameters:
        theta: [β, ln(σ²_v), ln(σ²_u), ln(σ²_w), δ]
        y: Dependent variable
        X: Frontier variables
        Z: Inefficiency determinants
        entity_id: Entity identifiers
        time_id: Time identifiers
        sign: +1 for production, -1 for cost
        n_quadrature: Number of quadrature points

    Returns:
        Log-likelihood value
    """
    n, k = X.shape
    m = Z.shape[1]

    # Extract parameters
    beta = theta[:k]
    ln_sigma_v_sq = theta[k]
    ln_sigma_u_sq = theta[k + 1]
    ln_sigma_w_sq = theta[k + 2]
    delta = theta[k + 3 : k + 3 + m]

    # Transform to natural scale
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)
    sigma_w_sq = np.exp(ln_sigma_w_sq)
    sigma_v = np.sqrt(sigma_v_sq)
    sigma_u = np.sqrt(sigma_u_sq)
    sigma_w = np.sqrt(sigma_w_sq)

    # Compute μ_{it} = Z_{it}δ
    mu_it = Z @ delta

    # Composite variance
    sigma_sq = sigma_v_sq + sigma_u_sq
    sigma = np.sqrt(sigma_sq)
    sigma_star_sq = (sigma_v_sq * sigma_u_sq) / sigma_sq
    sigma_star = np.sqrt(sigma_star_sq)

    # Residuals
    epsilon = y - X @ beta

    # Get quadrature nodes and weights
    nodes, weights = roots_hermite(n_quadrature)
    weights = weights / np.sqrt(np.pi)

    # Get unique entities
    unique_entities = np.unique(entity_id)

    loglik = 0.0

    # Loop over entities
    for i in unique_entities:
        # Get observations for entity i
        mask_i = entity_id == i
        epsilon_i = epsilon[mask_i]
        mu_it_i = mu_it[mask_i]
        T_i = len(epsilon_i)

        # Integrate over w_i
        integral = 0.0

        for node, weight in zip(nodes, weights):
            # Transform node to w_i space
            w_k = sigma_w * np.sqrt(2) * node

            # Conditional likelihood given w_k
            epsilon_it_conditional = epsilon_i - w_k

            # BC95 likelihood for each observation
            mu_star = (
                sigma_u_sq * (-sign * epsilon_it_conditional) + sigma_v_sq * mu_it_i
            ) / sigma_sq

            ll_conditional = np.sum(
                -np.log(sigma)
                - 0.5 * np.log(2 * np.pi)
                - 0.5 * (epsilon_it_conditional + sign * mu_it_i) ** 2 / sigma_sq
                + log_ndtr(mu_star / sigma_star)
                - log_ndtr(mu_it_i / sigma_u)
            )

            # Add weighted contribution
            integral += weight * np.exp(ll_conditional)

        # Convert to log
        if integral > 0:
            loglik_i = np.log(integral)
        else:
            return -np.inf

        if not np.isfinite(loglik_i):
            return -np.inf

        loglik += loglik_i

    return loglik
