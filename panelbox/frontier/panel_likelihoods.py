"""
Log-likelihood functions for panel stochastic frontier models.

This module implements log-likelihood functions for panel data SFA models,
including time-invariant and time-varying inefficiency specifications.

Panel models exploit the time dimension to:
1. Obtain more precise efficiency estimates (Pitt & Lee 1981)
2. Model time-varying inefficiency (Battese & Coelli 1992)
3. Include determinants of inefficiency (Battese & Coelli 1995)

All functions use the parameterization:
    θ = [β, ln(σ²_v), ln(σ²_u), ...]

References:
    Pitt, M. M., & Lee, L. F. (1981).
        The measurement and sources of technical inefficiency in Indonesian
        weaving industry. Journal of Development Economics, 9(1), 43-64.

    Battese, G. E., & Coelli, T. J. (1992).
        Frontier production functions, technical efficiency and panel data:
        with application to paddy farmers in India.
        Journal of Productivity Analysis, 3(1-2), 153-169.

    Battese, G. E., & Coelli, T. J. (1995).
        A model for technical inefficiency effects in a stochastic frontier
        production function for panel data. Empirical Economics, 20(2), 325-332.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.special import log_ndtr, ndtr

# Constants for numerical stability
SQRT_2PI = np.sqrt(2 * np.pi)
LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def loglik_pitt_lee_half_normal(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    sign: int = 1,
) -> float:
    """Log-likelihood for Pitt & Lee (1981) panel model with half-normal.

    Model: y_{it} = X_{it}β + v_{it} - sign*u_i

    where:
        v_{it} ~ N(0, σ²_v) is noise
        u_i ~ N⁺(0, σ²_u) is time-invariant inefficiency

    The likelihood integrates over u_i for each entity:
        L_i = ∫ ∏_t f(y_{it} | X_{it}, u_i) g(u_i) du_i

    For half-normal distribution, this has a closed-form solution.

    Parameters:
        theta: Parameter vector [β, ln(σ²_v), ln(σ²_u)]
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        entity_id: Entity identifiers (n,) - integer coded 0, 1, ..., N-1
        time_id: Time identifiers (n,) - integer coded 0, 1, ..., T-1
        sign: Sign convention (+1 for production, -1 for cost)

    Returns:
        Log-likelihood value (scalar)

    References:
        Pitt, M. M., & Lee, L. F. (1981).
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

    # Residuals
    epsilon = y - X @ beta

    # Get unique entities and their observation counts
    unique_entities = np.unique(entity_id)
    N = len(unique_entities)

    loglik = 0.0

    # Loop over entities
    for i in unique_entities:
        # Get observations for entity i
        mask_i = entity_id == i
        epsilon_i = epsilon[mask_i]
        T_i = len(epsilon_i)  # Number of time periods for entity i

        # Compute sufficient statistics for entity i
        # For half-normal, closed form exists

        # Sigma for composed error over time
        sigma_star_sq = sigma_u_sq * sigma_v_sq / (T_i * sigma_u_sq + sigma_v_sq)
        sigma_star = np.sqrt(sigma_star_sq)

        # Overall variance
        sigma_sq_i = sigma_v_sq + sigma_u_sq
        sigma_i = np.sqrt(sigma_sq_i)

        # Mean of epsilon_i
        epsilon_bar_i = np.mean(epsilon_i)

        # Conditional mean of u_i given epsilon_i
        mu_star = -sign * T_i * sigma_u_sq * epsilon_bar_i / (T_i * sigma_u_sq + sigma_v_sq)

        # Log-likelihood for entity i
        # Part 1: Normal density for epsilon_i given u_i=0
        ll_normal = (
            -0.5 * T_i * np.log(2 * np.pi)
            - 0.5 * T_i * np.log(sigma_v_sq)
            - np.sum(epsilon_i**2) / (2 * sigma_v_sq)
        )

        # Part 2: Adjustment for u_i distribution
        # ln ∫ exp(terms involving u_i) g(u_i) du_i
        ll_adjustment = (
            np.log(2)  # From half-normal
            + T_i * sigma_u_sq * epsilon_bar_i**2 / (2 * (T_i * sigma_u_sq + sigma_v_sq))
            + log_ndtr(mu_star / sigma_star)
            - 0.5 * np.log(T_i * sigma_u_sq + sigma_v_sq)
            + 0.5 * np.log(sigma_v_sq)
        )

        loglik_i = ll_normal + ll_adjustment

        if not np.isfinite(loglik_i):
            return -np.inf

        loglik += loglik_i

    return loglik


def loglik_pitt_lee_exponential(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    sign: int = 1,
) -> float:
    """Log-likelihood for Pitt & Lee (1981) panel model with exponential.

    Model: y_{it} = X_{it}β + v_{it} - sign*u_i

    where:
        v_{it} ~ N(0, σ²_v) is noise
        u_i ~ Exp(λ) is time-invariant inefficiency

    For exponential distribution, closed-form solution exists.

    Parameters:
        theta: Parameter vector [β, ln(σ²_v), ln(λ)]
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        entity_id: Entity identifiers (n,)
        time_id: Time identifiers (n,)
        sign: Sign convention (+1 for production, -1 for cost)

    Returns:
        Log-likelihood value (scalar)
    """
    n, k = X.shape

    # Extract parameters
    beta = theta[:k]
    ln_sigma_v_sq = theta[k]
    ln_lambda = theta[k + 1]

    # Transform to natural scale
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_v = np.sqrt(sigma_v_sq)
    lambda_param = np.exp(ln_lambda)

    # Residuals
    epsilon = y - X @ beta

    # Get unique entities
    unique_entities = np.unique(entity_id)

    loglik = 0.0

    # Loop over entities
    for i in unique_entities:
        # Get observations for entity i
        mask_i = entity_id == i
        epsilon_i = epsilon[mask_i]
        T_i = len(epsilon_i)

        # Sum of residuals
        sum_epsilon_i = np.sum(epsilon_i)

        # Terms involving exponential distribution
        # μ_i* = -sign * sum(ε_{it}) - T_i * σ²_v * λ
        mu_star = -sign * sum_epsilon_i - T_i * sigma_v_sq * lambda_param

        # σ_* = σ_v / sqrt(T_i)
        sigma_star = sigma_v / np.sqrt(T_i)

        # Log-likelihood for entity i
        ll_i = (
            -0.5 * T_i * np.log(2 * np.pi)
            - 0.5 * T_i * np.log(sigma_v_sq)
            - np.sum(epsilon_i**2) / (2 * sigma_v_sq)
            + np.log(lambda_param)
            + mu_star
            + 0.5 * T_i * sigma_v_sq * lambda_param**2
            + log_ndtr(mu_star / sigma_star + sigma_star * lambda_param)
        )

        if not np.isfinite(ll_i):
            return -np.inf

        loglik += ll_i

    return loglik


def loglik_pitt_lee_truncated_normal(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    sign: int = 1,
    n_quad_points: int = 12,
) -> float:
    """Log-likelihood for Pitt & Lee (1981) with truncated normal via quadrature.

    Model: y_{it} = X_{it}β + v_{it} - sign*u_i

    where:
        v_{it} ~ N(0, σ²_v) is noise
        u_i ~ N⁺(μ, σ²_u) is time-invariant inefficiency

    Uses Gauss-Hermite quadrature to integrate over u_i.

    Parameters:
        theta: Parameter vector [β, ln(σ²_v), ln(σ²_u), μ]
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        entity_id: Entity identifiers (n,)
        time_id: Time identifiers (n,)
        sign: Sign convention (+1 for production, -1 for cost)
        n_quad_points: Number of quadrature points (default 12)

    Returns:
        Log-likelihood value (scalar)
    """
    from scipy.special import roots_hermite

    n, k = X.shape

    # Extract parameters
    beta = theta[:k]
    ln_sigma_v_sq = theta[k]
    ln_sigma_u_sq = theta[k + 1]
    mu = theta[k + 2]

    # Transform to natural scale
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)
    sigma_v = np.sqrt(sigma_v_sq)
    sigma_u = np.sqrt(sigma_u_sq)

    # Residuals
    epsilon = y - X @ beta

    # Get Gauss-Hermite nodes and weights
    nodes, weights = roots_hermite(n_quad_points)

    # Unique entities
    unique_entities = np.unique(entity_id)

    loglik = 0.0

    # Loop over entities
    for i in unique_entities:
        # Get observations for entity i
        mask_i = entity_id == i
        epsilon_i = epsilon[mask_i]
        T_i = len(epsilon_i)

        # Integrate over u_i using quadrature
        # Transform: u = μ + σ_u * sqrt(2) * x where x ~ N(0,1)
        # But we need u ≥ 0, so we use truncated normal

        integral = 0.0

        for j in range(n_quad_points):
            # Transform node to u space (truncated at 0)
            u_j = mu + sigma_u * np.sqrt(2) * nodes[j]

            if u_j < 0:
                continue  # Skip negative values (truncation)

            # Likelihood contribution for this u_j
            # ∏_t f(y_{it} | X_{it}, u_j)
            ll_v = -0.5 * T_i * np.log(2 * np.pi) - 0.5 * T_i * np.log(sigma_v_sq)
            ll_v -= np.sum((epsilon_i + sign * u_j) ** 2) / (2 * sigma_v_sq)

            # Density of u_j under truncated normal
            # g(u) = φ((u-μ)/σ_u) / (σ_u * Φ(μ/σ_u))
            z = (u_j - mu) / sigma_u
            log_g_u = -0.5 * np.log(2 * np.pi) - np.log(sigma_u) - 0.5 * z**2
            log_g_u -= log_ndtr(mu / sigma_u)  # Normalizing constant

            # Add weighted contribution
            integral += weights[j] * np.exp(ll_v + log_g_u)

        # Convert to log
        if integral > 0:
            loglik_i = np.log(integral / np.sqrt(np.pi))
        else:
            return -np.inf

        if not np.isfinite(loglik_i):
            return -np.inf

        loglik += loglik_i

    return loglik


def loglik_battese_coelli_92(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    sign: int = 1,
    n_quad_points: int = 12,
) -> float:
    """Log-likelihood for Battese & Coelli (1992) time-varying model.

    Model: y_{it} = X_{it}β + v_{it} - sign*u_{it}

    where:
        u_{it} = exp[-η(t - T)] * u_i
        u_i ~ N⁺(μ, σ²_u) is base inefficiency
        η is the time decay parameter

    When η > 0: efficiency improves over time (u decreases)
    When η < 0: efficiency worsens over time (u increases)
    When η = 0: reduces to Pitt-Lee model

    Parameters:
        theta: Parameter vector [β, ln(σ²_v), ln(σ²_u), μ, η]
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        entity_id: Entity identifiers (n,)
        time_id: Time identifiers (n,)
        sign: Sign convention (+1 for production, -1 for cost)
        n_quad_points: Number of quadrature points

    Returns:
        Log-likelihood value (scalar)

    References:
        Battese, G. E., & Coelli, T. J. (1992).
    """
    from scipy.special import roots_hermite

    n, k = X.shape

    # Extract parameters
    beta = theta[:k]
    ln_sigma_v_sq = theta[k]
    ln_sigma_u_sq = theta[k + 1]
    mu = theta[k + 2]
    eta = theta[k + 3]

    # Transform to natural scale
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)
    sigma_v = np.sqrt(sigma_v_sq)
    sigma_u = np.sqrt(sigma_u_sq)

    # Residuals
    epsilon = y - X @ beta

    # Get maximum time period (T)
    T_max = time_id.max()

    # Get Gauss-Hermite nodes and weights
    nodes, weights = roots_hermite(n_quad_points)

    # Unique entities
    unique_entities = np.unique(entity_id)

    loglik = 0.0

    # Loop over entities
    for i in unique_entities:
        # Get observations for entity i
        mask_i = entity_id == i
        epsilon_i = epsilon[mask_i]
        time_i = time_id[mask_i]
        T_i = len(epsilon_i)

        # Integrate over u_i using quadrature
        integral = 0.0

        for j in range(n_quad_points):
            # Transform node to u space
            u_j = mu + sigma_u * np.sqrt(2) * nodes[j]

            if u_j < 0:
                continue  # Truncated normal

            # Compute u_{it} = exp[-η(t - T)] * u_j for each t
            decay_factors = np.exp(-eta * (time_i - T_max))
            u_it = decay_factors * u_j

            # Likelihood for all time periods
            ll_v = -0.5 * T_i * np.log(2 * np.pi) - 0.5 * T_i * np.log(sigma_v_sq)
            ll_v -= np.sum((epsilon_i + sign * u_it) ** 2) / (2 * sigma_v_sq)

            # Density of u_j
            z = (u_j - mu) / sigma_u
            log_g_u = -0.5 * np.log(2 * np.pi) - np.log(sigma_u) - 0.5 * z**2
            log_g_u -= log_ndtr(mu / sigma_u)

            # Add weighted contribution
            integral += weights[j] * np.exp(ll_v + log_g_u)

        # Convert to log
        if integral > 0:
            loglik_i = np.log(integral / np.sqrt(np.pi))
        else:
            return -np.inf

        if not np.isfinite(loglik_i):
            return -np.inf

        loglik += loglik_i

    return loglik


def loglik_battese_coelli_95(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    sign: int = 1,
    n_quad_points: int = 12,
) -> float:
    """Log-likelihood for Battese & Coelli (1995) with inefficiency effects.

    Model: y_{it} = X_{it}β + v_{it} - sign*u_{it}

    where:
        u_{it} ~ N⁺(μ_{it}, σ²_u)
        μ_{it} = Z_{it}δ (heterogeneous mean)

    This is a single-step estimation where inefficiency determinants
    are included directly in the likelihood.

    Parameters:
        theta: Parameter vector [β, ln(σ²_v), ln(σ²_u), δ]
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        Z: Inefficiency determinants (n, m)
        entity_id: Entity identifiers (n,)
        time_id: Time identifiers (n,)
        sign: Sign convention (+1 for production, -1 for cost)
        n_quad_points: Number of quadrature points

    Returns:
        Log-likelihood value (scalar)

    References:
        Battese, G. E., & Coelli, T. J. (1995).

    Note:
        This implementation uses single-step MLE, avoiding the bias
        of two-step estimation (Wang & Schmidt 2002).
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

    # Residuals
    epsilon = y - X @ beta

    # Compute σ² and σ_*²
    sigma_sq = sigma_v_sq + sigma_u_sq
    sigma = np.sqrt(sigma_sq)
    sigma_star_sq = (sigma_v_sq * sigma_u_sq) / sigma_sq
    sigma_star = np.sqrt(sigma_star_sq)

    # Compute μ_*
    mu_star = (sigma_u_sq * (-sign * epsilon) + sigma_v_sq * mu_it) / sigma_sq

    # Log-likelihood
    # ln f(ε) = constant - ln(σ) - 0.5*(ε + sign*μ)²/σ²
    #           + ln Φ(μ_*/σ_*) - ln Φ(μ/σ_u)

    loglik_i = (
        -np.log(sigma)
        - 0.5 * np.log(2 * np.pi)
        - 0.5 * (epsilon + sign * mu_it) ** 2 / sigma_sq
        + log_ndtr(mu_star / sigma_star)
        - log_ndtr(mu_it / sigma_u)
    )

    loglik = np.sum(loglik_i)

    if not np.isfinite(loglik):
        return -np.inf

    return loglik


def loglik_kumbhakar_1990(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    sign: int = 1,
    n_quad_points: int = 12,
) -> float:
    """Log-likelihood for Kumbhakar (1990) flexible time pattern.

    Model: y_{it} = X_{it}β + v_{it} - sign*u_{it}

    where:
        u_{it} = B(t) * u_i
        B(t) = 1 / [1 + exp(b*t + c*t²)]  (logistic function)
        u_i ~ N⁺(μ, σ²_u)

    This allows for non-monotonic time patterns (U-shape, inverted-U).

    Parameters:
        theta: Parameter vector [β, ln(σ²_v), ln(σ²_u), μ, b, c]
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        entity_id: Entity identifiers (n,)
        time_id: Time identifiers (n,)
        sign: Sign convention (+1 for production, -1 for cost)
        n_quad_points: Number of quadrature points

    Returns:
        Log-likelihood value (scalar)

    References:
        Kumbhakar, S. C. (1990).
            Production frontiers, panel data, and time-varying technical
            inefficiency. Journal of Econometrics, 46(1-2), 201-211.
    """
    from scipy.special import roots_hermite

    n, k = X.shape

    # Extract parameters
    beta = theta[:k]
    ln_sigma_v_sq = theta[k]
    ln_sigma_u_sq = theta[k + 1]
    mu = theta[k + 2]
    b = theta[k + 3]
    c = theta[k + 4]

    # Transform to natural scale
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)
    sigma_v = np.sqrt(sigma_v_sq)
    sigma_u = np.sqrt(sigma_u_sq)

    # Residuals
    epsilon = y - X @ beta

    # Get Gauss-Hermite nodes and weights
    nodes, weights = roots_hermite(n_quad_points)

    # Unique entities
    unique_entities = np.unique(entity_id)

    loglik = 0.0

    # Loop over entities
    for i in unique_entities:
        # Get observations for entity i
        mask_i = entity_id == i
        epsilon_i = epsilon[mask_i]
        time_i = time_id[mask_i]
        T_i = len(epsilon_i)

        # Integrate over u_i using quadrature
        integral = 0.0

        for j in range(n_quad_points):
            # Transform node to u space
            u_j = mu + sigma_u * np.sqrt(2) * nodes[j]

            if u_j < 0:
                continue

            # Compute B(t) = 1 / [1 + exp(b*t + c*t²)]
            B_t = 1.0 / (1.0 + np.exp(b * time_i + c * time_i**2))

            # Compute u_{it} = B(t) * u_j
            u_it = B_t * u_j

            # Likelihood for all time periods
            ll_v = -0.5 * T_i * np.log(2 * np.pi) - 0.5 * T_i * np.log(sigma_v_sq)
            ll_v -= np.sum((epsilon_i + sign * u_it) ** 2) / (2 * sigma_v_sq)

            # Density of u_j
            z = (u_j - mu) / sigma_u
            log_g_u = -0.5 * np.log(2 * np.pi) - np.log(sigma_u) - 0.5 * z**2
            log_g_u -= log_ndtr(mu / sigma_u)

            # Add weighted contribution
            integral += weights[j] * np.exp(ll_v + log_g_u)

        # Convert to log
        if integral > 0:
            loglik_i = np.log(integral / np.sqrt(np.pi))
        else:
            return -np.inf

        if not np.isfinite(loglik_i):
            return -np.inf

        loglik += loglik_i

    return loglik


def loglik_lee_schmidt_1993(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    sign: int = 1,
    n_quad_points: int = 12,
) -> float:
    """Log-likelihood for Lee & Schmidt (1993) with time dummies.

    Model: y_{it} = X_{it}β + v_{it} - sign*u_{it}

    where:
        u_{it} = δ_t * u_i
        δ_t are free parameters (one per time period)
        δ_T = 1 (normalization)
        u_i ~ N⁺(μ, σ²_u)

    This is the most flexible time pattern specification, but
    requires T-1 additional parameters.

    Parameters:
        theta: Parameter vector [β, ln(σ²_v), ln(σ²_u), μ, δ_1, ..., δ_{T-1}]
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        entity_id: Entity identifiers (n,)
        time_id: Time identifiers (n,)
        sign: Sign convention (+1 for production, -1 for cost)
        n_quad_points: Number of quadrature points

    Returns:
        Log-likelihood value (scalar)

    References:
        Lee, Y. H., & Schmidt, P. (1993).
            A production frontier model with flexible temporal variation in
            technical inefficiency. In The measurement of productive efficiency:
            Techniques and applications (pp. 237-255).
    """
    from scipy.special import roots_hermite

    n, k = X.shape

    # Get number of time periods
    T = time_id.max() + 1  # Assuming time coded as 0, 1, ..., T-1

    # Extract parameters
    beta = theta[:k]
    ln_sigma_v_sq = theta[k]
    ln_sigma_u_sq = theta[k + 1]
    mu = theta[k + 2]
    delta_params = theta[k + 3 : k + 3 + T - 1]

    # Construct full delta vector with normalization δ_T = 1
    delta = np.concatenate([delta_params, [1.0]])

    # Transform to natural scale
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)
    sigma_v = np.sqrt(sigma_v_sq)
    sigma_u = np.sqrt(sigma_u_sq)

    # Residuals
    epsilon = y - X @ beta

    # Get Gauss-Hermite nodes and weights
    nodes, weights = roots_hermite(n_quad_points)

    # Unique entities
    unique_entities = np.unique(entity_id)

    loglik = 0.0

    # Loop over entities
    for i in unique_entities:
        # Get observations for entity i
        mask_i = entity_id == i
        epsilon_i = epsilon[mask_i]
        time_i = time_id[mask_i]
        T_i = len(epsilon_i)

        # Integrate over u_i using quadrature
        integral = 0.0

        for j in range(n_quad_points):
            # Transform node to u space
            u_j = mu + sigma_u * np.sqrt(2) * nodes[j]

            if u_j < 0:
                continue

            # Compute u_{it} = δ_t * u_j for each t
            u_it = delta[time_i] * u_j

            # Likelihood for all time periods
            ll_v = -0.5 * T_i * np.log(2 * np.pi) - 0.5 * T_i * np.log(sigma_v_sq)
            ll_v -= np.sum((epsilon_i + sign * u_it) ** 2) / (2 * sigma_v_sq)

            # Density of u_j
            z = (u_j - mu) / sigma_u
            log_g_u = -0.5 * np.log(2 * np.pi) - np.log(sigma_u) - 0.5 * z**2
            log_g_u -= log_ndtr(mu / sigma_u)

            # Add weighted contribution
            integral += weights[j] * np.exp(ll_v + log_g_u)

        # Convert to log
        if integral > 0:
            loglik_i = np.log(integral / np.sqrt(np.pi))
        else:
            return -np.inf

        if not np.isfinite(loglik_i):
            return -np.inf

        loglik += loglik_i

    return loglik


def loglik_bc92(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    sign: int = 1,
) -> float:
    """Log-likelihood for Battese & Coelli (1992) time-decay model.

    Model: y_it = x_it'β + v_it - sign*u_it
           u_it = exp[-η(t - T_i)] · u_i
           u_i ~ N⁺(0, σ²_u)
           v_it ~ N(0, σ²_v)

    where T_i is the last period observed for entity i.

    Parameters:
        theta: [β, ln(σ²_v), ln(σ²_u), η]
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        entity_id: Entity identifiers (n,) - coded 0, 1, ..., N-1
        time_id: Time identifiers (n,) - coded 0, 1, ..., T-1
        sign: +1 for production, -1 for cost

    Returns:
        Log-likelihood value

    References:
        Battese, G. E., & Coelli, T. J. (1992).
            Frontier production functions, technical efficiency and panel data:
            with application to paddy farmers in India.
            Journal of Productivity Analysis, 3(1-2), 153-169.
    """
    n, k = X.shape

    # Extract parameters
    beta = theta[:k]
    ln_sigma_v_sq = theta[k]
    ln_sigma_u_sq = theta[k + 1]
    eta = theta[k + 2]  # Time-decay parameter

    # Transform to natural scale
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)
    sigma_v = np.sqrt(sigma_v_sq)
    sigma_u = np.sqrt(sigma_u_sq)

    # Residuals
    epsilon = y - X @ beta

    # Get unique entities
    unique_entities = np.unique(entity_id)

    loglik = 0.0

    # For each entity
    for i in unique_entities:
        # Get observations for entity i
        mask_i = entity_id == i
        epsilon_i = epsilon[mask_i]
        time_i = time_id[mask_i]
        T_i = len(epsilon_i)

        # Last period for entity i
        T_i_max = time_i.max()

        # Time-varying inefficiency scale
        # u_it = exp[-η(t - T_i)] · u_i
        decay_factors = np.exp(-eta * (time_i - T_i_max))

        # For each time period, compute contribution to likelihood
        for t_idx in range(T_i):
            t = time_i[t_idx]
            decay_t = decay_factors[t_idx]
            eps_it = epsilon_i[t_idx]

            # Adjusted variance for this observation
            sigma_u_it_sq = (decay_t**2) * sigma_u_sq
            sigma_it_sq = sigma_v_sq + sigma_u_it_sq
            sigma_it = np.sqrt(sigma_it_sq)

            # Check for numerical issues
            if sigma_it < 1e-10:
                return -np.inf

            # Conditional moments
            sigma_star_sq = (sigma_v_sq * sigma_u_it_sq) / sigma_it_sq
            sigma_star = np.sqrt(sigma_star_sq)
            mu_star = -sign * eps_it * sigma_u_it_sq / sigma_it_sq

            # Log-likelihood contribution
            # ln f(ε_it) = ln(2) - ln(σ_it) - ln(√(2π)) - 0.5*(ε_it/σ_it)² + ln Φ(μ*/σ*)
            ll_it = (
                np.log(2)
                - np.log(sigma_it)
                - LOG_SQRT_2PI
                - 0.5 * (eps_it**2) / sigma_it_sq
                + log_ndtr(mu_star / sigma_star)
            )

            if not np.isfinite(ll_it):
                return -np.inf

            loglik += ll_it

    return loglik
