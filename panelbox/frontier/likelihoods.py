"""
Log-likelihood functions for stochastic frontier models.

This module implements numerically stable log-likelihood functions for
various distributional assumptions about the inefficiency term.

All functions use the parameterization:
    θ = [β, ln(σ²_v), ln(σ²_u), ...]

where ln(·) parameterization ensures positivity constraints during optimization.

References:
    Aigner, D., Lovell, C. K., & Schmidt, P. (1977).
        Normal-half normal model.

    Meeusen, W., & van Den Broeck, J. (1977).
        Normal-exponential model.

    Stevenson, R. E. (1980).
        Truncated normal model.

    Greene, W. H. (1990).
        Gamma distribution model.
"""

from typing import Tuple

import numpy as np
from scipy import stats
from scipy.special import gammaln, log_ndtr, ndtr

# Constants for numerical stability
SQRT_2PI = np.sqrt(2 * np.pi)
SQRT_2_OVER_PI = np.sqrt(2 / np.pi)
LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def loglik_half_normal(theta: np.ndarray, y: np.ndarray, X: np.ndarray, sign: int = 1) -> float:
    """Log-likelihood for normal-half normal model.

    Model: ε = v - sign*u where v ~ N(0, σ²_v) and u ~ N⁺(0, σ²_u)

    The composed error density is:
        f(ε) = (2/σ) φ(ε/σ) Φ(-sign*ε*λ/σ)

    where σ² = σ²_v + σ²_u, λ = σ_u/σ_v, φ is standard normal pdf,
    and Φ is standard normal cdf.

    Parameters:
        theta: Parameter vector [β, ln(σ²_v), ln(σ²_u)]
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        sign: Sign convention (+1 for production, -1 for cost)

    Returns:
        Log-likelihood value (scalar)

    References:
        Aigner, D., Lovell, C. K., & Schmidt, P. (1977).
    """
    n, k = X.shape

    # Extract parameters
    beta = theta[:k]
    ln_sigma_v_sq = theta[k]
    ln_sigma_u_sq = theta[k + 1]

    # Transform to natural scale
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)

    # Compute derived quantities
    sigma_sq = sigma_v_sq + sigma_u_sq
    sigma = np.sqrt(sigma_sq)
    lambda_param = np.sqrt(sigma_u_sq / sigma_v_sq)

    # Residuals
    epsilon = y - X @ beta

    # Log-likelihood components
    # ln f(ε) = ln(2/σ) + ln φ(ε/σ) + ln Φ(-sign*ε*λ/σ)
    #         = ln(2) - ln(σ) - 0.5*ln(2π) - ε²/(2σ²) + ln Φ(-sign*ε*λ/σ)

    # Standardized values
    eps_over_sigma = epsilon / sigma
    arg = -sign * epsilon * lambda_param / sigma

    # Log-likelihood using stable functions
    loglik_i = (
        np.log(2)
        - np.log(sigma)
        - 0.5 * np.log(2 * np.pi)
        - 0.5 * eps_over_sigma**2
        + log_ndtr(arg)  # Stable log Φ(x)
    )

    loglik = np.sum(loglik_i)

    # Check for numerical issues
    if not np.isfinite(loglik):
        return -np.inf

    return loglik


def loglik_exponential(theta: np.ndarray, y: np.ndarray, X: np.ndarray, sign: int = 1) -> float:
    """Log-likelihood for normal-exponential model.

    Model: ε = v - sign*u where v ~ N(0, σ²_v) and u ~ Exp(λ_u)

    The composed error density involves:
        f(ε) = (1/σ_u) exp(μ_i + 0.5*σ²_v/σ²_u) Φ(μ_i/σ_v + σ_v/σ_u)

    where μ_i = -sign*ε - σ²_v/σ_u

    Parameters:
        theta: Parameter vector [β, ln(σ²_v), ln(σ²_u)]
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        sign: Sign convention (+1 for production, -1 for cost)

    Returns:
        Log-likelihood value (scalar)

    References:
        Meeusen, W., & van Den Broeck, J. (1977).
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

    # Compute μ_i
    mu_i = -sign * epsilon - sigma_v_sq / sigma_u

    # Log-likelihood
    # ln f(ε) = -ln(σ_u) + μ_i + 0.5*σ²_v/σ²_u + ln Φ(μ_i/σ_v + σ_v/σ_u)

    arg = mu_i / sigma_v + sigma_v / sigma_u

    loglik_i = -np.log(sigma_u) + mu_i + 0.5 * sigma_v_sq / sigma_u_sq + log_ndtr(arg)

    loglik = np.sum(loglik_i)

    if not np.isfinite(loglik):
        return -np.inf

    return loglik


def loglik_truncated_normal(
    theta: np.ndarray, y: np.ndarray, X: np.ndarray, Z: np.ndarray = None, sign: int = 1
) -> float:
    """Log-likelihood for normal-truncated normal model.

    Model: ε = v - sign*u where v ~ N(0, σ²_v) and u ~ N⁺(μ, σ²_u)

    The truncated normal has location parameter μ ≥ 0, which can be
    heterogeneous: μ_i = Z_i'δ (Battese-Coelli 1995).

    Parameters:
        theta: Parameter vector [β, ln(σ²_v), ln(σ²_u), δ] if Z provided,
               else [β, ln(σ²_v), ln(σ²_u), μ]
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        Z: Variables for μ_i = Z_i'δ (n, m) - optional
        sign: Sign convention (+1 for production, -1 for cost)

    Returns:
        Log-likelihood value (scalar)

    References:
        Stevenson, R. E. (1980).
        Battese, G. E., & Coelli, T. J. (1995).
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

    # Compute σ² and σ_*²
    sigma_sq = sigma_v_sq + sigma_u_sq
    sigma = np.sqrt(sigma_sq)
    sigma_star_sq = (sigma_v_sq * sigma_u_sq) / sigma_sq
    sigma_star = np.sqrt(sigma_star_sq)

    # Get μ (location parameter)
    if Z is not None:
        # Heterogeneous μ_i = Z_i'δ
        m = Z.shape[1]
        delta = theta[k + 2 : k + 2 + m]
        mu = Z @ delta
    else:
        # Homogeneous μ
        mu = theta[k + 2]
        mu = np.full(n, mu)  # Broadcast to array

    # Residuals
    epsilon = y - X @ beta

    # Compute μ_*
    mu_star = (sigma_u_sq * (-sign * epsilon) + sigma_v_sq * mu) / sigma_sq

    # Log-likelihood
    # ln f(ε) = constant - ln(σ) - 0.5*(ε + sign*μ)²/σ²
    #           + ln Φ(μ_*/σ_*) - ln Φ(μ/σ_u)

    loglik_i = (
        -np.log(sigma)
        - 0.5 * np.log(2 * np.pi)
        - 0.5 * (epsilon + sign * mu) ** 2 / sigma_sq
        + log_ndtr(mu_star / sigma_star)
        - log_ndtr(mu / sigma_u)
    )

    loglik = np.sum(loglik_i)

    if not np.isfinite(loglik):
        return -np.inf

    return loglik


def loglik_gamma(theta: np.ndarray, y: np.ndarray, X: np.ndarray, sign: int = 1) -> float:
    """Log-likelihood for normal-gamma model.

    Model: ε = v - sign*u where v ~ N(0, σ²_v) and u ~ Gamma(P, λ)

    The gamma distribution has shape P and scale 1/λ, so:
        E[u] = P/λ
        Var[u] = P/λ²

    This is a more flexible model but computationally intensive.

    Parameters:
        theta: Parameter vector [β, ln(σ²_v), ln(P), ln(λ)]
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        sign: Sign convention (+1 for production, -1 for cost)

    Returns:
        Log-likelihood value (scalar)

    References:
        Greene, W. H. (1990).
        Ritter, C., & Simar, L. (1997).

    Note:
        This implementation uses numerical approximation. For production
        use, consider using specialized numerical integration or the
        implementation from frontier package in R.
    """
    n, k = X.shape

    # Extract parameters
    beta = theta[:k]
    ln_sigma_v_sq = theta[k]
    ln_P = theta[k + 1]
    ln_lambda = theta[k + 2]

    # Transform to natural scale
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_v = np.sqrt(sigma_v_sq)
    P = np.exp(ln_P)
    lambda_param = np.exp(ln_lambda)

    # Residuals
    epsilon = y - X @ beta

    # For gamma model, we need to integrate over u
    # f(ε) = ∫ φ((ε+sign*u)/σ_v) · gamma(u; P, λ) du

    # Use Gauss-Hermite quadrature approximation
    # This is computationally expensive - for production, use better methods

    from scipy.special import roots_hermite

    # Number of quadrature points
    n_points = 20

    # Get Gauss-Hermite nodes and weights
    nodes, weights = roots_hermite(n_points)

    loglik = 0.0

    for i in range(n):
        eps_i = epsilon[i]

        # Integrate using quadrature
        # Transform: u = a + b*x where x ~ N(0,1)
        # Use E[u] = P/λ as center and √(P/λ²) as scale for transformation
        u_mean = P / lambda_param
        u_std = np.sqrt(P) / lambda_param

        # Evaluate integral
        integral = 0.0

        for j in range(n_points):
            # Transform node to u space
            u_j = u_mean + u_std * np.sqrt(2) * nodes[j]

            if u_j <= 0:
                continue  # Gamma is only defined for u > 0

            # Evaluate components
            # Normal part: φ((ε + sign*u)/σ_v)
            z = (eps_i + sign * u_j) / sigma_v
            log_normal = -0.5 * np.log(2 * np.pi) - np.log(sigma_v) - 0.5 * z**2

            # Gamma part: gamma(u; P, λ)
            log_gamma = (
                P * np.log(lambda_param) - gammaln(P) + (P - 1) * np.log(u_j) - lambda_param * u_j
            )

            # Add to integral (in log space, convert back)
            integral += weights[j] * np.exp(log_normal + log_gamma)

        # Convert to log
        if integral > 0:
            loglik += np.log(integral / np.sqrt(np.pi))  # Normalize for Hermite weights
        else:
            return -np.inf

    if not np.isfinite(loglik):
        return -np.inf

    return loglik


def gradient_half_normal(
    theta: np.ndarray, y: np.ndarray, X: np.ndarray, sign: int = 1
) -> np.ndarray:
    """Analytical gradient for half-normal log-likelihood.

    Returns gradient with respect to θ = [β, ln(σ²_v), ln(σ²_u)].

    Parameters:
        theta: Parameter vector
        y: Dependent variable
        X: Exogenous variables
        sign: Sign convention

    Returns:
        Gradient vector (same shape as theta)
    """
    n, k = X.shape

    # Extract parameters
    beta = theta[:k]
    ln_sigma_v_sq = theta[k]
    ln_sigma_u_sq = theta[k + 1]

    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)
    sigma_sq = sigma_v_sq + sigma_u_sq
    sigma = np.sqrt(sigma_sq)

    # Residuals
    epsilon = y - X @ beta

    # Compute λ and derived quantities
    lambda_param = np.sqrt(sigma_u_sq / sigma_v_sq)
    arg = -sign * epsilon * lambda_param / sigma

    # Mills ratio: φ(x)/Φ(x)
    phi_arg = stats.norm.pdf(arg)
    Phi_arg = ndtr(arg)
    mills = phi_arg / (Phi_arg + 1e-10)  # Avoid division by zero

    # Gradient w.r.t. β
    grad_beta = X.T @ (epsilon / sigma_sq - sign * lambda_param * mills / sigma)

    # Gradient w.r.t. ln(σ²_v)
    # Chain rule: ∂/∂ln(σ²_v) = σ²_v * ∂/∂σ²_v
    grad_ln_sigma_v = sigma_v_sq * np.sum(
        -1 / (2 * sigma_sq)
        + epsilon**2 / (2 * sigma_sq**2)
        - sign * epsilon * mills * (sigma_u_sq / (sigma * sigma_sq * lambda_param))
    )

    # Gradient w.r.t. ln(σ²_u)
    grad_ln_sigma_u = sigma_u_sq * np.sum(
        -1 / (2 * sigma_sq)
        + epsilon**2 / (2 * sigma_sq**2)
        + sign * epsilon * mills * (sigma_v_sq / (sigma * sigma_sq * lambda_param))
    )

    grad = np.concatenate([grad_beta, [grad_ln_sigma_v], [grad_ln_sigma_u]])

    return grad


def gradient_exponential(
    theta: np.ndarray, y: np.ndarray, X: np.ndarray, sign: int = 1
) -> np.ndarray:
    """Analytical gradient for exponential log-likelihood.

    Parameters:
        theta: Parameter vector
        y: Dependent variable
        X: Exogenous variables
        sign: Sign convention

    Returns:
        Gradient vector
    """
    n, k = X.shape

    # Extract parameters
    beta = theta[:k]
    ln_sigma_v_sq = theta[k]
    ln_sigma_u_sq = theta[k + 1]

    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)
    sigma_v = np.sqrt(sigma_v_sq)
    sigma_u = np.sqrt(sigma_u_sq)

    # Residuals
    epsilon = y - X @ beta

    # Compute μ_i
    mu_i = -sign * epsilon - sigma_v_sq / sigma_u

    # Argument for Φ
    arg = mu_i / sigma_v + sigma_v / sigma_u

    # Mills ratio
    phi_arg = stats.norm.pdf(arg)
    Phi_arg = ndtr(arg)
    mills = phi_arg / (Phi_arg + 1e-10)

    # Gradient w.r.t. β
    grad_beta = X.T @ (-sign + sign * mills / sigma_v)

    # Gradient w.r.t. ln(σ²_v)
    grad_ln_sigma_v = sigma_v_sq * np.sum(
        1 / sigma_u + 0.5 / sigma_u_sq + mills * (-mu_i / (2 * sigma_v_sq) + 1 / (2 * sigma_u))
    )

    # Gradient w.r.t. ln(σ²_u)
    grad_ln_sigma_u = sigma_u_sq * np.sum(
        -1 / (2 * sigma_u)
        + sigma_v_sq / (2 * sigma_u**3)
        - sigma_v_sq / sigma_u_sq
        - mills * sigma_v / (2 * sigma_u)
    )

    grad = np.concatenate([grad_beta, [grad_ln_sigma_v], [grad_ln_sigma_u]])

    return grad


# Note: Gradients for truncated normal and gamma are complex and typically
# handled via numerical differentiation in practice
