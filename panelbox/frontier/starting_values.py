"""
Starting value computation for SFA estimation.

This module implements methods for computing initial parameter values
for maximum likelihood estimation of stochastic frontier models.

The main approaches are:
1. Method of moments (Olson-Schmidt-Waldman 1980)
2. OLS-based estimation
3. Grid search over variance parameters
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats


def ols_starting_values(
    y: np.ndarray, X: np.ndarray, dist: str = "half_normal"
) -> Tuple[np.ndarray, float, float]:
    """Compute OLS-based starting values.

    Estimates β via OLS, then uses residual moments to estimate
    variance components.

    Parameters:
        y: Dependent variable (n,)
        X: Exogenous variables (n, k)
        dist: Distribution type ('half_normal', 'exponential', etc.)

    Returns:
        Tuple of (beta, sigma_v_sq, sigma_u_sq)

    References:
        Olson, J. A., Schmidt, P., & Waldman, D. M. (1980).
    """
    # OLS estimation
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta

    # Compute moments
    m2 = np.mean(residuals**2)  # Second moment
    m3 = np.mean(residuals**3)  # Third moment

    # Estimate variance components based on distribution
    if dist == "half_normal":
        sigma_u_sq, sigma_v_sq = _moments_half_normal(m2, m3)
    elif dist == "exponential":
        sigma_u_sq, sigma_v_sq = _moments_exponential(m2, m3)
    elif dist == "truncated_normal":
        # For truncated normal without heterogeneity, use half-normal moments
        sigma_u_sq, sigma_v_sq = _moments_half_normal(m2, m3)
    elif dist == "gamma":
        # Gamma moments are complex, use half-normal as approximation
        sigma_u_sq, sigma_v_sq = _moments_half_normal(m2, m3)
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    # Ensure positive variances
    sigma_v_sq = max(sigma_v_sq, 1e-6)
    sigma_u_sq = max(sigma_u_sq, 1e-6)

    # Correct intercept bias
    # For production frontier: E[ε] = E[v - u] = -E[u]
    # For half-normal: E[u] = σ_u * sqrt(2/π)
    # OLS intercept is biased downward by this amount
    if X.shape[1] > 0 and np.allclose(X[:, 0], 1.0):
        # First column is constant
        sqrt_2_pi = np.sqrt(2 / np.pi)
        sigma_u = np.sqrt(sigma_u_sq)

        if dist == "half_normal":
            bias = sigma_u * sqrt_2_pi
        elif dist == "exponential":
            # For exponential: E[u] = σ_u
            bias = sigma_u
        elif dist == "truncated_normal":
            # For truncated normal at zero: E[u] ≈ σ_u * sqrt(2/π)
            bias = sigma_u * sqrt_2_pi
        elif dist == "gamma":
            # For gamma: E[u] = P/λ, approximated by exponential
            bias = sigma_u
        else:
            bias = 0.0

        # Adjust intercept for production frontier
        # Note: for cost frontier, sign would be reversed
        beta = beta.copy()
        beta[0] += bias

    return beta, sigma_v_sq, sigma_u_sq


def _moments_half_normal(m2: float, m3: float) -> Tuple[float, float]:
    """Estimate variances using half-normal moments.

    For half-normal distribution:
        E[ε] = -σ_u √(2/π)
        E[ε²] = σ²_v + (1 - 2/π)σ²_u
        E[ε³] = -σ³_u √(2/π)(1 - 4/π)

    From third moment:
        σ_u = [E[ε³] / (-√(2/π)(1 - 4/π))]^(1/3)

    From second moment:
        σ_v = √(E[ε²] - (1 - 2/π)σ²_u)

    Parameters:
        m2: Second moment E[ε²]
        m3: Third moment E[ε³]

    Returns:
        (sigma_u_sq, sigma_v_sq)
    """
    # Constants
    sqrt_2_pi = np.sqrt(2 / np.pi)
    factor = sqrt_2_pi * (1 - 4 / np.pi)

    # From third moment (need to handle sign)
    if abs(m3) < 1e-10:
        # No skewness detected, assume σ_u ≈ 0
        sigma_u_sq = 1e-6
        sigma_v_sq = max(m2, 1e-6)
        warnings.warn(
            "Third moment near zero. Setting σ_u to small value. "
            "Model may be better estimated as OLS.",
            UserWarning,
        )
    else:
        # Solve for σ_u
        sigma_u = abs(m3 / (-factor)) ** (1 / 3)
        sigma_u_sq = sigma_u**2

        # From second moment
        sigma_v_sq = m2 - (1 - 2 / np.pi) * sigma_u_sq

        # Handle negative variance
        if sigma_v_sq < 0:
            # Redistribute variance
            total_var = m2
            sigma_u_sq = 0.5 * total_var
            sigma_v_sq = 0.5 * total_var
            warnings.warn(
                "Negative variance from moments. Using equal split of variance.", UserWarning
            )

    return sigma_u_sq, max(sigma_v_sq, 1e-6)


def _moments_exponential(m2: float, m3: float) -> Tuple[float, float]:
    """Estimate variances using exponential moments.

    For exponential distribution with parameter λ:
        E[u] = 1/λ = σ_u
        Var[u] = 1/λ² = σ²_u

    Moments:
        E[ε²] = σ²_v + σ²_u
        E[ε³] = -2σ³_u

    Parameters:
        m2: Second moment E[ε²]
        m3: Third moment E[ε³]

    Returns:
        (sigma_u_sq, sigma_v_sq)
    """
    if abs(m3) < 1e-10:
        # No skewness
        sigma_u_sq = 1e-6
        sigma_v_sq = max(m2, 1e-6)
        warnings.warn("Third moment near zero. Setting σ_u to small value.", UserWarning)
    else:
        # From third moment
        sigma_u = abs(m3 / (-2)) ** (1 / 3)
        sigma_u_sq = sigma_u**2

        # From second moment
        sigma_v_sq = m2 - sigma_u_sq

        if sigma_v_sq < 0:
            total_var = m2
            sigma_u_sq = 0.5 * total_var
            sigma_v_sq = 0.5 * total_var
            warnings.warn("Negative variance from moments. Using equal split.", UserWarning)

    return sigma_u_sq, max(sigma_v_sq, 1e-6)


def grid_search_starting_values(
    y: np.ndarray, X: np.ndarray, dist: str, likelihood_func, sign: int, n_points: int = 5
) -> Tuple[np.ndarray, float, float]:
    """Grid search for starting values.

    Performs a grid search over variance parameters to find good
    starting values for MLE optimization.

    Parameters:
        y: Dependent variable
        X: Exogenous variables
        dist: Distribution type
        likelihood_func: Log-likelihood function to evaluate
        sign: Sign convention
        n_points: Number of grid points per dimension

    Returns:
        Tuple of (beta, sigma_v_sq, sigma_u_sq)
    """
    # Start with OLS
    beta_ols, sigma_v_ols, sigma_u_ols = ols_starting_values(y, X, dist)

    # Create grid around OLS estimates
    sigma_v_grid = np.linspace(0.1 * sigma_v_ols, 2.0 * sigma_v_ols, n_points)
    sigma_u_grid = np.linspace(0.1 * sigma_u_ols, 2.0 * sigma_u_ols, n_points)

    best_loglik = -np.inf
    best_params = (beta_ols, sigma_v_ols, sigma_u_ols)

    # Grid search
    for sigma_v in sigma_v_grid:
        for sigma_u in sigma_u_grid:
            # Construct parameter vector
            k = X.shape[1]
            theta = np.concatenate([beta_ols, [np.log(sigma_v**2)], [np.log(sigma_u**2)]])

            # Additional parameters for truncated normal
            if dist == "truncated_normal":
                theta = np.concatenate([theta, [0.0]])  # μ = 0
            elif dist == "gamma":
                theta = np.concatenate([theta, [np.log(2.0)], [np.log(1.0)]])  # P = 2, θ = 1

            # Evaluate likelihood
            try:
                loglik = likelihood_func(theta, y, X, sign=sign)

                if loglik > best_loglik:
                    best_loglik = loglik
                    best_params = (beta_ols, sigma_v**2, sigma_u**2)
            except (ValueError, RuntimeError):
                continue

    return best_params


def get_starting_values(
    y: np.ndarray,
    X: np.ndarray,
    Z: Optional[np.ndarray],
    dist: str,
    grid_search: bool = False,
    likelihood_func=None,
    sign: int = 1,
) -> np.ndarray:
    """Get starting values for MLE estimation.

    Main entry point for computing starting values. Uses method of moments
    by default, with optional grid search for robustness.

    Parameters:
        y: Dependent variable
        X: Exogenous variables
        Z: Inefficiency covariates (for BC95)
        dist: Distribution type
        grid_search: Whether to use grid search
        likelihood_func: Likelihood function (required if grid_search=True)
        sign: Sign convention

    Returns:
        Starting parameter vector θ₀
    """
    if grid_search and likelihood_func is None:
        raise ValueError("likelihood_func required when grid_search=True")

    # Get base starting values
    if grid_search:
        beta, sigma_v_sq, sigma_u_sq = grid_search_starting_values(
            y, X, dist, likelihood_func, sign
        )
    else:
        beta, sigma_v_sq, sigma_u_sq = ols_starting_values(y, X, dist)

    # Construct parameter vector
    k = X.shape[1]
    theta = np.concatenate([beta, [np.log(sigma_v_sq)], [np.log(sigma_u_sq)]])

    # Additional parameters for specific distributions
    if dist == "truncated_normal":
        if Z is not None:
            # BC95 model: add δ parameters
            m = Z.shape[1]
            delta = np.zeros(m)
            theta = np.concatenate([theta, delta])
        else:
            # Simple truncated normal: add μ
            theta = np.concatenate([theta, [0.0]])

    elif dist == "gamma":
        # Add P (shape) and θ (rate) parameters
        # Start with P=2 (moderate shape)
        # Start with θ=1 (moderate rate)
        # E[u] = P/θ = 2/1 = 2
        theta = np.concatenate([theta, [np.log(2.0)], [np.log(1.0)]])

    return theta


def check_starting_values(
    theta: np.ndarray, y: np.ndarray, X: np.ndarray, likelihood_func, sign: int
) -> Dict[str, any]:
    """Check quality of starting values.

    Evaluates likelihood and gradient at starting values to ensure
    they are reasonable.

    Parameters:
        theta: Starting parameter vector
        y: Dependent variable
        X: Exogenous variables
        likelihood_func: Log-likelihood function
        sign: Sign convention

    Returns:
        Dictionary with diagnostic information
    """
    # Evaluate likelihood
    try:
        loglik = likelihood_func(theta, y, X, sign=sign)
        is_finite = np.isfinite(loglik)
    except (ValueError, RuntimeError) as e:
        return {"valid": False, "loglik": -np.inf, "error": str(e)}

    # Check parameter values
    k = X.shape[1]
    beta = theta[:k]
    ln_sigma_v_sq = theta[k]
    ln_sigma_u_sq = theta[k + 1]

    sigma_v = np.exp(0.5 * ln_sigma_v_sq)
    sigma_u = np.exp(0.5 * ln_sigma_u_sq)

    # Check for reasonable values
    reasonable = (
        sigma_v > 1e-6
        and sigma_v < 1e6
        and sigma_u > 1e-6
        and sigma_u < 1e6
        and np.all(np.isfinite(beta))
    )

    return {
        "valid": is_finite and reasonable,
        "loglik": loglik,
        "sigma_v": sigma_v,
        "sigma_u": sigma_u,
        "lambda": sigma_u / sigma_v,
        "finite": is_finite,
        "reasonable": reasonable,
    }
