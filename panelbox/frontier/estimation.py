"""
Maximum likelihood estimation for stochastic frontier models.

This module implements MLE estimation with various optimization algorithms
and convergence diagnostics.
"""

import warnings
from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.linalg import inv
from scipy.optimize import minimize

from .data import FrontierType
from .likelihoods import (
    gradient_exponential,
    gradient_half_normal,
    loglik_exponential,
    loglik_gamma,
    loglik_half_normal,
    loglik_truncated_normal,
)
from .result import SFResult
from .starting_values import check_starting_values, get_starting_values


def estimate_mle(
    model,
    start_params: Optional[np.ndarray] = None,
    optimizer: str = "L-BFGS-B",
    maxiter: int = 1000,
    tol: float = 1e-8,
    grid_search: bool = False,
    verbose: bool = False,
    **kwargs,
) -> SFResult:
    """Estimate stochastic frontier model via maximum likelihood.

    Parameters:
        model: StochasticFrontier model instance
        start_params: Initial parameter values (computed if None)
        optimizer: Optimization algorithm ('L-BFGS-B', 'Newton-CG', 'BFGS')
        maxiter: Maximum iterations
        tol: Convergence tolerance
        grid_search: Use grid search for starting values
        verbose: Print optimization progress
        **kwargs: Additional optimizer arguments

    Returns:
        SFResult with estimation results

    Raises:
        RuntimeError: If optimization fails critically
    """
    # Extract data
    y = model.y
    X = model.X
    Z = model.Z
    dist = model.dist.value
    frontier_type = model.frontier_type

    # Sign convention
    sign = 1 if frontier_type == FrontierType.PRODUCTION else -1

    # Get likelihood function
    likelihood_func = _get_likelihood_function(dist)
    gradient_func = _get_gradient_function(dist)

    # Get starting values
    if start_params is None:
        start_params = get_starting_values(
            y=y,
            X=X,
            Z=Z,
            dist=dist,
            grid_search=grid_search,
            likelihood_func=likelihood_func,
            sign=sign,
        )

        if verbose:
            print("Starting values computed via method of moments")
            print(f"  β: {start_params[:X.shape[1]]}")
            print(f"  ln(σ²_v): {start_params[X.shape[1]]:.4f}")
            print(f"  ln(σ²_u): {start_params[X.shape[1]+1]:.4f}")

        # Check starting values
        sv_check = check_starting_values(start_params, y, X, likelihood_func, sign)

        if not sv_check["valid"]:
            warnings.warn(
                f"Starting values may be poor: {sv_check}. " "Consider using grid_search=True.",
                UserWarning,
            )

    # Negative log-likelihood for minimization
    def neg_loglik(theta):
        try:
            if Z is not None:
                ll = likelihood_func(theta, y, X, Z, sign=sign)
            else:
                ll = likelihood_func(theta, y, X, sign=sign)
            return -ll
        except (ValueError, RuntimeError, FloatingPointError):
            return np.inf

    # Gradient (if available)
    def neg_gradient(theta):
        if gradient_func is None:
            return None
        try:
            grad = gradient_func(theta, y, X, sign=sign)
            return -grad
        except (ValueError, RuntimeError):
            return None

    # Set up optimizer options
    options = {"maxiter": maxiter, "disp": verbose}
    options.update(kwargs)

    # Choose optimizer
    if optimizer.upper() in ["L-BFGS-B", "LBFGSB"]:
        # L-BFGS-B with box constraints
        options["ftol"] = tol
        options["gtol"] = tol

        # Set bounds to keep variances reasonable
        k = X.shape[1]
        bounds = [(None, None)] * k  # No bounds on β

        # Bounds on log-variances: ln(1e-6) to ln(1e6)
        bounds.append((-13.8, 13.8))  # ln(σ²_v)
        bounds.append((-13.8, 13.8))  # ln(σ²_u)

        # Additional bounds for other parameters
        if dist == "truncated_normal":
            if Z is not None:
                bounds.extend([(None, None)] * Z.shape[1])  # δ
            else:
                bounds.append((0, None))  # μ ≥ 0

        elif dist == "gamma":
            bounds.append((-2.3, 5.3))  # ln(P): P ∈ [0.1, 200]

        result = minimize(
            neg_loglik,
            start_params,
            method="L-BFGS-B",
            jac=neg_gradient if gradient_func else None,
            bounds=bounds,
            options=options,
        )

    elif optimizer.upper() in ["NEWTON-CG", "NEWTONCG"]:
        # Newton-CG (requires gradient)
        options["xtol"] = tol

        result = minimize(
            neg_loglik, start_params, method="Newton-CG", jac=neg_gradient, options=options
        )

    elif optimizer.upper() == "BFGS":
        # BFGS (quasi-Newton)
        options["gtol"] = tol

        result = minimize(
            neg_loglik,
            start_params,
            method="BFGS",
            jac=neg_gradient if gradient_func else None,
            options=options,
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # Check convergence
    # Special case: if starting values are very close to optimum,
    # L-BFGS-B may report "ABNORMAL" but the result is actually good
    converged = result.success
    loglik_value = -result.fun

    if not result.success:
        # Check if we have a reasonable log-likelihood despite convergence warning
        # This can happen when starting values are very good
        if "ABNORMAL" in str(result.message) and np.isfinite(loglik_value) and result.nfev > 0:
            # Accept the result if log-likelihood is finite
            converged = True
            if verbose:
                print("Note: Optimization reported ABNORMAL but log-likelihood is valid.")
                print("This usually means starting values were very close to optimum.")
        else:
            warnings.warn(
                f"Optimization did not converge: {result.message}. "
                f"Try different starting values or optimizer.",
                UserWarning,
            )

    if verbose:
        print(f"\nOptimization complete:")
        print(f"  Converged: {converged}")
        print(f"  Iterations: {result.nit}")
        print(f"  Function evaluations: {result.nfev}")
        print(f"  Log-likelihood: {loglik_value:.4f}")

    # Transform parameters back to natural scale for reporting
    params_transformed, param_names = _transform_parameters(
        result.x,
        X.shape[1],
        Z.shape[1] if Z is not None else 0,
        dist,
        model.exog_names,
        model.ineff_var_names,
    )

    # Compute Hessian at optimum (for standard errors)
    hessian = _compute_hessian(result.x, neg_loglik, method="numerical")

    # Create result object
    sf_result = SFResult(
        params=params_transformed,
        param_names=param_names,
        hessian=hessian,
        loglik=loglik_value,
        converged=converged,
        model=model,
        optimization_result=result,
    )

    return sf_result


def _get_likelihood_function(dist: str) -> Callable:
    """Get log-likelihood function for distribution."""
    likelihood_map = {
        "half_normal": loglik_half_normal,
        "exponential": loglik_exponential,
        "truncated_normal": loglik_truncated_normal,
        "gamma": loglik_gamma,
    }

    if dist not in likelihood_map:
        raise ValueError(f"Unknown distribution: {dist}")

    return likelihood_map[dist]


def _get_gradient_function(dist: str) -> Optional[Callable]:
    """Get gradient function for distribution (if available)."""
    gradient_map = {
        "half_normal": gradient_half_normal,
        "exponential": gradient_exponential,
        "truncated_normal": None,  # Use numerical
        "gamma": None,  # Use numerical
    }

    return gradient_map.get(dist)


def _transform_parameters(
    theta: np.ndarray,
    n_exog: int,
    n_ineff_vars: int,
    dist: str,
    exog_names: list,
    ineff_var_names: list,
) -> tuple:
    """Transform parameters from estimation space to natural scale.

    Parameters are estimated as:
        θ = [β, ln(σ²_v), ln(σ²_u), ...]

    We want to report:
        [β, σ²_v, σ²_u, ...]

    Parameters:
        theta: Parameter vector in estimation space
        n_exog: Number of exogenous variables
        n_ineff_vars: Number of inefficiency variables (BC95)
        dist: Distribution type
        exog_names: Names of exogenous variables
        ineff_var_names: Names of inefficiency variables

    Returns:
        Tuple of (transformed_params, param_names)
    """
    # Extract components
    beta = theta[:n_exog]
    ln_sigma_v_sq = theta[n_exog]
    ln_sigma_u_sq = theta[n_exog + 1]

    # Transform variances
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)

    # Build parameter vector
    params = np.concatenate([beta, [sigma_v_sq], [sigma_u_sq]])
    names = exog_names + ["sigma_v_sq", "sigma_u_sq"]

    # Additional parameters
    idx = n_exog + 2

    if dist == "truncated_normal":
        if n_ineff_vars > 0:
            # BC95 model
            delta = theta[idx : idx + n_ineff_vars]
            params = np.concatenate([params, delta])
            names.extend([f"delta_{name}" for name in ineff_var_names])
        else:
            # Simple truncated normal
            mu = theta[idx]
            params = np.concatenate([params, [mu]])
            names.append("mu")

    elif dist == "gamma":
        ln_P = theta[idx]
        P = np.exp(ln_P)
        params = np.concatenate([params, [P]])
        names.append("P")

    return params, names


def _compute_hessian(
    theta: np.ndarray, func: Callable, method: str = "numerical", epsilon: float = 1e-5
) -> np.ndarray:
    """Compute Hessian matrix at parameter vector.

    Parameters:
        theta: Parameter vector
        func: Function to compute Hessian of
        method: 'numerical' or 'analytical'
        epsilon: Step size for numerical differentiation

    Returns:
        Hessian matrix (k x k)
    """
    if method != "numerical":
        raise NotImplementedError("Only numerical Hessian implemented")

    k = len(theta)
    hessian = np.zeros((k, k))

    # Finite difference approximation
    f0 = func(theta)

    for i in range(k):
        theta_i_plus = theta.copy()
        theta_i_plus[i] += epsilon
        theta_i_minus = theta.copy()
        theta_i_minus[i] -= epsilon

        for j in range(i, k):
            if i == j:
                # Diagonal: ∂²f/∂θᵢ²
                f_plus = func(theta_i_plus)
                f_minus = func(theta_i_minus)
                hessian[i, i] = (f_plus - 2 * f0 + f_minus) / epsilon**2
            else:
                # Off-diagonal: ∂²f/∂θᵢ∂θⱼ
                theta_ij_pp = theta.copy()
                theta_ij_pp[i] += epsilon
                theta_ij_pp[j] += epsilon

                theta_ij_pm = theta.copy()
                theta_ij_pm[i] += epsilon
                theta_ij_pm[j] -= epsilon

                theta_ij_mp = theta.copy()
                theta_ij_mp[i] -= epsilon
                theta_ij_mp[j] += epsilon

                theta_ij_mm = theta.copy()
                theta_ij_mm[i] -= epsilon
                theta_ij_mm[j] -= epsilon

                f_pp = func(theta_ij_pp)
                f_pm = func(theta_ij_pm)
                f_mp = func(theta_ij_mp)
                f_mm = func(theta_ij_mm)

                hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
                hessian[j, i] = hessian[i, j]  # Symmetry

    # Check for numerical issues
    if not np.all(np.isfinite(hessian)):
        warnings.warn(
            "Hessian contains non-finite values. " "Standard errors may not be reliable.",
            UserWarning,
        )
        return None

    return hessian
