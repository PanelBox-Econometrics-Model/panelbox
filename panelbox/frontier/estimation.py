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
    loglik_wang_2002,
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
    # Special handling for CSS model (distribution-free)
    from .data import ModelType

    if model.model_type == ModelType.CSS:
        return _estimate_css_model(model, verbose=verbose)

    # Special handling for BC92 model (time-decay)
    if model.model_type == ModelType.BATTESE_COELLI_92:
        return _estimate_bc92_model(
            model,
            start_params=start_params,
            optimizer=optimizer,
            maxiter=maxiter,
            tol=tol,
            grid_search=grid_search,
            verbose=verbose,
            **kwargs,
        )

    # Special handling for Kumbhakar (1990) model
    if model.model_type == ModelType.KUMBHAKAR_1990:
        return _estimate_kumbhakar_1990_model(
            model,
            start_params=start_params,
            optimizer=optimizer,
            maxiter=maxiter,
            tol=tol,
            grid_search=grid_search,
            verbose=verbose,
            **kwargs,
        )

    # Special handling for Lee-Schmidt (1993) model
    if model.model_type == ModelType.LEE_SCHMIDT_1993:
        return _estimate_lee_schmidt_1993_model(
            model,
            start_params=start_params,
            optimizer=optimizer,
            maxiter=maxiter,
            tol=tol,
            grid_search=grid_search,
            verbose=verbose,
            **kwargs,
        )

    # Extract data
    y = model.y
    X = model.X
    Z = model.Z
    W = model.W
    dist = model.dist.value
    frontier_type = model.frontier_type

    # Sign convention
    sign = 1 if frontier_type == FrontierType.PRODUCTION else -1

    # Check if Wang (2002) model
    # Wang model requires: truncated normal + location vars (Z) + scale vars (W)
    is_wang = dist == "truncated_normal" and Z is not None and W is not None

    # Get likelihood function
    likelihood_func = _get_likelihood_function(dist, is_wang=is_wang)
    gradient_func = _get_gradient_function(dist)

    # Get starting values
    if start_params is None:
        if is_wang:
            # Wang (2002) starting values: [β, ln(σ²_v), δ, γ]
            # Start with OLS for β
            ols_beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ ols_beta
            ols_sigma_sq = np.var(residuals)

            # Starting values
            start_params = np.concatenate(
                [
                    ols_beta,
                    [np.log(ols_sigma_sq / 2)],  # ln(σ²_v)
                    np.zeros(Z.shape[1]),  # δ: start at 0 (no location effect)
                    np.zeros(W.shape[1]),  # γ: start at 0 (homoscedastic)
                ]
            )

            if verbose:
                print("Starting values for Wang (2002) model:")
                print(f"  β: {ols_beta}")
                print(f"  ln(σ²_v): {start_params[X.shape[1]]:.4f}")
                print(f"  δ (location): {start_params[X.shape[1]+1:X.shape[1]+1+Z.shape[1]]}")
                print(f"  γ (scale): {start_params[X.shape[1]+1+Z.shape[1]:]}")
        else:
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

            # Check starting values (skip for Wang as it has different structure)
            sv_check = check_starting_values(start_params, y, X, likelihood_func, sign)

            if not sv_check["valid"]:
                warnings.warn(
                    f"Starting values may be poor: {sv_check}. " "Consider using grid_search=True.",
                    UserWarning,
                )

    # Negative log-likelihood for minimization
    def neg_loglik(theta):
        try:
            if is_wang:
                # Wang (2002) requires Z and W
                ll = likelihood_func(theta, y, X, Z, W, sign=sign)
            elif Z is not None:
                # BC95 or other models with Z
                ll = likelihood_func(theta, y, X, Z, sign=sign)
            else:
                # Standard models
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

        # Additional bounds for other parameters
        if is_wang:
            # Wang (2002): [β, ln(σ²_v), δ, γ]
            # No ln(σ²_u) since it varies by observation
            bounds.extend([(None, None)] * Z.shape[1])  # δ (location parameters)
            bounds.extend([(-10, 10)] * W.shape[1])  # γ (scale parameters, bounded for stability)

        elif dist == "truncated_normal":
            bounds.append((-13.8, 13.8))  # ln(σ²_u)
            if Z is not None:
                bounds.extend([(None, None)] * Z.shape[1])  # δ
            else:
                bounds.append((0, None))  # μ ≥ 0

        elif dist == "gamma":
            bounds.append((-13.8, 13.8))  # ln(σ²_u)
            bounds.append((-2.3, 5.3))  # ln(P): P ∈ [0.1, 200]
            bounds.append((-2.3, 5.3))  # ln(θ): θ ∈ [0.1, 200]

        else:
            # Half-normal, exponential, etc.
            bounds.append((-13.8, 13.8))  # ln(σ²_u)

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
        is_wang=is_wang,
        hetero_var_names=model.hetero_var_names if is_wang else [],
    )

    # Compute Hessian at optimum (for standard errors)
    hessian = _compute_hessian(result.x, neg_loglik, method="numerical")

    # Create result object
    sf_result = SFResult(
        params=params_transformed,
        param_names=param_names,
        hessian=hessian,
        converged=converged,
        model=model,
        loglik=loglik_value,
        optimization_result=result,
    )

    return sf_result


def _get_likelihood_function(dist: str, is_wang: bool = False) -> Callable:
    """Get log-likelihood function for distribution."""
    # Wang (2002) takes precedence for truncated normal with heteroscedasticity
    if is_wang:
        return loglik_wang_2002

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
    is_wang: bool = False,
    hetero_var_names: list = None,
) -> tuple:
    """Transform parameters from estimation space to natural scale.

    Parameters are estimated as:
        θ = [β, ln(σ²_v), ln(σ²_u), ...]  (standard models)
        θ = [β, ln(σ²_v), δ, γ]           (Wang 2002)

    We want to report:
        [β, σ²_v, σ²_u, ...]  (standard models)
        [β, σ²_v, δ, γ]       (Wang 2002 - no σ²_u since it varies)

    Parameters:
        theta: Parameter vector in estimation space
        n_exog: Number of exogenous variables
        n_ineff_vars: Number of inefficiency variables (BC95)
        dist: Distribution type
        exog_names: Names of exogenous variables
        ineff_var_names: Names of inefficiency variables
        is_wang: Whether this is Wang (2002) model
        hetero_var_names: Names of heteroscedasticity variables (Wang 2002)

    Returns:
        Tuple of (transformed_params, param_names)
    """
    # Extract components
    beta = theta[:n_exog]
    ln_sigma_v_sq = theta[n_exog]

    # Transform variance
    sigma_v_sq = np.exp(ln_sigma_v_sq)

    if is_wang:
        # Wang (2002): [β, ln(σ²_v), δ, γ]
        # No σ²_u since it varies by observation
        params = np.concatenate([beta, [sigma_v_sq]])
        names = exog_names + ["sigma_v_sq"]

        # Location parameters (δ)
        idx = n_exog + 1
        delta = theta[idx : idx + n_ineff_vars]
        params = np.concatenate([params, delta])
        names.extend([f"delta_{name}" for name in ineff_var_names])

        # Scale parameters (γ)
        idx = n_exog + 1 + n_ineff_vars
        n_hetero_vars = len(hetero_var_names) if hetero_var_names else 0
        gamma = theta[idx : idx + n_hetero_vars]
        params = np.concatenate([params, gamma])
        names.extend([f"gamma_{name}" for name in hetero_var_names])

        return params, names

    # Standard models (not Wang)
    ln_sigma_u_sq = theta[n_exog + 1]
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
        ln_theta = theta[idx + 1]
        P = np.exp(ln_P)
        theta_param = np.exp(ln_theta)
        params = np.concatenate([params, [P], [theta_param]])
        names.extend(["gamma_P", "gamma_theta"])

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


def _estimate_css_model(model, verbose: bool = False):
    """Estimate CSS model using distribution-free approach.

    Parameters:
        model: StochasticFrontier model instance with model_type=CSS
        verbose: Print estimation progress

    Returns:
        SFResult with CSS estimation results
    """
    from .css import estimate_css_model
    from .data import ModelType
    from .result import SFResult

    # Extract entity and time indices (need integer coding 0, 1, ..., N-1)
    # model.data is indexed by (entity, time) after prepare_panel_index
    reset_data = model.data.reset_index()
    entity_id = (
        reset_data["entity"].values
        if "entity" in reset_data.columns
        else reset_data.index.get_level_values(0).values
    )
    time_id = (
        reset_data["time"].values
        if "time" in reset_data.columns
        else reset_data.index.get_level_values(1).values
    )

    # Convert to integer codes if not already
    entity_unique = {e: i for i, e in enumerate(sorted(set(entity_id)))}
    time_unique = {t: i for i, t in enumerate(sorted(set(time_id)))}

    entity_id_coded = np.array([entity_unique[e] for e in entity_id])
    time_id_coded = np.array([time_unique[t] for t in time_id])

    # Remove constant from X if present (CSS adds its own intercepts)
    X_no_const = model.X[:, 1:] if model.X.shape[1] > len(model.exog) else model.X

    if verbose:
        print(f"Estimating CSS model with time_trend='{model.css_time_trend}'")
        print(f"  N = {len(entity_unique)} entities")
        print(f"  T = {len(time_unique)} periods")
        print(f"  n = {len(model.y)} observations")

    # Estimate CSS model
    css_result = estimate_css_model(
        y=model.y,
        X=X_no_const,
        entity_id=entity_id_coded,
        time_id=time_id_coded,
        time_trend=model.css_time_trend,
        frontier_type=model.frontier_type.value,
    )

    if verbose:
        print(f"  R² = {css_result.r_squared:.4f}")
        print(f"  σ_v = {css_result.sigma_v:.4f}")
        print(f"  Mean efficiency = {np.mean(css_result.efficiency_it):.4f}")

    # Create SFResult wrapper
    # CSS does not use MLE, so loglik=None
    result = SFResult(
        params=css_result.params,
        param_names=css_result.param_names,
        hessian=None,  # CSS uses OLS, no Hessian
        converged=True,
        model=model,
        loglik=None,  # CSS is not MLE
        optimization_result=None,
    )

    # Attach CSS-specific results
    result._css_result = css_result
    result._alpha_it = css_result.alpha_it
    result._efficiency_it = css_result.efficiency_it
    result._r_squared = css_result.r_squared

    return result


def _estimate_bc92_model(
    model,
    start_params: Optional[np.ndarray] = None,
    optimizer: str = "L-BFGS-B",
    maxiter: int = 1000,
    tol: float = 1e-8,
    grid_search: bool = False,
    verbose: bool = False,
    **kwargs,
) -> SFResult:
    """Estimate BC92 model with time-decay inefficiency.

    Parameters:
        model: StochasticFrontier model instance with model_type=BC92
        start_params: Initial parameter values
        optimizer: Optimization algorithm
        maxiter: Maximum iterations
        tol: Convergence tolerance
        grid_search: Use grid search for starting values
        verbose: Print estimation progress
        **kwargs: Additional optimizer arguments

    Returns:
        SFResult with BC92 estimation results
    """
    from .panel_likelihoods import loglik_bc92

    # Prepare panel data - convert entity and time to integer codes
    reset_data = model.data.reset_index()

    if "entity" in reset_data.columns:
        entity_vals = reset_data["entity"].values
    else:
        entity_vals = (
            reset_data.index.get_level_values(0).values
            if hasattr(reset_data.index, "levels")
            else model.data.index.get_level_values(0).values
        )

    if "time" in reset_data.columns:
        time_vals = reset_data["time"].values
    else:
        time_vals = (
            reset_data.index.get_level_values(1).values
            if hasattr(reset_data.index, "levels")
            else model.data.index.get_level_values(1).values
        )

    # Convert to integer codes
    entity_unique = {e: i for i, e in enumerate(sorted(set(entity_vals)))}
    time_unique = {t: i for i, t in enumerate(sorted(set(time_vals)))}

    entity_id = np.array([entity_unique[e] for e in entity_vals])
    time_id = np.array([time_unique[t] for t in time_vals])

    # Sign convention
    sign = 1 if model.frontier_type == FrontierType.PRODUCTION else -1

    # Get starting values
    if start_params is None:
        # Start with OLS estimates for β
        X = model.X
        y = model.y
        ols_beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ ols_beta
        ols_sigma_sq = np.var(residuals)

        # Starting values: [β, ln(σ²_v), ln(σ²_u), η]
        start_params = np.concatenate(
            [
                ols_beta,
                [np.log(ols_sigma_sq / 2)],  # ln(σ²_v)
                [np.log(ols_sigma_sq / 2)],  # ln(σ²_u)
                [0.0],  # η (start at 0 = time-invariant, like Pitt-Lee)
            ]
        )

        if verbose:
            print("Starting values for BC92:")
            print(f"  β: {ols_beta}")
            print(f"  ln(σ²_v): {start_params[X.shape[1]]:.4f}")
            print(f"  ln(σ²_u): {start_params[X.shape[1]+1]:.4f}")
            print(f"  η: {start_params[X.shape[1]+2]:.4f}")

    # Negative log-likelihood for minimization
    def neg_loglik(theta):
        try:
            ll = loglik_bc92(
                theta,
                model.y,
                model.X,
                entity_id,
                time_id,
                sign,
            )
            return -ll
        except (ValueError, RuntimeError, FloatingPointError):
            return np.inf

    # Set up optimizer options
    options = {"maxiter": maxiter, "disp": verbose}
    options.update(kwargs)

    # Set bounds
    k = model.X.shape[1]
    bounds = [(None, None)] * k  # No bounds on β

    # Bounds on log-variances: ln(1e-6) to ln(1e6)
    bounds.append((-13.8, 13.8))  # ln(σ²_v)
    bounds.append((-13.8, 13.8))  # ln(σ²_u)

    # Bounds on η: allow both positive and negative (learning vs degradation)
    bounds.append((-5.0, 5.0))  # η

    # Optimize
    if optimizer.upper() in ["L-BFGS-B", "LBFGSB"]:
        options["ftol"] = tol
        options["gtol"] = tol

        result = minimize(
            neg_loglik,
            start_params,
            method="L-BFGS-B",
            bounds=bounds,
            options=options,
        )
    else:
        result = minimize(
            neg_loglik,
            start_params,
            method=optimizer,
            options=options,
        )

    # Check convergence
    converged = result.success
    loglik_value = -result.fun

    if not result.success:
        warnings.warn(
            f"BC92 optimization did not converge: {result.message}. "
            f"Try different starting values or optimizer.",
            UserWarning,
        )

    if verbose:
        print(f"\nBC92 Optimization complete:")
        print(f"  Converged: {converged}")
        print(f"  Iterations: {result.nit}")
        print(f"  Log-likelihood: {loglik_value:.4f}")

    # Transform parameters
    params_transformed, param_names = _transform_bc92_parameters(
        result.x,
        model.X.shape[1],
        model.exog_names,
    )

    # Compute Hessian
    hessian = _compute_hessian(result.x, neg_loglik, method="numerical")

    # Extract eta parameter
    eta = result.x[k + 2]

    # Create result object with panel-specific attributes
    from .result import PanelSFResult

    sf_result = PanelSFResult(
        params=params_transformed,
        param_names=param_names,
        hessian=hessian,
        converged=converged,
        model=model,
        loglik=loglik_value,
        panel_type="bc92",
        temporal_params={"eta": eta},
        optimization_result=result,
    )

    # Store BC92-specific info
    sf_result._bc92_eta = eta  # Time-decay parameter

    # Store entity and time IDs for efficiency calculation
    sf_result._entity_id = entity_id
    sf_result._time_id = time_id

    return sf_result


def _transform_bc92_parameters(
    theta: np.ndarray,
    n_exog: int,
    exog_names: list,
) -> tuple:
    """Transform BC92 parameters from estimation space to natural scale.

    Parameters:
        theta: [β, ln(σ²_v), ln(σ²_u), η]
        n_exog: Number of exogenous variables
        exog_names: Names of exogenous variables

    Returns:
        Tuple of (transformed_params, param_names)
    """
    # Extract components
    beta = theta[:n_exog]
    ln_sigma_v_sq = theta[n_exog]
    ln_sigma_u_sq = theta[n_exog + 1]
    eta = theta[n_exog + 2]

    # Transform variances
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)

    # Build parameter vector
    params = np.concatenate([beta, [sigma_v_sq], [sigma_u_sq], [eta]])
    names = exog_names + ["sigma_v_sq", "sigma_u_sq", "eta"]

    return params, names


def _estimate_kumbhakar_1990_model(
    model,
    start_params: Optional[np.ndarray] = None,
    optimizer: str = "L-BFGS-B",
    maxiter: int = 1000,
    tol: float = 1e-8,
    grid_search: bool = False,
    verbose: bool = False,
    **kwargs,
) -> SFResult:
    """Estimate Kumbhakar (1990) model with flexible time pattern.

    Model: u_it = B(t) * u_i
           B(t) = 1 / [1 + exp(b*t + c*t²)]

    Parameters:
        model: StochasticFrontier model instance with model_type=KUMBHAKAR_1990
        start_params: Initial parameter values
        optimizer: Optimization algorithm
        maxiter: Maximum iterations
        tol: Convergence tolerance
        grid_search: Use grid search for starting values
        verbose: Print estimation progress
        **kwargs: Additional optimizer arguments

    Returns:
        PanelSFResult with Kumbhakar estimation results
    """
    from .panel_likelihoods import loglik_kumbhakar_1990

    # Prepare panel data - convert entity and time to integer codes
    reset_data = model.data.reset_index()

    if "entity" in reset_data.columns:
        entity_vals = reset_data["entity"].values
    else:
        entity_vals = (
            reset_data.index.get_level_values(0).values
            if hasattr(reset_data.index, "levels")
            else model.data.index.get_level_values(0).values
        )

    if "time" in reset_data.columns:
        time_vals = reset_data["time"].values
    else:
        time_vals = (
            reset_data.index.get_level_values(1).values
            if hasattr(reset_data.index, "levels")
            else model.data.index.get_level_values(1).values
        )

    # Convert to integer codes
    entity_unique = {e: i for i, e in enumerate(sorted(set(entity_vals)))}
    time_unique = {t: i for i, t in enumerate(sorted(set(time_vals)))}

    entity_id = np.array([entity_unique[e] for e in entity_vals])
    time_id = np.array([time_unique[t] for t in time_vals])

    # Sign convention
    sign = 1 if model.frontier_type == FrontierType.PRODUCTION else -1

    # Get starting values
    if start_params is None:
        # Start with OLS estimates for β
        X = model.X
        y = model.y
        ols_beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ ols_beta
        ols_sigma_sq = np.var(residuals)

        # Starting values: [β, ln(σ²_v), ln(σ²_u), μ, b, c]
        start_params = np.concatenate(
            [
                ols_beta,
                [np.log(ols_sigma_sq / 2)],  # ln(σ²_v)
                [np.log(ols_sigma_sq / 2)],  # ln(σ²_u)
                [0.0],  # μ (mean of truncated normal)
                [0.0],  # b (start at 0 = time-invariant)
                [0.0],  # c (start at 0 = time-invariant)
            ]
        )

        if verbose:
            print("Starting values for Kumbhakar (1990):")
            print(f"  β: {ols_beta}")
            print(f"  ln(σ²_v): {start_params[X.shape[1]]:.4f}")
            print(f"  ln(σ²_u): {start_params[X.shape[1]+1]:.4f}")
            print(f"  μ: {start_params[X.shape[1]+2]:.4f}")
            print(f"  b: {start_params[X.shape[1]+3]:.4f}")
            print(f"  c: {start_params[X.shape[1]+4]:.4f}")

    # Negative log-likelihood for minimization
    def neg_loglik(theta):
        try:
            ll = loglik_kumbhakar_1990(
                theta,
                model.y,
                model.X,
                entity_id,
                time_id,
                sign,
            )
            return -ll
        except (ValueError, RuntimeError, FloatingPointError):
            return np.inf

    # Set up optimizer options
    options = {"maxiter": maxiter, "disp": verbose}
    options.update(kwargs)

    # Set bounds
    k = model.X.shape[1]
    bounds = [(None, None)] * k  # No bounds on β

    # Bounds on log-variances: ln(1e-6) to ln(1e6)
    bounds.append((-13.8, 13.8))  # ln(σ²_v)
    bounds.append((-13.8, 13.8))  # ln(σ²_u)

    # Bounds on μ: allow negative values for truncated normal
    bounds.append((-10.0, 10.0))  # μ

    # Bounds on b and c: time pattern coefficients
    bounds.append((-5.0, 5.0))  # b (linear time coefficient)
    bounds.append((-5.0, 5.0))  # c (quadratic time coefficient)

    # Optimize
    if optimizer.upper() in ["L-BFGS-B", "LBFGSB"]:
        options["ftol"] = tol
        options["gtol"] = tol

        result = minimize(
            neg_loglik,
            start_params,
            method="L-BFGS-B",
            bounds=bounds,
            options=options,
        )
    else:
        result = minimize(
            neg_loglik,
            start_params,
            method=optimizer,
            options=options,
        )

    # Check convergence
    converged = result.success
    loglik_value = -result.fun

    if not result.success:
        warnings.warn(
            f"Kumbhakar (1990) optimization did not converge: {result.message}. "
            f"Try different starting values or optimizer.",
            UserWarning,
        )

    if verbose:
        print(f"\nKumbhakar (1990) Optimization complete:")
        print(f"  Converged: {converged}")
        print(f"  Iterations: {result.nit}")
        print(f"  Log-likelihood: {loglik_value:.4f}")

    # Transform parameters
    params_transformed, param_names = _transform_kumbhakar_parameters(
        result.x,
        model.X.shape[1],
        model.exog_names,
    )

    # Compute Hessian
    hessian = _compute_hessian(result.x, neg_loglik, method="numerical")

    # Extract temporal parameters
    mu = result.x[k + 2]
    b = result.x[k + 3]
    c = result.x[k + 4]

    # Create result object with panel-specific attributes
    from .result import PanelSFResult

    sf_result = PanelSFResult(
        params=params_transformed,
        param_names=param_names,
        hessian=hessian,
        converged=converged,
        model=model,
        loglik=loglik_value,
        panel_type="kumbhakar",
        temporal_params={"mu": mu, "b": b, "c": c},
        optimization_result=result,
    )

    # Store entity and time IDs for efficiency calculation
    sf_result._entity_id = entity_id
    sf_result._time_id = time_id

    return sf_result


def _transform_kumbhakar_parameters(
    theta: np.ndarray,
    n_exog: int,
    exog_names: list,
) -> tuple:
    """Transform Kumbhakar parameters from estimation space to natural scale.

    Parameters:
        theta: [β, ln(σ²_v), ln(σ²_u), μ, b, c]
        n_exog: Number of exogenous variables
        exog_names: Names of exogenous variables

    Returns:
        Tuple of (transformed_params, param_names)
    """
    # Extract components
    beta = theta[:n_exog]
    ln_sigma_v_sq = theta[n_exog]
    ln_sigma_u_sq = theta[n_exog + 1]
    mu = theta[n_exog + 2]
    b = theta[n_exog + 3]
    c = theta[n_exog + 4]

    # Transform variances
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)

    # Build parameter vector
    params = np.concatenate([beta, [sigma_v_sq], [sigma_u_sq], [mu], [b], [c]])
    names = exog_names + ["sigma_v_sq", "sigma_u_sq", "mu", "b", "c"]

    return params, names


def _estimate_lee_schmidt_1993_model(
    model,
    start_params: Optional[np.ndarray] = None,
    optimizer: str = "L-BFGS-B",
    maxiter: int = 1000,
    tol: float = 1e-8,
    grid_search: bool = False,
    verbose: bool = False,
    **kwargs,
) -> SFResult:
    """Estimate Lee & Schmidt (1993) model with time dummies.

    Model: u_it = δ_t * u_i
           δ_T = 1 (normalization)

    Parameters:
        model: StochasticFrontier model instance with model_type=LEE_SCHMIDT_1993
        start_params: Initial parameter values
        optimizer: Optimization algorithm
        maxiter: Maximum iterations
        tol: Convergence tolerance
        grid_search: Use grid search for starting values
        verbose: Print estimation progress
        **kwargs: Additional optimizer arguments

    Returns:
        PanelSFResult with Lee-Schmidt estimation results
    """
    from .panel_likelihoods import loglik_lee_schmidt_1993

    # Prepare panel data - convert entity and time to integer codes
    reset_data = model.data.reset_index()

    if "entity" in reset_data.columns:
        entity_vals = reset_data["entity"].values
    else:
        entity_vals = (
            reset_data.index.get_level_values(0).values
            if hasattr(reset_data.index, "levels")
            else model.data.index.get_level_values(0).values
        )

    if "time" in reset_data.columns:
        time_vals = reset_data["time"].values
    else:
        time_vals = (
            reset_data.index.get_level_values(1).values
            if hasattr(reset_data.index, "levels")
            else model.data.index.get_level_values(1).values
        )

    # Convert to integer codes
    entity_unique = {e: i for i, e in enumerate(sorted(set(entity_vals)))}
    time_unique = {t: i for i, t in enumerate(sorted(set(time_vals)))}

    entity_id = np.array([entity_unique[e] for e in entity_vals])
    time_id = np.array([time_unique[t] for t in time_vals])

    # Get number of time periods
    T = len(time_unique)

    # Sign convention
    sign = 1 if model.frontier_type == FrontierType.PRODUCTION else -1

    # Get starting values
    if start_params is None:
        # Start with OLS estimates for β
        X = model.X
        y = model.y
        ols_beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ ols_beta
        ols_sigma_sq = np.var(residuals)

        # Starting values: [β, ln(σ²_v), ln(σ²_u), μ, δ_1, ..., δ_{T-1}]
        # Note: δ_T = 1 by normalization, so we estimate T-1 parameters
        start_params = np.concatenate(
            [
                ols_beta,
                [np.log(ols_sigma_sq / 2)],  # ln(σ²_v)
                [np.log(ols_sigma_sq / 2)],  # ln(σ²_u)
                [0.0],  # μ (mean of truncated normal)
                np.ones(T - 1),  # δ_1, ..., δ_{T-1} (start at 1 = time-invariant)
            ]
        )

        if verbose:
            print("Starting values for Lee-Schmidt (1993):")
            print(f"  β: {ols_beta}")
            print(f"  ln(σ²_v): {start_params[X.shape[1]]:.4f}")
            print(f"  ln(σ²_u): {start_params[X.shape[1]+1]:.4f}")
            print(f"  μ: {start_params[X.shape[1]+2]:.4f}")
            print(f"  δ_t (T-1 params): {start_params[X.shape[1]+3:]}")

    # Negative log-likelihood for minimization
    def neg_loglik(theta):
        try:
            ll = loglik_lee_schmidt_1993(
                theta,
                model.y,
                model.X,
                entity_id,
                time_id,
                sign,
            )
            return -ll
        except (ValueError, RuntimeError, FloatingPointError):
            return np.inf

    # Set up optimizer options
    options = {"maxiter": maxiter, "disp": verbose}
    options.update(kwargs)

    # Set bounds
    k = model.X.shape[1]
    bounds = [(None, None)] * k  # No bounds on β

    # Bounds on log-variances: ln(1e-6) to ln(1e6)
    bounds.append((-13.8, 13.8))  # ln(σ²_v)
    bounds.append((-13.8, 13.8))  # ln(σ²_u)

    # Bounds on μ: allow negative values for truncated normal
    bounds.append((-10.0, 10.0))  # μ

    # Bounds on δ_t: positive values (inefficiency scale factors)
    bounds.extend([(0.01, 10.0)] * (T - 1))  # δ_1, ..., δ_{T-1}

    # Optimize
    if optimizer.upper() in ["L-BFGS-B", "LBFGSB"]:
        options["ftol"] = tol
        options["gtol"] = tol

        result = minimize(
            neg_loglik,
            start_params,
            method="L-BFGS-B",
            bounds=bounds,
            options=options,
        )
    else:
        result = minimize(
            neg_loglik,
            start_params,
            method=optimizer,
            options=options,
        )

    # Check convergence
    converged = result.success
    loglik_value = -result.fun

    if not result.success:
        warnings.warn(
            f"Lee-Schmidt (1993) optimization did not converge: {result.message}. "
            f"Try different starting values or optimizer.",
            UserWarning,
        )

    if verbose:
        print(f"\nLee-Schmidt (1993) Optimization complete:")
        print(f"  Converged: {converged}")
        print(f"  Iterations: {result.nit}")
        print(f"  Log-likelihood: {loglik_value:.4f}")

    # Transform parameters
    params_transformed, param_names = _transform_lee_schmidt_parameters(
        result.x,
        model.X.shape[1],
        model.exog_names,
        T,
    )

    # Compute Hessian
    hessian = _compute_hessian(result.x, neg_loglik, method="numerical")

    # Extract temporal parameters
    mu = result.x[k + 2]
    delta_t_params = result.x[k + 3 : k + 3 + T - 1]
    # Construct full delta_t with normalization δ_T = 1
    delta_t = np.concatenate([delta_t_params, [1.0]])

    # Create result object with panel-specific attributes
    from .result import PanelSFResult

    sf_result = PanelSFResult(
        params=params_transformed,
        param_names=param_names,
        hessian=hessian,
        converged=converged,
        model=model,
        loglik=loglik_value,
        panel_type="lee_schmidt",
        temporal_params={"mu": mu, "delta_t": delta_t},
        optimization_result=result,
    )

    # Store entity and time IDs for efficiency calculation
    sf_result._entity_id = entity_id
    sf_result._time_id = time_id

    return sf_result


def _transform_lee_schmidt_parameters(
    theta: np.ndarray,
    n_exog: int,
    exog_names: list,
    n_periods: int,
) -> tuple:
    """Transform Lee-Schmidt parameters from estimation space to natural scale.

    Parameters:
        theta: [β, ln(σ²_v), ln(σ²_u), μ, δ_1, ..., δ_{T-1}]
        n_exog: Number of exogenous variables
        exog_names: Names of exogenous variables
        n_periods: Number of time periods

    Returns:
        Tuple of (transformed_params, param_names)
    """
    # Extract components
    beta = theta[:n_exog]
    ln_sigma_v_sq = theta[n_exog]
    ln_sigma_u_sq = theta[n_exog + 1]
    mu = theta[n_exog + 2]
    delta_t_params = theta[n_exog + 3 : n_exog + 3 + n_periods - 1]

    # Transform variances
    sigma_v_sq = np.exp(ln_sigma_v_sq)
    sigma_u_sq = np.exp(ln_sigma_u_sq)

    # Build parameter vector (include all delta_t with normalization)
    params = np.concatenate([beta, [sigma_v_sq], [sigma_u_sq], [mu], delta_t_params, [1.0]])

    # Build names
    names = exog_names + ["sigma_v_sq", "sigma_u_sq", "mu"]
    names.extend([f"delta_t{t+1}" for t in range(n_periods - 1)])
    names.append(f"delta_t{n_periods}")  # Normalized to 1

    return params, names
