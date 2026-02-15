"""
Murphy-Topel variance correction for two-step estimators.

This module implements the Murphy-Topel (1985) correction for standard errors
in two-step estimation procedures, specifically for Heckman selection models.

References
----------
.. [1] Murphy, K. M., & Topel, R. H. (1985). "Estimation and Inference in
       Two-Step Econometric Models." Journal of Business & Economic Statistics,
       3(4), 370-379.
.. [2] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and
       Panel Data (2nd ed.). MIT Press. Section 12.5.
"""

from typing import Optional, Tuple

import numpy as np
from scipy import linalg


def murphy_topel_variance(
    vcov_step1: np.ndarray,
    vcov_step2_uncorrected: np.ndarray,
    cross_derivative: np.ndarray,
) -> np.ndarray:
    """
    Compute Murphy-Topel corrected variance-covariance matrix.

    The corrected variance for two-step estimator θ̂₂ is:

        V̂(θ̂₂) = (D̂'D̂)⁻¹ [D̂'ΩD̂ + R̂Ψ̂R̂'] (D̂'D̂)⁻¹

    where:
    - D̂ = ∂²Q/∂θ₂∂θ₂' (Hessian of second step)
    - Ω̂ = Var(score of second step)
    - R̂ = ∂²Q/∂θ₂∂θ₁' (cross-derivative between steps)
    - Ψ̂ = Var(score of first step)

    Parameters
    ----------
    vcov_step1 : np.ndarray
        Variance-covariance matrix from first step (k1 × k1)
    vcov_step2_uncorrected : np.ndarray
        Uncorrected variance-covariance from second step (k2 × k2)
    cross_derivative : np.ndarray
        Matrix of cross-derivatives ∂²Q/∂θ₂∂θ₁' (k2 × k1)

    Returns
    -------
    np.ndarray
        Corrected variance-covariance matrix (k2 × k2)

    Notes
    -----
    **When to Use:**

    Use Murphy-Topel correction when:
    1. First step parameters are estimated (not known)
    2. Second step depends on first step estimates
    3. You want asymptotically correct standard errors

    **Heckman Application:**

    In Heckman two-step:
    - Step 1: Probit for selection (γ̂)
    - Step 2: OLS with IMR (β̂, θ̂)
    - Cross-derivative accounts for estimation error in γ̂

    **Alternative: Bootstrap**

    If Murphy-Topel is too complex, bootstrap the entire procedure.

    Examples
    --------
    >>> # Simplified example
    >>> vcov_probit = np.eye(3) * 0.01  # From probit
    >>> vcov_ols = np.eye(5) * 0.04  # From OLS (uncorrected)
    >>> R = np.random.randn(5, 3) * 0.1  # Cross-derivative
    >>>
    >>> vcov_corrected = murphy_topel_variance(vcov_probit, vcov_ols, R)
    """
    # Ensure inputs are 2D
    if vcov_step1.ndim != 2 or vcov_step2_uncorrected.ndim != 2:
        raise ValueError("Variance-covariance matrices must be 2D arrays")

    if cross_derivative.ndim != 2:
        raise ValueError("Cross-derivative must be 2D array")

    # Dimensions
    k1 = vcov_step1.shape[0]
    k2 = vcov_step2_uncorrected.shape[0]

    if cross_derivative.shape != (k2, k1):
        raise ValueError(
            f"Cross-derivative shape {cross_derivative.shape} incompatible with "
            f"step dimensions ({k2}, {k1})"
        )

    # Murphy-Topel correction term
    # R Ψ R' where Ψ = Var(score₁) ≈ (∂²ℓ₁/∂θ₁²)⁻¹
    correction = cross_derivative @ vcov_step1 @ cross_derivative.T

    # Corrected variance
    # V̂ = V̂₂ + R Ψ R'
    # where V̂₂ is the uncorrected variance from step 2
    vcov_corrected = vcov_step2_uncorrected + correction

    # Ensure symmetry
    vcov_corrected = 0.5 * (vcov_corrected + vcov_corrected.T)

    return vcov_corrected


def compute_cross_derivative_heckman(
    X: np.ndarray,
    W: np.ndarray,
    imr: np.ndarray,
    imr_derivative: np.ndarray,
    beta: np.ndarray,
    theta: float,
    selected: np.ndarray,
) -> np.ndarray:
    """
    Compute cross-derivative for Heckman two-step estimator.

    This computes ∂²Q/∂θ₂∂γ where:
    - θ₂ = (β, θ) are outcome equation parameters
    - γ are selection equation parameters
    - Q is the second-step objective (OLS with IMR)

    Parameters
    ----------
    X : np.ndarray, shape (n, k_outcome)
        Outcome equation regressors (selected sample only)
    W : np.ndarray, shape (n, k_selection)
        Selection equation regressors (selected sample only)
    imr : np.ndarray, shape (n,)
        Inverse Mills Ratio (selected sample only)
    imr_derivative : np.ndarray, shape (n,)
        Derivative of IMR: dλ/dz (selected sample only)
    beta : np.ndarray, shape (k_outcome,)
        Outcome equation coefficients
    theta : float
        IMR coefficient (ρ σ_ε)
    selected : np.ndarray, shape (n,)
        Selection indicator (only selected obs passed)

    Returns
    -------
    np.ndarray, shape (k_outcome + 1, k_selection)
        Cross-derivative matrix

    Notes
    -----
    The cross-derivative arises because the IMR λ(W'γ) depends on γ,
    which was estimated in the first step.

    Derivation:

    Second step objective (for selected sample):
        Q = Σᵢ (yᵢ - Xᵢ'β - θ λᵢ)²

    Taking derivative w.r.t. γ:
        ∂Q/∂γ = -2 Σᵢ (yᵢ - Xᵢ'β - θ λᵢ) θ (∂λᵢ/∂γ)

    where:
        ∂λᵢ/∂γ = (dλᵢ/dz) Wᵢ

    At the optimum (where ∂Q/∂β = ∂Q/∂θ = 0), the Hessian is:
        ∂²Q/∂θ₂∂γ = -θ Σᵢ [Xᵢ; λᵢ] (dλᵢ/dz) Wᵢ'
    """
    n = X.shape[0]
    k_outcome = X.shape[1]
    k_selection = W.shape[1]

    # Augment X with IMR column: [X, λ]
    X_augmented = np.column_stack([X, imr])  # (n, k_outcome + 1)

    # Cross-derivative: Σᵢ [Xᵢ; λᵢ] (dλᵢ/dz) Wᵢ'
    # Shape: (k_outcome + 1, k_selection)
    cross_deriv = np.zeros((k_outcome + 1, k_selection))

    for i in range(n):
        # Outer product: [Xᵢ; λᵢ] Wᵢ' weighted by dλᵢ/dz
        cross_deriv += np.outer(
            X_augmented[i] * imr_derivative[i],
            W[i],
        )

    # Average and multiply by -θ
    cross_deriv = -theta * cross_deriv / n

    return cross_deriv


def heckman_two_step_variance(
    X: np.ndarray,
    W: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    theta: float,
    sigma: float,
    selected: np.ndarray,
    vcov_probit: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Murphy-Topel corrected variance for Heckman two-step.

    This is a convenience function that combines all steps:
    1. Compute IMR and derivatives
    2. Compute cross-derivative
    3. Apply Murphy-Topel correction

    Parameters
    ----------
    X : np.ndarray
        Outcome equation regressors (full sample)
    W : np.ndarray
        Selection equation regressors (full sample)
    y : np.ndarray
        Outcome variable (full sample, NaN for non-selected)
    beta : np.ndarray
        Outcome equation coefficients
    gamma : np.ndarray
        Selection equation coefficients
    theta : float
        IMR coefficient
    sigma : float
        Outcome equation error std dev
    selected : np.ndarray
        Binary selection indicator
    vcov_probit : np.ndarray
        Variance-covariance from first-step probit

    Returns
    -------
    vcov_corrected : np.ndarray
        Murphy-Topel corrected variance for (β, θ)
    se_corrected : np.ndarray
        Corrected standard errors

    Examples
    --------
    >>> # After fitting Heckman two-step
    >>> vcov_corrected, se_corrected = heckman_two_step_variance(
    ...     X, W, y, beta, gamma, theta, sigma, selected, vcov_probit
    ... )
    >>> print("Corrected SEs:", se_corrected)
    """
    from .inverse_mills import compute_imr
    from .inverse_mills import imr_derivative as compute_imr_derivative

    # Extract selected sample
    sel_mask = selected == 1
    X_sel = X[sel_mask]
    W_sel = W[sel_mask]
    y_sel = y[sel_mask]

    # Compute linear prediction and IMR
    linear_pred = W @ gamma
    linear_pred_sel = linear_pred[sel_mask]
    imr_sel = compute_imr(linear_pred_sel)
    imr_deriv_sel = compute_imr_derivative(linear_pred_sel)

    # Uncorrected variance from OLS (second step)
    # Residuals
    X_aug = np.column_stack([X_sel, imr_sel])
    params_aug = np.concatenate([beta, [theta]])
    residuals = y_sel - X_aug @ params_aug

    # OLS variance
    n_sel = len(y_sel)
    sigma_sq = np.sum(residuals**2) / (n_sel - len(params_aug))
    XtX_inv = np.linalg.inv(X_aug.T @ X_aug)
    vcov_ols = sigma_sq * XtX_inv

    # Cross-derivative
    cross_deriv = compute_cross_derivative_heckman(
        X_sel, W_sel, imr_sel, imr_deriv_sel, beta, theta, selected
    )

    # Murphy-Topel correction
    vcov_corrected = murphy_topel_variance(vcov_probit, vcov_ols, cross_deriv)

    # Standard errors
    se_corrected = np.sqrt(np.diag(vcov_corrected))

    return vcov_corrected, se_corrected


def bootstrap_two_step_variance(
    estimator_func,
    data,
    n_bootstrap: int = 500,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap variance estimation for two-step procedures.

    This is an alternative to Murphy-Topel that may be more robust
    but is computationally expensive.

    Parameters
    ----------
    estimator_func : callable
        Function that takes data and returns parameter estimates
    data : tuple or dict
        Data to resample (e.g., (y, X, W, selection))
    n_bootstrap : int, default=500
        Number of bootstrap replications
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    vcov_bootstrap : np.ndarray
        Bootstrap variance-covariance matrix
    se_bootstrap : np.ndarray
        Bootstrap standard errors

    Notes
    -----
    **Bootstrap Procedure:**

    1. Resample entities (with replacement) to preserve panel structure
    2. Re-estimate entire two-step procedure on bootstrap sample
    3. Repeat B times
    4. Compute variance across bootstrap estimates

    **Advantages:**

    - No need for analytical derivatives
    - Robust to model misspecification
    - Preserves panel structure if done correctly

    **Disadvantages:**

    - Computationally expensive (B × estimation time)
    - May be unstable with small samples

    Examples
    --------
    >>> def estimate(data):
    ...     y, X, W, selected = data
    ...     # ... fit Heckman two-step ...
    ...     return np.concatenate([beta, gamma, [theta]])
    >>>
    >>> vcov_boot, se_boot = bootstrap_two_step_variance(
    ...     estimate, (y, X, W, selected), n_bootstrap=500
    ... )
    """
    if seed is not None:
        np.random.seed(seed)

    # Placeholder implementation
    # Full implementation would:
    # 1. Identify entity structure
    # 2. Resample entities
    # 3. Re-fit model
    # 4. Store parameters
    # 5. Compute variance

    raise NotImplementedError(
        "Bootstrap variance estimation is not yet implemented. "
        "Use Murphy-Topel correction or implement panel bootstrap manually."
    )
