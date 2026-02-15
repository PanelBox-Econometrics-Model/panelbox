"""
Inverse Mills Ratio computation and diagnostics for selection models.

This module provides utilities for computing the Inverse Mills Ratio (IMR)
and related diagnostics for Heckman-type selection models.

References
----------
.. [1] Heckman, J.J. (1979). "Sample Selection Bias as a Specification Error."
       Econometrica, 47(1), 153-161.
.. [2] Wooldridge, J.M. (1995). "Selection Corrections for Panel Data Models Under
       Conditional Mean Independence Assumptions." Journal of Econometrics, 68(1), 115-132.
"""

from typing import Optional

import numpy as np
from scipy import stats


def compute_imr(
    linear_pred: np.ndarray,
    selected: Optional[np.ndarray] = None,
    clip_bounds: tuple[float, float] = (1e-10, 1 - 1e-10),
) -> np.ndarray:
    """
    Compute Inverse Mills Ratio (IMR) from linear predictions.

    The IMR is defined as:
        λ(z) = φ(z) / Φ(z)

    where φ is the standard normal PDF and Φ is the standard normal CDF.

    For selected observations (selection = 1):
        λᵢₜ = φ(Wᵢₜ'γ) / Φ(Wᵢₜ'γ)

    For non-selected observations (selection = 0):
        λᵢₜ = -φ(Wᵢₜ'γ) / [1 - Φ(Wᵢₜ'γ)]

    Parameters
    ----------
    linear_pred : np.ndarray
        Linear prediction from selection equation (W'γ)
    selected : np.ndarray, optional
        Binary selection indicator (1 if selected, 0 otherwise).
        If None, computes IMR for selected case only.
    clip_bounds : tuple[float, float], default=(1e-10, 1-1e-10)
        Bounds for clipping probabilities to avoid division by zero

    Returns
    -------
    np.ndarray
        Inverse Mills Ratio for each observation

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>>
    >>> # Selection equation predictions
    >>> z = np.array([0.0, 1.0, -1.0, 2.0])
    >>> selected = np.array([1, 1, 0, 1])
    >>>
    >>> # Compute IMR
    >>> imr = compute_imr(z, selected)
    >>> print(imr)

    Notes
    -----
    **Critical Implementation Detail:**

    The correct formula is φ/Φ, NOT Φ/φ. This is a common mistake.

    **Numerical Stability:**

    For extreme values of z (|z| > 8), numerical issues can occur.
    The function clips probabilities to avoid division by zero.

    **Interpretation:**

    - λ(z) decreases as z increases (selection probability increases)
    - λ(z) → ∞ as z → -∞ (very low selection probability)
    - λ(z) → 0 as z → +∞ (very high selection probability)
    """
    # Compute PDF and CDF
    pdf = stats.norm.pdf(linear_pred)
    cdf = stats.norm.cdf(linear_pred)

    # Clip to avoid division by zero
    cdf = np.clip(cdf, clip_bounds[0], clip_bounds[1])

    if selected is None:
        # Default: compute for selected case
        imr = pdf / cdf
    else:
        # Compute for both selected and non-selected
        imr = np.zeros_like(linear_pred)

        # Selected (d=1): λ = φ(z) / Φ(z)
        sel_mask = selected == 1
        imr[sel_mask] = pdf[sel_mask] / cdf[sel_mask]

        # Not selected (d=0): λ = -φ(z) / [1 - Φ(z)]
        not_sel_mask = selected == 0
        denominator = np.clip(1 - cdf[not_sel_mask], clip_bounds[0], clip_bounds[1])
        imr[not_sel_mask] = -pdf[not_sel_mask] / denominator

    return imr


def imr_derivative(linear_pred: np.ndarray) -> np.ndarray:
    """
    Compute derivative of Inverse Mills Ratio with respect to z.

    The derivative is:
        dλ/dz = -λ(λ + z)

    This is needed for Murphy-Topel variance correction.

    Parameters
    ----------
    linear_pred : np.ndarray
        Linear prediction from selection equation (W'γ)

    Returns
    -------
    np.ndarray
        Derivative of IMR

    Notes
    -----
    This derivative appears in the Murphy-Topel correction for two-step
    estimation standard errors.
    """
    imr = compute_imr(linear_pred)
    derivative = -imr * (imr + linear_pred)
    return derivative


def test_selection_effect(
    imr_coefficient: float,
    imr_se: float,
    alpha: float = 0.05,
) -> dict:
    """
    Test for presence of selection bias.

    Tests H0: ρ = 0 (no selection bias) against H1: ρ ≠ 0.

    The test is based on the coefficient of the IMR in the outcome equation.
    Under the two-step estimator:
        θ = ρ σ_ε

    where ρ is the correlation between selection and outcome errors.

    Parameters
    ----------
    imr_coefficient : float
        Coefficient on IMR in outcome equation (θ̂)
    imr_se : float
        Standard error of IMR coefficient
    alpha : float, default=0.05
        Significance level for test

    Returns
    -------
    dict
        Dictionary with test results:
        - 'statistic': t-statistic
        - 'pvalue': two-sided p-value
        - 'reject': bool indicating rejection at alpha level
        - 'interpretation': str describing result

    Examples
    --------
    >>> result = test_selection_effect(imr_coefficient=0.523, imr_se=0.145)
    >>> print(result['interpretation'])
    Selection bias detected (ρ ≠ 0, p=0.0003)
    """
    # t-statistic for H0: θ = 0
    t_stat = imr_coefficient / imr_se

    # Two-sided p-value
    pvalue = 2 * stats.norm.cdf(-np.abs(t_stat))

    # Rejection decision
    reject = pvalue < alpha

    # Interpretation
    if reject:
        interpretation = (
            f"Selection bias detected (ρ ≠ 0, p={pvalue:.4f}). "
            f"OLS would be biased. Heckman correction is necessary."
        )
    else:
        interpretation = (
            f"No significant selection bias (ρ ≈ 0, p={pvalue:.4f}). "
            f"OLS and Heckman should yield similar results."
        )

    return {
        "statistic": t_stat,
        "pvalue": pvalue,
        "reject": reject,
        "interpretation": interpretation,
        "imr_coefficient": imr_coefficient,
        "imr_se": imr_se,
    }


def imr_diagnostics(
    linear_pred: np.ndarray,
    selected: np.ndarray,
) -> dict:
    """
    Compute diagnostic statistics for Inverse Mills Ratio.

    Parameters
    ----------
    linear_pred : np.ndarray
        Linear prediction from selection equation
    selected : np.ndarray
        Binary selection indicator

    Returns
    -------
    dict
        Dictionary with diagnostic information:
        - 'imr_mean': mean IMR for selected observations
        - 'imr_std': std dev of IMR for selected
        - 'imr_min': minimum IMR
        - 'imr_max': maximum IMR
        - 'high_imr_count': count of obs with very high IMR (> 2)
        - 'selection_rate': fraction of observations selected

    Notes
    -----
    High IMR values (> 2) indicate strong selection.
    """
    imr = compute_imr(linear_pred, selected)
    selected_mask = selected == 1

    diagnostics = {
        "imr_mean": np.mean(imr[selected_mask]),
        "imr_std": np.std(imr[selected_mask]),
        "imr_min": np.min(imr[selected_mask]),
        "imr_max": np.max(imr[selected_mask]),
        "high_imr_count": np.sum(imr[selected_mask] > 2),
        "selection_rate": np.mean(selected),
        "n_selected": np.sum(selected),
        "n_total": len(selected),
    }

    return diagnostics
