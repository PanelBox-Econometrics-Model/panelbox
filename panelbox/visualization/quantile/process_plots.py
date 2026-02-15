"""
Visualization tools for quantile regression process plots.

This module provides plotting functions for visualizing quantile regression
results, including:
- Quantile process plots (coefficients across quantiles)
- Confidence band plots
- Quantile-quantile plots
- Residual plots
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    pass


def quantile_process_plot(
    quantiles: np.ndarray,
    params: np.ndarray,
    std_errors: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    ax=None,
    **kwargs,
):
    """
    Plot quantile regression coefficients across quantile levels.

    Parameters
    ----------
    quantiles : ndarray
        Quantile levels
    params : ndarray
        Coefficient estimates. Shape (n_vars, n_quantiles)
    std_errors : ndarray, optional
        Standard errors. Shape (n_vars, n_quantiles)
    alpha : float, default 0.05
        Significance level for confidence bands
    ax : matplotlib.axes.Axes, optional
        Axes object for plotting
    **kwargs
        Additional arguments passed to plot

    Returns
    -------
    fig, ax
        Figure and axes objects
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    n_vars = params.shape[0]

    # Plot each coefficient
    colors = plt.cm.Set1(np.linspace(0, 1, n_vars))

    for i in range(n_vars):
        ax.plot(quantiles, params[i], label=f"X{i}", color=colors[i], **kwargs)

        # Add confidence bands if provided
        if std_errors is not None:
            from scipy import stats

            z_alpha = stats.norm.ppf(1 - alpha / 2)

            lower = params[i] - z_alpha * std_errors[i]
            upper = params[i] + z_alpha * std_errors[i]

            ax.fill_between(quantiles, lower, upper, alpha=0.2, color=colors[i])

    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Quantile Level (τ)", fontsize=12)
    ax.set_ylabel("Coefficient Estimate", fontsize=12)
    ax.set_title("Quantile Regression Process Plot", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    return fig, ax


def residual_plot(residuals: np.ndarray, ax=None, **kwargs):
    """
    Create residual plot for quantile regression.

    Parameters
    ----------
    residuals : ndarray
        Model residuals
    ax : matplotlib.axes.Axes, optional
        Axes object for plotting
    **kwargs
        Additional arguments passed to plot

    Returns
    -------
    fig, ax
        Figure and axes objects
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # Scatter plot of residuals
    ax.scatter(range(len(residuals)), residuals, alpha=0.5, **kwargs)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Observation Index", fontsize=12)
    ax.set_ylabel("Residual", fontsize=12)
    ax.set_title("Residual Plot", fontsize=14)
    ax.grid(True, alpha=0.3)

    return fig, ax


def qq_plot(residuals: np.ndarray, ax=None, **kwargs):
    """
    Create Q-Q plot of residuals.

    Parameters
    ----------
    residuals : ndarray
        Model residuals
    ax : matplotlib.axes.Axes, optional
        Axes object for plotting
    **kwargs
        Additional arguments passed to plot

    Returns
    -------
    fig, ax
        Figure and axes objects
    """
    try:
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        raise ImportError("matplotlib and scipy are required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    # Compute quantiles
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    sample_quantiles = np.sort(residuals)

    # Plot
    ax.scatter(theoretical_quantiles, sample_quantiles, alpha=0.5, **kwargs)

    # Add diagonal line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "k--", lw=2)

    ax.set_xlabel("Theoretical Quantiles", fontsize=12)
    ax.set_ylabel("Sample Quantiles", fontsize=12)
    ax.set_title("Q-Q Plot", fontsize=14)
    ax.grid(True, alpha=0.3)

    return fig, ax
