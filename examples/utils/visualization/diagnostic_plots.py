"""
Diagnostic Visualization Utilities

This module provides diagnostic plotting functions for panel models.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def residual_plot(
    result: object, figsize: Tuple[int, int] = (12, 5)
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Create residual diagnostic plots.

    Parameters
    ----------
    result : PanelResults
        Fitted model result object
    figsize : tuple, optional
        Figure size (width, height)

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    axes : tuple of plt.Axes
        Tuple of (residuals_vs_fitted_ax, histogram_ax)

    Examples
    --------
    >>> residual_plot(fe_result)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    fitted_values = result.fitted_values
    residuals = result.resids

    # Residuals vs Fitted
    ax1.scatter(fitted_values, residuals, alpha=0.5, s=20)
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax1.set_xlabel("Fitted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted Values")
    ax1.grid(True, alpha=0.3)

    # Histogram of residuals
    ax2.hist(residuals, bins=30, edgecolor="black", alpha=0.7, density=True)

    # Overlay normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label="Normal")

    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of Residuals")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, (ax1, ax2)


def qq_plot(
    result: object, ax: Optional[plt.Axes] = None, figsize: Tuple[int, int] = (8, 8)
) -> plt.Axes:
    """
    Create Q-Q plot for residuals.

    Parameters
    ----------
    result : PanelResults
        Fitted model result object
    ax : plt.Axes, optional
        Matplotlib axes object. If None, creates new figure
    figsize : tuple, optional
        Figure size (width, height) if ax is None

    Returns
    -------
    plt.Axes
        Matplotlib axes object

    Examples
    --------
    >>> qq_plot(fe_result)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    residuals = result.resids
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot of Residuals")
    ax.grid(True, alpha=0.3)

    return ax


def leverage_plot(result: object, figsize: Tuple[int, int] = (10, 6)) -> plt.Axes:
    """
    Create leverage plot (if available from model).

    Parameters
    ----------
    result : PanelResults
        Fitted model result object
    figsize : tuple, optional
        Figure size (width, height)

    Returns
    -------
    plt.Axes
        Matplotlib axes object

    Examples
    --------
    >>> leverage_plot(pooled_result)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # This is a placeholder - actual implementation depends on PanelBox API
    # for computing leverage/influence statistics

    fitted_values = result.fitted_values
    residuals = result.resids
    standardized_residuals = residuals / residuals.std()

    ax.scatter(fitted_values, standardized_residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.axhline(y=2, color="red", linestyle="--", linewidth=1, label="±2σ")
    ax.axhline(y=-2, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Standardized Residuals")
    ax.set_title("Standardized Residuals vs Fitted Values")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def partial_residual_plot(
    data: pd.DataFrame, result: object, variable: str, figsize: Tuple[int, int] = (8, 6)
) -> plt.Axes:
    """
    Create partial residual plot for a specific variable.

    Parameters
    ----------
    data : pd.DataFrame
        Original panel data
    result : PanelResults
        Fitted model result object
    variable : str
        Variable name for partial residual plot
    figsize : tuple, optional
        Figure size (width, height)

    Returns
    -------
    plt.Axes
        Matplotlib axes object

    Examples
    --------
    >>> partial_residual_plot(data, fe_result, 'capital')
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute partial residuals
    beta = result.params[variable]
    x = data[variable]
    partial_resid = result.resids + beta * x

    ax.scatter(x, partial_resid, alpha=0.5, s=20)

    # Add regression line
    z = np.polyfit(x, partial_resid, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2, label="Linear fit")

    ax.set_xlabel(variable)
    ax.set_ylabel(f"Partial Residuals ({variable})")
    ax.set_title(f"Partial Residual Plot: {variable}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
