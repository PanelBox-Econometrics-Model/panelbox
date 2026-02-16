"""
Plotting utilities for standard errors tutorials.

This module provides visualization functions for standard error comparisons,
diagnostic plots, and result presentations across the tutorial series.

Functions
---------
plot_residuals : Plot residuals vs fitted values
plot_acf_pacf : Plot autocorrelation and partial autocorrelation functions
plot_se_comparison : Compare standard errors across methods
plot_quantile_process : Plot coefficient estimates across quantiles
plot_spatial_kernel : Plot kernel weights vs distance
plot_forest_ci : Forest plot with confidence intervals

Author: PanelBox Development Team
Date: 2026-02-16
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Set default plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10


def plot_residuals(
    fitted: np.ndarray,
    residuals: np.ndarray,
    title: str = "Residuals vs Fitted Values",
    xlabel: str = "Fitted Values",
    ylabel: str = "Residuals",
    alpha: float = 0.5,
    add_lowess: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot residuals vs fitted values to diagnose heteroskedasticity.

    Parameters
    ----------
    fitted : np.ndarray
        Fitted values from regression
    residuals : np.ndarray
        Residuals from regression
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    alpha : float, optional
        Point transparency (0-1)
    add_lowess : bool, optional
        Add LOWESS smoothing line
    figsize : tuple, optional
        Figure size (width, height)
    ax : plt.Axes, optional
        Matplotlib axes object. If None, creates new figure

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Scatter plot
    ax.scatter(fitted, residuals, alpha=alpha, s=30, edgecolors="k", linewidth=0.5)

    # Add horizontal line at zero
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2, label="Zero line")

    # Add LOWESS smoothing if requested
    if add_lowess:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            smoothed = lowess(residuals, fitted, frac=0.3)
            ax.plot(smoothed[:, 0], smoothed[:, 1], "blue", linewidth=2, label="LOWESS trend")
        except ImportError:
            # Fallback to simple moving average if statsmodels not available
            sorted_idx = np.argsort(fitted)
            window = max(len(fitted) // 20, 5)
            trend = pd.Series(residuals[sorted_idx]).rolling(window=window, center=True).mean()
            ax.plot(fitted[sorted_idx], trend, "blue", linewidth=2, label="Trend")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_acf_pacf(
    data: Union[np.ndarray, pd.Series],
    lags: int = 20,
    title: str = "ACF and PACF",
    figsize: Tuple[int, int] = (14, 5),
    alpha: float = 0.05,
) -> plt.Figure:
    """
    Plot autocorrelation function (ACF) and partial autocorrelation function (PACF).

    Parameters
    ----------
    data : np.ndarray or pd.Series
        Time series data
    lags : int, optional
        Number of lags to display
    title : str, optional
        Overall plot title
    figsize : tuple, optional
        Figure size (width, height)
    alpha : float, optional
        Significance level for confidence bands

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # ACF plot
        plot_acf(data, lags=lags, ax=ax1, alpha=alpha, title="Autocorrelation Function")
        ax1.set_xlabel("Lag")
        ax1.set_ylabel("ACF")

        # PACF plot
        plot_pacf(data, lags=lags, ax=ax2, alpha=alpha, title="Partial Autocorrelation Function")
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("PACF")

        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()

    except ImportError:
        # Fallback implementation if statsmodels not available
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Compute ACF manually
        data_centered = data - np.mean(data)
        acf_values = []
        for lag in range(lags + 1):
            if lag == 0:
                acf_values.append(1.0)
            else:
                c0 = np.dot(data_centered, data_centered) / len(data)
                c_lag = np.dot(data_centered[:-lag], data_centered[lag:]) / len(data)
                acf_values.append(c_lag / c0)

        # Plot ACF
        ax1.bar(range(lags + 1), acf_values, width=0.3)
        ax1.axhline(y=0, color="black", linewidth=0.8)

        # Confidence bands
        conf_level = 1.96 / np.sqrt(len(data))
        ax1.axhline(y=conf_level, color="blue", linestyle="--", linewidth=1)
        ax1.axhline(y=-conf_level, color="blue", linestyle="--", linewidth=1)
        ax1.set_xlabel("Lag")
        ax1.set_ylabel("ACF")
        ax1.set_title("Autocorrelation Function")
        ax1.grid(True, alpha=0.3)

        # Simple PACF approximation (note: not exact)
        ax2.bar(range(lags + 1), acf_values, width=0.3)  # Simplified
        ax2.axhline(y=0, color="black", linewidth=0.8)
        ax2.axhline(y=conf_level, color="blue", linestyle="--", linewidth=1)
        ax2.axhline(y=-conf_level, color="blue", linestyle="--", linewidth=1)
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("PACF (approx)")
        ax2.set_title("Partial Autocorrelation (Approximation)")
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()

    return fig


def plot_se_comparison(
    coef_name: str,
    estimates: Dict[str, float],
    std_errors: Dict[str, float],
    methods: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    orientation: str = "horizontal",
) -> plt.Figure:
    """
    Compare coefficient estimates and standard errors across methods.

    Parameters
    ----------
    coef_name : str
        Name of the coefficient being compared
    estimates : dict
        Dictionary mapping method names to coefficient estimates
    std_errors : dict
        Dictionary mapping method names to standard errors
    methods : list, optional
        List of methods to include (in order). If None, uses all methods
    title : str, optional
        Plot title. If None, auto-generated
    figsize : tuple, optional
        Figure size (width, height)
    orientation : str, optional
        'horizontal' or 'vertical' for bar orientation

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    if methods is None:
        methods = list(estimates.keys())

    # Prepare data
    coefs = [estimates[m] for m in methods]
    ses = [std_errors[m] for m in methods]

    # Compute 95% confidence intervals
    ci_lower = [coefs[i] - 1.96 * ses[i] for i in range(len(methods))]
    ci_upper = [coefs[i] + 1.96 * ses[i] for i in range(len(methods))]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left panel: Coefficient estimates with confidence intervals
    if orientation == "horizontal":
        y_pos = np.arange(len(methods))
        ax1.barh(y_pos, coefs, alpha=0.7, color="steelblue")
        ax1.errorbar(
            coefs,
            y_pos,
            xerr=[np.array(coefs) - np.array(ci_lower), np.array(ci_upper) - np.array(coefs)],
            fmt="none",
            ecolor="black",
            capsize=5,
            linewidth=2,
        )
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(methods)
        ax1.set_xlabel(f"Estimate ({coef_name})")
        ax1.set_ylabel("Method")
        ax1.axvline(x=0, color="red", linestyle="--", linewidth=1)
    else:
        x_pos = np.arange(len(methods))
        ax1.bar(x_pos, coefs, alpha=0.7, color="steelblue")
        ax1.errorbar(
            x_pos,
            coefs,
            yerr=[np.array(coefs) - np.array(ci_lower), np.array(ci_upper) - np.array(coefs)],
            fmt="none",
            ecolor="black",
            capsize=5,
            linewidth=2,
        )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(methods, rotation=45, ha="right")
        ax1.set_ylabel(f"Estimate ({coef_name})")
        ax1.set_xlabel("Method")
        ax1.axhline(y=0, color="red", linestyle="--", linewidth=1)

    ax1.set_title("Coefficient Estimates with 95% CI")
    ax1.grid(True, alpha=0.3)

    # Right panel: Standard errors comparison
    if orientation == "horizontal":
        ax2.barh(y_pos, ses, alpha=0.7, color="coral")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(methods)
        ax2.set_xlabel("Standard Error")
        ax2.set_ylabel("Method")
    else:
        ax2.bar(x_pos, ses, alpha=0.7, color="coral")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(methods, rotation=45, ha="right")
        ax2.set_ylabel("Standard Error")
        ax2.set_xlabel("Method")

    ax2.set_title("Standard Errors")
    ax2.grid(True, alpha=0.3)

    # Overall title
    if title is None:
        title = f"Comparison of Estimation Methods: {coef_name}"
    fig.suptitle(title, fontsize=14, y=1.00)

    plt.tight_layout()
    return fig


def plot_quantile_process(
    quantiles: np.ndarray,
    coefficients: np.ndarray,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    coef_name: str = "Coefficient",
    ols_estimate: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot coefficient estimates across quantiles (quantile process plot).

    Parameters
    ----------
    quantiles : np.ndarray
        Array of quantiles (e.g., [0.1, 0.2, ..., 0.9])
    coefficients : np.ndarray
        Coefficient estimates at each quantile
    ci_lower : np.ndarray, optional
        Lower confidence interval bounds
    ci_upper : np.ndarray, optional
        Upper confidence interval bounds
    coef_name : str, optional
        Name of the coefficient
    ols_estimate : float, optional
        OLS estimate to show as reference line
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height)

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot quantile regression estimates
    ax.plot(
        quantiles,
        coefficients,
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=6,
        label="Quantile Regression",
    )

    # Add confidence bands if provided
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(
            quantiles, ci_lower, ci_upper, alpha=0.2, color="steelblue", label="95% Confidence Band"
        )

    # Add OLS reference line if provided
    if ols_estimate is not None:
        ax.axhline(
            y=ols_estimate,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"OLS Estimate ({ols_estimate:.3f})",
        )

    ax.set_xlabel("Quantile", fontsize=12)
    ax.set_ylabel(f"{coef_name} Estimate", fontsize=12)

    if title is None:
        title = f"Quantile Regression Process: {coef_name}"
    ax.set_title(title, fontsize=14)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(quantiles.min() - 0.05, quantiles.max() + 0.05)

    plt.tight_layout()
    return fig


def plot_spatial_kernel(
    distances: np.ndarray,
    weights: np.ndarray,
    kernel_type: str = "Bartlett",
    bandwidth: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot kernel weights vs distance for spatial HAC standard errors.

    Parameters
    ----------
    distances : np.ndarray
        Array of distances
    weights : np.ndarray
        Kernel weights corresponding to distances
    kernel_type : str, optional
        Type of kernel used
    bandwidth : float, optional
        Bandwidth parameter (if applicable)
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height)

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by distance for smooth plot
    sorted_idx = np.argsort(distances)
    sorted_dist = distances[sorted_idx]
    sorted_weights = weights[sorted_idx]

    # Plot kernel weights
    ax.plot(sorted_dist, sorted_weights, linewidth=2, color="steelblue")
    ax.fill_between(sorted_dist, 0, sorted_weights, alpha=0.3, color="steelblue")

    # Add bandwidth reference if provided
    if bandwidth is not None:
        ax.axvline(
            x=bandwidth,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Bandwidth = {bandwidth:.2f}",
        )

    ax.set_xlabel("Distance", fontsize=12)
    ax.set_ylabel("Kernel Weight", fontsize=12)

    if title is None:
        title = f"{kernel_type} Kernel Weights vs Distance"
    ax.set_title(title, fontsize=14)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=1.05)

    plt.tight_layout()
    return fig


def plot_forest_ci(
    variables: List[str],
    estimates: List[float],
    ci_lower: List[float],
    ci_upper: List[float],
    groups: Optional[List[str]] = None,
    title: str = "Forest Plot with Confidence Intervals",
    xlabel: str = "Estimate",
    figsize: Tuple[int, int] = (10, 8),
    reference_line: float = 0.0,
) -> plt.Figure:
    """
    Create a forest plot showing estimates with confidence intervals.

    Commonly used to display multiple coefficient estimates and their
    confidence intervals in a compact visual format.

    Parameters
    ----------
    variables : list of str
        Variable names
    estimates : list of float
        Point estimates
    ci_lower : list of float
        Lower confidence interval bounds
    ci_upper : list of float
        Upper confidence interval bounds
    groups : list of str, optional
        Group labels for variables (for color coding)
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    figsize : tuple, optional
        Figure size (width, height)
    reference_line : float, optional
        Value for vertical reference line (e.g., 0 for no effect)

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    n_vars = len(variables)

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare colors
    if groups is not None:
        unique_groups = list(dict.fromkeys(groups))  # Preserve order
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_groups)))
        color_map = dict(zip(unique_groups, colors))
        point_colors = [color_map[g] for g in groups]
    else:
        point_colors = ["steelblue"] * n_vars

    # Plot each variable
    y_pos = np.arange(n_vars)

    for i in range(n_vars):
        # Plot confidence interval as line
        ax.plot(
            [ci_lower[i], ci_upper[i]],
            [y_pos[i], y_pos[i]],
            color=point_colors[i],
            linewidth=2,
            alpha=0.7,
        )

        # Plot point estimate
        ax.scatter(
            estimates[i],
            y_pos[i],
            color=point_colors[i],
            s=100,
            zorder=3,
            edgecolors="black",
            linewidth=1,
        )

    # Add reference line
    ax.axvline(
        x=reference_line,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Reference ({reference_line})",
    )

    # Format axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")

    # Add legend if groups provided
    if groups is not None:
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[g],
                markersize=8,
                label=g,
                markeredgecolor="black",
            )
            for g in unique_groups
        ]
        ax.legend(handles=legend_elements, loc="best")

    plt.tight_layout()
    return fig


def plot_heteroskedasticity_test(
    fitted: np.ndarray,
    squared_residuals: np.ndarray,
    title: str = "Heteroskedasticity Diagnostic",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot squared residuals vs fitted values to visualize heteroskedasticity.

    Parameters
    ----------
    fitted : np.ndarray
        Fitted values from regression
    squared_residuals : np.ndarray
        Squared residuals from regression
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height)

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot of squared residuals
    ax.scatter(fitted, squared_residuals, alpha=0.5, s=30, edgecolors="k", linewidth=0.5)

    # Add trend line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smoothed = lowess(squared_residuals, fitted, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], "red", linewidth=2, label="LOWESS trend")
    except ImportError:
        # Fallback to simple moving average
        sorted_idx = np.argsort(fitted)
        window = max(len(fitted) // 20, 5)
        trend = pd.Series(squared_residuals[sorted_idx]).rolling(window=window, center=True).mean()
        ax.plot(fitted[sorted_idx], trend, "red", linewidth=2, label="Trend")

    ax.set_xlabel("Fitted Values", fontsize=12)
    ax.set_ylabel("Squared Residuals", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# Export all functions
__all__ = [
    "plot_residuals",
    "plot_acf_pacf",
    "plot_se_comparison",
    "plot_quantile_process",
    "plot_spatial_kernel",
    "plot_forest_ci",
    "plot_heteroskedasticity_test",
]
