"""
Custom visualization functions for GMM tutorials.

Provides themed plotting functions for coefficient comparisons,
diagnostic visualizations, and bootstrap distributions.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Consistent style for all tutorial plots
TUTORIAL_STYLE = {
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def apply_tutorial_style():
    """Apply the tutorial plotting style globally."""
    plt.rcParams.update(TUTORIAL_STYLE)


def plot_coefficient_comparison(
    estimates: dict[str, tuple[float, float]],
    true_value: float | None = None,
    param_name: str = "Parameter",
    title: str | None = None,
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot coefficient estimates with confidence intervals across estimators.

    Parameters
    ----------
    estimates : dict
        Mapping of estimator name to (point_estimate, std_error).
    true_value : float, optional
        True parameter value (drawn as dashed line).
    param_name : str
        Name of the parameter being compared.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save the figure.
    ax : matplotlib Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    names = list(estimates.keys())
    points = [v[0] for v in estimates.values()]
    errors = [1.96 * v[1] for v in estimates.values()]
    y_pos = range(len(names))

    ax.errorbar(
        points,
        y_pos,
        xerr=errors,
        fmt="o",
        capsize=5,
        capthick=1.5,
        markersize=8,
        color="steelblue",
        ecolor="gray",
    )

    if true_value is not None:
        ax.axvline(
            true_value, color="red", linestyle="--", alpha=0.7, label=f"True value = {true_value}"
        )
        ax.legend()

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names)
    ax.set_xlabel(param_name)
    ax.set_title(title or f"Coefficient Comparison: {param_name}")
    ax.invert_yaxis()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_nickell_bias(
    bias_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot Nickell bias as a function of T for different rho values.

    Parameters
    ----------
    bias_df : pd.DataFrame
        DataFrame with columns: rho, T, fe_estimate, gmm_estimate.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for rho, group in bias_df.groupby("rho"):
        # FE bias
        axes[0].plot(group["T"], group["fe_estimate"], "o-", label=f"ρ = {rho}")
        axes[0].axhline(rho, color="gray", linestyle=":", alpha=0.5)

    axes[0].set_xlabel("T (panel length)")
    axes[0].set_ylabel("FE Estimate of ρ")
    axes[0].set_title("Fixed Effects: Nickell Bias")
    axes[0].legend()

    for rho, group in bias_df.groupby("rho"):
        # GMM estimate
        axes[1].plot(group["T"], group["gmm_estimate"], "s-", label=f"ρ = {rho}")
        axes[1].axhline(rho, color="gray", linestyle=":", alpha=0.5)

    axes[1].set_xlabel("T (panel length)")
    axes[1].set_ylabel("GMM Estimate of ρ")
    axes[1].set_title("Difference GMM: Consistent Estimation")
    axes[1].legend()

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_instrument_count_effects(
    results_by_count: dict[int, dict],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot how instrument count affects Hansen J p-value and coefficient estimates.

    Parameters
    ----------
    results_by_count : dict
        Mapping of instrument count to dict with keys:
        'hansen_p', 'coef', 'std_error'.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = sorted(results_by_count.keys())
    hansen_p = [results_by_count[c]["hansen_p"] for c in counts]
    coefs = [results_by_count[c]["coef"] for c in counts]
    std_errors = [results_by_count[c]["std_error"] for c in counts]

    # Hansen J p-value
    axes[0].plot(counts, hansen_p, "o-", color="steelblue", markersize=8)
    axes[0].axhline(0.05, color="red", linestyle="--", alpha=0.5, label="α = 0.05")
    axes[0].axhline(0.99, color="orange", linestyle="--", alpha=0.5, label="Suspiciously high")
    axes[0].set_xlabel("Number of Instruments")
    axes[0].set_ylabel("Hansen J p-value")
    axes[0].set_title("Hansen J Test vs Instrument Count")
    axes[0].legend()

    # Coefficient stability
    axes[1].errorbar(
        counts,
        coefs,
        yerr=[1.96 * se for se in std_errors],
        fmt="s-",
        capsize=4,
        color="darkgreen",
        markersize=8,
    )
    axes[1].set_xlabel("Number of Instruments")
    axes[1].set_ylabel("Coefficient Estimate")
    axes[1].set_title("Coefficient Stability vs Instrument Count")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_bootstrap_distribution(
    bootstrap_estimates: np.ndarray,
    point_estimate: float,
    true_value: float | None = None,
    param_name: str = "ρ",
    n_bins: int = 50,
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot bootstrap distribution of a parameter estimate.

    Parameters
    ----------
    bootstrap_estimates : array-like
        Bootstrap replications of the parameter.
    point_estimate : float
        Original point estimate.
    true_value : float, optional
        True parameter value.
    param_name : str
        Name of the parameter.
    n_bins : int
        Number of histogram bins.
    save_path : str, optional
        Path to save the figure.
    ax : matplotlib Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        bootstrap_estimates,
        bins=n_bins,
        density=True,
        alpha=0.6,
        color="steelblue",
        edgecolor="white",
    )
    ax.axvline(
        point_estimate,
        color="darkblue",
        linestyle="-",
        linewidth=2,
        label=f"Point estimate = {point_estimate:.4f}",
    )

    if true_value is not None:
        ax.axvline(
            true_value, color="red", linestyle="--", linewidth=2, label=f"True value = {true_value}"
        )

    # Confidence interval
    ci_lower = np.percentile(bootstrap_estimates, 2.5)
    ci_upper = np.percentile(bootstrap_estimates, 97.5)
    ax.axvspan(
        ci_lower,
        ci_upper,
        alpha=0.15,
        color="steelblue",
        label=f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]",
    )

    ax.set_xlabel(param_name)
    ax.set_ylabel("Density")
    ax.set_title(f"Bootstrap Distribution of {param_name}")
    ax.legend(fontsize=9)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_convergence(
    iterations: list[float],
    param_name: str = "Objective",
    title: str = "CUE-GMM Convergence",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot convergence of iterative GMM estimation (e.g., CUE).

    Parameters
    ----------
    iterations : list of float
        Objective function value at each iteration.
    param_name : str
        Label for the y-axis.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.
    ax : matplotlib Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(range(1, len(iterations) + 1), iterations, "o-", color="steelblue", markersize=5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(param_name)
    ax.set_title(title)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax
