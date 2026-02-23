"""
Visualization helpers for SFA tutorial notebooks.

Functions:
- plot_frontier_2d: 2D scatter with estimated frontier line
- plot_efficiency_histogram: Histogram + KDE of efficiency scores
- plot_efficiency_ranking: Horizontal bar chart of top/bottom entities
- plot_efficiency_evolution: Spaghetti plot of efficiency over time
- plot_variance_decomposition: Stacked bar or pie of variance components
- plot_model_comparison: Bar chart comparing models on AIC/BIC/LL
- set_sfa_style: Set matplotlib style for SFA figures
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_sfa_style():
    """Set matplotlib style for publication-quality SFA figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12


def plot_frontier_2d(
    data: pd.DataFrame,
    input_var: str,
    output_var: str = "log_output",
    frontier_params: Optional[dict] = None,
    show_inefficiency: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    2D scatter plot with estimated production frontier line.

    Shows observations, frontier line, and optionally efficiency gaps.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing input and output variables.
    input_var : str
        Name of the input variable (x-axis).
    output_var : str
        Name of the output variable (y-axis).
    frontier_params : dict, optional
        Dict with 'intercept' and 'slope' for the frontier line.
        If None, fits OLS upper envelope.
    show_inefficiency : bool
        Whether to show distance-to-frontier lines.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    x = data[input_var].values
    y = data[output_var].values

    # Scatter plot
    ax.scatter(x, y, alpha=0.5, s=30, color="steelblue", label="Observations")

    # Frontier line
    x_range = np.linspace(x.min(), x.max(), 100)
    if frontier_params is not None:
        frontier_y = frontier_params["intercept"] + frontier_params["slope"] * x_range
    else:
        # Simple upper envelope approximation using quantile regression-like approach
        # Fit to top 10% as proxy for frontier
        threshold = np.percentile(y, 90)
        mask = y >= threshold
        if mask.sum() >= 2:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            frontier_y = np.polyval(coeffs, x_range)
        else:
            coeffs = np.polyfit(x, y, 1)
            frontier_y = np.polyval(coeffs, x_range) + np.std(y) * 0.5

    ax.plot(x_range, frontier_y, "r-", linewidth=2, label="Estimated Frontier")

    # Show inefficiency distances
    if show_inefficiency:
        sample_idx = np.random.choice(len(x), size=min(20, len(x)), replace=False)
        for idx in sample_idx:
            if frontier_params is not None:
                y_frontier = frontier_params["intercept"] + frontier_params["slope"] * x[idx]
            else:
                y_frontier = np.polyval(coeffs, x[idx])
                if not mask.sum() >= 2:
                    y_frontier += np.std(y) * 0.5
            if y_frontier > y[idx]:
                ax.plot([x[idx], x[idx]], [y[idx], y_frontier], "gray", alpha=0.3, linewidth=0.8)

    ax.set_xlabel(input_var)
    ax.set_ylabel(output_var)
    ax.set_title(title or f"Production Frontier: {output_var} vs {input_var}")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_efficiency_histogram(
    efficiency_df: pd.DataFrame,
    efficiency_col: str = "efficiency",
    show_stats: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Histogram + KDE of efficiency scores with summary statistics.

    Parameters
    ----------
    efficiency_df : pd.DataFrame
        DataFrame containing efficiency estimates.
    efficiency_col : str
        Column name for efficiency scores.
    show_stats : bool
        Whether to annotate summary statistics.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    eff = efficiency_df[efficiency_col].dropna()

    ax.hist(
        eff,
        bins=30,
        density=True,
        alpha=0.6,
        color="steelblue",
        edgecolor="white",
        label="Histogram",
    )

    # KDE
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(eff)
        x_range = np.linspace(eff.min(), eff.max(), 200)
        ax.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
    except Exception:
        pass

    if show_stats:
        stats_text = (
            f"Mean: {eff.mean():.4f}\n"
            f"Median: {eff.median():.4f}\n"
            f"Std: {eff.std():.4f}\n"
            f"Min: {eff.min():.4f}\n"
            f"Max: {eff.max():.4f}\n"
            f"N: {len(eff)}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    ax.set_xlabel("Technical Efficiency")
    ax.set_ylabel("Density")
    ax.set_title(title or "Distribution of Technical Efficiency Scores")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_efficiency_ranking(
    efficiency_df: pd.DataFrame,
    efficiency_col: str = "efficiency",
    entity_col: str = "entity",
    top_n: int = 10,
    bottom_n: int = 10,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of top-N and bottom-N entities by efficiency.

    Parameters
    ----------
    efficiency_df : pd.DataFrame
        DataFrame with entity identifiers and efficiency scores.
    efficiency_col : str
        Column name for efficiency scores.
    entity_col : str
        Column name for entity identifiers.
    top_n : int
        Number of top-ranked entities to show.
    bottom_n : int
        Number of bottom-ranked entities to show.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    # Average efficiency per entity if multiple observations
    avg_eff = efficiency_df.groupby(entity_col)[efficiency_col].mean().sort_values()

    top = avg_eff.tail(top_n)
    bottom = avg_eff.head(bottom_n)
    combined = pd.concat([bottom, top])

    fig, ax = plt.subplots(figsize=(10, max(6, len(combined) * 0.35)))

    colors = ["#e74c3c"] * len(bottom) + ["#2ecc71"] * len(top)
    bars = ax.barh(range(len(combined)), combined.values, color=colors, edgecolor="white")

    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels([str(x) for x in combined.index])
    ax.set_xlabel("Technical Efficiency")
    ax.set_title(title or f"Efficiency Ranking (Top {top_n} and Bottom {bottom_n})")

    # Add value labels
    for bar, val in zip(bars, combined.values):
        ax.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9
        )

    # Add separator line
    ax.axhline(y=len(bottom) - 0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_efficiency_evolution(
    efficiency_df: pd.DataFrame,
    efficiency_col: str = "efficiency",
    entity_col: str = "entity",
    time_col: str = "year",
    highlight: Optional[list] = None,
    show_mean: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Spaghetti plot of entity-level efficiency over time.

    Parameters
    ----------
    efficiency_df : pd.DataFrame
        DataFrame with entity, time, and efficiency columns.
    efficiency_col : str
        Column name for efficiency scores.
    entity_col : str
        Column name for entity identifiers.
    time_col : str
        Column name for time variable.
    highlight : list, optional
        List of entity IDs to highlight.
    show_mean : bool
        Whether to show the cross-sectional mean trajectory.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    entities = efficiency_df[entity_col].unique()

    for entity in entities:
        mask = efficiency_df[entity_col] == entity
        entity_data = efficiency_df[mask].sort_values(time_col)

        if highlight and entity in highlight:
            ax.plot(
                entity_data[time_col],
                entity_data[efficiency_col],
                linewidth=2,
                alpha=0.9,
                label=f"{entity_col} {entity}",
            )
        else:
            ax.plot(
                entity_data[time_col],
                entity_data[efficiency_col],
                color="gray",
                alpha=0.15,
                linewidth=0.5,
            )

    if show_mean:
        mean_eff = efficiency_df.groupby(time_col)[efficiency_col].mean()
        ax.plot(mean_eff.index, mean_eff.values, "k-", linewidth=3, label="Cross-sectional Mean")

    ax.set_xlabel(time_col.capitalize())
    ax.set_ylabel("Technical Efficiency")
    ax.set_title(title or "Efficiency Evolution Over Time")
    ax.legend(loc="best")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_variance_decomposition(
    components: dict,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Stacked bar or pie chart showing variance decomposition.

    Parameters
    ----------
    components : dict
        Dictionary mapping component names to variance shares.
        Example: {'sigma_v_sq': 0.03, 'sigma_u_sq': 0.06}
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    labels = list(components.keys())
    values = np.array(list(components.values()))
    total = values.sum()
    shares = values / total if total > 0 else values

    # Clean up labels for display
    display_labels = []
    for label in labels:
        clean = label.replace("sigma_", r"$\sigma^2_{") + "}$"
        clean = clean.replace("_sq", "")
        if "sigma" not in label:
            clean = label.replace("_", " ").title()
        display_labels.append(clean)

    colors = sns.color_palette("husl", len(labels))

    # Bar chart of actual values
    bars = ax1.bar(display_labels, values, color=colors, edgecolor="white")
    ax1.set_ylabel("Variance")
    ax1.set_title("Variance Components (Absolute)")
    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Pie chart of shares
    ax2.pie(
        shares,
        labels=display_labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.85,
    )
    ax2.set_title("Variance Shares (%)")

    fig.suptitle(title or "Variance Decomposition", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "aic",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing models on AIC/BIC/log-likelihood.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with model names as index and metrics as columns.
    metric : str
        Metric to plot ('aic', 'bic', 'loglik').
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {list(comparison_df.columns)}")

    values = comparison_df[metric]
    models = (
        comparison_df.index if isinstance(comparison_df.index, pd.Index) else range(len(values))
    )

    # Color the best model differently
    if metric in ("aic", "bic"):
        best_idx = values.idxmin()
    else:
        best_idx = values.idxmax()

    colors = ["#2ecc71" if idx == best_idx else "steelblue" for idx in models]

    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(m) for m in models], rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Model Comparison ({metric.upper()})")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
