"""
Spatial visualization utilities for panel data diagnostics.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def create_moran_scatterplot(
    values: np.ndarray,
    W: np.ndarray,
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: str = "Moran Scatterplot",
    xlabel: str = "Standardized Values",
    ylabel: str = "Spatial Lag of Standardized Values",
    color: str = "blue",
    alpha: float = 0.5,
    show_regression: bool = True,
    show_quadrants: bool = True,
    **kwargs,
) -> "matplotlib.figure.Figure":
    """
    Create Moran scatterplot.

    Parameters
    ----------
    values : np.ndarray
        Variable values (N×1)
    W : np.ndarray
        Spatial weights matrix (N×N)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    color : str
        Point color
    alpha : float
        Point transparency
    show_regression : bool
        Whether to show regression line
    show_quadrants : bool
        Whether to show quadrant lines
    **kwargs
        Additional matplotlib scatter arguments

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt

    # Standardize values
    z = (values - np.mean(values)) / np.std(values)

    # Compute spatial lag
    Wz = W @ z

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    # Scatter plot
    ax.scatter(z, Wz, c=color, alpha=alpha, **kwargs)

    # Add regression line
    if show_regression:
        coef = np.polyfit(z, Wz, 1)
        x_line = np.array([z.min(), z.max()])
        y_line = coef[0] * x_line + coef[1]
        ax.plot(x_line, y_line, "r-", linewidth=2, label=f"Slope = {coef[0]:.3f}")
        ax.legend()

    # Add quadrant lines
    if show_quadrants:
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5)

        # Label quadrants
        ax.text(
            0.05, 0.95, "HH", transform=ax.transAxes, fontsize=12, color="gray", ha="left", va="top"
        )
        ax.text(
            0.95,
            0.95,
            "LH",
            transform=ax.transAxes,
            fontsize=12,
            color="gray",
            ha="right",
            va="top",
        )
        ax.text(
            0.05,
            0.05,
            "HL",
            transform=ax.transAxes,
            fontsize=12,
            color="gray",
            ha="left",
            va="bottom",
        )
        ax.text(
            0.95,
            0.05,
            "LL",
            transform=ax.transAxes,
            fontsize=12,
            color="gray",
            ha="right",
            va="bottom",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig


def create_lisa_cluster_map(
    lisa_results: pd.DataFrame,
    gdf: Optional["gpd.GeoDataFrame"] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: str = "LISA Cluster Map",
    figsize: tuple = (12, 8),
    color_map: Optional[Dict[str, str]] = None,
    legend: bool = True,
    **kwargs,
) -> "matplotlib.figure.Figure":
    """
    Create LISA cluster map visualization.

    Parameters
    ----------
    lisa_results : pd.DataFrame
        LISA results with 'cluster_type' column
    gdf : GeoDataFrame, optional
        Geometries for spatial units
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str
        Plot title
    figsize : tuple
        Figure size
    color_map : dict, optional
        Custom color mapping for clusters
    legend : bool
        Whether to show legend
    **kwargs
        Additional plot arguments

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Default color map
    if color_map is None:
        color_map = {
            "HH": "#d7191c",  # Red (hot spots)
            "LL": "#2c7bb6",  # Blue (cold spots)
            "HL": "#fdae61",  # Orange (high outlier)
            "LH": "#abd9e9",  # Light blue (low outlier)
            "Not significant": "#ffffbf",  # Light yellow
        }

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if gdf is not None:
        # Choropleth map with geometries
        try:
            # Merge LISA results with geometries
            gdf_plot = gdf.copy()
            gdf_plot["cluster_type"] = lisa_results["cluster_type"].values

            # Plot each cluster type
            for cluster_type, color in color_map.items():
                mask = gdf_plot["cluster_type"] == cluster_type
                if mask.any():
                    gdf_plot[mask].plot(ax=ax, color=color, label=cluster_type)

            # Remove axes
            ax.set_axis_off()

        except Exception as e:
            warnings.warn(f"Could not create map with geometries: {e}")
            # Fall back to bar plot
            _create_lisa_bar_plot(lisa_results, ax, color_map)

    else:
        # Create bar plot without geometries
        _create_lisa_bar_plot(lisa_results, ax, color_map)

    ax.set_title(title)

    # Add legend
    if legend:
        legend_elements = [
            Patch(facecolor=color_map[ct], label=ct)
            for ct in ["HH", "LL", "HL", "LH", "Not significant"]
            if ct in lisa_results["cluster_type"].values
        ]
        ax.legend(handles=legend_elements, loc="best")

    return fig


def _create_lisa_bar_plot(
    lisa_results: pd.DataFrame, ax: "matplotlib.axes.Axes", color_map: Dict[str, str]
):
    """Helper to create bar plot for LISA results."""
    # Create bar plot colored by cluster type
    entities = lisa_results.index.values
    Ii_values = lisa_results["Ii"].values
    colors = [color_map.get(ct, "#gray") for ct in lisa_results["cluster_type"]]

    bars = ax.bar(range(len(entities)), Ii_values, color=colors)

    ax.set_xlabel("Entity")
    ax.set_ylabel("Local Moran I")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Set x-axis labels if not too many
    if len(entities) <= 30:
        ax.set_xticks(range(len(entities)))
        ax.set_xticklabels(entities, rotation=45, ha="right")
    else:
        ax.set_xticks([])


def plot_morans_i_by_period(
    results: Union["MoranIByPeriodResult", pd.DataFrame],
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: str = "Moran's I by Period",
    xlabel: str = "Time Period",
    ylabel: str = "Moran's I",
    figsize: tuple = (10, 6),
    show_expected: bool = True,
    show_significance: bool = True,
    alpha: float = 0.05,
    **kwargs,
) -> "matplotlib.figure.Figure":
    """
    Plot Moran's I values over time periods.

    Parameters
    ----------
    results : MoranIByPeriodResult or pd.DataFrame
        Results from Moran's I by period analysis
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    show_expected : bool
        Whether to show expected value line
    show_significance : bool
        Whether to highlight significant periods
    alpha : float
        Significance level
    **kwargs
        Additional plot arguments

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt

    # Extract data from results
    if hasattr(results, "results_by_period"):
        # MoranIByPeriodResult object
        periods = list(results.results_by_period.keys())
        I_values = [results.results_by_period[t]["statistic"] for t in periods]
        pvalues = [results.results_by_period[t]["pvalue"] for t in periods]
        expected = [results.results_by_period[t]["expected_value"] for t in periods]
    else:
        # DataFrame
        periods = results.index.values
        I_values = results["statistic"].values
        pvalues = results["pvalue"].values
        expected = results["expected_value"].values if "expected_value" in results else None

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot Moran's I line
    ax.plot(periods, I_values, "b-o", linewidth=2, markersize=8, label="Moran's I")

    # Show expected value
    if show_expected and expected is not None:
        ax.axhline(y=expected[0], color="gray", linestyle="--", label="E[I] (no autocorrelation)")

    # Highlight significant periods
    if show_significance:
        significant = [p < alpha for p in pvalues]
        if any(significant):
            sig_periods = [periods[i] for i in range(len(periods)) if significant[i]]
            sig_values = [I_values[i] for i in range(len(periods)) if significant[i]]
            ax.scatter(
                sig_periods,
                sig_values,
                color="red",
                s=100,
                marker="*",
                label=f"Significant (p<{alpha})",
                zorder=5,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_spatial_weights_structure(
    W: np.ndarray,
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: str = "Spatial Weights Structure",
    figsize: tuple = (8, 8),
    cmap: str = "Blues",
    show_colorbar: bool = True,
    **kwargs,
) -> "matplotlib.figure.Figure":
    """
    Visualize spatial weights matrix structure.

    Parameters
    ----------
    W : np.ndarray
        Spatial weights matrix
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    show_colorbar : bool
        Whether to show colorbar
    **kwargs
        Additional imshow arguments

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot matrix
    im = ax.imshow(W, cmap=cmap, aspect="equal", **kwargs)

    # Add colorbar
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Spatial Unit j")
    ax.set_ylabel("Spatial Unit i")

    # Add grid for small matrices
    if W.shape[0] <= 50:
        ax.set_xticks(np.arange(W.shape[1]) - 0.5, minor=True)
        ax.set_yticks(np.arange(W.shape[0]) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

    return fig


def create_spatial_diagnostics_dashboard(
    spatial_diagnostics: Dict[str, Any],
    figsize: tuple = (16, 12),
    title: str = "Spatial Diagnostics Dashboard",
    **kwargs,
) -> "matplotlib.figure.Figure":
    """
    Create comprehensive dashboard with all spatial diagnostics.

    Parameters
    ----------
    spatial_diagnostics : dict
        Results from run_spatial_diagnostics()
    figsize : tuple
        Figure size
    title : str
        Dashboard title
    **kwargs
        Additional arguments

    Returns
    -------
    matplotlib.figure.Figure
        Figure object with multiple subplots
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Create grid
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Moran's I result (text)
    ax1 = fig.add_subplot(gs[0, 0])
    morans_result = spatial_diagnostics["morans_i"]
    text = f"Global Moran's I\n\n"
    text += f"Statistic: {morans_result.statistic:.4f}\n"
    text += f"P-value: {morans_result.pvalue:.4f}\n"
    text += f"Expected: {morans_result.additional_info['expected_value']:.4f}\n"
    text += f"\n{morans_result.conclusion}"
    ax1.text(
        0.05,
        0.95,
        text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )
    ax1.set_axis_off()

    # 2. LM Tests summary
    ax2 = fig.add_subplot(gs[0, 1:])
    lm_summary = spatial_diagnostics["lm_tests"]["summary"]
    ax2.axis("tight")
    ax2.axis("off")

    # Create table
    table_data = []
    for _, row in lm_summary.iterrows():
        table_data.append(
            [
                row["Test"],
                f"{row['Statistic']:.3f}",
                f"{row['p-value']:.4f}",
                "✓" if row["Significant"] else "✗",
            ]
        )

    table = ax2.table(
        cellText=table_data,
        colLabels=["Test", "Statistic", "P-value", "Sig."],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Add recommendation
    rec_text = (
        f"Recommendation: {spatial_diagnostics['recommendation']}\n"
        f"Reason: {spatial_diagnostics['lm_tests']['reason']}"
    )
    ax2.text(
        0.5, -0.1, rec_text, transform=ax2.transAxes, fontsize=11, ha="center", fontweight="bold"
    )

    # 3. LISA clusters summary
    ax3 = fig.add_subplot(gs[1, 0])
    lisa_results = spatial_diagnostics["morans_i_local"]
    cluster_counts = lisa_results["cluster_type"].value_counts()

    # Bar plot
    colors = {
        "HH": "#d7191c",
        "LL": "#2c7bb6",
        "HL": "#fdae61",
        "LH": "#abd9e9",
        "Not significant": "#ffffbf",
    }

    cluster_counts.plot(
        kind="bar", ax=ax3, color=[colors.get(x, "gray") for x in cluster_counts.index]
    )
    ax3.set_title("LISA Cluster Distribution")
    ax3.set_xlabel("Cluster Type")
    ax3.set_ylabel("Count")
    ax3.tick_params(axis="x", rotation=45)

    # 4. Moran scatterplot
    ax4 = fig.add_subplot(gs[1, 1])
    # Need to reconstruct from available data
    # This would require access to original residuals
    ax4.text(
        0.5,
        0.5,
        "Moran Scatterplot\n(Requires residuals)",
        transform=ax4.transAxes,
        ha="center",
        va="center",
    )
    ax4.set_axis_off()

    # 5. Spatial weights structure
    ax5 = fig.add_subplot(gs[1, 2])
    W = spatial_diagnostics["W"]
    im = ax5.imshow(W[:20, :20], cmap="Blues", aspect="equal")  # Show subset if large
    ax5.set_title(f'Spatial Weights ({"subset " if W.shape[0] > 20 else ""}N={W.shape[0]})')
    ax5.set_xlabel("Unit j")
    ax5.set_ylabel("Unit i")

    # 6. Distribution of Local Moran's I
    ax6 = fig.add_subplot(gs[2, :2])
    lisa_results["Ii"].hist(bins=30, ax=ax6, edgecolor="black")
    ax6.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax6.set_title("Distribution of Local Moran's I")
    ax6.set_xlabel("Local Moran's I")
    ax6.set_ylabel("Frequency")

    # 7. P-value distribution
    ax7 = fig.add_subplot(gs[2, 2])
    lisa_results["pvalue"].hist(bins=20, ax=ax7, edgecolor="black")
    ax7.axvline(x=0.05, color="red", linestyle="--", alpha=0.5, label="α=0.05")
    ax7.set_title("P-value Distribution (LISA)")
    ax7.set_xlabel("P-value")
    ax7.set_ylabel("Frequency")
    ax7.legend()

    plt.tight_layout()
    return fig


def plot_spatial_effects(
    effects_result: "SpatialEffectsResult",
    figsize: tuple = (12, 6),
    title: str = "Spatial Effects Decomposition",
    show_ci: bool = True,
    colors: Optional[Dict[str, str]] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    **kwargs,
) -> "matplotlib.figure.Figure":
    """
    Create bar plot of spatial effects (direct, indirect, total).

    Parameters
    ----------
    effects_result : SpatialEffectsResult
        Results from compute_spatial_effects()
    figsize : tuple
        Figure size
    title : str
        Plot title
    show_ci : bool
        Whether to show confidence intervals
    colors : dict, optional
        Custom colors for effect types
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    **kwargs
        Additional arguments

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Default colors
    if colors is None:
        colors = {"direct": "steelblue", "indirect": "coral", "total": "seagreen"}

    # Extract data
    variables = list(effects_result.effects.keys())
    n_vars = len(variables)

    direct_vals = np.array([effects_result.effects[v]["direct"] for v in variables])
    indirect_vals = np.array([effects_result.effects[v]["indirect"] for v in variables])
    total_vals = np.array([effects_result.effects[v]["total"] for v in variables])

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Set up positions
    x = np.arange(n_vars)
    width = 0.25

    # Plot bars
    bars_direct = ax.bar(x - width, direct_vals, width, label="Direct", color=colors["direct"])
    bars_indirect = ax.bar(x, indirect_vals, width, label="Indirect", color=colors["indirect"])
    bars_total = ax.bar(x + width, total_vals, width, label="Total", color=colors["total"])

    # Add error bars if requested
    if show_ci and "direct_se" in list(effects_result.effects.values())[0]:
        direct_se = np.array([effects_result.effects[v]["direct_se"] for v in variables])
        indirect_se = np.array([effects_result.effects[v]["indirect_se"] for v in variables])
        total_se = np.array([effects_result.effects[v]["total_se"] for v in variables])

        # Use 1.96 for 95% CI
        ax.errorbar(
            x - width,
            direct_vals,
            yerr=direct_se * 1.96,
            fmt="none",
            color="black",
            capsize=3,
            alpha=0.5,
        )
        ax.errorbar(
            x,
            indirect_vals,
            yerr=indirect_se * 1.96,
            fmt="none",
            color="black",
            capsize=3,
            alpha=0.5,
        )
        ax.errorbar(
            x + width,
            total_vals,
            yerr=total_se * 1.96,
            fmt="none",
            color="black",
            capsize=3,
            alpha=0.5,
        )

    # Customize plot
    ax.set_xlabel("Variable", fontsize=12)
    ax.set_ylabel("Effect Magnitude", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variables, rotation=45 if len(variables) > 5 else 0, ha="right")
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis="y")

    # Add horizontal line at zero
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Add value labels on bars if not too many variables
    if n_vars <= 6:

        def autolabel(bars, values):
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(
                    f"{val:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=9,
                )

        autolabel(bars_direct, direct_vals)
        autolabel(bars_indirect, indirect_vals)
        autolabel(bars_total, total_vals)

    plt.tight_layout()
    return fig


def plot_direct_vs_indirect(
    effects_result: "SpatialEffectsResult",
    figsize: tuple = (8, 8),
    title: str = "Direct vs Indirect Effects",
    show_diagonal: bool = True,
    show_labels: bool = True,
    ax: Optional["matplotlib.axes.Axes"] = None,
    **kwargs,
) -> "matplotlib.figure.Figure":
    """
    Create scatter plot of direct vs indirect effects.

    Parameters
    ----------
    effects_result : SpatialEffectsResult
        Results from compute_spatial_effects()
    figsize : tuple
        Figure size
    title : str
        Plot title
    show_diagonal : bool
        Whether to show 45-degree line
    show_labels : bool
        Whether to label points with variable names
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    **kwargs
        Additional scatter arguments

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract data
    variables = list(effects_result.effects.keys())
    direct_vals = np.array([effects_result.effects[v]["direct"] for v in variables])
    indirect_vals = np.array([effects_result.effects[v]["indirect"] for v in variables])

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Scatter plot
    ax.scatter(direct_vals, indirect_vals, s=100, alpha=0.6, **kwargs)

    # Add labels
    if show_labels:
        for i, var in enumerate(variables):
            ax.annotate(
                var,
                (direct_vals[i], indirect_vals[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                alpha=0.7,
            )

    # Add 45-degree line
    if show_diagonal:
        lim_min = min(direct_vals.min(), indirect_vals.min())
        lim_max = max(direct_vals.max(), indirect_vals.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", alpha=0.5, label="Direct = Indirect")

    # Add reference lines at zero
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    # Labels and title
    ax.set_xlabel("Direct Effects", fontsize=12)
    ax.set_ylabel("Indirect Effects (Spillovers)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if show_diagonal:
        ax.legend()

    # Add quadrant labels
    ax.text(
        0.95,
        0.95,
        "Strong\nSpillovers",
        transform=ax.transAxes,
        fontsize=10,
        color="gray",
        ha="right",
        va="top",
        alpha=0.5,
    )
    ax.text(
        0.05,
        0.05,
        "Weak\nSpillovers",
        transform=ax.transAxes,
        fontsize=10,
        color="gray",
        ha="left",
        va="bottom",
        alpha=0.5,
    )

    plt.tight_layout()
    return fig


def plot_effects_comparison(
    effects_results: List["SpatialEffectsResult"],
    model_names: List[str],
    variables: Optional[List[str]] = None,
    effect_type: str = "total",
    figsize: tuple = (12, 6),
    title: str = "Model Comparison: Spatial Effects",
    ax: Optional["matplotlib.axes.Axes"] = None,
    **kwargs,
) -> "matplotlib.figure.Figure":
    """
    Compare effects across multiple spatial models.

    Parameters
    ----------
    effects_results : list of SpatialEffectsResult
        Results from different models
    model_names : list of str
        Names for each model
    variables : list of str, optional
        Variables to compare (None = all common)
    effect_type : {'direct', 'indirect', 'total'}
        Which effect to compare
    figsize : tuple
        Figure size
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    **kwargs
        Additional arguments

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get common variables if not specified
    if variables is None:
        all_vars = [set(res.effects.keys()) for res in effects_results]
        variables = list(set.intersection(*all_vars))
        variables = sorted(variables)

    if not variables:
        raise ValueError("No common variables found across models")

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Set up positions
    n_vars = len(variables)
    n_models = len(model_names)
    x = np.arange(n_vars)
    width = 0.8 / n_models

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    # Plot bars for each model
    for i, (result, name) in enumerate(zip(effects_results, model_names)):
        values = [result.effects[v][effect_type] for v in variables]

        # Add error bars if available
        if f"{effect_type}_se" in list(result.effects.values())[0]:
            errors = [result.effects[v][f"{effect_type}_se"] * 1.96 for v in variables]
            ax.bar(
                x + i * width - width * n_models / 2 + width / 2,
                values,
                width,
                label=name,
                color=colors[i],
                yerr=errors,
                capsize=3,
            )
        else:
            ax.bar(
                x + i * width - width * n_models / 2 + width / 2,
                values,
                width,
                label=name,
                color=colors[i],
            )

    # Customize plot
    ax.set_xlabel("Variable", fontsize=12)
    ax.set_ylabel(f"{effect_type.capitalize()} Effect", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variables, rotation=45 if len(variables) > 5 else 0, ha="right")
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis="y")

    # Add horizontal line at zero
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    return fig
