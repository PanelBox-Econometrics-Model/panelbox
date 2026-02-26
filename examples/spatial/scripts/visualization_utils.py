"""
Visualization Utilities for Spatial Econometrics Tutorials

This module provides plotting functions for spatial data visualization and
spatial econometrics diagnostics.

Functions:
    plot_choropleth: Create choropleth map
    plot_spatial_connections: Visualize spatial weights connections
    plot_moran_scatterplot: Moran scatterplot for spatial autocorrelation
    plot_lisa_map: Local Moran's I (LISA) cluster map
    plot_effects_decomposition: Visualize direct/indirect/total effects
    plot_spatial_residuals: Map model residuals
"""

from typing import Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def plot_choropleth(
    gdf: gpd.GeoDataFrame,
    variable: str,
    title: Optional[str] = None,
    cmap: str = "YlOrRd",
    scheme: Optional[str] = "quantiles",
    k: int = 5,
    legend: bool = True,
    figsize: tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
):
    """
    Create choropleth map.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with geometry and variable to map
    variable : str
        Column name to visualize
    title : str, optional
        Map title
    cmap : str, default "YlOrRd"
        Matplotlib colormap
    scheme : str, optional
        Classification scheme ('quantiles', 'equal_interval', 'fisher_jenks', None)
    k : int, default 5
        Number of classes for classification
    legend : bool, default True
        Show legend
    figsize : tuple, default (12, 8)
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig, ax : matplotlib figure and axis

    Examples
    --------
    >>> plot_choropleth(
    ...     gdf, variable="income_pc", title="Per Capita Income by County", scheme="quantiles"
    ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    if scheme:
        gdf.plot(
            column=variable,
            ax=ax,
            cmap=cmap,
            scheme=scheme,
            k=k,
            legend=legend,
            edgecolor="black",
            linewidth=0.5,
        )
    else:
        gdf.plot(column=variable, ax=ax, cmap=cmap, legend=legend, edgecolor="black", linewidth=0.5)

    ax.set_title(title or f"Choropleth Map: {variable}", fontsize=14)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_spatial_connections(
    gdf: gpd.GeoDataFrame,
    W,
    sample_entities: Optional[list] = None,
    figsize: tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
):
    """
    Visualize spatial weights connections.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with geometries
    W : libpysal.weights.W
        Spatial weights object
    sample_entities : list, optional
        List of entity IDs to highlight connections (if None, shows all)
    figsize : tuple, default (12, 8)
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig, ax : matplotlib figure and axis

    Examples
    --------
    >>> plot_spatial_connections(gdf, W, sample_entities=["01001", "01003"])
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot base map
    gdf.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=0.5)

    # Get centroids
    gdf_copy = gdf.copy()
    gdf_copy["centroid"] = gdf_copy.geometry.centroid

    # Plot connections
    if sample_entities:
        entities = sample_entities
    else:
        entities = list(W.neighbors.keys())

    for entity in entities:
        if entity not in W.neighbors:
            continue

        # Get centroid of entity
        entity_geom = gdf_copy[gdf_copy.index == entity]["centroid"].iloc[0]

        # Plot connections to neighbors
        for neighbor in W.neighbors[entity]:
            neighbor_geom = gdf_copy[gdf_copy.index == neighbor]["centroid"].iloc[0]

            ax.plot(
                [entity_geom.x, neighbor_geom.x],
                [entity_geom.y, neighbor_geom.y],
                "b-",
                alpha=0.3,
                linewidth=0.5,
            )

    ax.set_title("Spatial Weights Connections", fontsize=14)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_moran_scatterplot(
    data: Union[pd.DataFrame, np.ndarray],
    W,
    variable: Optional[str] = None,
    standardize: bool = True,
    figsize: tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
):
    """
    Create Moran scatterplot for spatial autocorrelation.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Data with variable to analyze
    W : libpysal.weights.W
        Spatial weights object
    variable : str, optional
        Column name (if data is DataFrame)
    standardize : bool, default True
        Standardize variable to mean=0, std=1
    figsize : tuple, default (8, 8)
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig, ax : matplotlib figure and axis
    moran_i : float
        Moran's I statistic

    Examples
    --------
    >>> fig, ax, moran_i = plot_moran_scatterplot(df, W, variable="income_pc")
    >>> print(f"Moran's I: {moran_i:.4f}")
    """
    from esda import Moran
    from libpysal.weights import lag_spatial

    # Extract values
    if isinstance(data, pd.DataFrame):
        if variable is None:
            raise ValueError("variable must be specified when data is DataFrame")
        values = data[variable].values
    else:
        values = data

    # Standardize
    if standardize:
        values = (values - values.mean()) / values.std()

    # Compute spatial lag
    w_values = lag_spatial(W, values)

    # Compute Moran's I
    moran = Moran(values, W)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(values, w_values, alpha=0.5, edgecolors="k", linewidths=0.5)

    # Add regression line
    slope, intercept = np.polyfit(values, w_values, 1)
    x_line = np.array([values.min(), values.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "r--", linewidth=2, label=f"Slope = {slope:.4f}")

    # Add quadrant lines
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5)

    # Labels
    ax.set_xlabel("Standardized Variable", fontsize=12)
    ax.set_ylabel("Spatial Lag of Variable", fontsize=12)
    ax.set_title(
        f"Moran Scatterplot\nMoran's I = {moran.I:.4f} (p = {moran.p_sim:.4f})", fontsize=14
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax, moran.I


def plot_lisa_map(
    gdf: gpd.GeoDataFrame,
    variable: str,
    W,
    significance: float = 0.05,
    figsize: tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
):
    """
    Create LISA (Local Moran's I) cluster map.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with geometry and variable
    variable : str
        Column name to analyze
    W : libpysal.weights.W
        Spatial weights object
    significance : float, default 0.05
        Significance level for clusters
    figsize : tuple, default (12, 8)
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig, ax : matplotlib figure and axis

    Examples
    --------
    >>> plot_lisa_map(gdf, variable="income_pc", W=W)
    """
    from esda import Moran_Local

    # Compute Local Moran's I
    values = gdf[variable].values
    lisa = Moran_Local(values, W)

    # Create cluster categories
    # 1 = HH (High-High), 2 = LH (Low-High), 3 = LL (Low-Low), 4 = HL (High-Low)
    # 0 = Not significant
    sig = lisa.p_sim < significance
    clusters = lisa.q * sig

    # Create labels
    cluster_labels = {
        0: "Not Significant",
        1: "HH (High-High)",
        2: "LH (Low-High)",
        3: "LL (Low-Low)",
        4: "HL (High-Low)",
    }

    # Add to GeoDataFrame
    gdf_copy = gdf.copy()
    gdf_copy["lisa_cluster"] = clusters
    gdf_copy["lisa_cluster_label"] = gdf_copy["lisa_cluster"].map(cluster_labels)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors
    cluster_colors = {0: "lightgray", 1: "red", 2: "lightblue", 3: "blue", 4: "pink"}

    for cluster, color in cluster_colors.items():
        subset = gdf_copy[gdf_copy["lisa_cluster"] == cluster]
        if len(subset) > 0:
            subset.plot(
                ax=ax, color=color, edgecolor="black", linewidth=0.5, label=cluster_labels[cluster]
            )

    ax.set_title(f"LISA Cluster Map: {variable}", fontsize=14)
    ax.axis("off")
    ax.legend(loc="best")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_effects_decomposition(
    effects_df: pd.DataFrame,
    variables: Optional[list[str]] = None,
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Visualize spatial effects decomposition (direct, indirect, total).

    Parameters
    ----------
    effects_df : pd.DataFrame
        DataFrame with effects (from compute_spatial_effects)
    variables : list of str, optional
        Variables to plot (if None, plots all)
    figsize : tuple, default (10, 6)
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig, ax : matplotlib figure and axis

    Examples
    --------
    >>> from panelbox.effects import compute_spatial_effects
    >>> effects = compute_spatial_effects(results, W)
    >>> plot_effects_decomposition(effects)
    """
    if variables is None:
        variables = effects_df.index.tolist()

    # Prepare data
    effects_subset = effects_df.loc[variables]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(variables))
    width = 0.25

    # Bar positions
    ax.bar(x - width, effects_subset["Direct"], width, label="Direct", alpha=0.8)
    ax.bar(x, effects_subset["Indirect"], width, label="Indirect", alpha=0.8)
    ax.bar(x + width, effects_subset["Total"], width, label="Total", alpha=0.8)

    # Labels and title
    ax.set_xlabel("Variables", fontsize=12)
    ax.set_ylabel("Effect Size", fontsize=12)
    ax.set_title("Spatial Effects Decomposition", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(variables, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_spatial_residuals(
    gdf: gpd.GeoDataFrame,
    residuals: np.ndarray,
    title: str = "Spatial Distribution of Residuals",
    cmap: str = "RdBu_r",
    figsize: tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
):
    """
    Map model residuals spatially.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with geometries
    residuals : np.ndarray
        Model residuals
    title : str
        Map title
    cmap : str, default "RdBu_r"
        Colormap (diverging recommended)
    figsize : tuple, default (12, 8)
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig, ax : matplotlib figure and axis

    Examples
    --------
    >>> residuals = results.resid
    >>> plot_spatial_residuals(gdf, residuals)
    """
    gdf_copy = gdf.copy()
    gdf_copy["residuals"] = residuals

    fig, ax = plt.subplots(figsize=figsize)

    # Plot with diverging colormap centered at zero
    vmax = np.abs(residuals).max()
    gdf_copy.plot(
        column="residuals",
        ax=ax,
        cmap=cmap,
        edgecolor="black",
        linewidth=0.5,
        legend=True,
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_title(title, fontsize=14)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_variable_distribution(
    data: pd.DataFrame,
    variable: str,
    figsize: tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
):
    """
    Plot histogram and summary statistics for a variable.

    Parameters
    ----------
    data : pd.DataFrame
        Data with variable
    variable : str
        Column name to visualize
    figsize : tuple, default (10, 5)
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig, axes : matplotlib figure and axes

    Examples
    --------
    >>> plot_variable_distribution(df, variable="income_pc")
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    axes[0].hist(data[variable].dropna(), bins=30, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel(variable, fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title(f"Histogram: {variable}", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Box plot
    axes[1].boxplot(data[variable].dropna(), vert=True)
    axes[1].set_ylabel(variable, fontsize=12)
    axes[1].set_title(f"Box Plot: {variable}", fontsize=14)
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add summary statistics
    stats_text = f"""
    Mean: {data[variable].mean():.2f}
    Median: {data[variable].median():.2f}
    Std: {data[variable].std():.2f}
    Min: {data[variable].min():.2f}
    Max: {data[variable].max():.2f}
    """
    fig.text(0.5, -0.05, stats_text, ha="center", fontsize=10, family="monospace")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


if __name__ == "__main__":
    # Example usage
    print("Spatial Visualization Utilities")
    print("=" * 50)
    print()
    print("Available functions:")
    print("  - plot_choropleth() - Choropleth maps")
    print("  - plot_spatial_connections() - Weights connections")
    print("  - plot_moran_scatterplot() - Moran scatterplot")
    print("  - plot_lisa_map() - LISA cluster map")
    print("  - plot_effects_decomposition() - Effects bar chart")
    print("  - plot_spatial_residuals() - Residual maps")
    print("  - plot_variable_distribution() - Histogram and boxplot")
    print()
    print("Import this module in notebooks to access these utilities.")
