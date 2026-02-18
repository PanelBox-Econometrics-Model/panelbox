"""
Panel Data Visualization Utilities

This module provides plotting functions for panel data analysis.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def spaghetti_plot(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    y_col: str,
    ax: Optional[plt.Axes] = None,
    sample_entities: Optional[int] = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot individual trajectories over time (spaghetti plot).

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format
    entity_col : str
        Name of entity identifier column
    time_col : str
        Name of time identifier column
    y_col : str
        Name of variable to plot
    ax : plt.Axes, optional
        Matplotlib axes object. If None, creates new figure
    sample_entities : int, optional
        Number of entities to randomly sample. If None, plots all
    **kwargs
        Additional arguments passed to plt.plot

    Returns
    -------
    plt.Axes
        Matplotlib axes object

    Examples
    --------
    >>> import pandas as pd
    >>> from utils.visualization.panel_plots import spaghetti_plot
    >>> data = pd.DataFrame({
    ...     'firm': [1, 1, 2, 2],
    ...     'year': [2000, 2001, 2000, 2001],
    ...     'sales': [100, 120, 80, 95]
    ... })
    >>> spaghetti_plot(data, 'firm', 'year', 'sales')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    entities = data[entity_col].unique()
    if sample_entities is not None and sample_entities < len(entities):
        entities = np.random.choice(entities, size=sample_entities, replace=False)

    default_kwargs = {"alpha": 0.3, "linewidth": 0.8}
    default_kwargs.update(kwargs)

    for entity in entities:
        entity_data = data[data[entity_col] == entity].sort_values(time_col)
        ax.plot(entity_data[time_col], entity_data[y_col], **default_kwargs)

    ax.set_xlabel(time_col.capitalize())
    ax.set_ylabel(y_col.replace("_", " ").capitalize())
    ax.set_title(f'{y_col.replace("_", " ").capitalize()} Trajectories Over Time')
    ax.grid(True, alpha=0.3)

    return ax


def within_between_scatter(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    x_col: str,
    y_col: str,
    figsize: Tuple[int, int] = (14, 5),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Create within and between variation scatter plots.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format
    entity_col : str
        Name of entity identifier column
    time_col : str
        Name of time identifier column
    x_col : str
        Name of x-variable
    y_col : str
        Name of y-variable
    figsize : tuple, optional
        Figure size (width, height)

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    axes : tuple of plt.Axes
        Tuple of (within_ax, between_ax)

    Examples
    --------
    >>> fig, axes = within_between_scatter(data, 'firm', 'year', 'capital', 'invest')
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Within variation (demeaned)
    data_within = data.copy()
    entity_means_x = data.groupby(entity_col)[x_col].transform("mean")
    entity_means_y = data.groupby(entity_col)[y_col].transform("mean")
    data_within[f"{x_col}_demeaned"] = data[x_col] - entity_means_x
    data_within[f"{y_col}_demeaned"] = data[y_col] - entity_means_y

    ax1.scatter(data_within[f"{x_col}_demeaned"], data_within[f"{y_col}_demeaned"], alpha=0.5, s=20)
    ax1.set_xlabel(f"{x_col} (demeaned)")
    ax1.set_ylabel(f"{y_col} (demeaned)")
    ax1.set_title("Within Variation")
    ax1.grid(True, alpha=0.3)

    # Between variation (entity means)
    entity_means = data.groupby(entity_col)[[x_col, y_col]].mean().reset_index()
    ax2.scatter(entity_means[x_col], entity_means[y_col], alpha=0.5, s=50)
    ax2.set_xlabel(f"{x_col} (entity mean)")
    ax2.set_ylabel(f"{y_col} (entity mean)")
    ax2.set_title("Between Variation")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, (ax1, ax2)


def variance_decomposition_plot(
    data: pd.DataFrame, entity_col: str, variables: List[str], figsize: Tuple[int, int] = (10, 6)
) -> plt.Axes:
    """
    Plot variance decomposition (within vs between) for multiple variables.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format
    entity_col : str
        Name of entity identifier column
    variables : list of str
        List of variable names to decompose
    figsize : tuple, optional
        Figure size (width, height)

    Returns
    -------
    plt.Axes
        Matplotlib axes object

    Examples
    --------
    >>> variance_decomposition_plot(data, 'firm', ['invest', 'value', 'capital'])
    """
    decomposition = {}

    for var in variables:
        total_var = data[var].var()
        entity_means = data.groupby(entity_col)[var].mean()
        between_var = entity_means.var()
        within_var = total_var - between_var

        decomposition[var] = {
            "Within": within_var / total_var * 100,
            "Between": between_var / total_var * 100,
        }

    df_decomp = pd.DataFrame(decomposition).T

    fig, ax = plt.subplots(figsize=figsize)
    df_decomp.plot(kind="barh", stacked=True, ax=ax, color=["#2ecc71", "#3498db"])
    ax.set_xlabel("Percentage of Total Variance")
    ax.set_ylabel("Variables")
    ax.set_title("Variance Decomposition: Within vs Between")
    ax.legend(title="Component")
    ax.grid(True, alpha=0.3, axis="x")

    return ax
