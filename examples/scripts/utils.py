"""
Utility functions for PanelBox tutorial notebooks.

This module provides helper functions to reduce code duplication
across tutorial notebooks and enhance the learning experience.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_panel_structure(
    data: pd.DataFrame, entity_col: str, time_col: str, figsize: tuple[int, int] = (12, 6)
) -> None:
    """
    Visualize panel data structure showing entities over time.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset
    entity_col : str
        Name of entity identifier column
    time_col : str
        Name of time identifier column
    figsize : tuple
        Figure size (width, height)

    Examples
    --------
    >>> plot_panel_structure(df, "firm", "year")
    """
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Heatmap of data availability
    pivot = data.groupby([entity_col, time_col]).size().unstack(fill_value=0)
    pivot = (pivot > 0).astype(int)

    sns.heatmap(pivot, cmap="Blues", cbar=False, ax=ax1, linewidths=0.5, linecolor="gray")
    ax1.set_title("Panel Structure: Entity × Time Coverage")
    ax1.set_xlabel("Time Period")
    ax1.set_ylabel("Entity")

    # Plot 2: Observations per period
    obs_per_period = data.groupby(time_col).size()
    ax2.bar(obs_per_period.index, obs_per_period.values, color="steelblue")
    ax2.set_title("Observations per Time Period")
    ax2.set_xlabel("Time Period")
    ax2.set_ylabel("Number of Observations")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def summary_stats(
    data: pd.DataFrame, variables: Optional[list[str]] = None, by_group: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate comprehensive summary statistics table.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    variables : list of str, optional
        Variables to summarize (defaults to all numeric)
    by_group : str, optional
        Group variable for stratified statistics

    Returns
    -------
    pd.DataFrame
        Summary statistics table

    Examples
    --------
    >>> summary_stats(df, variables=["invest", "value", "capital"])
    >>> summary_stats(df, by_group="firm")
    """
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    if by_group is None:
        stats = data[variables].describe().T
        stats["cv"] = stats["std"] / stats["mean"]  # Coefficient of variation
        return stats.round(4)
    else:
        return data.groupby(by_group)[variables].describe().round(4)


def plot_residual_diagnostics(
    residuals: np.ndarray, fitted: np.ndarray, figsize: tuple[int, int] = (14, 5)
) -> None:
    """
    Create diagnostic plots for regression residuals.

    Parameters
    ----------
    residuals : np.ndarray
        Regression residuals
    fitted : np.ndarray
        Fitted values
    figsize : tuple
        Figure size

    Examples
    --------
    >>> plot_residual_diagnostics(model.resid, model.fitted_values)
    """
    _fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Residuals vs Fitted
    axes[0].scatter(fitted, residuals, alpha=0.5, s=30)
    axes[0].axhline(y=0, color="r", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")
    axes[0].grid(alpha=0.3)

    # Plot 2: Q-Q Plot
    from scipy import stats

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q Plot")
    axes[1].grid(alpha=0.3)

    # Plot 3: Histogram
    axes[2].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    axes[2].set_xlabel("Residuals")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Histogram of Residuals")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_models(
    *models,
    model_names: Optional[list[str]] = None,
    stats: list[str] = None,
) -> pd.DataFrame:
    """
    Compare fit statistics across multiple models.

    Parameters
    ----------
    *models : PanelResults objects
        Fitted model results to compare
    model_names : list of str, optional
        Names for models (defaults to Model 1, Model 2, ...)
    stats : list of str
        Statistics to compare

    Returns
    -------
    pd.DataFrame
        Comparison table

    Examples
    --------
    >>> compare_models(model1, model2, model3, model_names=["Pooled", "FE", "RE"])
    """
    if stats is None:
        stats = ["rsquared", "rsquared_adj", "aic", "bic"]
    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(len(models))]

    comparison = {}
    for name, model in zip(model_names, models):
        comparison[name] = {stat: getattr(model, stat, np.nan) for stat in stats}

    return pd.DataFrame(comparison).T.round(4)


def export_results_table(
    model, format: str = "latex", filename: Optional[str] = None, **kwargs
) -> str:
    """
    Export regression results to formatted table.

    Parameters
    ----------
    model : PanelResults
        Fitted model results
    format : str
        Output format ('latex', 'markdown', 'html')
    filename : str, optional
        File to save output (if None, returns string)
    **kwargs
        Additional arguments passed to formatter

    Returns
    -------
    str
        Formatted table

    Examples
    --------
    >>> latex_table = export_results_table(model, format="latex")
    >>> export_results_table(model, format="html", filename="results.html")
    """
    if format == "latex":
        output = model.summary.tables[1].as_latex_tabular()
    elif format == "markdown":
        output = model.summary.tables[1].as_text()
    elif format == "html":
        output = model.summary.tables[1].as_html()
    else:
        raise ValueError(f"Format '{format}' not supported")

    if filename:
        with open(filename, "w") as f:
            f.write(output)

    return output


def validate_panel_data(data: pd.DataFrame, entity_col: str, time_col: str) -> dict:
    """
    Validate panel data structure and report issues.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset
    entity_col : str
        Entity identifier column
    time_col : str
        Time identifier column

    Returns
    -------
    dict
        Validation report with issues and warnings

    Examples
    --------
    >>> report = validate_panel_data(df, "firm", "year")
    >>> print(report["summary"])
    """
    report = {
        "n_entities": data[entity_col].nunique(),
        "n_periods": data[time_col].nunique(),
        "n_obs": len(data),
        "is_balanced": False,
        "missing_cells": 0,
        "duplicate_entries": 0,
        "issues": [],
    }

    # Check for duplicates
    duplicates = data.duplicated(subset=[entity_col, time_col], keep=False)
    report["duplicate_entries"] = duplicates.sum()
    if duplicates.any():
        report["issues"].append(f"Found {duplicates.sum()} duplicate entries")

    # Check if balanced
    expected_obs = report["n_entities"] * report["n_periods"]
    report["missing_cells"] = expected_obs - report["n_obs"]
    report["is_balanced"] = report["n_obs"] == expected_obs

    if not report["is_balanced"]:
        report["issues"].append(
            f"Panel is unbalanced: {report['missing_cells']} missing observations"
        )

    # Summary message
    if not report["issues"]:
        report["summary"] = "✓ Panel structure is valid and balanced"
    else:
        report["summary"] = "⚠ Issues detected:\n  - " + "\n  - ".join(report["issues"])

    return report


# Configure default plotting style for tutorials
def set_tutorial_style():
    """Set consistent plotting style for all tutorial notebooks."""
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["legend.fontsize"] = 10


# Automatically apply style when module is imported
set_tutorial_style()
