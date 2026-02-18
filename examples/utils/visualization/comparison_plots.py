"""
Model Comparison Visualization Utilities

This module provides plotting functions for comparing panel model results.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def coefficient_plot(
    results_dict: Dict[str, object],
    variables: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    conf_level: float = 0.95,
) -> plt.Axes:
    """
    Create coefficient plot comparing estimates across models.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to PanelResults objects
    variables : list of str, optional
        Variables to include. If None, uses all common variables
    figsize : tuple, optional
        Figure size (width, height)
    conf_level : float, optional
        Confidence level for intervals (default: 0.95)

    Returns
    -------
    plt.Axes
        Matplotlib axes object

    Examples
    --------
    >>> results = {
    ...     'Pooled OLS': pooled_result,
    ...     'Fixed Effects': fe_result,
    ...     'Random Effects': re_result
    ... }
    >>> coefficient_plot(results)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract coefficients and standard errors
    n_models = len(results_dict)
    model_names = list(results_dict.keys())

    if variables is None:
        # Find common variables across all models
        all_vars = [set(res.params.index) for res in results_dict.values()]
        variables = sorted(list(set.intersection(*all_vars)))

    n_vars = len(variables)
    y_positions = np.arange(n_vars)
    offset = np.linspace(-0.3, 0.3, n_models)

    z_score = 1.96 if conf_level == 0.95 else 2.576  # 95% or 99%

    for i, (model_name, result) in enumerate(results_dict.items()):
        coeffs = [result.params.get(var, np.nan) for var in variables]
        std_errs = [result.std_errors.get(var, np.nan) for var in variables]
        ci_lower = [c - z_score * se for c, se in zip(coeffs, std_errs)]
        ci_upper = [c + z_score * se for c, se in zip(coeffs, std_errs)]

        ax.errorbar(
            coeffs,
            y_positions + offset[i],
            xerr=[(c - l, u - c) for c, l, u in zip(coeffs, ci_lower, ci_upper)],
            fmt="o",
            label=model_name,
            capsize=5,
            markersize=6,
        )

    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(variables)
    ax.set_xlabel("Coefficient Estimate")
    ax.set_ylabel("Variable")
    ax.set_title(f"Coefficient Comparison Across Models ({int(conf_level*100)}% CI)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return ax


def r_squared_comparison(
    results_dict: Dict[str, object], figsize: Tuple[int, int] = (10, 6)
) -> plt.Axes:
    """
    Compare R-squared values across models.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to PanelResults objects
    figsize : tuple, optional
        Figure size (width, height)

    Returns
    -------
    plt.Axes
        Matplotlib axes object

    Examples
    --------
    >>> r_squared_comparison({'Pooled': pooled, 'FE': fe, 'RE': re})
    """
    model_names = list(results_dict.keys())
    r2_values = []
    r2_adj_values = []

    for result in results_dict.values():
        r2_values.append(getattr(result, "r_squared", np.nan))
        r2_adj_values.append(getattr(result, "r_squared_adj", np.nan))

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, r2_values, width, label="R²", color="#3498db")
    ax.bar(x + width / 2, r2_adj_values, width, label="Adjusted R²", color="#2ecc71")

    ax.set_ylabel("R-squared Value")
    ax.set_title("R-squared Comparison Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    return ax


def model_metrics_table(
    results_dict: Dict[str, object], metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create comparison table of model fit metrics.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to PanelResults objects
    metrics : list of str, optional
        Metrics to include. Default: ['r_squared', 'r_squared_adj', 'aic', 'bic', 'nobs']

    Returns
    -------
    pd.DataFrame
        Comparison table

    Examples
    --------
    >>> table = model_metrics_table({'Pooled': pooled, 'FE': fe})
    >>> print(table)
    """
    if metrics is None:
        metrics = ["r_squared", "r_squared_adj", "aic", "bic", "nobs"]

    comparison = {}
    for model_name, result in results_dict.items():
        comparison[model_name] = {metric: getattr(result, metric, np.nan) for metric in metrics}

    df = pd.DataFrame(comparison).T
    return df.round(4)
