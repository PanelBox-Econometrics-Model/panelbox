"""Visualization helpers for prediction and deployment tutorials."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_production_style():
    """Set clean, professional matplotlib style for production reports."""
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_actual_vs_predicted(
    actual,
    predicted,
    title="Actual vs Predicted",
    save_path=None,
) -> plt.Figure:
    """
    Scatter plot of actual vs predicted with 45-degree line.

    Parameters
    ----------
    actual : array-like
        Observed values.
    predicted : array-like
        Predicted values.
    title : str
        Plot title.
    save_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    actual, predicted = np.asarray(actual), np.asarray(predicted)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual, predicted, alpha=0.5, s=20, color="steelblue")

    # 45-degree line
    lims = [
        min(actual.min(), predicted.min()),
        max(actual.max(), predicted.max()),
    ]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")

    # R-squared annotation
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    ax.annotate(
        f"R² = {r2:.3f}\nN = {len(actual)}",
        xy=(0.05, 0.92),
        xycoords="axes fraction",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5},
    )

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_prediction_errors(
    actual,
    predicted,
    title="Prediction Error Distribution",
    save_path=None,
) -> plt.Figure:
    """
    Histogram of prediction errors with statistics.

    Parameters
    ----------
    actual : array-like
        Observed values.
    predicted : array-like
        Predicted values.
    title : str
        Plot title.
    save_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    errors = np.asarray(actual) - np.asarray(predicted)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(
        errors.mean(),
        color="orange",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean = {errors.mean():.3f}",
    )

    stats_text = (
        f"Mean: {errors.mean():.4f}\n"
        f"Std:  {errors.std():.4f}\n"
        f"Min:  {errors.min():.4f}\n"
        f"Max:  {errors.max():.4f}"
    )
    ax.annotate(
        stats_text,
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.8},
        fontfamily="monospace",
    )

    ax.set_xlabel("Prediction Error (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_forecast_trajectory(
    historical: pd.Series,
    forecast: pd.Series,
    ci_lower=None,
    ci_upper=None,
    actual_future=None,
    title="Forecast",
    save_path=None,
) -> plt.Figure:
    """
    Plot historical + forecast trajectory with optional CI.

    Parameters
    ----------
    historical : pd.Series
        Historical observations (index = time).
    forecast : pd.Series
        Forecast values (index = future time).
    ci_lower : pd.Series, optional
        Lower confidence interval.
    ci_upper : pd.Series, optional
        Upper confidence interval.
    actual_future : pd.Series, optional
        Actual future values for comparison.
    title : str
        Plot title.
    save_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(historical.index, historical.values, "b-", linewidth=1.5, label="Historical")
    ax.plot(
        forecast.index,
        forecast.values,
        "r--",
        linewidth=2,
        marker="o",
        markersize=5,
        label="Forecast",
    )

    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(
            forecast.index,
            ci_lower.values,
            ci_upper.values,
            alpha=0.2,
            color="red",
            label="95% CI",
        )

    if actual_future is not None:
        ax.plot(
            actual_future.index,
            actual_future.values,
            "g-",
            linewidth=1.5,
            marker="s",
            markersize=5,
            label="Actual",
        )

    # Vertical line at forecast start
    ax.axvline(historical.index[-1], color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_model_comparison(
    evaluation_table: pd.DataFrame,
    metric="RMSE",
    save_path=None,
) -> plt.Figure:
    """
    Bar chart comparing models on a given metric.

    Parameters
    ----------
    evaluation_table : pd.DataFrame
        Output of forecast_evaluation_table().
    metric : str
        Column name to plot (e.g., "RMSE", "MAE").
    save_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(evaluation_table)))
    bars = ax.bar(evaluation_table.index, evaluation_table[metric], color=colors, edgecolor="gray")

    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    ax.set_ylabel(metric)
    ax.set_title(f"Model Comparison: {metric}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_coefficient_drift(
    versions: list,
    save_path=None,
) -> plt.Figure:
    """
    Show how coefficients change across model versions.

    Parameters
    ----------
    versions : list of dict
        Each dict has 'version' (str) and 'params' (dict of param_name: value).
    save_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Collect all param names
    all_params = set()
    for v in versions:
        all_params.update(v["params"].keys())
    all_params = sorted(all_params)

    version_labels = [v["version"] for v in versions]
    n_versions = len(versions)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_versions)
    width = 0.8 / len(all_params)

    for i, param in enumerate(all_params):
        values = [v["params"].get(param, np.nan) for v in versions]
        offset = (i - len(all_params) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=param, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(version_labels)
    ax.set_xlabel("Model Version")
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Coefficient Drift Across Versions")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
