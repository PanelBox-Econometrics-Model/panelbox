"""
Visualization helper functions for count models tutorials.

Provides consistent, publication-quality plotting functions.

Functions:
- plot_rootogram: Hanging rootogram for model fit
- plot_variance_mean: Variance-mean relationship
- plot_marginal_effects: Marginal effects with CIs
- plot_irr_forest: Forest plot of IRRs
- compare_models_plot: Model comparison visualization
- plot_panel_trends: Panel data trends
- plot_zero_inflation: Zero-inflation diagnostics
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def plot_rootogram(observed, expected, breaks=None, ax=None, **kwargs):
    """
    Create a hanging rootogram for assessing model fit.

    A rootogram displays the square roots of observed and expected frequencies,
    making it easier to detect patterns in the residuals.

    Parameters
    ----------
    observed : array-like
        Observed counts
    expected : array-like
        Expected counts from fitted model
    breaks : array-like, optional
        Bin edges. If None, uses unique values in observed.
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates new figure.
    **kwargs : dict
        Additional arguments passed to plt.bar()

    Returns
    -------
    ax : matplotlib axis
        The axis object with the plot

    Examples
    --------
    >>> from panelbox.models.count import PooledPoisson
    >>> model = PooledPoisson.from_formula('y ~ x', data=df)
    >>> result = model.fit()
    >>> plot_rootogram(df['y'], result.predict())

    Notes
    -----
    In a well-fitted model, the hanging bars should oscillate randomly
    around zero. Systematic patterns indicate misspecification.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    observed = np.asarray(observed)
    expected = np.asarray(expected)

    if breaks is None:
        breaks = np.arange(observed.min(), observed.max() + 2)

    # Compute frequencies
    obs_freq, _ = np.histogram(observed, bins=breaks)
    exp_freq, _ = np.histogram(expected, bins=breaks)

    # Square root transformation
    obs_sqrt = np.sqrt(obs_freq)
    exp_sqrt = np.sqrt(exp_freq)

    # Hanging: observed hangs from expected
    x = breaks[:-1] + 0.5
    height = obs_sqrt - exp_sqrt

    # Plot
    colors = ["red" if h < 0 else "steelblue" for h in height]
    ax.bar(x, height, bottom=exp_sqrt, color=colors, alpha=0.6, **kwargs)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.plot(x, exp_sqrt, "o-", color="darkred", linewidth=2, label="Expected")

    ax.set_xlabel("Count Value", fontsize=12)
    ax.set_ylabel("Square Root of Frequency", fontsize=12)
    ax.set_title("Hanging Rootogram", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    return ax


def plot_variance_mean(data, y_col, group_cols=None, ax=None):
    """
    Plot variance-mean relationship to assess overdispersion.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing outcome and grouping variables
    y_col : str
        Name of outcome column
    group_cols : list of str, optional
        Columns to group by for computing within-group means and variances
    ax : matplotlib axis, optional
        Axis to plot on

    Returns
    -------
    ax : matplotlib axis

    Notes
    -----
    For Poisson, variance = mean (45-degree line).
    Overdispersion: points above 45-degree line.
    Underdispersion: points below 45-degree line.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if group_cols is None:
        # Use rolling window or bins
        sorted_data = data.sort_values(y_col)
        window = max(10, len(data) // 20)
        means = sorted_data[y_col].rolling(window).mean()
        variances = sorted_data[y_col].rolling(window).var()
    else:
        # Group-wise means and variances
        grouped = data.groupby(group_cols)[y_col]
        means = grouped.mean()
        variances = grouped.var()

    # Remove NaN
    mask = ~(np.isnan(means) | np.isnan(variances))
    means = means[mask]
    variances = variances[mask]

    # Plot
    ax.scatter(means, variances, alpha=0.6, s=50)

    # Reference line (Poisson: Var = Mean)
    max_val = max(means.max(), variances.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Var = Mean (Poisson)")

    ax.set_xlabel("Mean", fontsize=12)
    ax.set_ylabel("Variance", fontsize=12)
    ax.set_title("Variance-Mean Relationship", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    return ax


def plot_marginal_effects(me_result, var_name=None, ax=None):
    """
    Plot marginal effects with confidence intervals.

    Parameters
    ----------
    me_result : MarginalEffectsResult
        Result from count_me() function
    var_name : str, optional
        Specific variable to plot. If None, plots all.
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Extract marginal effects and CIs
    summary = me_result.summary_frame()

    if var_name:
        summary = summary[summary.index == var_name]

    # Plot
    x = range(len(summary))
    ax.errorbar(
        x,
        summary["dy/dx"],
        yerr=[
            summary["dy/dx"] - summary["Conf. Int. Low"],
            summary["Conf. Int. High"] - summary["dy/dx"],
        ],
        fmt="o",
        markersize=8,
        capsize=5,
        capthick=2,
    )

    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=45, ha="right")
    ax.set_ylabel("Marginal Effect", fontsize=12)
    ax.set_title("Average Marginal Effects", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return ax


def plot_irr_forest(results_dict, ax=None):
    """
    Create forest plot of incidence rate ratios.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping variable names to (IRR, CI_low, CI_high) tuples
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    vars_list = list(results_dict.keys())
    y_pos = np.arange(len(vars_list))

    irrs = [results_dict[v][0] for v in vars_list]
    ci_lows = [results_dict[v][1] for v in vars_list]
    ci_highs = [results_dict[v][2] for v in vars_list]

    # Plot
    ax.errorbar(
        irrs,
        y_pos,
        xerr=[np.array(irrs) - np.array(ci_lows), np.array(ci_highs) - np.array(irrs)],
        fmt="o",
        markersize=8,
        capsize=5,
        capthick=2,
    )

    ax.axvline(1, color="red", linestyle="--", linewidth=2, label="IRR = 1 (no effect)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(vars_list)
    ax.set_xlabel("Incidence Rate Ratio", fontsize=12)
    ax.set_title("Forest Plot of IRRs with 95% CIs", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return ax


def compare_models_plot(models_dict, metric="aic", ax=None):
    """
    Compare multiple models using information criteria.

    Parameters
    ----------
    models_dict : dict
        Dictionary mapping model names to fitted results
    metric : str, default 'aic'
        Metric to plot: 'aic', 'bic', or 'loglik'
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    model_names = list(models_dict.keys())

    if metric == "aic":
        values = [models_dict[m].aic for m in model_names]
        ylabel = "AIC (lower is better)"
    elif metric == "bic":
        values = [models_dict[m].bic for m in model_names]
        ylabel = "BIC (lower is better)"
    elif metric == "loglik":
        values = [models_dict[m].llf for m in model_names]
        ylabel = "Log-Likelihood (higher is better)"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Plot
    x_pos = np.arange(len(model_names))
    bars = ax.bar(x_pos, values, alpha=0.7, color="steelblue", edgecolor="black")

    # Highlight best model
    if metric == "loglik":
        best_idx = np.argmax(values)
    else:
        best_idx = np.argmin(values)
    bars[best_idx].set_color("green")
    bars[best_idx].set_alpha(0.9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return ax


def plot_panel_trends(data, id_col, time_col, y_col, n_ids=10, ax=None):
    """
    Plot time trends for random sample of panel units.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    id_col : str
        Name of entity ID column
    time_col : str
        Name of time column
    y_col : str
        Name of outcome column
    n_ids : int, default 10
        Number of random units to plot
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Sample random IDs
    unique_ids = data[id_col].unique()
    sampled_ids = np.random.choice(unique_ids, min(n_ids, len(unique_ids)), replace=False)

    # Plot each unit
    for unit_id in sampled_ids:
        unit_data = data[data[id_col] == unit_id].sort_values(time_col)
        ax.plot(unit_data[time_col], unit_data[y_col], marker="o", alpha=0.6, label=f"ID {unit_id}")

    ax.set_xlabel(time_col.capitalize(), fontsize=12)
    ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)
    ax.set_title(
        f"Panel Trends (Random Sample of {len(sampled_ids)} Units)", fontsize=14, fontweight="bold"
    )
    ax.grid(alpha=0.3)
    if n_ids <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    return ax


def plot_zero_inflation(data, y_col, group_col=None, ax=None):
    """
    Visualize zero-inflation in count data.

    Parameters
    ----------
    data : pd.DataFrame
        Data
    y_col : str
        Outcome column
    group_col : str, optional
        Grouping variable for comparison
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if group_col is None:
        # Overall distribution
        counts = data[y_col].value_counts().sort_index()
        ax.bar(counts.index, counts.values, alpha=0.7, color="steelblue", edgecolor="black")

        zero_pct = (data[y_col] == 0).mean() * 100
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label=f"Zeros: {zero_pct:.1f}%")
    else:
        # By group
        for group in data[group_col].unique():
            group_data = data[data[group_col] == group][y_col]
            counts = group_data.value_counts().sort_index()
            ax.plot(counts.index, counts.values, marker="o", label=f"{group_col}={group}")

    ax.set_xlabel(y_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Counts (Zero-Inflation Check)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    return ax


if __name__ == "__main__":
    # Test plotting functions
    print("Visualization helpers module loaded successfully!")
    print("Functions available:")
    print("  - plot_rootogram")
    print("  - plot_variance_mean")
    print("  - plot_marginal_effects")
    print("  - plot_irr_forest")
    print("  - compare_models_plot")
    print("  - plot_panel_trends")
    print("  - plot_zero_inflation")
