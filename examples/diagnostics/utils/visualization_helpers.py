"""
Visualization utilities for diagnostics tutorials.

Provides consistent, publication-quality plotting functions for panel data
diagnostic test results, time series grids, and multi-panel dashboards.

Functions:
- plot_test_comparison: Bar chart comparing test statistics/p-values
- plot_time_series_grid: Grid of time series plots for selected entities
- create_diagnostic_dashboard: Multi-panel figure summarizing all diagnostics
- set_diagnostics_style: Configure matplotlib for publication-quality figures
"""

from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_diagnostics_style():
    """
    Set matplotlib style for publication-quality diagnostic figures.

    Configures seaborn-v0_8-whitegrid (with fallbacks), Set2 palette,
    standard figure size (12, 7), 100 dpi, and font sizes suitable for
    academic papers and presentations.
    """
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("ggplot")

    sns.set_palette("Set2")
    plt.rcParams["figure.figsize"] = (12, 7)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12


def plot_test_comparison(
    results_df: pd.DataFrame,
    metric: str = "pvalue",
    alpha: float = 0.05,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing test statistics or p-values across multiple tests.

    Draws a horizontal dashed line at the significance level ``alpha``
    when ``metric='pvalue'`` so the reader can immediately see which
    tests reject their respective null hypotheses.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with at least a ``'Test'`` column and a column matching
        ``metric`` (e.g. ``'pvalue'``, ``'statistic'``).  Each row is one
        diagnostic test.  Optional columns ``'H0'`` and ``'Decision'`` are
        used for annotation when present.
    metric : str, default 'pvalue'
        Column name in *results_df* to plot.  Typical choices are
        ``'pvalue'`` and ``'statistic'``.
    alpha : float, default 0.05
        Significance level.  A horizontal decision-threshold line is drawn
        at this value when ``metric='pvalue'``.
    save_path : str, optional
        If provided, the figure is saved to this path at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "Test": ["LLC", "IPS", "ADF-Fisher"],
    ...         "statistic": [-3.2, -2.8, 45.1],
    ...         "pvalue": [0.001, 0.003, 0.012],
    ...         "H0": ["Unit root"] * 3,
    ...         "Decision": ["Reject"] * 3,
    ...     }
    ... )
    >>> fig = plot_test_comparison(df, metric="pvalue")
    """
    set_diagnostics_style()

    if metric not in results_df.columns:
        raise ValueError(
            f"Column '{metric}' not found in results_df. "
            f"Available columns: {list(results_df.columns)}"
        )

    fig, ax = plt.subplots(figsize=(12, 7))

    test_names = results_df["Test"].values
    values = results_df[metric].astype(float).values
    n_tests = len(test_names)
    x_pos = np.arange(n_tests)

    # Color bars by reject/fail-to-reject when plotting p-values
    if metric == "pvalue":
        colors = []
        palette = sns.color_palette("Set2", n_colors=2)
        for v in values:
            if np.isfinite(v) and v < alpha:
                colors.append(palette[0])  # reject colour (green-ish)
            else:
                colors.append(palette[1])  # fail-to-reject (orange-ish)
    else:
        colors = sns.color_palette("Set2", n_colors=n_tests)

    bars = ax.bar(x_pos, values, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)

    # Annotate bar values on top
    for _i, (bar, val) in enumerate(zip(bars, values)):
        if np.isfinite(val):
            label_text = f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(abs(values[np.isfinite(values)])) * 0.02,
                label_text,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Decision threshold line for p-values
    if metric == "pvalue":
        ax.axhline(
            y=alpha,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Significance level ({alpha})",
        )
        ax.set_ylabel("P-value", fontsize=12)
        ax.set_title(
            "Diagnostic Test P-values Comparison",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=10, loc="best")
    else:
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(
            f"Diagnostic Test Comparison ({metric.replace('_', ' ').title()})",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(test_names, rotation=30, ha="right", fontsize=11)
    ax.set_xlabel("Test", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_time_series_grid(
    data: pd.DataFrame,
    variable: str,
    entity_col: str,
    time_col: str,
    n_entities: int = 9,
    ncols: int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grid of time series plots for a random sample of panel entities.

    Useful for quick visual inspection of the dependent variable across
    cross-sectional units before running formal diagnostic tests.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format containing at least ``variable``,
        ``entity_col``, and ``time_col`` columns.
    variable : str
        Name of the variable to plot (y-axis).
    entity_col : str
        Name of the entity/individual identifier column.
    time_col : str
        Name of the time column.
    n_entities : int, default 9
        Number of entities to sample.  If the dataset has fewer entities,
        all are plotted.
    ncols : int, default 3
        Number of columns in the subplot grid.
    save_path : str, optional
        If provided, the figure is saved to this path at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.

    Examples
    --------
    >>> fig = plot_time_series_grid(
    ...     panel_df,
    ...     variable="log_gdp",
    ...     entity_col="country",
    ...     time_col="year",
    ...     n_entities=9,
    ...     ncols=3,
    ... )
    """
    set_diagnostics_style()

    unique_entities = data[entity_col].unique()
    n_available = len(unique_entities)
    n_plot = min(n_entities, n_available)

    # Reproducible random sample
    rng = np.random.RandomState(42)
    sampled = rng.choice(unique_entities, size=n_plot, replace=False)

    nrows = int(np.ceil(n_plot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    palette = sns.color_palette("Set2", n_colors=n_plot)

    for idx, entity in enumerate(sampled):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        entity_data = data[data[entity_col] == entity].sort_values(time_col)
        time_vals = entity_data[time_col].values
        var_vals = entity_data[variable].values

        ax.plot(
            time_vals,
            var_vals,
            color=palette[idx],
            linewidth=1.5,
            marker="o",
            markersize=3,
        )

        # Add a subtle mean line
        if len(var_vals) > 0:
            mean_val = np.nanmean(var_vals)
            ax.axhline(
                mean_val,
                color="gray",
                linestyle=":",
                linewidth=1,
                alpha=0.6,
            )

        ax.set_title(f"{entity_col}: {entity}", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)

        # Only add axis labels at edges
        if row == nrows - 1:
            ax.set_xlabel(time_col, fontsize=10)
        if col == 0:
            ax.set_ylabel(variable, fontsize=10)

    # Hide unused subplots
    for idx in range(n_plot, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Time Series of '{variable}' (Sample of {n_plot} entities)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_diagnostic_dashboard(
    test_results: dict[str, dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-panel figure summarizing all diagnostic test results.

    Produces a four-panel dashboard:

    1. **P-value comparison** -- bar chart of all test p-values with an
       ``alpha = 0.05`` decision line.
    2. **Test statistics** -- bar chart of test-statistic magnitudes.
    3. **Decision summary** -- colour-coded table showing reject/fail.
    4. **Interpretation notes** -- text panel listing null hypotheses and
       conclusions.

    Parameters
    ----------
    test_results : dict of dict
        Mapping ``{test_name: info_dict}`` where each *info_dict* must
        contain at least ``'statistic'`` and ``'pvalue'`` keys.  Optional
        keys: ``'H0'`` (null-hypothesis string), ``'reject'`` (bool),
        ``'alpha'`` (float, default 0.05).
    save_path : str, optional
        If provided, the figure is saved to this path at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.

    Examples
    --------
    >>> results = {
    ...     "LLC": {"statistic": -3.21, "pvalue": 0.001, "H0": "Common unit root", "reject": True},
    ...     "IPS": {
    ...         "statistic": -2.80,
    ...         "pvalue": 0.003,
    ...         "H0": "Individual unit root",
    ...         "reject": True,
    ...     },
    ...     "Hadri": {"statistic": 4.50, "pvalue": 0.00, "H0": "Stationarity", "reject": True},
    ... }
    >>> fig = create_diagnostic_dashboard(results)
    """
    set_diagnostics_style()

    test_names = list(test_results.keys())
    statistics = [test_results[t].get("statistic", np.nan) for t in test_names]
    pvalues = [test_results[t].get("pvalue", np.nan) for t in test_names]
    hypotheses = [test_results[t].get("H0", "Not specified") for t in test_names]
    n_tests = len(test_names)

    # Resolve reject decisions
    reject_flags = []
    for t in test_names:
        info = test_results[t]
        if "reject" in info:
            reject_flags.append(bool(info["reject"]))
        else:
            test_alpha = info.get("alpha", 0.05)
            pval = info.get("pvalue", np.nan)
            reject_flags.append(bool(np.isfinite(pval) and pval < test_alpha))

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.10, wspace=0.10)

    palette = sns.color_palette("Set2", n_colors=max(n_tests, 2))

    # ---- Panel 1: P-value comparison (top-left) ----
    ax1 = fig.add_subplot(gs[0, 0])
    p_colors = [palette[0] if r else palette[1] for r in reject_flags]
    bars1 = ax1.bar(
        np.arange(n_tests),
        pvalues,
        color=p_colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )
    ax1.axhline(0.05, color="red", linestyle="--", linewidth=2, label="alpha = 0.05")
    ax1.set_xticks(np.arange(n_tests))
    ax1.set_xticklabels(test_names, rotation=30, ha="right", fontsize=10)
    ax1.set_ylabel("P-value", fontsize=11)
    ax1.set_title("P-value Comparison", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Annotate p-values
    for bar, pval in zip(bars1, pvalues):
        if np.isfinite(pval):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{pval:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # ---- Panel 2: Test statistics (top-right) ----
    ax2 = fig.add_subplot(gs[0, 1])
    stat_colors = sns.color_palette("Set2", n_colors=n_tests)
    bars2 = ax2.bar(
        np.arange(n_tests),
        statistics,
        color=stat_colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_xticks(np.arange(n_tests))
    ax2.set_xticklabels(test_names, rotation=30, ha="right", fontsize=10)
    ax2.set_ylabel("Test Statistic", fontsize=11)
    ax2.set_title("Test Statistics", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # Annotate statistics
    for bar, stat in zip(bars2, statistics):
        if np.isfinite(stat):
            y_offset = abs(stat) * 0.03 if stat >= 0 else -abs(stat) * 0.03
            va = "bottom" if stat >= 0 else "top"
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + y_offset,
                f"{stat:.3f}",
                ha="center",
                va=va,
                fontsize=8,
            )

    # ---- Panel 3: Decision summary table (bottom-left) ----
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")

    table_data = []
    cell_colours = []
    reject_color = "#a8d5a2"  # light green
    fail_color = "#f5b7b1"  # light red
    header_color = "#85c1e9"  # light blue

    for i, t in enumerate(test_names):
        decision = "Reject H0" if reject_flags[i] else "Fail to Reject"
        pval_str = f"{pvalues[i]:.4f}" if np.isfinite(pvalues[i]) else "N/A"
        table_data.append([t, pval_str, decision])
        row_color = reject_color if reject_flags[i] else fail_color
        cell_colours.append(["white", "white", row_color])

    col_labels = ["Test", "P-value", "Decision"]
    table = ax3.table(
        cellText=table_data,
        colLabels=col_labels,
        cellColours=cell_colours,
        colColours=[header_color] * 3,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    ax3.set_title("Decision Summary", fontsize=13, fontweight="bold", pad=20)

    # ---- Panel 4: Interpretation notes (bottom-right) ----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    note_lines = ["Null Hypotheses & Conclusions", "-" * 40]
    for i, t in enumerate(test_names):
        h0 = hypotheses[i]
        decision = "REJECT" if reject_flags[i] else "FAIL TO REJECT"
        pval_str = f"{pvalues[i]:.4f}" if np.isfinite(pvalues[i]) else "N/A"
        note_lines.append(f"{t}:")
        note_lines.append(f"  H0: {h0}")
        note_lines.append(f"  p = {pval_str}  =>  {decision}")
        note_lines.append("")

    notes_text = "\n".join(note_lines)
    ax4.text(
        0.05,
        0.95,
        notes_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#f8f9fa",
            "edgecolor": "#dee2e6",
            "alpha": 0.9,
        },
    )
    ax4.set_title("Interpretation", fontsize=13, fontweight="bold", pad=20)

    fig.suptitle(
        "Diagnostic Tests Dashboard",
        fontsize=16,
        fontweight="bold",
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "create_diagnostic_dashboard",
    "plot_test_comparison",
    "plot_time_series_grid",
    "set_diagnostics_style",
]


if __name__ == "__main__":
    print("Visualization helpers for diagnostics tutorials loaded successfully!")
    print("Functions available:")
    print("  - set_diagnostics_style")
    print("  - plot_test_comparison")
    print("  - plot_time_series_grid")
    print("  - create_diagnostic_dashboard")
