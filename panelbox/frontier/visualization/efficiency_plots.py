"""
Efficiency distribution and ranking plots for SFA results.

This module provides visualization functions for:
- Efficiency distribution (histogram + KDE)
- Efficiency rankings (top/bottom performers)
- Efficiency by groups (box plots)
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


def plot_efficiency_distribution(
    efficiency_df: pd.DataFrame,
    backend: str = "plotly",
    bins: int = 30,
    show_kde: bool = True,
    show_stats: bool = True,
    title: Optional[str] = None,
    **kwargs,
) -> Any:
    """Plot distribution of efficiency scores with histogram and KDE.

    Parameters:
        efficiency_df: DataFrame with 'efficiency' column (from result.efficiency())
        backend: 'plotly' for interactive or 'matplotlib' for static
        bins: Number of histogram bins
        show_kde: Whether to overlay kernel density estimate
        show_stats: Whether to annotate with statistics
        title: Custom plot title
        **kwargs: Additional arguments passed to plotting backend

    Returns:
        Plotly Figure or Matplotlib Figure object

    Example:
        >>> result = sf.fit()
        >>> eff_df = result.efficiency(estimator='bc')
        >>> fig = plot_efficiency_distribution(eff_df, backend='plotly')
        >>> fig.show()
    """
    efficiency = efficiency_df["efficiency"].values

    # Compute statistics
    mean_eff = np.mean(efficiency)
    median_eff = np.median(efficiency)
    std_eff = np.std(efficiency)
    min_eff = np.min(efficiency)
    max_eff = np.max(efficiency)

    if title is None:
        title = "Efficiency Distribution"

    if backend == "plotly":
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = go.Figure()

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=efficiency,
                nbinsx=bins,
                name="Distribution",
                marker_color="lightblue",
                marker_line_color="darkblue",
                marker_line_width=1,
                opacity=0.7,
                histnorm="probability density" if show_kde else "",
            )
        )

        # KDE overlay
        if show_kde:
            kde = stats.gaussian_kde(efficiency)
            x_range = np.linspace(efficiency.min(), efficiency.max(), 200)
            kde_values = kde(x_range)

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde_values,
                    mode="lines",
                    name="KDE",
                    line=dict(color="red", width=2),
                )
            )

        # Add vertical lines for statistics
        if show_stats:
            # Mean line
            fig.add_vline(
                x=mean_eff,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Mean: {mean_eff:.4f}",
                annotation_position="top",
            )

            # Median line
            fig.add_vline(
                x=median_eff,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"Median: {median_eff:.4f}",
                annotation_position="bottom",
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Technical Efficiency",
            yaxis_title="Density" if show_kde else "Count",
            showlegend=True,
            template="plotly_white",
            hovermode="x unified",
            **kwargs,
        )

        # Add annotation box with statistics
        if show_stats:
            stats_text = (
                f"<b>Statistics</b><br>"
                f"Mean: {mean_eff:.4f}<br>"
                f"Median: {median_eff:.4f}<br>"
                f"Std Dev: {std_eff:.4f}<br>"
                f"Min: {min_eff:.4f}<br>"
                f"Max: {max_eff:.4f}"
            )

            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.98,
                xanchor="right",
                yanchor="top",
                text=stats_text,
                showarrow=False,
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
                opacity=0.8,
            )

        return fig

    elif backend == "matplotlib":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))

        # Histogram
        n, bins_edges, patches = ax.hist(
            efficiency,
            bins=bins,
            density=show_kde,
            alpha=0.7,
            color="lightblue",
            edgecolor="darkblue",
            label="Distribution",
        )

        # KDE overlay
        if show_kde:
            kde = stats.gaussian_kde(efficiency)
            x_range = np.linspace(efficiency.min(), efficiency.max(), 200)
            kde_values = kde(x_range)
            ax.plot(x_range, kde_values, "r-", linewidth=2, label="KDE")

        # Statistics lines
        if show_stats:
            ax.axvline(
                mean_eff, color="green", linestyle="--", linewidth=2, label=f"Mean: {mean_eff:.4f}"
            )
            ax.axvline(
                median_eff,
                color="orange",
                linestyle=":",
                linewidth=2,
                label=f"Median: {median_eff:.4f}",
            )

        ax.set_xlabel("Technical Efficiency", fontsize=12)
        ax.set_ylabel("Density" if show_kde else "Count", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        if show_stats:
            stats_text = (
                f"Statistics:\n"
                f"Mean:   {mean_eff:.4f}\n"
                f"Median: {median_eff:.4f}\n"
                f"Std:    {std_eff:.4f}\n"
                f"Min:    {min_eff:.4f}\n"
                f"Max:    {max_eff:.4f}"
            )
            ax.text(
                0.98,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black"),
            )

        plt.tight_layout()
        return fig

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'plotly' or 'matplotlib'.")


def plot_efficiency_ranking(
    efficiency_df: pd.DataFrame,
    backend: str = "plotly",
    top_n: int = 10,
    bottom_n: int = 10,
    entity_col: str = "entity",
    title: Optional[str] = None,
    colorscale: str = "RdYlGn",
    **kwargs,
) -> Any:
    """Plot efficiency rankings showing top and bottom performers.

    Parameters:
        efficiency_df: DataFrame with 'efficiency' column and entity identifier
        backend: 'plotly' for interactive or 'matplotlib' for static
        top_n: Number of top performers to show
        bottom_n: Number of bottom performers to show
        entity_col: Name of column containing entity identifiers
        title: Custom plot title
        colorscale: Color gradient ('RdYlGn' = red-yellow-green)
        **kwargs: Additional arguments passed to plotting backend

    Returns:
        Plotly Figure or Matplotlib Figure object

    Example:
        >>> result = sf.fit()
        >>> eff_df = result.efficiency(estimator='bc')
        >>> fig = plot_efficiency_ranking(eff_df, top_n=15, bottom_n=15)
        >>> fig.show()
    """
    # Sort by efficiency
    df_sorted = efficiency_df.sort_values("efficiency", ascending=False).reset_index(drop=True)

    # Get top and bottom performers
    top_performers = df_sorted.head(top_n)
    bottom_performers = df_sorted.tail(bottom_n).iloc[::-1]  # Reverse for better visualization

    # Combine
    combined = pd.concat([top_performers, bottom_performers])

    # Create labels
    if entity_col in combined.columns:
        labels = combined[entity_col].astype(str).values
    else:
        labels = combined.index.astype(str).values

    efficiency_values = combined["efficiency"].values

    # Determine colors based on efficiency (gradient)
    # Normalize efficiency to [0, 1] for color mapping
    eff_min = efficiency_values.min()
    eff_max = efficiency_values.max()
    eff_normalized = (
        (efficiency_values - eff_min) / (eff_max - eff_min)
        if eff_max > eff_min
        else np.ones_like(efficiency_values)
    )

    if title is None:
        title = f"Efficiency Rankings (Top {top_n} & Bottom {bottom_n})"

    if backend == "plotly":
        import plotly.express as px
        import plotly.graph_objects as go

        # Map normalized efficiency to colors
        colormap = px.colors.sample_colorscale(colorscale, eff_normalized)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=labels,
                x=efficiency_values,
                orientation="h",
                marker=dict(
                    color=eff_normalized,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(title="Efficiency"),
                    line=dict(color="black", width=1),
                ),
                text=[f"{e:.4f}" for e in efficiency_values],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Efficiency: %{x:.4f}<extra></extra>",
            )
        )

        # Add separator line between top and bottom
        fig.add_hline(
            y=top_n - 0.5,
            line_dash="dash",
            line_color="gray",
            line_width=2,
            annotation_text="Top ↑ | Bottom ↓",
            annotation_position="right",
        )

        fig.update_layout(
            title=title,
            xaxis_title="Technical Efficiency",
            yaxis_title="Entity",
            showlegend=False,
            template="plotly_white",
            height=max(400, 30 * (top_n + bottom_n)),
            **kwargs,
        )

        return fig

    elif backend == "matplotlib":
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        fig, ax = plt.subplots(
            figsize=kwargs.get("figsize", (10, max(6, 0.3 * (top_n + bottom_n))))
        )

        # Color mapping
        norm = Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap(colorscale)
        colors = [cmap(norm(e)) for e in eff_normalized]

        # Horizontal bar chart
        y_positions = np.arange(len(labels))
        bars = ax.barh(y_positions, efficiency_values, color=colors, edgecolor="black", linewidth=1)

        # Add value labels
        for i, (pos, val) in enumerate(zip(y_positions, efficiency_values)):
            ax.text(val + 0.01, pos, f"{val:.4f}", va="center", fontsize=9)

        # Add separator line
        ax.axhline(y=top_n - 0.5, color="gray", linestyle="--", linewidth=2, label="Separator")

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Technical Efficiency", fontsize=12)
        ax.set_ylabel("Entity", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)

        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Efficiency (normalized)", rotation=270, labelpad=20)

        plt.tight_layout()
        return fig

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'plotly' or 'matplotlib'.")


def plot_efficiency_boxplot(
    efficiency_df: pd.DataFrame,
    group_var: str,
    backend: str = "plotly",
    test: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs,
) -> Any:
    """Plot box plots of efficiency by groups.

    Parameters:
        efficiency_df: DataFrame with 'efficiency' column and grouping variable
        group_var: Name of column to group by
        backend: 'plotly' for interactive or 'matplotlib' for static
        test: Statistical test to perform ('anova', 'kruskal', or None)
        title: Custom plot title
        **kwargs: Additional arguments passed to plotting backend

    Returns:
        Plotly Figure or Matplotlib Figure object

    Example:
        >>> result = sf.fit()
        >>> eff_df = result.efficiency(estimator='bc')
        >>> # Assuming 'region' column exists in original data
        >>> eff_df = eff_df.join(data['region'])
        >>> fig = plot_efficiency_boxplot(eff_df, group_var='region', test='kruskal')
        >>> fig.show()
    """
    if group_var not in efficiency_df.columns:
        raise ValueError(f"Group variable '{group_var}' not found in efficiency_df")

    # Perform statistical test if requested
    test_result = None
    if test is not None:
        groups = efficiency_df.groupby(group_var)["efficiency"].apply(list)

        if test == "anova":
            # One-way ANOVA
            f_stat, p_value = stats.f_oneway(*groups.values)
            test_result = {
                "test": "ANOVA",
                "statistic": f_stat,
                "pvalue": p_value,
                "conclusion": (
                    "Significant differences" if p_value < 0.05 else "No significant differences"
                ),
            }
        elif test == "kruskal":
            # Kruskal-Wallis H test (non-parametric)
            h_stat, p_value = stats.kruskal(*groups.values)
            test_result = {
                "test": "Kruskal-Wallis",
                "statistic": h_stat,
                "pvalue": p_value,
                "conclusion": (
                    "Significant differences" if p_value < 0.05 else "No significant differences"
                ),
            }
        else:
            raise ValueError(f"Unknown test: {test}. Use 'anova' or 'kruskal'.")

    if title is None:
        title = f"Efficiency Distribution by {group_var}"

    if backend == "plotly":
        import plotly.express as px

        fig = px.box(
            efficiency_df,
            x=group_var,
            y="efficiency",
            title=title,
            points="outliers",  # Show outliers
            color=group_var,
            **kwargs,
        )

        fig.update_layout(
            xaxis_title=group_var.replace("_", " ").title(),
            yaxis_title="Technical Efficiency",
            showlegend=False,
            template="plotly_white",
        )

        # Add test results annotation
        if test_result is not None:
            test_text = (
                f"<b>{test_result['test']} Test</b><br>"
                f"Statistic: {test_result['statistic']:.4f}<br>"
                f"P-value: {test_result['pvalue']:.4f}<br>"
                f"{test_result['conclusion']}"
            )

            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.02,
                xanchor="right",
                yanchor="bottom",
                text=test_text,
                showarrow=False,
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
                opacity=0.8,
            )

        return fig

    elif backend == "matplotlib":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))

        # Prepare data for box plot
        groups = efficiency_df.groupby(group_var)["efficiency"].apply(list)
        labels = list(groups.index)
        data = list(groups.values)

        # Box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True, showmeans=True)

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_xlabel(group_var.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Technical Efficiency", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        # Add test results
        if test_result is not None:
            test_text = (
                f"{test_result['test']} Test\n"
                f"Statistic: {test_result['statistic']:.4f}\n"
                f"P-value: {test_result['pvalue']:.4f}\n"
                f"{test_result['conclusion']}"
            )
            ax.text(
                0.98,
                0.02,
                test_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black"),
            )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'plotly' or 'matplotlib'.")
