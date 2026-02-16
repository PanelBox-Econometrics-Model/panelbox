"""
Temporal evolution plots for panel SFA results.

This module provides visualization functions for:
- Time series of mean efficiency
- Spaghetti plots (individual entity trajectories)
- Heatmaps (entity × time)
- Fan charts (percentile evolution)
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def plot_efficiency_timeseries(
    efficiency_df: pd.DataFrame,
    backend: str = "plotly",
    show_ci: bool = True,
    show_range: bool = False,
    show_median: bool = False,
    events: Optional[Dict[Union[int, str], str]] = None,
    title: Optional[str] = None,
    **kwargs,
) -> Any:
    """Plot time series of mean efficiency with confidence intervals.

    Parameters:
        efficiency_df: DataFrame with 'efficiency' and 'time' columns
        backend: 'plotly' for interactive or 'matplotlib' for static
        show_ci: Show confidence interval band (±2 std errors)
        show_range: Show min-max range band
        show_median: Show median line in addition to mean
        events: Dictionary mapping time points to event labels
                Example: {2008: 'Financial Crisis', 2015: 'Regulation'}
        title: Custom plot title
        **kwargs: Additional arguments passed to plotting backend

    Returns:
        Plotly Figure or Matplotlib Figure object

    Example:
        >>> result = panel_sf.fit()
        >>> eff_df = result.efficiency(estimator='bc', by_period=True)
        >>> fig = plot_efficiency_timeseries(
        ...     eff_df,
        ...     show_ci=True,
        ...     events={2008: 'Crisis', 2015: 'Reform'}
        ... )
        >>> fig.show()

    Notes:
        - Requires panel data with time-varying efficiency
        - For Pitt-Lee model, efficiency is constant over time
    """
    if "time" not in efficiency_df.columns:
        raise ValueError("efficiency_df must have 'time' column for time series plot")

    # Aggregate by time
    grouped = efficiency_df.groupby("time")["efficiency"]
    mean_eff = grouped.mean()
    median_eff = grouped.median()
    std_eff = grouped.std()
    n_entities = grouped.count()
    se_eff = std_eff / np.sqrt(n_entities)

    min_eff = grouped.min()
    max_eff = grouped.max()

    time_points = mean_eff.index

    if title is None:
        title = "Efficiency Evolution Over Time"

    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure()

        # Range band (min-max)
        if show_range:
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=max_eff,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=min_eff,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(200, 200, 200, 0.3)",
                    name="Range (Min-Max)",
                    hoverinfo="skip",
                )
            )

        # Confidence interval band
        if show_ci:
            ci_upper = mean_eff + 1.96 * se_eff
            ci_lower = mean_eff - 1.96 * se_eff

            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=ci_upper,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=ci_lower,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(100, 150, 250, 0.3)",
                    name="95% CI",
                    hoverinfo="skip",
                )
            )

        # Mean line
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=mean_eff,
                mode="lines+markers",
                line=dict(color="blue", width=3),
                marker=dict(size=8),
                name="Mean Efficiency",
                hovertemplate="<b>Time: %{x}</b><br>Mean: %{y:.4f}<extra></extra>",
            )
        )

        # Median line
        if show_median:
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=median_eff,
                    mode="lines+markers",
                    line=dict(color="orange", width=2, dash="dot"),
                    marker=dict(size=6),
                    name="Median Efficiency",
                    hovertemplate="<b>Time: %{x}</b><br>Median: %{y:.4f}<extra></extra>",
                )
            )

        # Event annotations
        if events is not None:
            for event_time, event_label in events.items():
                if event_time in time_points:
                    fig.add_vline(
                        x=event_time,
                        line_dash="dash",
                        line_color="red",
                        line_width=2,
                        annotation_text=event_label,
                        annotation_position="top",
                    )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Technical Efficiency",
            hovermode="x unified",
            template="plotly_white",
            **kwargs,
        )

        return fig

    elif backend == "matplotlib":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 6)))

        # Range band
        if show_range:
            ax.fill_between(
                time_points, min_eff, max_eff, alpha=0.2, color="gray", label="Range (Min-Max)"
            )

        # CI band
        if show_ci:
            ci_upper = mean_eff + 1.96 * se_eff
            ci_lower = mean_eff - 1.96 * se_eff
            ax.fill_between(
                time_points, ci_lower, ci_upper, alpha=0.3, color="blue", label="95% CI"
            )

        # Mean line
        ax.plot(time_points, mean_eff, "b-o", linewidth=3, markersize=8, label="Mean Efficiency")

        # Median line
        if show_median:
            ax.plot(
                time_points,
                median_eff,
                "orange",
                linestyle=":",
                linewidth=2,
                marker="s",
                markersize=6,
                label="Median Efficiency",
            )

        # Event annotations
        if events is not None:
            for event_time, event_label in events.items():
                if event_time in time_points.values:
                    ax.axvline(event_time, color="red", linestyle="--", linewidth=2)
                    ax.text(
                        event_time,
                        ax.get_ylim()[1],
                        event_label,
                        rotation=90,
                        va="top",
                        ha="right",
                        fontsize=10,
                    )

        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Technical Efficiency", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'plotly' or 'matplotlib'.")


def plot_efficiency_spaghetti(
    efficiency_df: pd.DataFrame,
    backend: str = "plotly",
    highlight: Optional[List[Union[str, int]]] = None,
    alpha: float = 0.2,
    show_mean: bool = True,
    entity_col: str = "entity",
    title: Optional[str] = None,
    **kwargs,
) -> Any:
    """Plot spaghetti plot with individual entity trajectories.

    Parameters:
        efficiency_df: DataFrame with 'efficiency', 'time', and entity identifier
        backend: 'plotly' for interactive or 'matplotlib' for static
        highlight: List of entity IDs to highlight with thicker lines
        alpha: Transparency for non-highlighted lines (0-1)
        show_mean: Show mean trajectory with thick line
        entity_col: Name of column containing entity identifiers
        title: Custom plot title
        **kwargs: Additional arguments passed to plotting backend

    Returns:
        Plotly Figure or Matplotlib Figure object

    Example:
        >>> result = panel_sf.fit()
        >>> eff_df = result.efficiency(estimator='bc', by_period=True)
        >>> fig = plot_efficiency_spaghetti(
        ...     eff_df,
        ...     highlight=['firm_A', 'firm_B'],
        ...     alpha=0.3,
        ...     show_mean=True
        ... )
        >>> fig.show()
    """
    if "time" not in efficiency_df.columns:
        raise ValueError("efficiency_df must have 'time' column")
    if entity_col not in efficiency_df.columns:
        raise ValueError(f"efficiency_df must have '{entity_col}' column")

    entities = efficiency_df[entity_col].unique()

    if title is None:
        title = "Efficiency Evolution by Entity"

    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure()

        # Plot each entity trajectory
        for entity in entities:
            entity_data = efficiency_df[efficiency_df[entity_col] == entity].sort_values("time")

            if highlight is not None and entity in highlight:
                # Highlighted entity
                fig.add_trace(
                    go.Scatter(
                        x=entity_data["time"],
                        y=entity_data["efficiency"],
                        mode="lines",
                        line=dict(width=3),
                        name=str(entity),
                        hovertemplate=f"<b>{entity}</b><br>Time: %{{x}}<br>Efficiency: %{{y:.4f}}<extra></extra>",
                    )
                )
            else:
                # Non-highlighted entity
                fig.add_trace(
                    go.Scatter(
                        x=entity_data["time"],
                        y=entity_data["efficiency"],
                        mode="lines",
                        line=dict(width=1),
                        opacity=alpha,
                        name=str(entity),
                        showlegend=False,
                        hovertemplate=f"<b>{entity}</b><br>Time: %{{x}}<br>Efficiency: %{{y:.4f}}<extra></extra>",
                    )
                )

        # Add mean trajectory
        if show_mean:
            mean_by_time = efficiency_df.groupby("time")["efficiency"].mean()
            fig.add_trace(
                go.Scatter(
                    x=mean_by_time.index,
                    y=mean_by_time.values,
                    mode="lines",
                    line=dict(color="black", width=4, dash="dash"),
                    name="Mean",
                    hovertemplate="<b>Mean</b><br>Time: %{x}<br>Efficiency: %{y:.4f}<extra></extra>",
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Technical Efficiency",
            hovermode="closest",
            template="plotly_white",
            **kwargs,
        )

        return fig

    elif backend == "matplotlib":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 6)))

        # Plot each entity
        for entity in entities:
            entity_data = efficiency_df[efficiency_df[entity_col] == entity].sort_values("time")

            if highlight is not None and entity in highlight:
                ax.plot(
                    entity_data["time"], entity_data["efficiency"], linewidth=3, label=str(entity)
                )
            else:
                ax.plot(
                    entity_data["time"],
                    entity_data["efficiency"],
                    linewidth=1,
                    alpha=alpha,
                    color="gray",
                )

        # Mean trajectory
        if show_mean:
            mean_by_time = efficiency_df.groupby("time")["efficiency"].mean()
            ax.plot(mean_by_time.index, mean_by_time.values, "k--", linewidth=4, label="Mean")

        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Technical Efficiency", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        if highlight is not None or show_mean:
            ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'plotly' or 'matplotlib'.")


def plot_efficiency_heatmap(
    efficiency_df: pd.DataFrame,
    backend: str = "plotly",
    order_by: str = "efficiency",
    entity_col: str = "entity",
    colorscale: str = "RdYlGn",
    title: Optional[str] = None,
    **kwargs,
) -> Any:
    """Plot heatmap of efficiency (entity × time).

    Parameters:
        efficiency_df: DataFrame with 'efficiency', 'time', and entity identifier
        backend: 'plotly' for interactive or 'matplotlib' for static
        order_by: How to order entities ('efficiency', 'alphabetical', or None)
        entity_col: Name of column containing entity identifiers
        colorscale: Color gradient ('RdYlGn' = red-yellow-green)
        title: Custom plot title
        **kwargs: Additional arguments passed to plotting backend

    Returns:
        Plotly Figure or Matplotlib Figure object

    Example:
        >>> result = panel_sf.fit()
        >>> eff_df = result.efficiency(estimator='bc', by_period=True)
        >>> fig = plot_efficiency_heatmap(eff_df, order_by='efficiency')
        >>> fig.show()
    """
    if "time" not in efficiency_df.columns:
        raise ValueError("efficiency_df must have 'time' column")
    if entity_col not in efficiency_df.columns:
        raise ValueError(f"efficiency_df must have '{entity_col}' column")

    # Pivot to create entity × time matrix
    heatmap_data = efficiency_df.pivot(index=entity_col, columns="time", values="efficiency")

    # Order entities
    if order_by == "efficiency":
        # Order by mean efficiency (descending)
        entity_order = heatmap_data.mean(axis=1).sort_values(ascending=False).index
        heatmap_data = heatmap_data.loc[entity_order]
    elif order_by == "alphabetical":
        heatmap_data = heatmap_data.sort_index()
    # else: keep original order

    if title is None:
        title = "Efficiency Heatmap (Entity × Time)"

    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale=colorscale,
                hovertemplate="<b>Entity: %{y}</b><br>Time: %{x}<br>Efficiency: %{z:.4f}<extra></extra>",
                colorbar=dict(title="Efficiency"),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Entity",
            template="plotly_white",
            height=max(400, 20 * len(heatmap_data)),
            **kwargs,
        )

        return fig

    elif backend == "matplotlib":
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, max(8, 0.3 * len(heatmap_data)))))

        # Create heatmap
        sns.heatmap(
            heatmap_data,
            cmap=colorscale,
            cbar_kws={"label": "Efficiency"},
            ax=ax,
            linewidths=0.5,
            linecolor="white",
        )

        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Entity", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()
        return fig

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'plotly' or 'matplotlib'.")


def plot_efficiency_fanchart(
    efficiency_df: pd.DataFrame,
    backend: str = "plotly",
    percentiles: Optional[List[int]] = None,
    title: Optional[str] = None,
    **kwargs,
) -> Any:
    """Plot fan chart showing percentile evolution over time.

    Parameters:
        efficiency_df: DataFrame with 'efficiency' and 'time' columns
        backend: 'plotly' for interactive or 'matplotlib' for static
        percentiles: List of percentiles to plot (default: [10, 25, 50, 75, 90])
        title: Custom plot title
        **kwargs: Additional arguments passed to plotting backend

    Returns:
        Plotly Figure or Matplotlib Figure object

    Example:
        >>> result = panel_sf.fit()
        >>> eff_df = result.efficiency(estimator='bc', by_period=True)
        >>> fig = plot_efficiency_fanchart(
        ...     eff_df,
        ...     percentiles=[10, 25, 50, 75, 90]
        ... )
        >>> fig.show()
    """
    if "time" not in efficiency_df.columns:
        raise ValueError("efficiency_df must have 'time' column")

    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    # Compute percentiles by time
    grouped = efficiency_df.groupby("time")["efficiency"]
    percentile_data = {}
    for p in percentiles:
        percentile_data[f"p{p}"] = grouped.quantile(p / 100)

    time_points = percentile_data[f"p{percentiles[0]}"].index

    if title is None:
        title = "Efficiency Fan Chart (Percentile Evolution)"

    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure()

        # Plot percentiles with filled areas
        colors = [
            "rgba(100,100,250,0.1)",
            "rgba(100,150,250,0.2)",
            "rgba(0,0,0,1)",
            "rgba(250,150,100,0.2)",
            "rgba(250,100,100,0.1)",
        ]

        for i, p in enumerate(sorted(percentiles)):
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=percentile_data[f"p{p}"],
                    mode="lines",
                    line=dict(
                        width=2 if p == 50 else 1, color=colors[i] if i < len(colors) else "gray"
                    ),
                    name=f"P{p}",
                    fill="tonexty" if i > 0 else None,
                    fillcolor=colors[i] if i < len(colors) else "rgba(200,200,200,0.2)",
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Technical Efficiency",
            hovermode="x unified",
            template="plotly_white",
            **kwargs,
        )

        return fig

    elif backend == "matplotlib":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 6)))

        # Plot median
        median_idx = percentiles.index(50) if 50 in percentiles else None
        if median_idx is not None:
            ax.plot(time_points, percentile_data["p50"], "k-", linewidth=3, label="P50 (Median)")

        # Fill between percentiles
        sorted_p = sorted(percentiles)
        n = len(sorted_p)
        for i in range(n // 2):
            lower_p = sorted_p[i]
            upper_p = sorted_p[n - 1 - i]
            ax.fill_between(
                time_points,
                percentile_data[f"p{lower_p}"],
                percentile_data[f"p{upper_p}"],
                alpha=0.2,
                label=f"P{lower_p}-P{upper_p}",
            )

        # Plot individual percentile lines
        for p in sorted_p:
            if p != 50:  # Median already plotted
                ax.plot(
                    time_points,
                    percentile_data[f"p{p}"],
                    linewidth=1,
                    linestyle="--",
                    label=f"P{p}",
                )

        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Technical Efficiency", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'plotly' or 'matplotlib'.")
