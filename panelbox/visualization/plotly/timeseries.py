"""
Time series visualization charts for PanelBox.

This module provides charts for visualizing panel data over time, including
multi-line plots, trend analysis, and faceted time series.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..base import PlotlyChartBase
from ..registry import register_chart


@register_chart("timeseries_panel")
class PanelTimeSeriesChart(PlotlyChartBase):
    """
    Panel time series plot with multiple entities.

    Multi-line plot showing time series for each entity in the panel,
    with interactive legend filtering.

    Examples
    --------
    >>> from panelbox.visualization import PanelTimeSeriesChart
    >>>
    >>> data = {
    >>>     'time': pd.date_range('2020-01-01', periods=100),
    >>>     'values': np.random.randn(500),  # 5 entities * 100 time periods
    >>>     'entity_id': np.repeat(['A', 'B', 'C', 'D', 'E'], 100)
    >>> }
    >>>
    >>> chart = PanelTimeSeriesChart()
    >>> chart.create(data)
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create panel time series chart."""
        time = data.get("time")
        values = np.asarray(data.get("values"))
        entity_id = data.get("entity_id")
        variable_name = data.get("variable_name", "Value")
        max_entities = data.get("max_entities", 20)
        show_mean = data.get("show_mean", False)

        if entity_id is None:
            raise ValueError("entity_id is required for panel time series")

        fig = go.Figure()

        # Get unique entities
        unique_entities = pd.unique(entity_id)

        # Limit number of entities for performance
        if len(unique_entities) > max_entities:
            unique_entities = unique_entities[:max_entities]
            import warnings

            warnings.warn(f"Too many entities. Limiting to first {max_entities} for performance.")

        # Create DataFrame for easier manipulation
        df = pd.DataFrame({"time": time, "value": values, "entity": entity_id})

        # Plot each entity
        colors = self.theme.color_scheme if self.theme and self.theme.color_scheme else None

        for i, entity in enumerate(unique_entities):
            entity_data = df[df["entity"] == entity].sort_values("time")

            fig.add_trace(
                go.Scatter(
                    x=entity_data["time"],
                    y=entity_data["value"],
                    mode="lines",
                    name=str(entity),
                    line=dict(color=colors[i % len(colors)] if colors else None, width=2),
                    hovertemplate=(
                        f"<b>{entity}</b><br>"
                        + "Time: %{x}<br>"
                        + f"{variable_name}: %{{y:.2f}}<br>"
                        + "<extra></extra>"
                    ),
                )
            )

        # Add mean line if requested
        if show_mean:
            mean_by_time = df.groupby("time")["value"].mean().reset_index()

            fig.add_trace(
                go.Scatter(
                    x=mean_by_time["time"],
                    y=mean_by_time["value"],
                    mode="lines",
                    name="Mean",
                    line=dict(color="black", width=3, dash="dash"),
                    hovertemplate=(
                        "<b>Mean</b><br>"
                        + "Time: %{x}<br>"
                        + f"{variable_name}: %{{y:.2f}}<br>"
                        + "<extra></extra>"
                    ),
                )
            )

        # Update layout
        fig.update_layout(
            title=data.get("title", f"Panel Time Series: {variable_name}"),
            xaxis_title=data.get("xaxis_title", "Time"),
            yaxis_title=data.get("yaxis_title", variable_name),
            hovermode="closest",
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        )

        return fig


@register_chart("timeseries_trend")
class TrendLineChart(PlotlyChartBase):
    """
    Time series with trend decomposition.

    Shows original time series with trend line, moving averages, and optional
    seasonal decomposition.

    Examples
    --------
    >>> from panelbox.visualization import TrendLineChart
    >>>
    >>> data = {
    >>>     'time': pd.date_range('2020-01-01', periods=100),
    >>>     'values': np.random.randn(100).cumsum(),
    >>>     'show_moving_average': True,
    >>>     'window': 7
    >>> }
    >>>
    >>> chart = TrendLineChart()
    >>> chart.create(data)
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create trend line chart."""
        time = data.get("time")
        values = np.asarray(data.get("values"))
        show_moving_average = data.get("show_moving_average", True)
        window = data.get("window", 7)
        show_trend = data.get("show_trend", True)

        fig = go.Figure()

        # Original time series
        fig.add_trace(
            go.Scatter(
                x=time,
                y=values,
                mode="lines",
                name="Original",
                line=dict(color="lightgray", width=1),
                opacity=0.7,
            )
        )

        # Moving average
        if show_moving_average and len(values) > window:
            ma = pd.Series(values).rolling(window=window, center=True).mean()

            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=ma,
                    mode="lines",
                    name=f"MA({window})",
                    line=dict(
                        color=(
                            self.theme.color_scheme[0]
                            if self.theme and self.theme.color_scheme
                            else "blue"
                        ),
                        width=2,
                    ),
                )
            )

        # Trend line (linear regression)
        if show_trend:
            # Create numeric time index
            x_numeric = np.arange(len(values))

            # Remove NaN values for regression
            mask = ~np.isnan(values)
            if mask.sum() > 1:
                # Linear regression
                coeffs = np.polyfit(x_numeric[mask], values[mask], 1)
                trend = np.polyval(coeffs, x_numeric)

                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=trend,
                        mode="lines",
                        name="Trend",
                        line=dict(color="red", width=2, dash="dash"),
                    )
                )

        # Update layout
        fig.update_layout(
            title=data.get("title", "Time Series with Trend"),
            xaxis_title=data.get("xaxis_title", "Time"),
            yaxis_title=data.get("yaxis_title", "Value"),
            hovermode="x unified",
        )

        return fig


@register_chart("timeseries_faceted")
class FacetedTimeSeriesChart(PlotlyChartBase):
    """
    Faceted time series with small multiples.

    Creates a grid of time series plots, one for each entity or group,
    with shared axes for easy comparison.

    Examples
    --------
    >>> from panelbox.visualization import FacetedTimeSeriesChart
    >>>
    >>> data = {
    >>>     'time': np.tile(pd.date_range('2020-01-01', periods=50), 6),
    >>>     'values': np.random.randn(300),
    >>>     'entity_id': np.repeat(['A', 'B', 'C', 'D', 'E', 'F'], 50)
    >>> }
    >>>
    >>> chart = FacetedTimeSeriesChart()
    >>> chart.create(data)
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create faceted time series chart."""
        time = data.get("time")
        values = np.asarray(data.get("values"))
        entity_id = data.get("entity_id")
        variable_name = data.get("variable_name", "Value")
        ncols = data.get("ncols", 3)
        shared_yaxis = data.get("shared_yaxis", True)

        if entity_id is None:
            raise ValueError("entity_id is required for faceted time series")

        # Create DataFrame
        df = pd.DataFrame({"time": time, "value": values, "entity": entity_id})

        # Get unique entities
        unique_entities = pd.unique(entity_id)
        n_entities = len(unique_entities)

        # Calculate grid dimensions
        nrows = int(np.ceil(n_entities / ncols))

        # Create subplots
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[str(e) for e in unique_entities],
            shared_xaxes=True,
            shared_yaxes=shared_yaxis,
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
        )

        # Plot each entity in its own subplot
        for i, entity in enumerate(unique_entities):
            row = (i // ncols) + 1
            col = (i % ncols) + 1

            entity_data = df[df["entity"] == entity].sort_values("time")

            fig.add_trace(
                go.Scatter(
                    x=entity_data["time"],
                    y=entity_data["value"],
                    mode="lines",
                    name=str(entity),
                    line=dict(
                        color=(
                            self.theme.color_scheme[0]
                            if self.theme and self.theme.color_scheme
                            else "blue"
                        ),
                        width=2,
                    ),
                    showlegend=False,
                    hovertemplate=(
                        "Time: %{x}<br>" + f"{variable_name}: %{{y:.2f}}<br>" + "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        # Update axes labels
        for i in range(1, nrows + 1):
            for j in range(1, ncols + 1):
                # X-axis label only on bottom row
                if i == nrows:
                    fig.update_xaxes(title_text=data.get("xaxis_title", "Time"), row=i, col=j)
                # Y-axis label only on left column
                if j == 1:
                    fig.update_yaxes(
                        title_text=data.get("yaxis_title", variable_name), row=i, col=j
                    )

        # Update layout
        fig.update_layout(
            title=data.get("title", f"Faceted Time Series: {variable_name}"),
            hovermode="closest",
            height=data.get("height", 250 * nrows),
            showlegend=False,
        )

        return fig
