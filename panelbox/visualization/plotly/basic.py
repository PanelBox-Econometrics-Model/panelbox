"""
Basic chart types (bar, line, scatter).

This module provides fundamental chart types that serve as examples
and building blocks for more complex visualizations.
"""

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

from ..base import PlotlyChartBase
from ..registry import register_chart


@register_chart("bar_chart")
class BarChart(PlotlyChartBase):
    """
    Interactive bar chart.

    Creates a customizable bar chart with support for grouped and stacked layouts.

    Data Format
    -----------
    data : dict
        Must contain:
        - 'x': list - Category labels
        - 'y': list or dict - Values (single list or dict of {name: values})

        Optional:
        - 'orientation': str - 'v' (vertical, default) or 'h' (horizontal)
        - 'barmode': str - 'group' (default), 'stack', or 'overlay'
        - 'text_position': str - 'auto', 'inside', 'outside', or 'none'
        - 'show_values': bool - Whether to show value labels (default True)

    Examples
    --------
    Simple bar chart:

    >>> chart = BarChart()
    >>> chart.create(data={
    ...     'x': ['Category A', 'Category B', 'Category C'],
    ...     'y': [10, 25, 15]
    ... })
    >>> html = chart.to_html()

    Grouped bar chart:

    >>> chart = BarChart()
    >>> chart.create(data={
    ...     'x': ['Q1', 'Q2', 'Q3', 'Q4'],
    ...     'y': {
    ...         '2023': [100, 120, 115, 140],
    ...         '2024': [110, 130, 125, 150]
    ...     },
    ...     'barmode': 'group'
    ... })

    Stacked bar chart:

    >>> chart = BarChart(config={'title': 'Sales by Region'})
    >>> chart.create(data={
    ...     'x': ['North', 'South', 'East', 'West'],
    ...     'y': {
    ...         'Product A': [30, 40, 35, 45],
    ...         'Product B': [20, 25, 30, 20],
    ...         'Product C': [15, 20, 18, 25]
    ...     },
    ...     'barmode': 'stack'
    ... })

    Horizontal bar chart:

    >>> chart = BarChart()
    >>> chart.create(data={
    ...     'x': [15, 25, 20, 30],
    ...     'y': ['Item 1', 'Item 2', 'Item 3', 'Item 4'],
    ...     'orientation': 'h'
    ... })
    """

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Validate bar chart data."""
        super()._validate_data(data)

        if "x" not in data:
            raise ValueError("Bar chart data must contain 'x' (categories)")
        if "y" not in data:
            raise ValueError("Bar chart data must contain 'y' (values)")

        # Validate x is a list
        if not isinstance(data["x"], list):
            raise ValueError("'x' must be a list")

        # Validate y is a list or dict
        if not isinstance(data["y"], (list, dict)):
            raise ValueError("'y' must be a list or dict")

        # If y is a list, check it matches x length
        if isinstance(data["y"], list) and len(data["y"]) != len(data["x"]):
            raise ValueError(
                f"Length of 'y' ({len(data['y'])}) must match length of 'x' ({len(data['x'])})"
            )

    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data to standardize format."""
        processed = data.copy()

        # Normalize y to always be a dict
        if isinstance(data["y"], list):
            processed["y"] = {"values": data["y"]}

        # Set defaults
        processed.setdefault("orientation", "v")
        processed.setdefault("barmode", "group")
        processed.setdefault("text_position", "auto")
        processed.setdefault("show_values", True)

        return processed

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create Plotly bar chart figure."""
        fig = go.Figure()

        # Extract data
        x_values = data["x"]
        y_data = data["y"]
        orientation = data["orientation"]
        barmode = data["barmode"]
        text_position = data["text_position"]
        show_values = data["show_values"]

        # Create traces (one per series)
        for i, (series_name, y_values) in enumerate(y_data.items()):
            # Determine bar orientation
            if orientation == "h":
                trace_data = {"y": x_values, "x": y_values, "orientation": "h"}
            else:
                trace_data = {"x": x_values, "y": y_values}

            # Add text labels if requested
            if show_values:
                text = [f"{val:.1f}" if isinstance(val, float) else str(val) for val in y_values]
                trace_data["text"] = text
                trace_data["textposition"] = text_position

            # Create bar trace
            trace = go.Bar(name=series_name if len(y_data) > 1 else None, **trace_data)

            fig.add_trace(trace)

        # Update layout for bar mode
        fig.update_layout(barmode=barmode)

        # Update axis labels from config
        if orientation == "h":
            xaxis_title = self.config.get("xaxis_title", "Value")
            yaxis_title = self.config.get("yaxis_title", "Category")
        else:
            xaxis_title = self.config.get("xaxis_title", "Category")
            yaxis_title = self.config.get("yaxis_title", "Value")

        fig.update_xaxes(title_text=xaxis_title)
        fig.update_yaxes(title_text=yaxis_title)

        return fig


@register_chart("line_chart")
class LineChart(PlotlyChartBase):
    """
    Interactive line chart.

    Creates a line chart for time series or continuous data visualization.

    Data Format
    -----------
    data : dict
        Must contain:
        - 'x': list - X-axis values
        - 'y': list or dict - Y-axis values (single list or dict of {name: values})

        Optional:
        - 'mode': str - 'lines' (default), 'markers', or 'lines+markers'
        - 'line_shape': str - 'linear' (default), 'spline', or 'hv'

    Examples
    --------
    >>> chart = LineChart()
    >>> chart.create(data={
    ...     'x': [1, 2, 3, 4, 5],
    ...     'y': [10, 12, 15, 14, 18]
    ... })
    """

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Validate line chart data."""
        super()._validate_data(data)

        if "x" not in data or "y" not in data:
            raise ValueError("Line chart requires both 'x' and 'y'")

    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data."""
        processed = data.copy()

        # Normalize y to dict
        if isinstance(data["y"], list):
            processed["y"] = {"line": data["y"]}

        processed.setdefault("mode", "lines+markers")
        processed.setdefault("line_shape", "linear")

        return processed

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create Plotly line chart."""
        fig = go.Figure()

        x_values = data["x"]
        y_data = data["y"]
        mode = data["mode"]
        line_shape = data["line_shape"]

        for series_name, y_values in y_data.items():
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode=mode,
                    name=series_name if len(y_data) > 1 else None,
                    line={"shape": line_shape},
                )
            )

        return fig
