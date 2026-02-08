"""
Distribution visualization charts for PanelBox.

This module provides charts for visualizing data distributions, including
histograms, kernel density estimates, violin plots, and box plots.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from ..base import PlotlyChartBase
from ..registry import register_chart


@register_chart("distribution_histogram")
class HistogramChart(PlotlyChartBase):
    """
    Histogram with optional KDE and normal distribution overlays.

    Examples
    --------
    >>> from panelbox.visualization import HistogramChart
    >>>
    >>> data = {
    >>>     'values': np.random.normal(0, 1, 1000),
    >>>     'show_kde': True,
    >>>     'show_normal': True
    >>> }
    >>>
    >>> chart = HistogramChart()
    >>> chart.create(data)
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create histogram chart."""
        values = np.asarray(data.get("values", []))
        groups = data.get("groups", None)
        show_kde = data.get("show_kde", False)
        show_normal = data.get("show_normal", False)
        bins = data.get("bins", "auto")

        fig = go.Figure()

        if groups is not None:
            # Grouped histogram
            unique_groups = np.unique(groups)
            for group in unique_groups:
                mask = groups == group
                group_values = values[mask]

                fig.add_trace(
                    go.Histogram(
                        x=group_values,
                        name=str(group),
                        opacity=0.7,
                        nbinsx=self._calculate_bins(group_values, bins),
                        histnorm="probability density" if show_kde or show_normal else "",
                    )
                )
        else:
            # Single histogram
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name="Distribution",
                    opacity=0.7,
                    nbinsx=self._calculate_bins(values, bins),
                    histnorm="probability density" if show_kde or show_normal else "",
                    marker_color=(
                        self.theme.color_scheme[0]
                        if self.theme and self.theme.color_scheme
                        else None
                    ),
                )
            )

        # Add KDE overlay
        if show_kde and len(values) > 1:
            try:
                from scipy.stats import gaussian_kde

                kde = gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 200)
                y_kde = kde(x_range)

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_kde,
                        mode="lines",
                        name="KDE",
                        line=dict(color="red", width=2),
                    )
                )
            except Exception:
                pass  # Skip KDE if it fails

        # Add normal distribution overlay
        if show_normal:
            mean = np.mean(values)
            std = np.std(values)
            x_range = np.linspace(values.min(), values.max(), 200)
            y_normal = stats.norm.pdf(x_range, mean, std)

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_normal,
                    mode="lines",
                    name="Normal",
                    line=dict(color="green", width=2, dash="dash"),
                )
            )

        # Update layout
        fig.update_layout(
            title=data.get("title", "Distribution Histogram"),
            xaxis_title=data.get("xaxis_title", "Value"),
            yaxis_title=data.get(
                "yaxis_title", "Frequency" if not (show_kde or show_normal) else "Density"
            ),
            barmode="overlay" if groups is not None else "stack",
            hovermode="closest",
        )

        return fig

    def _calculate_bins(self, values: np.ndarray, bins: Union[str, int]) -> int:
        """Calculate number of bins using Freedman-Diaconis rule or specified number."""
        if isinstance(bins, int):
            return bins
        elif bins == "auto" or bins == "fd":
            # Freedman-Diaconis rule
            iqr = np.percentile(values, 75) - np.percentile(values, 25)
            if iqr > 0:
                bin_width = 2 * iqr / (len(values) ** (1 / 3))
                n_bins = int((values.max() - values.min()) / bin_width)
                return max(10, min(n_bins, 100))  # Clamp between 10 and 100
            else:
                return 30
        else:
            return 30  # Default


@register_chart("distribution_kde")
class KDEChart(PlotlyChartBase):
    """
    Kernel Density Estimate plot.

    Examples
    --------
    >>> from panelbox.visualization import KDEChart
    >>>
    >>> data = {
    >>>     'values': np.random.normal(0, 1, 1000),
    >>>     'show_rug': True
    >>> }
    >>>
    >>> chart = KDEChart()
    >>> chart.create(data)
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create KDE chart."""
        values = np.asarray(data.get("values", []))
        groups = data.get("groups", None)
        show_rug = data.get("show_rug", True)
        fill = data.get("fill", True)

        fig = go.Figure()

        if groups is not None:
            # Multiple KDE curves for groups
            unique_groups = np.unique(groups)
            for i, group in enumerate(unique_groups):
                mask = groups == group
                group_values = values[mask]

                if len(group_values) > 1:
                    try:
                        from scipy.stats import gaussian_kde

                        kde = gaussian_kde(group_values)
                        x_range = np.linspace(group_values.min(), group_values.max(), 200)
                        y_kde = kde(x_range)

                        fig.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=y_kde,
                                mode="lines",
                                name=str(group),
                                fill="tozeroy" if fill else None,
                                opacity=0.6 if fill else 1.0,
                                line=dict(width=2),
                            )
                        )

                        # Add rug plot
                        if show_rug:
                            fig.add_trace(
                                go.Scatter(
                                    x=group_values,
                                    y=np.zeros_like(group_values),
                                    mode="markers",
                                    name=f"{group} (rug)",
                                    marker=dict(symbol="line-ns", size=10, line=dict(width=1)),
                                    showlegend=False,
                                )
                            )
                    except Exception:
                        pass  # Skip if KDE fails
        else:
            # Single KDE curve
            if len(values) > 1:
                try:
                    from scipy.stats import gaussian_kde

                    kde = gaussian_kde(values)
                    x_range = np.linspace(values.min(), values.max(), 200)
                    y_kde = kde(x_range)

                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_kde,
                            mode="lines",
                            name="Density",
                            fill="tozeroy" if fill else None,
                            opacity=0.6 if fill else 1.0,
                            line=dict(width=3),
                            marker_color=(
                                self.theme.color_scheme[0]
                                if self.theme and self.theme.color_scheme
                                else None
                            ),
                        )
                    )

                    # Add rug plot
                    if show_rug:
                        fig.add_trace(
                            go.Scatter(
                                x=values,
                                y=np.zeros_like(values),
                                mode="markers",
                                name="Observations",
                                marker=dict(
                                    symbol="line-ns",
                                    size=10,
                                    color="rgba(0,0,0,0.3)",
                                    line=dict(width=1),
                                ),
                            )
                        )

                    # Add summary statistics as annotations
                    mean_val = np.mean(values)
                    median_val = np.median(values)

                    fig.add_vline(
                        x=mean_val,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Mean: {mean_val:.2f}",
                        annotation_position="top",
                    )

                    fig.add_vline(
                        x=median_val,
                        line_dash="dot",
                        line_color="blue",
                        annotation_text=f"Median: {median_val:.2f}",
                        annotation_position="bottom",
                    )

                except Exception:
                    pass  # Skip if KDE fails

        # Update layout
        fig.update_layout(
            title=data.get("title", "Kernel Density Estimate"),
            xaxis_title=data.get("xaxis_title", "Value"),
            yaxis_title=data.get("yaxis_title", "Density"),
            hovermode="closest",
        )

        return fig


@register_chart("distribution_violin")
class ViolinPlotChart(PlotlyChartBase):
    """
    Violin plot combining box plot and KDE.

    Examples
    --------
    >>> from panelbox.visualization import ViolinPlotChart
    >>>
    >>> data = {
    >>>     'values': np.random.normal(0, 1, 1000),
    >>>     'groups': np.repeat(['A', 'B', 'C'], 333)
    >>> }
    >>>
    >>> chart = ViolinPlotChart()
    >>> chart.create(data)
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create violin plot."""
        values = np.asarray(data.get("values", []))
        groups = data.get("groups", None)
        show_box = data.get("show_box", True)
        show_points = data.get("show_points", False)

        fig = go.Figure()

        if groups is not None:
            unique_groups = np.unique(groups)
            for group in unique_groups:
                mask = groups == group
                group_values = values[mask]

                fig.add_trace(
                    go.Violin(
                        y=group_values,
                        name=str(group),
                        box_visible=show_box,
                        meanline_visible=True,
                        points="all" if show_points else False,
                        jitter=0.3 if show_points else 0,
                        pointpos=-1.5 if show_points else 0,
                    )
                )
        else:
            # Single violin (no grouping, use dummy x)
            fig.add_trace(
                go.Violin(
                    y=values,
                    name="Distribution",
                    box_visible=show_box,
                    meanline_visible=True,
                    points="all" if show_points else False,
                    jitter=0.3 if show_points else 0,
                    marker_color=(
                        self.theme.color_scheme[0]
                        if self.theme and self.theme.color_scheme
                        else None
                    ),
                )
            )

        # Update layout
        fig.update_layout(
            title=data.get("title", "Violin Plot"),
            xaxis_title=data.get("xaxis_title", "Group" if groups is not None else ""),
            yaxis_title=data.get("yaxis_title", "Value"),
            violinmode="group" if groups is not None else None,
            hovermode="closest",
        )

        return fig


@register_chart("distribution_boxplot")
class BoxPlotChart(PlotlyChartBase):
    """
    Box plot showing distribution summary.

    Examples
    --------
    >>> from panelbox.visualization import BoxPlotChart
    >>>
    >>> data = {
    >>>     'values': np.random.normal(0, 1, 1000),
    >>>     'groups': np.repeat(['A', 'B', 'C'], 333),
    >>>     'show_mean': True
    >>> }
    >>>
    >>> chart = BoxPlotChart()
    >>> chart.create(data)
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create box plot."""
        values = np.asarray(data.get("values", []))
        groups = data.get("groups", None)
        show_mean = data.get("show_mean", True)
        show_points = data.get("show_points", False)
        orientation = data.get("orientation", "v")  # 'v' or 'h'

        fig = go.Figure()

        if groups is not None:
            unique_groups = np.unique(groups)
            for group in unique_groups:
                mask = groups == group
                group_values = values[mask]

                if orientation == "v":
                    fig.add_trace(
                        go.Box(
                            y=group_values,
                            name=str(group),
                            boxmean="sd" if show_mean else False,
                            boxpoints="all" if show_points else "outliers",
                            jitter=0.3 if show_points else 0,
                            pointpos=-1.5 if show_points else 0,
                        )
                    )
                else:  # horizontal
                    fig.add_trace(
                        go.Box(
                            x=group_values,
                            name=str(group),
                            boxmean="sd" if show_mean else False,
                            boxpoints="all" if show_points else "outliers",
                            jitter=0.3 if show_points else 0,
                            pointpos=-1.5 if show_points else 0,
                        )
                    )
        else:
            # Single box plot
            if orientation == "v":
                fig.add_trace(
                    go.Box(
                        y=values,
                        name="Distribution",
                        boxmean="sd" if show_mean else False,
                        boxpoints="all" if show_points else "outliers",
                        marker_color=(
                            self.theme.color_scheme[0]
                            if self.theme and self.theme.color_scheme
                            else None
                        ),
                    )
                )
            else:  # horizontal
                fig.add_trace(
                    go.Box(
                        x=values,
                        name="Distribution",
                        boxmean="sd" if show_mean else False,
                        boxpoints="all" if show_points else "outliers",
                        marker_color=(
                            self.theme.color_scheme[0]
                            if self.theme and self.theme.color_scheme
                            else None
                        ),
                    )
                )

        # Update layout
        if orientation == "v":
            fig.update_layout(
                title=data.get("title", "Box Plot"),
                xaxis_title=data.get("xaxis_title", "Group" if groups is not None else ""),
                yaxis_title=data.get("yaxis_title", "Value"),
                boxmode="group" if groups is not None else None,
                hovermode="closest",
            )
        else:  # horizontal
            fig.update_layout(
                title=data.get("title", "Box Plot"),
                xaxis_title=data.get("xaxis_title", "Value"),
                yaxis_title=data.get("yaxis_title", "Group" if groups is not None else ""),
                boxmode="group" if groups is not None else None,
                hovermode="closest",
            )

        return fig
