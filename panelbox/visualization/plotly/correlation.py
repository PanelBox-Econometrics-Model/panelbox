"""
Correlation visualization charts for PanelBox.

This module provides charts for visualizing correlations between variables,
including heatmaps and pairwise scatter matrices.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..base import PlotlyChartBase
from ..registry import register_chart


@register_chart("correlation_heatmap")
class CorrelationHeatmapChart(PlotlyChartBase):
    """
    Correlation matrix heatmap.

    Visualizes correlation coefficients between variables with a diverging
    color scale and optional hierarchical clustering.

    Examples
    --------
    >>> from panelbox.visualization import CorrelationHeatmapChart
    >>>
    >>> # Create correlation matrix
    >>> data = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    >>> corr_matrix = data.corr()
    >>>
    >>> chart = CorrelationHeatmapChart()
    >>> chart.create({'correlation_matrix': corr_matrix})
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create correlation heatmap."""
        corr_matrix = data.get("correlation_matrix")
        variable_names = data.get("variable_names", None)
        show_values = data.get("show_values", True)
        mask_diagonal = data.get("mask_diagonal", False)
        mask_upper = data.get("mask_upper", False)
        threshold = data.get("threshold", None)

        # Convert to numpy if pandas
        if isinstance(corr_matrix, pd.DataFrame):
            if variable_names is None:
                variable_names = list(corr_matrix.columns)
            corr_matrix = corr_matrix.values

        if variable_names is None:
            variable_names = [f"Var{i+1}" for i in range(corr_matrix.shape[0])]

        # Apply masks
        plot_matrix = corr_matrix.copy()

        if mask_diagonal:
            np.fill_diagonal(plot_matrix, np.nan)

        if mask_upper:
            plot_matrix = np.tril(plot_matrix, k=-1)
            plot_matrix[plot_matrix == 0] = np.nan

        # Apply threshold
        if threshold is not None:
            plot_matrix[np.abs(plot_matrix) < threshold] = np.nan

        # Create text annotations
        if show_values:
            text = [
                [f"{val:.2f}" if not np.isnan(val) else "" for val in row] for row in plot_matrix
            ]
        else:
            text = None

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=plot_matrix,
                x=variable_names,
                y=variable_names,
                colorscale="RdBu_r",  # Red-Blue diverging
                zmid=0,
                zmin=-1,
                zmax=1,
                text=text,
                texttemplate="%{text}" if text else None,
                textfont={"size": 10},
                hovertemplate=(
                    "<b>%{y} vs %{x}</b><br>" + "Correlation: %{z:.3f}<br>" + "<extra></extra>"
                ),
                colorbar=dict(
                    title="Correlation",
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["-1", "-0.5", "0", "0.5", "1"],
                ),
            )
        )

        # Update layout
        fig.update_layout(
            title=data.get("title", "Correlation Heatmap"),
            xaxis_title=data.get("xaxis_title", ""),
            yaxis_title=data.get("yaxis_title", ""),
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed"),  # Top to bottom
            width=data.get("width", 700),
            height=data.get("height", 650),
        )

        # Make square aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig


@register_chart("correlation_pairwise")
class PairwiseCorrelationChart(PlotlyChartBase):
    """
    Pairwise correlation scatter matrix.

    Shows scatter plots for all pairwise relationships between variables,
    with histograms on the diagonal.

    Examples
    --------
    >>> from panelbox.visualization import PairwiseCorrelationChart
    >>>
    >>> # Create data
    >>> data_df = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])
    >>>
    >>> chart = PairwiseCorrelationChart()
    >>> chart.create({'data': data_df})
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create pairwise correlation scatter matrix."""
        df = data.get("data")
        variables = data.get("variables", None)
        group_col = data.get("group", None)
        show_diagonal_hist = data.get("show_diagonal_hist", True)

        # Convert to DataFrame if not already
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Select variables
        if variables is None:
            # Use all numeric columns
            variables = df.select_dtypes(include=[np.number]).columns.tolist()

        # Limit number of variables for performance
        if len(variables) > 8:
            variables = variables[:8]
            import warnings

            warnings.warn("Too many variables. Limiting to first 8 for performance.")

        n_vars = len(variables)

        # Create subplots
        fig = make_subplots(
            rows=n_vars,
            cols=n_vars,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.02,
            horizontal_spacing=0.02,
        )

        # Group data if specified
        if group_col and group_col in df.columns:
            groups = df[group_col].unique()
            colors = self.theme.color_scheme if self.theme and self.theme.color_scheme else None
        else:
            groups = [None]
            colors = None

        # Create scatter plots
        for i, var_y in enumerate(variables):
            for j, var_x in enumerate(variables):
                row = i + 1
                col = j + 1

                if i == j and show_diagonal_hist:
                    # Diagonal: histogram
                    for g_idx, group in enumerate(groups):
                        if group is not None:
                            subset = df[df[group_col] == group]
                        else:
                            subset = df

                        fig.add_trace(
                            go.Histogram(
                                x=subset[var_x],
                                name=str(group) if group is not None else var_x,
                                marker_color=colors[g_idx % len(colors)] if colors else None,
                                opacity=0.7,
                                showlegend=(row == 1 and col == 1 and group is not None),
                                legendgroup=str(group) if group is not None else None,
                            ),
                            row=row,
                            col=col,
                        )
                else:
                    # Off-diagonal: scatter plot
                    for g_idx, group in enumerate(groups):
                        if group is not None:
                            subset = df[df[group_col] == group]
                        else:
                            subset = df

                        fig.add_trace(
                            go.Scatter(
                                x=subset[var_x],
                                y=subset[var_y],
                                mode="markers",
                                name=str(group) if group is not None else None,
                                marker=dict(
                                    size=4,
                                    color=colors[g_idx % len(colors)] if colors else None,
                                    opacity=0.5,
                                ),
                                showlegend=False,
                                hovertemplate=(
                                    f"<b>{var_x}</b>: %{{x:.2f}}<br>"
                                    + f"<b>{var_y}</b>: %{{y:.2f}}<br>"
                                    + "<extra></extra>"
                                ),
                            ),
                            row=row,
                            col=col,
                        )

                # Add axis labels only on edges
                if row == n_vars:
                    fig.update_xaxes(title_text=var_x, row=row, col=col)
                if col == 1:
                    fig.update_yaxes(title_text=var_y, row=row, col=col)

        # Update layout
        fig.update_layout(
            title=data.get("title", "Pairwise Correlation Matrix"),
            showlegend=group_col is not None,
            hovermode="closest",
            height=data.get("height", 150 * n_vars),
            width=data.get("width", 150 * n_vars),
        )

        return fig
