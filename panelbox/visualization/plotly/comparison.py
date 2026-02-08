"""
Model comparison charts for PanelBox visualization system.

This module provides charts for comparing multiple models side-by-side,
including coefficient comparisons, forest plots, fit statistics, and
information criteria.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..base import PlotlyChartBase
from ..registry import register_chart


@register_chart("comparison_coefficients")
class CoefficientComparisonChart(PlotlyChartBase):
    """
    Coefficient comparison plot.

    Grouped bar chart comparing coefficient estimates across multiple models
    with error bars for confidence intervals and significance indicators.

    Examples
    --------
    >>> from panelbox.visualization import CoefficientComparisonChart
    >>>
    >>> # Data: list of model results
    >>> data = {
    >>>     'models': ['Model 1', 'Model 2', 'Model 3'],
    >>>     'coefficients': {
    >>>         'x1': [0.5, 0.6, 0.55],
    >>>         'x2': [0.3, 0.25, 0.28],
    >>>         'x3': [-0.2, -0.15, -0.18]
    >>>     },
    >>>     'std_errors': {
    >>>         'x1': [0.1, 0.12, 0.11],
    >>>         'x2': [0.08, 0.09, 0.07],
    >>>         'x3': [0.05, 0.06, 0.05]
    >>>     }
    >>> }
    >>>
    >>> chart = CoefficientComparisonChart()
    >>> chart.create(data)
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create coefficient comparison chart."""
        models = data.get("models", [])
        coefficients = data.get("coefficients", {})
        std_errors = data.get("std_errors", {})
        show_significance = data.get("show_significance", True)
        ci_level = data.get("ci_level", 0.95)

        # Calculate confidence intervals
        z_score = 1.96 if ci_level == 0.95 else 2.576  # 95% or 99%

        fig = go.Figure()

        # Add bars for each model
        variables = list(coefficients.keys())
        x_positions = np.arange(len(variables))
        bar_width = 0.8 / len(models)

        for i, model in enumerate(models):
            coef_values = [coefficients[var][i] for var in variables]

            # Calculate error bars if std_errors provided
            if std_errors:
                errors = [std_errors[var][i] * z_score for var in variables]
            else:
                errors = None

            # Offset for grouped bars
            offset = (i - len(models) / 2 + 0.5) * bar_width
            x_pos = x_positions + offset

            # Add bars
            fig.add_trace(
                go.Bar(
                    name=model,
                    x=variables,
                    y=coef_values,
                    error_y=dict(type="data", array=errors, visible=True) if errors else None,
                    offsetgroup=i,
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{model}</b><br>"
                        + "Variable: %{x}<br>"
                        + "Coefficient: %{y:.4f}<br>"
                        + "<extra></extra>"
                    ),
                )
            )

        # Add reference line at zero
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text="Zero",
            annotation_position="right",
        )

        # Update layout
        fig.update_layout(
            title=data.get("title", "Coefficient Comparison Across Models"),
            xaxis_title=data.get("xaxis_title", "Variables"),
            yaxis_title=data.get("yaxis_title", "Coefficient Estimate"),
            barmode="group",
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig


@register_chart("comparison_forest_plot")
class ForestPlotChart(PlotlyChartBase):
    """
    Forest plot for coefficient visualization.

    Horizontal layout showing point estimates with confidence intervals,
    commonly used in meta-analysis and model comparison.

    Examples
    --------
    >>> from panelbox.visualization import ForestPlotChart
    >>>
    >>> # Data format
    >>> data = {
    >>>     'variables': ['x1', 'x2', 'x3', 'x4'],
    >>>     'estimates': [0.5, 0.3, -0.2, 0.8],
    >>>     'ci_lower': [0.3, 0.1, -0.4, 0.6],
    >>>     'ci_upper': [0.7, 0.5, 0.0, 1.0],
    >>>     'pvalues': [0.001, 0.01, 0.05, 0.0001]
    >>> }
    >>>
    >>> chart = ForestPlotChart()
    >>> chart.create(data)
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create forest plot."""
        variables = data.get("variables", [])
        estimates = data.get("estimates", [])
        ci_lower = data.get("ci_lower", [])
        ci_upper = data.get("ci_upper", [])
        pvalues = data.get("pvalues", None)

        # Sort by effect size if requested
        if data.get("sort_by_size", False):
            sorted_indices = np.argsort(np.abs(estimates))[::-1]
            variables = [variables[i] for i in sorted_indices]
            estimates = [estimates[i] for i in sorted_indices]
            ci_lower = [ci_lower[i] for i in sorted_indices]
            ci_upper = [ci_upper[i] for i in sorted_indices]
            if pvalues:
                pvalues = [pvalues[i] for i in sorted_indices]

        # Determine significance colors
        if pvalues:
            colors = []
            for p in pvalues:
                if p < 0.001:
                    colors.append("darkgreen")
                elif p < 0.01:
                    colors.append("green")
                elif p < 0.05:
                    colors.append("orange")
                else:
                    colors.append("gray")
        else:
            colors = ["steelblue"] * len(variables)

        fig = go.Figure()

        # Add error bars (confidence intervals)
        for i, var in enumerate(variables):
            fig.add_trace(
                go.Scatter(
                    x=[ci_lower[i], estimates[i], ci_upper[i]],
                    y=[var, var, var],
                    mode="lines+markers",
                    marker=dict(size=[0, 10, 0], color=colors[i]),
                    line=dict(color=colors[i], width=2),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{var}</b><br>"
                        + f"Estimate: {estimates[i]:.4f}<br>"
                        + f"95% CI: [{ci_lower[i]:.4f}, {ci_upper[i]:.4f}]<br>"
                        + f"p-value: {pvalues[i]:.4f}<br>"
                        if pvalues
                        else "" + "<extra></extra>"
                    ),
                )
            )

        # Add vertical reference line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1, opacity=0.7)

        # Update layout
        fig.update_layout(
            title=data.get("title", "Forest Plot: Coefficient Estimates with 95% CI"),
            xaxis_title=data.get("xaxis_title", "Coefficient Estimate"),
            yaxis_title=data.get("yaxis_title", "Variables"),
            hovermode="closest",
            yaxis=dict(categoryorder="array", categoryarray=variables),
        )

        # Add legend for significance levels if pvalues provided
        if pvalues:
            # Add dummy traces for legend
            for label, color in [
                ("p < 0.001***", "darkgreen"),
                ("p < 0.01**", "green"),
                ("p < 0.05*", "orange"),
                ("p >= 0.05", "gray"),
            ]:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(size=10, color=color),
                        name=label,
                        showlegend=True,
                    )
                )

        return fig


@register_chart("comparison_model_fit")
class ModelFitComparisonChart(PlotlyChartBase):
    """
    Model fit comparison chart.

    Grouped bar chart comparing fit statistics (R², Adj. R², F-statistic, etc.)
    across multiple models.

    Examples
    --------
    >>> from panelbox.visualization import ModelFitComparisonChart
    >>>
    >>> data = {
    >>>     'models': ['OLS', 'Fixed Effects', 'Random Effects'],
    >>>     'metrics': {
    >>>         'R²': [0.75, 0.82, 0.78],
    >>>         'Adj. R²': [0.73, 0.80, 0.76],
    >>>         'F-statistic': [45.3, 52.1, 48.7]
    >>>     }
    >>> }
    >>>
    >>> chart = ModelFitComparisonChart()
    >>> chart.create(data)
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create model fit comparison chart."""
        models = data.get("models", [])
        metrics = data.get("metrics", {})
        normalize = data.get("normalize", False)

        fig = go.Figure()

        # Add bars for each model
        for model in models:
            metric_values = []
            metric_names = []

            for metric_name, values in metrics.items():
                model_idx = models.index(model)
                metric_values.append(values[model_idx])
                metric_names.append(metric_name)

            # Normalize if requested (for better comparison of different scales)
            if normalize:
                max_val = max(metric_values)
                if max_val > 0:
                    metric_values = [v / max_val for v in metric_values]

            fig.add_trace(
                go.Bar(
                    name=model,
                    x=metric_names,
                    y=metric_values,
                    text=[f"{v:.3f}" for v in metric_values],
                    textposition="auto",
                    hovertemplate=(
                        f"<b>{model}</b><br>"
                        + "Metric: %{x}<br>"
                        + "Value: %{y:.4f}<br>"
                        + "<extra></extra>"
                    ),
                )
            )

        # Update layout
        fig.update_layout(
            title=data.get("title", "Model Fit Comparison"),
            xaxis_title=data.get("xaxis_title", "Fit Metrics"),
            yaxis_title=data.get("yaxis_title", "Value" + (" (Normalized)" if normalize else "")),
            barmode="group",
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig


@register_chart("comparison_ic")
class InformationCriteriaChart(PlotlyChartBase):
    """
    Information criteria comparison chart.

    Bar chart comparing AIC, BIC, and other information criteria across models.
    Lower values indicate better fit.

    Examples
    --------
    >>> from panelbox.visualization import InformationCriteriaChart
    >>>
    >>> data = {
    >>>     'models': ['Model 1', 'Model 2', 'Model 3'],
    >>>     'aic': [1234.5, 1220.3, 1245.8],
    >>>     'bic': [1250.2, 1235.7, 1262.1]
    >>> }
    >>>
    >>> chart = InformationCriteriaChart()
    >>> chart.create(data)
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create information criteria comparison chart."""
        models = data.get("models", [])
        aic = data.get("aic", None)
        bic = data.get("bic", None)
        hqic = data.get("hqic", None)
        show_delta = data.get("show_delta", True)

        fig = go.Figure()

        # Determine which criteria to plot
        criteria = []
        if aic is not None:
            criteria.append(("AIC", aic))
        if bic is not None:
            criteria.append(("BIC", bic))
        if hqic is not None:
            criteria.append(("HQIC", hqic))

        # Add bars for each criterion
        for criterion_name, values in criteria:
            # Calculate delta from best (lowest)
            best_value = min(values)
            deltas = [v - best_value for v in values]

            # Find best model
            best_idx = values.index(best_value)

            # Create hover text
            hover_text = []
            for i, (model, value, delta) in enumerate(zip(models, values, deltas)):
                text = f"<b>{model}</b><br>"
                text += f"{criterion_name}: {value:.2f}<br>"
                if show_delta:
                    text += f"Δ{criterion_name}: {delta:.2f}<br>"
                if i == best_idx:
                    text += "<b>★ Best Model</b>"
                hover_text.append(text)

            fig.add_trace(
                go.Bar(
                    name=criterion_name,
                    x=models,
                    y=values,
                    text=[f"{v:.1f}" for v in values],
                    textposition="auto",
                    hovertemplate="%{hovertext}<extra></extra>",
                    hovertext=hover_text,
                    marker=dict(
                        # Highlight best model
                        line=dict(
                            color=[
                                "gold" if i == best_idx else "rgba(0,0,0,0)"
                                for i in range(len(models))
                            ],
                            width=3,
                        )
                    ),
                )
            )

        # Update layout
        fig.update_layout(
            title=data.get(
                "title",
                "Information Criteria Comparison<br><sub>Lower values indicate better fit</sub>",
            ),
            xaxis_title=data.get("xaxis_title", "Models"),
            yaxis_title=data.get("yaxis_title", "IC Value"),
            barmode="group",
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            annotations=[
                dict(
                    text="★ = Best Model (lowest IC)",
                    xref="paper",
                    yref="paper",
                    x=1,
                    y=-0.15,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                )
            ],
        )

        return fig
