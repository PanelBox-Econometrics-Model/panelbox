"""
Validation chart implementations.

This module provides interactive Plotly charts for visualizing
validation test results from panel data models.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..base import PlotlyChartBase
from ..config.color_schemes import SIGNIFICANCE_COLORS, get_color_for_pvalue
from ..registry import register_chart


@register_chart("validation_test_overview")
class TestOverviewChart(PlotlyChartBase):
    """
    Test overview stacked bar chart.

    Visualizes test results by category showing passed/failed counts
    in a stacked bar chart format.

    Data Format
    -----------
    data : dict
        Must contain:
        - 'categories': list[str] - Test category names
        - 'passed': list[int] - Number of passed tests per category
        - 'failed': list[int] - Number of failed tests per category

        Optional:
        - 'show_percentages': bool - Show percentage labels (default: True)

    Examples
    --------
    >>> chart = TestOverviewChart()
    >>> chart.create(data={
    ...     'categories': ['Specification', 'Serial Correlation',
    ...                    'Heteroskedasticity', 'Cross-Sectional'],
    ...     'passed': [5, 3, 2, 4],
    ...     'failed': [1, 2, 0, 1]
    ... })
    >>> html = chart.to_html()
    """

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Validate test overview data."""
        super()._validate_data(data)

        required = ["categories", "passed", "failed"]
        for field in required:
            if field not in data:
                raise ValueError(f"Test overview data must contain '{field}'")

        # Check lengths match
        if not (len(data["categories"]) == len(data["passed"]) == len(data["failed"])):
            raise ValueError("categories, passed, and failed must have same length")

    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data."""
        processed = data.copy()
        processed.setdefault("show_percentages", True)
        return processed

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create test overview stacked bar chart."""
        fig = go.Figure()

        categories = data["categories"]
        passed = data["passed"]
        failed = data["failed"]
        show_percentages = data["show_percentages"]

        # Calculate totals and percentages
        totals = [p + f for p, f in zip(passed, failed)]
        passed_pct = [100 * p / t if t > 0 else 0 for p, t in zip(passed, totals)]
        failed_pct = [100 * f / t if t > 0 else 0 for f, t in zip(failed, totals)]

        # Passed trace (green)
        passed_text = [
            f"{p} ({pct:.1f}%)" if show_percentages else str(p)
            for p, pct in zip(passed, passed_pct)
        ]

        fig.add_trace(
            go.Bar(
                name="Passed",
                x=categories,
                y=passed,
                text=passed_text,
                textposition="inside",
                marker_color=self.theme.success_color,
                hovertemplate="<b>%{x}</b><br>Passed: %{y}<br>%{text}<extra></extra>",
            )
        )

        # Failed trace (red)
        failed_text = [
            f"{f} ({pct:.1f}%)" if show_percentages else str(f)
            for f, pct in zip(failed, failed_pct)
        ]

        fig.add_trace(
            go.Bar(
                name="Failed",
                x=categories,
                y=failed,
                text=failed_text,
                textposition="inside",
                marker_color=self.theme.danger_color,
                hovertemplate="<b>%{x}</b><br>Failed: %{y}<br>%{text}<extra></extra>",
            )
        )

        # Update layout for stacked bars
        fig.update_layout(
            barmode="stack",
            xaxis_title=self.config.get("xaxis_title", "Test Category"),
            yaxis_title=self.config.get("yaxis_title", "Number of Tests"),
        )

        return fig


@register_chart("validation_pvalue_distribution")
class PValueDistributionChart(PlotlyChartBase):
    """
    P-value distribution bar chart.

    Visualizes p-values for all tests with color-coding based on
    significance levels and reference lines for common alpha thresholds.

    Data Format
    -----------
    data : dict
        Must contain:
        - 'test_names': list[str] - Names of tests
        - 'pvalues': list[float] - P-values for each test

        Optional:
        - 'alpha': float - Significance threshold (default: 0.05)
        - 'log_scale': bool - Use log scale for y-axis (default: True)

    Examples
    --------
    >>> chart = PValueDistributionChart()
    >>> chart.create(data={
    ...     'test_names': ['Hausman', 'Wooldridge', 'Pesaran CD'],
    ...     'pvalues': [0.001, 0.234, 0.012]
    ... })
    """

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Validate p-value distribution data."""
        super()._validate_data(data)

        if "test_names" not in data or "pvalues" not in data:
            raise ValueError("P-value data must contain 'test_names' and 'pvalues'")

        if len(data["test_names"]) != len(data["pvalues"]):
            raise ValueError("test_names and pvalues must have same length")

    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data."""
        processed = data.copy()
        processed.setdefault("alpha", 0.05)
        processed.setdefault("log_scale", True)
        return processed

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create p-value distribution chart."""
        fig = go.Figure()

        test_names = data["test_names"]
        pvalues = data["pvalues"]
        alpha = data["alpha"]
        log_scale = data["log_scale"]

        # Color-code bars by significance
        colors = [get_color_for_pvalue(p, alpha) for p in pvalues]

        # Create significance labels
        sig_labels = []
        for p in pvalues:
            if p < alpha / 10:
                sig_labels.append("***")
            elif p < alpha:
                sig_labels.append("**")
            elif p < alpha * 2:
                sig_labels.append("*")
            else:
                sig_labels.append("ns")

        # Bar chart
        fig.add_trace(
            go.Bar(
                x=test_names,
                y=pvalues,
                marker_color=colors,
                text=[f"p={p:.4f}<br>{sig}" for p, sig in zip(pvalues, sig_labels)],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>P-value: %{y:.4f}<extra></extra>",
            )
        )

        # Add reference lines for significance thresholds
        fig.add_hline(
            y=alpha,
            line_dash="dash",
            line_color="red",
            annotation_text=f"α = {alpha}",
            annotation_position="right",
        )

        fig.add_hline(
            y=alpha / 10,
            line_dash="dash",
            line_color="darkred",
            annotation_text=f"α/10 = {alpha/10}",
            annotation_position="right",
        )

        # Update layout
        yaxis_config = {"title": "P-value"}
        if log_scale:
            yaxis_config["type"] = "log"
            yaxis_config["range"] = [-4, 0]  # 0.0001 to 1

        fig.update_layout(
            xaxis_title=self.config.get("xaxis_title", "Test"), yaxis=yaxis_config, showlegend=False
        )

        return fig


@register_chart("validation_test_statistics")
class TestStatisticsChart(PlotlyChartBase):
    """
    Test statistics scatter plot.

    Visualizes the magnitude of test statistics with color-coding
    by test category.

    Data Format
    -----------
    data : dict
        Must contain:
        - 'test_names': list[str] - Names of tests
        - 'statistics': list[float] - Test statistics
        - 'categories': list[str] - Test categories

        Optional:
        - 'pvalues': list[float] - P-values (for size scaling)

    Examples
    --------
    >>> chart = TestStatisticsChart()
    >>> chart.create(data={
    ...     'test_names': ['Hausman', 'Wooldridge', 'Pesaran CD'],
    ...     'statistics': [15.3, 8.2, 2.1],
    ...     'categories': ['Specification', 'Serial', 'CD']
    ... })
    """

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Validate test statistics data."""
        super()._validate_data(data)

        required = ["test_names", "statistics", "categories"]
        for field in required:
            if field not in data:
                raise ValueError(f"Test statistics data must contain '{field}'")

        if not (len(data["test_names"]) == len(data["statistics"]) == len(data["categories"])):
            raise ValueError("test_names, statistics, and categories must have same length")

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create test statistics scatter plot."""
        fig = go.Figure()

        test_names = data["test_names"]
        statistics = data["statistics"]
        categories = data["categories"]
        pvalues = data.get("pvalues", None)

        # Get unique categories for color mapping
        unique_categories = list(set(categories))
        category_colors = {cat: self.theme.get_color(i) for i, cat in enumerate(unique_categories)}

        # Create traces by category
        for category in unique_categories:
            # Filter data for this category
            mask = [c == category for c in categories]
            cat_names = [n for n, m in zip(test_names, mask) if m]
            cat_stats = [s for s, m in zip(statistics, mask) if m]

            # Size by p-value if available
            if pvalues:
                cat_pvalues = [p for p, m in zip(pvalues, mask) if m]
                # Smaller p-value = larger marker
                sizes = [max(10, 30 * (1 - min(p, 1))) for p in cat_pvalues]
            else:
                sizes = 15

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(cat_names))),
                    y=cat_stats,
                    mode="markers",
                    name=category,
                    marker=dict(
                        size=sizes,
                        color=category_colors[category],
                        line=dict(width=1, color="white"),
                    ),
                    text=cat_names,
                    hovertemplate="<b>%{text}</b><br>Statistic: %{y:.2f}<extra></extra>",
                )
            )

        fig.update_layout(
            xaxis_title=self.config.get("xaxis_title", "Test Index"),
            yaxis_title=self.config.get("yaxis_title", "Test Statistic"),
            showlegend=True,
            legend_title_text="Category",
        )

        return fig


@register_chart("validation_comparison_heatmap")
class TestComparisonHeatmap(PlotlyChartBase):
    """
    Test comparison heatmap.

    Visualizes test results across multiple models or scenarios
    as a heatmap with color-coding based on p-values or pass/fail status.

    Data Format
    -----------
    data : dict
        Must contain:
        - 'models': list[str] - Model/scenario names (rows)
        - 'tests': list[str] - Test names (columns)
        - 'matrix': list[list[float]] - P-values or binary pass/fail

        Optional:
        - 'binary': bool - If True, matrix contains 0/1 (fail/pass)

    Examples
    --------
    >>> chart = TestComparisonHeatmap()
    >>> chart.create(data={
    ...     'models': ['Model 1', 'Model 2', 'Model 3'],
    ...     'tests': ['Hausman', 'Wooldridge', 'Pesaran CD'],
    ...     'matrix': [[0.01, 0.23, 0.001],
    ...                [0.45, 0.03, 0.12],
    ...                [0.001, 0.56, 0.08]]
    ... })
    """

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Validate heatmap data."""
        super()._validate_data(data)

        required = ["models", "tests", "matrix"]
        for field in required:
            if field not in data:
                raise ValueError(f"Heatmap data must contain '{field}'")

        # Validate matrix dimensions
        if len(data["matrix"]) != len(data["models"]):
            raise ValueError("Number of matrix rows must match number of models")

        for row in data["matrix"]:
            if len(row) != len(data["tests"]):
                raise ValueError("Number of matrix columns must match number of tests")

    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data."""
        processed = data.copy()
        processed.setdefault("binary", False)
        processed.setdefault("alpha", 0.05)
        return processed

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create comparison heatmap."""
        models = data["models"]
        tests = data["tests"]
        matrix = data["matrix"]
        binary = data["binary"]
        alpha = data["alpha"]

        # Create annotation text
        annotations = []
        for i, model in enumerate(models):
            for j, test in enumerate(tests):
                value = matrix[i][j]

                if binary:
                    text = "✓" if value == 1 else "✗"
                    text_color = "white"
                else:
                    text = f"{value:.3f}"
                    # Significance stars
                    if value < alpha / 10:
                        text += " ***"
                    elif value < alpha:
                        text += " **"
                    elif value < alpha * 2:
                        text += " *"
                    text_color = "black" if value > 0.5 else "white"

                annotations.append(
                    dict(x=j, y=i, text=text, showarrow=False, font=dict(color=text_color, size=10))
                )

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=tests,
                y=models,
                colorscale="RdYlGn" if not binary else [[0, "red"], [1, "green"]],
                reversescale=not binary,  # Lower p-values are better (green)
                showscale=True,
                hovertemplate="<b>%{y}</b><br>%{x}<br>Value: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            annotations=annotations,
            xaxis_title=self.config.get("xaxis_title", "Test"),
            yaxis_title=self.config.get("yaxis_title", "Model"),
        )

        return fig


@register_chart("validation_dashboard")
class ValidationDashboard(PlotlyChartBase):
    """
    Validation dashboard with multiple panels.

    Creates a comprehensive dashboard with 4 subplots:
    - Top-left: Test overview
    - Top-right: P-value distribution
    - Bottom-left: Test statistics
    - Bottom-right: Summary metrics

    Data Format
    -----------
    data : dict
        Must contain data for all subplots:
        - 'overview': dict - Data for TestOverviewChart
        - 'pvalues': dict - Data for PValueDistributionChart
        - 'statistics': dict - Data for TestStatisticsChart
        - 'summary': dict - Summary metrics

    Examples
    --------
    >>> chart = ValidationDashboard()
    >>> chart.create(data={
    ...     'overview': {...},
    ...     'pvalues': {...},
    ...     'statistics': {...},
    ...     'summary': {'total_tests': 10, 'passed': 8, 'failed': 2}
    ... })
    """

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Validate dashboard data."""
        super()._validate_data(data)

        required = ["overview", "pvalues", "statistics", "summary"]
        for field in required:
            if field not in data:
                raise ValueError(f"Dashboard data must contain '{field}'")

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create validation dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Test Overview", "P-value Distribution", "Test Statistics", "Summary"),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "indicator"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
        )

        # Subplot 1: Test Overview (stacked bar)
        overview_data = data["overview"]
        categories = overview_data["categories"]
        passed = overview_data["passed"]
        failed = overview_data["failed"]

        fig.add_trace(
            go.Bar(name="Passed", x=categories, y=passed, marker_color=self.theme.success_color),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(name="Failed", x=categories, y=failed, marker_color=self.theme.danger_color),
            row=1,
            col=1,
        )

        # Subplot 2: P-value Distribution
        pvalue_data = data["pvalues"]
        test_names = pvalue_data["test_names"]
        pvalues = pvalue_data["pvalues"]
        alpha = pvalue_data.get("alpha", 0.05)

        colors = [get_color_for_pvalue(p, alpha) for p in pvalues]

        fig.add_trace(
            go.Bar(x=test_names, y=pvalues, marker_color=colors, showlegend=False), row=1, col=2
        )

        # Subplot 3: Test Statistics
        stats_data = data["statistics"]
        stat_names = stats_data["test_names"]
        statistics = stats_data["statistics"]
        stat_categories = stats_data["categories"]

        unique_cats = list(set(stat_categories))
        for i, cat in enumerate(unique_cats):
            mask = [c == cat for c in stat_categories]
            cat_names = [n for n, m in zip(stat_names, mask) if m]
            cat_stats = [s for s, m in zip(statistics, mask) if m]

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(cat_names))),
                    y=cat_stats,
                    mode="markers",
                    name=cat,
                    marker=dict(size=12, color=self.theme.get_color(i)),
                ),
                row=2,
                col=1,
            )

        # Subplot 4: Summary Indicator
        summary = data["summary"]
        total = summary.get("total_tests", 0)
        passed_total = summary.get("passed", 0)
        pass_rate = 100 * passed_total / total if total > 0 else 0

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=pass_rate,
                title={"text": "Pass Rate (%)"},
                delta={"reference": 80},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": (
                            self.theme.success_color
                            if pass_rate >= 80
                            else self.theme.warning_color
                        )
                    },
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "gray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 80,
                    },
                },
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(barmode="stack", showlegend=True, height=self.config.get("height", 800))

        return fig
