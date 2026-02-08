"""
Tests for correlation chart implementations.

Tests all 2 correlation chart types:
- CorrelationHeatmapChart
- PairwiseCorrelationChart
"""

import pytest
import numpy as np
import pandas as pd

from panelbox.visualization.plotly.correlation import (
    CorrelationHeatmapChart,
    PairwiseCorrelationChart,
)
from panelbox.visualization.themes import PROFESSIONAL_THEME, ACADEMIC_THEME


@pytest.fixture
def sample_correlation_matrix():
    """Sample correlation matrix."""
    np.random.seed(42)
    # Create a realistic correlation matrix
    n_vars = 5
    data = np.random.randn(100, n_vars)
    df = pd.DataFrame(data, columns=[f'Var{i+1}' for i in range(n_vars)])
    return df.corr()


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for correlation tests."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'A': np.random.randn(n),
        'B': np.random.randn(n) * 2 + 1,
        'C': np.random.randn(n) * 0.5 - 0.5,
        'D': np.random.randn(n) * 1.5,
    })


@pytest.fixture
def sample_grouped_dataframe():
    """Sample DataFrame with groups for correlation tests."""
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        'A': np.random.randn(n),
        'B': np.random.randn(n) * 2,
        'C': np.random.randn(n) * 0.5,
        'group': np.repeat(['X', 'Y', 'Z'], 50)
    })


class TestCorrelationHeatmapChart:
    """Tests for CorrelationHeatmapChart."""

    def test_creation(self, sample_correlation_matrix):
        """Test correlation heatmap creation."""
        chart = CorrelationHeatmapChart()
        chart.create({'correlation_matrix': sample_correlation_matrix})

        assert chart.figure is not None
        assert len(chart.figure.data) == 1  # Single heatmap

    def test_with_theme(self, sample_correlation_matrix):
        """Test with theme."""
        chart = CorrelationHeatmapChart(theme=PROFESSIONAL_THEME)
        chart.create({'correlation_matrix': sample_correlation_matrix})

        assert chart.figure is not None

    def test_with_values_displayed(self, sample_correlation_matrix):
        """Test with values displayed on cells."""
        chart = CorrelationHeatmapChart()
        chart.create({
            'correlation_matrix': sample_correlation_matrix,
            'show_values': True
        })

        assert chart.figure is not None

    def test_without_values(self, sample_correlation_matrix):
        """Test without values displayed."""
        chart = CorrelationHeatmapChart()
        chart.create({
            'correlation_matrix': sample_correlation_matrix,
            'show_values': False
        })

        assert chart.figure is not None

    def test_mask_diagonal(self, sample_correlation_matrix):
        """Test with diagonal masked."""
        chart = CorrelationHeatmapChart()
        chart.create({
            'correlation_matrix': sample_correlation_matrix,
            'mask_diagonal': True
        })

        assert chart.figure is not None

    def test_mask_upper_triangle(self, sample_correlation_matrix):
        """Test with upper triangle masked."""
        chart = CorrelationHeatmapChart()
        chart.create({
            'correlation_matrix': sample_correlation_matrix,
            'mask_upper': True
        })

        assert chart.figure is not None

    def test_threshold_filtering(self, sample_correlation_matrix):
        """Test with threshold filtering."""
        chart = CorrelationHeatmapChart()
        chart.create({
            'correlation_matrix': sample_correlation_matrix,
            'threshold': 0.3
        })

        assert chart.figure is not None

    def test_custom_variable_names(self):
        """Test with custom variable names."""
        corr_matrix = np.array([
            [1.0, 0.5, -0.3],
            [0.5, 1.0, 0.2],
            [-0.3, 0.2, 1.0]
        ])
        chart = CorrelationHeatmapChart()
        chart.create({
            'correlation_matrix': corr_matrix,
            'variable_names': ['Income', 'Age', 'Education']
        })

        assert chart.figure is not None

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        corr_matrix = np.array([
            [1.0, 0.5, -0.3],
            [0.5, 1.0, 0.2],
            [-0.3, 0.2, 1.0]
        ])
        chart = CorrelationHeatmapChart()
        chart.create({'correlation_matrix': corr_matrix})

        assert chart.figure is not None

    def test_custom_title(self, sample_correlation_matrix):
        """Test custom title."""
        chart = CorrelationHeatmapChart()
        chart.create({
            'correlation_matrix': sample_correlation_matrix,
            'title': 'Custom Correlation Matrix'
        })

        assert chart.figure is not None
        assert 'Custom Correlation' in chart.figure.layout.title.text


class TestPairwiseCorrelationChart:
    """Tests for PairwiseCorrelationChart."""

    def test_creation(self, sample_dataframe):
        """Test pairwise correlation chart creation."""
        chart = PairwiseCorrelationChart()
        chart.create({'data': sample_dataframe})

        assert chart.figure is not None
        # Should have many traces (scatter + histogram for each pair)
        assert len(chart.figure.data) > 4

    def test_with_theme(self, sample_dataframe):
        """Test with theme."""
        chart = PairwiseCorrelationChart(theme=ACADEMIC_THEME)
        chart.create({'data': sample_dataframe})

        assert chart.figure is not None

    def test_subset_variables(self, sample_dataframe):
        """Test with subset of variables."""
        chart = PairwiseCorrelationChart()
        chart.create({
            'data': sample_dataframe,
            'variables': ['A', 'B', 'C']
        })

        assert chart.figure is not None

    def test_with_groups(self, sample_grouped_dataframe):
        """Test with group coloring."""
        chart = PairwiseCorrelationChart()
        chart.create({
            'data': sample_grouped_dataframe,
            'variables': ['A', 'B', 'C'],
            'group': 'group'
        })

        assert chart.figure is not None

    def test_without_diagonal_histogram(self, sample_dataframe):
        """Test without diagonal histograms."""
        chart = PairwiseCorrelationChart()
        chart.create({
            'data': sample_dataframe,
            'show_diagonal_hist': False
        })

        assert chart.figure is not None

    def test_variable_limit_warning(self):
        """Test that warning is raised for too many variables."""
        # Create DataFrame with 10 variables (more than limit of 8)
        np.random.seed(42)
        n = 100
        data = {f'Var{i}': np.random.randn(n) for i in range(10)}
        df = pd.DataFrame(data)

        chart = PairwiseCorrelationChart()
        with pytest.warns(UserWarning, match="Too many variables"):
            chart.create({'data': df})

        assert chart.figure is not None

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        np.random.seed(42)
        data = np.random.randn(100, 4)

        chart = PairwiseCorrelationChart()
        chart.create({'data': data})

        assert chart.figure is not None

    def test_custom_title(self, sample_dataframe):
        """Test custom title."""
        chart = PairwiseCorrelationChart()
        chart.create({
            'data': sample_dataframe,
            'title': 'Custom Pairwise Plot'
        })

        assert chart.figure is not None
        assert 'Custom Pairwise' in chart.figure.layout.title.text


class TestChartIntegration:
    """Integration tests for correlation charts."""

    def test_all_charts_create_html(self, sample_correlation_matrix, sample_dataframe):
        """Test that all charts can export to HTML."""
        charts_data = [
            (CorrelationHeatmapChart(), {'correlation_matrix': sample_correlation_matrix}),
            (PairwiseCorrelationChart(), {'data': sample_dataframe}),
        ]

        for chart, data in charts_data:
            chart.create(data)
            html = chart.to_html()
            assert html is not None
            assert isinstance(html, str)
            assert len(html) > 0

    def test_all_charts_create_json(self, sample_correlation_matrix, sample_dataframe):
        """Test that all charts can export to JSON."""
        charts_data = [
            (CorrelationHeatmapChart(), {'correlation_matrix': sample_correlation_matrix}),
            (PairwiseCorrelationChart(), {'data': sample_dataframe}),
        ]

        for chart, data in charts_data:
            chart.create(data)
            json_data = chart.to_json()
            assert json_data is not None
            assert isinstance(json_data, str)
            assert len(json_data) > 0

    def test_all_charts_with_all_themes(self, sample_correlation_matrix):
        """Test all charts work with all themes."""
        themes = [PROFESSIONAL_THEME, ACADEMIC_THEME, None]

        for theme in themes:
            chart = CorrelationHeatmapChart(theme=theme)
            chart.create({'correlation_matrix': sample_correlation_matrix})
            assert chart.figure is not None

    def test_workflow_correlation_analysis(self, sample_dataframe):
        """Test typical correlation analysis workflow."""
        # 1. Create correlation matrix
        corr_matrix = sample_dataframe.corr()

        # 2. Visualize with heatmap
        heatmap = CorrelationHeatmapChart()
        heatmap.create({
            'correlation_matrix': corr_matrix,
            'show_values': True,
            'mask_upper': True
        })
        assert heatmap.figure is not None

        # 3. Detailed pairwise view
        pairwise = PairwiseCorrelationChart()
        pairwise.create({'data': sample_dataframe})
        assert pairwise.figure is not None
