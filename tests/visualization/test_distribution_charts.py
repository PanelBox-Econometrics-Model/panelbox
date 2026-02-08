"""
Tests for distribution chart implementations.

Tests all 4 distribution chart types:
- HistogramChart
- KDEChart
- ViolinPlotChart
- BoxPlotChart
"""

import pytest
import numpy as np
import pandas as pd

from panelbox.visualization.plotly.distribution import (
    HistogramChart,
    KDEChart,
    ViolinPlotChart,
    BoxPlotChart,
)
from panelbox.visualization.themes import PROFESSIONAL_THEME, ACADEMIC_THEME


@pytest.fixture
def sample_values():
    """Sample values for distribution tests."""
    np.random.seed(42)
    return np.random.normal(0, 1, 200)


@pytest.fixture
def sample_grouped_data():
    """Sample grouped data for distribution tests."""
    np.random.seed(42)
    group_a = np.random.normal(0, 1, 100)
    group_b = np.random.normal(1, 1.5, 100)
    group_c = np.random.normal(-0.5, 0.8, 100)

    return {
        'values': np.concatenate([group_a, group_b, group_c]),
        'groups': np.repeat(['A', 'B', 'C'], 100)
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for distribution tests."""
    np.random.seed(42)
    return pd.DataFrame({
        'value': np.random.normal(0, 1, 150),
        'category': np.repeat(['X', 'Y', 'Z'], 50)
    })


class TestHistogramChart:
    """Tests for HistogramChart."""

    def test_creation(self, sample_values):
        """Test histogram creation."""
        chart = HistogramChart()
        chart.create({'values': sample_values})

        assert chart.figure is not None
        assert len(chart.figure.data) >= 1

    def test_with_theme(self, sample_values):
        """Test with theme."""
        chart = HistogramChart(theme=PROFESSIONAL_THEME)
        chart.create({'values': sample_values})

        assert chart.figure is not None

    def test_with_kde_overlay(self, sample_values):
        """Test with KDE overlay."""
        chart = HistogramChart()
        chart.create({
            'values': sample_values,
            'show_kde': True
        })

        assert chart.figure is not None
        # Should have histogram + KDE trace
        assert len(chart.figure.data) >= 2

    def test_with_normal_overlay(self, sample_values):
        """Test with normal distribution overlay."""
        chart = HistogramChart()
        chart.create({
            'values': sample_values,
            'show_normal': True
        })

        assert chart.figure is not None
        assert len(chart.figure.data) >= 2

    def test_custom_bins(self, sample_values):
        """Test with custom number of bins."""
        chart = HistogramChart()
        chart.create({
            'values': sample_values,
            'nbins': 30
        })

        assert chart.figure is not None

    def test_grouped_histogram(self, sample_grouped_data):
        """Test grouped histogram."""
        chart = HistogramChart()
        chart.create(sample_grouped_data)

        assert chart.figure is not None
        # Should have trace for each group
        assert len(chart.figure.data) >= 3

    def test_normalized_histogram(self, sample_values):
        """Test normalized histogram (density)."""
        chart = HistogramChart()
        chart.create({
            'values': sample_values,
            'histnorm': 'probability density'
        })

        assert chart.figure is not None

    def test_custom_title(self, sample_values):
        """Test custom title."""
        chart = HistogramChart()
        chart.create({
            'values': sample_values,
            'title': 'Custom Histogram Title'
        })

        assert chart.figure is not None
        assert 'Custom Histogram' in chart.figure.layout.title.text


class TestKDEChart:
    """Tests for KDEChart."""

    def test_creation(self, sample_values):
        """Test KDE chart creation."""
        chart = KDEChart()
        chart.create({'values': sample_values})

        assert chart.figure is not None
        assert len(chart.figure.data) >= 1

    def test_with_theme(self, sample_values):
        """Test with theme."""
        chart = KDEChart(theme=ACADEMIC_THEME)
        chart.create({'values': sample_values})

        assert chart.figure is not None

    def test_with_rug_plot(self, sample_values):
        """Test with rug plot showing observations."""
        chart = KDEChart()
        chart.create({
            'values': sample_values,
            'show_rug': True
        })

        assert chart.figure is not None
        # Should have KDE + rug traces
        assert len(chart.figure.data) >= 2

    def test_with_statistics(self, sample_values):
        """Test with mean/median lines."""
        chart = KDEChart()
        chart.create({
            'values': sample_values,
            'show_mean': True,
            'show_median': True
        })

        assert chart.figure is not None

    def test_grouped_kde(self, sample_grouped_data):
        """Test grouped KDE."""
        chart = KDEChart()
        chart.create(sample_grouped_data)

        assert chart.figure is not None
        # Should have trace for each group
        assert len(chart.figure.data) >= 3

    def test_filled_kde(self, sample_values):
        """Test filled KDE."""
        chart = KDEChart()
        chart.create({
            'values': sample_values,
            'fill': True
        })

        assert chart.figure is not None

    def test_custom_bandwidth(self, sample_values):
        """Test with custom bandwidth."""
        chart = KDEChart()
        chart.create({
            'values': sample_values,
            'bandwidth': 0.5
        })

        assert chart.figure is not None


class TestViolinPlotChart:
    """Tests for ViolinPlotChart."""

    def test_creation(self, sample_grouped_data):
        """Test violin plot creation."""
        chart = ViolinPlotChart()
        chart.create(sample_grouped_data)

        assert chart.figure is not None
        assert len(chart.figure.data) >= 3

    def test_with_theme(self, sample_grouped_data):
        """Test with theme."""
        chart = ViolinPlotChart(theme=PROFESSIONAL_THEME)
        chart.create(sample_grouped_data)

        assert chart.figure is not None

    def test_with_box_plot(self, sample_grouped_data):
        """Test with box plot inside violin."""
        chart = ViolinPlotChart()
        chart.create({
            **sample_grouped_data,
            'show_box': True
        })

        assert chart.figure is not None

    def test_with_points(self, sample_grouped_data):
        """Test with individual points."""
        chart = ViolinPlotChart()
        chart.create({
            **sample_grouped_data,
            'show_points': True
        })

        assert chart.figure is not None

    def test_single_group(self, sample_values):
        """Test with single group."""
        chart = ViolinPlotChart()
        chart.create({'values': sample_values})

        assert chart.figure is not None

    def test_horizontal_orientation(self, sample_grouped_data):
        """Test horizontal orientation."""
        chart = ViolinPlotChart()
        chart.create({
            **sample_grouped_data,
            'orientation': 'h'
        })

        assert chart.figure is not None

    def test_with_meanline(self, sample_grouped_data):
        """Test with meanline visible."""
        chart = ViolinPlotChart()
        chart.create({
            **sample_grouped_data,
            'meanline_visible': True
        })

        assert chart.figure is not None


class TestBoxPlotChart:
    """Tests for BoxPlotChart."""

    def test_creation(self, sample_grouped_data):
        """Test box plot creation."""
        chart = BoxPlotChart()
        chart.create(sample_grouped_data)

        assert chart.figure is not None
        assert len(chart.figure.data) >= 3

    def test_with_theme(self, sample_grouped_data):
        """Test with theme."""
        chart = BoxPlotChart(theme=ACADEMIC_THEME)
        chart.create(sample_grouped_data)

        assert chart.figure is not None

    def test_with_points(self, sample_grouped_data):
        """Test with all points displayed."""
        chart = BoxPlotChart()
        chart.create({
            **sample_grouped_data,
            'show_points': 'all'
        })

        assert chart.figure is not None

    def test_with_outliers_only(self, sample_grouped_data):
        """Test showing outliers only."""
        chart = BoxPlotChart()
        chart.create({
            **sample_grouped_data,
            'show_points': 'outliers'
        })

        assert chart.figure is not None

    def test_with_mean(self, sample_grouped_data):
        """Test with mean displayed."""
        chart = BoxPlotChart()
        chart.create({
            **sample_grouped_data,
            'show_mean': True
        })

        assert chart.figure is not None

    def test_single_group(self, sample_values):
        """Test with single group."""
        chart = BoxPlotChart()
        chart.create({'values': sample_values})

        assert chart.figure is not None

    def test_horizontal_orientation(self, sample_grouped_data):
        """Test horizontal orientation."""
        chart = BoxPlotChart()
        chart.create({
            **sample_grouped_data,
            'orientation': 'h'
        })

        assert chart.figure is not None

    def test_notched_boxplot(self, sample_grouped_data):
        """Test notched box plot."""
        chart = BoxPlotChart()
        chart.create({
            **sample_grouped_data,
            'notched': True
        })

        assert chart.figure is not None


class TestChartIntegration:
    """Integration tests for distribution charts."""

    def test_all_charts_create_html(self, sample_values, sample_grouped_data):
        """Test that all charts can export to HTML."""
        charts_data = [
            (HistogramChart(), {'values': sample_values}),
            (KDEChart(), {'values': sample_values}),
            (ViolinPlotChart(), sample_grouped_data),
            (BoxPlotChart(), sample_grouped_data),
        ]

        for chart, data in charts_data:
            chart.create(data)
            html = chart.to_html()
            assert html is not None
            assert isinstance(html, str)
            assert len(html) > 0

    def test_all_charts_create_json(self, sample_values, sample_grouped_data):
        """Test that all charts can export to JSON."""
        charts_data = [
            (HistogramChart(), {'values': sample_values}),
            (KDEChart(), {'values': sample_values}),
            (ViolinPlotChart(), sample_grouped_data),
            (BoxPlotChart(), sample_grouped_data),
        ]

        for chart, data in charts_data:
            chart.create(data)
            json_data = chart.to_json()
            assert json_data is not None
            assert isinstance(json_data, str)
            assert len(json_data) > 0

    def test_all_charts_with_all_themes(self, sample_values):
        """Test all charts work with all themes."""
        themes = [PROFESSIONAL_THEME, ACADEMIC_THEME, None]

        for theme in themes:
            chart = HistogramChart(theme=theme)
            chart.create({'values': sample_values})
            assert chart.figure is not None

    def test_consistent_api_across_charts(self, sample_grouped_data):
        """Test that all charts accept similar data structures."""
        charts = [
            HistogramChart(),
            KDEChart(),
            ViolinPlotChart(),
            BoxPlotChart(),
        ]

        for chart in charts:
            chart.create(sample_grouped_data)
            assert chart.figure is not None
