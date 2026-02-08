"""
Tests for time series chart implementations.

Tests all 3 time series chart types:
- PanelTimeSeriesChart
- TrendLineChart
- FacetedTimeSeriesChart
"""

import pytest
import numpy as np
import pandas as pd

from panelbox.visualization.plotly.timeseries import (
    PanelTimeSeriesChart,
    TrendLineChart,
    FacetedTimeSeriesChart,
)
from panelbox.visualization.themes import PROFESSIONAL_THEME, ACADEMIC_THEME


@pytest.fixture
def sample_panel_data():
    """Sample panel data with multiple entities."""
    np.random.seed(42)
    n_periods = 50
    n_entities = 5

    time = pd.date_range('2020-01-01', periods=n_periods)
    entities = ['A', 'B', 'C', 'D', 'E']

    data = {
        'time': np.tile(time, n_entities),
        'values': np.random.randn(n_periods * n_entities).cumsum(),
        'entity_id': np.repeat(entities, n_periods)
    }

    return data


@pytest.fixture
def sample_timeseries():
    """Sample single time series."""
    np.random.seed(42)
    n_periods = 100
    time = pd.date_range('2020-01-01', periods=n_periods)
    values = np.random.randn(n_periods).cumsum()

    return {
        'time': time,
        'values': values
    }


@pytest.fixture
def sample_faceted_data():
    """Sample data for faceted time series."""
    np.random.seed(42)
    n_periods = 30
    n_entities = 6

    time = pd.date_range('2020-01-01', periods=n_periods)
    entities = ['Entity1', 'Entity2', 'Entity3', 'Entity4', 'Entity5', 'Entity6']

    data = {
        'time': np.tile(time, n_entities),
        'values': np.random.randn(n_periods * n_entities) * 2 + np.repeat(np.arange(n_entities), n_periods),
        'entity_id': np.repeat(entities, n_periods)
    }

    return data


class TestPanelTimeSeriesChart:
    """Tests for PanelTimeSeriesChart."""

    def test_creation(self, sample_panel_data):
        """Test panel time series creation."""
        chart = PanelTimeSeriesChart()
        chart.create(sample_panel_data)

        assert chart.figure is not None
        # Should have one trace per entity (5)
        assert len(chart.figure.data) == 5

    def test_with_theme(self, sample_panel_data):
        """Test with theme."""
        chart = PanelTimeSeriesChart(theme=PROFESSIONAL_THEME)
        chart.create(sample_panel_data)

        assert chart.figure is not None

    def test_with_mean_line(self, sample_panel_data):
        """Test with mean line."""
        chart = PanelTimeSeriesChart()
        chart.create({
            **sample_panel_data,
            'show_mean': True
        })

        assert chart.figure is not None
        # Should have 5 entity traces + 1 mean trace
        assert len(chart.figure.data) == 6

    def test_custom_variable_name(self, sample_panel_data):
        """Test with custom variable name."""
        chart = PanelTimeSeriesChart()
        chart.create({
            **sample_panel_data,
            'variable_name': 'GDP Growth'
        })

        assert chart.figure is not None
        assert 'GDP Growth' in chart.figure.layout.title.text

    def test_entity_limit_warning(self):
        """Test warning for too many entities."""
        np.random.seed(42)
        n_periods = 20
        n_entities = 25  # More than default limit of 20

        time = pd.date_range('2020-01-01', periods=n_periods)
        entities = [f'Entity{i}' for i in range(n_entities)]

        data = {
            'time': np.tile(time, n_entities),
            'values': np.random.randn(n_periods * n_entities),
            'entity_id': np.repeat(entities, n_periods)
        }

        chart = PanelTimeSeriesChart()
        with pytest.warns(UserWarning, match="Too many entities"):
            chart.create(data)

        assert chart.figure is not None

    def test_missing_entity_id_raises_error(self, sample_timeseries):
        """Test that missing entity_id raises error."""
        chart = PanelTimeSeriesChart()

        with pytest.raises(ValueError, match="entity_id is required"):
            chart.create(sample_timeseries)

    def test_custom_title(self, sample_panel_data):
        """Test custom title."""
        chart = PanelTimeSeriesChart()
        chart.create({
            **sample_panel_data,
            'title': 'Custom Panel Plot'
        })

        assert chart.figure is not None
        assert 'Custom Panel Plot' in chart.figure.layout.title.text


class TestTrendLineChart:
    """Tests for TrendLineChart."""

    def test_creation(self, sample_timeseries):
        """Test trend line chart creation."""
        chart = TrendLineChart()
        chart.create(sample_timeseries)

        assert chart.figure is not None
        # Should have original + MA + trend
        assert len(chart.figure.data) >= 3

    def test_with_theme(self, sample_timeseries):
        """Test with theme."""
        chart = TrendLineChart(theme=ACADEMIC_THEME)
        chart.create(sample_timeseries)

        assert chart.figure is not None

    def test_with_moving_average(self, sample_timeseries):
        """Test with moving average."""
        chart = TrendLineChart()
        chart.create({
            **sample_timeseries,
            'show_moving_average': True,
            'window': 10
        })

        assert chart.figure is not None

    def test_without_moving_average(self, sample_timeseries):
        """Test without moving average."""
        chart = TrendLineChart()
        chart.create({
            **sample_timeseries,
            'show_moving_average': False
        })

        assert chart.figure is not None

    def test_with_trend_line(self, sample_timeseries):
        """Test with trend line."""
        chart = TrendLineChart()
        chart.create({
            **sample_timeseries,
            'show_trend': True
        })

        assert chart.figure is not None

    def test_without_trend_line(self, sample_timeseries):
        """Test without trend line."""
        chart = TrendLineChart()
        chart.create({
            **sample_timeseries,
            'show_trend': False,
            'show_moving_average': False
        })

        assert chart.figure is not None

    def test_custom_window_size(self, sample_timeseries):
        """Test custom MA window size."""
        chart = TrendLineChart()
        chart.create({
            **sample_timeseries,
            'window': 20
        })

        assert chart.figure is not None

    def test_short_series(self):
        """Test with series shorter than window."""
        short_data = {
            'time': pd.date_range('2020-01-01', periods=5),
            'values': np.array([1, 2, 3, 4, 5])
        }

        chart = TrendLineChart()
        chart.create({
            **short_data,
            'window': 10  # Longer than series
        })

        assert chart.figure is not None

    def test_with_nan_values(self):
        """Test with NaN values in series."""
        data = {
            'time': pd.date_range('2020-01-01', periods=50),
            'values': np.random.randn(50)
        }
        data['values'][10:15] = np.nan  # Add some NaN values

        chart = TrendLineChart()
        chart.create(data)

        assert chart.figure is not None


class TestFacetedTimeSeriesChart:
    """Tests for FacetedTimeSeriesChart."""

    def test_creation(self, sample_faceted_data):
        """Test faceted time series creation."""
        chart = FacetedTimeSeriesChart()
        chart.create(sample_faceted_data)

        assert chart.figure is not None
        # Should have 6 traces (one per entity)
        assert len(chart.figure.data) == 6

    def test_with_theme(self, sample_faceted_data):
        """Test with theme."""
        chart = FacetedTimeSeriesChart(theme=PROFESSIONAL_THEME)
        chart.create(sample_faceted_data)

        assert chart.figure is not None

    def test_custom_columns(self, sample_faceted_data):
        """Test with custom number of columns."""
        chart = FacetedTimeSeriesChart()
        chart.create({
            **sample_faceted_data,
            'ncols': 2
        })

        assert chart.figure is not None

    def test_shared_yaxis(self, sample_faceted_data):
        """Test with shared y-axis."""
        chart = FacetedTimeSeriesChart()
        chart.create({
            **sample_faceted_data,
            'shared_yaxis': True
        })

        assert chart.figure is not None

    def test_independent_yaxis(self, sample_faceted_data):
        """Test with independent y-axes."""
        chart = FacetedTimeSeriesChart()
        chart.create({
            **sample_faceted_data,
            'shared_yaxis': False
        })

        assert chart.figure is not None

    def test_custom_variable_name(self, sample_faceted_data):
        """Test with custom variable name."""
        chart = FacetedTimeSeriesChart()
        chart.create({
            **sample_faceted_data,
            'variable_name': 'Temperature'
        })

        assert chart.figure is not None

    def test_missing_entity_id_raises_error(self, sample_timeseries):
        """Test that missing entity_id raises error."""
        chart = FacetedTimeSeriesChart()

        with pytest.raises(ValueError, match="entity_id is required"):
            chart.create(sample_timeseries)

    def test_single_entity(self):
        """Test with single entity."""
        np.random.seed(42)
        data = {
            'time': pd.date_range('2020-01-01', periods=30),
            'values': np.random.randn(30),
            'entity_id': np.repeat(['A'], 30)
        }

        chart = FacetedTimeSeriesChart()
        chart.create(data)

        assert chart.figure is not None
        assert len(chart.figure.data) == 1

    def test_custom_height(self, sample_faceted_data):
        """Test with custom height."""
        chart = FacetedTimeSeriesChart()
        chart.create({
            **sample_faceted_data,
            'height': 800
        })

        assert chart.figure is not None
        assert chart.figure.layout.height == 800


class TestChartIntegration:
    """Integration tests for time series charts."""

    def test_all_charts_create_html(self, sample_panel_data, sample_timeseries, sample_faceted_data):
        """Test that all charts can export to HTML."""
        charts_data = [
            (PanelTimeSeriesChart(), sample_panel_data),
            (TrendLineChart(), sample_timeseries),
            (FacetedTimeSeriesChart(), sample_faceted_data),
        ]

        for chart, data in charts_data:
            chart.create(data)
            html = chart.to_html()
            assert html is not None
            assert isinstance(html, str)
            assert len(html) > 0

    def test_all_charts_create_json(self, sample_panel_data, sample_timeseries, sample_faceted_data):
        """Test that all charts can export to JSON."""
        charts_data = [
            (PanelTimeSeriesChart(), sample_panel_data),
            (TrendLineChart(), sample_timeseries),
            (FacetedTimeSeriesChart(), sample_faceted_data),
        ]

        for chart, data in charts_data:
            chart.create(data)
            json_data = chart.to_json()
            assert json_data is not None
            assert isinstance(json_data, str)
            assert len(json_data) > 0

    def test_all_charts_with_all_themes(self, sample_panel_data):
        """Test all charts work with all themes."""
        themes = [PROFESSIONAL_THEME, ACADEMIC_THEME, None]

        for theme in themes:
            chart = PanelTimeSeriesChart(theme=theme)
            chart.create(sample_panel_data)
            assert chart.figure is not None

    def test_panel_data_workflow(self, sample_panel_data):
        """Test typical panel data visualization workflow."""
        # 1. Overview with all entities
        panel_chart = PanelTimeSeriesChart()
        panel_chart.create({
            **sample_panel_data,
            'show_mean': True
        })
        assert panel_chart.figure is not None

        # 2. Faceted view for detailed comparison
        faceted_chart = FacetedTimeSeriesChart()
        faceted_chart.create(sample_panel_data)
        assert faceted_chart.figure is not None

        # Both charts should work with same data
        assert len(panel_chart.figure.data) > 0
        assert len(faceted_chart.figure.data) > 0
