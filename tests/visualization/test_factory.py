"""
Tests for chart factory.
"""

import pytest

from panelbox.visualization import ChartFactory, ChartRegistry, register_chart
from panelbox.visualization.base import BaseChart, PlotlyChartBase
from panelbox.visualization.themes import ACADEMIC_THEME, PROFESSIONAL_THEME


# Test chart for factory
@register_chart("test_factory_chart")
class TestFactoryChart(BaseChart):
    """Test chart for factory."""

    def _create_figure(self, data, **kwargs):
        return {"data": data}

    def to_json(self):
        return "{}"

    def to_html(self):
        return "<div></div>"


def test_factory_create_chart():
    """Test creating chart via factory."""
    chart = ChartFactory.create(
        chart_type="test_factory_chart",
        data={"x": [1, 2, 3]},
    )

    assert isinstance(chart, TestFactoryChart)
    assert chart.figure is not None


def test_factory_create_without_data():
    """Test creating chart without data."""
    chart = ChartFactory.create(chart_type="test_factory_chart")

    assert isinstance(chart, TestFactoryChart)
    assert chart.figure is None  # Not created yet


def test_factory_with_theme():
    """Test factory with theme."""
    chart = ChartFactory.create(chart_type="test_factory_chart", theme="academic")

    assert chart.theme == ACADEMIC_THEME


def test_factory_with_config():
    """Test factory with configuration."""
    config = {"title": "Test", "width": 1000}
    chart = ChartFactory.create(chart_type="test_factory_chart", config=config)

    assert chart.config == config


def test_factory_invalid_chart_type():
    """Test factory with invalid chart type."""
    with pytest.raises(ValueError, match="not registered"):
        ChartFactory.create(chart_type="nonexistent")


def test_factory_list_available_charts():
    """Test listing available charts."""
    charts = ChartFactory.list_available_charts()

    assert isinstance(charts, list)
    assert "test_factory_chart" in charts


def test_factory_get_chart_info():
    """Test getting chart info from factory."""
    info = ChartFactory.get_chart_info("test_factory_chart")

    assert info["name"] == "test_factory_chart"
    assert "class" in info


def test_factory_create_multiple():
    """Test creating multiple charts."""
    specs = [
        {"type": "test_factory_chart", "name": "chart1", "data": {"x": [1, 2]}},
        {"type": "test_factory_chart", "name": "chart2", "data": {"x": [3, 4]}},
    ]

    charts = ChartFactory.create_multiple(specs, common_theme="professional")

    assert len(charts) == 2
    assert "chart1" in charts
    assert "chart2" in charts
    assert charts["chart1"].theme == PROFESSIONAL_THEME


def test_factory_create_multiple_missing_type():
    """Test create_multiple with missing type."""
    specs = [{"data": {"x": [1, 2]}}]  # Missing 'type'

    with pytest.raises(ValueError, match="must have a 'type'"):
        ChartFactory.create_multiple(specs)


def test_factory_create_multiple_missing_data():
    """Test create_multiple with missing data."""
    specs = [{"type": "test_factory_chart"}]  # Missing 'data'

    with pytest.raises(ValueError, match="must have 'data'"):
        ChartFactory.create_multiple(specs)
