"""
Tests for base chart classes.
"""

import pytest

from panelbox.visualization.base import BaseChart, MatplotlibChartBase, PlotlyChartBase
from panelbox.visualization.themes import PROFESSIONAL_THEME


class ConcreteChart(BaseChart):
    """Concrete implementation for testing."""

    def _create_figure(self, data, **kwargs):
        return {"data": data}

    def to_json(self):
        return '{"chart": "test"}'

    def to_html(self):
        return "<div>Test Chart</div>"


def test_base_chart_initialization():
    """Test base chart initialization."""
    chart = ConcreteChart()
    assert chart.theme is not None
    assert chart.config == {}
    assert chart.figure is None


def test_base_chart_with_theme():
    """Test base chart with custom theme."""
    chart = ConcreteChart(theme=PROFESSIONAL_THEME)
    assert chart.theme == PROFESSIONAL_THEME
    assert chart.theme.name == "professional"


def test_base_chart_with_config():
    """Test base chart with configuration."""
    config = {"title": "Test Chart", "width": 1000}
    chart = ConcreteChart(config=config)
    assert chart.config == config


def test_base_chart_create():
    """Test chart creation template method."""
    chart = ConcreteChart()
    data = {"x": [1, 2, 3], "y": [4, 5, 6]}

    result = chart.create(data)

    assert result is chart  # Method chaining
    assert chart.figure is not None
    assert chart.figure["data"] == data


def test_base_chart_validate_data_invalid():
    """Test data validation with invalid data."""
    chart = ConcreteChart()

    with pytest.raises(ValueError, match="Data must be a dictionary"):
        chart.create("not a dict")


def test_plotly_chart_base():
    """Test Plotly chart base via concrete implementation."""
    pytest.importorskip("plotly")
    from panelbox.visualization.plotly.basic import BarChart

    # Test via a concrete implementation
    chart = BarChart()
    assert chart.theme is not None
    assert isinstance(chart, PlotlyChartBase)


def test_matplotlib_chart_base():
    """Test Matplotlib chart base initialization."""
    pytest.importorskip("matplotlib")

    # MatplotlibChartBase is abstract, so this test would need a concrete implementation
    # For now, just test that it can be subclassed
    class ConcreteMatplotlibChart(MatplotlibChartBase):
        def _create_figure(self, data, **kwargs):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            return fig, ax

    chart = ConcreteMatplotlibChart()
    assert chart.theme is not None
    assert chart.fig is None
    assert chart.ax is None
