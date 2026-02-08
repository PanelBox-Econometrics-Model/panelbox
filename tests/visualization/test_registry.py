"""
Tests for chart registry system.
"""

import pytest

from panelbox.visualization.base import BaseChart, PlotlyChartBase
from panelbox.visualization.registry import ChartRegistry, register_chart


# Test charts
class TestChart1(BaseChart):
    """Test chart 1."""

    def _create_figure(self, data, **kwargs):
        return data

    def to_json(self):
        return "{}"

    def to_html(self):
        return "<div></div>"


class TestChart2(PlotlyChartBase):
    """Test chart 2."""

    def _create_figure(self, data, **kwargs):
        import plotly.graph_objects as go

        return go.Figure()


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before each test."""
    # Store original registry
    original = ChartRegistry._registry.copy()

    yield

    # Restore original registry
    ChartRegistry._registry = original


def test_register_chart():
    """Test chart registration."""
    ChartRegistry.register("test_chart", TestChart1)

    assert ChartRegistry.is_registered("test_chart")
    assert ChartRegistry.get("test_chart") == TestChart1


def test_register_duplicate_chart():
    """Test registering duplicate chart raises error."""
    ChartRegistry.register("test_chart", TestChart1)

    with pytest.raises(ValueError, match="already registered"):
        ChartRegistry.register("test_chart", TestChart2)


def test_register_same_chart_twice():
    """Test registering same chart twice is okay."""
    ChartRegistry.register("test_chart", TestChart1)
    ChartRegistry.register("test_chart", TestChart1)  # Should not raise


def test_register_invalid_class():
    """Test registering invalid class."""
    with pytest.raises(TypeError, match="must be a class"):
        ChartRegistry.register("invalid", "not a class")


def test_register_non_basechart():
    """Test registering non-BaseChart class."""

    class NotAChart:
        pass

    with pytest.raises(ValueError, match="must inherit from BaseChart"):
        ChartRegistry.register("not_chart", NotAChart)


def test_get_unregistered_chart():
    """Test getting unregistered chart."""
    with pytest.raises(ValueError, match="not registered"):
        ChartRegistry.get("nonexistent")


def test_list_charts():
    """Test listing registered charts."""
    ChartRegistry.register("chart1", TestChart1)
    ChartRegistry.register("chart2", TestChart2)

    charts = ChartRegistry.list_charts()

    assert "chart1" in charts
    assert "chart2" in charts
    assert len(charts) >= 2  # May have others from imports


def test_is_registered():
    """Test checking if chart is registered."""
    ChartRegistry.register("test_chart", TestChart1)

    assert ChartRegistry.is_registered("test_chart")
    assert not ChartRegistry.is_registered("nonexistent")


def test_unregister():
    """Test unregistering a chart."""
    ChartRegistry.register("test_chart", TestChart1)
    ChartRegistry.unregister("test_chart")

    assert not ChartRegistry.is_registered("test_chart")


def test_unregister_nonexistent():
    """Test unregistering nonexistent chart."""
    with pytest.raises(ValueError, match="not registered"):
        ChartRegistry.unregister("nonexistent")


def test_get_chart_info():
    """Test getting chart information."""
    ChartRegistry.register("test_chart", TestChart1)

    info = ChartRegistry.get_chart_info("test_chart")

    assert info["name"] == "test_chart"
    assert info["class"] == "TestChart1"
    assert "description" in info


def test_register_chart_decorator():
    """Test @register_chart decorator."""

    @register_chart("decorated_chart")
    class DecoratedChart(BaseChart):
        """Decorated chart."""

        def _create_figure(self, data, **kwargs):
            return data

        def to_json(self):
            return "{}"

        def to_html(self):
            return "<div></div>"

    assert ChartRegistry.is_registered("decorated_chart")
    assert ChartRegistry.get("decorated_chart") == DecoratedChart
    assert DecoratedChart._registry_name == "decorated_chart"
