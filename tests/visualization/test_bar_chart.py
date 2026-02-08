"""
Tests for BarChart implementation.
"""

import pytest

plotly = pytest.importorskip("plotly")

from panelbox.visualization import ChartFactory
from panelbox.visualization.plotly.basic import BarChart


def test_bar_chart_simple():
    """Test simple bar chart creation."""
    chart = BarChart()
    data = {"x": ["A", "B", "C"], "y": [10, 20, 15]}

    chart.create(data)

    assert chart.figure is not None
    assert len(chart.figure.data) == 1
    assert chart.figure.data[0].type == "bar"


def test_bar_chart_grouped():
    """Test grouped bar chart."""
    chart = BarChart()
    data = {"x": ["Q1", "Q2", "Q3"], "y": {"2023": [10, 15, 12], "2024": [12, 18, 14]}, "barmode": "group"}

    chart.create(data)

    assert len(chart.figure.data) == 2  # Two series
    assert chart.figure.layout.barmode == "group"


def test_bar_chart_stacked():
    """Test stacked bar chart."""
    chart = BarChart()
    data = {
        "x": ["A", "B", "C"],
        "y": {"Series 1": [10, 20, 15], "Series 2": [5, 10, 8]},
        "barmode": "stack",
    }

    chart.create(data)

    assert chart.figure.layout.barmode == "stack"


def test_bar_chart_horizontal():
    """Test horizontal bar chart."""
    chart = BarChart()
    data = {"x": [10, 20, 15], "y": ["Item 1", "Item 2", "Item 3"], "orientation": "h"}

    chart.create(data)

    assert chart.figure.data[0].orientation == "h"


def test_bar_chart_validation_missing_x():
    """Test validation with missing x."""
    chart = BarChart()
    data = {"y": [10, 20, 15]}

    with pytest.raises(ValueError, match="must contain 'x'"):
        chart.create(data)


def test_bar_chart_validation_missing_y():
    """Test validation with missing y."""
    chart = BarChart()
    data = {"x": ["A", "B", "C"]}

    with pytest.raises(ValueError, match="must contain 'y'"):
        chart.create(data)


def test_bar_chart_validation_length_mismatch():
    """Test validation with mismatched lengths."""
    chart = BarChart()
    data = {"x": ["A", "B", "C"], "y": [10, 20]}  # Lengths don't match

    with pytest.raises(ValueError, match="must match length"):
        chart.create(data)


def test_bar_chart_via_factory():
    """Test creating bar chart via factory."""
    chart = ChartFactory.create(chart_type="bar_chart", data={"x": ["A", "B"], "y": [10, 20]}, theme="professional")

    assert isinstance(chart, BarChart)
    assert chart.figure is not None


def test_bar_chart_to_html():
    """Test HTML export."""
    chart = BarChart()
    chart.create(data={"x": ["A", "B"], "y": [10, 20]})

    html = chart.to_html()

    assert isinstance(html, str)
    assert len(html) > 0
    assert "plotly" in html.lower()


def test_bar_chart_to_json():
    """Test JSON export."""
    chart = BarChart()
    chart.create(data={"x": ["A", "B"], "y": [10, 20]})

    json_str = chart.to_json()

    assert isinstance(json_str, str)
    assert len(json_str) > 0

    # Validate it's valid JSON
    import json

    parsed = json.loads(json_str)
    assert "data" in parsed
