"""
Tests for base chart classes.

Tests BaseChart, PlotlyChartBase, MatplotlibChartBase, and NumpyEncoder.
"""

import datetime
import json
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest

from panelbox.visualization.base import (
    BaseChart,
    MatplotlibChartBase,
    NumpyEncoder,
    PlotlyChartBase,
)
from panelbox.visualization.themes import PROFESSIONAL_THEME

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# =====================================================================
# Concrete implementations for testing
# =====================================================================


class ConcreteChart(BaseChart):
    """Concrete implementation for testing."""

    def _create_figure(self, data, **kwargs):
        return {"data": data}

    def to_json(self):
        return '{"chart": "test"}'

    def to_html(self):
        return "<div>Test Chart</div>"


class ConcretePlotlyChart(PlotlyChartBase):
    """Concrete Plotly subclass for testing base class methods."""

    def _create_figure(self, data, **kwargs):
        fig = go.Figure(data=[go.Bar(x=[1, 2, 3], y=[4, 5, 6])])
        return fig


class ConcreteMatplotlibChart(MatplotlibChartBase):
    """Concrete Matplotlib subclass for testing base class methods."""

    def _create_figure(self, data, **kwargs):
        self.fig, self.ax = plt.subplots()
        self.ax.bar([1, 2, 3], [4, 5, 6])
        return self.fig


# =====================================================================
# BaseChart tests (existing)
# =====================================================================


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


def test_base_chart_repr():
    """Test BaseChart string representation."""
    chart = ConcreteChart(theme=PROFESSIONAL_THEME)
    repr_str = repr(chart)
    assert "ConcreteChart" in repr_str
    assert "professional" in repr_str


def test_base_chart_to_dict():
    """Test BaseChart to_dict returns proper dict."""
    chart = ConcreteChart(config={"width": 800})
    result = chart.to_dict()
    assert isinstance(result, dict)
    assert result["type"] == "ConcreteChart"
    assert result["config"] == {"width": 800}


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

    chart = ConcreteMatplotlibChart()
    assert chart.theme is not None
    assert chart.fig is None
    assert chart.ax is None


# =====================================================================
# PlotlyChartBase tests
# =====================================================================


class TestPlotlyChartBase:
    """Tests for PlotlyChartBase methods."""

    def test_to_json_no_figure_raises(self):
        """Test to_json raises ValueError without figure."""
        chart = ConcretePlotlyChart()
        with pytest.raises(ValueError, match="not been created"):
            chart.to_json()

    def test_to_html_no_figure_raises(self):
        """Test to_html raises ValueError without figure."""
        chart = ConcretePlotlyChart()
        with pytest.raises(ValueError, match="not been created"):
            chart.to_html()

    def test_to_json_with_figure(self):
        """Test to_json returns valid JSON after create."""
        chart = ConcretePlotlyChart()
        chart.create({"key": "value"})
        result = chart.to_json()
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_to_html_with_figure(self):
        """Test to_html returns HTML string after create."""
        chart = ConcretePlotlyChart()
        chart.create({"key": "value"})
        result = chart.to_html()
        assert isinstance(result, str)
        assert "<" in result

    def test_to_dict_no_figure(self):
        """Test to_dict without figure returns base dict."""
        chart = ConcretePlotlyChart(config={"width": 800})
        result = chart.to_dict()
        assert isinstance(result, dict)
        assert "figure" not in result
        assert result["config"] == {"width": 800}

    def test_to_dict_with_figure(self):
        """Test to_dict with figure includes figure data."""
        chart = ConcretePlotlyChart()
        chart.create({"key": "value"})
        result = chart.to_dict()
        assert isinstance(result, dict)
        assert "figure" in result
        assert isinstance(result["figure"], dict)

    def test_to_image_no_figure_raises(self):
        """Test to_image raises ValueError without figure."""
        chart = ConcretePlotlyChart()
        with pytest.raises(ValueError, match="not been created"):
            chart.to_image()

    def test_to_image_kaleido_import_error(self):
        """Test to_image raises ImportError when kaleido missing."""
        chart = ConcretePlotlyChart()
        chart.create({"key": "value"})

        with (
            patch(
                "plotly.io.to_image",
                side_effect=ValueError("kaleido is required"),
            ),
            pytest.raises(ImportError, match="kaleido"),
        ):
            chart.to_image()

    def test_to_image_non_kaleido_value_error(self):
        """Test to_image re-raises non-kaleido ValueError."""
        chart = ConcretePlotlyChart()
        chart.create({"key": "value"})

        with (
            patch(
                "plotly.io.to_image",
                side_effect=ValueError("some other error"),
            ),
            pytest.raises(ValueError, match="some other error"),
        ):
            chart.to_image()

    def test_save_image_no_figure_raises(self):
        """Test save_image raises ValueError without figure."""
        chart = ConcretePlotlyChart()
        with pytest.raises(ValueError, match="not been created"):
            chart.save_image("test.png")

    def test_save_image_no_extension_raises(self):
        """Test save_image raises for path without extension."""
        chart = ConcretePlotlyChart()
        chart.create({"key": "value"})
        with pytest.raises(ValueError, match="Cannot infer format"):
            chart.save_image("testfile")

    def test_save_image_invalid_format_raises(self):
        """Test save_image raises for invalid format."""
        chart = ConcretePlotlyChart()
        chart.create({"key": "value"})
        with pytest.raises(ValueError, match="Invalid format"):
            chart.save_image("test.xyz", format="xyz")

    def test_save_image_infers_format(self, tmp_path):
        """Test save_image infers format from file extension."""
        chart = ConcretePlotlyChart()
        chart.create({"key": "value"})
        file_path = str(tmp_path / "test.png")

        with patch.object(chart, "to_image", return_value=b"fake image data"):
            chart.save_image(file_path)

        with open(file_path, "rb") as f:
            assert f.read() == b"fake image data"

    def test_save_image_explicit_format(self, tmp_path):
        """Test save_image uses explicit format over extension."""
        chart = ConcretePlotlyChart()
        chart.create({"key": "value"})
        file_path = str(tmp_path / "test.png")

        with patch.object(chart, "to_image", return_value=b"svg data") as mock_to_image:
            chart.save_image(file_path, format="svg")
            mock_to_image.assert_called_once_with(format="svg", width=None, height=None, scale=1.0)

    def test_apply_theme_colors(self):
        """Test theme application sets colors on traces."""
        chart = ConcretePlotlyChart()
        chart.create({"key": "value"})
        # After create, _apply_theme is called, figure should have layout
        assert chart.figure is not None
        layout = chart.figure.layout
        assert layout.template is not None or layout.colorway is not None

    def test_create_base_layout_with_config(self):
        """Test _create_base_layout includes config overrides."""
        chart = ConcretePlotlyChart(config={"title": "Test", "width": 900, "height": 600})
        layout = chart._create_base_layout()
        assert layout["title"] == "Test"
        assert layout["width"] == 900
        assert layout["height"] == 600

    def test_to_png_convenience(self):
        """Test to_png convenience method."""
        chart = ConcretePlotlyChart()
        chart.create({"key": "value"})
        with patch.object(chart, "to_image", return_value=b"png data") as mock:
            result = chart.to_png(width=800, height=600)
            assert result == b"png data"
            mock.assert_called_once_with(format="png", width=800, height=600, scale=1.0)

    def test_to_svg_convenience(self):
        """Test to_svg convenience method."""
        chart = ConcretePlotlyChart()
        chart.create({"key": "value"})
        with patch.object(chart, "to_image", return_value=b"svg data") as mock:
            result = chart.to_svg(width=800)
            assert result == b"svg data"
            mock.assert_called_once_with(format="svg", width=800, height=None)

    def test_to_pdf_convenience(self):
        """Test to_pdf convenience method."""
        chart = ConcretePlotlyChart()
        chart.create({"key": "value"})
        with patch.object(chart, "to_image", return_value=b"pdf data") as mock:
            result = chart.to_pdf()
            assert result == b"pdf data"
            mock.assert_called_once_with(format="pdf", width=None, height=None)


# =====================================================================
# MatplotlibChartBase tests
# =====================================================================


class TestMatplotlibChartBase:
    """Tests for MatplotlibChartBase methods."""

    def test_create_and_theme_apply(self):
        """Test matplotlib chart creation and theme application."""
        chart = ConcreteMatplotlibChart()
        chart.create({"key": "value"})
        assert chart.fig is not None
        assert chart.ax is not None

    def test_to_base64(self):
        """Test to_base64 returns valid data URI."""
        chart = ConcreteMatplotlibChart()
        chart.create({"key": "value"})
        b64 = chart.to_base64()
        assert isinstance(b64, str)
        assert b64.startswith("data:image/png;base64,")
        assert len(b64) > 50

    def test_to_base64_no_figure_raises(self):
        """Test to_base64 raises ValueError without figure."""
        chart = ConcreteMatplotlibChart()
        with pytest.raises(ValueError, match="not been created"):
            chart.to_base64()

    def test_to_json(self):
        """Test matplotlib chart to_json returns JSON with image."""
        chart = ConcreteMatplotlibChart()
        chart.create({"key": "value"})
        json_str = chart.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["type"] == "matplotlib"
        assert "image" in parsed

    def test_to_html(self):
        """Test matplotlib chart to_html returns img tag."""
        chart = ConcreteMatplotlibChart()
        chart.create({"key": "value"})
        html = chart.to_html()
        assert isinstance(html, str)
        assert '<img src="data:image/' in html
        assert 'alt="Chart"' in html


# =====================================================================
# NumpyEncoder tests
# =====================================================================


class TestNumpyEncoder:
    """Tests for NumpyEncoder JSON serialization."""

    def test_numpy_array(self):
        """Test NumpyEncoder handles numpy arrays."""
        data = {"array": np.array([1, 2, 3])}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed["array"] == [1, 2, 3]

    def test_numpy_integer(self):
        """Test NumpyEncoder handles numpy integers."""
        data = {"val": np.int64(42)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed["val"] == 42

    def test_numpy_float(self):
        """Test NumpyEncoder handles numpy floats."""
        data = {"val": np.float64(3.14)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert abs(parsed["val"] - 3.14) < 1e-10

    def test_numpy_nan(self):
        """Test NumpyEncoder handles numpy NaN values."""
        # Use NumpyEncoder.default directly to verify conversion
        encoder = NumpyEncoder()
        result = encoder.default(np.float64("nan"))
        assert result is None

    def test_numpy_inf(self):
        """Test NumpyEncoder handles numpy Inf values."""
        encoder = NumpyEncoder()
        result = encoder.default(np.float64("inf"))
        assert result is None

    def test_numpy_bool(self):
        """Test NumpyEncoder handles numpy booleans."""
        data = {"val": np.bool_(True)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed["val"] is True

    def test_datetime(self):
        """Test NumpyEncoder handles datetime objects."""
        data = {"date": datetime.datetime(2024, 1, 15, 10, 30)}
        result = json.dumps(data, cls=NumpyEncoder)
        assert "2024" in result
        parsed = json.loads(result)
        assert "2024-01-15" in parsed["date"]

    def test_date(self):
        """Test NumpyEncoder handles date objects."""
        data = {"date": datetime.date(2024, 6, 15)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed["date"] == "2024-06-15"

    def test_time(self):
        """Test NumpyEncoder handles time objects."""
        data = {"time": datetime.time(14, 30, 0)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert "14:30" in parsed["time"]

    def test_unsupported_type_raises(self):
        """Test NumpyEncoder raises TypeError for unsupported types."""
        data = {"val": {1, 2, 3}}  # Sets are not JSON serializable
        with pytest.raises(TypeError):
            json.dumps(data, cls=NumpyEncoder)
