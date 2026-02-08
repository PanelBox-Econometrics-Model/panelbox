"""
Tests for validation chart implementations.

Tests all 5 validation chart types:
- TestOverviewChart
- PValueDistributionChart
- TestStatisticsChart
- TestComparisonHeatmap
- ValidationDashboard
"""

import pytest
import numpy as np

from panelbox.visualization.plotly.validation import (
    TestOverviewChart,
    PValueDistributionChart,
    TestStatisticsChart,
    TestComparisonHeatmap,
    ValidationDashboard,
)
from panelbox.visualization.themes import PROFESSIONAL_THEME, ACADEMIC_THEME


@pytest.fixture
def sample_test_overview_data():
    """Sample data for test overview chart."""
    return {
        "categories": ["Specification", "Serial Correlation", "Heteroskedasticity"],
        "passed": [2, 1, 3],
        "failed": [1, 2, 0],
    }


@pytest.fixture
def sample_pvalue_data():
    """Sample p-value distribution data."""
    return {
        "test_names": ["Hausman", "Wooldridge", "Breusch-Pagan", "Pesaran CD"],
        "pvalues": [0.001, 0.045, 0.234, 0.678],
        "alpha": 0.05,
    }


@pytest.fixture
def sample_test_statistics_data():
    """Sample test statistics data."""
    return {
        "test_names": ["Hausman", "Wooldridge", "Breusch-Pagan", "Pesaran CD"],
        "statistics": [15.2, 3.4, 2.1, 0.8],
        "categories": [
            "Specification",
            "Serial Correlation",
            "Heteroskedasticity",
            "Cross-Sectional Dep.",
        ],
        "pvalues": [0.001, 0.045, 0.234, 0.678],
    }


@pytest.fixture
def sample_heatmap_data():
    """Sample comparison heatmap data."""
    return {
        "models": ["Fixed Effects", "Random Effects", "Pooled OLS"],
        "tests": ["Hausman", "Wooldridge", "Breusch-Pagan"],
        "matrix": [[0.001, 0.045, 0.234], [0.678, 0.123, 0.456], [0.789, 0.012, 0.345]],
    }


@pytest.fixture
def sample_dashboard_data(
    sample_test_overview_data, sample_pvalue_data, sample_test_statistics_data
):
    """Sample dashboard data."""
    return {
        "overview": sample_test_overview_data,
        "pvalues": sample_pvalue_data,
        "statistics": sample_test_statistics_data,
        "summary": {"total_tests": 4, "passed": 2, "failed": 2},
    }


class TestTestOverviewChart:
    """Tests for TestOverviewChart."""

    def test_creation(self, sample_test_overview_data):
        """Test chart creation."""
        chart = TestOverviewChart()
        chart.create(sample_test_overview_data)

        assert chart.figure is not None
        assert len(chart.figure.data) == 2  # Passed and Failed traces

    def test_with_theme(self, sample_test_overview_data):
        """Test chart with theme."""
        chart = TestOverviewChart(theme=PROFESSIONAL_THEME)
        chart.create(sample_test_overview_data)

        assert chart.figure is not None
        # Check theme colors are applied
        assert chart.figure.data[0].marker.color is not None

    def test_stacked_mode(self, sample_test_overview_data):
        """Test stacked bar mode."""
        chart = TestOverviewChart()
        chart.create(sample_test_overview_data)

        # Default should be stacked
        assert chart.figure.layout.barmode == "stack"

    def test_percentages_display(self, sample_test_overview_data):
        """Test percentage display in bars."""
        chart = TestOverviewChart()
        chart.create(sample_test_overview_data)

        # Should have text labels with percentages
        assert chart.figure.data[0].text is not None
        assert chart.figure.data[1].text is not None

    def test_validation_missing_categories(self):
        """Test validation with missing categories."""
        chart = TestOverviewChart()

        with pytest.raises(ValueError, match="must contain 'categories'"):
            chart.create({"passed": [1, 2], "failed": [3, 4]})

    def test_validation_mismatched_lengths(self, sample_test_overview_data):
        """Test validation with mismatched array lengths."""
        chart = TestOverviewChart()

        bad_data = sample_test_overview_data.copy()
        bad_data["passed"] = [1, 2]  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            chart.create(bad_data)

    def test_to_html(self, sample_test_overview_data):
        """Test HTML export."""
        chart = TestOverviewChart()
        chart.create(sample_test_overview_data)

        html = chart.to_html()

        assert isinstance(html, str)
        assert "plotly" in html.lower()
        assert len(html) > 100

    def test_to_json(self, sample_test_overview_data):
        """Test JSON export."""
        chart = TestOverviewChart()
        chart.create(sample_test_overview_data)

        json_str = chart.to_json()

        assert isinstance(json_str, str)
        assert "data" in json_str
        assert "layout" in json_str


class TestPValueDistributionChart:
    """Tests for PValueDistributionChart."""

    def test_creation(self, sample_pvalue_data):
        """Test chart creation."""
        chart = PValueDistributionChart()
        chart.create(sample_pvalue_data)

        assert chart.figure is not None
        assert len(chart.figure.data) >= 1  # At least bar trace

    def test_color_coding(self, sample_pvalue_data):
        """Test p-value color coding."""
        chart = PValueDistributionChart()
        chart.create(sample_pvalue_data)

        # Should have colors based on significance
        assert chart.figure.data[0].marker.color is not None

    def test_log_scale(self, sample_pvalue_data):
        """Test log scale on y-axis."""
        chart = PValueDistributionChart()
        chart.create(sample_pvalue_data, log_scale=True)

        assert chart.figure.layout.yaxis.type == "log"

    def test_alpha_line(self, sample_pvalue_data):
        """Test alpha reference line."""
        chart = PValueDistributionChart()
        chart.create(sample_pvalue_data)

        # Should have horizontal line at alpha
        shapes = chart.figure.layout.shapes
        assert len(shapes) > 0
        assert shapes[0].y0 == 0.05

    def test_custom_alpha(self, sample_pvalue_data):
        """Test custom alpha threshold."""
        custom_data = sample_pvalue_data.copy()
        custom_data["alpha"] = 0.01

        chart = PValueDistributionChart()
        chart.create(custom_data)

        shapes = chart.figure.layout.shapes
        assert shapes[0].y0 == 0.01

    def test_validation_missing_pvalues(self):
        """Test validation with missing p-values."""
        chart = PValueDistributionChart()

        with pytest.raises(ValueError, match="must contain.*pvalues"):
            chart.create({"test_names": ["Test1", "Test2"]})

    def test_to_html(self, sample_pvalue_data):
        """Test HTML export."""
        chart = PValueDistributionChart()
        chart.create(sample_pvalue_data)

        html = chart.to_html()

        assert isinstance(html, str)
        assert len(html) > 100


class TestTestStatisticsChart:
    """Tests for TestStatisticsChart."""

    def test_creation(self, sample_test_statistics_data):
        """Test chart creation."""
        chart = TestStatisticsChart()
        chart.create(sample_test_statistics_data)

        assert chart.figure is not None
        assert len(chart.figure.data) > 0

    def test_size_scaling(self, sample_test_statistics_data):
        """Test point size scaling by p-value."""
        chart = TestStatisticsChart()
        chart.create(sample_test_statistics_data)

        # Points should have varying sizes based on p-value
        # Size is calculated from p-values if provided
        assert chart.figure.data[0].marker.size is not None

    def test_category_coloring(self, sample_test_statistics_data):
        """Test coloring by category."""
        chart = TestStatisticsChart()
        chart.create(sample_test_statistics_data)

        # Should have color mapping
        assert chart.figure.data[0].marker.color is not None

    def test_with_theme(self, sample_test_statistics_data):
        """Test with custom theme."""
        chart = TestStatisticsChart(theme=ACADEMIC_THEME)
        chart.create(sample_test_statistics_data)

        assert chart.figure is not None

    def test_validation_missing_statistics(self):
        """Test validation with missing statistics."""
        chart = TestStatisticsChart()

        with pytest.raises(ValueError, match="must contain 'statistics'"):
            chart.create({"test_names": ["Test1", "Test2"]})

    def test_validation_mismatched_lengths(self, sample_test_statistics_data):
        """Test validation with mismatched array lengths."""
        chart = TestStatisticsChart()

        bad_data = sample_test_statistics_data.copy()
        bad_data["statistics"] = [1.0, 2.0]  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            chart.create(bad_data)

    def test_to_html(self, sample_test_statistics_data):
        """Test HTML export."""
        chart = TestStatisticsChart()
        chart.create(sample_test_statistics_data)

        html = chart.to_html()

        assert isinstance(html, str)
        assert len(html) > 100


class TestTestComparisonHeatmap:
    """Tests for TestComparisonHeatmap."""

    def test_creation(self, sample_heatmap_data):
        """Test chart creation."""
        chart = TestComparisonHeatmap()
        chart.create(sample_heatmap_data)

        assert chart.figure is not None
        assert len(chart.figure.data) == 1  # Heatmap trace

    def test_annotations(self, sample_heatmap_data):
        """Test p-value annotations."""
        chart = TestComparisonHeatmap()
        chart.create(sample_heatmap_data)

        # Should have annotations for each cell (always shown)
        annotations = chart.figure.layout.annotations
        assert len(annotations) == 9  # 3 models * 3 tests

    def test_colorscale(self, sample_heatmap_data):
        """Test diverging colorscale."""
        chart = TestComparisonHeatmap()
        chart.create(sample_heatmap_data)

        # Should use RdYlGn or similar diverging scale
        colorscale = chart.figure.data[0].colorscale
        assert colorscale is not None

    def test_validation_missing_matrix(self):
        """Test validation with missing matrix."""
        chart = TestComparisonHeatmap()

        with pytest.raises(ValueError, match="must contain 'matrix'"):
            chart.create({"models": ["M1", "M2"], "tests": ["T1", "T2"]})

    def test_validation_matrix_shape(self, sample_heatmap_data):
        """Test validation of matrix shape."""
        chart = TestComparisonHeatmap()

        bad_data = sample_heatmap_data.copy()
        bad_data["matrix"] = [[0.1, 0.2]]  # Wrong shape

        with pytest.raises(ValueError, match="Number of matrix"):
            chart.create(bad_data)

    def test_to_html(self, sample_heatmap_data):
        """Test HTML export."""
        chart = TestComparisonHeatmap()
        chart.create(sample_heatmap_data)

        html = chart.to_html()

        assert isinstance(html, str)
        assert len(html) > 100


class TestValidationDashboard:
    """Tests for ValidationDashboard."""

    def test_creation(self, sample_dashboard_data):
        """Test dashboard creation."""
        chart = ValidationDashboard()
        chart.create(sample_dashboard_data)

        assert chart.figure is not None
        # Should have multiple traces from different subplots
        assert len(chart.figure.data) >= 4

    def test_subplot_layout(self, sample_dashboard_data):
        """Test 2x2 subplot layout."""
        chart = ValidationDashboard()
        chart.create(sample_dashboard_data)

        # Check for subplot structure
        layout = chart.figure.layout
        assert hasattr(layout, "xaxis")
        assert hasattr(layout, "xaxis2")
        assert hasattr(layout, "yaxis")
        assert hasattr(layout, "yaxis2")

    def test_summary_metrics(self, sample_dashboard_data):
        """Test summary metrics display."""
        chart = ValidationDashboard()
        chart.create(sample_dashboard_data)

        # Should display total_tests, passed, failed
        assert chart.figure is not None

    def test_validation_missing_sections(self):
        """Test validation with missing dashboard sections."""
        chart = ValidationDashboard()

        with pytest.raises(ValueError, match="must contain 'overview'"):
            chart.create({"summary": {"total_tests": 10}})

    def test_with_theme(self, sample_dashboard_data):
        """Test dashboard with theme."""
        chart = ValidationDashboard(theme=PROFESSIONAL_THEME)
        chart.create(sample_dashboard_data)

        assert chart.figure is not None

    def test_to_html(self, sample_dashboard_data):
        """Test HTML export."""
        chart = ValidationDashboard()
        chart.create(sample_dashboard_data)

        html = chart.to_html()

        assert isinstance(html, str)
        assert len(html) > 100
        # Dashboard HTML should be larger due to multiple charts
        assert len(html) > 500


class TestValidationChartsIntegration:
    """Integration tests for validation charts."""

    def test_all_charts_registered(self):
        """Test that all validation charts are registered."""
        from panelbox.visualization import ChartRegistry

        assert ChartRegistry.get("validation_test_overview") == TestOverviewChart
        assert ChartRegistry.get("validation_pvalue_distribution") == PValueDistributionChart
        assert ChartRegistry.get("validation_test_statistics") == TestStatisticsChart
        assert ChartRegistry.get("validation_comparison_heatmap") == TestComparisonHeatmap
        assert ChartRegistry.get("validation_dashboard") == ValidationDashboard

    def test_factory_creation(self, sample_test_overview_data):
        """Test creating charts via factory."""
        from panelbox.visualization import ChartFactory

        chart = ChartFactory.create(
            chart_type="validation_test_overview",
            data=sample_test_overview_data,
            theme="professional",
        )

        assert isinstance(chart, TestOverviewChart)
        assert chart.figure is not None

    def test_theme_application(self, sample_pvalue_data):
        """Test theme is properly applied."""
        from panelbox.visualization import ChartFactory

        chart = ChartFactory.create(
            chart_type="validation_pvalue_distribution",
            data=sample_pvalue_data,
            theme="academic",  # Use theme string, not object
        )

        assert chart.theme.name == "academic"

    def test_multiple_exports(self, sample_test_statistics_data):
        """Test multiple export formats."""
        chart = TestStatisticsChart()
        chart.create(sample_test_statistics_data)

        html = chart.to_html()
        json_str = chart.to_json()

        assert isinstance(html, str)
        assert isinstance(json_str, str)
        assert len(html) > 0
        assert len(json_str) > 0
