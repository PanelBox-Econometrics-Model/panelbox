"""
Tests for high-level validation chart API.

Tests the create_validation_charts() function and its integration
with transformers and chart factories.
"""

import pytest
from unittest.mock import Mock

from panelbox.visualization.api import create_validation_charts
from panelbox.visualization.plotly.validation import (
    TestOverviewChart,
    PValueDistributionChart,
    TestStatisticsChart,
    ValidationDashboard,
)


@pytest.fixture
def sample_validation_dict():
    """Sample validation data as dictionary."""
    return {
        "tests": [
            {
                "name": "Hausman Test",
                "category": "Specification",
                "statistic": 12.5,
                "pvalue": 0.006,
                "df": 2,
                "conclusion": "Use Fixed Effects",
                "passed": False,
                "alpha": 0.05,
            },
            {
                "name": "Wooldridge Test",
                "category": "Serial Correlation",
                "statistic": 3.4,
                "pvalue": 0.045,
                "df": 1,
                "conclusion": "Serial correlation detected",
                "passed": False,
                "alpha": 0.05,
            },
            {
                "name": "Breusch-Pagan",
                "category": "Heteroskedasticity",
                "statistic": 2.1,
                "pvalue": 0.234,
                "df": 3,
                "conclusion": "No heteroskedasticity",
                "passed": True,
                "alpha": 0.05,
            },
            {
                "name": "Pesaran CD",
                "category": "Cross-Sectional Dependence",
                "statistic": 0.8,
                "pvalue": 0.678,
                "df": 1,
                "conclusion": "No cross-sectional dependence",
                "passed": True,
                "alpha": 0.05,
            },
        ],
        "categories": {
            "Specification": [{"name": "Hausman Test", "passed": False}],
            "Serial Correlation": [{"name": "Wooldridge Test", "passed": False}],
            "Heteroskedasticity": [{"name": "Breusch-Pagan", "passed": True}],
            "Cross-Sectional Dependence": [{"name": "Pesaran CD", "passed": True}],
        },
        "summary": {"total_tests": 4, "passed": 2, "failed": 2, "pass_rate": 50.0},
        "model_info": {"estimator": "FixedEffects", "nobs": 1000},
    }


@pytest.fixture
def mock_validation_report():
    """Mock ValidationReport object."""
    report = Mock()

    # Specification tests
    spec_result = Mock()
    spec_result.statistic = 12.5
    spec_result.pvalue = 0.006
    spec_result.df = 2
    spec_result.conclusion = "Use Fixed Effects"
    spec_result.reject_null = True
    spec_result.alpha = 0.05
    spec_result.metadata = {}

    report.specification_tests = {"Hausman Test": spec_result}

    # Serial correlation tests
    serial_result = Mock()
    serial_result.statistic = 3.4
    serial_result.pvalue = 0.045
    serial_result.df = 1
    serial_result.conclusion = "Serial correlation detected"
    serial_result.reject_null = True
    serial_result.alpha = 0.05
    serial_result.metadata = {}

    report.serial_tests = {"Wooldridge Test": serial_result}

    # Heteroskedasticity tests
    het_result = Mock()
    het_result.statistic = 2.1
    het_result.pvalue = 0.234
    het_result.df = 3
    het_result.conclusion = "No heteroskedasticity"
    het_result.reject_null = False
    het_result.alpha = 0.05
    het_result.metadata = {}

    report.het_tests = {"Breusch-Pagan": het_result}

    # Cross-sectional dependence tests
    cd_result = Mock()
    cd_result.statistic = 0.8
    cd_result.pvalue = 0.678
    cd_result.df = 1
    cd_result.conclusion = "No cross-sectional dependence"
    cd_result.reject_null = False
    cd_result.alpha = 0.05
    cd_result.metadata = {}

    report.cd_tests = {"Pesaran CD": cd_result}

    # Model info
    report.model_info = {"estimator": "FixedEffects", "nobs": 1000}

    return report


class TestCreateValidationChartsBasic:
    """Basic tests for create_validation_charts()."""

    def test_with_dict_data(self, sample_validation_dict):
        """Test with dictionary data."""
        charts = create_validation_charts(sample_validation_dict)

        assert isinstance(charts, dict)
        assert len(charts) > 0

    def test_with_validation_report(self, mock_validation_report):
        """Test with ValidationReport object."""
        charts = create_validation_charts(mock_validation_report)

        assert isinstance(charts, dict)
        assert len(charts) > 0

    def test_default_charts(self, sample_validation_dict):
        """Test default chart selection."""
        charts = create_validation_charts(sample_validation_dict)

        # Should create default charts
        assert "test_overview" in charts
        assert "pvalue_distribution" in charts
        assert "test_statistics" in charts

    def test_specific_charts(self, sample_validation_dict):
        """Test requesting specific charts."""
        charts = create_validation_charts(
            sample_validation_dict, charts=["test_overview", "pvalue_distribution"]
        )

        assert "test_overview" in charts
        assert "pvalue_distribution" in charts
        assert "test_statistics" not in charts

    def test_single_chart(self, sample_validation_dict):
        """Test requesting single chart."""
        charts = create_validation_charts(sample_validation_dict, charts=["test_overview"])

        assert len(charts) == 1
        assert "test_overview" in charts


class TestCreateValidationChartsThemes:
    """Tests for theme handling."""

    def test_professional_theme_string(self, sample_validation_dict):
        """Test with professional theme string."""
        charts = create_validation_charts(sample_validation_dict, theme="professional")

        assert len(charts) > 0
        # Charts should be created successfully

    def test_academic_theme_string(self, sample_validation_dict):
        """Test with academic theme string."""
        charts = create_validation_charts(sample_validation_dict, theme="academic")

        assert len(charts) > 0

    def test_presentation_theme_string(self, sample_validation_dict):
        """Test with presentation theme string."""
        charts = create_validation_charts(sample_validation_dict, theme="presentation")

        assert len(charts) > 0

    def test_theme_object(self, sample_validation_dict):
        """Test with Theme object."""
        from panelbox.visualization.themes import PROFESSIONAL_THEME

        charts = create_validation_charts(sample_validation_dict, theme=PROFESSIONAL_THEME)

        assert len(charts) > 0

    def test_no_theme(self, sample_validation_dict):
        """Test with no theme."""
        charts = create_validation_charts(sample_validation_dict, theme=None)

        assert len(charts) > 0


class TestCreateValidationChartsOutputFormats:
    """Tests for different output formats."""

    def test_chart_objects(self, sample_validation_dict):
        """Test returning chart objects (default)."""
        charts = create_validation_charts(sample_validation_dict, include_html=False)

        # Should return chart objects
        assert isinstance(charts["test_overview"], TestOverviewChart)

    def test_html_strings(self, sample_validation_dict):
        """Test returning HTML strings."""
        charts = create_validation_charts(sample_validation_dict, include_html=True)

        # Should return HTML strings
        assert isinstance(charts["test_overview"], str)
        assert "<div" in charts["test_overview"]
        assert "plotly" in charts["test_overview"].lower()

    def test_html_validity(self, sample_validation_dict):
        """Test HTML output is valid."""
        charts = create_validation_charts(sample_validation_dict, include_html=True)

        for chart_name, html in charts.items():
            assert len(html) > 100
            assert "<div" in html or "<script" in html


class TestCreateValidationChartsOptions:
    """Tests for chart options."""

    def test_custom_alpha(self, sample_validation_dict):
        """Test custom alpha threshold."""
        charts = create_validation_charts(sample_validation_dict, alpha=0.01)

        assert len(charts) > 0

    def test_custom_config(self, sample_validation_dict):
        """Test custom chart configuration."""
        config = {"test_overview": {"width": 1200, "height": 800}}

        charts = create_validation_charts(sample_validation_dict, config=config)

        assert len(charts) > 0

    def test_interactive_true(self, sample_validation_dict):
        """Test interactive mode (default)."""
        charts = create_validation_charts(sample_validation_dict, interactive=True)

        # Should create Plotly charts
        assert isinstance(charts["test_overview"], TestOverviewChart)

    def test_interactive_false(self, sample_validation_dict):
        """Test non-interactive mode."""
        # Currently defaults to Plotly, but should handle gracefully
        charts = create_validation_charts(sample_validation_dict, interactive=False)

        # Should still create charts (Matplotlib support in future phases)
        assert len(charts) > 0


class TestCreateValidationChartsDashboard:
    """Tests for dashboard creation."""

    def test_dashboard_creation(self, sample_validation_dict):
        """Test creating dashboard."""
        charts = create_validation_charts(sample_validation_dict, charts=["dashboard"])

        assert "dashboard" in charts
        assert isinstance(charts["dashboard"], ValidationDashboard)

    def test_auto_dashboard_large_report(self, sample_validation_dict):
        """Test automatic dashboard for large reports."""
        # Add more tests to trigger dashboard
        large_data = sample_validation_dict.copy()
        large_data["tests"] = sample_validation_dict["tests"] * 2  # 8 tests

        charts = create_validation_charts(large_data)

        # Should include dashboard for reports with >5 tests
        assert "dashboard" in charts

    def test_no_dashboard_small_report(self, sample_validation_dict):
        """Test no auto-dashboard for small reports."""
        # Use minimal data
        small_data = sample_validation_dict.copy()
        small_data["tests"] = sample_validation_dict["tests"][:2]  # 2 tests

        charts = create_validation_charts(small_data)

        # Should not include dashboard automatically
        assert "dashboard" not in charts


class TestCreateValidationChartsIntegration:
    """Integration tests."""

    def test_end_to_end_dict(self, sample_validation_dict):
        """Test complete workflow with dict data."""
        charts = create_validation_charts(
            sample_validation_dict, theme="professional", charts=["test_overview", "pvalue_distribution"], include_html=True
        )

        # Verify all requested charts created as HTML
        assert "test_overview" in charts
        assert "pvalue_distribution" in charts
        assert isinstance(charts["test_overview"], str)
        assert isinstance(charts["pvalue_distribution"], str)

    def test_end_to_end_validation_report(self, mock_validation_report):
        """Test complete workflow with ValidationReport."""
        charts = create_validation_charts(
            mock_validation_report, theme="academic", charts=["test_overview", "test_statistics"], include_html=False
        )

        # Verify chart objects created
        assert "test_overview" in charts
        assert "test_statistics" in charts
        assert isinstance(charts["test_overview"], TestOverviewChart)
        assert isinstance(charts["test_statistics"], TestStatisticsChart)

    def test_all_chart_types(self, sample_validation_dict):
        """Test creating all chart types."""
        charts = create_validation_charts(
            sample_validation_dict,
            charts=["test_overview", "pvalue_distribution", "test_statistics", "dashboard"],
        )

        assert len(charts) == 4
        assert "test_overview" in charts
        assert "pvalue_distribution" in charts
        assert "test_statistics" in charts
        assert "dashboard" in charts

    def test_multiple_themes(self, sample_validation_dict):
        """Test creating charts with different themes."""
        themes = ["professional", "academic", "presentation"]

        for theme in themes:
            charts = create_validation_charts(sample_validation_dict, theme=theme, charts=["test_overview"])

            assert "test_overview" in charts
            assert charts["test_overview"].figure is not None


class TestCreateValidationChartsEdgeCases:
    """Edge case tests."""

    def test_empty_tests(self):
        """Test with no tests."""
        empty_data = {
            "tests": [],
            "categories": {},
            "summary": {"total_tests": 0, "passed": 0, "failed": 0, "pass_rate": 0.0},
            "model_info": {},
        }

        charts = create_validation_charts(empty_data, charts=["test_overview"])

        # Should handle gracefully
        assert "test_overview" in charts

    def test_single_test(self):
        """Test with single test."""
        single_test_data = {
            "tests": [
                {
                    "name": "Test1",
                    "category": "Specification",
                    "statistic": 5.0,
                    "pvalue": 0.05,
                    "passed": False,
                    "alpha": 0.05,
                }
            ],
            "categories": {"Specification": [{"name": "Test1", "passed": False}]},
            "summary": {"total_tests": 1, "passed": 0, "failed": 1, "pass_rate": 0.0},
            "model_info": {},
        }

        charts = create_validation_charts(single_test_data)

        assert len(charts) > 0

    def test_all_tests_passed(self):
        """Test with all tests passing."""
        all_passed_data = {
            "tests": [
                {
                    "name": "Test1",
                    "category": "Specification",
                    "statistic": 1.0,
                    "pvalue": 0.5,
                    "passed": True,
                    "alpha": 0.05,
                },
                {
                    "name": "Test2",
                    "category": "Serial Correlation",
                    "statistic": 0.5,
                    "pvalue": 0.8,
                    "passed": True,
                    "alpha": 0.05,
                },
            ],
            "categories": {
                "Specification": [{"name": "Test1", "passed": True}],
                "Serial Correlation": [{"name": "Test2", "passed": True}],
            },
            "summary": {"total_tests": 2, "passed": 2, "failed": 0, "pass_rate": 100.0},
            "model_info": {},
        }

        charts = create_validation_charts(all_passed_data)

        assert len(charts) > 0

    def test_all_tests_failed(self):
        """Test with all tests failing."""
        all_failed_data = {
            "tests": [
                {
                    "name": "Test1",
                    "category": "Specification",
                    "statistic": 50.0,
                    "pvalue": 0.001,
                    "passed": False,
                    "alpha": 0.05,
                },
                {
                    "name": "Test2",
                    "category": "Serial Correlation",
                    "statistic": 30.0,
                    "pvalue": 0.002,
                    "passed": False,
                    "alpha": 0.05,
                },
            ],
            "categories": {
                "Specification": [{"name": "Test1", "passed": False}],
                "Serial Correlation": [{"name": "Test2", "passed": False}],
            },
            "summary": {"total_tests": 2, "passed": 0, "failed": 2, "pass_rate": 0.0},
            "model_info": {},
        }

        charts = create_validation_charts(all_failed_data)

        assert len(charts) > 0
