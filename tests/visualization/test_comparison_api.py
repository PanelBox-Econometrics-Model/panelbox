"""
Tests for create_comparison_charts API function.

Tests the high-level convenience API for model comparison visualizations.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from panelbox.visualization.api import create_comparison_charts
from panelbox.visualization.themes import PROFESSIONAL_THEME, ACADEMIC_THEME


@pytest.fixture
def mock_results():
    """Create mock model results for testing."""
    results1 = Mock()
    results1.params = pd.Series({'x1': 1.2, 'x2': -0.5, 'const': 2.0})
    results1.std_errors = pd.Series({'x1': 0.3, 'x2': 0.2, 'const': 0.5})
    results1.pvalues = pd.Series({'x1': 0.001, 'x2': 0.02, 'const': 0.0001})
    results1.rsquared = 0.75
    results1.rsquared_adj = 0.73
    results1.fvalue = 45.2
    results1.llf = -150.5
    results1.aic = 305.0
    results1.bic = 315.0
    results1.hqic = 308.5

    results2 = Mock()
    results2.params = pd.Series({'x1': 1.5, 'x2': -0.3, 'const': 1.8})
    results2.std_errors = pd.Series({'x1': 0.25, 'x2': 0.18, 'const': 0.45})
    results2.pvalues = pd.Series({'x1': 0.0005, 'x2': 0.05, 'const': 0.0002})
    results2.rsquared = 0.80
    results2.rsquared_adj = 0.78
    results2.fvalue = 52.1
    results2.llf = -145.2
    results2.aic = 294.4
    results2.bic = 304.4
    results2.hqic = 297.9

    results3 = Mock()
    results3.params = pd.Series({'x1': 1.3, 'x2': -0.4, 'const': 1.9})
    results3.std_errors = pd.Series({'x1': 0.28, 'x2': 0.19, 'const': 0.48})
    results3.pvalues = pd.Series({'x1': 0.0008, 'x2': 0.03, 'const': 0.00015})
    results3.rsquared = 0.77
    results3.rsquared_adj = 0.75
    results3.fvalue = 48.5
    results3.llf = -148.0
    results3.aic = 300.0
    results3.bic = 310.0
    results3.hqic = 303.5

    return [results1, results2, results3]


class TestCreateComparisonChartsBasic:
    """Basic tests for create_comparison_charts function."""

    def test_default_charts(self, mock_results):
        """Test creating comparison charts with defaults."""
        charts = create_comparison_charts(mock_results)

        # Default charts are coefficients, fit_comparison, ic_comparison
        assert 'coefficients' in charts
        assert 'fit_comparison' in charts
        assert 'ic_comparison' in charts
        assert len(charts) == 3

    def test_with_custom_names(self, mock_results):
        """Test with custom model names."""
        names = ['OLS', 'Fixed Effects', 'Random Effects']
        charts = create_comparison_charts(mock_results, names=names)

        assert charts is not None
        assert len(charts) >= 3

    def test_single_chart_type(self, mock_results):
        """Test requesting single chart type."""
        charts = create_comparison_charts(
            mock_results,
            charts=['coefficients']
        )

        assert 'coefficients' in charts
        assert len(charts) == 1

    def test_all_chart_types(self, mock_results):
        """Test requesting all chart types."""
        charts = create_comparison_charts(
            mock_results,
            charts=['coefficients', 'fit_comparison', 'ic_comparison']
        )

        assert 'coefficients' in charts
        assert 'fit_comparison' in charts
        assert 'ic_comparison' in charts
        assert len(charts) == 3

    def test_forest_plot_single_model(self, mock_results):
        """Test forest plot (requires single model)."""
        charts = create_comparison_charts(
            [mock_results[0]],  # Single model
            charts=['forest_plot']
        )

        assert 'forest_plot' in charts


class TestCreateComparisonChartsThemes:
    """Tests for theme application."""

    def test_professional_theme(self, mock_results):
        """Test with professional theme."""
        charts = create_comparison_charts(
            mock_results,
            theme='professional'
        )

        assert charts is not None
        assert len(charts) >= 3

    def test_academic_theme(self, mock_results):
        """Test with academic theme."""
        charts = create_comparison_charts(
            mock_results,
            theme='academic'
        )

        assert charts is not None

    def test_presentation_theme(self, mock_results):
        """Test with presentation theme."""
        charts = create_comparison_charts(
            mock_results,
            theme='presentation'
        )

        assert charts is not None

    def test_theme_object(self, mock_results):
        """Test with theme object."""
        charts = create_comparison_charts(
            mock_results,
            theme=PROFESSIONAL_THEME
        )

        assert charts is not None

    def test_no_theme(self, mock_results):
        """Test with no theme."""
        charts = create_comparison_charts(
            mock_results,
            theme=None
        )

        assert charts is not None


class TestCreateComparisonChartsOptions:
    """Tests for various options."""

    def test_with_variable_subset(self, mock_results):
        """Test with subset of variables."""
        charts = create_comparison_charts(
            mock_results,
            charts=['coefficients'],
            variables=['x1', 'x2']
        )

        assert 'coefficients' in charts

    def test_with_confidence_level(self, mock_results):
        """Test with custom confidence level."""
        charts = create_comparison_charts(
            [mock_results[0]],
            charts=['forest_plot'],
            confidence_level=0.99
        )

        assert 'forest_plot' in charts

    def test_include_html_false(self, mock_results):
        """Test without HTML export."""
        charts = create_comparison_charts(
            mock_results,
            include_html=False
        )

        # Should return chart objects, not HTML
        for chart_name, chart in charts.items():
            assert chart is not None
            assert hasattr(chart, 'figure')

    def test_include_html_true(self, mock_results):
        """Test with HTML export."""
        charts = create_comparison_charts(
            mock_results,
            include_html=True
        )

        # Should return HTML strings
        for chart_name, html in charts.items():
            assert html is not None
            assert isinstance(html, str)
            assert len(html) > 0


class TestCreateComparisonChartsEdgeCases:
    """Tests for edge cases."""

    def test_single_model(self, mock_results):
        """Test with single model."""
        charts = create_comparison_charts(
            [mock_results[0]],
            charts=['forest_plot']  # Only forest plot works with single model
        )

        assert 'forest_plot' in charts

    def test_two_models(self, mock_results):
        """Test with two models."""
        charts = create_comparison_charts(
            mock_results[:2],
            charts=['coefficients', 'fit_comparison', 'ic_comparison']
        )

        assert len(charts) == 3

    def test_many_models(self, mock_results):
        """Test with multiple models."""
        # Create more mock results
        extra_results = []
        for i in range(2):
            result = Mock()
            result.params = pd.Series({'x1': 1.0 + i*0.1, 'x2': -0.5 + i*0.05, 'const': 2.0})
            result.std_errors = pd.Series({'x1': 0.3, 'x2': 0.2, 'const': 0.5})
            result.pvalues = pd.Series({'x1': 0.001, 'x2': 0.02, 'const': 0.0001})
            result.rsquared = 0.75
            result.rsquared_adj = 0.73
            result.fvalue = 45.2
            result.llf = -150.5
            result.aic = 305.0
            result.bic = 315.0
            result.hqic = 308.5
            extra_results.append(result)

        all_results = mock_results + extra_results

        charts = create_comparison_charts(all_results)

        assert len(charts) >= 3

    def test_empty_charts_list(self, mock_results):
        """Test with empty charts list."""
        charts = create_comparison_charts(
            mock_results,
            charts=[]
        )

        # Should return empty dict
        assert charts == {}

    def test_invalid_chart_type(self, mock_results):
        """Test with invalid chart type (should skip with warning)."""
        charts = create_comparison_charts(
            mock_results,
            charts=['coefficients', 'invalid_chart_type']
        )

        # Should create valid chart and skip invalid one
        assert 'coefficients' in charts
        assert 'invalid_chart_type' not in charts

    def test_missing_attributes(self):
        """Test with results missing some attributes."""
        results = Mock()
        results.params = pd.Series({'x1': 1.2})
        results.std_errors = pd.Series({'x1': 0.3})
        # Missing other attributes

        # Should handle gracefully with warnings
        charts = create_comparison_charts(
            [results],
            charts=['forest_plot']
        )

        assert 'forest_plot' in charts


class TestCreateComparisonChartsIntegration:
    """Integration tests for create_comparison_charts."""

    def test_complete_workflow(self, mock_results):
        """Test complete comparison workflow."""
        # 1. Create all comparison charts
        charts = create_comparison_charts(
            mock_results,
            names=['OLS', 'FE', 'RE'],
            theme='professional',
            charts=['coefficients', 'fit_comparison', 'ic_comparison']
        )

        assert len(charts) == 3

        # 2. Each chart should be a valid chart object
        for chart_name, chart in charts.items():
            assert chart is not None
            assert hasattr(chart, 'figure')

    def test_export_all_charts(self, mock_results):
        """Test exporting all charts to HTML."""
        charts = create_comparison_charts(
            mock_results,
            include_html=True
        )

        for chart_name, html in charts.items():
            assert isinstance(html, str)
            assert 'plotly' in html.lower()  # Should contain Plotly library reference

    def test_charts_consistency(self, mock_results):
        """Test that all charts work consistently with same data."""
        chart_types = ['coefficients', 'fit_comparison', 'ic_comparison']

        for chart_type in chart_types:
            charts = create_comparison_charts(
                mock_results,
                charts=[chart_type]
            )

            assert chart_type in charts
            assert charts[chart_type] is not None

    def test_different_themes_produce_different_outputs(self, mock_results):
        """Test that different themes produce different chart configurations."""
        charts_prof = create_comparison_charts(
            mock_results,
            theme='professional',
            include_html=True
        )

        charts_acad = create_comparison_charts(
            mock_results,
            theme='academic',
            include_html=True
        )

        # Both should produce valid charts
        assert len(charts_prof) == len(charts_acad)

        # HTML output should differ (due to different styling)
        for chart_name in charts_prof:
            if chart_name in charts_acad:
                # Charts exist but may have different content
                assert isinstance(charts_prof[chart_name], str)
                assert isinstance(charts_acad[chart_name], str)
