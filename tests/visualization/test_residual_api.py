"""
Tests for high-level residual diagnostics API.

Tests the create_residual_diagnostics() function and its integration
with transformers and chart factories.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from panelbox.visualization.api import create_residual_diagnostics
from panelbox.visualization.plotly.residuals import (
    QQPlot,
    ResidualVsFittedPlot,
    ScaleLocationPlot,
    ResidualVsLeveragePlot,
    ResidualTimeSeriesPlot,
    ResidualDistributionPlot,
)


@pytest.fixture
def mock_results():
    """Mock model results for testing."""
    results = Mock(spec=[
        'resid', 'fittedvalues', 'params', 'df_model', 'df_resid', 'nobs',
        'rsquared', 'scale', 'model'
    ])

    np.random.seed(42)
    n = 100

    # Basic data
    results.resid = np.random.normal(0, 1, n)
    results.fittedvalues = np.random.normal(5, 2, n)

    # Model info
    results.params = Mock()
    results.params.index = ['const', 'x1', 'x2']
    results.params.__len__ = Mock(return_value=3)

    results.df_model = 2
    results.df_resid = n - 3
    results.nobs = n
    results.rsquared = 0.75
    results.scale = 1.0

    results.model = Mock(spec=['__class__'])
    results.model.__class__ = Mock()
    results.model.__class__.__name__ = 'FixedEffects'

    return results


class TestCreateResidualDiagnosticsBasic:
    """Basic tests for create_residual_diagnostics()."""

    def test_with_mock_results(self, mock_results):
        """Test with mock results object."""
        charts = create_residual_diagnostics(mock_results)

        assert isinstance(charts, dict)
        assert len(charts) > 0

    def test_default_charts(self, mock_results):
        """Test default chart selection."""
        charts = create_residual_diagnostics(mock_results)

        # Should create all 6 default charts (not partial regression)
        expected_charts = [
            'qq_plot',
            'residual_vs_fitted',
            'scale_location',
            'residual_vs_leverage',
            'residual_timeseries',
            'residual_distribution'
        ]

        for chart_name in expected_charts:
            assert chart_name in charts

    def test_specific_charts(self, mock_results):
        """Test requesting specific charts."""
        charts = create_residual_diagnostics(
            mock_results,
            charts=['qq_plot', 'residual_vs_fitted']
        )

        assert 'qq_plot' in charts
        assert 'residual_vs_fitted' in charts
        assert len(charts) == 2

    def test_single_chart(self, mock_results):
        """Test requesting single chart."""
        charts = create_residual_diagnostics(
            mock_results,
            charts=['qq_plot']
        )

        assert len(charts) == 1
        assert 'qq_plot' in charts


class TestCreateResidualDiagnosticsThemes:
    """Tests for theme handling."""

    def test_professional_theme(self, mock_results):
        """Test with professional theme."""
        charts = create_residual_diagnostics(
            mock_results,
            theme='professional',
            charts=['qq_plot']
        )

        assert len(charts) > 0

    def test_academic_theme(self, mock_results):
        """Test with academic theme."""
        charts = create_residual_diagnostics(
            mock_results,
            theme='academic',
            charts=['qq_plot']
        )

        assert len(charts) > 0

    def test_presentation_theme(self, mock_results):
        """Test with presentation theme."""
        charts = create_residual_diagnostics(
            mock_results,
            theme='presentation',
            charts=['qq_plot']
        )

        assert len(charts) > 0

    def test_no_theme(self, mock_results):
        """Test with no theme."""
        charts = create_residual_diagnostics(
            mock_results,
            theme=None,
            charts=['qq_plot']
        )

        assert len(charts) > 0


class TestCreateResidualDiagnosticsOutputFormats:
    """Tests for different output formats."""

    def test_chart_objects(self, mock_results):
        """Test returning chart objects (default)."""
        charts = create_residual_diagnostics(
            mock_results,
            charts=['qq_plot'],
            include_html=False
        )

        # Should return chart object
        assert isinstance(charts['qq_plot'], QQPlot)

    def test_html_strings(self, mock_results):
        """Test returning HTML strings."""
        charts = create_residual_diagnostics(
            mock_results,
            charts=['qq_plot'],
            include_html=True
        )

        # Should return HTML string
        assert isinstance(charts['qq_plot'], str)
        assert '<div' in charts['qq_plot']

    def test_html_validity(self, mock_results):
        """Test HTML output is valid."""
        charts = create_residual_diagnostics(
            mock_results,
            charts=['qq_plot', 'residual_vs_fitted'],
            include_html=True
        )

        for chart_name, html in charts.items():
            assert len(html) > 100
            assert isinstance(html, str)


class TestCreateResidualDiagnosticsOptions:
    """Tests for chart options."""

    def test_custom_config(self, mock_results):
        """Test custom chart configuration."""
        config = {
            'qq_plot': {
                'title': 'Custom Q-Q Plot',
                'width': 1200,
                'height': 900
            }
        }

        charts = create_residual_diagnostics(
            mock_results,
            charts=['qq_plot'],
            config=config
        )

        assert len(charts) > 0

    def test_multiple_configs(self, mock_results):
        """Test configs for multiple charts."""
        config = {
            'qq_plot': {'title': 'Q-Q Plot'},
            'residual_vs_fitted': {'title': 'Residuals'}
        }

        charts = create_residual_diagnostics(
            mock_results,
            charts=['qq_plot', 'residual_vs_fitted'],
            config=config
        )

        assert len(charts) == 2


class TestCreateResidualDiagnosticsIntegration:
    """Integration tests."""

    def test_end_to_end_all_charts(self, mock_results):
        """Test complete workflow with all charts."""
        charts = create_residual_diagnostics(
            mock_results,
            theme='professional'
        )

        # Should create multiple charts
        assert len(charts) >= 6

        # All should be chart objects by default
        for chart in charts.values():
            assert hasattr(chart, 'figure')

    def test_end_to_end_html_output(self, mock_results):
        """Test complete workflow with HTML output."""
        charts = create_residual_diagnostics(
            mock_results,
            theme='academic',
            include_html=True
        )

        # All should be HTML strings
        for html in charts.values():
            assert isinstance(html, str)
            assert len(html) > 100

    def test_all_chart_types(self, mock_results):
        """Test creating all available chart types."""
        all_charts = [
            'qq_plot',
            'residual_vs_fitted',
            'scale_location',
            'residual_vs_leverage',
            'residual_timeseries',
            'residual_distribution'
        ]

        charts = create_residual_diagnostics(
            mock_results,
            charts=all_charts
        )

        assert len(charts) == len(all_charts)
        for chart_name in all_charts:
            assert chart_name in charts

    def test_theme_applied_to_all(self, mock_results):
        """Test theme is applied to all charts."""
        charts = create_residual_diagnostics(
            mock_results,
            theme='professional',
            charts=['qq_plot', 'residual_vs_fitted']
        )

        for chart in charts.values():
            assert chart.theme.name == 'professional'


class TestCreateResidualDiagnosticsErrorHandling:
    """Tests for error handling."""

    def test_invalid_chart_name(self, mock_results):
        """Test with invalid chart name."""
        # Should not raise, just skip invalid charts
        charts = create_residual_diagnostics(
            mock_results,
            charts=['qq_plot', 'invalid_chart']
        )

        # Should still create valid charts
        assert 'qq_plot' in charts

    def test_missing_data_continues(self):
        """Test that missing data doesn't stop all chart creation."""
        results = Mock(spec=['resid', 'fittedvalues', 'params', 'df_model', 'scale'])
        results.resid = np.random.normal(0, 1, 100)
        results.fittedvalues = np.random.normal(5, 2, 100)
        results.params = Mock()
        results.params.index = ['const', 'x1', 'x2']
        results.params.__len__ = Mock(return_value=3)
        results.df_model = 2
        results.scale = 1.0

        # Should create what it can
        charts = create_residual_diagnostics(
            results,
            charts=['qq_plot', 'residual_vs_fitted']
        )

        # Should succeed for at least some charts
        assert len(charts) > 0

    def test_empty_chart_list(self, mock_results):
        """Test with empty chart list."""
        charts = create_residual_diagnostics(
            mock_results,
            charts=[]
        )

        # Should return empty dict
        assert len(charts) == 0


class TestCreateResidualDiagnosticsEdgeCases:
    """Edge case tests."""

    def test_very_small_sample(self):
        """Test with very small sample."""
        results = Mock(spec=['resid', 'fittedvalues', 'params', 'df_model', 'scale'])
        results.resid = np.array([0.5, -0.3, 0.8])
        results.fittedvalues = np.array([5.0, 4.5, 5.5])
        results.params = Mock()
        results.params.index = ['const', 'x1']
        results.params.__len__ = Mock(return_value=2)
        results.df_model = 1
        results.scale = 1.0

        charts = create_residual_diagnostics(
            results,
            charts=['qq_plot']
        )

        assert 'qq_plot' in charts

    def test_perfect_fit(self):
        """Test with perfect fit (zero residuals)."""
        results = Mock(spec=['resid', 'fittedvalues', 'params', 'df_model', 'scale'])
        results.resid = np.zeros(100)
        results.fittedvalues = np.random.normal(5, 2, 100)
        results.params = Mock()
        results.params.index = ['const', 'x1', 'x2']
        results.params.__len__ = Mock(return_value=3)
        results.df_model = 2
        results.scale = 0.0

        charts = create_residual_diagnostics(
            results,
            charts=['residual_distribution']
        )

        # With zero variance, KDE may fail - the function should handle gracefully
        # Either creates chart or warns and skips
        assert isinstance(charts, dict)

    def test_all_charts_with_minimal_data(self):
        """Test all charts with minimal data."""
        results = Mock(spec=['resid', 'fittedvalues', 'params', 'df_model', 'scale', 'model'])
        results.resid = np.random.normal(0, 1, 50)
        results.fittedvalues = np.random.normal(5, 2, 50)
        results.params = Mock()
        results.params.index = ['const', 'x1', 'x2']
        results.params.__len__ = Mock(return_value=3)
        results.df_model = 2
        results.scale = 1.0
        results.model = Mock(spec=['__class__'])
        results.model.__class__ = Mock()
        results.model.__class__.__name__ = 'TestModel'

        # Try to create all charts
        charts = create_residual_diagnostics(results)

        # Should create at least some charts
        assert len(charts) > 0


class TestCreateResidualDiagnosticsIndividualCharts:
    """Tests for individual chart creation."""

    def test_qq_plot_creation(self, mock_results):
        """Test Q-Q plot creation."""
        charts = create_residual_diagnostics(
            mock_results,
            charts=['qq_plot']
        )

        assert isinstance(charts['qq_plot'], QQPlot)

    def test_residual_vs_fitted_creation(self, mock_results):
        """Test residual vs fitted creation."""
        charts = create_residual_diagnostics(
            mock_results,
            charts=['residual_vs_fitted']
        )

        assert isinstance(charts['residual_vs_fitted'], ResidualVsFittedPlot)

    def test_scale_location_creation(self, mock_results):
        """Test scale-location creation."""
        charts = create_residual_diagnostics(
            mock_results,
            charts=['scale_location']
        )

        assert isinstance(charts['scale_location'], ScaleLocationPlot)

    def test_residual_vs_leverage_creation(self, mock_results):
        """Test residual vs leverage creation."""
        charts = create_residual_diagnostics(
            mock_results,
            charts=['residual_vs_leverage']
        )

        assert isinstance(charts['residual_vs_leverage'], ResidualVsLeveragePlot)

    def test_timeseries_creation(self, mock_results):
        """Test time series creation."""
        charts = create_residual_diagnostics(
            mock_results,
            charts=['residual_timeseries']
        )

        assert isinstance(charts['residual_timeseries'], ResidualTimeSeriesPlot)

    def test_distribution_creation(self, mock_results):
        """Test distribution creation."""
        charts = create_residual_diagnostics(
            mock_results,
            charts=['residual_distribution']
        )

        assert isinstance(charts['residual_distribution'], ResidualDistributionPlot)
