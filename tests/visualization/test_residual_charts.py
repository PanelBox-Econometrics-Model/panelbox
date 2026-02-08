"""
Tests for residual diagnostic chart implementations.

Tests all 7 residual chart types:
- QQPlot
- ResidualVsFittedPlot
- ScaleLocationPlot
- ResidualVsLeveragePlot
- ResidualTimeSeriesPlot
- ResidualDistributionPlot
- PartialRegressionPlot
"""

import pytest
import numpy as np

from panelbox.visualization.plotly.residuals import (
    QQPlot,
    ResidualVsFittedPlot,
    ScaleLocationPlot,
    ResidualVsLeveragePlot,
    ResidualTimeSeriesPlot,
    ResidualDistributionPlot,
    PartialRegressionPlot,
)
from panelbox.visualization.themes import PROFESSIONAL_THEME, ACADEMIC_THEME


@pytest.fixture
def sample_residuals():
    """Sample residuals for testing."""
    np.random.seed(42)
    return np.random.normal(0, 1, 100)


@pytest.fixture
def sample_fitted():
    """Sample fitted values for testing."""
    np.random.seed(42)
    return np.random.normal(5, 2, 100)


@pytest.fixture
def sample_leverage():
    """Sample leverage values for testing."""
    np.random.seed(42)
    # Leverage values should be between 0 and 1
    return np.random.beta(2, 5, 100)


@pytest.fixture
def sample_cooks_d():
    """Sample Cook's distance values."""
    np.random.seed(42)
    # Most values small, few large
    return np.random.exponential(0.1, 100)


@pytest.fixture
def sample_time_index():
    """Sample time index."""
    return np.arange(100)


class TestQQPlot:
    """Tests for QQPlot."""

    def test_creation(self, sample_residuals):
        """Test Q-Q plot creation."""
        chart = QQPlot()
        chart.create({'residuals': sample_residuals})

        assert chart.figure is not None
        assert len(chart.figure.data) >= 2  # Points + diagonal line

    def test_with_theme(self, sample_residuals):
        """Test Q-Q plot with theme."""
        chart = QQPlot(theme=PROFESSIONAL_THEME)
        chart.create({'residuals': sample_residuals})

        assert chart.figure is not None

    def test_standardized_residuals(self, sample_residuals):
        """Test with standardized residuals option."""
        chart = QQPlot()
        chart.create({
            'residuals': sample_residuals,
            'standardized': True
        })

        assert chart.figure is not None

    def test_confidence_bands(self, sample_residuals):
        """Test with confidence bands."""
        chart = QQPlot()
        chart.create({
            'residuals': sample_residuals,
            'show_confidence': True,
            'confidence_level': 0.95
        })

        assert chart.figure is not None
        # Should have more traces with confidence bands
        assert len(chart.figure.data) >= 4

    def test_no_confidence_bands(self, sample_residuals):
        """Test without confidence bands."""
        chart = QQPlot()
        chart.create({
            'residuals': sample_residuals,
            'show_confidence': False
        })

        assert chart.figure is not None
        # Should have only points and line
        assert len(chart.figure.data) == 2

    def test_validation_missing_residuals(self):
        """Test validation with missing residuals."""
        chart = QQPlot()

        with pytest.raises(ValueError, match="must contain 'residuals'"):
            chart.create({})

    def test_validation_empty_residuals(self):
        """Test validation with empty residuals."""
        chart = QQPlot()

        with pytest.raises(ValueError, match="cannot be empty"):
            chart.create({'residuals': []})

    def test_to_html(self, sample_residuals):
        """Test HTML export."""
        chart = QQPlot()
        chart.create({'residuals': sample_residuals})

        html = chart.to_html()

        assert isinstance(html, str)
        assert len(html) > 100

    def test_to_json(self, sample_residuals):
        """Test JSON export."""
        chart = QQPlot()
        chart.create({'residuals': sample_residuals})

        json_str = chart.to_json()

        assert isinstance(json_str, str)
        assert 'data' in json_str


class TestResidualVsFittedPlot:
    """Tests for ResidualVsFittedPlot."""

    def test_creation(self, sample_fitted, sample_residuals):
        """Test plot creation."""
        chart = ResidualVsFittedPlot()
        chart.create({
            'fitted': sample_fitted,
            'residuals': sample_residuals
        })

        assert chart.figure is not None
        assert len(chart.figure.data) >= 1

    def test_with_lowess(self, sample_fitted, sample_residuals):
        """Test with LOWESS smoothing."""
        chart = ResidualVsFittedPlot()
        chart.create({
            'fitted': sample_fitted,
            'residuals': sample_residuals,
            'add_lowess': True
        })

        assert chart.figure is not None

    def test_without_lowess(self, sample_fitted, sample_residuals):
        """Test without LOWESS smoothing."""
        chart = ResidualVsFittedPlot()
        chart.create({
            'fitted': sample_fitted,
            'residuals': sample_residuals,
            'add_lowess': False
        })

        assert chart.figure is not None

    def test_with_reference_line(self, sample_fitted, sample_residuals):
        """Test with reference line at y=0."""
        chart = ResidualVsFittedPlot()
        chart.create({
            'fitted': sample_fitted,
            'residuals': sample_residuals,
            'add_reference': True
        })

        assert chart.figure is not None

    def test_validation_missing_fitted(self, sample_residuals):
        """Test validation with missing fitted values."""
        chart = ResidualVsFittedPlot()

        with pytest.raises(ValueError, match="must contain 'fitted'"):
            chart.create({'residuals': sample_residuals})

    def test_validation_missing_residuals(self, sample_fitted):
        """Test validation with missing residuals."""
        chart = ResidualVsFittedPlot()

        with pytest.raises(ValueError, match="must contain 'residuals'"):
            chart.create({'fitted': sample_fitted})

    def test_validation_mismatched_lengths(self, sample_fitted):
        """Test validation with mismatched array lengths."""
        chart = ResidualVsFittedPlot()

        with pytest.raises(ValueError, match="same length"):
            chart.create({
                'fitted': sample_fitted,
                'residuals': np.random.normal(0, 1, 50)  # Different length
            })

    def test_to_html(self, sample_fitted, sample_residuals):
        """Test HTML export."""
        chart = ResidualVsFittedPlot()
        chart.create({
            'fitted': sample_fitted,
            'residuals': sample_residuals
        })

        html = chart.to_html()

        assert isinstance(html, str)
        assert len(html) > 100


class TestScaleLocationPlot:
    """Tests for ScaleLocationPlot."""

    def test_creation(self, sample_fitted, sample_residuals):
        """Test plot creation."""
        chart = ScaleLocationPlot()
        chart.create({
            'fitted': sample_fitted,
            'residuals': sample_residuals
        })

        assert chart.figure is not None
        assert len(chart.figure.data) >= 1

    def test_with_lowess(self, sample_fitted, sample_residuals):
        """Test with LOWESS smoothing."""
        chart = ScaleLocationPlot()
        chart.create({
            'fitted': sample_fitted,
            'residuals': sample_residuals,
            'add_lowess': True
        })

        assert chart.figure is not None

    def test_without_lowess(self, sample_fitted, sample_residuals):
        """Test without LOWESS smoothing."""
        chart = ScaleLocationPlot()
        chart.create({
            'fitted': sample_fitted,
            'residuals': sample_residuals,
            'add_lowess': False
        })

        assert chart.figure is not None

    def test_validation_missing_fields(self, sample_fitted):
        """Test validation with missing fields."""
        chart = ScaleLocationPlot()

        with pytest.raises(ValueError, match="must contain"):
            chart.create({'fitted': sample_fitted})

    def test_to_html(self, sample_fitted, sample_residuals):
        """Test HTML export."""
        chart = ScaleLocationPlot()
        chart.create({
            'fitted': sample_fitted,
            'residuals': sample_residuals
        })

        html = chart.to_html()

        assert isinstance(html, str)
        assert len(html) > 100


class TestResidualVsLeveragePlot:
    """Tests for ResidualVsLeveragePlot."""

    def test_creation(self, sample_residuals, sample_leverage):
        """Test plot creation."""
        chart = ResidualVsLeveragePlot()
        chart.create({
            'residuals': sample_residuals,
            'leverage': sample_leverage
        })

        assert chart.figure is not None
        assert len(chart.figure.data) >= 1

    def test_with_cooks_distance(self, sample_residuals, sample_leverage, sample_cooks_d):
        """Test with Cook's distance."""
        chart = ResidualVsLeveragePlot()
        chart.create({
            'residuals': sample_residuals,
            'leverage': sample_leverage,
            'cooks_d': sample_cooks_d
        })

        assert chart.figure is not None

    def test_with_contours(self, sample_residuals, sample_leverage, sample_cooks_d):
        """Test with Cook's distance contours."""
        chart = ResidualVsLeveragePlot()
        chart.create({
            'residuals': sample_residuals,
            'leverage': sample_leverage,
            'cooks_d': sample_cooks_d,
            'show_contours': True
        })

        assert chart.figure is not None

    def test_without_contours(self, sample_residuals, sample_leverage):
        """Test without contours."""
        chart = ResidualVsLeveragePlot()
        chart.create({
            'residuals': sample_residuals,
            'leverage': sample_leverage,
            'show_contours': False
        })

        assert chart.figure is not None

    def test_with_labels(self, sample_residuals, sample_leverage):
        """Test with observation labels."""
        labels = [f"Obs{i}" for i in range(len(sample_residuals))]

        chart = ResidualVsLeveragePlot()
        chart.create({
            'residuals': sample_residuals,
            'leverage': sample_leverage,
            'labels': labels
        })

        assert chart.figure is not None

    def test_validation_missing_residuals(self, sample_leverage):
        """Test validation with missing residuals."""
        chart = ResidualVsLeveragePlot()

        with pytest.raises(ValueError, match="must contain 'residuals'"):
            chart.create({'leverage': sample_leverage})

    def test_validation_missing_leverage(self, sample_residuals):
        """Test validation with missing leverage."""
        chart = ResidualVsLeveragePlot()

        with pytest.raises(ValueError, match="must contain 'leverage'"):
            chart.create({'residuals': sample_residuals})

    def test_to_html(self, sample_residuals, sample_leverage):
        """Test HTML export."""
        chart = ResidualVsLeveragePlot()
        chart.create({
            'residuals': sample_residuals,
            'leverage': sample_leverage
        })

        html = chart.to_html()

        assert isinstance(html, str)
        assert len(html) > 100


class TestResidualTimeSeriesPlot:
    """Tests for ResidualTimeSeriesPlot."""

    def test_creation(self, sample_residuals):
        """Test plot creation."""
        chart = ResidualTimeSeriesPlot()
        chart.create({'residuals': sample_residuals})

        assert chart.figure is not None
        assert len(chart.figure.data) >= 1

    def test_with_time_index(self, sample_residuals, sample_time_index):
        """Test with custom time index."""
        chart = ResidualTimeSeriesPlot()
        chart.create({
            'residuals': sample_residuals,
            'time_index': sample_time_index
        })

        assert chart.figure is not None

    def test_with_bands(self, sample_residuals):
        """Test with ±2σ bands."""
        chart = ResidualTimeSeriesPlot()
        chart.create({
            'residuals': sample_residuals,
            'add_bands': True
        })

        assert chart.figure is not None

    def test_without_bands(self, sample_residuals):
        """Test without bands."""
        chart = ResidualTimeSeriesPlot()
        chart.create({
            'residuals': sample_residuals,
            'add_bands': False
        })

        assert chart.figure is not None

    def test_validation_missing_residuals(self):
        """Test validation with missing residuals."""
        chart = ResidualTimeSeriesPlot()

        with pytest.raises(ValueError, match="must contain 'residuals'"):
            chart.create({})

    def test_to_html(self, sample_residuals):
        """Test HTML export."""
        chart = ResidualTimeSeriesPlot()
        chart.create({'residuals': sample_residuals})

        html = chart.to_html()

        assert isinstance(html, str)
        assert len(html) > 100


class TestResidualDistributionPlot:
    """Tests for ResidualDistributionPlot."""

    def test_creation(self, sample_residuals):
        """Test plot creation."""
        chart = ResidualDistributionPlot()
        chart.create({'residuals': sample_residuals})

        assert chart.figure is not None
        assert len(chart.figure.data) >= 1

    def test_with_kde(self, sample_residuals):
        """Test with KDE overlay."""
        chart = ResidualDistributionPlot()
        chart.create({
            'residuals': sample_residuals,
            'show_kde': True
        })

        assert chart.figure is not None

    def test_without_kde(self, sample_residuals):
        """Test without KDE overlay."""
        chart = ResidualDistributionPlot()
        chart.create({
            'residuals': sample_residuals,
            'show_kde': False
        })

        assert chart.figure is not None

    def test_with_normal_overlay(self, sample_residuals):
        """Test with normal distribution overlay."""
        chart = ResidualDistributionPlot()
        chart.create({
            'residuals': sample_residuals,
            'show_normal': True
        })

        assert chart.figure is not None

    def test_without_normal_overlay(self, sample_residuals):
        """Test without normal overlay."""
        chart = ResidualDistributionPlot()
        chart.create({
            'residuals': sample_residuals,
            'show_normal': False
        })

        assert chart.figure is not None

    def test_custom_bins(self, sample_residuals):
        """Test with custom number of bins."""
        chart = ResidualDistributionPlot()
        chart.create({
            'residuals': sample_residuals,
            'bins': 20
        })

        assert chart.figure is not None

    def test_auto_bins(self, sample_residuals):
        """Test with auto bins."""
        chart = ResidualDistributionPlot()
        chart.create({
            'residuals': sample_residuals,
            'bins': 'auto'
        })

        assert chart.figure is not None

    def test_validation_missing_residuals(self):
        """Test validation with missing residuals."""
        chart = ResidualDistributionPlot()

        with pytest.raises(ValueError, match="must contain 'residuals'"):
            chart.create({})

    def test_to_html(self, sample_residuals):
        """Test HTML export."""
        chart = ResidualDistributionPlot()
        chart.create({'residuals': sample_residuals})

        html = chart.to_html()

        assert isinstance(html, str)
        assert len(html) > 100


class TestPartialRegressionPlot:
    """Tests for PartialRegressionPlot."""

    def test_creation(self):
        """Test plot creation."""
        np.random.seed(42)
        x_resid = np.random.normal(0, 1, 100)
        y_resid = 0.5 * x_resid + np.random.normal(0, 0.5, 100)

        chart = PartialRegressionPlot()
        chart.create({
            'x_resid': x_resid,
            'y_resid': y_resid
        })

        assert chart.figure is not None
        assert len(chart.figure.data) >= 1

    def test_with_variable_name(self):
        """Test with custom variable name."""
        np.random.seed(42)
        x_resid = np.random.normal(0, 1, 100)
        y_resid = 0.5 * x_resid + np.random.normal(0, 0.5, 100)

        chart = PartialRegressionPlot()
        chart.create({
            'x_resid': x_resid,
            'y_resid': y_resid,
            'variable_name': 'GDP'
        })

        assert chart.figure is not None

    def test_with_regression_line(self):
        """Test with regression line."""
        np.random.seed(42)
        x_resid = np.random.normal(0, 1, 100)
        y_resid = 0.5 * x_resid + np.random.normal(0, 0.5, 100)

        chart = PartialRegressionPlot()
        chart.create({
            'x_resid': x_resid,
            'y_resid': y_resid,
            'add_regression_line': True
        })

        assert chart.figure is not None

    def test_without_regression_line(self):
        """Test without regression line."""
        np.random.seed(42)
        x_resid = np.random.normal(0, 1, 100)
        y_resid = 0.5 * x_resid + np.random.normal(0, 0.5, 100)

        chart = PartialRegressionPlot()
        chart.create({
            'x_resid': x_resid,
            'y_resid': y_resid,
            'add_regression_line': False
        })

        assert chart.figure is not None

    def test_with_confidence_bands(self):
        """Test with confidence bands."""
        np.random.seed(42)
        x_resid = np.random.normal(0, 1, 100)
        y_resid = 0.5 * x_resid + np.random.normal(0, 0.5, 100)

        chart = PartialRegressionPlot()
        chart.create({
            'x_resid': x_resid,
            'y_resid': y_resid,
            'add_regression_line': True,
            'add_confidence': True
        })

        assert chart.figure is not None

    def test_without_confidence_bands(self):
        """Test without confidence bands."""
        np.random.seed(42)
        x_resid = np.random.normal(0, 1, 100)
        y_resid = 0.5 * x_resid + np.random.normal(0, 0.5, 100)

        chart = PartialRegressionPlot()
        chart.create({
            'x_resid': x_resid,
            'y_resid': y_resid,
            'add_regression_line': True,
            'add_confidence': False
        })

        assert chart.figure is not None

    def test_validation_missing_x_resid(self):
        """Test validation with missing x residuals."""
        chart = PartialRegressionPlot()

        with pytest.raises(ValueError, match="must contain 'x_resid'"):
            chart.create({'y_resid': np.random.normal(0, 1, 100)})

    def test_validation_missing_y_resid(self):
        """Test validation with missing y residuals."""
        chart = PartialRegressionPlot()

        with pytest.raises(ValueError, match="must contain 'y_resid'"):
            chart.create({'x_resid': np.random.normal(0, 1, 100)})

    def test_to_html(self):
        """Test HTML export."""
        np.random.seed(42)
        x_resid = np.random.normal(0, 1, 100)
        y_resid = 0.5 * x_resid + np.random.normal(0, 0.5, 100)

        chart = PartialRegressionPlot()
        chart.create({
            'x_resid': x_resid,
            'y_resid': y_resid
        })

        html = chart.to_html()

        assert isinstance(html, str)
        assert len(html) > 100


class TestResidualChartsIntegration:
    """Integration tests for residual charts."""

    def test_all_charts_registered(self):
        """Test that all residual charts are registered."""
        from panelbox.visualization import ChartRegistry

        assert ChartRegistry.get('residual_qq_plot') == QQPlot
        assert ChartRegistry.get('residual_vs_fitted') == ResidualVsFittedPlot
        assert ChartRegistry.get('residual_scale_location') == ScaleLocationPlot
        assert ChartRegistry.get('residual_vs_leverage') == ResidualVsLeveragePlot
        assert ChartRegistry.get('residual_timeseries') == ResidualTimeSeriesPlot
        assert ChartRegistry.get('residual_distribution') == ResidualDistributionPlot
        assert ChartRegistry.get('residual_partial_regression') == PartialRegressionPlot

    def test_factory_creation(self, sample_residuals):
        """Test creating charts via factory."""
        from panelbox.visualization import ChartFactory

        chart = ChartFactory.create(
            chart_type='residual_qq_plot',
            data={'residuals': sample_residuals},
            theme='professional'
        )

        assert isinstance(chart, QQPlot)
        assert chart.figure is not None

    def test_theme_application(self, sample_residuals):
        """Test theme is properly applied."""
        from panelbox.visualization import ChartFactory

        chart = ChartFactory.create(
            chart_type='residual_distribution',
            data={'residuals': sample_residuals},
            theme='academic'
        )

        assert chart.theme.name == 'academic'

    def test_multiple_exports(self, sample_residuals):
        """Test multiple export formats."""
        chart = QQPlot()
        chart.create({'residuals': sample_residuals})

        html = chart.to_html()
        json_str = chart.to_json()

        assert isinstance(html, str)
        assert isinstance(json_str, str)
        assert len(html) > 0
        assert len(json_str) > 0
