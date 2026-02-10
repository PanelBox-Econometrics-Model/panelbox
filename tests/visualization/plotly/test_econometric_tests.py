"""
Comprehensive tests for econometric test visualizations.
"""

import numpy as np
import pytest

from panelbox.visualization.plotly.econometric_tests import (
    ACFPACFPlot,
    CointegrationHeatmap,
    CrossSectionalDependencePlot,
    UnitRootTestPlot,
    calculate_acf,
    calculate_pacf,
    ljung_box_test,
)


class TestCalculateACF:
    """Test ACF calculation function."""

    def test_acf_basic(self):
        """Test basic ACF calculation."""
        residuals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        acf = calculate_acf(residuals, 3)
        assert len(acf) == 4
        assert acf[0] == 1.0

    def test_acf_flatten_input(self):
        """Test ACF with 2D input."""
        residuals = np.array([[1, 2, 3], [4, 5, 6]])
        acf = calculate_acf(residuals, 2)
        assert len(acf) == 3


class TestCalculatePACF:
    """Test PACF calculation function."""

    def test_pacf_basic(self):
        """Test basic PACF calculation."""
        residuals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        pacf = calculate_pacf(residuals, 3)
        assert len(pacf) == 4
        assert pacf[0] == 1.0

    def test_pacf_zero_lags(self):
        """Test PACF with max_lags=0."""
        residuals = np.array([1, 2, 3, 4, 5])
        pacf = calculate_pacf(residuals, 0)
        assert len(pacf) == 1
        assert pacf[0] == 1.0

    def test_pacf_one_lag(self):
        """Test PACF with max_lags=1."""
        residuals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        pacf = calculate_pacf(residuals, 1)
        assert len(pacf) == 2

    def test_pacf_singular_matrix(self):
        """Test PACF handles singular matrices."""
        residuals = np.ones(100)
        pacf = calculate_pacf(residuals, 5)
        assert len(pacf) == 6


class TestLjungBoxTest:
    """Test Ljung-Box test function."""

    def test_ljung_box_basic(self):
        """Test Ljung-Box test."""
        np.random.seed(42)
        residuals = np.random.randn(100)
        result = ljung_box_test(residuals, 10)
        assert "statistic" in result
        assert "pvalue" in result
        assert "df" in result


class TestACFPACFPlot:
    """Test ACF/PACF plot class."""

    def test_basic_plot(self):
        """Test basic ACF/PACF plot creation."""
        np.random.seed(42)
        residuals = np.random.randn(100)
        chart = ACFPACFPlot()
        data = {"residuals": residuals, "max_lags": 20}
        fig = chart.create(data)
        assert fig is not None

    def test_empty_residuals_error(self):
        """Test error on empty residuals."""
        chart = ACFPACFPlot()
        with pytest.raises(ValueError, match="Residuals cannot be empty"):
            chart.create({"residuals": []})

    def test_confidence_level_95(self):
        """Test with 95% confidence."""
        np.random.seed(42)
        chart = ACFPACFPlot()
        data = {"residuals": np.random.randn(100), "confidence_level": 0.95}
        fig = chart.create(data)
        assert fig is not None

    def test_confidence_level_99(self):
        """Test with 99% confidence."""
        np.random.seed(42)
        chart = ACFPACFPlot()
        data = {"residuals": np.random.randn(100), "confidence_level": 0.99}
        fig = chart.create(data)
        assert fig is not None

    def test_confidence_level_custom(self):
        """Test with custom confidence level."""
        np.random.seed(42)
        chart = ACFPACFPlot()
        data = {"residuals": np.random.randn(100), "confidence_level": 0.90}
        fig = chart.create(data)
        assert fig is not None

    def test_no_ljung_box(self):
        """Test without Ljung-Box test."""
        np.random.seed(42)
        chart = ACFPACFPlot()
        data = {"residuals": np.random.randn(100), "show_ljung_box": False}
        fig = chart.create(data)
        assert fig is not None

    def test_max_lags_auto(self):
        """Test automatic max_lags."""
        np.random.seed(42)
        chart = ACFPACFPlot()
        fig = chart.create({"residuals": np.random.randn(100)})
        assert fig is not None

    def test_custom_title(self):
        """Test custom title."""
        np.random.seed(42)
        chart = ACFPACFPlot()
        fig = chart.create({"residuals": np.random.randn(100)}, title="Custom Title")
        assert fig is not None


class TestUnitRootTestPlot:
    """Test unit root test visualization."""

    def test_basic_plot(self):
        """Test basic unit root test plot."""
        chart = UnitRootTestPlot()
        data = {
            "test_names": ["ADF", "PP"],
            "test_stats": [-3.5, -3.8],
            "critical_values": {"5%": -3.41},
            "pvalues": [0.008, 0.003],
        }
        fig = chart.create(data)
        assert fig is not None

    def test_empty_test_names_error(self):
        """Test error on empty test names."""
        chart = UnitRootTestPlot()
        with pytest.raises(ValueError, match="Test names cannot be empty"):
            chart.create({"test_names": [], "test_stats": [], "pvalues": []})

    def test_with_time_series(self):
        """Test with time series overlay."""
        chart = UnitRootTestPlot()
        data = {
            "test_names": ["ADF"],
            "test_stats": [-3.5],
            "pvalues": [0.01],
            "critical_values": {"5%": -3.41},
            "series": np.random.randn(100),
            "time_index": np.arange(100),
        }
        fig = chart.create(data)
        assert fig is not None

    def test_pvalue_levels(self):
        """Test p-value color coding."""
        chart = UnitRootTestPlot()
        data = {
            "test_names": ["T1", "T2", "T3", "T4"],
            "test_stats": [-4.0, -3.0, -2.5, -1.0],
            "pvalues": [0.005, 0.03, 0.07, 0.15],
            "critical_values": {"5%": -3.41},
        }
        fig = chart.create(data)
        assert fig is not None

    def test_custom_title(self):
        """Test custom title."""
        chart = UnitRootTestPlot()
        data = {
            "test_names": ["ADF"],
            "test_stats": [-3.5],
            "pvalues": [0.01],
            "critical_values": {},
        }
        fig = chart.create(data, title="Custom Title")
        assert fig is not None


class TestCointegrationHeatmap:
    """Test cointegration heatmap visualization."""

    def test_basic_heatmap(self):
        """Test basic cointegration heatmap."""
        chart = CointegrationHeatmap()
        data = {
            "variables": ["GDP", "Consumption"],
            "pvalues": [[1.0, 0.02], [0.02, 1.0]],
        }
        fig = chart.create(data)
        assert fig is not None

    def test_empty_variables_error(self):
        """Test error on empty variables."""
        chart = CointegrationHeatmap()
        with pytest.raises(ValueError, match="Variables cannot be empty"):
            chart.create({"variables": [], "pvalues": []})

    def test_with_test_statistics(self):
        """Test heatmap with test statistics."""
        chart = CointegrationHeatmap()
        data = {
            "variables": ["X", "Y"],
            "pvalues": [[1.0, 0.05], [0.05, 1.0]],
            "test_stats": [[0.0, -3.5], [-3.5, 0.0]],
        }
        fig = chart.create(data)
        assert fig is not None

    def test_custom_title(self):
        """Test custom title."""
        chart = CointegrationHeatmap()
        data = {"variables": ["X", "Y"], "pvalues": [[1.0, 0.05], [0.05, 1.0]]}
        fig = chart.create(data, title="Custom Title")
        assert fig is not None

    def test_default_test_name(self):
        """Test default test name."""
        chart = CointegrationHeatmap()
        data = {"variables": ["X", "Y"], "pvalues": [[1.0, 0.05], [0.05, 1.0]]}
        fig = chart.create(data)
        assert fig is not None


class TestCrossSectionalDependencePlot:
    """Test cross-sectional dependence visualization."""

    def test_basic_plot(self):
        """Test basic CD plot."""
        chart = CrossSectionalDependencePlot()
        data = {"cd_statistic": 5.23, "pvalue": 0.001}
        fig = chart.create(data)
        assert fig is not None

    def test_missing_cd_statistic_error(self):
        """Test error on missing CD statistic."""
        chart = CrossSectionalDependencePlot()
        with pytest.raises(ValueError, match="CD statistic is required"):
            chart.create({"pvalue": 0.05})

    def test_with_entity_correlations(self):
        """Test with entity correlations."""
        chart = CrossSectionalDependencePlot()
        data = {
            "cd_statistic": 5.23,
            "pvalue": 0.001,
            "entity_correlations": [0.3, 0.5, 0.6],
        }
        fig = chart.create(data)
        assert fig is not None

    def test_pvalue_none(self):
        """Test with None p-value."""
        chart = CrossSectionalDependencePlot()
        fig = chart.create({"cd_statistic": 3.5, "pvalue": None})
        assert fig is not None

    def test_pvalue_levels(self):
        """Test different p-value levels."""
        chart = CrossSectionalDependencePlot()
        for pval in [0.005, 0.03, 0.15]:
            fig = chart.create({"cd_statistic": 3.0, "pvalue": pval})
            assert fig is not None

    def test_custom_title(self):
        """Test custom title."""
        chart = CrossSectionalDependencePlot()
        fig = chart.create({"cd_statistic": 3.0, "pvalue": 0.05}, title="Custom Title")
        assert fig is not None

    def test_avg_correlation_none(self):
        """Test with None avg_correlation."""
        chart = CrossSectionalDependencePlot()
        fig = chart.create({"cd_statistic": 3.0, "pvalue": 0.05, "avg_correlation": None})
        assert fig is not None
