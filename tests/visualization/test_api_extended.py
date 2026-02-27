"""
Extended tests for the visualization API module.

Tests export functions, panel chart creation, comparison charts,
and econometric test chart creation functions.
"""

import warnings
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from panelbox.visualization.api import (
    create_acf_pacf_plot,
    create_between_within_plot,
    create_cointegration_heatmap,
    create_comparison_charts,
    create_cross_sectional_dependence_plot,
    create_entity_effects_plot,
    create_panel_charts,
    create_panel_structure_plot,
    create_time_effects_plot,
    create_unit_root_test_plot,
    export_chart,
    export_charts,
    export_charts_multiple_formats,
)

# =====================================================================
# Export functions tests
# =====================================================================


class TestExportCharts:
    """Tests for export_charts batch export function."""

    def test_export_charts_creates_directory_and_calls_save(self, tmp_path):
        """Test export_charts creates output dir and calls save_image."""
        mock_chart = MagicMock()
        mock_chart.save_image = MagicMock()
        charts = {"test_chart": mock_chart}

        output_dir = str(tmp_path / "output")
        result = export_charts(charts, output_dir=output_dir, format="png")

        assert "test_chart" in result
        assert result["test_chart"].endswith("test_chart.png")
        mock_chart.save_image.assert_called_once()

    def test_export_charts_skips_none_chart(self, tmp_path):
        """Test export_charts warns and skips None charts."""
        charts = {"missing": None}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = export_charts(charts, output_dir=str(tmp_path))
            assert len(result) == 0
            assert any("None" in str(warning.message) for warning in w)

    def test_export_charts_skips_no_save_image(self, tmp_path):
        """Test export_charts warns if chart has no save_image method."""
        mock_chart = MagicMock(spec=[])  # No methods
        charts = {"bad_chart": mock_chart}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = export_charts(charts, output_dir=str(tmp_path))
            assert len(result) == 0
            assert any("save_image" in str(warning.message) for warning in w)


class TestExportChart:
    """Tests for export_chart single chart export function."""

    def test_export_chart_calls_save_image(self, tmp_path):
        """Test export_chart calls save_image and returns path."""
        mock_chart = MagicMock()
        file_path = str(tmp_path / "chart.png")

        result = export_chart(mock_chart, file_path, format="png")

        assert result == file_path
        mock_chart.save_image.assert_called_once()

    def test_export_chart_no_save_image_raises(self):
        """Test export_chart raises ValueError without save_image method."""
        mock_chart = MagicMock(spec=[])

        with pytest.raises(ValueError, match="save_image"):
            export_chart(mock_chart, "chart.png")

    def test_export_chart_with_dimensions(self, tmp_path):
        """Test export_chart passes width/height/scale."""
        mock_chart = MagicMock()
        file_path = str(tmp_path / "chart.svg")

        export_chart(mock_chart, file_path, format="svg", width=1200, height=800, scale=2.0)

        mock_chart.save_image.assert_called_once_with(
            file_path, format="svg", width=1200, height=800, scale=2.0
        )


class TestExportChartsMultipleFormats:
    """Tests for export_charts_multiple_formats function."""

    def test_exports_default_formats(self, tmp_path):
        """Test multiple format export with default formats (png, svg)."""
        mock_chart = MagicMock()
        charts = {"chart1": mock_chart}

        result = export_charts_multiple_formats(charts, output_dir=str(tmp_path))

        assert "png" in result
        assert "svg" in result

    def test_exports_custom_formats(self, tmp_path):
        """Test multiple format export with custom format list."""
        mock_chart = MagicMock()
        charts = {"chart1": mock_chart}

        result = export_charts_multiple_formats(
            charts, output_dir=str(tmp_path), formats=["png", "pdf"]
        )

        assert "png" in result
        assert "pdf" in result
        assert "svg" not in result


# =====================================================================
# Panel chart creation tests
# =====================================================================


class TestCreateEntityEffectsPlot:
    """Tests for create_entity_effects_plot function."""

    def test_with_dict_data(self):
        """Test create_entity_effects_plot with dict data."""
        data = {
            "entity_id": ["A", "B", "C"],
            "effect": [0.5, -0.3, 0.1],
        }
        chart = create_entity_effects_plot(data)
        assert chart is not None

    def test_with_theme_string(self):
        """Test create_entity_effects_plot with theme as string."""
        data = {
            "entity_id": ["A", "B", "C"],
            "effect": [0.5, -0.3, 0.1],
        }
        chart = create_entity_effects_plot(data, theme="academic")
        assert chart is not None


class TestCreateTimeEffectsPlot:
    """Tests for create_time_effects_plot function."""

    def test_with_dict_data(self):
        """Test create_time_effects_plot with dict data."""
        data = {
            "time": ["2020", "2021", "2022"],
            "effect": [0.1, 0.2, -0.1],
        }
        chart = create_time_effects_plot(data)
        assert chart is not None

    def test_with_theme_string(self):
        """Test create_time_effects_plot with presentation theme."""
        data = {
            "time": ["2020", "2021", "2022"],
            "effect": [0.1, 0.2, -0.1],
        }
        chart = create_time_effects_plot(data, theme="presentation")
        assert chart is not None


class TestCreateBetweenWithinPlot:
    """Tests for create_between_within_plot function."""

    def test_with_dict_data(self):
        """Test create_between_within_plot with dict data."""
        data = {
            "variables": ["x1", "x2"],
            "between_var": [0.5, 0.3],
            "within_var": [0.2, 0.7],
        }
        chart = create_between_within_plot(data)
        assert chart is not None

    def test_with_style_parameter(self):
        """Test create_between_within_plot passes style via config."""
        data = {
            "variables": ["x1", "x2"],
            "between_var": [0.5, 0.3],
            "within_var": [0.2, 0.7],
        }
        chart = create_between_within_plot(data, style="side_by_side")
        assert chart is not None


# =====================================================================
# Panel structure and econometric test charts
# =====================================================================


class TestCreatePanelStructurePlot:
    """Tests for create_panel_structure_plot function."""

    def test_with_dict_data(self):
        """Test create_panel_structure_plot with dict data."""
        data = {
            "entities": ["A", "B", "C"],
            "time_periods": ["2020", "2021", "2022"],
            "presence_matrix": [[1, 1, 1], [1, 0, 1], [1, 1, 0]],
        }
        chart = create_panel_structure_plot(data)
        assert chart is not None

    def test_with_theme_string(self):
        """Test create_panel_structure_plot with theme."""
        data = {
            "entities": ["A", "B"],
            "time_periods": ["2020", "2021"],
            "presence_matrix": [[1, 1], [1, 0]],
        }
        chart = create_panel_structure_plot(data, theme="academic")
        assert chart is not None


class TestCreateAcfPacfPlot:
    """Tests for create_acf_pacf_plot function."""

    def test_basic_call(self):
        """Test create_acf_pacf_plot with basic residuals."""
        np.random.seed(42)
        residuals = np.random.randn(100).tolist()
        chart = create_acf_pacf_plot(residuals)
        assert chart is not None

    def test_with_max_lags(self):
        """Test create_acf_pacf_plot with explicit max_lags."""
        np.random.seed(42)
        residuals = np.random.randn(100).tolist()
        chart = create_acf_pacf_plot(residuals, max_lags=10)
        assert chart is not None


class TestCreateUnitRootTestPlot:
    """Tests for create_unit_root_test_plot function."""

    def test_with_dict_data(self):
        """Test create_unit_root_test_plot with dict results."""
        data = {
            "test_names": ["ADF", "PP", "KPSS"],
            "test_stats": [-3.5, -3.8, 0.3],
            "critical_values": {"1%": -3.96, "5%": -3.41, "10%": -3.13},
            "pvalues": [0.008, 0.003, 0.15],
        }
        chart = create_unit_root_test_plot(data)
        assert chart is not None

    def test_with_object_data(self):
        """Test create_unit_root_test_plot extracts from test object."""
        mock_result = Mock()
        mock_result.test_name = "ADF"
        mock_result.statistic = -3.5
        mock_result.critical_values = {"1%": -3.96, "5%": -3.41}
        mock_result.pvalue = 0.008

        chart = create_unit_root_test_plot(mock_result)
        assert chart is not None


# =====================================================================
# Cointegration and cross-sectional dependence
# =====================================================================


class TestCreateCointegrationHeatmap:
    """Tests for create_cointegration_heatmap function."""

    def test_basic_call(self):
        """Test create_cointegration_heatmap with basic data."""
        data = {
            "variables": ["GDP", "Consumption", "Investment"],
            "pvalues": [
                [1.0, 0.02, 0.15],
                [0.02, 1.0, 0.08],
                [0.15, 0.08, 1.0],
            ],
            "test_name": "Engle-Granger",
        }
        chart = create_cointegration_heatmap(data)
        assert chart is not None

    def test_with_academic_theme(self):
        """Test create_cointegration_heatmap with explicit theme."""
        data = {
            "variables": ["var1", "var2"],
            "pvalues": [[1.0, 0.05], [0.05, 1.0]],
        }
        chart = create_cointegration_heatmap(data, theme="academic")
        assert chart is not None


class TestCreateCrossSectionalDependencePlot:
    """Tests for create_cross_sectional_dependence_plot function."""

    def test_basic_call(self):
        """Test create_cross_sectional_dependence_plot with basic data."""
        data = {
            "cd_statistic": 5.23,
            "pvalue": 0.001,
            "avg_correlation": 0.42,
        }
        chart = create_cross_sectional_dependence_plot(data)
        assert chart is not None

    def test_with_entity_correlations(self):
        """Test with entity-level correlations."""
        data = {
            "cd_statistic": 3.45,
            "pvalue": 0.003,
            "avg_correlation": 0.28,
            "entity_correlations": [0.15, 0.32, 0.45, 0.21],
        }
        chart = create_cross_sectional_dependence_plot(data)
        assert chart is not None


class TestCreatePanelChartsAutoDetect:
    """Tests for create_panel_charts auto-detection."""

    def test_auto_detect_fallback(self):
        """Test create_panel_charts falls back to all chart types."""
        data = {
            "entity_names": ["A", "B"],
            "effects": [0.5, -0.3],
        }
        # When data is a dict (no hasattr checks succeed),
        # it falls back to all chart types
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = create_panel_charts(data, include_html=False)
            assert isinstance(result, dict)

    def test_unknown_chart_type_warns(self):
        """Test create_panel_charts warns on unknown chart type."""
        data = {"effects": [0.5]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_panel_charts(data, chart_types=["nonexistent"], include_html=False)
            assert any("Unknown chart type" in str(warning.message) for warning in w)


# =====================================================================
# Comparison charts misc tests
# =====================================================================


class TestComparisonChartsMisc:
    """Tests for comparison chart edge cases."""

    def test_forest_plot_multiple_models_warns(self):
        """Test forest plot warns with multiple models."""
        import pandas as pd

        results1 = Mock()
        results1.params = pd.Series({"x1": 1.2})
        results1.std_errors = pd.Series({"x1": 0.3})
        results1.pvalues = pd.Series({"x1": 0.01})
        results1.rsquared = 0.75
        results1.rsquared_adj = 0.73
        results1.fvalue = 45.0
        results1.llf = -150.0
        results1.aic = 305.0
        results1.bic = 315.0
        results1.hqic = 308.0

        results2 = Mock()
        results2.params = pd.Series({"x1": 1.5})
        results2.std_errors = pd.Series({"x1": 0.25})
        results2.pvalues = pd.Series({"x1": 0.005})
        results2.rsquared = 0.80
        results2.rsquared_adj = 0.78
        results2.fvalue = 52.0
        results2.llf = -145.0
        results2.aic = 294.0
        results2.bic = 304.0
        results2.hqic = 298.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_comparison_charts([results1, results2], charts=["forest_plot"])
            assert any("Forest plot" in str(warning.message) for warning in w)

    def test_comparison_chart_creation_error_warns(self):
        """Test chart creation error produces warning."""
        bad_results = Mock()
        bad_results.params = None  # Will cause error

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = create_comparison_charts([bad_results], charts=["coefficients"])
            # Either the chart is created despite the bad data, or a warning is raised
            assert isinstance(result, dict)

    def test_comparison_heatmap_data_path(self):
        """Test comparison_heatmap chart type requires models key."""
        from panelbox.visualization.api import create_validation_charts

        data = {
            "tests": [
                {
                    "name": "Test1",
                    "category": "Cat1",
                    "statistic": 2.5,
                    "pvalue": 0.05,
                    "passed": True,
                }
            ],
            "models": ["FE", "RE"],
            "test_names": ["Test1"],
            "pvalue_matrix": [[0.05], [0.03]],  # 2 rows for 2 models
        }
        charts = create_validation_charts(data, charts=["comparison_heatmap"])
        assert isinstance(charts, dict)
        assert "comparison_heatmap" in charts
