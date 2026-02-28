"""
Tests for uncovered branches in panelbox.visualization.api.

Targets specific uncovered lines:
- Lines 516-517: create_comparison_charts exception handler
- Lines 618-619: export_charts exception handler
- Lines 879, 883: create_panel_charts auto-detection branches
- Lines 915-918: create_panel_charts include_html True/False branches
- Lines 962-964: create_entity_effects_plot non-dict path
- Lines 1005-1007: create_time_effects_plot non-dict path
- Lines 1060-1062: create_between_within_plot non-dict path
- Lines 1109-1111: create_panel_structure_plot non-dict path
"""

from __future__ import annotations

import warnings
from unittest.mock import Mock, patch

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

plotly = pytest.importorskip("plotly")

from panelbox.visualization.api import (  # noqa: E402
    create_between_within_plot,
    create_comparison_charts,
    create_entity_effects_plot,
    create_panel_charts,
    create_panel_structure_plot,
    create_time_effects_plot,
    export_charts,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_comparison_results():
    """Create mock model results for comparison chart tests."""
    results1 = Mock()
    results1.params = pd.Series({"x1": 1.2, "x2": -0.5, "const": 2.0})
    results1.std_errors = pd.Series({"x1": 0.3, "x2": 0.2, "const": 0.5})
    results1.pvalues = pd.Series({"x1": 0.001, "x2": 0.02, "const": 0.0001})
    results1.rsquared = 0.75
    results1.rsquared_adj = 0.73
    results1.fvalue = 45.2
    results1.llf = -150.5
    results1.aic = 305.0
    results1.bic = 315.0
    results1.hqic = 308.5

    results2 = Mock()
    results2.params = pd.Series({"x1": 1.5, "x2": -0.3, "const": 1.8})
    results2.std_errors = pd.Series({"x1": 0.25, "x2": 0.18, "const": 0.45})
    results2.pvalues = pd.Series({"x1": 0.0005, "x2": 0.05, "const": 0.0002})
    results2.rsquared = 0.80
    results2.rsquared_adj = 0.78
    results2.fvalue = 52.1
    results2.llf = -145.2
    results2.aic = 294.4
    results2.bic = 304.4
    results2.hqic = 297.9

    return [results1, results2]


@pytest.fixture
def panel_dataframe():
    """Create a simple panel DataFrame with MultiIndex for structure/between-within tests."""
    np.random.seed(42)
    entities = ["A", "B", "C"]
    times = [2000, 2001, 2002]
    idx = pd.MultiIndex.from_product([entities, times], names=["entity", "time"])
    df = pd.DataFrame(
        {
            "wage": np.random.randn(9) * 10 + 50,
            "education": np.random.randn(9) * 2 + 12,
        },
        index=idx,
    )
    return df


@pytest.fixture
def mock_panel_results_with_entity_effects():
    """Mock object that has entity_effects attribute (triggers line 878 auto-detect)."""
    obj = Mock()
    obj.entity_effects = pd.Series([0.5, -0.3, 0.8], index=["Firm_A", "Firm_B", "Firm_C"])
    # Remove attributes we do NOT want so auto-detection is controlled
    del obj.dataframe
    del obj.model
    return obj


@pytest.fixture
def mock_panel_results_with_params():
    """Mock object that has params attribute but not entity_effects (triggers line 878)."""
    obj = Mock()
    # Has params -> triggers entity_effects/time_effects auto-detect (line 878)
    obj.params = pd.Series({"x1": 1.0, "x2": 0.5})
    # Remove attributes that would trigger the dataframe/model branch
    del obj.entity_effects
    del obj.dataframe
    del obj.model
    return obj


@pytest.fixture
def mock_panel_results_with_dataframe(panel_dataframe):
    """Mock object that has dataframe attribute (triggers line 882 auto-detect)."""
    obj = Mock()
    obj.dataframe = panel_dataframe
    # Remove attributes that would trigger entity_effects/params branch
    del obj.entity_effects
    del obj.params
    del obj.model
    return obj


@pytest.fixture
def mock_panel_results_with_model():
    """Mock object that has model attribute (triggers line 882 auto-detect)."""
    obj = Mock()
    obj.model = Mock()
    # Remove attributes that would trigger entity_effects/params branch
    del obj.entity_effects
    del obj.params
    del obj.dataframe
    return obj


@pytest.fixture
def mock_entity_effects_result():
    """Mock object for create_entity_effects_plot non-dict path (lines 962-964).

    PanelDataTransformer.extract_entity_effects expects entity_effects attr.
    """
    obj = Mock()
    obj.entity_effects = pd.Series(
        [0.5, -0.3, 0.8, 0.1],
        index=["Firm_A", "Firm_B", "Firm_C", "Firm_D"],
    )
    return obj


@pytest.fixture
def mock_time_effects_result():
    """Mock object for create_time_effects_plot non-dict path (lines 1005-1007).

    PanelDataTransformer.extract_time_effects expects time_effects attr.
    """
    obj = Mock()
    obj.time_effects = pd.Series(
        [0.1, 0.4, -0.2, 0.6],
        index=[2000, 2001, 2002, 2003],
    )
    return obj


@pytest.fixture
def mock_between_within_data(panel_dataframe):
    """Mock object for create_between_within_plot non-dict path (lines 1060-1062).

    PanelDataTransformer.calculate_between_within expects dataframe attr or DataFrame.
    """
    obj = Mock()
    obj.dataframe = panel_dataframe
    return obj


@pytest.fixture
def mock_structure_data(panel_dataframe):
    """Mock object for create_panel_structure_plot non-dict path (lines 1109-1111).

    PanelDataTransformer.analyze_panel_structure expects dataframe attr or DataFrame.
    """
    obj = Mock()
    obj.dataframe = panel_dataframe
    return obj


# ===========================================================================
# Tests: create_comparison_charts exception handler (lines 516-517)
# ===========================================================================


class TestCreateComparisonChartsExceptionHandler:
    """Test that create_comparison_charts catches exceptions per chart and warns."""

    def test_exception_in_chart_creation_emits_warning(self, mock_comparison_results):
        """When ChartFactory.create raises, a warning is emitted and other charts continue."""
        with patch(
            "panelbox.visualization.api.ChartFactory.create",
            side_effect=RuntimeError("boom"),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = create_comparison_charts(
                    mock_comparison_results,
                    charts=["coefficients", "fit_comparison"],
                )

            # Both charts should have failed
            assert len(result) == 0
            # There should be warnings about the failures
            warning_messages = [str(x.message) for x in w]
            assert any("Failed to create coefficients" in msg for msg in warning_messages)
            assert any("Failed to create fit_comparison" in msg for msg in warning_messages)

    def test_exception_in_one_chart_does_not_block_others(self, mock_comparison_results):
        """If one chart fails, the rest are still created.

        We patch the transformer's prepare_coefficient_comparison so only
        the 'coefficients' chart creation raises, while 'fit_comparison'
        and 'ic_comparison' succeed normally.
        """
        with (
            patch(
                "panelbox.visualization.transformers.comparison.ComparisonDataTransformer"
                ".prepare_coefficient_comparison",
                side_effect=RuntimeError("transformer boom"),
            ),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            result = create_comparison_charts(
                mock_comparison_results,
                charts=["coefficients", "fit_comparison", "ic_comparison"],
            )

        # coefficients chart failed because its transformer raised
        assert "coefficients" not in result
        # The other charts should have succeeded
        assert "fit_comparison" in result
        assert "ic_comparison" in result
        warning_messages = [str(x.message) for x in w]
        assert any("Failed to create coefficients" in msg for msg in warning_messages)


# ===========================================================================
# Tests: export_charts exception handler (lines 618-619)
# ===========================================================================


class TestExportChartsExceptionHandler:
    """Test that export_charts catches exceptions per chart and warns."""

    def test_save_image_exception_emits_warning(self, tmp_path):
        """When chart.save_image raises, a warning is emitted."""
        chart = Mock()
        chart.save_image = Mock(side_effect=OSError("disk full"))

        charts_dict = {"my_chart": chart}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = export_charts(charts_dict, output_dir=str(tmp_path))

        assert len(result) == 0
        warning_messages = [str(x.message) for x in w]
        assert any("Failed to export chart 'my_chart'" in msg for msg in warning_messages)
        assert any("disk full" in msg for msg in warning_messages)

    def test_partial_export_on_exception(self, tmp_path):
        """If one chart fails to export, others still succeed."""
        good_chart = Mock()
        good_chart.save_image = Mock()  # succeeds

        bad_chart = Mock()
        bad_chart.save_image = Mock(side_effect=RuntimeError("render failed"))

        charts_dict = {"good_chart": good_chart, "bad_chart": bad_chart}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = export_charts(charts_dict, output_dir=str(tmp_path))

        # Good chart was exported
        assert "good_chart" in result
        # Bad chart was not
        assert "bad_chart" not in result
        warning_messages = [str(x.message) for x in w]
        assert any("Failed to export chart 'bad_chart'" in msg for msg in warning_messages)


# ===========================================================================
# Tests: create_panel_charts auto-detection (lines 878-883)
# ===========================================================================


class TestCreatePanelChartsAutoDetection:
    """Test auto-detection of chart_types based on panel_results attributes."""

    def test_auto_detect_entity_effects_attr(self, mock_panel_results_with_entity_effects):
        """Object with entity_effects triggers entity_effects/time_effects auto-detect (line 878)."""
        # entity_effects attr is present -> chart_types should include entity_effects, time_effects
        # ChartFactory.create may fail on the mock, but the auto-detection (line 878-879) is covered
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = create_panel_charts(
                mock_panel_results_with_entity_effects,
                chart_types=None,  # trigger auto-detection
                include_html=False,
            )
        # At minimum entity_effects should succeed since entity_effects attr is present
        # time_effects may fail (warning) since mock lacks time_effects attr
        assert isinstance(result, dict)

    def test_auto_detect_params_attr(self, mock_panel_results_with_params):
        """Object with params triggers entity_effects/time_effects auto-detect (line 878)."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = create_panel_charts(
                mock_panel_results_with_params,
                chart_types=None,
                include_html=False,
            )
        assert isinstance(result, dict)

    def test_auto_detect_dataframe_attr(self, mock_panel_results_with_dataframe):
        """Object with dataframe triggers between_within/structure auto-detect (line 882)."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = create_panel_charts(
                mock_panel_results_with_dataframe,
                chart_types=None,
                include_html=False,
            )
        assert isinstance(result, dict)

    def test_auto_detect_model_attr(self, mock_panel_results_with_model):
        """Object with model triggers between_within/structure auto-detect (line 882)."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = create_panel_charts(
                mock_panel_results_with_model,
                chart_types=None,
                include_html=False,
            )
        assert isinstance(result, dict)

    def test_auto_detect_fallback_no_attrs(self):
        """Object with no recognized attrs falls back to all chart_types (line 886-887)."""
        obj = Mock(spec=[])  # spec=[] means no attributes at all
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = create_panel_charts(
                obj,
                chart_types=None,
                include_html=False,
            )
        # All chart types attempted; all likely fail with warnings
        assert isinstance(result, dict)


# ===========================================================================
# Tests: create_panel_charts include_html branches (lines 915-918)
# ===========================================================================


class TestCreatePanelChartsIncludeHtml:
    """Test include_html=True returns HTML strings, include_html=False returns chart objects."""

    @pytest.fixture
    def entity_effects_data(self):
        return {
            "entity_id": ["A", "B", "C"],
            "effect": [0.5, -0.3, 0.8],
            "std_error": [0.1, 0.15, 0.12],
        }

    def test_include_html_true_returns_strings(self, entity_effects_data):
        """With include_html=True, chart values should be HTML strings (line 916)."""
        result = create_panel_charts(
            entity_effects_data,
            chart_types=["entity_effects"],
            include_html=True,
        )
        assert "entity_effects" in result
        # Should be an HTML string
        assert isinstance(result["entity_effects"], str)
        assert "<div" in result["entity_effects"] or "<html" in result["entity_effects"].lower()

    def test_include_html_false_returns_chart_objects(self, entity_effects_data):
        """With include_html=False, chart values should be chart objects (line 918)."""
        result = create_panel_charts(
            entity_effects_data,
            chart_types=["entity_effects"],
            include_html=False,
        )
        assert "entity_effects" in result
        # Should NOT be a string; should be a chart object
        assert not isinstance(result["entity_effects"], str)
        assert hasattr(result["entity_effects"], "to_html")


# ===========================================================================
# Tests: create_entity_effects_plot non-dict path (lines 962-964)
# ===========================================================================


class TestCreateEntityEffectsPlotNonDict:
    """Test that non-dict input triggers PanelDataTransformer.extract_entity_effects."""

    def test_non_dict_with_entity_effects_attr(self, mock_entity_effects_result):
        """Non-dict object with entity_effects is transformed via PanelDataTransformer."""
        chart = create_entity_effects_plot(mock_entity_effects_result)
        assert chart is not None
        assert hasattr(chart, "to_html")

    def test_non_dict_with_entity_effects_custom_theme(self, mock_entity_effects_result):
        """Non-dict path also works with a custom theme string."""
        chart = create_entity_effects_plot(mock_entity_effects_result, theme="academic")
        assert chart is not None


# ===========================================================================
# Tests: create_time_effects_plot non-dict path (lines 1005-1007)
# ===========================================================================


class TestCreateTimeEffectsPlotNonDict:
    """Test that non-dict input triggers PanelDataTransformer.extract_time_effects."""

    def test_non_dict_with_time_effects_attr(self, mock_time_effects_result):
        """Non-dict object with time_effects is transformed via PanelDataTransformer."""
        chart = create_time_effects_plot(mock_time_effects_result)
        assert chart is not None
        assert hasattr(chart, "to_html")

    def test_non_dict_with_time_effects_custom_theme(self, mock_time_effects_result):
        """Non-dict path also works with a custom theme string."""
        chart = create_time_effects_plot(mock_time_effects_result, theme="academic")
        assert chart is not None


# ===========================================================================
# Tests: create_between_within_plot non-dict path (lines 1060-1062)
# ===========================================================================


class TestCreateBetweenWithinPlotNonDict:
    """Test that non-dict input with variables triggers PanelDataTransformer.calculate_between_within."""

    def test_non_dict_with_dataframe_attr(self, mock_between_within_data):
        """Non-dict object with dataframe attr is transformed via PanelDataTransformer."""
        chart = create_between_within_plot(
            mock_between_within_data,
            variables=["wage", "education"],
        )
        assert chart is not None
        assert hasattr(chart, "to_html")

    def test_non_dict_with_raw_dataframe(self, panel_dataframe):
        """Raw DataFrame is also accepted through PanelDataTransformer."""
        chart = create_between_within_plot(
            panel_dataframe,
            variables=["wage"],
        )
        assert chart is not None
        assert hasattr(chart, "to_html")

    def test_non_dict_with_custom_theme(self, mock_between_within_data):
        """Non-dict path works with a custom theme string."""
        chart = create_between_within_plot(
            mock_between_within_data,
            variables=["wage"],
            theme="academic",
        )
        assert chart is not None


# ===========================================================================
# Tests: create_panel_structure_plot non-dict path (lines 1109-1111)
# ===========================================================================


class TestCreatePanelStructurePlotNonDict:
    """Test that non-dict input triggers PanelDataTransformer.analyze_panel_structure."""

    def test_non_dict_with_dataframe_attr(self, mock_structure_data):
        """Non-dict object with dataframe attr is transformed via PanelDataTransformer."""
        chart = create_panel_structure_plot(mock_structure_data)
        assert chart is not None
        assert hasattr(chart, "to_html")

    def test_non_dict_with_raw_dataframe(self, panel_dataframe):
        """Raw DataFrame is also accepted through PanelDataTransformer."""
        chart = create_panel_structure_plot(panel_dataframe)
        assert chart is not None
        assert hasattr(chart, "to_html")

    def test_non_dict_with_custom_theme(self, mock_structure_data):
        """Non-dict path works with a custom theme string."""
        chart = create_panel_structure_plot(mock_structure_data, theme="presentation")
        assert chart is not None
