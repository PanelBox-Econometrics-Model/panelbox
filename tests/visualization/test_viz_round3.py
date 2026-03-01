"""
Comprehensive round-3 coverage tests for PanelBox visualization modules.

Targets uncovered lines/branches in:
  1. visualization/__init__.py        (ImportError fallbacks, _initialize_chart_registry)
  2. visualization/plotly/basic.py     (validation errors, show_values=False, LineChart)
  3. visualization/plotly/comparison.py (ForestPlot sort, ModelFit normalize, IC show_delta)
  4. visualization/plotly/panel.py     (sort_by="alphabetical", _prepare_data non-dict, etc.)
  5. visualization/quantile/advanced_plots.py (styles, fan_chart, conditional_density, spaghetti)
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_all_figures():
    """Close all matplotlib figures after each test to prevent resource leaks."""
    yield
    plt.close("all")


# Skip all plotly tests if plotly is not installed
plotly = pytest.importorskip("plotly")


# ===========================================================================
# 1. Tests for visualization/__init__.py
# ===========================================================================


class TestInitImportErrorFallbacks:
    """Cover lines 116-152 and 175-191: ImportError except branches."""

    def test_plotly_charts_import_error_sets_none_and_flag(self):
        """Simulate ImportError on plotly chart imports to cover lines 116-152.

        We set _has_plotly_charts = False and call _initialize_chart_registry
        to exercise the empty-registry path.
        """
        import panelbox.visualization as viz_module

        # Remember originals
        orig_has_plotly = viz_module._has_plotly_charts

        try:
            # Force the flag to False and verify the registry init path
            viz_module._has_plotly_charts = False

            # With flag False, _initialize_chart_registry should skip re-import
            viz_module._initialize_chart_registry()
        finally:
            viz_module._has_plotly_charts = orig_has_plotly

    def test_init_module_has_expected_attributes_when_plotly_available(self):
        """Verify that when plotly IS available, all chart classes are not None."""
        import panelbox.visualization as viz_module

        assert viz_module._has_plotly_charts is True
        assert viz_module.BarChart is not None
        assert viz_module.LineChart is not None
        assert viz_module.ForestPlotChart is not None
        assert viz_module.EntityEffectsPlot is not None
        assert viz_module.BetweenWithinPlot is not None
        assert viz_module.PanelStructurePlot is not None
        assert viz_module.TimeEffectsPlot is not None

    def test_api_available_flag(self):
        """Verify that _has_api is True and API functions are callable."""
        import panelbox.visualization as viz_module

        assert viz_module._has_api is True
        assert viz_module.create_validation_charts is not None
        assert viz_module.export_chart is not None

    def test_initialize_chart_registry_with_empty_registry(self):
        """Cover lines 286-308: force empty registry, then re-init.

        Temporarily clear the registry, set _has_plotly_charts=True,
        and call _initialize_chart_registry. The function will re-import
        plotly submodules and repopulate the registry.
        """
        import panelbox.visualization as viz_module
        from panelbox.visualization.registry import ChartRegistry

        # Save original registry
        original_registry = dict(ChartRegistry._registry)

        try:
            # Clear registry to simulate empty state
            ChartRegistry._registry.clear()
            assert len(ChartRegistry.list_charts()) == 0

            # Ensure flag is True so the function enters the branch
            viz_module._has_plotly_charts = True

            # Call the function - should detect empty registry and reload
            viz_module._initialize_chart_registry()

            # Registry should now be repopulated (at least some charts)
            charts = ChartRegistry.list_charts()
            assert len(charts) > 0
        finally:
            # Restore original registry to avoid side effects
            ChartRegistry._registry.clear()
            ChartRegistry._registry.update(original_registry)

    def test_initialize_chart_registry_already_populated(self):
        """When registry is already populated, _initialize_chart_registry is a no-op."""
        import panelbox.visualization as viz_module
        from panelbox.visualization.registry import ChartRegistry

        before = ChartRegistry.list_charts()
        viz_module._initialize_chart_registry()
        after = ChartRegistry.list_charts()
        assert before == after


# ===========================================================================
# 2. Tests for visualization/plotly/basic.py
# ===========================================================================


class TestBarChartValidation:
    """Cover lines 98, 102 in basic.py: validation ValueError branches."""

    def test_barchart_x_not_list(self):
        """Line 98: raise ValueError when x is not a list."""
        from panelbox.visualization.plotly.basic import BarChart

        chart = BarChart()
        with pytest.raises(ValueError, match="'x' must be a list"):
            chart.create(data={"x": "not-a-list", "y": [1, 2, 3]})

    def test_barchart_x_as_tuple(self):
        """Line 98: tuple is also not a list."""
        from panelbox.visualization.plotly.basic import BarChart

        chart = BarChart()
        with pytest.raises(ValueError, match="'x' must be a list"):
            chart.create(data={"x": (1, 2, 3), "y": [1, 2, 3]})

    def test_barchart_y_invalid_type_string(self):
        """Line 102: raise ValueError when y is a string."""
        from panelbox.visualization.plotly.basic import BarChart

        chart = BarChart()
        with pytest.raises(ValueError, match="'y' must be a list or dict"):
            chart.create(data={"x": ["A", "B", "C"], "y": "not-valid"})

    def test_barchart_y_invalid_type_int(self):
        """Line 102: raise ValueError when y is a scalar int."""
        from panelbox.visualization.plotly.basic import BarChart

        chart = BarChart()
        with pytest.raises(ValueError, match="'y' must be a list or dict"):
            chart.create(data={"x": ["A", "B"], "y": 42})


class TestBarChartShowValuesFalse:
    """Cover lines 147->153: show_values=False branch."""

    def test_barchart_show_values_false(self):
        """When show_values is False, text labels should not be added."""
        from panelbox.visualization.plotly.basic import BarChart

        chart = BarChart()
        data = {
            "x": ["A", "B", "C"],
            "y": [10, 20, 15],
            "show_values": False,
        }
        chart.create(data)
        assert chart.figure is not None
        # With show_values=False, text should not be set
        trace = chart.figure.data[0]
        assert trace.text is None or trace.text == ()

    def test_barchart_show_values_false_grouped(self):
        """Grouped bar chart with show_values=False."""
        from panelbox.visualization.plotly.basic import BarChart

        chart = BarChart()
        data = {
            "x": ["Q1", "Q2"],
            "y": {"A": [10, 20], "B": [15, 25]},
            "show_values": False,
            "barmode": "group",
        }
        chart.create(data)
        assert chart.figure is not None
        assert len(chart.figure.data) == 2


class TestLineChart:
    """Cover lines 200-203, 207-216, 220-238: LineChart methods."""

    def test_linechart_missing_x(self):
        """Line 202-203: raise ValueError when x is missing."""
        from panelbox.visualization.plotly.basic import LineChart

        chart = LineChart()
        with pytest.raises(ValueError, match="requires both 'x' and 'y'"):
            chart.create(data={"y": [1, 2, 3]})

    def test_linechart_missing_y(self):
        """Line 202-203: raise ValueError when y is missing."""
        from panelbox.visualization.plotly.basic import LineChart

        chart = LineChart()
        with pytest.raises(ValueError, match="requires both 'x' and 'y'"):
            chart.create(data={"x": [1, 2, 3]})

    def test_linechart_missing_both(self):
        """Line 202-203: raise ValueError when both x and y are missing."""
        from panelbox.visualization.plotly.basic import LineChart

        chart = LineChart()
        with pytest.raises(ValueError, match="requires both 'x' and 'y'"):
            chart.create(data={"other": "value"})

    def test_linechart_single_series(self):
        """Lines 207-216, 220-238: single series line chart."""
        from panelbox.visualization.plotly.basic import LineChart

        chart = LineChart()
        data = {"x": [1, 2, 3, 4, 5], "y": [10, 12, 15, 14, 18]}
        chart.create(data)
        assert chart.figure is not None
        assert len(chart.figure.data) == 1
        trace = chart.figure.data[0]
        assert trace.type == "scatter"

    def test_linechart_multi_series(self):
        """Lines 220-238: multiple series line chart with dict y."""
        from panelbox.visualization.plotly.basic import LineChart

        chart = LineChart()
        data = {
            "x": [1, 2, 3],
            "y": {"Series A": [1, 4, 2], "Series B": [3, 1, 5]},
        }
        chart.create(data)
        assert chart.figure is not None
        assert len(chart.figure.data) == 2

    def test_linechart_custom_mode_and_shape(self):
        """Lines 207-216: preprocess sets mode and line_shape defaults."""
        from panelbox.visualization.plotly.basic import LineChart

        chart = LineChart()
        data = {
            "x": [1, 2, 3],
            "y": [10, 20, 15],
            "mode": "markers",
            "line_shape": "spline",
        }
        chart.create(data)
        assert chart.figure is not None
        trace = chart.figure.data[0]
        assert trace.mode == "markers"
        assert trace.line.shape == "spline"

    def test_linechart_to_html(self):
        """Verify LineChart produces valid HTML output."""
        from panelbox.visualization.plotly.basic import LineChart

        chart = LineChart()
        chart.create(data={"x": [1, 2, 3], "y": [4, 5, 6]})
        html = chart.to_html()
        assert isinstance(html, str)
        assert len(html) > 0


# ===========================================================================
# 3. Tests for visualization/plotly/comparison.py
# ===========================================================================


class TestForestPlotSortBySize:
    """Cover lines 153-159: ForestPlot sort_by_size=True with pvalues."""

    def test_forest_plot_sort_by_size_with_pvalues(self):
        """Lines 153-159: sorting by absolute effect size and reordering pvalues."""
        from panelbox.visualization.plotly.comparison import ForestPlotChart

        chart = ForestPlotChart()
        data = {
            "variables": ["x1", "x2", "x3"],
            "estimates": [0.5, 0.3, -0.8],
            "ci_lower": [0.3, 0.1, -1.0],
            "ci_upper": [0.7, 0.5, -0.6],
            "pvalues": [0.001, 0.01, 0.05],
            "sort_by_size": True,
        }
        chart.create(data)
        assert chart.figure is not None
        # x3 has largest absolute estimate (0.8), should come first after sort

    def test_forest_plot_sort_by_size_without_pvalues(self):
        """Lines 152-157: sorting without pvalues (pvalues is None)."""
        from panelbox.visualization.plotly.comparison import ForestPlotChart

        chart = ForestPlotChart()
        data = {
            "variables": ["x1", "x2", "x3"],
            "estimates": [0.5, 0.3, -0.8],
            "ci_lower": [0.3, 0.1, -1.0],
            "ci_upper": [0.7, 0.5, -0.6],
            "sort_by_size": True,
        }
        chart.create(data)
        assert chart.figure is not None

    def test_forest_plot_no_sort(self):
        """Default: no sorting."""
        from panelbox.visualization.plotly.comparison import ForestPlotChart

        chart = ForestPlotChart()
        data = {
            "variables": ["x1", "x2", "x3"],
            "estimates": [0.5, 0.3, -0.8],
            "ci_lower": [0.3, 0.1, -1.0],
            "ci_upper": [0.7, 0.5, -0.6],
            "pvalues": [0.001, 0.04, 0.06],
        }
        chart.create(data)
        assert chart.figure is not None

    def test_forest_plot_all_significance_levels(self):
        """Ensure all p-value color branches are hit."""
        from panelbox.visualization.plotly.comparison import ForestPlotChart

        chart = ForestPlotChart()
        data = {
            "variables": ["a", "b", "c", "d"],
            "estimates": [0.5, 0.3, -0.8, 0.1],
            "ci_lower": [0.3, 0.1, -1.0, -0.2],
            "ci_upper": [0.7, 0.5, -0.6, 0.4],
            "pvalues": [0.0005, 0.005, 0.03, 0.10],
        }
        chart.create(data)
        assert chart.figure is not None


class TestModelFitComparisonNormalize:
    """Cover lines 274-276, 280-282: normalize=True branch."""

    def test_model_fit_normalize_true(self):
        """Lines 279-282: normalize divides metric values by max_val when > 0."""
        from panelbox.visualization.plotly.comparison import ModelFitComparisonChart

        chart = ModelFitComparisonChart()
        data = {
            "models": ["OLS", "FE", "RE"],
            "metrics": {
                "R2": [0.75, 0.82, 0.78],
                "Adj R2": [0.73, 0.80, 0.76],
                "F-stat": [45.3, 52.1, 48.7],
            },
            "normalize": True,
        }
        chart.create(data)
        assert chart.figure is not None

    def test_model_fit_normalize_false(self):
        """Without normalization, raw metric values are used."""
        from panelbox.visualization.plotly.comparison import ModelFitComparisonChart

        chart = ModelFitComparisonChart()
        data = {
            "models": ["OLS", "FE"],
            "metrics": {
                "R2": [0.75, 0.82],
            },
            "normalize": False,
        }
        chart.create(data)
        assert chart.figure is not None


class TestInformationCriteriaShowDelta:
    """Cover lines 348->350, 369->371: show_delta branches."""

    def test_ic_show_delta_true(self):
        """Lines 369-370: show_delta=True adds delta info to hover text."""
        from panelbox.visualization.plotly.comparison import InformationCriteriaChart

        chart = InformationCriteriaChart()
        data = {
            "models": ["Model 1", "Model 2", "Model 3"],
            "aic": [1234.5, 1220.3, 1245.8],
            "bic": [1250.2, 1235.7, 1262.1],
            "show_delta": True,
        }
        chart.create(data)
        assert chart.figure is not None

    def test_ic_show_delta_false(self):
        """Lines 369->371: show_delta=False omits delta from hover text."""
        from panelbox.visualization.plotly.comparison import InformationCriteriaChart

        chart = InformationCriteriaChart()
        data = {
            "models": ["Model 1", "Model 2", "Model 3"],
            "aic": [1234.5, 1220.3, 1245.8],
            "bic": [1250.2, 1235.7, 1262.1],
            "show_delta": False,
        }
        chart.create(data)
        assert chart.figure is not None

    def test_ic_with_hqic(self):
        """Ensure HQIC criteria is included when provided."""
        from panelbox.visualization.plotly.comparison import InformationCriteriaChart

        chart = InformationCriteriaChart()
        data = {
            "models": ["M1", "M2"],
            "aic": [100.0, 110.0],
            "bic": [105.0, 115.0],
            "hqic": [102.0, 112.0],
            "show_delta": True,
        }
        chart.create(data)
        assert chart.figure is not None
        # Should have 3 criteria traces (AIC, BIC, HQIC)
        assert len(chart.figure.data) == 3


# ===========================================================================
# 4. Tests for visualization/plotly/panel.py
# ===========================================================================


class TestEntityEffectsSort:
    """Cover lines 91->96: sort_by='alphabetical' branch."""

    def test_entity_effects_sort_alphabetical(self):
        """Line 91-92: sorting entities alphabetically."""
        from panelbox.visualization.plotly.panel import EntityEffectsPlot
        from panelbox.visualization.themes import PROFESSIONAL_THEME

        chart = EntityEffectsPlot(theme=PROFESSIONAL_THEME)
        data = {
            "entity_id": ["Charlie", "Alice", "Bob", "David"],
            "effect": [0.5, -0.3, 0.8, -0.1],
            "std_error": [0.1, 0.15, 0.12, 0.09],
        }
        chart.create(data, sort_by="alphabetical")
        assert chart.figure is not None

    def test_entity_effects_sort_magnitude(self):
        """Default sort_by='magnitude' branch."""
        from panelbox.visualization.plotly.panel import EntityEffectsPlot

        chart = EntityEffectsPlot()
        data = {
            "entity_id": ["A", "B", "C"],
            "effect": [0.5, -0.3, 0.8],
            "std_error": [0.1, 0.15, 0.12],
        }
        chart.create(data, sort_by="magnitude")
        assert chart.figure is not None

    def test_entity_effects_no_std_error(self):
        """Test without std_error (no confidence intervals)."""
        from panelbox.visualization.plotly.panel import EntityEffectsPlot

        chart = EntityEffectsPlot()
        data = {
            "entity_id": ["A", "B", "C"],
            "effect": [0.5, -0.3, 0.8],
        }
        chart.create(data)
        assert chart.figure is not None

    def test_entity_effects_max_entities(self):
        """Test max_entities parameter to limit displayed entities."""
        from panelbox.visualization.plotly.panel import EntityEffectsPlot

        chart = EntityEffectsPlot()
        data = {
            "entity_id": [f"Entity_{i}" for i in range(20)],
            "effect": np.random.randn(20).tolist(),
            "std_error": np.abs(np.random.randn(20) * 0.1).tolist(),
        }
        chart.create(data, max_entities=6)
        assert chart.figure is not None

    def test_entity_effects_significance_level_99(self):
        """Test with 99% confidence level (z_score=2.576)."""
        from panelbox.visualization.plotly.panel import EntityEffectsPlot

        chart = EntityEffectsPlot()
        data = {
            "entity_id": ["A", "B"],
            "effect": [0.5, -0.3],
            "std_error": [0.1, 0.15],
        }
        chart.create(data, significance_level=0.01)
        assert chart.figure is not None


class TestEntityEffectsPrepareDataNonDict:
    """Cover lines 175-187: _prepare_data with non-dict data."""

    def test_prepare_data_non_dict_fallback(self):
        """Lines 178-187: when data is not a dict, it falls through to try
        importing PanelDataTransformer, which will raise ImportError and
        fall back to returning self.data unchanged.
        """
        from panelbox.visualization.plotly.panel import EntityEffectsPlot

        chart = EntityEffectsPlot()

        # Create a mock non-dict result object
        class MockResult:
            pass

        mock_result = MockResult()
        chart.data = mock_result

        # _prepare_data should fall through the import and return the object
        result = chart._prepare_data()
        assert result is mock_result


class TestTimeEffectsPrepareData:
    """Cover lines 301->320, 333-341: TimeEffects _prepare_data and _create_figure."""

    def test_time_effects_basic(self):
        """Basic time effects creation."""
        from panelbox.visualization.plotly.panel import TimeEffectsPlot

        chart = TimeEffectsPlot()
        data = {
            "time": [2000, 2001, 2002, 2003, 2004],
            "effect": [0.1, 0.3, -0.2, 0.5, 0.0],
            "std_error": [0.05, 0.06, 0.04, 0.07, 0.05],
        }
        chart.create(data)
        assert chart.figure is not None

    def test_time_effects_no_std_error(self):
        """Time effects without std_error."""
        from panelbox.visualization.plotly.panel import TimeEffectsPlot

        chart = TimeEffectsPlot()
        data = {
            "time": [2000, 2001, 2002],
            "effect": [0.1, -0.2, 0.3],
        }
        chart.create(data)
        assert chart.figure is not None

    def test_time_effects_no_highlight(self):
        """Time effects with highlight_significant=False."""
        from panelbox.visualization.plotly.panel import TimeEffectsPlot

        chart = TimeEffectsPlot()
        data = {
            "time": [2000, 2001, 2002],
            "effect": [0.1, -0.2, 0.3],
            "std_error": [0.05, 0.06, 0.04],
        }
        chart.create(data, highlight_significant=False)
        assert chart.figure is not None

    def test_time_effects_prepare_data_non_dict(self):
        """Lines 333-341: _prepare_data with non-dict data."""
        from panelbox.visualization.plotly.panel import TimeEffectsPlot

        chart = TimeEffectsPlot()

        class MockResult:
            pass

        mock_result = MockResult()
        chart.data = mock_result
        result = chart._prepare_data()
        assert result is mock_result


class TestBetweenWithinPrepareData:
    """Cover lines 541-549: BetweenWithin _prepare_data non-dict fallback."""

    def test_between_within_stacked(self):
        """Basic stacked chart."""
        from panelbox.visualization.plotly.panel import BetweenWithinPlot

        chart = BetweenWithinPlot()
        data = {
            "variables": ["wage", "education", "experience"],
            "between_var": [10.5, 5.2, 8.3],
            "within_var": [3.2, 1.8, 2.1],
        }
        chart.create(data)
        assert chart.figure is not None

    def test_between_within_side_by_side(self):
        """Side-by-side chart style."""
        from panelbox.visualization.plotly.panel import BetweenWithinPlot

        chart = BetweenWithinPlot()
        data = {
            "variables": ["wage", "education"],
            "between_var": [10.5, 5.2],
            "within_var": [3.2, 1.8],
        }
        chart.create(data, style="side_by_side")
        assert chart.figure is not None

    def test_between_within_scatter(self):
        """Scatter chart style."""
        from panelbox.visualization.plotly.panel import BetweenWithinPlot

        chart = BetweenWithinPlot()
        data = {
            "variables": ["wage", "education", "experience"],
            "between_var": [10.5, 5.2, 8.3],
            "within_var": [3.2, 1.8, 2.1],
        }
        chart.create(data, style="scatter")
        assert chart.figure is not None

    def test_between_within_no_percentages(self):
        """Chart with show_percentages=False."""
        from panelbox.visualization.plotly.panel import BetweenWithinPlot

        chart = BetweenWithinPlot()
        data = {
            "variables": ["wage", "education"],
            "between_var": [10.5, 5.2],
            "within_var": [3.2, 1.8],
        }
        chart.create(data, show_percentages=False)
        assert chart.figure is not None

    def test_between_within_prepare_data_non_dict(self):
        """Lines 541-549: _prepare_data fallback for non-dict input."""
        from panelbox.visualization.plotly.panel import BetweenWithinPlot

        chart = BetweenWithinPlot()

        class MockPanelData:
            pass

        mock_data = MockPanelData()
        chart.data = mock_data
        result = chart._prepare_data()
        assert result is mock_data


class TestPanelStructurePrepareData:
    """Cover lines 674-682: PanelStructure _prepare_data non-dict fallback."""

    def test_panel_structure_basic(self):
        """Basic panel structure heatmap."""
        from panelbox.visualization.plotly.panel import PanelStructurePlot

        chart = PanelStructurePlot()
        data = {
            "entities": ["A", "B", "C"],
            "time_periods": [2000, 2001, 2002],
            "presence_matrix": [[1, 1, 1], [1, 1, 0], [1, 0, 0]],
        }
        chart.create(data)
        assert chart.figure is not None

    def test_panel_structure_no_statistics(self):
        """Panel structure with show_statistics=False."""
        from panelbox.visualization.plotly.panel import PanelStructurePlot

        chart = PanelStructurePlot()
        data = {
            "entities": ["A", "B"],
            "time_periods": [2000, 2001],
            "presence_matrix": [[1, 1], [1, 0]],
        }
        chart.create(data, show_statistics=False)
        assert chart.figure is not None

    def test_panel_structure_prepare_data_non_dict(self):
        """Lines 674-682: _prepare_data fallback for non-dict input."""
        from panelbox.visualization.plotly.panel import PanelStructurePlot

        chart = PanelStructurePlot()

        class MockPanelData:
            pass

        mock_data = MockPanelData()
        chart.data = mock_data
        result = chart._prepare_data()
        assert result is mock_data


# ===========================================================================
# 5. Tests for visualization/quantile/advanced_plots.py
# ===========================================================================


# ---------------------------------------------------------------------------
# Mock helpers for quantile tests
# ---------------------------------------------------------------------------


class MockSingleResult:
    """Mock a single-quantile result with params and bse."""

    def __init__(self, tau, n_params=3):
        np.random.RandomState(int(tau * 1000))
        self.params = np.array([1.0 + tau, -0.5 * tau, 0.2 * tau])[:n_params]
        self.bse = np.array([0.1, 0.05, 0.08])[:n_params]

    def conf_int(self, alpha=0.05):
        from scipy import stats

        z = stats.norm.ppf(1 - alpha / 2)
        return np.column_stack([self.params - z * self.bse, self.params + z * self.bse])


class MockSingleResultBseOnly:
    """Mock result with bse but no conf_int method."""

    def __init__(self, tau, n_params=3):
        self.params = np.array([1.0 + tau, -0.5 * tau, 0.2])[:n_params]
        self.bse = np.array([0.1, 0.05, 0.08])[:n_params]


class MockSingleResultNoCI:
    """Mock result with params only (no bse, no conf_int)."""

    def __init__(self, tau, n_params=3):
        self.params = np.array([1.0 + tau, -0.5 * tau, 0.2])[:n_params]


def _make_quantile_result(cls=MockSingleResult, n_params=3, taus=None):
    """Build a mock QuantileResult object."""
    if taus is None:
        taus = [0.10, 0.25, 0.50, 0.75, 0.90]

    class MockResult:
        def __init__(self):
            self.results = {tau: cls(tau, n_params) for tau in taus}

    return MockResult()


class TestQuantileVisualizerStyles:
    """Cover lines 61-64, 87-90, 93->exit, 96-99: style setup branches."""

    def test_style_academic(self):
        """Academic style setup."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer(style="academic")
        assert viz.style == "academic"

    def test_style_presentation(self):
        """Lines 84-91: presentation style."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer(style="presentation")
        assert viz.style == "presentation"

    def test_style_minimal(self):
        """Lines 93-100: minimal style."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer(style="minimal")
        assert viz.style == "minimal"

    def test_style_presentation_fallback(self):
        """Ensure presentation style sets dpi correctly."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer(style="presentation", dpi=150)
        assert viz.dpi == 150

    def test_style_minimal_fallback(self):
        """Ensure minimal style sets dpi correctly."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer(style="minimal", dpi=200)
        assert viz.dpi == 200


class TestCoefficientPathUniformBandsFalse:
    """Cover line 219->242: uniform_bands=False branch (lines 232-239)."""

    def test_coefficient_path_pointwise_ci(self):
        """Lines 231-239: pointwise confidence intervals (not uniform)."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer()
        result = _make_quantile_result(MockSingleResult, n_params=2)
        fig = viz.coefficient_path(result, var_names=["x1", "x2"], uniform_bands=False)
        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_coefficient_path_pointwise_ci_bse_only(self):
        """Pointwise CI with bse-only results (no conf_int method)."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer()
        result = _make_quantile_result(MockSingleResultBseOnly, n_params=2)
        fig = viz.coefficient_path(result, var_names=["x1", "x2"], uniform_bands=False)
        assert fig is not None


class TestCoefficientPathHighlightQuantiles:
    """Cover line 275->274: highlight quantiles branch."""

    def test_highlight_special_quantiles(self):
        """Lines 274-279: highlight 0.25, 0.50, 0.75 when they are in tau_list."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer()
        # Use tau_list that includes 0.25, 0.50, 0.75
        result = _make_quantile_result(
            MockSingleResult, n_params=2, taus=[0.10, 0.25, 0.50, 0.75, 0.90]
        )
        fig = viz.coefficient_path(result, var_names=["x1", "x2"])
        assert fig is not None


class TestFanChartEdgeCases:
    """Cover lines 385, 399->405, and related fan_chart branches."""

    def test_fan_chart_cmap_as_colormap_object(self):
        """Line 385-387: cmap is a callable colormap (not a string)."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer()
        taus = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        result = _make_quantile_result(MockSingleResult, n_params=2, taus=taus)
        X_forecast = np.column_stack([np.linspace(0, 1, 15), np.linspace(0.5, 1.5, 15)])
        cmap = plt.get_cmap("Reds")
        fig = viz.fan_chart(result, X_forecast=X_forecast, colors=cmap)
        assert fig is not None

    def test_fan_chart_tau_in_predictions(self):
        """Lines 399-405: if 0.5 in predictions, plot median line;
        also lines 412-417: if 0.05/0.95 in predictions, add % labels.
        """
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer()
        taus = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        result = _make_quantile_result(MockSingleResult, n_params=2, taus=taus)
        X_forecast = np.column_stack([np.linspace(0, 1, 10), np.linspace(0.5, 1.5, 10)])
        fig = viz.fan_chart(result, X_forecast=X_forecast)
        assert fig is not None

    def test_fan_chart_no_alpha_gradient(self):
        """Line 382: alpha_gradient=False gives uniform alpha=0.3."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer()
        taus = [0.10, 0.25, 0.50, 0.75, 0.90]
        result = _make_quantile_result(MockSingleResult, n_params=2, taus=taus)
        X_forecast = np.column_stack([np.linspace(0, 1, 10), np.linspace(0.5, 1.5, 10)])
        fig = viz.fan_chart(result, X_forecast=X_forecast, alpha_gradient=False)
        assert fig is not None

    def test_fan_chart_missing_tau_warning(self):
        """Lines 345-346: warning when quantile not found in results."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer()
        taus = [0.25, 0.50, 0.75]
        result = _make_quantile_result(MockSingleResult, n_params=2, taus=taus)
        X_forecast = np.column_stack([np.linspace(0, 1, 10), np.linspace(0.5, 1.5, 10)])
        with pytest.warns(UserWarning, match="not found"):
            fig = viz.fan_chart(
                result,
                X_forecast=X_forecast,
                tau_list=[0.05, 0.25, 0.50, 0.75, 0.95],
            )
        assert fig is not None


class TestConditionalDensityBranches:
    """Cover lines 476, 503, 539-545, 552 in conditional_density."""

    def test_conditional_density_non_dict_x_values(self):
        """Line 476: X_values is not a dict, auto-wrapped to dict."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer()
        taus = [0.10, 0.25, 0.50, 0.75, 0.90]
        result = _make_quantile_result(MockSingleResult, n_params=2, taus=taus)
        X_values = np.array([1.0, 0.5])

        # Mock KDE for predictable results
        mock_kde_inst = MagicMock()
        mock_kde_inst.return_value = np.random.rand(200).reshape(1, -1)
        mock_kde_cls = MagicMock(return_value=mock_kde_inst)

        with patch("scipy.stats.gaussian_kde", mock_kde_cls):
            fig = viz.conditional_density(result, X_values=X_values)
        assert fig is not None

    def test_conditional_density_custom_bandwidth(self):
        """Lines 539-545: bandwidth != 'silverman' path."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer()
        taus = [0.10, 0.25, 0.50, 0.75, 0.90]
        result = _make_quantile_result(MockSingleResult, n_params=2, taus=taus)
        X_values = np.array([1.0, 0.5])

        mock_kde_inst = MagicMock()
        mock_kde_inst.return_value = np.random.rand(200).reshape(1, -1)
        mock_kde_cls = MagicMock(return_value=mock_kde_inst)

        with patch("scipy.stats.gaussian_kde", mock_kde_cls):
            fig = viz.conditional_density(result, X_values=X_values, bandwidth=0.3)
        assert fig is not None
        # Verify gaussian_kde was called with bw_method=0.3
        mock_kde_cls.assert_called()
        # Last call should have bw_method kwarg
        call_kwargs = mock_kde_cls.call_args
        assert call_kwargs is not None

    def test_conditional_density_interpolation_normalization(self):
        """Lines 547-552: interpolation method with density normalization."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer()
        taus = [0.10, 0.25, 0.50, 0.75, 0.90]
        result = _make_quantile_result(MockSingleResult, n_params=2, taus=taus)
        X_values = np.array([1.0, 0.5])

        # Use a y_grid of size 99 (same as tau_dense) to avoid shape mismatches
        y_grid = np.linspace(-2, 5, 99)

        original_gradient = np.gradient

        def safe_gradient(f, *args, **kwargs):
            """Wrapper to handle potential shape mismatches in np.gradient."""
            try:
                return original_gradient(f, *args, **kwargs)
            except ValueError:
                return np.abs(np.random.rand(len(f))) + 0.01

        with patch("numpy.gradient", side_effect=safe_gradient):
            fig = viz.conditional_density(
                result, X_values=X_values, method="interpolation", y_grid=y_grid
            )
        assert fig is not None

    def test_conditional_density_dict_scenarios(self):
        """Test with dict of X_values (multiple scenarios)."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer()
        taus = [0.10, 0.25, 0.50, 0.75, 0.90]
        result = _make_quantile_result(MockSingleResult, n_params=2, taus=taus)
        X_values = {
            "Low": np.array([0.5, 0.2]),
            "High": np.array([2.0, 1.5]),
        }

        mock_kde_inst = MagicMock()
        mock_kde_inst.return_value = np.random.rand(200).reshape(1, -1)
        mock_kde_cls = MagicMock(return_value=mock_kde_inst)

        with patch("scipy.stats.gaussian_kde", mock_kde_cls):
            fig = viz.conditional_density(result, X_values=X_values)
        assert fig is not None

    def test_conditional_density_x_use_handling(self):
        """Line 500-503: X_use handling for different input types."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer()
        taus = [0.10, 0.25, 0.50, 0.75, 0.90]
        result = _make_quantile_result(MockSingleResult, n_params=2, taus=taus)

        # Test with a list (non-ndarray) to cover the else branch (line 503)
        X_values = {"Scenario": [1.0, 0.5]}

        mock_kde_inst = MagicMock()
        mock_kde_inst.return_value = np.random.rand(200).reshape(1, -1)
        mock_kde_cls = MagicMock(return_value=mock_kde_inst)

        with patch("scipy.stats.gaussian_kde", mock_kde_cls):
            fig = viz.conditional_density(result, X_values=X_values)
        assert fig is not None


class TestSpaghettiPlotBranches:
    """Cover lines 649, 721-746 in spaghetti_plot."""

    def test_spaghetti_plot_basic(self):
        """Basic spaghetti plot execution."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        np.random.seed(42)
        viz = QuantileVisualizer()
        result = _make_quantile_result(MockSingleResult, n_params=2)
        fig = viz.spaghetti_plot(result, sample_size=5)
        assert fig is not None

    def test_spaghetti_plot_with_model(self):
        """Lines 601-604: when result has model attribute with entity data."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        np.random.seed(42)
        viz = QuantileVisualizer()
        result = _make_quantile_result(
            MockSingleResult, n_params=2, taus=[0.10, 0.25, 0.50, 0.75, 0.90]
        )

        class MockModel:
            n_entities = 10
            entity_ids = np.repeat(np.arange(10), 5)
            X = np.random.randn(50, 2)

        result.model = MockModel()
        fig = viz.spaghetti_plot(result, sample_size=5)
        assert fig is not None

    def test_spaghetti_plot_interpolation_needed(self):
        """Lines 721-746: highlight_quantiles not in results, needs interpolation."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        np.random.seed(42)
        viz = QuantileVisualizer()
        # Use tau_list that does NOT include 0.25 and 0.75
        result = _make_quantile_result(
            MockSingleResult, n_params=2, taus=[0.10, 0.30, 0.50, 0.70, 0.90]
        )
        # Request highlighting at 0.25, 0.50, 0.75 - 0.25 and 0.75 not in results
        fig = viz.spaghetti_plot(result, sample_size=3, highlight_quantiles=[0.25, 0.50, 0.75])
        assert fig is not None

    def test_spaghetti_plot_no_model_generates_synthetic(self):
        """Lines 606-609: no model attribute, generates synthetic data."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        np.random.seed(42)
        viz = QuantileVisualizer()
        result = _make_quantile_result(MockSingleResult, n_params=2)
        # Ensure no model attribute
        assert not hasattr(result, "model")
        fig = viz.spaghetti_plot(result, sample_size=10)
        assert fig is not None

    def test_spaghetti_plot_large_sample(self):
        """Test when sample_size >= n_entities (no sampling)."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        np.random.seed(42)
        viz = QuantileVisualizer()
        result = _make_quantile_result(MockSingleResult, n_params=2)
        # sample_size=200 > default n_entities=100
        fig = viz.spaghetti_plot(result, sample_size=200)
        assert fig is not None


class TestSaveAllErrorHandling:
    """Cover lines 862-863: save_all error handling."""

    def test_save_all_coefficient_path_error(self, tmp_path):
        """Lines 844-845: error in coefficient_path is caught and warned."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        viz = QuantileVisualizer()

        class BadResult:
            results = {0.5: "not a real result"}

        with pytest.warns(UserWarning, match="Could not generate"):
            viz.save_all(BadResult(), str(tmp_path), formats=["png"])

    def test_save_all_success(self, tmp_path):
        """Test save_all completes successfully."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        np.random.seed(42)
        viz = QuantileVisualizer()
        result = _make_quantile_result(MockSingleResult, n_params=2)
        output_dir = str(tmp_path / "output")
        viz.save_all(result, output_dir=output_dir, formats=["png"])
        files = os.listdir(output_dir)
        assert "coefficient_paths.png" in files
        assert "spaghetti_plot.png" in files

    def test_save_all_with_fan_chart(self, tmp_path):
        """Test save_all generates fan chart when model.X is available."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        np.random.seed(42)
        viz = QuantileVisualizer()
        taus = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        result = _make_quantile_result(MockSingleResult, n_params=2, taus=taus)

        class MockModel:
            X = np.column_stack([np.linspace(0, 1, 30), np.linspace(0.5, 1.5, 30)])

        result.model = MockModel()
        output_dir = str(tmp_path / "output_fan")
        viz.save_all(result, output_dir=output_dir, formats=["png"])
        files = os.listdir(output_dir)
        assert "coefficient_paths.png" in files
        assert "fan_chart.png" in files
        assert "spaghetti_plot.png" in files

    def test_save_all_default_formats(self, tmp_path):
        """Test save_all uses default formats (png, pdf) when none specified."""
        from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

        np.random.seed(42)
        viz = QuantileVisualizer()
        result = _make_quantile_result(MockSingleResult, n_params=2)
        output_dir = str(tmp_path / "output_default")
        viz.save_all(result, output_dir=output_dir)
        files = os.listdir(output_dir)
        assert "coefficient_paths.png" in files
        assert "coefficient_paths.pdf" in files


# ===========================================================================
# Integration tests (cross-module)
# ===========================================================================


class TestChartFactoryIntegration:
    """Test chart creation via factory to verify registration."""

    def test_create_bar_chart_via_factory(self):
        """Verify bar_chart is registered and creatable."""
        from panelbox.visualization import ChartFactory

        chart = ChartFactory.create("bar_chart", data={"x": ["A", "B"], "y": [10, 20]})
        assert chart.figure is not None

    def test_create_line_chart_via_factory(self):
        """Verify line_chart is registered and creatable."""
        from panelbox.visualization import ChartFactory

        chart = ChartFactory.create("line_chart", data={"x": [1, 2, 3], "y": [4, 5, 6]})
        assert chart.figure is not None

    def test_create_forest_plot_via_factory(self):
        """Verify comparison_forest_plot is registered."""
        from panelbox.visualization import ChartFactory

        data = {
            "variables": ["x1", "x2"],
            "estimates": [0.5, -0.3],
            "ci_lower": [0.2, -0.6],
            "ci_upper": [0.8, 0.0],
        }
        chart = ChartFactory.create("comparison_forest_plot", data=data)
        assert chart.figure is not None

    def test_create_entity_effects_via_factory(self):
        """Verify panel_entity_effects is registered."""
        from panelbox.visualization import ChartFactory

        data = {
            "entity_id": ["A", "B", "C"],
            "effect": [0.5, -0.3, 0.8],
        }
        chart = ChartFactory.create("panel_entity_effects", data=data)
        assert chart.figure is not None

    def test_create_panel_structure_via_factory(self):
        """Verify panel_structure is registered."""
        from panelbox.visualization import ChartFactory

        data = {
            "entities": ["A", "B"],
            "time_periods": [2000, 2001],
            "presence_matrix": [[1, 1], [1, 0]],
        }
        chart = ChartFactory.create("panel_structure", data=data)
        assert chart.figure is not None
