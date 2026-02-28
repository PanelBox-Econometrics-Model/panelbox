"""Coverage tests for panelbox.visualization.spatial_plots module.

Targets uncovered lines: 177-232, 640-663, 862-864, 867, 873, 890-891
Focus on: create_lisa_cluster_map, plot_spatial_effects with CI,
plot_direct_vs_indirect, plot_effects_comparison
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def close_figs():
    """Close all figures after each test."""
    yield
    plt.close("all")


class MockSpatialEffectsResult:
    """Mock for SpatialEffectsResult."""

    def __init__(self, effects):
        self.effects = effects


def _make_effects(with_se=False):
    """Create test effects dict."""
    effects = {
        "x1": {"direct": 0.5, "indirect": 0.2, "total": 0.7},
        "x2": {"direct": -0.3, "indirect": 0.1, "total": -0.2},
    }
    if with_se:
        for v in effects.values():
            v["direct_se"] = 0.05
            v["indirect_se"] = 0.03
            v["total_se"] = 0.06
    return effects


class TestCreateMoranScatterplot:
    """Test create_moran_scatterplot function."""

    def test_basic(self):
        from panelbox.visualization.spatial_plots import create_moran_scatterplot

        np.random.seed(42)
        N = 30
        values = np.random.randn(N)
        W = np.eye(N)
        W = np.roll(W, 1, axis=1)
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

        fig = create_moran_scatterplot(values, W)
        assert fig is not None

    def test_with_ax(self):
        from panelbox.visualization.spatial_plots import create_moran_scatterplot

        np.random.seed(42)
        N = 20
        values = np.random.randn(N)
        W = np.eye(N)
        W = np.roll(W, 1, axis=1)
        np.fill_diagonal(W, 0)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

        fig_ext, ax_ext = plt.subplots()
        fig = create_moran_scatterplot(values, W, ax=ax_ext)
        assert fig is fig_ext

    def test_no_regression_no_quadrants(self):
        from panelbox.visualization.spatial_plots import create_moran_scatterplot

        np.random.seed(42)
        N = 20
        values = np.random.randn(N)
        W = np.eye(N)
        W = np.roll(W, 1, axis=1)
        np.fill_diagonal(W, 0)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

        fig = create_moran_scatterplot(values, W, show_regression=False, show_quadrants=False)
        assert fig is not None


class TestCreateLisaClusterMap:
    """Test create_lisa_cluster_map function - covers lines 177-232."""

    def test_without_gdf(self):
        from panelbox.visualization.spatial_plots import create_lisa_cluster_map

        lisa_df = pd.DataFrame(
            {
                "Ii": [1.2, -0.5, 0.8, -0.3, 0.1],
                "cluster_type": ["HH", "LL", "HL", "LH", "Not significant"],
            },
            index=["A", "B", "C", "D", "E"],
        )
        fig = create_lisa_cluster_map(lisa_df)
        assert fig is not None

    def test_without_gdf_many_entities(self):
        from panelbox.visualization.spatial_plots import create_lisa_cluster_map

        n = 50
        lisa_df = pd.DataFrame(
            {
                "Ii": np.random.randn(n),
                "cluster_type": np.random.choice(["HH", "LL", "HL", "LH", "Not significant"], n),
            },
            index=[f"E{i}" for i in range(n)],
        )
        fig = create_lisa_cluster_map(lisa_df)
        assert fig is not None

    def test_with_custom_color_map(self):
        from panelbox.visualization.spatial_plots import create_lisa_cluster_map

        lisa_df = pd.DataFrame(
            {
                "Ii": [1.0, -1.0],
                "cluster_type": ["HH", "LL"],
            },
            index=["A", "B"],
        )
        custom_colors = {"HH": "red", "LL": "blue", "Not significant": "gray"}
        fig = create_lisa_cluster_map(lisa_df, color_map=custom_colors)
        assert fig is not None

    def test_with_ax(self):
        from panelbox.visualization.spatial_plots import create_lisa_cluster_map

        lisa_df = pd.DataFrame(
            {
                "Ii": [0.5, -0.5],
                "cluster_type": ["HH", "LL"],
            },
            index=["A", "B"],
        )
        fig_ext, ax_ext = plt.subplots()
        fig = create_lisa_cluster_map(lisa_df, ax=ax_ext)
        assert fig is fig_ext

    def test_legend_false(self):
        from panelbox.visualization.spatial_plots import create_lisa_cluster_map

        lisa_df = pd.DataFrame(
            {
                "Ii": [0.5],
                "cluster_type": ["HH"],
            },
            index=["A"],
        )
        fig = create_lisa_cluster_map(lisa_df, legend=False)
        assert fig is not None


class TestPlotMoransIByPeriod:
    """Test plot_morans_i_by_period function."""

    def test_with_dataframe(self):
        from panelbox.visualization.spatial_plots import plot_morans_i_by_period

        df = pd.DataFrame(
            {
                "statistic": [0.3, 0.25, 0.4, 0.1],
                "pvalue": [0.01, 0.03, 0.001, 0.2],
                "expected_value": [-0.05, -0.05, -0.05, -0.05],
            },
            index=[2000, 2001, 2002, 2003],
        )
        fig = plot_morans_i_by_period(df)
        assert fig is not None

    def test_with_result_object(self):
        from panelbox.visualization.spatial_plots import plot_morans_i_by_period

        class MockResult:
            def __init__(self):
                self.results_by_period = {
                    2000: {"statistic": 0.3, "pvalue": 0.01, "expected_value": -0.05},
                    2001: {"statistic": 0.2, "pvalue": 0.08, "expected_value": -0.05},
                }

        fig = plot_morans_i_by_period(MockResult())
        assert fig is not None

    def test_no_expected_no_significance(self):
        from panelbox.visualization.spatial_plots import plot_morans_i_by_period

        df = pd.DataFrame(
            {
                "statistic": [0.3, 0.25],
                "pvalue": [0.1, 0.2],
            },
            index=[2000, 2001],
        )
        fig = plot_morans_i_by_period(df, show_expected=False, show_significance=False)
        assert fig is not None

    def test_no_significant_periods(self):
        from panelbox.visualization.spatial_plots import plot_morans_i_by_period

        df = pd.DataFrame(
            {
                "statistic": [0.1, 0.05],
                "pvalue": [0.5, 0.8],
            },
            index=[2000, 2001],
        )
        fig = plot_morans_i_by_period(df, show_significance=True)
        assert fig is not None


class TestPlotSpatialWeightsStructure:
    """Test plot_spatial_weights_structure function."""

    def test_small_matrix(self):
        from panelbox.visualization.spatial_plots import plot_spatial_weights_structure

        W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        fig = plot_spatial_weights_structure(W)
        assert fig is not None

    def test_large_matrix_no_grid(self):
        from panelbox.visualization.spatial_plots import plot_spatial_weights_structure

        W = np.random.rand(60, 60)
        fig = plot_spatial_weights_structure(W)
        assert fig is not None

    def test_no_colorbar(self):
        from panelbox.visualization.spatial_plots import plot_spatial_weights_structure

        W = np.array([[0, 1], [1, 0]], dtype=float)
        fig = plot_spatial_weights_structure(W, show_colorbar=False)
        assert fig is not None

    def test_with_ax(self):
        from panelbox.visualization.spatial_plots import plot_spatial_weights_structure

        W = np.array([[0, 1], [1, 0]], dtype=float)
        fig_ext, ax_ext = plt.subplots()
        fig = plot_spatial_weights_structure(W, ax=ax_ext)
        assert fig is fig_ext


class TestCreateSpatialDiagnosticsDashboard:
    """Test create_spatial_diagnostics_dashboard function."""

    def test_basic_dashboard(self):
        from panelbox.visualization.spatial_plots import (
            create_spatial_diagnostics_dashboard,
        )

        class MockMoransI:
            statistic = 0.35
            pvalue = 0.001
            additional_info = {"expected_value": -0.05}
            conclusion = "Significant positive spatial autocorrelation"

        N = 10
        spatial_diags = {
            "morans_i": MockMoransI(),
            "lm_tests": {
                "summary": pd.DataFrame(
                    {
                        "Test": ["LM-Error", "LM-Lag"],
                        "Statistic": [5.2, 3.1],
                        "p-value": [0.02, 0.08],
                        "Significant": [True, False],
                    }
                ),
                "reason": "LM-Error is significant",
            },
            "recommendation": "SEM",
            "morans_i_local": pd.DataFrame(
                {
                    "Ii": np.random.randn(N),
                    "pvalue": np.random.uniform(0, 0.2, N),
                    "cluster_type": np.random.choice(["HH", "LL", "Not significant"], N),
                }
            ),
            "W": np.random.rand(N, N),
        }
        fig = create_spatial_diagnostics_dashboard(spatial_diags)
        assert fig is not None


class TestPlotSpatialEffects:
    """Test plot_spatial_effects function - covers lines 640-663."""

    def test_basic_no_ci(self):
        from panelbox.visualization.spatial_plots import plot_spatial_effects

        result = MockSpatialEffectsResult(_make_effects(with_se=False))
        fig = plot_spatial_effects(result, show_ci=False)
        assert fig is not None

    def test_with_ci(self):
        from panelbox.visualization.spatial_plots import plot_spatial_effects

        result = MockSpatialEffectsResult(_make_effects(with_se=True))
        fig = plot_spatial_effects(result, show_ci=True)
        assert fig is not None

    def test_with_custom_colors(self):
        from panelbox.visualization.spatial_plots import plot_spatial_effects

        result = MockSpatialEffectsResult(_make_effects(with_se=False))
        custom_colors = {"direct": "red", "indirect": "blue", "total": "green"}
        fig = plot_spatial_effects(result, colors=custom_colors)
        assert fig is not None

    def test_many_variables_no_labels(self):
        from panelbox.visualization.spatial_plots import plot_spatial_effects

        effects = {}
        for i in range(8):
            effects[f"var{i}"] = {
                "direct": np.random.randn(),
                "indirect": np.random.randn(),
                "total": np.random.randn(),
            }
        result = MockSpatialEffectsResult(effects)
        fig = plot_spatial_effects(result)
        assert fig is not None

    def test_with_ax(self):
        from panelbox.visualization.spatial_plots import plot_spatial_effects

        result = MockSpatialEffectsResult(_make_effects(with_se=False))
        fig_ext, ax_ext = plt.subplots()
        fig = plot_spatial_effects(result, ax=ax_ext)
        assert fig is fig_ext


class TestPlotDirectVsIndirect:
    """Test plot_direct_vs_indirect function."""

    def test_basic(self):
        from panelbox.visualization.spatial_plots import plot_direct_vs_indirect

        result = MockSpatialEffectsResult(_make_effects())
        fig = plot_direct_vs_indirect(result)
        assert fig is not None

    def test_no_diagonal_no_labels(self):
        from panelbox.visualization.spatial_plots import plot_direct_vs_indirect

        result = MockSpatialEffectsResult(_make_effects())
        fig = plot_direct_vs_indirect(result, show_diagonal=False, show_labels=False)
        assert fig is not None

    def test_with_ax(self):
        from panelbox.visualization.spatial_plots import plot_direct_vs_indirect

        result = MockSpatialEffectsResult(_make_effects())
        fig_ext, ax_ext = plt.subplots()
        fig = plot_direct_vs_indirect(result, ax=ax_ext)
        assert fig is fig_ext


class TestPlotEffectsComparison:
    """Test plot_effects_comparison function."""

    def test_basic(self):
        from panelbox.visualization.spatial_plots import plot_effects_comparison

        r1 = MockSpatialEffectsResult(_make_effects())
        r2 = MockSpatialEffectsResult(_make_effects())
        fig = plot_effects_comparison([r1, r2], ["Model1", "Model2"])
        assert fig is not None

    def test_with_specified_variables(self):
        from panelbox.visualization.spatial_plots import plot_effects_comparison

        r1 = MockSpatialEffectsResult(_make_effects())
        r2 = MockSpatialEffectsResult(_make_effects())
        fig = plot_effects_comparison([r1, r2], ["M1", "M2"], variables=["x1"])
        assert fig is not None

    def test_with_se(self):
        from panelbox.visualization.spatial_plots import plot_effects_comparison

        r1 = MockSpatialEffectsResult(_make_effects(with_se=True))
        r2 = MockSpatialEffectsResult(_make_effects(with_se=True))
        fig = plot_effects_comparison([r1, r2], ["M1", "M2"])
        assert fig is not None

    def test_direct_effect_type(self):
        from panelbox.visualization.spatial_plots import plot_effects_comparison

        r1 = MockSpatialEffectsResult(_make_effects())
        fig = plot_effects_comparison([r1], ["M1"], effect_type="direct")
        assert fig is not None

    def test_no_common_variables_raises(self):
        from panelbox.visualization.spatial_plots import plot_effects_comparison

        e1 = {"x1": {"direct": 0.5, "indirect": 0.2, "total": 0.7}}
        e2 = {"x99": {"direct": 0.1, "indirect": 0.1, "total": 0.2}}
        r1 = MockSpatialEffectsResult(e1)
        r2 = MockSpatialEffectsResult(e2)
        with pytest.raises(ValueError, match="No common variables"):
            plot_effects_comparison([r1, r2], ["M1", "M2"])

    def test_with_ax(self):
        from panelbox.visualization.spatial_plots import plot_effects_comparison

        r1 = MockSpatialEffectsResult(_make_effects())
        fig_ext, ax_ext = plt.subplots()
        fig = plot_effects_comparison([r1], ["M1"], ax=ax_ext)
        assert fig is fig_ext
