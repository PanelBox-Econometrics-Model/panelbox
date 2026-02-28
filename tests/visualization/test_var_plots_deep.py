"""Coverage tests for panelbox.visualization.var_plots module.

Targets remaining uncovered lines: 23-24, 79, 171-172, 287-288, 351,
411-412, 435, 495-496, 558, 675-676, 693-694, 812-813, 899, 1022-1023,
1039-1040, 1138-1139

Focus on: show=True paths (plt.show/fig.show), HAS_PLOTLY=False fallback,
plotly import error in submodule functions, empty changes dict in sensitivity.
"""

import matplotlib

matplotlib.use("Agg")
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def close_figs():
    """Close all figures after each test."""
    yield
    plt.close("all")


# ============================================================
# Helpers: mock data for IRF and FEVD results
# ============================================================


def _make_irf_result():
    """Create a mock IRFResult-like object."""
    periods = 10
    n_vars = 2
    var_names = ["y1", "y2"]
    rng = np.random.RandomState(42)
    # irf_matrix shape: (periods+1, n_vars, n_vars)
    irf_matrix = rng.randn(periods + 1, n_vars, n_vars) * 0.1

    class MockIRFResult:
        def __init__(self):
            self.irf_matrix = irf_matrix
            self.var_names = var_names
            self.periods = periods
            self.ci_lower = irf_matrix - 0.05
            self.ci_upper = irf_matrix + 0.05
            self.ci_level = 0.95
            self.method = "oirf"

        def __getitem__(self, key):
            response_var, impulse_var = key
            resp_idx = self.var_names.index(response_var)
            imp_idx = self.var_names.index(impulse_var)
            return self.irf_matrix[:, resp_idx, imp_idx]

    return MockIRFResult()


def _make_fevd_result():
    """Create a mock FEVDResult-like object."""
    periods = 10
    n_vars = 2
    var_names = ["y1", "y2"]
    # decomposition shape: (periods+1, n_vars, n_vars) - [horizon, target_var, source_var]
    rng = np.random.RandomState(42)
    decomp = np.zeros((periods + 1, n_vars, n_vars))
    for t in range(periods + 1):
        for i in range(n_vars):
            vals = rng.dirichlet(np.ones(n_vars))
            decomp[t, i, :] = vals

    class MockFEVDResult:
        def __init__(self):
            self.decomposition = decomp
            self.var_names = var_names
            self.periods = periods
            self.K = n_vars
            self.method = "cholesky"

    return MockFEVDResult()


def _make_sensitivity_results(with_changes=True, many_coefs=False):
    """Create sensitivity_results dict for plot_instrument_sensitivity."""
    n_coefs = 8 if many_coefs else 3
    coefficients = {}
    for i in range(n_coefs):
        coefficients[f"coef_{i}"] = [0.5 + i * 0.1, 0.6 + i * 0.1, 0.55 + i * 0.1]

    result = {
        "coefficients": coefficients,
        "n_instruments_actual": [5, 10, 15],
    }
    if with_changes:
        result["coefficient_changes"] = {f"coef_{i}": abs(0.1 * (i + 1)) for i in range(n_coefs)}
    return result


# ============================================================
# Tests for show=True matplotlib paths (plt.show called)
# Lines: 171-172, 411-412, 675-676, 1022-1023
# ============================================================


class TestStabilityShowTrue:
    """Cover lines 170-172: _plot_stability_matplotlib show=True."""

    def test_show_true_returns_none(self):
        from panelbox.visualization.var_plots import plot_stability

        eigenvalues = np.array([0.5 + 0.3j, 0.5 - 0.3j, -0.4, 0.2])
        with patch.object(plt, "show"):
            result = plot_stability(eigenvalues, show=True)
        assert result is None


class TestSensitivityShowTrue:
    """Cover lines 410-412: _plot_sensitivity_matplotlib show=True."""

    def test_show_true_returns_none(self):
        from panelbox.visualization.var_plots import plot_instrument_sensitivity

        data = _make_sensitivity_results()
        with patch.object(plt, "show"):
            result = plot_instrument_sensitivity(data, show=True)
        assert result is None


class TestIRFShowTrue:
    """Cover lines 674-676: _plot_irf_matplotlib show=True."""

    def test_show_true_returns_none(self):
        from panelbox.visualization.var_plots import plot_irf

        irf = _make_irf_result()
        with patch.object(plt, "show"):
            result = plot_irf(irf, show=True)
        assert result is None


class TestFEVDShowTrue:
    """Cover lines 1021-1023: _plot_fevd_matplotlib show=True."""

    def test_show_true_returns_none(self):
        from panelbox.visualization.var_plots import plot_fevd

        fevd = _make_fevd_result()
        with patch.object(plt, "show"):
            result = plot_fevd(fevd, show=True)
        assert result is None


# ============================================================
# Tests for show=True plotly paths (fig.show called)
# Lines: 287-288, 495-496, 812-813, 1138-1139
# ============================================================


class TestStabilityPlotlyShowTrue:
    """Cover lines 286-288: _plot_stability_plotly show=True."""

    def test_show_true_returns_none(self):
        pytest.importorskip("plotly")
        from panelbox.visualization.var_plots import plot_stability

        eigenvalues = np.array([0.5 + 0.3j, 0.5 - 0.3j, -0.4])
        with patch("plotly.graph_objects.Figure.show"):
            result = plot_stability(eigenvalues, backend="plotly", show=True)
        assert result is None


class TestSensitivityPlotlyShowTrue:
    """Cover lines 494-496: _plot_sensitivity_plotly show=True."""

    def test_show_true_returns_none(self):
        pytest.importorskip("plotly")
        from panelbox.visualization.var_plots import plot_instrument_sensitivity

        data = _make_sensitivity_results()
        with patch("plotly.graph_objects.Figure.show"):
            result = plot_instrument_sensitivity(data, backend="plotly", show=True)
        assert result is None


class TestIRFPlotlyShowTrue:
    """Cover lines 811-813: _plot_irf_plotly show=True."""

    def test_show_true_returns_none(self):
        pytest.importorskip("plotly")
        from panelbox.visualization.var_plots import plot_irf

        irf = _make_irf_result()
        with patch("plotly.graph_objects.Figure.show"):
            result = plot_irf(irf, backend="plotly", show=True)
        assert result is None


class TestFEVDPlotlyShowTrue:
    """Cover lines 1137-1139: _plot_fevd_plotly show=True."""

    def test_show_true_returns_none(self):
        pytest.importorskip("plotly")
        from panelbox.visualization.var_plots import plot_fevd

        fevd = _make_fevd_result()
        with patch("plotly.graph_objects.Figure.show"):
            result = plot_fevd(fevd, backend="plotly", show=True)
        assert result is None


# ============================================================
# Tests for HAS_PLOTLY=False paths
# Lines: 79, 351, 558, 899
# ============================================================


class TestPlotlyNotInstalled:
    """Cover lines 79, 351, 558, 899: HAS_PLOTLY=False branches."""

    def test_stability_plotly_not_installed(self):
        from panelbox.visualization import var_plots

        eigenvalues = np.array([0.5, -0.3])
        with (
            patch.object(var_plots, "HAS_PLOTLY", False),
            pytest.raises(ImportError, match="Plotly is required"),
        ):
            var_plots.plot_stability(eigenvalues, backend="plotly")

    def test_sensitivity_plotly_not_installed(self):
        from panelbox.visualization import var_plots

        data = _make_sensitivity_results()
        with (
            patch.object(var_plots, "HAS_PLOTLY", False),
            pytest.raises(ImportError, match="Plotly is required"),
        ):
            var_plots.plot_instrument_sensitivity(data, backend="plotly")

    def test_irf_plotly_not_installed(self):
        from panelbox.visualization import var_plots

        irf = _make_irf_result()
        with (
            patch.object(var_plots, "HAS_PLOTLY", False),
            pytest.raises(ImportError, match="Plotly is required"),
        ):
            var_plots.plot_irf(irf, backend="plotly")

    def test_fevd_plotly_not_installed(self):
        from panelbox.visualization import var_plots

        fevd = _make_fevd_result()
        with (
            patch.object(var_plots, "HAS_PLOTLY", False),
            pytest.raises(ImportError, match="Plotly is required"),
        ):
            var_plots.plot_fevd(fevd, backend="plotly")


# ============================================================
# Test for empty changes dict in sensitivity plotly
# Line: 435
# ============================================================


class TestSensitivityPlotlyEmptyChanges:
    """Cover line 435: empty coefficient_changes in _plot_sensitivity_plotly."""

    def test_empty_changes_fallback(self):
        pytest.importorskip("plotly")
        from panelbox.visualization.var_plots import plot_instrument_sensitivity

        # many_coefs=True so len(coef_names) > max_coefs_to_plot triggers
        # the branch checking changes dict; with_changes=False => empty
        data = _make_sensitivity_results(with_changes=False, many_coefs=True)
        fig = plot_instrument_sensitivity(data, backend="plotly", show=False, max_coefs_to_plot=3)
        assert fig is not None
