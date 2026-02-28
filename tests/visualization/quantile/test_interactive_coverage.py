"""Coverage tests for panelbox.visualization.quantile.interactive module.

Targets uncovered lines: 69-70, 72, 439-440, 449, 470-471, 542-604
Focus on: coefficient_dashboard with bse/pvalues, quantile_surface_interactive,
parallel_coordinates
"""

import numpy as np
import pytest

plotly = pytest.importorskip("plotly")


class MockResultWithBse:
    """Mock result with bse and pvalues for full dashboard coverage."""

    def __init__(self, n_params=3, tau_list=None):
        if tau_list is None:
            tau_list = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.results = {}
        np.random.seed(42)
        for tau in tau_list:
            mock = type(
                "Res",
                (),
                {
                    "params": np.array([1.0 + tau, 0.5 - tau * 0.3, -0.2 + tau * 0.1]),
                    "bse": np.array([0.1, 0.05, 0.08]),
                    "pvalues": np.array([0.01, 0.05 * tau, 0.5]),
                },
            )()
            self.results[tau] = mock


class MockResultSimple:
    """Mock result with params only, no bse."""

    def __init__(self, n_params=3, tau_list=None):
        if tau_list is None:
            tau_list = [0.25, 0.5, 0.75]
        self.results = {}
        for tau in tau_list:
            mock = type(
                "Res",
                (),
                {
                    "params": np.array([1.0 + tau, 0.5 - tau, -0.2 + tau]),
                },
            )()
            self.results[tau] = mock


class MockResultArrayOnly:
    """Mock result with plain arrays (no params attribute)."""

    def __init__(self, tau_list=None):
        if tau_list is None:
            tau_list = [0.25, 0.5, 0.75]
        self.results = {}
        for tau in tau_list:
            self.results[tau] = np.array([1.0 + tau, 0.5 - tau])


class TestInteractivePlotterInit:
    """Test InteractivePlotter initialization."""

    def test_default_theme(self):
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        assert plotter.theme == "plotly_white"

    def test_custom_theme(self):
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter(theme="plotly_dark")
        assert plotter.theme == "plotly_dark"


class TestCoefficientDashboard:
    """Test coefficient_dashboard with various result types."""

    def test_with_bse_and_pvalues(self):
        """Cover lines 69-70, 72 (bse and pvalues branches)."""
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        result = MockResultWithBse()
        fig = plotter.coefficient_dashboard(result)
        assert fig is not None
        assert len(fig.data) > 3  # Multiple traces

    def test_without_bse(self):
        """Cover else branch for no bse."""
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        result = MockResultSimple()
        fig = plotter.coefficient_dashboard(result)
        assert fig is not None

    def test_with_array_results(self):
        """Cover lines 69-70: array result format."""
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        result = MockResultArrayOnly()
        fig = plotter.coefficient_dashboard(result)
        assert fig is not None

    def test_with_custom_var_names(self):
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        result = MockResultSimple()
        fig = plotter.coefficient_dashboard(result, var_names=["A", "B", "C"])
        assert fig is not None


class TestAnimatedCoefficientPath:
    """Test animated_coefficient_path method."""

    def test_basic(self):
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        result = MockResultSimple()
        fig = plotter.animated_coefficient_path(result, var_idx=0)
        assert fig is not None
        assert len(fig.frames) > 0

    def test_with_custom_var_name(self):
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        result = MockResultSimple()
        fig = plotter.animated_coefficient_path(result, var_idx=1, var_name="Education")
        assert fig is not None

    def test_with_array_results(self):
        """Cover the branch where result is array, not params object."""
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        result = MockResultArrayOnly()
        fig = plotter.animated_coefficient_path(result, var_idx=0)
        assert fig is not None


class TestQuantileSurfaceInteractive:
    """Test quantile_surface_interactive - covers lines 439-440, 449, 470-471."""

    def test_basic(self):
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        result = MockResultSimple()
        fig = plotter.quantile_surface_interactive(result)
        assert fig is not None

    def test_with_custom_grid(self):
        """Cover lines 439-440: X_grid branch."""
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        result = MockResultSimple()
        X_grid = np.array([np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)])
        fig = plotter.quantile_surface_interactive(result, X_grid=X_grid)
        assert fig is not None

    def test_with_array_results(self):
        """Cover lines 470-471: array results branch."""
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        result = MockResultArrayOnly()
        fig = plotter.quantile_surface_interactive(result)
        assert fig is not None


class TestParallelCoordinates:
    """Test parallel_coordinates - covers lines 542-604."""

    def test_basic(self):
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        result = MockResultSimple()
        fig = plotter.parallel_coordinates(result)
        assert fig is not None

    def test_with_many_quantiles_sampling(self):
        """Cover the sampling branch when len(tau_list) > n_samples."""
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        # Create result with many quantiles
        tau_list = [round(t, 2) for t in np.linspace(0.05, 0.95, 50)]
        result = MockResultSimple(tau_list=tau_list)
        fig = plotter.parallel_coordinates(result, n_samples=10)
        assert fig is not None

    def test_with_array_results(self):
        """Cover array result branch in parallel_coordinates."""
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        result = MockResultArrayOnly()
        fig = plotter.parallel_coordinates(result)
        assert fig is not None

    def test_with_bse_result(self):
        from panelbox.visualization.quantile.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        result = MockResultWithBse()
        fig = plotter.parallel_coordinates(result)
        assert fig is not None
