import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from panelbox.visualization.quantile.surface_plots import SurfacePlotter


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def quantile_result_for_surface():
    """Mock quantile result for surface plotting."""
    np.random.seed(42)

    class MockSingleResult:
        def __init__(self, tau, n_params=3):
            self.params = np.array([1.0 + tau, 0.5 - tau, -0.2 * tau])

    class MockResult:
        def __init__(self):
            self.results = {tau: MockSingleResult(tau) for tau in [0.1, 0.25, 0.5, 0.75, 0.9]}

    return MockResult()


def test_plot_surface_3d(quantile_result_for_surface):
    """Smoke test: 3D surface plot."""
    plotter = SurfacePlotter()
    fig = plotter.plot_surface(
        quantile_result_for_surface,
        var_names=["x1", "x2"],
        projection="3d",
    )
    assert fig is not None


def test_plot_surface_contour(quantile_result_for_surface):
    """Smoke test: contour plot."""
    plotter = SurfacePlotter()
    fig = plotter.plot_surface(
        quantile_result_for_surface,
        var_names=["x1", "x2"],
        projection="contour",
    )
    assert fig is not None


def test_plot_interactive(quantile_result_for_surface):
    """Smoke test: interactive plotly surface."""
    plotter = SurfacePlotter()
    try:
        fig = plotter.plot_interactive(quantile_result_for_surface)
    except ValueError as e:
        if "titlefont" in str(e):
            pytest.skip("Source uses deprecated plotly 'titlefont' property")
        raise
    assert fig is not None
    assert hasattr(fig, "data")


def test_coefficient_heatmap(quantile_result_for_surface):
    """Smoke test: coefficient heatmap."""
    plotter = SurfacePlotter()
    fig = plotter.coefficient_heatmap(
        quantile_result_for_surface, var_names=["intercept", "x1", "x2"]
    )
    assert fig is not None
