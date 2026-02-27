import numpy as np
import pytest


@pytest.fixture
def quantile_result_for_interactive():
    """Mock quantile results for interactive plots."""
    np.random.seed(42)

    class MockSingleResult:
        def __init__(self, tau, n_params=2):
            self.params = np.array([0.5 + tau, -0.3 * tau])
            self.bse = np.array([0.1, 0.05])
            self.pvalues = np.array([0.01, 0.05 * tau])

    class MockResult:
        def __init__(self):
            self.results = {tau: MockSingleResult(tau) for tau in [0.25, 0.50, 0.75]}

    return MockResult()


def test_coefficient_dashboard(quantile_result_for_interactive):
    """Smoke test: coefficient_dashboard."""
    from panelbox.visualization.quantile.interactive import InteractivePlotter

    plotter = InteractivePlotter()
    fig = plotter.coefficient_dashboard(quantile_result_for_interactive)
    assert fig is not None
    assert hasattr(fig, "data")


def test_animated_coefficient_path(quantile_result_for_interactive):
    """Smoke test: animated_coefficient_path."""
    from panelbox.visualization.quantile.interactive import InteractivePlotter

    plotter = InteractivePlotter()
    fig = plotter.animated_coefficient_path(
        quantile_result_for_interactive, var_idx=0, var_name="x1"
    )
    assert fig is not None
    assert hasattr(fig, "data")


def test_quantile_surface_interactive(quantile_result_for_interactive):
    """Smoke test: quantile_surface_interactive."""
    from panelbox.visualization.quantile.interactive import InteractivePlotter

    plotter = InteractivePlotter()
    fig = plotter.quantile_surface_interactive(quantile_result_for_interactive)
    assert fig is not None
    assert hasattr(fig, "data")
