import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def quantile_result_mock():
    """Mock quantile regression result with .results dict."""
    np.random.seed(42)

    class MockSingleResult:
        def __init__(self, tau, n_params=2):
            self.params = np.array([0.5 + tau, -0.3 * tau])
            self.bse = np.array([0.1, 0.05])

        def conf_int(self, alpha=0.05):
            from scipy import stats

            z = stats.norm.ppf(1 - alpha / 2)
            ci = np.column_stack([self.params - z * self.bse, self.params + z * self.bse])
            return ci

    class MockResult:
        def __init__(self):
            self.results = {tau: MockSingleResult(tau) for tau in [0.25, 0.50, 0.75]}

    return MockResult()


def test_init_default():
    """Test QuantileVisualizer default init."""
    viz = QuantileVisualizer()
    assert viz is not None
    assert viz.style == "academic"


def test_init_academic():
    """Test QuantileVisualizer with academic theme."""
    viz = QuantileVisualizer(style="academic")
    assert viz is not None


def test_init_presentation():
    """Test QuantileVisualizer with presentation theme."""
    viz = QuantileVisualizer(style="presentation")
    assert viz is not None


def test_init_minimal():
    """Test QuantileVisualizer with minimal theme."""
    viz = QuantileVisualizer(style="minimal")
    assert viz is not None


def test_coefficient_path(quantile_result_mock):
    """Smoke test: coefficient_path plot."""
    viz = QuantileVisualizer()
    fig = viz.coefficient_path(quantile_result_mock, var_names=["x1", "x2"])
    assert fig is not None


def test_fan_chart(quantile_result_mock):
    """Smoke test: fan_chart plot."""
    viz = QuantileVisualizer()
    X_forecast = np.column_stack([np.linspace(0, 1, 20), np.linspace(0, 1, 20)])
    fig = viz.fan_chart(quantile_result_mock, X_forecast=X_forecast)
    assert fig is not None


def test_spaghetti_plot(quantile_result_mock):
    """Smoke test: spaghetti_plot."""
    viz = QuantileVisualizer()
    fig = viz.spaghetti_plot(quantile_result_mock, sample_size=5)
    assert fig is not None
