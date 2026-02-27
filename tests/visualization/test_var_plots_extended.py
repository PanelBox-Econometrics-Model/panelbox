import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from panelbox.visualization.var_plots import (
    _get_theme_config,
    plot_fevd,
    plot_instrument_sensitivity,
    plot_irf,
    plot_stability,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def mock_irf_result():
    """Mock IRFResult object with required attributes."""
    np.random.seed(42)
    n_steps = 10
    n_vars = 2

    class MockIRFResult:
        def __init__(self):
            self.irf_matrix = np.random.randn(n_steps + 1, n_vars, n_vars)
            self.var_names = ["var1", "var2"]
            self.periods = n_steps
            self.K = n_vars
            self.method = "cholesky"
            self.ci_lower = np.random.randn(n_steps + 1, n_vars, n_vars) - 0.5
            self.ci_upper = np.random.randn(n_steps + 1, n_vars, n_vars) + 0.5
            self.ci_level = 0.95

        def __getitem__(self, key):
            resp, imp = key
            resp_idx = self.var_names.index(resp)
            imp_idx = self.var_names.index(imp)
            return self.irf_matrix[:, resp_idx, imp_idx]

    return MockIRFResult()


@pytest.fixture
def mock_fevd_result():
    """Mock FEVDResult object with valid decomposition."""
    np.random.seed(42)
    n_steps = 10
    n_vars = 2

    class MockFEVDResult:
        def __init__(self):
            # Each row must sum to 1.0 for valid FEVD
            raw = np.random.dirichlet([1, 1], size=(n_steps + 1, n_vars))
            self.decomposition = raw.reshape(n_steps + 1, n_vars, n_vars)
            self.var_names = ["var1", "var2"]
            self.periods = n_steps
            self.K = n_vars
            self.method = "cholesky"
            self.ci_lower = None
            self.ci_upper = None
            self.ci_level = None

    return MockFEVDResult()


@pytest.fixture
def eigenvalues():
    """Mock eigenvalues for stability plot."""
    return np.array([0.5 + 0.3j, 0.5 - 0.3j, 0.2, -0.1])


# --- Theme config tests ---


def test_get_theme_config_academic():
    """Test _get_theme_config returns valid config for academic theme."""
    config = _get_theme_config("academic")
    assert isinstance(config, dict)
    assert "fontsize" in config
    assert "line_color" in config


def test_get_theme_config_presentation():
    """Test _get_theme_config returns valid config for presentation theme."""
    config = _get_theme_config("presentation")
    assert isinstance(config, dict)


def test_get_theme_config_professional():
    """Test _get_theme_config returns valid config for professional theme."""
    config = _get_theme_config("professional")
    assert isinstance(config, dict)


# --- Stability plot tests ---


def test_plot_stability_matplotlib(eigenvalues):
    """Smoke test: plot_stability with matplotlib backend."""
    fig = plot_stability(eigenvalues, backend="matplotlib", show=False)
    assert fig is not None


def test_plot_stability_plotly(eigenvalues):
    """Smoke test: plot_stability with plotly backend."""
    fig = plot_stability(eigenvalues, backend="plotly", show=False)
    assert fig is not None
    assert hasattr(fig, "data")


# --- IRF plot tests ---


def test_plot_irf_matplotlib(mock_irf_result):
    """Smoke test: plot_irf with matplotlib backend."""
    fig = plot_irf(mock_irf_result, backend="matplotlib", show=False)
    assert fig is not None


def test_plot_irf_plotly(mock_irf_result):
    """Smoke test: plot_irf with plotly backend."""
    fig = plot_irf(mock_irf_result, backend="plotly", show=False)
    assert fig is not None
    assert hasattr(fig, "data")


# --- FEVD plot tests ---


def test_plot_fevd_matplotlib(mock_fevd_result):
    """Smoke test: plot_fevd with matplotlib backend."""
    fig = plot_fevd(mock_fevd_result, backend="matplotlib", show=False)
    assert fig is not None


def test_plot_fevd_plotly(mock_fevd_result):
    """Smoke test: plot_fevd with plotly backend."""
    fig = plot_fevd(mock_fevd_result, backend="plotly", show=False)
    assert fig is not None


# --- Instrument sensitivity test ---


def test_plot_instrument_sensitivity():
    """Smoke test: plot_instrument_sensitivity."""
    data = {
        "coefficients": {
            "lag1": [0.5, 0.48, 0.52],
            "lag2": [0.3, 0.28, 0.32],
        },
        "n_instruments_actual": [10, 15, 20],
    }
    fig = plot_instrument_sensitivity(data, show=False)
    assert fig is not None
