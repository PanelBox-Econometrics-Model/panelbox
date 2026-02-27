import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from panelbox.visualization.quantile.process_plots import (
    qq_plot,
    quantile_process_plot,
    residual_plot,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def test_quantile_process_plot():
    """Smoke test: quantile_process_plot."""
    np.random.seed(42)
    quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    n_vars = 2
    params = np.random.randn(n_vars, len(quantiles))
    fig, ax = quantile_process_plot(quantiles, params)
    assert fig is not None
    assert ax is not None


def test_quantile_process_plot_with_se():
    """Smoke test: quantile_process_plot with standard errors."""
    np.random.seed(42)
    quantiles = np.array([0.25, 0.5, 0.75])
    n_vars = 2
    params = np.random.randn(n_vars, len(quantiles))
    std_errors = np.abs(np.random.randn(n_vars, len(quantiles))) * 0.1
    fig, _ax = quantile_process_plot(quantiles, params, std_errors=std_errors)
    assert fig is not None


def test_residual_plot():
    """Smoke test: residual_plot."""
    np.random.seed(42)
    residuals = np.random.randn(50)
    fig, ax = residual_plot(residuals)
    assert fig is not None
    assert ax is not None


def test_qq_plot():
    """Smoke test: qq_plot."""
    np.random.seed(42)
    residuals = np.random.randn(50)
    fig, ax = qq_plot(residuals)
    assert fig is not None
    assert ax is not None
