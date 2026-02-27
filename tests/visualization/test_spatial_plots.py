import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from panelbox.visualization.spatial_plots import (
    _create_lisa_bar_plot,
    create_moran_scatterplot,
    plot_direct_vs_indirect,
    plot_effects_comparison,
    plot_morans_i_by_period,
    plot_spatial_effects,
    plot_spatial_weights_structure,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def spatial_data():
    """Grid 5x5 with queen contiguity W."""
    np.random.seed(42)
    n = 25
    W = np.zeros((n, n))
    grid = 5
    for i in range(grid):
        for j in range(grid):
            idx = i * grid + j
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni_, nj_ = i + di, j + dj
                if 0 <= ni_ < grid and 0 <= nj_ < grid:
                    W[idx, ni_ * grid + nj_] = 1.0
    W = W / W.sum(axis=1, keepdims=True)
    values = np.random.randn(n)
    return values, W


@pytest.fixture
def mock_effects_result():
    """Mock SpatialEffectsResult with proper .effects attribute."""

    class MockEffectsResult:
        def __init__(self):
            self.effects = {
                "x1": {
                    "direct": 0.5,
                    "indirect": 0.2,
                    "total": 0.7,
                },
                "x2": {
                    "direct": -0.3,
                    "indirect": 0.1,
                    "total": -0.2,
                },
            }

    return MockEffectsResult()


# --- Group 1: Basic spatial plots ---


def test_moran_scatterplot(spatial_data):
    """Smoke test: create_moran_scatterplot."""
    values, W = spatial_data
    fig = create_moran_scatterplot(values, W)
    assert fig is not None


def test_lisa_bar_plot():
    """Smoke test: _create_lisa_bar_plot."""
    lisa_results = pd.DataFrame(
        {
            "Ii": np.random.randn(10),
            "cluster_type": ["HH", "LL", "HL", "LH", "Not significant"] * 2,
        },
        index=[f"unit_{i}" for i in range(10)],
    )
    color_map = {
        "HH": "#d7191c",
        "LL": "#2c7bb6",
        "HL": "#fdae61",
        "LH": "#abd9e9",
        "Not significant": "#ffffbf",
    }
    fig, ax = plt.subplots()
    _create_lisa_bar_plot(lisa_results, ax, color_map)
    assert fig is not None


def test_morans_i_by_period():
    """Smoke test: plot_morans_i_by_period with DataFrame input."""
    morans_df = pd.DataFrame(
        {
            "statistic": [0.3, 0.25, 0.35],
            "pvalue": [0.01, 0.05, 0.001],
            "expected_value": [-0.04, -0.04, -0.04],
        },
        index=["2020", "2021", "2022"],
    )
    fig = plot_morans_i_by_period(morans_df)
    assert fig is not None


def test_spatial_weights_structure(spatial_data):
    """Smoke test: plot_spatial_weights_structure."""
    _, W = spatial_data
    fig = plot_spatial_weights_structure(W)
    assert fig is not None


# --- Group 2: Effects and diagnostics plots ---


def test_spatial_effects(mock_effects_result):
    """Smoke test: plot_spatial_effects."""
    fig = plot_spatial_effects(mock_effects_result)
    assert fig is not None


def test_direct_vs_indirect(mock_effects_result):
    """Smoke test: plot_direct_vs_indirect."""
    fig = plot_direct_vs_indirect(mock_effects_result)
    assert fig is not None


def test_effects_comparison():
    """Smoke test: plot_effects_comparison."""

    class MockEffects:
        def __init__(self, effects):
            self.effects = effects

    result1 = MockEffects({"x1": {"direct": 0.5, "indirect": 0.2, "total": 0.7}})
    result2 = MockEffects({"x1": {"direct": 0.4, "indirect": 0.3, "total": 0.7}})
    fig = plot_effects_comparison(
        [result1, result2],
        model_names=["SAR", "SDM"],
        variables=["x1"],
    )
    assert fig is not None


def test_spatial_diagnostics_dashboard():
    """Smoke test: create_spatial_diagnostics_dashboard with mock data."""
    from panelbox.visualization.spatial_plots import (
        create_spatial_diagnostics_dashboard,
    )

    # Build mock objects matching expected structure
    class MockMoranResult:
        statistic = 0.3
        pvalue = 0.01
        additional_info = {"expected_value": -0.04}
        conclusion = "Significant spatial autocorrelation"

    lm_summary = pd.DataFrame(
        {
            "Test": ["LM-lag", "LM-error"],
            "Statistic": [5.0, 3.0],
            "p-value": [0.01, 0.05],
            "Significant": [True, True],
        }
    )

    lisa_df = pd.DataFrame(
        {
            "Ii": np.random.randn(25),
            "pvalue": np.random.rand(25),
            "cluster_type": np.random.choice(["HH", "LL", "HL", "LH", "Not significant"], 25),
        }
    )

    spatial_diagnostics = {
        "morans_i": MockMoranResult(),
        "lm_tests": {"summary": lm_summary, "reason": "LM-lag preferred"},
        "morans_i_local": lisa_df,
        "recommendation": "SAR model",
        "W": np.eye(25),
    }
    fig = create_spatial_diagnostics_dashboard(spatial_diagnostics)
    assert fig is not None
