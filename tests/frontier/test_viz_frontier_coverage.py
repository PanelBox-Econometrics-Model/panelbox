"""Tests for panelbox.frontier.visualization.frontier_plots module.

Covers frontier 2D (matplotlib), 3D (plotly), contour, partial,
and error paths to increase coverage from ~66% to 80%+.
"""

import matplotlib

matplotlib.use("Agg")
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def close_figs():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def mock_sf_result():
    """Create a mock SFResult for frontier plotting functions."""
    np.random.seed(42)
    n_obs = 50

    result = MagicMock()

    # Input data
    x1 = np.random.uniform(1, 5, n_obs)
    x2 = np.random.uniform(1, 5, n_obs)
    X_df = pd.DataFrame({"log_labor": x1, "log_capital": x2})
    result.model.X_df = X_df
    result.model.exog = ["log_labor", "log_capital"]

    # Design matrix with intercept
    X_matrix = np.column_stack([np.ones(n_obs), x1, x2])
    result.model.X = X_matrix

    # True coefficients
    beta = np.array([1.0, 0.5, 0.3])
    noise = np.random.normal(0, 0.1, n_obs)
    y = X_matrix @ beta + noise
    result.model.y = y

    # Parameters (including sigma params that should be excluded)
    params = pd.Series(
        [1.0, 0.5, 0.3, 0.15, 0.20],
        index=["const", "log_labor", "log_capital", "sigma_v", "sigma_u"],
    )
    result.params = params

    # Frontier type
    result.model.frontier_type.value = "production"

    # Efficiency method
    eff_values = np.random.uniform(0.6, 0.95, n_obs)
    eff_df = pd.DataFrame({"efficiency": eff_values})
    result.efficiency.return_value = eff_df

    return result


class TestPlotFrontier2d:
    """Tests for plot_frontier_2d."""

    def test_matplotlib_backend(self, mock_sf_result):
        """Test 2D frontier with matplotlib backend."""
        from panelbox.frontier.visualization.frontier_plots import plot_frontier_2d

        fig = plot_frontier_2d(mock_sf_result, input_var="log_labor", backend="matplotlib")
        assert fig is not None

    def test_matplotlib_with_distance(self, mock_sf_result):
        """Test 2D frontier with show_distance on matplotlib."""
        from panelbox.frontier.visualization.frontier_plots import plot_frontier_2d

        fig = plot_frontier_2d(
            mock_sf_result,
            input_var="log_labor",
            backend="matplotlib",
            show_distance=True,
            n_observations=20,
        )
        assert fig is not None

    def test_plotly_with_distance(self, mock_sf_result):
        """Test 2D frontier with plotly and show_distance."""
        from panelbox.frontier.visualization.frontier_plots import plot_frontier_2d

        fig = plot_frontier_2d(
            mock_sf_result,
            input_var="log_labor",
            backend="plotly",
            show_distance=True,
            n_observations=10,
            title="Custom 2D",
        )
        assert fig is not None

    def test_invalid_input_var(self, mock_sf_result):
        """Test invalid input variable raises ValueError."""
        from panelbox.frontier.visualization.frontier_plots import plot_frontier_2d

        with pytest.raises(ValueError, match="not found"):
            plot_frontier_2d(mock_sf_result, input_var="nonexistent")

    def test_invalid_backend(self, mock_sf_result):
        """Test invalid backend raises ValueError."""
        from panelbox.frontier.visualization.frontier_plots import plot_frontier_2d

        with pytest.raises(ValueError, match="Unknown backend"):
            plot_frontier_2d(mock_sf_result, input_var="log_labor", backend="bad")


class TestPlotFrontier3d:
    """Tests for plot_frontier_3d."""

    def test_plotly_backend(self, mock_sf_result):
        """Test 3D frontier with plotly backend."""
        from panelbox.frontier.visualization.frontier_plots import plot_frontier_3d

        fig = plot_frontier_3d(
            mock_sf_result,
            input_vars=["log_labor", "log_capital"],
            backend="plotly",
            n_grid=10,
        )
        assert fig is not None

    def test_matplotlib_backend(self, mock_sf_result):
        """Test 3D frontier with matplotlib backend."""
        from panelbox.frontier.visualization.frontier_plots import plot_frontier_3d

        fig = plot_frontier_3d(
            mock_sf_result,
            input_vars=["log_labor", "log_capital"],
            backend="matplotlib",
            n_grid=10,
        )
        assert fig is not None

    def test_wrong_number_of_vars(self, mock_sf_result):
        """Test providing != 2 input vars raises ValueError."""
        from panelbox.frontier.visualization.frontier_plots import plot_frontier_3d

        with pytest.raises(ValueError, match="exactly 2"):
            plot_frontier_3d(mock_sf_result, input_vars=["log_labor"])

    def test_invalid_backend(self, mock_sf_result):
        """Test invalid backend raises ValueError."""
        from panelbox.frontier.visualization.frontier_plots import plot_frontier_3d

        with pytest.raises(ValueError, match="Unknown backend"):
            plot_frontier_3d(
                mock_sf_result,
                input_vars=["log_labor", "log_capital"],
                backend="bad",
            )


class TestPlotFrontierContour:
    """Tests for plot_frontier_contour."""

    def test_plotly_backend(self, mock_sf_result):
        """Test contour plot with plotly backend."""
        from panelbox.frontier.visualization.frontier_plots import (
            plot_frontier_contour,
        )

        fig = plot_frontier_contour(
            mock_sf_result,
            input_vars=["log_labor", "log_capital"],
            backend="plotly",
            n_grid=10,
        )
        assert fig is not None

    def test_matplotlib_backend(self, mock_sf_result):
        """Test contour plot with matplotlib backend."""
        from panelbox.frontier.visualization.frontier_plots import (
            plot_frontier_contour,
        )

        fig = plot_frontier_contour(
            mock_sf_result,
            input_vars=["log_labor", "log_capital"],
            backend="matplotlib",
            n_grid=10,
        )
        assert fig is not None

    def test_invalid_backend(self, mock_sf_result):
        """Test invalid backend raises ValueError."""
        from panelbox.frontier.visualization.frontier_plots import (
            plot_frontier_contour,
        )

        with pytest.raises(ValueError, match="Unknown backend"):
            plot_frontier_contour(
                mock_sf_result,
                input_vars=["log_labor", "log_capital"],
                backend="bad",
            )


class TestPlotFrontierPartial:
    """Tests for plot_frontier_partial."""

    def test_plotly_fix_at_mean(self, mock_sf_result):
        """Test partial frontier with plotly, fix_others_at='mean'."""
        from panelbox.frontier.visualization.frontier_plots import (
            plot_frontier_partial,
        )

        fig = plot_frontier_partial(
            mock_sf_result,
            input_var="log_labor",
            fix_others_at="mean",
            backend="plotly",
        )
        assert fig is not None

    def test_matplotlib_fix_at_median(self, mock_sf_result):
        """Test partial frontier with matplotlib, fix_others_at='median'."""
        from panelbox.frontier.visualization.frontier_plots import (
            plot_frontier_partial,
        )

        fig = plot_frontier_partial(
            mock_sf_result,
            input_var="log_labor",
            fix_others_at="median",
            backend="matplotlib",
        )
        assert fig is not None

    def test_fix_at_dict(self, mock_sf_result):
        """Test partial frontier with dict of fixed values."""
        from panelbox.frontier.visualization.frontier_plots import (
            plot_frontier_partial,
        )

        fig = plot_frontier_partial(
            mock_sf_result,
            input_var="log_labor",
            fix_others_at={"log_capital": 3.0},
            backend="plotly",
        )
        assert fig is not None

    def test_invalid_fix_others_string(self, mock_sf_result):
        """Test invalid fix_others_at string raises ValueError."""
        from panelbox.frontier.visualization.frontier_plots import (
            plot_frontier_partial,
        )

        with pytest.raises(ValueError, match="Unknown fix_others_at"):
            plot_frontier_partial(
                mock_sf_result,
                input_var="log_labor",
                fix_others_at="invalid",
            )

    def test_invalid_fix_others_type(self, mock_sf_result):
        """Test invalid fix_others_at type raises ValueError."""
        from panelbox.frontier.visualization.frontier_plots import (
            plot_frontier_partial,
        )

        with pytest.raises(ValueError, match="must be 'mean', 'median', or dict"):
            plot_frontier_partial(
                mock_sf_result,
                input_var="log_labor",
                fix_others_at=42,
            )

    def test_invalid_backend(self, mock_sf_result):
        """Test invalid backend raises ValueError."""
        from panelbox.frontier.visualization.frontier_plots import (
            plot_frontier_partial,
        )

        with pytest.raises(ValueError, match="Unknown backend"):
            plot_frontier_partial(
                mock_sf_result,
                input_var="log_labor",
                backend="bad",
            )
