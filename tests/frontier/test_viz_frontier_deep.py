"""
Deep coverage tests for frontier/visualization/frontier_plots.py.

Targets specific uncovered lines and branch partials to push
coverage from ~94.88% toward 100%.

Uncovered items:
- Line 26: _get_X_df else branch (model without X_df)
- Line 262: ValueError for invalid var in 3D plot
- Line 423: ValueError for wrong num vars in contour
- Line 429: ValueError for invalid var in contour
- Line 596: ValueError for invalid var in partial
- Branch 146->159: plotly distance lines (show_distance=True)
- Branch 301->305: plotly 3D missing title (cost frontier)
- Branch 465->468: plotly contour custom title
- Branch 635->634: fixed_values dict with var not in X_df
- Branch 646->649: partial with custom title
"""

import matplotlib

matplotlib.use("Agg")

from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from panelbox.frontier.visualization.frontier_plots import (
    _get_X_df,
    plot_frontier_2d,
    plot_frontier_3d,
    plot_frontier_contour,
    plot_frontier_partial,
)


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

    x1 = np.random.uniform(1, 5, n_obs)
    x2 = np.random.uniform(1, 5, n_obs)
    X_df = pd.DataFrame({"log_labor": x1, "log_capital": x2})
    result.model.X_df = X_df
    result.model.exog = ["log_labor", "log_capital"]

    X_matrix = np.column_stack([np.ones(n_obs), x1, x2])
    result.model.X = X_matrix

    beta = np.array([1.0, 0.5, 0.3])
    noise = np.random.normal(0, 0.1, n_obs)
    y = X_matrix @ beta + noise
    result.model.y = y

    params = pd.Series(
        [1.0, 0.5, 0.3, 0.15, 0.20],
        index=["const", "log_labor", "log_capital", "sigma_v", "sigma_u"],
    )
    result.params = params

    result.model.frontier_type.value = "production"

    eff_values = np.random.uniform(0.6, 0.95, n_obs)
    eff_df = pd.DataFrame({"efficiency": eff_values})
    result.efficiency.return_value = eff_df

    return result


class TestGetXDf:
    """Test _get_X_df helper function."""

    def test_with_X_df_attribute(self, mock_sf_result):
        """Test when model has X_df attribute (covered path)."""
        X_df = _get_X_df(mock_sf_result)
        assert isinstance(X_df, pd.DataFrame)
        assert "log_labor" in X_df.columns

    def test_without_X_df_attribute(self):
        """Cover line 26: model without X_df, uses data[exog]."""
        result = MagicMock(spec=[])
        result.model = MagicMock(spec=["data", "exog"])
        del result.model.X_df  # Ensure no X_df attribute
        result.model.exog = ["log_labor", "log_capital"]
        result.model.data = pd.DataFrame(
            {
                "log_labor": [1.0, 2.0, 3.0],
                "log_capital": [4.0, 5.0, 6.0],
                "y": [7.0, 8.0, 9.0],
            }
        )
        X_df = _get_X_df(result)
        assert isinstance(X_df, pd.DataFrame)
        assert list(X_df.columns) == ["log_labor", "log_capital"]


class TestPlotFrontier2dBranches:
    """Test uncovered branches in plot_frontier_2d."""

    def test_plotly_no_distance(self, mock_sf_result):
        """Cover branch 146->159: show_distance=False in plotly (False path)."""
        fig = plot_frontier_2d(
            mock_sf_result,
            input_var="log_labor",
            backend="plotly",
            show_distance=False,
        )
        assert fig is not None


class TestPlotFrontier3dBranches:
    """Test uncovered branches in plot_frontier_3d."""

    def test_invalid_var_name(self, mock_sf_result):
        """Cover line 262: ValueError for invalid input var."""
        with pytest.raises(ValueError, match="not found in model data"):
            plot_frontier_3d(
                mock_sf_result,
                input_vars=["log_labor", "nonexistent_var"],
                backend="plotly",
            )

    def test_cost_frontier_title(self, mock_sf_result):
        """Cover branch 301->305: cost frontier auto-generates 'Cost' title."""
        mock_sf_result.model.frontier_type.value = "cost"
        fig = plot_frontier_3d(
            mock_sf_result,
            input_vars=["log_labor", "log_capital"],
            backend="plotly",
        )
        assert fig is not None

    def test_custom_title_3d(self, mock_sf_result):
        """Cover branch 301->305: custom title skips auto-generation."""
        fig = plot_frontier_3d(
            mock_sf_result,
            input_vars=["log_labor", "log_capital"],
            backend="plotly",
            title="My Custom 3D Title",
        )
        assert fig is not None


class TestPlotFrontierContourBranches:
    """Test uncovered branches in plot_frontier_contour."""

    def test_wrong_number_of_vars(self, mock_sf_result):
        """Cover line 423: ValueError for != 2 vars."""
        with pytest.raises(ValueError, match="exactly 2"):
            plot_frontier_contour(
                mock_sf_result,
                input_vars=["log_labor"],
                backend="plotly",
            )

    def test_invalid_var_in_contour(self, mock_sf_result):
        """Cover line 429: ValueError for invalid var."""
        with pytest.raises(ValueError, match="not found in model data"):
            plot_frontier_contour(
                mock_sf_result,
                input_vars=["log_labor", "missing_var"],
                backend="plotly",
            )

    def test_custom_title(self, mock_sf_result):
        """Cover branch 465->468: custom title provided."""
        fig = plot_frontier_contour(
            mock_sf_result,
            input_vars=["log_labor", "log_capital"],
            backend="plotly",
            title="My Custom Contour",
        )
        assert fig is not None


class TestPlotFrontierPartialBranches:
    """Test uncovered branches in plot_frontier_partial."""

    def test_invalid_input_var(self, mock_sf_result):
        """Cover line 596: ValueError for invalid input var."""
        with pytest.raises(ValueError, match="not found in model data"):
            plot_frontier_partial(
                mock_sf_result,
                input_var="nonexistent_var",
                backend="plotly",
            )

    def test_dict_with_unknown_var(self, mock_sf_result):
        """Cover branch 635->634: dict key not in X_df columns."""
        fig = plot_frontier_partial(
            mock_sf_result,
            input_var="log_labor",
            fix_others_at={"log_capital": 3.0, "unknown_var": 1.0},
            backend="plotly",
        )
        assert fig is not None

    def test_custom_title(self, mock_sf_result):
        """Cover branch 646->649: custom title provided."""
        fig = plot_frontier_partial(
            mock_sf_result,
            input_var="log_labor",
            fix_others_at="mean",
            backend="plotly",
            title="Custom Partial Frontier",
        )
        assert fig is not None
