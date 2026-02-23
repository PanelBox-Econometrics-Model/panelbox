"""
Test utility functions for diagnostics tutorials.

Tests data generators, helpers, and visualization utilities.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))


# -----------------------------------------------------------------------
# Data generators
# -----------------------------------------------------------------------


class TestDataGenerators:
    """Test that all data generators return correct output."""

    def test_generate_penn_world_table(self):
        from data_generators import generate_penn_world_table

        df = generate_penn_world_table()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1500, 8)
        assert "countrycode" in df.columns
        assert "rgdpna" in df.columns
        assert df["countrycode"].nunique() == 30
        assert df["year"].nunique() == 50

    def test_generate_prices_panel(self):
        from data_generators import generate_prices_panel

        df = generate_prices_panel()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1200, 5)
        assert df["region"].nunique() == 40

    def test_generate_oecd_macro(self):
        from data_generators import generate_oecd_macro

        df = generate_oecd_macro()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (800, 7)
        assert df["country"].nunique() == 20

    def test_generate_ppp_data(self):
        from data_generators import generate_ppp_data

        df = generate_ppp_data()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (875, 7)
        assert df["country"].nunique() == 25

    def test_generate_interest_rates(self):
        from data_generators import generate_interest_rates

        df = generate_interest_rates()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (450, 6)
        assert df["country"].nunique() == 15

    def test_generate_nlswork(self):
        from data_generators import generate_nlswork

        df = generate_nlswork()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (60000, 10)
        assert df["idcode"].nunique() == 4000

    def test_generate_firm_productivity(self):
        from data_generators import generate_firm_productivity

        df = generate_firm_productivity()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (4000, 9)
        assert df["firm_id"].nunique() == 200

    def test_generate_trade_panel(self):
        from data_generators import generate_trade_panel

        df = generate_trade_panel()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (4500, 9)
        assert df["pair_id"].nunique() == 300

    def test_generate_us_counties(self):
        from data_generators import generate_us_counties

        result = generate_us_counties()
        assert len(result) == 4
        df, W_cont, W_dist, coords = result
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2000, 8)
        assert W_cont.shape == (200, 200)
        assert W_dist.shape == (200, 200)
        assert coords.shape == (200, 3)
        # Weight matrix properties
        assert np.allclose(np.diag(W_cont), 0)
        assert np.allclose(np.diag(W_dist), 0)

    def test_generate_eu_regions(self):
        from data_generators import generate_eu_regions

        result = generate_eu_regions()
        assert len(result) == 3
        df, W, coords = result
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1500, 8)
        assert W.shape == (100, 100)
        assert coords.shape == (100, 3)

    def test_reproducibility(self):
        """Test that generators produce identical output with same seed."""
        from data_generators import generate_penn_world_table

        df1 = generate_penn_world_table(seed=123)
        df2 = generate_penn_world_table(seed=123)
        pd.testing.assert_frame_equal(df1, df2)


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------


class TestVisualizationHelpers:
    """Test visualization helper functions."""

    def test_set_diagnostics_style(self):
        from visualization_helpers import set_diagnostics_style

        # Should not raise
        set_diagnostics_style()

    def test_plot_test_comparison(self):
        from visualization_helpers import plot_test_comparison

        results_df = pd.DataFrame(
            {
                "Test": ["Test A", "Test B"],
                "pvalue": [0.01, 0.15],
            }
        )
        fig = plot_test_comparison(results_df, metric="pvalue")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_time_series_grid(self):
        from visualization_helpers import plot_time_series_grid

        # Create small test data
        data = pd.DataFrame(
            {
                "entity": np.repeat(["A", "B", "C", "D"], 10),
                "time": np.tile(range(10), 4),
                "value": np.random.randn(40),
            }
        )
        fig = plot_time_series_grid(
            data,
            "value",
            "entity",
            "time",
            n_entities=4,
            ncols=2,
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close("all")


class TestDiagnosticsUtils:
    """Test general diagnostics utilities."""

    def test_format_test_results(self):
        from diagnostics_utils import format_test_results

        result = format_test_results("ADF", -3.45, 0.01, True)
        assert isinstance(result, str)
        assert "ADF" in result

    def test_create_results_table(self):
        from diagnostics_utils import create_results_table

        results = {
            "Test A": {"H0": "Unit root", "statistic": -2.5, "pvalue": 0.03, "reject": True},
            "Test B": {"H0": "Stationarity", "statistic": 1.8, "pvalue": 0.12, "reject": False},
        }
        table = create_results_table(results)
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2


class TestSpatialHelpers:
    """Test spatial helper functions."""

    def test_build_weight_matrix_knn(self):
        from spatial_helpers import build_weight_matrix

        coords = pd.DataFrame(
            {
                "id": range(10),
                "latitude": np.random.uniform(30, 50, 10),
                "longitude": np.random.uniform(-100, -70, 10),
            }
        )
        W = build_weight_matrix(coords, method="knn", k=3)
        assert W.shape == (10, 10)
        assert np.allclose(np.diag(W), 0)
        # Row-normalized
        row_sums = W.sum(axis=1)
        assert np.allclose(row_sums[row_sums > 0], 1.0, atol=1e-10)

    def test_lm_decision_tree_summary(self):
        from spatial_helpers import lm_decision_tree_summary

        lm_results = {
            "lm_lag": type("R", (), {"pvalue": 0.01, "statistic": 8.5})(),
            "lm_error": type("R", (), {"pvalue": 0.02, "statistic": 7.2})(),
            "robust_lm_lag": type("R", (), {"pvalue": 0.03, "statistic": 5.1})(),
            "robust_lm_error": type("R", (), {"pvalue": 0.15, "statistic": 2.0})(),
        }
        summary = lm_decision_tree_summary(lm_results)
        assert isinstance(summary, str)
        assert len(summary) > 0
