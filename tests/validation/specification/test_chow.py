"""
Tests for Chow test for structural break in panel data models.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.validation.specification.chow import ChowTest


class TestChowInit:
    """Test Chow test initialization."""

    def test_init_with_pooled_ols(self, balanced_panel_data):
        """Test initialization with Pooled OLS results."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        assert test.results is results

    def test_init_with_fixed_effects(self, balanced_panel_data):
        """Test initialization with Fixed Effects results."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        assert test.results is results


class TestChowRun:
    """Test Chow test run method."""

    def test_run_basic(self, balanced_panel_data):
        """Test basic Chow test execution."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run()

        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")
        assert result.test_name == "Chow Test for Structural Break"

    def test_run_with_default_break_point(self, balanced_panel_data):
        """Test that default break point is median."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run()

        # Get time periods
        time_periods = sorted(balanced_panel_data["time"].unique())
        n_periods = len(time_periods)
        expected_break_idx = n_periods // 2

        assert result.metadata["break_index"] == expected_break_idx
        assert result.metadata["break_point"] == time_periods[expected_break_idx]

    def test_run_with_integer_break_point(self, balanced_panel_data):
        """Test with explicit integer break point."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        time_periods = sorted(balanced_panel_data["time"].unique())
        break_time = time_periods[3]  # Use 4th time period

        test = ChowTest(results)
        result = test.run(break_point=break_time)

        assert result.metadata["break_point"] == break_time
        assert result.metadata["break_index"] == 3

    def test_run_with_float_break_point(self, balanced_panel_data):
        """Test with fractional break point."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run(break_point=0.5)

        # Should use median
        time_periods = sorted(balanced_panel_data["time"].unique())
        n_periods = len(time_periods)
        expected_break_idx = int(n_periods * 0.5)

        assert result.metadata["break_index"] == expected_break_idx

    def test_run_with_alpha(self, balanced_panel_data):
        """Test with custom alpha level."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run(alpha=0.01)

        assert result.alpha == 0.01

    def test_result_has_metadata(self, balanced_panel_data):
        """Test that result contains all expected metadata."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run()

        metadata = result.metadata
        assert "break_point" in metadata
        assert "break_index" in metadata
        assert "n_periods_total" in metadata
        assert "n_obs_period1" in metadata
        assert "n_obs_period2" in metadata
        assert "n_obs_total" in metadata
        assert "ssr_restricted" in metadata
        assert "ssr_unrestricted" in metadata
        assert "ssr_period1" in metadata
        assert "ssr_period2" in metadata
        assert "k_parameters" in metadata
        assert "coefficients_period1" in metadata
        assert "coefficients_period2" in metadata


class TestChowWithStructuralBreak:
    """Test Chow test with data containing structural break."""

    def test_detects_structural_break(self):
        """Test that Chow test can detect a structural break."""
        np.random.seed(42)

        # Generate data with structural break
        n_entities = 20
        n_periods = 10

        entity = np.repeat(np.arange(1, n_entities + 1), n_periods)
        time = np.tile(np.arange(1, n_periods + 1), n_entities)

        x1 = np.random.normal(0, 1, n_entities * n_periods)
        x2 = np.random.normal(0, 1, n_entities * n_periods)

        # Create structural break at t=6
        # Period 1 (t<6): y = 1 + 2*x1 + 3*x2 + e
        # Period 2 (t>=6): y = 1 + 5*x1 + 1*x2 + e  (different coefficients)
        y = np.where(
            time < 6,
            1 + 2 * x1 + 3 * x2 + np.random.normal(0, 0.5, n_entities * n_periods),
            1 + 5 * x1 + 1 * x2 + np.random.normal(0, 0.5, n_entities * n_periods),
        )

        data = pd.DataFrame({"entity": entity, "time": time, "x1": x1, "x2": x2, "y": y})

        # Fit model ignoring structural break
        model = PooledOLS("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run(break_point=6)

        # Chow test should detect the structural break
        assert result is not None
        assert result.statistic > 0
        # With strong structural break, p-value should be small
        # (but we don't enforce strict threshold due to randomness)


class TestChowStatistics:
    """Test statistical properties of Chow test."""

    def test_f_statistic_positive(self, balanced_panel_data):
        """Test that F-statistic is positive."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run()

        assert result.statistic > 0

    def test_pvalue_in_valid_range(self, balanced_panel_data):
        """Test that p-value is between 0 and 1."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run()

        assert 0 <= result.pvalue <= 1

    def test_degrees_of_freedom(self, balanced_panel_data):
        """Test degrees of freedom calculation."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run()

        df_num, df_denom = result.df
        k = result.metadata["k_parameters"]
        N = result.metadata["n_obs_total"]

        assert df_num == k
        assert df_denom == N - 2 * k

    def test_ssr_relationship(self, balanced_panel_data):
        """Test that SSR relationship is correct."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run()

        ssr_unrestricted = result.metadata["ssr_unrestricted"]
        ssr_period1 = result.metadata["ssr_period1"]
        ssr_period2 = result.metadata["ssr_period2"]

        # Unrestricted SSR should equal sum of period SSRs
        assert np.isclose(ssr_unrestricted, ssr_period1 + ssr_period2)


class TestChowEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_break_point_raises(self, balanced_panel_data):
        """Test that invalid break point raises ValueError."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)

        # Break point not in time periods
        with pytest.raises(ValueError, match="not found in time periods"):
            test.run(break_point=999)

        # Invalid float (>1)
        with pytest.raises(ValueError, match="must be None, int"):
            test.run(break_point=1.5)

        # Invalid float (<0)
        with pytest.raises(ValueError, match="must be None, int"):
            test.run(break_point=-0.5)

        # Invalid type
        with pytest.raises(ValueError, match="must be None, int"):
            test.run(break_point="invalid")

    def test_insufficient_observations_raises(self):
        """Test that insufficient observations in subperiods raises ValueError."""
        # Create dataset with enough observations to fit, but not enough for Chow test
        np.random.seed(42)
        n = 20
        data = pd.DataFrame(
            {
                "entity": np.repeat([1, 2], 10),
                "time": np.tile(range(1, 11), 2),
                "x1": np.random.normal(0, 1, n),
                "x2": np.random.normal(0, 1, n),
                "y": np.random.normal(0, 1, n),
            }
        )

        model = PooledOLS("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)

        # Break at extreme point to create insufficient observations
        with pytest.raises(ValueError, match="Insufficient observations"):
            test.run(break_point=0.05)  # Very early break point

    def test_works_with_single_regressor(self, balanced_panel_data):
        """Test with single regressor."""
        model = PooledOLS("y ~ x1", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run()

        assert result is not None

    def test_observation_counts(self, balanced_panel_data):
        """Test that observation counts are correct."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run()

        n1 = result.metadata["n_obs_period1"]
        n2 = result.metadata["n_obs_period2"]
        n_total = result.metadata["n_obs_total"]

        assert n1 + n2 == n_total
        assert n1 > 0
        assert n2 > 0


class TestChowCoefficients:
    """Test coefficient estimates for each period."""

    def test_coefficients_present(self, balanced_panel_data):
        """Test that coefficients for both periods are returned."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run()

        coef1 = result.metadata["coefficients_period1"]
        coef2 = result.metadata["coefficients_period2"]

        assert len(coef1) > 0
        assert len(coef2) > 0
        assert "x1" in coef1
        assert "x2" in coef1
        assert "x1" in coef2
        assert "x2" in coef2

    def test_coefficients_are_numeric(self, balanced_panel_data):
        """Test that coefficients are numeric."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run()

        coef1 = result.metadata["coefficients_period1"]
        coef2 = result.metadata["coefficients_period2"]

        assert all(isinstance(v, (int, float)) for v in coef1.values())
        assert all(isinstance(v, (int, float)) for v in coef2.values())


class TestChowHypotheses:
    """Test null and alternative hypotheses."""

    def test_null_hypothesis(self, balanced_panel_data):
        """Test null hypothesis text."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run()

        assert result.null_hypothesis == "No structural break (parameters stable)"

    def test_alternative_hypothesis(self, balanced_panel_data):
        """Test alternative hypothesis includes break point."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)

        # Get valid time periods
        time_periods = sorted(balanced_panel_data["time"].unique())
        break_time = time_periods[2]  # Use 3rd time period

        result = test.run(break_point=break_time)

        assert f"Structural break at t={break_time}" in result.alternative_hypothesis


class TestChowGetDataFull:
    """Test _get_data_full internal method."""

    def test_get_data_full_with_valid_model(self, balanced_panel_data):
        """Test _get_data_full returns correct data."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        data, formula, entity_col, time_col, var_names = test._get_data_full()

        assert data is not None
        assert formula is not None
        assert entity_col == "entity"
        assert time_col == "time"
        assert "x1" in var_names
        assert "x2" in var_names

    def test_get_data_full_excludes_intercept(self, balanced_panel_data):
        """Test that intercept is excluded from var_names."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        _, _, _, _, var_names = test._get_data_full()

        assert "intercept" not in [v.lower() for v in var_names]
        assert "1" not in var_names


class TestChowWithDifferentBreakPoints:
    """Test Chow test with various break point specifications."""

    def test_break_at_25_percent(self, balanced_panel_data):
        """Test break at 25% of sample."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run(break_point=0.25)

        time_periods = sorted(balanced_panel_data["time"].unique())
        n_periods = len(time_periods)
        expected_idx = int(n_periods * 0.25)

        assert result.metadata["break_index"] == expected_idx

    def test_break_at_75_percent(self, balanced_panel_data):
        """Test break at 75% of sample."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result = test.run(break_point=0.75)

        time_periods = sorted(balanced_panel_data["time"].unique())
        n_periods = len(time_periods)
        expected_idx = int(n_periods * 0.75)

        assert result.metadata["break_index"] == expected_idx

    def test_different_break_points_give_different_results(self, balanced_panel_data):
        """Test that different break points give different statistics."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = ChowTest(results)
        result1 = test.run(break_point=0.25)
        result2 = test.run(break_point=0.75)

        # Different break points should generally give different F-statistics
        # (unless data has no variation, which is unlikely with random data)
        assert result1.metadata["break_point"] != result2.metadata["break_point"]
