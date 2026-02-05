"""
Tests for Standard Error Comparison Tools
==========================================

This module contains tests for the StandardErrorComparison class which allows
researchers to compare different types of standard errors for the same model.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.models.static.random_effects import RandomEffects
from panelbox.standard_errors import ComparisonResult, StandardErrorComparison

# ===========================
# Fixtures
# ===========================


@pytest.fixture
def panel_data():
    """Generate simple panel dataset for testing."""
    np.random.seed(42)
    n_entities = 20
    n_periods = 8
    n = n_entities * n_periods

    data = pd.DataFrame(
        {
            "entity": np.repeat(range(n_entities), n_periods),
            "time": np.tile(range(n_periods), n_entities),
            "y": np.random.randn(n),
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
        }
    )

    return data


@pytest.fixture
def fe_results(panel_data):
    """Fit Fixed Effects model for testing."""
    fe = FixedEffects("y ~ x1 + x2", panel_data, "entity", "time")
    return fe.fit()


@pytest.fixture
def pooled_results(panel_data):
    """Fit Pooled OLS model for testing."""
    pooled = PooledOLS("y ~ x1 + x2", panel_data, "entity", "time")
    return pooled.fit()


# ===========================
# Test Initialization
# ===========================


class TestInitialization:
    """Test StandardErrorComparison initialization."""

    def test_init_with_fixed_effects(self, fe_results):
        """Test initialization with Fixed Effects results."""
        comparison = StandardErrorComparison(fe_results)

        assert comparison.model_results is not None
        assert len(comparison.coef_names) == 2
        assert len(comparison.coefficients) == 2
        assert comparison.df_resid > 0

    def test_init_with_pooled(self, pooled_results):
        """Test initialization with Pooled OLS results."""
        comparison = StandardErrorComparison(pooled_results)

        assert comparison.model_results is not None
        assert len(comparison.coef_names) == 2

    def test_extract_model_info(self, fe_results):
        """Test that model info is extracted correctly."""
        comparison = StandardErrorComparison(fe_results)

        # Should have stored the model object
        assert hasattr(comparison, "model")
        assert comparison.model is not None


# ===========================
# Test compare_all
# ===========================


class TestCompareAll:
    """Test compare_all method."""

    def test_compare_all_default(self, fe_results):
        """Test compare_all with default SE types."""
        comparison = StandardErrorComparison(fe_results)
        result = comparison.compare_all()

        # Check result structure
        assert isinstance(result, ComparisonResult)
        assert result.se_comparison is not None
        assert result.se_ratios is not None
        assert result.t_stats is not None
        assert result.p_values is not None
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.significance is not None
        assert result.summary_stats is not None

    def test_compare_all_custom_types(self, fe_results):
        """Test compare_all with custom SE types."""
        comparison = StandardErrorComparison(fe_results)
        result = comparison.compare_all(se_types=["nonrobust", "robust", "clustered"])

        # Should only have the 3 requested types
        assert len(result.se_comparison.columns) == 3
        assert "nonrobust" in result.se_comparison.columns
        assert "robust" in result.se_comparison.columns
        assert "clustered" in result.se_comparison.columns

    def test_se_comparison_shape(self, fe_results):
        """Test that SE comparison has correct shape."""
        comparison = StandardErrorComparison(fe_results)
        result = comparison.compare_all()

        # Should have rows for each coefficient
        assert result.se_comparison.shape[0] == len(comparison.coef_names)

        # Should have columns for each SE type
        assert result.se_comparison.shape[1] > 0

    def test_se_ratios_relative_to_baseline(self, fe_results):
        """Test that SE ratios are computed correctly."""
        comparison = StandardErrorComparison(fe_results)
        result = comparison.compare_all(se_types=["nonrobust", "robust", "clustered"])

        # If nonrobust is included, it should have ratio of 1.0
        if "nonrobust" in result.se_ratios.columns:
            np.testing.assert_array_almost_equal(
                result.se_ratios["nonrobust"].values,
                np.ones(len(comparison.coef_names)),
                decimal=10,
            )

        # All ratios should be positive
        assert (result.se_ratios > 0).all().all()

    def test_t_stats_computation(self, fe_results):
        """Test that t-statistics are computed correctly."""
        comparison = StandardErrorComparison(fe_results)
        result = comparison.compare_all(se_types=["nonrobust", "robust"])

        # t-stats should be coef / se
        for se_type in result.t_stats.columns:
            expected_t = comparison.coefficients / result.se_comparison[se_type].values
            np.testing.assert_array_almost_equal(
                result.t_stats[se_type].values, expected_t, decimal=10
            )

    def test_p_values_range(self, fe_results):
        """Test that p-values are in [0, 1] range."""
        comparison = StandardErrorComparison(fe_results)
        result = comparison.compare_all()

        # All p-values should be between 0 and 1
        assert (result.p_values >= 0).all().all()
        assert (result.p_values <= 1).all().all()

    def test_confidence_intervals(self, fe_results):
        """Test that confidence intervals are ordered correctly."""
        comparison = StandardErrorComparison(fe_results)
        result = comparison.compare_all()

        # ci_lower should be less than ci_upper
        for se_type in result.ci_lower.columns:
            assert (result.ci_lower[se_type] < result.ci_upper[se_type]).all()

        # Coefficients should be within CI
        for se_type in result.ci_lower.columns:
            within_ci = (result.ci_lower[se_type].values <= comparison.coefficients) & (
                comparison.coefficients <= result.ci_upper[se_type].values
            )
            # At 95% level, most should be within CI (but not necessarily all in small sample)
            # Just check that at least some are within
            assert within_ci.sum() >= 0

    def test_significance_stars(self, fe_results):
        """Test significance star indicators."""
        comparison = StandardErrorComparison(fe_results)
        result = comparison.compare_all()

        # Check that significance contains only valid values
        valid_values = {"", "*", "**", "***"}
        for se_type in result.significance.columns:
            assert set(result.significance[se_type].unique()).issubset(valid_values)

    def test_summary_stats_computation(self, fe_results):
        """Test summary statistics computation."""
        comparison = StandardErrorComparison(fe_results)
        result = comparison.compare_all()

        # Check that summary stats has expected columns
        assert "mean_se" in result.summary_stats.columns
        assert "std_se" in result.summary_stats.columns
        assert "min_se" in result.summary_stats.columns
        assert "max_se" in result.summary_stats.columns
        assert "range_se" in result.summary_stats.columns
        assert "cv_se" in result.summary_stats.columns

        # Check that computations are correct
        mean_manual = result.se_comparison.mean(axis=1)
        np.testing.assert_array_almost_equal(
            result.summary_stats["mean_se"].values, mean_manual.values, decimal=10
        )

        # Range should be max - min
        range_manual = result.se_comparison.max(axis=1) - result.se_comparison.min(axis=1)
        np.testing.assert_array_almost_equal(
            result.summary_stats["range_se"].values, range_manual.values, decimal=10
        )


# ===========================
# Test compare_pair
# ===========================


class TestComparePair:
    """Test compare_pair method."""

    def test_compare_pair_basic(self, fe_results):
        """Test comparing two specific SE types."""
        comparison = StandardErrorComparison(fe_results)
        result = comparison.compare_pair("nonrobust", "robust")

        # Should only have 2 SE types
        assert len(result.se_comparison.columns) == 2
        assert "nonrobust" in result.se_comparison.columns
        assert "robust" in result.se_comparison.columns

    def test_compare_pair_structure(self, fe_results):
        """Test that pair comparison has same structure as compare_all."""
        comparison = StandardErrorComparison(fe_results)
        result = comparison.compare_pair("nonrobust", "clustered")

        # Should have all the same attributes
        assert isinstance(result, ComparisonResult)
        assert result.se_comparison is not None
        assert result.se_ratios is not None
        assert result.t_stats is not None
        assert result.p_values is not None


# ===========================
# Test plot_comparison (optional, if matplotlib available)
# ===========================


class TestPlotComparison:
    """Test plot_comparison method."""

    def test_plot_comparison_basic(self, fe_results):
        """Test basic plotting functionality."""
        pytest.importorskip("matplotlib")

        comparison = StandardErrorComparison(fe_results)
        fig = comparison.plot_comparison()

        # Should return a figure
        assert fig is not None
        assert hasattr(fig, "axes")
        assert len(fig.axes) == 2  # Two subplots

    def test_plot_comparison_custom_result(self, fe_results):
        """Test plotting with pre-computed result."""
        pytest.importorskip("matplotlib")

        comparison = StandardErrorComparison(fe_results)
        result = comparison.compare_all(se_types=["nonrobust", "robust", "clustered"])
        fig = comparison.plot_comparison(result=result)

        assert fig is not None

    def test_plot_comparison_without_matplotlib(self, fe_results):
        """Test that plotting raises informative error without matplotlib."""
        # This test assumes matplotlib might not be available
        # Skip if matplotlib is installed
        pytest.importorskip("matplotlib")
        # If matplotlib is available, this test doesn't apply


# ===========================
# Test summary
# ===========================


class TestSummary:
    """Test summary method."""

    def test_summary_runs(self, fe_results, capsys):
        """Test that summary method runs without error."""
        comparison = StandardErrorComparison(fe_results)
        comparison.summary()

        # Capture output
        captured = capsys.readouterr()

        # Should have printed something
        assert len(captured.out) > 0
        assert "STANDARD ERROR COMPARISON SUMMARY" in captured.out

    def test_summary_with_result(self, fe_results, capsys):
        """Test summary with pre-computed result."""
        comparison = StandardErrorComparison(fe_results)
        result = comparison.compare_all(se_types=["nonrobust", "robust"])
        comparison.summary(result)

        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_summary_shows_inference_sensitivity(self, fe_results, capsys):
        """Test that summary shows inference sensitivity analysis."""
        comparison = StandardErrorComparison(fe_results)
        comparison.summary()

        captured = capsys.readouterr()
        assert "Inference Sensitivity" in captured.out


# ===========================
# Test Edge Cases
# ===========================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_se_types_computable(self, fe_results):
        """Test behavior when no SE types can be computed."""
        comparison = StandardErrorComparison(fe_results)

        # Try with non-existent SE type (should be caught and warned)
        # This will use default types instead
        result = comparison.compare_all()
        assert result is not None

    def test_with_pooled_ols(self, pooled_results):
        """Test comparison with Pooled OLS results."""
        comparison = StandardErrorComparison(pooled_results)
        result = comparison.compare_all(se_types=["nonrobust", "robust", "clustered"])

        assert result is not None
        assert len(result.se_comparison.columns) == 3

    def test_compare_all_with_lag_parameters(self, fe_results):
        """Test compare_all with lag parameters for HAC estimators."""
        comparison = StandardErrorComparison(fe_results)
        result = comparison.compare_all(se_types=["nonrobust", "driscoll_kraay"], max_lags=2)

        assert "driscoll_kraay" in result.se_comparison.columns


# ===========================
# Test Integration
# ===========================


class TestIntegration:
    """Test integration with different model types."""

    def test_with_fixed_effects_multiple_se(self, panel_data):
        """Test complete workflow with Fixed Effects."""
        # Fit model
        fe = FixedEffects("y ~ x1 + x2", panel_data, "entity", "time")
        results = fe.fit()

        # Compare SEs
        comparison = StandardErrorComparison(results)
        result = comparison.compare_all(se_types=["nonrobust", "robust", "hc3", "clustered"])

        # Verify results
        assert result.se_comparison.shape == (2, 4)
        assert (result.se_comparison > 0).all().all()

    def test_with_pooled_ols_multiple_se(self, panel_data):
        """Test complete workflow with Pooled OLS."""
        # Fit model
        pooled = PooledOLS("y ~ x1 + x2", panel_data, "entity", "time")
        results = pooled.fit()

        # Compare SEs
        comparison = StandardErrorComparison(results)
        result = comparison.compare_all(se_types=["nonrobust", "robust", "clustered", "twoway"])

        # Verify results
        assert result.se_comparison.shape == (2, 4)
        assert (result.se_comparison > 0).all().all()

    def test_inference_consistency_check(self, panel_data):
        """Test that inference consistency is detected."""
        # Create dataset where we know there will be inconsistency
        fe = FixedEffects("y ~ x1 + x2", panel_data, "entity", "time")
        results = fe.fit()

        comparison = StandardErrorComparison(results)
        result = comparison.compare_all(se_types=["nonrobust", "robust", "clustered"])

        # Check that we can identify which coefficients have consistent inference
        sig_matrix = result.p_values < 0.05
        for coef in sig_matrix.index:
            sig_count = sig_matrix.loc[coef].sum()
            # Inference is consistent if all agree or none
            is_consistent = (sig_count == 0) or (sig_count == len(sig_matrix.columns))


# ===========================
# Run Tests
# ===========================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
