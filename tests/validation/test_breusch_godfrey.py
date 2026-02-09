"""
Unit tests for Breusch-Godfrey LM test.

Tests the BreuschGodfreyTest which detects AR(p) serial correlation
in panel models using the Lagrange Multiplier approach.
"""

import numpy as np
import pytest

from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.validation.serial_correlation.breusch_godfrey import BreuschGodfreyTest


class TestBreuschGodfrey:
    """Test suite for Breusch-Godfrey LM test."""

    def test_detects_ar1_correlation(self, panel_with_ar1):
        """Test that BG test detects AR(1) serial correlation."""
        fe = FixedEffects("y ~ x1 + x2", panel_with_ar1, "entity", "time")
        results = fe.fit()

        test = BreuschGodfreyTest(results)
        result = test.run(lags=1, alpha=0.05)

        # Should detect AR(1) correlation
        assert result.reject_null is True, "Should reject null of no serial correlation"
        assert result.pvalue < 0.05, f"P-value {result.pvalue} should be < 0.05"
        assert result.statistic > 0, "LM statistic should be positive"
        # Conclusion format changed - check reject_null instead
        assert result.reject_null is not None

    def test_no_false_positive_clean_data(self, clean_panel_data):
        """Test that BG doesn't reject when no AR exists."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschGodfreyTest(results)
        result = test.run(lags=1, alpha=0.05)

        # Should NOT detect serial correlation
        assert result.reject_null is False, "Should not reject null with clean data"
        assert result.pvalue >= 0.05, f"P-value {result.pvalue} should be >= 0.05"

    def test_works_with_pooled_ols(self, panel_with_ar1):
        """Test that BG test works with Pooled OLS."""
        pooled = PooledOLS("y ~ x1 + x2", panel_with_ar1, "entity", "time")
        results = pooled.fit()

        test = BreuschGodfreyTest(results)
        result = test.run(lags=1)

        # Should run without error
        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_higher_order_lags(self, panel_with_ar1):
        """Test BG with higher order lags."""
        fe = FixedEffects("y ~ x1 + x2", panel_with_ar1, "entity", "time")
        results = fe.fit()

        test = BreuschGodfreyTest(results)

        # Test with lags=2
        result_lag2 = test.run(lags=2, alpha=0.05)
        assert result_lag2 is not None
        assert result_lag2.details["lags"] == 2

        # Test with lags=3
        result_lag3 = test.run(lags=3, alpha=0.05)
        assert result_lag3 is not None
        assert result_lag3.details["lags"] == 3

    def test_requires_minimum_periods_for_lags(self, balanced_panel_data):
        """Test that BG requires sufficient periods for lag order."""
        # Create data with only 3 periods
        data_short = balanced_panel_data[
            balanced_panel_data["time"].isin([2020, 2021, 2022])
        ].copy()

        fe = FixedEffects("y ~ x1 + x2", data_short, "entity", "time")
        results = fe.fit()

        test = BreuschGodfreyTest(results)

        # lags=1 should work (need at least 2 periods after lag)
        result = test.run(lags=1)
        assert result is not None

        # lags=3 should fail (insufficient periods)
        with pytest.raises(ValueError, match="(Insufficient observations|No valid observations)"):
            test.run(lags=3)

    def test_result_attributes(self, panel_with_ar1):
        """Test that result has all required attributes."""
        fe = FixedEffects("y ~ x1 + x2", panel_with_ar1, "entity", "time")
        results = fe.fit()

        test = BreuschGodfreyTest(results)
        result = test.run(lags=1)

        # Check all attributes exist
        assert hasattr(result, "test_name")
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")
        assert hasattr(result, "reject_null")
        assert hasattr(result, "conclusion")
        assert hasattr(result, "details")

        # Check details contains lag info
        assert "lags" in result.details
        assert result.details["lags"] == 1

        # Check test name
        assert "Breusch-Godfrey" in result.test_name

    def test_statistic_non_negative(self, clean_panel_data):
        """Test that LM statistic is non-negative."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschGodfreyTest(results)
        result = test.run(lags=1)

        assert result.statistic >= 0, "LM statistic must be non-negative"

    def test_different_alpha_levels(self, panel_with_ar1):
        """Test behavior with different significance levels."""
        fe = FixedEffects("y ~ x1 + x2", panel_with_ar1, "entity", "time")
        results = fe.fit()

        test = BreuschGodfreyTest(results)

        # Test with different alphas
        result_001 = test.run(lags=1, alpha=0.01)
        result_010 = test.run(lags=1, alpha=0.10)

        # P-value and statistic should be the same
        assert result_001.pvalue == result_010.pvalue
        assert result_001.statistic == result_010.statistic

    def test_lags_parameter_validation(self, clean_panel_data):
        """Test that lags parameter is validated."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschGodfreyTest(results)

        # lags must be positive integer
        with pytest.raises((ValueError, TypeError)):
            test.run(lags=0)

        with pytest.raises((ValueError, TypeError)):
            test.run(lags=-1)

    def test_with_unbalanced_panel(self, unbalanced_panel_data):
        """Test BG test with unbalanced panel."""
        fe = FixedEffects("y ~ x1 + x2", unbalanced_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschGodfreyTest(results)
        # Should handle unbalanced data
        result = test.run(lags=1)

        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_design_matrix_not_available(self, clean_panel_data):
        """Test ValueError when design matrix is not available."""
        from unittest.mock import Mock

        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        # Mock _get_design_matrix to return None
        test = BreuschGodfreyTest(results)
        original_method = test._get_design_matrix
        test._get_design_matrix = lambda: None

        # Should raise ValueError (line 112)
        with pytest.raises(ValueError, match="Design matrix not available"):
            test.run(lags=1)

        # Restore original method
        test._get_design_matrix = original_method

    def test_missing_entity_time_index(self, clean_panel_data):
        """Test AttributeError when entity_index/time_index missing."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        # Remove entity_index to trigger AttributeError (line 246)
        if hasattr(results, "entity_index"):
            delattr(results, "entity_index")

        test = BreuschGodfreyTest(results)

        # Should raise AttributeError
        with pytest.raises(AttributeError, match="entity_index"):
            test.run(lags=1)

    def test_model_none_in_design_matrix(self, clean_panel_data):
        """Test case where _model is None in _get_design_matrix."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschGodfreyTest(results)

        # Set _model to None to trigger line 253
        original_model = results._model
        results._model = None

        # Should return None
        design_matrix = test._get_design_matrix()
        assert design_matrix is None

        # Restore
        results._model = original_model

    def test_exception_in_design_matrix_building(self, clean_panel_data):
        """Test exception handling in design matrix building."""
        from unittest.mock import Mock, patch

        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschGodfreyTest(results)

        # Mock build_design_matrices to raise an exception (lines 264-265)
        with patch.object(
            results._model.formula_parser,
            "build_design_matrices",
            side_effect=Exception("Test error"),
        ):
            # Should catch exception and return None
            design_matrix = test._get_design_matrix()
            assert design_matrix is None
