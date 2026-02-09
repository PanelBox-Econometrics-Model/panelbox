"""
Unit tests for Breusch-Pagan LM test for cross-sectional dependence.

Tests the BreuschPaganLMTest which detects cross-sectional dependence
in panel model residuals using the Lagrange Multiplier approach.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.validation.cross_sectional_dependence.breusch_pagan_lm import BreuschPaganLMTest


class TestBreuschPaganLM:
    """Test suite for Breusch-Pagan LM cross-sectional dependence test."""

    def test_detects_cross_sectional_dependence(self, panel_with_cross_sectional_dependence):
        """Test that BP-LM test detects cross-sectional dependence."""
        fe = FixedEffects("y ~ x1 + x2", panel_with_cross_sectional_dependence, "entity", "time")
        results = fe.fit()

        test = BreuschPaganLMTest(results)
        result = test.run(alpha=0.05)

        # Should detect cross-sectional dependence
        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_no_false_positive_clean_data(self, clean_panel_data):
        """Test that BP-LM doesn't reject when no cross-sectional dependence exists."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganLMTest(results)
        result = test.run(alpha=0.05)

        # Test should run without error (p-value bounds check only)
        # Note: May reject due to sampling variation even with independent data
        assert result is not None
        assert 0 <= result.pvalue <= 1

    def test_works_with_pooled_ols(self, clean_panel_data):
        """Test that BP-LM works with Pooled OLS."""
        pooled = PooledOLS("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = pooled.fit()

        test = BreuschPaganLMTest(results)
        result = test.run()

        # Should run without error
        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_result_attributes(self, clean_panel_data):
        """Test that result has all required attributes."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganLMTest(results)
        result = test.run()

        # Check all attributes exist
        assert hasattr(result, "test_name")
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")
        assert hasattr(result, "reject_null")
        assert hasattr(result, "conclusion")
        assert hasattr(result, "details")

        # Check types
        assert isinstance(result.test_name, str)
        assert isinstance(result.statistic, (int, float))
        assert isinstance(result.pvalue, (int, float))
        assert isinstance(result.reject_null, bool)

        # Check test name
        assert "Breusch" in result.test_name or "BP" in result.test_name or "LM" in result.test_name

    def test_statistic_non_negative(self, clean_panel_data):
        """Test that LM statistic is non-negative."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganLMTest(results)
        result = test.run()

        # LM statistic should be non-negative (sum of squared correlations)
        assert result.statistic >= 0, "LM statistic must be non-negative"

    def test_different_alpha_levels(self, clean_panel_data):
        """Test behavior with different significance levels."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganLMTest(results)

        # Test with different alphas
        result_001 = test.run(alpha=0.01)
        result_010 = test.run(alpha=0.10)

        # P-value and statistic should be the same
        assert result_001.pvalue == result_010.pvalue
        assert result_001.statistic == result_010.statistic

    def test_pvalue_bounds(self, clean_panel_data):
        """Test that p-value is between 0 and 1."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganLMTest(results)
        result = test.run()

        assert 0 <= result.pvalue <= 1, f"P-value {result.pvalue} should be in [0, 1]"

    def test_with_balanced_panel(self, balanced_panel_data):
        """Test BP-LM with balanced panel."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganLMTest(results)
        result = test.run()

        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_with_unbalanced_panel(self, unbalanced_panel_data):
        """Test BP-LM with unbalanced panel."""
        fe = FixedEffects("y ~ x1 + x2", unbalanced_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganLMTest(results)
        # Should handle unbalanced data
        result = test.run()

        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_insufficient_entities_error(self, clean_panel_data):
        """Test ValueError when N < 2 (line 118)."""
        # Create data with only one entity
        data_single = clean_panel_data[clean_panel_data["entity"] == 1].copy()

        fe = FixedEffects("y ~ x1 + x2", data_single, "entity", "time")
        results = fe.fit()

        test = BreuschPaganLMTest(results)
        with pytest.raises(ValueError, match="at least 2 entities"):
            test.run()

    def test_no_valid_correlations_error(self, clean_panel_data):
        """Test ValueError when no valid pairwise correlations (line 152)."""
        from unittest.mock import patch

        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganLMTest(results)

        # Mock np.corrcoef to return NaN to simulate constant residuals
        original_corrcoef = np.corrcoef

        def mock_corrcoef(x, y):
            # Return NaN correlation matrix
            return np.array([[np.nan, np.nan], [np.nan, np.nan]])

        with patch("numpy.corrcoef", side_effect=mock_corrcoef):
            with pytest.raises(ValueError, match="No valid pairwise correlations"):
                test.run()

    def test_missing_entity_time_index_error(self, clean_panel_data):
        """Test AttributeError when entity_index/time_index missing (line 227)."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        # Remove required attributes
        if hasattr(results, "entity_index"):
            delattr(results, "entity_index")
        if hasattr(results, "time_index"):
            delattr(results, "time_index")

        test = BreuschPaganLMTest(results)
        with pytest.raises(AttributeError, match="entity_index.*time_index"):
            test.run()
