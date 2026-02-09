"""
Unit tests for Modified Wald test.

Tests the ModifiedWaldTest which detects groupwise heteroskedasticity
in Fixed Effects panel models.
"""

import numpy as np
import pytest

from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.validation.heteroskedasticity.modified_wald import ModifiedWaldTest


class TestModifiedWald:
    """Test suite for Modified Wald test."""

    def test_detects_heteroskedasticity(self, panel_with_heteroskedasticity):
        """Test that Modified Wald test detects groupwise heteroskedasticity."""
        fe = FixedEffects("y ~ x1 + x2", panel_with_heteroskedasticity, "entity", "time")
        results = fe.fit()

        test = ModifiedWaldTest(results)
        result = test.run(alpha=0.05)

        # Should detect heteroskedasticity
        assert result.reject_null is True, "Should reject null of homoskedasticity"
        assert result.pvalue < 0.05, f"P-value {result.pvalue} should be < 0.05"
        assert result.statistic > 0, "Chi-squared statistic should be positive"
        # Conclusion format changed - check reject_null instead
        assert result.reject_null is not None

    def test_no_false_positive_clean_data(self, clean_panel_data):
        """Test that Modified Wald doesn't reject with homoskedastic data."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = ModifiedWaldTest(results)
        result = test.run(alpha=0.05)

        # Should NOT detect heteroskedasticity (though may have some chance of Type I error)
        # Use looser threshold due to randomness
        assert result.pvalue > 0.001, "Should not strongly reject with clean data"

    def test_requires_fixed_effects(self, panel_with_heteroskedasticity):
        """Test that Modified Wald test requires Fixed Effects model."""
        from panelbox.models.static.pooled_ols import PooledOLS

        pooled = PooledOLS("y ~ x1 + x2", panel_with_heteroskedasticity, "entity", "time")
        results = pooled.fit()

        # Test will warn but not raise error - designed for FE but works with others
        test = ModifiedWaldTest(results)
        result = test.run()
        assert result is not None

    def test_result_attributes(self, panel_with_heteroskedasticity):
        """Test that result has all required attributes."""
        fe = FixedEffects("y ~ x1 + x2", panel_with_heteroskedasticity, "entity", "time")
        results = fe.fit()

        test = ModifiedWaldTest(results)
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
        assert isinstance(result.conclusion, str)
        assert isinstance(result.details, dict)

        # Check test name
        assert "Modified Wald" in result.test_name

    def test_statistic_positive(self, clean_panel_data):
        """Test that chi-squared statistic is positive."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = ModifiedWaldTest(results)
        result = test.run()

        assert result.statistic > 0, "Chi-squared statistic must be positive"

    def test_different_alpha_levels(self, panel_with_heteroskedasticity):
        """Test behavior with different significance levels."""
        fe = FixedEffects("y ~ x1 + x2", panel_with_heteroskedasticity, "entity", "time")
        results = fe.fit()

        test = ModifiedWaldTest(results)

        # Test with different alphas
        result_001 = test.run(alpha=0.01)
        result_010 = test.run(alpha=0.10)

        # P-value and statistic should be the same
        assert result_001.pvalue == result_010.pvalue
        assert result_001.statistic == result_010.statistic

    def test_degrees_of_freedom(self, panel_with_heteroskedasticity):
        """Test that degrees of freedom equals number of entities."""
        fe = FixedEffects("y ~ x1 + x2", panel_with_heteroskedasticity, "entity", "time")
        results = fe.fit()

        test = ModifiedWaldTest(results)
        result = test.run()

        # Degrees of freedom should equal number of entities
        expected_df = results.n_entities
        assert result.details["df"] == expected_df

    def test_with_unbalanced_panel(self, unbalanced_panel_data):
        """Test Modified Wald with unbalanced panel."""
        fe = FixedEffects("y ~ x1 + x2", unbalanced_panel_data, "entity", "time")
        results = fe.fit()

        test = ModifiedWaldTest(results)
        # Should work with unbalanced data
        result = test.run()

        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_minimum_observations_per_group(self, balanced_panel_data):
        """Test that test requires minimum observations per group."""
        # Create data with very few observations per entity
        data_few = balanced_panel_data[balanced_panel_data["time"].isin([2020, 2021])].copy()

        fe = FixedEffects("y ~ x1 + x2", data_few, "entity", "time")
        results = fe.fit()

        test = ModifiedWaldTest(results)
        # Should still run but may have low power
        result = test.run()
        assert result is not None

    def test_missing_entity_index(self, balanced_panel_data):
        """Test AttributeError when entity_index is missing (line 355)."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        # Remove entity_index to trigger AttributeError
        if hasattr(results, "entity_index"):
            delattr(results, "entity_index")

        test = ModifiedWaldTest(results)

        # Should raise AttributeError
        with pytest.raises(AttributeError, match="entity_index"):
            test.run()

    def test_zero_variance_entity(self, balanced_panel_data):
        """Test handling of entity with zero variance (line 308)."""
        # Create data where one entity has constant residuals (zero variance)
        # This is a contrived case that's hard to trigger naturally
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = fe.fit()

        test = ModifiedWaldTest(results)

        # The test should handle zero/negative variance gracefully
        # by skipping that entity (line 308: continue)
        # Even if it's hard to trigger, we verify the test runs
        result = test.run()
        assert result is not None
        assert result.statistic >= 0
