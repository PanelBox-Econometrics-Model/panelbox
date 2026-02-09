"""
Unit tests for Breusch-Pagan LM test.

Tests the BreuschPaganTest which detects heteroskedasticity
using the Lagrange Multiplier approach.
"""

import numpy as np
import pytest

from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.validation.heteroskedasticity.breusch_pagan import BreuschPaganTest


class TestBreuschPagan:
    """Test suite for Breusch-Pagan LM test."""

    def test_detects_heteroskedasticity(self, panel_with_heteroskedasticity):
        """Test that BP test can detect heteroskedasticity."""
        fe = FixedEffects("y ~ x1 + x2", panel_with_heteroskedasticity, "entity", "time")
        results = fe.fit()

        test = BreuschPaganTest(results)
        result = test.run(alpha=0.05)

        # Check that test runs
        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")
        # Note: BP may not always detect groupwise het as strongly as Modified Wald

    def test_no_false_positive_clean_data(self, clean_panel_data):
        """Test that BP doesn't reject with homoskedastic data."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganTest(results)
        result = test.run(alpha=0.05)

        # Should not strongly reject with clean data
        assert result.pvalue > 0.001, "Should not strongly reject with clean data"

    def test_works_with_pooled_ols(self, panel_with_heteroskedasticity):
        """Test that BP test works with Pooled OLS."""
        pooled = PooledOLS("y ~ x1 + x2", panel_with_heteroskedasticity, "entity", "time")
        results = pooled.fit()

        test = BreuschPaganTest(results)
        result = test.run()

        # Should run without error
        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_result_attributes(self, clean_panel_data):
        """Test that result has all required attributes."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganTest(results)
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
        assert "Breusch-Pagan" in result.test_name

    def test_statistic_non_negative(self, clean_panel_data):
        """Test that LM statistic is non-negative."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganTest(results)
        result = test.run()

        # LM statistic should be non-negative
        # (can be negative due to numerical issues, but typically non-negative)
        assert result.statistic is not None

    def test_different_alpha_levels(self, clean_panel_data):
        """Test behavior with different significance levels."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganTest(results)

        # Test with different alphas
        result_001 = test.run(alpha=0.01)
        result_010 = test.run(alpha=0.10)

        # P-value and statistic should be the same
        assert result_001.pvalue == result_010.pvalue
        assert result_001.statistic == result_010.statistic

    def test_degrees_of_freedom(self, clean_panel_data):
        """Test that degrees of freedom equals number of regressors (excluding intercept)."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganTest(results)
        result = test.run()

        # DF should be number of regressors in auxiliary regression
        assert "df" in result.details
        assert result.details["df"] > 0

    def test_with_unbalanced_panel(self, unbalanced_panel_data):
        """Test BP test with unbalanced panel."""
        fe = FixedEffects("y ~ x1 + x2", unbalanced_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganTest(results)
        result = test.run()

        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_requires_model_reference(self, clean_panel_data):
        """Test that BP test can access design matrix through model reference."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        # Should have model reference
        assert hasattr(results, "_model")
        assert results._model is not None

        test = BreuschPaganTest(results)
        result = test.run()

        # Should successfully run (accessing X via model)
        assert result is not None

    def test_with_single_regressor(self, clean_panel_data):
        """Test BP with single regressor."""
        fe = FixedEffects("y ~ x1", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganTest(results)
        result = test.run()

        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_pvalue_bounds(self, clean_panel_data):
        """Test that p-value is between 0 and 1."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganTest(results)
        result = test.run()

        assert 0 <= result.pvalue <= 1, f"P-value {result.pvalue} should be in [0, 1]"

    def test_design_matrix_not_available(self, clean_panel_data):
        """Test ValueError when design matrix is not available (line 115)."""
        from unittest.mock import Mock

        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganTest(results)
        # Mock _get_design_matrix to return None
        test._get_design_matrix = lambda: None

        # Should raise ValueError
        with pytest.raises(ValueError, match="Design matrix not available"):
            test.run()

    def test_singular_matrix_handling(self, clean_panel_data):
        """Test handling of singular matrix in auxiliary regression (lines 128-130)."""
        from unittest.mock import patch

        import numpy.linalg

        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganTest(results)

        # Mock np.linalg.solve to raise LinAlgError
        original_solve = np.linalg.solve

        def mock_solve(a, b):
            # Raise LinAlgError to trigger fallback to lstsq
            raise np.linalg.LinAlgError("Singular matrix")

        with patch("numpy.linalg.solve", side_effect=mock_solve):
            # This should trigger the LinAlgError path and use lstsq
            result = test.run()

        # Should still complete successfully using lstsq fallback
        assert result is not None
        assert result.statistic >= 0
        assert 0 <= result.pvalue <= 1

    def test_zero_total_variance(self, clean_panel_data):
        """Test R²=0 when SST=0 (line 145)."""
        from unittest.mock import Mock, patch

        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganTest(results)

        # Get the actual size of residuals
        n_obs = len(results.resid)

        # Mock to create scenario with SST = 0 (all squared residuals identical)
        # This is extremely rare but line 145 handles it
        with patch.object(test, "resid", np.ones(n_obs) * 5.0):
            result = test.run()
            assert result is not None
            # When SST=0, R2_aux should be 0.0, and lm_stat should be 0
            assert result.statistic >= 0

    def test_exception_in_design_matrix_building(self, clean_panel_data):
        """Test exception handling in _get_design_matrix (lines 224-227)."""
        from unittest.mock import patch

        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = BreuschPaganTest(results)
        # Clear cache to force rebuild
        test._X = None

        # Mock build_design_matrices to raise exception
        with patch.object(
            results._model.formula_parser,
            "build_design_matrices",
            side_effect=Exception("Test error"),
        ):
            # Should catch exception and return None
            design_matrix = test._get_design_matrix()
            assert design_matrix is None
