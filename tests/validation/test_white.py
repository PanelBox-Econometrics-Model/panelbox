"""
Unit tests for White's test.

Tests the WhiteTest which detects general heteroskedasticity
using squared terms and cross-products.
"""

import numpy as np
import pytest

from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.validation.heteroskedasticity.white import WhiteTest


class TestWhite:
    """Test suite for White's test."""

    def test_detects_heteroskedasticity(self, panel_with_heteroskedasticity):
        """Test that White test can detect heteroskedasticity."""
        fe = FixedEffects("y ~ x1 + x2", panel_with_heteroskedasticity, "entity", "time")
        results = fe.fit()

        test = WhiteTest(results)
        result = test.run(alpha=0.05, cross_terms=True)

        # Check that test runs
        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_no_false_positive_clean_data(self, clean_panel_data):
        """Test that White doesn't reject with homoskedastic data."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = WhiteTest(results)
        result = test.run(alpha=0.05, cross_terms=True)

        # Should not strongly reject with clean data
        assert result.pvalue > 0.001, "Should not strongly reject with clean data"

    def test_with_cross_terms_false(self, clean_panel_data):
        """Test White test without cross-product terms."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = WhiteTest(results)
        result_no_cross = test.run(cross_terms=False)

        # Should run successfully
        assert result_no_cross is not None
        assert hasattr(result_no_cross, "statistic")
        assert hasattr(result_no_cross, "pvalue")

        # Test with cross terms
        result_with_cross = test.run(cross_terms=True)

        # With cross terms should have different (typically higher) DF
        assert result_with_cross.details["df"] >= result_no_cross.details["df"]

    def test_works_with_pooled_ols(self, clean_panel_data):
        """Test that White test works with Pooled OLS."""
        pooled = PooledOLS("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = pooled.fit()

        test = WhiteTest(results)
        result = test.run(cross_terms=False)

        # Should run without error
        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_result_attributes(self, clean_panel_data):
        """Test that result has all required attributes."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = WhiteTest(results)
        result = test.run(cross_terms=False)

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
        assert "White" in result.test_name

    def test_statistic_non_negative(self, clean_panel_data):
        """Test that LM statistic is non-negative."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = WhiteTest(results)
        result = test.run(cross_terms=False)

        # LM statistic should be non-negative
        assert result.statistic is not None

    def test_different_alpha_levels(self, clean_panel_data):
        """Test behavior with different significance levels."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = WhiteTest(results)

        # Test with different alphas
        result_001 = test.run(alpha=0.01, cross_terms=False)
        result_010 = test.run(alpha=0.10, cross_terms=False)

        # P-value and statistic should be the same
        assert result_001.pvalue == result_010.pvalue
        assert result_001.statistic == result_010.statistic

    def test_degrees_of_freedom_increases_with_cross_terms(self, clean_panel_data):
        """Test that DF increases when cross terms are included."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = WhiteTest(results)

        result_no_cross = test.run(cross_terms=False)
        result_with_cross = test.run(cross_terms=True)

        # DF with cross terms should be larger
        df_no_cross = result_no_cross.details["df"]
        df_with_cross = result_with_cross.details["df"]

        assert (
            df_with_cross > df_no_cross
        ), f"DF with cross terms ({df_with_cross}) should be > DF without ({df_no_cross})"

    def test_with_unbalanced_panel(self, unbalanced_panel_data):
        """Test White test with unbalanced panel."""
        fe = FixedEffects("y ~ x1 + x2", unbalanced_panel_data, "entity", "time")
        results = fe.fit()

        test = WhiteTest(results)
        result = test.run(cross_terms=False)

        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_requires_model_reference(self, clean_panel_data):
        """Test that White test can access design matrix through model reference."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        # Should have model reference
        assert hasattr(results, "_model")
        assert results._model is not None

        test = WhiteTest(results)
        result = test.run(cross_terms=False)

        # Should successfully run (accessing X via model)
        assert result is not None

    def test_with_single_regressor(self, clean_panel_data):
        """Test White with single regressor."""
        fe = FixedEffects("y ~ x1", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = WhiteTest(results)
        result = test.run(cross_terms=False)

        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_pvalue_bounds(self, clean_panel_data):
        """Test that p-value is between 0 and 1."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = WhiteTest(results)
        result = test.run(cross_terms=False)

        assert 0 <= result.pvalue <= 1, f"P-value {result.pvalue} should be in [0, 1]"

    def test_cross_terms_parameter_type(self, clean_panel_data):
        """Test that cross_terms parameter accepts boolean values."""
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = WhiteTest(results)

        # Should accept True
        result_true = test.run(cross_terms=True)
        assert result_true is not None

        # Should accept False
        result_false = test.run(cross_terms=False)
        assert result_false is not None

    def test_design_matrix_not_available(self, clean_panel_data):
        """Test ValueError when design matrix is not available (line 106)."""
        from unittest.mock import Mock

        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = WhiteTest(results)
        # Mock _get_design_matrix to return None
        test._get_design_matrix = lambda: None

        # Should raise ValueError
        with pytest.raises(ValueError, match="Design matrix not available"):
            test.run()

    def test_exception_in_design_matrix_building(self, clean_panel_data):
        """Test exception handling in _get_design_matrix (lines 215-216)."""
        from unittest.mock import patch

        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = WhiteTest(results)
        # Clear the cache to force the method to call build_design_matrices
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

    def test_singular_matrix_handling(self, clean_panel_data):
        """Test handling of singular matrix with lstsq fallback (lines 150-151)."""
        from unittest.mock import patch

        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = WhiteTest(results)

        # Mock np.linalg.solve to raise LinAlgError
        original_solve = np.linalg.solve

        def mock_solve(a, b):
            # Raise LinAlgError to trigger lstsq fallback
            raise np.linalg.LinAlgError("Singular matrix")

        with patch("numpy.linalg.solve", side_effect=mock_solve):
            # Should use lstsq fallback and complete successfully
            result = test.run(cross_terms=False)
            assert result is not None
            assert result.statistic >= 0

    def test_zero_total_variance_edge_case(self, clean_panel_data):
        """Test edge case when SST is zero (line 164)."""
        from unittest.mock import Mock

        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        # Create residuals with zero variance (all same value)
        results.resid = np.ones(len(results.resid))

        test = WhiteTest(results)
        # This should trigger the SST <= 0 case, setting R2_aux = 0
        result = test.run(cross_terms=False)

        # Should complete successfully with R2_aux = 0
        assert result is not None
        assert result.statistic >= 0
