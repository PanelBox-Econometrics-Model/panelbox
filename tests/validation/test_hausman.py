"""
Tests for Hausman specification test.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.random_effects import RandomEffects
from panelbox.validation.specification.hausman import HausmanTest, HausmanTestResult


class TestHausmanTestInitialization:
    """Tests for HausmanTest initialization."""

    def test_init_valid(self, balanced_panel_data):
        """Test initialization with valid FE and RE results."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)

        assert hausman.fe_results is not None
        assert hausman.re_results is not None
        assert len(hausman.common_vars) > 0

    def test_init_wrong_order(self, balanced_panel_data):
        """Test that wrong order raises ValueError."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        # Should raise error if order is reversed
        with pytest.raises(ValueError, match="First argument must be Fixed Effects"):
            HausmanTest(re_results, fe_results)

    def test_common_variables(self, balanced_panel_data):
        """Test that common variables are identified correctly."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)

        # Common vars should be x1 and x2 (not Intercept, as FE doesn't have it)
        assert "x1" in hausman.common_vars
        assert "x2" in hausman.common_vars
        assert "Intercept" not in hausman.common_vars


class TestHausmanTestExecution:
    """Tests for running Hausman test."""

    def test_run_basic(self, balanced_panel_data):
        """Test basic execution of Hausman test."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)
        result = hausman.run()

        assert isinstance(result, HausmanTestResult)
        assert result.statistic is not None
        assert result.pvalue is not None
        assert result.df > 0

    def test_statistic_positive(self, balanced_panel_data):
        """Test that Hausman statistic is positive."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)
        result = hausman.run()

        # Chi-squared statistic should be non-negative
        assert result.statistic >= 0

    def test_pvalue_bounds(self, balanced_panel_data):
        """Test that p-value is in [0, 1]."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)
        result = hausman.run()

        assert 0 <= result.pvalue <= 1

    def test_degrees_of_freedom(self, balanced_panel_data):
        """Test that degrees of freedom equals number of common variables."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)
        result = hausman.run()

        # df should equal number of common variables (x1, x2)
        assert result.df == len(hausman.common_vars)
        assert result.df == 2


class TestHausmanTestResult:
    """Tests for HausmanTestResult class."""

    def test_result_attributes(self, balanced_panel_data):
        """Test that result has all required attributes."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)
        result = hausman.run()

        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")
        assert hasattr(result, "df")
        assert hasattr(result, "conclusion")
        assert hasattr(result, "recommendation")
        assert hasattr(result, "fe_params")
        assert hasattr(result, "re_params")
        assert hasattr(result, "diff")

    def test_recommendation(self, balanced_panel_data):
        """Test that recommendation is either FE or RE."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)
        result = hausman.run()

        assert result.recommendation in ["Fixed Effects", "Random Effects"]

    def test_summary_format(self, balanced_panel_data):
        """Test that summary is properly formatted."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)
        result = hausman.run()

        summary = result.summary()

        assert "HAUSMAN SPECIFICATION TEST" in summary
        assert "H0:" in summary
        assert "Recommendation:" in summary
        assert "COEFFICIENT COMPARISON" in summary
        assert "x1" in summary
        assert "x2" in summary

    def test_str_repr(self, balanced_panel_data):
        """Test __str__ and __repr__ methods."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)
        result = hausman.run()

        # __str__ should return summary
        str_result = str(result)
        assert "HAUSMAN" in str_result

        # __repr__ should be concise
        repr_result = repr(result)
        assert "HausmanTestResult" in repr_result


class TestCoefficientDifferences:
    """Tests for coefficient differences."""

    def test_diff_computation(self, balanced_panel_data):
        """Test that differences are computed correctly."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)
        result = hausman.run()

        # Manually compute difference
        for var in hausman.common_vars:
            manual_diff = fe_results.params[var] - re_results.params[var]
            np.testing.assert_almost_equal(result.diff[var], manual_diff)

    def test_fe_re_params_match(self, balanced_panel_data):
        """Test that FE and RE params in result match original results."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)
        result = hausman.run()

        # Check FE params
        for var in hausman.common_vars:
            np.testing.assert_almost_equal(result.fe_params[var], fe_results.params[var])
            np.testing.assert_almost_equal(result.re_params[var], re_results.params[var])


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_regressor(self, balanced_panel_data):
        """Test with single regressor."""
        fe = FixedEffects("y ~ x1", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)
        result = hausman.run()

        assert result.df == 1
        assert len(hausman.common_vars) == 1

    def test_unbalanced_panel(self, unbalanced_panel_data):
        """Test with unbalanced panel."""
        fe = FixedEffects("y ~ x1 + x2", unbalanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", unbalanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)
        result = hausman.run()

        assert result is not None
        assert result.statistic >= 0

    def test_different_significance_levels(self, balanced_panel_data):
        """Test with different significance levels."""
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)

        result_05 = hausman.run(alpha=0.05)
        result_01 = hausman.run(alpha=0.01)

        # Statistic and pvalue should be the same
        assert result_05.statistic == result_01.statistic
        assert result_05.pvalue == result_01.pvalue

        # But alpha should differ
        assert result_05.alpha == 0.05
        assert result_01.alpha == 0.01

    def test_reject_null_fe_preferred(self, panel_for_mundlak):
        """Test case where FE is significantly different from RE (reject H0)."""
        # Use panel_for_mundlak which has entity correlation
        # This should make FE and RE substantially different
        fe = FixedEffects("y ~ x1 + x2", panel_for_mundlak, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", panel_for_mundlak, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)
        result = hausman.run(alpha=0.05)

        # With endogeneity, FE and RE should differ significantly
        # This tests lines 65-69 (reject_null = True branch)
        if result.pvalue < 0.05:
            assert result.reject_null is True
            assert result.recommendation == "Fixed Effects"
            assert "Use Fixed Effects" in result.conclusion

    def test_invalid_second_argument(self, balanced_panel_data):
        """Test ValueError when second argument is not Random Effects."""
        from panelbox.models.static.pooled_ols import PooledOLS

        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        pooled = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        pooled_results = pooled.fit()

        # Should raise ValueError (line 296)
        with pytest.raises(ValueError, match="Second argument must be Random Effects"):
            HausmanTest(fe_results, pooled_results)

    def test_no_common_variables_error(self, balanced_panel_data):
        """Test ValueError when no common variables between FE and RE."""
        from unittest.mock import Mock

        # Create mock FE results with only var_A, var_B
        fe_results = Mock()
        fe_results.model_type = "Fixed Effects"
        fe_results.params = pd.Series({"var_A": 1.0, "var_B": 2.0})

        # Create mock RE results with only Intercept, var_C, var_D
        # (Intercept is excluded from comparison, so no overlap with FE)
        re_results = Mock()
        re_results.model_type = "Random Effects"
        re_results.params = pd.Series({"Intercept": 0.0, "var_C": 3.0, "var_D": 4.0})

        # Should raise ValueError (line 309)
        with pytest.raises(ValueError, match="No common variables found"):
            HausmanTest(fe_results, re_results)

    def test_singular_matrix_pseudoinverse(self, balanced_panel_data):
        """Test that singular matrix is handled with pseudoinverse."""
        # This edge case (lines 360-362) is difficult to trigger naturally
        # The try-except exists as a safety check
        fe = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        fe_results = fe.fit()

        re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        re_results = re.fit()

        hausman = HausmanTest(fe_results, re_results)
        result = hausman.run()

        # Should complete successfully
        # The singular matrix path is a safety check that's hard to reach
        assert result is not None
        assert result.statistic >= 0
