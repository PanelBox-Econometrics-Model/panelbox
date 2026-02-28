"""Deep coverage tests for panelbox.frontier.tests module.

Targets the remaining uncovered lines:
- Lines 137-140: LinAlgError fallback in hausman_test_tfe_tre (PD but singular)
- Branch 598->608: cost frontier skewness warning in inefficiency_presence_test
"""

import warnings
from unittest.mock import patch

import numpy as np

from panelbox.frontier.tests import hausman_test_tfe_tre, inefficiency_presence_test


class TestHausmanSingularButPositiveDefinite:
    """Cover lines 137-140: V_diff passes Cholesky but inv() raises LinAlgError."""

    def test_hausman_inv_linalg_error_fallback(self):
        """When V_diff is positive definite but inv() raises LinAlgError,
        the code should fall back to pinv with a warning."""
        # Create valid positive definite V_diff so Cholesky passes
        params_tfe = np.array([1.0, 2.0, 3.0, 0.5, 0.3])
        params_tre = np.array([0.9, 1.8, 2.7, 0.4, 0.2, 0.1])

        # TFE vcov: 5x5 (3 beta + 2 variance)
        V_tfe = np.eye(5) * 0.1
        # TRE vcov: 6x6 (3 beta + 3 variance)
        V_tre = np.eye(6) * 0.05

        # V_diff = V_tfe[:3,:3] - V_tre[:3,:3] = 0.1*I - 0.05*I = 0.05*I
        # This is PD, so Cholesky passes. Now mock inv to raise LinAlgError.
        with (
            patch("panelbox.frontier.tests.inv", side_effect=np.linalg.LinAlgError("singular")),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            result = hausman_test_tfe_tre(
                params_tfe=params_tfe,
                params_tre=params_tre,
                vcov_tfe=V_tfe,
                vcov_tre=V_tre,
            )
            # Check that the singular warning was issued (line 140)
            singular_warnings = [x for x in w if "singular" in str(x.message).lower()]
            assert len(singular_warnings) >= 1, (
                f"Expected singular warning, got: {[str(x.message) for x in w]}"
            )

        # Result should still be valid (computed via pinv)
        assert "statistic" in result
        assert "pvalue" in result
        assert result["is_positive_definite"]  # Cholesky passed


class TestInefficiencyPresenceCostSkewness:
    """Cover branch 598->608: cost frontier with negative skewness warning."""

    def test_cost_frontier_negative_skewness_triggers_warning(self):
        """When frontier_type='cost' and residuals have negative skewness,
        the skewness_warning should be set (lines 598-606)."""
        np.random.seed(42)
        # Create residuals with strong negative skewness
        # Use exponential and negate to get negative skew
        residuals = -np.random.exponential(1.0, 500)

        result = inefficiency_presence_test(
            loglik_sfa=-100.0,
            loglik_ols=-120.0,
            residuals_ols=residuals,
            frontier_type="cost",
            distribution="half_normal",
        )

        assert result["skewness_warning"] is not None
        assert "negative skewness" in result["skewness_warning"]
        assert "cost frontier" in result["skewness_warning"]

    def test_cost_frontier_positive_skewness_no_warning(self):
        """When frontier_type='cost' and residuals have positive skewness,
        no skewness_warning should be set (correct behavior for cost)."""
        np.random.seed(42)
        # Create residuals with positive skewness (correct for cost)
        residuals = np.random.exponential(1.0, 500)

        result = inefficiency_presence_test(
            loglik_sfa=-100.0,
            loglik_ols=-120.0,
            residuals_ols=residuals,
            frontier_type="cost",
            distribution="half_normal",
        )

        assert result["skewness_warning"] is None

    def test_other_frontier_type_no_skewness_check(self):
        """When frontier_type is neither 'production' nor 'cost',
        skewness_warning should remain None (branch 598->608 False path)."""
        np.random.seed(42)
        residuals = -np.random.exponential(1.0, 500)

        result = inefficiency_presence_test(
            loglik_sfa=-100.0,
            loglik_ols=-120.0,
            residuals_ols=residuals,
            frontier_type="other",
            distribution="half_normal",
        )

        assert result["skewness_warning"] is None
