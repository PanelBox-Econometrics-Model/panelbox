"""
Validation tests comparing PanelBox Conditional Logit with R mlogit.
"""

import pytest

pytestmark = pytest.mark.r_validation


class TestConditionalLogitvsR:
    """Compare Conditional Logit against R mlogit package."""

    def test_parameter_estimates(self, conditional_logit_r_results):
        """Compare parameter estimates with R."""
        from panelbox.models.discrete import ConditionalLogit

        r_results = conditional_logit_r_results["results"]
        data = conditional_logit_r_results["data"]

        # Fit PanelBox model
        model = ConditionalLogit(
            data=data,
            choice_col="trip_id",
            alt_col="mode",
            chosen_col="chosen",
            alt_varying_vars=["cost", "time"],
        )
        result = model.fit()

        # Get R coefficients
        r_coefs = r_results.set_index("param")["coef"]

        # Compare cost coefficient
        python_cost = result.params[0]  # Assuming cost is first
        r_cost = r_coefs["cost"]
        rel_diff_cost = abs(python_cost - r_cost) / abs(r_cost)

        assert rel_diff_cost < 0.15, (
            f"Cost: Python={python_cost:.4f}, R={r_cost:.4f}, rel_diff={rel_diff_cost:.2%}"
        )

        # Compare time coefficient
        python_time = result.params[1]  # Assuming time is second
        r_time = r_coefs["time"]
        rel_diff_time = abs(python_time - r_time) / abs(r_time)

        assert rel_diff_time < 0.15, (
            f"Time: Python={python_time:.4f}, R={r_time:.4f}, rel_diff={rel_diff_time:.2%}"
        )

    def test_signs_match_r(self, conditional_logit_r_results):
        """Check that coefficient signs match R."""
        from panelbox.models.discrete import ConditionalLogit

        r_results = conditional_logit_r_results["results"]
        data = conditional_logit_r_results["data"]

        model = ConditionalLogit(
            data=data,
            choice_col="trip_id",
            alt_col="mode",
            chosen_col="chosen",
            alt_varying_vars=["cost", "time"],
        )
        result = model.fit()

        r_coefs = r_results.set_index("param")["coef"]

        # Both should be negative
        assert result.params[0] < 0, "Cost coefficient should be negative"
        assert result.params[1] < 0, "Time coefficient should be negative"
        assert r_coefs["cost"] < 0, "R cost coefficient should be negative"
        assert r_coefs["time"] < 0, "R time coefficient should be negative"
