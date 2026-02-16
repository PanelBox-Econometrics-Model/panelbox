"""
Integration tests for TFP decomposition and marginal effects.

Tests that result.tfp_decomposition() and result.marginal_effects()
work correctly for various model types.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier.advanced.four_component import FourComponentSFA


def generate_test_data(n_firms=30, n_years=4, seed=42):
    """Generate synthetic panel data for testing."""
    np.random.seed(seed)

    firms = np.repeat(range(1, n_firms + 1), n_years)
    years = np.tile(range(2015, 2015 + n_years), n_firms)
    n_obs = n_firms * n_years

    # Inputs
    log_K = np.random.uniform(3, 5, n_obs)
    log_L = np.random.uniform(2, 4, n_obs)

    # Output
    log_Y = 2.0 + 0.4 * log_K + 0.5 * log_L + np.random.normal(0, 0.1, n_obs)

    data = pd.DataFrame(
        {
            "firm": firms,
            "year": years,
            "log_output": log_Y,
            "log_capital": log_K,
            "log_labor": log_L,
        }
    )

    return data


class TestTFPIntegration:
    """Test TFP decomposition integration with result objects."""

    def test_tfp_decomposition_four_component(self):
        """Test that tfp_decomposition() works for FourComponentResult."""
        # Generate data
        data = generate_test_data(n_firms=30, n_years=4)

        # Fit model
        model = FourComponentSFA(
            data=data,
            depvar="log_output",
            exog=["log_capital", "log_labor"],
            entity="firm",
            time="year",
            frontier_type="production",
        )

        result = model.fit(verbose=False)

        # Test that tfp_decomposition() method exists
        assert hasattr(result, "tfp_decomposition"), "Result should have tfp_decomposition() method"

        # Call tfp_decomposition()
        tfp = result.tfp_decomposition()

        # Test that it returns TFPDecomposition object
        from panelbox.frontier.utils import TFPDecomposition

        assert isinstance(
            tfp, TFPDecomposition
        ), "tfp_decomposition() should return TFPDecomposition object"

        # Test decompose()
        decomp = tfp.decompose()

        # Check structure
        assert isinstance(decomp, pd.DataFrame)
        assert "delta_tfp" in decomp.columns
        assert "delta_tc" in decomp.columns
        assert "delta_te" in decomp.columns
        assert "delta_se" in decomp.columns
        assert len(decomp) > 0

        # Check that components sum to total
        for _, row in decomp.iterrows():
            total = row["delta_tfp"]
            components = row["delta_tc"] + row["delta_te"] + row["delta_se"]
            assert (
                abs(total - components) < 1e-5
            ), f"Components should sum to total: {total} vs {components}"

        # Test aggregate_decomposition()
        agg = tfp.aggregate_decomposition()
        assert "mean_delta_tfp" in agg
        assert "mean_delta_tc" in agg
        assert "mean_delta_te" in agg
        assert "mean_delta_se" in agg

        # Test summary()
        summary = tfp.summary()
        assert isinstance(summary, str)
        assert "TFP DECOMPOSITION" in summary

        print("✓ TFP decomposition integration test passed!")


class TestMarginalEffectsIntegration:
    """Test marginal effects integration with result objects."""

    def test_marginal_effects_method_exists(self):
        """Test that marginal_effects() method exists on result."""
        # Generate data
        data = generate_test_data(n_firms=30, n_years=4)

        # Fit model
        model = FourComponentSFA(
            data=data,
            depvar="log_output",
            exog=["log_capital", "log_labor"],
            entity="firm",
            time="year",
            frontier_type="production",
        )

        result = model.fit(verbose=False)

        # Test that marginal_effects() method exists
        assert hasattr(result, "marginal_effects"), "Result should have marginal_effects() method"

        print("✓ Marginal effects method exists!")


if __name__ == "__main__":
    # Run tests
    test_tfp = TestTFPIntegration()
    test_tfp.test_tfp_decomposition_four_component()

    test_me = TestMarginalEffectsIntegration()
    test_me.test_marginal_effects_method_exists()

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED!")
    print("=" * 60)
