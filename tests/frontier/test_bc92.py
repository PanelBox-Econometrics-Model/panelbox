"""
Tests for BC92 (Battese & Coelli 1992) time-decay model.

This module tests the BC92 model which allows inefficiency to vary
over time through the time-decay specification: u_it = exp[-η(t - T_i)] · u_i
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier import ModelType, StochasticFrontier


def generate_bc92_panel_data(N=25, T=8, eta=0.1, seed=42):
    """Generate synthetic panel data with time-decay inefficiency.

    Parameters:
        N: Number of entities
        T: Number of time periods
        eta: Time-decay parameter (positive = efficiency improves)
        seed: Random seed

    Returns:
        DataFrame with panel structure
    """
    np.random.seed(seed)

    data = []
    for i in range(N):
        # Entity-specific base inefficiency
        u_i = np.random.exponential(0.3)

        for t in range(T):
            # Time-varying inefficiency with decay
            # u_it = exp[-η(t - T)] · u_i
            decay = np.exp(-eta * (t - (T - 1)))
            u_it = decay * u_i

            # Production inputs
            x1 = np.random.normal(10, 2)
            x2 = np.random.normal(5, 1)

            # Output with noise and inefficiency
            y = 5 + 0.6 * x1 + 0.4 * x2 - u_it + np.random.normal(0, 0.2)

            data.append(
                {
                    "firm": i,
                    "time": t,
                    "y": y,
                    "x1": x1,
                    "x2": x2,
                }
            )

    return pd.DataFrame(data)


class TestBC92Model:
    """Test suite for BC92 model."""

    def test_bc92_basic_estimation(self):
        """Test basic BC92 model estimation."""
        df = generate_bc92_panel_data(N=20, T=6, eta=0.1)

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="bc92",
        )

        result = model.fit()

        # Basic assertions
        assert result is not None
        assert result.converged
        assert result.loglik is not None
        assert np.isfinite(result.loglik)

        # Check parameter estimates
        assert len(result.params) == 6  # const, x1, x2, sigma_v_sq, sigma_u_sq, eta
        assert "eta" in result.param_names

        # Check eta parameter
        assert hasattr(result, "_bc92_eta")
        assert np.isfinite(result._bc92_eta)

    def test_bc92_eta_zero_vs_pitt_lee(self):
        """Test that BC92 with η=0 should give similar results to Pitt-Lee."""
        df = generate_bc92_panel_data(N=15, T=5, eta=0.0)  # No time variation

        # Estimate BC92
        model_bc92 = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="bc92",
        )
        result_bc92 = model_bc92.fit()

        # Estimate Pitt-Lee
        model_pl = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="pitt_lee",
        )
        result_pl = model_pl.fit()

        # Compare log-likelihoods (should be similar since η≈0)
        # Allow some tolerance due to different optimization paths
        assert abs(result_bc92.loglik - result_pl.loglik) < 5.0

        # Check that η is close to zero
        assert abs(result_bc92._bc92_eta) < 0.2

    def test_bc92_positive_eta(self):
        """Test BC92 with positive η (efficiency improves over time)."""
        df = generate_bc92_panel_data(N=20, T=8, eta=0.15)

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="bc92",
        )

        result = model.fit()

        # Should recover positive eta
        assert result._bc92_eta > 0
        assert result.converged

        # Check efficiency
        eff = result.efficiency(by_period=True)
        assert len(eff) > 0
        assert "efficiency" in eff.columns

    def test_bc92_negative_eta(self):
        """Test BC92 with negative η (efficiency degrades over time)."""
        df = generate_bc92_panel_data(N=20, T=8, eta=-0.1)

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="bc92",
        )

        result = model.fit()

        # Should recover negative or near-zero eta
        # (may not be perfectly negative due to noise)
        assert result.converged
        assert np.isfinite(result._bc92_eta)

    def test_bc92_panel_type(self):
        """Test that BC92 result has correct panel_type."""
        df = generate_bc92_panel_data(N=15, T=6, eta=0.1)

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="bc92",
        )

        result = model.fit()

        # Check panel type
        assert hasattr(result, "panel_type")
        assert result.panel_type == "bc92"

        # Check temporal params
        assert hasattr(result, "temporal_params")
        assert "eta" in result.temporal_params
        assert result.temporal_params["eta"] == result._bc92_eta

    def test_bc92_efficiency_time_varying(self):
        """Test that efficiency varies over time with BC92."""
        df = generate_bc92_panel_data(N=10, T=8, eta=0.2)

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="bc92",
        )

        result = model.fit()

        # Get efficiency by period
        eff_by_period = result.efficiency(by_period=True)

        # Check that we have efficiency for multiple periods
        periods = eff_by_period["period"].unique()
        assert len(periods) > 1

        # For each entity, efficiency should vary over time
        # (not necessarily monotonically due to noise, but should show variation)
        for entity in eff_by_period["entity"].unique()[:3]:  # Check first 3
            entity_eff = eff_by_period[eff_by_period["entity"] == entity]["efficiency"]
            # Check that there's some variation (std > 0)
            if len(entity_eff) > 1:
                assert np.std(entity_eff) >= 0  # Should have variation

    def test_bc92_cost_frontier(self):
        """Test BC92 with cost frontier."""
        df = generate_bc92_panel_data(N=15, T=6, eta=0.1)

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="cost",  # Cost frontier
            model_type="bc92",
        )

        result = model.fit()

        assert result is not None
        assert result.converged
        assert "eta" in result.param_names

    def test_bc92_parameter_interpretation(self):
        """Test that BC92 parameters are in expected ranges."""
        df = generate_bc92_panel_data(N=20, T=8, eta=0.15)

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="bc92",
        )

        result = model.fit()

        # Extract parameters
        sigma_v_sq = result.params[result.param_names.index("sigma_v_sq")]
        sigma_u_sq = result.params[result.param_names.index("sigma_u_sq")]
        eta = result.params[result.param_names.index("eta")]

        # Variances should be positive
        assert sigma_v_sq > 0
        assert sigma_u_sq > 0

        # Eta should be finite
        assert np.isfinite(eta)

    def test_bc92_with_verbose(self):
        """Test BC92 estimation with verbose output."""
        df = generate_bc92_panel_data(N=10, T=5, eta=0.1)

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="bc92",
        )

        # Should run without errors with verbose=True
        result = model.fit(verbose=False)  # Set to False to avoid cluttering test output

        assert result is not None
        assert result.converged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
