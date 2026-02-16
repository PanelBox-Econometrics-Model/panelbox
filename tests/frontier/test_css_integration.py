"""
Tests for CSS (Cornwell, Schmidt & Sickles 1990) model integration.

This module tests the integration of the distribution-free CSS model
into the main StochasticFrontier API.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier import ModelType, StochasticFrontier


def generate_css_panel_data(N=20, T=10, seed=42):
    """Generate synthetic panel data for CSS testing.

    Parameters:
        N: Number of entities
        T: Number of time periods
        seed: Random seed for reproducibility

    Returns:
        DataFrame with panel structure
    """
    np.random.seed(seed)

    data = []
    for i in range(N):
        # Entity-specific base productivity (time-invariant)
        alpha_i_base = 5 + np.random.normal(0, 0.5)

        # Entity-specific time trend coefficients
        theta_i_linear = np.random.normal(0.05, 0.02)
        theta_i_quad = np.random.normal(-0.005, 0.001)

        for t in range(T):
            # Time-varying productivity
            alpha_it = alpha_i_base + theta_i_linear * t + theta_i_quad * (t**2)

            # Exogenous variables
            x1 = np.random.normal(10, 2)
            x2 = np.random.normal(5, 1)

            # Output with noise
            y = alpha_it + 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1)

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


class TestCSSIntegration:
    """Test suite for CSS model integration with StochasticFrontier API."""

    def test_css_model_quadratic_time_trend(self):
        """Test CSS model with quadratic time trend (default)."""
        df = generate_css_panel_data(N=20, T=10)

        # Estimate CSS model via StochasticFrontier
        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="css",
            css_time_trend="quadratic",
        )

        result = model.fit()

        # Assertions
        assert result is not None
        assert hasattr(result, "_css_result")
        assert result._efficiency_it.shape == (20, 10)

        # Check efficiency in valid range
        assert np.all(result._efficiency_it > 0)
        assert np.all(result._efficiency_it <= 1)

        # Check R² is reasonable
        assert result._r_squared > 0.5

        # Check parameter estimates are reasonable
        assert len(result.params) == 2  # x1, x2
        assert 0.3 < result.params[0] < 0.7  # Approximately 0.5
        assert 0.1 < result.params[1] < 0.5  # Approximately 0.3

    def test_css_model_linear_time_trend(self):
        """Test CSS model with linear time trend."""
        df = generate_css_panel_data(N=15, T=8)

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="css",
            css_time_trend="linear",
        )

        result = model.fit()

        assert result is not None
        assert result._efficiency_it.shape == (15, 8)
        assert np.all(result._efficiency_it > 0)
        assert np.all(result._efficiency_it <= 1)

    def test_css_model_no_time_trend(self):
        """Test CSS model with no time trend (pure fixed effects)."""
        df = generate_css_panel_data(N=25, T=6)

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="css",
            css_time_trend="none",
        )

        result = model.fit()

        assert result is not None
        assert result._efficiency_it.shape == (25, 6)

        # With no time trend, efficiency should be constant over time for each entity
        # (but can vary across entities)
        for i in range(25):
            eff_i = result._efficiency_it[i, :]
            # Check that efficiency is relatively constant over time
            assert np.std(eff_i) < 0.1  # Small variation allowed due to frontier changes

    def test_css_auto_detection(self):
        """Test that CSS is auto-detected when css_time_trend is specified."""
        df = generate_css_panel_data(N=15, T=10)

        # Don't specify model_type, but specify css_time_trend
        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            css_time_trend="quadratic",  # This should trigger CSS
        )

        assert model.model_type == ModelType.CSS

        result = model.fit()
        assert hasattr(result, "_css_result")

    def test_css_default_time_trend(self):
        """Test that CSS defaults to quadratic when model_type='css' without css_time_trend."""
        df = generate_css_panel_data(N=15, T=10)

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="css",
            # No css_time_trend specified
        )

        assert model.css_time_trend == "quadratic"

        result = model.fit()
        assert result is not None

    def test_css_requires_panel_data(self):
        """Test that CSS raises error when entity or time is missing."""
        df = generate_css_panel_data(N=15, T=10)

        # Missing time
        with pytest.raises(ValueError, match="CSS model requires both entity and time"):
            StochasticFrontier(
                data=df,
                depvar="y",
                exog=["x1", "x2"],
                entity="firm",
                # time missing
                frontier="production",
                model_type="css",
            )

        # Missing entity
        with pytest.raises(ValueError, match="CSS model requires both entity and time"):
            StochasticFrontier(
                data=df,
                depvar="y",
                exog=["x1", "x2"],
                # entity missing
                time="time",
                frontier="production",
                model_type="css",
            )

    def test_css_invalid_time_trend(self):
        """Test that invalid time_trend specification raises error."""
        df = generate_css_panel_data(N=15, T=10)

        with pytest.raises(ValueError, match="css_time_trend must be"):
            StochasticFrontier(
                data=df,
                depvar="y",
                exog=["x1", "x2"],
                entity="firm",
                time="time",
                frontier="production",
                model_type="css",
                css_time_trend="invalid",
            )

    def test_css_cost_frontier(self):
        """Test CSS model with cost frontier."""
        df = generate_css_panel_data(N=20, T=10)

        # For cost frontier, reverse the efficiency logic
        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="cost",  # Cost frontier
            model_type="css",
            css_time_trend="quadratic",
        )

        result = model.fit()

        assert result is not None
        assert np.all(result._efficiency_it > 0)
        assert np.all(result._efficiency_it <= 1)

    def test_css_efficiency_by_entity(self):
        """Test CSS result methods for entity-level efficiency."""
        df = generate_css_panel_data(N=20, T=10)

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="css",
            css_time_trend="quadratic",
        )

        result = model.fit()

        # Test efficiency_by_entity method
        eff_by_entity = result._css_result.efficiency_by_entity()

        assert isinstance(eff_by_entity, pd.DataFrame)
        assert len(eff_by_entity) == 20
        assert "mean_efficiency" in eff_by_entity.columns
        assert "min_efficiency" in eff_by_entity.columns
        assert "max_efficiency" in eff_by_entity.columns
        assert "trend" in eff_by_entity.columns

    def test_css_efficiency_by_period(self):
        """Test CSS result methods for period-level efficiency."""
        df = generate_css_panel_data(N=20, T=10)

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="css",
            css_time_trend="quadratic",
        )

        result = model.fit()

        # Test efficiency_by_period method
        eff_by_period = result._css_result.efficiency_by_period()

        assert isinstance(eff_by_period, pd.DataFrame)
        assert len(eff_by_period) == 10
        assert "mean_efficiency" in eff_by_period.columns
        assert "min_efficiency" in eff_by_period.columns
        assert "max_efficiency" in eff_by_period.columns

    def test_css_with_few_periods(self):
        """Test CSS warning with T < 5."""
        df = generate_css_panel_data(N=20, T=4)

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            entity="firm",
            time="time",
            frontier="production",
            model_type="css",
            css_time_trend="quadratic",
        )

        # Should emit a warning but still run
        with pytest.warns(UserWarning, match="T = 4 is less than recommended minimum"):
            result = model.fit()

        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
