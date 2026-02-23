"""
Tests for Panel VAR forecasting functionality.

This module tests forecasting methods including:
- h-step-ahead forecasts
- Analytical confidence intervals
- Bootstrap confidence intervals
- Forecast evaluation metrics
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.var import PanelVAR, PanelVARData


@pytest.fixture
def stable_var_data():
    """
    Generate synthetic data from a stable Panel VAR(1) process.

    DGP:
    y_{it} = 0.5*y_{it-1} + e_{it}, where e ~ N(0, I)

    This is a stable process (eigenvalue = 0.5 < 1).
    """
    np.random.seed(42)
    N, T, K = 30, 50, 2

    # True parameters
    A1_true = np.array([[0.5, 0.0], [0.0, 0.5]])
    Sigma_true = np.eye(K)

    # Generate data
    data_list = []
    for i in range(N):
        y = np.zeros((T, K))
        y[0, :] = np.random.randn(K)

        for t in range(1, T):
            y[t, :] = A1_true @ y[t - 1, :] + np.random.multivariate_normal(np.zeros(K), Sigma_true)

        entity_data = pd.DataFrame(
            {"entity": i, "time": np.arange(T), "x1": y[:, 0], "x2": y[:, 1]}
        )
        data_list.append(entity_data)

    return pd.concat(data_list, ignore_index=True)


@pytest.fixture
def estimated_var(stable_var_data):
    """Estimate Panel VAR on stable data."""
    pvar_data = PanelVARData(
        stable_var_data, endog_vars=["x1", "x2"], entity_col="entity", time_col="time", lags=1
    )

    model = PanelVAR(pvar_data)
    result = model.fit(method="ols")
    return result


class TestForecastBasic:
    """Basic forecast functionality tests."""

    def test_forecast_returns_correct_shape(self, estimated_var):
        """Test that forecast returns correct shape."""
        steps = 10
        fcst = estimated_var.forecast(steps=steps)

        # Forecast should have shape (steps, N, K)
        assert fcst.forecasts.shape == (steps, estimated_var.N, estimated_var.K)

    def test_forecast_one_step(self, estimated_var):
        """Test 1-step-ahead forecast."""
        fcst = estimated_var.forecast(steps=1)

        # Should have 1 step
        assert fcst.forecasts.shape[0] == 1
        assert fcst.forecasts.shape[1] == estimated_var.N
        assert fcst.forecasts.shape[2] == estimated_var.K

    def test_forecast_multi_step(self, estimated_var):
        """Test multi-step-ahead forecast."""
        for steps in [5, 10, 20]:
            fcst = estimated_var.forecast(steps=steps)
            assert fcst.forecasts.shape[0] == steps

    def test_forecast_no_nans(self, estimated_var):
        """Test that forecasts don't contain NaNs."""
        fcst = estimated_var.forecast(steps=10)
        assert not np.any(np.isnan(fcst.forecasts))

    def test_forecast_deterministic(self, estimated_var):
        """Test that forecasts are deterministic (no randomness without bootstrap)."""
        fcst1 = estimated_var.forecast(steps=10, ci_method=None)
        fcst2 = estimated_var.forecast(steps=10, ci_method=None)

        np.testing.assert_allclose(fcst1.forecasts, fcst2.forecasts)


class TestForecastConvergence:
    """Test forecast convergence for stable VAR."""

    def test_stable_var_forecast_converges(self, estimated_var):
        """Test that forecasts converge to unconditional mean for stable VAR."""
        # For stable VAR, long-horizon forecasts → unconditional mean (≈ 0)
        fcst = estimated_var.forecast(steps=100)

        # Check that forecasts at horizon 100 are close to 0
        # (unconditional mean of stationary process with no intercept)
        fcst_h100 = fcst.forecasts[-1, :, :]  # Last forecast
        assert np.abs(fcst_h100).max() < 0.5  # Should be close to 0

    def test_forecast_uncertainty_increases_with_horizon(self, estimated_var):
        """Test that forecast uncertainty increases with horizon (analytical CIs)."""
        fcst = estimated_var.forecast(steps=20, ci_method="analytical")

        # CI width should increase with horizon
        widths = []
        for h in range(20):
            width_h = np.mean(fcst.ci_upper[h, :, :] - fcst.ci_lower[h, :, :])
            widths.append(width_h)

        # Check that widths are increasing (at least on average)
        # First half should be narrower than second half
        assert np.mean(widths[:10]) < np.mean(widths[10:])


class TestForecastConfidenceIntervals:
    """Test confidence interval methods."""

    def test_analytical_ci_shape(self, estimated_var):
        """Test analytical CI returns correct shape."""
        fcst = estimated_var.forecast(steps=10, ci_method="analytical", ci_level=0.95)

        assert fcst.ci_lower.shape == fcst.forecasts.shape
        assert fcst.ci_upper.shape == fcst.forecasts.shape

    def test_analytical_ci_bounds(self, estimated_var):
        """Test that analytical CIs bound the forecast."""
        fcst = estimated_var.forecast(steps=10, ci_method="analytical")

        # Lower bound <= forecast <= upper bound
        assert np.all(fcst.ci_lower <= fcst.forecasts)
        assert np.all(fcst.forecasts <= fcst.ci_upper)

    def test_bootstrap_ci_shape(self, estimated_var):
        """Test bootstrap CI returns correct shape."""
        fcst = estimated_var.forecast(steps=5, ci_method="bootstrap", n_bootstrap=100)

        assert fcst.ci_lower.shape == fcst.forecasts.shape
        assert fcst.ci_upper.shape == fcst.forecasts.shape

    def test_bootstrap_ci_bounds(self, estimated_var):
        """Test that bootstrap CIs bound the forecast (approximately)."""
        fcst = estimated_var.forecast(steps=5, ci_method="bootstrap", n_bootstrap=100, seed=42)

        # Bootstrap CIs should contain forecast (mostly)
        # Allow some tolerance as bootstrap is stochastic
        pct_in_ci = np.mean((fcst.ci_lower <= fcst.forecasts) & (fcst.forecasts <= fcst.ci_upper))
        assert pct_in_ci > 0.7  # At least 70% should be in CI

    def test_ci_level_affects_width(self, estimated_var):
        """Test that higher confidence level gives wider CIs."""
        fcst_90 = estimated_var.forecast(steps=10, ci_method="analytical", ci_level=0.90)
        fcst_95 = estimated_var.forecast(steps=10, ci_method="analytical", ci_level=0.95)

        # 95% CI should be wider than 90% CI
        width_90 = np.mean(fcst_90.ci_upper - fcst_90.ci_lower)
        width_95 = np.mean(fcst_95.ci_upper - fcst_95.ci_lower)

        assert width_95 > width_90

    def test_bootstrap_reproducibility(self, estimated_var):
        """Test that bootstrap with same seed gives same results."""
        fcst1 = estimated_var.forecast(steps=5, ci_method="bootstrap", n_bootstrap=100, seed=42)
        fcst2 = estimated_var.forecast(steps=5, ci_method="bootstrap", n_bootstrap=100, seed=42)

        np.testing.assert_allclose(fcst1.ci_lower, fcst2.ci_lower)
        np.testing.assert_allclose(fcst1.ci_upper, fcst2.ci_upper)


class TestForecastEvaluation:
    """Test forecast evaluation metrics."""

    def test_to_dataframe(self, estimated_var):
        """Test that forecasts can be converted to DataFrame."""
        fcst = estimated_var.forecast(steps=10)

        # Get forecasts for first entity
        df = fcst.to_dataframe(entity=0)

        assert isinstance(df, pd.DataFrame)
        assert "x1" in df.columns
        assert "x2" in df.columns
        assert len(df) == 10  # steps

    def test_plot_returns_figure(self, estimated_var):
        """Test that plot method returns a figure."""
        pytest.importorskip("matplotlib")

        fcst = estimated_var.forecast(steps=10)
        fig = fcst.plot(entity=0, variable="x1", show=False)

        # Should return matplotlib Figure
        assert fig is not None

    def test_evaluate_with_actual_data(self, stable_var_data, estimated_var):
        """Test forecast evaluation metrics with actual data."""
        # Split data: use last 10 periods as test set
        train_data = stable_var_data[stable_var_data["time"] < 40]
        test_data = stable_var_data[stable_var_data["time"] >= 40]

        # Re-estimate on train
        pvar_data = PanelVARData(
            train_data, endog_vars=["x1", "x2"], entity_col="entity", time_col="time", lags=1
        )
        model = PanelVAR(pvar_data)
        result = model.fit(method="ols")

        # Forecast
        fcst = result.forecast(steps=10)

        # Evaluate (passing actual test data)
        metrics = fcst.evaluate(test_data)

        # Should have RMSE, MAE, MAPE
        assert "RMSE" in metrics
        assert "MAE" in metrics
        assert "MAPE" in metrics

        # Metrics should be positive
        assert metrics["RMSE"] > 0
        assert metrics["MAE"] > 0

    def test_forecast_without_ci(self, estimated_var):
        """Test forecast without confidence intervals."""
        fcst = estimated_var.forecast(steps=10, ci_method=None)

        # CIs should be None
        assert fcst.ci_lower is None
        assert fcst.ci_upper is None


class TestForecastEdgeCases:
    """Test edge cases and error handling."""

    def test_forecast_zero_steps_raises(self, estimated_var):
        """Test that zero steps raises error."""
        with pytest.raises(ValueError, match="steps must be positive"):
            estimated_var.forecast(steps=0)

    def test_forecast_negative_steps_raises(self, estimated_var):
        """Test that negative steps raises error."""
        with pytest.raises(ValueError, match="steps must be positive"):
            estimated_var.forecast(steps=-5)

    def test_invalid_ci_method_raises(self, estimated_var):
        """Test that invalid ci_method raises error."""
        with pytest.raises(ValueError, match="Unknown ci_method"):
            estimated_var.forecast(steps=10, ci_method="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
