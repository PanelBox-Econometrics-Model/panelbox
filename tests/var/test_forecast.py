"""
Tests for Panel VAR forecasting functionality.

This module tests forecasting methods including:
- h-step-ahead forecasts
- Analytical confidence intervals
- Bootstrap confidence intervals
- Forecast evaluation metrics
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from panelbox.var import PanelVAR, PanelVARData
from panelbox.var.forecast import ForecastResult


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


# ---------------------------------------------------------------------------
# Helper to build a ForecastResult without fitting a model
# ---------------------------------------------------------------------------
def _make_forecast_result(
    horizon=5,
    n_entities=3,
    n_vars=2,
    with_ci=True,
    endog_names=None,
    entity_names=None,
    seed=42,
):
    """Build a synthetic ForecastResult for unit testing."""
    rng = np.random.default_rng(seed)
    forecasts = rng.standard_normal((horizon, n_entities, n_vars))
    ci_lower = forecasts - 0.5 if with_ci else None
    ci_upper = forecasts + 0.5 if with_ci else None
    endog_names = endog_names or [f"y{k + 1}" for k in range(n_vars)]
    entity_names = entity_names or [f"E{i}" for i in range(n_entities)]
    return ForecastResult(
        forecasts=forecasts,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        endog_names=endog_names,
        entity_names=entity_names,
        ci_level=0.95,
        method="iterative",
        ci_method="bootstrap" if with_ci else None,
    )


# ===================================================================
# Tests for ForecastResult.__init__ with 2D input  (lines 96-104)
# ===================================================================
class TestForecastResultInit2D:
    """Tests for the 2D-to-3D expansion logic in ForecastResult.__init__."""

    def test_2d_forecasts_expanded_to_3d(self):
        """2D (steps, K) forecasts are expanded to (steps, 1, K)."""
        rng = np.random.default_rng(0)
        fc2d = rng.standard_normal((5, 2))
        fr = ForecastResult(forecasts=fc2d, endog_names=["y1", "y2"])

        assert fr.forecasts.ndim == 3
        assert fr.forecasts.shape == (5, 1, 2)
        assert fr.N == 1
        assert fr.K == 2
        assert fr.horizon == 5

    def test_2d_ci_lower_expanded(self):
        """2D ci_lower is expanded to 3D alongside forecasts."""
        rng = np.random.default_rng(1)
        fc2d = rng.standard_normal((4, 3))
        ci_lo = fc2d - 1.0
        fr = ForecastResult(
            forecasts=fc2d,
            ci_lower=ci_lo,
            endog_names=["a", "b", "c"],
        )
        assert fr.ci_lower.shape == (4, 1, 3)
        np.testing.assert_allclose(fr.ci_lower[:, 0, :], fc2d - 1.0)

    def test_2d_ci_upper_expanded(self):
        """2D ci_upper is expanded to 3D alongside forecasts."""
        rng = np.random.default_rng(2)
        fc2d = rng.standard_normal((6, 2))
        ci_hi = fc2d + 2.0
        fr = ForecastResult(
            forecasts=fc2d,
            ci_upper=ci_hi,
            endog_names=["y1", "y2"],
        )
        assert fr.ci_upper.shape == (6, 1, 2)
        np.testing.assert_allclose(fr.ci_upper[:, 0, :], fc2d + 2.0)

    def test_2d_both_ci_expanded(self):
        """Both ci_lower and ci_upper are expanded when forecasts are 2D."""
        rng = np.random.default_rng(3)
        fc2d = rng.standard_normal((3, 2))
        ci_lo = fc2d - 0.5
        ci_hi = fc2d + 0.5
        fr = ForecastResult(
            forecasts=fc2d,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            endog_names=["y1", "y2"],
        )
        assert fr.ci_lower.shape == (3, 1, 2)
        assert fr.ci_upper.shape == (3, 1, 2)

    def test_2d_default_entity_names(self):
        """Default entity name is generated for single-entity 2D input."""
        fc2d = np.zeros((2, 2))
        fr = ForecastResult(forecasts=fc2d, endog_names=["y1", "y2"])
        assert fr.entity_names == ["entity_0"]

    def test_3d_passthrough(self):
        """3D input is kept as-is."""
        fc3d = np.zeros((4, 3, 2))
        fr = ForecastResult(
            forecasts=fc3d,
            endog_names=["y1", "y2"],
            entity_names=["A", "B", "C"],
        )
        assert fr.forecasts.shape == (4, 3, 2)
        assert fr.N == 3

    def test_horizon_mismatch_raises(self):
        """Explicit horizon that conflicts with forecasts shape raises ValueError."""
        fc2d = np.zeros((5, 2))
        with pytest.raises(ValueError, match="inconsistent"):
            ForecastResult(forecasts=fc2d, horizon=10, endog_names=["y1", "y2"])

    def test_endog_names_length_mismatch_raises(self):
        """Wrong number of endog_names raises ValueError."""
        fc3d = np.zeros((3, 2, 2))
        with pytest.raises(ValueError, match="endog_names"):
            ForecastResult(
                forecasts=fc3d,
                endog_names=["y1"],  # only 1, but K=2
                entity_names=["A", "B"],
            )

    def test_entity_names_length_mismatch_raises(self):
        """Wrong number of entity_names raises ValueError."""
        fc3d = np.zeros((3, 2, 2))
        with pytest.raises(ValueError, match="entity_names"):
            ForecastResult(
                forecasts=fc3d,
                endog_names=["y1", "y2"],
                entity_names=["A"],  # only 1, but N=2
            )


# ===================================================================
# Tests for to_dataframe()  (lines 123-205)
# ===================================================================
class TestToDataFrame:
    """Tests for ForecastResult.to_dataframe() method."""

    def test_single_entity_wide_format(self):
        """entity=0 with variable=None returns wide format."""
        fr = _make_forecast_result(horizon=5, n_entities=3, n_vars=2)
        df = fr.to_dataframe(entity=0)

        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "horizon"
        assert len(df) == 5
        # Should have variable columns + CI columns
        assert "y1" in df.columns
        assert "y2" in df.columns
        assert "y1_ci_lower" in df.columns
        assert "y2_ci_upper" in df.columns

    def test_single_entity_wide_format_no_ci(self):
        """Wide format without CIs has only variable columns."""
        fr = _make_forecast_result(with_ci=False)
        df = fr.to_dataframe(entity=0)
        assert "y1" in df.columns
        assert "y1_ci_lower" not in df.columns
        assert "y1_ci_upper" not in df.columns

    def test_entity_none_returns_long_format(self):
        """entity=None returns long format with all entities."""
        fr = _make_forecast_result(horizon=3, n_entities=2, n_vars=2)
        df = fr.to_dataframe(entity=None)

        assert isinstance(df, pd.DataFrame)
        # Long format: 3 horizons * 2 entities * 2 variables = 12 rows
        assert len(df) == 3 * 2 * 2
        assert "forecast" in df.columns or "forecast" in df.reset_index().columns

    def test_entity_as_string_name(self):
        """Passing entity as string name works."""
        fr = _make_forecast_result(
            entity_names=["Alpha", "Beta", "Gamma"],
        )
        df = fr.to_dataframe(entity="Beta")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # horizon=5

    def test_variable_filter(self):
        """Passing variable filters to single variable (long format)."""
        fr = _make_forecast_result(horizon=4, n_entities=3, n_vars=2)
        df = fr.to_dataframe(entity=None, variable="y1")

        # Long format: 4 horizons * 3 entities * 1 variable = 12 rows
        assert len(df) == 4 * 3

    def test_single_entity_single_variable_long_format(self):
        """entity=0, variable='y1' returns long format with 1 variable."""
        fr = _make_forecast_result(horizon=5, n_entities=3, n_vars=2)
        df = fr.to_dataframe(entity=0, variable="y1")

        # 5 horizons * 1 entity * 1 variable = 5 rows
        assert len(df) == 5
        # Only 1 variable and 1 entity: should have horizon index only
        # (no entity or variable in index because both have len 1)
        assert "forecast" in df.columns

    def test_long_format_has_ci_columns(self):
        """Long format includes ci_lower/ci_upper when CIs exist."""
        fr = _make_forecast_result(horizon=3, n_entities=2, n_vars=1)
        df = fr.to_dataframe(entity=None)
        flat = df.reset_index()
        assert "ci_lower" in flat.columns
        assert "ci_upper" in flat.columns

    def test_long_format_no_ci_columns(self):
        """Long format omits ci columns when CIs are None."""
        fr = _make_forecast_result(horizon=3, n_entities=2, n_vars=1, with_ci=False)
        df = fr.to_dataframe(entity=None)
        flat = df.reset_index()
        assert "ci_lower" not in flat.columns
        assert "ci_upper" not in flat.columns

    def test_wide_values_match_forecasts(self):
        """Wide-format values match the underlying forecast array."""
        fr = _make_forecast_result(horizon=4, n_entities=2, n_vars=2)
        df = fr.to_dataframe(entity=1)

        np.testing.assert_allclose(df["y1"].values, fr.forecasts[:, 1, 0])
        np.testing.assert_allclose(df["y2"].values, fr.forecasts[:, 1, 1])


# ===================================================================
# Tests for _plot_plotly()  (lines 270-352)
# ===================================================================
class TestPlotPlotly:
    """Tests for Plotly plotting backend."""

    @pytest.fixture(autouse=True)
    def _require_plotly(self):
        pytest.importorskip("plotly")

    def test_plotly_basic(self):
        """Basic plotly plot returns a Figure object."""
        import plotly.graph_objects as go

        fr = _make_forecast_result()
        fig = fr.plot(entity=0, variable="y1", backend="plotly", show=False)
        assert isinstance(fig, go.Figure)

    def test_plotly_without_ci(self):
        """Plotly plot without CIs should still work."""
        import plotly.graph_objects as go

        fr = _make_forecast_result(with_ci=False)
        fig = fr.plot(entity=0, variable="y1", backend="plotly", show=False)
        assert isinstance(fig, go.Figure)
        # Only 1 trace (forecast line), no CI traces
        trace_names = [t.name for t in fig.data]
        assert "Forecast" in trace_names
        assert all("CI" not in n for n in trace_names)

    def test_plotly_with_ci_traces(self):
        """Plotly plot with CIs has upper/lower traces."""
        fr = _make_forecast_result(with_ci=True)
        fig = fr.plot(entity=0, variable="y1", backend="plotly", show=False)
        trace_names = [t.name for t in fig.data]
        assert any("CI Upper" in n for n in trace_names)
        assert any("CI Lower" in n for n in trace_names)

    def test_plotly_with_actual(self):
        """Plotly plot with actual values adds an Actual trace."""
        fr = _make_forecast_result(horizon=5)
        actual = np.ones(5)
        fig = fr.plot(entity=0, variable="y1", actual=actual, backend="plotly", show=False)
        trace_names = [t.name for t in fig.data]
        assert "Actual" in trace_names

    def test_plotly_entity_as_string(self):
        """Plotly plot works when entity is passed as string name."""
        import plotly.graph_objects as go

        fr = _make_forecast_result(entity_names=["Alpha", "Beta", "Gamma"])
        fig = fr.plot(entity="Beta", variable="y1", backend="plotly", show=False)
        assert isinstance(fig, go.Figure)
        assert "Beta" in fig.layout.title.text

    def test_plotly_title_contains_variable_and_entity(self):
        """Title includes the variable and entity name."""
        fr = _make_forecast_result(entity_names=["City_A", "City_B", "City_C"])
        fig = fr.plot(entity=0, variable="y2", backend="plotly", show=False)
        assert "y2" in fig.layout.title.text
        assert "City_A" in fig.layout.title.text


# ===================================================================
# Tests for _plot_matplotlib()  (lines 354-407)
# ===================================================================
class TestPlotMatplotlib:
    """Tests for Matplotlib plotting backend."""

    def setup_method(self):
        plt.close("all")

    def test_matplotlib_basic(self):
        """Basic matplotlib plot returns a Figure."""
        fr = _make_forecast_result()
        fig = fr.plot(entity=0, variable="y1", backend="matplotlib", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_matplotlib_without_ci(self):
        """Matplotlib plot without CIs shows no fill."""
        fr = _make_forecast_result(with_ci=False)
        fig = fr.plot(entity=0, variable="y1", backend="matplotlib", show=False)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        # No PolyCollection (fill_between) when CIs are absent
        from matplotlib.collections import PolyCollection

        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(polys) == 0
        plt.close(fig)

    def test_matplotlib_with_ci_fill(self):
        """Matplotlib plot with CIs has a fill_between region."""
        fr = _make_forecast_result(with_ci=True)
        fig = fr.plot(entity=0, variable="y1", backend="matplotlib", show=False)
        ax = fig.axes[0]
        from matplotlib.collections import PolyCollection

        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(polys) >= 1
        plt.close(fig)

    def test_matplotlib_with_actual(self):
        """Matplotlib plot with actual values adds an extra line."""
        fr = _make_forecast_result(horizon=5)
        actual = np.linspace(0, 1, 5)
        fig = fr.plot(entity=0, variable="y1", actual=actual, backend="matplotlib", show=False)
        ax = fig.axes[0]
        labels = [l.get_label() for l in ax.get_lines()]
        assert "Actual" in labels
        plt.close(fig)

    def test_matplotlib_entity_as_string(self):
        """Matplotlib plot works when entity is passed as string name."""
        fr = _make_forecast_result(entity_names=["X", "Y", "Z"])
        fig = fr.plot(entity="Y", variable="y1", backend="matplotlib", show=False)
        assert isinstance(fig, plt.Figure)
        assert "Y" in fig.axes[0].get_title()
        plt.close(fig)

    def test_matplotlib_axis_labels(self):
        """Matplotlib plot has correct axis labels."""
        fr = _make_forecast_result()
        fig = fr.plot(entity=0, variable="y2", backend="matplotlib", show=False)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Horizon"
        assert ax.get_ylabel() == "y2"
        plt.close(fig)

    def test_matplotlib_custom_figsize(self):
        """Passing figsize to matplotlib backend works."""
        fr = _make_forecast_result()
        fig = fr.plot(
            entity=0,
            variable="y1",
            backend="matplotlib",
            show=False,
            figsize=(12, 4),
        )
        w, h = fig.get_size_inches()
        assert w == pytest.approx(12)
        assert h == pytest.approx(4)
        plt.close(fig)


# ===================================================================
# Tests for evaluate()  (lines 409-501)
# ===================================================================
class TestEvaluateMethod:
    """Tests for ForecastResult.evaluate() forecast accuracy method."""

    def test_evaluate_basic_array(self):
        """evaluate() with 3D numpy array returns metrics."""
        fr = _make_forecast_result(horizon=5, n_entities=3, n_vars=2)
        # Perfect forecast: actual == forecasts
        actual = fr.forecasts.copy()
        metrics = fr.evaluate(actual)

        # With multiple entities, evaluate returns aggregated Series
        assert isinstance(metrics, pd.Series)
        assert "RMSE" in metrics.index
        assert "MAE" in metrics.index
        assert "MAPE" in metrics.index
        # Perfect forecast should give 0 error
        assert metrics["RMSE"] == pytest.approx(0.0, abs=1e-12)
        assert metrics["MAE"] == pytest.approx(0.0, abs=1e-12)

    def test_evaluate_with_errors(self):
        """evaluate() with different actual values returns positive metrics."""
        fr = _make_forecast_result(horizon=5, n_entities=3, n_vars=2)
        rng = np.random.default_rng(99)
        actual = fr.forecasts + rng.standard_normal(fr.forecasts.shape)
        metrics = fr.evaluate(actual)

        assert metrics["RMSE"] > 0
        assert metrics["MAE"] > 0

    def test_evaluate_single_entity(self):
        """evaluate() with entity filter returns DataFrame for that entity."""
        fr = _make_forecast_result(horizon=5, n_entities=3, n_vars=2)
        actual = fr.forecasts.copy() + 0.1
        result = fr.evaluate(actual, entity=1)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 2 variables
        assert "RMSE" in result.columns
        assert all(result["entity"] == "E1")

    def test_evaluate_entity_by_name(self):
        """evaluate() with entity as string name works."""
        fr = _make_forecast_result(horizon=5, n_entities=2, entity_names=["Alpha", "Beta"])
        actual = fr.forecasts.copy() + 0.2
        result = fr.evaluate(actual, entity="Alpha")

        assert isinstance(result, pd.DataFrame)
        assert all(result["entity"] == "Alpha")

    def test_evaluate_2d_actual(self):
        """evaluate() with 2D actual (horizon, K) expands to 3D internally."""
        rng = np.random.default_rng(7)
        fc2d = rng.standard_normal((5, 2))
        fr = ForecastResult(forecasts=fc2d, endog_names=["y1", "y2"])

        actual_2d = fc2d + 0.1
        result = fr.evaluate(actual_2d)

        # 1 entity * 2 variables = 2 results -> aggregated Series
        assert isinstance(result, pd.Series)
        assert "RMSE" in result.index
        assert result["RMSE"] > 0

    def test_evaluate_dataframe_input(self):
        """evaluate() with DataFrame input works."""
        fr = _make_forecast_result(
            horizon=5,
            n_entities=2,
            n_vars=2,
            endog_names=["y1", "y2"],
            entity_names=["E0", "E1"],
        )
        # Build a DataFrame mimicking panel data sorted by entity then time
        rows = []
        for i in range(2):  # 2 entities
            for h in range(5):  # 5 periods
                rows.append(
                    {
                        "entity": f"E{i}",
                        "y1": fr.forecasts[h, i, 0] + 0.05,
                        "y2": fr.forecasts[h, i, 1] + 0.05,
                    }
                )
        actual_df = pd.DataFrame(rows)
        metrics = fr.evaluate(actual_df)

        assert isinstance(metrics, pd.Series)
        assert metrics["RMSE"] > 0

    def test_evaluate_nan_handling(self):
        """evaluate() skips NaN values in actual data."""
        fr = _make_forecast_result(horizon=5, n_entities=1, n_vars=1)
        actual = fr.forecasts.copy() + 0.5
        # Inject NaN
        actual[2, 0, 0] = np.nan

        result = fr.evaluate(actual)
        # Should still produce valid results (single entity returns DataFrame)
        assert isinstance(result, pd.DataFrame)
        assert not np.isnan(result["RMSE"].iloc[0])

    def test_evaluate_all_nan_skips_entity(self):
        """evaluate() skips entity/variable combination if all values are NaN."""
        rng = np.random.default_rng(10)
        fc2d = rng.standard_normal((3, 1))
        fr = ForecastResult(forecasts=fc2d, endog_names=["y1"])

        actual = np.full((3, 1, 1), np.nan)
        result = fr.evaluate(actual)

        # All NaN: the loop's `continue` branch fires, so result is empty
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_evaluate_actual_too_short_raises(self):
        """evaluate() raises ValueError if actual has fewer periods than horizon."""
        fr = _make_forecast_result(horizon=5, n_entities=1, n_vars=1)
        actual_short = np.zeros((3, 1, 1))  # only 3, need 5
        with pytest.raises(ValueError, match="actual has 3 periods"):
            fr.evaluate(actual_short)

    def test_evaluate_mape_zero_actual(self):
        """MAPE handles zero actual values gracefully."""
        rng = np.random.default_rng(20)
        fc = rng.standard_normal((4, 1, 1))
        fr = ForecastResult(forecasts=fc, endog_names=["y1"])

        actual = np.zeros((4, 1, 1))  # All zeros -> division by zero in MAPE
        result = fr.evaluate(actual)

        # Should not crash; MAPE may be NaN because all pct_errors are inf
        assert isinstance(result, pd.DataFrame)


# ===================================================================
# Tests for summary()  (lines 503-534)
# ===================================================================
class TestSummaryMethod:
    """Tests for ForecastResult.summary() method."""

    def test_summary_basic(self):
        """summary() returns a non-empty string."""
        fr = _make_forecast_result()
        s = fr.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contains_header(self):
        """summary() contains the Panel VAR Forecast Results header."""
        fr = _make_forecast_result()
        s = fr.summary()
        assert "Panel VAR Forecast Results" in s

    def test_summary_contains_dimensions(self):
        """summary() reports N, K, and horizon."""
        fr = _make_forecast_result(horizon=7, n_entities=4, n_vars=3)
        s = fr.summary()
        assert "Number of entities: 4" in s
        assert "Number of variables: 3" in s
        assert "Forecast horizon: 7" in s

    def test_summary_contains_method(self):
        """summary() reports the forecast method."""
        fr = _make_forecast_result()
        s = fr.summary()
        assert "Forecast method: iterative" in s

    def test_summary_with_ci(self):
        """summary() reports confidence level and CI method when CIs exist."""
        fr = _make_forecast_result(with_ci=True)
        s = fr.summary()
        assert "Confidence level: 95.0%" in s
        assert "CI method: bootstrap" in s

    def test_summary_without_ci(self):
        """summary() omits CI lines when no CIs are present."""
        fr = _make_forecast_result(with_ci=False)
        s = fr.summary()
        assert "Confidence level" not in s
        assert "CI method" not in s

    def test_summary_variable_names(self):
        """summary() lists variable names."""
        fr = _make_forecast_result(endog_names=["gdp", "inflation"])
        s = fr.summary()
        assert "gdp" in s
        assert "inflation" in s

    def test_summary_entity_names_short(self):
        """summary() lists all entities when N <= 10."""
        fr = _make_forecast_result(n_entities=3, entity_names=["US", "UK", "DE"])
        s = fr.summary()
        assert "US" in s
        assert "UK" in s
        assert "DE" in s

    def test_summary_entity_names_long(self):
        """summary() truncates entity listing when N > 10."""
        names = [f"country_{i}" for i in range(15)]
        fr = _make_forecast_result(n_entities=15, entity_names=names)
        s = fr.summary()
        assert "country_0" in s
        assert "country_14" in s
        assert "..." in s

    def test_summary_separator_lines(self):
        """summary() contains separator lines."""
        fr = _make_forecast_result()
        s = fr.summary()
        assert "=" * 70 in s


# ===================================================================
# Tests for __repr__
# ===================================================================
class TestRepr:
    """Tests for ForecastResult.__repr__."""

    def test_repr_content(self):
        """repr contains key information."""
        fr = _make_forecast_result(horizon=5, n_entities=3, n_vars=2, with_ci=True)
        r = repr(fr)
        assert "ForecastResult" in r
        assert "horizon=5" in r
        assert "N=3" in r
        assert "K=2" in r
        assert "has_ci=True" in r

    def test_repr_no_ci(self):
        """repr shows has_ci=False when no CIs."""
        fr = _make_forecast_result(with_ci=False)
        r = repr(fr)
        assert "has_ci=False" in r


# ===================================================================
# Additional coverage tests for uncovered lines in forecast.py
# ===================================================================


class TestToDataFrameLongFormatCoverage:
    """
    Additional tests for to_dataframe() long-format branches
    (lines 176-205) to improve coverage of multi-entity and
    variable-filter paths.
    """

    def test_multi_entity_no_filter_long_format(self):
        """All entities, all variables -> long format with entity+variable+horizon index."""
        fr = _make_forecast_result(horizon=3, n_entities=3, n_vars=2)
        df = fr.to_dataframe()  # entity=None, variable=None
        assert isinstance(df, pd.DataFrame)
        # 3 horizons * 3 entities * 2 variables = 18 rows
        assert len(df) == 18
        # Should be MultiIndex with entity, variable, horizon
        flat = df.reset_index()
        assert "entity" in flat.columns
        assert "variable" in flat.columns
        assert "horizon" in flat.columns

    def test_multi_entity_single_variable_long_format(self):
        """All entities, single variable filter -> long format."""
        fr = _make_forecast_result(horizon=4, n_entities=3, n_vars=2)
        df = fr.to_dataframe(entity=None, variable="y2")
        # 4 horizons * 3 entities = 12 rows
        assert len(df) == 12
        flat = df.reset_index()
        assert "entity" in flat.columns

    def test_single_entity_by_name_wide_format(self):
        """Pass entity as string name -> wide format."""
        fr = _make_forecast_result(
            horizon=3,
            n_entities=2,
            entity_names=["Alpha", "Beta"],
        )
        df = fr.to_dataframe(entity="Alpha")
        assert len(df) == 3
        assert "y1" in df.columns
        assert "y2" in df.columns

    def test_single_entity_single_variable_returns_forecast_column(self):
        """entity=1, variable='y2' -> long format with 'forecast' column."""
        fr = _make_forecast_result(horizon=5, n_entities=3, n_vars=2)
        df = fr.to_dataframe(entity=1, variable="y2")
        assert len(df) == 5
        assert "forecast" in df.columns

    def test_long_format_values_match_forecasts(self):
        """Values in long-format DataFrame match underlying array."""
        fr = _make_forecast_result(horizon=3, n_entities=2, n_vars=2)
        df = fr.to_dataframe(entity=None)
        flat = df.reset_index()
        # Check first entity, first variable, first horizon
        row = flat[(flat["entity"] == "E0") & (flat["variable"] == "y1") & (flat["horizon"] == 1)]
        assert len(row) == 1
        np.testing.assert_allclose(row["forecast"].values[0], fr.forecasts[0, 0, 0])

    def test_wide_format_ci_values_match(self):
        """CI values in wide format match underlying arrays."""
        fr = _make_forecast_result(horizon=4, n_entities=2, with_ci=True)
        df = fr.to_dataframe(entity=0)
        np.testing.assert_allclose(df["y1_ci_lower"].values, fr.ci_lower[:, 0, 0])
        np.testing.assert_allclose(df["y2_ci_upper"].values, fr.ci_upper[:, 0, 1])


class TestPlotlyAdditionalCoverage:
    """Additional Plotly plot tests for uncovered branches (lines 262-315)."""

    @pytest.fixture(autouse=True)
    def _require_plotly(self):
        pytest.importorskip("plotly")

    def test_plotly_plot_all_traces_with_ci_and_actual(self):
        """Plotly plot with CI and actual has at least 4 traces."""
        import plotly.graph_objects as go

        fr = _make_forecast_result(horizon=5, with_ci=True)
        actual = np.ones(5) * 0.5
        fig = fr.plot(entity=0, variable="y1", actual=actual, backend="plotly", show=False)
        assert isinstance(fig, go.Figure)
        # Forecast + CI Upper + CI Lower + Actual = 4 traces
        assert len(fig.data) >= 4
        trace_names = [t.name for t in fig.data]
        assert "Forecast" in trace_names
        assert "Actual" in trace_names
        assert any("CI Upper" in n for n in trace_names)
        assert any("CI Lower" in n for n in trace_names)

    def test_plotly_entity_by_index(self):
        """Plotly plot using integer entity index."""
        import plotly.graph_objects as go

        fr = _make_forecast_result(entity_names=["City_A", "City_B", "City_C"])
        fig = fr.plot(entity=2, variable="y1", backend="plotly", show=False)
        assert isinstance(fig, go.Figure)
        assert "City_C" in fig.layout.title.text


class TestPlotMatplotlibAdditionalCoverage:
    """Additional matplotlib plot tests for uncovered branches (lines 367-407)."""

    def setup_method(self):
        plt.close("all")

    def test_matplotlib_with_ci_and_actual_together(self):
        """Matplotlib plot with both CI and actual data."""
        fr = _make_forecast_result(horizon=5, with_ci=True)
        actual = np.linspace(-1, 1, 5)
        fig = fr.plot(entity=0, variable="y1", actual=actual, backend="matplotlib", show=False)
        ax = fig.axes[0]

        # Should have forecast line + actual line
        labels = [line.get_label() for line in ax.get_lines()]
        assert "Forecast" in labels
        assert "Actual" in labels

        # Should have CI fill
        from matplotlib.collections import PolyCollection

        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(polys) >= 1
        plt.close(fig)

    def test_matplotlib_title_format(self):
        """Matplotlib title includes variable and entity name."""
        fr = _make_forecast_result(entity_names=["US", "UK", "JP"])
        fig = fr.plot(entity=2, variable="y2", backend="matplotlib", show=False)
        title = fig.axes[0].get_title()
        assert "y2" in title
        assert "JP" in title
        plt.close(fig)


class TestEvaluateAdditionalCoverage:
    """Additional tests for evaluate() to cover uncovered branches (lines 451-528)."""

    def test_evaluate_returns_series_for_multi_entity(self):
        """evaluate() with multiple entities and no entity filter returns aggregated Series."""
        fr = _make_forecast_result(horizon=5, n_entities=3, n_vars=2)
        actual = fr.forecasts.copy() + 0.1
        metrics = fr.evaluate(actual)
        assert isinstance(metrics, pd.Series)
        assert "RMSE" in metrics.index
        assert "MAE" in metrics.index
        assert "MAPE" in metrics.index

    def test_evaluate_single_entity_filter_returns_dataframe(self):
        """evaluate() with entity filter returns per-variable DataFrame."""
        fr = _make_forecast_result(horizon=5, n_entities=3, n_vars=2)
        actual = fr.forecasts.copy() + 0.2
        result = fr.evaluate(actual, entity=0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 2 variables
        assert "RMSE" in result.columns
        assert "MAE" in result.columns
        assert "MAPE" in result.columns
        assert all(result["entity"] == "E0")

    def test_evaluate_entity_by_string(self):
        """evaluate() with entity passed as string name."""
        fr = _make_forecast_result(horizon=5, n_entities=2, entity_names=["Region_A", "Region_B"])
        actual = fr.forecasts.copy() + 0.15
        result = fr.evaluate(actual, entity="Region_B")
        assert isinstance(result, pd.DataFrame)
        assert all(result["entity"] == "Region_B")

    def test_evaluate_2d_actual_expands(self):
        """evaluate() with 2D actual (horizon, K) expands to 3D."""
        rng = np.random.default_rng(42)
        fc2d = rng.standard_normal((5, 2))
        fr = ForecastResult(forecasts=fc2d, endog_names=["y1", "y2"])
        actual_2d = fc2d + 0.3
        metrics = fr.evaluate(actual_2d)
        assert isinstance(metrics, pd.Series)
        assert metrics["RMSE"] > 0

    def test_evaluate_dataframe_actual(self):
        """evaluate() with DataFrame actual works correctly."""
        fr = _make_forecast_result(
            horizon=4,
            n_entities=2,
            n_vars=2,
            endog_names=["gdp", "inf"],
            entity_names=["US", "UK"],
        )
        rows = []
        for i in range(2):
            for h in range(4):
                rows.append(
                    {
                        "country": ["US", "UK"][i],
                        "gdp": fr.forecasts[h, i, 0] + 0.1,
                        "inf": fr.forecasts[h, i, 1] + 0.1,
                    }
                )
        actual_df = pd.DataFrame(rows)
        metrics = fr.evaluate(actual_df)
        assert isinstance(metrics, pd.Series)
        assert metrics["RMSE"] > 0
        assert metrics["MAE"] > 0


class TestSummaryAdditionalCoverage:
    """Additional summary() tests for edge cases (lines 503-528)."""

    def test_summary_entity_list_truncated_for_many_entities(self):
        """summary() truncates entity list when N > 10."""
        names = [f"entity_{i}" for i in range(20)]
        fr = _make_forecast_result(n_entities=20, entity_names=names)
        s = fr.summary()
        assert "entity_0" in s
        assert "entity_19" in s
        assert "..." in s

    def test_summary_no_ci_omits_ci_lines(self):
        """summary() omits CI-related lines when no CIs."""
        fr = _make_forecast_result(with_ci=False)
        s = fr.summary()
        assert "Confidence level" not in s
        assert "CI method" not in s

    def test_summary_with_ci_includes_ci_info(self):
        """summary() includes CI info when CIs exist."""
        fr = _make_forecast_result(with_ci=True)
        s = fr.summary()
        assert "Confidence level: 95.0%" in s
        assert "CI method: bootstrap" in s


class TestReprAdditionalCoverage:
    """Additional __repr__ tests."""

    def test_repr_method_name(self):
        """repr includes method name."""
        fr = _make_forecast_result()
        r = repr(fr)
        assert "method='iterative'" in r

    def test_repr_2d_input(self):
        """repr works after 2D input expansion."""
        fc2d = np.zeros((3, 2))
        fr = ForecastResult(forecasts=fc2d, endog_names=["a", "b"])
        r = repr(fr)
        assert "N=1" in r
        assert "K=2" in r
        assert "horizon=3" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
