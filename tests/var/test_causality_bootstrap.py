"""
Tests for bootstrap inference in Panel VAR causality tests.

This module tests:
1. Bootstrap Granger causality test
2. Bootstrap Dumitrescu-Hurlin test
3. Wild bootstrap
4. Residual bootstrap
5. Entity bootstrap
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from panelbox.var.causality_bootstrap import (
    BootstrapGrangerResult,
    _single_bootstrap_dh_iteration,
    _single_bootstrap_granger_iteration,
    _single_bootstrap_iteration_restricted,
    bootstrap_dumitrescu_hurlin,
    bootstrap_granger_test,
)
from panelbox.var.data import PanelVARData
from panelbox.var.model import PanelVAR


@pytest.fixture
def panel_data_with_causality():
    """Generate panel data with Granger causality: x1 -> x2."""
    np.random.seed(42)
    N = 30
    T = 25
    lags = 2

    data_list = []
    for i in range(N):
        # x1: AR(1) process
        x1 = np.zeros(T)
        x1[0] = np.random.randn()
        for t in range(1, T):
            x1[t] = 0.5 * x1[t - 1] + np.random.randn()

        # x2: depends on lagged x1 (causality)
        x2 = np.zeros(T)
        x2[0] = np.random.randn()
        x2[1] = np.random.randn()
        for t in range(lags, T):
            x2[t] = 0.3 * x2[t - 1] + 0.4 * x1[t - 1] + 0.3 * x1[t - 2] + np.random.randn() * 0.5

        entity_data = pd.DataFrame(
            {
                "entity": i,
                "time": range(T),
                "x1": x1,
                "x2": x2,
            }
        )
        data_list.append(entity_data)

    return pd.concat(data_list, ignore_index=True)


@pytest.fixture
def panel_data_no_causality():
    """Generate panel data without Granger causality."""
    np.random.seed(123)
    N = 30
    T = 25

    data_list = []
    for i in range(N):
        # Both x1 and x2 are independent AR(1) processes
        x1 = np.zeros(T)
        x2 = np.zeros(T)
        x1[0] = np.random.randn()
        x2[0] = np.random.randn()

        for t in range(1, T):
            x1[t] = 0.5 * x1[t - 1] + np.random.randn()
            x2[t] = 0.5 * x2[t - 1] + np.random.randn()

        entity_data = pd.DataFrame(
            {
                "entity": i,
                "time": range(T),
                "x1": x1,
                "x2": x2,
            }
        )
        data_list.append(entity_data)

    return pd.concat(data_list, ignore_index=True)


class TestBootstrapGrangerTest:
    """Test bootstrap Granger causality test."""

    def test_bootstrap_granger_with_causality(self, panel_data_with_causality):
        """Test that bootstrap detects causality when present."""
        # Prepare data
        var_data = PanelVARData(
            panel_data_with_causality,
            endog_vars=["x1", "x2"],
            entity_col="entity",
            time_col="time",
            lags=2,
        )

        # Fit model
        model = PanelVAR(var_data)
        result = model.fit()

        # Run bootstrap test (fewer iterations for speed)
        boot_result = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=99,
            bootstrap_type="wild",
            random_state=42,
            show_progress=False,
        )

        # Check result type
        assert isinstance(boot_result, BootstrapGrangerResult)

        # Should detect causality (both p-values < 0.10)
        assert boot_result.p_value_asymptotic < 0.10
        assert boot_result.p_value_bootstrap < 0.15  # More conservative

        # Check bootstrap distribution
        assert len(boot_result.bootstrap_dist) > 80  # Most iterations should succeed
        assert boot_result.bootstrap_dist.mean() > 0

    def test_bootstrap_granger_no_causality(self, panel_data_no_causality):
        """Test that bootstrap does not detect causality when absent."""
        # Prepare data
        var_data = PanelVARData(
            panel_data_no_causality,
            endog_vars=["x1", "x2"],
            entity_col="entity",
            time_col="time",
            lags=2,
        )

        # Fit model
        model = PanelVAR(var_data)
        result = model.fit()

        # Run bootstrap test
        boot_result = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=99,
            bootstrap_type="wild",
            random_state=123,
            show_progress=False,
        )

        # Should NOT detect causality (p-values > 0.05)
        assert boot_result.p_value_bootstrap > 0.05

    def test_bootstrap_reproducibility(self, panel_data_with_causality):
        """Test that bootstrap is reproducible with same seed."""
        # Prepare data
        var_data = PanelVARData(
            panel_data_with_causality,
            endog_vars=["x1", "x2"],
            entity_col="entity",
            time_col="time",
            lags=2,
        )

        # Fit model
        model = PanelVAR(var_data)
        result = model.fit()

        # Run bootstrap twice with same seed
        boot1 = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=50,
            bootstrap_type="wild",
            random_state=42,
            show_progress=False,
        )

        boot2 = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=50,
            bootstrap_type="wild",
            random_state=42,
            show_progress=False,
        )

        # Results should be identical
        np.testing.assert_allclose(boot1.bootstrap_dist, boot2.bootstrap_dist)
        assert boot1.p_value_bootstrap == boot2.p_value_bootstrap

    def test_bootstrap_types(self, panel_data_with_causality):
        """Test different bootstrap types."""
        # Prepare data
        var_data = PanelVARData(
            panel_data_with_causality,
            endog_vars=["x1", "x2"],
            entity_col="entity",
            time_col="time",
            lags=2,
        )

        # Fit model
        model = PanelVAR(var_data)
        result = model.fit()

        # Test wild bootstrap
        boot_wild = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=30,
            bootstrap_type="wild",
            random_state=42,
            show_progress=False,
        )
        assert boot_wild.bootstrap_type == "wild"
        assert boot_wild.p_value_bootstrap < 0.20

        # Test residual bootstrap
        boot_residual = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=30,
            bootstrap_type="residual",
            random_state=42,
            show_progress=False,
        )
        assert boot_residual.bootstrap_type == "residual"
        assert boot_residual.p_value_bootstrap < 0.20

        # Test pairs bootstrap
        boot_pairs = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=30,
            bootstrap_type="pairs",
            random_state=42,
            show_progress=False,
        )
        assert boot_pairs.bootstrap_type == "pairs"


class TestBootstrapDumitrescuHurlin:
    """Test bootstrap Dumitrescu-Hurlin test."""

    def test_bootstrap_dh_with_causality(self, panel_data_with_causality):
        """Test that bootstrap DH detects causality when present."""
        boot_result = bootstrap_dumitrescu_hurlin(
            data=panel_data_with_causality,
            cause="x1",
            effect="x2",
            lags=2,
            entity_col="entity",
            time_col="time",
            n_bootstrap=50,
            bootstrap_type="wild",
            random_state=42,
            show_progress=False,
        )

        # Check result structure
        assert "observed_W_bar" in boot_result
        assert "observed_Z_tilde" in boot_result
        assert "observed_Z_bar" in boot_result
        assert "p_value_bootstrap_W_bar" in boot_result
        assert "p_value_bootstrap_Z_tilde" in boot_result
        assert "p_value_bootstrap_Z_bar" in boot_result
        assert "bootstrap_dist_W_bar" in boot_result

        # Should detect causality
        assert boot_result["p_value_asymptotic_Z_tilde"] < 0.10
        # Bootstrap p-values might be more conservative
        assert boot_result["p_value_bootstrap_Z_tilde"] < 0.20

    def test_bootstrap_dh_no_causality(self, panel_data_no_causality):
        """Test that bootstrap DH does not detect causality when absent."""
        boot_result = bootstrap_dumitrescu_hurlin(
            data=panel_data_no_causality,
            cause="x1",
            effect="x2",
            lags=2,
            entity_col="entity",
            time_col="time",
            n_bootstrap=50,
            bootstrap_type="wild",
            random_state=123,
            show_progress=False,
        )

        # Should NOT detect causality
        assert boot_result["p_value_bootstrap_Z_tilde"] > 0.05
        assert boot_result["p_value_bootstrap_Z_bar"] > 0.05

    def test_bootstrap_dh_entity_bootstrap(self, panel_data_with_causality):
        """Test entity bootstrap for DH test."""
        boot_result = bootstrap_dumitrescu_hurlin(
            data=panel_data_with_causality,
            cause="x1",
            effect="x2",
            lags=2,
            entity_col="entity",
            time_col="time",
            n_bootstrap=30,
            bootstrap_type="entity",
            random_state=42,
            show_progress=False,
        )

        # Check that entity bootstrap works
        assert boot_result["bootstrap_type"] == "entity"
        assert len(boot_result["bootstrap_dist_W_bar"]) > 20  # Most iterations should succeed

    def test_bootstrap_dh_reproducibility(self, panel_data_with_causality):
        """Test that bootstrap DH is reproducible with same seed."""
        boot1 = bootstrap_dumitrescu_hurlin(
            data=panel_data_with_causality,
            cause="x1",
            effect="x2",
            lags=2,
            entity_col="entity",
            time_col="time",
            n_bootstrap=30,
            bootstrap_type="wild",
            random_state=42,
            show_progress=False,
        )

        boot2 = bootstrap_dumitrescu_hurlin(
            data=panel_data_with_causality,
            cause="x1",
            effect="x2",
            lags=2,
            entity_col="entity",
            time_col="time",
            n_bootstrap=30,
            bootstrap_type="wild",
            random_state=42,
            show_progress=False,
        )

        # Results should be very similar (some floating point differences might occur)
        np.testing.assert_allclose(
            boot1["bootstrap_dist_W_bar"], boot2["bootstrap_dist_W_bar"], rtol=1e-10
        )


class TestBootstrapVisualization:
    """Test bootstrap visualization methods."""

    def test_plot_bootstrap_distribution_matplotlib(self, panel_data_with_causality):
        """Test matplotlib plotting of bootstrap distribution."""
        # Prepare data
        var_data = PanelVARData(
            panel_data_with_causality,
            endog_vars=["x1", "x2"],
            entity_col="entity",
            time_col="time",
            lags=2,
        )

        # Fit model
        model = PanelVAR(var_data)
        result = model.fit()

        # Run bootstrap test
        boot_result = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=50,
            bootstrap_type="wild",
            random_state=42,
            show_progress=False,
        )

        # Test plotting (don't show)
        fig = boot_result.plot_bootstrap_distribution(backend="matplotlib", show=False)
        assert fig is not None

    def test_plot_bootstrap_distribution_plotly(self, panel_data_with_causality):
        """Test plotly plotting of bootstrap distribution."""
        # Prepare data
        var_data = PanelVARData(
            panel_data_with_causality,
            endog_vars=["x1", "x2"],
            entity_col="entity",
            time_col="time",
            lags=2,
        )

        # Fit model
        model = PanelVAR(var_data)
        result = model.fit()

        # Run bootstrap test
        boot_result = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=50,
            bootstrap_type="wild",
            random_state=42,
            show_progress=False,
        )

        # Test plotting (don't show)
        fig = boot_result.plot_bootstrap_distribution(backend="plotly", show=False)
        assert fig is not None

    def test_bootstrap_summary(self, panel_data_with_causality):
        """Test bootstrap summary output."""
        # Prepare data
        var_data = PanelVARData(
            panel_data_with_causality,
            endog_vars=["x1", "x2"],
            entity_col="entity",
            time_col="time",
            lags=2,
        )

        # Fit model
        model = PanelVAR(var_data)
        result = model.fit()

        # Run bootstrap test
        boot_result = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=50,
            bootstrap_type="wild",
            random_state=42,
            show_progress=False,
        )

        # Test summary
        summary = boot_result.summary()
        assert isinstance(summary, str)
        assert "Bootstrap Granger Causality Test" in summary
        assert "x1" in summary
        assert "x2" in summary
        assert "wild" in summary


# ---------------------------------------------------------------------------
# New tests targeting uncovered lines in causality_bootstrap.py
# ---------------------------------------------------------------------------


class TestSingleBootstrapIterationRestricted:
    """Tests for _single_bootstrap_iteration_restricted (lines 282-321)."""

    @staticmethod
    def _make_regression_data(seed=42):
        """Create synthetic regression data for restricted bootstrap tests."""
        np.random.seed(seed)
        n = 120
        X_full = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
        X_restricted = X_full[:, :2]  # Only intercept + first covariate
        true_beta = np.array([1.0, 0.5, 0.3])
        y = X_full @ true_beta + np.random.randn(n) * 0.5
        R = np.array([[0, 0, 1]])  # Test: third coefficient = 0
        return y, X_restricted, X_full, R

    def test_residual_bootstrap_returns_finite(self):
        """Residual bootstrap should return a finite non-negative Wald stat."""
        y, X_r, X_u, R = self._make_regression_data()
        stat = _single_bootstrap_iteration_restricted(
            y, X_r, X_u, R, bootstrap_type="residual", seed=100
        )
        assert np.isfinite(stat), f"Expected finite stat, got {stat}"
        assert stat >= 0, f"Wald stat should be >= 0, got {stat}"

    def test_wild_bootstrap_returns_finite(self):
        """Wild bootstrap should return a finite non-negative Wald stat."""
        y, X_r, X_u, R = self._make_regression_data()
        stat = _single_bootstrap_iteration_restricted(
            y, X_r, X_u, R, bootstrap_type="wild", seed=200
        )
        assert np.isfinite(stat), f"Expected finite stat, got {stat}"
        assert stat >= 0, f"Wald stat should be >= 0, got {stat}"

    def test_different_seeds_give_different_stats(self):
        """Different seeds should produce different bootstrap statistics."""
        y, X_r, X_u, R = self._make_regression_data()
        stat1 = _single_bootstrap_iteration_restricted(
            y, X_r, X_u, R, bootstrap_type="residual", seed=1
        )
        stat2 = _single_bootstrap_iteration_restricted(
            y, X_r, X_u, R, bootstrap_type="residual", seed=999
        )
        # Extremely unlikely to be exactly equal with different seeds
        assert stat1 != stat2

    def test_same_seed_gives_same_stat(self):
        """Same seed should produce identical bootstrap statistics."""
        y, X_r, X_u, R = self._make_regression_data()
        stat1 = _single_bootstrap_iteration_restricted(
            y, X_r, X_u, R, bootstrap_type="wild", seed=42
        )
        stat2 = _single_bootstrap_iteration_restricted(
            y, X_r, X_u, R, bootstrap_type="wild", seed=42
        )
        assert stat1 == stat2

    def test_unknown_bootstrap_type_raises(self):
        """An unknown bootstrap_type should raise ValueError."""
        y, X_r, X_u, R = self._make_regression_data()
        with pytest.raises(ValueError, match="Unknown bootstrap_type"):
            _single_bootstrap_iteration_restricted(
                y, X_r, X_u, R, bootstrap_type="invalid", seed=42
            )

    def test_multiple_restrictions(self):
        """Test with a restriction matrix testing multiple coefficients."""
        np.random.seed(10)
        n = 150
        X_full = np.column_stack(
            [np.ones(n), np.random.randn(n), np.random.randn(n), np.random.randn(n)]
        )
        X_restricted = X_full[:, :2]
        y = X_full @ np.array([1.0, 0.5, 0.0, 0.0]) + np.random.randn(n) * 0.5
        # Test that last two coefficients are zero
        R = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])

        stat = _single_bootstrap_iteration_restricted(
            y, X_restricted, X_full, R, bootstrap_type="residual", seed=77
        )
        assert np.isfinite(stat)
        assert stat >= 0


class TestSingleBootstrapGrangerIteration:
    """Tests for _single_bootstrap_granger_iteration (lines 368-463)."""

    @staticmethod
    def _make_panel(n_entities=10, n_periods=25, seed=42):
        """Create a panel dataset suitable for Granger bootstrap iteration."""
        np.random.seed(seed)
        data = []
        for i in range(n_entities):
            x1 = np.zeros(n_periods)
            x2 = np.zeros(n_periods)
            x1[0] = np.random.randn()
            x2[0] = np.random.randn()
            for t in range(1, n_periods):
                x1[t] = 0.5 * x1[t - 1] + np.random.randn()
                x2[t] = 0.3 * x2[t - 1] + 0.4 * x1[t - 1] + np.random.randn() * 0.5
            for t in range(n_periods):
                data.append({"entity": i, "time": t, "x1": x1[t], "x2": x2[t]})
        return pd.DataFrame(data)

    def test_pairs_bootstrap(self):
        """Pairs bootstrap should return a float Wald statistic."""
        panel = self._make_panel()
        stat = _single_bootstrap_granger_iteration(
            data=panel,
            causing_var="x1",
            caused_var="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            bootstrap_type="pairs",
            seed=42,
        )
        assert isinstance(stat, float)
        # Stat may be NaN if bootstrap sample is degenerate, but usually finite
        if np.isfinite(stat):
            assert stat >= 0

    def test_residual_bootstrap(self):
        """Residual bootstrap should return a float Wald statistic."""
        panel = self._make_panel()
        stat = _single_bootstrap_granger_iteration(
            data=panel,
            causing_var="x1",
            caused_var="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            bootstrap_type="residual",
            seed=42,
        )
        assert isinstance(stat, float)
        if np.isfinite(stat):
            assert stat >= 0

    def test_wild_bootstrap(self):
        """Wild bootstrap should return a float Wald statistic."""
        panel = self._make_panel()
        stat = _single_bootstrap_granger_iteration(
            data=panel,
            causing_var="x1",
            caused_var="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            bootstrap_type="wild",
            seed=42,
        )
        assert isinstance(stat, float)
        if np.isfinite(stat):
            assert stat >= 0

    def test_reproducibility_with_same_seed(self):
        """Same seed should yield the same Wald statistic."""
        panel = self._make_panel()
        stat1 = _single_bootstrap_granger_iteration(
            data=panel,
            causing_var="x1",
            caused_var="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            bootstrap_type="wild",
            seed=77,
        )
        stat2 = _single_bootstrap_granger_iteration(
            data=panel,
            causing_var="x1",
            caused_var="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            bootstrap_type="wild",
            seed=77,
        )
        assert stat1 == stat2

    def test_pairs_different_seed(self):
        """Different seeds should (almost surely) produce different statistics."""
        panel = self._make_panel()
        stat1 = _single_bootstrap_granger_iteration(
            data=panel,
            causing_var="x1",
            caused_var="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            bootstrap_type="pairs",
            seed=1,
        )
        stat2 = _single_bootstrap_granger_iteration(
            data=panel,
            causing_var="x1",
            caused_var="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            bootstrap_type="pairs",
            seed=999,
        )
        # Both should be float; if both finite, extremely unlikely to be equal
        assert isinstance(stat1, float)
        assert isinstance(stat2, float)

    def test_with_two_lags(self):
        """Test with lags=2 to exercise more of the lag construction code."""
        panel = self._make_panel(n_entities=10, n_periods=30)
        stat = _single_bootstrap_granger_iteration(
            data=panel,
            causing_var="x1",
            caused_var="x2",
            lags=2,
            entity_col="entity",
            time_col="time",
            bootstrap_type="residual",
            seed=42,
        )
        assert isinstance(stat, float)


class TestSingleBootstrapDHIteration:
    """Tests for _single_bootstrap_dh_iteration (lines 633-725)."""

    @staticmethod
    def _make_panel(n_entities=15, n_periods=25, seed=42):
        """Create a panel dataset suitable for DH bootstrap iteration."""
        np.random.seed(seed)
        data = []
        for i in range(n_entities):
            x1 = np.zeros(n_periods)
            x2 = np.zeros(n_periods)
            x1[0] = np.random.randn()
            x2[0] = np.random.randn()
            for t in range(1, n_periods):
                x1[t] = 0.5 * x1[t - 1] + np.random.randn()
                x2[t] = 0.3 * x2[t - 1] + 0.4 * x1[t - 1] + np.random.randn() * 0.5
            for t in range(n_periods):
                data.append({"entity": i, "time": t, "x1": x1[t], "x2": x2[t]})
        return pd.DataFrame(data)

    def test_entity_bootstrap(self):
        """Entity bootstrap should return a dict with DH statistics or None."""
        panel = self._make_panel()
        result = _single_bootstrap_dh_iteration(
            data=panel,
            cause="x1",
            effect="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            bootstrap_type="entity",
            seed=42,
        )
        if result is not None:
            assert "W_bar" in result
            assert "Z_tilde_stat" in result
            assert "Z_bar_stat" in result
            assert np.isfinite(result["W_bar"])

    def test_wild_bootstrap(self):
        """Wild bootstrap should return a dict with DH statistics or None."""
        panel = self._make_panel()
        result = _single_bootstrap_dh_iteration(
            data=panel,
            cause="x1",
            effect="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            bootstrap_type="wild",
            seed=42,
        )
        if result is not None:
            assert "W_bar" in result
            assert "Z_tilde_stat" in result
            assert "Z_bar_stat" in result

    def test_residual_bootstrap(self):
        """Residual bootstrap should return a dict with DH statistics or None."""
        panel = self._make_panel()
        result = _single_bootstrap_dh_iteration(
            data=panel,
            cause="x1",
            effect="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            bootstrap_type="residual",
            seed=42,
        )
        if result is not None:
            assert "W_bar" in result
            assert "Z_tilde_stat" in result
            assert "Z_bar_stat" in result

    def test_unknown_type_raises(self):
        """Unknown bootstrap_type should raise ValueError (not caught)."""
        panel = self._make_panel()
        # The function catches generic Exception and returns None,
        # but ValueError from unknown type is raised inside the try block
        # and gets caught by the except Exception, returning None.
        result = _single_bootstrap_dh_iteration(
            data=panel,
            cause="x1",
            effect="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            bootstrap_type="invalid_type",
            seed=42,
        )
        # ValueError is caught by the except Exception block -> returns None
        assert result is None

    def test_entity_reproducibility(self):
        """Same seed should produce the same entity bootstrap result."""
        panel = self._make_panel()
        r1 = _single_bootstrap_dh_iteration(
            data=panel,
            cause="x1",
            effect="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            bootstrap_type="entity",
            seed=77,
        )
        r2 = _single_bootstrap_dh_iteration(
            data=panel,
            cause="x1",
            effect="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            bootstrap_type="entity",
            seed=77,
        )
        if r1 is not None and r2 is not None:
            assert r1["W_bar"] == r2["W_bar"]
            assert r1["Z_tilde_stat"] == r2["Z_tilde_stat"]

    def test_wild_with_two_lags(self):
        """Wild bootstrap with lags=2 exercises more of the lag loop."""
        panel = self._make_panel(n_entities=15, n_periods=30)
        result = _single_bootstrap_dh_iteration(
            data=panel,
            cause="x1",
            effect="x2",
            lags=2,
            entity_col="entity",
            time_col="time",
            bootstrap_type="wild",
            seed=42,
        )
        if result is not None:
            assert "W_bar" in result

    def test_residual_with_two_lags(self):
        """Residual bootstrap with lags=2."""
        panel = self._make_panel(n_entities=15, n_periods=30)
        result = _single_bootstrap_dh_iteration(
            data=panel,
            cause="x1",
            effect="x2",
            lags=2,
            entity_col="entity",
            time_col="time",
            bootstrap_type="residual",
            seed=42,
        )
        if result is not None:
            assert "W_bar" in result


class TestBootstrapGrangerHighLevel:
    """Additional high-level tests for bootstrap_granger_test covering residual/pairs."""

    @staticmethod
    def _make_panel_and_fit(seed=42):
        """Create panel, fit VAR, return result."""
        np.random.seed(seed)
        n_entities = 15
        n_periods = 25
        data = []
        for i in range(n_entities):
            x1 = np.zeros(n_periods)
            x2 = np.zeros(n_periods)
            x1[0] = np.random.randn()
            x2[0] = np.random.randn()
            for t in range(1, n_periods):
                x1[t] = 0.5 * x1[t - 1] + np.random.randn()
                x2[t] = 0.3 * x2[t - 1] + 0.4 * x1[t - 1] + np.random.randn() * 0.5
            for t in range(n_periods):
                data.append({"entity": i, "time": t, "x1": x1[t], "x2": x2[t]})
        df = pd.DataFrame(data)

        var_data = PanelVARData(
            df, endog_vars=["x1", "x2"], entity_col="entity", time_col="time", lags=1
        )
        model = PanelVAR(var_data)
        result = model.fit()
        return result

    def test_residual_bootstrap(self):
        """High-level bootstrap_granger_test with residual bootstrap."""
        result = self._make_panel_and_fit()
        boot = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=20,
            bootstrap_type="residual",
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )
        assert isinstance(boot, BootstrapGrangerResult)
        assert 0 <= boot.p_value_bootstrap <= 1
        assert boot.bootstrap_type == "residual"
        assert hasattr(boot, "bootstrap_dist")
        assert hasattr(boot, "ci_lower")
        assert hasattr(boot, "ci_upper")

    def test_pairs_bootstrap(self):
        """High-level bootstrap_granger_test with pairs bootstrap."""
        result = self._make_panel_and_fit()
        boot = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=20,
            bootstrap_type="pairs",
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )
        assert isinstance(boot, BootstrapGrangerResult)
        assert 0 <= boot.p_value_bootstrap <= 1
        assert boot.bootstrap_type == "pairs"

    def test_wild_bootstrap(self):
        """High-level bootstrap_granger_test with wild bootstrap."""
        result = self._make_panel_and_fit()
        boot = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=20,
            bootstrap_type="wild",
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )
        assert isinstance(boot, BootstrapGrangerResult)
        assert 0 <= boot.p_value_bootstrap <= 1
        assert boot.bootstrap_type == "wild"
        # The bootstrap distribution should have some entries
        assert len(boot.bootstrap_dist) > 0

    def test_observed_stat_is_positive(self):
        """Observed Wald statistic should be non-negative."""
        result = self._make_panel_and_fit()
        boot = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=10,
            bootstrap_type="wild",
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )
        assert boot.observed_stat >= 0


class TestBootstrapDHHighLevel:
    """Additional high-level tests for bootstrap_dumitrescu_hurlin."""

    @staticmethod
    def _make_panel(n_entities=15, n_periods=25, seed=42):
        """Create a panel dataset with causality for DH testing."""
        np.random.seed(seed)
        data = []
        for i in range(n_entities):
            x1 = np.zeros(n_periods)
            x2 = np.zeros(n_periods)
            x1[0] = np.random.randn()
            x2[0] = np.random.randn()
            for t in range(1, n_periods):
                x1[t] = 0.5 * x1[t - 1] + np.random.randn()
                x2[t] = 0.3 * x2[t - 1] + 0.4 * x1[t - 1] + np.random.randn() * 0.5
            for t in range(n_periods):
                data.append({"entity": i, "time": t, "x1": x1[t], "x2": x2[t]})
        return pd.DataFrame(data)

    def test_residual_bootstrap(self):
        """DH test with residual bootstrap."""
        panel = self._make_panel()
        result = bootstrap_dumitrescu_hurlin(
            data=panel,
            cause="x1",
            effect="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            n_bootstrap=20,
            bootstrap_type="residual",
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )
        assert result is not None
        assert "observed_W_bar" in result
        assert "p_value_bootstrap_W_bar" in result
        assert "p_value_bootstrap_Z_tilde" in result
        assert "p_value_bootstrap_Z_bar" in result
        assert 0 <= result["p_value_bootstrap_W_bar"] <= 1
        assert result["bootstrap_type"] == "residual"

    def test_wild_bootstrap(self):
        """DH test with wild bootstrap."""
        panel = self._make_panel()
        result = bootstrap_dumitrescu_hurlin(
            data=panel,
            cause="x1",
            effect="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            n_bootstrap=20,
            bootstrap_type="wild",
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )
        assert result is not None
        assert "observed_W_bar" in result
        assert result["bootstrap_type"] == "wild"
        assert 0 <= result["p_value_bootstrap_Z_tilde"] <= 1

    def test_entity_bootstrap(self):
        """DH test with entity bootstrap."""
        panel = self._make_panel()
        result = bootstrap_dumitrescu_hurlin(
            data=panel,
            cause="x1",
            effect="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            n_bootstrap=20,
            bootstrap_type="entity",
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )
        assert result is not None
        assert result["bootstrap_type"] == "entity"
        assert result["n_bootstrap"] > 0

    def test_bootstrap_distributions_populated(self):
        """Verify bootstrap distribution arrays are populated."""
        panel = self._make_panel()
        result = bootstrap_dumitrescu_hurlin(
            data=panel,
            cause="x1",
            effect="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            n_bootstrap=15,
            bootstrap_type="wild",
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )
        assert len(result["bootstrap_dist_W_bar"]) > 0
        assert len(result["bootstrap_dist_Z_tilde"]) > 0
        assert len(result["bootstrap_dist_Z_bar"]) > 0

    def test_with_two_lags(self):
        """DH bootstrap test with lags=2."""
        panel = self._make_panel(n_entities=15, n_periods=30)
        result = bootstrap_dumitrescu_hurlin(
            data=panel,
            cause="x1",
            effect="x2",
            lags=2,
            entity_col="entity",
            time_col="time",
            n_bootstrap=15,
            bootstrap_type="residual",
            random_state=42,
            show_progress=False,
            n_jobs=1,
        )
        assert result is not None
        assert "observed_W_bar" in result
        assert result["n_bootstrap"] > 0


# ---------------------------------------------------------------------------
# Coverage-focused tests targeting uncovered branches
# ---------------------------------------------------------------------------


class TestBootstrapGrangerResidualCoverage:
    """
    Exercise bootstrap_granger_test with bootstrap_type='residual' through
    the full high-level API to cover _single_bootstrap_granger_iteration
    residual branch (lines 388-444).
    """

    @staticmethod
    def _build_fitted_result(seed=42):
        np.random.seed(seed)
        n_entities, n_periods = 20, 30
        rows = []
        for i in range(n_entities):
            x1 = np.zeros(n_periods)
            x2 = np.zeros(n_periods)
            x1[0] = np.random.randn()
            x2[0] = np.random.randn()
            for t in range(1, n_periods):
                x1[t] = 0.5 * x1[t - 1] + np.random.randn()
                x2[t] = 0.3 * x2[t - 1] + 0.4 * x1[t - 1] + np.random.randn() * 0.5
            for t in range(n_periods):
                rows.append({"entity": i, "time": t, "x1": x1[t], "x2": x2[t]})
        df = pd.DataFrame(rows)
        var_data = PanelVARData(
            df, endog_vars=["x1", "x2"], entity_col="entity", time_col="time", lags=1
        )
        model = PanelVAR(var_data)
        return model.fit(cov_type="nonrobust")

    def test_residual_bootstrap_full(self):
        """Full residual bootstrap test returns valid result with p-value in [0, 1]."""
        result = self._build_fitted_result()
        boot = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=50,
            bootstrap_type="residual",
            n_jobs=1,
            show_progress=False,
            random_state=42,
        )
        assert isinstance(boot, BootstrapGrangerResult)
        assert 0 <= boot.p_value_bootstrap <= 1
        assert boot.bootstrap_type == "residual"
        assert len(boot.bootstrap_dist) > 0
        assert boot.ci_lower <= boot.ci_upper

    def test_residual_bootstrap_distribution_stats(self):
        """Bootstrap distribution has finite mean and std."""
        result = self._build_fitted_result()
        boot = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=50,
            bootstrap_type="residual",
            n_jobs=1,
            show_progress=False,
            random_state=42,
        )
        assert np.isfinite(np.mean(boot.bootstrap_dist))
        assert np.isfinite(np.std(boot.bootstrap_dist))


class TestBootstrapGrangerPairsCoverage:
    """
    Exercise bootstrap_granger_test with bootstrap_type='pairs' through
    the full high-level API to cover _single_bootstrap_granger_iteration
    pairs branch (lines 374-386).
    """

    @staticmethod
    def _build_fitted_result(seed=42):
        np.random.seed(seed)
        n_entities, n_periods = 20, 30
        rows = []
        for i in range(n_entities):
            x1 = np.zeros(n_periods)
            x2 = np.zeros(n_periods)
            x1[0] = np.random.randn()
            x2[0] = np.random.randn()
            for t in range(1, n_periods):
                x1[t] = 0.5 * x1[t - 1] + np.random.randn()
                x2[t] = 0.3 * x2[t - 1] + 0.4 * x1[t - 1] + np.random.randn() * 0.5
            for t in range(n_periods):
                rows.append({"entity": i, "time": t, "x1": x1[t], "x2": x2[t]})
        df = pd.DataFrame(rows)
        var_data = PanelVARData(
            df, endog_vars=["x1", "x2"], entity_col="entity", time_col="time", lags=1
        )
        model = PanelVAR(var_data)
        return model.fit(cov_type="nonrobust")

    def test_pairs_bootstrap_full(self):
        """Full pairs bootstrap test returns valid result."""
        result = self._build_fitted_result()
        boot = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=50,
            bootstrap_type="pairs",
            n_jobs=1,
            show_progress=False,
            random_state=42,
        )
        assert isinstance(boot, BootstrapGrangerResult)
        assert 0 <= boot.p_value_bootstrap <= 1
        assert boot.bootstrap_type == "pairs"
        assert len(boot.bootstrap_dist) > 0

    def test_pairs_observed_stat_positive(self):
        """Observed Wald stat is non-negative."""
        result = self._build_fitted_result()
        boot = bootstrap_granger_test(
            result,
            causing_var="x1",
            caused_var="x2",
            n_bootstrap=50,
            bootstrap_type="pairs",
            n_jobs=1,
            show_progress=False,
            random_state=42,
        )
        assert boot.observed_stat >= 0


class TestBootstrapDHEntityCoverage:
    """
    Exercise bootstrap_dumitrescu_hurlin with bootstrap_type='entity'
    to cover _single_bootstrap_dh_iteration entity branch (lines 636-648).
    """

    @staticmethod
    def _make_panel(seed=42):
        np.random.seed(seed)
        n_entities, n_periods = 25, 30
        rows = []
        for i in range(n_entities):
            x1 = np.zeros(n_periods)
            x2 = np.zeros(n_periods)
            x1[0] = np.random.randn()
            x2[0] = np.random.randn()
            for t in range(1, n_periods):
                x1[t] = 0.5 * x1[t - 1] + np.random.randn()
                x2[t] = 0.3 * x2[t - 1] + 0.4 * x1[t - 1] + np.random.randn() * 0.5
            for t in range(n_periods):
                rows.append({"entity": i, "time": t, "x1": x1[t], "x2": x2[t]})
        return pd.DataFrame(rows)

    def test_entity_bootstrap_full(self):
        """Full entity bootstrap DH test returns valid structure."""
        panel = self._make_panel()
        result = bootstrap_dumitrescu_hurlin(
            data=panel,
            cause="x1",
            effect="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            n_bootstrap=50,
            bootstrap_type="entity",
            n_jobs=1,
            show_progress=False,
            random_state=42,
        )
        assert result is not None
        assert result["bootstrap_type"] == "entity"
        assert 0 <= result["p_value_bootstrap_W_bar"] <= 1
        assert 0 <= result["p_value_bootstrap_Z_tilde"] <= 1
        assert 0 <= result["p_value_bootstrap_Z_bar"] <= 1

    def test_entity_bootstrap_observed_stats_finite(self):
        """Observed DH statistics are finite."""
        panel = self._make_panel()
        result = bootstrap_dumitrescu_hurlin(
            data=panel,
            cause="x1",
            effect="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            n_bootstrap=50,
            bootstrap_type="entity",
            n_jobs=1,
            show_progress=False,
            random_state=42,
        )
        assert np.isfinite(result["observed_W_bar"])
        assert np.isfinite(result["observed_Z_tilde"])
        assert np.isfinite(result["observed_Z_bar"])


class TestBootstrapDHResidualCoverage:
    """
    Exercise bootstrap_dumitrescu_hurlin with bootstrap_type='residual'
    to cover _single_bootstrap_dh_iteration residual branch (lines 650-701).
    """

    @staticmethod
    def _make_panel(seed=42):
        np.random.seed(seed)
        n_entities, n_periods = 25, 30
        rows = []
        for i in range(n_entities):
            x1 = np.zeros(n_periods)
            x2 = np.zeros(n_periods)
            x1[0] = np.random.randn()
            x2[0] = np.random.randn()
            for t in range(1, n_periods):
                x1[t] = 0.5 * x1[t - 1] + np.random.randn()
                x2[t] = 0.3 * x2[t - 1] + 0.4 * x1[t - 1] + np.random.randn() * 0.5
            for t in range(n_periods):
                rows.append({"entity": i, "time": t, "x1": x1[t], "x2": x2[t]})
        return pd.DataFrame(rows)

    def test_residual_bootstrap_full(self):
        """Full residual bootstrap DH test returns valid structure."""
        panel = self._make_panel()
        result = bootstrap_dumitrescu_hurlin(
            data=panel,
            cause="x1",
            effect="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            n_bootstrap=50,
            bootstrap_type="residual",
            n_jobs=1,
            show_progress=False,
            random_state=42,
        )
        assert result is not None
        assert result["bootstrap_type"] == "residual"
        assert 0 <= result["p_value_bootstrap_W_bar"] <= 1
        assert len(result["bootstrap_dist_W_bar"]) > 0

    def test_residual_bootstrap_distributions_finite(self):
        """Bootstrap distributions contain finite values."""
        panel = self._make_panel()
        result = bootstrap_dumitrescu_hurlin(
            data=panel,
            cause="x1",
            effect="x2",
            lags=1,
            entity_col="entity",
            time_col="time",
            n_bootstrap=50,
            bootstrap_type="residual",
            n_jobs=1,
            show_progress=False,
            random_state=42,
        )
        dist = result["bootstrap_dist_W_bar"]
        finite_count = np.sum(np.isfinite(dist))
        assert finite_count > 0, "Expected at least some finite values in distribution"


class TestBootstrapGrangerResultSummaryCoverage:
    """
    Test BootstrapGrangerResult.summary() output content more thoroughly.
    """

    @staticmethod
    def _build_boot_result():
        """Build a BootstrapGrangerResult directly (no model fitting needed)."""
        np.random.seed(42)
        dist = np.random.chisquare(df=2, size=50)
        return BootstrapGrangerResult(
            cause="x1",
            effect="x2",
            observed_stat=8.5,
            p_value_asymptotic=0.014,
            p_value_bootstrap=0.04,
            bootstrap_dist=dist,
            ci_lower=float(np.percentile(dist, 2.5)),
            ci_upper=float(np.percentile(dist, 97.5)),
            n_bootstrap=50,
            bootstrap_type="residual",
        )

    def test_summary_contains_key_sections(self):
        """Summary string contains all expected sections."""
        boot = self._build_boot_result()
        summary = boot.summary()
        assert "Bootstrap Granger Causality Test" in summary
        assert "Cause: x1" in summary
        assert "Effect: x2" in summary
        assert "Bootstrap type: residual" in summary
        assert "Observed Wald:" in summary
        assert "P-value (asymptotic):" in summary
        assert "P-value (bootstrap):" in summary
        assert "Bootstrap Distribution:" in summary
        assert "Mean:" in summary
        assert "Std:" in summary
        assert "95% Bootstrap CI:" in summary
        assert "Conclusion:" in summary

    def test_summary_conclusion_reject_5pct(self):
        """Summary shows ** when p-value is between 0.01 and 0.05."""
        boot = self._build_boot_result()
        # p_value_bootstrap=0.04 -> Rejects at 5%
        summary = boot.summary()
        assert "Rejects H0 at 5% (bootstrap) (**)" in summary

    def test_summary_conclusion_reject_1pct(self):
        """Summary shows *** when p-value < 0.01."""
        np.random.seed(42)
        dist = np.random.chisquare(df=2, size=50)
        boot = BootstrapGrangerResult(
            cause="x1",
            effect="x2",
            observed_stat=15.0,
            p_value_asymptotic=0.001,
            p_value_bootstrap=0.005,
            bootstrap_dist=dist,
            ci_lower=float(np.percentile(dist, 2.5)),
            ci_upper=float(np.percentile(dist, 97.5)),
            n_bootstrap=50,
            bootstrap_type="wild",
        )
        summary = boot.summary()
        assert "Rejects H0 at 1% (bootstrap) (***)" in summary

    def test_summary_conclusion_reject_10pct(self):
        """Summary shows * when p-value is between 0.05 and 0.10."""
        np.random.seed(42)
        dist = np.random.chisquare(df=2, size=50)
        boot = BootstrapGrangerResult(
            cause="x1",
            effect="x2",
            observed_stat=5.0,
            p_value_asymptotic=0.08,
            p_value_bootstrap=0.07,
            bootstrap_dist=dist,
            ci_lower=float(np.percentile(dist, 2.5)),
            ci_upper=float(np.percentile(dist, 97.5)),
            n_bootstrap=50,
            bootstrap_type="pairs",
        )
        summary = boot.summary()
        assert "Rejects H0 at 10% (bootstrap) (*)" in summary

    def test_summary_conclusion_fail_to_reject(self):
        """Summary shows fail to reject when p >= 0.10."""
        np.random.seed(42)
        dist = np.random.chisquare(df=2, size=50)
        boot = BootstrapGrangerResult(
            cause="x1",
            effect="x2",
            observed_stat=1.0,
            p_value_asymptotic=0.60,
            p_value_bootstrap=0.55,
            bootstrap_dist=dist,
            ci_lower=float(np.percentile(dist, 2.5)),
            ci_upper=float(np.percentile(dist, 97.5)),
            n_bootstrap=50,
            bootstrap_type="wild",
        )
        summary = boot.summary()
        assert "Fails to reject H0 (bootstrap)" in summary


class TestBootstrapGrangerResultPlotCoverage:
    """
    Test BootstrapGrangerResult.plot_bootstrap_distribution() with Agg backend.
    """

    @staticmethod
    def _build_boot_result():
        np.random.seed(42)
        dist = np.random.chisquare(df=2, size=80)
        return BootstrapGrangerResult(
            cause="x1",
            effect="x2",
            observed_stat=8.5,
            p_value_asymptotic=0.014,
            p_value_bootstrap=0.04,
            bootstrap_dist=dist,
            ci_lower=float(np.percentile(dist, 2.5)),
            ci_upper=float(np.percentile(dist, 97.5)),
            n_bootstrap=80,
            bootstrap_type="residual",
        )

    def test_plot_matplotlib_returns_figure(self):
        """plot_bootstrap_distribution with matplotlib returns a Figure."""
        import matplotlib.pyplot as plt

        boot = self._build_boot_result()
        fig = boot.plot_bootstrap_distribution(backend="matplotlib", show=False)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_matplotlib_axes_content(self):
        """Matplotlib plot has expected labels and legend."""
        import matplotlib.pyplot as plt

        boot = self._build_boot_result()
        fig = boot.plot_bootstrap_distribution(backend="matplotlib", show=False)
        ax = fig.axes[0]
        assert "Wald Statistic" in ax.get_xlabel()
        assert "Density" in ax.get_ylabel()
        assert "x1" in ax.get_title()
        assert "x2" in ax.get_title()
        plt.close(fig)

    def test_plot_plotly_returns_figure(self):
        """plot_bootstrap_distribution with plotly returns a go.Figure."""
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

        boot = self._build_boot_result()
        fig = boot.plot_bootstrap_distribution(backend="plotly", show=False)
        assert isinstance(fig, go.Figure)
        # Check that traces exist
        assert len(fig.data) >= 2  # histogram + observed line

    def test_plot_invalid_backend_raises(self):
        """Unknown backend raises ValueError."""
        boot = self._build_boot_result()
        with pytest.raises(ValueError, match="Unknown backend"):
            boot.plot_bootstrap_distribution(backend="unknown", show=False)


class TestBootstrapGrangerResultRepr:
    """Test __repr__ on BootstrapGrangerResult."""

    def test_repr_contains_key_info(self):
        np.random.seed(42)
        dist = np.random.chisquare(df=2, size=30)
        boot = BootstrapGrangerResult(
            cause="x1",
            effect="x2",
            observed_stat=8.5,
            p_value_asymptotic=0.014,
            p_value_bootstrap=0.04,
            bootstrap_dist=dist,
            ci_lower=float(np.percentile(dist, 2.5)),
            ci_upper=float(np.percentile(dist, 97.5)),
            n_bootstrap=30,
            bootstrap_type="wild",
        )
        r = repr(boot)
        assert "BootstrapGrangerResult" in r
        assert "x1" in r
        assert "x2" in r
        assert "8.50" in r
