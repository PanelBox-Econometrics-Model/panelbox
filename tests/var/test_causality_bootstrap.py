"""
Tests for bootstrap inference in Panel VAR causality tests.

This module tests:
1. Bootstrap Granger causality test
2. Bootstrap Dumitrescu-Hurlin test
3. Wild bootstrap
4. Residual bootstrap
5. Entity bootstrap
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.var.causality_bootstrap import (
    BootstrapGrangerResult,
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
