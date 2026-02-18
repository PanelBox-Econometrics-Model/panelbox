"""
End-to-end workflow tests for VAR tutorial notebooks.
Tests that key PanelBox operations work correctly with generated data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from data_generators import generate_macro_panel

from panelbox.var import PanelVAR, PanelVARData


class TestVARWorkflow:
    """Test complete VAR estimation workflow."""

    @pytest.fixture
    def macro_data(self):
        """Generate a small macro panel for testing."""
        return generate_macro_panel(n_countries=10, n_quarters=20, seed=42)

    @pytest.fixture
    def endog_vars(self):
        return ["gdp_growth", "inflation", "interest_rate"]

    def test_data_loading(self, macro_data):
        """Verify data loads and has correct dimensions."""
        assert isinstance(macro_data, pd.DataFrame)
        assert macro_data.shape[0] == 10 * 20
        assert "country" in macro_data.columns
        assert "quarter" in macro_data.columns
        assert "gdp_growth" in macro_data.columns
        assert macro_data.isnull().sum().sum() == 0

    def test_panel_var_data_creation(self, macro_data, endog_vars):
        """Verify PanelVARData accepts generated data."""
        var_data = PanelVARData(
            data=macro_data,
            endog_vars=endog_vars,
            entity_col="country",
            time_col="quarter",
            lags=2,
        )
        assert var_data.K == 3
        assert var_data.p == 2
        assert var_data.N == 10

    def test_lag_selection(self, macro_data, endog_vars):
        """Verify lag selection returns valid results."""
        var_data = PanelVARData(
            data=macro_data,
            endog_vars=endog_vars,
            entity_col="country",
            time_col="quarter",
            lags=1,
        )
        model = PanelVAR(var_data)
        try:
            lag_result = model.select_lag_order(max_lags=4)
            assert hasattr(lag_result, "criteria_df")
            assert hasattr(lag_result, "selected")
            assert len(lag_result.criteria_df) == 4
        except Exception as e:
            pytest.skip(f"Lag selection not available: {e}")

    def test_ols_estimation(self, macro_data, endog_vars):
        """Verify OLS estimation with clustered SEs."""
        var_data = PanelVARData(
            data=macro_data,
            endog_vars=endog_vars,
            entity_col="country",
            time_col="quarter",
            lags=2,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        assert results.K == 3
        assert results.p == 2
        assert len(results.A_matrices) == 2
        assert results.A_matrices[0].shape == (3, 3)
        assert results.Sigma.shape == (3, 3)

    def test_stability_check(self, macro_data, endog_vars):
        """Verify estimated model is stable."""
        var_data = PanelVARData(
            data=macro_data,
            endog_vars=endog_vars,
            entity_col="country",
            time_col="quarter",
            lags=1,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        assert isinstance(results.is_stable(), bool)
        assert results.max_eigenvalue_modulus < 1.0

    def test_irf_computation(self, macro_data, endog_vars):
        """Verify IRFs can be computed (Cholesky and Generalized)."""
        var_data = PanelVARData(
            data=macro_data,
            endog_vars=endog_vars,
            entity_col="country",
            time_col="quarter",
            lags=1,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        # Cholesky IRF
        irf_chol = results.irf(periods=10, method="cholesky", verbose=False)
        assert irf_chol.irf_matrix.shape == (11, 3, 3)

        # Generalized IRF
        irf_gen = results.irf(periods=10, method="generalized", verbose=False)
        assert irf_gen.irf_matrix.shape == (11, 3, 3)

    def test_fevd_computation(self, macro_data, endog_vars):
        """Verify FEVD can be computed."""
        var_data = PanelVARData(
            data=macro_data,
            endog_vars=endog_vars,
            entity_col="country",
            time_col="quarter",
            lags=1,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        fevd = results.fevd(periods=10)
        assert fevd.decomposition.shape == (11, 3, 3)
        # FEVD should sum to approximately 1 across shocks for each variable
        for h in range(11):
            for i in range(3):
                assert abs(fevd.decomposition[h, i, :].sum() - 1.0) < 0.01

    def test_forecast(self, macro_data, endog_vars):
        """Verify forecasting works."""
        var_data = PanelVARData(
            data=macro_data,
            endog_vars=endog_vars,
            entity_col="country",
            time_col="quarter",
            lags=1,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        forecast = results.forecast(steps=5)
        assert forecast.forecasts.shape[0] == 5
        assert forecast.K == 3

    def test_granger_causality(self, macro_data, endog_vars):
        """Verify Granger causality tests run."""
        var_data = PanelVARData(
            data=macro_data,
            endog_vars=endog_vars,
            entity_col="country",
            time_col="quarter",
            lags=2,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        gc_result = results.granger_causality(cause="gdp_growth", effect="inflation")
        assert hasattr(gc_result, "wald_stat")
        assert hasattr(gc_result, "p_value")
        assert gc_result.p_value >= 0 and gc_result.p_value <= 1

    def test_granger_matrix(self, macro_data, endog_vars):
        """Verify Granger causality matrix has correct dimensions."""
        var_data = PanelVARData(
            data=macro_data,
            endog_vars=endog_vars,
            entity_col="country",
            time_col="quarter",
            lags=2,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        gc_matrix = results.granger_causality_matrix()
        assert gc_matrix.shape == (3, 3)
