# tests/spatial/test_sar_re_validation.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelbox.models.spatial import SpatialLag

FIXTURES_PATH = Path(__file__).parent / "fixtures"


@pytest.fixture
def spatial_test_data():
    """Load spatial test data."""
    df = pd.read_csv(FIXTURES_PATH / "spatial_test_data.csv")
    W = np.loadtxt(FIXTURES_PATH / "spatial_weights.csv", delimiter=",")
    return df, W


@pytest.fixture
def r_sar_results():
    """Load R SAR results."""
    with open(FIXTURES_PATH / "r_sar_re_results.json", "r") as f:
        return json.load(f)


class TestSARRandomEffects:
    """Validate SAR Random Effects against R splm."""

    def test_sar_re_rho(self, spatial_test_data, r_sar_results):
        """Test SAR RE spatial lag coefficient."""
        df, W = spatial_test_data

        model = SpatialLag(
            formula="y ~ x1 + x2 + x3", data=df, entity_col="entity", time_col="time", W=W
        )

        result = model.fit(effects="random", method="ml")

        r_rho = r_sar_results["sar_re"]["rho"]

        # Use rho attribute
        py_rho = result.rho if hasattr(result, "rho") else result.params["rho"]

        assert np.isclose(py_rho, r_rho, rtol=0.20), f"SAR RE rho: {py_rho} vs R: {r_rho}"

    def test_sar_re_beta(self, spatial_test_data, r_sar_results):
        """Test SAR RE coefficients."""
        df, W = spatial_test_data

        model = SpatialLag(
            formula="y ~ x1 + x2 + x3", data=df, entity_col="entity", time_col="time", W=W
        )

        result = model.fit(effects="random", method="ml")

        r_betas = r_sar_results["sar_re"]["beta"]

        for var_name, r_beta in r_betas.items():
            py_beta = result.params[var_name]

            assert np.isclose(
                py_beta, r_beta, rtol=0.20
            ), f"SAR RE {var_name}: {py_beta} vs R: {r_beta}"

    def test_sar_re_variance_components(self, spatial_test_data, r_sar_results):
        """Test SAR RE variance components."""
        df, W = spatial_test_data

        model = SpatialLag(
            formula="y ~ x1 + x2 + x3", data=df, entity_col="entity", time_col="time", W=W
        )

        result = model.fit(effects="random", method="ml")

        r_sigma_alpha2 = r_sar_results["sar_re"]["sigma_alpha2"]
        r_sigma_eps2 = r_sar_results["sar_re"]["sigma_epsilon2"]

        py_sigma_alpha2 = result.variance_components["sigma_alpha2"]
        py_sigma_eps2 = result.variance_components["sigma_epsilon2"]

        # Variance components may have more variation
        assert np.isclose(
            py_sigma_alpha2, r_sigma_alpha2, rtol=0.30
        ), f"SAR RE sigma_alpha^2: {py_sigma_alpha2} vs R: {r_sigma_alpha2}"

        assert np.isclose(
            py_sigma_eps2, r_sigma_eps2, rtol=0.30
        ), f"SAR RE sigma_epsilon^2: {py_sigma_eps2} vs R: {r_sigma_eps2}"

    def test_sar_re_log_likelihood(self, spatial_test_data, r_sar_results):
        """Test SAR RE log-likelihood."""
        df, W = spatial_test_data

        model = SpatialLag(
            formula="y ~ x1 + x2 + x3", data=df, entity_col="entity", time_col="time", W=W
        )

        result = model.fit(effects="random", method="ml")

        r_loglik = r_sar_results["sar_re"]["logLik"]

        # Log-likelihood should be reasonably close
        assert np.isclose(
            result.llf, r_loglik, rtol=0.10
        ), f"SAR RE log-likelihood: {result.llf} vs R: {r_loglik}"

    def test_sar_re_convergence(self, spatial_test_data):
        """Test that SAR RE estimation converges."""
        df, W = spatial_test_data

        model = SpatialLag(
            formula="y ~ x1 + x2 + x3", data=df, entity_col="entity", time_col="time", W=W
        )

        result = model.fit(effects="random", method="ml", maxiter=200)

        assert result.convergence_info[
            "success"
        ], f"SAR RE did not converge: {result.convergence_info['message']}"


class TestSARREvsRFE:
    """Compare SAR RE vs FE."""

    def test_re_vs_fe_rho(self, spatial_test_data, r_sar_results):
        """Compare rho estimates between RE and FE."""
        df, W = spatial_test_data

        model = SpatialLag(
            formula="y ~ x1 + x2 + x3", data=df, entity_col="entity", time_col="time", W=W
        )

        result_re = model.fit(effects="random", method="ml")
        result_fe = model.fit(effects="fixed", method="qml")

        # Get rho values
        rho_re = result_re.rho if hasattr(result_re, "rho") else result_re.params["rho"]
        rho_fe = result_fe.params["rho"]

        # Also compare with R results
        r_rho_re = r_sar_results["sar_re"]["rho"]
        r_rho_fe = r_sar_results["sar_fe"]["rho"]

        # RE and FE should give reasonably similar rho
        # (exact relationship depends on data)
        print(f"\nPython RE rho: {rho_re}, FE rho: {rho_fe}")
        print(f"R RE rho: {r_rho_re}, FE rho: {r_rho_fe}")

        # Test that our RE is close to R RE
        assert np.isclose(
            rho_re, r_rho_re, rtol=0.20
        ), f"Python RE rho ({rho_re}) differs from R RE rho ({r_rho_re})"

        # Test that our FE is close to R FE
        assert np.isclose(
            rho_fe, r_rho_fe, rtol=0.20
        ), f"Python FE rho ({rho_fe}) differs from R FE rho ({r_rho_fe})"
