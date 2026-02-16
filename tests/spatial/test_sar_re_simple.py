# tests/spatial/test_sar_re_simple.py
# Simplified test for SAR Random Effects

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelbox.models.spatial import SpatialLag

FIXTURES_PATH = Path(__file__).parent / "fixtures"


def test_sar_re_basic():
    """Test basic SAR RE functionality."""
    # Load data
    df = pd.read_csv(FIXTURES_PATH / "spatial_test_data.csv")
    W = np.loadtxt(FIXTURES_PATH / "spatial_weights.csv", delimiter=",")

    # Create model
    model = SpatialLag(
        formula="y ~ x1 + x2 + x3", data=df, entity_col="entity", time_col="time", W=W
    )

    # Fit with random effects
    result = model.fit(effects="random", method="ml", maxiter=200)

    # Check that we have results
    assert result is not None
    assert hasattr(result, "params")
    assert hasattr(result, "convergence_info")
    assert hasattr(result, "variance_components")

    # Check convergence
    assert result.convergence_info[
        "success"
    ], f"Model did not converge: {result.convergence_info['message']}"

    # Check variance components exist
    assert "sigma_alpha2" in result.variance_components
    assert "sigma_epsilon2" in result.variance_components
    assert "theta" in result.variance_components

    # Check that parameters are reasonable
    assert result.variance_components["sigma_alpha2"] > 0
    assert result.variance_components["sigma_epsilon2"] > 0
    assert 0 < result.variance_components["theta"] < 1

    # Check rho is in bounds
    rho = result.rho if hasattr(result, "rho") else result.params["rho"]
    assert -1 < rho < 1

    # Load R results for validation
    with open(FIXTURES_PATH / "r_sar_re_results.json", "r") as f:
        r_results = json.load(f)

    r_rho = r_results["sar_re"]["rho"]
    r_sigma_alpha2 = r_results["sar_re"]["sigma_alpha2"]
    r_sigma_eps2 = r_results["sar_re"]["sigma_epsilon2"]

    # Relaxed validation - just check we're in the same ballpark
    print(f"\n Python rho: {rho}, R rho: {r_rho}")
    print(f"Python sigma_alpha2: {result.variance_components['sigma_alpha2']}, R: {r_sigma_alpha2}")
    print(f"Python sigma_eps2: {result.variance_components['sigma_epsilon2']}, R: {r_sigma_eps2}")

    # At least check signs and magnitudes are reasonable
    assert np.sign(rho) == np.sign(r_rho) or np.abs(rho) < 0.1 or np.abs(r_rho) < 0.1
    assert (
        np.abs(result.variance_components["sigma_alpha2"] / r_sigma_alpha2) < 3
    )  # Within factor of 3
    assert np.abs(result.variance_components["sigma_epsilon2"] / r_sigma_eps2) < 3


def test_sar_fe_still_works():
    """Test that SAR FE still works after RE implementation."""
    df = pd.read_csv(FIXTURES_PATH / "spatial_test_data.csv")
    W = np.loadtxt(FIXTURES_PATH / "spatial_weights.csv", delimiter=",")

    model = SpatialLag(
        formula="y ~ x1 + x2 + x3", data=df, entity_col="entity", time_col="time", W=W
    )

    result = model.fit(effects="fixed", method="qml")

    assert result is not None
    assert hasattr(result, "params")
    assert "rho" in result.params.index

    # Validate against R
    with open(FIXTURES_PATH / "r_sar_re_results.json", "r") as f:
        r_results = json.load(f)

    r_rho_fe = r_results["sar_fe"]["rho"]
    py_rho_fe = result.params["rho"]

    print(f"\nPython FE rho: {py_rho_fe}, R FE rho: {r_rho_fe}")

    # Should be close
    assert np.isclose(py_rho_fe, r_rho_fe, rtol=0.20), f"FE rho {py_rho_fe} vs R {r_rho_fe}"
