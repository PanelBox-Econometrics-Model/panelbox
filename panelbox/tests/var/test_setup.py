# tests/var/test_setup.py
"""
Test to verify that the validation infrastructure is set up correctly.
This is a sanity check for FASE 1 completion.
"""

import numpy as np
import pytest


def test_panel_data_fixture(panel_data):
    """Test that panel_data fixture generates data correctly."""
    # Check shape
    assert panel_data.shape == (1000, 5), "Expected 50 entities x 20 periods = 1000 rows"

    # Check columns
    expected_cols = ["entity", "time", "y1", "y2", "y3"]
    assert list(panel_data.columns) == expected_cols

    # Check data types
    assert panel_data["entity"].dtype == np.int64
    assert panel_data["time"].dtype == np.int64
    assert panel_data["y1"].dtype == np.float64

    # Check no NaN/Inf
    assert not panel_data.isna().any().any()
    assert not np.isinf(panel_data.select_dtypes(include=[np.number])).any().any()


def test_true_params_fixture(true_params):
    """Test that true_params fixture contains correct DGP parameters."""
    # Check keys
    assert "A1" in true_params
    assert "Sigma" in true_params

    # Check shapes
    assert true_params["A1"].shape == (3, 3)
    assert true_params["Sigma"].shape == (3, 3)

    # Check specific values (DGP definition)
    assert true_params["A1"][0, 0] == 0.5  # y1 <- y1 coefficient
    assert true_params["A1"][0, 1] == 0.2  # y1 <- y2 coefficient
    assert true_params["Sigma"][0, 0] == 1.0  # variance of y1 error


def test_r_benchmarks_fixture(r_benchmarks):
    """Test that R benchmarks were loaded correctly."""
    # Check top-level keys
    assert "gmm" in r_benchmarks
    assert "irf" in r_benchmarks
    assert "fevd" in r_benchmarks
    assert "metadata" in r_benchmarks

    # Check GMM section
    assert "coefficients" in r_benchmarks["gmm"]
    assert "standard_errors" in r_benchmarks["gmm"]
    assert "vcov" in r_benchmarks["gmm"]
    assert len(r_benchmarks["gmm"]["coefficients"]) == 12  # 3 vars x (3 lags + 1 const)

    # Check IRF section
    assert "orthogonalized" in r_benchmarks["irf"]
    assert "generalized" in r_benchmarks["irf"]
    assert "y1" in r_benchmarks["irf"]["orthogonalized"]

    # Check FEVD section
    assert "y1" in r_benchmarks["fevd"]
    assert "y2" in r_benchmarks["fevd"]
    assert "y3" in r_benchmarks["fevd"]

    # Check metadata
    assert r_benchmarks["metadata"]["n_entities"] == 50
    assert r_benchmarks["metadata"]["n_periods"] == 20
    assert r_benchmarks["metadata"]["lags"] == 1


def test_data_generation_seed_consistency():
    """Test that data generation is consistent with the same seed."""
    from panelbox.tests.var.fixtures.var_test_data import generate_panel_var_data

    data1 = generate_panel_var_data(n_entities=10, n_periods=10, seed=123)
    data2 = generate_panel_var_data(n_entities=10, n_periods=10, seed=123)

    # Should be identical with same seed
    np.testing.assert_array_equal(data1.values, data2.values)


def test_fevd_sums_to_one(r_benchmarks):
    """Test that FEVD decompositions sum to 1 at each horizon."""
    for var in ["y1", "y2", "y3"]:
        fevd = r_benchmarks["fevd"][var]
        for period_fevd in fevd:
            # Each row should sum to 1 (100% variance explained)
            assert abs(sum(period_fevd) - 1.0) < 1e-10, f"FEVD for {var} doesn't sum to 1"
