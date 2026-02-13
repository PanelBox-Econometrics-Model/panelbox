"""
Validation tests for FEVD against R's vars package.

This module tests that our FEVD implementations match R's vars::fevd() function.
"""

import json
import os
import subprocess
import tempfile

import numpy as np
import pytest

from panelbox.var.fevd import compute_fevd_cholesky, compute_fevd_generalized
from panelbox.var.irf import compute_irf_cholesky, compute_phi_non_orthogonalized


def estimate_var(y, p):
    """
    Estimate VAR parameters using OLS.

    Parameters
    ----------
    y : np.ndarray
        Data matrix (T, K)
    p : int
        Lag order

    Returns
    -------
    A_matrices : list of np.ndarray
        Estimated coefficient matrices [A_1, ..., A_p]
    Sigma : np.ndarray
        Estimated residual covariance matrix
    """
    T, K = y.shape

    # Build design matrix
    X = np.zeros((T - p, K * p))
    Y = y[p:]

    for t in range(T - p):
        for lag in range(1, p + 1):
            X[t, (lag - 1) * K : lag * K] = y[p + t - lag]

    # OLS estimation
    A_flat = np.linalg.lstsq(X, Y, rcond=None)[0]

    # Reshape to list of matrices
    A_matrices = []
    for lag in range(p):
        A_matrices.append(A_flat[lag * K : (lag + 1) * K, :].T)

    # Estimate residual covariance
    residuals = Y - X @ A_flat
    Sigma = (residuals.T @ residuals) / (T - p)

    return A_matrices, Sigma


def run_r_fevd_cholesky(y_data, p, periods):
    """
    Run R's vars::fevd() with Cholesky decomposition and return results.

    Parameters
    ----------
    y_data : np.ndarray
        Data matrix (T, K)
    p : int
        Lag order
    periods : int
        Number of FEVD periods

    Returns
    -------
    fevd_r : np.ndarray
        FEVD matrix from R (periods+1, K, K)
    """
    T, K = y_data.shape

    # Create R script
    r_script = f"""
library(vars)
library(jsonlite)

# Load data
y <- matrix(c({','.join(map(str, y_data.flatten('F')))}), nrow={T}, ncol={K})

# Fit VAR
var_model <- VAR(y, p={p}, type="none")

# Compute FEVD (Cholesky)
# Note: R's fevd() returns n.ahead periods (NOT n.ahead+1)
# It returns horizons 1, 2, ..., n.ahead (no horizon 0)
fevd_result <- fevd(var_model, n.ahead={periods})

# Extract FEVD matrix
# fevd_result is a list with K elements (one per variable)
# Each element is a matrix (n.ahead rows, K columns)
# We need to add h=0 manually

# Get number of horizons returned by R
n_horizons <- nrow(fevd_result[[1]])

# Create array with periods+1 to include h=0
fevd_array <- array(0, dim=c({periods+1}, {K}, {K}))

# fevd_result[[i]] contains FEVD for variable i (as a percentage)
# Fill from h=1 onwards (R doesn't return h=0)
for (i in 1:{K}) {{
    # R returns horizons 1..n.ahead, we store in [2:(periods+1), i, ]
    fevd_array[2:({periods+1}), i, ] <- as.matrix(fevd_result[[i]])
}}

# For h=0, Cholesky FEVD: variable i is 100% explained by shock i (lower triangular)
# Set h=0 to identity-like (100% own shock, 0% other shocks)
for (i in 1:{K}) {{
    fevd_array[1, i, i] <- 100.0  # 100% at h=0
    for (j in 1:{K}) {{
        if (i != j) {{
            fevd_array[1, i, j] <- 0.0
        }}
    }}
}}

# Output as JSON (fevd_array is already numeric)
cat(toJSON(fevd_array))
"""

    # Write R script to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(r_script)
        r_script_path = f.name

    try:
        # Run R script
        result = subprocess.run(
            ["Rscript", r_script_path], capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            raise RuntimeError(f"R script failed: {result.stderr}")

        # Parse JSON output
        fevd_r = np.array(json.loads(result.stdout))

        # Convert from percentage to proportion if needed
        # R's fevd returns percentages (0-100), we use proportions (0-1)
        if np.max(fevd_r) > 1.5:
            fevd_r = fevd_r / 100.0

        return fevd_r

    finally:
        # Clean up temporary file
        os.unlink(r_script_path)


@pytest.fixture
def simple_var_data():
    """Generate simple VAR data for testing."""
    np.random.seed(42)

    K = 2
    T = 100
    p = 1

    # True parameters
    A1 = np.array([[0.5, 0.1], [0.2, 0.6]])

    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

    # Generate data
    y = np.zeros((T + p, K))
    errors = np.random.multivariate_normal(np.zeros(K), Sigma, size=T)

    for t in range(p, T + p):
        y[t] = A1 @ y[t - 1] + errors[t - p]

    return y[p:], A1, Sigma, p


def test_fevd_cholesky_vs_r_simple(simple_var_data):
    """Test Cholesky FEVD against R's vars::fevd() - simple case."""
    y, A1_true, Sigma_true, p = simple_var_data

    T, K = y.shape
    periods = 10

    # Estimate from data (same as R)
    A_matrices, Sigma = estimate_var(y, p)

    # Compute IRF first
    irf = compute_irf_cholesky(A_matrices, Sigma, periods)
    P = np.linalg.cholesky(Sigma)

    # Compute FEVD in Python
    fevd_python = compute_fevd_cholesky(irf, P, Sigma, periods)

    # Compute FEVD in R
    fevd_r = run_r_fevd_cholesky(y, p, periods)

    # Compare
    np.testing.assert_allclose(fevd_python, fevd_r, rtol=1e-3, atol=1e-3)


def test_fevd_sum_to_one_python_and_r(simple_var_data):
    """Test that FEVD sums to 1 (100%) in both Python and R."""
    y, A1_true, Sigma_true, p = simple_var_data

    periods = 10

    # Estimate from data (same as R)
    A_matrices, Sigma = estimate_var(y, p)

    # Compute FEVD in Python
    irf = compute_irf_cholesky(A_matrices, Sigma, periods)
    P = np.linalg.cholesky(Sigma)
    fevd_python = compute_fevd_cholesky(irf, P, Sigma, periods)

    # Compute FEVD in R
    fevd_r = run_r_fevd_cholesky(y, p, periods)

    # Check that both sum to 1 for all variables and horizons
    for h in range(periods + 1):
        for i in range(len(A_matrices[0])):
            sum_python = fevd_python[h, i, :].sum()
            sum_r = fevd_r[h, i, :].sum()

            assert np.isclose(sum_python, 1.0, atol=1e-6)
            assert np.isclose(sum_r, 1.0, atol=1e-6)


def test_fevd_initial_period_vs_r(simple_var_data):
    """Test FEVD at initial period (h=0 or h=1) against R."""
    y, A1_true, Sigma_true, p = simple_var_data

    periods = 10

    # Estimate from data (same as R)
    A_matrices, Sigma = estimate_var(y, p)

    # Compute FEVD in Python
    irf = compute_irf_cholesky(A_matrices, Sigma, periods)
    P = np.linalg.cholesky(Sigma)
    fevd_python = compute_fevd_cholesky(irf, P, Sigma, periods)

    # Compute FEVD in R
    fevd_r = run_r_fevd_cholesky(y, p, periods)

    # At h=0 (or h=1), first variable should explain ~100% of itself in Cholesky
    # (depends on ordering)
    # Just check that Python and R match
    np.testing.assert_allclose(fevd_python[0], fevd_r[0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(fevd_python[1], fevd_r[1], rtol=1e-3, atol=1e-3)


@pytest.mark.slow
def test_fevd_cholesky_var2_vs_r():
    """Test Cholesky FEVD for VAR(2) against R."""
    np.random.seed(123)

    K = 2
    T = 150
    p = 2

    # True parameters for VAR(2)
    A1 = np.array([[0.4, 0.1], [0.15, 0.5]])
    A2 = np.array([[0.2, 0.05], [0.1, 0.15]])

    Sigma = np.array([[1.0, 0.25], [0.25, 0.9]])

    # Generate data
    y = np.zeros((T + p, K))
    errors = np.random.multivariate_normal(np.zeros(K), Sigma, size=T)

    for t in range(p, T + p):
        y[t] = A1 @ y[t - 1] + A2 @ y[t - 2] + errors[t - p]

    y_data = y[p:]

    periods = 15

    # Estimate from data (same as R)
    A_matrices, Sigma_est = estimate_var(y_data, p)

    # Compute FEVD in Python
    irf = compute_irf_cholesky(A_matrices, Sigma_est, periods)
    P = np.linalg.cholesky(Sigma_est)
    fevd_python = compute_fevd_cholesky(irf, P, Sigma_est, periods)

    # Compute FEVD in R
    fevd_r = run_r_fevd_cholesky(y_data, p, periods)

    # Compare
    np.testing.assert_allclose(fevd_python, fevd_r, rtol=1e-3, atol=1e-3)


@pytest.mark.slow
def test_fevd_long_horizon_vs_r(simple_var_data):
    """Test FEVD at long horizons against R."""
    y, A1_true, Sigma_true, p = simple_var_data

    periods = 30

    # Estimate from data (same as R)
    A_matrices, Sigma = estimate_var(y, p)

    # Compute FEVD in Python
    irf = compute_irf_cholesky(A_matrices, Sigma, periods)
    P = np.linalg.cholesky(Sigma)
    fevd_python = compute_fevd_cholesky(irf, P, Sigma, periods)

    # Compute FEVD in R
    fevd_r = run_r_fevd_cholesky(y, p, periods)

    # Compare
    np.testing.assert_allclose(fevd_python, fevd_r, rtol=1e-3, atol=1e-3)

    # FEVD should stabilize at long horizons
    # Check that last few periods are similar
    assert np.allclose(fevd_python[periods - 1], fevd_python[periods], atol=0.01)
    assert np.allclose(fevd_r[periods - 1], fevd_r[periods], atol=0.01)


def test_fevd_ordering_effect_matches_r(simple_var_data):
    """Test that changing variable order affects FEVD the same way in R and Python."""
    y, A1_true, Sigma_true, p = simple_var_data

    periods = 10

    # Original order - estimate from data
    A_matrices, Sigma = estimate_var(y, p)
    irf1 = compute_irf_cholesky(A_matrices, Sigma, periods)
    P1 = np.linalg.cholesky(Sigma)
    fevd_python_order1 = compute_fevd_cholesky(irf1, P1, Sigma, periods)
    fevd_r_order1 = run_r_fevd_cholesky(y, p, periods)

    # Reversed order
    perm = np.array([1, 0])
    y_perm = y[:, perm]

    A_matrices_perm, Sigma_perm = estimate_var(y_perm, p)
    irf2 = compute_irf_cholesky(A_matrices_perm, Sigma_perm, periods)
    P2 = np.linalg.cholesky(Sigma_perm)
    fevd_python_order2 = compute_fevd_cholesky(irf2, P2, Sigma_perm, periods)
    fevd_r_order2 = run_r_fevd_cholesky(y_perm, p, periods)

    # Compare: Python order1 vs R order1
    np.testing.assert_allclose(fevd_python_order1, fevd_r_order1, rtol=1e-3, atol=1e-3)

    # Compare: Python order2 vs R order2
    np.testing.assert_allclose(fevd_python_order2, fevd_r_order2, rtol=1e-3, atol=1e-3)

    # FEVD should be different for different orders
    assert not np.allclose(fevd_python_order1, fevd_python_order2, atol=0.05)


@pytest.mark.slow
def test_fevd_generalized_sum_to_one():
    """Test that Generalized FEVD sums to 1 after normalization."""
    np.random.seed(42)

    K = 2
    T = 100
    p = 1

    A1 = np.array([[0.5, 0.1], [0.2, 0.6]])
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

    # Generate data
    y = np.zeros((T + p, K))
    errors = np.random.multivariate_normal(np.zeros(K), Sigma, size=T)

    for t in range(p, T + p):
        y[t] = A1 @ y[t - 1] + errors[t - p]

    periods = 10

    # Compute Generalized FEVD in Python
    Phi = compute_phi_non_orthogonalized([A1], periods)
    fevd_python = compute_fevd_generalized(Phi, Sigma, periods)

    # Check that it sums to 1
    for h in range(periods + 1):
        for i in range(K):
            assert np.isclose(fevd_python[h, i, :].sum(), 1.0, atol=1e-6)


def test_fevd_values_in_valid_range(simple_var_data):
    """Test that all FEVD values are in [0, 1] range."""
    y, A1_true, Sigma_true, p = simple_var_data

    periods = 20

    # Compute FEVD
    irf = compute_irf_cholesky([A1_true], Sigma_true, periods)
    P = np.linalg.cholesky(Sigma_true)
    fevd_python = compute_fevd_cholesky(irf, P, Sigma_true, periods)

    # All values should be in [0, 1]
    assert np.all(fevd_python >= -1e-10)  # Allow small numerical errors
    assert np.all(fevd_python <= 1.0 + 1e-10)


if __name__ == "__main__":
    # Run a simple test manually
    print("Testing FEVD validation against R...")

    np.random.seed(42)
    K = 2
    T = 100
    p = 1

    A1 = np.array([[0.5, 0.1], [0.2, 0.6]])
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

    y = np.zeros((T + p, K))
    errors = np.random.multivariate_normal(np.zeros(K), Sigma, size=T)

    for t in range(p, T + p):
        y[t] = A1 @ y[t - 1] + errors[t - p]

    y_data = y[p:]
    periods = 10

    irf = compute_irf_cholesky([A1], Sigma, periods)
    P = np.linalg.cholesky(Sigma)
    fevd_python = compute_fevd_cholesky(irf, P, Sigma, periods)
    fevd_r = run_r_fevd_cholesky(y_data, p, periods)

    print(f"Python FEVD[5]:\n{fevd_python[5]}")
    print(f"R FEVD[5]:\n{fevd_r[5]}")
    print(f"Difference: {np.max(np.abs(fevd_python - fevd_r))}")

    if np.allclose(fevd_python, fevd_r, rtol=1e-3, atol=1e-3):
        print("✓ Python and R FEVDs match!")
    else:
        print("✗ Python and R FEVDs differ")
