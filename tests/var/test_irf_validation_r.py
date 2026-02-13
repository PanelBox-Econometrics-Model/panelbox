"""
Validation tests for IRFs against R's vars package.

This module tests that our IRF implementations match R's vars::irf() function.
"""

import json
import os
import subprocess
import tempfile

import numpy as np
import pytest

from panelbox.var.irf import (
    compute_irf_cholesky,
    compute_irf_generalized,
    compute_phi_non_orthogonalized,
)


def run_r_irf_cholesky(y_data, p, periods):
    """
    Run R's vars::irf() with Cholesky decomposition and return results.

    Parameters
    ----------
    y_data : np.ndarray
        Data matrix (T, K)
    p : int
        Lag order
    periods : int
        Number of IRF periods

    Returns
    -------
    irf_r : np.ndarray
        IRF matrix from R (periods+1, K, K)
    """
    T, K = y_data.shape

    # Create R script
    r_script = f"""
library(vars)

# Load data
y <- matrix(c({','.join(map(str, y_data.flatten('F')))}), nrow={T}, ncol={K})

# Fit VAR
var_model <- VAR(y, p={p}, type="none")

# Compute IRF (Cholesky)
irf_result <- irf(var_model, n.ahead={periods}, ortho=TRUE, boot=FALSE)

# Extract IRF matrix
# irf_result$irf is a list with K elements (one per impulse variable)
# Each element is a matrix (periods+1, K) of responses

irf_array <- array(0, dim=c({periods+1}, {K}, {K}))

for (j in 1:{K}) {{
    irf_array[, , j] <- irf_result$irf[[j]]
}}

# Output as JSON
library(jsonlite)
cat(toJSON(irf_array))
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
        irf_r = np.array(json.loads(result.stdout))

        return irf_r

    finally:
        # Clean up temporary file
        os.unlink(r_script_path)


def run_r_irf_generalized(y_data, p, periods):
    """
    Run R's vars::irf() with generalized (non-orthogonal) IRF.

    Parameters
    ----------
    y_data : np.ndarray
        Data matrix (T, K)
    p : int
        Lag order
    periods : int
        Number of IRF periods

    Returns
    -------
    irf_r : np.ndarray
        Generalized IRF matrix from R (periods+1, K, K)
    """
    T, K = y_data.shape

    # Create R script
    r_script = f"""
library(vars)

# Load data
y <- matrix(c({','.join(map(str, y_data.flatten('F')))}), nrow={T}, ncol={K})

# Fit VAR
var_model <- VAR(y, p={p}, type="none")

# Compute Generalized IRF (ortho=FALSE)
irf_result <- irf(var_model, n.ahead={periods}, ortho=FALSE, boot=FALSE)

# Extract IRF matrix
irf_array <- array(0, dim=c({periods+1}, {K}, {K}))

for (j in 1:{K}) {{
    irf_array[, , j] <- irf_result$irf[[j]]
}}

# Output as JSON
library(jsonlite)
cat(toJSON(irf_array))
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
        irf_r = np.array(json.loads(result.stdout))

        return irf_r

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

    # Reshape into A matrices
    A_matrices = []
    for lag in range(p):
        A_l = A_flat[lag * K : (lag + 1) * K, :].T
        A_matrices.append(A_l)

    # Compute residuals and Sigma
    resid = Y - X @ A_flat
    Sigma = (resid.T @ resid) / (T - p)

    return A_matrices, Sigma


def test_irf_cholesky_vs_r_simple(simple_var_data):
    """Test Cholesky IRF against R's vars::irf() - simple case."""
    y, A1_true, Sigma_true, p = simple_var_data

    T, K = y.shape
    periods = 10

    # Estimate VAR in Python (same as R)
    A_matrices, Sigma = estimate_var(y, p)

    # Compute IRF in Python
    irf_python = compute_irf_cholesky(A_matrices, Sigma, periods)

    # Compute IRF in R
    irf_r = run_r_irf_cholesky(y, p, periods)

    # Compare
    # Note: R may have small differences in OLS estimation details
    # We compare element-wise with reasonable tolerance
    np.testing.assert_allclose(irf_python, irf_r, rtol=1e-2, atol=1e-2)


def test_irf_cholesky_vs_r_larger(simple_var_data):
    """Test Cholesky IRF against R with longer horizon."""
    y, A1_true, Sigma_true, p = simple_var_data

    periods = 20

    # Estimate VAR in Python (same as R)
    A_matrices, Sigma = estimate_var(y, p)

    # Compute IRF in Python
    irf_python = compute_irf_cholesky(A_matrices, Sigma, periods)

    # Compute IRF in R
    irf_r = run_r_irf_cholesky(y, p, periods)

    # Compare
    np.testing.assert_allclose(irf_python, irf_r, rtol=1e-2, atol=1e-2)


@pytest.mark.slow
def test_irf_generalized_vs_r_simple(simple_var_data):
    """
    Test non-orthogonalized IRF (Phi_h) against R's vars::irf() with ortho=FALSE.

    Note: R's ortho=FALSE returns Phi_h (MA coefficients), not Pesaran-Shin GIRF.
    Our GIRF implementation follows Pesaran & Shin (1998), which is different.
    """
    y, A1_true, Sigma_true, p = simple_var_data

    periods = 10

    # Estimate VAR in Python (same as R)
    A_matrices, Sigma = estimate_var(y, p)

    # Compute non-orthogonalized IRF (Phi_h) in Python
    # This should match R's ortho=FALSE
    Phi = compute_phi_non_orthogonalized(A_matrices, periods)

    # Compute non-orthogonalized IRF in R
    irf_r = run_r_irf_generalized(y, p, periods)

    # Compare Phi_h directly (not GIRF)
    # R's ortho=FALSE returns Phi_h, which is what we compute in Phi
    np.testing.assert_allclose(Phi, irf_r, rtol=1e-4, atol=1e-5)


def test_irf_cholesky_convergence_vs_r(simple_var_data):
    """Test that IRFs converge to zero (stable VAR)."""
    y, A1_true, Sigma_true, p = simple_var_data

    periods = 50

    # Estimate VAR in Python (same as R)
    A_matrices, Sigma = estimate_var(y, p)

    # Compute IRF in Python
    irf_python = compute_irf_cholesky(A_matrices, Sigma, periods)

    # Compute IRF in R
    irf_r = run_r_irf_cholesky(y, p, periods)

    # Both should converge to near-zero
    assert np.allclose(irf_python[periods], 0, atol=1e-3)
    assert np.allclose(irf_r[periods], 0, atol=1e-3)

    # And they should converge to the same value
    np.testing.assert_allclose(irf_python[periods], irf_r[periods], atol=1e-4)


@pytest.mark.slow
def test_irf_cholesky_var2_vs_r():
    """Test Cholesky IRF for VAR(2) against R."""
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

    # Estimate VAR in Python (same as R)
    A_matrices, Sigma_est = estimate_var(y_data, p)

    # Compute IRF in Python
    irf_python = compute_irf_cholesky(A_matrices, Sigma_est, periods)

    # Compute IRF in R
    irf_r = run_r_irf_cholesky(y_data, p, periods)

    # Compare
    np.testing.assert_allclose(irf_python, irf_r, rtol=1e-2, atol=1e-2)


@pytest.mark.slow
def test_irf_ordering_effect_matches_r(simple_var_data):
    """Test that changing variable order affects Cholesky IRF the same way in R and Python."""
    y, A1_true, Sigma_true, p = simple_var_data

    periods = 10

    # Original order - estimate from data
    A_matrices, Sigma = estimate_var(y, p)
    irf_python_order1 = compute_irf_cholesky(A_matrices, Sigma, periods)
    irf_r_order1 = run_r_irf_cholesky(y, p, periods)

    # Reversed order
    # Permute variables: swap columns 0 and 1
    perm = np.array([1, 0])
    y_perm = y[:, perm]

    A_matrices_perm, Sigma_perm = estimate_var(y_perm, p)
    irf_python_order2 = compute_irf_cholesky(A_matrices_perm, Sigma_perm, periods)
    irf_r_order2 = run_r_irf_cholesky(y_perm, p, periods)

    # Compare: Python order1 vs R order1
    np.testing.assert_allclose(irf_python_order1, irf_r_order1, rtol=1e-2, atol=1e-2)

    # Compare: Python order2 vs R order2
    np.testing.assert_allclose(irf_python_order2, irf_r_order2, rtol=1e-2, atol=1e-2)

    # IRFs should be different for different orders
    assert not np.allclose(irf_python_order1, irf_python_order2, atol=1e-3)


def test_irf_initial_impact_vs_r(simple_var_data):
    """Test that initial impact (h=0) matches between Python and R."""
    y, A1_true, Sigma_true, p = simple_var_data

    periods = 10

    # Estimate from data (same as R)
    A_matrices, Sigma = estimate_var(y, p)

    # Compute IRF in Python
    irf_python = compute_irf_cholesky(A_matrices, Sigma, periods)

    # Compute IRF in R
    irf_r = run_r_irf_cholesky(y, p, periods)

    # Initial impact should be Cholesky factor
    P = np.linalg.cholesky(Sigma)

    # Check h=0
    np.testing.assert_allclose(irf_python[0], P, rtol=1e-10)
    np.testing.assert_allclose(irf_r[0], P, rtol=1e-3)  # R may have slight numerical differences


if __name__ == "__main__":
    # Run a simple test manually
    print("Testing IRF validation against R...")

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

    # Estimate VAR (same as R)
    A_matrices, Sigma_est = estimate_var(y_data, p)

    irf_python = compute_irf_cholesky(A_matrices, Sigma_est, periods)
    irf_r = run_r_irf_cholesky(y_data, p, periods)

    print(f"Python IRF[0]:\n{irf_python[0]}")
    print(f"R IRF[0]:\n{irf_r[0]}")
    print(f"Difference: {np.max(np.abs(irf_python - irf_r))}")

    if np.allclose(irf_python, irf_r, rtol=1e-2, atol=1e-2):
        print("✓ Python and R IRFs match!")
    else:
        print("✗ Python and R IRFs differ")
