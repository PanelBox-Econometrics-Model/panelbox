"""
Tests for Generalized Impulse Response Functions (Pesaran-Shin).

This module tests:
- GIRF computation
- Order invariance property
- Comparison with Cholesky for diagonal covariance
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.var import PanelVAR, PanelVARData


@pytest.fixture
def simple_var_data():
    """
    Create simple VAR(1) data for testing.

    DGP:
    y1_t = 0.5 * y1_{t-1} + 0.2 * y2_{t-1} + e1_t
    y2_t = 0.1 * y1_{t-1} + 0.6 * y2_{t-1} + e2_t

    with Cov(e1, e2) = [[1.0, 0.3], [0.3, 0.8]]
    """
    np.random.seed(42)

    # True parameters
    A1 = np.array([[0.5, 0.2], [0.1, 0.6]])
    Sigma = np.array([[1.0, 0.3], [0.3, 0.8]])

    # Simulate data
    N = 30  # entities
    T = 50  # time periods
    K = 2  # variables

    data_list = []

    for i in range(N):
        # Generate errors
        errors = np.random.multivariate_normal(np.zeros(K), Sigma, size=T + 20)

        # Generate VAR(1) data
        y = np.zeros((T + 20, K))
        for t in range(1, T + 20):
            y[t] = A1 @ y[t - 1] + errors[t]

        # Discard burn-in
        y = y[20:]

        # Create DataFrame
        df_entity = pd.DataFrame(
            {
                "entity": i,
                "time": range(T),
                "y1": y[:, 0],
                "y2": y[:, 1],
            }
        )
        data_list.append(df_entity)

    df = pd.concat(data_list, ignore_index=True)

    return df, A1, Sigma


@pytest.fixture
def diagonal_var_data():
    """Create VAR data with diagonal covariance (no contemporaneous correlation)."""
    np.random.seed(123)

    A1 = np.array([[0.6, 0.15], [0.2, 0.5]])
    Sigma = np.array([[1.0, 0.0], [0.0, 0.8]])  # Diagonal!

    N = 30
    T = 50
    K = 2

    data_list = []

    for i in range(N):
        errors = np.random.multivariate_normal(np.zeros(K), Sigma, size=T + 20)

        y = np.zeros((T + 20, K))
        for t in range(1, T + 20):
            y[t] = A1 @ y[t - 1] + errors[t]

        y = y[20:]

        df_entity = pd.DataFrame(
            {
                "entity": i,
                "time": range(T),
                "y1": y[:, 0],
                "y2": y[:, 1],
            }
        )
        data_list.append(df_entity)

    df = pd.concat(data_list, ignore_index=True)

    return df, A1, Sigma


def test_girf_basic(simple_var_data):
    """Test basic GIRF computation."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Compute GIRFs
    girf_result = result.irf(periods=10, method="generalized")

    # Check shape
    assert girf_result.irf_matrix.shape == (11, 2, 2)

    # Check attributes
    assert girf_result.method == "generalized"
    assert girf_result.K == 2
    assert girf_result.periods == 10


def test_girf_order_invariance(simple_var_data):
    """Test that GIRF is invariant to variable ordering."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Default order
    girf_order1 = result.irf(periods=10, method="generalized")

    # Reversed order
    girf_order2 = result.irf(periods=10, method="generalized", order=["y2", "y1"])

    # For GIRF, we need to reorder the second result back to original order
    # to compare properly
    # girf_order2 is in order [y2, y1], so:
    #   girf_order2[:, 0, 0] = response of y2 to y2
    #   girf_order2[:, 0, 1] = response of y2 to y1
    #   girf_order2[:, 1, 0] = response of y1 to y2
    #   girf_order2[:, 1, 1] = response of y1 to y1

    # We need to remap to [y1, y2] order:
    #   girf_reordered[:, 0, 0] = response of y1 to y1 = girf_order2[:, 1, 1]
    #   girf_reordered[:, 0, 1] = response of y1 to y2 = girf_order2[:, 1, 0]
    #   girf_reordered[:, 1, 0] = response of y2 to y1 = girf_order2[:, 0, 1]
    #   girf_reordered[:, 1, 1] = response of y2 to y2 = girf_order2[:, 0, 0]

    girf_reordered = np.zeros_like(girf_order2.irf_matrix)
    girf_reordered[:, 0, 0] = girf_order2.irf_matrix[:, 1, 1]  # y1 to y1
    girf_reordered[:, 0, 1] = girf_order2.irf_matrix[:, 1, 0]  # y1 to y2
    girf_reordered[:, 1, 0] = girf_order2.irf_matrix[:, 0, 1]  # y2 to y1
    girf_reordered[:, 1, 1] = girf_order2.irf_matrix[:, 0, 0]  # y2 to y2

    # GIRFs should be invariant to ordering
    assert np.allclose(
        girf_order1.irf_matrix, girf_reordered, atol=1e-8
    ), "GIRF should be invariant to variable ordering"


def test_girf_equals_cholesky_for_diagonal_sigma(diagonal_var_data):
    """Test that GIRF equals Cholesky IRF when Sigma is diagonal."""
    df, A1_true, Sigma_true = diagonal_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Check that estimated Sigma is approximately diagonal
    off_diag = result.Sigma[0, 1]
    assert abs(off_diag) < 0.15, "Sigma should be approximately diagonal"

    # Compute both
    chol_irf = result.irf(periods=10, method="cholesky")
    girf = result.irf(periods=10, method="generalized")

    # For diagonal Sigma, GIRF should be very close to Cholesky IRF
    # (up to scaling, since GIRF normalizes by sqrt(sigma_jj))
    # Actually, for diagonal Sigma, they should be identical after proper normalization

    # Check that they're close (allowing for estimation noise)
    assert np.allclose(
        chol_irf.irf_matrix, girf.irf_matrix, atol=0.05
    ), "For diagonal Sigma, GIRF should approximate Cholesky IRF"


def test_girf_differs_from_cholesky_for_nondiagonal(simple_var_data):
    """Test that GIRF differs from Cholesky when Sigma is non-diagonal."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Check that Sigma has significant off-diagonal
    off_diag = result.Sigma[0, 1]
    assert abs(off_diag) > 0.1, "Sigma should have non-trivial correlation"

    # Compute both
    chol_irf = result.irf(periods=10, method="cholesky")
    girf = result.irf(periods=10, method="generalized")

    # They should differ
    assert not np.allclose(
        chol_irf.irf_matrix, girf.irf_matrix, atol=1e-6
    ), "GIRF should differ from Cholesky for non-diagonal Sigma"


def test_girf_cumulative(simple_var_data):
    """Test cumulative GIRFs."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Regular GIRF
    girf_regular = result.irf(periods=10, method="generalized", cumulative=False)

    # Cumulative GIRF
    girf_cumulative = result.irf(periods=10, method="generalized", cumulative=True)

    # Manual cumsum
    manual_cumsum = np.cumsum(girf_regular.irf_matrix, axis=0)

    assert np.allclose(girf_cumulative.irf_matrix, manual_cumsum, atol=1e-10)


def test_girf_convergence(simple_var_data):
    """Test that GIRFs converge to zero for stable system."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    assert result.is_stable(), "VAR should be stable"

    # Long horizon GIRF
    girf = result.irf(periods=50, method="generalized")

    # Should converge to zero
    final_girf = girf.irf_matrix[-1]
    assert np.allclose(final_girf, 0, atol=1e-3), "GIRF should converge to zero"


def test_girf_manual_validation():
    """Manually validate GIRF computation for simple case."""
    # Simple VAR(1) with diagonal Sigma
    A1 = np.array([[0.5, 0.2], [0.1, 0.4]])
    Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])

    from panelbox.var.irf import compute_irf_generalized, compute_phi_non_orthogonalized

    # Compute non-orthogonalized Phi
    Phi = compute_phi_non_orthogonalized([A1], periods=3)

    # Should have Phi[0] = I
    assert np.allclose(Phi[0], np.eye(2), atol=1e-10)

    # Phi[1] = A1 @ Phi[0] = A1 @ I = A1
    assert np.allclose(Phi[1], A1, atol=1e-10)

    # Phi[2] = A1 @ Phi[1] = A1 @ A1 = A1²
    assert np.allclose(Phi[2], A1 @ A1, atol=1e-10)

    # Compute GIRF
    GIRF = compute_irf_generalized(Phi, Sigma, periods=3)

    # For diagonal Sigma with σ_11 = σ_22 = 1:
    # GIRF[h, i, j] = Phi[h, i, :] @ Sigma @ e_j / sqrt(σ_jj)
    #               = Phi[h, i, :] @ e_j  (since Sigma = I and sqrt(1) = 1)
    #               = Phi[h, i, j]

    assert np.allclose(GIRF, Phi, atol=1e-10), "GIRF should equal Phi for Sigma=I"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
