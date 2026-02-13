"""
Tests for orthogonalized Impulse Response Functions (Cholesky).

This module tests:
- IRF computation using Cholesky decomposition
- Recursive vs companion matrix methods
- IRF properties (convergence, h=0 values)
- Cumulative IRFs
- Variable ordering effects
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


def test_irf_cholesky_basic(simple_var_data):
    """Test basic IRF computation with Cholesky."""
    df, A1_true, Sigma_true = simple_var_data

    # Prepare data
    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    # Estimate VAR
    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Compute IRFs
    irf_result = result.irf(periods=10, method="cholesky")

    # Check shape
    assert irf_result.irf_matrix.shape == (11, 2, 2)

    # Check attributes
    assert irf_result.K == 2
    assert irf_result.periods == 10
    assert irf_result.method == "cholesky"
    assert irf_result.var_names == ["y1", "y2"]


def test_irf_h0_equals_cholesky_diagonal(simple_var_data):
    """Test that IRF at h=0 on diagonal equals sqrt(sigma_ii)."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    irf_result = result.irf(periods=10, method="cholesky")

    # At h=0, IRF[i,i] should be sqrt(Sigma[i,i]) (diagonal of P)
    P = np.linalg.cholesky(result.Sigma)

    for i in range(2):
        assert np.isclose(
            irf_result.irf_matrix[0, i, i], P[i, i], atol=1e-6
        ), f"IRF[0, {i}, {i}] should equal P[{i}, {i}]"


def test_irf_convergence_stable_system(simple_var_data):
    """Test that IRFs converge to zero for stable system."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Check stability
    assert result.is_stable(), "VAR system should be stable"

    # Compute IRFs for long horizon
    irf_result = result.irf(periods=50, method="cholesky")

    # IRFs should converge to zero
    final_irf = irf_result.irf_matrix[-1]
    assert np.allclose(final_irf, 0, atol=1e-3), "IRFs should converge to zero"


def test_irf_cumulative(simple_var_data):
    """Test cumulative IRFs."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Compute regular IRFs
    irf_regular = result.irf(periods=10, method="cholesky", cumulative=False)

    # Compute cumulative IRFs
    irf_cumulative = result.irf(periods=10, method="cholesky", cumulative=True)

    # Manual cumulative sum
    manual_cumsum = np.cumsum(irf_regular.irf_matrix, axis=0)

    # Check that cumulative matches manual cumsum
    assert np.allclose(
        irf_cumulative.irf_matrix, manual_cumsum, atol=1e-10
    ), "Cumulative IRFs should match manual cumsum"


def test_irf_ordering_affects_cholesky(simple_var_data):
    """Test that variable ordering affects Cholesky IRFs."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Default order
    irf_order1 = result.irf(periods=10, method="cholesky")

    # Reversed order
    irf_order2 = result.irf(periods=10, method="cholesky", order=["y2", "y1"])

    # IRFs should differ (ordering matters for Cholesky)
    assert not np.allclose(
        irf_order1.irf_matrix, irf_order2.irf_matrix, atol=1e-6
    ), "Cholesky IRFs should depend on ordering"


def test_irf_shock_size(simple_var_data):
    """Test different shock sizes."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # One std shock
    irf_1std = result.irf(periods=10, method="cholesky", shock_size="one_std")

    # 2x shock
    irf_2x = result.irf(periods=10, method="cholesky", shock_size=2.0)

    # IRF with 2x shock should be exactly 2x the 1std IRF
    assert np.allclose(
        irf_2x.irf_matrix, 2.0 * irf_1std.irf_matrix, atol=1e-10
    ), "IRF with 2x shock should be 2x the 1std IRF"


def test_irf_result_getitem(simple_var_data):
    """Test IRFResult __getitem__ accessor."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    irf_result = result.irf(periods=10, method="cholesky")

    # Access specific IRF
    irf_y1_to_y2 = irf_result["y1", "y2"]

    assert irf_y1_to_y2.shape == (11,)
    assert np.allclose(irf_y1_to_y2, irf_result.irf_matrix[:, 0, 1])


def test_irf_to_dataframe(simple_var_data):
    """Test IRFResult to_dataframe method."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    irf_result = result.irf(periods=10, method="cholesky")

    # Full DataFrame
    df_full = irf_result.to_dataframe()
    assert df_full.shape == (11, 5)  # horizon + 4 IRFs (2x2)

    # Filter by impulse
    df_impulse = irf_result.to_dataframe(impulse="y1")
    assert df_impulse.shape == (11, 3)  # horizon + 2 responses

    # Filter by response
    df_response = irf_result.to_dataframe(response="y2")
    assert df_response.shape == (11, 3)  # horizon + 2 impulses


def test_irf_summary(simple_var_data):
    """Test IRFResult summary method."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    irf_result = result.irf(periods=10, method="cholesky")

    # Get summary
    summary = irf_result.summary()

    assert isinstance(summary, str)
    assert "Impulse Response Functions" in summary
    assert "cholesky" in summary.lower()


def test_irf_manual_validation_var1():
    """
    Manually validate IRF computation for simple VAR(1) case.

    For VAR(1): Φ_1 = A_1 · P, Φ_2 = A_1 · Φ_1 = A_1^2 · P, etc.
    """
    # Define simple VAR(1)
    A1 = np.array([[0.5, 0.2], [0.1, 0.4]])
    Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])  # Diagonal for simplicity

    # Cholesky
    P = np.linalg.cholesky(Sigma)  # = I for diagonal Sigma

    # Manual IRF calculation
    Phi0_manual = P
    Phi1_manual = A1 @ P
    Phi2_manual = A1 @ Phi1_manual
    Phi3_manual = A1 @ Phi2_manual

    # Compute using function
    from panelbox.var.irf import compute_irf_cholesky

    Phi_computed = compute_irf_cholesky([A1], Sigma, periods=3)

    # Validate
    assert np.allclose(Phi_computed[0], Phi0_manual, atol=1e-10)
    assert np.allclose(Phi_computed[1], Phi1_manual, atol=1e-10)
    assert np.allclose(Phi_computed[2], Phi2_manual, atol=1e-10)
    assert np.allclose(Phi_computed[3], Phi3_manual, atol=1e-10)


def test_irf_recursive_vs_companion(simple_var_data):
    """Test that recursive and companion methods give same IRFs."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Compute using recursive method (default)
    from panelbox.var.irf import compute_irf_cholesky

    Phi_recursive = compute_irf_cholesky(result.A_matrices, result.Sigma, periods=10)

    # Compute using companion method
    from panelbox.var.irf import compute_irf_companion

    P = np.linalg.cholesky(result.Sigma)
    companion = result.companion_matrix()
    Phi_companion = compute_irf_companion(companion, P, periods=10, K=2)

    # Should be identical
    assert np.allclose(
        Phi_recursive, Phi_companion, atol=1e-8
    ), "Recursive and companion methods should give identical IRFs"


def test_irf_invalid_method():
    """Test that invalid method raises error."""
    np.random.seed(42)

    # Create dummy data
    df = pd.DataFrame(
        {
            "entity": [0] * 50,
            "time": range(50),
            "y1": np.random.randn(50),
            "y2": np.random.randn(50),
        }
    )

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Invalid method should raise ValueError
    with pytest.raises(ValueError, match="Unknown method"):
        result.irf(periods=10, method="invalid_method")


def test_irf_invalid_ordering():
    """Test that invalid ordering raises error."""
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "entity": [0] * 50,
            "time": range(50),
            "y1": np.random.randn(50),
            "y2": np.random.randn(50),
        }
    )

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Invalid ordering (wrong variables)
    with pytest.raises(ValueError, match="order must contain exactly"):
        result.irf(periods=10, order=["y1", "y3"])

    # Invalid ordering (missing variable)
    with pytest.raises(ValueError, match="order must contain exactly"):
        result.irf(periods=10, order=["y1"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
