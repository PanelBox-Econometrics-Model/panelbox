"""
Tests for Forecast Error Variance Decomposition (FEVD).

This module tests:
- FEVD computation (Cholesky and Generalized)
- Sum-to-100% property
- Order invariance of Generalized FEVD
- h=0 properties for Cholesky FEVD
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

    A1 = np.array([[0.5, 0.2], [0.1, 0.6]])
    Sigma = np.array([[1.0, 0.3], [0.3, 0.8]])

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


def test_fevd_cholesky_basic(simple_var_data):
    """Test basic Cholesky FEVD computation."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Compute FEVD
    fevd_result = result.fevd(periods=10, method="cholesky")

    # Check shape
    assert fevd_result.decomposition.shape == (11, 2, 2)

    # Check attributes
    assert fevd_result.K == 2
    assert fevd_result.periods == 10
    assert fevd_result.method == "cholesky"
    assert fevd_result.var_names == ["y1", "y2"]


def test_fevd_sums_to_one(simple_var_data):
    """Test that FEVD sums to 100% for each variable at each horizon."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Cholesky FEVD
    fevd_chol = result.fevd(periods=10, method="cholesky")

    # Check that each row sums to 1
    for h in range(11):
        for i in range(2):
            row_sum = fevd_chol.decomposition[h, i, :].sum()
            assert np.isclose(
                row_sum, 1.0, atol=1e-6
            ), f"FEVD at h={h}, var={i} should sum to 1, got {row_sum}"

    # Generalized FEVD
    fevd_gen = result.fevd(periods=10, method="generalized")

    for h in range(11):
        for i in range(2):
            row_sum = fevd_gen.decomposition[h, i, :].sum()
            assert np.isclose(
                row_sum, 1.0, atol=1e-6
            ), f"Generalized FEVD at h={h}, var={i} should sum to 1, got {row_sum}"


def test_fevd_cholesky_h0_first_variable(simple_var_data):
    """
    Test that at h=0, first variable is 100% explained by itself (Cholesky).

    With Cholesky ordering, the first variable is contemporaneously exogenous,
    so at h=0 it should be 100% explained by its own shock.
    """
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    fevd_result = result.fevd(periods=10, method="cholesky")

    # At h=0, first variable (y1) should be 100% explained by itself
    assert np.isclose(
        fevd_result.decomposition[0, 0, 0], 1.0, atol=1e-6
    ), "First variable at h=0 should be 100% self-explained"

    # And 0% explained by second variable
    assert np.isclose(
        fevd_result.decomposition[0, 0, 1], 0.0, atol=1e-6
    ), "First variable at h=0 should not be explained by second variable"


def test_fevd_generalized_order_invariance(simple_var_data):
    """Test that Generalized FEVD is invariant to variable ordering."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Default order
    fevd_order1 = result.fevd(periods=10, method="generalized")

    # Reversed order
    fevd_order2 = result.fevd(periods=10, method="generalized", order=["y2", "y1"])

    # Reorder fevd_order2 back to original order for comparison
    # fevd_order2 is in order [y2, y1], so:
    #   fevd_order2[:, 0, :] = FEVD for y2
    #   fevd_order2[:, 1, :] = FEVD for y1
    # And for sources:
    #   fevd_order2[:, :, 0] = contribution from y2
    #   fevd_order2[:, :, 1] = contribution from y1

    # Reorder to [y1, y2]:
    fevd_reordered = np.zeros_like(fevd_order2.decomposition)
    # Variable y1 (was index 1 in order2, now index 0)
    # from shock y1 (was index 1 in order2, now index 0)
    fevd_reordered[:, 0, 0] = fevd_order2.decomposition[:, 1, 1]  # y1 from y1
    # from shock y2 (was index 0 in order2, now index 1)
    fevd_reordered[:, 0, 1] = fevd_order2.decomposition[:, 1, 0]  # y1 from y2
    # Variable y2 (was index 0 in order2, now index 1)
    fevd_reordered[:, 1, 0] = fevd_order2.decomposition[:, 0, 1]  # y2 from y1
    fevd_reordered[:, 1, 1] = fevd_order2.decomposition[:, 0, 0]  # y2 from y2

    # GFEVD should be invariant to ordering
    assert np.allclose(
        fevd_order1.decomposition, fevd_reordered, atol=1e-6
    ), "Generalized FEVD should be invariant to variable ordering"


def test_fevd_cholesky_depends_on_order(simple_var_data):
    """Test that Cholesky FEVD depends on variable ordering."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Check that Sigma has non-zero off-diagonal
    assert abs(result.Sigma[0, 1]) > 0.1, "Need non-diagonal Sigma for this test"

    # Default order
    fevd_order1 = result.fevd(periods=10, method="cholesky")

    # Reversed order
    fevd_order2 = result.fevd(periods=10, method="cholesky", order=["y2", "y1"])

    # Cholesky FEVD should differ
    assert not np.allclose(
        fevd_order1.decomposition, fevd_order2.decomposition, atol=1e-6
    ), "Cholesky FEVD should depend on ordering"


def test_fevd_getitem(simple_var_data):
    """Test FEVDResult __getitem__ accessor."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    fevd_result = result.fevd(periods=10, method="cholesky")

    # Access FEVD for y1
    fevd_y1 = fevd_result["y1"]

    assert fevd_y1.shape == (11, 2)
    assert np.allclose(fevd_y1, fevd_result.decomposition[:, 0, :])


def test_fevd_to_dataframe(simple_var_data):
    """Test FEVDResult to_dataframe method."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    fevd_result = result.fevd(periods=10, method="cholesky")

    # DataFrame for single variable
    df_y1 = fevd_result.to_dataframe(variable="y1")
    assert df_y1.shape == (11, 3)  # horizon + 2 shocks

    # DataFrame for all variables
    df_all = fevd_result.to_dataframe()
    assert df_all.shape == (22, 4)  # 2 vars × 11 horizons, 4 columns


def test_fevd_summary(simple_var_data):
    """Test FEVDResult summary method."""
    df, A1_true, Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    fevd_result = result.fevd(periods=10, method="cholesky")

    # Get summary
    summary = fevd_result.summary()

    assert isinstance(summary, str)
    assert "Forecast Error Variance Decomposition" in summary
    assert "cholesky" in summary.lower()


def test_fevd_invalid_method():
    """Test that invalid method raises error."""
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

    with pytest.raises(ValueError, match="Unknown method"):
        result.fevd(periods=10, method="invalid")


def test_fevd_manual_validation():
    """
    Manually validate FEVD computation for simple diagonal case.

    For diagonal Sigma with no correlations, each variable should be
    100% explained by itself at all horizons (in the limit).
    """
    np.random.seed(123)

    # Simple diagonal case
    A1 = np.array([[0.3, 0.0], [0.0, 0.4]])  # Diagonal dynamics
    Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])  # Diagonal covariance

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

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Check that estimated A and Sigma are approximately diagonal
    assert abs(result.A_matrices[0][0, 1]) < 0.1, "A should be approximately diagonal"
    assert abs(result.A_matrices[0][1, 0]) < 0.1, "A should be approximately diagonal"
    assert abs(result.Sigma[0, 1]) < 0.15, "Sigma should be approximately diagonal"

    # Compute FEVD
    fevd_result = result.fevd(periods=20, method="cholesky")

    # For diagonal system, each variable should be mostly self-explained
    # At h=0, first variable is 100% self-explained (Cholesky property)
    assert fevd_result.decomposition[0, 0, 0] > 0.99

    # At longer horizons, still mostly self-explained (but with noise due to estimation)
    # y1 explained by itself
    assert fevd_result.decomposition[20, 0, 0] > 0.85
    # y2 explained by itself
    assert fevd_result.decomposition[20, 1, 1] > 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
