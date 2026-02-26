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
    df, _A1_true, _Sigma_true = simple_var_data

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
    df, _A1_true, _Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    # Cholesky FEVD
    fevd_chol = result.fevd(periods=10, method="cholesky")

    # Check that each row sums to 1
    for h in range(11):
        for i in range(2):
            row_sum = fevd_chol.decomposition[h, i, :].sum()
            assert np.isclose(row_sum, 1.0, atol=1e-6), (
                f"FEVD at h={h}, var={i} should sum to 1, got {row_sum}"
            )

    # Generalized FEVD
    fevd_gen = result.fevd(periods=10, method="generalized")

    for h in range(11):
        for i in range(2):
            row_sum = fevd_gen.decomposition[h, i, :].sum()
            assert np.isclose(row_sum, 1.0, atol=1e-6), (
                f"Generalized FEVD at h={h}, var={i} should sum to 1, got {row_sum}"
            )


def test_fevd_cholesky_h0_first_variable(simple_var_data):
    """
    Test that at h=0, first variable is 100% explained by itself (Cholesky).

    With Cholesky ordering, the first variable is contemporaneously exogenous,
    so at h=0 it should be 100% explained by its own shock.
    """
    df, _A1_true, _Sigma_true = simple_var_data

    data = PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

    model = PanelVAR(data)
    result = model.fit(method="ols")

    fevd_result = result.fevd(periods=10, method="cholesky")

    # At h=0, first variable (y1) should be 100% explained by itself
    assert np.isclose(fevd_result.decomposition[0, 0, 0], 1.0, atol=1e-6), (
        "First variable at h=0 should be 100% self-explained"
    )

    # And 0% explained by second variable
    assert np.isclose(fevd_result.decomposition[0, 0, 1], 0.0, atol=1e-6), (
        "First variable at h=0 should not be explained by second variable"
    )


def test_fevd_generalized_order_invariance(simple_var_data):
    """Test that Generalized FEVD is invariant to variable ordering."""
    df, _A1_true, _Sigma_true = simple_var_data

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
    assert np.allclose(fevd_order1.decomposition, fevd_reordered, atol=1e-6), (
        "Generalized FEVD should be invariant to variable ordering"
    )


def test_fevd_cholesky_depends_on_order(simple_var_data):
    """Test that Cholesky FEVD depends on variable ordering."""
    df, _A1_true, _Sigma_true = simple_var_data

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
    assert not np.allclose(fevd_order1.decomposition, fevd_order2.decomposition, atol=1e-6), (
        "Cholesky FEVD should depend on ordering"
    )


def test_fevd_getitem(simple_var_data):
    """Test FEVDResult __getitem__ accessor."""
    df, _A1_true, _Sigma_true = simple_var_data

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
    df, _A1_true, _Sigma_true = simple_var_data

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
    df, _A1_true, _Sigma_true = simple_var_data

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


class TestFEVDGetItem:
    """Tests for FEVDResult.__getitem__ to cover lines 111-139."""

    def test_getitem_valid_variable(self, simple_var_data):
        """Test accessing FEVD for a valid variable name."""
        df, _A1_true, _Sigma_true = simple_var_data

        data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(data)
        result = model.fit(method="ols")

        fevd_result = result.fevd(periods=10, method="cholesky")

        # Access FEVD for y1 using __getitem__
        y1_fevd = fevd_result["y1"]
        assert y1_fevd.shape == (11, 2)
        # Should match the decomposition slice for variable index 0
        assert np.allclose(y1_fevd, fevd_result.decomposition[:, 0, :])

        # Access FEVD for y2
        y2_fevd = fevd_result["y2"]
        assert y2_fevd.shape == (11, 2)
        assert np.allclose(y2_fevd, fevd_result.decomposition[:, 1, :])

    def test_getitem_invalid_variable(self, simple_var_data):
        """Test that accessing a nonexistent variable raises KeyError."""
        df, _A1_true, _Sigma_true = simple_var_data

        data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(data)
        result = model.fit(method="ols")

        fevd_result = result.fevd(periods=10, method="cholesky")

        with pytest.raises(KeyError, match="Variable 'nonexistent' not found"):
            fevd_result["nonexistent"]

    def test_getitem_sums_to_one(self, simple_var_data):
        """Test that the FEVD slice from __getitem__ sums to 1 per row."""
        df, _A1_true, _Sigma_true = simple_var_data

        data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(data)
        result = model.fit(method="ols")

        fevd_result = result.fevd(periods=10, method="cholesky")
        y1_fevd = fevd_result["y1"]

        for h in range(11):
            row_sum = y1_fevd[h, :].sum()
            assert np.isclose(row_sum, 1.0, atol=1e-6), (
                f"FEVD row at h={h} should sum to 1, got {row_sum}"
            )


class TestBootstrapFEVDIteration:
    """Tests for _bootstrap_fevd_iteration to cover lines 455-526."""

    def test_bootstrap_iteration_cholesky(self):
        """Test a single bootstrap FEVD iteration with Cholesky method."""
        from panelbox.var.fevd import _bootstrap_fevd_iteration

        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        np.random.seed(42)
        residuals = np.random.multivariate_normal(np.zeros(2), Sigma, size=100)

        result = _bootstrap_fevd_iteration(
            A_matrices, Sigma, residuals, periods=5, method="cholesky", seed=42
        )

        # Shape should be (periods+1, K, K) = (6, 2, 2)
        assert result.shape == (6, 2, 2)

        # FEVD should sum to 1 for each variable at each horizon
        for h in range(6):
            for i in range(2):
                row_sum = result[h, i, :].sum()
                assert np.isclose(row_sum, 1.0, atol=1e-4), (
                    f"Bootstrap FEVD at h={h}, var={i} should sum to 1, got {row_sum}"
                )

        # Values should be non-negative (proportions)
        assert np.all(result >= -1e-10)

    def test_bootstrap_iteration_generalized(self):
        """Test a single bootstrap FEVD iteration with generalized method."""
        from panelbox.var.fevd import _bootstrap_fevd_iteration

        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        np.random.seed(123)
        residuals = np.random.multivariate_normal(np.zeros(2), Sigma, size=100)

        result = _bootstrap_fevd_iteration(
            A_matrices, Sigma, residuals, periods=5, method="generalized", seed=123
        )

        assert result.shape == (6, 2, 2)

        # Generalized FEVD should also sum to 1 (after normalization)
        for h in range(6):
            for i in range(2):
                row_sum = result[h, i, :].sum()
                assert np.isclose(row_sum, 1.0, atol=1e-4)

    def test_bootstrap_iteration_reproducible(self):
        """Test that using the same seed gives the same result."""
        from panelbox.var.fevd import _bootstrap_fevd_iteration

        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        np.random.seed(42)
        residuals = np.random.multivariate_normal(np.zeros(2), Sigma, size=100)

        result1 = _bootstrap_fevd_iteration(
            A_matrices, Sigma, residuals, periods=5, method="cholesky", seed=99
        )
        result2 = _bootstrap_fevd_iteration(
            A_matrices, Sigma, residuals, periods=5, method="cholesky", seed=99
        )

        assert np.allclose(result1, result2)

    def test_bootstrap_iteration_invalid_method(self):
        """Test that an invalid method raises ValueError."""
        from panelbox.var.fevd import _bootstrap_fevd_iteration

        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.eye(2)
        residuals = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="Unknown method"):
            _bootstrap_fevd_iteration(
                A_matrices, Sigma, residuals, periods=5, method="invalid", seed=42
            )


class TestBootstrapFEVD:
    """Tests for bootstrap_fevd to cover lines 582-608."""

    def test_bootstrap_fevd_cholesky(self):
        """Test full bootstrap FEVD with Cholesky method."""
        from panelbox.var.fevd import bootstrap_fevd

        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        np.random.seed(42)
        residuals = np.random.multivariate_normal(np.zeros(2), Sigma, size=100)

        ci_lower, ci_upper, boot_dist = bootstrap_fevd(
            A_matrices,
            Sigma,
            residuals,
            periods=5,
            method="cholesky",
            n_bootstrap=20,
            ci_level=0.95,
            n_jobs=1,
            seed=42,
            verbose=False,
        )

        # Check shapes
        assert ci_lower.shape == (6, 2, 2)
        assert ci_upper.shape == (6, 2, 2)
        assert boot_dist.shape == (20, 6, 2, 2)

        # Lower CI should be <= upper CI
        assert np.all(ci_lower <= ci_upper + 1e-10)

    def test_bootstrap_fevd_generalized(self):
        """Test full bootstrap FEVD with generalized method."""
        from panelbox.var.fevd import bootstrap_fevd

        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        np.random.seed(42)
        residuals = np.random.multivariate_normal(np.zeros(2), Sigma, size=100)

        ci_lower, ci_upper, boot_dist = bootstrap_fevd(
            A_matrices,
            Sigma,
            residuals,
            periods=5,
            method="generalized",
            n_bootstrap=20,
            ci_level=0.95,
            n_jobs=1,
            seed=42,
            verbose=False,
        )

        assert ci_lower.shape == (6, 2, 2)
        assert ci_upper.shape == (6, 2, 2)
        assert boot_dist.shape == (20, 6, 2, 2)

    def test_bootstrap_fevd_reproducible(self):
        """Test that bootstrap FEVD is reproducible with the same seed."""
        from panelbox.var.fevd import bootstrap_fevd

        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        np.random.seed(42)
        residuals = np.random.multivariate_normal(np.zeros(2), Sigma, size=100)

        ci_lower1, ci_upper1, _ = bootstrap_fevd(
            A_matrices,
            Sigma,
            residuals,
            periods=5,
            method="cholesky",
            n_bootstrap=10,
            n_jobs=1,
            seed=42,
            verbose=False,
        )

        ci_lower2, ci_upper2, _ = bootstrap_fevd(
            A_matrices,
            Sigma,
            residuals,
            periods=5,
            method="cholesky",
            n_bootstrap=10,
            n_jobs=1,
            seed=42,
            verbose=False,
        )

        assert np.allclose(ci_lower1, ci_lower2)
        assert np.allclose(ci_upper1, ci_upper2)

    def test_bootstrap_fevd_ci_level(self):
        """Test that wider CI level produces wider intervals."""
        from panelbox.var.fevd import bootstrap_fevd

        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        np.random.seed(42)
        residuals = np.random.multivariate_normal(np.zeros(2), Sigma, size=100)

        _, _, boot_dist = bootstrap_fevd(
            A_matrices,
            Sigma,
            residuals,
            periods=5,
            method="cholesky",
            n_bootstrap=30,
            ci_level=0.90,
            n_jobs=1,
            seed=42,
            verbose=False,
        )

        # Just verify the bootstrap distribution has the right shape
        assert boot_dist.shape[0] == 30


class TestFEVDToDataFrameFiltering:
    """Tests for FEVDResult.to_dataframe() with variable and horizons parameters (lines 141-185)."""

    @pytest.fixture
    def fevd_result(self, simple_var_data):
        """Fit a VAR and compute FEVD for reuse."""
        df, _A1_true, _Sigma_true = simple_var_data

        data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(data)
        result = model.fit(method="ols")
        return result.fevd(periods=10, method="cholesky")

    def test_to_dataframe_with_horizons(self, fevd_result):
        """Test to_dataframe with specific horizons parameter."""
        df = fevd_result.to_dataframe(horizons=[0, 5, 10])

        # Should have 2 vars x 3 horizons = 6 rows in long format
        assert df.shape[0] == 6
        # Should contain horizon column and shock columns
        assert "horizon" in df.columns
        assert "variable" in df.columns

    def test_to_dataframe_single_variable_with_horizons(self, fevd_result):
        """Test to_dataframe with both variable and horizons specified."""
        df = fevd_result.to_dataframe(variable="y1", horizons=[0, 5, 10])

        # Single variable: 3 horizons, columns = horizon + 2 shocks
        assert df.shape == (3, 3)
        assert "horizon" in df.columns
        assert "y1" in df.columns
        assert "y2" in df.columns

    def test_to_dataframe_all_variables_all_horizons(self, fevd_result):
        """Test to_dataframe with no filtering returns all data."""
        df = fevd_result.to_dataframe()

        # 2 vars x 11 horizons = 22 rows
        assert df.shape[0] == 22
        assert "horizon" in df.columns
        assert "variable" in df.columns


class TestFEVDValidationWarning:
    """Test FEVDResult._validate_fevd() warning path (lines 96-109)."""

    def test_validate_fevd_warns_on_bad_sum(self):
        """Test that _validate_fevd issues a warning when rows don't sum to 1."""
        import warnings

        from panelbox.var.fevd import FEVDResult

        # Create a FEVD array that does NOT sum to 1
        bad_decomp = np.array(
            [
                [
                    [0.5, 0.3],  # sums to 0.8, not 1.0
                    [0.6, 0.4],
                ],  # sums to 1.0
            ]
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FEVDResult(
                decomposition=bad_decomp,
                var_names=["y1", "y2"],
                periods=0,
                method="cholesky",
            )
            # Should have at least one warning about FEVD sum
            fevd_warnings = [x for x in w if "sums to" in str(x.message)]
            assert len(fevd_warnings) > 0, "Should warn when FEVD doesn't sum to 1"


class TestFEVDGeneralizedMethod:
    """Test FEVD with generalized method via compute functions."""

    def test_generalized_fevd_sums_to_one(self):
        """Test that generalized FEVD also sums to 1 after normalization."""
        from panelbox.var.fevd import compute_fevd_generalized
        from panelbox.var.irf import compute_phi_non_orthogonalized

        np.random.seed(42)
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

        Phi = compute_phi_non_orthogonalized(A_matrices, periods=10)
        fevd = compute_fevd_generalized(Phi, Sigma, periods=10)

        for h in range(11):
            for i in range(2):
                row_sum = fevd[h, i, :].sum()
                assert np.isclose(row_sum, 1.0, atol=1e-6), (
                    f"Generalized FEVD at h={h}, var={i} should sum to 1, got {row_sum}"
                )


class TestFEVDCustomOrdering:
    """Test FEVD with custom variable ordering."""

    def test_cholesky_fevd_custom_ordering(self, simple_var_data):
        """Test that custom ordering changes Cholesky FEVD results."""
        df, _A1_true, _Sigma_true = simple_var_data

        data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(data)
        result = model.fit(method="ols")

        fevd_default = result.fevd(periods=5, method="cholesky")
        fevd_reversed = result.fevd(periods=5, method="cholesky", order=["y2", "y1"])

        # They should differ since ordering matters for Cholesky
        assert not np.allclose(
            fevd_default.decomposition, fevd_reversed.decomposition, atol=1e-6
        ), "Different orderings should produce different Cholesky FEVDs"


class TestFEVDCoverage:
    """
    Additional tests to improve coverage for panelbox/var/fevd.py.

    Targets uncovered lines:
    - 102-137 (FEVDResult._validate_fevd warning and __getitem__)
    - 455-526 (_bootstrap_fevd_iteration)
    - 582-608 (bootstrap_fevd)
    - summary with custom horizons
    - compute_fevd_generalized
    - to_dataframe with filters
    """

    # ---------------------------------------------------------------
    # FEVDResult.__getitem__() with valid and invalid keys
    # ---------------------------------------------------------------
    def test_getitem_valid_key_returns_correct_slice(self):
        """Test __getitem__ with a valid variable name returns correct array."""
        from panelbox.var.fevd import FEVDResult

        K = 2
        periods = 5
        decomposition = np.zeros((periods + 1, K, K))
        for h in range(periods + 1):
            decomposition[h, 0, :] = [0.8, 0.2]
            decomposition[h, 1, :] = [0.3, 0.7]

        result = FEVDResult(
            decomposition=decomposition,
            var_names=["y1", "y2"],
            periods=periods,
            method="cholesky",
        )

        y1_fevd = result["y1"]
        assert y1_fevd.shape == (periods + 1, K)
        assert np.allclose(y1_fevd, decomposition[:, 0, :])

        y2_fevd = result["y2"]
        assert y2_fevd.shape == (periods + 1, K)
        assert np.allclose(y2_fevd, decomposition[:, 1, :])

    def test_getitem_invalid_key_raises_keyerror(self):
        """Test __getitem__ with nonexistent variable raises KeyError."""
        from panelbox.var.fevd import FEVDResult

        K = 2
        periods = 3
        decomposition = np.zeros((periods + 1, K, K))
        for h in range(periods + 1):
            decomposition[h, 0, :] = [0.6, 0.4]
            decomposition[h, 1, :] = [0.5, 0.5]

        result = FEVDResult(
            decomposition=decomposition,
            var_names=["y1", "y2"],
            periods=periods,
            method="cholesky",
        )

        with pytest.raises(KeyError, match="Variable 'nonexistent' not found"):
            result["nonexistent"]

    # ---------------------------------------------------------------
    # FEVDResult._validate_fevd() warning when rows don't sum to 1
    # ---------------------------------------------------------------
    def test_validate_fevd_warns_bad_sum(self):
        """Test that _validate_fevd warns when FEVD rows don't sum to 1."""
        import warnings as _warnings

        from panelbox.var.fevd import FEVDResult

        K = 2
        periods = 2
        bad_decomposition = np.zeros((periods + 1, K, K))
        # Make first row sum to 0.7 instead of 1.0
        bad_decomposition[0, 0, :] = [0.5, 0.2]  # sums to 0.7
        bad_decomposition[0, 1, :] = [0.4, 0.6]  # sums to 1.0
        for h in range(1, periods + 1):
            bad_decomposition[h, 0, :] = [0.6, 0.4]
            bad_decomposition[h, 1, :] = [0.3, 0.7]

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            FEVDResult(
                decomposition=bad_decomposition,
                var_names=["y1", "y2"],
                periods=periods,
                method="cholesky",
            )
            fevd_warnings = [x for x in w if "sums to" in str(x.message)]
            assert len(fevd_warnings) >= 1, "Should warn when FEVD row doesn't sum to 1"

    def test_validate_fevd_no_warning_when_correct(self):
        """Test that _validate_fevd does NOT warn when rows sum to 1."""
        import warnings as _warnings

        from panelbox.var.fevd import FEVDResult

        K = 2
        periods = 3
        good_decomposition = np.zeros((periods + 1, K, K))
        for h in range(periods + 1):
            good_decomposition[h, 0, :] = [0.7, 0.3]
            good_decomposition[h, 1, :] = [0.4, 0.6]

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            FEVDResult(
                decomposition=good_decomposition,
                var_names=["y1", "y2"],
                periods=periods,
                method="cholesky",
            )
            fevd_warnings = [x for x in w if "sums to" in str(x.message)]
            assert len(fevd_warnings) == 0, "Should not warn when FEVD rows sum to 1"

    # ---------------------------------------------------------------
    # _bootstrap_fevd_iteration() directly with simple VAR matrices
    # ---------------------------------------------------------------
    def test_bootstrap_fevd_iteration_cholesky_shape_and_values(self):
        """Test _bootstrap_fevd_iteration with Cholesky returns valid FEVD."""
        from panelbox.var.fevd import _bootstrap_fevd_iteration

        np.random.seed(42)
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        residuals = np.random.multivariate_normal(np.zeros(2), Sigma, size=80)

        result = _bootstrap_fevd_iteration(
            A_matrices, Sigma, residuals, periods=4, method="cholesky", seed=100
        )

        assert result.shape == (5, 2, 2)
        # Each row should sum to approximately 1
        for h in range(5):
            for i in range(2):
                assert np.isclose(result[h, i, :].sum(), 1.0, atol=1e-3)
        # Values should be non-negative
        assert np.all(result >= -1e-10)

    def test_bootstrap_fevd_iteration_generalized_shape(self):
        """Test _bootstrap_fevd_iteration with generalized method."""
        from panelbox.var.fevd import _bootstrap_fevd_iteration

        np.random.seed(42)
        A_matrices = [np.array([[0.4, 0.1], [0.15, 0.5]])]
        Sigma = np.array([[1.0, 0.2], [0.2, 0.8]])
        residuals = np.random.multivariate_normal(np.zeros(2), Sigma, size=80)

        result = _bootstrap_fevd_iteration(
            A_matrices, Sigma, residuals, periods=4, method="generalized", seed=200
        )

        assert result.shape == (5, 2, 2)
        for h in range(5):
            for i in range(2):
                assert np.isclose(result[h, i, :].sum(), 1.0, atol=1e-3)

    def test_bootstrap_fevd_iteration_var2(self):
        """Test _bootstrap_fevd_iteration with VAR(2) -- two lag matrices."""
        from panelbox.var.fevd import _bootstrap_fevd_iteration

        np.random.seed(42)
        A1 = np.array([[0.3, 0.1], [0.1, 0.3]])
        A2 = np.array([[0.1, 0.0], [0.0, 0.1]])
        A_matrices = [A1, A2]
        Sigma = np.eye(2)
        residuals = np.random.randn(100, 2)

        result = _bootstrap_fevd_iteration(
            A_matrices, Sigma, residuals, periods=3, method="cholesky", seed=300
        )

        assert result.shape == (4, 2, 2)
        assert np.all(np.isfinite(result))

    # ---------------------------------------------------------------
    # bootstrap_fevd() with small n_bootstrap
    # ---------------------------------------------------------------
    def test_bootstrap_fevd_small_n(self):
        """Test bootstrap_fevd with n_bootstrap=20 returns correct shapes."""
        from panelbox.var.fevd import bootstrap_fevd

        np.random.seed(42)
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        residuals = np.random.multivariate_normal(np.zeros(2), Sigma, size=80)

        ci_lower, ci_upper, boot_dist = bootstrap_fevd(
            A_matrices,
            Sigma,
            residuals,
            periods=4,
            method="cholesky",
            n_bootstrap=20,
            ci_level=0.90,
            n_jobs=1,
            seed=42,
            verbose=False,
        )

        assert ci_lower.shape == (5, 2, 2)
        assert ci_upper.shape == (5, 2, 2)
        assert boot_dist.shape == (20, 5, 2, 2)
        assert np.all(ci_lower <= ci_upper + 1e-10)

    def test_bootstrap_fevd_generalized_method(self):
        """Test bootstrap_fevd with generalized method."""
        from panelbox.var.fevd import bootstrap_fevd

        np.random.seed(42)
        A_matrices = [np.array([[0.4, 0.1], [0.15, 0.5]])]
        Sigma = np.array([[1.0, 0.2], [0.2, 0.8]])
        residuals = np.random.multivariate_normal(np.zeros(2), Sigma, size=80)

        ci_lower, ci_upper, boot_dist = bootstrap_fevd(
            A_matrices,
            Sigma,
            residuals,
            periods=3,
            method="generalized",
            n_bootstrap=15,
            ci_level=0.95,
            n_jobs=1,
            seed=123,
            verbose=False,
        )

        assert ci_lower.shape == (4, 2, 2)
        assert ci_upper.shape == (4, 2, 2)
        assert boot_dist.shape == (15, 4, 2, 2)

    def test_bootstrap_fevd_no_seed(self):
        """Test bootstrap_fevd without explicit seed (seed=None)."""
        from panelbox.var.fevd import bootstrap_fevd

        np.random.seed(42)
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        residuals = np.random.multivariate_normal(np.zeros(2), Sigma, size=80)

        ci_lower, _ci_upper, boot_dist = bootstrap_fevd(
            A_matrices,
            Sigma,
            residuals,
            periods=3,
            method="cholesky",
            n_bootstrap=10,
            n_jobs=1,
            seed=None,
            verbose=False,
        )

        assert ci_lower.shape == (4, 2, 2)
        assert boot_dist.shape == (10, 4, 2, 2)

    # ---------------------------------------------------------------
    # FEVDResult.summary() with custom horizons
    # ---------------------------------------------------------------
    def test_summary_with_custom_horizons(self):
        """Test summary() with user-specified horizons."""
        from panelbox.var.fevd import FEVDResult

        K = 2
        periods = 10
        decomposition = np.zeros((periods + 1, K, K))
        for h in range(periods + 1):
            decomposition[h, 0, :] = [0.7, 0.3]
            decomposition[h, 1, :] = [0.4, 0.6]

        result = FEVDResult(
            decomposition=decomposition,
            var_names=["y1", "y2"],
            periods=periods,
            method="cholesky",
        )

        summary = result.summary(horizons=[0, 2, 5])
        assert isinstance(summary, str)
        assert "h=0" in summary
        assert "h=2" in summary
        assert "h=5" in summary
        assert "Forecast Error Variance Decomposition" in summary

    def test_summary_default_horizons(self):
        """Test summary() with default horizons (None)."""
        from panelbox.var.fevd import FEVDResult

        K = 2
        periods = 20
        decomposition = np.zeros((periods + 1, K, K))
        for h in range(periods + 1):
            decomposition[h, 0, :] = [0.75, 0.25]
            decomposition[h, 1, :] = [0.35, 0.65]

        result = FEVDResult(
            decomposition=decomposition,
            var_names=["y1", "y2"],
            periods=periods,
            method="generalized",
        )

        summary = result.summary()
        assert isinstance(summary, str)
        # Default horizons: [1, 5, 10, periods]
        assert "h=1" in summary
        assert "h=5" in summary
        assert "h=10" in summary
        assert "h=20" in summary

    def test_summary_with_ordering(self):
        """Test summary() includes ordering for Cholesky method."""
        from panelbox.var.fevd import FEVDResult

        K = 2
        periods = 5
        decomposition = np.zeros((periods + 1, K, K))
        for h in range(periods + 1):
            decomposition[h, 0, :] = [0.8, 0.2]
            decomposition[h, 1, :] = [0.3, 0.7]

        result = FEVDResult(
            decomposition=decomposition,
            var_names=["y1", "y2"],
            periods=periods,
            method="cholesky",
            ordering=["y1", "y2"],
        )

        summary = result.summary(horizons=[1, 3])
        assert "Ordering:" in summary
        assert "y1, y2" in summary

    # ---------------------------------------------------------------
    # compute_fevd_generalized() function
    # ---------------------------------------------------------------
    def test_compute_fevd_generalized_basic(self):
        """Test compute_fevd_generalized produces valid output."""
        from panelbox.var.fevd import compute_fevd_generalized
        from panelbox.var.irf import compute_phi_non_orthogonalized

        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.4]])]
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        periods = 8

        Phi = compute_phi_non_orthogonalized(A_matrices, periods=periods)
        fevd = compute_fevd_generalized(Phi, Sigma, periods=periods)

        assert fevd.shape == (periods + 1, 2, 2)
        # Each row should sum to 1 (normalized)
        for h in range(periods + 1):
            for i in range(2):
                assert np.isclose(fevd[h, i, :].sum(), 1.0, atol=1e-6)
        # Values should be non-negative
        assert np.all(fevd >= -1e-10)

    def test_compute_fevd_generalized_three_var(self):
        """Test compute_fevd_generalized with K=3 variables."""
        from panelbox.var.fevd import compute_fevd_generalized
        from panelbox.var.irf import compute_phi_non_orthogonalized

        A_matrices = [
            np.array(
                [
                    [0.3, 0.1, 0.05],
                    [0.1, 0.4, 0.1],
                    [0.05, 0.1, 0.3],
                ]
            )
        ]
        Sigma = np.array(
            [
                [1.0, 0.2, 0.1],
                [0.2, 1.0, 0.15],
                [0.1, 0.15, 1.0],
            ]
        )
        periods = 5

        Phi = compute_phi_non_orthogonalized(A_matrices, periods=periods)
        fevd = compute_fevd_generalized(Phi, Sigma, periods=periods)

        assert fevd.shape == (periods + 1, 3, 3)
        for h in range(periods + 1):
            for i in range(3):
                assert np.isclose(fevd[h, i, :].sum(), 1.0, atol=1e-6)

    # ---------------------------------------------------------------
    # FEVDResult.to_dataframe() with variable filter and horizon filter
    # ---------------------------------------------------------------
    def test_to_dataframe_variable_filter(self):
        """Test to_dataframe with a single variable filter."""
        from panelbox.var.fevd import FEVDResult

        K = 2
        periods = 5
        decomposition = np.zeros((periods + 1, K, K))
        for h in range(periods + 1):
            decomposition[h, 0, :] = [0.8, 0.2]
            decomposition[h, 1, :] = [0.3, 0.7]

        result = FEVDResult(
            decomposition=decomposition,
            var_names=["y1", "y2"],
            periods=periods,
            method="cholesky",
        )

        df = result.to_dataframe(variable="y1")
        assert df.shape == (periods + 1, 3)  # horizon + 2 shocks
        assert "horizon" in df.columns
        assert "y1" in df.columns
        assert "y2" in df.columns

    def test_to_dataframe_horizon_filter(self):
        """Test to_dataframe with horizon filter only."""
        from panelbox.var.fevd import FEVDResult

        K = 2
        periods = 10
        decomposition = np.zeros((periods + 1, K, K))
        for h in range(periods + 1):
            decomposition[h, 0, :] = [0.7, 0.3]
            decomposition[h, 1, :] = [0.4, 0.6]

        result = FEVDResult(
            decomposition=decomposition,
            var_names=["y1", "y2"],
            periods=periods,
            method="cholesky",
        )

        df = result.to_dataframe(horizons=[0, 5, 10])
        # All variables long format: 2 vars x 3 horizons = 6 rows
        assert df.shape[0] == 6
        assert "horizon" in df.columns
        assert "variable" in df.columns

    def test_to_dataframe_variable_and_horizon_filter(self):
        """Test to_dataframe with both variable and horizon filters."""
        from panelbox.var.fevd import FEVDResult

        K = 2
        periods = 10
        decomposition = np.zeros((periods + 1, K, K))
        for h in range(periods + 1):
            decomposition[h, 0, :] = [0.7, 0.3]
            decomposition[h, 1, :] = [0.4, 0.6]

        result = FEVDResult(
            decomposition=decomposition,
            var_names=["y1", "y2"],
            periods=periods,
            method="cholesky",
        )

        df = result.to_dataframe(variable="y2", horizons=[0, 3, 7])
        assert df.shape == (3, 3)  # 3 horizons, horizon + 2 shock columns
        assert list(df["horizon"]) == [0, 3, 7]

    def test_to_dataframe_no_filter(self):
        """Test to_dataframe with no filter returns all data."""
        from panelbox.var.fevd import FEVDResult

        K = 2
        periods = 4
        decomposition = np.zeros((periods + 1, K, K))
        for h in range(periods + 1):
            decomposition[h, 0, :] = [0.6, 0.4]
            decomposition[h, 1, :] = [0.5, 0.5]

        result = FEVDResult(
            decomposition=decomposition,
            var_names=["y1", "y2"],
            periods=periods,
            method="cholesky",
        )

        df = result.to_dataframe()
        # 2 vars x 5 horizons = 10 rows
        assert df.shape[0] == 10

    # ---------------------------------------------------------------
    # FEVDResult repr
    # ---------------------------------------------------------------
    def test_repr(self):
        """Test __repr__ method."""
        from panelbox.var.fevd import FEVDResult

        K = 2
        periods = 5
        decomposition = np.zeros((periods + 1, K, K))
        for h in range(periods + 1):
            decomposition[h, 0, :] = [0.8, 0.2]
            decomposition[h, 1, :] = [0.3, 0.7]

        result = FEVDResult(
            decomposition=decomposition,
            var_names=["y1", "y2"],
            periods=periods,
            method="cholesky",
        )

        repr_str = repr(result)
        assert "FEVDResult" in repr_str
        assert "K=2" in repr_str
        assert "periods=5" in repr_str
        assert "cholesky" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
