"""
Tests for spatial diagnostics module.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.diagnostics.spatial_tests import LocalMoranI
from panelbox.validation.spatial import (
    LMErrorTest,
    LMLagTest,
    MoranIPanelTest,
    RobustLMErrorTest,
    RobustLMLagTest,
    SpatialHausmanTest,
    run_lm_tests,
)
from panelbox.validation.spatial.utils import standardize_spatial_weights, validate_spatial_weights


def create_spatial_weights(N: int, style: str = "queen", row_standardize: bool = True):
    """Create a simple spatial weights matrix."""
    # Create queen contiguity for a regular grid
    grid_size = int(np.sqrt(N))
    if grid_size**2 != N:
        # Fall back to simple nearest neighbor
        W = np.zeros((N, N))
        for i in range(N):
            # Connect to immediate neighbors
            if i > 0:
                W[i, i - 1] = 1
            if i < N - 1:
                W[i, i + 1] = 1
    else:
        # Create queen contiguity for grid
        W = np.zeros((N, N))
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                # Right neighbor
                if j < grid_size - 1:
                    W[idx, idx + 1] = 1
                # Bottom neighbor
                if i < grid_size - 1:
                    W[idx, idx + grid_size] = 1
                # Left neighbor
                if j > 0:
                    W[idx, idx - 1] = 1
                # Top neighbor
                if i > 0:
                    W[idx, idx - grid_size] = 1

    # Ensure symmetry
    W = (W + W.T) / 2

    # Row standardize if requested
    if row_standardize:
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        W = W / row_sums[:, np.newaxis]

    return W


def generate_spatial_panel_data(
    N: int = 25, T: int = 10, rho: float = 0.0, lambda_: float = 0.0, seed: int = 42
):
    """
    Generate panel data with spatial dependence.

    Parameters
    ----------
    N : int
        Number of spatial units
    T : int
        Number of time periods
    rho : float
        Spatial lag parameter (SAR)
    lambda_ : float
        Spatial error parameter (SEM)
    seed : int
        Random seed

    Returns
    -------
    dict
        Dictionary with data, W matrix, and true parameters
    """
    np.random.seed(seed)

    # Create spatial weights
    W = create_spatial_weights(N, row_standardize=True)

    # Generate exogenous variables
    X = np.random.randn(N * T, 3)
    beta = np.array([1.0, -0.5, 0.3])

    # Generate data for each period
    y_list = []
    e_list = []

    for t in range(T):
        # Extract X for this period
        X_t = X[t * N : (t + 1) * N]

        # Generate spatial error if lambda != 0
        if lambda_ != 0:
            # u = (I - λW)^{-1} ε
            epsilon_t = np.random.randn(N)
            I_lambdaW_inv = np.linalg.inv(np.eye(N) - lambda_ * W)
            u_t = I_lambdaW_inv @ epsilon_t
        else:
            u_t = np.random.randn(N)

        # Generate y with spatial lag if rho != 0
        Xbeta_t = X_t @ beta

        if rho != 0:
            # y = (I - ρW)^{-1} (Xβ + u)
            I_rhoW_inv = np.linalg.inv(np.eye(N) - rho * W)
            y_t = I_rhoW_inv @ (Xbeta_t + u_t)
        else:
            y_t = Xbeta_t + u_t

        y_list.append(y_t)
        e_list.append(u_t)

    # Combine all periods
    y = np.concatenate(y_list)
    e = np.concatenate(e_list)

    # Create entity and time indices
    entity_index = np.repeat(range(N), T)
    time_index = np.tile(range(T), N)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "entity": entity_index,
            "time": time_index,
            "y": y,
            "x1": X[:, 0],
            "x2": X[:, 1],
            "x3": X[:, 2],
        }
    )

    return {
        "data": data,
        "W": W,
        "residuals": e,
        "entity_index": entity_index,
        "time_index": time_index,
        "true_rho": rho,
        "true_lambda": lambda_,
        "true_beta": beta,
        "N": N,
        "T": T,
    }


class TestMoransI:
    """Tests for Moran's I."""

    def test_no_spatial_correlation(self):
        """Test Moran's I with no spatial autocorrelation."""
        # Generate data without spatial correlation
        data_dict = generate_spatial_panel_data(N=25, T=10, rho=0.0, lambda_=0.0)

        # Run test
        test = MoranIPanelTest(
            residuals=data_dict["residuals"],
            W=data_dict["W"],
            entity_index=data_dict["entity_index"],
            time_index=data_dict["time_index"],
            method="pooled",
        )
        result = test.run(alpha=0.05)

        # Should not reject H0 (p-value > 0.05 in most cases)
        # Allow for some Type I error
        assert result.pvalue > 0.01  # Very conservative
        assert abs(result.statistic - result.metadata["expected_value"]) < 0.2

    def test_positive_spatial_correlation(self):
        """Test Moran's I with positive spatial autocorrelation."""
        # Generate data with strong spatial lag
        data_dict = generate_spatial_panel_data(N=25, T=10, rho=0.7, lambda_=0.0)

        # Run test on y (which has spatial correlation)
        test = MoranIPanelTest(
            residuals=data_dict["data"]["y"].values,
            W=data_dict["W"],
            entity_index=data_dict["entity_index"],
            time_index=data_dict["time_index"],
            method="pooled",
        )
        result = test.run(alpha=0.05)

        # Should reject H0
        assert result.pvalue < 0.05
        assert result.statistic > result.metadata["expected_value"]

    def test_by_period_method(self):
        """Test Moran's I by period."""
        data_dict = generate_spatial_panel_data(N=25, T=5, rho=0.5, lambda_=0.0)

        test = MoranIPanelTest(
            residuals=data_dict["data"]["y"].values,
            W=data_dict["W"],
            entity_index=data_dict["entity_index"],
            time_index=data_dict["time_index"],
            method="period",
        )
        result = test.run()

        # Should return MoranIByPeriodResult
        assert hasattr(result, "results_by_period")
        assert len(result.results_by_period) == 5  # One for each period

        # Check summary DataFrame
        summary = result.summary()
        assert len(summary) == 5
        assert "statistic" in summary.columns
        assert "pvalue" in summary.columns

    def test_permutation_inference(self):
        """Test permutation-based inference."""
        data_dict = generate_spatial_panel_data(N=16, T=5, rho=0.6, lambda_=0.0)

        test = MoranIPanelTest(
            residuals=data_dict["data"]["y"].values,
            W=data_dict["W"],
            entity_index=data_dict["entity_index"],
            time_index=data_dict["time_index"],
            method="pooled",
        )

        # Run with permutation
        result_perm = test.run(inference="permutation", n_permutations=99)

        # Run with normal
        result_norm = test.run(inference="normal")

        # Both should detect spatial correlation (allow some variance)
        assert result_perm.pvalue < 0.15  # More lenient for permutation
        assert result_norm.pvalue < 0.15

        # Statistics should be identical
        assert abs(result_perm.statistic - result_norm.statistic) < 1e-10


@pytest.mark.skip(reason="LocalMoranI class not yet implemented")
class TestLocalMoranI:
    """Tests for Local Moran's I (LISA)."""

    def test_cluster_detection(self):
        """Test LISA cluster detection."""
        # Create data with clear spatial clusters
        N = 25
        T = 5
        np.random.seed(42)

        # Create values with spatial clusters
        values = np.random.randn(N * T)
        entity_index = np.repeat(range(N), T)
        time_index = np.tile(range(T), N)

        # Create a hot spot (high values cluster) in units 0-4
        for i in range(5):
            mask = entity_index == i
            values[mask] += 3.0

        # Create a cold spot (low values cluster) in units 20-24
        for i in range(20, 25):
            mask = entity_index == i
            values[mask] -= 3.0

        # Create weights matrix
        W = create_spatial_weights(N)

        # Run LISA
        lisa = LocalMoranI(
            variable=values, W=W, entity_index=entity_index, time_index=time_index, threshold=0.05
        )
        results = lisa.run(n_permutations=99, seed=42)

        # Check results structure
        assert isinstance(results, pd.DataFrame)
        assert "cluster_type" in results.columns
        assert "Ii" in results.columns
        assert "pvalue" in results.columns

        # Should detect some clusters
        cluster_counts = results["cluster_type"].value_counts()
        assert "HH" in cluster_counts.index or "LL" in cluster_counts.index

    def test_summary_statistics(self):
        """Test LISA summary statistics."""
        data_dict = generate_spatial_panel_data(N=16, T=3, rho=0.5)

        lisa = LocalMoranI(
            variable=data_dict["data"]["y"].values,
            W=data_dict["W"],
            entity_index=data_dict["entity_index"],
            time_index=data_dict["time_index"],
        )
        results = lisa.run(n_permutations=99)

        # Test summary method
        summary = lisa.summary(results)
        assert isinstance(summary, pd.DataFrame)
        assert "Count" in summary.columns
        assert "Percentage" in summary.columns


class TestLMTests:
    """Tests for LM spatial dependence tests."""

    def setup_method(self):
        """Create mock OLS result for testing."""
        # Generate data
        self.data_dict = generate_spatial_panel_data(N=25, T=10, rho=0.3, lambda_=0.0)

        # Create mock OLS result
        class MockOLSResult:
            def __init__(self, data_dict):
                # Simple OLS estimation
                X = data_dict["data"][["x1", "x2", "x3"]].values
                y = data_dict["data"]["y"].values

                # Add constant
                X_with_const = np.column_stack([np.ones(len(X)), X])

                # OLS estimates
                beta_hat = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                self.params = pd.Series(beta_hat, index=["const", "x1", "x2", "x3"])

                # Fitted values and residuals
                self.fittedvalues = X_with_const @ beta_hat
                self.resid = y - self.fittedvalues
                self.nobs = len(y)

                # For panel compatibility
                self.model = type(
                    "Model",
                    (),
                    {"N": data_dict["N"], "T": data_dict["T"], "data": type("Data", (), {"y": y})},
                )()

        self.ols_result = MockOLSResult(self.data_dict)

    def test_lm_lag(self):
        """Test LM-lag test."""
        test = LMLagTest(self.ols_result, self.data_dict["W"])
        result = test.run(alpha=0.05)

        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert 0 <= result.pvalue <= 1
        assert result.statistic >= 0

    def test_lm_error(self):
        """Test LM-error test."""
        test = LMErrorTest(self.ols_result, self.data_dict["W"])
        result = test.run(alpha=0.05)

        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert 0 <= result.pvalue <= 1
        assert result.statistic >= 0

    def test_robust_lm_lag(self):
        """Test Robust LM-lag test."""
        test = RobustLMLagTest(self.ols_result, self.data_dict["W"])
        result = test.run(alpha=0.05)

        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert 0 <= result.pvalue <= 1
        assert result.statistic >= 0

    def test_robust_lm_error(self):
        """Test Robust LM-error test."""
        test = RobustLMErrorTest(self.ols_result, self.data_dict["W"])
        result = test.run(alpha=0.05)

        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert 0 <= result.pvalue <= 1
        assert result.statistic >= 0

    def test_run_lm_tests(self):
        """Test complete LM test battery."""
        results = run_lm_tests(self.ols_result, self.data_dict["W"], verbose=False)

        # Check structure
        assert "lm_lag" in results
        assert "lm_error" in results
        assert "robust_lm_lag" in results
        assert "robust_lm_error" in results
        assert "recommendation" in results
        assert "reason" in results
        assert "summary" in results

        # Check recommendation
        assert results["recommendation"] in ["OLS", "SAR", "SEM", "SDM"]

        # Check summary DataFrame
        summary = results["summary"]
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 4  # Four tests
        assert "Statistic" in summary.columns
        assert "p-value" in summary.columns

    def test_lm_tests_detect_sar(self):
        """Test that LM tests detect SAR structure."""
        # Generate data with strong SAR structure
        data_dict = generate_spatial_panel_data(N=25, T=10, rho=0.7, lambda_=0.0)

        # Create mock OLS result (misspecified)
        class MockOLSResult:
            def __init__(self, data_dict):
                X = data_dict["data"][["x1", "x2", "x3"]].values
                y = data_dict["data"]["y"].values
                X_with_const = np.column_stack([np.ones(len(X)), X])
                beta_hat = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                self.params = pd.Series(beta_hat)
                self.fittedvalues = X_with_const @ beta_hat
                self.resid = y - self.fittedvalues
                self.nobs = len(y)
                self.model = type(
                    "Model",
                    (),
                    {"N": data_dict["N"], "T": data_dict["T"], "data": type("Data", (), {"y": y})},
                )()

        ols_result = MockOLSResult(data_dict)
        results = run_lm_tests(ols_result, data_dict["W"], verbose=False)

        # Should recommend SAR
        assert results["recommendation"] in ["SAR", "SDM"]

    def test_lm_tests_detect_sem(self):
        """Test that LM tests detect SEM structure."""
        # Generate data with strong SEM structure
        data_dict = generate_spatial_panel_data(N=25, T=10, rho=0.0, lambda_=0.7)

        # Create mock OLS result (misspecified)
        class MockOLSResult:
            def __init__(self, data_dict):
                X = data_dict["data"][["x1", "x2", "x3"]].values
                y = data_dict["data"]["y"].values
                X_with_const = np.column_stack([np.ones(len(X)), X])
                beta_hat = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                self.params = pd.Series(beta_hat)
                self.fittedvalues = X_with_const @ beta_hat
                self.resid = y - self.fittedvalues
                self.nobs = len(y)
                self.model = type(
                    "Model",
                    (),
                    {"N": data_dict["N"], "T": data_dict["T"], "data": type("Data", (), {"y": y})},
                )()

        ols_result = MockOLSResult(data_dict)
        results = run_lm_tests(ols_result, data_dict["W"], verbose=False)

        # Should recommend SEM
        assert results["recommendation"] in ["SEM", "SDM"]


class TestSpatialHausman:
    """Tests for Spatial Hausman test."""

    def test_basic_functionality(self):
        """Test basic Hausman test functionality."""

        # Create two mock results with different parameters
        class MockResult1:
            def __init__(self):
                self.params = pd.Series([1.0, 0.5, -0.3], index=["x1", "x2", "x3"])
                self.bse = pd.Series([0.1, 0.15, 0.12], index=["x1", "x2", "x3"])

            def cov_params(self):
                # Simple diagonal covariance
                return pd.DataFrame(
                    np.diag(self.bse**2), index=self.params.index, columns=self.params.index
                )

        class MockResult2:
            def __init__(self):
                self.params = pd.Series([1.1, 0.6, -0.25], index=["x1", "x2", "x3"])
                self.bse = pd.Series([0.12, 0.18, 0.14], index=["x1", "x2", "x3"])

            def cov_params(self):
                return pd.DataFrame(
                    np.diag(self.bse**2), index=self.params.index, columns=self.params.index
                )

        result1 = MockResult1()
        result2 = MockResult2()

        # Run Hausman test
        test = SpatialHausmanTest(result1, result2)
        result = test.run(alpha=0.05)

        # Check output
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert 0 <= result.pvalue <= 1
        assert result.statistic >= 0

        # Check summary
        summary = test.summary()
        assert isinstance(summary, pd.DataFrame)
        assert "Difference" in summary.columns

    def test_parameter_subset(self):
        """Test Hausman test on parameter subset."""

        class MockResult:
            def __init__(self, params):
                self.params = pd.Series(params)
                self.bse = pd.Series(np.abs(self.params) * 0.1, index=self.params.index)

            def cov_params(self):
                return pd.DataFrame(
                    np.diag(self.bse**2), index=self.params.index, columns=self.params.index
                )

        result1 = MockResult({"const": 1.0, "x1": 0.5, "x2": -0.3, "rho": 0.4})
        result2 = MockResult({"const": 1.1, "x1": 0.6, "x2": -0.25, "lambda": 0.3})

        test = SpatialHausmanTest(result1, result2)

        # Test on subset (exclude spatial parameters)
        result = test.run(subset=["x1", "x2"])

        assert result.metadata["n_parameters_tested"] == 2


class TestSpatialWeightsUtils:
    """Tests for spatial weights utilities."""

    def test_validate_spatial_weights(self):
        """Test spatial weights validation."""
        N = 10
        W = create_spatial_weights(N, row_standardize=False)

        # Should validate correctly
        W_valid = validate_spatial_weights(W)
        assert W_valid.shape == (N, N)

        # Test with invalid matrix (non-square)
        W_invalid = np.random.randn(N, N + 1)
        with pytest.raises(ValueError, match="must be square"):
            validate_spatial_weights(W_invalid)

        # Test with non-zero diagonal
        W_invalid = W.copy()
        np.fill_diagonal(W_invalid, 1)
        with pytest.raises(ValueError, match=r"Diagonal.*must be zero"):
            validate_spatial_weights(W_invalid)

        # Test with NaN values
        W_invalid = W.copy()
        W_invalid[0, 1] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_spatial_weights(W_invalid)

    def test_standardize_spatial_weights(self):
        """Test spatial weights standardization."""
        N = 10
        W = create_spatial_weights(N, row_standardize=False)

        # Row standardization
        W_std = standardize_spatial_weights(W, style="row")
        row_sums = W_std.sum(axis=1)
        # All non-zero rows should sum to 1
        non_zero_rows = row_sums > 0
        np.testing.assert_allclose(row_sums[non_zero_rows], 1.0, rtol=1e-10)

        # Spectral normalization
        W_spec = standardize_spatial_weights(W, style="spectral")
        eigenvalues = np.linalg.eigvalsh(W_spec)
        max_eig = np.max(np.abs(eigenvalues))
        assert max_eig <= 1.0 + 1e-10

        # No standardization
        W_none = standardize_spatial_weights(W, style="none")
        np.testing.assert_array_equal(W_none, W)
