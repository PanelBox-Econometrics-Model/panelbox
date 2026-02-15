"""
Test suite for Spatial Error Model (SEM) estimation.

Tests SEM-FE estimation with GMM using spatial instruments.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from panelbox.core.spatial_weights import SpatialWeights
from panelbox.models.spatial.spatial_error import SpatialError


class TestSEMDataGeneration:
    """Tests for SEM data generation process."""

    @staticmethod
    def generate_spatial_weights(n, type="random", density=0.3):
        """Generate spatial weight matrix for testing."""
        if type == "random":
            # Random sparse matrix
            W = stats.bernoulli.rvs(density, size=(n, n))
            W = W.astype(float)
            # Make symmetric
            W = (W + W.T) / 2
            # Zero diagonal
            np.fill_diagonal(W, 0)
            # Row normalize
            row_sums = W.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            W = W / row_sums
        elif type == "rook":
            # Simple rook contiguity (1D chain)
            W = np.zeros((n, n))
            for i in range(n - 1):
                W[i, i + 1] = 1
                W[i + 1, i] = 1
            # Row normalize
            row_sums = W.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            W = W / row_sums
        elif type == "queen":
            # Grid-based queen contiguity
            grid_size = int(np.sqrt(n))
            assert grid_size**2 == n, "n must be perfect square for queen"
            W = np.zeros((n, n))
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    # Add neighbors
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                                nidx = ni * grid_size + nj
                                W[idx, nidx] = 1
            # Row normalize
            row_sums = W.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            W = W / row_sums
        else:
            raise ValueError(f"Unknown type: {type}")

        return W

    @staticmethod
    def generate_sem_panel_data(n, t, lambda_, beta, W, sigma2=1.0, alpha_std=1.0, seed=None):
        """
        Generate panel data from SEM model.

        Model: y = Xβ + α + u, where u = λWu + ε

        Parameters
        ----------
        n : int
            Number of entities
        t : int
            Number of time periods
        lambda_ : float
            Spatial error parameter
        beta : array-like
            Regression coefficients
        W : array-like
            Spatial weight matrix (n x n)
        sigma2 : float
            Innovation variance
        alpha_std : float
            Standard deviation of fixed effects
        seed : int, optional
            Random seed

        Returns
        -------
        pd.DataFrame
            Panel data with columns: entity, time, y, x1, x2, ...
        """
        if seed is not None:
            np.random.seed(seed)

        k = len(beta)

        # Generate fixed effects
        alpha = np.random.normal(0, alpha_std, n)

        # Generate exogenous variables
        X = np.random.normal(0, 1, (n * t, k))

        # Generate innovations
        epsilon = np.random.normal(0, np.sqrt(sigma2), n * t)

        # Build panel structure
        entities = np.repeat(np.arange(n), t)
        times = np.tile(np.arange(t), n)

        # Expand fixed effects
        alpha_expanded = np.repeat(alpha, t)

        # Generate spatially correlated errors
        # u = λWu + ε => u = (I - λW)^{-1}ε
        y = np.zeros(n * t)
        for period in range(t):
            idx_t = times == period
            X_t = X[idx_t]
            alpha_t = alpha
            epsilon_t = epsilon[idx_t]

            # Spatial error process
            I_lambdaW = np.eye(n) - lambda_ * W
            I_lambdaW_inv = np.linalg.inv(I_lambdaW)
            u_t = I_lambdaW_inv @ epsilon_t

            # Generate y
            y_t = X_t @ beta + alpha_t + u_t
            y[idx_t] = y_t

        # Create DataFrame
        data = pd.DataFrame({"entity": entities, "time": times, "y": y})

        # Add X variables
        for j in range(k):
            data[f"x{j+1}"] = X[:, j]

        return data


class TestSEMEstimation:
    """Tests for SEM model estimation."""

    def test_sem_fe_basic(self):
        """Test basic SEM-FE estimation on small dataset."""
        # Generate data
        n, t = 25, 10
        lambda_true = 0.4
        beta_true = np.array([1.0, -0.5])

        W = TestSEMDataGeneration.generate_spatial_weights(n, type="rook")
        data = TestSEMDataGeneration.generate_sem_panel_data(
            n, t, lambda_true, beta_true, W, seed=42
        )

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Estimate model
        model = SpatialError(
            formula="y ~ x1 + x2", data=data, entity_col="entity", time_col="time", W=W_obj
        )

        result = model.fit(effects="fixed", method="gmm")

        # Check that estimation completed
        assert result is not None
        assert "lambda" in result.params.index
        assert "x1" in result.params.index
        assert "x2" in result.params.index

        # Check parameter recovery (loose tolerance for small sample)
        assert abs(result.params["lambda"] - lambda_true) < 0.3
        assert abs(result.params["x1"] - beta_true[0]) < 0.3
        assert abs(result.params["x2"] - beta_true[1]) < 0.3

    def test_sem_fe_larger_sample(self):
        """Test SEM-FE on larger dataset for better convergence."""
        # Generate data
        n, t = 100, 20
        lambda_true = 0.5
        beta_true = np.array([2.0, -1.0, 0.5])

        W = TestSEMDataGeneration.generate_spatial_weights(n, type="random", density=0.1)
        data = TestSEMDataGeneration.generate_sem_panel_data(
            n, t, lambda_true, beta_true, W, seed=123
        )

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Estimate model
        model = SpatialError(
            formula="y ~ x1 + x2 + x3", data=data, entity_col="entity", time_col="time", W=W_obj
        )

        result = model.fit(effects="fixed", method="gmm")

        # Check parameter recovery (tighter tolerance with larger sample)
        assert abs(result.params["lambda"] - lambda_true) < 0.15
        assert abs(result.params["x1"] - beta_true[0]) < 0.2
        assert abs(result.params["x2"] - beta_true[1]) < 0.2
        assert abs(result.params["x3"] - beta_true[2]) < 0.2

    def test_sem_fe_grid_based_weights(self):
        """Test SEM-FE with grid-based (queen) spatial weights."""
        # Generate data with grid structure
        n, t = 49, 15  # 7x7 grid
        lambda_true = 0.3
        beta_true = np.array([1.5, -0.8])

        W = TestSEMDataGeneration.generate_spatial_weights(n, type="queen")
        data = TestSEMDataGeneration.generate_sem_panel_data(
            n, t, lambda_true, beta_true, W, seed=456
        )

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Estimate model
        model = SpatialError(
            formula="y ~ x1 + x2", data=data, entity_col="entity", time_col="time", W=W_obj
        )

        result = model.fit(effects="fixed", method="gmm")

        # Check results
        assert result is not None
        assert result.method == "GMM (Spatial Instruments)"
        assert result.effects == "fixed"

        # Check spatial parameter bounds
        assert -0.99 <= result.params["lambda"] <= 0.99

    def test_sem_fe_with_different_n_lags(self):
        """Test SEM-FE with different numbers of spatial lags as instruments."""
        # Generate data
        n, t = 36, 12  # 6x6 grid
        lambda_true = 0.4
        beta_true = np.array([1.0, -0.5])

        W = TestSEMDataGeneration.generate_spatial_weights(n, type="queen")
        data = TestSEMDataGeneration.generate_sem_panel_data(
            n, t, lambda_true, beta_true, W, seed=789
        )

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Test with different numbers of lags
        for n_lags in [1, 2, 3]:
            model = SpatialError(
                formula="y ~ x1 + x2", data=data, entity_col="entity", time_col="time", W=W_obj
            )

            result = model.fit(effects="fixed", method="gmm", n_lags=n_lags)

            # Should converge for all reasonable n_lags
            assert result is not None
            assert "lambda" in result.params.index

    def test_sem_fe_standard_errors(self):
        """Test that standard errors are computed correctly."""
        # Generate data
        n, t = 64, 12  # 8x8 grid
        lambda_true = 0.4
        beta_true = np.array([1.0, -0.5])

        W = TestSEMDataGeneration.generate_spatial_weights(n, type="queen")
        data = TestSEMDataGeneration.generate_sem_panel_data(
            n, t, lambda_true, beta_true, W, seed=321
        )

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Estimate model
        model = SpatialError(
            formula="y ~ x1 + x2", data=data, entity_col="entity", time_col="time", W=W_obj
        )

        result = model.fit(effects="fixed", method="gmm")

        # Check that we have standard errors
        assert result.std_errors is not None
        assert len(result.std_errors) == len(result.params)
        assert all(result.std_errors > 0)

        # Check t-statistics
        assert result.t_statistics is not None
        assert len(result.t_statistics) == len(result.params)

        # Check p-values
        assert result.p_values is not None
        assert len(result.p_values) == len(result.params)
        assert all((0 <= p <= 1) for p in result.p_values)

    def test_sem_fe_residuals(self):
        """Test residuals from SEM-FE model."""
        # Generate data
        n, t = 25, 10
        lambda_true = 0.4
        beta_true = np.array([1.0, -0.5])

        W = TestSEMDataGeneration.generate_spatial_weights(n, type="rook")
        data = TestSEMDataGeneration.generate_sem_panel_data(
            n, t, lambda_true, beta_true, W, seed=654
        )

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Estimate model
        model = SpatialError(
            formula="y ~ x1 + x2", data=data, entity_col="entity", time_col="time", W=W_obj
        )

        result = model.fit(effects="fixed", method="gmm")

        # Check residuals
        assert result.resid is not None
        assert len(result.resid) == len(data)

        # Residuals should have approximately zero mean
        assert abs(np.mean(result.resid)) < 0.1

    def test_sem_fe_edge_cases(self):
        """Test SEM-FE with edge cases."""
        # Test with lambda near boundary
        n, t = 36, 8  # 6x6 grid
        lambda_true = 0.85  # Near upper bound
        beta_true = np.array([1.0])

        W = TestSEMDataGeneration.generate_spatial_weights(n, type="queen")
        data = TestSEMDataGeneration.generate_sem_panel_data(
            n, t, lambda_true, beta_true, W, seed=987
        )

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Estimate model
        model = SpatialError(
            formula="y ~ x1", data=data, entity_col="entity", time_col="time", W=W_obj
        )

        result = model.fit(effects="fixed", method="gmm")

        # Should still converge even with high lambda
        assert result is not None
        assert 0 < result.params["lambda"] < 1


class TestSEMGMMInstruments:
    """Tests for GMM instruments construction."""

    def test_gmm_instruments_construction(self):
        """Test that spatial instruments are constructed correctly."""
        n = 25
        t = 10
        k = 2
        W = TestSEMDataGeneration.generate_spatial_weights(n, type="rook")
        W_obj = SpatialWeights(W)

        # Create dummy data
        data = pd.DataFrame(
            {
                "entity": np.repeat(np.arange(n), t),
                "time": np.tile(np.arange(t), n),
                "y": np.random.normal(0, 1, n * t),
                "x1": np.random.normal(0, 1, n * t),
                "x2": np.random.normal(0, 1, n * t),
            }
        )

        model = SpatialError(
            formula="y ~ x1 + x2", data=data, entity_col="entity", time_col="time", W=W_obj
        )

        # Access internal methods to check instrument construction
        X_within = model._within_transformation(model.exog)

        # Construct instruments with n_lags=2
        n_lags = 2
        Z = [X_within]
        WkX = X_within
        for _ in range(n_lags):
            WkX = model.W_normalized @ WkX
            Z.append(WkX)

        Z = np.hstack(Z)

        # Check dimensions
        # Should have X, WX, W²X columns
        expected_cols = k * (n_lags + 1)
        assert Z.shape[1] == expected_cols

        # Check that instruments are different from original X
        # (they should be spatial lags)
        assert not np.allclose(Z[:, :k], Z[:, k : 2 * k])
        assert not np.allclose(Z[:, :k], Z[:, 2 * k : 3 * k])

    def test_gmm_weight_matrix_computation(self):
        """Test optimal GMM weight matrix computation."""
        n = 36  # 6x6 grid
        t = 8
        W = TestSEMDataGeneration.generate_spatial_weights(n, type="queen")
        W_obj = SpatialWeights(W)

        # Generate data
        lambda_true = 0.3
        beta_true = np.array([1.0, -0.5])
        data = TestSEMDataGeneration.generate_sem_panel_data(
            n, t, lambda_true, beta_true, W, seed=555
        )

        model = SpatialError(
            formula="y ~ x1 + x2", data=data, entity_col="entity", time_col="time", W=W_obj
        )

        # Perform within transformation
        y_within = model._within_transformation(model.endog)
        X_within = model._within_transformation(model.exog)

        # Construct instruments
        Z = [X_within]
        WX = model.W_normalized @ X_within
        Z.append(WX)
        Z = np.hstack(Z)

        # First-stage residuals
        beta_1 = np.linalg.lstsq(Z.T @ X_within, Z.T @ y_within, rcond=None)[0]
        u_1 = y_within - X_within @ beta_1

        # Compute weight matrix
        Omega = model._compute_spatial_gmm_weight_matrix(u_1, Z)

        # Check properties of weight matrix
        assert Omega.shape[0] == Z.shape[1]
        assert Omega.shape[1] == Z.shape[1]

        # Should be positive definite
        eigenvalues = np.linalg.eigvals(Omega)
        assert all(eigenvalues > -1e-10)  # Allow small numerical errors

        # Should be symmetric
        assert_allclose(Omega, Omega.T, rtol=1e-10)


class TestSEMComparison:
    """Tests comparing SEM with other models."""

    def test_sem_vs_ols_no_spatial_correlation(self):
        """Test that SEM reduces to OLS when lambda=0."""
        # Generate data with no spatial correlation
        n, t = 49, 10  # 7x7 grid
        lambda_true = 0.0  # No spatial correlation
        beta_true = np.array([1.0, -0.5])

        W = TestSEMDataGeneration.generate_spatial_weights(n, type="queen")
        data = TestSEMDataGeneration.generate_sem_panel_data(
            n, t, lambda_true, beta_true, W, sigma2=0.5, seed=888
        )

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Estimate SEM model
        model_sem = SpatialError(
            formula="y ~ x1 + x2", data=data, entity_col="entity", time_col="time", W=W_obj
        )

        result_sem = model.fit(effects="fixed", method="gmm")

        # Lambda should be close to zero
        assert abs(result_sem.params["lambda"]) < 0.1

        # Coefficients should be close to true values
        assert abs(result_sem.params["x1"] - beta_true[0]) < 0.2
        assert abs(result_sem.params["x2"] - beta_true[1]) < 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
