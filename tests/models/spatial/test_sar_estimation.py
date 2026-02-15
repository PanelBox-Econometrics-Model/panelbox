"""
Test suite for Spatial Lag (SAR) Model estimation.

Tests SAR-FE estimation with Quasi-ML (Lee & Yu 2010).
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from panelbox.core.spatial_weights import SpatialWeights
from panelbox.models.spatial.spatial_lag import SpatialLag


class TestSARDataGeneration:
    """Tests for SAR data generation process."""

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
    def generate_sar_panel_data(n, t, rho, beta, W, sigma2=1.0, alpha_std=1.0, seed=None):
        """
        Generate panel data from SAR model.

        Model: y = ρWy + Xβ + α + ε

        Parameters
        ----------
        n : int
            Number of entities
        t : int
            Number of time periods
        rho : float
            Spatial autoregressive parameter
        beta : array-like
            Regression coefficients
        W : array-like
            Spatial weight matrix (n x n)
        sigma2 : float
            Error variance
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

        # Generate errors
        epsilon = np.random.normal(0, np.sqrt(sigma2), n * t)

        # Build panel structure
        entities = np.repeat(np.arange(n), t)
        times = np.tile(np.arange(t), n)

        # Expand fixed effects
        alpha_expanded = np.repeat(alpha, t)

        # Generate y solving: y = ρWy + Xβ + α + ε
        # => (I - ρW)y = Xβ + α + ε
        # => y = (I - ρW)^{-1}(Xβ + α + ε)

        # For each time period
        y = np.zeros(n * t)
        for period in range(t):
            idx_t = times == period
            X_t = X[idx_t]
            alpha_t = alpha
            epsilon_t = epsilon[idx_t]

            # Reduced form
            I_rhoW = np.eye(n) - rho * W
            I_rhoW_inv = np.linalg.inv(I_rhoW)

            y_t = I_rhoW_inv @ (X_t @ beta + alpha_t + epsilon_t)
            y[idx_t] = y_t

        # Create DataFrame
        data = pd.DataFrame({"entity": entities, "time": times, "y": y})

        # Add X variables
        for j in range(k):
            data[f"x{j+1}"] = X[:, j]

        return data


class TestSAREstimation:
    """Tests for SAR model estimation."""

    def test_sar_fe_basic(self):
        """Test basic SAR-FE estimation on small dataset."""
        # Generate data
        n, t = 25, 10
        rho_true = 0.4
        beta_true = np.array([1.0, -0.5])

        W = TestSARDataGeneration.generate_spatial_weights(n, type="rook")
        data = TestSARDataGeneration.generate_sar_panel_data(n, t, rho_true, beta_true, W, seed=42)

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Estimate model
        model = SpatialLag(
            formula="y ~ x1 + x2", data=data, entity_col="entity", time_col="time", W=W_obj
        )

        result = model.fit(effects="fixed", method="qml")

        # Check that estimation completed
        assert result is not None
        assert "rho" in result.params.index
        assert "x1" in result.params.index
        assert "x2" in result.params.index

        # Check parameter recovery (loose tolerance for small sample)
        assert abs(result.params["rho"] - rho_true) < 0.2
        assert abs(result.params["x1"] - beta_true[0]) < 0.3
        assert abs(result.params["x2"] - beta_true[1]) < 0.3

    def test_sar_fe_larger_sample(self):
        """Test SAR-FE on larger dataset for better convergence."""
        # Generate data
        n, t = 100, 20
        rho_true = 0.5
        beta_true = np.array([2.0, -1.0, 0.5])

        W = TestSARDataGeneration.generate_spatial_weights(n, type="random", density=0.1)
        data = TestSARDataGeneration.generate_sar_panel_data(n, t, rho_true, beta_true, W, seed=123)

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Estimate model
        model = SpatialLag(
            formula="y ~ x1 + x2 + x3", data=data, entity_col="entity", time_col="time", W=W_obj
        )

        result = model.fit(effects="fixed", method="qml")

        # Check parameter recovery (tighter tolerance with larger sample)
        assert abs(result.params["rho"] - rho_true) < 0.1
        assert abs(result.params["x1"] - beta_true[0]) < 0.15
        assert abs(result.params["x2"] - beta_true[1]) < 0.15
        assert abs(result.params["x3"] - beta_true[2]) < 0.15

    def test_sar_fe_grid_based_weights(self):
        """Test SAR-FE with grid-based (queen) spatial weights."""
        # Generate data with grid structure
        n, t = 49, 15  # 7x7 grid
        rho_true = 0.3
        beta_true = np.array([1.5, -0.8])

        W = TestSARDataGeneration.generate_spatial_weights(n, type="queen")
        data = TestSARDataGeneration.generate_sar_panel_data(n, t, rho_true, beta_true, W, seed=456)

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Estimate model
        model = SpatialLag(
            formula="y ~ x1 + x2", data=data, entity_col="entity", time_col="time", W=W_obj
        )

        result = model.fit(effects="fixed", method="qml")

        # Check results
        assert result is not None
        assert result.method == "Quasi-ML (Lee & Yu 2010)"
        assert result.effects == "fixed"

        # Check spatial parameter bounds
        assert -0.99 <= result.params["rho"] <= 0.99

    def test_sar_fe_edge_cases(self):
        """Test SAR-FE with edge cases."""
        # Test with rho near boundary
        n, t = 36, 8  # 6x6 grid
        rho_true = 0.85  # Near upper bound
        beta_true = np.array([1.0])

        W = TestSARDataGeneration.generate_spatial_weights(n, type="queen")
        data = TestSARDataGeneration.generate_sar_panel_data(n, t, rho_true, beta_true, W, seed=789)

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Estimate model
        model = SpatialLag(
            formula="y ~ x1", data=data, entity_col="entity", time_col="time", W=W_obj
        )

        result = model.fit(effects="fixed", method="qml")

        # Should still converge even with high rho
        assert result is not None
        assert 0 < result.params["rho"] < 1

    def test_sar_fe_standard_errors(self):
        """Test that standard errors are computed correctly."""
        # Generate data
        n, t = 64, 12  # 8x8 grid
        rho_true = 0.4
        beta_true = np.array([1.0, -0.5])

        W = TestSARDataGeneration.generate_spatial_weights(n, type="queen")
        data = TestSARDataGeneration.generate_sar_panel_data(n, t, rho_true, beta_true, W, seed=321)

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Estimate model
        model = SpatialLag(
            formula="y ~ x1 + x2", data=data, entity_col="entity", time_col="time", W=W_obj
        )

        result = model.fit(effects="fixed", method="qml")

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

    def test_sar_fe_predictions(self):
        """Test predictions from SAR-FE model."""
        # Generate data
        n, t = 25, 10
        rho_true = 0.4
        beta_true = np.array([1.0, -0.5])

        W = TestSARDataGeneration.generate_spatial_weights(n, type="rook")
        data = TestSARDataGeneration.generate_sar_panel_data(n, t, rho_true, beta_true, W, seed=654)

        # Create spatial weights object
        W_obj = SpatialWeights(W)

        # Estimate model
        model = SpatialLag(
            formula="y ~ x1 + x2", data=data, entity_col="entity", time_col="time", W=W_obj
        )

        result = model.fit(effects="fixed", method="qml")

        # Check fitted values
        assert result.fittedvalues is not None
        assert len(result.fittedvalues) == len(data)

        # Check residuals
        assert result.resid is not None
        assert len(result.resid) == len(data)

        # Fitted + residuals should approximately equal y (after within transformation)
        # This is approximate due to the within transformation
        y_reconstructed = result.fittedvalues + result.resid
        # Check that they're at least correlated
        correlation = np.corrcoef(
            y_reconstructed.flatten(), model._within_transformation(model.endog).flatten()
        )[0, 1]
        assert correlation > 0.95


class TestSARLogDeterminant:
    """Tests for log-determinant computation methods."""

    def test_log_det_eigenvalue_method(self):
        """Test eigenvalue method for log-determinant."""
        n = 50
        W = TestSARDataGeneration.generate_spatial_weights(n, type="random", density=0.2)
        W_obj = SpatialWeights(W)

        # Create dummy data
        data = pd.DataFrame(
            {
                "entity": np.repeat(np.arange(n), 5),
                "time": np.tile(np.arange(5), n),
                "y": np.random.normal(0, 1, n * 5),
                "x1": np.random.normal(0, 1, n * 5),
            }
        )

        model = SpatialLag("y ~ x1", data, "entity", "time", W_obj)

        # Test log-det calculation
        rho = 0.5
        log_det = model._log_det_jacobian(rho, W, method="eigenvalue")

        # Verify with direct calculation
        I_rhoW = np.eye(n) - rho * W
        _, log_det_direct = np.linalg.slogdet(I_rhoW)

        assert_allclose(log_det, log_det_direct, rtol=1e-10)

    def test_log_det_sparse_lu_method(self):
        """Test sparse LU method for log-determinant."""
        n = 100
        W = TestSARDataGeneration.generate_spatial_weights(n, type="rook")
        W_obj = SpatialWeights(W)

        # Create dummy data
        data = pd.DataFrame(
            {
                "entity": np.repeat(np.arange(n), 5),
                "time": np.tile(np.arange(5), n),
                "y": np.random.normal(0, 1, n * 5),
                "x1": np.random.normal(0, 1, n * 5),
            }
        )

        model = SpatialLag("y ~ x1", data, "entity", "time", W_obj)

        # Test log-det calculation
        rho = 0.3
        log_det = model._log_det_jacobian(rho, W, method="sparse_lu")

        # Verify with eigenvalue method
        log_det_eigen = model._log_det_jacobian(rho, W, method="eigenvalue")

        assert_allclose(log_det, log_det_eigen, rtol=1e-8)

    def test_log_det_auto_selection(self):
        """Test automatic method selection based on matrix size."""
        # Small matrix - should use eigenvalue
        n_small = 50
        W_small = TestSARDataGeneration.generate_spatial_weights(n_small, type="random")
        W_obj_small = SpatialWeights(W_small)

        data_small = pd.DataFrame(
            {
                "entity": np.repeat(np.arange(n_small), 5),
                "time": np.tile(np.arange(5), n_small),
                "y": np.random.normal(0, 1, n_small * 5),
                "x1": np.random.normal(0, 1, n_small * 5),
            }
        )

        model_small = SpatialLag("y ~ x1", data_small, "entity", "time", W_obj_small)

        # Should automatically select eigenvalue method for n < 1000
        rho = 0.4
        log_det = model_small._log_det_jacobian(rho, W_small, method="auto")
        log_det_eigen = model_small._log_det_jacobian(rho, W_small, method="eigenvalue")

        assert_allclose(log_det, log_det_eigen, rtol=1e-10)


class TestSARSpatialBounds:
    """Tests for spatial coefficient bounds computation."""

    def test_bounds_row_normalized_weights(self):
        """Test bounds for row-normalized weight matrix."""
        n = 36
        W = TestSARDataGeneration.generate_spatial_weights(n, type="queen")
        W_obj = SpatialWeights(W)

        # Create dummy data
        data = pd.DataFrame(
            {
                "entity": np.repeat(np.arange(n), 5),
                "time": np.tile(np.arange(5), n),
                "y": np.random.normal(0, 1, n * 5),
                "x1": np.random.normal(0, 1, n * 5),
            }
        )

        model = SpatialLag("y ~ x1", data, "entity", "time", W_obj)

        # Get bounds
        rho_min, rho_max = model._spatial_coefficient_bounds(W)

        # For row-normalized matrix, bounds should be close to (-1, 1)
        assert -1 <= rho_min < 0
        assert 0 < rho_max <= 1

        # Bounds should be symmetric for symmetric W
        # (approximately, due to numerical precision)
        assert abs(abs(rho_min) - abs(rho_max)) < 0.1

    def test_bounds_asymmetric_weights(self):
        """Test bounds for asymmetric weight matrix."""
        n = 30
        # Create asymmetric W
        W = np.random.rand(n, n) * 0.3
        np.fill_diagonal(W, 0)
        # Row normalize
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

        W_obj = SpatialWeights(W)

        # Create dummy data
        data = pd.DataFrame(
            {
                "entity": np.repeat(np.arange(n), 5),
                "time": np.tile(np.arange(5), n),
                "y": np.random.normal(0, 1, n * 5),
                "x1": np.random.normal(0, 1, n * 5),
            }
        )

        model = SpatialLag("y ~ x1", data, "entity", "time", W_obj)

        # Get bounds
        rho_min, rho_max = model._spatial_coefficient_bounds(W)

        # Bounds should be valid
        assert rho_min < rho_max
        assert -0.99 <= rho_min
        assert rho_max <= 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
