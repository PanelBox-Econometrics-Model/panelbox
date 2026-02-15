"""
Tests for General Nesting Spatial (GNS) Model.

This module tests the GNS model implementation, including:
- Parameter estimation
- Model type identification
- Likelihood ratio tests
- Special cases recovery
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from panelbox.models.spatial import GeneralNestingSpatial, SpatialDurbin, SpatialError, SpatialLag


class TestGeneralNestingSpatial:
    """Test suite for GNS model."""

    @pytest.fixture
    def panel_data(self):
        """Generate panel data with spatial structure."""
        np.random.seed(42)

        # Dimensions
        N = 25  # entities
        T = 10  # time periods
        K = 3  # exogenous variables

        # Create spatial weight matrix (queen contiguity on 5x5 grid)
        W = self._create_queen_weights(5, 5)

        # Generate exogenous variables
        X = np.random.randn(N * T, K)

        # True parameters
        beta_true = np.array([2.0, -1.5, 0.8])
        rho_true = 0.4
        lambda_true = 0.3
        theta_true = np.array([0.5, -0.3, 0.2])  # SDM parameters

        # Generate spatial data following GNS DGP
        y = np.zeros(N * T)

        for t in range(T):
            start_idx = t * N
            end_idx = (t + 1) * N

            # Slice for current period
            X_t = X[start_idx:end_idx]

            # Generate errors with spatial correlation
            eps = np.random.randn(N)
            u = np.linalg.inv(np.eye(N) - lambda_true * W) @ eps

            # Generate y with spatial lag and spatial Durbin terms
            WX_t = W @ X_t
            Xb = X_t @ beta_true + WX_t @ theta_true

            # Solve for y: (I - ρW)y = Xβ + WXθ + u
            y_t = np.linalg.inv(np.eye(N) - rho_true * W) @ (Xb + u)
            y[start_idx:end_idx] = y_t

        # Create DataFrame
        entities = np.repeat(np.arange(N), T)
        time_periods = np.tile(np.arange(T), N)

        data = pd.DataFrame(
            {
                "entity": entities,
                "time": time_periods,
                "y": y,
                "x1": X[:, 0],
                "x2": X[:, 1],
                "x3": X[:, 2],
            }
        )

        data = data.set_index(["entity", "time"])

        return {
            "data": data,
            "W": W,
            "beta_true": beta_true,
            "rho_true": rho_true,
            "lambda_true": lambda_true,
            "theta_true": theta_true,
        }

    def _create_queen_weights(self, rows, cols):
        """Create queen contiguity weight matrix for grid."""
        N = rows * cols
        W = np.zeros((N, N))

        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j

                # Neighbors
                neighbors = []

                # Horizontal neighbors
                if j > 0:
                    neighbors.append(i * cols + (j - 1))
                if j < cols - 1:
                    neighbors.append(i * cols + (j + 1))

                # Vertical neighbors
                if i > 0:
                    neighbors.append((i - 1) * cols + j)
                if i < rows - 1:
                    neighbors.append((i + 1) * cols + j)

                # Diagonal neighbors
                if i > 0 and j > 0:
                    neighbors.append((i - 1) * cols + (j - 1))
                if i > 0 and j < cols - 1:
                    neighbors.append((i - 1) * cols + (j + 1))
                if i < rows - 1 and j > 0:
                    neighbors.append((i + 1) * cols + (j - 1))
                if i < rows - 1 and j < cols - 1:
                    neighbors.append((i + 1) * cols + (j + 1))

                # Set weights
                for neighbor in neighbors:
                    W[idx, neighbor] = 1

        # Row-normalize
        row_sums = W.sum(axis=1)
        W = W / row_sums[:, np.newaxis]

        return W

    def test_gns_initialization(self, panel_data):
        """Test GNS model initialization."""
        data = panel_data["data"]
        W = panel_data["W"]

        # Initialize with single W matrix
        model = GeneralNestingSpatial(
            formula="y ~ x1 + x2 + x3",
            data=data.reset_index(),
            entity_col="entity",
            time_col="time",
            W1=W,
        )

        assert model.W1 is not None
        assert model.W2 is not None
        assert model.W3 is not None
        assert model.W1.shape == (25, 25)

        # Initialize with different W matrices
        W2 = W @ W  # W-squared
        model2 = GeneralNestingSpatial(
            formula="y ~ x1 + x2 + x3",
            data=data.reset_index(),
            entity_col="entity",
            time_col="time",
            W1=W,
            W2=W2,
            W3=W,
        )

        assert not np.allclose(model2.W2, model2.W1)

    def test_gns_full_model_estimation(self, panel_data):
        """Test estimation of full GNS model."""
        data = panel_data["data"]
        W = panel_data["W"]

        model = GeneralNestingSpatial(
            formula="y ~ x1 + x2 + x3",
            data=data.reset_index(),
            entity_col="entity",
            time_col="time",
            W1=W,
            W2=W,
            W3=W,
        )

        # Fit full model
        result = model.fit(effects="fixed", method="ml", include_wx=True, maxiter=100)

        # Check that all parameters are estimated
        assert "rho" in result.params.index
        assert "lambda" in result.params.index
        assert any("theta" in idx for idx in result.params.index)

        # Parameters should be reasonably close to true values
        # (with some tolerance due to fixed effects and finite sample)
        assert abs(result.params.loc["rho", "coefficient"] - panel_data["rho_true"]) < 0.2
        assert abs(result.params.loc["lambda", "coefficient"] - panel_data["lambda_true"]) < 0.3

    def test_model_type_identification(self, panel_data):
        """Test automatic identification of nested model types."""
        data = panel_data["data"]
        W = panel_data["W"]

        # Generate SAR data (rho != 0, theta = 0, lambda = 0)
        N = 25
        T = 10
        X = np.random.randn(N * T, 2)
        beta = np.array([1.0, -0.5])
        rho = 0.5

        y = np.zeros(N * T)
        for t in range(T):
            start_idx = t * N
            end_idx = (t + 1) * N
            X_t = X[start_idx:end_idx]
            eps = np.random.randn(N)
            y_t = np.linalg.inv(np.eye(N) - rho * W) @ (X_t @ beta + eps)
            y[start_idx:end_idx] = y_t

        sar_data = pd.DataFrame(
            {
                "entity": np.repeat(np.arange(N), T),
                "time": np.tile(np.arange(T), N),
                "y": y,
                "x1": X[:, 0],
                "x2": X[:, 1],
            }
        ).set_index(["entity", "time"])

        # Fit and identify as SAR
        model = GeneralNestingSpatial(
            formula="y ~ x1 + x2",
            data=sar_data.reset_index(),
            entity_col="entity",
            time_col="time",
            W1=W,
        )

        result = model.fit(effects="fixed", include_wx=False)
        identified_type = model.identify_model_type(result)

        # Should identify as SAR (or close to it)
        assert identified_type in ["SAR", "OLS"]  # Might be OLS if rho not significant

    def test_likelihood_ratio_tests(self, panel_data):
        """Test LR tests for nested models."""
        data = panel_data["data"]
        W = panel_data["W"]

        model = GeneralNestingSpatial(
            formula="y ~ x1 + x2 + x3",
            data=data.reset_index(),
            entity_col="entity",
            time_col="time",
            W1=W,
            W2=W,
            W3=W,
        )

        # Fit unrestricted model
        full_result = model.fit(effects="fixed", include_wx=True, maxiter=50)

        # Test restriction to SAR (theta = 0, lambda = 0)
        lr_test_sar = model.test_restrictions(
            restrictions={"theta": 0, "lambda": 0}, full_model=full_result
        )

        assert "lr_statistic" in lr_test_sar
        assert "p_value" in lr_test_sar
        assert lr_test_sar["lr_statistic"] >= 0

        # Test restriction to SEM (rho = 0, theta = 0)
        lr_test_sem = model.test_restrictions(
            restrictions={"rho": 0, "theta": 0}, full_model=full_result
        )

        assert lr_test_sem["restricted_model_type"] == "SEM"

    def test_gns_recovers_sar(self, panel_data):
        """Test that GNS recovers SAR when theta=0, lambda=0."""
        data = panel_data["data"]
        W = panel_data["W"]

        # Generate pure SAR data
        N = 25
        T = 10
        X = np.random.randn(N * T, 2)
        beta = np.array([1.0, -0.5])
        rho = 0.4

        y = np.zeros(N * T)
        for t in range(T):
            start_idx = t * N
            end_idx = (t + 1) * N
            X_t = X[start_idx:end_idx]
            eps = np.random.randn(N) * 0.5
            y_t = np.linalg.inv(np.eye(N) - rho * W) @ (X_t @ beta + eps)
            y[start_idx:end_idx] = y_t

        sar_data = pd.DataFrame(
            {
                "entity": np.repeat(np.arange(N), T),
                "time": np.tile(np.arange(T), N),
                "y": y,
                "x1": X[:, 0],
                "x2": X[:, 1],
            }
        ).set_index(["entity", "time"])

        # Fit GNS with restrictions
        gns_model = GeneralNestingSpatial(
            formula="y ~ x1 + x2",
            data=sar_data.reset_index(),
            entity_col="entity",
            time_col="time",
            W1=W,
            W3=W,
        )

        gns_result = gns_model.fit(
            effects="fixed",
            include_wx=False,  # No WX terms
            lambda_init=0.0,  # Start lambda at 0
            maxiter=50,
        )

        # Compare with pure SAR model
        sar_model = SpatialLag(
            formula="y ~ x1 + x2",
            data=sar_data.reset_index(),
            entity_col="entity",
            time_col="time",
            W=W,
        )

        sar_result = sar_model.fit(effects="fixed")

        # Results should be similar
        assert abs(gns_result.rho - sar_result.rho) < 0.1

    def test_gns_with_different_weight_matrices(self):
        """Test GNS with different W matrices for each component."""
        np.random.seed(123)

        N = 16  # 4x4 grid
        T = 8
        K = 2

        # Create different weight matrices
        W1 = self._create_queen_weights(4, 4)  # Queen contiguity
        W2 = self._create_rook_weights(4, 4)  # Rook contiguity
        W3 = self._create_distance_weights(4, 4, cutoff=1.5)  # Distance-based

        # Generate data with different spatial structures
        X = np.random.randn(N * T, K)
        beta = np.array([1.5, -1.0])
        rho = 0.3
        theta = np.array([0.4, -0.2])
        lambda_ = 0.25

        y = np.zeros(N * T)
        for t in range(T):
            start_idx = t * N
            end_idx = (t + 1) * N
            X_t = X[start_idx:end_idx]

            # Different W for different components
            WX_t = W2 @ X_t  # W2 for WX
            eps = np.random.randn(N) * 0.5
            u = np.linalg.inv(np.eye(N) - lambda_ * W3) @ eps  # W3 for errors
            y_t = np.linalg.inv(np.eye(N) - rho * W1) @ (X_t @ beta + WX_t @ theta + u)  # W1 for Wy
            y[start_idx:end_idx] = y_t

        data = pd.DataFrame(
            {
                "entity": np.repeat(np.arange(N), T),
                "time": np.tile(np.arange(T), N),
                "y": y,
                "x1": X[:, 0],
                "x2": X[:, 1],
            }
        ).set_index(["entity", "time"])

        # Fit model with different W matrices
        model = GeneralNestingSpatial(
            formula="y ~ x1 + x2",
            data=data.reset_index(),
            entity_col="entity",
            time_col="time",
            W1=W1,
            W2=W2,
            W3=W3,
        )

        result = model.fit(effects="fixed", include_wx=True, maxiter=50)

        # Check that parameters are estimated
        assert result.rho is not None
        assert "lambda" in result.params.index
        assert result.params is not None

    def _create_rook_weights(self, rows, cols):
        """Create rook contiguity weight matrix (only horizontal/vertical neighbors)."""
        N = rows * cols
        W = np.zeros((N, N))

        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                neighbors = []

                # Only horizontal and vertical neighbors (no diagonals)
                if j > 0:
                    neighbors.append(i * cols + (j - 1))
                if j < cols - 1:
                    neighbors.append(i * cols + (j + 1))
                if i > 0:
                    neighbors.append((i - 1) * cols + j)
                if i < rows - 1:
                    neighbors.append((i + 1) * cols + j)

                for neighbor in neighbors:
                    W[idx, neighbor] = 1

        # Row-normalize
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1
        W = W / row_sums[:, np.newaxis]

        return W

    def _create_distance_weights(self, rows, cols, cutoff):
        """Create distance-based weight matrix."""
        N = rows * cols
        W = np.zeros((N, N))

        # Create coordinates
        coords = []
        for i in range(rows):
            for j in range(cols):
                coords.append([i, j])
        coords = np.array(coords)

        # Compute distances and weights
        for i in range(N):
            for j in range(N):
                if i != j:
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist <= cutoff:
                        W[i, j] = 1 / dist

        # Row-normalize
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1
        W = W / row_sums[:, np.newaxis]

        return W
