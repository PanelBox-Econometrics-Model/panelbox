"""
Tests for Dynamic Spatial Panel Model.

This module tests the Dynamic Spatial Panel implementation including:
- GMM estimation
- Temporal and spatial lag creation
- Instrument construction
- Hansen J-test
- Impulse response functions
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from panelbox.models.spatial import DynamicSpatialPanel


class TestDynamicSpatialPanel:
    """Test suite for Dynamic Spatial Panel model."""

    @pytest.fixture
    def dynamic_spatial_data(self):
        """Generate dynamic spatial panel data."""
        np.random.seed(42)

        # Dimensions
        N = 25  # 5x5 grid
        T = 15  # Need more periods for dynamics
        K = 2  # Exogenous variables

        # Create spatial weight matrix
        W = self._create_queen_weights(5, 5)

        # True parameters
        gamma_true = 0.3  # Temporal persistence
        rho_true = 0.4  # Spatial spillover
        beta_true = np.array([1.5, -1.0])

        # Generate data following dynamic spatial DGP
        X = np.random.randn(N * T, K)
        y = np.zeros(N * T)

        # Initial period (no lag available)
        X_0 = X[:N]
        eps_0 = np.random.randn(N)
        y_0 = np.linalg.inv(np.eye(N) - rho_true * W) @ (X_0 @ beta_true + eps_0)
        y[:N] = y_0

        # Generate remaining periods
        for t in range(1, T):
            start_idx = t * N
            end_idx = (t + 1) * N

            # Previous period y
            y_lag = y[(t - 1) * N : t * N]

            # Current period X
            X_t = X[start_idx:end_idx]

            # Error term
            eps_t = np.random.randn(N) * 0.5

            # Generate y_t: (I - ρW)y_t = γy_{t-1} + Xβ + ε
            rhs = gamma_true * y_lag + X_t @ beta_true + eps_t
            y_t = np.linalg.inv(np.eye(N) - rho_true * W) @ rhs
            y[start_idx:end_idx] = y_t

        # Create DataFrame
        entities = np.repeat(np.arange(N), T)
        time_periods = np.tile(np.arange(T), N)

        data = pd.DataFrame(
            {"entity": entities, "time": time_periods, "y": y, "x1": X[:, 0], "x2": X[:, 1]}
        )

        data = data.set_index(["entity", "time"])

        return {
            "data": data,
            "W": W,
            "gamma_true": gamma_true,
            "rho_true": rho_true,
            "beta_true": beta_true,
            "N": N,
            "T": T,
        }

    def _create_queen_weights(self, rows, cols):
        """Create queen contiguity weight matrix."""
        N = rows * cols
        W = np.zeros((N, N))

        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                neighbors = []

                # All 8 neighbors (queen contiguity)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            neighbors.append(ni * cols + nj)

                for neighbor in neighbors:
                    W[idx, neighbor] = 1

        # Row-normalize
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1
        W = W / row_sums[:, np.newaxis]

        return W

    def test_dynamic_spatial_initialization(self, dynamic_spatial_data):
        """Test Dynamic Spatial Panel initialization."""
        data = dynamic_spatial_data["data"]
        W = dynamic_spatial_data["W"]

        model = DynamicSpatialPanel(
            formula="y ~ x1 + x2",
            data=data.reset_index(),
            entity_col="entity",
            time_col="time",
            W=W,
        )

        assert model.model_type == "Dynamic Spatial Panel"
        assert model.W_normalized is not None
        assert model.gamma is None  # Not fitted yet
        assert model.rho is None

    def test_temporal_lag_creation(self, dynamic_spatial_data):
        """Test creation of temporal lags."""
        data = dynamic_spatial_data["data"]
        W = dynamic_spatial_data["W"]
        N = dynamic_spatial_data["N"]
        T = dynamic_spatial_data["T"]

        model = DynamicSpatialPanel(
            formula="y ~ x1 + x2",
            data=data.reset_index(),
            entity_col="entity",
            time_col="time",
            W=W,
        )

        # Create temporal lag
        y = data["y"].values
        y_lag = model._create_temporal_lag(y, N, T, lags=1)

        # Check dimensions
        assert y_lag.shape == y.shape

        # Check that lag is correct
        # For t=1, entity 0: y_lag should equal y from t=0, entity 0
        assert_allclose(y_lag[N], y[0])  # First entity, second period

        # First period should have zeros (no lag available)
        assert_allclose(y_lag[:N], np.zeros(N))

    def test_spatial_lag_creation(self, dynamic_spatial_data):
        """Test creation of spatial lags."""
        data = dynamic_spatial_data["data"]
        W = dynamic_spatial_data["W"]
        N = dynamic_spatial_data["N"]
        T = dynamic_spatial_data["T"]

        model = DynamicSpatialPanel(
            formula="y ~ x1 + x2",
            data=data.reset_index(),
            entity_col="entity",
            time_col="time",
            W=W,
        )

        # Create spatial lag
        y = data["y"].values
        Wy = model._create_spatial_lag(y, N, T)

        # Check dimensions
        assert Wy.shape == y.shape

        # Check that spatial lag is computed correctly
        # For first time period
        y_0 = y[:N]
        Wy_0_expected = W @ y_0
        assert_allclose(Wy[:N], Wy_0_expected)

    def test_instrument_construction(self, dynamic_spatial_data):
        """Test GMM instrument construction."""
        data = dynamic_spatial_data["data"]
        W = dynamic_spatial_data["W"]
        N = dynamic_spatial_data["N"]
        T = dynamic_spatial_data["T"]

        model = DynamicSpatialPanel(
            formula="y ~ x1 + x2",
            data=data.reset_index(),
            entity_col="entity",
            time_col="time",
            W=W,
        )

        # Prepare data
        y, X = model.prepare_data("fixed")

        # Construct instruments
        Z = model._construct_instruments(y=y, X=X, N=N, T=T, lags=1, spatial_lags=2, time_lags=3)

        # Check dimensions
        assert Z.shape[0] == N * T  # Same number of observations

        # Should have multiple instruments:
        # - Lagged y (t-2, t-3)
        # - X
        # - WX, W²X
        min_instruments = X.shape[1] + 2 + 2 * X.shape[1]  # X + 2 lags of y + WX + W²X
        assert Z.shape[1] >= min_instruments

    def test_gmm_estimation(self, dynamic_spatial_data):
        """Test GMM estimation of Dynamic Spatial Panel."""
        data = dynamic_spatial_data["data"]
        W = dynamic_spatial_data["W"]

        model = DynamicSpatialPanel(
            formula="y ~ x1 + x2",
            data=data.reset_index(),
            entity_col="entity",
            time_col="time",
            W=W,
        )

        # Fit model with GMM
        result = model.fit(
            effects="fixed", method="gmm", lags=1, spatial_lags=2, time_lags=3, verbose=False
        )

        # Check that parameters are estimated
        assert "gamma" in result.params.index
        assert "rho" in result.params.index
        assert "beta_0" in result.params.index
        assert "beta_1" in result.params.index

        # Check parameter values are reasonable
        gamma_est = result.params.loc["gamma", "coefficient"]
        rho_est = result.params.loc["rho", "coefficient"]

        assert -1 < gamma_est < 1  # Stationarity
        assert -1 < rho_est < 1  # Spatial stationarity

        # Parameters should be somewhat close to true values
        # (allowing for bias due to fixed effects and finite sample)
        gamma_true = dynamic_spatial_data["gamma_true"]
        rho_true = dynamic_spatial_data["rho_true"]

        assert abs(gamma_est - gamma_true) < 0.3
        assert abs(rho_est - rho_true) < 0.3

    def test_hansen_j_test(self, dynamic_spatial_data):
        """Test Hansen J-test for overidentifying restrictions."""
        data = dynamic_spatial_data["data"]
        W = dynamic_spatial_data["W"]

        model = DynamicSpatialPanel(
            formula="y ~ x1 + x2",
            data=data.reset_index(),
            entity_col="entity",
            time_col="time",
            W=W,
        )

        # Fit with many instruments
        result = model.fit(
            effects="fixed",
            method="gmm",
            lags=1,
            spatial_lags=3,  # More spatial lags = more instruments
            time_lags=4,  # More time lags = more instruments
            verbose=False,
        )

        # Check J-test results
        assert hasattr(result, "j_statistic")
        assert hasattr(result, "j_pvalue")
        assert hasattr(result, "n_instruments")

        # J-statistic should be non-negative
        assert result.j_statistic >= 0 or np.isnan(result.j_statistic)

        # P-value should be between 0 and 1 (if not NaN)
        if not np.isnan(result.j_pvalue):
            assert 0 <= result.j_pvalue <= 1

    def test_impulse_response(self, dynamic_spatial_data):
        """Test spatial-temporal impulse response function."""
        data = dynamic_spatial_data["data"]
        W = dynamic_spatial_data["W"]
        N = dynamic_spatial_data["N"]

        model = DynamicSpatialPanel(
            formula="y ~ x1 + x2",
            data=data.reset_index(),
            entity_col="entity",
            time_col="time",
            W=W,
        )

        # Fit model
        result = model.fit(effects="fixed", method="gmm", lags=1, verbose=False)

        # Compute impulse response
        shock_entity = 12  # Middle of 5x5 grid
        periods = 10

        irf = model.compute_impulse_response(shock_entity=shock_entity, periods=periods)

        # Check dimensions
        assert irf.shape == (periods, N)

        # Initial shock should be at specified entity
        assert irf[0, shock_entity] == 1
        assert np.sum(irf[0]) == 1  # Only shocked entity

        # Response should decay over time (stability)
        total_response = np.sum(np.abs(irf), axis=1)
        assert total_response[-1] < total_response[0]

        # Spatial spillovers: neighbors should be affected
        # Find neighbors of shocked entity
        row = shock_entity // 5
        col = shock_entity % 5
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 5 and 0 <= new_col < 5:
                    neighbors.append(new_row * 5 + new_col)

        # Neighbors should show response in period 1
        neighbor_response = np.sum([irf[1, n] for n in neighbors])
        assert neighbor_response > 0

    def test_model_with_no_temporal_lag(self, dynamic_spatial_data):
        """Test that model reduces to spatial lag when gamma=0."""
        data = dynamic_spatial_data["data"]
        W = dynamic_spatial_data["W"]
        N = dynamic_spatial_data["N"]

        # Generate data without temporal dependence (gamma=0)
        T = 10
        X = np.random.randn(N * T, 2)
        beta = np.array([1.0, -0.5])
        rho = 0.5
        y = np.zeros(N * T)

        for t in range(T):
            start_idx = t * N
            end_idx = (t + 1) * N
            X_t = X[start_idx:end_idx]
            eps = np.random.randn(N) * 0.5
            y_t = np.linalg.inv(np.eye(N) - rho * W) @ (X_t @ beta + eps)
            y[start_idx:end_idx] = y_t

        static_data = pd.DataFrame(
            {
                "entity": np.repeat(np.arange(N), T),
                "time": np.tile(np.arange(T), N),
                "y": y,
                "x1": X[:, 0],
                "x2": X[:, 1],
            }
        ).set_index(["entity", "time"])

        model = DynamicSpatialPanel(
            formula="y ~ x1 + x2",
            data=static_data.reset_index(),
            entity_col="entity",
            time_col="time",
            W=W,
        )

        result = model.fit(effects="fixed", method="gmm", lags=1, verbose=False)

        # Gamma should be close to zero
        gamma_est = result.params.loc["gamma", "coefficient"]
        assert abs(gamma_est) < 0.2  # Should be small

        # Rho should be close to true value
        rho_est = result.params.loc["rho", "coefficient"]
        assert abs(rho_est - rho) < 0.2

    def test_prediction(self, dynamic_spatial_data):
        """Test multi-step prediction."""
        data = dynamic_spatial_data["data"]
        W = dynamic_spatial_data["W"]

        model = DynamicSpatialPanel(
            formula="y ~ x1 + x2",
            data=data.reset_index(),
            entity_col="entity",
            time_col="time",
            W=W,
        )

        # Fit model
        result = model.fit(effects="fixed", method="gmm", lags=1, verbose=False)

        # Store result for prediction
        model.last_result = result

        # Predict next periods
        predictions = model.predict(steps=3)

        # Check dimensions
        N = dynamic_spatial_data["N"]
        assert predictions.shape == (3, N)

        # Predictions should be finite
        assert np.all(np.isfinite(predictions))
