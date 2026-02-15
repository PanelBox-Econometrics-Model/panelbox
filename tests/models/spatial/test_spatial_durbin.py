"""
Tests for Spatial Durbin Model (SDM) implementation.

This module tests the SDM estimation for both fixed and random effects,
parameter recovery, and convergence properties.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.sparse import csr_matrix

from panelbox.core.spatial_weights import SpatialWeights
from panelbox.models.spatial import SpatialDurbin, SpatialPanelResults


class TestSpatialDurbinModel:
    """Tests for Spatial Durbin Model estimation."""

    @pytest.fixture
    def setup_sdm_data(self):
        """Generate simulated panel data with SDM structure."""
        np.random.seed(42)

        # Panel dimensions
        N = 50  # Number of entities
        T = 10  # Number of time periods
        K = 3  # Number of covariates

        # Generate spatial weights matrix (rook contiguity for grid)
        W = self._generate_spatial_weights(N)

        # True parameters
        rho_true = 0.4
        beta_true = np.array([1.5, -0.8, 0.3])
        theta_true = np.array([0.6, -0.3, 0.2])  # Spatial spillover coefficients
        sigma2_true = 1.0

        # Generate exogenous variables
        X = np.random.randn(N * T, K)

        # Entity fixed effects
        alpha = np.random.randn(N)
        alpha_panel = np.repeat(alpha, T)

        # Spatial lag of X
        WX = np.zeros_like(X)
        for t in range(T):
            start_idx = t * N
            end_idx = (t + 1) * N
            for k in range(K):
                WX[start_idx:end_idx, k] = W @ X[start_idx:end_idx, k]

        # Generate y according to SDM model
        # y = ρWy + Xβ + WXθ + α + ε
        epsilon = np.random.randn(N * T) * np.sqrt(sigma2_true)

        # Initialize y
        y = np.zeros(N * T)

        # Generate y by time period (accounting for spatial dependence)
        for t in range(T):
            start_idx = t * N
            end_idx = (t + 1) * N

            # Construct (I - ρW)^{-1}
            I_rhoW_inv = np.linalg.inv(np.eye(N) - rho_true * W)

            # Direct and spillover effects
            Xbeta = X[start_idx:end_idx] @ beta_true
            WXtheta = WX[start_idx:end_idx] @ theta_true

            # Generate y_t
            y_t = I_rhoW_inv @ (Xbeta + WXtheta + alpha + epsilon[start_idx:end_idx])
            y[start_idx:end_idx] = y_t

        # Create panel DataFrame
        entity_ids = np.repeat(np.arange(N), T)
        time_ids = np.tile(np.arange(T), N)

        data = pd.DataFrame(
            {
                "entity": entity_ids,
                "time": time_ids,
                "y": y,
                "x1": X[:, 0],
                "x2": X[:, 1],
                "x3": X[:, 2],
            }
        )

        # Set MultiIndex
        data = data.set_index(["entity", "time"])

        return {
            "data": data,
            "W": W,
            "true_params": {
                "rho": rho_true,
                "beta": beta_true,
                "theta": theta_true,
                "sigma2": sigma2_true,
            },
            "N": N,
            "T": T,
            "K": K,
        }

    def _generate_spatial_weights(self, N):
        """Generate row-normalized spatial weights matrix."""
        # Simple nearest neighbor structure
        W = np.zeros((N, N))

        # Create chain structure (each unit connected to neighbors)
        for i in range(N):
            if i > 0:
                W[i, i - 1] = 1
            if i < N - 1:
                W[i, i + 1] = 1

        # Row normalize
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1
        W = W / row_sums[:, np.newaxis]

        return W

    def test_sdm_fixed_effects_estimation(self, setup_sdm_data):
        """Test SDM estimation with fixed effects."""
        data = setup_sdm_data["data"]
        W = setup_sdm_data["W"]
        true_params = setup_sdm_data["true_params"]

        # Create model
        model = SpatialDurbin(
            formula="y ~ x1 + x2 + x3",
            data=data,
            entity_col="entity",
            time_col="time",
            W=W,
            effects="fixed",
        )

        # Fit model
        result = model.fit(method="qml")

        # Check that model converged
        assert isinstance(result, SpatialPanelResults)
        assert result.params is not None

        # Check parameter recovery (with some tolerance)
        rho_est = result.params["rho"]
        assert (
            abs(rho_est - true_params["rho"]) < 0.15
        ), f"rho estimate {rho_est:.3f} far from true value {true_params['rho']:.3f}"

        # Check that we have both β and θ parameters
        assert "x1" in result.params
        assert "W*x1" in result.params
        assert "x2" in result.params
        assert "W*x2" in result.params

        # Check signs are correct
        assert result.params["x1"] > 0  # True β1 = 1.5
        assert result.params["x2"] < 0  # True β2 = -0.8
        assert result.params["W*x1"] > 0  # True θ1 = 0.6

    def test_sdm_random_effects_estimation(self, setup_sdm_data):
        """Test SDM estimation with random effects."""
        data = setup_sdm_data["data"]
        W = setup_sdm_data["W"]

        # Create model
        model = SpatialDurbin(
            formula="y ~ x1 + x2 + x3",
            data=data,
            entity_col="entity",
            time_col="time",
            W=W,
            effects="random",
        )

        # Fit model
        result = model.fit(method="ml")

        # Check that model converged
        assert isinstance(result, SpatialPanelResults)
        assert result.params is not None

        # Check that we have variance components
        assert "sigma_alpha" in result.params
        assert "sigma_epsilon" in result.params
        assert result.params["sigma_alpha"] > 0
        assert result.params["sigma_epsilon"] > 0

    def test_sdm_vs_sar_nested(self, setup_sdm_data):
        """Test that SDM nests SAR when θ=0."""
        # Generate data with θ=0 (SAR case)
        np.random.seed(123)
        N = 30
        T = 8
        K = 2

        W = self._generate_spatial_weights(N)
        rho_true = 0.3
        beta_true = np.array([1.0, -0.5])

        # Generate data without spatial spillovers (θ=0)
        X = np.random.randn(N * T, K)
        epsilon = np.random.randn(N * T)

        y = np.zeros(N * T)
        for t in range(T):
            start_idx = t * N
            end_idx = (t + 1) * N
            I_rhoW_inv = np.linalg.inv(np.eye(N) - rho_true * W)
            Xbeta = X[start_idx:end_idx] @ beta_true
            y[start_idx:end_idx] = I_rhoW_inv @ (Xbeta + epsilon[start_idx:end_idx])

        # Create DataFrame
        entity_ids = np.repeat(np.arange(N), T)
        time_ids = np.tile(np.arange(T), N)

        data = pd.DataFrame(
            {"entity": entity_ids, "time": time_ids, "y": y, "x1": X[:, 0], "x2": X[:, 1]}
        ).set_index(["entity", "time"])

        # Fit SDM
        model_sdm = SpatialDurbin(
            formula="y ~ x1 + x2",
            data=data,
            entity_col="entity",
            time_col="time",
            W=W,
            effects="fixed",
        )
        result_sdm = model_sdm.fit(method="qml")

        # Check that θ estimates are close to zero
        theta1 = result_sdm.params.get("W*x1", 0)
        theta2 = result_sdm.params.get("W*x2", 0)

        assert abs(theta1) < 0.2, f"θ1 should be near 0, got {theta1:.3f}"
        assert abs(theta2) < 0.2, f"θ2 should be near 0, got {theta2:.3f}"

    def test_sdm_prediction(self, setup_sdm_data):
        """Test prediction with fitted SDM."""
        data = setup_sdm_data["data"]
        W = setup_sdm_data["W"]

        # Fit model
        model = SpatialDurbin(
            formula="y ~ x1 + x2 + x3",
            data=data,
            entity_col="entity",
            time_col="time",
            W=W,
            effects="fixed",
        )
        result = model.fit(method="qml")

        # Make predictions
        y_pred_direct = model.predict(effects_type="direct")
        y_pred_total = model.predict(effects_type="total")

        # Check predictions are reasonable
        assert len(y_pred_direct) == len(data)
        assert len(y_pred_total) == len(data)
        assert not np.any(np.isnan(y_pred_direct))
        assert not np.any(np.isnan(y_pred_total))

        # Total effects should generally be larger in magnitude than direct
        # due to spatial multiplier
        assert np.abs(y_pred_total).mean() >= np.abs(y_pred_direct).mean()

    def test_sdm_parameter_bounds(self, setup_sdm_data):
        """Test that spatial parameter respects theoretical bounds."""
        data = setup_sdm_data["data"]
        W = setup_sdm_data["W"]

        model = SpatialDurbin(
            formula="y ~ x1 + x2 + x3",
            data=data,
            entity_col="entity",
            time_col="time",
            W=W,
            effects="fixed",
        )

        # Get bounds
        rho_min, rho_max = model._spatial_coefficient_bounds()

        # Fit model
        result = model.fit(method="qml")

        # Check that estimated rho is within bounds
        rho_est = result.params["rho"]
        assert (
            rho_min <= rho_est <= rho_max
        ), f"rho {rho_est:.3f} outside bounds [{rho_min:.3f}, {rho_max:.3f}]"

    def test_sdm_sparse_weights(self, setup_sdm_data):
        """Test SDM with sparse weight matrix."""
        data = setup_sdm_data["data"]
        W_dense = setup_sdm_data["W"]

        # Convert to sparse
        W_sparse = csr_matrix(W_dense)

        # Fit with sparse W
        model = SpatialDurbin(
            formula="y ~ x1 + x2 + x3",
            data=data,
            entity_col="entity",
            time_col="time",
            W=SpatialWeights.from_matrix(W_sparse),
            effects="fixed",
        )

        result = model.fit(method="qml")

        # Should still work
        assert result.params is not None
        assert "rho" in result.params

    def test_sdm_statistical_inference(self, setup_sdm_data):
        """Test statistical inference for SDM parameters."""
        data = setup_sdm_data["data"]
        W = setup_sdm_data["W"]

        model = SpatialDurbin(
            formula="y ~ x1 + x2 + x3",
            data=data,
            entity_col="entity",
            time_col="time",
            W=W,
            effects="fixed",
        )

        result = model.fit(method="qml")

        # Check that we have standard errors
        assert hasattr(result, "bse")
        assert len(result.bse) == len(result.params)
        assert all(result.bse > 0)

        # Check t-values and p-values
        assert hasattr(result, "tvalues")
        assert hasattr(result, "pvalues")
        assert len(result.pvalues) == len(result.params)
        assert all(0 <= p <= 1 for p in result.pvalues)

    @pytest.mark.parametrize("effects", ["fixed", "random"])
    def test_sdm_model_fit_statistics(self, setup_sdm_data, effects):
        """Test model fit statistics for different effect types."""
        data = setup_sdm_data["data"]
        W = setup_sdm_data["W"]

        model = SpatialDurbin(
            formula="y ~ x1 + x2 + x3",
            data=data,
            entity_col="entity",
            time_col="time",
            W=W,
            effects=effects,
        )

        method = "qml" if effects == "fixed" else "ml"
        result = model.fit(method=method)

        # Check fit statistics
        assert hasattr(result, "llf")
        assert hasattr(result, "aic")
        assert hasattr(result, "bic")

        # AIC = -2*llf + 2*k
        expected_aic = -2 * result.llf + 2 * result.df_model
        assert abs(result.aic - expected_aic) < 1e-6

        # BIC = -2*llf + k*log(n)
        expected_bic = -2 * result.llf + result.df_model * np.log(result.nobs)
        assert abs(result.bic - expected_bic) < 1e-6
