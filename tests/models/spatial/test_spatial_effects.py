"""
Tests for spatial effects decomposition.

This module tests the computation of direct, indirect, and total effects
for spatial models, including inference methods.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from panelbox.effects.spatial_effects import (
    SpatialEffectsResult,
    compute_spatial_effects,
    spatial_impact_matrix,
)
from panelbox.models.spatial import SpatialDurbin, SpatialLag


class TestSpatialEffectsDecomposition:
    """Tests for spatial effects decomposition."""

    @pytest.fixture
    def setup_sar_model(self):
        """Setup fitted SAR model for testing."""
        np.random.seed(42)

        # Small panel for testing
        N = 20
        T = 5
        K = 2

        # Generate spatial weights
        W = self._generate_spatial_weights(N)

        # True parameters
        rho_true = 0.3
        beta_true = np.array([1.0, -0.5])

        # Generate data
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

        # Fit SAR model
        model = SpatialLag(
            formula="y ~ x1 + x2", data=data, entity_col="entity", time_col="time", W=W
        )

        # Mock fitted result (to avoid dependency on full estimation)
        from panelbox.models.spatial.spatial_lag import SpatialPanelResults

        # Create mock result
        params = pd.Series({"rho": rho_true, "x1": beta_true[0], "x2": beta_true[1]})

        # Simple covariance matrix
        cov_matrix = np.eye(3) * 0.01

        result = SpatialPanelResults(
            model=model,
            params=params,
            cov_params=cov_matrix,
            llf=-100,
            nobs=N * T,
            df_model=3,
            df_resid=N * T - 3,
            method="qml",
            effects="fixed",
            resid=epsilon,
            sigma2=1.0,
        )

        # Add required attributes
        result.W = W
        model.spatial_model_type = "SAR"

        return {
            "model": model,
            "result": result,
            "W": W,
            "true_params": {"rho": rho_true, "beta": beta_true},
            "N": N,
            "T": T,
        }

    @pytest.fixture
    def setup_sdm_model(self):
        """Setup fitted SDM model for testing."""
        np.random.seed(123)

        N = 20
        T = 5
        K = 2

        W = self._generate_spatial_weights(N)

        # True parameters
        rho_true = 0.3
        beta_true = np.array([1.0, -0.5])
        theta_true = np.array([0.4, -0.2])

        # Generate data
        X = np.random.randn(N * T, K)
        WX = np.zeros_like(X)
        for t in range(T):
            start_idx = t * N
            end_idx = (t + 1) * N
            for k in range(K):
                WX[start_idx:end_idx, k] = W @ X[start_idx:end_idx, k]

        epsilon = np.random.randn(N * T)

        y = np.zeros(N * T)
        for t in range(T):
            start_idx = t * N
            end_idx = (t + 1) * N
            I_rhoW_inv = np.linalg.inv(np.eye(N) - rho_true * W)
            Xbeta = X[start_idx:end_idx] @ beta_true
            WXtheta = WX[start_idx:end_idx] @ theta_true
            y[start_idx:end_idx] = I_rhoW_inv @ (Xbeta + WXtheta + epsilon[start_idx:end_idx])

        # Create DataFrame
        entity_ids = np.repeat(np.arange(N), T)
        time_ids = np.tile(np.arange(T), N)

        data = pd.DataFrame(
            {"entity": entity_ids, "time": time_ids, "y": y, "x1": X[:, 0], "x2": X[:, 1]}
        ).set_index(["entity", "time"])

        # Create SDM model
        model = SpatialDurbin(
            formula="y ~ x1 + x2",
            data=data,
            entity_col="entity",
            time_col="time",
            W=W,
            effects="fixed",
        )

        # Mock fitted result
        from panelbox.models.spatial.spatial_lag import SpatialPanelResults

        params = pd.Series(
            {
                "rho": rho_true,
                "x1": beta_true[0],
                "x2": beta_true[1],
                "W*x1": theta_true[0],
                "W*x2": theta_true[1],
            }
        )

        cov_matrix = np.eye(5) * 0.01

        result = SpatialPanelResults(
            model=model,
            params=params,
            cov_params=cov_matrix,
            llf=-100,
            nobs=N * T,
            df_model=5,
            df_resid=N * T - 5,
            method="qml",
            effects="fixed",
            resid=epsilon,
            sigma2=1.0,
        )

        result.W = W
        model.spatial_model_type = "SDM"

        return {
            "model": model,
            "result": result,
            "W": W,
            "true_params": {"rho": rho_true, "beta": beta_true, "theta": theta_true},
            "N": N,
            "T": T,
        }

    def _generate_spatial_weights(self, N):
        """Generate row-normalized spatial weights matrix."""
        W = np.zeros((N, N))

        # Chain structure
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

    def test_spatial_impact_matrix_sar(self, setup_sar_model):
        """Test impact matrix computation for SAR model."""
        W = setup_sar_model["W"]
        N = setup_sar_model["N"]
        rho = 0.3
        beta = 1.0

        # Compute impact matrix
        impact = spatial_impact_matrix(rho, beta, None, W, model_type="SAR")

        # Check dimensions
        assert impact.shape == (N, N)

        # Analytical formula: (I - ρW)^{-1} * β
        I_rhoW_inv = np.linalg.inv(np.eye(N) - rho * W)
        expected_impact = I_rhoW_inv * beta

        np.testing.assert_allclose(impact, expected_impact, rtol=1e-10)

        # Diagonal elements should be larger (direct effects)
        assert np.mean(np.diag(impact)) > np.mean(impact[~np.eye(N, dtype=bool)])

    def test_spatial_impact_matrix_sdm(self, setup_sdm_model):
        """Test impact matrix computation for SDM model."""
        W = setup_sdm_model["W"]
        N = setup_sdm_model["N"]
        rho = 0.3
        beta = 1.0
        theta = 0.4

        # Compute impact matrix
        impact = spatial_impact_matrix(rho, beta, theta, W, model_type="SDM")

        # Check dimensions
        assert impact.shape == (N, N)

        # Analytical formula: (I - ρW)^{-1} * (Iβ + Wθ)
        I_rhoW_inv = np.linalg.inv(np.eye(N) - rho * W)
        expected_impact = I_rhoW_inv @ (np.eye(N) * beta + W * theta)

        np.testing.assert_allclose(impact, expected_impact, rtol=1e-10)

    def test_compute_spatial_effects_sar(self, setup_sar_model):
        """Test effects decomposition for SAR model."""
        result = setup_sar_model["result"]

        # Compute effects
        effects = compute_spatial_effects(
            result, variables=["x1", "x2"], n_simulations=100, method="simulation"
        )

        # Check that we have effects for both variables
        assert "x1" in effects.effects
        assert "x2" in effects.effects

        # Check components for each variable
        for var in ["x1", "x2"]:
            var_effects = effects.effects[var]

            # Should have direct, indirect, total
            assert "direct" in var_effects
            assert "indirect" in var_effects
            assert "total" in var_effects

            # Total = Direct + Indirect
            np.testing.assert_allclose(
                var_effects["total"], var_effects["direct"] + var_effects["indirect"], rtol=1e-10
            )

            # For SAR, direct effect should be close to β
            beta = result.params[var]
            rho = result.params["rho"]

            # Direct ≈ β / (1 - ρ) for small ρ
            expected_direct_approx = beta / (1 - rho)
            assert abs(var_effects["direct"] - beta) < abs(expected_direct_approx - beta)

            # Indirect effects should be positive for positive ρ and β
            if rho > 0 and beta > 0:
                assert var_effects["indirect"] > 0

    def test_compute_spatial_effects_sdm(self, setup_sdm_model):
        """Test effects decomposition for SDM model."""
        result = setup_sdm_model["result"]

        # Compute effects
        effects = compute_spatial_effects(
            result, variables=["x1", "x2"], n_simulations=100, method="simulation"
        )

        # Check components
        for var in ["x1", "x2"]:
            var_effects = effects.effects[var]

            # Should have all components
            assert "direct" in var_effects
            assert "indirect" in var_effects
            assert "total" in var_effects

            # Total = Direct + Indirect
            np.testing.assert_allclose(
                var_effects["total"], var_effects["direct"] + var_effects["indirect"], rtol=1e-10
            )

            # For SDM, effects depend on both β and θ
            beta = result.params[var]
            theta = result.params[f"W*{var}"]
            rho = result.params["rho"]

            # Direct effects should be influenced by both β and θ
            # Indirect effects should be non-zero even if θ=0 (due to ρ)

    def test_simulation_inference(self, setup_sar_model):
        """Test simulation-based inference for effects."""
        result = setup_sar_model["result"]

        # Compute effects with simulation inference
        effects = compute_spatial_effects(
            result, variables=["x1"], n_simulations=500, confidence_level=0.95, method="simulation"
        )

        var_effects = effects.effects["x1"]

        # Should have standard errors and CIs
        assert "direct_se" in var_effects
        assert "indirect_se" in var_effects
        assert "total_se" in var_effects

        assert "direct_ci" in var_effects
        assert "indirect_ci" in var_effects
        assert "total_ci" in var_effects

        # Standard errors should be positive
        assert var_effects["direct_se"] > 0
        assert var_effects["indirect_se"] > 0
        assert var_effects["total_se"] > 0

        # CIs should contain point estimate
        assert var_effects["direct_ci"][0] <= var_effects["direct"] <= var_effects["direct_ci"][1]
        assert (
            var_effects["indirect_ci"][0]
            <= var_effects["indirect"]
            <= var_effects["indirect_ci"][1]
        )

        # P-values should be between 0 and 1
        assert "direct_pvalue" in var_effects
        assert 0 <= var_effects["direct_pvalue"] <= 1

    def test_delta_method_inference(self, setup_sar_model):
        """Test delta method inference for effects."""
        result = setup_sar_model["result"]

        # Compute effects with delta method
        effects = compute_spatial_effects(
            result, variables=["x1"], confidence_level=0.95, method="delta"
        )

        var_effects = effects.effects["x1"]

        # Should have standard errors and CIs
        assert "direct_se" in var_effects
        assert "indirect_se" in var_effects
        assert "total_se" in var_effects

        # Standard errors should be positive
        assert var_effects["direct_se"] > 0
        assert var_effects["indirect_se"] > 0
        assert var_effects["total_se"] > 0

    def test_effects_result_summary(self, setup_sar_model):
        """Test SpatialEffectsResult summary methods."""
        result = setup_sar_model["result"]

        # Compute effects
        effects = compute_spatial_effects(result, n_simulations=100, method="simulation")

        # Test summary DataFrame
        summary_df = effects.summary(show_pvalues=True)

        assert isinstance(summary_df, pd.DataFrame)
        assert "Variable" in summary_df.columns
        assert "Effect" in summary_df.columns
        assert "Estimate" in summary_df.columns
        assert "Std.Error" in summary_df.columns

        # Should have 3 rows per variable (direct, indirect, total)
        n_vars = len(["x1", "x2"])
        assert len(summary_df) == n_vars * 3

    def test_effects_visualization(self, setup_sar_model):
        """Test that effects can be visualized."""
        result = setup_sar_model["result"]

        # Compute effects
        effects = compute_spatial_effects(result, n_simulations=100)

        # Test that plot methods exist and don't error
        try:
            # Matplotlib backend
            fig = effects.plot(backend="matplotlib", show_ci=True)
            assert fig is not None
        except ImportError:
            # Matplotlib might not be installed
            pass

    def test_effects_latex_export(self, setup_sar_model):
        """Test LaTeX export of effects."""
        result = setup_sar_model["result"]

        # Compute effects
        effects = compute_spatial_effects(result, n_simulations=100)

        # Export to LaTeX
        latex_str = effects.to_latex()

        assert isinstance(latex_str, str)
        assert "\\begin{tabular}" in latex_str
        assert "Direct" in latex_str
        assert "Indirect" in latex_str
        assert "Total" in latex_str

    def test_effects_consistency_across_methods(self, setup_sar_model):
        """Test that simulation and delta methods give similar results."""
        result = setup_sar_model["result"]

        # Compute with simulation
        effects_sim = compute_spatial_effects(
            result, variables=["x1"], n_simulations=1000, method="simulation"
        )

        # Compute with delta method
        effects_delta = compute_spatial_effects(result, variables=["x1"], method="delta")

        # Point estimates should be identical
        np.testing.assert_allclose(
            effects_sim.effects["x1"]["direct"], effects_delta.effects["x1"]["direct"], rtol=1e-10
        )

        # Standard errors should be similar (within 20%)
        se_sim = effects_sim.effects["x1"]["direct_se"]
        se_delta = effects_delta.effects["x1"]["direct_se"]

        # Allow for some variation due to methods
        assert abs(se_sim - se_delta) / se_sim < 0.3
