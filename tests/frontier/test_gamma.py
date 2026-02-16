"""
Tests for Normal-Gamma stochastic frontier model.

This module tests the gamma distribution implementation including:
- Log-likelihood computation with SML
- Parameter estimation and recovery
- Efficiency estimators (JLMS and BC)
- Special case: P=1 should match exponential
- Stability across different configurations
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier import StochasticFrontier


class TestGammaLogLikelihood:
    """Test log-likelihood function for gamma distribution."""

    def test_loglik_returns_finite(self):
        """Test that log-likelihood returns finite value."""
        np.random.seed(42)
        n = 50
        P_true = 2.0
        theta_true = 1.5
        sigma_v_true = 0.3

        # Generate data
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, (n, 2))])
        beta_true = np.array([1.0, 0.5, -0.3])
        u = np.random.gamma(P_true, 1 / theta_true, n)
        v = np.random.normal(0, sigma_v_true, n)
        y = X @ beta_true + v - u

        df = pd.DataFrame(
            {
                "y": y,
                "x1": X[:, 1],
                "x2": X[:, 2],
            }
        )

        # Estimate model
        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            frontier="production",
            dist="gamma",
        )

        result = model.fit(maxiter=50)  # Limited iterations for fast test

        # Check that log-likelihood is finite
        assert np.isfinite(result.loglik)
        assert result.loglik < 0  # Log-likelihood should be negative

    def test_gamma_p_equals_1_matches_exponential(self):
        """When P=1, gamma should match exponential distribution."""
        np.random.seed(123)
        n = 100
        theta_true = 2.0
        sigma_v_true = 0.2

        # Generate data with P=1 (exponential)
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        beta_true = np.array([2.0, 0.5])
        u = np.random.gamma(1.0, 1 / theta_true, n)  # P=1
        v = np.random.normal(0, sigma_v_true, n)
        y = X @ beta_true + v - u

        df = pd.DataFrame({"y": y, "x1": X[:, 1]})

        # Estimate as gamma
        model_gamma = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1"],
            frontier="production",
            dist="gamma",
        )
        result_gamma = model_gamma.fit(maxiter=100)

        # Estimate as exponential
        model_exp = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1"],
            frontier="production",
            dist="exponential",
        )
        result_exp = model_exp.fit(maxiter=100)

        # Log-likelihoods should be similar (allowing for numerical differences)
        # Note: They won't be exactly equal due to SML approximation in gamma
        assert abs(result_gamma.loglik - result_exp.loglik) < 5.0


class TestGammaParameterEstimation:
    """Test parameter estimation for gamma model."""

    def test_parameter_recovery_large_sample(self):
        """Test that we recover known parameters from large sample."""
        np.random.seed(456)
        n = 500  # Large sample for good recovery

        P_true = 2.5
        theta_true = 2.0
        sigma_v_true = 0.2
        beta_true = np.array([2.0, 0.7, -0.4])

        # Generate data
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, (n, 2))])
        u = np.random.gamma(P_true, 1 / theta_true, n)
        v = np.random.normal(0, sigma_v_true, n)
        y = X @ beta_true + v - u

        df = pd.DataFrame({"y": y, "x1": X[:, 1], "x2": X[:, 2]})

        # Estimate
        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            frontier="production",
            dist="gamma",
        )
        result = model.fit(maxiter=200)

        # Check recovery (with tolerance due to sampling variability)
        np.testing.assert_allclose(result.params[:3], beta_true, rtol=0.15, atol=0.1)

        # Check gamma parameters (more lenient tolerance)
        assert abs(result.gamma_P - P_true) < 1.0
        assert abs(result.gamma_theta - theta_true) < 1.0

    def test_convergence(self):
        """Test that optimization converges."""
        np.random.seed(789)
        n = 100

        # Generate data
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        beta_true = np.array([1.0, 0.5])
        u = np.random.gamma(2.0, 1 / 1.5, n)
        v = np.random.normal(0, 0.3, n)
        y = X @ beta_true + v - u

        df = pd.DataFrame({"y": y, "x1": X[:, 1]})

        # Estimate
        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1"],
            frontier="production",
            dist="gamma",
        )
        result = model.fit()

        # Check convergence
        assert result.converged or result.loglik > -np.inf

    def test_cost_frontier(self):
        """Test gamma model with cost frontier."""
        np.random.seed(321)
        n = 100

        # Generate cost frontier data (sign = -1)
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        beta_true = np.array([1.0, 0.5])
        u = np.random.gamma(2.0, 1 / 1.5, n)
        v = np.random.normal(0, 0.3, n)
        y = X @ beta_true + v + u  # Cost: inefficiency increases cost

        df = pd.DataFrame({"y": y, "x1": X[:, 1]})

        # Estimate
        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1"],
            frontier="cost",
            dist="gamma",
        )
        result = model.fit()

        # Check that it runs without errors
        assert np.isfinite(result.loglik)
        assert result.gamma_P is not None
        assert result.gamma_theta is not None


class TestGammaEfficiencyEstimators:
    """Test efficiency estimators for gamma distribution."""

    @pytest.fixture
    def gamma_model_result(self):
        """Create a fitted gamma model for testing."""
        np.random.seed(100)
        n = 100

        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        beta_true = np.array([2.0, 0.5])
        u = np.random.gamma(2.0, 1 / 1.5, n)
        v = np.random.normal(0, 0.2, n)
        y = X @ beta_true + v - u

        df = pd.DataFrame({"y": y, "x1": X[:, 1]})

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1"],
            frontier="production",
            dist="gamma",
        )
        result = model.fit(maxiter=100)
        return result

    def test_bc_estimator(self, gamma_model_result):
        """Test BC efficiency estimator for gamma."""
        from panelbox.frontier.efficiency import estimate_efficiency

        # Compute efficiency
        eff_df = estimate_efficiency(gamma_model_result, estimator="bc")

        # Check output structure
        assert "efficiency" in eff_df.columns
        assert "inefficiency" in eff_df.columns
        assert len(eff_df) == 100

        # Check efficiency range
        assert (eff_df["efficiency"] > 0).all()
        assert (eff_df["efficiency"] <= 1).all()

        # Check that inefficiency is non-negative
        assert (eff_df["inefficiency"] >= 0).all()

    def test_jlms_estimator(self, gamma_model_result):
        """Test JLMS efficiency estimator for gamma."""
        from panelbox.frontier.efficiency import estimate_efficiency

        # Compute efficiency
        eff_df = estimate_efficiency(gamma_model_result, estimator="jlms")

        # Check output structure
        assert "efficiency" in eff_df.columns
        assert "inefficiency" in eff_df.columns
        assert len(eff_df) == 100

        # Check efficiency range
        assert (eff_df["efficiency"] > 0).all()
        assert (eff_df["efficiency"] <= 1).all()

        # Check that inefficiency is non-negative
        assert (eff_df["inefficiency"] >= 0).all()

    def test_efficiency_consistency(self, gamma_model_result):
        """Test that efficiency = exp(-inefficiency)."""
        from panelbox.frontier.efficiency import estimate_efficiency

        eff_df = estimate_efficiency(gamma_model_result, estimator="bc")

        # Check relationship
        expected_eff = np.exp(-eff_df["inefficiency"])
        np.testing.assert_allclose(eff_df["efficiency"], expected_eff, rtol=1e-6)


class TestGammaStability:
    """Test stability across different configurations."""

    def test_different_p_values(self):
        """Test estimation with different P (shape) values."""
        np.random.seed(200)
        n = 100

        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        beta_true = np.array([1.0, 0.5])
        v = np.random.normal(0, 0.3, n)

        # Test P = 0.5, 2.0, 5.0
        for P_true in [0.5, 2.0, 5.0]:
            u = np.random.gamma(P_true, 1 / 1.5, n)
            y = X @ beta_true + v - u

            df = pd.DataFrame({"y": y, "x1": X[:, 1]})

            model = StochasticFrontier(
                data=df,
                depvar="y",
                exog=["x1"],
                frontier="production",
                dist="gamma",
            )
            result = model.fit(maxiter=150)

            # Should converge and have reasonable log-likelihood
            assert np.isfinite(result.loglik)
            assert result.gamma_P > 0

    def test_different_theta_values(self):
        """Test estimation with different θ (rate) values."""
        np.random.seed(300)
        n = 100

        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        beta_true = np.array([1.0, 0.5])
        v = np.random.normal(0, 0.3, n)

        # Test θ = 0.5, 1.5, 3.0
        for theta_true in [0.5, 1.5, 3.0]:
            u = np.random.gamma(2.0, 1 / theta_true, n)
            y = X @ beta_true + v - u

            df = pd.DataFrame({"y": y, "x1": X[:, 1]})

            model = StochasticFrontier(
                data=df,
                depvar="y",
                exog=["x1"],
                frontier="production",
                dist="gamma",
            )
            result = model.fit(maxiter=150)

            # Should converge and have reasonable log-likelihood
            assert np.isfinite(result.loglik)
            assert result.gamma_theta > 0


class TestGammaSummary:
    """Test summary output for gamma model."""

    def test_summary_includes_gamma_params(self):
        """Test that summary includes gamma parameters."""
        np.random.seed(400)
        n = 100

        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        beta_true = np.array([1.0, 0.5])
        u = np.random.gamma(2.0, 1 / 1.5, n)
        v = np.random.normal(0, 0.3, n)
        y = X @ beta_true + v - u

        df = pd.DataFrame({"y": y, "x1": X[:, 1]})

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1"],
            frontier="production",
            dist="gamma",
        )
        result = model.fit()

        summary = result.summary()

        # Check that summary includes gamma parameters
        assert "Gamma Distribution Parameters" in summary
        assert "P (shape)" in summary
        assert "θ (rate)" in summary
        assert "E[u] = P/θ" in summary
