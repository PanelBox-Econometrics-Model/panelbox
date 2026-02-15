"""
Tests for MLE estimation of stochastic frontier models.

Tests:
    - Parameter recovery from DGP
    - Convergence with different starting values
    - Model comparison across distributions
    - Robustness to sample size
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier import StochasticFrontier


class TestMLEEstimation:
    """Tests for MLE estimation."""

    @pytest.fixture
    def production_data(self):
        """Simulated production data with known parameters."""
        np.random.seed(42)
        n = 500

        # True parameters
        beta_true = np.array([2.0, 0.6, 0.3])
        sigma_v_true = 0.1
        sigma_u_true = 0.2

        # Generate data
        const = np.ones(n)
        log_labor = np.random.uniform(0, 3, n)
        log_capital = np.random.uniform(0, 3, n)

        X = np.column_stack([const, log_labor, log_capital])

        v = np.random.normal(0, sigma_v_true, n)
        u = np.abs(np.random.normal(0, sigma_u_true, n))

        log_output = X @ beta_true + v - u

        df = pd.DataFrame(
            {"log_output": log_output, "log_labor": log_labor, "log_capital": log_capital}
        )

        return {
            "data": df,
            "beta_true": beta_true,
            "sigma_v_true": sigma_v_true,
            "sigma_u_true": sigma_u_true,
        }

    def test_parameter_recovery_half_normal(self, production_data):
        """Test that MLE recovers true parameters (half-normal)."""
        df = production_data["data"]
        beta_true = production_data["beta_true"]
        sigma_v_true = production_data["sigma_v_true"]
        sigma_u_true = production_data["sigma_u_true"]

        # Estimate model
        sf = StochasticFrontier(
            data=df,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist="half_normal",
        )

        result = sf.fit(method="mle", verbose=False)

        # Check convergence
        assert result.converged, "Optimization did not converge"

        # Check parameter recovery (within 10% for variance components)
        # Beta parameters
        beta_est = result.params[["const", "log_labor", "log_capital"]].values
        np.testing.assert_allclose(
            beta_est,
            beta_true,
            rtol=0.1,
            atol=0.1,
            err_msg="Beta parameters not recovered accurately",
        )

        # Variance components
        assert abs(result.sigma_v - sigma_v_true) / sigma_v_true < 0.2
        assert abs(result.sigma_u - sigma_u_true) / sigma_u_true < 0.2

    def test_parameter_recovery_exponential(self, production_data):
        """Test parameter recovery with exponential distribution."""
        # Generate exponential data
        np.random.seed(123)
        n = 500
        beta_true = np.array([2.0, 0.6, 0.3])
        sigma_v_true = 0.1
        sigma_u_true = 0.15

        const = np.ones(n)
        log_labor = np.random.uniform(0, 3, n)
        log_capital = np.random.uniform(0, 3, n)
        X = np.column_stack([const, log_labor, log_capital])

        v = np.random.normal(0, sigma_v_true, n)
        u = np.random.exponential(sigma_u_true, n)

        log_output = X @ beta_true + v - u

        df = pd.DataFrame(
            {"log_output": log_output, "log_labor": log_labor, "log_capital": log_capital}
        )

        # Estimate
        sf = StochasticFrontier(
            data=df,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist="exponential",
        )

        result = sf.fit(method="mle", verbose=False)

        assert result.converged
        # Looser tolerance for exponential (harder to estimate)
        assert abs(result.sigma_v - sigma_v_true) / sigma_v_true < 0.3

    def test_convergence_with_different_starting_values(self, production_data):
        """Test that different starting values converge to same optimum."""
        df = production_data["data"]

        sf = StochasticFrontier(
            data=df,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist="half_normal",
        )

        # Estimate with default starting values
        result1 = sf.fit(method="mle", verbose=False)

        # Estimate with custom starting values
        n_params = 5  # β0, β1, β2, ln(σ²_v), ln(σ²_u)
        start_params = np.array([1.5, 0.5, 0.4, np.log(0.05), np.log(0.15)])

        result2 = sf.fit(method="mle", start_params=start_params, verbose=False)

        # Should converge to similar log-likelihood
        assert abs(result1.loglik - result2.loglik) < 0.1

    def test_cost_frontier(self):
        """Test cost frontier estimation."""
        np.random.seed(999)
        n = 500

        beta_true = np.array([1.0, 0.4, 0.5])
        sigma_v_true = 0.1
        sigma_u_true = 0.15

        const = np.ones(n)
        log_labor = np.random.uniform(0, 2, n)
        log_capital = np.random.uniform(0, 2, n)
        X = np.column_stack([const, log_labor, log_capital])

        v = np.random.normal(0, sigma_v_true, n)
        u = np.abs(np.random.normal(0, sigma_u_true, n))

        # Cost frontier: u increases cost
        log_cost = X @ beta_true + v + u

        df = pd.DataFrame(
            {"log_cost": log_cost, "log_labor": log_labor, "log_capital": log_capital}
        )

        sf = StochasticFrontier(
            data=df,
            depvar="log_cost",
            exog=["log_labor", "log_capital"],
            frontier="cost",  # Cost frontier
            dist="half_normal",
        )

        result = sf.fit(method="mle", verbose=False)

        assert result.converged
        # Basic sanity checks
        assert result.sigma_v > 0
        assert result.sigma_u > 0
        assert result.lambda_param > 0

    def test_model_type_detection(self, production_data):
        """Test automatic model type detection."""
        df = production_data["data"]

        # Cross-sectional (no entity/time)
        sf = StochasticFrontier(
            data=df,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist="half_normal",
        )

        assert sf.model_type.value == "cross_section"
        assert not sf.is_panel

    def test_summary_output(self, production_data):
        """Test that summary() produces valid output."""
        df = production_data["data"]

        sf = StochasticFrontier(
            data=df,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist="half_normal",
        )

        result = sf.fit(method="mle", verbose=False)

        summary = result.summary()

        # Check summary contains key information
        assert "Stochastic Frontier Analysis" in summary
        assert "Log-Likelihood" in summary
        assert "AIC" in summary
        assert "BIC" in summary
        # Check for variance components (either sigma_v or σ_v format)
        assert ("sigma_v" in summary.lower()) or ("σ_v" in summary.lower())
        assert ("sigma_u" in summary.lower()) or ("σ_u" in summary.lower())

    def test_compare_distributions(self, production_data):
        """Test distribution comparison functionality."""
        df = production_data["data"]

        # Estimate with two distributions
        sf_hn = StochasticFrontier(
            data=df,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist="half_normal",
        )
        result_hn = sf_hn.fit(method="mle", verbose=False)

        sf_exp = StochasticFrontier(
            data=df,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist="exponential",
        )
        result_exp = sf_exp.fit(method="mle", verbose=False)

        # Compare
        comparison = result_hn.compare_distributions([result_exp])

        assert len(comparison) == 2
        assert "Distribution" in comparison.columns
        assert "AIC" in comparison.columns
        assert "BIC" in comparison.columns


class TestRobustness:
    """Test robustness of estimation."""

    def test_small_sample(self):
        """Test estimation with small sample."""
        np.random.seed(42)
        n = 100  # Small sample

        beta_true = np.array([1.0, 0.5])
        sigma_v_true = 0.2
        sigma_u_true = 0.3

        const = np.ones(n)
        x = np.random.normal(0, 1, n)
        X = np.column_stack([const, x])

        v = np.random.normal(0, sigma_v_true, n)
        u = np.abs(np.random.normal(0, sigma_u_true, n))

        y = X @ beta_true + v - u

        df = pd.DataFrame({"y": y, "x": x})

        sf = StochasticFrontier(
            data=df, depvar="y", exog=["x"], frontier="production", dist="half_normal"
        )

        result = sf.fit(method="mle", verbose=False)

        # Should still converge (though estimates may be less precise)
        assert result.converged or result.loglik > -np.inf

    def test_large_sample(self):
        """Test estimation with large sample."""
        np.random.seed(42)
        n = 2000  # Large sample

        beta_true = np.array([1.0, 0.5, -0.3])
        sigma_v_true = 0.1
        sigma_u_true = 0.2

        const = np.ones(n)
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        X = np.column_stack([const, x1, x2])

        v = np.random.normal(0, sigma_v_true, n)
        u = np.abs(np.random.normal(0, sigma_u_true, n))

        y = X @ beta_true + v - u

        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        sf = StochasticFrontier(
            data=df, depvar="y", exog=["x1", "x2"], frontier="production", dist="half_normal"
        )

        result = sf.fit(method="mle", verbose=False)

        assert result.converged

        # With large sample, should recover parameters well
        beta_est = result.params[["const", "x1", "x2"]].values
        np.testing.assert_allclose(beta_est, beta_true, rtol=0.05, atol=0.05)


class TestModelCreation:
    """Test model creation with various parameter combinations."""

    def test_all_distribution_types(self):
        """Test model creation with all distribution types."""
        np.random.seed(42)
        n = 100

        df = pd.DataFrame(
            {
                "y": np.random.normal(0, 1, n),
                "x1": np.random.normal(0, 1, n),
                "x2": np.random.normal(0, 1, n),
            }
        )

        distributions = ["half_normal", "exponential", "truncated_normal"]

        for dist in distributions:
            sf = StochasticFrontier(
                data=df, depvar="y", exog=["x1", "x2"], frontier="production", dist=dist
            )

            assert sf.dist.value == dist
            assert sf.frontier_type.value == "production"

    def test_production_vs_cost_frontier(self):
        """Test model creation with production vs cost frontier."""
        np.random.seed(42)
        n = 100

        df = pd.DataFrame({"y": np.random.normal(0, 1, n), "x": np.random.normal(0, 1, n)})

        for frontier_type in ["production", "cost"]:
            sf = StochasticFrontier(
                data=df, depvar="y", exog=["x"], frontier=frontier_type, dist="half_normal"
            )

            assert sf.frontier_type.value == frontier_type

    def test_single_vs_multiple_exog(self):
        """Test model with single vs multiple exogenous variables."""
        np.random.seed(42)
        n = 100

        df = pd.DataFrame(
            {
                "y": np.random.normal(0, 1, n),
                "x1": np.random.normal(0, 1, n),
                "x2": np.random.normal(0, 1, n),
                "x3": np.random.normal(0, 1, n),
            }
        )

        # Single exog
        sf1 = StochasticFrontier(
            data=df, depvar="y", exog=["x1"], frontier="production", dist="half_normal"
        )

        assert len(sf1.exog) == 1

        # Multiple exog
        sf2 = StochasticFrontier(
            data=df, depvar="y", exog=["x1", "x2", "x3"], frontier="production", dist="half_normal"
        )

        assert len(sf2.exog) == 3

    def test_cross_section_model_type(self):
        """Test that cross-section model type is detected."""
        np.random.seed(42)
        n = 100

        df = pd.DataFrame({"y": np.random.normal(0, 1, n), "x": np.random.normal(0, 1, n)})

        sf = StochasticFrontier(
            data=df, depvar="y", exog=["x"], frontier="production", dist="half_normal"
        )

        assert sf.model_type.value == "cross_section"
        assert not sf.is_panel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
