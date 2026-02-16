"""
Tests for Wang (2002) heteroscedastic inefficiency model.

This module tests:
1. Basic estimation with heteroscedastic inefficiency
2. Marginal effects on location and scale
3. Special cases (reduces to BC95, half-normal)
4. Parameter recovery with simulated data
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from panelbox.frontier import StochasticFrontier


class TestWang2002Basic:
    """Basic tests for Wang (2002) model estimation."""

    def test_wang_2002_estimation(self):
        """Test basic Wang (2002) model estimation."""
        np.random.seed(42)
        n = 200

        # Generate data with heteroscedastic inefficiency
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, (n, 2))])
        Z = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])  # age
        W = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])  # size

        beta_true = np.array([2.0, 0.5, -0.3])
        delta_true = np.array([0.2, 0.5])  # Higher age → more inefficiency
        gamma_true = np.array([-1.0, 0.3])  # Bigger size → more variance

        # Generate inefficiency
        mu_i = Z @ delta_true
        ln_sigma_u_sq_i = W @ gamma_true
        sigma_u_i = np.sqrt(np.exp(ln_sigma_u_sq_i))

        u = np.abs(np.random.normal(mu_i, sigma_u_i))
        v = np.random.normal(0, 0.2, n)
        y = X @ beta_true + v - u

        df = pd.DataFrame(
            {
                "y": y,
                "x1": X[:, 1],
                "x2": X[:, 2],
                "age": Z[:, 1],
                "size": W[:, 1],
            }
        )

        # Estimate Wang (2002) model
        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1", "x2"],
            frontier="production",
            dist="truncated_normal",
            inefficiency_vars=["age"],
            het_vars=["size"],
        )

        result = model.fit(verbose=False)

        # Check convergence
        assert result.converged, "Wang (2002) model did not converge"

        # Check number of parameters
        # [β0, β1, β2, σ²_v, δ0, δ1, γ0, γ1] = 8 parameters
        assert len(result.params) == 8, f"Expected 8 parameters, got {len(result.params)}"

        # Check parameter names
        param_names = result.params.index.tolist()
        assert "const" in param_names
        assert "x1" in param_names
        assert "x2" in param_names
        assert "sigma_v_sq" in param_names
        assert "delta_const" in param_names
        assert "delta_age" in param_names
        assert "gamma_const" in param_names
        assert "gamma_size" in param_names

        # Check that log-likelihood is finite
        assert np.isfinite(result.loglik), "Log-likelihood is not finite"

        # Check that AIC and BIC are computed
        assert result.aic is not None
        assert result.bic is not None

    def test_wang_2002_parameter_recovery(self):
        """Test parameter recovery with known DGP."""
        np.random.seed(123)
        n = 500  # Larger sample for better recovery

        # Simple DGP with clear heteroscedasticity
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        Z = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])
        W = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])

        beta_true = np.array([3.0, 0.8])
        delta_true = np.array([0.1, 0.4])  # Moderate location effect
        gamma_true = np.array([-1.5, 0.5])  # Moderate scale effect

        # Generate inefficiency
        mu_i = Z @ delta_true
        ln_sigma_u_sq_i = W @ gamma_true
        sigma_u_i = np.sqrt(np.exp(ln_sigma_u_sq_i))

        u = np.abs(np.random.normal(mu_i, sigma_u_i))
        v = np.random.normal(0, 0.3, n)
        y = X @ beta_true + v - u

        df = pd.DataFrame(
            {
                "y": y,
                "x1": X[:, 1],
                "z1": Z[:, 1],
                "w1": W[:, 1],
            }
        )

        # Estimate
        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1"],
            frontier="production",
            dist="truncated_normal",
            inefficiency_vars=["z1"],
            het_vars=["w1"],
        )

        result = model.fit(verbose=False)

        # Check parameter recovery (allow for sampling variability)
        # Frontier parameters
        assert abs(result.params["const"] - beta_true[0]) < 0.3
        assert abs(result.params["x1"] - beta_true[1]) < 0.3

        # Location parameters (δ)
        assert abs(result.params["delta_const"] - delta_true[0]) < 0.5
        assert abs(result.params["delta_z1"] - delta_true[1]) < 0.5

        # Scale parameters (γ) - these are harder to estimate precisely
        assert abs(result.params["gamma_const"] - gamma_true[0]) < 1.0
        # At least check sign is correct for gamma_w1
        assert np.sign(result.params["gamma_w1"]) == np.sign(gamma_true[1])

    def test_wang_2002_standard_errors(self):
        """Test that standard errors are computed and reasonable."""
        np.random.seed(42)
        n = 150

        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        Z = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])
        W = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])

        beta = np.array([2.0, 0.5])
        delta = np.array([0.3, 0.3])
        gamma = np.array([-1.0, 0.2])

        mu_i = Z @ delta
        sigma_u_i = np.sqrt(np.exp(W @ gamma))
        u = np.abs(np.random.normal(mu_i, sigma_u_i))
        v = np.random.normal(0, 0.2, n)
        y = X @ beta + v - u

        df = pd.DataFrame(
            {
                "y": y,
                "x1": X[:, 1],
                "z1": Z[:, 1],
                "w1": W[:, 1],
            }
        )

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1"],
            frontier="production",
            dist="truncated_normal",
            inefficiency_vars=["z1"],
            het_vars=["w1"],
        )

        result = model.fit(verbose=False)

        # Check that all standard errors are positive and finite
        assert all(result.se > 0), "Some standard errors are non-positive"
        assert all(np.isfinite(result.se)), "Some standard errors are not finite"

        # Check that t-statistics are computed
        assert all(np.isfinite(result.tvalues)), "Some t-statistics are not finite"

        # Check that p-values are in [0, 1]
        assert all((result.pvalues >= 0) & (result.pvalues <= 1)), "P-values out of range"


class TestWang2002Restrictions:
    """Test that Wang (2002) reduces to special cases."""

    def test_wang_reduces_to_homoscedastic(self):
        """Test that when γ ≈ 0, model behaves like BC95."""
        np.random.seed(42)
        n = 200

        # Generate data with NO heteroscedasticity (constant σ_u)
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        Z = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])
        W = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])

        beta = np.array([2.0, 0.5])
        delta = np.array([0.2, 0.4])  # Location effect
        gamma = np.array([-1.0, 0.0])  # NO scale effect (γ_w1 = 0)

        mu_i = Z @ delta
        sigma_u = 0.5  # Constant σ_u
        u = np.abs(np.random.normal(mu_i, sigma_u, n))
        v = np.random.normal(0, 0.2, n)
        y = X @ beta + v - u

        df = pd.DataFrame(
            {
                "y": y,
                "x1": X[:, 1],
                "z1": Z[:, 1],
                "w1": W[:, 1],
            }
        )

        # Estimate Wang model
        model_wang = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1"],
            frontier="production",
            dist="truncated_normal",
            inefficiency_vars=["z1"],
            het_vars=["w1"],
        )

        result_wang = model_wang.fit(verbose=False)

        # Check that γ_w1 is close to 0
        assert (
            abs(result_wang.params["gamma_w1"]) < 0.3
        ), f"Expected γ_w1 ≈ 0, got {result_wang.params['gamma_w1']:.4f}"

        # Check that δ parameters are still recovered
        assert abs(result_wang.params["delta_const"] - delta[0]) < 0.4
        assert abs(result_wang.params["delta_z1"] - delta[1]) < 0.4

    def test_wang_convergence(self):
        """Test that Wang model converges in various settings."""
        np.random.seed(42)

        for n in [100, 200, 300]:
            X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
            Z = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])
            W = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])

            beta = np.array([2.0, 0.5])
            delta = np.array([0.2, 0.3])
            gamma = np.array([-1.0, 0.2])

            mu_i = Z @ delta
            sigma_u_i = np.sqrt(np.exp(W @ gamma))
            u = np.abs(np.random.normal(mu_i, sigma_u_i))
            v = np.random.normal(0, 0.2, n)
            y = X @ beta + v - u

            df = pd.DataFrame(
                {
                    "y": y,
                    "x1": X[:, 1],
                    "z1": Z[:, 1],
                    "w1": W[:, 1],
                }
            )

            model = StochasticFrontier(
                data=df,
                depvar="y",
                exog=["x1"],
                frontier="production",
                dist="truncated_normal",
                inefficiency_vars=["z1"],
                het_vars=["w1"],
            )

            result = model.fit(verbose=False)

            assert result.converged, f"Model failed to converge with n={n}"
            assert np.isfinite(result.loglik), f"Log-likelihood not finite with n={n}"


class TestWang2002MarginalEffects:
    """Test marginal effects computation for Wang (2002)."""

    def test_marginal_effects_location(self):
        """Test marginal effects on location (mean inefficiency)."""
        np.random.seed(42)
        n = 200

        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        Z = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])
        W = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])

        beta = np.array([2.0, 0.5])
        delta = np.array([0.3, 0.6])  # Positive effect
        gamma = np.array([-1.0, 0.2])

        mu_i = Z @ delta
        sigma_u_i = np.sqrt(np.exp(W @ gamma))
        u = np.abs(np.random.normal(mu_i, sigma_u_i))
        v = np.random.normal(0, 0.2, n)
        y = X @ beta + v - u

        df = pd.DataFrame(
            {
                "y": y,
                "x1": X[:, 1],
                "age": Z[:, 1],
                "size": W[:, 1],
            }
        )

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1"],
            frontier="production",
            dist="truncated_normal",
            inefficiency_vars=["age"],
            het_vars=["size"],
        )

        result = model.fit(verbose=False)

        # Compute marginal effects on location
        me_location = result.marginal_effects(method="location")

        # Check structure
        assert isinstance(me_location, pd.DataFrame)
        assert "variable" in me_location.columns
        assert "marginal_effect" in me_location.columns
        assert "std_error" in me_location.columns
        assert "t_stat" in me_location.columns
        assert "p_value" in me_location.columns
        assert "ci_lower" in me_location.columns
        assert "ci_upper" in me_location.columns

        # Check that ME for 'age' is present
        assert "age" in me_location["variable"].values

        # Check that ME ≈ δ (for location, ME is just delta)
        age_me = me_location[me_location["variable"] == "age"]["marginal_effect"].values[0]
        delta_age = result.params["delta_age"]
        assert abs(age_me - delta_age) < 1e-6, "Location ME should equal delta"

        # Check that standard errors are positive
        assert all(me_location["std_error"] > 0)

    def test_marginal_effects_scale(self):
        """Test marginal effects on scale (variance of inefficiency)."""
        np.random.seed(42)
        n = 200

        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        Z = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])
        W = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])

        beta = np.array([2.0, 0.5])
        delta = np.array([0.3, 0.4])
        gamma = np.array([-1.0, 0.5])  # Positive scale effect

        mu_i = Z @ delta
        sigma_u_i = np.sqrt(np.exp(W @ gamma))
        u = np.abs(np.random.normal(mu_i, sigma_u_i))
        v = np.random.normal(0, 0.2, n)
        y = X @ beta + v - u

        df = pd.DataFrame(
            {
                "y": y,
                "x1": X[:, 1],
                "age": Z[:, 1],
                "size": W[:, 1],
            }
        )

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1"],
            frontier="production",
            dist="truncated_normal",
            inefficiency_vars=["age"],
            het_vars=["size"],
        )

        result = model.fit(verbose=False)

        # Compute marginal effects on scale
        me_scale = result.marginal_effects(method="scale")

        # Check structure
        assert isinstance(me_scale, pd.DataFrame)
        assert "variable" in me_scale.columns
        assert "marginal_effect" in me_scale.columns
        assert "std_error" in me_scale.columns

        # Check that ME for 'size' is present
        assert "size" in me_scale["variable"].values

        # Check that standard errors are positive
        assert all(me_scale["std_error"] > 0)

        # Check sign of ME (should match sign of gamma)
        size_me = me_scale[me_scale["variable"] == "size"]["marginal_effect"].values[0]
        gamma_size = result.params["gamma_size"]
        # ME = (σ_u / 2) · γ, so signs should match
        assert np.sign(size_me) == np.sign(gamma_size)

    def test_marginal_effects_error_handling(self):
        """Test error handling for marginal effects."""
        np.random.seed(42)
        n = 100

        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        y = X @ np.array([2.0, 0.5]) + np.random.normal(0, 0.2, n)

        df = pd.DataFrame(
            {
                "y": y,
                "x1": X[:, 1],
            }
        )

        # Model without inefficiency determinants
        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1"],
            frontier="production",
            dist="half_normal",
        )

        result = model.fit(verbose=False)

        # Should raise error
        with pytest.raises(ValueError, match="no inefficiency determinants"):
            result.marginal_effects()


class TestWang2002Efficiency:
    """Test efficiency predictions from Wang (2002) model."""

    def test_efficiency_estimation(self):
        """Test that efficiency can be estimated."""
        np.random.seed(42)
        n = 150

        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        Z = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])
        W = np.column_stack([np.ones(n), np.random.uniform(0, 1, n)])

        beta = np.array([2.0, 0.5])
        delta = np.array([0.2, 0.3])
        gamma = np.array([-1.0, 0.2])

        mu_i = Z @ delta
        sigma_u_i = np.sqrt(np.exp(W @ gamma))
        u = np.abs(np.random.normal(mu_i, sigma_u_i))
        v = np.random.normal(0, 0.2, n)
        y = X @ beta + v - u

        df = pd.DataFrame(
            {
                "y": y,
                "x1": X[:, 1],
                "z1": Z[:, 1],
                "w1": W[:, 1],
            }
        )

        model = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x1"],
            frontier="production",
            dist="truncated_normal",
            inefficiency_vars=["z1"],
            het_vars=["w1"],
        )

        result = model.fit(verbose=False)

        # Get efficiency estimates
        eff_bc = result.efficiency(estimator="bc")

        # Check structure
        assert isinstance(eff_bc, pd.DataFrame)
        assert "efficiency" in eff_bc.columns
        assert len(eff_bc) == n

        # Check that efficiencies are in (0, 1]
        assert all(
            (eff_bc["efficiency"] > 0) & (eff_bc["efficiency"] <= 1)
        ), "Efficiencies out of range"

        # Check that mean efficiency is reasonable
        mean_eff = eff_bc["efficiency"].mean()
        assert 0.3 < mean_eff < 0.95, f"Mean efficiency {mean_eff:.3f} seems unreasonable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
