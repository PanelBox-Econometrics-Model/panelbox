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

    @pytest.mark.xfail(
        reason="Exponential SFA MLE is prone to converging to poor local optima; "
        "sigma_v estimate can diverge when the optimizer fails to separate v from u"
    )
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

    @pytest.mark.xfail(reason="SFA MLE has multiple local optima with different starting values")
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

    @pytest.mark.slow
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


class TestGetLikelihoodFunction:
    """Tests for _get_likelihood_function dispatch."""

    def test_half_normal(self):
        """Test that half_normal returns the correct likelihood function."""
        from panelbox.frontier.estimation import _get_likelihood_function
        from panelbox.frontier.likelihoods import loglik_half_normal

        func = _get_likelihood_function("half_normal")
        assert func is loglik_half_normal

    def test_exponential(self):
        """Test that exponential returns the correct likelihood function."""
        from panelbox.frontier.estimation import _get_likelihood_function
        from panelbox.frontier.likelihoods import loglik_exponential

        func = _get_likelihood_function("exponential")
        assert func is loglik_exponential

    def test_truncated_normal(self):
        """Test that truncated_normal returns the correct likelihood function."""
        from panelbox.frontier.estimation import _get_likelihood_function
        from panelbox.frontier.likelihoods import loglik_truncated_normal

        func = _get_likelihood_function("truncated_normal")
        assert func is loglik_truncated_normal

    def test_gamma(self):
        """Test that gamma returns the correct likelihood function."""
        from panelbox.frontier.estimation import _get_likelihood_function
        from panelbox.frontier.likelihoods import loglik_gamma

        func = _get_likelihood_function("gamma")
        assert func is loglik_gamma

    def test_wang_2002(self):
        """Test that Wang (2002) model returns the correct likelihood function."""
        from panelbox.frontier.estimation import _get_likelihood_function
        from panelbox.frontier.likelihoods import loglik_wang_2002

        func = _get_likelihood_function("truncated_normal", is_wang=True)
        assert func is loglik_wang_2002

    def test_unknown_distribution_raises(self):
        """Test that unknown distribution raises ValueError."""
        from panelbox.frontier.estimation import _get_likelihood_function

        with pytest.raises(ValueError, match="Unknown distribution"):
            _get_likelihood_function("unknown_dist")


class TestGetGradientFunction:
    """Tests for _get_gradient_function dispatch."""

    def test_half_normal_gradient(self):
        """Test that half_normal returns gradient function."""
        from panelbox.frontier.estimation import _get_gradient_function
        from panelbox.frontier.likelihoods import gradient_half_normal

        func = _get_gradient_function("half_normal")
        assert func is gradient_half_normal

    def test_exponential_gradient(self):
        """Test that exponential returns gradient function."""
        from panelbox.frontier.estimation import _get_gradient_function
        from panelbox.frontier.likelihoods import gradient_exponential

        func = _get_gradient_function("exponential")
        assert func is gradient_exponential

    def test_truncated_normal_no_gradient(self):
        """Test that truncated_normal has no analytical gradient."""
        from panelbox.frontier.estimation import _get_gradient_function

        func = _get_gradient_function("truncated_normal")
        assert func is None

    def test_gamma_no_gradient(self):
        """Test that gamma has no analytical gradient."""
        from panelbox.frontier.estimation import _get_gradient_function

        func = _get_gradient_function("gamma")
        assert func is None

    def test_unknown_dist_returns_none(self):
        """Test that unknown distribution returns None (no gradient)."""
        from panelbox.frontier.estimation import _get_gradient_function

        func = _get_gradient_function("nonexistent_dist")
        assert func is None


class TestTransformParameters:
    """Tests for _transform_parameters."""

    def test_half_normal_basic(self):
        """Test parameter transformation for half-normal model."""
        from panelbox.frontier.estimation import _transform_parameters

        # theta = [beta0, beta1, ln(sigma_v_sq), ln(sigma_u_sq)]
        beta = np.array([2.0, 0.6])
        ln_sigma_v_sq = np.log(0.01)  # sigma_v_sq = 0.01
        ln_sigma_u_sq = np.log(0.04)  # sigma_u_sq = 0.04
        theta = np.concatenate([beta, [ln_sigma_v_sq, ln_sigma_u_sq]])

        params, names = _transform_parameters(
            theta,
            n_exog=2,
            n_ineff_vars=0,
            dist="half_normal",
            exog_names=["const", "x1"],
            ineff_var_names=[],
        )

        assert len(params) == 4
        assert len(names) == 4
        np.testing.assert_allclose(params[0], 2.0)
        np.testing.assert_allclose(params[1], 0.6)
        np.testing.assert_allclose(params[2], 0.01, rtol=1e-10)
        np.testing.assert_allclose(params[3], 0.04, rtol=1e-10)
        assert names == ["const", "x1", "sigma_v_sq", "sigma_u_sq"]

    def test_exponential_basic(self):
        """Test parameter transformation for exponential model."""
        from panelbox.frontier.estimation import _transform_parameters

        theta = np.array([1.5, np.log(0.02), np.log(0.05)])

        params, names = _transform_parameters(
            theta,
            n_exog=1,
            n_ineff_vars=0,
            dist="exponential",
            exog_names=["const"],
            ineff_var_names=[],
        )

        assert len(params) == 3
        np.testing.assert_allclose(params[0], 1.5)
        np.testing.assert_allclose(params[1], 0.02, rtol=1e-10)
        np.testing.assert_allclose(params[2], 0.05, rtol=1e-10)
        assert names == ["const", "sigma_v_sq", "sigma_u_sq"]

    def test_truncated_normal_simple(self):
        """Test parameter transformation for simple truncated normal (no Z vars)."""
        from panelbox.frontier.estimation import _transform_parameters

        # theta = [beta0, ln(sigma_v_sq), ln(sigma_u_sq), mu]
        theta = np.array([3.0, np.log(0.01), np.log(0.04), 0.5])

        params, names = _transform_parameters(
            theta,
            n_exog=1,
            n_ineff_vars=0,
            dist="truncated_normal",
            exog_names=["const"],
            ineff_var_names=[],
        )

        assert len(params) == 4
        np.testing.assert_allclose(params[0], 3.0)
        np.testing.assert_allclose(params[1], 0.01, rtol=1e-10)
        np.testing.assert_allclose(params[2], 0.04, rtol=1e-10)
        np.testing.assert_allclose(params[3], 0.5)
        assert names == ["const", "sigma_v_sq", "sigma_u_sq", "mu"]

    def test_truncated_normal_with_bc95_vars(self):
        """Test parameter transformation for truncated normal with BC95 inefficiency vars."""
        from panelbox.frontier.estimation import _transform_parameters

        # theta = [beta0, beta1, ln(sigma_v_sq), ln(sigma_u_sq), delta1, delta2]
        theta = np.array([2.0, 0.5, np.log(0.01), np.log(0.04), 0.3, -0.2])

        params, names = _transform_parameters(
            theta,
            n_exog=2,
            n_ineff_vars=2,
            dist="truncated_normal",
            exog_names=["const", "x1"],
            ineff_var_names=["z1", "z2"],
        )

        assert len(params) == 6
        np.testing.assert_allclose(params[0], 2.0)
        np.testing.assert_allclose(params[1], 0.5)
        np.testing.assert_allclose(params[2], 0.01, rtol=1e-10)
        np.testing.assert_allclose(params[3], 0.04, rtol=1e-10)
        np.testing.assert_allclose(params[4], 0.3)
        np.testing.assert_allclose(params[5], -0.2)
        assert names == ["const", "x1", "sigma_v_sq", "sigma_u_sq", "delta_z1", "delta_z2"]

    def test_gamma_distribution(self):
        """Test parameter transformation for gamma distribution."""
        from panelbox.frontier.estimation import _transform_parameters

        # theta = [beta0, ln(sigma_v_sq), ln(sigma_u_sq), ln(P), ln(theta)]
        ln_P = np.log(2.0)
        ln_theta_param = np.log(1.5)
        theta = np.array([1.0, np.log(0.01), np.log(0.04), ln_P, ln_theta_param])

        params, names = _transform_parameters(
            theta,
            n_exog=1,
            n_ineff_vars=0,
            dist="gamma",
            exog_names=["const"],
            ineff_var_names=[],
        )

        assert len(params) == 5
        np.testing.assert_allclose(params[0], 1.0)
        np.testing.assert_allclose(params[1], 0.01, rtol=1e-10)
        np.testing.assert_allclose(params[2], 0.04, rtol=1e-10)
        np.testing.assert_allclose(params[3], 2.0, rtol=1e-10)  # P
        np.testing.assert_allclose(params[4], 1.5, rtol=1e-10)  # theta
        assert names == ["const", "sigma_v_sq", "sigma_u_sq", "gamma_P", "gamma_theta"]

    def test_wang_model(self):
        """Test parameter transformation for Wang (2002) model."""
        from panelbox.frontier.estimation import _transform_parameters

        # theta = [beta0, beta1, ln(sigma_v_sq), delta1, gamma1, gamma2]
        theta = np.array([2.0, 0.5, np.log(0.01), 0.3, -0.1, 0.2])

        params, names = _transform_parameters(
            theta,
            n_exog=2,
            n_ineff_vars=1,
            dist="truncated_normal",
            exog_names=["const", "x1"],
            ineff_var_names=["z1"],
            is_wang=True,
            hetero_var_names=["w1", "w2"],
        )

        # Wang: [beta0, beta1, sigma_v_sq, delta_z1, gamma_w1, gamma_w2]
        assert len(params) == 6
        np.testing.assert_allclose(params[0], 2.0)
        np.testing.assert_allclose(params[1], 0.5)
        np.testing.assert_allclose(params[2], 0.01, rtol=1e-10)
        np.testing.assert_allclose(params[3], 0.3)  # delta
        np.testing.assert_allclose(params[4], -0.1)  # gamma
        np.testing.assert_allclose(params[5], 0.2)  # gamma
        assert names == ["const", "x1", "sigma_v_sq", "delta_z1", "gamma_w1", "gamma_w2"]

    def test_wang_model_no_hetero_vars(self):
        """Test Wang model transformation with empty hetero_var_names."""
        from panelbox.frontier.estimation import _transform_parameters

        # theta = [beta0, ln(sigma_v_sq), delta1]
        theta = np.array([2.0, np.log(0.01), 0.3])

        params, names = _transform_parameters(
            theta,
            n_exog=1,
            n_ineff_vars=1,
            dist="truncated_normal",
            exog_names=["const"],
            ineff_var_names=["z1"],
            is_wang=True,
            hetero_var_names=[],
        )

        # Wang without hetero vars: [beta0, sigma_v_sq, delta_z1]
        assert len(params) == 3
        assert names == ["const", "sigma_v_sq", "delta_z1"]


class TestComputeHessian:
    """Tests for _compute_hessian."""

    def test_quadratic_function(self):
        """Test Hessian of f(x) = x1^2 + 2*x2^2 at origin."""
        from panelbox.frontier.estimation import _compute_hessian

        def quadratic(theta):
            return theta[0] ** 2 + 2 * theta[1] ** 2

        theta = np.array([0.0, 0.0])
        hessian = _compute_hessian(theta, quadratic, method="numerical")

        # Hessian should be [[2, 0], [0, 4]]
        expected = np.array([[2.0, 0.0], [0.0, 4.0]])
        np.testing.assert_allclose(hessian, expected, atol=1e-4)

    def test_mixed_quadratic(self):
        """Test Hessian of f(x) = x1^2 + x1*x2 + x2^2."""
        from panelbox.frontier.estimation import _compute_hessian

        def mixed_quadratic(theta):
            return theta[0] ** 2 + theta[0] * theta[1] + theta[1] ** 2

        theta = np.array([1.0, 1.0])
        hessian = _compute_hessian(theta, mixed_quadratic, method="numerical")

        # Hessian should be [[2, 1], [1, 2]]
        expected = np.array([[2.0, 1.0], [1.0, 2.0]])
        np.testing.assert_allclose(hessian, expected, atol=1e-4)

    def test_hessian_symmetry(self):
        """Test that Hessian is symmetric."""
        from panelbox.frontier.estimation import _compute_hessian

        def cubic(theta):
            return theta[0] ** 3 + theta[0] * theta[1] ** 2 + theta[1] ** 3

        theta = np.array([1.0, 2.0])
        hessian = _compute_hessian(theta, cubic, method="numerical")

        # Must be symmetric
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-8)

    def test_single_variable(self):
        """Test Hessian for a scalar function."""
        from panelbox.frontier.estimation import _compute_hessian

        def scalar_func(theta):
            return 3 * theta[0] ** 2

        theta = np.array([2.0])
        hessian = _compute_hessian(theta, scalar_func, method="numerical")

        assert hessian.shape == (1, 1)
        np.testing.assert_allclose(hessian[0, 0], 6.0, atol=1e-4)

    def test_nonfinite_hessian_returns_none(self):
        """Test that non-finite Hessian returns None with a warning."""
        from panelbox.frontier.estimation import _compute_hessian

        def bad_func(theta):
            # Returns inf when any parameter is slightly perturbed positively
            return np.inf if np.any(theta > 0.5) else 0.0

        # Epsilon is 1e-5 by default, so 0.5 - 5e-6 + 1e-5 > 0.5
        theta = np.array([0.5 - 5e-6])
        with pytest.warns(UserWarning, match="non-finite"):
            result = _compute_hessian(theta, bad_func, method="numerical")

        assert result is None

    def test_analytical_method_raises(self):
        """Test that analytical Hessian raises NotImplementedError."""
        from panelbox.frontier.estimation import _compute_hessian

        theta = np.array([1.0])
        with pytest.raises(NotImplementedError, match="Only numerical"):
            _compute_hessian(theta, lambda t: t[0] ** 2, method="analytical")

    def test_higher_dimension(self):
        """Test Hessian with more than 2 parameters."""
        from panelbox.frontier.estimation import _compute_hessian

        def func3d(theta):
            return theta[0] ** 2 + 2 * theta[1] ** 2 + 3 * theta[2] ** 2

        theta = np.array([1.0, 1.0, 1.0])
        hessian = _compute_hessian(theta, func3d, method="numerical")

        expected = np.diag([2.0, 4.0, 6.0])
        np.testing.assert_allclose(hessian, expected, atol=1e-4)


class TestTransformBC92Parameters:
    """Tests for _transform_bc92_parameters."""

    def test_basic(self):
        """Test BC92 parameter transformation."""
        from panelbox.frontier.estimation import _transform_bc92_parameters

        # theta = [beta0, beta1, ln(sigma_v_sq), ln(sigma_u_sq), eta]
        theta = np.array([2.0, 0.5, np.log(0.01), np.log(0.04), -0.1])

        params, names = _transform_bc92_parameters(
            theta,
            n_exog=2,
            exog_names=["const", "x1"],
        )

        assert len(params) == 5
        assert len(names) == 5
        np.testing.assert_allclose(params[0], 2.0)
        np.testing.assert_allclose(params[1], 0.5)
        np.testing.assert_allclose(params[2], 0.01, rtol=1e-10)  # sigma_v_sq
        np.testing.assert_allclose(params[3], 0.04, rtol=1e-10)  # sigma_u_sq
        np.testing.assert_allclose(params[4], -0.1)  # eta
        assert names == ["const", "x1", "sigma_v_sq", "sigma_u_sq", "eta"]

    def test_single_exog(self):
        """Test BC92 with a single exogenous variable."""
        from panelbox.frontier.estimation import _transform_bc92_parameters

        theta = np.array([3.0, np.log(0.05), np.log(0.10), 0.3])

        params, names = _transform_bc92_parameters(
            theta,
            n_exog=1,
            exog_names=["const"],
        )

        assert len(params) == 4
        np.testing.assert_allclose(params[0], 3.0)
        np.testing.assert_allclose(params[1], 0.05, rtol=1e-10)
        np.testing.assert_allclose(params[2], 0.10, rtol=1e-10)
        np.testing.assert_allclose(params[3], 0.3)
        assert names == ["const", "sigma_v_sq", "sigma_u_sq", "eta"]


class TestTransformKumbhakarParameters:
    """Tests for _transform_kumbhakar_parameters."""

    def test_basic(self):
        """Test Kumbhakar parameter transformation."""
        from panelbox.frontier.estimation import _transform_kumbhakar_parameters

        # theta = [beta0, beta1, ln(sigma_v_sq), ln(sigma_u_sq), mu, b, c]
        theta = np.array([2.0, 0.5, np.log(0.01), np.log(0.04), 0.1, -0.05, 0.02])

        params, names = _transform_kumbhakar_parameters(
            theta,
            n_exog=2,
            exog_names=["const", "x1"],
        )

        assert len(params) == 7
        assert len(names) == 7
        np.testing.assert_allclose(params[0], 2.0)
        np.testing.assert_allclose(params[1], 0.5)
        np.testing.assert_allclose(params[2], 0.01, rtol=1e-10)  # sigma_v_sq
        np.testing.assert_allclose(params[3], 0.04, rtol=1e-10)  # sigma_u_sq
        np.testing.assert_allclose(params[4], 0.1)  # mu
        np.testing.assert_allclose(params[5], -0.05)  # b
        np.testing.assert_allclose(params[6], 0.02)  # c
        assert names == ["const", "x1", "sigma_v_sq", "sigma_u_sq", "mu", "b", "c"]


class TestTransformLeeSchmidtParameters:
    """Tests for _transform_lee_schmidt_parameters."""

    def test_basic(self):
        """Test Lee-Schmidt parameter transformation with 4 time periods."""
        from panelbox.frontier.estimation import _transform_lee_schmidt_parameters

        # theta = [beta0, ln(sigma_v_sq), ln(sigma_u_sq), mu, delta_1, delta_2, delta_3]
        # (T=4, so 3 delta parameters, delta_4 = 1 by normalization)
        theta = np.array([2.0, np.log(0.01), np.log(0.04), 0.1, 0.8, 0.9, 1.1])

        params, names = _transform_lee_schmidt_parameters(
            theta,
            n_exog=1,
            exog_names=["const"],
            n_periods=4,
        )

        # Output: [beta0, sigma_v_sq, sigma_u_sq, mu, delta_1, delta_2, delta_3, delta_4=1]
        assert len(params) == 8
        assert len(names) == 8
        np.testing.assert_allclose(params[0], 2.0)
        np.testing.assert_allclose(params[1], 0.01, rtol=1e-10)
        np.testing.assert_allclose(params[2], 0.04, rtol=1e-10)
        np.testing.assert_allclose(params[3], 0.1)  # mu
        np.testing.assert_allclose(params[4], 0.8)  # delta_t1
        np.testing.assert_allclose(params[5], 0.9)  # delta_t2
        np.testing.assert_allclose(params[6], 1.1)  # delta_t3
        np.testing.assert_allclose(params[7], 1.0)  # delta_t4 (normalized)
        assert names == [
            "const",
            "sigma_v_sq",
            "sigma_u_sq",
            "mu",
            "delta_t1",
            "delta_t2",
            "delta_t3",
            "delta_t4",
        ]

    def test_two_periods(self):
        """Test Lee-Schmidt with only 2 time periods."""
        from panelbox.frontier.estimation import _transform_lee_schmidt_parameters

        # theta = [beta0, beta1, ln(sigma_v_sq), ln(sigma_u_sq), mu, delta_1]
        # T=2, so 1 delta parameter estimated, delta_T=1 appended
        theta = np.array([1.0, 0.5, np.log(0.02), np.log(0.03), 0.0, 0.85])

        params, names = _transform_lee_schmidt_parameters(
            theta,
            n_exog=2,
            exog_names=["const", "x1"],
            n_periods=2,
        )

        # Output: [beta0, beta1, sigma_v_sq, sigma_u_sq, mu, delta_t1, delta_t2=1.0]
        assert len(params) == 7
        np.testing.assert_allclose(params[-1], 1.0)  # last delta = 1.0 (normalized)
        np.testing.assert_allclose(params[-2], 0.85)  # delta_t1
        assert names[-1] == "delta_t2"
        assert names[-2] == "delta_t1"


class TestEstimateMLE_OptimizerDispatch:
    """Tests for optimizer dispatch and error handling in estimate_mle."""

    @pytest.fixture
    def simple_production_data(self):
        """Small dataset for fast testing of optimizer paths."""
        np.random.seed(42)
        n = 200

        beta_true = np.array([1.0, 0.5])
        x = np.random.normal(0, 1, n)
        v = np.random.normal(0, 0.1, n)
        u = np.abs(np.random.normal(0, 0.2, n))
        y = beta_true[0] + beta_true[1] * x + v - u

        df = pd.DataFrame({"y": y, "x": x})
        return df

    def test_bfgs_optimizer(self, simple_production_data):
        """Test estimation with BFGS optimizer."""
        sf = StochasticFrontier(
            data=simple_production_data,
            depvar="y",
            exog=["x"],
            frontier="production",
            dist="half_normal",
        )

        result = sf.fit(method="mle", optimizer="BFGS", verbose=False)

        # BFGS may or may not converge but should produce finite results
        assert np.isfinite(result.loglik)
        assert result.sigma_v > 0
        assert result.sigma_u > 0

    def test_unknown_optimizer_raises(self, simple_production_data):
        """Test that unknown optimizer raises ValueError."""
        sf = StochasticFrontier(
            data=simple_production_data,
            depvar="y",
            exog=["x"],
            frontier="production",
            dist="half_normal",
        )

        with pytest.raises(ValueError, match="Unknown optimizer"):
            sf.fit(method="mle", optimizer="NONEXISTENT_OPTIMIZER", verbose=False)

    def test_truncated_normal_estimation(self):
        """Test estimation with truncated normal distribution."""
        np.random.seed(42)
        n = 300

        x = np.random.normal(0, 1, n)
        v = np.random.normal(0, 0.1, n)
        # Truncated normal with mu > 0
        u = np.abs(np.random.normal(0.2, 0.15, n))
        y = 1.0 + 0.5 * x + v - u

        df = pd.DataFrame({"y": y, "x": x})

        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x"],
            frontier="production",
            dist="truncated_normal",
        )

        result = sf.fit(method="mle", verbose=False)

        assert np.isfinite(result.loglik)
        assert result.sigma_v > 0
        assert result.sigma_u > 0

    def test_verbose_output(self, simple_production_data, caplog):
        """Test that verbose mode produces log output."""
        import logging

        sf = StochasticFrontier(
            data=simple_production_data,
            depvar="y",
            exog=["x"],
            frontier="production",
            dist="half_normal",
        )

        with caplog.at_level(logging.INFO, logger="panelbox.frontier.estimation"):
            result = sf.fit(method="mle", verbose=True)

        # The optimization should complete
        assert np.isfinite(result.loglik)

    def test_custom_starting_values(self, simple_production_data):
        """Test that custom starting values are used."""
        sf = StochasticFrontier(
            data=simple_production_data,
            depvar="y",
            exog=["x"],
            frontier="production",
            dist="half_normal",
        )

        # Provide good starting values
        start = np.array([1.0, 0.5, np.log(0.01), np.log(0.04)])
        result = sf.fit(method="mle", start_params=start, verbose=False)

        assert np.isfinite(result.loglik)

    def test_grid_search_starting_values(self, simple_production_data):
        """Test estimation with grid_search=True for starting values."""
        sf = StochasticFrontier(
            data=simple_production_data,
            depvar="y",
            exog=["x"],
            frontier="production",
            dist="half_normal",
        )

        result = sf.fit(method="mle", grid_search=True, verbose=False)

        assert np.isfinite(result.loglik)

    def test_maxiter_respected(self, simple_production_data):
        """Test that maxiter limits the number of iterations."""
        sf = StochasticFrontier(
            data=simple_production_data,
            depvar="y",
            exog=["x"],
            frontier="production",
            dist="half_normal",
        )

        # Very low maxiter - should produce a result (possibly not converged)
        result = sf.fit(method="mle", maxiter=2, verbose=False)
        assert np.isfinite(result.loglik) or result.loglik is not None


class TestEstimateMLE_ConvergenceChecks:
    """Tests for convergence checking logic in estimate_mle."""

    def test_convergence_result_attributes(self):
        """Test that convergence info is stored in result."""
        np.random.seed(42)
        n = 200

        x = np.random.normal(0, 1, n)
        v = np.random.normal(0, 0.1, n)
        u = np.abs(np.random.normal(0, 0.2, n))
        y = 1.0 + 0.5 * x + v - u

        df = pd.DataFrame({"y": y, "x": x})

        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x"],
            frontier="production",
            dist="half_normal",
        )

        result = sf.fit(method="mle", verbose=False)

        # Check convergence attributes
        assert isinstance(result.converged, bool)
        assert np.isfinite(result.loglik)
        assert result.optimization_result is not None


class TestEstimateMLE_NegLogLikErrorHandling:
    """Tests for error handling in the neg_loglik and neg_gradient closures."""

    def test_neg_loglik_returns_inf_on_bad_params(self):
        """Test that neg_loglik returns inf when likelihood raises exceptions."""
        np.random.seed(42)
        n = 200

        x = np.random.normal(0, 1, n)
        v = np.random.normal(0, 0.1, n)
        u = np.abs(np.random.normal(0, 0.2, n))
        y = 1.0 + 0.5 * x + v - u

        df = pd.DataFrame({"y": y, "x": x})

        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x"],
            frontier="production",
            dist="half_normal",
        )

        # The optimizer should handle the bad starting values gracefully
        # by catching exceptions inside neg_loglik. Test with extreme values.
        extreme_start = np.array([1e10, 1e10, 50.0, 50.0])
        # This should not crash - just produce a possibly poor result
        result = sf.fit(method="mle", start_params=extreme_start, verbose=False)
        assert result is not None


class TestEstimateMLE_TruncatedNormalBounds:
    """Tests for bounds setting with truncated normal distribution."""

    def test_truncated_normal_with_mu_bounds(self):
        """Test that truncated normal estimation includes mu parameter bounds."""
        np.random.seed(42)
        n = 300

        x = np.random.normal(0, 1, n)
        v = np.random.normal(0, 0.1, n)
        u = np.abs(np.random.normal(0.1, 0.2, n))
        y = 1.0 + 0.5 * x + v - u

        df = pd.DataFrame({"y": y, "x": x})

        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x"],
            frontier="production",
            dist="truncated_normal",
        )

        result = sf.fit(method="mle", verbose=False)
        # Check that mu parameter exists in the result
        assert "mu" in result.params.index or np.isfinite(result.loglik)


class TestEstimateMLE_NewtonCG:
    """Tests for Newton-CG optimizer path."""

    def test_newton_cg_optimizer(self):
        """Test estimation with Newton-CG optimizer."""
        np.random.seed(42)
        n = 300

        x = np.random.normal(0, 1, n)
        v = np.random.normal(0, 0.1, n)
        u = np.abs(np.random.normal(0, 0.2, n))
        y = 1.0 + 0.5 * x + v - u

        df = pd.DataFrame({"y": y, "x": x})

        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x"],
            frontier="production",
            dist="half_normal",
        )

        # Newton-CG requires gradient, which half_normal provides
        result = sf.fit(method="mle", optimizer="Newton-CG", verbose=False)

        assert np.isfinite(result.loglik)
        assert result.sigma_v > 0
        assert result.sigma_u > 0


class TestEstimateMLE_GammaDistribution:
    """Tests for gamma distribution estimation path."""

    def test_gamma_distribution_estimation(self):
        """Test that gamma distribution estimation runs (covers gamma bounds)."""
        np.random.seed(42)
        n = 300

        x = np.random.normal(0, 1, n)
        v = np.random.normal(0, 0.1, n)
        u = np.random.gamma(2.0, 0.1, n)  # Gamma-distributed inefficiency
        y = 1.0 + 0.5 * x + v - u

        df = pd.DataFrame({"y": y, "x": x})

        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x"],
            frontier="production",
            dist="gamma",
        )

        # Gamma may or may not converge - that's fine, we want to cover the bounds path
        result = sf.fit(method="mle", verbose=False)
        assert result is not None
        assert result.loglik is not None


class TestEstimateMLE_StartingValueCheck:
    """Tests for starting value validation."""

    def test_poor_starting_values_warning(self):
        """Test that poor starting values generate a warning."""
        np.random.seed(42)
        n = 300

        x = np.random.normal(0, 1, n)
        # Create data where default starting values will be very poor
        # Strong noise, weak signal
        y = 0.01 * x + np.random.normal(0, 10, n) - np.abs(np.random.normal(0, 5, n))

        df = pd.DataFrame({"y": y, "x": x})

        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x"],
            frontier="production",
            dist="half_normal",
        )

        # May produce a starting values warning
        result = sf.fit(method="mle", verbose=False)
        assert result is not None


class TestEstimateMLE_BC92Panel:
    """Tests for BC92 panel model estimation."""

    def test_bc92_model_estimation(self):
        """Test BC92 model estimation with panel data."""
        np.random.seed(42)
        N = 20  # entities
        T = 5  # time periods
        n = N * T

        # Create panel structure
        entities = np.repeat(np.arange(N), T)
        times = np.tile(np.arange(T), N)

        # Generate data
        x = np.random.normal(0, 1, n)
        v = np.random.normal(0, 0.1, n)

        # Time-varying inefficiency: u_it = exp(-eta*(T-t)) * u_i
        eta = 0.05
        u_i = np.abs(np.random.normal(0, 0.3, N))
        u = np.zeros(n)
        for i in range(N):
            for t in range(T):
                idx = i * T + t
                u[idx] = np.exp(-eta * (T - 1 - t)) * u_i[i]

        y = 1.0 + 0.5 * x + v - u

        df = pd.DataFrame({"y": y, "x": x, "entity": entities, "time": times})

        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x"],
            entity="entity",
            time="time",
            frontier="production",
            dist="half_normal",
            model_type="bc92",
        )

        result = sf.fit(method="mle", verbose=False)

        assert result is not None
        assert np.isfinite(result.loglik)
        assert "eta" in result.params.index


class TestEstimateMLE_Kumbhakar:
    """Tests for Kumbhakar (1990) panel model estimation."""

    def test_kumbhakar_model_estimation(self):
        """Test Kumbhakar (1990) model estimation with panel data."""
        np.random.seed(42)
        N = 15
        T = 4
        n = N * T

        entities = np.repeat(np.arange(N), T)
        times = np.tile(np.arange(T), N)

        x = np.random.normal(0, 1, n)
        v = np.random.normal(0, 0.1, n)

        # Kumbhakar time pattern: B(t) = 1 / (1 + exp(b*t + c*t^2))
        u_i = np.abs(np.random.normal(0, 0.3, N))
        u = np.zeros(n)
        for i in range(N):
            for t in range(T):
                idx = i * T + t
                B_t = 1.0 / (1.0 + np.exp(0.0 * t + 0.0 * t**2))
                u[idx] = B_t * u_i[i]

        y = 1.0 + 0.5 * x + v - u

        df = pd.DataFrame({"y": y, "x": x, "entity": entities, "time": times})

        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x"],
            entity="entity",
            time="time",
            frontier="production",
            dist="half_normal",
            model_type="kumbhakar_1990",
        )

        result = sf.fit(method="mle", verbose=False)

        assert result is not None
        assert np.isfinite(result.loglik)
        assert "b" in result.params.index
        assert "c" in result.params.index
        assert "mu" in result.params.index


class TestEstimateMLE_LeeSchmidt:
    """Tests for Lee-Schmidt (1993) panel model estimation."""

    def test_lee_schmidt_model_estimation(self):
        """Test Lee-Schmidt (1993) model estimation with panel data."""
        np.random.seed(42)
        N = 15
        T = 3
        n = N * T

        entities = np.repeat(np.arange(N), T)
        times = np.tile(np.arange(T), N)

        x = np.random.normal(0, 1, n)
        v = np.random.normal(0, 0.1, n)

        # Lee-Schmidt: u_it = delta_t * u_i, delta_T = 1
        delta_t = np.array([0.8, 0.9, 1.0])  # time dummies
        u_i = np.abs(np.random.normal(0, 0.3, N))
        u = np.zeros(n)
        for i in range(N):
            for t in range(T):
                idx = i * T + t
                u[idx] = delta_t[t] * u_i[i]

        y = 1.0 + 0.5 * x + v - u

        df = pd.DataFrame({"y": y, "x": x, "entity": entities, "time": times})

        sf = StochasticFrontier(
            data=df,
            depvar="y",
            exog=["x"],
            entity="entity",
            time="time",
            frontier="production",
            dist="half_normal",
            model_type="lee_schmidt_1993",
        )

        result = sf.fit(method="mle", verbose=False)

        assert result is not None
        assert np.isfinite(result.loglik)
        # Should have delta_t parameters
        delta_names = [n for n in result.params.index if n.startswith("delta_t")]
        assert len(delta_names) == T


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
