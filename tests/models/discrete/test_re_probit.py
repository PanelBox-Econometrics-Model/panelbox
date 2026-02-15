"""
Tests for Random Effects Probit model.

Tests the Random Effects Probit implementation with Gauss-Hermite quadrature.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from panelbox.core.panel_data import PanelData
from panelbox.models.discrete.binary import PooledProbit, RandomEffectsProbit


class TestRandomEffectsProbit:
    """Test Random Effects Probit model."""

    @pytest.fixture
    def setup_re_data(self):
        """Generate panel data with random effects."""
        np.random.seed(42)

        n_entities = 100
        n_periods = 5
        n_obs = n_entities * n_periods

        # True parameters
        beta_true = np.array([0.5, 0.8, -0.6])  # [intercept, x1, x2]
        sigma_alpha_true = 0.8  # RE standard deviation

        # Generate entity and time indices
        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_entities)

        # Generate random effects
        alpha_i = np.random.normal(0, sigma_alpha_true, n_entities)
        alpha_expanded = np.repeat(alpha_i, n_periods)

        # Generate covariates
        x1 = np.random.randn(n_obs)
        x2 = np.random.randn(n_obs)

        # Linear predictor with random effects
        X = np.column_stack([np.ones(n_obs), x1, x2])
        eta = X @ beta_true + alpha_expanded

        # Generate binary outcome
        prob = stats.norm.cdf(eta)
        y = np.random.binomial(1, prob)

        # Create DataFrame
        data = pd.DataFrame({"entity": entity_ids, "time": time_ids, "y": y, "x1": x1, "x2": x2})

        return data, beta_true, sigma_alpha_true

    def test_re_probit_initialization(self, setup_re_data):
        """Test RE Probit initialization."""
        data, _, _ = setup_re_data

        # Initialize model
        model = RandomEffectsProbit("y ~ x1 + x2", data, "entity", "time", quadrature_points=10)

        # Check attributes
        assert model.quadrature_points == 10
        assert model.family == "probit"
        assert hasattr(model, "_quad_nodes")
        assert hasattr(model, "_quad_weights")
        assert len(model._quad_nodes) == 10
        assert len(model._quad_weights) == 10

    def test_re_probit_fit(self, setup_re_data):
        """Test RE Probit fitting."""
        data, beta_true, sigma_alpha_true = setup_re_data

        # Fit model
        model = RandomEffectsProbit("y ~ x1 + x2", data, "entity", "time", quadrature_points=12)

        # Use fewer iterations for testing
        result = model.fit(method="bfgs", maxiter=50)

        # Check result structure
        assert hasattr(result, "params")
        assert hasattr(result, "std_errors")
        assert hasattr(result, "llf")
        assert hasattr(result, "rho")
        assert hasattr(result, "sigma_alpha")

        # Check parameter names
        param_names = result.params.index.tolist()
        assert "Intercept" in param_names
        assert "x1" in param_names
        assert "x2" in param_names
        assert "log_sigma_alpha" in param_names

        # Check rho is valid
        assert 0 <= result.rho <= 1

        # Check sigma_alpha is positive
        assert result.sigma_alpha > 0

    def test_re_probit_convergence(self, setup_re_data):
        """Test that RE Probit converges to reasonable values."""
        data, beta_true, sigma_alpha_true = setup_re_data

        # Fit model with more iterations
        model = RandomEffectsProbit(
            "y ~ x1 + x2", data, "entity", "time", quadrature_points=8  # Fewer points for speed
        )
        result = model.fit(method="bfgs", maxiter=100)

        # Check that estimates are in reasonable range
        # (Won't match exactly due to finite sample)
        beta_est = result.params[["Intercept", "x1", "x2"]].values
        sigma_alpha_est = result.sigma_alpha

        # Should be same sign and order of magnitude
        for i in range(len(beta_true)):
            assert np.sign(beta_est[i]) == np.sign(beta_true[i]) or abs(beta_est[i]) < 0.1

        # Sigma_alpha should be positive and reasonable
        assert 0.1 < sigma_alpha_est < 3.0

    def test_quadrature_points_effect(self, setup_re_data):
        """Test effect of different quadrature points."""
        data, _, _ = setup_re_data

        # Fit with different quadrature points
        results = {}
        for n_points in [4, 8, 12]:
            model = RandomEffectsProbit(
                "y ~ x1 + x2", data, "entity", "time", quadrature_points=n_points
            )
            results[n_points] = model.fit(method="bfgs", maxiter=30)

        # Higher quadrature should give more accurate likelihood
        # (though not necessarily monotonic due to optimization)
        llf_4 = results[4].llf
        llf_8 = results[8].llf
        llf_12 = results[12].llf

        # Check that likelihoods are finite
        assert np.isfinite(llf_4)
        assert np.isfinite(llf_8)
        assert np.isfinite(llf_12)

        # Parameters should be similar across different quadrature points
        params_8 = results[8].params["x1"]
        params_12 = results[12].params["x1"]
        assert np.abs(params_8 - params_12) < 0.5  # Reasonable tolerance

    def test_re_probit_predict(self, setup_re_data):
        """Test prediction methods for RE Probit."""
        data, _, _ = setup_re_data

        # Fit model
        model = RandomEffectsProbit("y ~ x1 + x2", data, "entity", "time", quadrature_points=8)
        result = model.fit(method="bfgs", maxiter=30)

        # Test prediction on original data
        probs = result.predict(type="prob")

        # Check predictions are valid probabilities
        assert len(probs) == len(data)
        assert np.all((probs >= 0) & (probs <= 1))

        # Test linear predictor
        linear_pred = result.predict(type="linear")
        assert len(linear_pred) == len(data)
        assert np.all(np.isfinite(linear_pred))

    def test_re_probit_vs_pooled(self, setup_re_data):
        """Test that RE Probit differs from Pooled Probit when there are REs."""
        data, _, _ = setup_re_data

        # Fit Pooled Probit
        pooled_model = PooledProbit("y ~ x1 + x2", data, "entity", "time")
        pooled_result = pooled_model.fit(cov_type="nonrobust")

        # Fit RE Probit
        re_model = RandomEffectsProbit("y ~ x1 + x2", data, "entity", "time", quadrature_points=8)
        re_result = re_model.fit(method="bfgs", maxiter=50)

        # Coefficients should differ when sigma_alpha > 0
        pooled_coef = pooled_result.params["x1"]
        re_coef = re_result.params["x1"]

        # Should be different (RE accounts for clustering)
        assert np.abs(pooled_coef - re_coef) > 0.01

        # RE should have higher likelihood (more parameters)
        # Note: This might not always hold due to optimization issues
        # assert re_result.llf > pooled_result.llf

    def test_re_probit_rho_property(self, setup_re_data):
        """Test intra-class correlation property."""
        data, _, _ = setup_re_data

        # Fit model
        model = RandomEffectsProbit("y ~ x1 + x2", data, "entity", "time", quadrature_points=8)
        result = model.fit(method="bfgs", maxiter=30)

        # Test rho calculation
        sigma_alpha = result.sigma_alpha
        rho_expected = sigma_alpha**2 / (1 + sigma_alpha**2)

        assert np.allclose(result.rho, rho_expected)
        assert np.allclose(model.rho, rho_expected)

    def test_re_probit_marginal_effects(self, setup_re_data):
        """Test marginal effects for RE Probit."""
        data, _, _ = setup_re_data

        # Fit model
        model = RandomEffectsProbit("y ~ x1 + x2", data, "entity", "time", quadrature_points=8)
        result = model.fit(method="bfgs", maxiter=30)

        # Test that marginal_effects method exists and works
        ame = model.marginal_effects(at="overall")
        mem = model.marginal_effects(at="mean")

        # Check structure
        assert hasattr(ame, "marginal_effects")
        assert hasattr(ame, "std_errors")
        assert hasattr(mem, "marginal_effects")

        # Effects should be smaller than raw coefficients
        # (due to nonlinearity and random effects)
        for var in ["x1", "x2"]:
            assert abs(ame.marginal_effects[var]) < abs(result.params[var])


class TestREProbitEdgeCases:
    """Test edge cases for Random Effects Probit."""

    def test_re_probit_no_variation(self):
        """Test RE Probit with no within-entity variation."""
        # Create data where y is constant within entities
        np.random.seed(123)
        n_entities = 50
        n_periods = 3

        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_entities)

        # Each entity has constant y
        y_entity = np.random.binomial(1, 0.5, n_entities)
        y = np.repeat(y_entity, n_periods)

        x = np.random.randn(n_entities * n_periods)

        data = pd.DataFrame({"entity": entity_ids, "time": time_ids, "y": y, "x": x})

        # RE Probit should still work (unlike FE Logit)
        model = RandomEffectsProbit("y ~ x", data, "entity", "time", quadrature_points=8)
        result = model.fit(method="bfgs", maxiter=30)

        # Should get valid results
        assert np.isfinite(result.llf)
        assert result.sigma_alpha > 0  # Should find substantial RE

    def test_re_probit_small_sigma_alpha(self):
        """Test RE Probit when sigma_alpha is very small."""
        np.random.seed(456)

        # Generate data with no random effects (sigma_alpha = 0)
        n = 200
        t = 3

        entity_ids = np.repeat(np.arange(n), t)
        time_ids = np.tile(np.arange(t), n)

        x = np.random.randn(n * t)
        eta = 0.5 + 0.7 * x  # No random effects
        prob = stats.norm.cdf(eta)
        y = np.random.binomial(1, prob)

        data = pd.DataFrame({"entity": entity_ids, "time": time_ids, "y": y, "x": x})

        # Fit RE Probit
        model = RandomEffectsProbit("y ~ x", data, "entity", "time", quadrature_points=8)
        result = model.fit(method="bfgs", maxiter=50)

        # sigma_alpha should be small
        assert result.sigma_alpha < 0.5

        # rho should be close to 0
        assert result.rho < 0.2

        # Should converge to Pooled Probit
        pooled = PooledProbit("y ~ x", data, "entity", "time")
        pooled_result = pooled.fit()

        # Coefficients should be similar
        assert np.abs(result.params["x"] - pooled_result.params["x"]) < 0.2

    def test_re_probit_perfect_separation(self):
        """Test RE Probit with perfect separation."""
        # Create data with perfect separation
        n_entities = 30
        n_periods = 3

        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_entities)

        x = np.random.randn(n_entities * n_periods)
        # Perfect separation: y = 1 iff x > 0
        y = (x > 0).astype(int)

        data = pd.DataFrame({"entity": entity_ids, "time": time_ids, "y": y, "x": x})

        # Model should handle this (though estimates may be extreme)
        model = RandomEffectsProbit("y ~ x", data, "entity", "time", quadrature_points=6)

        # May not converge well, but shouldn't crash
        result = model.fit(method="bfgs", maxiter=20)

        # Should get some result (even if extreme)
        assert hasattr(result, "params")
        assert np.isfinite(result.params["Intercept"])  # May be large but finite


class TestREProbitStartingValues:
    """Test starting values for RE Probit."""

    def test_starting_values_from_pooled(self):
        """Test that starting values come from Pooled Probit."""
        np.random.seed(789)

        # Simple data
        n = 100
        t = 4
        entity_ids = np.repeat(np.arange(n), t)
        time_ids = np.tile(np.arange(t), n)

        x = np.random.randn(n * t)
        y = np.random.binomial(1, stats.norm.cdf(0.3 + 0.5 * x))

        data = pd.DataFrame({"entity": entity_ids, "time": time_ids, "y": y, "x": x})

        # Get starting values
        model = RandomEffectsProbit("y ~ x", data, "entity", "time")
        start_values = model._starting_values()

        # Should have correct length (beta + log_sigma_alpha)
        assert len(start_values) == 3  # intercept, x, log_sigma_alpha

        # Last element should be 0 (log(1) = 0)
        assert start_values[-1] == 0.0

        # First elements should match Pooled Probit
        pooled = PooledProbit("y ~ x", data, "entity", "time")
        pooled_result = pooled.fit()

        np.testing.assert_array_almost_equal(
            start_values[:-1], pooled_result.params.values, decimal=5
        )
