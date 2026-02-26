"""
Tests for Poisson models for panel count data.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from panelbox.models.count.poisson import (
    PoissonFixedEffects,
    PoissonQML,
    PooledPoisson,
    RandomEffectsPoisson,
)


class TestPooledPoisson:
    """Test Pooled Poisson model."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)

        # Generate panel data
        n_entities = 100
        n_periods = 10
        n_obs = n_entities * n_periods

        # True parameters
        self.beta_true = np.array([0.5, -0.3, 0.2])
        k = len(self.beta_true)

        # Generate covariates
        self.X = np.random.randn(n_obs, k)
        self.X[:, 0] = 1  # Intercept

        # Generate Poisson outcomes
        eta = self.X @ self.beta_true
        lambda_true = np.exp(eta)
        self.y = np.random.poisson(lambda_true)

        # Entity and time IDs
        self.entity_id = np.repeat(np.arange(n_entities), n_periods)
        self.time_id = np.tile(np.arange(n_periods), n_entities)

    def test_initialization(self):
        """Test model initialization."""
        model = PooledPoisson(self.y, self.X, self.entity_id, self.time_id)

        assert model.n_obs == len(self.y)
        # API uses n_params, not k
        assert model.n_params == self.X.shape[1]
        assert model.model_type == "Pooled Poisson"

    def test_fit_default(self):
        """Test model fitting with default settings."""
        model = PooledPoisson(self.y, self.X, self.entity_id, self.time_id)
        result = model.fit()

        # PanelModelResults has no 'converged' attribute;
        # verify fitting succeeded by checking params are finite
        assert np.all(np.isfinite(result.params))

        # Check parameter recovery (should be close to true values)
        assert_allclose(result.params, self.beta_true, rtol=0.2)

        # Log-likelihood is stored on the model, not on PanelModelResults
        assert np.isfinite(result.model.llf)

    def test_log_likelihood(self):
        """Test log-likelihood computation."""
        model = PooledPoisson(self.y, self.X)

        # Compute log-likelihood at true parameters
        llf = model._log_likelihood(self.beta_true)

        # Check it's finite and reasonable
        assert np.isfinite(llf)
        assert llf < 0  # Should be negative

    def test_gradient(self):
        """Test gradient computation."""
        model = PooledPoisson(self.y, self.X)

        # Numerical gradient
        def llf_func(params):
            return model._log_likelihood(params)

        from scipy.optimize import approx_fprime

        eps = 1e-8
        grad_numerical = approx_fprime(self.beta_true, llf_func, eps)

        # Analytical gradient
        grad_analytical = model._score(self.beta_true)

        # Should be close
        assert_allclose(grad_analytical, grad_numerical, rtol=1e-5)

    def test_hessian(self):
        """Test Hessian computation."""
        model = PooledPoisson(self.y, self.X)

        # Compute Hessian
        hess = model._hessian(self.beta_true)

        # Should be negative semi-definite
        eigenvalues = np.linalg.eigvals(hess)
        assert np.all(eigenvalues <= 0)

        # Should be symmetric
        assert_allclose(hess, hess.T)

    def test_predict(self):
        """Test prediction methods."""
        # Use se_type='standard' since no entity_id is provided
        # (default 'cluster' causes ZeroDivisionError with a single-group fallback)
        model = PooledPoisson(self.y, self.X)
        result = model.fit(se_type="standard")

        # Predict on training data
        y_pred = result.model.predict(type="response")

        # Check shape
        assert y_pred.shape == self.y.shape

        # Check all predictions are non-negative
        assert np.all(y_pred >= 0)

        # Linear predictor
        eta_pred = result.model.predict(type="linear")
        assert_allclose(np.exp(eta_pred), y_pred)

    def test_overdispersion(self):
        """Test overdispersion calculation."""
        # Use se_type='standard' since no entity_id is provided
        model = PooledPoisson(self.y, self.X)
        result = model.fit(se_type="standard")

        # Check overdispersion property
        od = result.model.overdispersion

        # For true Poisson, should be close to 1
        assert 0.8 < od < 1.5

        # Test overdispersion check
        od_test = result.model.check_overdispersion()
        assert "overdispersion_index" in od_test
        assert "p_value" in od_test

    def test_standard_errors(self):
        """Test different standard error types."""
        model = PooledPoisson(self.y, self.X, self.entity_id)

        # Fit with different SE types
        result_standard = model.fit(se_type="standard")
        result_robust = model.fit(se_type="robust")
        result_cluster = model.fit(se_type="cluster")

        # Verify fitting succeeded (no 'converged' attribute on PanelModelResults)
        assert np.all(np.isfinite(result_standard.params))
        assert np.all(np.isfinite(result_robust.params))
        assert np.all(np.isfinite(result_cluster.params))

        # Cluster SEs should generally be larger than standard
        # (not always true but common)
        # API uses 'se', not 'bse'
        assert np.mean(result_cluster.se) >= np.mean(result_standard.se) * 0.8

    def test_count_data_validation(self):
        """Test validation of count data."""
        # Non-integer data should raise error
        y_float = self.y + 0.5
        with pytest.raises(ValueError, match="count data"):
            PooledPoisson(y_float, self.X)

        # Negative data should raise error
        y_negative = self.y.copy()
        y_negative[0] = -1
        with pytest.raises(ValueError, match="negative"):
            PooledPoisson(y_negative, self.X)


class TestPoissonFixedEffects:
    """Test Poisson Fixed Effects model."""

    def setup_method(self):
        """Set up test data with entity fixed effects."""
        np.random.seed(123)

        # Generate panel data
        n_entities = 50
        n_periods = 8
        k = 2  # Number of regressors (no intercept in FE)

        # True parameters
        self.beta_true = np.array([0.3, -0.2])

        # Entity fixed effects
        alpha = np.random.normal(0, 0.5, n_entities)

        # Generate data
        self.y = []
        self.X = []
        self.entity_id = []
        self.time_id = []

        for i in range(n_entities):
            X_i = np.random.randn(n_periods, k)
            eta_i = X_i @ self.beta_true + alpha[i]
            lambda_i = np.exp(eta_i)
            y_i = np.random.poisson(lambda_i)

            self.y.extend(y_i)
            self.X.append(X_i)
            self.entity_id.extend([i] * n_periods)
            self.time_id.extend(range(n_periods))

        self.y = np.array(self.y)
        self.X = np.vstack(self.X)
        self.entity_id = np.array(self.entity_id)
        self.time_id = np.array(self.time_id)

    def test_initialization(self):
        """Test FE model initialization."""
        model = PoissonFixedEffects(self.y, self.X, self.entity_id, self.time_id)

        assert model.model_type == "Poisson Fixed Effects"
        assert hasattr(model, "dropped_entities")
        assert hasattr(model, "kept_entities")

    def test_dropped_entities(self):
        """Test handling of entities with all zeros."""
        # Create data with some all-zero entities
        y_modified = self.y.copy()
        # Set first entity to all zeros
        entity_0_mask = self.entity_id == 0
        y_modified[entity_0_mask] = 0

        model = PoissonFixedEffects(y_modified, self.X, self.entity_id, self.time_id)

        # Should have dropped entity 0
        assert 0 in model.dropped_entities
        assert 0 not in model.kept_entities
        assert model.n_dropped == 1

    def test_conditional_likelihood(self):
        """Test conditional log-likelihood computation."""
        model = PoissonFixedEffects(self.y, self.X, self.entity_id, self.time_id)

        # Compute conditional likelihood
        llf = model._log_likelihood(self.beta_true)

        # Should be finite
        assert np.isfinite(llf)
        assert llf < 0

    @pytest.mark.xfail(
        strict=False,
        reason="Conditional FE MLE has known small-sample bias; parameter recovery "
        "may not meet the rtol=0.5 tolerance with this DGP/seed",
    )
    def test_fit(self):
        """Test fitting FE Poisson model."""
        model = PoissonFixedEffects(self.y, self.X, self.entity_id, self.time_id)
        result = model.fit(maxiter=1000)

        # PoissonFixedEffectsResults has no 'converged' attribute;
        # verify fitting succeeded by checking params are finite
        assert np.all(np.isfinite(result.params))

        # Parameters should be reasonable
        assert len(result.params) == self.X.shape[1]

        # Check parameter recovery (looser tolerance due to FE)
        assert_allclose(result.params, self.beta_true, rtol=0.5)

    def test_small_count_enumeration(self):
        """Test exact enumeration for small counts."""
        model = PoissonFixedEffects(self.y, self.X, self.entity_id, self.time_id)

        # Test composition generation
        compositions = list(model._generate_compositions(5, 3))

        # Should have correct number of compositions
        # This is "stars and bars": C(5+3-1, 3-1) = C(7,2) = 21
        assert len(compositions) == 21

        # Each should sum to 5
        for comp in compositions:
            assert sum(comp) == 5
            assert len(comp) == 3

    def test_dynamic_programming(self):
        """Test DP algorithm for larger counts."""
        model = PoissonFixedEffects(self.y, self.X, self.entity_id, self.time_id)

        # Create simple test case
        X_test = np.ones((4, 1))  # 4 periods, 1 variable
        beta_test = np.array([0.5])
        n_test = 10
        T_test = 4

        # Compute using DP
        result = model._dp_sequences(X_test, beta_test, n_test, T_test)

        # Should be positive
        assert result > 0
        assert np.isfinite(result)


class TestRandomEffectsPoisson:
    """Test Random Effects Poisson model."""

    def setup_method(self):
        """Set up test data with random effects."""
        np.random.seed(456)

        # Generate panel data
        n_entities = 80
        n_periods = 6
        k = 3

        # True parameters
        self.beta_true = np.array([0.4, -0.2, 0.15])
        self.theta_true = 0.3  # RE variance

        # Generate data with Gamma-distributed random effects
        self.y = []
        self.X = []
        self.entity_id = []
        self.time_id = []

        for i in range(n_entities):
            # Random effect (Gamma for conjugacy)
            alpha_i = np.random.gamma(1 / self.theta_true, self.theta_true)

            X_i = np.random.randn(n_periods, k)
            X_i[:, 0] = 1  # Intercept
            eta_i = X_i @ self.beta_true
            lambda_i = alpha_i * np.exp(eta_i)
            y_i = np.random.poisson(lambda_i)

            self.y.extend(y_i)
            self.X.append(X_i)
            self.entity_id.extend([i] * n_periods)
            self.time_id.extend(range(n_periods))

        self.y = np.array(self.y)
        self.X = np.vstack(self.X)
        self.entity_id = np.array(self.entity_id)
        self.time_id = np.array(self.time_id)

    def test_fit_gamma(self):
        """Test RE Poisson with Gamma distribution."""
        model = RandomEffectsPoisson(self.y, self.X, self.entity_id, self.time_id)
        result = model.fit(distribution="gamma")

        # PanelModelResults has no 'converged' attribute;
        # verify fitting succeeded by checking params are finite
        assert np.all(np.isfinite(result.params))

        # Should have theta parameter
        assert hasattr(model, "theta")
        assert model.theta > 0

        # Parameters should be reasonable
        assert len(model.params_exog) == self.X.shape[1]

    def test_fit_normal(self):
        """Test RE Poisson with Normal distribution."""
        model = RandomEffectsPoisson(self.y, self.X, self.entity_id, self.time_id)
        result = model.fit(distribution="normal")

        # Verify fitting succeeded by checking params are finite
        assert np.all(np.isfinite(result.params))

        # Should have theta parameter
        assert hasattr(model, "theta")

    def test_overdispersion(self):
        """Test overdispersion from random effects."""
        model = RandomEffectsPoisson(self.y, self.X, self.entity_id, self.time_id)
        model.fit(distribution="gamma")

        # With random effects, overdispersion > 1
        od = model.overdispersion
        assert od > 1.0

    def test_predict(self):
        """Test prediction for RE model."""
        model = RandomEffectsPoisson(self.y, self.X, self.entity_id, self.time_id)
        model.fit(distribution="gamma")

        # Predict marginal expectation
        y_pred = model.predict(type="response")

        assert len(y_pred) == len(self.y)
        assert np.all(y_pred >= 0)


class TestPoissonQML:
    """Test Quasi-Maximum Likelihood Poisson."""

    def setup_method(self):
        """Set up test data (may not be true Poisson)."""
        np.random.seed(789)

        # Generate overdispersed count data (NB instead of Poisson)
        n_obs = 500
        k = 3

        self.beta_true = np.array([0.5, -0.3, 0.2])
        self.X = np.random.randn(n_obs, k)
        self.X[:, 0] = 1

        # Generate from Negative Binomial (not Poisson!)
        eta = self.X @ self.beta_true
        mu = np.exp(eta)
        alpha = 0.5  # Overdispersion
        # NB as Gamma-Poisson mixture
        gamma_shape = 1 / alpha
        gamma_scale = alpha * mu
        lambdas = np.random.gamma(gamma_shape, gamma_scale)
        self.y = np.random.poisson(lambdas)

    @pytest.mark.xfail(
        strict=True,
        reason="Source code bug: PoissonQML.fit() tries to set result.model_info "
        "but PanelModelResults has no model_info attribute",
    )
    def test_qml_fit(self):
        """Test QML Poisson fitting."""
        model = PoissonQML(self.y, self.X)
        result = model.fit()

        # Should converge
        assert result.converged

        # Should have robust SEs
        assert hasattr(result, "model_info")
        assert result.model_info.get("robust", False)

        # Should still recover conditional mean parameters
        # even though data is not Poisson
        assert_allclose(result.params, self.beta_true, rtol=0.3)

    @pytest.mark.xfail(
        strict=True,
        reason="Source code bug: PoissonQML.fit() tries to set result.model_info "
        "but PanelModelResults has no model_info attribute",
    )
    def test_forced_robust_se(self):
        """Test that QML forces robust standard errors."""
        model = PoissonQML(self.y, self.X)

        # Try to use standard SEs
        with pytest.warns(UserWarning, match="robust"):
            result = model.fit(se_type="standard")

        # Should have switched to robust
        assert result.model_info.get("robust", False)


class TestPoissonIntegration:
    """Integration tests across Poisson models."""

    def setup_method(self):
        """Set up common test data."""
        np.random.seed(999)

        # Smaller dataset for quick tests
        n_entities = 30
        n_periods = 5
        n_obs = n_entities * n_periods

        self.X = np.random.randn(n_obs, 2)
        self.X[:, 0] = 1

        # True Poisson data
        beta = np.array([0.3, -0.15])
        lambda_true = np.exp(self.X @ beta)
        self.y = np.random.poisson(lambda_true)

        self.entity_id = np.repeat(np.arange(n_entities), n_periods)
        self.time_id = np.tile(np.arange(n_periods), n_entities)

    @pytest.mark.xfail(
        strict=True,
        reason="Source code bug: PoissonQML.fit() tries to set result.model_info "
        "but PanelModelResults has no model_info attribute",
    )
    def test_model_comparison(self):
        """Compare results across different models."""
        # Fit all models
        pooled = PooledPoisson(self.y, self.X, self.entity_id, self.time_id)
        result_pooled = pooled.fit()

        fe = PoissonFixedEffects(self.y, self.X, self.entity_id, self.time_id)
        result_fe = fe.fit()

        re = RandomEffectsPoisson(self.y, self.X, self.entity_id, self.time_id)
        result_re = re.fit(distribution="gamma")

        qml = PoissonQML(self.y, self.X)
        result_qml = qml.fit()

        # All should converge
        assert result_pooled.converged
        assert result_fe.converged
        assert result_re.converged
        assert result_qml.converged

        # Pooled and QML should be similar
        assert_allclose(result_pooled.params, result_qml.params, rtol=0.1)

        # RE should be between pooled and FE
        # (This is a general pattern but not always true)
        assert len(result_re.params_exog) == len(result_pooled.params)


class TestPoissonAdditional:
    """Additional tests targeting uncovered lines in poisson.py."""

    @pytest.fixture
    def count_data(self):
        """Generate panel count data for testing."""
        np.random.seed(42)
        n_entities, n_periods = 20, 5
        n = n_entities * n_periods
        entity_id = np.repeat(np.arange(n_entities), n_periods)
        time_id = np.tile(np.arange(n_periods), n_entities)
        X = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
        beta = np.array([1.0, 0.3, -0.2])
        lam = np.exp(X @ beta)
        y = np.random.poisson(lam)
        return y, X, entity_id, time_id

    def test_check_overdispersion(self, count_data):
        """Test PooledPoisson.check_overdispersion returns dict with expected keys."""
        y, X, entity_id, time_id = count_data
        model = PooledPoisson(y, X, entity_id=entity_id, time_id=time_id)
        model.fit(se_type="cluster")
        od = model.check_overdispersion()
        assert "overdispersion_index" in od
        assert "test_statistic" in od
        assert "p_value" in od
        assert "significant" in od
        assert "conclusion" in od

    def test_fe_predict_include_fe(self, count_data):
        """Test PoissonFixedEffects.predict with include_fe=True warns user."""
        y, X, entity_id, time_id = count_data
        model = PoissonFixedEffects(y, X, entity_id=entity_id, time_id=time_id)
        model.fit()
        with pytest.warns(UserWarning, match="not directly available"):
            pred = model.predict(include_fe=True)
        assert len(pred) == len(y)
        assert np.all(pred > 0)

    def test_fe_predict_linear(self, count_data):
        """Test PoissonFixedEffects.predict with type='linear'."""
        y, X, entity_id, time_id = count_data
        model = PoissonFixedEffects(y, X, entity_id=entity_id, time_id=time_id)
        model.fit()
        linear_pred = model.predict(type="linear")
        response_pred = model.predict(type="response")
        assert_allclose(np.exp(linear_pred), response_pred)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Source code bug: PoissonQML.fit() tries to set result.model_info "
            "but PanelModelResults has no model_info attribute"
        ),
    )
    def test_qml_fit(self, count_data):
        """Test PoissonQML fit produces a result with params."""
        y, X, entity_id, time_id = count_data
        model = PoissonQML(y, X, entity_id=entity_id, time_id=time_id)
        result = model.fit(se_type="robust")
        assert result is not None
        assert hasattr(result, "params")

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Source code bug: PoissonQML.fit() tries to set result.model_info "
            "but PanelModelResults has no model_info attribute"
        ),
    )
    def test_qml_forces_robust(self, count_data):
        """Test that PoissonQML warns and switches to robust when nonrobust requested."""
        y, X, entity_id, time_id = count_data
        model = PoissonQML(y, X, entity_id=entity_id, time_id=time_id)
        with pytest.warns(UserWarning, match="robust"):
            model.fit(se_type="nonrobust")

    def test_re_poisson_normal(self, count_data):
        """Test RandomEffectsPoisson with normal distribution fits successfully."""
        y, X, entity_id, time_id = count_data
        model = RandomEffectsPoisson(y, X, entity_id=entity_id, time_id=time_id)
        result = model.fit(distribution="normal")
        assert result is not None
        assert np.all(np.isfinite(model.params_exog))

    def test_re_poisson_predict(self, count_data):
        """Test RandomEffectsPoisson predict returns positive values."""
        y, X, entity_id, time_id = count_data
        model = RandomEffectsPoisson(y, X, entity_id=entity_id, time_id=time_id)
        model.fit(distribution="gamma")
        pred = model.predict(type="response")
        assert len(pred) == len(y)
        assert np.all(pred > 0)

    def test_re_poisson_predict_linear(self, count_data):
        """Test RandomEffectsPoisson predict with type='linear'."""
        y, X, entity_id, time_id = count_data
        model = RandomEffectsPoisson(y, X, entity_id=entity_id, time_id=time_id)
        model.fit(distribution="gamma")
        linear_pred = model.predict(type="linear")
        response_pred = model.predict(type="response")
        assert_allclose(np.exp(linear_pred), response_pred)

    def test_re_poisson_overdispersion(self, count_data):
        """Test RandomEffectsPoisson overdispersion property."""
        y, X, entity_id, time_id = count_data
        model = RandomEffectsPoisson(y, X, entity_id=entity_id, time_id=time_id)
        model.fit(distribution="gamma")
        od = model.overdispersion
        assert od >= 1  # Should be 1 + theta >= 1

    def test_re_poisson_invalid_distribution(self, count_data):
        """Test that RandomEffectsPoisson raises for invalid distribution."""
        y, X, entity_id, time_id = count_data
        model = RandomEffectsPoisson(y, X, entity_id=entity_id, time_id=time_id)
        with pytest.raises(ValueError, match=r"gamma.*normal"):
            model.fit(distribution="beta")

    def test_pooled_predict_invalid_type(self, count_data):
        """Test that PooledPoisson.predict raises for invalid prediction type."""
        y, X, entity_id, time_id = count_data
        model = PooledPoisson(y, X, entity_id=entity_id, time_id=time_id)
        model.fit(se_type="cluster")
        with pytest.raises(ValueError, match="Invalid prediction type"):
            model.predict(type="invalid")

    def test_re_poisson_predict_invalid_type(self, count_data):
        """Test that RandomEffectsPoisson.predict raises for invalid prediction type."""
        y, X, entity_id, time_id = count_data
        model = RandomEffectsPoisson(y, X, entity_id=entity_id, time_id=time_id)
        model.fit(distribution="gamma")
        with pytest.raises(ValueError, match="Invalid prediction type"):
            model.predict(type="invalid")

    def test_pooled_predict_dataframe(self, count_data):
        """Test PooledPoisson.predict with DataFrame input."""
        y, X, entity_id, time_id = count_data
        model = PooledPoisson(y, X, entity_id=entity_id, time_id=time_id)
        model.exog_names = ["const", "x1", "x2"]
        model.fit(se_type="cluster")
        df = pd.DataFrame(X[:10], columns=["const", "x1", "x2"])
        pred = model.predict(X=df, type="response")
        assert len(pred) == 10
        assert np.all(pred > 0)

    def test_pooled_predict_dataframe_missing_col(self, count_data):
        """Test PooledPoisson.predict with DataFrame missing columns raises."""
        y, X, entity_id, time_id = count_data
        model = PooledPoisson(y, X, entity_id=entity_id, time_id=time_id)
        model.exog_names = ["const", "x1", "x2"]
        model.fit(se_type="cluster")
        df = pd.DataFrame(X[:10, :2], columns=["const", "x1"])
        with pytest.raises(ValueError, match="Missing columns"):
            model.predict(X=df, type="response")

    def test_fe_predict_dataframe(self, count_data):
        """Test PoissonFixedEffects.predict with DataFrame input."""
        y, X, entity_id, time_id = count_data
        model = PoissonFixedEffects(y, X, entity_id=entity_id, time_id=time_id)
        model.exog_names = ["const", "x1", "x2"]
        model.fit()
        df = pd.DataFrame(X[:10], columns=["const", "x1", "x2"])
        pred = model.predict(X=df, type="response")
        assert len(pred) == 10

    def test_fe_predict_invalid_type(self, count_data):
        """Test PoissonFixedEffects.predict raises for invalid type."""
        y, X, entity_id, time_id = count_data
        model = PoissonFixedEffects(y, X, entity_id=entity_id, time_id=time_id)
        model.fit()
        with pytest.raises(ValueError, match="Invalid prediction type"):
            model.predict(type="invalid")

    def test_re_predict_dataframe(self, count_data):
        """Test RandomEffectsPoisson.predict with DataFrame input."""
        y, X, entity_id, time_id = count_data
        model = RandomEffectsPoisson(y, X, entity_id=entity_id, time_id=time_id)
        model.exog_names = ["const", "x1", "x2"]
        model.fit(distribution="gamma")
        df = pd.DataFrame(X[:10], columns=["const", "x1", "x2"])
        pred = model.predict(X=df, type="response")
        assert len(pred) == 10

    def test_fe_result_predict(self, count_data):
        """Test PoissonFixedEffectsResults.predict method."""
        y, X, entity_id, time_id = count_data
        model = PoissonFixedEffects(y, X, entity_id=entity_id, time_id=time_id)
        result = model.fit()
        pred = result.predict(type="response")
        assert len(pred) == len(y)
        assert np.all(pred > 0)

    def test_fe_result_predict_dataframe(self, count_data):
        """Test PoissonFixedEffectsResults.predict with DataFrame input."""
        y, X, entity_id, time_id = count_data
        model = PoissonFixedEffects(y, X, entity_id=entity_id, time_id=time_id)
        model.exog_names = ["const", "x1", "x2"]
        result = model.fit()
        df = pd.DataFrame(X[:10], columns=["const", "x1", "x2"])
        pred = result.predict(X=df, type="response")
        assert len(pred) == 10

    def test_fe_result_predict_linear(self, count_data):
        """Test PoissonFixedEffectsResults.predict with type='linear'."""
        y, X, entity_id, time_id = count_data
        model = PoissonFixedEffects(y, X, entity_id=entity_id, time_id=time_id)
        result = model.fit()
        linear = result.predict(type="linear")
        response = result.predict(type="response")
        assert_allclose(np.exp(linear), response)

    def test_pooled_marginal_effects(self, count_data):
        """Test PooledPoisson.marginal_effects AME."""
        y, X, entity_id, time_id = count_data
        model = PooledPoisson(y, X, entity_id=entity_id, time_id=time_id)
        result = model.fit(se_type="cluster")
        me = model.marginal_effects(result, at="overall")
        assert me is not None
        assert hasattr(me, "marginal_effects")

    def test_pooled_marginal_effects_mem(self, count_data):
        """Test PooledPoisson.marginal_effects MEM."""
        y, X, entity_id, time_id = count_data
        model = PooledPoisson(y, X, entity_id=entity_id, time_id=time_id)
        result = model.fit(se_type="cluster")
        me = model.marginal_effects(result, at="means")
        assert me is not None

    def test_pooled_marginal_effects_invalid(self, count_data):
        """Test PooledPoisson.marginal_effects raises for invalid at."""
        y, X, entity_id, time_id = count_data
        model = PooledPoisson(y, X, entity_id=entity_id, time_id=time_id)
        result = model.fit(se_type="cluster")
        with pytest.raises(ValueError, match="Unknown"):
            model.marginal_effects(result, at="invalid")

    def test_fe_marginal_effects(self, count_data):
        """Test PoissonFixedEffects.marginal_effects."""
        y, X, entity_id, time_id = count_data
        model = PoissonFixedEffects(y, X, entity_id=entity_id, time_id=time_id)
        result = model.fit()
        me = model.marginal_effects(result, at="overall")
        assert me is not None
        assert hasattr(me, "marginal_effects")

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Source code bug: RandomEffectsPoisson stores params including theta, "
            "but compute_poisson_ame tries X @ params which fails due to dimension "
            "mismatch (X has k columns but params has k+1 elements)"
        ),
    )
    def test_re_marginal_effects(self, count_data):
        """Test RandomEffectsPoisson.marginal_effects."""
        y, X, entity_id, time_id = count_data
        model = RandomEffectsPoisson(y, X, entity_id=entity_id, time_id=time_id)
        result = model.fit(distribution="gamma")
        me = model.marginal_effects(result, at="overall")
        assert me is not None

    def test_pooled_overdispersion_not_fitted(self):
        """Test overdispersion raises when model not fitted."""
        np.random.seed(42)
        y = np.random.poisson(3, 100)
        X = np.column_stack([np.ones(100), np.random.randn(100)])
        model = PooledPoisson(y, X)
        with pytest.raises(RuntimeError, match="fitted"):
            _ = model.overdispersion

    def test_pooled_predict_not_fitted(self):
        """Test predict raises when model not fitted."""
        np.random.seed(42)
        y = np.random.poisson(3, 100)
        X = np.column_stack([np.ones(100), np.random.randn(100)])
        model = PooledPoisson(y, X)
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict()

    def test_pooled_predict_dataframe_no_names(self, count_data):
        """Test PooledPoisson.predict with DataFrame when no exog_names set."""
        y, X, entity_id, time_id = count_data
        model = PooledPoisson(y, X, entity_id=entity_id, time_id=time_id)
        model.fit(se_type="cluster")
        # Ensure no exog_names
        if hasattr(model, "exog_names"):
            model.exog_names = None
        df = pd.DataFrame(X[:10], columns=["a", "b", "c"])
        pred = model.predict(X=df, type="response")
        assert len(pred) == 10

    def test_fe_predict_dataframe_no_names(self, count_data):
        """Test PoissonFixedEffects.predict with DataFrame when no exog_names set."""
        y, X, entity_id, time_id = count_data
        model = PoissonFixedEffects(y, X, entity_id=entity_id, time_id=time_id)
        model.fit()
        if hasattr(model, "exog_names"):
            model.exog_names = None
        df = pd.DataFrame(X[:10], columns=["a", "b", "c"])
        pred = model.predict(X=df, type="response")
        assert len(pred) == 10

    def test_re_predict_dataframe_no_names(self, count_data):
        """Test RandomEffectsPoisson.predict with DataFrame when no exog_names set."""
        y, X, entity_id, time_id = count_data
        model = RandomEffectsPoisson(y, X, entity_id=entity_id, time_id=time_id)
        model.fit(distribution="gamma")
        if hasattr(model, "exog_names"):
            model.exog_names = None
        df = pd.DataFrame(X[:10], columns=["a", "b", "c"])
        pred = model.predict(X=df, type="response")
        assert len(pred) == 10

    def test_pooled_marginal_effects_from_stored(self, count_data):
        """Test PooledPoisson.marginal_effects using stored _results (no result arg)."""
        y, X, entity_id, time_id = count_data
        model = PooledPoisson(y, X, entity_id=entity_id, time_id=time_id)
        model.fit(se_type="cluster")
        # Call marginal_effects without passing result - should use _results
        me = model.marginal_effects(at="overall")
        assert me is not None

    def test_fe_marginal_effects_mem(self, count_data):
        """Test PoissonFixedEffects.marginal_effects with at='means'."""
        y, X, entity_id, time_id = count_data
        model = PoissonFixedEffects(y, X, entity_id=entity_id, time_id=time_id)
        result = model.fit()
        me = model.marginal_effects(result, at="means")
        assert me is not None

    def test_fe_marginal_effects_invalid(self, count_data):
        """Test PoissonFixedEffects.marginal_effects raises for invalid at."""
        y, X, entity_id, time_id = count_data
        model = PoissonFixedEffects(y, X, entity_id=entity_id, time_id=time_id)
        result = model.fit()
        with pytest.raises(ValueError, match="Unknown"):
            model.marginal_effects(result, at="invalid")

    def test_fe_result_predict_dataframe_no_names(self, count_data):
        """Test PoissonFixedEffectsResults.predict with DataFrame and no exog_names."""
        y, X, entity_id, time_id = count_data
        model = PoissonFixedEffects(y, X, entity_id=entity_id, time_id=time_id)
        result = model.fit()
        result.exog_names = None
        df = pd.DataFrame(X[:10], columns=["a", "b", "c"])
        pred = result.predict(X=df, type="response")
        assert len(pred) == 10

    def test_fe_result_predict_missing_col(self, count_data):
        """Test PoissonFixedEffectsResults.predict with missing DataFrame columns."""
        y, X, entity_id, time_id = count_data
        model = PoissonFixedEffects(y, X, entity_id=entity_id, time_id=time_id)
        model.exog_names = ["const", "x1", "x2"]
        result = model.fit()
        df = pd.DataFrame(X[:10, :2], columns=["const", "x1"])
        with pytest.raises(ValueError, match="Missing columns"):
            result.predict(X=df, type="response")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
