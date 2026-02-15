"""
Test suite for Negative Binomial regression models.

Tests NegativeBinomial and NegativeBinomialFixedEffects for overdispersed count data.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from panelbox.models.count.negbin import NegativeBinomial, NegativeBinomialFixedEffects
from panelbox.models.count.poisson import PooledPoisson


class TestNegativeBinomial:
    """Test Negative Binomial (NB2) model."""

    def setup_method(self):
        """Set up test data with overdispersion."""
        np.random.seed(42)

        # Generate panel data
        n_entities = 100
        n_periods = 10
        n_obs = n_entities * n_periods

        # True parameters
        self.beta_true = np.array([0.5, -0.3, 0.2])
        self.alpha_true = 0.5  # Overdispersion parameter
        k = len(self.beta_true)

        # Generate covariates
        self.X = np.random.randn(n_obs, k)
        self.X[:, 0] = 1  # Intercept

        # Generate NB2 outcomes
        eta = self.X @ self.beta_true
        mu = np.exp(eta)

        # NB2 as Gamma-Poisson mixture
        r = 1 / self.alpha_true
        p = r / (r + mu)
        self.y = np.random.negative_binomial(r, p)

        # Entity and time IDs
        self.entity_id = np.repeat(np.arange(n_entities), n_periods)
        self.time_id = np.tile(np.arange(n_periods), n_entities)

    def test_initialization(self):
        """Test NB model initialization."""
        model = NegativeBinomial(self.y, self.X, self.entity_id, self.time_id)

        assert model.model_type == "Negative Binomial (NB2)"
        assert model.n_obs == len(self.y)

    def test_fit(self):
        """Test NB model fitting."""
        model = NegativeBinomial(self.y, self.X, self.entity_id, self.time_id)
        result = model.fit()

        # Check convergence
        assert hasattr(result, "params")
        assert hasattr(result, "alpha")

        # Check parameter recovery
        beta_est = result.params[:-1]
        assert_allclose(beta_est, self.beta_true, rtol=0.3)

        # Check alpha recovery
        assert result.alpha > 0
        assert_allclose(result.alpha, self.alpha_true, rtol=0.5)

    def test_log_likelihood(self):
        """Test NB2 log-likelihood computation."""
        model = NegativeBinomial(self.y, self.X)

        # Parameters with log(alpha)
        params = np.append(self.beta_true, np.log(self.alpha_true))

        llf = model._log_likelihood(params)

        # Should be finite and negative
        assert np.isfinite(llf)
        assert llf < 0

    def test_score(self):
        """Test score function."""
        model = NegativeBinomial(self.y, self.X)

        params = np.append(self.beta_true, np.log(self.alpha_true))

        # Analytical score
        score_analytical = model._score(params)

        # Numerical score
        def llf_func(p):
            return model._log_likelihood(p)

        from scipy.optimize import approx_fprime

        eps = 1e-8
        score_numerical = approx_fprime(params, llf_func, eps)

        # Should be close
        assert_allclose(score_analytical, score_numerical, rtol=1e-4)

    def test_predict(self):
        """Test prediction."""
        model = NegativeBinomial(self.y, self.X)
        result = model.fit()

        # Predict expected counts
        y_pred = model.predict(type="response")

        assert y_pred.shape == self.y.shape
        assert np.all(y_pred >= 0)

        # Linear predictor
        eta_pred = model.predict(type="linear")
        assert_allclose(np.exp(eta_pred), y_pred)

    def test_overdispersion(self):
        """Test overdispersion calculation."""
        model = NegativeBinomial(self.y, self.X)
        result = model.fit()

        od = model.overdispersion

        # Should be > 1 for NB model
        assert od > 1.0

        # Should reflect alpha parameter
        fitted = model.predict(type="response")
        mean_fitted = np.mean(fitted)
        expected_od = 1 + result.alpha * mean_fitted
        assert_allclose(od, expected_od)

    def test_lr_test_poisson(self):
        """Test LR test against Poisson."""
        model_nb = NegativeBinomial(self.y, self.X)
        result_nb = model_nb.fit()

        # Perform LR test
        lr_test = model_nb.lr_test_poisson()

        assert "lr_statistic" in lr_test
        assert "p_value" in lr_test
        assert "alpha_estimate" in lr_test

        # With overdispersed data, should reject Poisson
        assert lr_test["p_value"] < 0.05
        assert lr_test["recommendation"] == "Use Negative Binomial"

    def test_cluster_robust_se(self):
        """Test cluster-robust standard errors."""
        model = NegativeBinomial(self.y, self.X, self.entity_id)
        result = model.fit()

        # Should have standard errors
        assert hasattr(result, "bse")
        assert len(result.bse) == len(result.params)

        # SE for alpha should be positive
        assert result.bse[-1] > 0

    def test_comparison_with_poisson(self):
        """Test that NB reduces to Poisson when alpha → 0."""
        # Generate true Poisson data (no overdispersion)
        np.random.seed(123)
        n = 500
        X = np.random.randn(n, 2)
        X[:, 0] = 1
        beta = np.array([0.3, -0.2])
        lambda_true = np.exp(X @ beta)
        y_poisson = np.random.poisson(lambda_true)

        # Fit both models
        model_poisson = PooledPoisson(y_poisson, X)
        result_poisson = model_poisson.fit()

        model_nb = NegativeBinomial(y_poisson, X)
        result_nb = model_nb.fit()

        # Alpha should be small
        assert result_nb.alpha < 0.1

        # Parameters should be similar
        assert_allclose(result_nb.params[:-1], result_poisson.params, rtol=0.1)

        # LR test should not reject Poisson
        lr_test = model_nb.lr_test_poisson()
        assert lr_test["p_value"] > 0.01  # Don't reject at 1% level


class TestNegativeBinomialFixedEffects:
    """Test Fixed Effects Negative Binomial model."""

    def setup_method(self):
        """Set up test data with fixed effects and overdispersion."""
        np.random.seed(456)

        # Smaller dataset for FE (computationally intensive)
        n_entities = 30
        n_periods = 8
        k = 2

        # True parameters
        self.beta_true = np.array([0.4, -0.25])
        self.alpha_true = 0.3

        # Entity fixed effects
        alpha_fe = np.random.normal(0, 0.5, n_entities)

        # Generate data
        self.y = []
        self.X = []
        self.entity_id = []
        self.time_id = []

        for i in range(n_entities):
            X_i = np.random.randn(n_periods, k)
            eta_i = X_i @ self.beta_true + alpha_fe[i]
            mu_i = np.exp(eta_i)

            # Generate NB outcomes
            r = 1 / self.alpha_true
            p = r / (r + mu_i)
            y_i = np.random.negative_binomial(r, p)

            self.y.extend(y_i)
            self.X.append(X_i)
            self.entity_id.extend([i] * n_periods)
            self.time_id.extend(range(n_periods))

        self.y = np.array(self.y)
        self.X = np.vstack(self.X)
        self.entity_id = np.array(self.entity_id)
        self.time_id = np.array(self.time_id)

    def test_initialization(self):
        """Test NB FE initialization."""
        model = NegativeBinomialFixedEffects(self.y, self.X, self.entity_id, self.time_id)

        assert model.model_type == "Negative Binomial Fixed Effects"
        assert hasattr(model, "entity_dummies")
        assert hasattr(model, "n_fe")

    def test_entity_dummies_creation(self):
        """Test creation of entity dummy variables."""
        model = NegativeBinomialFixedEffects(self.y, self.X, self.entity_id, self.time_id)

        # Should have N-1 dummies
        n_unique_entities = len(np.unique(self.entity_id))
        assert model.n_fe == n_unique_entities - 1
        assert len(model.entity_dummies) == n_unique_entities - 1

    def test_fit(self):
        """Test fitting NB FE model."""
        model = NegativeBinomialFixedEffects(self.y, self.X, self.entity_id, self.time_id)
        result = model.fit(maxiter=100)  # May need more iterations

        # Check parameters
        assert hasattr(result, "params")
        assert hasattr(result, "alpha")

        # Main parameters
        assert len(model.params_exog) == self.X.shape[1]

        # Alpha should be positive
        assert result.alpha > 0

    def test_predict_with_fe(self):
        """Test prediction with fixed effects."""
        model = NegativeBinomialFixedEffects(self.y, self.X, self.entity_id, self.time_id)
        result = model.fit(maxiter=50)

        # Predict with FE
        y_pred_fe = model.predict(
            X=self.X, entity_id=self.entity_id, include_fe=True, type="response"
        )

        # Predict without FE
        y_pred_no_fe = model.predict(X=self.X, include_fe=False, type="response")

        # Should be different
        assert not np.allclose(y_pred_fe, y_pred_no_fe)

        # Both should be positive
        assert np.all(y_pred_fe >= 0)
        assert np.all(y_pred_no_fe >= 0)

    def test_standard_errors_main_params(self):
        """Test that SEs focus on main parameters."""
        model = NegativeBinomialFixedEffects(self.y, self.X, self.entity_id, self.time_id)
        result = model.fit(maxiter=50)

        # Should have SEs for main parameters
        assert hasattr(result, "bse")

        # Main parameter SEs should be non-zero
        assert np.all(result.bse[: len(model.params_exog)] > 0)
        assert result.bse[-1] > 0  # Alpha SE

    def test_computational_warning(self):
        """Test warning for many fixed effects."""
        # Create data with many entities
        n_entities = 150
        n_periods = 3
        n_obs = n_entities * n_periods

        y = np.random.poisson(5, n_obs)
        X = np.random.randn(n_obs, 2)
        entity_id = np.repeat(np.arange(n_entities), n_periods)
        time_id = np.tile(np.arange(n_periods), n_entities)

        # Should warn about computational cost
        with pytest.warns(UserWarning, match="fixed effects parameters"):
            model = NegativeBinomialFixedEffects(y, X, entity_id, time_id)


class TestNBResults:
    """Test NegativeBinomialResults class."""

    def setup_method(self):
        """Set up a fitted model."""
        np.random.seed(789)

        n = 200
        X = np.random.randn(n, 2)
        X[:, 0] = 1

        beta = np.array([0.3, -0.2])
        alpha = 0.4

        # Generate NB data
        mu = np.exp(X @ beta)
        r = 1 / alpha
        p = r / (r + mu)
        y = np.random.negative_binomial(r, p)

        self.model = NegativeBinomial(y, X)
        self.result = self.model.fit()

    def test_result_attributes(self):
        """Test result object attributes."""
        assert hasattr(self.result, "params")
        assert hasattr(self.result, "alpha")
        assert hasattr(self.result, "llf")
        assert hasattr(self.result, "aic")
        assert hasattr(self.result, "bic")

    def test_summary(self):
        """Test summary method."""
        summary = self.result.summary()

        assert isinstance(summary, str)
        assert "Negative Binomial" in summary
        assert "Alpha" in summary
        assert "Log-Likelihood" in summary

    def test_predict_method(self):
        """Test prediction through results."""
        y_pred = self.result.predict(type="response")

        assert len(y_pred) == len(self.model.endog)
        assert np.all(y_pred >= 0)


class TestNBIntegration:
    """Integration tests for NB models."""

    def setup_method(self):
        """Set up common test data."""
        np.random.seed(999)

        # Generate overdispersed count data
        n = 300
        self.X = np.random.randn(n, 3)
        self.X[:, 0] = 1

        beta = np.array([0.5, -0.3, 0.2])
        alpha = 0.6

        # NB data
        mu = np.exp(self.X @ beta)
        r = 1 / alpha
        p = r / (r + mu)
        self.y = np.random.negative_binomial(r, p)

        self.entity_id = np.random.randint(0, 30, n)

    def test_nb_vs_poisson_with_overdispersion(self):
        """Test that NB fits better than Poisson with overdispersed data."""
        # Fit Poisson
        model_poisson = PooledPoisson(self.y, self.X)
        result_poisson = model_poisson.fit()

        # Fit NB
        model_nb = NegativeBinomial(self.y, self.X)
        result_nb = model_nb.fit()

        # NB should have higher log-likelihood
        assert result_nb.llf > result_poisson.llf

        # AIC should favor NB
        assert result_nb.aic < result_poisson.aic

    def test_model_selection(self):
        """Test model selection based on data characteristics."""
        # True Poisson data
        np.random.seed(111)
        n = 200
        X = np.random.randn(n, 2)
        X[:, 0] = 1
        lambda_true = np.exp(X @ np.array([0.3, -0.2]))
        y_poisson = np.random.poisson(lambda_true)

        # Overdispersed data
        alpha = 0.8
        mu = lambda_true
        r = 1 / alpha
        p = r / (r + mu)
        y_nb = np.random.negative_binomial(r, p)

        # For Poisson data, LR test should not reject
        model1 = NegativeBinomial(y_poisson, X)
        result1 = model1.fit()
        lr_test1 = model1.lr_test_poisson()
        assert lr_test1["p_value"] > 0.01

        # For NB data, LR test should reject
        model2 = NegativeBinomial(y_nb, X)
        result2 = model2.fit()
        lr_test2 = model2.lr_test_poisson()
        assert lr_test2["p_value"] < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
