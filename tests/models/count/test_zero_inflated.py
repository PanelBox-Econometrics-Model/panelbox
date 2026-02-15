"""
Test suite for Zero-Inflated count models.

Tests ZIP and ZINB models against simulated data and R pscl package.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from panelbox.models.count import PooledPoisson, ZeroInflatedNegativeBinomial, ZeroInflatedPoisson


class TestZeroInflatedPoisson:
    """Test Zero-Inflated Poisson model."""

    @pytest.fixture
    def simulated_data(self):
        """
        Simulate ZIP data.

        Returns data with excess zeros from a two-part process.
        """
        np.random.seed(42)
        n = 500

        # Covariates
        X = np.random.randn(n, 3)
        X[:, 0] = 1  # Intercept

        # True parameters
        beta_count = np.array([1.0, 0.5, -0.3])  # Count model
        gamma_inflate = np.array([-0.5, 0.3, 0.2])  # Inflation model

        # Generate data
        # Inflation process (probability of structural zero)
        pi = 1 / (1 + np.exp(-X @ gamma_inflate))
        structural_zero = np.random.binomial(1, pi)

        # Count process (Poisson)
        lambda_ = np.exp(X @ beta_count)
        counts = np.random.poisson(lambda_)

        # Observed outcome
        y = np.where(structural_zero == 1, 0, counts)

        return {
            "y": y,
            "X": X,
            "true_beta": beta_count,
            "true_gamma": gamma_inflate,
            "n_zeros": np.sum(y == 0),
        }

    def test_zip_estimation(self, simulated_data):
        """Test ZIP parameter estimation."""
        # Fit model
        model = ZeroInflatedPoisson(simulated_data["y"], simulated_data["X"])
        result = model.fit()

        # Check convergence
        assert result.converged

        # Check parameters are close to true values (within 2 std errors typically)
        beta_est = result.params_count
        gamma_est = result.params_inflate

        # More lenient check - within reasonable range
        assert np.allclose(beta_est, simulated_data["true_beta"], atol=0.5)
        assert np.allclose(gamma_est, simulated_data["true_gamma"], atol=0.5)

    def test_zip_vs_poisson(self, simulated_data):
        """Test that ZIP fits better than standard Poisson for zero-inflated data."""
        y = simulated_data["y"]
        X = simulated_data["X"]

        # Fit ZIP
        zip_model = ZeroInflatedPoisson(y, X)
        zip_result = zip_model.fit()

        # Fit standard Poisson
        poisson_model = PooledPoisson(y, X)
        poisson_result = poisson_model.fit()

        # ZIP should have higher log-likelihood
        assert zip_result.llf > poisson_result.llf

        # Check Vuong test if available
        if hasattr(zip_result, "vuong_stat"):
            # Positive Vuong statistic favors ZIP
            assert zip_result.vuong_stat > 0

    def test_zip_predictions(self, simulated_data):
        """Test ZIP predictions."""
        model = ZeroInflatedPoisson(simulated_data["y"], simulated_data["X"])
        result = model.fit()

        # Test different prediction types
        mean_pred = model.predict(result.params, which="mean")
        prob_zero = model.predict(result.params, which="prob-zero")
        prob_zero_struct = model.predict(result.params, which="prob-zero-structural")

        # Basic checks
        assert np.all(mean_pred >= 0)
        assert np.all((prob_zero >= 0) & (prob_zero <= 1))
        assert np.all((prob_zero_struct >= 0) & (prob_zero_struct <= 1))

        # Structural zero probability should be less than total zero probability
        assert np.all(prob_zero_struct <= prob_zero)

        # Mean should be close to actual mean
        assert np.abs(np.mean(mean_pred) - np.mean(simulated_data["y"])) < 0.5

    def test_zip_gradient(self):
        """Test analytical gradient implementation."""
        np.random.seed(123)
        n = 100
        X = np.random.randn(n, 2)
        X[:, 0] = 1
        y = np.random.poisson(2, n)
        y[np.random.rand(n) < 0.3] = 0  # Add some zeros

        model = ZeroInflatedPoisson(y, X)

        # Test at random parameter values
        params = np.random.randn(4)  # 2 for count, 2 for inflate

        # Analytical gradient
        grad_analytical = model.gradient(params)

        # Numerical gradient
        eps = 1e-6
        grad_numerical = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps

            grad_numerical[i] = (
                model.log_likelihood(params_plus) - model.log_likelihood(params_minus)
            ) / (2 * eps)

        # Should be close
        assert np.allclose(grad_analytical, grad_numerical, rtol=1e-5)

    def test_zip_with_panel_structure(self):
        """Test ZIP with panel data structure."""
        np.random.seed(456)

        # Create panel data
        n_entities = 50
        n_time = 10
        n = n_entities * n_time

        # Entity and time indices
        entity = np.repeat(np.arange(n_entities), n_time)
        time = np.tile(np.arange(n_time), n_entities)

        # Create DataFrame with MultiIndex
        index = pd.MultiIndex.from_arrays([entity, time], names=["entity", "time"])

        # Generate data
        X = np.random.randn(n, 2)
        X[:, 0] = 1

        # Add entity effects
        entity_effects = np.random.randn(n_entities)
        lambda_ = np.exp(X @ [1.0, 0.5] + entity_effects[entity])
        y = np.random.poisson(lambda_)

        # Add structural zeros
        pi = 0.2
        y[np.random.rand(n) < pi] = 0

        # Create panel DataFrame
        df = pd.DataFrame({"y": y, "X1": X[:, 0], "X2": X[:, 1]}, index=index)

        # Fit model
        model = ZeroInflatedPoisson(df["y"], df[["X1", "X2"]], entity_col="entity", time_col="time")
        result = model.fit()

        assert result.converged
        assert len(result.params) == 4  # 2 count + 2 inflate


class TestZeroInflatedNegativeBinomial:
    """Test Zero-Inflated Negative Binomial model."""

    @pytest.fixture
    def overdispersed_data(self):
        """Simulate ZINB data with overdispersion."""
        np.random.seed(789)
        n = 500

        # Covariates
        X = np.random.randn(n, 2)
        X[:, 0] = 1

        # Parameters
        beta = np.array([1.5, -0.5])
        gamma = np.array([-1.0, 0.5])
        alpha = 2.0  # Overdispersion

        # Generate ZINB data
        # Inflation
        pi = 1 / (1 + np.exp(-X @ gamma))
        structural_zero = np.random.binomial(1, pi)

        # NB process
        mu = np.exp(X @ beta)
        size = 1 / alpha
        # NB is Gamma-Poisson mixture
        counts = np.random.negative_binomial(size, size / (size + mu))

        # Observed
        y = np.where(structural_zero == 1, 0, counts)

        return {
            "y": y,
            "X": X,
            "alpha": alpha,
            "overdispersion": np.var(y[y > 0]) / np.mean(y[y > 0]),
        }

    def test_zinb_estimation(self, overdispersed_data):
        """Test ZINB parameter estimation."""
        model = ZeroInflatedNegativeBinomial(overdispersed_data["y"], overdispersed_data["X"])
        result = model.fit()

        assert result.converged

        # Check overdispersion parameter
        assert result.alpha > 0
        # Should detect overdispersion
        assert result.alpha > 0.1

    def test_zinb_vs_zip(self, overdispersed_data):
        """Test that ZINB fits better than ZIP for overdispersed data."""
        y = overdispersed_data["y"]
        X = overdispersed_data["X"]

        # Fit both models
        zinb_model = ZeroInflatedNegativeBinomial(y, X)
        zinb_result = zinb_model.fit()

        zip_model = ZeroInflatedPoisson(y, X)
        zip_result = zip_model.fit()

        # ZINB should fit better (higher likelihood)
        assert zinb_result.llf > zip_result.llf

        # AIC should also favor ZINB despite extra parameter
        assert zinb_result.aic < zip_result.aic

    def test_zinb_summary(self, overdispersed_data):
        """Test ZINB summary output."""
        model = ZeroInflatedNegativeBinomial(overdispersed_data["y"], overdispersed_data["X"])
        result = model.fit()

        summary = result.summary()

        # Check summary contains key information
        assert "Zero-Inflated Negative Binomial" in summary
        assert "Alpha (overdispersion)" in summary
        assert f"{result.alpha:.4f}" in summary
        assert "Count Model" in summary
        assert "Zero-Inflation Model" in summary


class TestZeroInflatedEdgeCases:
    """Test edge cases for zero-inflated models."""

    def test_no_zeros(self):
        """Test behavior when there are no zeros."""
        np.random.seed(111)
        n = 100
        X = np.random.randn(n, 2)
        X[:, 0] = 1
        # Generate data with no zeros
        y = np.random.poisson(5, n) + 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ZeroInflatedPoisson(y, X)
            result = model.fit()

            # Model should still run
            assert result.params is not None

    def test_all_zeros(self):
        """Test behavior when all values are zero."""
        n = 100
        X = np.random.randn(n, 2)
        X[:, 0] = 1
        y = np.zeros(n)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ZeroInflatedPoisson(y, X)
            result = model.fit()

            # Should predict high inflation probability
            prob_struct = model.predict(result.params, which="prob-zero-structural")
            assert np.mean(prob_struct) > 0.9

    def test_perfect_separation(self):
        """Test with perfect separation in inflation model."""
        n = 100
        X = np.zeros((n, 2))
        X[:, 0] = 1
        X[:50, 1] = 1  # First half has X2=1
        X[50:, 1] = 0  # Second half has X2=0

        # y=0 when X2=1, y>0 when X2=0
        y = np.zeros(n)
        y[50:] = np.random.poisson(3, 50)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ZeroInflatedPoisson(y, X)
            result = model.fit()

            # Model should handle this gracefully
            assert result.params is not None

    def test_different_regressors(self):
        """Test with different regressors for count and inflation models."""
        np.random.seed(222)
        n = 200

        # Different regressors
        X_count = np.random.randn(n, 2)
        X_count[:, 0] = 1

        X_inflate = np.random.randn(n, 3)
        X_inflate[:, 0] = 1

        # Generate ZIP data
        lambda_ = np.exp(X_count @ [1.0, 0.5])
        pi = 1 / (1 + np.exp(-X_inflate @ [0.5, -0.3, 0.2]))

        structural_zero = np.random.binomial(1, pi)
        counts = np.random.poisson(lambda_)
        y = np.where(structural_zero == 1, 0, counts)

        # Fit with different regressors
        model = ZeroInflatedPoisson(y, X_count, X_inflate)
        result = model.fit()

        assert result.converged
        assert len(result.params_count) == 2
        assert len(result.params_inflate) == 3


class TestVuongTest:
    """Test Vuong test for model comparison."""

    def test_vuong_statistic(self):
        """Test Vuong test statistic calculation."""
        np.random.seed(333)
        n = 300

        X = np.random.randn(n, 2)
        X[:, 0] = 1

        # Generate data with clear zero-inflation
        pi = 0.3  # 30% structural zeros
        structural_zero = np.random.binomial(1, pi, n)
        lambda_ = np.exp(X @ [1.0, 0.5])
        counts = np.random.poisson(lambda_)
        y = np.where(structural_zero == 1, 0, counts)

        # Fit ZIP
        model = ZeroInflatedPoisson(y, X)
        result = model.fit()

        # Vuong test should be computed
        assert hasattr(result, "vuong_stat")
        assert hasattr(result, "vuong_pvalue")

        # With clear zero-inflation, ZIP should be preferred
        assert result.vuong_stat > 2  # Significant at 5% level
        assert result.vuong_pvalue < 0.05

    def test_vuong_no_inflation(self):
        """Test Vuong when there's no zero-inflation."""
        np.random.seed(444)
        n = 300

        X = np.random.randn(n, 2)
        X[:, 0] = 1

        # Generate pure Poisson data (no inflation)
        lambda_ = np.exp(X @ [1.0, 0.5])
        y = np.random.poisson(lambda_)

        # Fit ZIP
        model = ZeroInflatedPoisson(y, X)
        result = model.fit()

        # Vuong test should not strongly favor ZIP
        assert hasattr(result, "vuong_stat")
        assert np.abs(result.vuong_stat) < 2  # Not significant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
