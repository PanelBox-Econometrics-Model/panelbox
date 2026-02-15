"""
Test suite for count model marginal effects.

Tests marginal effects calculations for Poisson and Negative Binomial models.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from panelbox.marginal_effects.count_me import CountMarginalEffects, CountMarginalEffectsResults
from panelbox.models.count.negbin import NegativeBinomial
from panelbox.models.count.poisson import PooledPoisson


class TestCountMarginalEffectsPoisson:
    """Test marginal effects for Poisson models."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)

        # Generate simple data
        n = 500
        self.X = np.random.randn(n, 3)
        self.X[:, 0] = 1  # Intercept

        self.beta_true = np.array([0.5, -0.3, 0.2])
        lambda_true = np.exp(self.X @ self.beta_true)
        self.y = np.random.poisson(lambda_true)

        # Fit model
        self.model = PooledPoisson(self.y, self.X)
        self.result = self.model.fit()

    def test_initialization(self):
        """Test ME calculator initialization."""
        me_calc = CountMarginalEffects(self.result)

        assert me_calc.model == self.model
        assert hasattr(me_calc, "params")
        assert me_calc.method == "ame"  # Default

    def test_ame_count(self):
        """Test Average Marginal Effects for count."""
        me_calc = CountMarginalEffects(self.result)
        results = me_calc.compute(effect_type="count")

        # Check structure
        assert isinstance(results, CountMarginalEffectsResults)
        assert hasattr(results, "effects")
        assert hasattr(results, "standard_errors")
        assert len(results.effects) == self.X.shape[1]

        # For Poisson, AME_k = beta_k * mean(exp(X'beta))
        fitted = self.model.predict(type="response")
        expected_me = self.result.params * np.mean(fitted)

        assert_allclose(results.effects, expected_me, rtol=0.01)

    def test_ame_rate(self):
        """Test AME for rate parameter."""
        me_calc = CountMarginalEffects(self.result)
        results = me_calc.compute(effect_type="rate")

        # For rate, effects are just beta coefficients
        expected_me = self.result.params

        assert_allclose(results.effects, expected_me, rtol=0.01)

    def test_ame_elasticity(self):
        """Test AME elasticities."""
        me_calc = CountMarginalEffects(self.result)
        results = me_calc.compute(effect_type="elasticity")

        # Elasticity = beta_k * mean(X_k)
        expected_elasticity = self.result.params * np.mean(self.X, axis=0)

        assert_allclose(results.effects, expected_elasticity, rtol=0.01)

    def test_mem_count(self):
        """Test Marginal Effects at Mean."""
        me_calc = CountMarginalEffects(self.result, method="mem")
        results = me_calc.compute(effect_type="count")

        # MEM evaluated at mean X
        X_mean = np.mean(self.X, axis=0)
        lambda_at_mean = np.exp(X_mean @ self.result.params)
        expected_mem = self.result.params * lambda_at_mean

        assert_allclose(results.effects, expected_mem, rtol=0.01)

    def test_mer_count(self):
        """Test Marginal Effects at Representative values."""
        me_calc = CountMarginalEffects(self.result, method="mer")

        # Specify representative values
        X_repr = np.array([[1.0, 0.0, 0.0]])  # At intercept only
        results = me_calc.compute(effect_type="count", at_values=X_repr)

        # MER at specified point
        lambda_repr = np.exp(X_repr @ self.result.params)
        expected_mer = self.result.params * lambda_repr

        assert_allclose(results.effects, expected_mer.flatten(), rtol=0.01)

    def test_standard_errors(self):
        """Test standard error calculation via delta method."""
        me_calc = CountMarginalEffects(self.result)
        results = me_calc.compute(effect_type="count")

        # Should have standard errors
        assert hasattr(results, "standard_errors")
        assert len(results.standard_errors) == len(results.effects)
        assert np.all(results.standard_errors > 0)

        # Confidence intervals should be reasonable
        assert np.all(results.ci_lower < results.effects)
        assert np.all(results.ci_upper > results.effects)

    def test_summary(self):
        """Test summary output."""
        me_calc = CountMarginalEffects(self.result)
        results = me_calc.compute(effect_type="count")

        summary = results.summary()
        assert isinstance(summary, str)
        assert "Average Marginal Effects" in summary
        assert "Effect on Expected Count" in summary

        # With variable names
        var_names = ["Intercept", "X1", "X2"]
        summary_named = results.summary(variable_names=var_names)
        assert "Intercept" in summary_named
        assert "X1" in summary_named

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        me_calc = CountMarginalEffects(self.result)
        results = me_calc.compute(effect_type="count")

        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == self.X.shape[1]
        assert "Marginal_Effect" in df.columns
        assert "Std_Error" in df.columns
        assert "p_value" in df.columns


class TestCountMarginalEffectsNB:
    """Test marginal effects for Negative Binomial models."""

    def setup_method(self):
        """Set up test data with overdispersion."""
        np.random.seed(123)

        # Generate NB data
        n = 400
        self.X = np.random.randn(n, 2)
        self.X[:, 0] = 1

        self.beta_true = np.array([0.4, -0.25])
        self.alpha_true = 0.5

        mu = np.exp(self.X @ self.beta_true)
        r = 1 / self.alpha_true
        p = r / (r + mu)
        self.y = np.random.negative_binomial(r, p)

        # Fit model
        self.model = NegativeBinomial(self.y, self.X)
        self.result = self.model.fit()

    def test_nb_ame_count(self):
        """Test AME for NB model."""
        me_calc = CountMarginalEffects(self.result)
        results = me_calc.compute(effect_type="count")

        # For NB, ME similar to Poisson: beta_k * exp(X'beta)
        fitted = self.model.predict(type="response")
        beta_main = self.result.params[:-1]  # Exclude alpha
        expected_me = beta_main * np.mean(fitted)

        assert_allclose(results.effects, expected_me, rtol=0.05)

    def test_nb_parameter_extraction(self):
        """Test correct parameter extraction for NB."""
        me_calc = CountMarginalEffects(self.result)

        # Should extract main parameters only (not alpha)
        assert len(me_calc.params) == self.X.shape[1]
        assert_allclose(me_calc.params, self.result.params[:-1])

    def test_nb_model_identification(self):
        """Test model type identification."""
        me_calc = CountMarginalEffects(self.result)
        assert me_calc.model_type == "negbin"

    def test_nb_different_methods(self):
        """Test different ME methods for NB."""
        # AME
        me_ame = CountMarginalEffects(self.result, method="ame")
        results_ame = me_ame.compute(effect_type="count")

        # MEM
        me_mem = CountMarginalEffects(self.result, method="mem")
        results_mem = me_mem.compute(effect_type="count")

        # MER
        me_mer = CountMarginalEffects(self.result, method="mer")
        X_median = np.median(self.X, axis=0, keepdims=True)
        results_mer = me_mer.compute(effect_type="count", at_values=X_median)

        # All should give different results
        assert not np.allclose(results_ame.effects, results_mem.effects)
        assert not np.allclose(results_ame.effects, results_mer.effects)

        # But all should be reasonable
        for results in [results_ame, results_mem, results_mer]:
            assert np.all(np.isfinite(results.effects))
            assert np.all(results.standard_errors > 0)


class TestJointSignificance:
    """Test joint significance tests for marginal effects."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(456)

        n = 300
        X = np.random.randn(n, 4)
        X[:, 0] = 1

        # Make some coefficients zero
        beta = np.array([0.3, 0.0, 0.0, 0.5])  # X2 and X3 have no effect
        lambda_true = np.exp(X @ beta)
        y = np.random.poisson(lambda_true)

        model = PooledPoisson(y, X)
        self.result = model.fit()
        self.me_calc = CountMarginalEffects(self.result)

    def test_joint_test_implementation(self):
        """Test joint significance test."""
        # Compute marginal effects first
        me_results = self.me_calc.compute(effect_type="count")

        # Store for testing
        self.me_calc.marginal_effects = me_results.effects
        self.me_calc.cov_marginal_effects = np.diag(me_results.standard_errors**2)

        # Test joint significance
        test_result = self.me_calc.test_joint_significance()

        assert "statistic" in test_result
        assert "p_value" in test_result
        assert "df" in test_result
        assert "conclusion" in test_result

    def test_joint_test_subset(self):
        """Test joint test for subset of variables."""
        me_results = self.me_calc.compute(effect_type="count")

        self.me_calc.marginal_effects = me_results.effects
        self.me_calc.cov_marginal_effects = np.diag(me_results.standard_errors**2)

        # Test only variables 2 and 3 (should be jointly insignificant)
        test_result = self.me_calc.test_joint_significance(variables=[1, 2])

        # These should be insignificant
        assert test_result["p_value"] > 0.05


class TestMarginalEffectsIntegration:
    """Integration tests across different models and methods."""

    def setup_method(self):
        """Set up common data."""
        np.random.seed(789)

        n = 200
        self.X = np.random.randn(n, 2)
        self.X[:, 0] = 1

        beta = np.array([0.3, -0.2])
        lambda_true = np.exp(self.X @ beta)
        self.y = np.random.poisson(lambda_true)

    def test_consistency_across_models(self):
        """Test that ME are consistent across similar models."""
        # Poisson
        model_pois = PooledPoisson(self.y, self.X)
        result_pois = model_pois.fit()
        me_pois = CountMarginalEffects(result_pois)
        me_results_pois = me_pois.compute(effect_type="count")

        # NB (should be similar for low alpha)
        model_nb = NegativeBinomial(self.y, self.X)
        result_nb = model_nb.fit()
        me_nb = CountMarginalEffects(result_nb)
        me_results_nb = me_nb.compute(effect_type="count")

        # Should be reasonably close
        assert_allclose(me_results_pois.effects, me_results_nb.effects, rtol=0.2)

    def test_confidence_interval_coverage(self):
        """Test that confidence intervals have reasonable coverage."""
        # Simulate multiple datasets and check CI coverage
        n_sims = 100
        coverage = []

        for _ in range(n_sims):
            # Generate data
            X = np.random.randn(100, 2)
            X[:, 0] = 1
            beta_true = np.array([0.3, -0.2])
            lambda_true = np.exp(X @ beta_true)
            y = np.random.poisson(lambda_true)

            # Fit and get ME
            model = PooledPoisson(y, X)
            result = model.fit()
            me_calc = CountMarginalEffects(result)
            me_results = me_calc.compute(effect_type="rate", confidence_level=0.95)

            # Check if true values in CI
            # For rate, ME = beta
            in_ci = (beta_true >= me_results.ci_lower) & (beta_true <= me_results.ci_upper)
            coverage.append(in_ci)

        coverage = np.mean(coverage, axis=0)

        # Should be close to 95%
        assert np.all(coverage > 0.85)  # Allow some tolerance
        assert np.all(coverage < 0.99)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
