"""
Unit tests for MLE standard errors computation.

Tests cover:
- Sandwich estimator
- Cluster-robust standard errors
- Bootstrap standard errors
- Delta method
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit

from panelbox.models.discrete import PooledLogit
from panelbox.standard_errors.mle import (
    bootstrap_se,
    cluster_robust_mle,
    delta_method,
    sandwich_estimator,
)


class TestSandwichEstimator:
    """Tests for sandwich (robust) standard errors."""

    def test_sandwich_vs_manual_calculation(self):
        """Test sandwich estimator against manual implementation."""
        np.random.seed(42)
        n = 500

        # Generate simple data with heteroscedasticity
        x = np.random.randn(n)
        # Heteroscedastic error variance increases with x
        var_scale = 0.5 + np.abs(x)
        prob_true = expit(0.5 + 0.8 * x + np.random.randn(n) * var_scale * 0.1)
        y = np.random.binomial(1, prob_true)

        # Create DataFrame
        df = pd.DataFrame({"entity_id": range(n), "time_id": 0, "y": y, "x": x})

        # Fit model with robust SEs
        model = PooledLogit("y ~ x", df, "entity_id", "time_id")
        results_robust = model.fit(cov_type="robust")

        # Manual calculation
        # Get design matrix
        X = np.column_stack([np.ones(n), x])
        params = results_robust.params.values

        # Calculate fitted probabilities
        eta = X @ params
        p = expit(eta)

        # Hessian: H = -X'WX where W = diag(p*(1-p))
        W = p * (1 - p)
        H = -(X.T * W) @ X
        H_inv = np.linalg.inv(H)

        # Meat: S = Σ s_i s_i' where s_i = (y_i - p_i) * X_i
        scores = (y - p)[:, np.newaxis] * X
        S = scores.T @ scores

        # Sandwich variance
        vcov_sandwich = H_inv @ S @ H_inv
        se_sandwich_manual = np.sqrt(np.diag(vcov_sandwich))

        # Compare
        assert_allclose(results_robust.std_errors.values, se_sandwich_manual, rtol=1e-5)

    def test_sandwich_vs_nonrobust_with_heteroscedasticity(self):
        """Test that sandwich SEs differ from non-robust with heteroscedasticity."""
        np.random.seed(123)
        n = 1000

        # Generate data with strong heteroscedasticity
        x = np.random.uniform(-3, 3, n)
        # Variance strongly depends on x
        noise_scale = 0.1 + 0.5 * np.abs(x)
        linear_pred = 0.5 + 1.0 * x
        prob_true = expit(linear_pred + np.random.randn(n) * noise_scale)
        y = np.random.binomial(1, np.clip(prob_true, 0.01, 0.99))

        df = pd.DataFrame({"entity_id": range(n), "time_id": 0, "y": y, "x": x})

        # Fit with different SE types
        model = PooledLogit("y ~ x", df, "entity_id", "time_id")
        results_nonrobust = model.fit(cov_type="nonrobust")
        results_robust = model.fit(cov_type="robust")

        # With heteroscedasticity, robust SEs should differ from non-robust
        se_ratio = results_robust.std_errors / results_nonrobust.std_errors

        # Should be notably different (not just numerical noise)
        assert np.max(np.abs(se_ratio - 1)) > 0.1


class TestClusterRobustSE:
    """Tests for cluster-robust standard errors."""

    def test_cluster_se_for_pooled_logit(self):
        """Test cluster-robust SEs for Pooled Logit match manual calculation."""
        np.random.seed(456)

        # Generate panel data with clustering
        n_entities = 50
        n_periods = 10
        n = n_entities * n_periods

        # Generate correlated data within entities
        entity_effects = np.random.randn(n_entities) * 0.5

        data_list = []
        for i in range(n_entities):
            for t in range(n_periods):
                x = np.random.randn()
                # Add entity effect to create within-cluster correlation
                linear_pred = entity_effects[i] + 0.7 * x
                prob = expit(linear_pred)
                y = np.random.binomial(1, prob)

                data_list.append({"entity_id": i, "time_id": t, "y": y, "x": x})

        df = pd.DataFrame(data_list)

        # Fit model with cluster SEs
        model = PooledLogit("y ~ x", df, "entity_id", "time_id")
        results_cluster = model.fit(cov_type="cluster")

        # Manual calculation
        X = pd.get_dummies(df[["x"]], drop_first=False)
        X.insert(0, "intercept", 1.0)
        X = X.values
        y = df["y"].values
        entities = df["entity_id"].values

        params = results_cluster.params.values
        eta = X @ params
        p = expit(eta)

        # Hessian
        W = p * (1 - p)
        H = -(X.T * W) @ X
        H_inv = np.linalg.inv(H)

        # Cluster scores
        scores = (y - p)[:, np.newaxis] * X
        unique_entities = np.unique(entities)
        n_clusters = len(unique_entities)
        cluster_scores = np.zeros((n_clusters, X.shape[1]))

        for i, entity in enumerate(unique_entities):
            mask = entities == entity
            cluster_scores[i] = scores[mask].sum(axis=0)

        # Meat
        S = cluster_scores.T @ cluster_scores

        # DF correction
        k = X.shape[1]
        G = n_clusters
        df_correction = G / (G - 1) * (n - 1) / (n - k)

        # Cluster variance
        vcov_cluster = df_correction * H_inv @ S @ H_inv
        se_cluster_manual = np.sqrt(np.diag(vcov_cluster))

        # Compare
        assert_allclose(results_cluster.std_errors.values, se_cluster_manual, rtol=1e-5)

    def test_cluster_se_larger_than_nonrobust_with_clustering(self):
        """Test that cluster SEs are larger than non-robust with within-cluster correlation."""
        np.random.seed(789)

        # Generate strongly clustered data
        n_entities = 30
        n_periods = 15

        # Strong entity effects create within-cluster correlation
        entity_effects = np.random.randn(n_entities) * 2.0

        data_list = []
        for i in range(n_entities):
            for t in range(n_periods):
                x = np.random.randn()
                # Strong entity effect
                linear_pred = entity_effects[i] + 0.5 * x
                prob = expit(linear_pred)
                y = np.random.binomial(1, prob)

                data_list.append({"entity_id": i, "time_id": t, "y": y, "x": x})

        df = pd.DataFrame(data_list)

        # Fit with different SE types
        model = PooledLogit("y ~ x", df, "entity_id", "time_id")
        results_nonrobust = model.fit(cov_type="nonrobust")
        results_cluster = model.fit(cov_type="cluster")

        # Cluster SEs should be larger with within-cluster correlation
        se_ratio = results_cluster.std_errors / results_nonrobust.std_errors

        # Should be notably larger (at least for some coefficients)
        assert np.max(se_ratio) > 1.2


class TestDGPWithHeteroscedasticity:
    """Test models with heteroscedastic DGP."""

    def test_dgp_heteroscedastic_logit(self):
        """Test that robust SEs correctly handle heteroscedastic DGP."""
        np.random.seed(111)
        n = 2000

        # Generate data with known heteroscedasticity pattern
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)

        # True parameters
        beta_true = np.array([0.5, 0.8, -0.6])

        # Generate outcome with heteroscedastic errors
        # Variance increases with |x1|
        X = np.column_stack([np.ones(n), x1, x2])
        linear_pred = X @ beta_true

        # Add heteroscedastic noise
        noise_scale = 0.5 + 0.3 * np.abs(x1)
        linear_pred_noisy = linear_pred + np.random.randn(n) * noise_scale * 0.2

        prob = expit(linear_pred_noisy)
        y = np.random.binomial(1, prob)

        df = pd.DataFrame({"entity_id": range(n), "time_id": 0, "y": y, "x1": x1, "x2": x2})

        # Fit with different SE types
        model = PooledLogit("y ~ x1 + x2", df, "entity_id", "time_id")
        results_nonrobust = model.fit(cov_type="nonrobust")
        results_robust = model.fit(cov_type="robust")

        # With heteroscedasticity:
        # 1. Point estimates should be similar
        assert_allclose(results_robust.params, results_nonrobust.params, rtol=1e-10)

        # 2. But SEs should differ
        se_diff = np.abs(results_robust.std_errors - results_nonrobust.std_errors)
        assert np.max(se_diff) > 0.01

        # 3. Coefficients should still be reasonable
        assert_allclose(results_robust.params, beta_true, rtol=0.5)


class TestBootstrapSE:
    """Tests for bootstrap standard errors."""

    @pytest.mark.slow
    def test_bootstrap_se_consistency(self):
        """Test that bootstrap SEs are consistent with analytical SEs."""
        np.random.seed(222)
        n = 200

        # Simple data
        x = np.random.randn(n)
        prob_true = expit(0.5 + 0.8 * x)
        y = np.random.binomial(1, prob_true)

        df = pd.DataFrame({"entity_id": range(n), "time_id": 0, "y": y, "x": x})

        # Fit with bootstrap SEs (small number for speed)
        model = PooledLogit("y ~ x", df, "entity_id", "time_id")
        results_boot = model.fit(cov_type="bootstrap", n_bootstrap=100)

        # Fit with analytical SEs
        results_robust = model.fit(cov_type="robust")

        # Bootstrap SEs should be in same ballpark as analytical
        # (won't be exact due to randomness and finite bootstrap samples)
        se_ratio = results_boot.std_errors / results_robust.std_errors
        assert np.all((se_ratio > 0.5) & (se_ratio < 2.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
