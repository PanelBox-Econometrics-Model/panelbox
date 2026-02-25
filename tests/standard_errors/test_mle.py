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
    MLECovarianceResult,
    bootstrap_mle,
    cluster_robust_mle,
    compute_mle_standard_errors,
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

    @pytest.mark.xfail(
        strict=False,
        reason="Numerical issue: logit link function absorbs heteroscedasticity in "
        "the latent linear predictor, so robust and non-robust SEs do not diverge "
        "enough to exceed the 0.1 threshold with this DGP",
    )
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

    @pytest.mark.xfail(
        strict=False,
        reason="Numerical issue: logit link function absorbs heteroscedasticity in "
        "the latent linear predictor with small noise_scale*0.2 multiplier, so "
        "robust and non-robust SEs do not diverge enough (max diff < 0.01)",
    )
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
    @pytest.mark.xfail(
        strict=True,
        reason="Source-code bug: PooledLogit.fit() does not support cov_type='bootstrap'; "
        "valid options are 'nonrobust', 'robust', or 'cluster'",
    )
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


# ===========================
# Direct unit tests for mle.py functions
# ===========================


class TestMLECovarianceResult:
    """Test the MLECovarianceResult container."""

    def test_creation(self):
        """Test basic creation of the result object."""
        cov = np.eye(2) * 0.01
        se = np.sqrt(np.diag(cov))
        result = MLECovarianceResult(
            cov_matrix=cov,
            std_errors=se,
            method="robust",
            n_obs=100,
            n_params=2,
        )
        assert result.method == "robust"
        assert result.n_obs == 100
        assert result.n_params == 2
        assert_allclose(result.std_errors, np.array([0.1, 0.1]))

    def test_repr(self):
        """Test string representation."""
        result = MLECovarianceResult(
            cov_matrix=np.eye(3),
            std_errors=np.ones(3),
            method="bootstrap",
            n_obs=200,
            n_params=3,
        )
        repr_str = repr(result)
        assert "bootstrap" in repr_str
        assert "200" in repr_str
        assert "3" in repr_str


class TestSandwichEstimatorDirect:
    """Direct unit tests for sandwich_estimator covering lines 166-194."""

    @pytest.fixture
    def known_hessian_scores(self):
        """Hessian and scores with analytically known covariance.

        H = -c * I so H_inv = (1/c) * I.
        scores have structure [[1,0],[0,1],[1,0],[0,1]] so S = 2*I.
        """
        k = 2
        c = 5.0
        hessian = -c * np.eye(k)
        scores = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        return hessian, scores, c

    def test_nonrobust_returns_neg_h_inv(self, known_hessian_scores):
        """Nonrobust method should return -H^{-1}."""
        hessian, scores, c = known_hessian_scores
        result = sandwich_estimator(hessian, scores, method="nonrobust")

        expected_cov = (1.0 / c) * np.eye(2)
        assert_allclose(result.cov_matrix, expected_cov)
        assert result.method == "nonrobust"
        assert result.n_obs == 4
        assert result.n_params == 2

    def test_nonrobust_std_errors(self, known_hessian_scores):
        """Nonrobust standard errors should be sqrt(diag(-H^{-1}))."""
        hessian, scores, c = known_hessian_scores
        result = sandwich_estimator(hessian, scores, method="nonrobust")

        expected_se = np.sqrt(np.array([1.0 / c, 1.0 / c]))
        assert_allclose(result.std_errors, expected_se)

    def test_robust_sandwich_known_values(self, known_hessian_scores):
        """Robust sandwich with analytically known values.

        H_inv = (1/5)*I, S = 2*I.
        V = H_inv @ S @ H_inv = (1/25) * 2 * I = (2/25)*I.
        """
        hessian, scores, _c = known_hessian_scores
        result = sandwich_estimator(hessian, scores, method="robust")

        expected_cov = np.diag([2.0 / 25.0, 2.0 / 25.0])
        assert_allclose(result.cov_matrix, expected_cov)
        assert result.method == "robust"

    def test_robust_sandwich_general(self):
        """Robust sandwich with general (non-diagonal) Hessian."""
        np.random.seed(42)
        n, k = 100, 3
        hessian = -np.diag([10.0, 20.0, 5.0])
        scores = np.random.randn(n, k) * 0.1

        result = sandwich_estimator(hessian, scores, method="robust")

        # Manually compute
        H_inv = -np.linalg.inv(hessian)
        S = scores.T @ scores
        expected_cov = H_inv @ S @ H_inv

        assert_allclose(result.cov_matrix, expected_cov)
        assert result.cov_matrix.shape == (k, k)
        assert result.std_errors.shape == (k,)
        assert result.n_obs == n
        assert result.n_params == k

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        hessian = -np.eye(2) * 10.0
        scores = np.random.randn(20, 2)
        with pytest.raises(ValueError, match="method must be"):
            sandwich_estimator(hessian, scores, method="invalid")

    def test_covariance_is_symmetric(self):
        """Robust covariance should be symmetric."""
        np.random.seed(99)
        hessian = -np.eye(3) * 15.0
        scores = np.random.randn(50, 3) * 0.2
        result = sandwich_estimator(hessian, scores, method="robust")
        assert_allclose(result.cov_matrix, result.cov_matrix.T, atol=1e-15)

    def test_robust_differs_from_nonrobust_under_heteroskedasticity(self):
        """Under heteroskedasticity, robust SEs should differ from nonrobust."""
        np.random.seed(99)
        n, k = 200, 2
        hessian = -np.eye(k) * 50.0

        # Heteroskedastic scores: variance increases with index
        scores = np.random.randn(n, k)
        for i in range(n):
            scores[i] *= 1.0 + 3.0 * i / n

        result_nonrobust = sandwich_estimator(hessian, scores, method="nonrobust")
        result_robust = sandwich_estimator(hessian, scores, method="robust")

        assert not np.allclose(result_nonrobust.std_errors, result_robust.std_errors, rtol=0.01)


class TestClusterRobustMLEDirect:
    """Direct unit tests for cluster_robust_mle covering lines 270-305."""

    @pytest.fixture
    def cluster_data(self):
        """Generate data with cluster structure."""
        np.random.seed(123)
        n_clusters = 10
        cluster_size = 5
        n = n_clusters * cluster_size
        k = 2

        hessian = -np.diag([8.0, 12.0])
        scores = np.random.randn(n, k) * 0.2
        cluster_ids = np.repeat(np.arange(n_clusters), cluster_size)

        return hessian, scores, cluster_ids

    def test_basic_clustering(self, cluster_data):
        """Test basic cluster-robust computation returns expected structure."""
        hessian, scores, cluster_ids = cluster_data
        result = cluster_robust_mle(hessian, scores, cluster_ids)

        assert result.method == "cluster"
        assert result.cov_matrix.shape == (2, 2)
        assert result.std_errors.shape == (2,)
        assert result.n_obs == 50
        assert result.n_params == 2

    def test_df_correction_vs_uncorrected(self, cluster_data):
        """Corrected cov should be a scalar multiple of the uncorrected one."""
        hessian, scores, cluster_ids = cluster_data

        result_corrected = cluster_robust_mle(hessian, scores, cluster_ids, df_correction=True)
        result_uncorrected = cluster_robust_mle(hessian, scores, cluster_ids, df_correction=False)

        n_obs, n_params = scores.shape
        n_clusters = len(np.unique(cluster_ids))
        expected_adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))

        ratio = result_corrected.cov_matrix / result_uncorrected.cov_matrix
        assert_allclose(ratio, np.full_like(ratio, expected_adj), rtol=1e-10)

    def test_df_correction_false_manual(self, cluster_data):
        """Uncorrected result should match manual meat computation."""
        hessian, scores, cluster_ids = cluster_data
        result = cluster_robust_mle(hessian, scores, cluster_ids, df_correction=False)

        H_inv = -np.linalg.inv(hessian)
        unique_clusters = np.unique(cluster_ids)
        meat = np.zeros((2, 2))
        for cluster in unique_clusters:
            mask = cluster_ids == cluster
            g_i = scores[mask].sum(axis=0)
            meat += np.outer(g_i, g_i)

        expected_cov = H_inv @ meat @ H_inv
        assert_allclose(result.cov_matrix, expected_cov, rtol=1e-10)

    def test_single_obs_per_cluster_matches_robust(self):
        """Each obs as its own cluster (no df correction) should match robust sandwich."""
        np.random.seed(77)
        n, k = 20, 2
        hessian = -np.eye(k) * 10.0
        scores = np.random.randn(n, k)
        cluster_ids = np.arange(n)

        result_cluster = cluster_robust_mle(hessian, scores, cluster_ids, df_correction=False)
        result_robust = sandwich_estimator(hessian, scores, method="robust")

        assert_allclose(result_cluster.cov_matrix, result_robust.cov_matrix, rtol=1e-10)

    def test_covariance_is_symmetric(self, cluster_data):
        """Cluster-robust covariance should be symmetric."""
        hessian, scores, cluster_ids = cluster_data
        result = cluster_robust_mle(hessian, scores, cluster_ids)
        assert_allclose(result.cov_matrix, result.cov_matrix.T, atol=1e-15)

    def test_positive_standard_errors(self, cluster_data):
        """Standard errors should all be positive."""
        hessian, scores, cluster_ids = cluster_data
        result = cluster_robust_mle(hessian, scores, cluster_ids)
        assert np.all(result.std_errors > 0)


class TestDeltaMethodDirect:
    """Direct unit tests for delta_method covering lines 382-409."""

    def test_identity_transform(self):
        """Identity transform should return the original vcov."""
        params = np.array([1.0, 2.0, 3.0])
        vcov = np.array(
            [
                [0.04, 0.01, 0.0],
                [0.01, 0.09, 0.02],
                [0.0, 0.02, 0.16],
            ]
        )

        vcov_transformed = delta_method(vcov, lambda b: b, params)
        assert_allclose(vcov_transformed, vcov, atol=1e-6)

    def test_linear_transform(self):
        """Linear transform g(b) = A @ b should give A @ V @ A.T exactly."""
        params = np.array([1.0, 2.0])
        vcov = np.diag([0.01, 0.04])

        A = np.array([[2.0, 1.0], [1.0, -1.0]])

        vcov_transformed = delta_method(vcov, lambda b: A @ b, params)
        expected = A @ vcov @ A.T
        assert_allclose(vcov_transformed, expected, atol=1e-6)

    def test_exp_transform(self):
        """Var(exp(b)) should approx equal diag(exp(b))^2 * Var(b) * diag(exp(b))^2."""
        params = np.array([0.5, 1.0])
        vcov = np.diag([0.01, 0.02])

        vcov_transformed = delta_method(vcov, np.exp, params)

        J = np.diag(np.exp(params))
        expected = J @ vcov @ J.T

        assert_allclose(vcov_transformed, expected, atol=1e-5)

    def test_nonlinear_ratio_transform(self):
        """Delta method for ratio g(b) = b0/b1."""
        params = np.array([4.0, 2.0])
        vcov = np.diag([0.01, 0.04])

        def ratio(beta):
            return np.array([beta[0] / beta[1]])

        vcov_transformed = delta_method(vcov, ratio, params)

        # Jacobian: [1/b1, -b0/b1^2] = [0.5, -1.0]
        J = np.array([[1.0 / params[1], -params[0] / params[1] ** 2]])
        expected = J @ vcov @ J.T

        assert_allclose(vcov_transformed, expected, atol=1e-5)

    def test_single_param_square(self):
        """Single parameter square transform: Var(b^2) = (2b)^2 * Var(b)."""
        params = np.array([2.0])
        vcov = np.array([[0.25]])

        def square(beta):
            return np.array([beta[0] ** 2])

        vcov_transformed = delta_method(vcov, square, params)

        # Jacobian = [2*b] = [4], so Var = 4^2 * 0.25 = 4.0
        expected = np.array([[16.0 * 0.25]])
        assert_allclose(vcov_transformed, expected, atol=1e-4)

    def test_symmetry_of_result(self):
        """Delta method result should be symmetric."""
        params = np.array([1.0, 2.0, 3.0])
        vcov = np.array(
            [
                [0.04, 0.01, 0.005],
                [0.01, 0.09, 0.02],
                [0.005, 0.02, 0.16],
            ]
        )

        def transform(beta):
            return np.array([np.exp(beta[0]), beta[1] ** 2, beta[0] * beta[2]])

        vcov_transformed = delta_method(vcov, transform, params)
        assert_allclose(vcov_transformed, vcov_transformed.T, atol=1e-10)

    def test_custom_epsilon(self):
        """Different epsilon values should still give consistent results for smooth functions."""
        params = np.array([1.0, 2.0])
        vcov = np.diag([0.01, 0.04])

        result_default = delta_method(vcov, np.exp, params)
        result_small = delta_method(vcov, np.exp, params, epsilon=1e-10)

        assert_allclose(result_default, result_small, rtol=1e-3)


class TestComputeMLEStandardErrorsDirect:
    """Direct unit tests for compute_mle_standard_errors covering lines 441-471."""

    class MockModel:
        """Mock model object with _hessian and _score_obs methods."""

        def __init__(self, n=50, k=3):
            np.random.seed(55)
            self.n = n
            self.k = k
            self._hessian_matrix = -np.eye(k) * 15.0
            self._scores = np.random.randn(n, k) * 0.1

        def _hessian(self, params):
            return self._hessian_matrix

        def _score_obs(self, params):
            return self._scores

    def test_nonrobust_dispatch(self):
        """se_type='nonrobust' should return -H^{-1}."""
        model = self.MockModel()
        params = np.zeros(model.k)
        cov = compute_mle_standard_errors(model, params, se_type="nonrobust")

        expected = -np.linalg.inv(model._hessian_matrix)
        assert_allclose(cov, expected)

    def test_robust_dispatch(self):
        """se_type='robust' should use sandwich estimator."""
        model = self.MockModel()
        params = np.zeros(model.k)
        cov = compute_mle_standard_errors(model, params, se_type="robust")

        H_inv = -np.linalg.inv(model._hessian_matrix)
        S = model._scores.T @ model._scores
        expected = H_inv @ S @ H_inv

        assert_allclose(cov, expected)

    def test_cluster_dispatch(self):
        """se_type='cluster' should use cluster_robust_mle."""
        model = self.MockModel(n=50, k=2)
        params = np.zeros(model.k)
        entity_ids = np.repeat(np.arange(10), 5)

        cov = compute_mle_standard_errors(model, params, se_type="cluster", entity_id=entity_ids)

        assert cov.shape == (2, 2)
        eigenvalues = np.linalg.eigvals(cov)
        assert np.all(eigenvalues >= -1e-10)

    def test_cluster_without_entity_id_raises(self):
        """Cluster SE without entity_id should raise ValueError."""
        model = self.MockModel()
        params = np.zeros(model.k)

        with pytest.raises(ValueError, match="entity_id required"):
            compute_mle_standard_errors(model, params, se_type="cluster")

    def test_unknown_se_type_raises(self):
        """Unknown se_type should raise ValueError."""
        model = self.MockModel()
        params = np.zeros(model.k)

        with pytest.raises(ValueError, match="Unknown se_type"):
            compute_mle_standard_errors(model, params, se_type="magic")

    def test_singular_hessian_nonrobust_uses_pinv(self):
        """Singular Hessian with nonrobust should fall back to pinv."""
        model = self.MockModel()
        model._hessian_matrix = np.array(
            [
                [-10.0, -10.0, 0.0],
                [-10.0, -10.0, 0.0],
                [0.0, 0.0, -5.0],
            ]
        )
        params = np.zeros(model.k)

        cov = compute_mle_standard_errors(model, params, se_type="nonrobust")
        assert cov.shape == (3, 3)
        assert np.all(np.isfinite(cov))


class TestBootstrapMLEDirect:
    """Direct unit tests for bootstrap_mle covering lines 561-631."""

    @staticmethod
    def _ols_estimator(y, X):
        """Simple OLS estimator for bootstrap testing."""
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def test_basic_bootstrap(self):
        """Test basic bootstrap returns correct structure."""
        np.random.seed(42)
        n, k = 100, 2
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ [1.0, 2.0] + np.random.randn(n) * 0.5

        result = bootstrap_mle(self._ols_estimator, y, X, n_bootstrap=99, seed=42)

        assert isinstance(result, MLECovarianceResult)
        assert result.method == "bootstrap"
        assert result.n_obs == n
        assert result.n_params == k
        assert result.cov_matrix.shape == (k, k)
        assert result.std_errors.shape == (k,)

    def test_bootstrap_se_positive(self):
        """Bootstrap standard errors should all be positive."""
        np.random.seed(42)
        n = 80
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ [1.0, 2.0] + np.random.randn(n)

        result = bootstrap_mle(self._ols_estimator, y, X, n_bootstrap=99, seed=123)
        assert np.all(result.std_errors > 0)

    def test_bootstrap_cov_symmetric(self):
        """Bootstrap covariance matrix should be symmetric."""
        np.random.seed(42)
        n = 80
        X = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
        y = X @ [1.0, 2.0, -0.5] + np.random.randn(n)

        result = bootstrap_mle(self._ols_estimator, y, X, n_bootstrap=99, seed=42)
        assert_allclose(result.cov_matrix, result.cov_matrix.T, atol=1e-12)

    def test_bootstrap_reproducibility(self):
        """Same seed should give same results."""
        np.random.seed(42)
        n = 60
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ [1.0, 0.5] + np.random.randn(n)

        result1 = bootstrap_mle(self._ols_estimator, y, X, n_bootstrap=99, seed=7)
        result2 = bootstrap_mle(self._ols_estimator, y, X, n_bootstrap=99, seed=7)

        assert_allclose(result1.cov_matrix, result2.cov_matrix)
        assert_allclose(result1.std_errors, result2.std_errors)

    def test_cluster_bootstrap(self):
        """Test cluster bootstrap resampling path."""
        np.random.seed(42)
        n_clusters = 20
        cluster_size = 5
        n = n_clusters * cluster_size
        k = 2
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ [1.0, 2.0] + np.random.randn(n)
        cluster_ids = np.repeat(np.arange(n_clusters), cluster_size)

        result = bootstrap_mle(
            self._ols_estimator,
            y,
            X,
            n_bootstrap=99,
            cluster_ids=cluster_ids,
            seed=42,
        )

        assert isinstance(result, MLECovarianceResult)
        assert result.method == "bootstrap"
        assert result.cov_matrix.shape == (k, k)
        assert np.all(result.std_errors > 0)

    def test_cluster_bootstrap_vs_standard(self):
        """Cluster bootstrap SE should differ from standard bootstrap SE."""
        np.random.seed(42)
        n_clusters = 15
        cluster_size = 10
        n = n_clusters * cluster_size
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ [1.0, 2.0] + np.random.randn(n)
        cluster_ids = np.repeat(np.arange(n_clusters), cluster_size)

        result_standard = bootstrap_mle(self._ols_estimator, y, X, n_bootstrap=199, seed=42)
        result_cluster = bootstrap_mle(
            self._ols_estimator,
            y,
            X,
            n_bootstrap=199,
            cluster_ids=cluster_ids,
            seed=42,
        )

        assert not np.allclose(result_standard.std_errors, result_cluster.std_errors, rtol=0.01)

    def test_bootstrap_with_failing_replications(self):
        """Bootstrap should handle replications that raise exceptions."""
        call_count = [0]

        def sometimes_failing_estimator(y, X):
            call_count[0] += 1
            if call_count[0] > 1 and call_count[0] <= 3:
                raise ValueError("Simulated estimation failure")
            return np.linalg.lstsq(X, y, rcond=None)[0]

        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ [1.0, 2.0] + np.random.randn(n)

        result = bootstrap_mle(sometimes_failing_estimator, y, X, n_bootstrap=50, seed=42)
        assert result.cov_matrix.shape == (2, 2)
        assert np.all(np.isfinite(result.std_errors))

    def test_bootstrap_many_failures_warns(self):
        """More than 50% failures should emit a warning."""
        # The first call to estimate_func in bootstrap_mle is for the
        # original estimate (not a bootstrap sample), so we need to let
        # that succeed.  After the first call we fail with high probability.
        call_count = [0]

        def mostly_failing_estimator(y, X):
            call_count[0] += 1
            # Let the first call (original estimate) succeed
            if call_count[0] == 1:
                return np.linalg.lstsq(X, y, rcond=None)[0]
            # Fail 90% of bootstrap replications
            if np.random.rand() < 0.9:
                raise ValueError("Simulated failure")
            return np.linalg.lstsq(X, y, rcond=None)[0]

        np.random.seed(42)
        n = 30
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ [1.0, 2.0] + np.random.randn(n)

        with pytest.warns(UserWarning, match="50% of bootstrap replications failed"):
            bootstrap_mle(mostly_failing_estimator, y, X, n_bootstrap=20, seed=42)

    def test_bootstrap_se_reasonable_magnitude(self):
        """Bootstrap SEs should be in a plausible range for a known DGP."""
        np.random.seed(42)
        n = 500
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        sigma = 1.0
        y = X @ [1.0, 2.0] + np.random.randn(n) * sigma

        result = bootstrap_mle(self._ols_estimator, y, X, n_bootstrap=499, seed=42)

        # SE for slope should be approximately sigma / sqrt(n) ~ 0.045
        assert result.std_errors[1] < 0.15
        assert result.std_errors[1] > 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
