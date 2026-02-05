"""
Unit tests for cluster-robust standard errors.

Tests cover:
- One-way clustering by entity/time
- Two-way clustering
- Finite-sample corrections
- Cluster size diagnostics
- Cameron-Gelbach-Miller (2011) formula
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from panelbox.standard_errors import (
    ClusteredCovarianceResult,
    ClusteredStandardErrors,
    cluster_by_entity,
    cluster_by_time,
    twoway_cluster,
)
from panelbox.standard_errors.utils import (
    compute_bread,
    compute_clustered_meat,
    compute_twoway_clustered_meat,
    sandwich_covariance,
)


class TestClusteredMeat:
    """Test clustered meat computation."""

    @pytest.fixture
    def setup_panel(self):
        """Create panel data setup."""
        np.random.seed(42)
        n_entities = 10
        n_periods = 5
        n = n_entities * n_periods
        k = 3

        X = np.random.randn(n, k)
        resid = np.random.randn(n)

        # Entity clusters
        entity_ids = np.repeat(np.arange(n_entities), n_periods)

        # Time clusters
        time_ids = np.tile(np.arange(n_periods), n_entities)

        return X, resid, entity_ids, time_ids

    def test_meat_shape(self, setup_panel):
        """Clustered meat should be k x k."""
        X, resid, entity_ids, _ = setup_panel
        k = X.shape[1]

        meat = compute_clustered_meat(X, resid, entity_ids)
        assert meat.shape == (k, k)

    def test_meat_symmetric(self, setup_panel):
        """Clustered meat should be symmetric."""
        X, resid, entity_ids, _ = setup_panel

        meat = compute_clustered_meat(X, resid, entity_ids)
        assert_allclose(meat, meat.T, rtol=1e-10)

    def test_meat_formula(self, setup_panel):
        """Verify meat = Σ_g (X_g' ε_g)(X_g' ε_g)'."""
        X, resid, entity_ids, _ = setup_panel
        k = X.shape[1]

        meat = compute_clustered_meat(X, resid, entity_ids, df_correction=False)

        # Manual computation
        unique_clusters = np.unique(entity_ids)
        meat_manual = np.zeros((k, k))

        for cluster_id in unique_clusters:
            mask = entity_ids == cluster_id
            X_c = X[mask]
            resid_c = resid[mask]
            score_c = X_c.T @ resid_c
            meat_manual += np.outer(score_c, score_c)

        assert_allclose(meat, meat_manual, rtol=1e-10)

    def test_df_correction(self, setup_panel):
        """Test finite-sample correction."""
        X, resid, entity_ids, _ = setup_panel
        n, k = X.shape
        n_clusters = len(np.unique(entity_ids))

        meat_no_corr = compute_clustered_meat(X, resid, entity_ids, df_correction=False)
        meat_with_corr = compute_clustered_meat(X, resid, entity_ids, df_correction=True)

        # Correction: G/(G-1) × (N-1)/(N-K)
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        expected = correction * meat_no_corr

        assert_allclose(meat_with_corr, expected, rtol=1e-10)

    def test_single_cluster(self):
        """Test with single cluster (should work but give warning-worthy results)."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        resid = np.random.randn(50)
        clusters = np.zeros(50)  # All same cluster

        # Should not raise error
        meat = compute_clustered_meat(X, resid, clusters, df_correction=False)
        assert meat.shape == (3, 3)


class TestTwoWayClusteredMeat:
    """Test two-way clustered meat computation."""

    @pytest.fixture
    def setup_twoway(self):
        """Create two-way clustered data."""
        np.random.seed(42)
        n_entities = 8
        n_periods = 6
        n = n_entities * n_periods
        k = 4

        X = np.random.randn(n, k)
        resid = np.random.randn(n)

        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_entities)

        return X, resid, entity_ids, time_ids

    def test_twoway_formula(self, setup_twoway):
        """Verify V = V_1 + V_2 - V_12."""
        X, resid, entity_ids, time_ids = setup_twoway

        meat_twoway = compute_twoway_clustered_meat(
            X, resid, entity_ids, time_ids, df_correction=False
        )

        # Manual computation
        meat1 = compute_clustered_meat(X, resid, entity_ids, df_correction=False)
        meat2 = compute_clustered_meat(X, resid, time_ids, df_correction=False)

        # Intersection clusters
        clusters_12 = np.array([f"{e}_{t}" for e, t in zip(entity_ids, time_ids)])
        meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction=False)

        expected = meat1 + meat2 - meat12

        assert_allclose(meat_twoway, expected, rtol=1e-10)

    def test_twoway_symmetric(self, setup_twoway):
        """Two-way meat should be symmetric."""
        X, resid, entity_ids, time_ids = setup_twoway

        meat = compute_twoway_clustered_meat(X, resid, entity_ids, time_ids)
        assert_allclose(meat, meat.T, rtol=1e-10)


class TestClusteredStandardErrors:
    """Test ClusteredStandardErrors class."""

    @pytest.fixture
    def setup_panel(self):
        """Create panel data."""
        np.random.seed(42)
        n_entities = 12
        n_periods = 8
        n = n_entities * n_periods
        k = 5

        X = np.random.randn(n, k)

        # Create clustered errors
        entity_effects = np.random.randn(n_entities) * 2
        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        resid = entity_effects[entity_ids] + np.random.randn(n) * 0.5

        time_ids = np.tile(np.arange(n_periods), n_entities)

        return X, resid, entity_ids, time_ids

    def test_initialization_oneway(self, setup_panel):
        """Test initialization with one-way clustering."""
        X, resid, entity_ids, _ = setup_panel

        clustered = ClusteredStandardErrors(X, resid, entity_ids)

        assert clustered.n_obs == len(X)
        assert clustered.n_params == 5
        assert clustered.cluster_dims == 1
        assert clustered.df_correction is True

    def test_initialization_twoway(self, setup_panel):
        """Test initialization with two-way clustering."""
        X, resid, entity_ids, time_ids = setup_panel

        clustered = ClusteredStandardErrors(X, resid, (entity_ids, time_ids))

        assert clustered.cluster_dims == 2
        assert hasattr(clustered, "clusters1")
        assert hasattr(clustered, "clusters2")

    def test_invalid_cluster_dims(self, setup_panel):
        """Test that invalid cluster dimensions raise error."""
        X, resid, entity_ids, time_ids = setup_panel

        with pytest.raises(ValueError, match="exactly 2 cluster dimensions"):
            ClusteredStandardErrors(X, resid, (entity_ids, time_ids, entity_ids))

    def test_dimension_mismatch(self, setup_panel):
        """Test that dimension mismatch raises error."""
        X, resid, entity_ids, _ = setup_panel

        # Wrong length clusters
        wrong_clusters = entity_ids[:-5]

        with pytest.raises(ValueError, match="Cluster dimension mismatch"):
            ClusteredStandardErrors(X, resid, wrong_clusters)

    def test_n_clusters_oneway(self, setup_panel):
        """Test n_clusters property for one-way."""
        X, resid, entity_ids, _ = setup_panel

        clustered = ClusteredStandardErrors(X, resid, entity_ids)
        assert clustered.n_clusters == 12

    def test_n_clusters_twoway(self, setup_panel):
        """Test n_clusters property for two-way."""
        X, resid, entity_ids, time_ids = setup_panel

        clustered = ClusteredStandardErrors(X, resid, (entity_ids, time_ids))
        assert clustered.n_clusters == (12, 8)

    def test_compute_oneway(self, setup_panel):
        """Test compute() for one-way clustering."""
        X, resid, entity_ids, _ = setup_panel

        clustered = ClusteredStandardErrors(X, resid, entity_ids)
        result = clustered.compute()

        assert isinstance(result, ClusteredCovarianceResult)
        assert result.cov_matrix.shape == (5, 5)
        assert result.std_errors.shape == (5,)
        assert result.n_clusters == 12
        assert result.cluster_dims == 1
        assert result.df_correction is True

    def test_compute_twoway(self, setup_panel):
        """Test compute() for two-way clustering."""
        X, resid, entity_ids, time_ids = setup_panel

        clustered = ClusteredStandardErrors(X, resid, (entity_ids, time_ids))
        result = clustered.compute()

        assert result.cluster_dims == 2
        assert result.n_clusters == (12, 8)

    def test_bread_caching(self, setup_panel):
        """Test that bread is cached."""
        X, resid, entity_ids, _ = setup_panel

        clustered = ClusteredStandardErrors(X, resid, entity_ids)
        assert clustered._bread is None

        bread1 = clustered.bread
        assert clustered._bread is not None

        bread2 = clustered.bread
        assert bread1 is bread2

    def test_positive_standard_errors(self, setup_panel):
        """Standard errors should be positive."""
        X, resid, entity_ids, time_ids = setup_panel

        # One-way
        result1 = ClusteredStandardErrors(X, resid, entity_ids).compute()
        assert np.all(result1.std_errors > 0)

        # Two-way
        result2 = ClusteredStandardErrors(X, resid, (entity_ids, time_ids)).compute()
        assert np.all(result2.std_errors > 0)

    def test_symmetric_covariance(self, setup_panel):
        """Covariance should be symmetric."""
        X, resid, entity_ids, _ = setup_panel

        result = ClusteredStandardErrors(X, resid, entity_ids).compute()
        assert_allclose(result.cov_matrix, result.cov_matrix.T, rtol=1e-10)

    def test_df_correction_effect(self, setup_panel):
        """Test effect of df_correction."""
        X, resid, entity_ids, _ = setup_panel

        result_no_corr = ClusteredStandardErrors(
            X, resid, entity_ids, df_correction=False
        ).compute()

        result_with_corr = ClusteredStandardErrors(
            X, resid, entity_ids, df_correction=True
        ).compute()

        # With correction should have larger SEs
        assert np.all(result_with_corr.std_errors >= result_no_corr.std_errors - 1e-10)


class TestDiagnosticSummary:
    """Test diagnostic_summary() method."""

    def test_oneway_summary(self):
        """Test diagnostic summary for one-way clustering."""
        np.random.seed(42)
        n_entities = 25
        n_periods = 10
        n = n_entities * n_periods

        X = np.random.randn(n, 3)
        resid = np.random.randn(n)
        entity_ids = np.repeat(np.arange(n_entities), n_periods)

        clustered = ClusteredStandardErrors(X, resid, entity_ids)
        summary = clustered.diagnostic_summary()

        assert "Cluster-Robust Standard Errors Diagnostics" in summary
        assert "Number of clusters: 25" in summary
        assert "Observations: 250" in summary
        assert "Clustering dimension: 1" in summary

    def test_twoway_summary(self):
        """Test diagnostic summary for two-way clustering."""
        np.random.seed(42)
        n_entities = 15
        n_periods = 12
        n = n_entities * n_periods

        X = np.random.randn(n, 3)
        resid = np.random.randn(n)
        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_entities)

        clustered = ClusteredStandardErrors(X, resid, (entity_ids, time_ids))
        summary = clustered.diagnostic_summary()

        assert "Clustering dimensions: 2" in summary
        assert f"Number of clusters (dim 1): {n_entities}" in summary
        assert f"Number of clusters (dim 2): {n_periods}" in summary

    def test_warning_few_clusters(self):
        """Test warning for few clusters."""
        np.random.seed(42)
        n_clusters = 8
        n_per_cluster = 10
        n = n_clusters * n_per_cluster

        X = np.random.randn(n, 3)
        resid = np.random.randn(n)
        clusters = np.repeat(np.arange(n_clusters), n_per_cluster)

        clustered = ClusteredStandardErrors(X, resid, clusters)
        summary = clustered.diagnostic_summary()

        assert "CRITICAL" in summary
        assert "Very few clusters (<10)" in summary


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture
    def setup_data(self):
        """Create test data."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 4)
        resid = np.random.randn(n)
        entity_ids = np.repeat(np.arange(20), 5)
        time_ids = np.tile(np.arange(5), 20)
        return X, resid, entity_ids, time_ids

    def test_cluster_by_entity(self, setup_data):
        """Test cluster_by_entity() function."""
        X, resid, entity_ids, _ = setup_data

        result = cluster_by_entity(X, resid, entity_ids)

        assert isinstance(result, ClusteredCovarianceResult)
        assert result.n_clusters == 20
        assert result.cluster_dims == 1

    def test_cluster_by_time(self, setup_data):
        """Test cluster_by_time() function."""
        X, resid, _, time_ids = setup_data

        result = cluster_by_time(X, resid, time_ids)

        assert isinstance(result, ClusteredCovarianceResult)
        assert result.n_clusters == 5
        assert result.cluster_dims == 1

    def test_twoway_cluster(self, setup_data):
        """Test twoway_cluster() function."""
        X, resid, entity_ids, time_ids = setup_data

        result = twoway_cluster(X, resid, entity_ids, time_ids)

        assert isinstance(result, ClusteredCovarianceResult)
        assert result.n_clusters == (20, 5)
        assert result.cluster_dims == 2

    def test_convenience_matches_class(self, setup_data):
        """Test that convenience functions match class methods."""
        X, resid, entity_ids, _ = setup_data

        # Using class
        clustered = ClusteredStandardErrors(X, resid, entity_ids)
        result1 = clustered.compute()

        # Using function
        result2 = cluster_by_entity(X, resid, entity_ids)

        assert_allclose(result1.std_errors, result2.std_errors)
        assert_allclose(result1.cov_matrix, result2.cov_matrix)


class TestClusterPatterns:
    """Test different cluster patterns and edge cases."""

    def test_balanced_clusters(self):
        """Test with perfectly balanced clusters."""
        np.random.seed(42)
        n_clusters = 10
        n_per_cluster = 15
        n = n_clusters * n_per_cluster

        X = np.random.randn(n, 3)
        resid = np.random.randn(n)
        clusters = np.repeat(np.arange(n_clusters), n_per_cluster)

        result = cluster_by_entity(X, resid, clusters)
        assert result.n_clusters == n_clusters

    def test_unbalanced_clusters(self):
        """Test with unbalanced clusters."""
        np.random.seed(42)
        # Create clusters of different sizes
        cluster_sizes = [5, 10, 3, 8, 12, 7]
        n = sum(cluster_sizes)

        X = np.random.randn(n, 3)
        resid = np.random.randn(n)

        clusters = np.concatenate([np.full(size, i) for i, size in enumerate(cluster_sizes)])

        result = cluster_by_entity(X, resid, clusters)
        assert result.n_clusters == len(cluster_sizes)

    def test_singleton_clusters(self):
        """Test with some singleton clusters (size 1)."""
        np.random.seed(42)
        n = 20
        X = np.random.randn(n, 3)
        resid = np.random.randn(n)

        # Half singletons, half in one cluster
        clusters = np.concatenate(
            [np.arange(10), np.full(10, 10)]  # 10 singletons  # 1 cluster of size 10
        )

        result = cluster_by_entity(X, resid, clusters)
        assert result.n_clusters == 11

    def test_string_cluster_ids(self):
        """Test with string cluster identifiers."""
        np.random.seed(42)
        n = 30
        X = np.random.randn(n, 3)
        resid = np.random.randn(n)

        # String cluster IDs
        clusters = np.array(["A", "B", "C"] * 10)

        result = cluster_by_entity(X, resid, clusters)
        assert result.n_clusters == 3


class TestComparisonWithRobust:
    """Compare clustered SEs with robust SEs."""

    def test_clustered_vs_robust(self):
        """Clustered SEs should generally be larger than robust SEs."""
        np.random.seed(42)
        n_entities = 20
        n_periods = 5
        n = n_entities * n_periods

        X = np.random.randn(n, 4)

        # Create clustered errors
        entity_effects = np.random.randn(n_entities)
        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        resid = entity_effects[entity_ids] + np.random.randn(n) * 0.1

        # Clustered SEs
        from panelbox.standard_errors import robust_covariance

        result_robust = robust_covariance(X, resid, method="HC1")
        result_clustered = cluster_by_entity(X, resid, entity_ids)

        # Clustered SEs should be larger (accounting for within-cluster correlation)
        # Note: This is not always true, but generally holds with clustered errors
        # We just check that both are positive and finite
        assert np.all(result_robust.std_errors > 0)
        assert np.all(result_clustered.std_errors > 0)
        assert np.all(np.isfinite(result_robust.std_errors))
        assert np.all(np.isfinite(result_clustered.std_errors))


class TestEdgeCases:
    """Test edge cases."""

    def test_single_observation_per_cluster(self):
        """Test when each cluster has one observation."""
        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 3)
        resid = np.random.randn(n)
        clusters = np.arange(n)  # Each obs is its own cluster

        result = cluster_by_entity(X, resid, clusters)
        assert result.n_clusters == n

    def test_all_same_cluster(self):
        """Test when all observations in same cluster."""
        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 3)
        resid = np.random.randn(n)
        clusters = np.zeros(n)  # All same cluster

        result = cluster_by_entity(X, resid, clusters)
        assert result.n_clusters == 1

    def test_perfect_fit(self):
        """Test with zero residuals."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        resid = np.zeros(n)
        clusters = np.repeat(np.arange(20), 5)

        result = cluster_by_entity(X, resid, clusters)

        # With zero residuals, covariance should be zero
        assert_allclose(result.cov_matrix, 0, atol=1e-10)
        assert_allclose(result.std_errors, 0, atol=1e-10)


class TestNumericalStability:
    """Test numerical stability."""

    def test_large_residuals(self):
        """Test with very large residuals."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        resid = 1e10 * np.random.randn(n)
        clusters = np.repeat(np.arange(20), 5)

        result = cluster_by_entity(X, resid, clusters)

        assert np.all(np.isfinite(result.std_errors))
        assert np.all(result.std_errors > 0)

    def test_many_clusters(self):
        """Test with large number of clusters."""
        np.random.seed(42)
        n_clusters = 500
        n_per_cluster = 3
        n = n_clusters * n_per_cluster

        X = np.random.randn(n, 4)
        resid = np.random.randn(n)
        clusters = np.repeat(np.arange(n_clusters), n_per_cluster)

        result = cluster_by_entity(X, resid, clusters)
        assert result.n_clusters == n_clusters
        assert np.all(np.isfinite(result.std_errors))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
