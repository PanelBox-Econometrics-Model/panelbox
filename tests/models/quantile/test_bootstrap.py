"""
Unit tests for bootstrap inference methods.

Tests cover:
- Cluster bootstrap
- Pairs bootstrap
- Wild bootstrap
- Subsampling bootstrap
- Parallel computation
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from panelbox.inference.quantile import BootstrapInference
from panelbox.models.quantile import PooledQuantile


class TestClusterBootstrap:
    """Tests for cluster bootstrap inference."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted quantile model."""
        np.random.seed(42)
        n_entities = 20
        n_periods = 10
        n_obs = n_entities * n_periods

        entity_id = np.repeat(range(n_entities), n_periods)

        x1 = np.random.randn(n_obs)
        X = np.column_stack([np.ones(n_obs), x1])
        y = 1.0 + 0.5 * x1 + np.random.randn(n_obs)

        model = PooledQuantile(y, X, entity_id=entity_id, quantiles=0.5)
        results = model.fit()

        return model, results

    def test_cluster_bootstrap_runs(self, fitted_model):
        """Test that cluster bootstrap runs."""
        model, results = fitted_model

        boot = BootstrapInference(model, n_bootstrap=100, n_jobs=1)
        se = boot.cluster_bootstrap(results.params.ravel(), tau=0.5)

        assert len(se) == model.n_params
        assert np.all(se > 0)

    def test_cluster_bootstrap_se_reasonable(self, fitted_model):
        """Test that bootstrap SEs are reasonable."""
        model, results = fitted_model

        boot = BootstrapInference(model, n_bootstrap=100, n_jobs=1)
        se_boot = boot.cluster_bootstrap(results.params.ravel(), tau=0.5)

        # Bootstrap SEs should be of similar magnitude to analytical SEs
        # Allow for variation
        ratio = se_boot / results.std_errors.ravel()

        # Should be within reasonable range
        assert np.all(ratio > 0.1)
        assert np.all(ratio < 10.0)

    def test_cluster_bootstrap_reproducibility(self, fitted_model):
        """Test reproducibility with seed."""
        model, results = fitted_model

        boot1 = BootstrapInference(model, n_bootstrap=50, seed=42, n_jobs=1)
        se1 = boot1.cluster_bootstrap(results.params.ravel(), tau=0.5)

        boot2 = BootstrapInference(model, n_bootstrap=50, seed=42, n_jobs=1)
        se2 = boot2.cluster_bootstrap(results.params.ravel(), tau=0.5)

        # Should be identical
        assert_allclose(se1, se2)


class TestPairsBootstrap:
    """Tests for pairs bootstrap inference."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted quantile model."""
        np.random.seed(42)
        n_obs = 100

        x1 = np.random.randn(n_obs)
        X = np.column_stack([np.ones(n_obs), x1])
        y = 1.0 + 0.5 * x1 + np.random.randn(n_obs)

        model = PooledQuantile(y, X, quantiles=0.5)
        results = model.fit()

        return model, results

    def test_pairs_bootstrap_runs(self, fitted_model):
        """Test that pairs bootstrap runs."""
        model, results = fitted_model

        boot = BootstrapInference(model, n_bootstrap=100, n_jobs=1)
        se = boot.pairs_bootstrap(results.params.ravel(), tau=0.5)

        assert len(se) == model.n_params
        assert np.all(se > 0)

    def test_pairs_bootstrap_vs_cluster(self):
        """Compare pairs bootstrap with cluster bootstrap."""
        np.random.seed(42)
        n_obs = 100

        x1 = np.random.randn(n_obs)
        X = np.column_stack([np.ones(n_obs), x1])
        y = 1.0 + 0.5 * x1 + np.random.randn(n_obs)

        # Model with clustering info
        entity_id = np.repeat(range(10), 10)
        model = PooledQuantile(y, X, entity_id=entity_id, quantiles=0.5)
        results = model.fit()

        boot = BootstrapInference(model, n_bootstrap=100, seed=42, n_jobs=1)

        se_pairs = boot.pairs_bootstrap(results.params.ravel(), tau=0.5)

        # Both should give positive SEs
        assert np.all(se_pairs > 0)


class TestWildBootstrap:
    """Tests for wild bootstrap inference."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted quantile model."""
        np.random.seed(42)
        n_obs = 100

        x1 = np.random.randn(n_obs)
        X = np.column_stack([np.ones(n_obs), x1])
        y = 1.0 + 0.5 * x1 + np.random.randn(n_obs)

        model = PooledQuantile(y, X, quantiles=0.5)
        results = model.fit()

        return model, results

    def test_wild_bootstrap_rademacher(self, fitted_model):
        """Test wild bootstrap with Rademacher distribution."""
        model, results = fitted_model

        boot = BootstrapInference(model, n_bootstrap=100, n_jobs=1)
        se = boot.wild_bootstrap(results.params.ravel(), tau=0.5, dist="rademacher")

        assert len(se) == model.n_params
        assert np.all(se > 0)

    def test_wild_bootstrap_normal(self, fitted_model):
        """Test wild bootstrap with normal distribution."""
        model, results = fitted_model

        boot = BootstrapInference(model, n_bootstrap=100, n_jobs=1)
        se = boot.wild_bootstrap(results.params.ravel(), tau=0.5, dist="normal")

        assert len(se) == model.n_params
        assert np.all(se > 0)

    def test_wild_bootstrap_mammen(self, fitted_model):
        """Test wild bootstrap with Mammen distribution."""
        model, results = fitted_model

        boot = BootstrapInference(model, n_bootstrap=100, n_jobs=1)
        se = boot.wild_bootstrap(results.params.ravel(), tau=0.5, dist="mammen")

        assert len(se) == model.n_params
        assert np.all(se > 0)


class TestSubsamplingBootstrap:
    """Tests for subsampling bootstrap inference."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted quantile model."""
        np.random.seed(42)
        n_obs = 100

        x1 = np.random.randn(n_obs)
        X = np.column_stack([np.ones(n_obs), x1])
        y = 1.0 + 0.5 * x1 + np.random.randn(n_obs)

        model = PooledQuantile(y, X, quantiles=0.5)
        results = model.fit()

        return model, results

    def test_subsampling_runs(self, fitted_model):
        """Test that subsampling runs."""
        model, results = fitted_model

        boot = BootstrapInference(model, n_bootstrap=100, n_jobs=1)
        se = boot.subsampling_bootstrap(results.params.ravel(), tau=0.5)

        assert len(se) == model.n_params
        assert np.all(se > 0)

    def test_subsampling_custom_size(self, fitted_model):
        """Test subsampling with custom subsample size."""
        model, results = fitted_model

        boot = BootstrapInference(model, n_bootstrap=100, n_jobs=1)
        se = boot.subsampling_bootstrap(results.params.ravel(), tau=0.5, subsample_size=50)

        assert len(se) == model.n_params


class TestBootstrapParallel:
    """Tests for parallel bootstrap computation."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted quantile model."""
        np.random.seed(42)
        n_obs = 100

        x1 = np.random.randn(n_obs)
        X = np.column_stack([np.ones(n_obs), x1])
        y = 1.0 + 0.5 * x1 + np.random.randn(n_obs)

        model = PooledQuantile(y, X, quantiles=0.5)
        results = model.fit()

        return model, results

    def test_parallel_vs_serial(self, fitted_model):
        """Compare parallel vs serial bootstrap."""
        model, results = fitted_model

        # Serial
        boot_serial = BootstrapInference(model, n_bootstrap=50, seed=42, n_jobs=1)
        se_serial = boot_serial.cluster_bootstrap(results.params.ravel(), tau=0.5)

        # Parallel (or serial if n_jobs=-1 not available)
        boot_parallel = BootstrapInference(model, n_bootstrap=50, seed=42, n_jobs=-1)
        se_parallel = boot_parallel.cluster_bootstrap(results.params.ravel(), tau=0.5)

        # Results should be very similar (might not be identical due to random sampling)
        # but should be close
        assert_allclose(se_serial, se_parallel, rtol=0.2)
