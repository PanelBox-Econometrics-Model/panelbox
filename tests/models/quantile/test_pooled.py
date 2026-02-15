"""
Unit tests for pooled quantile regression model.

Tests cover:
- Basic estimation
- Multiple quantiles
- Standard errors (cluster, robust, nonrobust)
- Predictions
- Results object
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_less

from panelbox.models.quantile import PooledQuantile, PooledQuantileResults


class TestPooledQuantileBasic:
    """Tests for basic pooled quantile regression."""

    @pytest.fixture
    def panel_data(self):
        """Generate panel data for testing."""
        np.random.seed(42)
        n_entities = 50
        n_periods = 10
        n_obs = n_entities * n_periods

        # Generate panel structure
        entity_id = np.repeat(range(n_entities), n_periods)
        time_id = np.tile(range(n_periods), n_entities)

        # Generate covariates
        x1 = np.random.randn(n_obs)
        x2 = np.random.randn(n_obs)

        # True parameters for median regression
        beta_true = np.array([0.5, 0.8, -0.3])

        # Generate outcome
        X = np.column_stack([np.ones(n_obs), x1, x2])
        linear_pred = X @ beta_true
        y = linear_pred + np.random.randn(n_obs)

        return {
            "y": y,
            "X": X,
            "entity_id": entity_id,
            "time_id": time_id,
            "beta_true": beta_true,
            "x1": x1,
            "x2": x2,
        }

    def test_pooled_quantile_basic(self, panel_data):
        """Test basic pooled quantile estimation."""
        model = PooledQuantile(
            panel_data["y"], panel_data["X"], entity_id=panel_data["entity_id"], quantiles=0.5
        )

        results = model.fit()

        # Check that results were obtained
        assert results is not None
        assert hasattr(results, "params")
        assert len(results.params) == panel_data["X"].shape[1]

    def test_pooled_quantile_single_quantile(self, panel_data):
        """Test with single quantile."""
        model = PooledQuantile(
            panel_data["y"], panel_data["X"], entity_id=panel_data["entity_id"], quantiles=0.5
        )

        results = model.fit()

        # Check parameter shape
        assert results.params.shape[0] == panel_data["X"].shape[1]

        # Check standard errors
        assert results.std_errors is not None
        assert np.all(results.std_errors > 0)

    def test_pooled_quantile_multiple_quantiles(self, panel_data):
        """Test with multiple quantiles."""
        quantiles = [0.25, 0.5, 0.75]

        model = PooledQuantile(
            panel_data["y"], panel_data["X"], entity_id=panel_data["entity_id"], quantiles=quantiles
        )

        results = model.fit()

        # Check parameter shape
        assert results.params.shape == (panel_data["X"].shape[1], len(quantiles))

        # Check that different quantiles give different estimates
        assert not np.allclose(results.params[:, 0], results.params[:, 1])
        assert not np.allclose(results.params[:, 1], results.params[:, 2])

    def test_pooled_quantile_parameter_variation(self, panel_data):
        """Test that parameters vary across quantiles."""
        quantiles = [0.1, 0.5, 0.9]

        model = PooledQuantile(
            panel_data["y"], panel_data["X"], entity_id=panel_data["entity_id"], quantiles=quantiles
        )

        results = model.fit()

        # Different quantiles should give different results
        for i in range(len(quantiles) - 1):
            diff = np.abs(results.params[:, i] - results.params[:, i + 1])
            assert np.max(diff) > 1e-6, "Parameters should differ across quantiles"


class TestPooledQuantileStandardErrors:
    """Tests for standard error computation."""

    @pytest.fixture
    def panel_data(self):
        """Generate panel data."""
        np.random.seed(42)
        n_entities = 100
        n_periods = 10
        n_obs = n_entities * n_periods

        entity_id = np.repeat(range(n_entities), n_periods)
        time_id = np.tile(range(n_periods), n_entities)

        x1 = np.random.randn(n_obs)
        x2 = np.random.randn(n_obs)

        X = np.column_stack([np.ones(n_obs), x1, x2])
        beta = np.array([0.5, 0.8, -0.3])
        y = X @ beta + np.random.randn(n_obs)

        return {"y": y, "X": X, "entity_id": entity_id, "time_id": time_id}

    def test_se_types(self, panel_data):
        """Test different standard error types."""
        model = PooledQuantile(
            panel_data["y"], panel_data["X"], entity_id=panel_data["entity_id"], quantiles=0.5
        )

        # Cluster SE
        results_cluster = model.fit(se_type="cluster")
        assert results_cluster.std_errors is not None

        # Robust SE
        results_robust = model.fit(se_type="robust")
        assert results_robust.std_errors is not None

        # Nonrobust SE
        results_nonrobust = model.fit(se_type="nonrobust")
        assert results_nonrobust.std_errors is not None

        # All should be positive
        assert np.all(results_cluster.std_errors > 0)
        assert np.all(results_robust.std_errors > 0)
        assert np.all(results_nonrobust.std_errors > 0)

    def test_cluster_se_larger_than_nonrobust(self, panel_data):
        """Test that cluster SEs account for clustering."""
        model = PooledQuantile(
            panel_data["y"], panel_data["X"], entity_id=panel_data["entity_id"], quantiles=0.5
        )

        results_cluster = model.fit(se_type="cluster")
        results_nonrobust = model.fit(se_type="nonrobust")

        # Cluster SEs should typically be >= nonrobust SEs
        # (due to within-cluster correlation)
        se_ratio = results_cluster.std_errors.ravel() / results_nonrobust.std_errors.ravel()

        # At least half should be >= 1
        assert np.sum(se_ratio >= 1) >= len(se_ratio) / 2


class TestPooledQuantilePredictions:
    """Tests for predictions."""

    @pytest.fixture
    def simple_model(self):
        """Create simple fitted model."""
        np.random.seed(42)
        n_obs = 100

        X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, 2)])
        y = X @ np.array([1.0, 0.5, -0.3]) + np.random.randn(n_obs)
        entity_id = np.repeat(range(10), 10)

        model = PooledQuantile(y, X, entity_id=entity_id, quantiles=0.5)
        return model

    def test_predict_basic(self, simple_model):
        """Test basic predictions."""
        results = simple_model.fit()

        # Predict on training data
        pred = results.predict()

        assert len(pred) == len(simple_model.endog)
        assert np.all(np.isfinite(pred))

    def test_predict_new_data(self, simple_model):
        """Test predictions on new data."""
        results = simple_model.fit()

        # New data
        X_new = np.column_stack([np.ones(10), np.random.randn(10, 2)])

        pred = results.predict(exog=X_new)

        assert len(pred) == 10
        assert np.all(np.isfinite(pred))

    def test_predict_multiple_quantiles(self):
        """Test predictions with multiple quantiles."""
        np.random.seed(42)
        n_obs = 100

        X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, 2)])
        y = np.random.randn(n_obs)
        entity_id = np.repeat(range(10), 10)

        model = PooledQuantile(y, X, entity_id=entity_id, quantiles=[0.25, 0.5, 0.75])
        results = model.fit()

        # Predictions for different quantiles
        for q_idx in range(3):
            pred = results.predict(quantile_idx=q_idx)
            assert len(pred) == n_obs


class TestPooledQuantileResults:
    """Tests for results object."""

    @pytest.fixture
    def results_object(self):
        """Create results object."""
        np.random.seed(42)
        n_obs = 100

        X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, 2)])
        y = np.random.randn(n_obs)
        entity_id = np.repeat(range(10), 10)

        model = PooledQuantile(y, X, entity_id=entity_id, quantiles=[0.5])
        results = model.fit()

        return results

    def test_summary_output(self, results_object):
        """Test that summary produces output."""
        summary = results_object.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Quantile" in summary or "quantile" in summary

    def test_confidence_intervals(self, results_object):
        """Test confidence interval computation."""
        lower, upper = results_object.conf_int(alpha=0.05)

        # Check shape
        assert lower.shape == results_object.params.shape
        assert upper.shape == results_object.params.shape

        # Lower should be less than upper
        assert np.all(lower < upper)

    def test_results_attributes(self, results_object):
        """Test that results object has required attributes."""
        assert hasattr(results_object, "params")
        assert hasattr(results_object, "std_errors")
        assert hasattr(results_object, "tvalues")
        assert hasattr(results_object, "pvalues")
        assert hasattr(results_object, "quantiles")


class TestPooledQuantileEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_quantile(self):
        """Test error for invalid quantiles."""
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        # Quantile = 0 should fail
        with pytest.raises(ValueError):
            PooledQuantile(y, X, quantiles=0.0)

        # Quantile = 1 should fail
        with pytest.raises(ValueError):
            PooledQuantile(y, X, quantiles=1.0)

        # Quantile > 1 should fail
        with pytest.raises(ValueError):
            PooledQuantile(y, X, quantiles=1.5)

    def test_single_observation(self):
        """Test behavior with very small samples."""
        X = np.ones((1, 1))
        y = np.array([1.0])

        model = PooledQuantile(y, X, quantiles=0.5)

        # Should not crash, though convergence might not be achieved
        results = model.fit()
        assert results is not None

    def test_perfect_separation(self):
        """Test with data that has perfect fit."""
        X = np.column_stack([np.ones(100), np.arange(100)])
        y = X @ np.array([1.0, 2.0])
        entity_id = np.repeat(range(10), 10)

        model = PooledQuantile(y, X, entity_id=entity_id, quantiles=0.5)
        results = model.fit()

        # Should recover true parameters
        assert_allclose(results.params.ravel(), [1.0, 2.0], atol=1e-6)
