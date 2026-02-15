"""
Unit tests for quantile regression optimization algorithms.

Tests cover:
- Interior point method convergence
- Smooth approximation method
- Optimization accuracy
- Performance on various sample sizes
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from panelbox.optimization.quantile import check_loss, interior_point_qr, smooth_qr


class TestInteriorPoint:
    """Tests for interior point optimization method."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        np.random.seed(42)
        n_obs = 100
        n_vars = 3

        X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, n_vars - 1)])
        true_params = np.array([0.5, 0.8, -0.3])
        y = X @ true_params + np.random.randn(n_obs)

        return y, X, true_params

    def test_interior_point_convergence(self, simple_data):
        """Test that interior point method converges."""
        y, X, _ = simple_data

        params, converged = interior_point_qr(y, X, tau=0.5, maxiter=1000)

        assert converged, "Interior point method should converge"
        assert not np.any(np.isnan(params)), "Parameters should not contain NaN"
        assert len(params) == X.shape[1], "Wrong number of parameters"

    def test_interior_point_multiple_quantiles(self, simple_data):
        """Test interior point on multiple quantiles."""
        y, X, _ = simple_data

        for tau in [0.25, 0.5, 0.75]:
            params, converged = interior_point_qr(y, X, tau=tau)

            assert converged, f"Should converge for tau={tau}"
            assert len(params) == X.shape[1]

    def test_interior_point_accuracy(self, simple_data):
        """Test accuracy of interior point estimates."""
        y, X, true_params = simple_data

        # For median regression, should recover roughly correct parameters
        params, _ = interior_point_qr(y, X, tau=0.5, maxiter=1000)

        # Parameters should be within reasonable distance of true values
        # (accounting for noise and estimation error)
        assert_allclose(params, true_params, rtol=0.5, atol=0.2)

    def test_interior_point_performance_large_sample(self):
        """Test interior point performance on large sample."""
        np.random.seed(42)
        n_obs = 1000
        n_vars = 10

        X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, n_vars - 1)])
        y = X @ np.ones(n_vars) + np.random.randn(n_obs)

        import time

        start = time.time()
        params, converged = interior_point_qr(y, X, tau=0.5, maxiter=1000)
        elapsed = time.time() - start

        assert converged, "Should converge"
        assert elapsed < 5.0, f"Should converge in <5 seconds, took {elapsed:.2f}s"

    def test_interior_point_initial_params(self, simple_data):
        """Test that initial parameters are used."""
        y, X, true_params = simple_data

        # With good initial parameters
        params_init = true_params + 0.1 * np.random.randn(len(true_params))
        params, converged = interior_point_qr(y, X, tau=0.5, params_init=params_init)

        assert converged
        assert len(params) == len(params_init)

    def test_check_loss_computation(self, simple_data):
        """Test check loss computation."""
        y, X, true_params = simple_data

        residuals = y - X @ true_params
        loss = check_loss(residuals, tau=0.5)

        # Check loss should be positive and finite
        assert loss > 0, "Check loss should be positive"
        assert np.isfinite(loss), "Check loss should be finite"

        # Check loss computation formula
        expected = np.sum((0.5 - (residuals < 0)) * residuals)
        assert_allclose(loss, expected)


class TestSmoothApproximation:
    """Tests for smooth approximation method."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        np.random.seed(42)
        n_obs = 100
        n_vars = 3

        X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, n_vars - 1)])
        true_params = np.array([0.5, 0.8, -0.3])
        y = X @ true_params + np.random.randn(n_obs)

        return y, X, true_params

    def test_smooth_qr_convergence(self, simple_data):
        """Test that smooth QR method converges."""
        y, X, _ = simple_data

        params, converged = smooth_qr(y, X, tau=0.5, maxiter=1000)

        assert converged, "Smooth QR should converge"
        assert not np.any(np.isnan(params)), "Parameters should not contain NaN"
        assert len(params) == X.shape[1]

    def test_smooth_qr_vs_interior_point(self, simple_data):
        """Compare smooth QR with interior point."""
        y, X, _ = simple_data

        params_smooth, _ = smooth_qr(y, X, tau=0.5, maxiter=1000)
        params_ip, _ = interior_point_qr(y, X, tau=0.5, maxiter=1000)

        # Both methods should give similar results
        assert_allclose(params_smooth, params_ip, rtol=0.1, atol=0.05)

    def test_smooth_qr_bandwidth_selection(self, simple_data):
        """Test automatic bandwidth selection."""
        y, X, _ = simple_data

        # Should work with automatic bandwidth
        params, converged = smooth_qr(y, X, tau=0.5, bandwidth=None)

        assert converged
        assert len(params) == X.shape[1]

    def test_smooth_qr_custom_bandwidth(self, simple_data):
        """Test with custom bandwidth."""
        y, X, _ = simple_data

        params, converged = smooth_qr(y, X, tau=0.5, bandwidth=0.1)

        assert converged
        assert len(params) == X.shape[1]


class TestOptimizationEdgeCases:
    """Test edge cases and robustness."""

    def test_perfect_fit(self):
        """Test when perfect fit is possible."""
        np.random.seed(42)
        n_obs = 50
        n_vars = 3

        true_params = np.array([1.0, 0.5, -0.3])
        X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, n_vars - 1)])
        y = X @ true_params  # Perfect fit, no noise

        params, converged = interior_point_qr(y, X, tau=0.5)

        assert converged
        assert_allclose(params, true_params, atol=1e-6)

    def test_high_dimensional(self):
        """Test with moderate high-dimensional data."""
        np.random.seed(42)
        n_obs = 100
        n_vars = 20

        X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, n_vars - 1)])
        y = np.random.randn(n_obs)

        params, converged = interior_point_qr(y, X, tau=0.5, maxiter=100)

        assert len(params) == n_vars
        # Might not converge with fewer iterations for higher dimensions

    def test_collinear_columns(self):
        """Test behavior with nearly collinear columns."""
        np.random.seed(42)
        n_obs = 50

        x1 = np.random.randn(n_obs)
        X = np.column_stack([np.ones(n_obs), x1, 1.01 * x1 + 1e-8 * np.random.randn(n_obs)])
        y = X[:, 1] + np.random.randn(n_obs)

        params, _ = interior_point_qr(y, X, tau=0.5)

        # Should still produce valid estimates
        assert len(params) == 3
        assert not np.any(np.isnan(params))


class TestConvergenceCriteria:
    """Test convergence criteria and stopping rules."""

    def test_max_iterations(self):
        """Test that max iterations is respected."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        params, _ = interior_point_qr(y, X, tau=0.5, maxiter=2)

        # Should still return valid parameters
        assert len(params) == 5
        assert not np.any(np.isnan(params))

    def test_tolerance_effect(self):
        """Test that tolerance affects convergence."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        # Loose tolerance
        params_loose, _ = interior_point_qr(y, X, tau=0.5, tol=1e-3, maxiter=100)

        # Tight tolerance
        params_tight, _ = interior_point_qr(y, X, tau=0.5, tol=1e-8, maxiter=100)

        # Tight tolerance should give better convergence (closer to optimal)
        y_pred_loose = X @ params_loose
        y_pred_tight = X @ params_tight

        loss_loose = check_loss(y - y_pred_loose, tau=0.5)
        loss_tight = check_loss(y - y_pred_tight, tau=0.5)

        assert loss_tight <= loss_loose + 1e-6
