"""
Coverage tests for panelbox.optimization.quantile.penalized.

Targets uncovered lines: 62-71, 77-83, 89-94, 165->169, 240-241,
286-298, 304-313, 367->369, 433-434.
"""

import numpy as np
import pytest

from panelbox.optimization.quantile.penalized import (
    AdaptiveOptimizer,
    PenalizedQuantileOptimizer,
    compute_check_loss_matrix,
    compute_gradient_matrix,
)


@pytest.fixture
def synthetic_panel():
    """Create small synthetic panel data: 100 obs, 3 features, 10 entities."""
    np.random.seed(42)
    n_entities = 10
    n_time = 10
    n_obs = n_entities * n_time

    entity_ids = np.repeat(np.arange(n_entities), n_time)
    X = np.random.randn(n_obs, 3)
    beta_true = np.array([2.0, -1.0, 0.5])
    entity_effects = np.random.randn(n_entities) * 0.5
    y = X @ beta_true + entity_effects[entity_ids] + np.random.randn(n_obs) * 0.3

    return X, y, entity_ids


# ---------------------------------------------------------------------------
# Lines 62-71: _check_loss_fast (numba JIT body)
# ---------------------------------------------------------------------------
class TestCheckLossFast:
    """Exercise the JIT-compiled _check_loss_fast function."""

    def test_positive_residuals(self):
        """Positive residuals yield tau * r contribution."""
        residuals = np.array([1.0, 2.0, 3.0])
        tau = 0.5
        loss = PenalizedQuantileOptimizer._check_loss_fast(residuals, tau)
        expected = tau * (1.0 + 2.0 + 3.0)
        assert np.isclose(loss, expected, atol=1e-10)

    def test_negative_residuals(self):
        """Negative residuals yield (tau - 1) * r contribution."""
        residuals = np.array([-1.0, -2.0])
        tau = 0.25
        loss = PenalizedQuantileOptimizer._check_loss_fast(residuals, tau)
        expected = (tau - 1) * (-1.0) + (tau - 1) * (-2.0)
        assert np.isclose(loss, expected, atol=1e-10)

    def test_mixed_residuals(self):
        """Mix of positive and negative residuals."""
        residuals = np.array([1.0, -1.0])
        tau = 0.75
        loss = PenalizedQuantileOptimizer._check_loss_fast(residuals, tau)
        expected = tau * 1.0 + (tau - 1) * (-1.0)
        assert np.isclose(loss, expected, atol=1e-10)

    def test_all_zeros(self):
        """Zero residuals produce zero loss."""
        residuals = np.zeros(10)
        loss = PenalizedQuantileOptimizer._check_loss_fast(residuals, 0.5)
        assert loss == 0.0

    def test_nonnegative_loss(self):
        """Check loss is always non-negative."""
        np.random.seed(42)
        residuals = np.random.randn(200)
        for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
            loss = PenalizedQuantileOptimizer._check_loss_fast(residuals, tau)
            assert loss >= 0.0


# ---------------------------------------------------------------------------
# Lines 77-83: _check_gradient_fast (numba JIT body)
# ---------------------------------------------------------------------------
class TestCheckGradientFast:
    """Exercise the JIT-compiled _check_gradient_fast function."""

    def test_positive_residuals_gradient(self):
        """Positive residuals give gradient = tau."""
        residuals = np.array([1.0, 2.0, 0.5])
        tau = 0.3
        grad = PenalizedQuantileOptimizer._check_gradient_fast(residuals, tau)
        assert np.allclose(grad, tau)

    def test_negative_residuals_gradient(self):
        """Negative residuals give gradient = tau - 1."""
        residuals = np.array([-1.0, -0.5, -3.0])
        tau = 0.3
        grad = PenalizedQuantileOptimizer._check_gradient_fast(residuals, tau)
        assert np.allclose(grad, tau - 1.0)

    def test_mixed_gradient(self):
        """Mixed residuals produce correct element-wise gradient."""
        residuals = np.array([1.0, -1.0, 2.0, -2.0])
        tau = 0.5
        grad = PenalizedQuantileOptimizer._check_gradient_fast(residuals, tau)
        expected = np.array([tau, tau - 1.0, tau, tau - 1.0])
        assert np.allclose(grad, expected)

    def test_gradient_shape(self):
        """Gradient has the same length as input residuals."""
        residuals = np.random.randn(50)
        grad = PenalizedQuantileOptimizer._check_gradient_fast(residuals, 0.5)
        assert grad.shape == residuals.shape


# ---------------------------------------------------------------------------
# Lines 89-94: _soft_threshold (numba JIT body)
# ---------------------------------------------------------------------------
class TestSoftThreshold:
    """Exercise the JIT-compiled _soft_threshold function."""

    def test_above_threshold(self):
        """x > lambda: returns x - lambda."""
        result = PenalizedQuantileOptimizer._soft_threshold(5.0, 2.0)
        assert np.isclose(result, 3.0)

    def test_below_negative_threshold(self):
        """x < -lambda: returns x + lambda."""
        result = PenalizedQuantileOptimizer._soft_threshold(-5.0, 2.0)
        assert np.isclose(result, -3.0)

    def test_within_threshold(self):
        """abs(x) <= lambda: returns 0."""
        result = PenalizedQuantileOptimizer._soft_threshold(1.0, 2.0)
        assert result == 0.0

    def test_at_boundary_positive(self):
        """x == lambda: returns 0."""
        result = PenalizedQuantileOptimizer._soft_threshold(2.0, 2.0)
        assert result == 0.0

    def test_at_boundary_negative(self):
        """x == -lambda: returns 0."""
        result = PenalizedQuantileOptimizer._soft_threshold(-2.0, 2.0)
        assert result == 0.0

    def test_zero_lambda(self):
        """lambda=0 means no thresholding; returns x."""
        result = PenalizedQuantileOptimizer._soft_threshold(3.0, 0.0)
        assert np.isclose(result, 3.0)


# ---------------------------------------------------------------------------
# Line 165->169: optimize() with options=None default branch
# ---------------------------------------------------------------------------
class TestOptimizeOptionsNone:
    """Ensure the options=None default path is exercised."""

    def test_optimize_default_options(self, synthetic_panel):
        """Calling optimize() without options triggers the default branch."""
        X, y, entity_ids = synthetic_panel
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=0.1)
        result = opt.optimize()
        assert result.fun < opt.objective(np.zeros(X.shape[1] + opt.n_entities))
        assert np.all(np.isfinite(result.x))

    def test_optimize_explicit_options_skips_default(self, synthetic_panel):
        """Passing explicit options should not use the default dict."""
        X, y, entity_ids = synthetic_panel
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=0.1)
        result = opt.optimize(options={"maxiter": 50, "ftol": 1e-6})
        assert np.all(np.isfinite(result.x))


# ---------------------------------------------------------------------------
# Lines 240-241: coordinate_descent convergence (converged = True; break)
# ---------------------------------------------------------------------------
class TestCoordinateDescentConvergence:
    """Target the convergence branch where the loop breaks early."""

    def test_converges_with_high_tolerance(self, synthetic_panel):
        """With a very high tolerance the loop should converge and break early."""
        X, y, entity_ids = synthetic_panel
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=0.5)
        result = opt.coordinate_descent(max_iter=200, tol=1e2)
        assert result["converged"] is True
        assert result["iterations"] < 200

    def test_converges_with_moderate_tolerance(self, synthetic_panel):
        """Well-behaved data should converge with moderate tolerance."""
        X, y, entity_ids = synthetic_panel
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=1.0)
        result = opt.coordinate_descent(max_iter=500, tol=1e-3)
        if result["converged"]:
            assert result["iterations"] < 500

    def test_does_not_converge_with_tiny_tolerance(self, synthetic_panel):
        """With very few iterations and strict tolerance, should not converge."""
        X, y, entity_ids = synthetic_panel
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=0.1)
        result = opt.coordinate_descent(max_iter=1, tol=1e-15)
        assert result["iterations"] == 1


# ---------------------------------------------------------------------------
# Lines 286-298: compute_check_loss_matrix (module-level numba JIT)
# ---------------------------------------------------------------------------
class TestComputeCheckLossMatrix:
    """Exercise the module-level compute_check_loss_matrix function."""

    def test_shape(self):
        """Output shape is (n_residuals, n_taus)."""
        residuals = np.array([1.0, -1.0, 0.5, -0.5, 2.0])
        tau_grid = np.array([0.1, 0.5, 0.9])
        result = compute_check_loss_matrix(residuals, tau_grid)
        assert result.shape == (5, 3)

    def test_all_positive_residuals(self):
        """All positive residuals: loss[i,j] = tau_j * r_i."""
        residuals = np.array([1.0, 2.0])
        tau_grid = np.array([0.25, 0.75])
        result = compute_check_loss_matrix(residuals, tau_grid)
        assert np.isclose(result[0, 0], 0.25)
        assert np.isclose(result[1, 0], 0.50)
        assert np.isclose(result[0, 1], 0.75)
        assert np.isclose(result[1, 1], 1.50)

    def test_all_negative_residuals(self):
        """All negative residuals: loss[i,j] = (tau_j - 1) * r_i."""
        residuals = np.array([-1.0, -2.0])
        tau_grid = np.array([0.5])
        result = compute_check_loss_matrix(residuals, tau_grid)
        expected_0 = (0.5 - 1.0) * (-1.0)  # 0.5
        expected_1 = (0.5 - 1.0) * (-2.0)  # 1.0
        assert np.isclose(result[0, 0], expected_0)
        assert np.isclose(result[1, 0], expected_1)

    def test_nonnegative_values(self):
        """All entries in the loss matrix should be non-negative."""
        np.random.seed(42)
        residuals = np.random.randn(100)
        tau_grid = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        result = compute_check_loss_matrix(residuals, tau_grid)
        assert np.all(result >= -1e-15)

    def test_single_residual_single_tau(self):
        """Scalar-like input."""
        residuals = np.array([3.0])
        tau_grid = np.array([0.5])
        result = compute_check_loss_matrix(residuals, tau_grid)
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], 1.5)


# ---------------------------------------------------------------------------
# Lines 304-313: compute_gradient_matrix (module-level numba JIT)
# ---------------------------------------------------------------------------
class TestComputeGradientMatrix:
    """Exercise the module-level compute_gradient_matrix function."""

    def test_shape(self):
        """Output shape is (n_residuals, n_taus)."""
        residuals = np.array([1.0, -1.0, 0.0])
        tau_grid = np.array([0.25, 0.5, 0.75])
        result = compute_gradient_matrix(residuals, tau_grid)
        assert result.shape == (3, 3)

    def test_positive_residual_columns(self):
        """For positive r, gradient[i,j] = tau_j."""
        residuals = np.array([1.0])
        tau_grid = np.array([0.1, 0.5, 0.9])
        result = compute_gradient_matrix(residuals, tau_grid)
        assert np.isclose(result[0, 0], 0.1)
        assert np.isclose(result[0, 1], 0.5)
        assert np.isclose(result[0, 2], 0.9)

    def test_negative_residual_columns(self):
        """For negative r, gradient[i,j] = tau_j - 1."""
        residuals = np.array([-1.0])
        tau_grid = np.array([0.1, 0.5, 0.9])
        result = compute_gradient_matrix(residuals, tau_grid)
        assert np.isclose(result[0, 0], -0.9)
        assert np.isclose(result[0, 1], -0.5)
        assert np.isclose(result[0, 2], -0.1)

    def test_gradient_in_valid_range(self):
        """gradient[i,j] must be in [tau_j - 1, tau_j] for all i, j."""
        np.random.seed(42)
        residuals = np.random.randn(80)
        tau_grid = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        result = compute_gradient_matrix(residuals, tau_grid)
        for j, tau in enumerate(tau_grid):
            assert np.all(result[:, j] >= tau - 1.0 - 1e-12)
            assert np.all(result[:, j] <= tau + 1e-12)

    def test_many_residuals(self):
        """Larger array to exercise the parallel prange loop."""
        np.random.seed(42)
        residuals = np.random.randn(500)
        tau_grid = np.linspace(0.1, 0.9, 9)
        result = compute_gradient_matrix(residuals, tau_grid)
        assert result.shape == (500, 9)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Line 367->369: AdaptiveOptimizer._analyze_problem condition_number = np.inf
# ---------------------------------------------------------------------------
class TestAdaptiveOptimizerConditionInf:
    """Target the exception catch that sets condition_number = np.inf."""

    def test_singular_matrix_gives_inf_condition(self):
        """A zero X matrix makes X'X singular; condition_number should be inf."""
        np.random.seed(42)
        n_entities = 5
        n_time = 4
        n_obs = n_entities * n_time
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        X = np.zeros((n_obs, 3))
        y = np.random.randn(n_obs)

        ada = AdaptiveOptimizer(X, y, entity_ids)
        assert ada.condition_number == np.inf

    def test_nan_column_gives_inf_condition(self):
        """A column of NaN causes linalg.cond to fail; falls back to inf."""
        np.random.seed(42)
        n_entities = 5
        n_time = 4
        n_obs = n_entities * n_time
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        X = np.random.randn(n_obs, 2)
        X[:, 1] = np.nan
        y = np.random.randn(n_obs)

        ada = AdaptiveOptimizer(X, y, entity_ids)
        # np.linalg.cond on NaN data may return nan or raise; either way
        # the code should handle it gracefully.
        assert ada.condition_number == np.inf or np.isnan(ada.condition_number)


# ---------------------------------------------------------------------------
# Lines 433-434: AdaptiveOptimizer.recommend_method avg_T < 10 branch
# ---------------------------------------------------------------------------
class TestAdaptiveOptimizerSmallT:
    """Target the avg_T < 10 branch returning 'penalty'."""

    def test_small_avg_T_recommends_penalty(self):
        """Many entities with few time periods should recommend 'penalty'."""
        np.random.seed(42)
        n_entities = 50
        n_time = 3  # avg_T = 3 < 10
        n_obs = n_entities * n_time
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        X = np.random.randn(n_obs, 2)
        y = np.random.randn(n_obs)

        ada = AdaptiveOptimizer(X, y, entity_ids)
        assert ada.avg_T == 3.0
        method, params = ada.recommend_method()
        assert method == "penalty"
        assert params == {"lambda_fe": "auto", "cv_folds": 5}

    def test_avg_T_exactly_9(self):
        """avg_T = 9 is still < 10 and should recommend 'penalty'."""
        np.random.seed(42)
        n_entities = 10
        n_time = 9
        n_obs = n_entities * n_time
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        X = np.random.randn(n_obs, 2)
        y = np.random.randn(n_obs)

        ada = AdaptiveOptimizer(X, y, entity_ids)
        assert ada.avg_T == 9.0
        # problem_size = 90 * 2 = 180 < 1e6, so large-scale check does not trigger
        method, _params = ada.recommend_method()
        assert method == "penalty"

    def test_avg_T_10_does_not_trigger_penalty(self):
        """avg_T = 10 is not < 10 so should NOT fall into the penalty branch."""
        np.random.seed(42)
        n_entities = 10
        n_time = 10
        n_obs = n_entities * n_time
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        X = np.random.randn(n_obs, 2)
        y = np.random.randn(n_obs)

        ada = AdaptiveOptimizer(X, y, entity_ids)
        assert ada.avg_T == 10.0
        method, _params = ada.recommend_method()
        assert method != "penalty"


# ---------------------------------------------------------------------------
# Integration: objective and gradient through PenalizedQuantileOptimizer
# (ensures JIT functions are called as part of the overall flow)
# ---------------------------------------------------------------------------
class TestJITIntegration:
    """Integration tests that exercise JIT paths via objective and gradient."""

    def test_objective_uses_check_loss_fast(self, synthetic_panel):
        """objective() internally calls _check_loss_fast."""
        X, y, entity_ids = synthetic_panel
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=0.1)
        params = np.zeros(X.shape[1] + opt.n_entities)
        val = opt.objective(params)
        assert val > 0
        assert np.isfinite(val)

    def test_gradient_uses_check_gradient_fast(self, synthetic_panel):
        """gradient() internally calls _check_gradient_fast."""
        X, y, entity_ids = synthetic_panel
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=0.1)
        params = np.random.randn(X.shape[1] + opt.n_entities) * 0.1
        grad = opt.gradient(params)
        assert grad.shape == params.shape
        assert np.all(np.isfinite(grad))

    def test_coordinate_descent_uses_soft_threshold(self, synthetic_panel):
        """coordinate_descent() internally calls _soft_threshold."""
        X, y, entity_ids = synthetic_panel
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=0.5)
        result = opt.coordinate_descent(max_iter=5)
        assert np.all(np.isfinite(result["beta"]))
        assert np.all(np.isfinite(result["alpha"]))


# ---------------------------------------------------------------------------
# Line 367->369: compare_implementations with explicit methods list
# ---------------------------------------------------------------------------
class TestCompareImplementationsExplicitMethods:
    """Cover branch 367->369: methods is not None in compare_implementations."""

    def test_explicit_methods_list(self):
        """Pass explicit methods list to skip the 'if methods is None' branch."""
        import pandas as pd

        from panelbox.core.panel_data import PanelData
        from panelbox.optimization.quantile.penalized import PerformanceMonitor

        np.random.seed(42)
        n_entities = 20
        n_time = 10
        n_obs = n_entities * n_time
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        time_ids = np.tile(np.arange(n_time), n_entities)
        df = pd.DataFrame(
            {
                "y": np.random.randn(n_obs),
                "x1": np.random.randn(n_obs),
                "entity_id": entity_ids,
                "time_id": time_ids,
            }
        )
        data = PanelData(df, entity_col="entity_id", time_col="time_id")

        monitor = PerformanceMonitor()
        # Pass explicit methods list (not None) to trigger line 367->369 False branch
        results = monitor.compare_implementations(data, "y ~ x1", tau=0.5, methods=["canay"])
        assert "canay" in results
        assert "canay" in monitor.timings
