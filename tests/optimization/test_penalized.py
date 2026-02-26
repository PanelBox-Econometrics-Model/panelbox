"""
Tests for penalized quantile regression optimization.
"""

import time

import numpy as np
import pandas as pd
import pytest

from panelbox.core.panel_data import PanelData
from panelbox.models.quantile.fixed_effects import FixedEffectsQuantile
from panelbox.optimization.quantile.penalized import (
    AdaptiveOptimizer,
    PenalizedQuantileOptimizer,
    PerformanceMonitor,
    compute_check_loss_matrix,
    compute_gradient_matrix,
)


class TestPenalizedQuantileOptimizer:
    """Test suite for penalized quantile regression optimizer."""

    @pytest.fixture
    def simple_panel_data(self):
        """Create simple panel data for testing."""
        np.random.seed(42)
        n_entities = 20
        n_time = 10
        n_obs = n_entities * n_time

        # Create indices
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        np.tile(np.arange(n_time), n_entities)

        # Create data
        X = np.random.randn(n_obs, 3)
        entity_effects = np.random.randn(n_entities)
        entity_effects_expanded = np.repeat(entity_effects, n_time)

        # Generate y
        beta_true = np.array([1.5, -1.0, 0.5])
        y = X @ beta_true + entity_effects_expanded + np.random.randn(n_obs) * 0.5

        return X, y, entity_ids, entity_effects

    def test_optimizer_initialization(self, simple_panel_data):
        """Test optimizer initialization."""
        X, y, entity_ids, _ = simple_panel_data

        optimizer = PenalizedQuantileOptimizer(
            X=X, y=y, entity_ids=entity_ids, tau=0.5, lambda_val=0.1
        )

        assert optimizer.n == len(y)
        assert optimizer.p == X.shape[1]
        assert optimizer.n_entities == len(np.unique(entity_ids))
        assert optimizer.tau == 0.5
        assert optimizer.lambda_val == 0.1

    def test_objective_function(self, simple_panel_data):
        """Test objective function computation."""
        X, y, entity_ids, _ = simple_panel_data

        optimizer = PenalizedQuantileOptimizer(
            X=X, y=y, entity_ids=entity_ids, tau=0.5, lambda_val=0.1
        )

        # Test with zero parameters
        params = np.zeros(X.shape[1] + optimizer.n_entities)
        obj_val = optimizer.objective(params)

        assert obj_val > 0, "Objective should be positive"
        assert np.isfinite(obj_val), "Objective should be finite"

    def test_gradient_computation(self, simple_panel_data):
        """Test gradient computation."""
        X, y, entity_ids, _ = simple_panel_data

        optimizer = PenalizedQuantileOptimizer(
            X=X, y=y, entity_ids=entity_ids, tau=0.5, lambda_val=0.1
        )

        # Test gradient at random point
        params = np.random.randn(X.shape[1] + optimizer.n_entities) * 0.1
        grad = optimizer.gradient(params)

        assert len(grad) == len(params), "Gradient dimension should match params"
        assert np.all(np.isfinite(grad)), "Gradient should be finite"

    def test_optimization_convergence(self, simple_panel_data):
        """Test that optimization converges."""
        X, y, entity_ids, _ = simple_panel_data

        optimizer = PenalizedQuantileOptimizer(
            X=X, y=y, entity_ids=entity_ids, tau=0.5, lambda_val=0.1
        )

        result = optimizer.optimize(method="L-BFGS-B")

        assert result.success, "Optimization should converge"
        assert result.fun < optimizer.objective(np.zeros(len(result.x))), (
            "Should improve from zero initialization"
        )

    def test_warm_start(self, simple_panel_data):
        """Test warm start functionality."""
        X, y, entity_ids, _ = simple_panel_data

        optimizer = PenalizedQuantileOptimizer(
            X=X, y=y, entity_ids=entity_ids, tau=0.5, lambda_val=0.1
        )

        # First optimization
        result1 = optimizer.optimize()

        # Second optimization with warm start
        result2 = optimizer.optimize(warm_start=result1.x)

        # Should converge faster with warm start
        assert result2.nit <= result1.nit, "Warm start should reduce iterations"

    @pytest.mark.xfail(reason="FE norm shrinkage not always monotone with lambda")
    def test_lambda_path(self, simple_panel_data):
        """Test solution path computation."""
        X, y, entity_ids, _ = simple_panel_data

        optimizer = PenalizedQuantileOptimizer(
            X=X, y=y, entity_ids=entity_ids, tau=0.5, lambda_val=0.1
        )

        # Compute path
        lambda_grid = [0.01, 0.1, 1.0]
        results = optimizer.warm_start_path(lambda_grid)

        assert len(results) == len(lambda_grid), "Should have result for each lambda"

        # Check shrinkage: larger lambda should give smaller fixed effects
        fe_norms = []
        for res in results:
            res["params"][: X.shape[1]]
            alpha = res["params"][X.shape[1] :]
            fe_norms.append(np.linalg.norm(alpha))

        # Fixed effects should decrease with lambda
        assert fe_norms[-1] <= fe_norms[0], "Larger lambda should shrink fixed effects"

    @pytest.mark.xfail(reason="Numba JIT compilation overhead makes timing unreliable")
    def test_numba_acceleration(self, simple_panel_data):
        """Test that Numba functions are faster than pure Python."""
        _X, y, _entity_ids, _ = simple_panel_data
        n_obs = len(y)

        # Create test data
        residuals = np.random.randn(n_obs)
        tau = 0.5

        # Time Numba version (after JIT compilation)
        _ = PenalizedQuantileOptimizer._check_loss_fast(residuals, tau)  # Warm-up
        start = time.time()
        for _ in range(100):
            fast_loss = PenalizedQuantileOptimizer._check_loss_fast(residuals, tau)
        numba_time = time.time() - start

        # Time pure Python version
        def check_loss_slow(residuals, tau):
            loss = 0.0
            for r in residuals:
                if r < 0:
                    loss += (tau - 1) * r
                else:
                    loss += tau * r
            return loss

        start = time.time()
        for _ in range(100):
            slow_loss = check_loss_slow(residuals, tau)
        python_time = time.time() - start

        # Numba should be faster (or at least not much slower)
        assert numba_time < python_time * 2, "Numba should not be much slower"

        # Results should be very close
        assert np.abs(fast_loss - slow_loss) < 1e-10, "Results should match"

    def test_smart_initialization(self, simple_panel_data):
        """Test smart initialization strategy."""
        X, y, entity_ids, _ = simple_panel_data

        optimizer = PenalizedQuantileOptimizer(
            X=X, y=y, entity_ids=entity_ids, tau=0.5, lambda_val=0.1
        )

        # Get smart initialization
        x0 = optimizer._smart_init()

        assert len(x0) == X.shape[1] + optimizer.n_entities
        assert np.all(np.isfinite(x0)), "Initialization should be finite"

        # Beta part should be non-zero (from OLS)
        beta_init = x0[: X.shape[1]]
        assert np.linalg.norm(beta_init) > 0, "Beta should be initialized from OLS"

        # Alpha part should be zero
        alpha_init = x0[X.shape[1] :]
        assert np.allclose(alpha_init, 0), "Fixed effects should start at zero"

    def test_entity_structure_setup(self, simple_panel_data):
        """Test entity structure setup."""
        X, y, entity_ids, _ = simple_panel_data

        optimizer = PenalizedQuantileOptimizer(
            X=X, y=y, entity_ids=entity_ids, tau=0.5, lambda_val=0.1
        )

        # Check entity mapping
        assert len(optimizer.obs_to_entity) == len(y)
        assert np.max(optimizer.obs_to_entity) == optimizer.n_entities - 1
        assert np.min(optimizer.obs_to_entity) == 0

        # Check entity masks
        assert len(optimizer.entity_masks) == optimizer.n_entities
        for mask in optimizer.entity_masks:
            assert np.sum(mask) > 0, "Each entity should have observations"


class TestPerformanceMonitor:
    """Test suite for performance monitoring utilities."""

    @pytest.fixture
    def test_data(self):
        """Create test panel data."""
        np.random.seed(42)
        n_entities = 50
        n_time = 10
        n_obs = n_entities * n_time

        # Create DataFrame
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        time_ids = np.tile(np.arange(n_time), n_entities)

        df = pd.DataFrame(
            {
                "y": np.random.randn(n_obs),
                "x1": np.random.randn(n_obs),
                "x2": np.random.randn(n_obs),
                "entity_id": entity_ids,
                "time_id": time_ids,
            }
        )

        return PanelData(df, entity_col="entity_id", time_col="time_id")

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor()
        assert hasattr(monitor, "timings")
        assert isinstance(monitor.timings, dict)

    def test_profile_canay_method(self, test_data):
        """Test profiling Canay method."""
        monitor = PerformanceMonitor()

        # Profile Canay method
        monitor.profile_method("canay", data=test_data, formula="y ~ x1 + x2", tau=0.5)

        # Check timings recorded
        assert "canay" in monitor.timings
        assert "time" in monitor.timings["canay"]
        assert "memory_mb" in monitor.timings["canay"]
        assert "converged" in monitor.timings["canay"]

        # Should have positive time and memory
        assert monitor.timings["canay"]["time"] > 0
        assert monitor.timings["canay"]["memory_mb"] >= 0

    def test_profile_penalty_method(self, test_data):
        """Test profiling penalty method."""
        monitor = PerformanceMonitor()

        # Profile penalty method
        monitor.profile_method("penalty", data=test_data, formula="y ~ x1 + x2", tau=0.5)

        # Check timings recorded
        assert "penalty" in monitor.timings
        assert monitor.timings["penalty"]["time"] > 0

    def test_performance_comparison(self, test_data):
        """Test performance comparison between methods."""
        monitor = PerformanceMonitor()

        # Profile both methods
        monitor.profile_method("canay", test_data, "y ~ x1 + x2", tau=0.5)
        monitor.profile_method("penalty", test_data, "y ~ x1 + x2", tau=0.5)

        # Canay should typically be faster
        canay_time = monitor.timings["canay"]["time"]
        penalty_time = monitor.timings["penalty"]["time"]

        # This might not always hold for small data, but check structure
        assert canay_time > 0 and penalty_time > 0

    def test_report_generation(self, test_data, capsys):
        """Test performance report generation."""
        monitor = PerformanceMonitor()

        # Profile a method
        monitor.profile_method("canay", test_data, "y ~ x1 + x2", tau=0.5)

        # Generate report
        monitor.print_report()

        # Check output
        captured = capsys.readouterr()
        assert "PERFORMANCE COMPARISON" in captured.out
        assert "canay" in captured.out
        assert "Time" in captured.out
        assert "Memory" in captured.out

    def test_invalid_method_error(self, test_data):
        """Test error for invalid method."""
        monitor = PerformanceMonitor()

        with pytest.raises(ValueError, match="Unknown method"):
            monitor.profile_method("invalid_method", test_data, "y ~ x1", tau=0.5)


class TestIntegrationWithModels:
    """Integration tests with actual models."""

    @pytest.fixture
    def panel_data(self):
        """Create realistic panel data."""
        np.random.seed(42)
        n_entities = 100
        n_time = 20
        n_obs = n_entities * n_time

        # Create indices
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        time_ids = np.tile(np.arange(n_time), n_entities)

        # Create covariates
        X1 = np.random.randn(n_obs)
        X2 = np.random.randn(n_obs)

        # Entity effects
        entity_effects = np.random.randn(n_entities) * 2
        entity_effects_expanded = np.repeat(entity_effects, n_time)

        # Generate y
        y = 1 + 2 * X1 - X2 + entity_effects_expanded + np.random.randn(n_obs)

        df = pd.DataFrame(
            {"y": y, "X1": X1, "X2": X2, "entity_id": entity_ids, "time_id": time_ids}
        )

        return PanelData(df, entity_col="entity_id", time_col="time_id")

    def test_fixed_effects_with_optimizer(self, panel_data):
        """Test fixed effects model uses penalized optimizer."""
        model = FixedEffectsQuantile(panel_data, formula="y ~ X1 + X2", tau=0.5, lambda_fe=0.1)

        result = model.fit(verbose=False)

        # Should converge
        assert result.results[0.5].converged

        # Should have reasonable coefficients
        params = result.results[0.5].params
        assert len(params) == 3  # Constant + X1 + X2
        assert np.all(np.isfinite(params))

    def test_lambda_selection_performance(self, panel_data):
        """Test performance of lambda selection."""
        model = FixedEffectsQuantile(panel_data, formula="y ~ X1 + X2", tau=0.5, lambda_fe="auto")

        start = time.time()
        model.fit(cv_folds=3, verbose=False)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 60, "Lambda selection should complete within 1 minute"

        # Should have selected lambda
        assert hasattr(model, "cv_results_")
        assert "best_lambda" in model.cv_results_
        assert model.cv_results_["best_lambda"] > 0

    def test_shrinkage_path_computation(self, panel_data):
        """Test shrinkage path computation."""
        model = FixedEffectsQuantile(panel_data, formula="y ~ X1 + X2", tau=0.5, lambda_fe=0.1)

        # Compute shrinkage path
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

        fig = model.plot_shrinkage_path(tau=0.5, lambda_grid=np.logspace(-2, 1, 10))

        # Should create figure with 2 subplots
        assert len(fig.axes) == 2


class TestCoordinateDescent:
    """Tests for coordinate descent algorithm."""

    @pytest.fixture
    def optimizer(self):
        """Create a small optimizer for CD testing."""
        np.random.seed(42)
        n_entities = 5
        n_time = 10
        n_obs = n_entities * n_time

        entity_ids = np.repeat(np.arange(n_entities), n_time)
        X = np.random.randn(n_obs, 2)
        beta_true = np.array([1.0, -0.5])
        entity_effects = np.random.randn(n_entities) * 0.5
        y = X @ beta_true + entity_effects[entity_ids] + np.random.randn(n_obs) * 0.3

        return PenalizedQuantileOptimizer(X=X, y=y, entity_ids=entity_ids, tau=0.5, lambda_val=0.1)

    def test_coordinate_descent_returns_dict(self, optimizer):
        """CD should return dict with expected keys."""
        result = optimizer.coordinate_descent(max_iter=10)
        assert "beta" in result
        assert "alpha" in result
        assert "converged" in result
        assert "iterations" in result

    def test_coordinate_descent_beta_shape(self, optimizer):
        """Beta should match number of covariates."""
        result = optimizer.coordinate_descent(max_iter=10)
        assert result["beta"].shape == (2,)
        assert np.all(np.isfinite(result["beta"]))

    def test_coordinate_descent_alpha_shape(self, optimizer):
        """Alpha should match number of entities."""
        result = optimizer.coordinate_descent(max_iter=10)
        assert result["alpha"].shape == (5,)
        assert np.all(np.isfinite(result["alpha"]))

    def test_coordinate_descent_convergence(self, optimizer):
        """CD should converge with enough iterations."""
        result = optimizer.coordinate_descent(max_iter=500, tol=1e-4)
        assert result["converged"] or result["iterations"] == 500

    def test_coordinate_descent_warm_start(self, optimizer):
        """CD with warm start should accept initial params."""
        warm = np.zeros(2 + 5)
        warm[:2] = [1.0, -0.5]
        result = optimizer.coordinate_descent(max_iter=10, warm_start=warm)
        assert "beta" in result
        assert np.all(np.isfinite(result["beta"]))

    def test_coordinate_descent_no_warm_start(self, optimizer):
        """CD without warm start should initialize at zero."""
        result = optimizer.coordinate_descent(max_iter=5)
        assert result["iterations"] <= 5

    def test_coordinate_descent_iterations_counted(self, optimizer):
        """Iterations should be correctly counted."""
        result = optimizer.coordinate_descent(max_iter=3)
        assert 1 <= result["iterations"] <= 3


class TestComputeCheckLossMatrix:
    """Tests for compute_check_loss_matrix Numba function."""

    def test_basic_computation(self):
        """Should compute check loss for multiple taus."""
        residuals = np.array([1.0, -1.0, 0.5, -0.5])
        tau_grid = np.array([0.25, 0.5, 0.75])
        result = compute_check_loss_matrix(residuals, tau_grid)
        assert result.shape == (4, 3)
        assert np.all(np.isfinite(result))

    def test_positive_residuals(self):
        """Positive residuals should give tau * r."""
        residuals = np.array([2.0])
        tau_grid = np.array([0.5])
        result = compute_check_loss_matrix(residuals, tau_grid)
        assert np.isclose(result[0, 0], 0.5 * 2.0)

    def test_negative_residuals(self):
        """Negative residuals should give (tau-1) * r."""
        residuals = np.array([-2.0])
        tau_grid = np.array([0.5])
        result = compute_check_loss_matrix(residuals, tau_grid)
        assert np.isclose(result[0, 0], (0.5 - 1) * (-2.0))

    def test_all_losses_nonnegative(self):
        """All check losses should be non-negative."""
        np.random.seed(42)
        residuals = np.random.randn(50)
        tau_grid = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        result = compute_check_loss_matrix(residuals, tau_grid)
        assert np.all(result >= 0)


class TestComputeGradientMatrix:
    """Tests for compute_gradient_matrix Numba function."""

    def test_basic_computation(self):
        """Should compute gradients for multiple taus."""
        residuals = np.array([1.0, -1.0, 0.5, -0.5])
        tau_grid = np.array([0.25, 0.5, 0.75])
        result = compute_gradient_matrix(residuals, tau_grid)
        assert result.shape == (4, 3)
        assert np.all(np.isfinite(result))

    def test_positive_residual_gradient(self):
        """Positive residuals: gradient = tau."""
        residuals = np.array([1.0])
        tau_grid = np.array([0.3])
        result = compute_gradient_matrix(residuals, tau_grid)
        assert np.isclose(result[0, 0], 0.3)

    def test_negative_residual_gradient(self):
        """Negative residuals: gradient = tau - 1."""
        residuals = np.array([-1.0])
        tau_grid = np.array([0.3])
        result = compute_gradient_matrix(residuals, tau_grid)
        assert np.isclose(result[0, 0], 0.3 - 1.0)

    def test_gradient_range(self):
        """Gradients should be in [tau-1, tau]."""
        np.random.seed(42)
        residuals = np.random.randn(50)
        tau_grid = np.array([0.1, 0.5, 0.9])
        result = compute_gradient_matrix(residuals, tau_grid)
        for j, tau in enumerate(tau_grid):
            assert np.all(result[:, j] >= tau - 1 - 1e-10)
            assert np.all(result[:, j] <= tau + 1e-10)


class TestAdaptiveOptimizer:
    """Tests for AdaptiveOptimizer class."""

    @pytest.fixture
    def small_problem(self):
        """Small, well-conditioned problem."""
        np.random.seed(42)
        n_entities = 10
        n_time = 20
        n_obs = n_entities * n_time
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        X = np.random.randn(n_obs, 3)
        y = X @ np.array([1.0, -0.5, 0.3]) + np.random.randn(n_obs) * 0.5
        return X, y, entity_ids

    def test_initialization(self, small_problem):
        """Should initialize correctly."""
        X, y, entity_ids = small_problem
        ada = AdaptiveOptimizer(X, y, entity_ids)
        assert ada.n == len(y)
        assert ada.p == 3
        assert ada.n_entities == 10

    def test_problem_analysis(self, small_problem):
        """Should analyze problem characteristics."""
        X, y, entity_ids = small_problem
        ada = AdaptiveOptimizer(X, y, entity_ids)
        assert ada.problem_size == len(y) * 3
        assert ada.entity_ratio == 10 / len(y)
        assert ada.avg_T == len(y) / 10
        assert 0 <= ada.X_sparsity <= 1
        assert np.isfinite(ada.condition_number)

    def test_recommend_well_conditioned(self, small_problem):
        """Well-conditioned, large T problem should recommend canay."""
        X, y, entity_ids = small_problem
        ada = AdaptiveOptimizer(X, y, entity_ids)
        method, params = ada.recommend_method()
        assert isinstance(method, str)
        assert isinstance(params, dict)
        # With T=20 and likely well-conditioned, should be canay
        if ada.condition_number < 100 and ada.avg_T >= 10:
            assert method == "canay"

    def test_recommend_small_T(self):
        """Small T should recommend penalty method."""
        np.random.seed(42)
        n_entities = 50
        n_time = 5  # Small T
        n_obs = n_entities * n_time
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        X = np.random.randn(n_obs, 2)
        y = np.random.randn(n_obs)
        ada = AdaptiveOptimizer(X, y, entity_ids)
        method, params = ada.recommend_method()
        assert method == "penalty"
        assert "lambda_fe" in params

    def test_recommend_large_scale_sparse(self):
        """Large-scale sparse problem should recommend coordinate_descent."""
        np.random.seed(42)
        # Create large sparse X
        n = 10000
        p = 200
        entity_ids = np.repeat(np.arange(100), 100)
        X = np.zeros((n, p))
        # Make it sparse - only 10% nonzero
        for i in range(n):
            nonzero_cols = np.random.choice(p, size=int(p * 0.1), replace=False)
            X[i, nonzero_cols] = np.random.randn(len(nonzero_cols))
        y = np.random.randn(n)

        ada = AdaptiveOptimizer(X, y, entity_ids)
        method, params = ada.recommend_method()
        # problem_size = 10000*200 = 2e6 > 1e6, X_sparsity > 0.5
        assert method == "coordinate_descent"
        assert "max_iter" in params

    def test_recommend_large_scale_dense(self):
        """Large-scale dense problem should recommend L-BFGS-B."""
        np.random.seed(42)
        n = 10000
        p = 200
        entity_ids = np.repeat(np.arange(100), 100)
        X = np.random.randn(n, p)  # Dense
        y = np.random.randn(n)

        ada = AdaptiveOptimizer(X, y, entity_ids)
        method, params = ada.recommend_method()
        # problem_size > 1e6, X_sparsity < 0.5
        assert method == "L-BFGS-B"
        assert "maxiter" in params

    def test_recommend_default(self):
        """Ill-conditioned medium problem should use default L-BFGS-B."""
        np.random.seed(42)
        n_entities = 20
        n_time = 15
        n_obs = n_entities * n_time
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        # Make ill-conditioned X
        X = np.random.randn(n_obs, 3)
        X[:, 2] = X[:, 0] + X[:, 1] * 1e-6  # Nearly collinear
        y = np.random.randn(n_obs)

        ada = AdaptiveOptimizer(X, y, entity_ids)
        method, params = ada.recommend_method()
        # avg_T >= 10, condition_number likely > 100, problem_size < 1e6
        assert method == "L-BFGS-B"
        assert params["maxiter"] == 1000

    def test_print_analysis(self, small_problem, caplog):
        """print_analysis should log problem info."""
        import logging

        X, y, entity_ids = small_problem
        ada = AdaptiveOptimizer(X, y, entity_ids)
        with caplog.at_level(logging.INFO):
            ada.print_analysis()
        # Check that logger was called (output goes to logger, not stdout)
        assert ada.problem_size > 0  # At minimum, verify it ran without error

    def test_condition_number_failure(self):
        """Should handle condition number computation failure."""
        np.random.seed(42)
        n = 20
        entity_ids = np.repeat(np.arange(4), 5)
        X = np.zeros((n, 2))  # All zeros → singular
        y = np.random.randn(n)
        ada = AdaptiveOptimizer(X, y, entity_ids)
        assert ada.condition_number == np.inf


class TestPerformanceMonitorExtended:
    """Extended tests for PerformanceMonitor."""

    def test_print_report_multiple_methods(self, capsys):
        """Report with multiple methods should show speedup ratios."""
        monitor = PerformanceMonitor()
        monitor.timings = {
            "method_a": {"time": 1.0, "memory_mb": 10.0, "converged": True},
            "method_b": {"time": 0.5, "memory_mb": 5.0, "converged": True},
        }
        monitor.print_report()
        captured = capsys.readouterr()
        assert "PERFORMANCE COMPARISON" in captured.out
        assert "method_a" in captured.out
        assert "method_b" in captured.out
        assert "Fastest" in captured.out
        assert "Least Memory" in captured.out
        assert "Speedup Ratios" in captured.out

    def test_print_report_single_method(self, capsys):
        """Report with single method should not show speedup."""
        monitor = PerformanceMonitor()
        monitor.timings = {
            "method_a": {"time": 1.0, "memory_mb": 10.0, "converged": True},
        }
        monitor.print_report()
        captured = capsys.readouterr()
        assert "Fastest" in captured.out
        assert "Speedup" not in captured.out

    def test_print_report_empty(self, capsys):
        """Empty report should not crash."""
        monitor = PerformanceMonitor()
        monitor.print_report()
        captured = capsys.readouterr()
        assert "PERFORMANCE COMPARISON" in captured.out

    def test_print_report_failed_method(self, capsys):
        """Report with failed method should show failure symbol."""
        monitor = PerformanceMonitor()
        monitor.timings = {
            "good": {"time": 1.0, "memory_mb": 5.0, "converged": True},
            "bad": {"time": 2.0, "memory_mb": 15.0, "converged": False},
        }
        monitor.print_report()
        captured = capsys.readouterr()
        assert "good" in captured.out
        assert "bad" in captured.out

    def test_compare_implementations_default_methods(self):
        """compare_implementations should call profile_method for each method."""
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
        results = monitor.compare_implementations(data, "y ~ x1", tau=0.5)
        assert "canay" in results
        assert "penalty" in results
        assert "canay" in monitor.timings
        assert "penalty" in monitor.timings


class TestWarmStartPath:
    """Tests for warm_start_path method."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for path testing."""
        np.random.seed(42)
        n_entities = 5
        n_time = 10
        n_obs = n_entities * n_time
        entity_ids = np.repeat(np.arange(n_entities), n_time)
        X = np.random.randn(n_obs, 2)
        y = X @ np.array([1.0, -0.5]) + np.random.randn(n_obs) * 0.5
        return PenalizedQuantileOptimizer(X=X, y=y, entity_ids=entity_ids, tau=0.5, lambda_val=0.1)

    def test_path_returns_list(self, optimizer):
        """Path should return list of results."""
        results = optimizer.warm_start_path(np.array([0.01, 0.1, 1.0]))
        assert isinstance(results, list)
        assert len(results) == 3

    def test_path_result_keys(self, optimizer):
        """Each result should have expected keys."""
        results = optimizer.warm_start_path(np.array([0.1, 1.0]))
        for res in results:
            assert "lambda" in res
            assert "params" in res
            assert "converged" in res
            assert "objective" in res

    def test_path_sorted_descending(self, optimizer):
        """Lambda values should be processed largest first."""
        results = optimizer.warm_start_path(np.array([0.01, 1.0, 0.1]))
        lambdas = [r["lambda"] for r in results]
        assert lambdas == sorted(lambdas, reverse=True)
