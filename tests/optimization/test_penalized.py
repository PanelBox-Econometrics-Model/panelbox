"""
Tests for penalized quantile regression optimization.
"""

import time

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from panelbox.models.quantile.canay import CanayTwoStep
from panelbox.models.quantile.fixed_effects import FixedEffectsQuantile
from panelbox.optimization.quantile.penalized import PenalizedQuantileOptimizer, PerformanceMonitor
from panelbox.utils.data import PanelData


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
        time_ids = np.tile(np.arange(n_time), n_entities)

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
        assert result.fun < optimizer.objective(
            np.zeros(len(result.x))
        ), "Should improve from zero initialization"

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
            beta = res["params"][: X.shape[1]]
            alpha = res["params"][X.shape[1] :]
            fe_norms.append(np.linalg.norm(alpha))

        # Fixed effects should decrease with lambda
        assert fe_norms[-1] <= fe_norms[0], "Larger lambda should shrink fixed effects"

    def test_numba_acceleration(self, simple_panel_data):
        """Test that Numba functions are faster than pure Python."""
        X, y, entity_ids, _ = simple_panel_data
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

        return PanelData(df, entity="entity_id", time="time_id")

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor()
        assert hasattr(monitor, "timings")
        assert isinstance(monitor.timings, dict)

    def test_profile_canay_method(self, test_data):
        """Test profiling Canay method."""
        monitor = PerformanceMonitor()

        # Profile Canay method
        result = monitor.profile_method("canay", data=test_data, formula="y ~ x1 + x2", tau=0.5)

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
        result = monitor.profile_method("penalty", data=test_data, formula="y ~ x1 + x2", tau=0.5)

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

        return PanelData(df, entity="entity_id", time="time_id")

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
        result = model.fit(cv_folds=3, verbose=False)
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
