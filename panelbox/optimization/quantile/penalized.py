"""
Optimized solvers for penalized quantile regression.

This module provides high-performance implementations using Numba JIT compilation
and sparse matrices for solving large-scale penalized quantile regression problems.
"""

import time
from typing import Any, Dict, Optional, Tuple

import numba
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, eye


class PenalizedQuantileOptimizer:
    """
    Optimized solver for penalized quantile regression.

    Uses Numba JIT compilation and sparse matrices for speed.
    """

    def __init__(
        self, X: np.ndarray, y: np.ndarray, entity_ids: np.ndarray, tau: float, lambda_val: float
    ):
        self.X = X
        self.y = y
        self.entity_ids = entity_ids
        self.tau = tau
        self.lambda_val = lambda_val

        self.n, self.p = X.shape
        self.n_entities = len(np.unique(entity_ids))

        # Pre-compute entity mappings
        self._setup_entity_structure()

    def _setup_entity_structure(self):
        """Pre-compute entity-related structures for efficiency."""
        # Map observations to entity indices
        entity_map = {eid: i for i, eid in enumerate(np.unique(self.entity_ids))}
        self.obs_to_entity = np.array([entity_map[eid] for eid in self.entity_ids])

        # Pre-compute entity masks (sparse for memory efficiency)
        self.entity_masks = []
        for i in range(self.n_entities):
            mask = self.obs_to_entity == i
            self.entity_masks.append(mask)

        # Pre-compute entity counts for efficiency
        self.entity_counts = np.array([np.sum(mask) for mask in self.entity_masks])

    @staticmethod
    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def _check_loss_fast(residuals: np.ndarray, tau: float) -> float:
        """JIT-compiled check loss computation."""
        n = len(residuals)
        loss = 0.0

        for i in numba.prange(n):
            if residuals[i] < 0:
                loss += (tau - 1) * residuals[i]
            else:
                loss += tau * residuals[i]

        return loss

    @staticmethod
    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def _check_gradient_fast(residuals: np.ndarray, tau: float) -> np.ndarray:
        """JIT-compiled check loss gradient."""
        n = len(residuals)
        grad = np.zeros(n)

        for i in numba.prange(n):
            grad[i] = tau - (1.0 if residuals[i] < 0 else 0.0)

        return grad

    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def _soft_threshold(x: float, lambda_val: float) -> float:
        """Soft thresholding operator for L1 penalty."""
        if x > lambda_val:
            return x - lambda_val
        elif x < -lambda_val:
            return x + lambda_val
        else:
            return 0.0

    def objective(self, params: np.ndarray) -> float:
        """Penalized objective function (optimized)."""
        beta = params[: self.p]
        alpha = params[self.p :]

        # Vectorized residual computation
        Xbeta = self.X @ beta
        alpha_expanded = alpha[self.obs_to_entity]
        residuals = self.y - Xbeta - alpha_expanded

        # Fast check loss
        check_loss = self._check_loss_fast(residuals, self.tau)

        # L1 penalty
        penalty = self.lambda_val * np.sum(np.abs(alpha))

        return check_loss + penalty

    def gradient(self, params: np.ndarray) -> np.ndarray:
        """Gradient of penalized objective (optimized)."""
        beta = params[: self.p]
        alpha = params[self.p :]

        # Residuals
        Xbeta = self.X @ beta
        alpha_expanded = alpha[self.obs_to_entity]
        residuals = self.y - Xbeta - alpha_expanded

        # Fast gradient computation
        psi = self._check_gradient_fast(residuals, self.tau)

        # Gradient w.r.t. beta (vectorized)
        grad_beta = -self.X.T @ psi

        # Gradient w.r.t. alpha (optimized aggregation)
        grad_alpha = np.zeros(self.n_entities)

        # Use pre-computed masks for efficiency
        for i in range(self.n_entities):
            mask = self.entity_masks[i]
            grad_alpha[i] = -np.sum(psi[mask])

        # Add subgradient of L1 penalty
        grad_alpha += self.lambda_val * np.sign(alpha)

        return np.concatenate([grad_beta, grad_alpha])

    def optimize(
        self,
        method: str = "L-BFGS-B",
        options: Optional[Dict] = None,
        warm_start: Optional[np.ndarray] = None,
    ) -> Any:
        """
        Optimize penalized QR objective.

        Parameters
        ----------
        method : str
            Optimization method
        options : dict
            Optimizer options
        warm_start : array, optional
            Initial values (for warm starting)

        Returns
        -------
        OptimizeResult
        """
        if options is None:
            options = {"maxiter": 1000, "ftol": 1e-8}

        # Initial values
        if warm_start is not None:
            x0 = warm_start
        else:
            # Smart initialization
            x0 = self._smart_init()

        # Optimize
        result = minimize(
            fun=self.objective, x0=x0, method=method, jac=self.gradient, options=options
        )

        return result

    def _smart_init(self) -> np.ndarray:
        """Smart initialization for faster convergence."""
        # Start with OLS for beta
        beta_ols = np.linalg.lstsq(self.X, self.y, rcond=None)[0]

        # Initialize alpha at zero (will be adjusted by optimizer)
        alpha_init = np.zeros(self.n_entities)

        return np.concatenate([beta_ols, alpha_init])

    def coordinate_descent(
        self, max_iter: int = 100, tol: float = 1e-6, warm_start: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Coordinate descent algorithm for penalized QR.

        More efficient for very sparse solutions.
        """
        # Initialize
        if warm_start is not None:
            beta = warm_start[: self.p].copy()
            alpha = warm_start[self.p :].copy()
        else:
            beta = np.zeros(self.p)
            alpha = np.zeros(self.n_entities)

        converged = False
        for iteration in range(max_iter):
            params_old = np.concatenate([beta, alpha])

            # Update beta coordinates
            for j in range(self.p):
                # Compute partial residuals
                beta_j_old = beta[j]
                beta[j] = 0
                alpha_expanded = alpha[self.obs_to_entity]
                residuals = self.y - self.X @ beta - alpha_expanded

                # Update beta[j]
                gradient_j = -np.sum(self.X[:, j] * self._check_gradient_fast(residuals, self.tau))
                hessian_j = np.sum(self.X[:, j] ** 2) / self.n  # Approximate

                beta[j] = beta_j_old - gradient_j / (hessian_j + 1e-10)

            # Update alpha with soft thresholding
            for i in range(self.n_entities):
                mask = self.entity_masks[i]
                residuals_i = self.y[mask] - self.X[mask] @ beta
                gradient_i = -np.sum(self._check_gradient_fast(residuals_i, self.tau))

                # Soft thresholding for L1 penalty
                alpha[i] = self._soft_threshold(
                    -gradient_i / self.entity_counts[i], self.lambda_val / self.entity_counts[i]
                )

            # Check convergence
            params_new = np.concatenate([beta, alpha])
            if np.linalg.norm(params_new - params_old) < tol:
                converged = True
                break

        return {"beta": beta, "alpha": alpha, "converged": converged, "iterations": iteration + 1}

    def warm_start_path(self, lambda_grid: np.ndarray) -> list:
        """
        Compute solution path using warm starts.

        Significantly faster for multiple lambda values.
        """
        results = []
        params_prev = None

        # Start from largest lambda (most shrinkage)
        for lambda_val in sorted(lambda_grid, reverse=True):
            self.lambda_val = lambda_val

            # Use previous solution as warm start
            if params_prev is not None:
                result = self.optimize(warm_start=params_prev)
            else:
                result = self.optimize()

            results.append(
                {
                    "lambda": lambda_val,
                    "params": result.x,
                    "converged": result.success,
                    "objective": result.fun,
                }
            )

            params_prev = result.x.copy()

        return results


# Additional optimized functions using Numba
@numba.jit(nopython=True, parallel=True)
def compute_check_loss_matrix(residuals: np.ndarray, tau_grid: np.ndarray) -> np.ndarray:
    """
    Compute check loss for multiple quantiles simultaneously.

    Optimized for computing results across many quantiles.
    """
    n = len(residuals)
    m = len(tau_grid)
    losses = np.zeros((n, m))

    for j in numba.prange(m):
        tau = tau_grid[j]
        for i in range(n):
            if residuals[i] < 0:
                losses[i, j] = (tau - 1) * residuals[i]
            else:
                losses[i, j] = tau * residuals[i]

    return losses


@numba.jit(nopython=True, parallel=True)
def compute_gradient_matrix(residuals: np.ndarray, tau_grid: np.ndarray) -> np.ndarray:
    """
    Compute gradients for multiple quantiles simultaneously.
    """
    n = len(residuals)
    m = len(tau_grid)
    gradients = np.zeros((n, m))

    for j in numba.prange(m):
        tau = tau_grid[j]
        for i in range(n):
            gradients[i, j] = tau - (1.0 if residuals[i] < 0 else 0.0)

    return gradients


class PerformanceMonitor:
    """Monitor and profile QR performance."""

    def __init__(self):
        self.timings = {}
        self.memory_usage = {}

    def profile_method(self, method_name: str, data, formula: Optional[str], tau: float) -> Any:
        """Profile a specific QR method."""
        import tracemalloc

        # Memory profiling
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]

        # Time profiling
        start_time = time.time()

        # Run method
        if method_name == "canay":
            from panelbox.models.quantile.canay import CanayTwoStep

            model = CanayTwoStep(data, formula, tau)
        elif method_name == "penalty":
            from panelbox.models.quantile.fixed_effects import FixedEffectsQuantile

            model = FixedEffectsQuantile(data, formula, tau)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        result = model.fit(verbose=False)

        # Collect metrics
        elapsed_time = time.time() - start_time
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_used = (peak_memory - start_memory) / 1024 / 1024  # MB

        self.timings[method_name] = {
            "time": elapsed_time,
            "memory_mb": memory_used,
            "converged": all(r.converged for r in result.results.values()),
        }

        return result

    def compare_implementations(
        self, data, formula: Optional[str], tau: float, methods: list = ["canay", "penalty"]
    ) -> Dict:
        """Compare different implementations."""
        results = {}

        for method in methods:
            print(f"Profiling {method}...")
            results[method] = self.profile_method(method, data, formula, tau)

        return results

    def print_report(self):
        """Print performance comparison report."""
        print("\nPERFORMANCE COMPARISON")
        print("=" * 50)
        print(f"{'Method':<15} {'Time (s)':>10} {'Memory (MB)':>12} {'Status':>10}")
        print("-" * 50)

        for method, metrics in self.timings.items():
            status = "✓" if metrics["converged"] else "✗"
            print(
                f"{method:<15} {metrics['time']:10.2f} "
                f"{metrics['memory_mb']:12.1f} {status:>10}"
            )

        # Find best
        if self.timings:
            fastest = min(self.timings, key=lambda x: self.timings[x]["time"])
            least_memory = min(self.timings, key=lambda x: self.timings[x]["memory_mb"])

            print("\nBest Performance:")
            print(f"  Fastest: {fastest}")
            print(f"  Least Memory: {least_memory}")

            # Speedup ratios
            if len(self.timings) > 1:
                print("\nSpeedup Ratios:")
                baseline = list(self.timings.keys())[0]
                baseline_time = self.timings[baseline]["time"]
                for method in self.timings:
                    if method != baseline:
                        speedup = baseline_time / self.timings[method]["time"]
                        print(f"  {method} vs {baseline}: {speedup:.2f}x")


class AdaptiveOptimizer:
    """
    Adaptive optimizer that chooses the best algorithm based on problem characteristics.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, entity_ids: np.ndarray):
        self.X = X
        self.y = y
        self.entity_ids = entity_ids
        self.n, self.p = X.shape
        self.n_entities = len(np.unique(entity_ids))

        # Compute problem characteristics
        self._analyze_problem()

    def _analyze_problem(self):
        """Analyze problem characteristics to guide algorithm selection."""
        self.problem_size = self.n * self.p
        self.entity_ratio = self.n_entities / self.n
        self.avg_T = self.n / self.n_entities

        # Sparsity of X
        self.X_sparsity = np.sum(self.X == 0) / self.problem_size

        # Condition number (approximate)
        try:
            self.condition_number = np.linalg.cond(self.X.T @ self.X)
        except:
            self.condition_number = np.inf

    def recommend_method(self) -> Tuple[str, Dict[str, Any]]:
        """
        Recommend the best optimization method based on problem characteristics.

        Returns
        -------
        method : str
            Recommended method name
        params : dict
            Recommended parameters
        """
        # Large-scale problems
        if self.problem_size > 1e6:
            if self.X_sparsity > 0.5:
                return "coordinate_descent", {"max_iter": 200, "tol": 1e-5}
            else:
                return "L-BFGS-B", {"maxiter": 500, "ftol": 1e-6}

        # Small T (few time periods)
        if self.avg_T < 10:
            return "penalty", {"lambda_fe": "auto", "cv_folds": 5}

        # Well-conditioned problems
        if self.condition_number < 100:
            return "canay", {"se_adjustment": "two-step"}

        # Default
        return "L-BFGS-B", {"maxiter": 1000, "ftol": 1e-8}

    def print_analysis(self):
        """Print problem analysis."""
        print("\nPROBLEM ANALYSIS")
        print("=" * 50)
        print(f"Problem size (n×p): {self.n} × {self.p} = {self.problem_size:.0f}")
        print(f"Number of entities: {self.n_entities}")
        print(f"Average T: {self.avg_T:.1f}")
        print(f"Sparsity of X: {self.X_sparsity:.2%}")
        print(f"Condition number: {self.condition_number:.2f}")

        method, params = self.recommend_method()
        print(f"\nRecommended method: {method}")
        print(f"Parameters: {params}")
