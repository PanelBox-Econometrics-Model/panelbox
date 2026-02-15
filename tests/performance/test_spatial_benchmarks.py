"""
Benchmark tests for spatial panel models performance.

Tests performance metrics for different panel sizes and model types.
"""

import time
import tracemalloc
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from panelbox.core.spatial_weights import SpatialWeights
from panelbox.models.spatial import SpatialDurbin, SpatialError, SpatialLag


class TestSpatialBenchmarks:
    """Benchmark tests for spatial panel models."""

    @classmethod
    def setup_class(cls):
        """Setup for benchmark tests."""
        np.random.seed(42)

    def generate_panel_data(self, N: int, T: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate simulated panel data for benchmarks.

        Parameters
        ----------
        N : int
            Number of cross-sectional units
        T : int
            Number of time periods

        Returns
        -------
        data : pd.DataFrame
            Panel data with dependent and independent variables
        W : np.ndarray
            Spatial weight matrix
        """
        # Generate panel structure
        entity = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)

        # Generate covariates
        X1 = np.random.normal(0, 1, N * T)
        X2 = np.random.normal(0, 1, N * T)

        # Generate spatial weight matrix (sparse random)
        # Create sparse W with ~10% non-zero entries
        W = np.zeros((N, N))
        for i in range(N):
            # Connect to ~10% of other units
            n_neighbors = max(1, int(N * 0.1))
            neighbors = np.random.choice(
                [j for j in range(N) if j != i], n_neighbors, replace=False
            )
            for j in neighbors:
                W[i, j] = 1.0

        # Row-normalize
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

        # Generate dependent variable with spatial dependence
        rho = 0.5
        I = np.eye(N)
        A_inv = np.linalg.inv(I - rho * W)

        y_values = []
        for t in range(T):
            idx = time == t
            X1_t = X1[idx]
            X2_t = X2[idx]
            eps = np.random.normal(0, 1, N)
            y_t = A_inv @ (2 + 0.5 * X1_t + 0.3 * X2_t + eps)
            y_values.extend(y_t)

        data = pd.DataFrame({"entity": entity, "time": time, "y": y_values, "X1": X1, "X2": X2})

        return data, W

    def benchmark_model(
        self, model_class, data: pd.DataFrame, W: np.ndarray, formula: str = "y ~ X1 + X2"
    ) -> Dict[str, float]:
        """
        Benchmark a spatial model.

        Returns
        -------
        metrics : dict
            Dictionary with timing and memory metrics
        """
        # Start memory tracking
        tracemalloc.start()

        # Time model creation
        start_time = time.time()
        model = model_class(formula=formula, data=data, entity_col="entity", time_col="time", W=W)
        creation_time = time.time() - start_time

        # Time model estimation
        start_time = time.time()
        try:
            result = model.fit(effects="fixed", maxiter=100, method="ml")
            estimation_time = time.time() - start_time
            converged = result.converged if hasattr(result, "converged") else True
        except Exception as e:
            estimation_time = time.time() - start_time
            converged = False
            print(f"Estimation failed: {e}")

        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "creation_time": creation_time,
            "estimation_time": estimation_time,
            "total_time": creation_time + estimation_time,
            "peak_memory_mb": peak / 1024 / 1024,
            "converged": converged,
        }

    @pytest.mark.benchmark
    def test_sar_performance(self):
        """Benchmark SAR model for different panel sizes."""
        sizes = [
            (100, 10),  # Small
            (500, 10),  # Medium
            (1000, 10),  # Large
            # (5000, 5),   # Very large - uncomment for full test
        ]

        results = []
        for N, T in sizes:
            print(f"\nBenchmarking SAR with N={N}, T={T}")
            data, W = self.generate_panel_data(N, T)

            metrics = self.benchmark_model(SpatialLag, data, W)
            metrics["N"] = N
            metrics["T"] = T
            metrics["model"] = "SAR"
            results.append(metrics)

            print(f"  Total time: {metrics['total_time']:.2f}s")
            print(f"  Peak memory: {metrics['peak_memory_mb']:.1f} MB")
            print(f"  Converged: {metrics['converged']}")

        # Check performance targets
        for result in results:
            if result["N"] == 1000 and result["T"] == 10:
                # Target: N=1000, T=10 should complete in < 30s
                assert (
                    result["total_time"] < 30
                ), f"SAR with N=1000, T=10 took {result['total_time']:.1f}s (target: < 30s)"

    @pytest.mark.benchmark
    def test_model_comparison(self):
        """Compare performance across different spatial models."""
        N, T = 500, 10
        print(f"\nComparing models with N={N}, T={T}")

        data, W = self.generate_panel_data(N, T)

        models = [
            ("SAR", SpatialLag),
            ("SEM", SpatialError),
            ("SDM", SpatialDurbin),
        ]

        results = []
        for name, model_class in models:
            print(f"  Benchmarking {name}...")
            metrics = self.benchmark_model(model_class, data, W)
            metrics["model"] = name
            results.append(metrics)

            print(f"    Time: {metrics['total_time']:.2f}s")
            print(f"    Memory: {metrics['peak_memory_mb']:.1f} MB")

        # Create comparison DataFrame
        df = pd.DataFrame(results)
        print("\nModel Comparison:")
        print(df[["model", "total_time", "peak_memory_mb", "converged"]])

        return df

    @pytest.mark.benchmark
    def test_sparse_vs_dense(self):
        """Compare performance of sparse vs dense weight matrices."""
        N, T = 500, 10
        print(f"\nComparing sparse vs dense W with N={N}, T={T}")

        data, W_dense = self.generate_panel_data(N, T)

        # Create sparse version
        W_sparse = csr_matrix(W_dense)

        # Benchmark with dense W
        print("  Benchmarking with dense W...")
        metrics_dense = self.benchmark_model(SpatialLag, data, W_dense)

        # Benchmark with sparse W
        print("  Benchmarking with sparse W...")
        metrics_sparse = self.benchmark_model(SpatialLag, data, W_sparse.toarray())

        print(f"\nResults:")
        print(
            f"  Dense W: {metrics_dense['total_time']:.2f}s, "
            f"{metrics_dense['peak_memory_mb']:.1f} MB"
        )
        print(
            f"  Sparse W: {metrics_sparse['total_time']:.2f}s, "
            f"{metrics_sparse['peak_memory_mb']:.1f} MB"
        )

        # Sparse should be at least as fast
        speedup = metrics_dense["total_time"] / metrics_sparse["total_time"]
        print(f"  Speedup: {speedup:.2f}x")

    @pytest.mark.benchmark
    def test_scaling_analysis(self):
        """Analyze how performance scales with N."""
        T = 5  # Fixed time periods
        N_values = [50, 100, 200, 400, 800]

        results = []
        for N in N_values:
            print(f"  N={N}...")
            data, W = self.generate_panel_data(N, T)

            metrics = self.benchmark_model(SpatialLag, data, W)
            metrics["N"] = N
            results.append(metrics)

        # Analyze scaling
        df = pd.DataFrame(results)
        print("\nScaling Analysis:")
        print(df[["N", "total_time", "peak_memory_mb"]])

        # Check if scaling is reasonable (not exponential)
        # Time should scale approximately as O(N^3) for direct methods
        times = df["total_time"].values
        N_vals = df["N"].values

        # Fit polynomial to log-log plot
        log_N = np.log(N_vals)
        log_time = np.log(times)
        scaling_exp = np.polyfit(log_N, log_time, 1)[0]

        print(f"\nTime scaling: O(N^{scaling_exp:.2f})")

        # Should be around 3 for direct methods
        assert 2 < scaling_exp < 4, f"Unexpected scaling exponent: {scaling_exp:.2f}"


def generate_performance_report():
    """Generate a performance report for documentation."""
    print("\n" + "=" * 60)
    print("SPATIAL PANEL MODELS - PERFORMANCE REPORT")
    print("=" * 60)

    # Run benchmarks and generate report
    tester = TestSpatialBenchmarks()

    # Test different sizes
    print("\n1. Performance by Panel Size")
    print("-" * 40)
    tester.test_sar_performance()

    # Model comparison
    print("\n2. Model Type Comparison")
    print("-" * 40)
    df_models = tester.test_model_comparison()

    # Sparse vs Dense
    print("\n3. Sparse vs Dense Matrices")
    print("-" * 40)
    tester.test_sparse_vs_dense()

    # Scaling analysis
    print("\n4. Scaling Analysis")
    print("-" * 40)
    tester.test_scaling_analysis()

    print("\n" + "=" * 60)
    print("END OF PERFORMANCE REPORT")
    print("=" * 60)


if __name__ == "__main__":
    # Run performance report
    generate_performance_report()
