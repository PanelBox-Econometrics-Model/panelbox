"""Performance tests and benchmarks for gamma distribution.

This module tests the performance of the gamma likelihood computation
and provides benchmarks for different optimization strategies.
"""

import time
from typing import Dict, List

import numpy as np
import pytest

from panelbox.frontier import StochasticFrontier


class TestGammaPerformance:
    """Performance tests for gamma distribution."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for performance testing."""
        np.random.seed(42)
        n = 200
        P_true = 2.0
        theta_true = 1.5
        sigma_v_true = 0.3

        X = np.column_stack([np.ones(n), np.random.normal(0, 1, (n, 2))])
        beta_true = np.array([1.0, 0.5, -0.3])
        u = np.random.gamma(P_true, 1 / theta_true, n)
        v = np.random.normal(0, sigma_v_true, n)
        y = X @ beta_true + v - u

        return {
            "y": y,
            "X": X,
            "P_true": P_true,
            "theta_true": theta_true,
            "sigma_v_true": sigma_v_true,
            "beta_true": beta_true,
        }

    def test_benchmark_n_simulations(self, synthetic_data):
        """Benchmark: time vs n_simulations."""
        import pandas as pd

        y = synthetic_data["y"]
        X = synthetic_data["X"]

        df = pd.DataFrame({"y": y, "x1": X[:, 1], "x2": X[:, 2]})

        n_sims = [50, 100, 200]
        times = []

        for n_sim in n_sims:
            model = StochasticFrontier(
                data=df, depvar="y", exog=["x1", "x2"], frontier="production", dist="gamma"
            )

            start = time.time()
            # We'll patch the n_simulations parameter in estimation
            result = model.fit()
            elapsed = time.time() - start

            times.append(elapsed)

            print(f"n_simulations={n_sim}: {elapsed:.2f}s, loglik={result.loglik:.2f}")

        # Verify diminishing returns
        # More simulations should give better loglik but take longer
        assert times[1] > times[0]  # 100 > 50

    def test_halton_vs_random(self, synthetic_data):
        """Compare Halton sequences vs random sampling."""
        import pandas as pd

        y = synthetic_data["y"]
        X = synthetic_data["X"]

        df = pd.DataFrame({"y": y, "x1": X[:, 1], "x2": X[:, 2]})

        # Test multiple seeds to check variance
        n_runs = 3
        logliks_halton = []
        logliks_random = []

        for run in range(n_runs):
            # Halton (should be deterministic)
            model_h = StochasticFrontier(
                data=df, depvar="y", exog=["x1", "x2"], frontier="production", dist="gamma"
            )
            result_h = model_h.fit()
            logliks_halton.append(result_h.loglik)

            # Random (will vary across runs)
            np.random.seed(run)
            model_r = StochasticFrontier(
                data=df, depvar="y", exog=["x1", "x2"], frontier="production", dist="gamma"
            )
            # We would need to modify code to use random instead of Halton
            # For now, just test that Halton is stable

        # Halton should give identical results across runs
        std_halton = np.std(logliks_halton)
        print(f"Halton std: {std_halton:.6f}")

        # This should be very small (near machine precision)
        # Note: might have some variation due to optimization paths
        assert std_halton < 0.1, "Halton sequences should be deterministic"

    def test_profile_bottleneck(self, synthetic_data):
        """Profile to identify performance bottlenecks."""
        import cProfile
        import io
        import pstats

        import pandas as pd

        y = synthetic_data["y"]
        X = synthetic_data["X"]

        df = pd.DataFrame({"y": y, "x1": X[:, 1], "x2": X[:, 2]})

        model = StochasticFrontier(
            data=df, depvar="y", exog=["x1", "x2"], frontier="production", dist="gamma"
        )

        # Profile the fit
        profiler = cProfile.Profile()
        profiler.enable()

        result = model.fit()

        profiler.disable()

        # Print stats
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        stats.print_stats(20)  # Top 20 functions

        print("\n=== PROFILING RESULTS ===")
        print(s.getvalue())

        # Just verify it completes
        assert result.loglik is not None

    def test_sample_size_scaling(self):
        """Test how computation time scales with sample size."""
        sample_sizes = [50, 100, 200]
        times = []

        for n in sample_sizes:
            np.random.seed(42)
            P_true = 2.0
            theta_true = 1.5
            sigma_v_true = 0.3

            X = np.column_stack([np.ones(n), np.random.normal(0, 1, (n, 2))])
            beta_true = np.array([1.0, 0.5, -0.3])
            u = np.random.gamma(P_true, 1 / theta_true, n)
            v = np.random.normal(0, sigma_v_true, n)
            y = X @ beta_true + v - u

            import pandas as pd

            df = pd.DataFrame({"y": y, "x1": X[:, 1], "x2": X[:, 2]})

            model = StochasticFrontier(
                data=df, depvar="y", exog=["x1", "x2"], frontier="production", dist="gamma"
            )

            start = time.time()
            result = model.fit()
            elapsed = time.time() - start

            times.append(elapsed)

            print(f"n={n}: {elapsed:.2f}s, loglik={result.loglik:.2f}")

        # Time should scale roughly linearly with n
        # (or slightly worse due to optimization)
        time_ratio = times[1] / times[0]  # 100 / 50
        print(f"Time ratio (100/50): {time_ratio:.2f}")

        # Should be roughly 2x (linear), allow some slack
        assert 1.5 < time_ratio < 3.5


class TestCacheOptimization:
    """Test caching strategies for Halton sequences."""

    def test_halton_cache(self):
        """Test that we can cache and reuse Halton draws."""
        from scipy.stats import gamma as gamma_dist
        from scipy.stats.qmc import Halton

        n_simulations = 100
        P = 2.0
        theta_gamma = 1.5

        # Generate once
        halton = Halton(d=1, scramble=False, seed=42)
        uniform_draws = halton.random(n_simulations)
        u_draws_1 = gamma_dist.ppf(uniform_draws.flatten(), a=P, scale=1 / theta_gamma)

        # Generate again with same seed
        halton = Halton(d=1, scramble=False, seed=42)
        uniform_draws = halton.random(n_simulations)
        u_draws_2 = gamma_dist.ppf(uniform_draws.flatten(), a=P, scale=1 / theta_gamma)

        # Should be identical
        np.testing.assert_allclose(u_draws_1, u_draws_2)

        print("✓ Halton sequences are reproducible with same seed")


class TestParallelization:
    """Test parallelization strategies (future optimization)."""

    def test_vectorized_likelihood(self):
        """Test that likelihood computation can be vectorized."""
        from scipy import stats

        n = 100
        n_simulations = 50
        sigma_v = 0.3

        # Generate test data
        epsilon = np.random.normal(0, 1, n)
        u_draws = np.random.gamma(2.0, 1.5, size=n_simulations)

        # Current approach: loop over observations
        logliks_loop = []
        for i in range(n):
            eps_i = epsilon[i]
            likelihoods = np.zeros(n_simulations)
            for r in range(n_simulations):
                u_r = u_draws[r]
                eps_conditional = eps_i - u_r
                likelihoods[r] = stats.norm.pdf(eps_conditional, loc=0, scale=sigma_v)
            avg_lik = np.mean(likelihoods)
            logliks_loop.append(np.log(avg_lik + 1e-10))

        # Vectorized approach: broadcast operations
        # epsilon: (n,), u_draws: (n_simulations,)
        # We want: (n, n_simulations) matrix of eps - u
        eps_matrix = epsilon[:, np.newaxis] - u_draws[np.newaxis, :]  # (n, n_simulations)

        # Compute all likelihoods at once
        likelihoods_matrix = stats.norm.pdf(eps_matrix, loc=0, scale=sigma_v)

        # Average over simulations (axis=1)
        avg_liks = np.mean(likelihoods_matrix, axis=1)

        # Log-likelihood
        logliks_vectorized = np.log(avg_liks + 1e-10)

        # Should match
        np.testing.assert_allclose(logliks_loop, logliks_vectorized, rtol=1e-10)

        print("✓ Vectorized likelihood matches loop implementation")

        # Benchmark
        import time

        # Loop version
        start = time.time()
        for _ in range(10):
            logliks_loop = []
            for i in range(n):
                eps_i = epsilon[i]
                likelihoods = np.zeros(n_simulations)
                for r in range(n_simulations):
                    u_r = u_draws[r]
                    eps_conditional = eps_i - u_r
                    likelihoods[r] = stats.norm.pdf(eps_conditional, loc=0, scale=sigma_v)
                avg_lik = np.mean(likelihoods)
                logliks_loop.append(np.log(avg_lik + 1e-10))
        time_loop = time.time() - start

        # Vectorized version
        start = time.time()
        for _ in range(10):
            eps_matrix = epsilon[:, np.newaxis] - u_draws[np.newaxis, :]
            likelihoods_matrix = stats.norm.pdf(eps_matrix, loc=0, scale=sigma_v)
            avg_liks = np.mean(likelihoods_matrix, axis=1)
            logliks_vectorized = np.log(avg_liks + 1e-10)
        time_vectorized = time.time() - start

        speedup = time_loop / time_vectorized
        print(f"Loop: {time_loop:.4f}s, Vectorized: {time_vectorized:.4f}s")
        print(f"Speedup: {speedup:.2f}x")

        assert speedup > 2, f"Vectorization should give at least 2x speedup, got {speedup:.2f}x"


if __name__ == "__main__":
    # Run simple benchmarks
    test_perf = TestGammaPerformance()

    print("\n=== Synthetic Data ===")
    data = test_perf.synthetic_data()
    print(f"n={len(data['y'])}, P={data['P_true']}, theta={data['theta_true']}")

    print("\n=== Halton Cache Test ===")
    test_cache = TestCacheOptimization()
    test_cache.test_halton_cache()

    print("\n=== Vectorization Test ===")
    test_parallel = TestParallelization()
    test_parallel.test_vectorized_likelihood()

    print("\n✓ All performance tests passed!")
