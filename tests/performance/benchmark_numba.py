"""
Benchmark Numba optimizations vs pure Python implementations.

This script measures the speedup achieved by Numba-compiled functions
for the most performance-critical operations in PanelBox.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from panelbox.utils.numba_optimized import (
    NUMBA_AVAILABLE,
    demean_within_1d_numba,
    demean_within_numba,
    fill_iv_instruments_numba,
    get_numba_status,
)


def generate_panel_data(n_entities=100, t_periods=20, n_vars=5, seed=42):
    """Generate synthetic panel data for benchmarking."""
    np.random.seed(seed)

    n_obs = n_entities * t_periods

    entity_ids = np.repeat(np.arange(n_entities), t_periods)
    time_periods = np.tile(np.arange(t_periods), n_entities)

    X = np.random.randn(n_obs, n_vars)
    y = np.random.randn(n_obs)

    return X, y, entity_ids, time_periods


# ============================================================================
# Pure Python Implementations (for comparison)
# ============================================================================


def demean_within_python(X, entity_ids):
    """Pure Python version of within-transformation."""
    X_demeaned = np.copy(X)
    unique_entities = np.unique(entity_ids)

    for entity in unique_entities:
        mask = entity_ids == entity
        entity_mean = X[mask].mean(axis=0)
        X_demeaned[mask] -= entity_mean

    return X_demeaned


def demean_within_1d_python(x, entity_ids):
    """Pure Python version for 1D array."""
    x_demeaned = np.copy(x)
    unique_entities = np.unique(entity_ids)

    for entity in unique_entities:
        mask = entity_ids == entity
        entity_mean = x[mask].mean()
        x_demeaned[mask] -= entity_mean

    return x_demeaned


def fill_iv_instruments_python(Z, var_data, ids, times, min_lag, max_lag, equation="diff"):
    """Pure Python version of IV instrument filling."""
    n_obs = len(ids)
    n_lags = max_lag - min_lag + 1

    for i in range(n_obs):
        current_id = ids[i]
        current_time = times[i]

        for lag_idx in range(n_lags):
            lag = min_lag + lag_idx
            lag_time = current_time - lag

            # Find lagged value
            for j in range(n_obs):
                if ids[j] == current_id and times[j] == lag_time:
                    if equation == "diff":
                        Z[i, lag_idx] = var_data[j]
                    else:
                        # For level equation
                        lag1_time = current_time - lag - 1
                        for k in range(n_obs):
                            if ids[k] == current_id and times[k] == lag1_time:
                                Z[i, lag_idx] = var_data[j] - var_data[k]
                                break
                    break

    return Z


# ============================================================================
# Benchmarking Functions
# ============================================================================


def benchmark_function(func, *args, n_runs=10, warmup=2):
    """
    Benchmark a function with multiple runs.

    Parameters
    ----------
    func : callable
        Function to benchmark
    *args
        Arguments to pass to function
    n_runs : int
        Number of runs for timing
    warmup : int
        Number of warmup runs (to compile Numba)

    Returns
    -------
    dict
        Timing results
    """
    # Warmup runs (important for Numba JIT compilation)
    for _ in range(warmup):
        _ = func(*args)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "median": np.median(times),
    }


def validate_results(result1, result2, tolerance=1e-10):
    """
    Validate that two results are numerically equivalent.

    Returns
    -------
    bool
        True if results match within tolerance
    """
    if isinstance(result1, np.ndarray):
        max_diff = np.max(np.abs(result1 - result2))
    else:
        max_diff = abs(result1 - result2)

    return max_diff < tolerance


# ============================================================================
# Main Benchmarking
# ============================================================================


def main():
    print("=" * 80)
    print("NUMBA OPTIMIZATION BENCHMARKS")
    print("=" * 80)
    print()

    # Check Numba status
    status = get_numba_status()
    print(f"Numba available: {status['available']}")
    if status["available"]:
        print(f"Numba version: {status['version']}")
        print(f"Parallel support: {status['parallel_available']}")
    print()

    if not NUMBA_AVAILABLE:
        print("ERROR: Numba not available. Install with: pip install numba")
        return 1

    # Test sizes
    test_configs = [
        {"name": "Small", "n_entities": 50, "t_periods": 10, "n_vars": 3},
        {"name": "Medium", "n_entities": 100, "t_periods": 20, "n_vars": 5},
        {"name": "Large", "n_entities": 200, "t_periods": 30, "n_vars": 10},
    ]

    results_summary = []

    for config in test_configs:
        print("=" * 80)
        print(f"TEST SIZE: {config['name']}")
        print(f"  N entities: {config['n_entities']}")
        print(f"  T periods: {config['t_periods']}")
        print(f"  Variables: {config['n_vars']}")
        print("=" * 80)
        print()

        # Generate data
        X, y, entity_ids, time_periods = generate_panel_data(
            n_entities=config["n_entities"], t_periods=config["t_periods"], n_vars=config["n_vars"]
        )

        # ----------------------------------------------------------------
        # 1. Demean Within (2D)
        # ----------------------------------------------------------------
        print("-" * 80)
        print("1. Demean Within (2D matrix)")
        print("-" * 80)

        # Python version
        time_python = benchmark_function(demean_within_python, X.copy(), entity_ids)
        result_python = demean_within_python(X.copy(), entity_ids)

        # Numba version
        time_numba = benchmark_function(demean_within_numba, X.copy(), entity_ids)
        result_numba = demean_within_numba(X.copy(), entity_ids)

        # Validate
        match = validate_results(result_python, result_numba)
        speedup = time_python["mean"] / time_numba["mean"]

        print(f"Python:  {time_python['mean']*1000:8.2f} ms (± {time_python['std']*1000:.2f} ms)")
        print(f"Numba:   {time_numba['mean']*1000:8.2f} ms (± {time_numba['std']*1000:.2f} ms)")
        print(f"Speedup: {speedup:8.2f}x")
        print(f"Match:   {'✓' if match else '✗'}")
        print()

        results_summary.append(
            {
                "size": config["name"],
                "operation": "Demean 2D",
                "python_ms": time_python["mean"] * 1000,
                "numba_ms": time_numba["mean"] * 1000,
                "speedup": speedup,
                "match": match,
            }
        )

        # ----------------------------------------------------------------
        # 2. Demean Within (1D)
        # ----------------------------------------------------------------
        print("-" * 80)
        print("2. Demean Within (1D vector)")
        print("-" * 80)

        # Python version
        time_python = benchmark_function(demean_within_1d_python, y.copy(), entity_ids)
        result_python = demean_within_1d_python(y.copy(), entity_ids)

        # Numba version
        time_numba = benchmark_function(demean_within_1d_numba, y.copy(), entity_ids)
        result_numba = demean_within_1d_numba(y.copy(), entity_ids)

        # Validate
        match = validate_results(result_python, result_numba)
        speedup = time_python["mean"] / time_numba["mean"]

        print(f"Python:  {time_python['mean']*1000:8.2f} ms (± {time_python['std']*1000:.2f} ms)")
        print(f"Numba:   {time_numba['mean']*1000:8.2f} ms (± {time_numba['std']*1000:.2f} ms)")
        print(f"Speedup: {speedup:8.2f}x")
        print(f"Match:   {'✓' if match else '✗'}")
        print()

        results_summary.append(
            {
                "size": config["name"],
                "operation": "Demean 1D",
                "python_ms": time_python["mean"] * 1000,
                "numba_ms": time_numba["mean"] * 1000,
                "speedup": speedup,
                "match": match,
            }
        )

        # ----------------------------------------------------------------
        # 3. Fill IV Instruments
        # ----------------------------------------------------------------
        print("-" * 80)
        print("3. Fill IV Instruments (GMM)")
        print("-" * 80)

        min_lag, max_lag = 2, 4
        n_lags = max_lag - min_lag + 1
        n_obs = len(entity_ids)

        Z_python = np.zeros((n_obs, n_lags))
        Z_numba = np.zeros((n_obs, n_lags))

        # Python version
        time_python = benchmark_function(
            fill_iv_instruments_python,
            Z_python.copy(),
            y,
            entity_ids,
            time_periods,
            min_lag,
            max_lag,
            "diff",
        )
        result_python = fill_iv_instruments_python(
            Z_python.copy(), y, entity_ids, time_periods, min_lag, max_lag, "diff"
        )

        # Numba version
        time_numba = benchmark_function(
            fill_iv_instruments_numba,
            Z_numba.copy(),
            y,
            entity_ids,
            time_periods,
            min_lag,
            max_lag,
            "diff",
        )
        result_numba = fill_iv_instruments_numba(
            Z_numba.copy(), y, entity_ids, time_periods, min_lag, max_lag, "diff"
        )

        # Validate
        match = validate_results(result_python, result_numba)
        speedup = time_python["mean"] / time_numba["mean"]

        print(f"Python:  {time_python['mean']*1000:8.2f} ms (± {time_python['std']*1000:.2f} ms)")
        print(f"Numba:   {time_numba['mean']*1000:8.2f} ms (± {time_numba['std']*1000:.2f} ms)")
        print(f"Speedup: {speedup:8.2f}x")
        print(f"Match:   {'✓' if match else '✗'}")
        print()

        results_summary.append(
            {
                "size": config["name"],
                "operation": "Fill IV Instruments",
                "python_ms": time_python["mean"] * 1000,
                "numba_ms": time_numba["mean"] * 1000,
                "speedup": speedup,
                "match": match,
            }
        )

    # ----------------------------------------------------------------
    # Summary Table
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY OF SPEEDUPS")
    print("=" * 80)
    print()

    df = pd.DataFrame(results_summary)

    print(df.to_string(index=False))
    print()

    # Overall statistics
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print()
    print(f"Average speedup: {df['speedup'].mean():.2f}x")
    print(f"Median speedup:  {df['speedup'].median():.2f}x")
    print(f"Min speedup:     {df['speedup'].min():.2f}x")
    print(f"Max speedup:     {df['speedup'].max():.2f}x")
    print()

    all_match = all(df["match"])
    print(f"All validations passed: {'✓ YES' if all_match else '✗ NO'}")
    print()

    if not all_match:
        print("WARNING: Some validations failed!")
        print("Failed operations:")
        for _, row in df[~df["match"]].iterrows():
            print(f"  - {row['size']} {row['operation']}")
        return 1

    print("=" * 80)
    print("✓ ALL BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
