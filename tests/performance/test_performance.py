"""
Performance Tests for PanelBox

Measures execution time of critical operations and compares against targets.

Target: PanelBox should be ≤ 2x slower than Stata/R compiled code
        (This is reasonable for pure Python vs C/Fortran implementations)

Usage:
    python3 test_performance.py
"""

import sys

sys.path.insert(0, "/home/guhaase/projetos/panelbox")

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import panelbox as pb


class PerformanceTester:
    """Test performance of PanelBox operations."""

    def __init__(self):
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.results = []

    def generate_panel_data(
        self, n_entities: int, n_time: int, n_vars: int = 3, seed: int = 42
    ) -> pd.DataFrame:
        """Generate synthetic panel data."""
        np.random.seed(seed)

        data_list = []
        for i in range(n_entities):
            entity_data = {"entity": i, "time": list(range(n_time))}

            # Dependent variable with persistence
            y = np.zeros(n_time)
            y[0] = np.random.randn()
            for t in range(1, n_time):
                y[t] = 0.5 * y[t - 1] + np.random.randn()
            entity_data["y"] = y

            # Independent variables
            for j in range(n_vars):
                entity_data[f"x{j+1}"] = np.random.randn(n_time)

            data_list.append(pd.DataFrame(entity_data))

        return pd.concat(data_list, ignore_index=True)

    def time_function(self, func, n_runs: int = 3) -> Tuple[float, float, bool, str]:
        """
        Time a function over multiple runs.

        Returns:
            mean_time, std_time, success, error_msg
        """
        times = []
        success = True
        error_msg = None

        for i in range(n_runs):
            start = time.time()
            try:
                result = func()
                end = time.time()
                times.append(end - start)
            except Exception as e:
                success = False
                error_msg = str(e)
                break

        if not success:
            return None, None, False, error_msg

        mean_time = np.mean(times)
        std_time = np.std(times)

        return mean_time, std_time, True, None

    def test_pooled_ols_scaling(self):
        """Test Pooled OLS performance at different scales."""
        print("\n" + "=" * 80)
        print("PERFORMANCE TEST: Pooled OLS Scaling")
        print("=" * 80)

        scales = [
            (100, 20, "Small"),
            (500, 20, "Medium"),
            (1000, 50, "Large"),
            (2000, 100, "Very Large"),
        ]

        results = []

        for n, t, label in scales:
            print(f"\n{label}: N={n}, T={t}")
            data = self.generate_panel_data(n, t)

            def run_model():
                model = pb.PooledOLS("y ~ x1 + x2 + x3", data, "entity", "time")
                return model.fit()

            mean_time, std_time, success, error = self.time_function(run_model, n_runs=3)

            if success:
                print(f"  Time: {mean_time:.4f} ± {std_time:.4f} seconds")
                results.append(
                    {
                        "model": "Pooled OLS",
                        "scale": label,
                        "n_entities": n,
                        "n_time": t,
                        "mean_time": mean_time,
                        "std_time": std_time,
                        "success": True,
                    }
                )
            else:
                print(f"  Error: {error}")
                results.append(
                    {
                        "model": "Pooled OLS",
                        "scale": label,
                        "n_entities": n,
                        "n_time": t,
                        "error": error,
                        "success": False,
                    }
                )

        return results

    def test_fixed_effects_scaling(self):
        """Test Fixed Effects performance at different scales."""
        print("\n" + "=" * 80)
        print("PERFORMANCE TEST: Fixed Effects Scaling")
        print("=" * 80)

        scales = [(100, 20, "Small"), (500, 20, "Medium"), (1000, 50, "Large")]

        results = []

        for n, t, label in scales:
            print(f"\n{label}: N={n}, T={t}")
            data = self.generate_panel_data(n, t)

            def run_model():
                model = pb.FixedEffects("y ~ x1 + x2 + x3", data, "entity", "time")
                return model.fit()

            mean_time, std_time, success, error = self.time_function(run_model, n_runs=3)

            if success:
                print(f"  Time: {mean_time:.4f} ± {std_time:.4f} seconds")
                results.append(
                    {
                        "model": "Fixed Effects",
                        "scale": label,
                        "n_entities": n,
                        "n_time": t,
                        "mean_time": mean_time,
                        "std_time": std_time,
                        "success": True,
                    }
                )
            else:
                print(f"  Error: {error}")
                results.append(
                    {
                        "model": "Fixed Effects",
                        "scale": label,
                        "n_entities": n,
                        "n_time": t,
                        "error": error,
                        "success": False,
                    }
                )

        return results

    def test_gmm_performance(self):
        """Test GMM performance (most intensive operation)."""
        print("\n" + "=" * 80)
        print("PERFORMANCE TEST: GMM Estimators")
        print("=" * 80)

        # Smaller scales for GMM (more intensive)
        scales = [(50, 10, "Small"), (100, 20, "Medium"), (200, 30, "Large")]

        results = []

        for n, t, label in scales:
            print(f"\n{label}: N={n}, T={t}")
            data = self.generate_panel_data(n, t)

            # Test Difference GMM
            print(f"  Difference GMM...")

            def run_diff_gmm():
                model = pb.DifferenceGMM(
                    data=data,
                    dep_var="y",
                    lags=1,
                    id_var="entity",
                    time_var="time",
                    exog_vars=["x1", "x2"],
                    collapse=True,
                    two_step=True,
                )
                return model.fit()

            mean_time, std_time, success, error = self.time_function(run_diff_gmm, n_runs=2)

            if success:
                print(f"    Time: {mean_time:.4f} ± {std_time:.4f} seconds")
                results.append(
                    {
                        "model": "Difference GMM",
                        "scale": label,
                        "n_entities": n,
                        "n_time": t,
                        "mean_time": mean_time,
                        "std_time": std_time,
                        "success": True,
                    }
                )
            else:
                print(f"    Error: {error}")
                results.append(
                    {
                        "model": "Difference GMM",
                        "scale": label,
                        "n_entities": n,
                        "n_time": t,
                        "error": error,
                        "success": False,
                    }
                )

            # Test System GMM
            print(f"  System GMM...")

            def run_sys_gmm():
                model = pb.SystemGMM(
                    data=data,
                    dep_var="y",
                    lags=1,
                    id_var="entity",
                    time_var="time",
                    exog_vars=["x1", "x2"],
                    collapse=True,
                    two_step=True,
                )
                return model.fit()

            mean_time, std_time, success, error = self.time_function(run_sys_gmm, n_runs=2)

            if success:
                print(f"    Time: {mean_time:.4f} ± {std_time:.4f} seconds")
                results.append(
                    {
                        "model": "System GMM",
                        "scale": label,
                        "n_entities": n,
                        "n_time": t,
                        "mean_time": mean_time,
                        "std_time": std_time,
                        "success": True,
                    }
                )
            else:
                print(f"    Error: {error}")
                results.append(
                    {
                        "model": "System GMM",
                        "scale": label,
                        "n_entities": n,
                        "n_time": t,
                        "error": error,
                        "success": False,
                    }
                )

        return results

    def run_all_tests(self):
        """Run all performance tests."""
        print("\n" + "=" * 80)
        print("PANELBOX PERFORMANCE TEST SUITE")
        print("=" * 80)
        print(f"\nTimestamp: {datetime.now().isoformat()}")

        all_results = []

        # Test 1: Pooled OLS scaling
        pooled_results = self.test_pooled_ols_scaling()
        all_results.extend(pooled_results)

        # Test 2: Fixed Effects scaling
        fe_results = self.test_fixed_effects_scaling()
        all_results.extend(fe_results)

        # Test 3: GMM performance
        gmm_results = self.test_gmm_performance()
        all_results.extend(gmm_results)

        self.results = all_results
        return all_results

    def save_results(self):
        """Save results to JSON."""
        output_file = (
            self.results_dir
            / f"performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        output = {
            "timestamp": datetime.now().isoformat(),
            "platform": sys.platform,
            "python_version": sys.version,
            "results": self.results,
        }

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

        # Also save latest
        latest_file = self.results_dir / "performance_results_latest.json"
        with open(latest_file, "w") as f:
            json.dump(output, f, indent=2)

    def generate_summary(self):
        """Generate summary of performance results."""
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)

        # Group by model
        by_model = {}
        for result in self.results:
            if not result.get("success"):
                continue

            model = result["model"]
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(result)

        # Print summary
        for model, results in by_model.items():
            print(f"\n{model}:")
            print(f"  {'Scale':<15} {'N':>6} {'T':>6} {'Time (s)':>10}")
            print("  " + "-" * 40)

            for r in results:
                scale = r["scale"]
                n = r["n_entities"]
                t = r["n_time"]
                time_val = r["mean_time"]
                print(f"  {scale:<15} {n:>6} {t:>6} {time_val:>10.4f}")

        # Check against targets
        print("\n" + "=" * 80)
        print("PERFORMANCE TARGETS")
        print("=" * 80)

        print("\nTarget: ≤ 2x slower than Stata/R")
        print("\nNOTE: Actual comparison requires running equivalent Stata/R code.")
        print("      These are absolute timings for Python implementation.")

        # Identify slow operations (> 5 seconds)
        slow_ops = [r for r in self.results if r.get("success") and r.get("mean_time", 0) > 5.0]

        if slow_ops:
            print("\n⚠️  Operations taking > 5 seconds:")
            for op in slow_ops:
                print(f"  - {op['model']} ({op['scale']}): {op['mean_time']:.2f}s")
            print("\n  → Consider optimization with Numba or Cython")
        else:
            print("\n✓ All operations complete in reasonable time (< 5s)")


def main():
    """Main entry point."""
    tester = PerformanceTester()

    # Run all tests
    tester.run_all_tests()

    # Save results
    tester.save_results()

    # Generate summary
    tester.generate_summary()

    print("\n" + "=" * 80)
    print("Performance testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
