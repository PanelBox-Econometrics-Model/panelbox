"""
Performance Profiling for PanelBox

This module profiles critical operations to identify bottlenecks.

Usage:
    python3 profiling.py [--model MODEL] [--n N] [--t T]

Models: pooled, fe, re, diff_gmm, sys_gmm
"""

import sys

sys.path.insert(0, "/home/guhaase/projetos/panelbox")

import cProfile
import io
import pstats
import time
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

import panelbox as pb


class PerformanceProfiler:
    """Profile performance of PanelBox operations."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(__file__).parent / "profiles"
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}

    def generate_panel_data(self, n_entities: int, n_time: int, n_vars: int = 3) -> pd.DataFrame:
        """Generate synthetic panel data for testing."""
        np.random.seed(42)

        data_list = []
        for i in range(n_entities):
            entity_data = {"entity": i, "time": list(range(n_time))}

            # Dependent variable (with some persistence)
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

    def profile_function(self, func: Callable, name: str, *args, **kwargs) -> Dict[str, Any]:
        """Profile a function and return statistics."""
        print(f"\n{'=' * 80}")
        print(f"Profiling: {name}")
        print(f"{'=' * 80}")

        # Create profiler
        profiler = cProfile.Profile()

        # Time execution
        start_time = time.time()
        profiler.enable()

        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)

        profiler.disable()
        end_time = time.time()

        execution_time = end_time - start_time

        # Get statistics
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats("cumulative")

        print(f"\nExecution time: {execution_time:.4f} seconds")
        print(f"Success: {success}")
        if error:
            print(f"Error: {error}")

        # Print top 20 functions
        print("\nTop 20 functions by cumulative time:")
        print("-" * 80)
        stats.print_stats(20)

        # Save full profile
        profile_file = self.output_dir / f"{name.replace(' ', '_')}.prof"
        stats.dump_stats(str(profile_file))
        print(f"\nFull profile saved to: {profile_file}")

        # Save text report
        text_file = self.output_dir / f"{name.replace(' ', '_')}.txt"
        with open(text_file, "w") as f:
            f.write(f"Profile: {name}\n")
            f.write(f"Execution time: {execution_time:.4f} seconds\n")
            f.write(f"Success: {success}\n")
            if error:
                f.write(f"Error: {error}\n")
            f.write("\n" + "=" * 80 + "\n")
            stats.stream = f
            stats.print_stats()

        return {
            "name": name,
            "execution_time": execution_time,
            "success": success,
            "error": error,
            "profile_file": str(profile_file),
            "text_file": str(text_file),
        }

    def profile_pooled_ols(self, n_entities: int = 100, n_time: int = 20):
        """Profile Pooled OLS estimation."""
        data = self.generate_panel_data(n_entities, n_time)

        def run_pooled():
            model = pb.PooledOLS("y ~ x1 + x2 + x3", data, "entity", "time")
            return model.fit()

        return self.profile_function(run_pooled, f"PooledOLS_N{n_entities}_T{n_time}")

    def profile_fixed_effects(self, n_entities: int = 100, n_time: int = 20):
        """Profile Fixed Effects estimation."""
        data = self.generate_panel_data(n_entities, n_time)

        def run_fe():
            model = pb.FixedEffects("y ~ x1 + x2 + x3", data, "entity", "time")
            return model.fit()

        return self.profile_function(run_fe, f"FixedEffects_N{n_entities}_T{n_time}")

    def profile_random_effects(self, n_entities: int = 100, n_time: int = 20):
        """Profile Random Effects estimation."""
        data = self.generate_panel_data(n_entities, n_time)

        def run_re():
            model = pb.RandomEffects("y ~ x1 + x2 + x3", data, "entity", "time")
            return model.fit()

        return self.profile_function(run_re, f"RandomEffects_N{n_entities}_T{n_time}")

    def profile_difference_gmm(self, n_entities: int = 50, n_time: int = 10):
        """Profile Difference GMM estimation."""
        data = self.generate_panel_data(n_entities, n_time)

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

        return self.profile_function(run_diff_gmm, f"DifferenceGMM_N{n_entities}_T{n_time}")

    def profile_system_gmm(self, n_entities: int = 50, n_time: int = 10):
        """Profile System GMM estimation."""
        data = self.generate_panel_data(n_entities, n_time)

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

        return self.profile_function(run_sys_gmm, f"SystemGMM_N{n_entities}_T{n_time}")

    def profile_all_models(self):
        """Profile all models with default parameters."""
        print("\n" + "=" * 80)
        print("PROFILING ALL MODELS")
        print("=" * 80)

        models = [
            ("Pooled OLS", lambda: self.profile_pooled_ols(100, 20)),
            ("Fixed Effects", lambda: self.profile_fixed_effects(100, 20)),
            ("Random Effects", lambda: self.profile_random_effects(100, 20)),
            ("Difference GMM", lambda: self.profile_difference_gmm(50, 10)),
            ("System GMM", lambda: self.profile_system_gmm(50, 10)),
        ]

        results = []
        for name, func in models:
            try:
                result = func()
                results.append(result)
            except Exception as e:
                print(f"\nError profiling {name}: {e}")
                results.append(
                    {"name": name, "execution_time": None, "success": False, "error": str(e)}
                )

        self.results = results
        return results

    def generate_summary_report(self):
        """Generate summary report of profiling results."""
        print("\n" + "=" * 80)
        print("PROFILING SUMMARY")
        print("=" * 80)

        print(f"\n{'Model':<20} {'Time (s)':>12} {'Status':>10}")
        print("-" * 45)

        for result in self.results:
            name = result["name"]
            time_str = f"{result['execution_time']:.4f}" if result["execution_time"] else "N/A"
            status = "✓" if result["success"] else "✗"
            print(f"{name:<20} {time_str:>12} {status:>10}")

        # Save summary
        summary_file = self.output_dir / "PROFILING_SUMMARY.txt"
        with open(summary_file, "w") as f:
            f.write("PROFILING SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"{'Model':<20} {'Time (s)':>12} {'Status':>10}\n")
            f.write("-" * 45 + "\n")

            for result in self.results:
                name = result["name"]
                time_str = f"{result['execution_time']:.4f}" if result["execution_time"] else "N/A"
                status = "✓" if result["success"] else "✗"
                f.write(f"{name:<20} {time_str:>12} {status:>10}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("\nProfile files saved in: {}\n".format(self.output_dir))

        print(f"\nSummary saved to: {summary_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Profile PanelBox performance")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "pooled", "fe", "re", "diff_gmm", "sys_gmm"],
        help="Model to profile",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of entities")
    parser.add_argument("--t", type=int, default=20, help="Number of time periods")

    args = parser.parse_args()

    profiler = PerformanceProfiler()

    if args.model == "all":
        profiler.profile_all_models()
        profiler.generate_summary_report()
    elif args.model == "pooled":
        profiler.profile_pooled_ols(args.n, args.t)
    elif args.model == "fe":
        profiler.profile_fixed_effects(args.n, args.t)
    elif args.model == "re":
        profiler.profile_random_effects(args.n, args.t)
    elif args.model == "diff_gmm":
        profiler.profile_difference_gmm(args.n, args.t)
    elif args.model == "sys_gmm":
        profiler.profile_system_gmm(args.n, args.t)


if __name__ == "__main__":
    main()
