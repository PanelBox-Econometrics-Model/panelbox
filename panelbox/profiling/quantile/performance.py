"""
Performance profiling suite for quantile regression.

This module provides tools for measuring and optimizing the performance
of quantile regression implementations, including memory usage, computation
time, and scalability analysis.
"""

import json
import time
import tracemalloc
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil


@dataclass
class ProfileResult:
    """Container for profiling results."""

    name: str
    time: float
    memory_delta: float
    peak_memory: float
    cpu_percent: float = 0.0
    n_iterations: int = 0
    convergence: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """
    Performance profiling suite for quantile regression.

    Provides tools for measuring computation time, memory usage, and
    identifying performance bottlenecks in quantile regression implementations.

    Examples
    --------
    >>> profiler = PerformanceProfiler()
    >>> with profiler.profile('optimization'):
    ...     result = model.fit()
    >>> report = profiler.generate_report()
    >>> report.print_summary()
    """

    def __init__(self):
        """Initialize performance profiler."""
        self.metrics = []
        self.current_run = {}
        self.process = psutil.Process()

    @contextmanager
    def profile(self, name: str, **metadata):
        """
        Context manager for profiling code blocks.

        Parameters
        ----------
        name : str
            Name of the code block being profiled
        **metadata
            Additional metadata to store with profile results

        Examples
        --------
        >>> with profiler.profile('optimization', tau=0.5):
        ...     # Code to profile
        ...     result = model.fit()
        """
        # Start monitoring
        tracemalloc.start()

        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self.process.cpu_percent(interval=None)

        try:
            yield
        finally:
            # Stop monitoring
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            end_cpu = self.process.cpu_percent(interval=None)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Create profile result
            result = ProfileResult(
                name=name,
                time=end_time - start_time,
                memory_delta=end_memory - start_memory,
                peak_memory=peak / 1024 / 1024,
                cpu_percent=end_cpu - start_cpu,
                metadata=metadata,
            )

            # Store metrics
            self.metrics.append(result)
            self.current_run[name] = result

    def benchmark_estimators(
        self, data, formula, tau: float = 0.5, methods: Optional[List[str]] = None, n_runs: int = 3
    ) -> "BenchmarkReport":
        """
        Benchmark different QR estimators.

        Parameters
        ----------
        data : DataFrame
            Panel data
        formula : str
            Model formula
        tau : float
            Quantile level
        methods : list, optional
            Methods to benchmark
        n_runs : int
            Number of runs for each method

        Returns
        -------
        BenchmarkReport
            Comprehensive benchmark results
        """
        if methods is None:
            methods = ["pooled", "canay", "fe", "location_scale"]

        results = {}

        for method in methods:
            print(f"Benchmarking {method}...")
            method_results = []

            for run in range(n_runs):
                with self.profile(f"{method}_run{run}", method=method, tau=tau):
                    try:
                        # Import and run the appropriate model
                        if method == "pooled":
                            from ...models.quantile.pooled import PooledQuantile

                            model = PooledQuantile(data, formula, tau)
                        elif method == "canay":
                            from ...models.quantile.canay import CanayTwoStep

                            model = CanayTwoStep(data, formula, tau)
                        elif method == "fe":
                            from ...models.quantile.fixed_effects import FixedEffectsQuantile

                            model = FixedEffectsQuantile(data, formula, tau)
                        elif method == "location_scale":
                            from ...models.quantile.location_scale import LocationScale

                            model = LocationScale(data, formula, tau)
                        else:
                            warnings.warn(f"Unknown method: {method}")
                            continue

                        # Fit model
                        with self.profile(f"{method}_fit_run{run}"):
                            result = model.fit(verbose=False)

                        # Store result
                        method_results.append(self.current_run[f"{method}_run{run}"])

                    except Exception as e:
                        warnings.warn(f"Method {method} failed: {e}")
                        method_results.append(
                            ProfileResult(
                                name=method,
                                time=np.nan,
                                memory_delta=np.nan,
                                peak_memory=np.nan,
                                convergence=False,
                                metadata={"error": str(e)},
                            )
                        )

            results[method] = method_results

        return self._create_benchmark_report(results)

    def profile_scaling(
        self, data_generator: Callable, sizes: List[int], method: str = "pooled", tau: float = 0.5
    ) -> pd.DataFrame:
        """
        Profile performance scaling with data size.

        Parameters
        ----------
        data_generator : callable
            Function that generates data given size
        sizes : list
            Data sizes to test
        method : str
            Method to profile
        tau : float
            Quantile level

        Returns
        -------
        DataFrame
            Scaling results
        """
        results = []

        for size in sizes:
            print(f"Testing size: {size}")

            # Generate data
            data = data_generator(size)

            # Profile
            with self.profile(f"size_{size}", size=size):
                try:
                    # Import model
                    if method == "pooled":
                        from ...models.quantile.pooled import PooledQuantile

                        model = PooledQuantile(data, "y ~ x", tau)
                    else:
                        warnings.warn(f"Unknown method: {method}")
                        continue

                    # Fit
                    result = model.fit(verbose=False)

                    # Get metrics
                    profile_result = self.current_run[f"size_{size}"]

                    results.append(
                        {
                            "size": size,
                            "time": profile_result.time,
                            "memory": profile_result.peak_memory,
                            "time_per_obs": profile_result.time / size,
                            "memory_per_obs": profile_result.peak_memory / size,
                        }
                    )

                except Exception as e:
                    results.append(
                        {
                            "size": size,
                            "time": np.nan,
                            "memory": np.nan,
                            "time_per_obs": np.nan,
                            "memory_per_obs": np.nan,
                            "error": str(e),
                        }
                    )

        return pd.DataFrame(results)

    def identify_bottlenecks(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Identify performance bottlenecks.

        Parameters
        ----------
        detailed : bool
            Include detailed analysis

        Returns
        -------
        dict
            Bottleneck analysis
        """
        if not self.metrics:
            return {"message": "No profiling data available"}

        # Analyze time distribution
        total_time = sum(m.time for m in self.metrics)
        time_distribution = {
            m.name: {
                "time": m.time,
                "percentage": 100 * m.time / total_time,
                "memory": m.peak_memory,
            }
            for m in self.metrics
        }

        # Find bottlenecks
        sorted_by_time = sorted(self.metrics, key=lambda x: x.time, reverse=True)
        sorted_by_memory = sorted(self.metrics, key=lambda x: x.peak_memory, reverse=True)

        bottlenecks = {
            "time": {
                "top_3": [
                    {"name": m.name, "time": m.time, "percentage": 100 * m.time / total_time}
                    for m in sorted_by_time[:3]
                ]
            },
            "memory": {
                "top_3": [
                    {"name": m.name, "peak_memory": m.peak_memory, "memory_delta": m.memory_delta}
                    for m in sorted_by_memory[:3]
                ]
            },
        }

        if detailed:
            # Add detailed metrics
            bottlenecks["detailed"] = {
                "time_distribution": time_distribution,
                "total_time": total_time,
                "total_peak_memory": max(m.peak_memory for m in self.metrics),
                "n_operations": len(self.metrics),
            }

        return bottlenecks

    def _create_benchmark_report(
        self, results: Dict[str, List[ProfileResult]]
    ) -> "BenchmarkReport":
        """Create formatted benchmark report."""
        data = []

        for method, runs in results.items():
            if not runs:
                continue

            # Calculate statistics across runs
            valid_runs = [r for r in runs if not np.isnan(r.time)]

            if valid_runs:
                avg_time = np.mean([r.time for r in valid_runs])
                std_time = np.std([r.time for r in valid_runs])
                avg_memory = np.mean([r.peak_memory for r in valid_runs])
                success_rate = len(valid_runs) / len(runs)
            else:
                avg_time = np.nan
                std_time = np.nan
                avg_memory = np.nan
                success_rate = 0.0

            data.append(
                {
                    "Method": method,
                    "Avg Time (s)": avg_time,
                    "Std Time (s)": std_time,
                    "Avg Memory (MB)": avg_memory,
                    "Success Rate": success_rate,
                    "Runs": len(runs),
                }
            )

        df = pd.DataFrame(data)
        return BenchmarkReport(df, results)

    def generate_report(self) -> "ProfileReport":
        """
        Generate comprehensive profiling report.

        Returns
        -------
        ProfileReport
            Complete profiling analysis
        """
        return ProfileReport(self.metrics)

    def save_results(self, filename: str):
        """
        Save profiling results to file.

        Parameters
        ----------
        filename : str
            Output filename (JSON format)
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": [
                {
                    "name": m.name,
                    "time": m.time,
                    "memory_delta": m.memory_delta,
                    "peak_memory": m.peak_memory,
                    "cpu_percent": m.cpu_percent,
                    "metadata": m.metadata,
                }
                for m in self.metrics
            ],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {filename}")


class BenchmarkReport:
    """Benchmark report with analysis."""

    def __init__(self, df: pd.DataFrame, raw_results: Dict[str, List[ProfileResult]]):
        self.df = df
        self.raw_results = raw_results
        self._analyze()

    def _analyze(self):
        """Analyze benchmark results."""
        success_df = self.df[self.df["Success Rate"] > 0]

        if not success_df.empty:
            self.fastest = success_df.loc[success_df["Avg Time (s)"].idxmin(), "Method"]
            self.most_memory_efficient = success_df.loc[
                success_df["Avg Memory (MB)"].idxmin(), "Method"
            ]
            self.most_reliable = success_df.loc[success_df["Success Rate"].idxmax(), "Method"]
        else:
            self.fastest = None
            self.most_memory_efficient = None
            self.most_reliable = None

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 80)

        print(self.df.to_string(index=False))

        print("\n" + "-" * 80)
        print("SUMMARY")
        print("-" * 80)

        if self.fastest:
            print(f"✓ Fastest Method: {self.fastest}")
        if self.most_memory_efficient:
            print(f"✓ Most Memory Efficient: {self.most_memory_efficient}")
        if self.most_reliable:
            print(f"✓ Most Reliable: {self.most_reliable}")

        # Performance comparison
        if len(self.df) > 1 and self.fastest:
            fastest_time = self.df[self.df["Method"] == self.fastest]["Avg Time (s)"].values[0]
            print("\n" + "-" * 80)
            print("RELATIVE PERFORMANCE")
            print("-" * 80)

            for _, row in self.df.iterrows():
                if not np.isnan(row["Avg Time (s)"]):
                    relative = row["Avg Time (s)"] / fastest_time
                    print(f"{row['Method']:20} {relative:>6.2f}x slower than {self.fastest}")

        print("=" * 80)

    def to_latex(self) -> str:
        """Convert to LaTeX table."""
        latex = self.df.to_latex(
            index=False,
            float_format="%.3f",
            caption="Quantile Regression Performance Benchmarks",
            label="tab:benchmarks",
        )
        return latex

    def plot_comparison(self):
        """Create comparison plots."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Time comparison
        ax = axes[0]
        valid_df = self.df[~self.df["Avg Time (s)"].isna()]
        ax.bar(valid_df["Method"], valid_df["Avg Time (s)"])
        ax.set_xlabel("Method")
        ax.set_ylabel("Average Time (seconds)")
        ax.set_title("Computation Time Comparison")
        ax.tick_params(axis="x", rotation=45)

        # Memory comparison
        ax = axes[1]
        valid_df = self.df[~self.df["Avg Memory (MB)"].isna()]
        ax.bar(valid_df["Method"], valid_df["Avg Memory (MB)"])
        ax.set_xlabel("Method")
        ax.set_ylabel("Average Memory (MB)")
        ax.set_title("Memory Usage Comparison")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        return fig


class ProfileReport:
    """Comprehensive profiling report."""

    def __init__(self, metrics: List[ProfileResult]):
        self.metrics = metrics
        self._analyze()

    def _analyze(self):
        """Analyze profiling data."""
        if not self.metrics:
            self.total_time = 0
            self.total_memory = 0
            self.n_operations = 0
            return

        self.total_time = sum(m.time for m in self.metrics)
        self.total_memory = max(m.peak_memory for m in self.metrics)
        self.n_operations = len(self.metrics)

        # Time breakdown
        self.time_breakdown = {m.name: m.time for m in self.metrics}

        # Memory breakdown
        self.memory_breakdown = {m.name: m.peak_memory for m in self.metrics}

    def print_summary(self):
        """Print profile summary."""
        print("\n" + "=" * 80)
        print("PROFILING SUMMARY")
        print("=" * 80)

        print(f"Total Operations: {self.n_operations}")
        print(f"Total Time: {self.total_time:.3f} seconds")
        print(f"Peak Memory: {self.total_memory:.1f} MB")

        if self.metrics:
            print("\n" + "-" * 80)
            print("TIME BREAKDOWN")
            print("-" * 80)

            sorted_metrics = sorted(self.metrics, key=lambda x: x.time, reverse=True)
            for m in sorted_metrics[:10]:  # Top 10
                pct = 100 * m.time / self.total_time if self.total_time > 0 else 0
                print(f"{m.name:30} {m.time:>8.3f}s ({pct:>5.1f}%)")

            print("\n" + "-" * 80)
            print("MEMORY USAGE")
            print("-" * 80)

            sorted_metrics = sorted(self.metrics, key=lambda x: x.peak_memory, reverse=True)
            for m in sorted_metrics[:5]:  # Top 5
                print(f"{m.name:30} {m.peak_memory:>8.1f} MB")

        print("=" * 80)
