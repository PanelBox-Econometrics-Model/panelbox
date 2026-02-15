"""
Comprehensive performance benchmarks for PanelBox advanced methods.

This script benchmarks all advanced methods implemented in Phases 1-5:
- CUE-GMM vs Two-Step GMM
- Bias-Corrected GMM
- Panel Heckman MLE
- Westerlund cointegration tests
- Hadri & Breitung unit root tests
- Multinomial FE
- PPML FE

Results are saved to /home/guhaase/projetos/panelbox/docs/benchmarks/
"""

import time
import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    method: str
    params: Dict
    time_seconds: float
    converged: bool
    notes: str = ""


class PerformanceBenchmarker:
    """Main benchmarking class."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def run_all_benchmarks(self):
        """Run all benchmarks."""
        print("=" * 80)
        print("PANELBOX ADVANCED METHODS - PERFORMANCE BENCHMARKS")
        print("=" * 80)
        print()

        self.benchmark_cue_gmm()
        self.benchmark_bias_corrected_gmm()
        self.benchmark_panel_heckman()
        self.benchmark_westerlund()
        self.benchmark_unit_root()
        self.benchmark_multinomial()
        self.benchmark_ppml()

        self.save_results()
        self.print_summary()

    def benchmark_cue_gmm(self):
        """Benchmark CUE-GMM vs Two-Step GMM."""
        print("\n" + "=" * 80)
        print("1. CUE-GMM vs Two-Step GMM")
        print("=" * 80)

        from panelbox.gmm import ContinuousUpdatedGMM, SystemGMM

        configs = [
            {"N": 100, "T": 10, "n_moments": 10},
            {"N": 500, "T": 10, "n_moments": 20},
            {"N": 1000, "T": 10, "n_moments": 20},
        ]

        for config in configs:
            N, T, n_moments = config["N"], config["T"], config["n_moments"]
            print(f"\n  Config: N={N}, T={T}, moments={n_moments}")

            # Generate data
            np.random.seed(42)
            y = np.random.randn(N * T)
            X = np.random.randn(N * T, 2)
            Z = np.random.randn(N * T, n_moments)
            entity_id = np.repeat(np.arange(N), T)
            time_id = np.tile(np.arange(T), N)

            data = pd.DataFrame(
                {"y": y, "x1": X[:, 0], "x2": X[:, 1], "entity": entity_id, "time": time_id}
            )
            for i in range(n_moments):
                data[f"z{i}"] = Z[:, i]

            # Benchmark Two-Step GMM
            try:
                start = time.time()
                model_2step = SystemGMM(
                    data=data, dependent="y", lags=1, entity_var="entity", time_var="time"
                )
                result_2step = model_2step.fit()
                time_2step = time.time() - start
                converged_2step = True
                print(f"    Two-Step GMM: {time_2step:.3f}s")

                self.results.append(
                    BenchmarkResult(
                        method="Two-Step GMM",
                        params=config,
                        time_seconds=time_2step,
                        converged=converged_2step,
                    )
                )
            except Exception as e:
                print(f"    Two-Step GMM: FAILED ({str(e)[:50]})")
                self.results.append(
                    BenchmarkResult(
                        method="Two-Step GMM",
                        params=config,
                        time_seconds=0.0,
                        converged=False,
                        notes=str(e)[:100],
                    )
                )

            # Benchmark CUE-GMM
            try:
                start = time.time()
                model_cue = ContinuousUpdatedGMM(
                    data=data, dependent="y", lags=1, entity_var="entity", time_var="time"
                )
                result_cue = model_cue.fit(max_iter=100)
                time_cue = time.time() - start
                converged_cue = True
                print(f"    CUE-GMM: {time_cue:.3f}s (ratio: {time_cue/time_2step:.2f}x)")

                self.results.append(
                    BenchmarkResult(
                        method="CUE-GMM",
                        params=config,
                        time_seconds=time_cue,
                        converged=converged_cue,
                    )
                )
            except Exception as e:
                print(f"    CUE-GMM: FAILED ({str(e)[:50]})")
                self.results.append(
                    BenchmarkResult(
                        method="CUE-GMM",
                        params=config,
                        time_seconds=0.0,
                        converged=False,
                        notes=str(e)[:100],
                    )
                )

    def benchmark_bias_corrected_gmm(self):
        """Benchmark Bias-Corrected GMM."""
        print("\n" + "=" * 80)
        print("2. Bias-Corrected GMM")
        print("=" * 80)

        from panelbox.gmm import BiasCorrectedGMM

        configs = [
            {"N": 50, "T": 10},
            {"N": 100, "T": 20},
            {"N": 500, "T": 50},
        ]

        for config in configs:
            N, T = config["N"], config["T"]
            print(f"\n  Config: N={N}, T={T}")

            # Generate data
            np.random.seed(42)
            y = np.random.randn(N * T)
            X = np.random.randn(N * T, 2)
            entity_id = np.repeat(np.arange(N), T)
            time_id = np.tile(np.arange(T), N)

            data = pd.DataFrame(
                {"y": y, "x1": X[:, 0], "x2": X[:, 1], "entity": entity_id, "time": time_id}
            )

            try:
                start = time.time()
                model = BiasCorrectedGMM(
                    data=data, dependent="y", lags=1, entity_var="entity", time_var="time"
                )
                result = model.fit()
                elapsed = time.time() - start
                print(f"    Bias-Corrected GMM: {elapsed:.3f}s")

                self.results.append(
                    BenchmarkResult(
                        method="Bias-Corrected GMM",
                        params=config,
                        time_seconds=elapsed,
                        converged=True,
                    )
                )
            except Exception as e:
                print(f"    Bias-Corrected GMM: FAILED ({str(e)[:50]})")
                self.results.append(
                    BenchmarkResult(
                        method="Bias-Corrected GMM",
                        params=config,
                        time_seconds=0.0,
                        converged=False,
                        notes=str(e)[:100],
                    )
                )

    def benchmark_panel_heckman(self):
        """Benchmark Panel Heckman MLE."""
        print("\n" + "=" * 80)
        print("3. Panel Heckman MLE")
        print("=" * 80)

        from panelbox.models.selection import PanelHeckman

        configs = [
            {"N": 100, "T": 10, "quad_points": 10},
            {"N": 200, "T": 10, "quad_points": 15},
        ]

        for config in configs:
            N, T, quad_points = config["N"], config["T"], config["quad_points"]
            print(f"\n  Config: N={N}, T={T}, quadrature_points={quad_points}")

            # Generate data
            np.random.seed(42)
            entity_id = np.repeat(np.arange(N), T)
            time_id = np.tile(np.arange(T), N)

            X = np.random.randn(N * T, 2)
            Z = np.random.randn(N * T, 2)

            # Selection equation
            selection = (Z[:, 0] + Z[:, 1] + np.random.randn(N * T)) > 0

            # Outcome equation (only observed if selected)
            y_star = X[:, 0] + X[:, 1] + np.random.randn(N * T)
            y = np.where(selection, y_star, np.nan)

            data = pd.DataFrame(
                {
                    "y": y,
                    "x1": X[:, 0],
                    "x2": X[:, 1],
                    "z1": Z[:, 0],
                    "z2": Z[:, 1],
                    "entity": entity_id,
                    "time": time_id,
                }
            )

            # Two-step (baseline)
            try:
                start = time.time()
                model_2step = PanelHeckman(
                    data=data,
                    outcome_formula="y ~ x1 + x2",
                    selection_formula="z1 + z2",
                    entity_var="entity",
                    time_var="time",
                    method="two-step",
                )
                result_2step = model_2step.fit()
                time_2step = time.time() - start
                print(f"    Two-Step: {time_2step:.3f}s")

                self.results.append(
                    BenchmarkResult(
                        method="Heckman Two-Step",
                        params=config,
                        time_seconds=time_2step,
                        converged=True,
                    )
                )
            except Exception as e:
                print(f"    Two-Step: FAILED ({str(e)[:50]})")
                self.results.append(
                    BenchmarkResult(
                        method="Heckman Two-Step",
                        params=config,
                        time_seconds=0.0,
                        converged=False,
                        notes=str(e)[:100],
                    )
                )

            # MLE (slower but more efficient)
            try:
                start = time.time()
                model_mle = PanelHeckman(
                    data=data,
                    outcome_formula="y ~ x1 + x2",
                    selection_formula="z1 + z2",
                    entity_var="entity",
                    time_var="time",
                    method="mle",
                    quadrature_points=quad_points,
                )
                result_mle = model_mle.fit()
                time_mle = time.time() - start
                print(f"    MLE: {time_mle:.3f}s (ratio: {time_mle/time_2step:.2f}x)")

                self.results.append(
                    BenchmarkResult(
                        method="Heckman MLE", params=config, time_seconds=time_mle, converged=True
                    )
                )
            except Exception as e:
                print(f"    MLE: FAILED ({str(e)[:50]})")
                self.results.append(
                    BenchmarkResult(
                        method="Heckman MLE",
                        params=config,
                        time_seconds=0.0,
                        converged=False,
                        notes=str(e)[:100],
                    )
                )

    def benchmark_westerlund(self):
        """Benchmark Westerlund cointegration tests."""
        print("\n" + "=" * 80)
        print("4. Westerlund Cointegration Tests")
        print("=" * 80)

        from panelbox.diagnostics.cointegration import westerlund_test

        configs = [
            {"N": 50, "T": 30, "bootstrap_reps": 100},
            {"N": 100, "T": 50, "bootstrap_reps": 100},
        ]

        for config in configs:
            N, T, bootstrap_reps = config["N"], config["T"], config["bootstrap_reps"]
            print(f"\n  Config: N={N}, T={T}, bootstrap_reps={bootstrap_reps}")

            # Generate cointegrated data
            np.random.seed(42)
            entity_id = np.repeat(np.arange(N), T)
            time_id = np.tile(np.arange(T), N)

            # Random walk
            y = np.zeros(N * T)
            for i in range(N):
                y[i * T : (i + 1) * T] = np.cumsum(np.random.randn(T))

            x = y + np.random.randn(N * T) * 0.1  # Cointegrated

            data = pd.DataFrame({"y": y, "x": x, "entity": entity_id, "time": time_id})

            try:
                start = time.time()
                result = westerlund_test(
                    data=data,
                    variables=["y", "x"],
                    entity_var="entity",
                    time_var="time",
                    bootstrap=True,
                    n_bootstrap=bootstrap_reps,
                )
                elapsed = time.time() - start
                print(f"    Westerlund (bootstrap): {elapsed:.3f}s")

                self.results.append(
                    BenchmarkResult(
                        method="Westerlund Bootstrap",
                        params=config,
                        time_seconds=elapsed,
                        converged=True,
                    )
                )
            except Exception as e:
                print(f"    Westerlund: FAILED ({str(e)[:50]})")
                self.results.append(
                    BenchmarkResult(
                        method="Westerlund Bootstrap",
                        params=config,
                        time_seconds=0.0,
                        converged=False,
                        notes=str(e)[:100],
                    )
                )

    def benchmark_unit_root(self):
        """Benchmark unit root tests."""
        print("\n" + "=" * 80)
        print("5. Unit Root Tests (Hadri & Breitung)")
        print("=" * 80)

        from panelbox.diagnostics.unit_root import breitung_test, hadri_test

        configs = [
            {"N": 50, "T": 30},
            {"N": 100, "T": 50},
        ]

        for config in configs:
            N, T = config["N"], config["T"]
            print(f"\n  Config: N={N}, T={T}")

            # Generate non-stationary data
            np.random.seed(42)
            entity_id = np.repeat(np.arange(N), T)
            time_id = np.tile(np.arange(T), N)

            y = np.zeros(N * T)
            for i in range(N):
                y[i * T : (i + 1) * T] = np.cumsum(np.random.randn(T))

            data = pd.DataFrame({"y": y, "entity": entity_id, "time": time_id})

            # Hadri test
            try:
                start = time.time()
                result = hadri_test(data=data, variable="y", entity_var="entity", time_var="time")
                elapsed = time.time() - start
                print(f"    Hadri test: {elapsed:.3f}s")

                self.results.append(
                    BenchmarkResult(
                        method="Hadri Test", params=config, time_seconds=elapsed, converged=True
                    )
                )
            except Exception as e:
                print(f"    Hadri: FAILED ({str(e)[:50]})")
                self.results.append(
                    BenchmarkResult(
                        method="Hadri Test",
                        params=config,
                        time_seconds=0.0,
                        converged=False,
                        notes=str(e)[:100],
                    )
                )

            # Breitung test
            try:
                start = time.time()
                result = breitung_test(
                    data=data, variable="y", entity_var="entity", time_var="time"
                )
                elapsed = time.time() - start
                print(f"    Breitung test: {elapsed:.3f}s")

                self.results.append(
                    BenchmarkResult(
                        method="Breitung Test", params=config, time_seconds=elapsed, converged=True
                    )
                )
            except Exception as e:
                print(f"    Breitung: FAILED ({str(e)[:50]})")
                self.results.append(
                    BenchmarkResult(
                        method="Breitung Test",
                        params=config,
                        time_seconds=0.0,
                        converged=False,
                        notes=str(e)[:100],
                    )
                )

    def benchmark_multinomial(self):
        """Benchmark Multinomial FE."""
        print("\n" + "=" * 80)
        print("6. Multinomial Logit Fixed Effects")
        print("=" * 80)

        from panelbox.models.discrete import MultinomialLogit

        configs = [
            {"N": 100, "T": 5, "J": 3},
            {"N": 100, "T": 10, "J": 3},
        ]

        for config in configs:
            N, T, J = config["N"], config["T"], config["J"]
            print(f"\n  Config: N={N}, T={T}, J={J} choices")

            # Generate data
            np.random.seed(42)
            entity_id = np.repeat(np.arange(N), T)
            time_id = np.tile(np.arange(T), N)

            X = np.random.randn(N * T, 2)

            # Generate choice probabilities
            choices = np.random.choice(J, size=N * T)

            data = pd.DataFrame(
                {
                    "choice": choices,
                    "x1": X[:, 0],
                    "x2": X[:, 1],
                    "entity": entity_id,
                    "time": time_id,
                }
            )

            try:
                start = time.time()
                model = MultinomialLogit(
                    data=data,
                    choice_var="choice",
                    formula="x1 + x2",
                    entity_var="entity",
                    time_var="time",
                    method="fixed_effects",
                )
                result = model.fit()
                elapsed = time.time() - start
                print(f"    Multinomial FE: {elapsed:.3f}s")

                if elapsed > 10:
                    print(f"    ⚠️  WARNING: Slow for J={J}, T={T}")

                self.results.append(
                    BenchmarkResult(
                        method="Multinomial FE", params=config, time_seconds=elapsed, converged=True
                    )
                )
            except Exception as e:
                print(f"    Multinomial FE: FAILED ({str(e)[:50]})")
                self.results.append(
                    BenchmarkResult(
                        method="Multinomial FE",
                        params=config,
                        time_seconds=0.0,
                        converged=False,
                        notes=str(e)[:100],
                    )
                )

    def benchmark_ppml(self):
        """Benchmark PPML FE."""
        print("\n" + "=" * 80)
        print("7. PPML Fixed Effects")
        print("=" * 80)

        from panelbox.models.count import PPML

        configs = [
            {"N": 500, "T": 10},
            {"N": 1000, "T": 20},
        ]

        for config in configs:
            N, T = config["N"], config["T"]
            print(f"\n  Config: N={N}, T={T}")

            # Generate count data
            np.random.seed(42)
            entity_id = np.repeat(np.arange(N), T)
            time_id = np.tile(np.arange(T), N)

            X = np.random.randn(N * T, 2)
            lambda_ = np.exp(X[:, 0] + X[:, 1])
            y = np.random.poisson(lambda_)

            data = pd.DataFrame(
                {"y": y, "x1": X[:, 0], "x2": X[:, 1], "entity": entity_id, "time": time_id}
            )

            try:
                start = time.time()
                model = PPML(
                    data=data,
                    dependent="y",
                    formula="x1 + x2",
                    entity_var="entity",
                    time_var="time",
                    method="fixed_effects",
                )
                result = model.fit()
                elapsed = time.time() - start
                print(f"    PPML FE: {elapsed:.3f}s")

                self.results.append(
                    BenchmarkResult(
                        method="PPML FE", params=config, time_seconds=elapsed, converged=True
                    )
                )
            except Exception as e:
                print(f"    PPML FE: FAILED ({str(e)[:50]})")
                self.results.append(
                    BenchmarkResult(
                        method="PPML FE",
                        params=config,
                        time_seconds=0.0,
                        converged=False,
                        notes=str(e)[:100],
                    )
                )

    def save_results(self):
        """Save results to CSV."""
        import os

        output_dir = "/home/guhaase/projetos/panelbox/docs/benchmarks"
        os.makedirs(output_dir, exist_ok=True)

        # Convert to DataFrame
        df = pd.DataFrame(
            [
                {
                    "method": r.method,
                    "time_seconds": r.time_seconds,
                    "converged": r.converged,
                    "notes": r.notes,
                    **r.params,
                }
                for r in self.results
            ]
        )

        output_path = os.path.join(output_dir, "benchmark_results.csv")
        df.to_csv(output_path, index=False)
        print(f"\n\nResults saved to: {output_path}")

    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        df = pd.DataFrame(
            [
                {"method": r.method, "time": r.time_seconds, "converged": r.converged}
                for r in self.results
            ]
        )

        # Group by method
        summary = df.groupby("method").agg({"time": ["mean", "min", "max"], "converged": "sum"})

        print(summary)

        # Warnings
        print("\n" + "=" * 80)
        print("PERFORMANCE RECOMMENDATIONS")
        print("=" * 80)

        slow_methods = df[df["time"] > 5]["method"].unique()
        if len(slow_methods) > 0:
            print("\n⚠️  Methods that may be slow for large datasets:")
            for method in slow_methods:
                print(f"  - {method}")

        failed_methods = df[~df["converged"]]["method"].unique()
        if len(failed_methods) > 0:
            print("\n❌ Methods that failed in some configurations:")
            for method in failed_methods:
                print(f"  - {method}")


if __name__ == "__main__":
    benchmarker = PerformanceBenchmarker()
    benchmarker.run_all_benchmarks()
