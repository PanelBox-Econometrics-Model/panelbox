"""
Performance Benchmarks for PanelBox Visualization System.

This script benchmarks chart creation performance across different data sizes
and chart types to ensure the system remains performant.

Run:
    python benchmarks/visualization_performance.py

Output:
    - Performance metrics for each chart type
    - Recommendations for optimization
    - Performance report (JSON and text)
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    chart_type: str
    data_size: str
    n_obs: int
    n_entities: int
    n_periods: int
    execution_time_ms: float
    memory_mb: float
    success: bool
    error: str = None


class VisualizationBenchmark:
    """Performance benchmark suite for visualization charts."""

    def __init__(self):
        """Initialize benchmark suite."""
        self.results: List[BenchmarkResult] = []

    def run_all(self) -> List[BenchmarkResult]:
        """
        Run all benchmarks.

        Returns
        -------
        list of BenchmarkResult
            All benchmark results
        """
        print("="*80)
        print("PanelBox Visualization Performance Benchmarks")
        print("="*80)
        print()

        # Test configurations: (n_entities, n_periods, label)
        test_configs = [
            (10, 10, "Small"),
            (50, 20, "Medium"),
            (100, 30, "Large"),
            (200, 50, "Very Large"),
        ]

        # Benchmark each chart type
        chart_types = [
            'residual_qq_plot',
            'residual_vs_fitted',
            'entity_effects_plot',
            'time_effects_plot',
            'between_within_plot',
            'panel_structure_plot',
            'acf_pacf_plot',
            'unit_root_test_plot',
            'cointegration_heatmap',
            'cross_sectional_dependence_plot',
        ]

        for chart_type in chart_types:
            print(f"\nBenchmarking: {chart_type}")
            print("-" * 60)

            for n_entities, n_periods, size_label in test_configs:
                result = self._benchmark_chart(
                    chart_type,
                    n_entities,
                    n_periods,
                    size_label
                )
                self.results.append(result)

                status = "✅" if result.success else "❌"
                print(f"  {status} {size_label:12} ({n_entities}x{n_periods}): "
                      f"{result.execution_time_ms:6.1f} ms")

        return self.results

    def _benchmark_chart(
        self,
        chart_type: str,
        n_entities: int,
        n_periods: int,
        size_label: str
    ) -> BenchmarkResult:
        """
        Benchmark a single chart type.

        Parameters
        ----------
        chart_type : str
            Chart type to benchmark
        n_entities : int
            Number of entities
        n_periods : int
            Number of time periods
        size_label : str
            Size label (Small, Medium, Large)

        Returns
        -------
        BenchmarkResult
            Benchmark result
        """
        n_obs = n_entities * n_periods

        try:
            # Generate test data
            data = self._generate_test_data(chart_type, n_entities, n_periods)

            # Measure execution time
            start_time = time.perf_counter()

            chart = self._create_chart(chart_type, data)

            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000

            # Estimate memory (rough approximation)
            memory_mb = n_obs * 0.001  # Rough estimate

            return BenchmarkResult(
                chart_type=chart_type,
                data_size=size_label,
                n_obs=n_obs,
                n_entities=n_entities,
                n_periods=n_periods,
                execution_time_ms=execution_time_ms,
                memory_mb=memory_mb,
                success=True
            )

        except Exception as e:
            return BenchmarkResult(
                chart_type=chart_type,
                data_size=size_label,
                n_obs=n_obs,
                n_entities=n_entities,
                n_periods=n_periods,
                execution_time_ms=0.0,
                memory_mb=0.0,
                success=False,
                error=str(e)
            )

    def _generate_test_data(
        self,
        chart_type: str,
        n_entities: int,
        n_periods: int
    ) -> Dict:
        """Generate test data for chart type."""
        n_obs = n_entities * n_periods

        if chart_type in ['residual_qq_plot', 'residual_vs_fitted']:
            return {
                'residuals': np.random.randn(n_obs),
                'fitted': np.random.randn(n_obs),
            }

        elif chart_type == 'entity_effects_plot':
            return {
                'entity_id': [f'Entity_{i}' for i in range(n_entities)],
                'effect': np.random.randn(n_entities),
                'std_error': np.random.uniform(0.1, 0.5, n_entities),
            }

        elif chart_type == 'time_effects_plot':
            return {
                'time_id': list(range(n_periods)),
                'effect': np.random.randn(n_periods),
                'std_error': np.random.uniform(0.1, 0.5, n_periods),
            }

        elif chart_type == 'between_within_plot':
            return {
                'variables': ['var1', 'var2', 'var3'],
                'between_var': [2.5, 3.0, 1.5],
                'within_var': [1.5, 2.0, 2.5],
                'total_var': [4.0, 5.0, 4.0],
            }

        elif chart_type == 'panel_structure_plot':
            # Create presence matrix
            entities = [f'E{i}' for i in range(n_entities)]
            periods = [f'T{i}' for i in range(n_periods)]
            presence = np.random.choice([0, 1], size=(n_entities, n_periods), p=[0.2, 0.8])

            return {
                'entities': entities,
                'periods': periods,
                'presence_matrix': presence.tolist(),
                'n_obs_per_entity': presence.sum(axis=1).tolist(),
                'n_obs_per_period': presence.sum(axis=0).tolist(),
            }

        elif chart_type == 'acf_pacf_plot':
            return {
                'residuals': np.random.randn(n_obs),
                'max_lags': min(20, n_obs // 4),
            }

        elif chart_type == 'unit_root_test_plot':
            return {
                'test_names': ['ADF', 'PP', 'KPSS'],
                'test_stats': [-3.5, -3.8, 0.3],
                'critical_values': {'1%': -3.96, '5%': -3.41, '10%': -3.13},
                'pvalues': [0.008, 0.003, 0.15],
            }

        elif chart_type == 'cointegration_heatmap':
            n_vars = min(10, n_entities)
            pvalues = np.random.uniform(0.01, 0.5, (n_vars, n_vars))
            np.fill_diagonal(pvalues, 1.0)

            return {
                'variables': [f'Var_{i}' for i in range(n_vars)],
                'pvalues': pvalues.tolist(),
                'test_name': 'Engle-Granger',
            }

        elif chart_type == 'cross_sectional_dependence_plot':
            return {
                'cd_statistic': 3.45,
                'pvalue': 0.003,
                'avg_correlation': 0.28,
                'entity_correlations': np.random.uniform(0.1, 0.5, n_entities).tolist(),
            }

        else:
            return {'data': np.random.randn(n_obs)}

    def _create_chart(self, chart_type: str, data: Dict):
        """Create chart (imports done here to measure true import + creation time)."""
        from panelbox.visualization import ChartFactory

        chart = ChartFactory.create(chart_type, data=data, theme='professional')
        return chart

    def generate_report(self, output_dir: str = 'benchmarks/results'):
        """
        Generate performance report.

        Parameters
        ----------
        output_dir : str, default 'benchmarks/results'
            Output directory for reports
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate JSON report
        json_path = output_path / 'performance_report.json'
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        # Generate text report
        txt_path = output_path / 'performance_report.txt'
        with open(txt_path, 'w') as f:
            f.write(self._format_text_report())

        print(f"\n✅ Reports generated:")
        print(f"   - JSON: {json_path}")
        print(f"   - Text: {txt_path}")

    def _format_text_report(self) -> str:
        """Format text report."""
        report = []
        report.append("="*80)
        report.append("PANELBOX VISUALIZATION PERFORMANCE REPORT")
        report.append("="*80)
        report.append("")

        # Group by chart type
        chart_types = sorted(set(r.chart_type for r in self.results))

        for chart_type in chart_types:
            chart_results = [r for r in self.results if r.chart_type == chart_type]

            report.append(f"\n{chart_type}")
            report.append("-" * 60)

            for result in chart_results:
                status = "✅" if result.success else "❌"
                report.append(
                    f"  {status} {result.data_size:12} "
                    f"({result.n_entities:3}x{result.n_periods:2}): "
                    f"{result.execution_time_ms:7.1f} ms"
                )

        # Summary statistics
        successful_results = [r for r in self.results if r.success]

        if successful_results:
            report.append("\n" + "="*80)
            report.append("SUMMARY STATISTICS")
            report.append("="*80)

            times = [r.execution_time_ms for r in successful_results]
            report.append(f"\nExecution Time:")
            report.append(f"  Mean:   {np.mean(times):7.1f} ms")
            report.append(f"  Median: {np.median(times):7.1f} ms")
            report.append(f"  Min:    {np.min(times):7.1f} ms")
            report.append(f"  Max:    {np.max(times):7.1f} ms")
            report.append(f"  Std:    {np.std(times):7.1f} ms")

            report.append(f"\nTotal Tests: {len(self.results)}")
            report.append(f"Successful:  {len(successful_results)}")
            report.append(f"Failed:      {len(self.results) - len(successful_results)}")

        # Recommendations
        report.append("\n" + "="*80)
        report.append("PERFORMANCE RECOMMENDATIONS")
        report.append("="*80)
        report.append("")

        if successful_results:
            slow_results = [r for r in successful_results if r.execution_time_ms > 1000]

            if slow_results:
                report.append("⚠️  Slow Charts (>1 second):")
                for result in slow_results:
                    report.append(f"  • {result.chart_type} ({result.data_size}): "
                                f"{result.execution_time_ms:.1f} ms")
                report.append("")
                report.append("Recommendations:")
                report.append("  - Consider lazy loading for large datasets")
                report.append("  - Use data aggregation before visualization")
                report.append("  - Enable caching for repeated renders")
            else:
                report.append("✅ All charts render in under 1 second!")

        return "\n".join(report)


def run_benchmarks():
    """Main function to run all benchmarks."""
    benchmark = VisualizationBenchmark()

    # Run all benchmarks
    results = benchmark.run_all()

    # Generate reports
    benchmark.generate_report()

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"\nTotal tests: {len(results)}")
    print(f"Successful:  {sum(1 for r in results if r.success)}")
    print(f"Failed:      {sum(1 for r in results if not r.success)}")

    successful = [r for r in results if r.success]
    if successful:
        times = [r.execution_time_ms for r in successful]
        print(f"\nPerformance:")
        print(f"  Mean execution time: {np.mean(times):.1f} ms")
        print(f"  Median: {np.median(times):.1f} ms")
        print(f"  Max: {np.max(times):.1f} ms")


if __name__ == '__main__':
    run_benchmarks()
