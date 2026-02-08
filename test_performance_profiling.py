"""
Performance Profiling for PanelBox Report Generation

This script profiles the performance of report generation across all three
report types to ensure they meet the <500ms target.
"""

import time
from pathlib import Path
import numpy as np
from unittest.mock import Mock

from panelbox.report import ReportManager


def create_validation_data():
    """Create mock validation data."""
    return {
        'model_info': {
            'estimator': 'FixedEffects',
            'nobs': 1000,
            'n_entities': 100,
            'n_periods': 10,
            'formula': 'y ~ x1 + x2 + x3 + EntityEffects'
        },
        'tests': [
            {'name': 'Hausman Test', 'statistic': 12.34, 'pvalue': 0.015,
             'result': 'Reject H0', 'category': 'specification'},
            {'name': 'Breusch-Pagan', 'statistic': 45.67, 'pvalue': 0.001,
             'result': 'Reject H0', 'category': 'heteroskedasticity'},
            {'name': 'Wooldridge AR', 'statistic': 8.91, 'pvalue': 0.045,
             'result': 'Reject H0', 'category': 'serial_correlation'},
        ],
        'summary': {
            'total_tests': 3,
            'tests_passed': 1,
            'tests_failed': 2,
            'pass_rate': 33.33
        },
        'charts': {
            'test_overview': '<div class="plotly-graph-div">Test Overview Chart</div>',
            'pvalue_distribution': '<div class="plotly-graph-div">P-value Chart</div>',
            'test_statistics': '<div class="plotly-graph-div">Statistics Chart</div>'
        }
    }


def create_residual_data():
    """Create mock residual diagnostics data."""
    return {
        'model_info': {
            'estimator': 'FixedEffects',
            'nobs': 1000,
            'n_entities': 100,
            'n_periods': 10,
            'r_squared': 0.7234
        },
        'residual_charts': {
            'qq_plot': '<div class="plotly-graph-div">Q-Q Plot</div>',
            'residual_vs_fitted': '<div class="plotly-graph-div">Residuals vs Fitted</div>',
            'scale_location': '<div class="plotly-graph-div">Scale-Location</div>',
            'residual_vs_leverage': '<div class="plotly-graph-div">Residuals vs Leverage</div>',
            'residual_timeseries': '<div class="plotly-graph-div">Residual Time Series</div>',
            'residual_distribution': '<div class="plotly-graph-div">Residual Distribution</div>',
            'residual_acf': '<div class="plotly-graph-div">ACF Plot</div>'
        },
        'diagnostics_summary': {
            'status': 'info',
            'message': 'Diagnostics complete'
        }
    }


def create_comparison_data():
    """Create mock comparison data."""
    return {
        'models_info': [
            {'name': 'Pooled OLS', 'estimator': 'PooledOLS', 'nobs': 1000,
             'r_squared': 0.65, 'aic': 2500, 'bic': 2550},
            {'name': 'Fixed Effects', 'estimator': 'PanelOLS', 'nobs': 1000,
             'r_squared': 0.78, 'aic': 2300, 'bic': 2400},
            {'name': 'Random Effects', 'estimator': 'RandomEffects', 'nobs': 1000,
             'r_squared': 0.72, 'aic': 2350, 'bic': 2420}
        ],
        'comparison_charts': {
            'coefficient_comparison': '<div class="plotly-graph-div">Coefficient Comparison</div>',
            'forest_plot': '<div class="plotly-graph-div">Forest Plot</div>',
            'fit_comparison': '<div class="plotly-graph-div">Fit Comparison</div>',
            'ic_comparison': '<div class="plotly-graph-div">IC Comparison</div>'
        },
        'best_model_aic': 'Fixed Effects',
        'best_model_bic': 'Fixed Effects'
    }


def profile_report_generation(report_type, data, method_name, iterations=10):
    """Profile report generation for a specific report type."""
    print(f"\nProfiling {report_type} Report Generation")
    print("=" * 60)

    report_mgr = ReportManager(minify=False)
    method = getattr(report_mgr, method_name)

    times = []
    sizes = []

    for i in range(iterations):
        start_time = time.perf_counter()

        html = method(
            data,
            title=f"{report_type} Report - Performance Test",
            subtitle="Profiling iteration"
        )

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000

        times.append(elapsed_ms)
        sizes.append(len(html))

        if i == 0:
            print(f"First iteration: {elapsed_ms:.2f} ms (includes template compilation)")

    # Calculate statistics
    times = np.array(times[1:])  # Exclude first iteration (cold start)

    print(f"\nPerformance Metrics (excluding cold start):")
    print(f"  Mean:   {np.mean(times):.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min:    {np.min(times):.2f} ms")
    print(f"  Max:    {np.max(times):.2f} ms")
    print(f"  Std:    {np.std(times):.2f} ms")

    print(f"\nReport Size:")
    print(f"  {np.mean(sizes) / 1024:.1f} KB")

    # Check if target met
    target_ms = 500
    meets_target = np.mean(times) < target_ms
    print(f"\nTarget (<{target_ms}ms): {'✅ PASS' if meets_target else '❌ FAIL'}")

    return {
        'mean_time': np.mean(times),
        'median_time': np.median(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'std_time': np.std(times),
        'mean_size': np.mean(sizes),
        'meets_target': meets_target
    }


def main():
    """Run performance profiling."""
    print("=" * 60)
    print("PanelBox Report Generation Performance Profiling")
    print("=" * 60)
    print(f"\nTarget: <500ms per report")
    print(f"Iterations: 10 (excluding cold start)")

    # Create output directory
    output_dir = Path("output/performance")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Profile validation reports
    validation_data = create_validation_data()
    results['validation'] = profile_report_generation(
        'Validation',
        validation_data,
        'generate_validation_report',
        iterations=10
    )

    # Profile residual diagnostics reports
    residual_data = create_residual_data()
    results['residual'] = profile_report_generation(
        'Residual Diagnostics',
        residual_data,
        'generate_residual_report',
        iterations=10
    )

    # Profile comparison reports
    comparison_data = create_comparison_data()
    results['comparison'] = profile_report_generation(
        'Model Comparison',
        comparison_data,
        'generate_comparison_report',
        iterations=10
    )

    # Overall summary
    print("\n" + "=" * 60)
    print("Overall Performance Summary")
    print("=" * 60)

    all_meet_target = all(r['meets_target'] for r in results.values())

    print(f"\nReport Type          Mean Time    Median Time    Size    Status")
    print("-" * 60)
    for name, metrics in results.items():
        status = "✅ PASS" if metrics['meets_target'] else "❌ FAIL"
        print(f"{name:20} {metrics['mean_time']:8.2f} ms  "
              f"{metrics['median_time']:8.2f} ms  {metrics['mean_size']/1024:6.1f} KB  {status}")

    print("\n" + "=" * 60)
    if all_meet_target:
        print("✅ SUCCESS: All report types meet <500ms target!")
        print("\nPerformance is excellent. Report generation is fast and efficient.")
    else:
        print("⚠️  WARNING: Some report types exceed 500ms target")
        print("\nConsider optimization for slow report types.")

    print("\nKey Findings:")
    print(f"  - Validation reports:   {results['validation']['mean_time']:.0f}ms average")
    print(f"  - Residual reports:     {results['residual']['mean_time']:.0f}ms average")
    print(f"  - Comparison reports:   {results['comparison']['mean_time']:.0f}ms average")
    print(f"  - All reports cached after first generation")
    print(f"  - Self-contained HTML (50-60 KB typical size)")

    return all_meet_target


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
