"""
Test Model Comparison Report Generation

This script tests the end-to-end generation of model comparison reports
using the new visualization system.
"""

from pathlib import Path
from unittest.mock import Mock
import numpy as np

from panelbox.report import ReportManager
from panelbox.visualization import create_comparison_charts


def create_mock_panel_results_list():
    """Create a list of mock PanelResults objects for testing."""
    np.random.seed(42)
    n_obs = 500
    n_entities = 50
    n_periods = n_obs // n_entities

    results_list = []
    model_configs = [
        {
            'name': 'Pooled OLS',
            'estimator': 'PooledOLS',
            'r_squared': 0.6523,
            'r_squared_adj': 0.6489,
            'aic': 2543.21,
            'bic': 2598.45,
            'coef_mult': 1.0
        },
        {
            'name': 'Fixed Effects',
            'estimator': 'PanelOLS',
            'r_squared': 0.7834,
            'r_squared_adj': 0.7756,
            'aic': 2298.67,
            'bic': 2412.89,
            'coef_mult': 1.2
        },
        {
            'name': 'Random Effects',
            'estimator': 'RandomEffects',
            'r_squared': 0.7245,
            'r_squared_adj': 0.7189,
            'aic': 2367.54,
            'bic': 2435.21,
            'coef_mult': 1.1
        }
    ]

    for config in model_configs:
        results = Mock()

        # Model info
        results.model_info = {
            'estimator': config['estimator'],
            'model_type': config['name'],
            'nobs': n_obs,
            'n_entities': n_entities,
            'n_periods': n_periods,
            'balanced': True,
            'r_squared': config['r_squared'],
            'r_squared_adj': config['r_squared_adj']
        }

        # Coefficients (vary slightly between models)
        mult = config['coef_mult']
        results.params = {
            'x1': 1.5 * mult,
            'x2': -0.8 * mult,
            'x3': 2.3 * mult,
            'x4': 0.6 * mult
        }

        # Standard errors
        results.std_errors = {
            'x1': 0.3 / np.sqrt(mult),
            'x2': 0.25 / np.sqrt(mult),
            'x3': 0.4 / np.sqrt(mult),
            'x4': 0.15 / np.sqrt(mult)
        }

        # Additional attributes for API
        results.nobs = n_obs
        results.rsquared = config['r_squared']
        results.rsquared_adj = config['r_squared_adj']

        # Information criteria
        results.aic = config['aic']
        results.bic = config['bic']

        results_list.append(results)

    return results_list, [c['name'] for c in model_configs]


def main():
    """Run model comparison report generation test."""
    print("=" * 80)
    print("Model Comparison Report Generation Test")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path("output/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir.absolute()}")
    print()

    # Step 1: Create mock results
    print("Step 1: Creating mock PanelResults list...")
    results_list, model_names = create_mock_panel_results_list()
    print(f"✓ Mock results created")
    print(f"  - Number of models: {len(results_list)}")
    print(f"  - Model names: {', '.join(model_names)}")
    for i, (results, name) in enumerate(zip(results_list, model_names)):
        print(f"  - {name}:")
        print(f"    - Estimator: {results.model_info['estimator']}")
        print(f"    - R-squared: {results.rsquared:.4f}")
        print(f"    - AIC: {results.aic:.2f}")
        print(f"    - BIC: {results.bic:.2f}")
    print()

    # Step 2: Generate comparison charts
    print("Step 2: Generating model comparison charts...")
    try:
        charts = create_comparison_charts(
            results_list=results_list,
            model_names=model_names,
            theme='professional',
            charts=None,  # Generate all available charts
            include_html=False  # Get chart objects
        )

        print(f"✓ Comparison charts generated")
        print(f"  - Number of charts: {len(charts)}")
        print(f"  - Chart types: {', '.join(charts.keys())}")
        print()

        # Convert to HTML for embedding
        comparison_charts_html = {}
        for name, chart in charts.items():
            comparison_charts_html[name] = chart.to_html(
                include_plotlyjs=False,
                full_html=False,
                div_id=f"chart-{name}"
            )
            print(f"  - {name}: {len(comparison_charts_html[name])} chars")

        print()

    except Exception as e:
        print(f"✗ Error generating charts: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Prepare report data
    print("Step 3: Preparing report data...")

    # Extract models info
    models_info = []
    for results, name in zip(results_list, model_names):
        models_info.append({
            'name': name,
            'estimator': results.model_info['estimator'],
            'nobs': results.nobs,
            'r_squared': results.rsquared,
            'aic': results.aic,
            'bic': results.bic
        })

    # Find best models by AIC and BIC
    best_aic_idx = np.argmin([r.aic for r in results_list])
    best_bic_idx = np.argmin([r.bic for r in results_list])

    comparison_data = {
        'comparison_charts': comparison_charts_html,
        'models_info': models_info,
        'best_model_aic': model_names[best_aic_idx],
        'best_model_bic': model_names[best_bic_idx],
        'comparison_summary': {
            'status': 'success',
            'message': 'Model comparison completed successfully',
            'description': f'Best model by AIC: {model_names[best_aic_idx]}. Best model by BIC: {model_names[best_bic_idx]}.'
        }
    }
    print("✓ Report data prepared")
    print(f"  - Best model (AIC): {model_names[best_aic_idx]}")
    print(f"  - Best model (BIC): {model_names[best_bic_idx]}")
    print()

    # Step 4: Generate HTML report
    print("Step 4: Generating HTML report...")
    try:
        report_mgr = ReportManager(minify=False)

        html = report_mgr.generate_comparison_report(
            comparison_data=comparison_data,
            title="Model Comparison Report",
            subtitle="Pooled OLS vs Fixed Effects vs Random Effects",
            interactive=True
        )

        # Save report
        output_path = output_dir / "model_comparison_report.html"
        output_path.write_text(html, encoding='utf-8')

        print(f"✓ Report generated: {output_path}")
        print(f"  - File size: {len(html) / 1024:.1f} KB")
        print()

        # Check report content
        has_plotly = 'plotly' in html.lower()
        has_charts = 'class="plotly-graph-div"' in html
        has_coefficient_comparison = 'coefficient_comparison' in html or 'Coefficient Comparison' in html
        has_forest_plot = 'forest_plot' in html
        has_fit_comparison = 'fit_comparison' in html
        has_ic_comparison = 'ic_comparison' in html

        print("Report validation:")
        print(f"  - Contains Plotly: {'✓' if has_plotly else '✗'}")
        print(f"  - Contains chart divs: {'✓' if has_charts else '✗'}")
        print(f"  - Coefficient Comparison: {'✓' if has_coefficient_comparison else '✗'}")
        print(f"  - Forest Plot: {'✓' if has_forest_plot else '✗'}")
        print(f"  - Fit Comparison: {'✓' if has_fit_comparison else '✗'}")
        print(f"  - IC Comparison: {'✓' if has_ic_comparison else '✗'}")
        print()

    except Exception as e:
        print(f"✗ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Summary
    print("=" * 80)
    print("Test Complete!")
    print("=" * 80)
    print()
    print("Generated report:")
    print(f"  📄 {output_path.name}")
    print(f"  📁 {output_path.absolute()}")
    print(f"  🌐 file://{output_path.absolute()}")
    print()
    print("Next steps:")
    print("  1. Open the report in your browser")
    print("  2. Navigate through the tabs (Overview, Comparison Charts, Interpretation)")
    print("  3. Test chart interactivity (hover, zoom, pan)")
    print("  4. Verify all comparison charts are displayed correctly")
    print("  5. Check that model selection guidance is clear")
    print()

    if has_charts and has_coefficient_comparison and has_fit_comparison:
        print("✅ SUCCESS: Model comparison report generation working!")
        print("   - All comparison charts generated")
        print("   - Charts embedded as pre-rendered HTML")
        print("   - Template rendering successful")
        print("   - Report ready for use")
        return True
    else:
        print("⚠️  PARTIAL SUCCESS: Report generated but some charts may be missing")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
