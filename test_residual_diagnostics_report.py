"""
Test Residual Diagnostics Report Generation

This script tests the end-to-end generation of residual diagnostics reports
using the new visualization system.
"""

from pathlib import Path
from unittest.mock import Mock
import numpy as np

from panelbox.report import ReportManager
from panelbox.visualization import create_residual_diagnostics


def create_mock_panel_results():
    """Create a mock PanelResults object for testing."""
    results = Mock()

    # Generate synthetic residual data
    np.random.seed(42)
    n_obs = 500

    # Simulated residuals (mostly normal with some outliers)
    residuals = np.random.normal(0, 1, n_obs)
    residuals[10] = 4.5  # Add outlier
    residuals[50] = -3.8  # Add outlier

    # Fitted values
    fitted = np.random.uniform(5, 15, n_obs)

    # Model attributes
    results.resids = residuals
    results.fitted_values = fitted
    results.nobs = n_obs

    # Entity and time info (panel structure)
    n_entities = 50
    n_periods = n_obs // n_entities
    results.entity_ids = np.repeat(np.arange(n_entities), n_periods)[:n_obs]
    results.time_ids = np.tile(np.arange(n_periods), n_entities)[:n_obs]

    # Leverage values (hat matrix diagonal)
    results.leverage = np.random.uniform(0.01, 0.15, n_obs)
    results.leverage[10] = 0.25  # High leverage for outlier

    # Model info
    results.model_info = {
        'estimator': 'FixedEffects',
        'model_type': 'Fixed Effects',
        'formula': 'y ~ x1 + x2 + x3 + EntityEffects',
        'nobs': n_obs,
        'n_entities': n_entities,
        'n_periods': n_periods,
        'balanced': True,
        'r_squared': 0.7234,
        'r_squared_adj': 0.7189
    }

    # Additional required attributes
    results.params = {'x1': 1.5, 'x2': -0.8, 'x3': 2.3}
    results.std_errors = {'x1': 0.3, 'x2': 0.25, 'x3': 0.4}

    return results


def main():
    """Run residual diagnostics report generation test."""
    print("=" * 80)
    print("Residual Diagnostics Report Generation Test")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path("output/residual_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir.absolute()}")
    print()

    # Step 1: Create mock results
    print("Step 1: Creating mock PanelResults...")
    results = create_mock_panel_results()
    print(f"✓ Mock results created")
    print(f"  - Observations: {results.nobs}")
    print(f"  - Entities: {results.model_info['n_entities']}")
    print(f"  - Time periods: {results.model_info['n_periods']}")
    print(f"  - R-squared: {results.model_info['r_squared']:.4f}")
    print()

    # Step 2: Generate residual diagnostics charts
    print("Step 2: Generating residual diagnostic charts...")
    try:
        diagnostics = create_residual_diagnostics(
            results=results,
            theme='professional',
            charts=None,  # Generate all available charts
            include_html=False  # Get chart objects
        )

        print(f"✓ Diagnostic charts generated")
        print(f"  - Number of charts: {len(diagnostics)}")
        print(f"  - Chart types: {', '.join(diagnostics.keys())}")
        print()

        # Convert to HTML for embedding
        residual_charts_html = {}
        for name, chart in diagnostics.items():
            residual_charts_html[name] = chart.to_html(
                include_plotlyjs=False,
                full_html=False,
                div_id=f"chart-{name}"
            )
            print(f"  - {name}: {len(residual_charts_html[name])} chars")

        print()

    except Exception as e:
        print(f"✗ Error generating charts: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Prepare report data
    print("Step 3: Preparing report data...")
    residual_data = {
        'residual_charts': residual_charts_html,
        'model_info': results.model_info,
        'diagnostics_summary': {
            'status': 'info',
            'message': 'Residual diagnostics completed successfully',
            'description': 'Review diagnostic plots below for potential model specification issues.'
        }
    }
    print("✓ Report data prepared")
    print()

    # Step 4: Generate HTML report
    print("Step 4: Generating HTML report...")
    try:
        report_mgr = ReportManager(minify=False)

        html = report_mgr.generate_residual_report(
            residual_data=residual_data,
            title="Residual Diagnostics Report",
            subtitle="Fixed Effects Model - Specification Checks",
            interactive=True
        )

        # Save report
        output_path = output_dir / "residual_diagnostics_report.html"
        output_path.write_text(html, encoding='utf-8')

        print(f"✓ Report generated: {output_path}")
        print(f"  - File size: {len(html) / 1024:.1f} KB")
        print()

        # Check report content
        has_plotly = 'plotly' in html.lower()
        has_charts = 'class="plotly-graph-div"' in html
        has_qq_plot = 'qq_plot' in html or 'Q-Q Plot' in html
        has_residual_fitted = 'residual_vs_fitted' in html
        has_scale_location = 'scale_location' in html

        print("Report validation:")
        print(f"  - Contains Plotly: {'✓' if has_plotly else '✗'}")
        print(f"  - Contains chart divs: {'✓' if has_charts else '✗'}")
        print(f"  - Q-Q Plot: {'✓' if has_qq_plot else '✗'}")
        print(f"  - Residuals vs Fitted: {'✓' if has_residual_fitted else '✗'}")
        print(f"  - Scale-Location: {'✓' if has_scale_location else '✗'}")
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
    print("  2. Navigate through the tabs (Overview, Diagnostic Plots, Interpretation)")
    print("  3. Test chart interactivity (hover, zoom, pan)")
    print("  4. Verify all diagnostic plots are displayed correctly")
    print()

    if has_charts and has_qq_plot and has_residual_fitted:
        print("✅ SUCCESS: Residual diagnostics report generation working!")
        print("   - All diagnostic charts generated")
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
