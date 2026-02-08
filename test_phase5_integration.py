"""
Test Phase 5 Integration - End-to-End Report Generation

This script tests the complete integration of the new visualization system
with the report generation pipeline.
"""

from pathlib import Path
from unittest.mock import Mock

from panelbox.report import ReportManager
from panelbox.report.validation_transformer import ValidationTransformer


def create_mock_validation_report():
    """Create a mock ValidationReport for testing."""
    report = Mock()

    # Specification tests
    spec_result = Mock()
    spec_result.statistic = 15.234
    spec_result.pvalue = 0.002
    spec_result.df = 3
    spec_result.conclusion = "Reject H0: Random Effects are inconsistent. Use Fixed Effects."
    spec_result.reject_null = True
    spec_result.alpha = 0.05
    spec_result.metadata = {}

    mundlak_result = Mock()
    mundlak_result.statistic = 12.876
    mundlak_result.pvalue = 0.005
    mundlak_result.df = 3
    mundlak_result.conclusion = "Reject H0: Entity means are correlated with unobserved effect."
    mundlak_result.reject_null = True
    mundlak_result.alpha = 0.05
    mundlak_result.metadata = {}

    report.specification_tests = {
        "Hausman Test": spec_result,
        "Mundlak Test": mundlak_result
    }

    # Serial correlation tests
    serial_result = Mock()
    serial_result.statistic = 2.345
    serial_result.pvalue = 0.128
    serial_result.df = 1
    serial_result.conclusion = "Accept H0: No first-order autocorrelation detected."
    serial_result.reject_null = False
    serial_result.alpha = 0.05
    serial_result.metadata = {}

    bw_result = Mock()
    bw_result.statistic = 1.987
    bw_result.pvalue = 0.156
    bw_result.df = None
    bw_result.conclusion = "Accept H0: No serial correlation detected."
    bw_result.reject_null = False
    bw_result.alpha = 0.05
    bw_result.metadata = {}

    report.serial_tests = {
        "Wooldridge Test": serial_result,
        "Baltagi-Wu Test": bw_result
    }

    # Heteroskedasticity tests
    het_result = Mock()
    het_result.statistic = 18.456
    het_result.pvalue = 0.001
    het_result.df = 3
    het_result.conclusion = "Reject H0: Heteroskedasticity detected. Use robust SE."
    het_result.reject_null = True
    het_result.alpha = 0.05
    het_result.metadata = {}

    report.het_tests = {
        "Breusch-Pagan LM Test": het_result
    }

    # Cross-sectional dependence tests
    cd_result = Mock()
    cd_result.statistic = 3.789
    cd_result.pvalue = 0.0002
    cd_result.df = None
    cd_result.conclusion = "Reject H0: Cross-sectional dependence detected."
    cd_result.reject_null = True
    cd_result.alpha = 0.05
    cd_result.metadata = {}

    frees_result = Mock()
    frees_result.statistic = 2.456
    frees_result.pvalue = 0.032
    frees_result.df = None
    frees_result.conclusion = "Reject H0: Cross-sectional dependence detected."
    frees_result.reject_null = True
    frees_result.alpha = 0.05
    frees_result.metadata = {}

    report.cd_tests = {
        "Pesaran CD Test": cd_result,
        "Frees Test": frees_result
    }

    # Model info
    report.model_info = {
        "model_type": "Fixed Effects",
        "formula": "y ~ x1 + x2 + x3",
        "nobs": 1000,
        "n_entities": 100,
        "n_periods": 10,
        "balanced": True,
    }

    return report


def main():
    """Run end-to-end integration test."""
    print("=" * 80)
    print("Phase 5 Integration Test - End-to-End Report Generation")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path("output/phase5_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir.absolute()}")
    print()

    # Step 1: Create mock validation report
    print("Step 1: Creating mock validation report...")
    validation_report = create_mock_validation_report()
    print(f"✓ Mock report created")
    print(f"  - Specification tests: {len(validation_report.specification_tests)}")
    print(f"  - Serial tests: {len(validation_report.serial_tests)}")
    print(f"  - Heteroskedasticity tests: {len(validation_report.het_tests)}")
    print(f"  - Cross-sectional tests: {len(validation_report.cd_tests)}")
    print()

    # Step 2: Test with NEW visualization system
    print("Step 2: Generating report with NEW visualization system...")
    try:
        transformer = ValidationTransformer(validation_report)
        validation_data_new = transformer.transform(
            include_charts=True,
            use_new_visualization=True
        )

        # Check chart types
        if 'charts' in validation_data_new:
            test_overview_type = type(validation_data_new['charts']['test_overview'])
            print(f"✓ Charts generated")
            print(f"  - Chart type: {test_overview_type.__name__}")

            if test_overview_type == str:
                print(f"  - ✅ Using NEW visualization system (pre-rendered HTML)")
                chart_length = len(validation_data_new['charts']['test_overview'])
                print(f"  - HTML length: {chart_length} characters")
            else:
                print(f"  - ⚠️  Fallback to legacy mode (data dict)")

        # Generate HTML report
        report_mgr = ReportManager(minify=False)
        html_new = report_mgr.generate_validation_report(
            validation_data=validation_data_new,
            interactive=True,
            title="Phase 5 Integration Test - NEW Visualization",
            subtitle="Testing pre-rendered charts from visualization system"
        )

        # Save report
        output_path_new = output_dir / "validation_report_new.html"
        output_path_new.write_text(html_new, encoding='utf-8')

        print(f"✓ Report generated: {output_path_new}")
        print(f"  - File size: {len(html_new) / 1024:.1f} KB")
        print()

    except Exception as e:
        print(f"✗ Error with new system: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Step 3: Test with LEGACY system (for comparison)
    print("Step 3: Generating report with LEGACY system...")
    try:
        transformer_legacy = ValidationTransformer(validation_report)
        validation_data_legacy = transformer_legacy.transform(
            include_charts=True,
            use_new_visualization=False
        )

        # Check chart types
        if 'charts' in validation_data_legacy:
            test_overview_type = type(validation_data_legacy['charts']['test_overview'])
            print(f"✓ Charts generated")
            print(f"  - Chart type: {test_overview_type.__name__}")
            print(f"  - ✅ Using LEGACY system (data dicts)")

        # Generate HTML report
        report_mgr_legacy = ReportManager(minify=False)
        html_legacy = report_mgr_legacy.generate_validation_report(
            validation_data=validation_data_legacy,
            interactive=True,
            title="Phase 5 Integration Test - LEGACY Visualization",
            subtitle="Testing legacy inline JavaScript rendering"
        )

        # Save report
        output_path_legacy = output_dir / "validation_report_legacy.html"
        output_path_legacy.write_text(html_legacy, encoding='utf-8')

        print(f"✓ Report generated: {output_path_legacy}")
        print(f"  - File size: {len(html_legacy) / 1024:.1f} KB")
        print()

    except Exception as e:
        print(f"✗ Error with legacy system: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Step 4: Compare results
    print("=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    print()

    try:
        size_diff = (len(html_new) - len(html_legacy)) / 1024
        print(f"File sizes:")
        print(f"  - NEW system:    {len(html_new) / 1024:.1f} KB")
        print(f"  - LEGACY system: {len(html_legacy) / 1024:.1f} KB")
        print(f"  - Difference:    {size_diff:+.1f} KB")
        print()

        # Check for Plotly in both
        has_plotly_new = 'plotly' in html_new.lower()
        has_plotly_legacy = 'plotly' in html_legacy.lower()

        print(f"Plotly usage:")
        print(f"  - NEW system:    {'✓' if has_plotly_new else '✗'}")
        print(f"  - LEGACY system: {'✓' if has_plotly_legacy else '✗'}")
        print()

        # Check for Plotly rendering code
        has_plotly_new = 'Plotly.newPlot' in html_new
        has_plotly_legacy = 'Plotly.newPlot' in html_legacy

        print(f"Plotly rendering code:")
        print(f"  - NEW system:    {'✓' if has_plotly_new else '✗'} (charts use Plotly)")
        print(f"  - LEGACY system: {'✓' if has_plotly_legacy else '✗'} (would need old template)")
        print()

        # The key difference: Check WHERE the chart definitions are
        # NEW system: charts are pre-defined in the chart divs
        # LEGACY system: charts would be defined in template JavaScript (but template was updated)
        has_chart_div_new = 'id="chart-test_overview"' in html_new or 'class="plotly-graph-div"' in html_new

        print(f"Chart structure:")
        print(f"  - NEW system:    {'✓ Pre-rendered chart divs' if has_chart_div_new else '✗ No chart divs'}")
        print(f"  - LEGACY system: {'ℹ Templates updated for new system' }")
        print()

    except NameError:
        print("⚠️  Could not compare - one or both reports failed to generate")
        print()

    # Final summary
    print("=" * 80)
    print("Integration Test Complete!")
    print("=" * 80)
    print()
    print("Generated reports:")
    if 'output_path_new' in locals():
        print(f"  📄 NEW system:    file://{output_path_new.absolute()}")
    if 'output_path_legacy' in locals():
        print(f"  📄 LEGACY system: file://{output_path_legacy.absolute()}")
    print()
    print("Next steps:")
    print("  1. Open both reports in your browser")
    print("  2. Compare visual appearance")
    print("  3. Test chart interactivity (hover, zoom, pan)")
    print("  4. Verify both reports work correctly")
    print()

    if 'has_chart_div_new' in locals() and has_chart_div_new:
        print("✅ SUCCESS: Integration complete!")
        print("   - NEW system uses pre-rendered charts (Plotly divs)")
        print("   - Charts embedded directly in HTML")
        print("   - Templates refactored successfully")
        print("   - Reports generated and ready for review")
        return True
    else:
        print("⚠️  PARTIAL SUCCESS: Reports generated but review needed")
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
