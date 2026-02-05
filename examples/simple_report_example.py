"""
PanelBox Report Generation - Simple Example.

This example demonstrates report generation using mock data,
without requiring the full PanelBox models to be implemented.

Author: PanelBox Team
Date: January 2026
"""

from pathlib import Path

from panelbox.report import ReportManager
from panelbox.report.exporters import HTMLExporter, LaTeXExporter, MarkdownExporter
from panelbox.report.validation_transformer import ValidationTransformer


def create_mock_validation_report():
    """
    Create a mock ValidationReport for demonstration.

    In production, this would come from:
    - panelbox.validation.ValidationReport
    - After running actual validation tests

    Returns
    -------
    dict
        Mock validation report structure
    """

    class MockTestResult:
        def __init__(self, statistic, pvalue, df, reject_null, conclusion, metadata=None):
            self.statistic = statistic
            self.pvalue = pvalue
            self.df = df
            self.reject_null = reject_null
            self.conclusion = conclusion
            self.metadata = metadata or {}

    class MockValidationReport:
        def __init__(self):
            # Model information
            self.model_info = {
                "model_type": "Fixed Effects",
                "formula": "y ~ x1 + x2 + x3",
                "nobs": 1000,
                "n_entities": 100,
                "n_periods": 10,
                "balanced": True,
            }

            # Specification tests
            self.specification_tests = {
                "Hausman Test": MockTestResult(
                    statistic=15.234,
                    pvalue=0.002,
                    df=3,
                    reject_null=True,
                    conclusion="Reject H0: Random Effects are inconsistent. Use Fixed Effects.",
                ),
                "Mundlak Test": MockTestResult(
                    statistic=12.876,
                    pvalue=0.005,
                    df=3,
                    reject_null=True,
                    conclusion="Reject H0: Entity means are correlated with unobserved effect.",
                ),
            }

            # Serial correlation tests
            self.serial_tests = {
                "Wooldridge Test": MockTestResult(
                    statistic=2.345,
                    pvalue=0.128,
                    df=1,
                    reject_null=False,
                    conclusion="Accept H0: No first-order autocorrelation detected.",
                ),
                "Baltagi-Wu Test": MockTestResult(
                    statistic=1.987,
                    pvalue=0.156,
                    df=None,
                    reject_null=False,
                    conclusion="Accept H0: No serial correlation detected.",
                ),
            }

            # Heteroskedasticity tests
            self.het_tests = {
                "Breusch-Pagan LM Test": MockTestResult(
                    statistic=18.456,
                    pvalue=0.001,
                    df=3,
                    reject_null=True,
                    conclusion="Reject H0: Heteroskedasticity detected. Use robust SE.",
                )
            }

            # Cross-sectional dependence tests
            self.cd_tests = {
                "Pesaran CD Test": MockTestResult(
                    statistic=3.789,
                    pvalue=0.0002,
                    df=None,
                    reject_null=True,
                    conclusion="Reject H0: Cross-sectional dependence detected.",
                ),
                "Frees Test": MockTestResult(
                    statistic=2.456,
                    pvalue=0.032,
                    df=None,
                    reject_null=True,
                    conclusion="Reject H0: Cross-sectional dependence detected.",
                ),
            }

    return MockValidationReport()


def main():
    """Run simple report generation example."""
    print("=" * 70)
    print("PanelBox Report Generation - Simple Example")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path("output/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    print()

    # Step 1: Create mock validation report
    print("Step 1: Creating mock validation report...")
    validation_report = create_mock_validation_report()
    print(f"✓ Mock validation report created")
    print(f"  - Model: {validation_report.model_info['model_type']}")
    print(f"  - Formula: {validation_report.model_info['formula']}")
    print(f"  - Observations: {validation_report.model_info['nobs']}")
    print()

    # Step 2: Transform validation data
    print("Step 2: Transforming validation data...")
    transformer = ValidationTransformer(validation_report)
    validation_data = transformer.transform(include_charts=True)
    print("✓ Data transformed for reporting")
    print(f"  - Total tests: {validation_data['summary']['total_tests']}")
    print(f"  - Tests passed: {validation_data['summary']['total_passed']}")
    print(f"  - Tests failed: {validation_data['summary']['total_failed']}")
    print()

    # Step 3: Generate HTML report
    print("Step 3: Generating HTML report...")
    report_mgr = ReportManager(minify=False)

    html = report_mgr.generate_validation_report(
        validation_data=validation_data,
        interactive=True,
        title="Panel Data Validation Report",
        subtitle="Example demonstration with mock data",
    )

    html_exporter = HTMLExporter()
    html_path = html_exporter.export(
        html, output_dir / "validation_report.html", overwrite=True, add_metadata=True
    )

    html_size = html_exporter.get_file_size(html)
    print(f"✓ HTML report generated: {html_path}")
    print(f"  - File size: {html_size['kb']:.1f} KB")
    print()

    # Step 4: Generate LaTeX table
    print("Step 4: Generating LaTeX table...")
    latex_exporter = LaTeXExporter(table_style="booktabs")

    latex = latex_exporter.export_validation_tests(
        validation_data["tests"],
        caption="Panel Data Validation Test Results",
        label="tab:validation",
    )

    latex_path = latex_exporter.save(
        latex, output_dir / "validation_tests.tex", overwrite=True, add_preamble=False
    )

    print(f"✓ LaTeX table generated: {latex_path}")
    print()

    # Step 5: Generate Markdown report
    print("Step 5: Generating Markdown report...")
    md_exporter = MarkdownExporter(include_toc=True, github_flavor=True)

    markdown = md_exporter.export_validation_report(
        validation_data, title="Panel Data Validation Report"
    )

    md_path = md_exporter.save(markdown, output_dir / "VALIDATION_REPORT.md", overwrite=True)

    print(f"✓ Markdown report generated: {md_path}")
    print()

    # Summary
    print("=" * 70)
    print("Report Generation Complete!")
    print("=" * 70)
    print()
    print("Generated files:")
    print(f"  📄 HTML (interactive): {html_path}")
    print(f"     Size: {html_size['kb']:.1f} KB (self-contained)")
    print()
    print(f"  📄 LaTeX table:        {latex_path}")
    print(f"     For academic papers")
    print()
    print(f"  📄 Markdown:           {md_path}")
    print(f"     For GitHub documentation")
    print()
    print("Next steps:")
    print("  1. Open HTML report in your browser:")
    print(f"     file://{html_path.absolute()}")
    print()
    print("  2. Include LaTeX table in your paper:")
    print(f"     \\input{{{latex_path.name}}}")
    print()
    print("  3. Add Markdown to your repository:")
    print(f"     git add {md_path}")
    print()

    # Show summary of test results
    print("=" * 70)
    print("Test Results Summary")
    print("=" * 70)
    print()
    summary = validation_data["summary"]
    print(f"Status: {summary['status_message']}")
    print(f"Pass Rate: {summary['pass_rate_formatted']}")
    print()
    print("Issues by category:")
    for category, count in summary["failed_by_category"].items():
        if count > 0:
            print(f"  - {category.replace('_', ' ').title()}: {count} test(s) failed")
    print()

    if validation_data["recommendations"]:
        print("Recommendations:")
        for i, rec in enumerate(validation_data["recommendations"], 1):
            print(f"  {i}. {rec['category']}: {rec['issue']}")
        print()

    print("✓ All reports generated successfully!")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
