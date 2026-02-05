"""
Minimal Report Generation Example.

Tests the report system with minimal mock data.
"""

from pathlib import Path

from panelbox.report.exporters import HTMLExporter, LaTeXExporter, MarkdownExporter


def main():
    print("Testing PanelBox Report Exporters...")
    print()

    # Create output directory
    output_dir = Path("output/test_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: HTML Export
    print("1. Testing HTMLExporter...")
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        h1 { color: #2563eb; }
    </style>
</head>
<body>
    <h1>PanelBox Test Report</h1>
    <p>This is a test of the HTML exporter.</p>
    <p>Generated successfully!</p>
</body>
</html>
"""

    html_exporter = HTMLExporter()
    html_path = html_exporter.export(html_content, output_dir / "test.html", overwrite=True)
    print(f"   ✓ HTML exported to: {html_path}")

    # Test 2: LaTeX Export
    print()
    print("2. Testing LaTeXExporter...")
    latex_exporter = LaTeXExporter(table_style="booktabs")

    sample_tests = [
        {
            "category": "Specification",
            "name": "Hausman Test",
            "statistic": 12.5,
            "statistic_formatted": "12.500",
            "pvalue": 0.014,
            "pvalue_formatted": "0.0140",
            "df": 2,
            "result": "REJECT",
            "result_class": "reject",
            "significance": "**",
            "conclusion": "Reject null hypothesis",
        }
    ]

    latex = latex_exporter.export_validation_tests(
        sample_tests, caption="Test Results", label="tab:test"
    )

    latex_path = latex_exporter.save(latex, output_dir / "test.tex", overwrite=True)
    print(f"   ✓ LaTeX exported to: {latex_path}")

    # Test 3: Markdown Export
    print()
    print("3. Testing MarkdownExporter...")
    md_exporter = MarkdownExporter()

    sample_validation_data = {
        "model_info": {
            "model_type": "Fixed Effects",
            "formula": "y ~ x1 + x2",
            "nobs": 1000,
            "nobs_formatted": "1,000",
        },
        "tests": sample_tests,
        "summary": {
            "total_tests": 1,
            "total_passed": 0,
            "total_failed": 1,
            "pass_rate": 0.0,
            "pass_rate_formatted": "0.0%",
            "has_issues": True,
            "status_message": "Issues detected",
        },
        "recommendations": [],
    }

    markdown = md_exporter.export_validation_report(sample_validation_data, title="Test Report")

    md_path = md_exporter.save(markdown, output_dir / "test.md", overwrite=True)
    print(f"   ✓ Markdown exported to: {md_path}")

    print()
    print("=" * 60)
    print("All exporters working correctly!")
    print("=" * 60)
    print()
    print(f"Output directory: {output_dir.absolute()}")
    print()
    print("Files created:")
    print(f"  - {html_path.name}")
    print(f"  - {latex_path.name}")
    print(f"  - {md_path.name}")
    print()


if __name__ == "__main__":
    main()
