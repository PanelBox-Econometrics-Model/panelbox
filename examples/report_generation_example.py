"""
PanelBox Report Generation - Complete Example.

This example demonstrates the full report generation workflow:
1. Loading panel data
2. Running validation tests
3. Generating HTML, LaTeX, and Markdown reports
4. Exporting in multiple formats

Author: PanelBox Team
Date: January 2026
"""

from pathlib import Path

import numpy as np
import pandas as pd

# PanelBox imports
from panelbox.data import PanelData
from panelbox.models import FixedEffects

# Report imports
from panelbox.report import ReportManager
from panelbox.report.exporters import HTMLExporter, LaTeXExporter, MarkdownExporter
from panelbox.report.validation_transformer import ValidationTransformer
from panelbox.validation import HausmanTest, MundlakTest, PesaranCDTest, WooldridgeTest
from panelbox.validation.validation_report import ValidationReport


def create_sample_panel_data(n_entities=100, n_periods=10, seed=42):
    """
    Create sample panel data for demonstration.

    Parameters
    ----------
    n_entities : int
        Number of cross-sectional units
    n_periods : int
        Number of time periods
    seed : int
        Random seed for reproducibility

    Returns
    -------
    PanelData
        Sample panel dataset
    """
    np.random.seed(seed)

    # Generate entity and time indices
    entities = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)

    # Generate data
    # Entity fixed effects
    entity_effects = np.repeat(np.random.normal(0, 1, n_entities), n_periods)

    # Explanatory variables
    x1 = np.random.normal(0, 1, n_entities * n_periods)
    x2 = np.random.normal(0, 1, n_entities * n_periods)
    x3 = np.random.normal(0, 1, n_entities * n_periods)

    # Outcome variable (with entity fixed effects)
    y = (
        3.0  # Intercept
        + 1.5 * x1  # Effect of x1
        + -0.8 * x2  # Effect of x2
        + 0.5 * x3  # Effect of x3
        + entity_effects  # Entity fixed effects
        + np.random.normal(0, 0.5, n_entities * n_periods)  # Random error
    )

    # Create DataFrame
    df = pd.DataFrame(
        {"entity_id": entities, "time_id": time, "y": y, "x1": x1, "x2": x2, "x3": x3}
    )

    # Create PanelData
    panel = PanelData(df, entity_col="entity_id", time_col="time_id")

    return panel


def run_validation_tests(panel, model):
    """
    Run comprehensive validation tests on panel data model.

    Parameters
    ----------
    panel : PanelData
        Panel dataset
    model : fitted model
        Estimated panel data model

    Returns
    -------
    ValidationReport
        Complete validation report
    """
    print("Running validation tests...")

    # Specification tests
    print("  - Hausman Test")
    hausman = HausmanTest()
    hausman_result = hausman.test(panel, "y ~ x1 + x2 + x3")

    print("  - Mundlak Test")
    mundlak = MundlakTest()
    mundlak_result = mundlak.test(panel, "y ~ x1 + x2 + x3")

    # Serial correlation tests
    print("  - Wooldridge Test")
    wooldridge = WooldridgeTest()
    wooldridge_result = wooldridge.test(panel, "y ~ x1 + x2 + x3")

    # Cross-sectional dependence tests
    print("  - Pesaran CD Test")
    pesaran = PesaranCDTest()
    pesaran_result = pesaran.test(panel, "y ~ x1 + x2 + x3")

    # Create validation report
    model_info = {
        "model_type": "Fixed Effects",
        "formula": "y ~ x1 + x2 + x3",
        "nobs": len(panel.data),
        "n_entities": panel.n_entities,
        "n_periods": panel.n_periods,
        "balanced": panel.is_balanced,
    }

    report = ValidationReport(
        model_info=model_info,
        specification_tests={"Hausman Test": hausman_result, "Mundlak Test": mundlak_result},
        serial_tests={"Wooldridge Test": wooldridge_result},
        cd_tests={"Pesaran CD Test": pesaran_result},
    )

    print("✓ Validation tests completed")
    print()

    return report


def generate_html_report(validation_report, output_dir):
    """
    Generate interactive HTML report.

    Parameters
    ----------
    validation_report : ValidationReport
        Validation report
    output_dir : Path
        Output directory

    Returns
    -------
    Path
        Path to HTML report
    """
    print("Generating HTML report...")

    # Initialize report manager
    report_mgr = ReportManager(minify=False)

    # Transform validation report
    transformer = ValidationTransformer(validation_report)
    validation_data = transformer.transform(include_charts=True)

    # Generate HTML
    html = report_mgr.generate_validation_report(
        validation_data=validation_data,
        interactive=True,
        title="Panel Data Validation Report",
        subtitle="Comprehensive validation of Fixed Effects model",
    )

    # Export HTML
    exporter = HTMLExporter()
    html_path = exporter.export(html, output_dir / "validation_report.html", overwrite=True)

    print(f"✓ HTML report saved to: {html_path}")
    print()

    return html_path


def generate_latex_tables(validation_data, output_dir):
    """
    Generate LaTeX tables for academic papers.

    Parameters
    ----------
    validation_data : dict
        Transformed validation data
    output_dir : Path
        Output directory

    Returns
    -------
    list of Path
        Paths to LaTeX files
    """
    print("Generating LaTeX tables...")

    exporter = LaTeXExporter(table_style="booktabs")

    # Validation tests table
    tests = validation_data["tests"]
    latex = exporter.export_validation_tests(
        tests, caption="Panel Data Validation Test Results", label="tab:validation"
    )

    latex_path = exporter.save(
        latex, output_dir / "validation_tests.tex", overwrite=True, add_preamble=False
    )

    print(f"✓ LaTeX table saved to: {latex_path}")
    print()

    return [latex_path]


def generate_markdown_report(validation_data, output_dir):
    """
    Generate Markdown report for GitHub.

    Parameters
    ----------
    validation_data : dict
        Transformed validation data
    output_dir : Path
        Output directory

    Returns
    -------
    Path
        Path to Markdown report
    """
    print("Generating Markdown report...")

    exporter = MarkdownExporter(include_toc=True, github_flavor=True)

    # Generate Markdown
    markdown = exporter.export_validation_report(
        validation_data, title="Panel Data Validation Report"
    )

    # Save
    md_path = exporter.save(markdown, output_dir / "VALIDATION_REPORT.md", overwrite=True)

    print(f"✓ Markdown report saved to: {md_path}")
    print()

    return md_path


def main():
    """Run complete report generation example."""
    print("=" * 70)
    print("PanelBox Report Generation - Complete Example")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path("output/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create sample panel data
    print("Step 1: Creating sample panel data...")
    panel = create_sample_panel_data(n_entities=100, n_periods=10)
    print(f"✓ Panel data created: {panel.n_entities} entities, {panel.n_periods} periods")
    print()

    # Step 2: Estimate Fixed Effects model
    print("Step 2: Estimating Fixed Effects model...")
    model = FixedEffects("y ~ x1 + x2 + x3", panel)
    results = model.fit()
    print("✓ Model estimated")
    print()

    # Step 3: Run validation tests
    validation_report = run_validation_tests(panel, results)

    # Print text summary
    print("Validation Summary:")
    print(validation_report.summary(verbose=False))
    print()

    # Step 4: Transform for report generation
    print("Step 4: Transforming validation results...")
    transformer = ValidationTransformer(validation_report)
    validation_data = transformer.transform(include_charts=True)
    print("✓ Data transformed for reporting")
    print()

    # Step 5: Generate reports in multiple formats
    print("Step 5: Generating reports in multiple formats...")
    print()

    # HTML (interactive)
    html_path = generate_html_report(validation_report, output_dir)

    # LaTeX (for papers)
    latex_paths = generate_latex_tables(validation_data, output_dir)

    # Markdown (for GitHub)
    md_path = generate_markdown_report(validation_data, output_dir)

    # Summary
    print("=" * 70)
    print("Report Generation Complete!")
    print("=" * 70)
    print()
    print("Generated files:")
    print(f"  📄 HTML:     {html_path}")
    print(f"  📄 LaTeX:    {latex_paths[0]}")
    print(f"  📄 Markdown: {md_path}")
    print()
    print("Next steps:")
    print("  1. Open the HTML report in your browser for interactive exploration")
    print("  2. Include LaTeX tables in your academic paper")
    print("  3. Add Markdown report to your GitHub repository")
    print()

    # File size information
    html_size = html_path.stat().st_size / 1024
    print(f"HTML report size: {html_size:.1f} KB (self-contained)")
    print()


if __name__ == "__main__":
    main()
