"""
Sprint 2 - Complete Validation Report Test
===========================================

Generates a complete validation report using the existing ValidationTransformer.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Import PanelBox components
import panelbox as pb
from panelbox.report.report_manager import ReportManager
from panelbox.report.validation_transformer import ValidationTransformer

# Set random seed for reproducibility
np.random.seed(42)


def create_panel_data():
    """Create panel data with known issues."""
    print("Creating panel data...")

    n_firms = 50
    n_years = 10
    n_obs = n_firms * n_years

    # Create panel structure
    data = pd.DataFrame(
        {
            "firm": np.repeat(range(1, n_firms + 1), n_years),
            "year": np.tile(range(2010, 2010 + n_years), n_firms),
        }
    )

    # Add firm-specific fixed effects
    firm_effect = {i: np.random.normal(0, 5) for i in range(1, n_firms + 1)}
    data["firm_effect"] = data["firm"].map(firm_effect)

    # Generate regressors
    data["capital"] = np.random.uniform(100, 1000, n_obs)
    data["labor"] = np.random.uniform(50, 500, n_obs)

    # Generate errors with serial correlation
    rho = 0.5
    errors = np.zeros(n_obs)
    for i in range(n_firms):
        start_idx = i * n_years
        end_idx = (i + 1) * n_years
        firm_errors = np.zeros(n_years)
        firm_errors[0] = np.random.normal(0, 10)
        for t in range(1, n_years):
            firm_errors[t] = rho * firm_errors[t - 1] + np.random.normal(0, 10)
        errors[start_idx:end_idx] = firm_errors

    # Generate dependent variable
    data["output"] = 10 + data["firm_effect"] + 0.5 * data["capital"] + 0.3 * data["labor"] + errors
    data = data.drop("firm_effect", axis=1)

    print(f"  ✅ Created panel: {n_firms} firms, {n_years} years")
    return data


def main():
    print("=" * 80)
    print("SPRINT 2: COMPLETE VALIDATION REPORT TEST")
    print("=" * 80)
    print()

    # 1. Create panel data
    data = create_panel_data()
    print()

    # 2. Estimate Fixed Effects model
    print("Estimating Fixed Effects model...")
    fe = pb.FixedEffects("output ~ capital + labor", data, "firm", "year")
    fe_results = fe.fit(cov_type="clustered")
    print(f"  ✅ Model estimated")
    print()

    # 3. Run validation tests
    print("Running validation tests...")
    validation = fe_results.validate(tests="default", alpha=0.05, verbose=False)

    # Count total tests
    total_tests = (
        len(validation.specification_tests or {})
        + len(validation.serial_tests or {})
        + len(validation.het_tests or {})
        + len(validation.cd_tests or {})
    )
    print(f"  ✅ Tests completed: {total_tests} tests run")
    print()

    # 4. Transform validation results
    print("Transforming validation results...")
    transformer = ValidationTransformer(validation)
    report_data = transformer.transform(include_charts=False)  # Charts optional for now
    print(f"  ✅ Data transformed")
    print()

    # 5. Generate HTML report
    print("Generating HTML report...")
    report_mgr = ReportManager(enable_cache=True, minify=False)

    html = report_mgr.generate_report(
        report_type="validation",
        template="validation/interactive/index.html",
        context=report_data,
        embed_assets=True,
        include_plotly=True,
    )

    print(f"  ✅ Report generated: {len(html):,} characters")
    print()

    # 6. Save report
    print("Saving report...")
    output_path = Path("/home/guhaase/projetos/panelbox/sprint2_complete_validation_report.html")
    output_path.write_text(html, encoding="utf-8")
    file_size_kb = output_path.stat().st_size / 1024

    print(f"  ✅ Report saved: {output_path}")
    print(f"  ✅ File size: {file_size_kb:.1f} KB")
    print()

    # 7. Summary
    print("=" * 80)
    print("✅ COMPLETE VALIDATION REPORT GENERATED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Report Details:")
    print(f"  • Total tests: {report_data['summary']['total_tests']}")
    print(f"  • Tests passed: {report_data['summary']['total_passed']}")
    print(f"  • Tests failed: {report_data['summary']['total_failed']}")
    print(f"  • Pass rate: {report_data['summary']['pass_rate_formatted']}")
    print(f"  • Recommendations: {len(report_data.get('recommendations', []))}")
    print(f"  • Model: {report_data['model_info'].get('model_type', 'N/A')}")
    print(f"  • Observations: {report_data['model_info'].get('nobs_formatted', 'N/A')}")
    print()
    print(f"Open the report: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
