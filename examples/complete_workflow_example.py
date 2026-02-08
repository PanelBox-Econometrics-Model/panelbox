"""
Complete PanelBox Workflow Example
===================================

This example demonstrates the complete PanelBox workflow using the
Experiment Pattern introduced in Sprints 3-4.

Features demonstrated:
1. Creating a PanelExperiment
2. Fitting multiple models (pooled, fixed effects, random effects)
3. Validating a model and generating ValidationResult
4. Comparing models and generating ComparisonResult
5. Saving professional HTML reports
6. Getting text summaries

Author: PanelBox Development Team
Date: 2026-02-08
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Import using the new public API
import panelbox as pb

# Set random seed for reproducibility
np.random.seed(42)


def create_sample_data():
    """
    Create sample panel data for demonstration.

    Returns
    -------
    pd.DataFrame
        Panel dataset with columns: firm, year, output, capital, labor
    """
    print("Creating sample panel data...")

    n_firms = 50
    n_years = 10
    n_obs = n_firms * n_years

    data = pd.DataFrame(
        {
            "firm": np.repeat(range(1, n_firms + 1), n_years),
            "year": np.tile(range(2010, 2010 + n_years), n_firms),
        }
    )

    firm_effect = {i: np.random.normal(0, 5) for i in range(1, n_firms + 1)}
    data["firm_effect"] = data["firm"].map(firm_effect)
    data["capital"] = np.random.uniform(100, 1000, n_obs)
    data["labor"] = np.random.uniform(50, 500, n_obs)

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

    data["output"] = 10 + data["firm_effect"] + 0.5 * data["capital"] + 0.3 * data["labor"] + errors
    data = data.drop("firm_effect", axis=1)

    print(f"✅ Created panel: {n_firms} firms × {n_years} years = {n_obs} observations")
    return data


def main():
    """Run complete workflow example."""

    print("=" * 80)
    print("PANELBOX - COMPLETE WORKFLOW EXAMPLE")
    print("=" * 80)
    print()

    # STEP 1: Create Panel Data
    print("STEP 1: CREATE PANEL DATA")
    print("-" * 80)
    data = create_sample_data()
    print()

    # STEP 2: Create PanelExperiment
    print("=" * 80)
    print("STEP 2: CREATE PANELEXPERIMENT")
    print("-" * 80)

    experiment = pb.PanelExperiment(
        data=data, formula="output ~ capital + labor", entity_col="firm", time_col="year"
    )

    print("✅ PanelExperiment created")
    print(experiment)
    print()

    # STEP 3: Fit Multiple Models
    print("=" * 80)
    print("STEP 3: FIT MULTIPLE MODELS")
    print("-" * 80)

    experiment.fit_all_models(names=["pooled", "fe", "re"])
    print(f"✅ Fitted models: {experiment.list_models()}")
    print()

    # STEP 4: Validate Model
    print("=" * 80)
    print("STEP 4: VALIDATE MODEL")
    print("-" * 80)

    val_result = experiment.validate_model("fe", tests="default", alpha=0.05)
    print(
        f"✅ ValidationResult: {val_result.total_tests} tests, {val_result.pass_rate:.1%} pass rate"
    )
    print()

    # STEP 5: Save Validation Report
    print("=" * 80)
    print("STEP 5: SAVE VALIDATION REPORT")
    print("-" * 80)

    val_html = Path("example_validation.html")
    val_result.save_html(str(val_html), test_type="validation", theme="professional")
    print(f"✅ HTML saved: {val_html} ({val_html.stat().st_size / 1024:.1f} KB)")
    print()

    # STEP 6: Compare Models
    print("=" * 80)
    print("STEP 6: COMPARE MODELS")
    print("-" * 80)

    comp_result = experiment.compare_models()
    best_model = comp_result.best_model("rsquared")
    print(f"✅ Best model by R²: {best_model}")
    print()

    # STEP 7: Save Comparison Report
    print("=" * 80)
    print("STEP 7: SAVE COMPARISON REPORT")
    print("-" * 80)

    comp_html = Path("example_comparison.html")
    comp_result.save_html(str(comp_html), test_type="comparison", theme="professional")
    print(f"✅ HTML saved: {comp_html} ({comp_html.stat().st_size / 1024:.1f} KB)")
    print()

    # FINAL SUMMARY
    print("=" * 80)
    print("✅ COMPLETE WORKFLOW FINISHED!")
    print("=" * 80)
    print()
    print("Generated Files:")
    print(f"  • {val_html}")
    print(f"  • {comp_html}")
    print()


if __name__ == "__main__":
    main()
