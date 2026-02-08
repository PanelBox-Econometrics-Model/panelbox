"""
Test Sprint 4 - Complete End-to-End Workflow
============================================

This test demonstrates the complete Sprint 4 functionality:
- ValidationResult
- ComparisonResult
- Enhanced PanelExperiment (validate_model, compare_models, fit_all_models)
- Complete workflow from experiment to reports
"""

from pathlib import Path

import numpy as np
import pandas as pd

import panelbox as pb
from panelbox.experiment import PanelExperiment
from panelbox.experiment.results import ComparisonResult, ValidationResult

np.random.seed(42)


def create_panel_data():
    """Create panel data with known properties."""
    print("Creating panel data...")

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

    print(f"  ✅ Created panel: {n_firms} firms, {n_years} years")
    return data


def main():
    print("=" * 80)
    print("TEST: SPRINT 4 - COMPLETE END-TO-END WORKFLOW")
    print("=" * 80)
    print()

    # Phase 1: Create Experiment
    print("=" * 80)
    print("PHASE 1: CREATE PANELEXPERIMENT")
    print("=" * 80)
    print()

    data = create_panel_data()
    print()

    experiment = PanelExperiment(
        data=data, formula="output ~ capital + labor", entity_col="firm", time_col="year"
    )
    print("✅ PanelExperiment created")
    print(f"  {experiment}")
    print()

    # Phase 2: Fit Multiple Models (New: fit_all_models)
    print("=" * 80)
    print("PHASE 2: FIT MULTIPLE MODELS (fit_all_models)")
    print("=" * 80)
    print()

    fitted_models = experiment.fit_all_models(
        model_types=["pooled_ols", "fixed_effects", "random_effects"], names=["pooled", "fe", "re"]
    )
    print(f"✅ All models fitted: {list(fitted_models.keys())}")
    print()

    # Phase 3: Validate a Model (New: validate_model)
    print("=" * 80)
    print("PHASE 3: VALIDATE MODEL (validate_model)")
    print("=" * 80)
    print()

    val_result = experiment.validate_model("fe", tests="default", alpha=0.05)
    print(f"✅ ValidationResult created via experiment.validate_model()")
    print(f"  {val_result}")
    print()

    # Phase 4: Save Validation Report
    print("=" * 80)
    print("PHASE 4: SAVE VALIDATION REPORT")
    print("=" * 80)
    print()

    val_json = Path("/home/guhaase/projetos/panelbox/sprint4_validation.json")
    val_result.save_json(str(val_json))
    print(f"  ✅ JSON: {val_json} ({val_json.stat().st_size / 1024:.1f} KB)")

    val_html = Path("/home/guhaase/projetos/panelbox/sprint4_validation.html")
    val_result.save_html(
        file_path=str(val_html),
        test_type="validation",
        theme="professional",
        title="Sprint 4 Validation Report",
    )
    print(f"  ✅ HTML: {val_html} ({val_html.stat().st_size / 1024:.1f} KB)")
    print()

    # Phase 5: Compare Models (New: compare_models)
    print("=" * 80)
    print("PHASE 5: COMPARE MODELS (compare_models)")
    print("=" * 80)
    print()

    comp_result = experiment.compare_models()  # Compare all models
    print(f"✅ ComparisonResult created via experiment.compare_models()")
    print(f"  {comp_result}")
    print()

    print(f"  Best model by R²: {comp_result.best_model('rsquared')}")
    print()

    # Phase 6: Save Comparison Report
    print("=" * 80)
    print("PHASE 6: SAVE COMPARISON REPORT")
    print("=" * 80)
    print()

    comp_json = Path("/home/guhaase/projetos/panelbox/sprint4_comparison.json")
    comp_result.save_json(str(comp_json))
    print(f"  ✅ JSON: {comp_json} ({comp_json.stat().st_size / 1024:.1f} KB)")

    comp_html = Path("/home/guhaase/projetos/panelbox/sprint4_comparison.html")
    comp_result.save_html(
        file_path=str(comp_html),
        test_type="comparison",
        theme="professional",
        title="Sprint 4 Comparison Report",
    )
    print(f"  ✅ HTML: {comp_html} ({comp_html.stat().st_size / 1024:.1f} KB)")
    print()

    # Phase 7: Alternative Workflows
    print("=" * 80)
    print("PHASE 7: ALTERNATIVE WORKFLOWS")
    print("=" * 80)
    print()

    print("Testing ValidationResult.from_model_results()...")
    fe_model = experiment.get_model("fe")
    val_result2 = ValidationResult.from_model_results(
        model_results=fe_model, alpha=0.05, tests="default", metadata={"method": "alternative"}
    )
    print(f"  ✅ ValidationResult created: {val_result2.total_tests} tests")
    print()

    print("Testing ComparisonResult.from_experiment() with filter...")
    comp_result2 = ComparisonResult.from_experiment(
        experiment=experiment, model_names=["fe", "re"]  # Only compare these two
    )
    print(f"  ✅ ComparisonResult created: {comp_result2.n_models} models")
    print(f"  • Models: {comp_result2.model_names}")
    print()

    # Phase 8: Summary Statistics
    print("=" * 80)
    print("PHASE 8: SUMMARY STATISTICS")
    print("=" * 80)
    print()

    print("Validation Summary (first 300 chars):")
    print("-" * 80)
    summary = val_result.summary()
    for line in summary[:300].split("\n"):
        print(f"  {line}")
    print("-" * 80)
    print()

    print("Comparison Summary (first 400 chars):")
    print("-" * 80)
    comp_summary = comp_result.summary()
    for line in comp_summary[:400].split("\n"):
        print(f"  {line}")
    print("-" * 80)
    print()

    # Final Summary
    print("=" * 80)
    print("✅ SPRINT 4 COMPLETE END-TO-END WORKFLOW - SUCCESS!")
    print("=" * 80)
    print()

    print("Sprint 4 Components Tested:")
    print("  ✅ PanelExperiment.fit_all_models() - Fit multiple models at once")
    print("  ✅ PanelExperiment.validate_model() - Validate and get ValidationResult")
    print("  ✅ PanelExperiment.compare_models() - Compare and get ComparisonResult")
    print("  ✅ ValidationResult - Complete validation workflow")
    print("  ✅ ComparisonResult - Complete comparison workflow")
    print("  ✅ save_json() and save_html() for both result types")
    print("  ✅ summary() for both result types")
    print("  ✅ Alternative factory methods")
    print()

    print("Files Generated:")
    print(f"  • Validation JSON: {val_json} ({val_json.stat().st_size / 1024:.1f} KB)")
    print(f"  • Validation HTML: {val_html} ({val_html.stat().st_size / 1024:.1f} KB)")
    print(f"  • Comparison JSON: {comp_json} ({comp_json.stat().st_size / 1024:.1f} KB)")
    print(f"  • Comparison HTML: {comp_html} ({comp_html.stat().st_size / 1024:.1f} KB)")
    print()

    print("Complete Workflow:")
    print("  1. Create PanelExperiment")
    print("  2. Fit multiple models (fit_all_models)")
    print("  3. Validate specific model (validate_model → ValidationResult)")
    print("  4. Save validation report (save_json, save_html)")
    print("  5. Compare all models (compare_models → ComparisonResult)")
    print("  6. Save comparison report (save_json, save_html)")
    print("  7. Use alternative factory methods")
    print("  8. Generate summaries")
    print()

    print("=" * 80)
    print("SPRINT 4 - READY FOR REVIEW")
    print("=" * 80)


if __name__ == "__main__":
    main()
