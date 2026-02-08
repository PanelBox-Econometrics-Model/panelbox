"""
Sprint 3 - Complete Workflow Test
==================================

Tests the complete workflow:
1. Create PanelExperiment
2. Fit multiple models
3. Create Result container
4. Generate HTML report using BaseResult.save_html()
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Import Sprint 3 components
from panelbox.experiment import PanelExperiment
from panelbox.experiment.results import BaseResult

# Import existing components
from panelbox.report.validation_transformer import ValidationTransformer

np.random.seed(42)


class ValidationResultContainer(BaseResult):
    """
    Concrete implementation of BaseResult for validation tests.

    This is a simple example showing how to create a Result container.
    """

    def __init__(self, validation_report, model_results, **kwargs):
        super().__init__(**kwargs)
        self.validation_report = validation_report
        self.model_results = model_results

    def to_dict(self):
        """Convert to dictionary using ValidationTransformer."""
        transformer = ValidationTransformer(self.validation_report)
        return transformer.transform(include_charts=True, use_new_visualization=True)

    def summary(self):
        """Generate text summary."""
        return self.validation_report.summary(verbose=True)


def create_panel_data():
    """Create sample panel data."""
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
    print("SPRINT 3: COMPLETE WORKFLOW TEST")
    print("=" * 80)
    print()

    # PHASE 1: PANEL EXPERIMENT
    print("PHASE 1: PANEL EXPERIMENT")
    print("-" * 80)
    print()

    # 1. Create panel data
    data = create_panel_data()
    print()

    # 2. Create PanelExperiment
    print("Creating PanelExperiment...")
    experiment = PanelExperiment(
        data=data, formula="output ~ capital + labor", entity_col="firm", time_col="year"
    )
    print(f"  ✅ Experiment created")
    print(f"  {experiment}")
    print()

    # 3. Fit multiple models
    print("Fitting models...")
    experiment.fit_model("pooled_ols", name="pooled")
    experiment.fit_model("fixed_effects", name="fe", cov_type="clustered")
    experiment.fit_model("random_effects", name="re")
    print(f"  ✅ All models fitted")
    print(f"  - Models: {', '.join(experiment.list_models())}")
    print()

    # PHASE 2: VALIDATION & RESULT CONTAINER
    print("PHASE 2: VALIDATION & RESULT CONTAINER")
    print("-" * 80)
    print()

    # 4. Get a model and run validation
    print("Running validation tests on Fixed Effects model...")
    fe_model = experiment.get_model("fe")
    validation = fe_model.validate(tests="default", alpha=0.05, verbose=False)

    # Count total tests
    total_tests = (
        len(validation.specification_tests or {})
        + len(validation.serial_tests or {})
        + len(validation.het_tests or {})
        + len(validation.cd_tests or {})
    )
    print(f"  ✅ Validation completed: {total_tests} tests run")
    print()

    # 5. Create Result container
    print("Creating ValidationResultContainer...")
    result_container = ValidationResultContainer(
        validation_report=validation,
        model_results=fe_model,
        metadata={
            "experiment_id": "sprint3_test",
            "model_name": "fe",
            "formula": experiment.formula,
        },
    )
    print(f"  ✅ Result container created")
    print(f"  {result_container}")
    print()

    # PHASE 3: HTML REPORT GENERATION
    print("PHASE 3: HTML REPORT GENERATION")
    print("-" * 80)
    print()

    # 6. Save as JSON
    print("Saving result as JSON...")
    json_path = Path("/home/guhaase/projetos/panelbox/sprint3_validation_result.json")
    result_container.save_json(str(json_path))
    print(f"  ✅ JSON saved: {json_path}")
    print(f"  - File size: {json_path.stat().st_size / 1024:.1f} KB")
    print()

    # 7. Generate HTML report using BaseResult.save_html()
    print("Generating HTML report via BaseResult.save_html()...")
    html_path = Path("/home/guhaase/projetos/panelbox/sprint3_validation_report.html")

    saved_html = result_container.save_html(
        file_path=str(html_path),
        test_type="validation",
        theme="professional",
        title="Sprint 3 - Complete Workflow Validation Report",
    )

    print(f"  ✅ HTML report generated: {saved_html}")
    print(f"  - File size: {saved_html.stat().st_size / 1024:.1f} KB")
    print()

    # PHASE 4: SUMMARY
    print("=" * 80)
    print("✅ SPRINT 3 COMPLETE WORKFLOW TEST: SUCCESS!")
    print("=" * 80)
    print()
    print("Workflow Summary:")
    print("-" * 80)
    print(f"1. PanelExperiment:")
    print(f"   • Formula: {experiment.formula}")
    print(f"   • Observations: {len(experiment.data)}")
    print(f"   • Models fitted: {len(experiment.list_models())}")
    print(f"   • Model types: {', '.join(experiment.list_models())}")
    print()
    print(f"2. Validation:")
    print(f"   • Model tested: Fixed Effects")
    print(f"   • Total tests: {total_tests}")
    print(
        f"   • Test categories: specification, serial correlation, heteroskedasticity, cross-section"
    )
    print()
    print(f"3. Result Container:")
    print(f"   • Type: ValidationResultContainer")
    print(f"   • Inherits from: BaseResult")
    print(f"   • JSON export: ✅")
    print(f"   • HTML export: ✅")
    print()
    print(f"4. Reports Generated:")
    print(f"   • JSON: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")
    print(f"   • HTML: {saved_html} ({saved_html.stat().st_size / 1024:.1f} KB)")
    print()
    print("=" * 80)
    print()
    print("Sprint 3 Deliverables:")
    print("  ✅ PanelExperiment - Factory pattern for model fitting")
    print("  ✅ BaseResult - Abstract base with save_html() and save_json()")
    print("  ✅ Result Container - Concrete implementation example")
    print("  ✅ Complete Workflow - Experiment → Model → Validation → Report")
    print("=" * 80)


if __name__ == "__main__":
    main()
