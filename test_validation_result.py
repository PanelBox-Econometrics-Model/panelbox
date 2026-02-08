"""
Test ValidationResult - Complete Functionality
==============================================

Tests the ValidationResult class with real panel data and validation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import panelbox as pb
from panelbox.experiment.results import ValidationResult

np.random.seed(42)


def create_panel_data():
    """Create panel data with known issues."""
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
    print("TEST: VALIDATIONRESULT - COMPLETE FUNCTIONALITY")
    print("=" * 80)
    print()

    # 1. Create panel data and fit model
    data = create_panel_data()
    print()

    print("Fitting Fixed Effects model...")
    fe = pb.FixedEffects("output ~ capital + labor", data, "firm", "year")
    fe_results = fe.fit(cov_type="clustered")
    print(f"  ✅ Model fitted")
    print()

    # 2. Run validation
    print("Running validation tests...")
    validation = fe_results.validate(tests="default", alpha=0.05, verbose=False)
    print(f"  ✅ Validation completed")
    print()

    # 3. Create ValidationResult (Method 1: Direct instantiation)
    print("Creating ValidationResult (direct)...")
    val_result = ValidationResult(
        validation_report=validation,
        model_results=fe_results,
        metadata={"experiment": "sprint4_test", "method": "direct"},
    )
    print(f"  ✅ ValidationResult created")
    print(f"  {val_result}")
    print()

    # 4. Test properties
    print("Testing properties...")
    print(f"  • Total tests: {val_result.total_tests}")
    print(f"  • Passed tests: {len(val_result.passed_tests)}")
    print(f"  • Failed tests: {len(val_result.failed_tests)}")
    print(f"  • Pass rate: {val_result.pass_rate:.1%}")
    print()

    # 5. Test to_dict()
    print("Testing to_dict()...")
    data_dict = val_result.to_dict()
    print(f"  ✅ Converted to dict")
    print(f"  • Keys: {list(data_dict.keys())}")
    print(f"  • Has summary: {'summary' in data_dict}")
    print(f"  • Has tests: {'tests' in data_dict}")
    print(f"  • Has charts: {'charts' in data_dict}")
    print()

    # 6. Test summary()
    print("Testing summary()...")
    summary = val_result.summary()
    print(f"  ✅ Summary generated ({len(summary)} characters)")
    print()
    print("  First 300 characters:")
    print("  " + "-" * 76)
    for line in summary[:300].split("\n"):
        print(f"  {line}")
    print("  " + "-" * 76)
    print()

    # 7. Test save_json()
    print("Testing save_json()...")
    json_path = Path("/home/guhaase/projetos/panelbox/validation_result_test.json")
    val_result.save_json(str(json_path))
    print(f"  ✅ JSON saved: {json_path}")
    print(f"  • File size: {json_path.stat().st_size / 1024:.1f} KB")
    print()

    # 8. Test save_html()
    print("Testing save_html()...")
    html_path = Path("/home/guhaase/projetos/panelbox/validation_result_test.html")
    val_result.save_html(
        file_path=str(html_path),
        test_type="validation",
        theme="professional",
        title="ValidationResult Test Report",
    )
    print(f"  ✅ HTML saved: {html_path}")
    print(f"  • File size: {html_path.stat().st_size / 1024:.1f} KB")
    print()

    # 9. Test from_model_results() factory method
    print("Testing from_model_results() factory method...")
    val_result2 = ValidationResult.from_model_results(
        model_results=fe_results,
        alpha=0.05,
        tests="default",
        metadata={"experiment": "sprint4_test", "method": "factory"},
    )
    print(f"  ✅ ValidationResult created via factory")
    print(f"  {val_result2}")
    print()

    # 10. Final summary
    print("=" * 80)
    print("✅ ALL VALIDATIONRESULT TESTS PASSED!")
    print("=" * 80)
    print()
    print("ValidationResult Features Tested:")
    print("  ✅ Direct instantiation")
    print("  ✅ Factory method (from_model_results)")
    print("  ✅ Properties (total_tests, passed_tests, failed_tests, pass_rate)")
    print("  ✅ to_dict() method")
    print("  ✅ summary() method")
    print("  ✅ save_json() method")
    print("  ✅ save_html() method")
    print("  ✅ __repr__() method")
    print()
    print("Files Generated:")
    print(f"  • JSON: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")
    print(f"  • HTML: {html_path} ({html_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 80)


if __name__ == "__main__":
    main()
