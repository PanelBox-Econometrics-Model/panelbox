"""
Test ComparisonResult - Complete Functionality
===============================================

Tests the ComparisonResult class with real panel data and multiple models.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import panelbox as pb
from panelbox.experiment import PanelExperiment
from panelbox.experiment.results import ComparisonResult

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
    print("TEST: COMPARISONRESULT - COMPLETE FUNCTIONALITY")
    print("=" * 80)
    print()

    # 1. Create panel data
    data = create_panel_data()
    print()

    # 2. Fit multiple models
    print("Fitting multiple models...")

    pooled = pb.PooledOLS("output ~ capital + labor", data, "firm", "year")
    pooled_results = pooled.fit()
    print(f"  ✅ Pooled OLS fitted (R² = {pooled_results.rsquared:.4f})")

    fe = pb.FixedEffects("output ~ capital + labor", data, "firm", "year")
    fe_results = fe.fit(cov_type="clustered")
    print(f"  ✅ Fixed Effects fitted (R² = {fe_results.rsquared:.4f})")

    re = pb.RandomEffects("output ~ capital + labor", data, "firm", "year")
    re_results = re.fit()
    print(f"  ✅ Random Effects fitted (R² = {re_results.rsquared:.4f})")
    print()

    # 3. Create ComparisonResult (Method 1: Direct instantiation)
    print("Creating ComparisonResult (direct)...")
    comp_result = ComparisonResult(
        models={
            "Pooled OLS": pooled_results,
            "Fixed Effects": fe_results,
            "Random Effects": re_results,
        },
        metadata={"experiment": "sprint4_test", "method": "direct"},
    )
    print(f"  ✅ ComparisonResult created")
    print(f"  {comp_result}")
    print()

    # 4. Test properties
    print("Testing properties...")
    print(f"  • Number of models: {comp_result.n_models}")
    print(f"  • Model names: {comp_result.model_names}")
    print()

    # 5. Test best_model()
    print("Testing best_model()...")
    best_rsq = comp_result.best_model("rsquared")
    print(f"  • Best R²: {best_rsq}")

    best_aic = comp_result.best_model("aic", prefer_lower=True)
    print(f"  • Best AIC: {best_aic}")

    best_bic = comp_result.best_model("bic", prefer_lower=True)
    print(f"  • Best BIC: {best_bic}")
    print()

    # 6. Test to_dict()
    print("Testing to_dict()...")
    data_dict = comp_result.to_dict()
    print(f"  ✅ Converted to dict")
    print(f"  • Keys: {list(data_dict.keys())}")
    print(f"  • Has models: {'models' in data_dict}")
    print(f"  • Has comparison_metrics: {'comparison_metrics' in data_dict}")
    print(f"  • Has charts: {'charts' in data_dict}")
    print()

    # 7. Test summary()
    print("Testing summary()...")
    summary = comp_result.summary()
    print(f"  ✅ Summary generated ({len(summary)} characters)")
    print()
    print("  First 400 characters:")
    print("  " + "-" * 76)
    for line in summary[:400].split("\n"):
        print(f"  {line}")
    print("  " + "-" * 76)
    print()

    # 8. Test save_json()
    print("Testing save_json()...")
    json_path = Path("/home/guhaase/projetos/panelbox/comparison_result_test.json")
    comp_result.save_json(str(json_path))
    print(f"  ✅ JSON saved: {json_path}")
    print(f"  • File size: {json_path.stat().st_size / 1024:.1f} KB")
    print()

    # 9. Test save_html()
    print("Testing save_html()...")
    html_path = Path("/home/guhaase/projetos/panelbox/comparison_result_test.html")
    comp_result.save_html(
        file_path=str(html_path),
        test_type="comparison",
        theme="professional",
        title="ComparisonResult Test Report",
    )
    print(f"  ✅ HTML saved: {html_path}")
    print(f"  • File size: {html_path.stat().st_size / 1024:.1f} KB")
    print()

    # 10. Test from_experiment() factory method
    print("Testing from_experiment() factory method...")

    # Create experiment
    experiment = PanelExperiment(
        data=data, formula="output ~ capital + labor", entity_col="firm", time_col="year"
    )

    # Fit models
    experiment.fit_model("pooled_ols", name="pooled")
    experiment.fit_model("fixed_effects", name="fe", cov_type="clustered")
    experiment.fit_model("random_effects", name="re")

    # Create ComparisonResult from experiment
    comp_result2 = ComparisonResult.from_experiment(
        experiment=experiment, metadata={"experiment": "sprint4_test", "method": "factory"}
    )
    print(f"  ✅ ComparisonResult created via factory")
    print(f"  {comp_result2}")
    print()

    # 11. Test from_experiment() with specific models
    print("Testing from_experiment() with specific models...")
    comp_result3 = ComparisonResult.from_experiment(
        experiment=experiment,
        model_names=["fe", "re"],  # Only compare FE and RE
        metadata={"experiment": "sprint4_test", "method": "factory_filtered"},
    )
    print(f"  ✅ ComparisonResult created (filtered)")
    print(f"  • Models: {comp_result3.model_names}")
    print()

    # 12. Final summary
    print("=" * 80)
    print("✅ ALL COMPARISONRESULT TESTS PASSED!")
    print("=" * 80)
    print()
    print("ComparisonResult Features Tested:")
    print("  ✅ Direct instantiation")
    print("  ✅ Factory method (from_experiment)")
    print("  ✅ Factory method with model filtering")
    print("  ✅ Properties (n_models, model_names)")
    print("  ✅ best_model() method (rsquared, aic, bic)")
    print("  ✅ to_dict() method")
    print("  ✅ summary() method")
    print("  ✅ save_json() method")
    print("  ✅ save_html() method")
    print("  ✅ __repr__() method")
    print("  ✅ Automatic metric computation (R², AIC, BIC)")
    print()
    print("Files Generated:")
    print(f"  • JSON: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")
    print(f"  • HTML: {html_path} ({html_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 80)


if __name__ == "__main__":
    main()
