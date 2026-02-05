"""
Test PanelBox Random Effects vs R plm package

Compares numerical results between PanelBox's RandomEffects and R's plm random model.

Dataset: Grunfeld (200 observations, 10 firms, 20 years)
Model: invest ~ value + capital (GLS transformation)

Expected tolerance: < 1e-6 for coefficients and standard errors
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import panelbox as pb


def load_grunfeld_data():
    """Load Grunfeld dataset - use R version for comparison"""
    try:
        # Use R's Grunfeld dataset for exact comparison
        r_data_path = os.path.join(os.path.dirname(__file__), "grunfeld_r.csv")
        if os.path.exists(r_data_path):
            data = pd.read_csv(r_data_path)
            # Rename 'inv' to 'invest' for PanelBox
            if "inv" in data.columns:
                data = data.rename(columns={"inv": "invest"})
            return data
        else:
            # Fallback to PanelBox dataset
            print("Warning: R dataset not found, using PanelBox default (may differ)")
            data = pb.load_grunfeld()
            return data
    except Exception as e:
        print(f"Error loading Grunfeld data: {e}")
        sys.exit(1)


def main():
    print("=" * 80)
    print("BENCHMARK: Random Effects - PanelBox vs R plm")
    print("=" * 80)
    print()

    # Load data
    data = load_grunfeld_data()

    print(f"Dataset: {len(data)} observations, {data['firm'].nunique()} firms")
    print()

    # Estimate with PanelBox
    print("-" * 80)
    print("PanelBox Estimation")
    print("-" * 80)
    print()

    model = pb.RandomEffects(
        formula="invest ~ value + capital", data=data, entity_col="firm", time_col="year"
    )

    results = model.fit()
    print(results.summary())
    print()

    # R plm reference values
    # From random.R output
    plm_results = {
        "coef": {
            "Intercept": -57.834415,  # From R plm random.R
            "value": 0.109781,  # From R plm random.R
            "capital": 0.308113,  # From R plm random.R
        },
        "se": {
            "Intercept": 28.898935,  # From R plm random.R
            "value": 0.010493,  # From R plm random.R
            "capital": 0.017180,  # From R plm random.R
        },
        "r2": 0.7695,  # From R plm random.R
        "sigma_u": 84.20,  # From R plm random.R (sqrt(7089.80))
        "sigma_e": 52.77,  # From R plm random.R (sqrt(2784.46))
        "theta": 0.8612,  # From R plm random.R
        "n_obs": 200,
        "n_groups": 10,
    }

    # Compare results
    print("-" * 80)
    print("Comparison with R plm")
    print("-" * 80)
    print()

    print("Coefficients:")
    print(f"{'Variable':<15} {'PanelBox':>12} {'R plm':>12} {'Diff':>12} {'Rel Error':>12}")
    print("-" * 67)

    all_passed = True
    tolerance = 1e-6

    for var in results.params.index:
        panelbox_coef = results.params[var]
        plm_coef = plm_results["coef"].get(var, np.nan)

        if not np.isnan(plm_coef):
            diff = panelbox_coef - plm_coef
            rel_error = abs(diff / plm_coef) if plm_coef != 0 else abs(diff)
            status = "✓" if abs(diff) < tolerance else "✗"
            if abs(diff) >= tolerance:
                all_passed = False
            print(
                f"{var:<15} {panelbox_coef:12.7f} {plm_coef:12.7f} "
                f"{diff:12.2e} {rel_error:12.2e} {status}"
            )

    print()
    print("Standard Errors:")
    print(f"{'Variable':<15} {'PanelBox':>12} {'R plm':>12} {'Diff':>12} {'Rel Error':>12}")
    print("-" * 67)

    for var in results.std_errors.index:
        panelbox_se = results.std_errors[var]
        plm_se = plm_results["se"].get(var, np.nan)

        if not np.isnan(plm_se):
            diff = panelbox_se - plm_se
            rel_error = abs(diff / plm_se) if plm_se != 0 else abs(diff)
            status = "✓" if abs(diff) < tolerance else "✗"
            if abs(diff) >= tolerance:
                all_passed = False
            print(
                f"{var:<15} {panelbox_se:12.7f} {plm_se:12.7f} "
                f"{diff:12.2e} {rel_error:12.2e} {status}"
            )

    print()
    print("=" * 80)

    if all_passed:
        print("✓ All comparisons passed")
        return 0
    else:
        print("✗ Some comparisons failed - Update plm_results from random.R output")
        return 1


if __name__ == "__main__":
    sys.exit(main())
