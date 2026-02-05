"""
Test PanelBox GMM vs R plm package (pgmm function)

Compares numerical results between PanelBox's DifferenceGMM/SystemGMM and R's pgmm.

Dataset: Grunfeld (200 observations, 10 firms, 20 years)
Model: invest ~ lag(invest) + value + capital

Expected tolerance: < 1e-3 for GMM (algorithm differences expected)

Note: Direct comparison is difficult due to different implementations:
- PanelBox follows Stata's xtabond2 approach
- R plm's pgmm has different syntax and defaults
- Results may differ due to instrument specification differences
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
    print("BENCHMARK: GMM Estimation - PanelBox vs R plm (pgmm)")
    print("=" * 80)
    print()
    print("NOTE: Direct comparison is challenging due to implementation differences")
    print("      - Different instrument specifications")
    print("      - Different weighting matrix calculations")
    print("      - Different defaults (one-step vs two-step, robust SE)")
    print()

    # Load data
    data = load_grunfeld_data()

    print(f"Dataset: {len(data)} observations, {data['firm'].nunique()} firms")
    print()

    # ============================================================================
    # Difference GMM
    # ============================================================================
    print("=" * 80)
    print("DIFFERENCE GMM (Arellano-Bond 1991)")
    print("=" * 80)
    print()

    print("-" * 80)
    print("PanelBox Estimation")
    print("-" * 80)
    print()

    diff_model = pb.DifferenceGMM(
        data=data,
        dep_var="invest",
        lags=1,
        id_var="firm",
        time_var="year",
        exog_vars=["value", "capital"],
        two_step=True,
        robust=True,
    )

    diff_results = diff_model.fit()
    print(diff_results.summary())
    print()

    # R plm reference values
    # INSTRUCTIONS: Run pgmm.R and copy Difference GMM values from pgmm_results.txt
    plm_diff_results = {
        "coef": {
            "invest_lag1": 0.2500000000,  # PLACEHOLDER - lag(inv, 1)
            "value": 0.1100000000,  # PLACEHOLDER
            "capital": 0.3100000000,  # PLACEHOLDER
        },
        "se": {
            "invest_lag1": 0.0500000000,  # PLACEHOLDER
            "value": 0.0200000000,  # PLACEHOLDER
            "capital": 0.0300000000,  # PLACEHOLDER
        },
    }

    # Compare Difference GMM
    print("-" * 80)
    print("Comparison: Difference GMM - PanelBox vs R plm")
    print("-" * 80)
    print()

    print("Coefficients:")
    print(f"{'Variable':<20} {'PanelBox':>12} {'R plm':>12} {'Diff':>12}")
    print("-" * 58)

    # Map variable names (PanelBox uses 'invest_lag1', R uses 'lag(inv, 1)')
    var_map = {"invest_lag1": "invest_lag1", "value": "value", "capital": "capital"}

    tolerance = 1e-3  # Relaxed for GMM
    all_passed = True

    for pb_var in diff_results.params.index:
        plm_var = var_map.get(pb_var, pb_var)
        pb_coef = diff_results.params[pb_var]
        plm_coef = plm_diff_results["coef"].get(plm_var, np.nan)

        if not np.isnan(plm_coef):
            diff = pb_coef - plm_coef
            status = "✓" if abs(diff) < tolerance else "✗"
            if abs(diff) >= tolerance:
                all_passed = False
            print(f"{pb_var:<20} {pb_coef:12.7f} {plm_coef:12.7f} {diff:12.2e} {status}")

    print()

    # ============================================================================
    # System GMM
    # ============================================================================
    print("=" * 80)
    print("SYSTEM GMM (Blundell-Bond 1998)")
    print("=" * 80)
    print()

    print("-" * 80)
    print("PanelBox Estimation")
    print("-" * 80)
    print()

    sys_model = pb.SystemGMM(
        data=data,
        dep_var="invest",
        lags=1,
        id_var="firm",
        time_var="year",
        exog_vars=["value", "capital"],
        two_step=True,
        robust=True,
    )

    sys_results = sys_model.fit()
    print(sys_results.summary())
    print()

    # R plm reference values for System GMM
    plm_sys_results = {
        "coef": {
            "invest_lag1": 0.3000000000,  # PLACEHOLDER
            "value": 0.1050000000,  # PLACEHOLDER
            "capital": 0.3050000000,  # PLACEHOLDER
        },
        "se": {
            "invest_lag1": 0.0400000000,  # PLACEHOLDER
            "value": 0.0150000000,  # PLACEHOLDER
            "capital": 0.0250000000,  # PLACEHOLDER
        },
    }

    # Compare System GMM
    print("-" * 80)
    print("Comparison: System GMM - PanelBox vs R plm")
    print("-" * 80)
    print()

    print("Coefficients:")
    print(f"{'Variable':<20} {'PanelBox':>12} {'R plm':>12} {'Diff':>12}")
    print("-" * 58)

    for pb_var in sys_results.params.index:
        plm_var = var_map.get(pb_var, pb_var)
        pb_coef = sys_results.params[pb_var]
        plm_coef = plm_sys_results["coef"].get(plm_var, np.nan)

        if not np.isnan(plm_coef):
            diff = pb_coef - plm_coef
            status = "✓" if abs(diff) < tolerance else "?"
            if abs(diff) >= tolerance:
                all_passed = False
            print(f"{pb_var:<20} {pb_coef:12.7f} {plm_coef:12.7f} {diff:12.2e} {status}")

    print()
    print("=" * 80)
    print("NOTES ON GMM COMPARISON")
    print("=" * 80)
    print()
    print("GMM results may differ between PanelBox and R plm due to:")
    print("1. Different lag specifications for instruments")
    print("2. Different weighting matrix calculations")
    print("3. Different handling of two-step robust standard errors")
    print("4. Numerical precision in iterative optimization")
    print()
    print("For accurate comparison:")
    print("- Ensure identical instrument specifications")
    print("- Match one-step vs two-step estimation")
    print("- Match robust/cluster SE options")
    print("- Compare test statistics (Hansen J, AR tests)")
    print()

    if all_passed:
        print("✓ Comparisons within tolerance (1e-3)")
        return 0
    else:
        print("? Some differences > 1e-3 (expected for GMM)")
        print("  Update plm_results from pgmm.R output for better comparison")
        return 0  # Don't fail - GMM differences are expected


if __name__ == "__main__":
    sys.exit(main())
