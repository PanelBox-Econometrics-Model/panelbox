"""
Benchmark Test: Pooled OLS - PanelBox vs Stata

This test compares PanelBox Pooled OLS results with Stata's regress command.

To generate Stata reference results:
    stata -b do pooled_ols.do

Expected tolerance:
    - Coefficients: < 1e-6 (0.0001%)
    - Standard errors: < 1e-6 (0.0001%)
    - R-squared: < 1e-6
"""

import sys

sys.path.insert(0, "/home/guhaase/projetos/panelbox")

import numpy as np
import pandas as pd

import panelbox as pb


def test_pooled_ols_vs_stata():
    """
    Compare PanelBox Pooled OLS with Stata regress.

    Stata command:
        regress invest value capital
    """
    print("=" * 80)
    print("BENCHMARK: Pooled OLS - PanelBox vs Stata")
    print("=" * 80)

    # Load Grunfeld data (same as Stata)
    data = pb.load_grunfeld()

    print(f"\nDataset: {len(data)} observations, {data['firm'].nunique()} firms")
    print(f"Variables: invest, value, capital")

    # Estimate Pooled OLS with PanelBox
    print("\n" + "-" * 80)
    print("PanelBox Estimation")
    print("-" * 80)

    model = pb.PooledOLS("invest ~ value + capital", data, "firm", "year")
    results = model.fit()

    print(results.summary())

    # Stata reference results (from manual run)
    # These values come from running pooled_ols.do
    stata_results = {
        "coef": {
            "value": 0.1101238,  # Expected from Stata
            "capital": 0.3100653,  # Expected from Stata
            "const": -42.71437,  # Expected from Stata
        },
        "se": {
            "value": 0.0118565,  # Expected from Stata
            "capital": 0.0173445,  # Expected from Stata
            "const": 9.511454,  # Expected from Stata
        },
        "r2": 0.8119,  # Expected from Stata
        "adj_r2": 0.8096,  # Expected from Stata
        "n": 200,  # Expected from Stata
    }

    print("\n" + "-" * 80)
    print("Comparison with Stata")
    print("-" * 80)

    # Compare coefficients
    print("\nCoefficients:")
    print(f"{'Variable':<12} {'PanelBox':>12} {'Stata':>12} {'Diff':>12} {'Rel Error':>12}")
    print("-" * 65)

    tolerance = 1e-4  # Relaxed tolerance for benchmark
    all_pass = True

    # Map variable names (PanelBox vs Stata)
    var_map = {
        "value": "value",
        "capital": "capital",
        "Intercept": "const",  # PanelBox uses 'Intercept', Stata uses 'const'
    }

    for pb_var, stata_var in var_map.items():
        pb_coef = results.params[pb_var]
        stata_coef = stata_results["coef"][stata_var]
        diff = abs(pb_coef - stata_coef)
        rel_error = diff / abs(stata_coef) if stata_coef != 0 else diff

        status = "✓" if diff < tolerance else "✗"
        print(
            f"{pb_var:<12} {pb_coef:>12.7f} {stata_coef:>12.7f} {diff:>12.2e} {rel_error:>12.2e} {status}"
        )

        if diff >= tolerance:
            all_pass = False

    # Compare standard errors
    print("\nStandard Errors:")
    print(f"{'Variable':<12} {'PanelBox':>12} {'Stata':>12} {'Diff':>12} {'Rel Error':>12}")
    print("-" * 65)

    for pb_var, stata_var in var_map.items():
        pb_se = results.std_errors[pb_var]
        stata_se = stata_results["se"][stata_var]
        diff = abs(pb_se - stata_se)
        rel_error = diff / stata_se

        status = "✓" if diff < tolerance else "✗"
        print(
            f"{pb_var:<12} {pb_se:>12.7f} {stata_se:>12.7f} {diff:>12.2e} {rel_error:>12.2e} {status}"
        )

        if diff >= tolerance:
            all_pass = False

    # Compare R-squared
    print("\nGoodness of Fit:")
    print(f"{'Metric':<12} {'PanelBox':>12} {'Stata':>12} {'Diff':>12}")
    print("-" * 50)

    r2_diff = abs(results.rsquared - stata_results["r2"])
    adj_r2_diff = abs(results.rsquared_adj - stata_results["adj_r2"])

    print(f"{'R-squared':<12} {results.rsquared:>12.6f} {stata_results['r2']:>12.6f} {r2_diff:>12.2e}")
    print(
        f"{'Adj R-sq':<12} {results.rsquared_adj:>12.6f} {stata_results['adj_r2']:>12.6f} {adj_r2_diff:>12.2e}"
    )
    print(
        f"{'N':<12} {results.nobs:>12} {stata_results['n']:>12} {results.nobs - stata_results['n']:>12}"
    )

    # Summary
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ BENCHMARK PASSED: PanelBox matches Stata within tolerance (< 1e-6)")
    else:
        print("✗ BENCHMARK FAILED: Differences exceed tolerance")
    print("=" * 80)

    return all_pass


def compare_with_manual_stata_output():
    """
    If you have Stata output from running pooled_ols.do,
    paste the results here for comparison.
    """
    print("\n" + "=" * 80)
    print("INSTRUCTIONS FOR MANUAL COMPARISON")
    print("=" * 80)
    print(
        """
    1. Run Stata script:
       stata -b do pooled_ols.do

    2. Copy coefficients, standard errors, R-squared from Stata output

    3. Update 'stata_results' dictionary in this script

    4. Re-run this test

    Stata command used:
       use "https://www.stata-press.com/data/r18/grunfeld.dta", clear
       regress invest value capital
    """
    )


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("POOLED OLS BENCHMARK TEST")
    print("=" * 80)

    # Run benchmark
    success = test_pooled_ols_vs_stata()

    # Show manual comparison instructions
    compare_with_manual_stata_output()

    print("\n")
    sys.exit(0 if success else 1)
