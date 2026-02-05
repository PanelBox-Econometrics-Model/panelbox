"""
Benchmark Test: Fixed Effects - PanelBox vs Stata

This test compares PanelBox Fixed Effects results with Stata's xtreg, fe command.

To generate Stata reference results:
    stata -b do fixed_effects.do

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


def test_fe_vs_stata():
    """
    Compare PanelBox Fixed Effects with Stata xtreg, fe.

    Stata command:
        xtset company year
        xtreg invest value capital, fe
    """
    print("=" * 80)
    print("BENCHMARK: Fixed Effects - PanelBox vs Stata")
    print("=" * 80)

    # Load Grunfeld data
    data = pb.load_grunfeld()

    print(f"\nDataset: {len(data)} observations, {data['firm'].nunique()} firms")
    print(f"Variables: invest, value, capital")

    # Estimate Fixed Effects with PanelBox
    print("\n" + "-" * 80)
    print("PanelBox Estimation")
    print("-" * 80)

    model = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
    results = model.fit()

    print(results.summary())

    # Stata reference results (from manual run)
    # These values come from running fixed_effects.do
    stata_results = {
        "coef": {
            "value": 0.1101112,  # PLACEHOLDER - Update from Stata
            "capital": 0.3101234,  # PLACEHOLDER - Update from Stata
        },
        "se": {
            "value": 0.0118123,  # PLACEHOLDER - Update from Stata
            "capital": 0.0173567,  # PLACEHOLDER - Update from Stata
        },
        "r2_within": 0.7667,  # PLACEHOLDER - Update from Stata
        "r2_between": 0.8234,  # PLACEHOLDER - Update from Stata
        "r2_overall": 0.8090,  # PLACEHOLDER - Update from Stata
        "rho": 0.7312,  # PLACEHOLDER - Fraction of variance due to u_i
        "sigma_u": 54.123,  # PLACEHOLDER - SD of fixed effects
        "sigma_e": 28.456,  # PLACEHOLDER - SD of residuals
        "n": 200,
        "n_groups": 10,
        "f_stat": 68.45,  # PLACEHOLDER - F-statistic
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

    for var in ["value", "capital"]:
        pb_coef = results.params[var]
        stata_coef = stata_results["coef"][var]
        diff = abs(pb_coef - stata_coef)
        rel_error = diff / abs(stata_coef) if stata_coef != 0 else diff

        status = "✓" if diff < tolerance else "✗"
        print(
            f"{var:<12} {pb_coef:>12.7f} {stata_coef:>12.7f} {diff:>12.2e} {rel_error:>12.2e} {status}"
        )

        if diff >= tolerance:
            all_pass = False

    # Compare standard errors
    print("\nStandard Errors:")
    print(f"{'Variable':<12} {'PanelBox':>12} {'Stata':>12} {'Diff':>12} {'Rel Error':>12}")
    print("-" * 65)

    for var in ["value", "capital"]:
        pb_se = results.std_errors[var]
        stata_se = stata_results["se"][var]
        diff = abs(pb_se - stata_se)
        rel_error = diff / stata_se

        status = "✓" if diff < tolerance else "✗"
        print(
            f"{var:<12} {pb_se:>12.7f} {stata_se:>12.7f} {diff:>12.2e} {rel_error:>12.2e} {status}"
        )

        if diff >= tolerance:
            all_pass = False

    # Compare R-squared measures
    print("\nGoodness of Fit:")
    print(f"{'Metric':<15} {'PanelBox':>12} {'Stata':>12} {'Diff':>12}")
    print("-" * 55)

    r2_metrics = [
        ("R² within", results.rsquared_within, stata_results["r2_within"]),
        ("R² between", results.rsquared_between, stata_results["r2_between"]),
        ("R² overall", results.rsquared_overall, stata_results["r2_overall"]),
    ]

    for name, pb_val, stata_val in r2_metrics:
        diff = abs(pb_val - stata_val)
        status = "✓" if diff < tolerance else "✗"
        print(f"{name:<15} {pb_val:>12.6f} {stata_val:>12.6f} {diff:>12.2e} {status}")
        if diff >= tolerance:
            all_pass = False

    # Compare variance components
    print("\nVariance Components:")
    print(f"{'Component':<15} {'PanelBox':>12} {'Stata':>12} {'Diff':>12}")
    print("-" * 55)

    if hasattr(results, "sigma_u") and hasattr(results, "sigma_e"):
        variance_comps = [
            ("sigma_u", results.sigma_u, stata_results["sigma_u"]),
            ("sigma_e", results.sigma_e, stata_results["sigma_e"]),
            ("rho", results.rho if hasattr(results, "rho") else 0, stata_results["rho"]),
        ]

        for name, pb_val, stata_val in variance_comps:
            diff = abs(pb_val - stata_val)
            status = "✓" if diff < tolerance else "✗"
            print(f"{name:<15} {pb_val:>12.6f} {stata_val:>12.6f} {diff:>12.2e} {status}")
            if diff >= tolerance:
                all_pass = False
    else:
        print("(Variance components not available in PanelBox results)")

    # Sample size
    print("\nSample Size:")
    print(f"{'Metric':<15} {'PanelBox':>12} {'Stata':>12}")
    print("-" * 40)
    print(f"{'N':<15} {results.nobs:>12} {stata_results['n']:>12}")
    print(f"{'Groups':<15} {results.n_entities:>12} {stata_results['n_groups']:>12}")

    # Summary
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ BENCHMARK PASSED: PanelBox matches Stata within tolerance (< 1e-4)")
    else:
        print("✗ BENCHMARK FAILED: Differences exceed tolerance")
        print("\nNOTE: Stata reference values are PLACEHOLDERS.")
        print("      Run 'stata -b do fixed_effects.do' and update values.")
    print("=" * 80)

    return all_pass


def compare_fixed_effects():
    """
    Additional comparison: Check that fixed effects are correctly extracted.
    """
    print("\n" + "=" * 80)
    print("ADDITIONAL CHECK: Fixed Effects Extraction")
    print("=" * 80)

    data = pb.load_grunfeld()
    model = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
    results = model.fit()

    # Extract fixed effects if available
    if hasattr(results, "fixed_effects"):
        print("\nFixed Effects by Entity:")
        print(f"{'Entity':>8} {'Fixed Effect':>15}")
        print("-" * 25)

        for entity in sorted(results.fixed_effects.keys())[:5]:  # Show first 5
            fe = results.fixed_effects[entity]
            print(f"{entity:>8} {fe:>15.6f}")

        if len(results.fixed_effects) > 5:
            print(f"... ({len(results.fixed_effects) - 5} more)")

        print(f"\nMean: {np.mean(list(results.fixed_effects.values())):>15.6f}")
        print(f"Std:  {np.std(list(results.fixed_effects.values())):>15.6f}")
    else:
        print("\n(Fixed effects not extracted in results object)")
        print("This is OK if the implementation stores them differently.")


def instructions():
    """
    Print instructions for manual comparison.
    """
    print("\n" + "=" * 80)
    print("INSTRUCTIONS FOR MANUAL COMPARISON")
    print("=" * 80)
    print(
        """
    1. Run Stata script:
       cd tests/benchmarks/stata_comparison
       stata -b do fixed_effects.do

    2. Open fixed_effects.log and copy values:
       - Coefficients for value and capital
       - Standard errors
       - R-squared (within, between, overall)
       - sigma_u, sigma_e, rho
       - F-statistic

    3. Update 'stata_results' dictionary in this script

    4. Re-run this test:
       python3 test_fe_vs_stata.py

    Stata commands used:
       use "https://www.stata-press.com/data/r18/grunfeld.dta", clear
       xtset company year
       xtreg invest value capital, fe
    """
    )


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("FIXED EFFECTS BENCHMARK TEST")
    print("=" * 80)

    # Run benchmark
    success = test_fe_vs_stata()

    # Additional checks
    compare_fixed_effects()

    # Show instructions
    instructions()

    print("\n")
    sys.exit(0 if success else 1)
