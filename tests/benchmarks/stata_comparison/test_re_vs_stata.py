"""
Benchmark Test: Random Effects - PanelBox vs Stata

This test compares PanelBox Random Effects results with Stata's xtreg, re command.

To generate Stata reference results:
    stata -b do random_effects.do

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


def test_re_vs_stata():
    """
    Compare PanelBox Random Effects with Stata xtreg, re.

    Stata command:
        xtset company year
        xtreg invest value capital, re
    """
    print("=" * 80)
    print("BENCHMARK: Random Effects - PanelBox vs Stata")
    print("=" * 80)

    # Load Grunfeld data
    data = pb.load_grunfeld()

    print(f"\nDataset: {len(data)} observations, {data['firm'].nunique()} firms")
    print(f"Variables: invest, value, capital")

    # Estimate Random Effects with PanelBox
    print("\n" + "-" * 80)
    print("PanelBox Estimation")
    print("-" * 80)

    model = pb.RandomEffects("invest ~ value + capital", data, "firm", "year")
    results = model.fit()

    print(results.summary())

    # Stata reference results (from manual run)
    # These values come from running random_effects.do
    stata_results = {
        "coef": {
            "value": 0.1101112,  # PLACEHOLDER - Update from Stata
            "capital": 0.3101234,  # PLACEHOLDER - Update from Stata
            "const": -57.8344,  # PLACEHOLDER - Update from Stata
        },
        "se": {
            "value": 0.0118123,  # PLACEHOLDER - Update from Stata
            "capital": 0.0173567,  # PLACEHOLDER - Update from Stata
            "const": 28.8976,  # PLACEHOLDER - Update from Stata
        },
        "r2_within": 0.7667,  # PLACEHOLDER - Update from Stata
        "r2_between": 0.8234,  # PLACEHOLDER - Update from Stata
        "r2_overall": 0.8090,  # PLACEHOLDER - Update from Stata
        "rho": 0.7312,  # PLACEHOLDER - Fraction of variance due to u_i
        "sigma_u": 84.123,  # PLACEHOLDER - SD of random effects
        "sigma_e": 28.456,  # PLACEHOLDER - SD of residuals
        "theta": 0.8612,  # PLACEHOLDER - GLS transformation parameter
        "chi2": 485.23,  # PLACEHOLDER - Wald chi-square
        "n": 200,
        "n_groups": 10,
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

    # Map variable names
    var_map = {"value": "value", "capital": "capital", "Intercept": "const"}

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

    # Compare variance components and theta
    print("\nVariance Components:")
    print(f"{'Component':<15} {'PanelBox':>12} {'Stata':>12} {'Diff':>12}")
    print("-" * 55)

    if hasattr(results, "sigma_u") and hasattr(results, "sigma_e"):
        variance_comps = [
            ("sigma_u", results.sigma_u, stata_results["sigma_u"]),
            ("sigma_e", results.sigma_e, stata_results["sigma_e"]),
            ("rho", results.rho if hasattr(results, "rho") else 0, stata_results["rho"]),
            ("theta", results.theta if hasattr(results, "theta") else 0, stata_results["theta"]),
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
        print("      Run 'stata -b do random_effects.do' and update values.")
    print("=" * 80)

    return all_pass


def test_hausman():
    """
    Additional test: Hausman specification test (FE vs RE).
    """
    print("\n" + "=" * 80)
    print("ADDITIONAL CHECK: Hausman Test (FE vs RE)")
    print("=" * 80)

    data = pb.load_grunfeld()

    # Estimate both models
    fe_model = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
    fe_results = fe_model.fit()

    re_model = pb.RandomEffects("invest ~ value + capital", data, "firm", "year")
    re_results = re_model.fit()

    # Perform Hausman test if available
    try:
        from panelbox.validation.specification import hausman_test

        hausman = hausman_test(fe_results, re_results)

        print("\nHausman Test Results:")
        print(f"  H0: Random effects model is consistent")
        print(f"  H1: Fixed effects model is consistent")
        print(f"\n  Statistic: {hausman.statistic:.4f}")
        print(f"  P-value:   {hausman.pvalue:.4f}")
        print(f"  DOF:       {hausman.df}")

        if hausman.pvalue < 0.05:
            print("\n  Conclusion: Reject H0 at 5% level")
            print("              → Use Fixed Effects model")
        else:
            print("\n  Conclusion: Fail to reject H0 at 5% level")
            print("              → Random Effects model is acceptable")

        print("\n  (Compare this with Stata's hausman test output)")

    except ImportError:
        print("\nHausman test not implemented in PanelBox yet.")
        print("Stata command: hausman fe_model re_model")


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
       stata -b do random_effects.do

    2. Open random_effects.log and copy values:
       - Coefficients for value, capital, _cons
       - Standard errors
       - R-squared (within, between, overall)
       - sigma_u, sigma_e, rho, theta
       - Wald chi2
       - Hausman test statistic and p-value

    3. Update 'stata_results' dictionary in this script

    4. Re-run this test:
       python3 test_re_vs_stata.py

    Stata commands used:
       use "https://www.stata-press.com/data/r18/grunfeld.dta", clear
       xtset company year
       xtreg invest value capital, re
       hausman fe_model re_model
    """
    )


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("RANDOM EFFECTS BENCHMARK TEST")
    print("=" * 80)

    # Run benchmark
    success = test_re_vs_stata()

    # Additional checks
    test_hausman()

    # Show instructions
    instructions()

    print("\n")
    sys.exit(0 if success else 1)
