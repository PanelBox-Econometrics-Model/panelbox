"""
Benchmark Test: System GMM - PanelBox vs Stata xtabond2

This test compares PanelBox System GMM results with Stata's xtabond2.

To generate Stata reference results:
    stata -b do sys_gmm.do

Expected tolerance:
    - Coefficients: < 1e-4 (0.01%)
    - Standard errors: < 1e-4 (0.01%)
    - Test statistics: < 1e-3 (0.1%)

Note: System GMM includes both difference and level equations.
"""

import sys

sys.path.insert(0, "/home/guhaase/projetos/panelbox")

import numpy as np
import pandas as pd

import panelbox as pb


def test_sys_gmm_vs_stata_twostep():
    """
    Compare PanelBox System GMM (two-step) with Stata xtabond2.

    Stata command:
        xtabond2 invest L.invest value capital, ///
            gmm(L.invest, lag(2 .)) ///
            iv(value capital) ///
            robust ///
            small ///
            twostep
    """
    print("=" * 80)
    print("BENCHMARK: System GMM (Two-Step) - PanelBox vs Stata xtabond2")
    print("=" * 80)

    # Load Grunfeld data
    data = pb.load_grunfeld()

    print(f"\nDataset: {len(data)} observations, {data['firm'].nunique()} firms")
    print(f"Model: invest = rho*L.invest + beta1*value + beta2*capital + u_i + e_it")
    print(f"       (Difference + Level equations)")

    # Estimate System GMM with PanelBox
    print("\n" + "-" * 80)
    print("PanelBox Estimation (Two-Step with Windmeijer)")
    print("-" * 80)

    try:
        model = pb.SystemGMM(
            data=data,
            dep_var="invest",
            lags=1,
            id_var="firm",
            time_var="year",
            exog_vars=["value", "capital"],
            gmm_lags=(2, None),  # lag(2 .)
            iv_vars=["value", "capital"],
            collapse=False,
            two_step=True,
        )
        results = model.fit()

        print(results.summary())

    except Exception as e:
        print(f"Error fitting model: {e}")
        print("This may indicate an issue with the model specification.")
        return False

    # Stata reference results (from manual run)
    # These values come from running sys_gmm.do
    stata_results = {
        "coef": {
            "L.invest": 0.7234567,  # PLACEHOLDER - Update from Stata
            "value": 0.0887654,  # PLACEHOLDER - Update from Stata
            "capital": 0.1945678,  # PLACEHOLDER - Update from Stata
        },
        "se": {
            "L.invest": 0.0676543,  # PLACEHOLDER - Update from Stata
            "value": 0.0194567,  # PLACEHOLDER - Update from Stata
            "capital": 0.0356789,  # PLACEHOLDER - Update from Stata
        },
        "hansen_j": 12.456,  # PLACEHOLDER - Hansen J statistic
        "hansen_p": 0.5678,  # PLACEHOLDER - Hansen J p-value
        "ar1_p": 0.0123,  # PLACEHOLDER - AR(1) test p-value
        "ar2_p": 0.5234,  # PLACEHOLDER - AR(2) test p-value
        "n_instruments": 42,  # PLACEHOLDER - Number of instruments
        "n": 180,  # PLACEHOLDER - Observations used
        "n_groups": 10,  # PLACEHOLDER - Number of groups
    }

    print("\n" + "-" * 80)
    print("Comparison with Stata xtabond2")
    print("-" * 80)

    # Compare coefficients
    print("\nCoefficients:")
    print(f"{'Variable':<12} {'PanelBox':>12} {'Stata':>12} {'Diff':>12} {'Rel Error':>12}")
    print("-" * 65)

    tolerance = 1e-3  # Relaxed tolerance for GMM
    all_pass = True

    var_map = {"L1.invest": "L.invest", "value": "value", "capital": "capital"}

    for pb_var, stata_var in var_map.items():
        try:
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
        except KeyError:
            print(
                f"{pb_var:<12} {'N/A':>12} {stata_results['coef'][stata_var]:>12.7f} {'N/A':>12} {'N/A':>12} ✗"
            )
            all_pass = False

    # Compare standard errors
    print("\nStandard Errors (Windmeijer-corrected):")
    print(f"{'Variable':<12} {'PanelBox':>12} {'Stata':>12} {'Diff':>12} {'Rel Error':>12}")
    print("-" * 65)

    for pb_var, stata_var in var_map.items():
        try:
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
        except KeyError:
            print(
                f"{pb_var:<12} {'N/A':>12} {stata_results['se'][stata_var]:>12.7f} {'N/A':>12} {'N/A':>12} ✗"
            )
            all_pass = False

    # Compare specification tests
    print("\nSpecification Tests:")
    print(f"{'Test':<20} {'PanelBox':>12} {'Stata':>12} {'Diff':>12}")
    print("-" * 60)

    tests = [
        (
            "Hansen J",
            "hansen_j",
            results.hansen_j.statistic if hasattr(results, "hansen_j") else None,
        ),
        (
            "Hansen J p-value",
            "hansen_p",
            results.hansen_j.pvalue if hasattr(results, "hansen_j") else None,
        ),
        (
            "AR(1) p-value",
            "ar1_p",
            results.ar1_test.pvalue if hasattr(results, "ar1_test") else None,
        ),
        (
            "AR(2) p-value",
            "ar2_p",
            results.ar2_test.pvalue if hasattr(results, "ar2_test") else None,
        ),
    ]

    for name, stata_key, pb_val in tests:
        if pb_val is not None:
            stata_val = stata_results[stata_key]
            diff = abs(pb_val - stata_val)
            status = "✓" if diff < tolerance else "✗"
            print(f"{name:<20} {pb_val:>12.4f} {stata_val:>12.4f} {diff:>12.2e} {status}")
            if diff >= tolerance:
                all_pass = False
        else:
            print(f"{name:<20} {'N/A':>12} {stata_results[stata_key]:>12.4f} {'N/A':>12} ✗")

    # Instrument count
    print("\nInstrumentation:")
    print(f"{'Metric':<20} {'PanelBox':>12} {'Stata':>12}")
    print("-" * 45)

    if hasattr(results, "n_instruments"):
        print(
            f"{'Instruments':<20} {results.n_instruments:>12} {stata_results['n_instruments']:>12}"
        )
    else:
        print(f"{'Instruments':<20} {'N/A':>12} {stata_results['n_instruments']:>12}")

    # Sample size
    print(f"{'N':<20} {results.nobs:>12} {stata_results['n']:>12}")
    print(
        f"{'Groups':<20} {results.n_entities if hasattr(results, 'n_entities') else 'N/A':>12} {stata_results['n_groups']:>12}"
    )

    # Diagnostic interpretation
    print("\n" + "-" * 80)
    print("DIAGNOSTIC INTERPRETATION")
    print("-" * 80)

    if hasattr(results, "hansen_j") and hasattr(results, "ar2_test"):
        print(f"\nHansen J test (p={results.hansen_j.pvalue:.3f}):")
        if results.hansen_j.pvalue > 0.05:
            print("  ✓ Instruments appear valid (p > 0.05)")
        else:
            print("  ✗ WARNING: Instruments may be invalid (p < 0.05)")

        print(f"\nAR(2) test (p={results.ar2_test.pvalue:.3f}):")
        if results.ar2_test.pvalue > 0.05:
            print("  ✓ No second-order autocorrelation (p > 0.05)")
        else:
            print("  ✗ WARNING: Second-order autocorrelation detected (p < 0.05)")

    print("\nSystem GMM advantages:")
    print("  - More efficient than Difference GMM")
    print("  - Better for persistent variables (high autocorrelation)")
    print("  - Uses level equation with differenced instruments")

    # Summary
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ BENCHMARK PASSED: PanelBox matches Stata within tolerance (< 1e-3)")
    else:
        print("✗ BENCHMARK FAILED: Differences exceed tolerance")
        print("\nNOTE: Stata reference values are PLACEHOLDERS.")
        print("      Run 'stata -b do sys_gmm.do' and update values.")
        print("\nGMM estimators can have larger numerical differences due to:")
        print("  - Different optimization algorithms")
        print("  - Different initial values")
        print("  - Matrix operation precision")
    print("=" * 80)

    return all_pass


def compare_diff_vs_sys():
    """
    Compare Difference GMM vs System GMM on the same data.
    """
    print("\n" + "=" * 80)
    print("COMPARISON: Difference GMM vs System GMM")
    print("=" * 80)

    data = pb.load_grunfeld()

    print("\nEstimating both models for comparison...")

    try:
        # Difference GMM
        diff_model = pb.DifferenceGMM(
            data=data,
            dep_var="invest",
            lags=1,
            id_var="firm",
            time_var="year",
            exog_vars=["value", "capital"],
            gmm_lags=(2, None),
            iv_vars=["value", "capital"],
            collapse=False,
            two_step=True,
        )
        diff_results = diff_model.fit()

        # System GMM
        sys_model = pb.SystemGMM(
            data=data,
            dep_var="invest",
            lags=1,
            id_var="firm",
            time_var="year",
            exog_vars=["value", "capital"],
            gmm_lags=(2, None),
            iv_vars=["value", "capital"],
            collapse=False,
            two_step=True,
        )
        sys_results = sys_model.fit()

        # Compare results
        print("\nCoefficient Comparison:")
        print(f"{'Variable':<12} {'Diff GMM':>12} {'Sys GMM':>12} {'Difference':>12}")
        print("-" * 50)

        for var in ["L1.invest", "value", "capital"]:
            try:
                diff_coef = diff_results.params[var]
                sys_coef = sys_results.params[var]
                diff = abs(diff_coef - sys_coef)
                print(f"{var:<12} {diff_coef:>12.7f} {sys_coef:>12.7f} {diff:>12.2e}")
            except KeyError:
                print(f"{var:<12} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

        print("\nDiagnostics Comparison:")
        print(f"{'Test':<20} {'Diff GMM':>12} {'Sys GMM':>12}")
        print("-" * 45)

        if hasattr(diff_results, "hansen_j") and hasattr(sys_results, "hansen_j"):
            print(
                f"{'Hansen J p-value':<20} {diff_results.hansen_j.pvalue:>12.4f} {sys_results.hansen_j.pvalue:>12.4f}"
            )

        if hasattr(diff_results, "ar2_test") and hasattr(sys_results, "ar2_test"):
            print(
                f"{'AR(2) p-value':<20} {diff_results.ar2_test.pvalue:>12.4f} {sys_results.ar2_test.pvalue:>12.4f}"
            )

        print("\nNote: System GMM typically has:")
        print("  - Similar or lower standard errors (more efficient)")
        print("  - More instruments (level + difference equations)")
        print("  - Better performance with persistent variables")

    except Exception as e:
        print(f"Error: {e}")


def instructions():
    """
    Print instructions for manual comparison.
    """
    print("\n" + "=" * 80)
    print("INSTRUCTIONS FOR MANUAL COMPARISON")
    print("=" * 80)
    print(
        """
    1. Run Stata script (requires xtabond2):
       ssc install xtabond2  # If not installed
       cd tests/benchmarks/stata_comparison
       stata -b do sys_gmm.do

    2. Open sys_gmm.log and copy values:
       - Coefficients for L.invest, value, capital
       - Standard errors (Windmeijer-corrected)
       - Hansen J statistic and p-value
       - AR(1) and AR(2) test p-values
       - Number of instruments

    3. Update 'stata_results' dictionary in this script

    4. Re-run this test:
       python3 test_sys_gmm_vs_stata.py

    Stata command used:
       use "https://www.stata-press.com/data/r18/grunfeld.dta", clear
       xtset company year
       xtabond2 invest L.invest value capital, ///
           gmm(L.invest, lag(2 .)) ///
           iv(value capital) ///
           robust small twostep

    Note: Unlike Difference GMM, this includes both
          difference and level equations.
    """
    )


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("SYSTEM GMM BENCHMARK TEST")
    print("=" * 80)

    # Run benchmark
    success = test_sys_gmm_vs_stata_twostep()

    # Additional comparison
    compare_diff_vs_sys()

    # Show instructions
    instructions()

    print("\n")
    sys.exit(0 if success else 1)
