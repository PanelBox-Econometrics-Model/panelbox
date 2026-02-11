"""
Panel VAR GMM Instrument Diagnostics Example

This example focuses on advanced instrument diagnostics for Panel VAR GMM estimation.

Topics covered:
1. Detecting instrument proliferation
2. Hansen J test interpretation
3. Difference-in-Hansen test for instrument subsets
4. Instrument sensitivity analysis
5. Visualizing coefficient stability
6. Comparing one-step vs two-step GMM

Author: PanelBox Team
Date: 2025-02-12
"""

import warnings

import numpy as np
import pandas as pd

from panelbox.var.gmm import estimate_panel_var_gmm
from panelbox.visualization.var_plots import plot_instrument_sensitivity


def generate_panel_data(N=50, T=20, K=2, dgp_type="valid", seed=42):
    """
    Generate panel data with different DGP characteristics.

    Parameters
    ----------
    N : int
        Number of entities
    T : int
        Number of time periods
    K : int
        Number of variables
    dgp_type : str
        Type of DGP: 'valid', 'invalid_instruments', 'weak_instruments'
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data
    """
    np.random.seed(seed)

    # VAR(1) coefficients
    A = np.array([[0.5, 0.2], [0.1, 0.6]])

    data = []
    for i in range(N):
        alpha_i = np.random.randn(K) * 0.5
        y = np.zeros((T, K))
        y[0] = alpha_i + np.random.randn(K) * 0.5

        for t in range(1, T):
            if dgp_type == "valid":
                # Standard VAR(1) with valid instruments
                y[t] = alpha_i + A @ y[t - 1] + np.random.randn(K) * 0.3

            elif dgp_type == "invalid_instruments":
                # Add correlation with future errors (violates exogeneity)
                y[t] = alpha_i + A @ y[t - 1] + np.random.randn(K) * 0.3
                if t < T - 1:
                    # Contaminate with future error
                    future_shock = np.random.randn(K) * 0.1
                    y[t] += future_shock

            elif dgp_type == "weak_instruments":
                # VAR with very persistent process (instruments are weak)
                A_weak = np.array([[0.95, 0.05], [0.05, 0.95]])
                y[t] = alpha_i + A_weak @ y[t - 1] + np.random.randn(K) * 0.3

        for t in range(T):
            data.append({"entity": i, "time": t, "y1": y[t, 0], "y2": y[t, 1]})

    return pd.DataFrame(data)


def example_1_instrument_proliferation_detection():
    """
    Example 1: Detecting Instrument Proliferation

    Shows how to detect when you have too many instruments relative to
    the number of entities (Roodman 2009 rule of thumb).
    """
    print("=" * 80)
    print("EXAMPLE 1: Detecting Instrument Proliferation")
    print("=" * 80)

    # Generate data with SMALL N and LARGE T
    # This is the classic setup for proliferation
    data = generate_panel_data(N=30, T=25, K=2)  # Small N, large T

    # First: Estimate with NO limit on instruments
    print("\n[CASE A] No instrument limit (instrument_type='all')")
    print("-" * 60)

    result_unlimited = estimate_panel_var_gmm(
        data=data,
        var_lags=1,
        value_cols=["y1", "y2"],
        instrument_type="all",
        max_instruments=None,  # No limit!
    )

    print(f"Number of instruments: {result_unlimited.n_instruments}")
    print(f"Number of entities: {result_unlimited.n_entities}")
    print(
        f"Ratio instruments/entities: {result_unlimited.n_instruments / result_unlimited.n_entities:.2f}"
    )

    # Check diagnostics
    print(result_unlimited.instrument_diagnostics())

    # Second: Estimate with collapsed instruments
    print("\n[CASE B] With collapsed instruments")
    print("-" * 60)

    result_collapsed = estimate_panel_var_gmm(
        data=data,
        var_lags=1,
        value_cols=["y1", "y2"],
        instrument_type="collapsed",
        max_instruments=10,
    )

    print(f"Number of instruments: {result_collapsed.n_instruments}")
    print(f"Number of entities: {result_collapsed.n_entities}")
    print(
        f"Ratio instruments/entities: {result_collapsed.n_instruments / result_collapsed.n_entities:.2f}"
    )

    print(result_collapsed.instrument_diagnostics())

    print("\n" + "=" * 80)
    print("KEY LESSON:")
    print("=" * 80)
    print("When #instruments > #entities:")
    print("  - Hansen J test loses power (doesn't reject bad instruments)")
    print("  - GMM estimates become biased toward OLS")
    print("  - SOLUTION: Use instrument_type='collapsed' and set max_instruments")

    return result_unlimited, result_collapsed


def example_2_hansen_j_interpretation():
    """
    Example 2: Hansen J Test Interpretation

    Demonstrates how to interpret Hansen J test p-values in different scenarios.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Hansen J Test Interpretation")
    print("=" * 80)

    # Case 1: Valid instruments (should NOT reject)
    print("\n[CASE 1] Valid Instruments")
    print("-" * 60)
    data_valid = generate_panel_data(N=50, T=18, K=2, dgp_type="valid")

    result_valid = estimate_panel_var_gmm(
        data=data_valid,
        var_lags=1,
        value_cols=["y1", "y2"],
        instrument_type="collapsed",
        max_instruments=10,
    )

    j_stat, j_pval = result_valid.diagnostics.hansen_j_test()
    print(f"Hansen J statistic: {j_stat:.4f}")
    print(f"P-value: {j_pval:.4f}")

    if 0.1 <= j_pval <= 0.9:
        print("✓ IDEAL: p-value in reasonable range [0.1, 0.9]")
    elif j_pval < 0.05:
        print("⚠ REJECT: Instruments may be invalid")
    elif j_pval > 0.99:
        print("⚠ WARNING: p-value too high - possible weak instruments")

    # Case 2: Too many instruments (p-value will be very high)
    print("\n[CASE 2] Instrument Proliferation (Too Many Instruments)")
    print("-" * 60)

    # Use a lot of instruments
    result_many = estimate_panel_var_gmm(
        data=data_valid,
        var_lags=1,
        value_cols=["y1", "y2"],
        instrument_type="all",
        max_instruments=None,
    )

    j_stat_many, j_pval_many = result_many.diagnostics.hansen_j_test()
    print(f"Hansen J statistic: {j_stat_many:.4f}")
    print(f"P-value: {j_pval_many:.4f}")
    print(f"Number of instruments: {result_many.n_instruments}")

    if j_pval_many > 0.99:
        print("⚠ WARNING: p-value suspiciously high!")
        print("  → Likely due to instrument proliferation")
        print("  → Test has lost power to detect invalid instruments")

    print("\n" + "=" * 80)
    print("HANSEN J INTERPRETATION GUIDE:")
    print("=" * 80)
    print("p < 0.05:     Reject H0 - instruments invalid or model misspecified")
    print("0.05 ≤ p ≤ 0.10:  Marginal - check sensitivity")
    print("0.10 < p < 0.90:  IDEAL RANGE - instruments appear valid")
    print("0.90 ≤ p < 0.99:  Acceptable but check for weak instruments")
    print("p ≥ 0.99:     WARNING - likely instrument proliferation")

    return result_valid, result_many


def example_3_difference_hansen_test():
    """
    Example 3: Difference-in-Hansen Test

    Tests the validity of subsets of instruments by comparing models with
    different instrument sets.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Difference-in-Hansen Test")
    print("=" * 80)

    data = generate_panel_data(N=60, T=20, K=2, dgp_type="valid")

    # Full model (more instruments)
    print("\n[STEP 1] Estimate full model (more instruments)")
    print("-" * 60)
    result_full = estimate_panel_var_gmm(
        data=data,
        var_lags=1,
        value_cols=["y1", "y2"],
        instrument_type="collapsed",
        max_instruments=15,
    )

    j_full, p_full = result_full.diagnostics.hansen_j_test()
    print(f"Full model - Hansen J: {j_full:.4f} (p = {p_full:.4f})")
    print(f"Number of instruments: {result_full.n_instruments}")

    # Restricted model (fewer instruments)
    print("\n[STEP 2] Estimate restricted model (fewer instruments)")
    print("-" * 60)
    result_restricted = estimate_panel_var_gmm(
        data=data,
        var_lags=1,
        value_cols=["y1", "y2"],
        instrument_type="collapsed",
        max_instruments=8,
    )

    j_restricted, p_restricted = result_restricted.diagnostics.hansen_j_test()
    print(f"Restricted model - Hansen J: {j_restricted:.4f} (p = {p_restricted:.4f})")
    print(f"Number of instruments: {result_restricted.n_instruments}")

    # Difference-in-Hansen test
    print("\n[STEP 3] Difference-in-Hansen Test")
    print("-" * 60)

    # Test additional instruments
    diff_result = result_full.diagnostics.difference_hansen_test(
        restricted_result=result_restricted
    )

    print(f"Difference-in-Hansen statistic: {diff_result['statistic']:.4f}")
    print(f"P-value: {diff_result['p_value']:.4f}")
    print(f"Testing {diff_result['df_diff']} additional instruments")

    if diff_result["p_value"] >= 0.05:
        print("\n✓ CONCLUSION: Additional instruments appear valid")
        print("  → Safe to use full instrument set")
    else:
        print("\n⚠ CONCLUSION: Additional instruments may be problematic")
        print("  → Use restricted instrument set instead")

    return result_full, result_restricted, diff_result


def example_4_sensitivity_analysis():
    """
    Example 4: Instrument Sensitivity Analysis

    Tests coefficient stability across different numbers of instruments.
    Unstable coefficients indicate invalid or weak instruments.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Instrument Sensitivity Analysis")
    print("=" * 80)

    data = generate_panel_data(N=55, T=20, K=2, dgp_type="valid")

    # Estimate base model
    result = estimate_panel_var_gmm(
        data=data,
        var_lags=1,
        value_cols=["y1", "y2"],
        instrument_type="collapsed",
        max_instruments=12,
    )

    # Run sensitivity analysis
    print("\n[STEP 1] Running sensitivity analysis...")
    print("-" * 60)
    print("Testing coefficient stability across different instrument counts...")

    sensitivity = result.diagnostics.instrument_sensitivity_analysis(
        max_instrument_counts=[4, 6, 8, 10, 12, 15, 18]
    )

    print("\nResults:")
    print(f"Instrument counts tested: {sensitivity['instrument_counts']}")
    print(f"Max coefficient change: {sensitivity['max_coef_change']:.4f}")
    print(f"Max coefficient change (%): {sensitivity['max_coef_change_pct']:.1f}%")
    print(f"Mean coefficient change: {sensitivity['mean_coef_change']:.4f}")
    print(f"Stable? {sensitivity['is_stable']}")

    # Display coefficient trajectories
    print("\n[STEP 2] Coefficient Trajectories")
    print("-" * 60)
    print("Coefficient | " + " | ".join([f"{n:3d} inst" for n in sensitivity["instrument_counts"]]))
    print("-" * 80)

    for var_from in ["y1", "y2"]:
        for var_to in ["y1", "y2"]:
            coef_name = f"{var_to}.L1.{var_from}"
            if coef_name in sensitivity["coefficients"]:
                values = sensitivity["coefficients"][coef_name]
                value_str = " | ".join([f"{v:7.4f}" for v in values])
                print(f"{coef_name:11s} | {value_str}")

    print("\n[STEP 3] Interpretation")
    print("-" * 60)

    if sensitivity["is_stable"]:
        print("✓ STABLE: Coefficients do not vary significantly")
        print("  → Instruments are likely valid")
        print("  → Results are robust to instrument count")
    else:
        print("⚠ UNSTABLE: Coefficients vary significantly")
        print("  → Instruments may be weak or invalid")
        print("  → Reduce number of instruments")
        print("  → Consider alternative specification")

    # Visualize (if plotly available)
    print("\n[STEP 4] Visualization")
    print("-" * 60)
    try:
        fig = plot_instrument_sensitivity(
            sensitivity_results=sensitivity, title="Instrument Sensitivity Analysis - VAR(1)"
        )
        print("✓ Sensitivity plot created")
        print("  To view: fig.show() or fig.write_html('sensitivity.html')")
        # Optionally save
        # fig.write_html('instrument_sensitivity.html')
        return result, sensitivity, fig
    except Exception as e:
        print(f"⚠ Could not create plot: {e}")
        return result, sensitivity, None


def example_5_comparing_gmm_steps_diagnostics():
    """
    Example 5: Comparing One-Step vs Two-Step GMM Diagnostics

    Shows how diagnostics differ between one-step and two-step GMM.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Comparing One-Step vs Two-Step GMM Diagnostics")
    print("=" * 80)

    data = generate_panel_data(N=50, T=18, K=2, dgp_type="valid")

    # Estimate with two-step
    print("\n[STEP 1] Two-Step GMM")
    print("-" * 60)
    result_2step = estimate_panel_var_gmm(
        data=data,
        var_lags=1,
        value_cols=["y1", "y2"],
        gmm_step="two-step",
        instrument_type="collapsed",
        max_instruments=10,
    )

    j_2step, p_2step = result_2step.diagnostics.hansen_j_test()
    ar1_2step, p_ar1_2step = result_2step.diagnostics.ar_test(order=1)
    ar2_2step, p_ar2_2step = result_2step.diagnostics.ar_test(order=2)

    print(f"Hansen J: {j_2step:.4f} (p = {p_2step:.4f})")
    print(f"AR(1): {ar1_2step:.4f} (p = {p_ar1_2step:.4f})")
    print(f"AR(2): {ar2_2step:.4f} (p = {p_ar2_2step:.4f})")

    # Compare with one-step
    print("\n[STEP 2] Comparing with One-Step GMM")
    print("-" * 60)
    comparison = result_2step.compare_one_step_two_step()

    print(comparison["summary"])

    print("\n[STEP 3] Diagnostic Comparison")
    print("-" * 60)

    # Get one-step result from comparison
    result_1step = comparison["one_step_result"]
    j_1step, p_1step = result_1step.diagnostics.hansen_j_test()

    print(f"Hansen J (one-step): {j_1step:.4f} (p = {p_1step:.4f})")
    print(f"Hansen J (two-step): {j_2step:.4f} (p = {p_2step:.4f})")

    print("\n[STEP 4] Interpretation")
    print("-" * 60)

    if abs(p_1step - p_2step) < 0.1:
        print("✓ Hansen J tests are similar across GMM steps")
        print("  → Estimation is stable")
    else:
        print("⚠ Hansen J tests differ significantly")
        print("  → Investigate further")

    if comparison["coef_diff_pct_max"] < 5:
        print("✓ Coefficients are very similar (<5% difference)")
        print("  → Two-step is preferred (more efficient)")
    elif comparison["coef_diff_pct_max"] < 15:
        print("✓ Coefficients are moderately similar (5-15% difference)")
        print("  → Two-step is still preferred but check diagnostics")
    else:
        print("⚠ Coefficients differ substantially (>15%)")
        print("  → Possible misspecification - investigate")

    return result_1step, result_2step, comparison


def example_6_complete_diagnostic_report():
    """
    Example 6: Complete Diagnostic Report

    Generates a comprehensive diagnostic report for a Panel VAR GMM model.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Complete Diagnostic Report")
    print("=" * 80)

    data = generate_panel_data(N=60, T=20, K=2, dgp_type="valid")

    # Estimate model
    result = estimate_panel_var_gmm(
        data=data,
        var_lags=1,
        value_cols=["y1", "y2"],
        transform="fod",
        gmm_step="two-step",
        instrument_type="collapsed",
        max_instruments=10,
    )

    print("\n" + "=" * 80)
    print("COMPLETE GMM DIAGNOSTIC REPORT")
    print("=" * 80)

    # Section 1: Model specification
    print("\n[1] MODEL SPECIFICATION")
    print("-" * 60)
    print(f"Variables: {result.var_names}")
    print(f"VAR lags: {result.lags}")
    print(f"Transformation: {result.transform}")
    print(f"GMM step: {result.gmm_step}")
    print(f"Instrument type: {result.instrument_type}")

    # Section 2: Sample information
    print("\n[2] SAMPLE INFORMATION")
    print("-" * 60)
    print(f"Number of entities: {result.n_entities}")
    print(f"Number of observations: {result.n_obs}")
    print(f"Number of instruments: {result.n_instruments}")
    print(f"Number of parameters: {len(result.params)}")

    # Section 3: Instrument diagnostics
    print("\n[3] INSTRUMENT DIAGNOSTICS")
    print("-" * 60)
    print(result.instrument_diagnostics())

    # Section 4: Hansen J test
    print("\n[4] OVERIDENTIFICATION TEST")
    print("-" * 60)
    j_stat, j_pval = result.diagnostics.hansen_j_test()
    print(f"Hansen J statistic: {j_stat:.4f}")
    print(f"P-value: {j_pval:.4f}")
    print(f"Degrees of freedom: {result.n_instruments - len(result.params)}")

    if j_pval < 0.05:
        verdict = "⚠ REJECT - Instruments may be invalid"
    elif j_pval > 0.99:
        verdict = "⚠ WARNING - Possible weak instruments"
    else:
        verdict = "✓ PASS - Instruments appear valid"
    print(f"Verdict: {verdict}")

    # Section 5: Serial correlation tests
    print("\n[5] SERIAL CORRELATION TESTS")
    print("-" * 60)
    ar1_stat, ar1_pval = result.diagnostics.ar_test(order=1)
    ar2_stat, ar2_pval = result.diagnostics.ar_test(order=2)

    print(f"AR(1) test: z = {ar1_stat:.4f}, p-value = {ar1_pval:.4f}")
    if ar1_pval < 0.05:
        print("  ✓ Expected rejection (by construction)")
    else:
        print("  ⚠ Unexpected - AR(1) should typically reject")

    print(f"AR(2) test: z = {ar2_stat:.4f}, p-value = {ar2_pval:.4f}")
    if ar2_pval >= 0.05:
        print("  ✓ PASS - No second-order serial correlation")
    else:
        print("  ⚠ FAIL - Consider adding lags or check specification")

    # Section 6: Coefficient stability
    print("\n[6] INSTRUMENT SENSITIVITY ANALYSIS")
    print("-" * 60)
    sensitivity = result.diagnostics.instrument_sensitivity_analysis(
        max_instrument_counts=[6, 8, 10, 12]
    )
    print(
        f"Max coefficient change: {sensitivity['max_coef_change']:.4f} ({sensitivity['max_coef_change_pct']:.1f}%)"
    )
    if sensitivity["is_stable"]:
        print("✓ Coefficients stable across instrument counts")
    else:
        print("⚠ Coefficients vary - reduce instruments")

    # Section 7: Overall assessment
    print("\n" + "=" * 80)
    print("[7] OVERALL ASSESSMENT")
    print("=" * 80)

    checks = {
        "Instrument count acceptable": result.n_instruments <= result.n_entities,
        "Hansen J valid": 0.05 <= j_pval <= 0.99,
        "AR(2) no serial correlation": ar2_pval >= 0.05,
        "Coefficient stability": sensitivity["is_stable"],
    }

    passed = sum(checks.values())
    total = len(checks)

    for check_name, check_passed in checks.items():
        status = "✓" if check_passed else "✗"
        print(f"  {status} {check_name}")

    print(f"\nPassed: {passed}/{total} checks")

    if passed == total:
        print("\n🎉 EXCELLENT: All diagnostic checks passed!")
        print("   Model is well-specified and ready for inference.")
    elif passed >= total - 1:
        print("\n✓ GOOD: Most diagnostic checks passed.")
        print("   Review failed checks but model appears reasonable.")
    else:
        print("\n⚠ WARNING: Multiple diagnostic checks failed.")
        print("   Review specification before proceeding with inference.")

    return result


def main():
    """Run all diagnostic examples."""
    print(
        """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              Panel VAR GMM Instrument Diagnostics Examples                  ║
║                                                                              ║
║  Advanced diagnostic techniques for Panel VAR GMM estimation.               ║
║  Learn to detect and resolve instrument-related issues.                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    )

    # Run examples
    print("\n🔍 Running diagnostic examples...\n")

    r1_unlimited, r1_collapsed = example_1_instrument_proliferation_detection()
    r2_valid, r2_many = example_2_hansen_j_interpretation()
    r3_full, r3_restricted, r3_diff = example_3_difference_hansen_test()
    r4_result, r4_sensitivity, r4_fig = example_4_sensitivity_analysis()
    r5_1step, r5_2step, r5_comp = example_5_comparing_gmm_steps_diagnostics()
    r6_result = example_6_complete_diagnostic_report()

    print("\n" + "=" * 80)
    print("ALL DIAGNOSTIC EXAMPLES COMPLETED")
    print("=" * 80)

    print("\n📚 KEY TAKEAWAYS:")
    print("=" * 80)
    print("1. INSTRUMENT PROLIFERATION:")
    print("   - Keep #instruments ≤ #entities (Roodman rule)")
    print("   - Use instrument_type='collapsed'")
    print("   - Set max_instruments conservatively")
    print()
    print("2. HANSEN J TEST:")
    print("   - p < 0.05: Invalid instruments or misspecification")
    print("   - 0.1 < p < 0.9: Ideal range")
    print("   - p > 0.99: Warning sign of weak instruments")
    print()
    print("3. AR TESTS:")
    print("   - AR(1) should reject (expected)")
    print("   - AR(2) should NOT reject (if well-specified)")
    print()
    print("4. SENSITIVITY ANALYSIS:")
    print("   - Coefficients should be stable across instrument counts")
    print("   - Instability indicates weak/invalid instruments")
    print()
    print("5. GENERAL WORKFLOW:")
    print("   - Start conservative (collapsed, low max_instruments)")
    print("   - Check all diagnostics")
    print("   - Adjust only if necessary")
    print("=" * 80)

    return {
        "example_1": (r1_unlimited, r1_collapsed),
        "example_2": (r2_valid, r2_many),
        "example_3": (r3_full, r3_restricted, r3_diff),
        "example_4": (r4_result, r4_sensitivity, r4_fig),
        "example_5": (r5_1step, r5_2step, r5_comp),
        "example_6": r6_result,
    }


if __name__ == "__main__":
    # Suppress some warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)

    results = main()

    print("\n💡 Next steps:")
    print("   1. Try these diagnostics on your own data")
    print("   2. Experiment with different instrument settings")
    print("   3. Combine with gmm_estimation.py examples")
    print("   4. Read the documentation: docs/api/var_gmm.md")
