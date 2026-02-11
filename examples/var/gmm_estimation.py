"""
Panel VAR GMM Estimation Example

This example demonstrates how to estimate Panel Vector Autoregression (VAR) models
using Generalized Method of Moments (GMM) with panelbox.

Topics covered:
1. Basic GMM estimation with FOD transformation
2. Comparing one-step vs two-step GMM
3. Using collapsed instruments
4. Comparing FOD vs FD transformations
5. Interpreting GMM diagnostics

Author: PanelBox Team
Date: 2025-02-12
"""

import numpy as np
import pandas as pd

from panelbox.var.diagnostics import compare_transforms
from panelbox.var.gmm import estimate_panel_var_gmm


def generate_panel_var_data(N=50, T=20, K=2, seed=42):
    """
    Generate simulated panel VAR data for demonstration.

    Parameters
    ----------
    N : int
        Number of entities
    T : int
        Number of time periods
    K : int
        Number of variables
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Panel data with entity, time, and variable columns
    """
    np.random.seed(seed)

    # True VAR(1) coefficients
    # y1_t = 0.5*y1_{t-1} + 0.2*y2_{t-1} + u1_t
    # y2_t = 0.1*y1_{t-1} + 0.6*y2_{t-1} + u2_t
    A = np.array([[0.5, 0.2], [0.1, 0.6]])

    data = []
    for i in range(N):
        # Entity-specific fixed effect
        alpha_i = np.random.randn(K) * 0.5

        # Initialize time series
        y = np.zeros((T, K))
        y[0] = alpha_i + np.random.randn(K) * 0.5

        # Generate VAR(1) process
        for t in range(1, T):
            # VAR dynamics
            y[t] = alpha_i + A @ y[t - 1] + np.random.randn(K) * 0.3

        # Store as dataframe rows
        for t in range(T):
            data.append({"entity": i, "time": t, "y1": y[t, 0], "y2": y[t, 1]})

    return pd.DataFrame(data)


def example_1_basic_gmm_estimation():
    """
    Example 1: Basic GMM Estimation with FOD

    This example shows the simplest use case: estimating a VAR(1) model
    using two-step GMM with Forward Orthogonal Deviations.
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic GMM Estimation with FOD")
    print("=" * 80)

    # Generate data
    data = generate_panel_var_data(N=50, T=20, K=2)

    # Estimate VAR(1) with GMM
    result = estimate_panel_var_gmm(
        data=data,
        var_lags=1,
        value_cols=["y1", "y2"],
        transform="fod",
        gmm_step="two-step",
        instrument_type="collapsed",
        max_instruments=10,
    )

    # Display results
    print(f"\nEstimation Results:")
    print(f"  GMM Step: {result.gmm_step}")
    print(f"  Transform: {result.transform}")
    print(f"  Instrument Type: {result.instrument_type}")
    print(f"  Number of Observations: {result.n_obs}")
    print(f"  Number of Entities: {result.n_entities}")
    print(f"  Number of Instruments: {result.n_instruments}")
    print(f"  Windmeijer Corrected: {result.windmeijer_corrected}")

    print(f"\nCoefficients (shape {result.coefficients.shape}):")
    print(result.coefficients)

    print(f"\nStandard Errors (shape {result.standard_errors.shape}):")
    print(result.standard_errors)

    print("\nKey takeaways:")
    print("- Two-step GMM with Windmeijer correction provides efficient estimates")
    print("- FOD transformation removes fixed effects while preserving efficiency")
    print("- Collapsed instruments prevent proliferation in this sample size")

    return result


def example_2_comparing_gmm_steps():
    """
    Example 2: Comparing One-Step vs Two-Step GMM

    Two-step GMM is more efficient asymptotically but requires Windmeijer
    correction. This example compares both approaches.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Comparing One-Step vs Two-Step GMM")
    print("=" * 80)

    # Generate data
    data = generate_panel_var_data(N=60, T=15, K=2)

    # Estimate with two-step (default)
    result_2step = estimate_panel_var_gmm(
        data=data,
        var_lags=1,
        value_cols=["y1", "y2"],
        gmm_step="two-step",
        instrument_type="collapsed",
        max_instruments=8,
    )

    # Compare with one-step
    comparison = result_2step.compare_one_step_two_step()

    print(comparison["summary"])

    print("\nInterpretation:")
    print(f"- Max coefficient difference: {comparison['coef_diff_max']:.4f}")
    print(f"- Mean coefficient difference: {comparison['coef_diff_mean']:.4f}")

    if comparison["coef_diff_pct_max"] < 5:
        print("- One-step and two-step agree closely (good sign)")
    elif comparison["coef_diff_pct_max"] < 15:
        print("- Moderate difference - prefer two-step for efficiency")
    else:
        print("- Large difference - investigate potential misspecification")

    return result_2step, comparison


def example_3_instrument_types():
    """
    Example 3: Comparing 'all' vs 'collapsed' Instruments

    Demonstrates the difference between using all available lags as instruments
    vs collapsed instruments (Roodman 2009) to prevent proliferation.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Comparing 'all' vs 'collapsed' Instruments")
    print("=" * 80)

    # Generate data with moderate N
    data = generate_panel_var_data(N=40, T=18, K=2)

    # Estimate with 'all' instruments
    print("\n--- Estimation with instrument_type='all' ---")
    result_all = estimate_panel_var_gmm(
        data=data,
        var_lags=1,
        value_cols=["y1", "y2"],
        instrument_type="all",  # All lags as instruments
        max_instruments=None,  # No limit
    )

    print(f"\nNumber of instruments: {result_all.n_instruments}")
    print(f"Number of entities: {result_all.n_entities}")
    print(f"Ratio instruments/entities: {result_all.n_instruments / result_all.n_entities:.2f}")

    # Estimate with 'collapsed' instruments
    print("\n--- Estimation with instrument_type='collapsed' ---")
    result_collapsed = estimate_panel_var_gmm(
        data=data,
        var_lags=1,
        value_cols=["y1", "y2"],
        instrument_type="collapsed",  # Collapsed instruments
        max_instruments=10,
    )

    print(f"\nNumber of instruments: {result_collapsed.n_instruments}")
    print(f"Number of entities: {result_collapsed.n_entities}")
    print(
        f"Ratio instruments/entities: {result_collapsed.n_instruments / result_collapsed.n_entities:.2f}"
    )

    print("\nComparison:")
    print(f"- 'all' uses {result_all.n_instruments} instruments")
    print(f"- 'collapsed' uses {result_collapsed.n_instruments} instruments")
    print(
        f"- Reduction: {100 * (1 - result_collapsed.n_instruments / result_all.n_instruments):.1f}%"
    )

    print("\nRecommendation:")
    if result_all.n_instruments > result_all.n_entities:
        print("  WARNING: 'all' leads to instrument proliferation!")
        print("  Always use 'collapsed' for this sample size.")
    else:
        print("  'all' is acceptable but 'collapsed' is safer and faster.")

    return result_all, result_collapsed


def example_4_comparing_transformations():
    """
    Example 4: Comparing FOD vs FD Transformations

    Shows how to compare Forward Orthogonal Deviations vs First-Differences
    transformations. FOD is generally preferred but FD may be used for
    comparability with older papers.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Comparing FOD vs FD Transformations")
    print("=" * 80)

    # Generate balanced panel
    data_balanced = generate_panel_var_data(N=45, T=16, K=2)

    print("\n--- Balanced Panel ---")
    comparison_balanced = compare_transforms(
        data=data_balanced,
        var_lags=1,
        value_cols=["y1", "y2"],
        gmm_step="two-step",
        instrument_type="collapsed",
        max_instruments=8,
    )

    print(comparison_balanced["summary"])

    # Generate unbalanced panel
    print("\n--- Unbalanced Panel ---")
    # Create unbalanced version by randomly dropping observations
    np.random.seed(123)
    mask = np.random.rand(len(data_balanced)) > 0.15  # Drop 15% of obs
    data_unbalanced = data_balanced[mask].copy().reset_index(drop=True)

    comparison_unbalanced = compare_transforms(
        data=data_unbalanced,
        var_lags=1,
        value_cols=["y1", "y2"],
        gmm_step="two-step",
        instrument_type="collapsed",
        max_instruments=8,
    )

    print(comparison_unbalanced["summary"])

    print("\nKey insights:")
    print("1. In balanced panels: FOD and FD should give similar results")
    print("2. In unbalanced panels: FOD preserves more observations")
    print("3. FOD is generally preferred (Abrigo & Love 2016 implementation)")

    return comparison_balanced, comparison_unbalanced


def example_5_diagnostic_workflow():
    """
    Example 5: Complete Diagnostic Workflow

    Demonstrates a comprehensive workflow for GMM estimation including
    all diagnostic checks.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Complete Diagnostic Workflow")
    print("=" * 80)

    # Generate data
    data = generate_panel_var_data(N=55, T=18, K=2)

    # Step 1: Estimate model
    print("\n[STEP 1] Estimating model...")
    result = estimate_panel_var_gmm(
        data=data,
        var_lags=1,
        value_cols=["y1", "y2"],
        transform="fod",
        gmm_step="two-step",
        instrument_type="collapsed",
        max_instruments=10,
    )

    # Step 2: Check Hansen J test
    print("\n[STEP 2] Hansen J Test (Overidentification)")
    print("-" * 60)
    j_stat, j_pval = result.diagnostics.hansen_j_test()
    print(f"Hansen J statistic: {j_stat:.4f}")
    print(f"P-value: {j_pval:.4f}")

    if j_pval < 0.05:
        print("⚠ FAIL: Instruments may be invalid or model misspecified")
    elif j_pval > 0.99:
        print("⚠ WARNING: p-value very high - possible weak instruments")
    else:
        print("✓ PASS: Instruments appear valid")

    # Step 3: Check AR tests
    print("\n[STEP 3] Serial Correlation Tests")
    print("-" * 60)
    ar1_stat, ar1_pval = result.diagnostics.ar_test(order=1)
    ar2_stat, ar2_pval = result.diagnostics.ar_test(order=2)

    print(f"AR(1) test: z = {ar1_stat:.4f}, p-value = {ar1_pval:.4f}")
    if ar1_pval < 0.05:
        print("  ✓ Expected: AR(1) should reject (by construction)")
    else:
        print("  ⚠ Unexpected: AR(1) should typically reject")

    print(f"AR(2) test: z = {ar2_stat:.4f}, p-value = {ar2_pval:.4f}")
    if ar2_pval >= 0.05:
        print("  ✓ PASS: No second-order serial correlation (model adequate)")
    else:
        print("  ⚠ FAIL: AR(2) rejects - consider adding lags or check specification")

    # Step 4: Instrument diagnostics
    print("\n[STEP 4] Instrument Proliferation Diagnostics")
    print("-" * 60)
    print(result.instrument_diagnostics())

    # Step 5: Sensitivity analysis
    print("\n[STEP 5] Instrument Sensitivity Analysis")
    print("-" * 60)
    sensitivity = result.diagnostics.instrument_sensitivity_analysis(
        max_instrument_counts=[4, 6, 8, 10, 12]
    )

    print(f"Testing stability across {len(sensitivity['instrument_counts'])} instrument counts...")
    print(
        f"Max coefficient change: {sensitivity['max_coef_change']:.4f} ({sensitivity['max_coef_change_pct']:.1f}%)"
    )

    if sensitivity["is_stable"]:
        print("✓ STABLE: Coefficients stable across instrument counts")
    else:
        print("⚠ UNSTABLE: Coefficients vary significantly - reduce instruments")

    # Final verdict
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    checks = {
        "Hansen J": j_pval >= 0.05 and j_pval <= 0.99,
        "AR(2) no serial correlation": ar2_pval >= 0.05,
        "Instrument count acceptable": result.n_instruments <= result.n_entities,
        "Coefficient stability": sensitivity["is_stable"],
    }

    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")

    if all(checks.values()):
        print("\n🎉 All diagnostics PASSED - model is well-specified!")
    else:
        print("\n⚠ Some diagnostics FAILED - review model specification")

    return result


def example_6_var2_model():
    """
    Example 6: Estimating a VAR(2) Model

    Demonstrates estimation of higher-order VAR models.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: VAR(2) Model Estimation")
    print("=" * 80)

    # Generate VAR(2) data
    np.random.seed(456)
    N, T, K = 50, 22, 3

    # VAR(2) coefficients
    A1 = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.2, 0.3]])
    A2 = np.array([[0.2, 0.0, 0.1], [0.1, 0.2, 0.0], [0.1, 0.1, 0.2]])

    data = []
    for i in range(N):
        alpha_i = np.random.randn(K) * 0.5
        y = np.zeros((T, K))
        y[0] = alpha_i + np.random.randn(K) * 0.5
        y[1] = alpha_i + A1 @ y[0] + np.random.randn(K) * 0.3

        for t in range(2, T):
            y[t] = alpha_i + A1 @ y[t - 1] + A2 @ y[t - 2] + np.random.randn(K) * 0.3

        for t in range(T):
            data.append(
                {
                    "entity": i,
                    "time": t,
                    "gdp": y[t, 0],
                    "investment": y[t, 1],
                    "consumption": y[t, 2],
                }
            )

    df = pd.DataFrame(data)

    # Estimate VAR(2)
    result = estimate_panel_var_gmm(
        data=df,
        var_lags=2,  # VAR(2)
        value_cols=["gdp", "investment", "consumption"],
        transform="fod",
        gmm_step="two-step",
        instrument_type="collapsed",
        max_instruments=15,
    )

    print(result.summary())

    print("\nNote: VAR(2) models:")
    print("- Require more observations (loses 2 periods per entity)")
    print("- Have more parameters (K² × p)")
    print("- May need more instruments (adjust max_instruments)")

    return result


def main():
    """Run all examples."""
    print(
        """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    Panel VAR GMM Estimation Examples                        ║
║                                                                              ║
║  This script demonstrates various GMM estimation techniques for Panel VAR   ║
║  models using panelbox. Each example builds on the previous one.            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    )

    # Run examples
    result_1 = example_1_basic_gmm_estimation()
    result_2, comp_2 = example_2_comparing_gmm_steps()
    result_all, result_collapsed = example_3_instrument_types()
    comp_balanced, comp_unbalanced = example_4_comparing_transformations()
    result_5 = example_5_diagnostic_workflow()
    result_6 = example_6_var2_model()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Try with your own data")
    print("2. Experiment with different instrument counts")
    print("3. Run impulse response functions: result.irf()")
    print(
        "4. Run forecast error variance decomposition: result.forecast_error_variance_decomposition()"
    )
    print("5. Check out examples/var/instrument_diagnostics.py for advanced diagnostics")

    return {
        "example_1": result_1,
        "example_2": (result_2, comp_2),
        "example_3": (result_all, result_collapsed),
        "example_4": (comp_balanced, comp_unbalanced),
        "example_5": result_5,
        "example_6": result_6,
    }


if __name__ == "__main__":
    results = main()
