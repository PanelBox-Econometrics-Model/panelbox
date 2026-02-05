"""
Firm Growth Example
===================

Estimating dynamic firm growth using Difference and System GMM.

Model: size_it = γ size_{i,t-1} + β investment_it + δ age_it + η_i + ε_it

This example demonstrates:
- Realistic firm growth model with persistence
- Data preparation and exploration
- Comparison of Difference vs System GMM
- Complete diagnostic workflow
- Economic interpretation of results

Based on typical firm-level panel data structure.

Author: PanelBox Development Team
Date: January 2026
"""

import os
import sys

import numpy as np
import pandas as pd

# Add panelbox to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.linear_model import LinearRegression

from panelbox.gmm import DifferenceGMM, SystemGMM


def generate_firm_data(n_firms=300, n_years=10, gamma=0.65, beta_inv=0.15, beta_age=-0.02, seed=42):
    """
    Generate synthetic firm growth panel data.

    Parameters
    ----------
    n_firms : int
        Number of firms
    n_years : int
        Number of years
    gamma : float
        True persistence coefficient (size_t-1)
    beta_inv : float
        True effect of investment
    beta_age : float
        True effect of age (negative = older firms grow slower)
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Firm panel data
    """
    np.random.seed(seed)

    data = []

    for i in range(1, n_firms + 1):
        # Firm fixed effect (persistent productivity/quality)
        eta_i = np.random.normal(0, 0.5)

        # Firm age (varies across firms, grows over time)
        base_age = np.random.randint(1, 20)

        # Generate data over time
        size_it = np.zeros(n_years)
        size_it[0] = eta_i + np.random.normal(0, 0.3)  # Initial size

        for t in range(1, n_years):
            # Investment rate (somewhat persistent, with noise)
            if t == 1:
                investment_it = np.random.uniform(0.05, 0.25)
            else:
                investment_it = 0.6 * data[-1]["investment"] + 0.4 * np.random.uniform(0.05, 0.25)

            # Age increases
            age_it = base_age + t

            # Size evolution (dynamic panel)
            epsilon_it = np.random.normal(0, 0.25)
            size_it[t] = (
                gamma * size_it[t - 1]
                + beta_inv * investment_it
                + beta_age * age_it
                + eta_i
                + epsilon_it
            )

            # Store observation
            data.append(
                {
                    "firm_id": i,
                    "year": 2000 + t,
                    "size": size_it[t],
                    "investment": investment_it,
                    "age": age_it,
                }
            )

        # First observation (t=0)
        data.insert(
            len(data) - (n_years - 1),
            {
                "firm_id": i,
                "year": 2000,
                "size": size_it[0],
                "investment": np.random.uniform(0.05, 0.25),
                "age": base_age,
            },
        )

    df = pd.DataFrame(data)

    print("=" * 70)
    print("GENERATED FIRM GROWTH PANEL DATA")
    print("=" * 70)
    print(f"\nData structure:")
    print(f"  Firms: {n_firms}")
    print(f"  Years: {n_years} (2000-{2000 + n_years - 1})")
    print(f"  Total observations: {len(df)}")
    print(f"\nTrue parameters:")
    print(f"  γ (persistence): {gamma}")
    print(f"  β_inv (investment effect): {beta_inv}")
    print(f"  β_age (age effect): {beta_age}")
    print(f"\nDescriptive statistics:")
    print(df[["size", "investment", "age"]].describe())
    print()

    return df, gamma, beta_inv, beta_age


def estimate_bounds(df):
    """
    Estimate OLS and FE bounds for persistence coefficient.
    """
    print("=" * 70)
    print("STEP 1: Estimate OLS and FE Bounds")
    print("=" * 70)
    print()

    # Create lagged size
    df_sorted = df.sort_values(["firm_id", "year"]).copy()
    df_sorted["size_lag"] = df_sorted.groupby("firm_id")["size"].shift(1)
    df_clean = df_sorted.dropna().copy()  # Explicit copy to avoid SettingWithCopyWarning

    # OLS (upper bound - biased upward)
    X_ols = df_clean[["size_lag", "investment", "age"]].values
    y_ols = df_clean["size"].values
    ols = LinearRegression().fit(X_ols, y_ols)

    gamma_ols = ols.coef_[0]
    beta_inv_ols = ols.coef_[1]
    beta_age_ols = ols.coef_[2]

    print("OLS Results (Upper Bound):")
    print(f"  γ (size_lag):   {gamma_ols:.4f}")
    print(f"  β (investment): {beta_inv_ols:.4f}")
    print(f"  β (age):        {beta_age_ols:.4f}")
    print(f"  R²: {ols.score(X_ols, y_ols):.4f}")
    print()

    # Fixed Effects (lower bound - biased downward due to Nickell)
    for var in ["size", "size_lag", "investment", "age"]:
        df_clean[f"{var}_dm"] = df_clean[var] - df_clean.groupby("firm_id")[var].transform("mean")

    X_fe = df_clean[["size_lag_dm", "investment_dm", "age_dm"]].values
    y_fe = df_clean["size_dm"].values
    fe = LinearRegression(fit_intercept=False).fit(X_fe, y_fe)

    gamma_fe = fe.coef_[0]
    beta_inv_fe = fe.coef_[1]
    beta_age_fe = fe.coef_[2]

    print("Fixed Effects Results (Lower Bound):")
    print(f"  γ (size_lag):   {gamma_fe:.4f}")
    print(f"  β (investment): {beta_inv_fe:.4f}")
    print(f"  β (age):        {beta_age_fe:.4f}")
    print(f"  R²: {fe.score(X_fe, y_fe):.4f}")
    print()

    print(f"Credible Range for γ: [{gamma_fe:.3f}, {gamma_ols:.3f}]")
    print(f"Range width: {gamma_ols - gamma_fe:.3f}")
    print()

    if gamma_ols - gamma_fe > 0.1:
        print("✓ Large gap confirms fixed effects matter - GMM is appropriate")
    else:
        print("⚠ Small gap - GMM may not be necessary, but proceeding for demonstration")
    print()

    return gamma_ols, gamma_fe


def estimate_difference_gmm(df):
    """
    Estimate Difference GMM (Arellano-Bond).
    """
    print("=" * 70)
    print("STEP 2: Difference GMM (Arellano-Bond 1991)")
    print("=" * 70)
    print()

    gmm_diff = DifferenceGMM(
        data=df,
        dep_var="size",
        lags=1,
        id_var="firm_id",
        time_var="year",
        exog_vars=["investment", "age"],
        time_dummies=False,  # Not needed for simulated data
        collapse=True,  # Recommended to avoid instrument proliferation
        two_step=True,  # Two-step with Windmeijer correction
        robust=True,  # Robust standard errors
    )

    results_diff = gmm_diff.fit()
    print(results_diff.summary())
    print()

    # Extract coefficients
    gamma_diff = results_diff.params["L1.size"]
    beta_inv_diff = results_diff.params["investment"]
    beta_age_diff = results_diff.params["age"]

    se_gamma = results_diff.std_errors["L1.size"]
    se_inv = results_diff.std_errors["investment"]
    se_age = results_diff.std_errors["age"]

    return results_diff, gamma_diff, beta_inv_diff, beta_age_diff


def estimate_system_gmm(df):
    """
    Estimate System GMM (Blundell-Bond).

    Note: System GMM may fail with some synthetic data due to
    collinearity in level instruments. This is normal and shows
    that Difference GMM is sometimes more robust.
    """
    print("=" * 70)
    print("STEP 3: System GMM (Blundell-Bond 1998)")
    print("=" * 70)
    print()

    try:
        gmm_sys = SystemGMM(
            data=df,
            dep_var="size",
            lags=1,
            id_var="firm_id",
            time_var="year",
            exog_vars=["investment", "age"],
            time_dummies=False,
            collapse=True,
            two_step=True,
            robust=True,
            level_instruments={"max_lags": 1},
        )

        results_sys = gmm_sys.fit()
        print(results_sys.summary())
        print()

        # Extract coefficients
        gamma_sys = results_sys.params["L1.size"]
        beta_inv_sys = results_sys.params["investment"]
        beta_age_sys = results_sys.params["age"]

        return results_sys, gamma_sys, beta_inv_sys, beta_age_sys

    except (ValueError, np.linalg.LinAlgError) as e:
        print("⚠ System GMM failed to converge")
        print(f"  Error: {str(e)[:100]}")
        print()
        print("Possible reasons:")
        print("  - Collinearity in level instruments")
        print("  - Insufficient variation in data")
        print("  - Stationarity assumptions violated")
        print()
        print("→ This demonstrates that System GMM is not always applicable")
        print("→ Difference GMM is more robust in this case")
        print()

        return None, None, None, None


def compare_results(
    gamma_true,
    beta_inv_true,
    beta_age_true,
    gamma_ols,
    gamma_fe,
    gamma_diff,
    beta_inv_diff,
    beta_age_diff,
    results_diff,
    gamma_sys,
    beta_inv_sys,
    beta_age_sys,
    results_sys,
):
    """
    Compare all estimation methods.
    """
    print("=" * 70)
    print("STEP 4: Comparison of Methods")
    print("=" * 70)
    print()

    # Check if System GMM succeeded
    sys_available = results_sys is not None

    # Create comparison table
    comp_data = {
        "True": [gamma_true, beta_inv_true, beta_age_true],
        "OLS": [gamma_ols, np.nan, np.nan],
        "FE": [gamma_fe, np.nan, np.nan],
        "Diff GMM": [gamma_diff, beta_inv_diff, beta_age_diff],
        "Diff SE": [
            results_diff.std_errors["L1.size"],
            results_diff.std_errors["investment"],
            results_diff.std_errors["age"],
        ],
    }

    if sys_available:
        comp_data["Sys GMM"] = [gamma_sys, beta_inv_sys, beta_age_sys]
        comp_data["Sys SE"] = [
            results_sys.std_errors["L1.size"],
            results_sys.std_errors["investment"],
            results_sys.std_errors["age"],
        ]

    comparison = pd.DataFrame(
        comp_data, index=["γ (persistence)", "β_inv (investment)", "β_age (age)"]
    )

    print("Coefficient Estimates:")
    if sys_available:
        print(comparison[["True", "OLS", "FE", "Diff GMM", "Sys GMM"]].to_string())
    else:
        print(comparison[["True", "OLS", "FE", "Diff GMM"]].to_string())
    print()

    # Bias analysis
    bias_diff = gamma_diff - gamma_true
    bias_pct_diff = 100 * bias_diff / gamma_true

    print("Bias Analysis (γ persistence coefficient):")
    print(f"  True γ:          {gamma_true:.4f}")
    print(f"  OLS (upper):     {gamma_ols:.4f}  (bias: {gamma_ols - gamma_true:+.4f})")
    print(f"  FE (lower):      {gamma_fe:.4f}  (bias: {gamma_fe - gamma_true:+.4f})")
    print(f"  Difference GMM:  {gamma_diff:.4f}  (bias: {bias_diff:+.4f}, {bias_pct_diff:+.1f}%)")

    if sys_available:
        bias_sys = gamma_sys - gamma_true
        bias_pct_sys = 100 * bias_sys / gamma_true
        print(f"  System GMM:      {gamma_sys:.4f}  (bias: {bias_sys:+.4f}, {bias_pct_sys:+.1f}%)")
    print()

    # Check credible range
    print("Credible Range Check:")
    diff_in_range = gamma_fe < gamma_diff < gamma_ols
    print(f"  Difference GMM in [{gamma_fe:.3f}, {gamma_ols:.3f}]? {diff_in_range}")

    if sys_available:
        sys_in_range = gamma_fe < gamma_sys < gamma_ols
        print(f"  System GMM in [{gamma_fe:.3f}, {gamma_ols:.3f}]? {sys_in_range}")
    print()

    # Efficiency comparison (only if both available)
    if sys_available:
        se_diff = results_diff.std_errors["L1.size"]
        se_sys = results_sys.std_errors["L1.size"]
        efficiency_gain = (se_diff - se_sys) / se_diff * 100

        print("Efficiency Comparison:")
        print(f"  Difference GMM SE: {se_diff:.4f}")
        print(f"  System GMM SE:     {se_sys:.4f}")
        print(f"  Efficiency gain:   {efficiency_gain:.1f}% reduction in SE")
        print()


def diagnostic_summary(results_diff, results_sys):
    """
    Comprehensive diagnostic summary.
    """
    print("=" * 70)
    print("STEP 5: Diagnostic Tests Summary")
    print("=" * 70)
    print()

    sys_available = results_sys is not None

    # Difference GMM diagnostics
    print("Difference GMM Diagnostics:")
    print(f"  Observations: {results_diff.nobs}")
    print(f"  Instruments: {results_diff.n_instruments}")
    print(f"  Instrument ratio: {results_diff.instrument_ratio:.3f}")
    print()

    hansen_diff = results_diff.hansen_j.pvalue
    ar2_diff = results_diff.ar2_test.pvalue

    print(f"  Hansen J-test: p = {hansen_diff:.3f}")
    if 0.10 < hansen_diff < 0.25:
        print("    ✓ Instruments appear valid")
    elif hansen_diff < 0.10:
        print("    ✗ Instruments rejected")
    else:
        print("    ⚠ Possible weak instruments (very high p-value)")

    print(f"  AR(2) test: p = {ar2_diff:.3f}")
    if ar2_diff > 0.10:
        print("    ✓ Moment conditions valid")
    else:
        print("    ✗ Moment conditions invalid!")
    print()

    # System GMM diagnostics (if available)
    if sys_available:
        print("System GMM Diagnostics:")
        print(f"  Observations: {results_sys.nobs}")
        print(f"  Instruments: {results_sys.n_instruments}")
        print(f"  Instrument ratio: {results_sys.instrument_ratio:.3f}")
        print()

        hansen_sys = results_sys.hansen_j.pvalue
        ar2_sys = results_sys.ar2_test.pvalue

        print(f"  Hansen J-test: p = {hansen_sys:.3f}")
        if 0.10 < hansen_sys < 0.25:
            print("    ✓ Instruments appear valid")
        elif hansen_sys < 0.10:
            print("    ✗ Instruments rejected")
        else:
            print("    ⚠ Possible weak instruments")

        print(f"  AR(2) test: p = {ar2_sys:.3f}")
        if ar2_sys > 0.10:
            print("    ✓ Moment conditions valid")
        else:
            print("    ✗ Moment conditions invalid!")
        print()
    else:
        print("System GMM: Not available (estimation failed)")
        print()

    # Model selection
    print("=" * 70)
    print("MODEL SELECTION RECOMMENDATION")
    print("=" * 70)
    print()

    diff_valid = ar2_diff > 0.10 and 0.10 < hansen_diff < 0.50

    if not sys_available:
        if diff_valid:
            print("✓ RECOMMENDATION: Use Difference GMM")
            print("  Reason: Difference GMM valid, System GMM not available")
        else:
            print("✗ WARNING: Difference GMM fails diagnostics")
            print("  Action: Check model specification or try different data")
    else:
        sys_valid = ar2_sys > 0.10 and 0.10 < hansen_sys < 0.50
        se_diff = results_diff.std_errors["L1.size"]
        se_sys = results_sys.std_errors["L1.size"]

        if diff_valid and sys_valid:
            if se_sys < se_diff * 0.9:
                print("✓ RECOMMENDATION: Use System GMM")
                print("  Reason: Both valid, System GMM more efficient (smaller SE)")
            else:
                print("✓ RECOMMENDATION: Use Difference GMM")
                print("  Reason: Both valid, similar efficiency, Difference more robust")
        elif diff_valid:
            print("✓ RECOMMENDATION: Use Difference GMM")
            print("  Reason: Difference GMM valid, System GMM fails diagnostics")
        elif sys_valid:
            print("✓ RECOMMENDATION: Use System GMM")
            print("  Reason: System GMM valid, Difference GMM fails diagnostics")
        else:
            print("✗ WARNING: Both models fail diagnostics")
            print("  Action: Check model specification")
    print()


def main():
    """
    Run complete firm growth example.
    """
    print()
    print("=" * 70)
    print("FIRM GROWTH EXAMPLE: Dynamic Panel Data with GMM")
    print("=" * 70)
    print()
    print("This example estimates a firm growth model:")
    print("  size_it = γ size_{i,t-1} + β_inv investment_it + β_age age_it + η_i + ε_it")
    print()
    print("Demonstrating:")
    print("  - Difference GMM (Arellano-Bond 1991)")
    print("  - System GMM (Blundell-Bond 1998)")
    print("  - Complete diagnostic workflow")
    print("  - Model selection criteria")
    print()

    # Generate data
    df, gamma_true, beta_inv_true, beta_age_true = generate_firm_data(
        n_firms=300, n_years=10, gamma=0.65, beta_inv=0.15, beta_age=-0.02
    )

    # Estimate bounds
    gamma_ols, gamma_fe = estimate_bounds(df)

    # Estimate GMM models
    results_diff, gamma_diff, beta_inv_diff, beta_age_diff = estimate_difference_gmm(df)
    results_sys, gamma_sys, beta_inv_sys, beta_age_sys = estimate_system_gmm(df)

    # Compare results
    compare_results(
        gamma_true,
        beta_inv_true,
        beta_age_true,
        gamma_ols,
        gamma_fe,
        gamma_diff,
        beta_inv_diff,
        beta_age_diff,
        results_diff,
        gamma_sys,
        beta_inv_sys,
        beta_age_sys,
        results_sys,
    )

    # Diagnostic summary
    diagnostic_summary(results_diff, results_sys)

    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. OLS is biased UPWARD (ignores fixed effects)")
    print("2. FE is biased DOWNWARD (Nickell bias with small T)")
    print("3. GMM provides consistent estimates")
    print()
    print("4. System GMM is often more efficient than Difference GMM")
    print("   → Smaller standard errors with valid diagnostics")
    print()
    print("5. Always check diagnostics:")
    print("   → AR(2) test: p > 0.10 (critical)")
    print("   → Hansen J: 0.10 < p < 0.25 (ideal)")
    print("   → Instrument ratio < 1.0")
    print()
    print("6. Choose model based on:")
    print("   → Diagnostic test results")
    print("   → Standard error comparison")
    print("   → Economic plausibility")
    print()
    print("=" * 70)
    print("✓ Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
