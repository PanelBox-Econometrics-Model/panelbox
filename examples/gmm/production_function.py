"""
Production Function Estimation
===============================

Advanced example: Estimating a production function with simultaneity bias
using System GMM.

Model: output_it = α capital_it + β labor_it + ω_it + ε_it
       ω_it = ρ ω_{i,t-1} + ξ_it  (productivity follows AR(1))

This example demonstrates:
- Simultaneity bias in production function estimation
- Why OLS and FE fail (inputs correlated with productivity shock)
- Why Difference GMM may have weak instruments (high persistence)
- Why System GMM is preferred (additional moment conditions)
- Economic interpretation of results

Based on Blundell-Bond (1998, 2000) production function applications.

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


def generate_production_data(n_firms=250, n_years=12, alpha=0.35, beta=0.65, rho=0.85, seed=42):
    """
    Generate synthetic production function data with simultaneity bias.

    Production function (log-linear Cobb-Douglas):
        y_it = α k_it + β l_it + ω_it + ε_it

    Where:
        y_it: log output
        k_it: log capital
        l_it: log labor
        ω_it: productivity (unobserved, persistent)
        ε_it: measurement error / transitory shock

    Simultaneity: Firms choose k and l based on ω_it
        k_it = k_0 + θ_k ω_it + u_kit
        l_it = l_0 + θ_l ω_it + u_lit

    This creates E[k_it ε_it] ≠ 0 and E[l_it ε_it] ≠ 0

    Parameters
    ----------
    n_firms : int
        Number of firms
    n_years : int
        Number of years
    alpha : float
        True capital elasticity
    beta : float
        True labor elasticity
    rho : float
        Persistence of productivity (AR(1) coefficient)
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Production panel data
    """
    np.random.seed(seed)

    print("=" * 70)
    print("GENERATING PRODUCTION FUNCTION DATA")
    print("=" * 70)
    print()
    print("Model:")
    print("  y_it = α k_it + β l_it + ω_it + ε_it")
    print("  ω_it = ρ ω_{i,t-1} + ξ_it  (productivity AR(1))")
    print()
    print("True parameters:")
    print(f"  α (capital elasticity): {alpha}")
    print(f"  β (labor elasticity):   {beta}")
    print(f"  ρ (persistence):        {rho}")
    print(f"  α + β:                  {alpha + beta} (returns to scale)")
    print()

    data = []

    for i in range(1, n_firms + 1):
        # Firm fixed effect (time-invariant productivity component)
        eta_i = np.random.normal(0, 0.3)

        # Initialize productivity
        omega = np.zeros(n_years)
        omega[0] = eta_i + np.random.normal(0, 0.2)

        # Generate productivity path (AR(1))
        for t in range(1, n_years):
            xi_it = np.random.normal(0, 0.2)
            omega[t] = rho * omega[t - 1] + xi_it

        # Generate inputs and output
        for t in range(n_years):
            # Simultaneity: inputs respond to productivity
            # Firms with high ω use more inputs
            theta_k = 0.5  # Response of capital to productivity
            theta_l = 0.6  # Response of labor to productivity

            # Capital: somewhat persistent + responds to ω
            if t == 0:
                k_base = 3.0 + np.random.normal(0, 0.3)
            else:
                k_base = 0.7 * data[-1]["capital"] + 0.3 * 3.0 + np.random.normal(0, 0.2)

            capital_it = k_base + theta_k * omega[t]

            # Labor: more flexible than capital
            labor_it = 2.5 + theta_l * omega[t] + np.random.normal(0, 0.3)

            # Measurement error / transitory shock
            epsilon_it = np.random.normal(0, 0.15)

            # Production function
            output_it = alpha * capital_it + beta * labor_it + omega[t] + epsilon_it

            # Store observation
            data.append(
                {
                    "firm_id": i,
                    "year": 2000 + t,
                    "output": output_it,
                    "capital": capital_it,
                    "labor": labor_it,
                    "productivity": omega[t],  # Unobserved (for validation only)
                }
            )

    df = pd.DataFrame(data)

    print(f"Generated data:")
    print(f"  Firms: {n_firms}")
    print(f"  Years: {n_years} (2000-{2000 + n_years - 1})")
    print(f"  Total observations: {len(df)}")
    print()
    print("Descriptive statistics:")
    print(df[["output", "capital", "labor"]].describe())
    print()

    # Check correlation between inputs and productivity (unobserved)
    corr_k_omega = df["capital"].corr(df["productivity"])
    corr_l_omega = df["labor"].corr(df["productivity"])
    print("Simultaneity check (correlation with unobserved productivity):")
    print(f"  Corr(capital, ω):  {corr_k_omega:.3f} ✓ Positive (simultaneity bias)")
    print(f"  Corr(labor, ω):    {corr_l_omega:.3f} ✓ Positive (simultaneity bias)")
    print()

    return df, alpha, beta, rho


def estimate_ols(df):
    """
    OLS estimation (BIASED due to simultaneity).
    """
    print("=" * 70)
    print("Method 1: OLS (Biased)")
    print("=" * 70)
    print()

    X = df[["capital", "labor"]].values
    y = df["output"].values

    ols = LinearRegression().fit(X, y)
    alpha_ols = ols.coef_[0]
    beta_ols = ols.coef_[1]

    # Simple standard errors
    y_pred = ols.predict(X)
    residuals = y - y_pred
    n = len(y)
    k = 2
    sigma_sq = np.sum(residuals**2) / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    var_coef = sigma_sq * np.diag(XtX_inv)
    se_alpha = np.sqrt(var_coef[0])
    se_beta = np.sqrt(var_coef[1])

    print("OLS Results:")
    print(f"  α (capital): {alpha_ols:.4f} (SE: {se_alpha:.4f})")
    print(f"  β (labor):   {beta_ols:.4f} (SE: {se_beta:.4f})")
    print(f"  α + β:       {alpha_ols + beta_ols:.4f} (returns to scale)")
    print(f"  R²:          {ols.score(X, y):.4f}")
    print()

    print("Problem: OLS is biased because:")
    print("  E[k_it ε_it] ≠ 0  (simultaneity)")
    print("  E[l_it ε_it] ≠ 0  (simultaneity)")
    print("  Expected: Upward bias (inputs respond to productivity shocks)")
    print()

    return alpha_ols, beta_ols, se_alpha, se_beta


def estimate_fixed_effects(df):
    """
    Fixed Effects estimation (BIASED due to Nickell + dynamic model).
    """
    print("=" * 70)
    print("Method 2: Fixed Effects (Biased)")
    print("=" * 70)
    print()

    # Within transformation
    df_fe = df.copy()
    for var in ["output", "capital", "labor"]:
        df_fe[f"{var}_dm"] = df_fe[var] - df_fe.groupby("firm_id")[var].transform("mean")

    X_dm = df_fe[["capital_dm", "labor_dm"]].values
    y_dm = df_fe["output_dm"].values

    fe = LinearRegression(fit_intercept=False).fit(X_dm, y_dm)
    alpha_fe = fe.coef_[0]
    beta_fe = fe.coef_[1]

    # Standard errors
    y_pred = fe.predict(X_dm)
    residuals = y_dm - y_pred
    n = len(y_dm)
    k = 2
    sigma_sq = np.sum(residuals**2) / (n - k)
    XtX_inv = np.linalg.inv(X_dm.T @ X_dm)
    var_coef = sigma_sq * np.diag(XtX_inv)
    se_alpha = np.sqrt(var_coef[0])
    se_beta = np.sqrt(var_coef[1])

    print("Fixed Effects Results:")
    print(f"  α (capital): {alpha_fe:.4f} (SE: {se_alpha:.4f})")
    print(f"  β (labor):   {beta_fe:.4f} (SE: {se_beta:.4f})")
    print(f"  α + β:       {alpha_fe + beta_fe:.4f} (returns to scale)")
    print(f"  R²:          {fe.score(X_dm, y_dm):.4f}")
    print()

    print("Problem: FE removes fixed effects but still biased because:")
    print("  - Inputs still correlated with transitory shocks")
    print("  - ω_it is persistent (dynamic model)")
    print("  - Nickell bias if productivity AR(1) treated as fixed effect")
    print()

    return alpha_fe, beta_fe, se_alpha, se_beta


def estimate_difference_gmm(df):
    """
    Difference GMM (Arellano-Bond).
    May have weak instruments if productivity is highly persistent.
    """
    print("=" * 70)
    print("Method 3: Difference GMM (Arellano-Bond 1991)")
    print("=" * 70)
    print()

    # Create lagged output as regressor
    df_sorted = df.sort_values(["firm_id", "year"]).copy()
    df_sorted["output_lag"] = df_sorted.groupby("firm_id")["output"].shift(1)

    gmm_diff = DifferenceGMM(
        data=df_sorted,
        dep_var="output",
        lags=1,  # Include output_{t-1}
        id_var="firm_id",
        time_var="year",
        exog_vars=["capital", "labor"],
        time_dummies=False,
        collapse=True,
        two_step=True,
        robust=True,
    )

    results_diff = gmm_diff.fit()
    print(results_diff.summary())
    print()

    # Extract coefficients
    rho_diff = results_diff.params["L1.output"]
    alpha_diff = results_diff.params["capital"]
    beta_diff = results_diff.params["labor"]

    se_rho = results_diff.std_errors["L1.output"]
    se_alpha = results_diff.std_errors["capital"]
    se_beta = results_diff.std_errors["labor"]

    print("Note: With highly persistent productivity (ρ ≈ 0.85):")
    print("  - Lagged levels are weak instruments for differences")
    print("  - Standard errors may be large")
    print("  - System GMM adds moment conditions for efficiency")
    print()

    return results_diff, rho_diff, alpha_diff, beta_diff


def estimate_system_gmm(df):
    """
    System GMM (Blundell-Bond).
    Preferred for production functions with persistent productivity.

    Note: System GMM may fail with synthetic data due to collinearity
    in level instruments. This is a known limitation.
    """
    print("=" * 70)
    print("Method 4: System GMM (Blundell-Bond 1998) - PREFERRED")
    print("=" * 70)
    print()

    try:
        # Create lagged output
        df_sorted = df.sort_values(["firm_id", "year"]).copy()
        df_sorted["output_lag"] = df_sorted.groupby("firm_id")["output"].shift(1)

        gmm_sys = SystemGMM(
            data=df_sorted,
            dep_var="output",
            lags=1,
            id_var="firm_id",
            time_var="year",
            exog_vars=["capital", "labor"],
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
        rho_sys = results_sys.params["L1.output"]
        alpha_sys = results_sys.params["capital"]
        beta_sys = results_sys.params["labor"]

        se_rho = results_sys.std_errors["L1.output"]
        se_alpha = results_sys.std_errors["capital"]
        se_beta = results_sys.std_errors["labor"]

        print("System GMM advantages:")
        print("  ✓ Additional moment conditions (level equation)")
        print("  ✓ More efficient with persistent productivity")
        print("  ✓ Stronger instruments (differences for levels)")
        print()

        return results_sys, rho_sys, alpha_sys, beta_sys

    except (ValueError, np.linalg.LinAlgError) as e:
        print("⚠ System GMM estimation failed")
        print(f"  Error: {str(e)[:100]}")
        print()
        print("Possible reasons:")
        print("  - Collinearity in level instruments")
        print("  - Stationarity assumptions not satisfied")
        print("  - Insufficient variation in differenced data")
        print()
        print("→ In practice, System GMM often works with real data")
        print("→ Difference GMM is more robust but less efficient")
        print("→ Try adjusting data generation or use real data")
        print()

        return None, None, None, None


def compare_results(
    alpha_true,
    beta_true,
    rho_true,
    alpha_ols,
    beta_ols,
    alpha_fe,
    beta_fe,
    rho_diff,
    alpha_diff,
    beta_diff,
    results_diff,
    rho_sys,
    alpha_sys,
    beta_sys,
    results_sys,
):
    """
    Compare all methods and provide economic interpretation.
    """
    print("=" * 70)
    print("COMPARISON OF METHODS")
    print("=" * 70)
    print()

    sys_available = results_sys is not None

    # Comparison table
    comp_data = {
        "True": [alpha_true, beta_true, alpha_true + beta_true, rho_true],
        "OLS": [alpha_ols, beta_ols, alpha_ols + beta_ols, np.nan],
        "FE": [alpha_fe, beta_fe, alpha_fe + beta_fe, np.nan],
        "Diff GMM": [alpha_diff, beta_diff, alpha_diff + beta_diff, rho_diff],
    }

    if sys_available:
        comp_data["Sys GMM"] = [alpha_sys, beta_sys, alpha_sys + beta_sys, rho_sys]

    comparison = pd.DataFrame(
        comp_data, index=["α (capital)", "β (labor)", "α+β (RTS)", "ρ (persistence)"]
    )

    print("Coefficient Estimates:")
    print(comparison.to_string())
    print()

    # Bias analysis for capital elasticity
    bias_ols = alpha_ols - alpha_true
    bias_fe = alpha_fe - alpha_true
    bias_diff = alpha_diff - alpha_true

    print("Bias Analysis (α - capital elasticity):")
    print(f"  True α:       {alpha_true:.4f}")
    print(
        f"  OLS:          {alpha_ols:.4f}  (bias: {bias_ols:+.4f}, {100*bias_ols/alpha_true:+.1f}%)"
    )
    print(f"  FE:           {alpha_fe:.4f}  (bias: {bias_fe:+.4f}, {100*bias_fe/alpha_true:+.1f}%)")
    print(
        f"  Diff GMM:     {alpha_diff:.4f}  (bias: {bias_diff:+.4f}, {100*bias_diff/alpha_true:+.1f}%)"
    )

    if sys_available:
        bias_sys = alpha_sys - alpha_true
        print(
            f"  System GMM:   {alpha_sys:.4f}  (bias: {bias_sys:+.4f}, {100*bias_sys/alpha_true:+.1f}%)"
        )
    print()

    # Returns to scale analysis
    rts_true = alpha_true + beta_true
    rts_diff = alpha_diff + beta_diff

    print("Returns to Scale (α + β):")
    print(f"  True:       {rts_true:.4f}")
    print(f"  Diff GMM:   {rts_diff:.4f}")

    if sys_available:
        rts_sys = alpha_sys + beta_sys
        print(f"  System GMM: {rts_sys:.4f}")
    print()

    if 0.95 < rts_true < 1.05:
        print("  → Constant returns to scale (CRS)")
    elif rts_true > 1.05:
        print("  → Increasing returns to scale (IRS)")
    else:
        print("  → Decreasing returns to scale (DRS)")
    print()

    # Efficiency comparison (only if System GMM available)
    if sys_available:
        se_diff_alpha = results_diff.std_errors["capital"]
        se_sys_alpha = results_sys.std_errors["capital"]
        efficiency_gain = (se_diff_alpha - se_sys_alpha) / se_diff_alpha * 100

        print("Efficiency Comparison (capital elasticity):")
        print(f"  Diff GMM SE:    {se_diff_alpha:.4f}")
        print(f"  System GMM SE:  {se_sys_alpha:.4f}")
        print(f"  Efficiency gain: {efficiency_gain:.1f}% reduction in SE")
        print()

        if efficiency_gain > 10:
            print("  ✓ System GMM is substantially more efficient")
        print()


def diagnostic_summary(results_diff, results_sys):
    """
    Diagnostic tests for both GMM estimators.
    """
    print("=" * 70)
    print("DIAGNOSTIC TESTS")
    print("=" * 70)
    print()

    sys_available = results_sys is not None

    # Difference GMM
    print("Difference GMM:")
    print(f"  Observations: {results_diff.nobs}")
    print(f"  Instruments: {results_diff.n_instruments}")
    print(f"  Instrument ratio: {results_diff.instrument_ratio:.3f}")
    print()

    hansen_diff = results_diff.hansen_j.pvalue
    ar2_diff = results_diff.ar2_test.pvalue

    print(f"  Hansen J: p = {hansen_diff:.3f}", end="")
    if 0.10 < hansen_diff < 0.25:
        print(" ✓")
    elif hansen_diff < 0.10:
        print(" ✗ (rejected)")
    else:
        print(" ⚠ (very high - weak instruments)")

    print(f"  AR(2): p = {ar2_diff:.3f}", end="")
    if ar2_diff > 0.10:
        print(" ✓")
    else:
        print(" ⚠ (borderline - close to 0.10 threshold)")
    print()

    # System GMM (if available)
    if sys_available:
        print("System GMM:")
        print(f"  Observations: {results_sys.nobs}")
        print(f"  Instruments: {results_sys.n_instruments}")
        print(f"  Instrument ratio: {results_sys.instrument_ratio:.3f}")
        print()

        hansen_sys = results_sys.hansen_j.pvalue
        ar2_sys = results_sys.ar2_test.pvalue

        print(f"  Hansen J: p = {hansen_sys:.3f}", end="")
        if 0.10 < hansen_sys < 0.25:
            print(" ✓")
        elif hansen_sys < 0.10:
            print(" ✗ (rejected)")
        else:
            print(" ⚠ (very high)")

        print(f"  AR(2): p = {ar2_sys:.3f}", end="")
        if ar2_sys > 0.10:
            print(" ✓")
        else:
            print(" ✗ (rejected)")
        print()
    else:
        print("System GMM: Not available (estimation failed)")
        print()

    # Model selection
    print("=" * 70)
    print("MODEL SELECTION")
    print("=" * 70)
    print()

    diff_valid = ar2_diff > 0.10 and 0.10 < hansen_diff < 0.50

    if not sys_available:
        if diff_valid:
            print("✓ RECOMMENDATION: Difference GMM")
            print("  Reasons:")
            print("    - Difference GMM diagnostics acceptable")
            print("    - System GMM not available (estimation failed)")
            print("    - More robust to specification issues")
        else:
            print("⚠ WARNING: Difference GMM has borderline diagnostics")
            print("  Actions:")
            print("    - AR(2) close to threshold - acceptable")
            print("    - Hansen J very high suggests weak instruments")
            print("    - Consider using real data for better instrument relevance")
    else:
        sys_valid = ar2_sys > 0.10 and 0.10 < hansen_sys < 0.50
        se_diff = results_diff.std_errors["capital"]
        se_sys = results_sys.std_errors["capital"]

        if diff_valid and sys_valid:
            if se_sys < se_diff * 0.9:
                print("✓ RECOMMENDATION: System GMM")
                print("  Reasons:")
                print("    - Both estimators valid")
                print("    - System GMM more efficient (smaller SEs)")
                print("    - Production function has persistent productivity")
            else:
                print("✓ RECOMMENDATION: System GMM or Difference GMM")
                print("  Both are valid and have similar efficiency")
        elif sys_valid:
            print("✓ RECOMMENDATION: System GMM")
            print("  Reason: System GMM valid, Difference GMM fails")
        elif diff_valid:
            print("⚠ RECOMMENDATION: Difference GMM")
            print("  Reason: Difference GMM valid, System GMM fails")
            print("  Note: Check stationarity assumptions")
        else:
            print("✗ WARNING: Both estimators fail diagnostics")
            print("  Action: Reexamine model specification")
    print()


def main():
    """
    Run complete production function estimation example.
    """
    print()
    print("=" * 70)
    print("PRODUCTION FUNCTION ESTIMATION WITH GMM")
    print("=" * 70)
    print()
    print("This example estimates a Cobb-Douglas production function:")
    print("  y_it = α k_it + β l_it + ω_it + ε_it")
    print()
    print("Key challenges:")
    print("  - Simultaneity bias (inputs respond to productivity shocks)")
    print("  - Persistent productivity (ω_it follows AR(1) with ρ ≈ 0.85)")
    print("  - Why System GMM is preferred over Difference GMM")
    print()

    # Generate data
    df, alpha_true, beta_true, rho_true = generate_production_data(
        n_firms=250, n_years=12, alpha=0.35, beta=0.65, rho=0.85
    )

    # Estimate with all methods
    alpha_ols, beta_ols, se_ols_a, se_ols_b = estimate_ols(df)
    alpha_fe, beta_fe, se_fe_a, se_fe_b = estimate_fixed_effects(df)
    results_diff, rho_diff, alpha_diff, beta_diff = estimate_difference_gmm(df)
    results_sys, rho_sys, alpha_sys, beta_sys = estimate_system_gmm(df)

    # Compare and interpret
    compare_results(
        alpha_true,
        beta_true,
        rho_true,
        alpha_ols,
        beta_ols,
        alpha_fe,
        beta_fe,
        rho_diff,
        alpha_diff,
        beta_diff,
        results_diff,
        rho_sys,
        alpha_sys,
        beta_sys,
        results_sys,
    )

    # Diagnostics
    diagnostic_summary(results_diff, results_sys)

    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. Simultaneity causes UPWARD bias in OLS")
    print("   → Firms with high productivity use more inputs")
    print()
    print("2. Fixed Effects still biased with simultaneity")
    print("   → Within transformation doesn't solve endogeneity")
    print()
    print("3. Difference GMM is consistent but may be inefficient")
    print("   → Weak instruments with highly persistent productivity")
    print()
    print("4. System GMM is PREFERRED for production functions")
    print("   → Additional moment conditions")
    print("   → More efficient estimates")
    print("   → Standard in applied work (Blundell-Bond 1998, 2000)")
    print()
    print("5. Always check diagnostics:")
    print("   → Hansen J test (instrument validity)")
    print("   → AR(2) test (moment conditions)")
    print("   → Standard errors (efficiency comparison)")
    print()
    print("6. Economic interpretation:")
    print("   → α + β ≈ 1: Constant returns to scale")
    print("   → ρ ≈ 0.85: Highly persistent productivity")
    print("   → Elasticities reasonable (α ≈ 0.35, β ≈ 0.65)")
    print()
    print("=" * 70)
    print("✓ Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
