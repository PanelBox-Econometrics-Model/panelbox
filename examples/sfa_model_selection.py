"""
Stochastic Frontier Analysis: Model Selection Workflow

This example demonstrates a comprehensive workflow for selecting
the best SFA model specification:

1. Test for presence of inefficiency (OLS vs SFA)
2. Compare distributional assumptions
3. Variance decomposition
4. Test functional form (Cobb-Douglas vs Translog)
5. Test returns to scale

References:
    - Kodde & Palm (1986). Wald criteria for jointly testing equality
      and inequality restrictions. Econometrica.
    - Vuong (1989). Likelihood ratio tests for model selection.
    - Coelli (1995). Estimators and hypothesis tests for a stochastic
      frontier function: A Monte Carlo analysis.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from panelbox.frontier import (
    StochasticFrontier,
    add_translog,
    compare_nested_distributions,
    inefficiency_presence_test,
    skewness_test,
    vuong_test,
)


def generate_production_data(n=200, sigma_v=0.2, sigma_u=0.3, returns_to_scale=1.0):
    """Generate simulated production data for demonstration.

    Parameters:
        n: Number of observations
        sigma_v: Standard deviation of noise
        sigma_u: Standard deviation of inefficiency
        returns_to_scale: Returns to scale (1.0 = CRS, >1 = IRS, <1 = DRS)

    Returns:
        DataFrame with production data
    """
    np.random.seed(42)

    # Generate inputs (in levels)
    capital = np.random.lognormal(mean=2, sigma=0.5, size=n)
    labor = np.random.lognormal(mean=1.5, sigma=0.5, size=n)

    # Take logs for Cobb-Douglas
    ln_K = np.log(capital)
    ln_L = np.log(labor)

    # Generate errors
    v = np.random.normal(0, sigma_v, n)  # Random noise
    u = np.abs(np.random.normal(0, sigma_u, n))  # Inefficiency (always positive)

    # Production function: y = exp(β₀) * K^β_K * L^β_L * exp(v - u)
    # In logs: ln(y) = β₀ + β_K·ln(K) + β_L·ln(L) + v - u

    # Set elasticities to achieve desired RTS
    beta_K = 0.4 * returns_to_scale
    beta_L = 0.6 * returns_to_scale
    beta_0 = 1.0

    ln_y = beta_0 + beta_K * ln_K + beta_L * ln_L + v - u

    return pd.DataFrame(
        {
            "output": np.exp(ln_y),
            "ln_output": ln_y,
            "capital": capital,
            "labor": labor,
            "ln_capital": ln_K,
            "ln_labor": ln_L,
        }
    )


def main():
    """Run complete model selection workflow."""

    print("=" * 80)
    print("STOCHASTIC FRONTIER ANALYSIS: MODEL SELECTION WORKFLOW")
    print("=" * 80)
    print()

    # ========================================================================
    # STEP 1: Generate data and preliminary analysis
    # ========================================================================

    print("STEP 1: Generate Data and Preliminary Analysis")
    print("-" * 80)

    data = generate_production_data(n=200, sigma_v=0.2, sigma_u=0.3, returns_to_scale=1.0)

    print(f"Sample size: {len(data)}")
    print(f"\nDescriptive statistics:")
    print(data[["ln_output", "ln_capital", "ln_labor"]].describe())
    print()

    # ========================================================================
    # STEP 2: Test for presence of inefficiency (OLS vs SFA)
    # ========================================================================

    print("\nSTEP 2: Test for Presence of Inefficiency")
    print("-" * 80)

    # Fit OLS for baseline
    X = sm.add_constant(data[["ln_capital", "ln_labor"]])
    y = data["ln_output"].values
    ols = sm.OLS(y, X).fit()
    residuals_ols = ols.resid.values

    # Compute OLS log-likelihood
    sigma_ols = np.std(residuals_ols, ddof=3)
    n = len(y)
    loglik_ols = (
        -0.5 * n * np.log(2 * np.pi * sigma_ols**2) - 0.5 * np.sum(residuals_ols**2) / sigma_ols**2
    )

    print(f"OLS log-likelihood: {loglik_ols:.4f}")
    print(f"OLS residual std: {sigma_ols:.4f}")

    # Skewness test (preliminary diagnostic)
    skew_result = skewness_test(residuals_ols, frontier_type="production")
    print(f"\nSkewness test:")
    print(f"  Skewness: {skew_result['skewness']:.4f}")
    print(f"  Expected sign: {skew_result['expected_sign']}")
    print(f"  Correct sign: {skew_result['correct_sign']}")
    if skew_result["warning"]:
        print(f"  {skew_result['warning']}")

    # Fit SFA with half-normal
    print(f"\nFitting SFA model (half-normal)...")
    sf_half = StochasticFrontier(
        data=data,
        depvar="ln_output",
        exog=["ln_capital", "ln_labor"],
        frontier="production",
        dist="half_normal",
    )
    result_half = sf_half.fit()
    print(f"SFA log-likelihood: {result_half.loglik:.4f}")

    # Run inefficiency presence test
    print(f"\nInefficiency presence test (LR with mixed chi-square):")
    ineff_test = inefficiency_presence_test(
        loglik_sfa=result_half.loglik,
        loglik_ols=loglik_ols,
        residuals_ols=residuals_ols,
        frontier_type="production",
        distribution="half_normal",
    )

    print(f"  LR statistic: {ineff_test['lr_statistic']:.4f}")
    print(f"  P-value: {ineff_test['pvalue']:.4f}")
    print(f"  Critical value (5%): {ineff_test['critical_values']['5%']:.4f}")
    print(f"  Conclusion: {ineff_test['conclusion']}")
    print(f"  {ineff_test['interpretation']}")

    # ========================================================================
    # STEP 3: Compare distributional assumptions
    # ========================================================================

    print("\n\nSTEP 3: Compare Distributional Assumptions")
    print("-" * 80)

    # Fit models with different distributions
    print("Fitting models with different distributions...")

    distributions = ["half_normal", "exponential", "truncated_normal"]
    results = {}

    for dist in distributions:
        print(f"  - {dist}...", end="")
        sf = StochasticFrontier(
            data=data,
            depvar="ln_output",
            exog=["ln_capital", "ln_labor"],
            frontier="production",
            dist=dist,
        )
        results[dist] = sf.fit()
        print(f" done (LL: {results[dist].loglik:.4f})")

    # Compare using information criteria
    print(f"\nModel comparison:")
    comparison_data = []
    for dist, result in results.items():
        comparison_data.append(
            {
                "Distribution": dist,
                "LogLik": result.loglik,
                "AIC": result.aic,
                "BIC": result.bic,
                "σ_v": result.sigma_v,
                "σ_u": result.sigma_u,
                "γ": result.gamma,
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df["Best_AIC"] = comparison_df["AIC"] == comparison_df["AIC"].min()
    comparison_df["Best_BIC"] = comparison_df["BIC"] == comparison_df["BIC"].min()
    print(comparison_df.to_string(index=False))

    # Nested model test: half-normal vs truncated-normal
    print(f"\n\nNested model test: half-normal vs truncated-normal")
    nested_test = compare_nested_distributions(
        loglik_restricted=results["half_normal"].loglik,
        loglik_unrestricted=results["truncated_normal"].loglik,
        dist_restricted="half_normal",
        dist_unrestricted="truncated_normal",
    )
    print(f"  LR statistic: {nested_test['lr_statistic']:.4f}")
    print(f"  P-value: {nested_test['pvalue']:.4f}")
    print(f"  Conclusion: {nested_test['conclusion']}")
    print(f"  {nested_test['interpretation']}")

    # ========================================================================
    # STEP 4: Variance decomposition
    # ========================================================================

    print("\n\nSTEP 4: Variance Decomposition")
    print("-" * 80)

    # Use best model (by AIC)
    best_dist = comparison_df.loc[comparison_df["AIC"].idxmin(), "Distribution"]
    best_result = results[best_dist]

    print(f"Using best model: {best_dist}")
    print()

    decomp = best_result.variance_decomposition(ci_level=0.95)

    print(f"Variance decomposition:")
    print(f"  σ²_v (noise):         {decomp['sigma_sq_v']:.6f}")
    print(f"  σ²_u (inefficiency):  {decomp['sigma_sq_u']:.6f}")
    print(f"  σ² (total):           {decomp['sigma_sq']:.6f}")
    print()
    print(f"  γ = σ²_u / σ²:        {decomp['gamma']:.4f}")
    print(f"  95% CI for γ:         [{decomp['gamma_ci'][0]:.4f}, {decomp['gamma_ci'][1]:.4f}]")
    print()
    print(f"  λ = σ_u / σ_v:        {decomp['lambda_param']:.4f}")
    print(f"  95% CI for λ:         [{decomp['lambda_ci'][0]:.4f}, {decomp['lambda_ci'][1]:.4f}]")
    print()
    print(f"Interpretation:")
    print(f"  {decomp['interpretation']}")

    # ========================================================================
    # STEP 5: Test returns to scale
    # ========================================================================

    print("\n\nSTEP 5: Test Returns to Scale")
    print("-" * 80)

    rts_test = best_result.returns_to_scale_test(input_vars=["ln_capital", "ln_labor"], alpha=0.05)

    print(f"Returns to scale test:")
    print(f"  RTS estimate:         {rts_test['rts']:.4f}")
    print(f"  Standard error:       {rts_test['rts_se']:.4f}")
    print(f"  95% CI:               [{rts_test['ci'][0]:.4f}, {rts_test['ci'][1]:.4f}]")
    print()
    print(f"  Test H0: RTS = 1")
    print(f"  Wald statistic:       {rts_test['test_statistic']:.4f}")
    print(f"  P-value:              {rts_test['pvalue']:.4f}")
    print(f"  Conclusion:           {rts_test['conclusion']}")
    print()
    print(f"  {rts_test['interpretation']}")

    # ========================================================================
    # STEP 6: Test functional form (Cobb-Douglas vs Translog)
    # ========================================================================

    print("\n\nSTEP 6: Test Functional Form (Cobb-Douglas vs Translog)")
    print("-" * 80)

    # Generate Translog terms
    print("Generating Translog terms...")
    data_translog = add_translog(data, variables=["ln_capital", "ln_labor"])

    print(f"New variables created:")
    print(f"  - ln_capital_sq")
    print(f"  - ln_labor_sq")
    print(f"  - ln_capital_ln_labor")

    # Fit Translog model
    print(f"\nFitting Translog model...")
    sf_translog = StochasticFrontier(
        data=data_translog,
        depvar="ln_output",
        exog=["ln_capital", "ln_labor", "ln_capital_sq", "ln_labor_sq", "ln_capital_ln_labor"],
        frontier="production",
        dist=best_dist,
    )
    result_translog = sf_translog.fit()

    print(f"  Cobb-Douglas LL:      {best_result.loglik:.4f}")
    print(f"  Translog LL:          {result_translog.loglik:.4f}")

    # LR test for functional form
    lr_stat = 2 * (result_translog.loglik - best_result.loglik)
    from scipy import stats

    df_diff = 3  # Three additional terms
    pvalue = 1 - stats.chi2.cdf(lr_stat, df=df_diff)

    print(f"\nLR test for functional form:")
    print(f"  H0: Cobb-Douglas is adequate")
    print(f"  H1: Translog provides better fit")
    print(f"  LR statistic:         {lr_stat:.4f}")
    print(f"  Degrees of freedom:   {df_diff}")
    print(f"  P-value:              {pvalue:.4f}")

    if pvalue < 0.05:
        print(f"  Conclusion: Reject H0. Translog is preferred.")
    else:
        print(f"  Conclusion: Do not reject H0. Cobb-Douglas is adequate.")

    # Compare AIC/BIC
    print(f"\nInformation criteria:")
    print(f"  Cobb-Douglas AIC:     {best_result.aic:.4f}")
    print(f"  Translog AIC:         {result_translog.aic:.4f}")
    print(f"  Cobb-Douglas BIC:     {best_result.bic:.4f}")
    print(f"  Translog BIC:         {result_translog.bic:.4f}")

    # ========================================================================
    # STEP 7: Final recommendations
    # ========================================================================

    print("\n\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)

    print(f"\n1. Inefficiency is present: {ineff_test['conclusion'] == 'SFA needed'}")
    print(f"   SFA is preferred over OLS (p = {ineff_test['pvalue']:.4f})")

    print(f"\n2. Best distribution: {best_dist}")
    print(f"   Selected by AIC/BIC")

    print(f"\n3. Variance decomposition:")
    print(f"   γ = {decomp['gamma']:.2%} of variance is due to inefficiency")

    print(f"\n4. Returns to scale: {rts_test['conclusion']}")
    print(f"   RTS = {rts_test['rts']:.3f}")

    print(f"\n5. Functional form:")
    if pvalue < 0.05:
        print(f"   Translog is preferred (p = {pvalue:.4f})")
        final_model = "Translog"
    else:
        print(f"   Cobb-Douglas is adequate (p = {pvalue:.4f})")
        final_model = "Cobb-Douglas"

    print(f"\n\nRECOMMENDED MODEL:")
    print(f"  - Functional form: {final_model}")
    print(f"  - Distribution: {best_dist}")
    print(f"  - Frontier type: Production")

    print("\n" + "=" * 80)
    print()


if __name__ == "__main__":
    main()
