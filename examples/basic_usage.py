"""
PanelBox - Basic Usage Example

This example demonstrates the basic workflow of the PanelBox library:
1. Loading and exploring panel data
2. Estimating Pooled OLS, Fixed Effects, and Random Effects models
3. Running Hausman test to choose between FE and RE
4. Interpreting results
"""

import numpy as np
import pandas as pd

import panelbox as pb

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("PANELBOX - BASIC USAGE EXAMPLE")
print("=" * 80)
print()

# ============================================================================
# 1. CREATE SAMPLE PANEL DATA
# ============================================================================
print("1. Creating sample panel data...")
print("-" * 80)

# Create balanced panel: 50 firms over 10 years
n_firms = 50
n_years = 10
n_obs = n_firms * n_years

# Generate data
data = pd.DataFrame(
    {
        "firm": np.repeat(range(1, n_firms + 1), n_years),
        "year": np.tile(range(2010, 2010 + n_years), n_firms),
    }
)

# Add firm-specific fixed effect
firm_effect = {i: np.random.normal(0, 5) for i in range(1, n_firms + 1)}
data["firm_effect"] = data["firm"].map(firm_effect)

# Generate regressors
data["capital"] = np.random.uniform(100, 1000, n_obs)
data["labor"] = np.random.uniform(50, 500, n_obs)

# Generate dependent variable with firm fixed effects
data["output"] = (
    10  # Intercept
    + data["firm_effect"]  # Firm-specific effect
    + 0.5 * data["capital"]  # Capital coefficient
    + 0.3 * data["labor"]  # Labor coefficient
    + np.random.normal(0, 10, n_obs)  # Random error
)

# Drop the true firm_effect (we pretend we don't observe it)
data = data.drop("firm_effect", axis=1)

print(f"Panel structure: {n_firms} firms, {n_years} years, {n_obs} observations")
print(f"\nFirst few rows:")
print(data.head(10))
print()

# ============================================================================
# 2. EXPLORE PANEL DATA
# ============================================================================
print("\n2. Exploring panel data structure...")
print("-" * 80)

panel = pb.PanelData(data, entity_col="firm", time_col="year")
print(panel.summary())
print()

# ============================================================================
# 3. POOLED OLS (BASELINE)
# ============================================================================
print("\n3. Estimating Pooled OLS (ignoring panel structure)...")
print("-" * 80)

pooled = pb.PooledOLS("output ~ capital + labor", data, "firm", "year")
pooled_results = pooled.fit(cov_type="robust")

print(pooled_results.summary())
print()

# ============================================================================
# 4. FIXED EFFECTS (WITHIN ESTIMATOR)
# ============================================================================
print("\n4. Estimating Fixed Effects model...")
print("-" * 80)

fe = pb.FixedEffects("output ~ capital + labor", data, "firm", "year")
fe_results = fe.fit(cov_type="clustered")

print(fe_results.summary())
print()

# Show first few estimated firm fixed effects
print("First 10 estimated firm fixed effects:")
print(fe.entity_fe.head(10))
print()

# ============================================================================
# 5. RANDOM EFFECTS (GLS)
# ============================================================================
print("\n5. Estimating Random Effects model...")
print("-" * 80)

re = pb.RandomEffects("output ~ capital + labor", data, "firm", "year")
re_results = re.fit()

print(re_results.summary())
print()

# Show variance components
print("Variance Components:")
print(f"  sigma2_u (firm effect variance): {re.sigma2_u:.4f}")
print(f"  sigma2_e (error variance):       {re.sigma2_e:.4f}")
print(f"  theta (GLS parameter):           {re.theta:.4f}")
print()

# ============================================================================
# 6. HAUSMAN TEST (FE vs RE)
# ============================================================================
print("\n6. Running Hausman specification test...")
print("-" * 80)

hausman = pb.HausmanTest(fe_results, re_results)
hausman_result = hausman.run(alpha=0.05)

print(hausman_result.summary())
print()

# ============================================================================
# 7. MODEL COMPARISON
# ============================================================================
print("\n7. Model Comparison Summary...")
print("-" * 80)

comparison_data = {
    "Model": ["Pooled OLS", "Fixed Effects", "Random Effects"],
    "capital_coef": [
        pooled_results.params["capital"],
        fe_results.params["capital"],
        re_results.params["capital"],
    ],
    "labor_coef": [
        pooled_results.params["labor"],
        fe_results.params["labor"],
        re_results.params["labor"],
    ],
    "R2_overall": [
        pooled_results.rsquared_overall,
        fe_results.rsquared_overall,
        re_results.rsquared_overall,
    ],
    "R2_within": [np.nan, fe_results.rsquared_within, re_results.rsquared_within],
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))
print()

# ============================================================================
# 8. FINAL RECOMMENDATION
# ============================================================================
print("\n8. Final Recommendation...")
print("-" * 80)

print(f"Based on Hausman test: Use {hausman_result.recommendation}")
print()

if hausman_result.recommendation == "Fixed Effects":
    final_model = fe_results
    print("Final Model: Fixed Effects")
else:
    final_model = re_results
    print("Final Model: Random Effects")

print("\nFinal Coefficients:")
for var, coef in final_model.params.items():
    se = final_model.std_errors[var]
    t = final_model.tvalues[var]
    p = final_model.pvalues[var]
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {var:<12} {coef:>8.4f} ({se:>6.4f})  t={t:>6.2f}  p={p:.4f} {stars}")

print()
print("=" * 80)
print("EXAMPLE COMPLETE")
print("=" * 80)
print()
print("Key Takeaways:")
print("1. Pooled OLS ignores panel structure (biased if firm effects exist)")
print("2. Fixed Effects removes firm-specific effects (consistent)")
print("3. Random Effects is efficient if firm effects uncorrelated with X")
print("4. Hausman test helps choose between FE and RE")
print()
print("For more examples, see: examples/notebooks/")
print("=" * 80)
