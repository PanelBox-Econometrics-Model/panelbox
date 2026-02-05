"""
Bootstrap Inference for Panel Data Models - Example

This script demonstrates how to use PanelBootstrap for inference in
panel data models, comparing bootstrap and asymptotic confidence intervals.

Author: PanelBox Development Team
Date: 2026-01-22
"""

import numpy as np
import pandas as pd

import panelbox as pb

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("Bootstrap Inference for Panel Data Models")
print("=" * 70)

# ============================================================================
# 1. Generate synthetic panel data
# ============================================================================

print("\n1. Generating synthetic panel data...")

n_entities = 20
n_periods = 10
n_obs = n_entities * n_periods

# True parameters
beta_0 = 10.0  # Intercept
beta_1 = 2.5  # Effect of x1
beta_2 = -1.8  # Effect of x2

# Generate data
entities = np.repeat(range(1, n_entities + 1), n_periods)
times = np.tile(range(1, n_periods + 1), n_entities)

# Predictors
x1 = np.random.randn(n_obs) * 2 + 5
x2 = np.random.randn(n_obs) * 3 + 10

# Entity fixed effects
entity_effects = np.repeat(np.random.randn(n_entities) * 5, n_periods)

# Random errors (heteroskedastic)
errors = np.random.randn(n_obs) * (1 + 0.5 * np.abs(x1))

# Outcome variable
y = beta_0 + beta_1 * x1 + beta_2 * x2 + entity_effects + errors

# Create DataFrame
data = pd.DataFrame({"entity": entities, "time": times, "y": y, "x1": x1, "x2": x2})

print(f"   Generated panel: {n_entities} entities, {n_periods} periods")
print(f"   Total observations: {n_obs}")
print(f"   True parameters: β₀={beta_0}, β₁={beta_1}, β₂={beta_2}")

# ============================================================================
# 2. Fit Fixed Effects model
# ============================================================================

print("\n2. Fitting Fixed Effects model...")

model = pb.FixedEffects(
    formula="y ~ x1 + x2",
    data=data,
    entity_col="entity",
    time_col="time",
    entity_effects=True,
    time_effects=False,
)

results = model.fit(cov_type="robust")

print("\nModel Results:")
print(results.summary())

# ============================================================================
# 3. Bootstrap inference (Pairs Bootstrap)
# ============================================================================

print("\n3. Running Bootstrap inference...")
print("   Method: Pairs (Entity) Bootstrap")
print("   Replications: 1000")
print("   (This may take 30-60 seconds...)")

bootstrap = pb.PanelBootstrap(
    results=results,
    n_bootstrap=1000,
    method="pairs",
    random_state=42,
    show_progress=True,  # Show progress bar
)

# Run bootstrap
bootstrap.run()

print(f"\n   Bootstrap completed successfully!")
print(f"   Successful replications: {1000 - bootstrap.n_failed_}")
print(f"   Failed replications: {bootstrap.n_failed_}")

# ============================================================================
# 4. Compare Standard Errors
# ============================================================================

print("\n4. Comparing Standard Errors:")
print("=" * 70)

summary = bootstrap.summary()
print(summary.to_string())

print("\nInterpretation:")
print("  - 'Bootstrap Bias' shows the bias in coefficient estimates")
print("  - 'SE Ratio' > 1 means bootstrap SE is larger than asymptotic SE")
print("  - 'SE Ratio' < 1 means bootstrap SE is smaller than asymptotic SE")

# ============================================================================
# 5. Compare Confidence Intervals
# ============================================================================

print("\n5. Comparing Confidence Intervals (95%):")
print("=" * 70)

# Asymptotic CI
ci_asymp = results.conf_int(alpha=0.05)
ci_asymp.columns = ["Asymptotic Lower", "Asymptotic Upper"]

# Bootstrap percentile CI
ci_boot_perc = bootstrap.conf_int(alpha=0.05, method="percentile")
ci_boot_perc.columns = ["Bootstrap Lower", "Bootstrap Upper"]

# Bootstrap basic CI
ci_boot_basic = bootstrap.conf_int(alpha=0.05, method="basic")
ci_boot_basic.columns = ["Basic Lower", "Basic Upper"]

# Combine
ci_comparison = pd.concat(
    [results.params.rename("Estimate"), ci_asymp, ci_boot_perc, ci_boot_basic], axis=1
)

print(ci_comparison.to_string())

print("\nInterpretation:")
print("  - Asymptotic CIs use asymptotic theory (normality assumption)")
print("  - Bootstrap Percentile CIs use bootstrap distribution quantiles")
print("  - Bootstrap Basic CIs use reflection method")
print("  - Differences indicate departure from asymptotic normality")

# ============================================================================
# 6. Compute CI Widths
# ============================================================================

print("\n6. Confidence Interval Widths:")
print("=" * 70)

widths = pd.DataFrame(
    {
        "Asymptotic Width": ci_asymp["Asymptotic Upper"] - ci_asymp["Asymptotic Lower"],
        "Bootstrap Width": ci_boot_perc["Bootstrap Upper"] - ci_boot_perc["Bootstrap Lower"],
        "Ratio (Boot/Asymp)": (
            (ci_boot_perc["Bootstrap Upper"] - ci_boot_perc["Bootstrap Lower"])
            / (ci_asymp["Asymptotic Upper"] - ci_asymp["Asymptotic Lower"])
        ),
    }
)

print(widths.to_string())

print("\nInterpretation:")
print("  - Ratio > 1: Bootstrap CI is wider (more conservative)")
print("  - Ratio < 1: Bootstrap CI is narrower (less conservative)")
print("  - Large differences may indicate small sample issues")

# ============================================================================
# 7. Hypothesis Testing with Bootstrap
# ============================================================================

print("\n7. Hypothesis Testing using Bootstrap:")
print("=" * 70)

# Test H0: β₁ = 2.5 (true value)
# Calculate bootstrap p-value

param_idx = results.params.index.get_loc("x1")
true_beta1 = 2.5
estimated_beta1 = results.params["x1"]

# Bootstrap distribution
boot_beta1 = bootstrap.bootstrap_estimates_[:, param_idx]

# Two-sided p-value: proportion of bootstrap estimates farther from H0 than observed
boot_t = np.abs((boot_beta1 - true_beta1) / bootstrap.bootstrap_se_[param_idx])
obs_t = np.abs((estimated_beta1 - true_beta1) / results.std_errors["x1"])
p_value_boot = np.mean(boot_t >= obs_t)

# Asymptotic p-value
from scipy import stats

p_value_asymp = 2 * (1 - stats.t.cdf(np.abs(obs_t), results.df_resid))

print(f"Testing H₀: β₁ = {true_beta1} (true value)")
print(f"Estimate: {estimated_beta1:.4f}")
print(f"\nAsymptotic p-value: {p_value_asymp:.4f}")
print(f"Bootstrap p-value:  {p_value_boot:.4f}")

if p_value_boot > 0.05:
    print(f"✓ Cannot reject H₀ at 5% level (bootstrap)")
else:
    print(f"✗ Reject H₀ at 5% level (bootstrap)")

# ============================================================================
# 8. Visualize Bootstrap Distribution (Optional)
# ============================================================================

print("\n8. Visualizing Bootstrap Distribution:")
print("   (Uncomment the plot_distribution() line to see the plots)")

# Uncomment to show plots:
# bootstrap.plot_distribution(param='x1')
# bootstrap.plot_distribution()  # All parameters

print("   To visualize, run: bootstrap.plot_distribution('x1')")

# ============================================================================
# 9. Summary and Recommendations
# ============================================================================

print("\n" + "=" * 70)
print("Summary and Recommendations")
print("=" * 70)

print("\n✓ Bootstrap Inference Completed Successfully")
print(f"\n  - Used {1000 - bootstrap.n_failed_} bootstrap replications")
print("  - Method: Pairs (Entity) Bootstrap")
print("  - Preserved within-entity correlation structure")

print("\nWhen to use bootstrap:")
print("  ✓ Small samples (N < 30 entities)")
print("  ✓ Uncertainty about distributional assumptions")
print("  ✓ Heteroskedasticity or serial correlation concerns")
print("  ✓ Non-standard estimators or test statistics")

print("\nBootstrap vs Asymptotic:")
if np.max(widths["Ratio (Boot/Asymp)"]) > 1.2:
    print("  ⚠  Bootstrap CIs notably wider → More conservative")
    print("     Consider using bootstrap for inference")
elif np.min(widths["Ratio (Boot/Asymp)"]) < 0.8:
    print("  ⚠  Bootstrap CIs notably narrower → Less conservative")
    print("     Investigate potential issues")
else:
    print("  ✓  Bootstrap and asymptotic CIs similar")
    print("     Asymptotic theory appears adequate")

print("\n" + "=" * 70)
print("Example completed!")
print("=" * 70)

# ============================================================================
# Optional: Save results to CSV
# ============================================================================

save_results = False  # Set to True to save

if save_results:
    # Save summary
    summary.to_csv("bootstrap_summary.csv")

    # Save CI comparison
    ci_comparison.to_csv("bootstrap_ci_comparison.csv")

    print("\nResults saved to:")
    print("  - bootstrap_summary.csv")
    print("  - bootstrap_ci_comparison.csv")
