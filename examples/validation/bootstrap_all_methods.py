"""
Comparing All Bootstrap Methods for Panel Data

This script demonstrates all four bootstrap methods available in PanelBox:
1. Pairs (Entity) Bootstrap - Most robust, general purpose
2. Wild Bootstrap - For heteroskedastic errors
3. Block Bootstrap - For time-series dependence
4. Residual Bootstrap - Assumes i.i.d. errors

Author: PanelBox Development Team
Date: 2026-01-22
"""

import numpy as np
import pandas as pd

import panelbox as pb

# Set random seed
np.random.seed(42)

print("=" * 80)
print("Comparing All Bootstrap Methods for Panel Data")
print("=" * 80)

# ============================================================================
# 1. Generate Panel Data
# ============================================================================

print("\n1. Generating panel data...")

n_entities = 25
n_periods = 10
n_obs = n_entities * n_periods

# True parameters
beta_1 = 2.0
beta_2 = -1.5

# Generate data
entities = np.repeat(range(1, n_entities + 1), n_periods)
times = np.tile(range(1, n_periods + 1), n_entities)

x1 = np.random.randn(n_obs) * 2 + 5
x2 = np.random.randn(n_obs) * 3 + 10

# Entity fixed effects
entity_effects = np.repeat(np.random.randn(n_entities) * 3, n_periods)

# Heteroskedastic errors (variance increases with x1)
errors = np.random.randn(n_obs) * (1 + 0.3 * np.abs(x1))

y = beta_1 * x1 + beta_2 * x2 + entity_effects + errors

data = pd.DataFrame({"entity": entities, "time": times, "y": y, "x1": x1, "x2": x2})

print(f"   Panel: {n_entities} entities × {n_periods} periods = {n_obs} observations")
print(f"   True parameters: β₁={beta_1}, β₂={beta_2}")

# ============================================================================
# 2. Fit Fixed Effects Model
# ============================================================================

print("\n2. Fitting Fixed Effects model...")

model = pb.FixedEffects("y ~ x1 + x2", data, "entity", "time")
results = model.fit(cov_type="robust")

print("\nEstimated coefficients:")
print(results.params.to_string())

# ============================================================================
# 3. Run All Bootstrap Methods
# ============================================================================

print("\n3. Running all bootstrap methods...")
print("   (This may take 1-2 minutes for 1000 replications each)")

methods = ["pairs", "wild", "block", "residual"]
bootstrap_results = {}

for method in methods:
    print(f"\n   Running {method.upper()} bootstrap...")

    bootstrap = pb.PanelBootstrap(
        results,
        n_bootstrap=1000,
        method=method,
        block_size=3 if method == "block" else None,
        random_state=42,
        show_progress=False,  # Set to True to see progress bars
    )

    bootstrap.run()

    bootstrap_results[method] = {
        "bootstrap": bootstrap,
        "se": bootstrap.bootstrap_se_,
        "ci": bootstrap.conf_int(alpha=0.05, method="percentile"),
        "summary": bootstrap.summary(),
    }

    print(f"   ✓ {method.upper()} complete: {1000 - bootstrap.n_failed_}/1000 successful")

# ============================================================================
# 4. Compare Standard Errors
# ============================================================================

print("\n4. Comparing Standard Errors Across Methods:")
print("=" * 80)

se_comparison = pd.DataFrame(
    {
        "Asymptotic": results.std_errors,
        "Pairs": bootstrap_results["pairs"]["se"],
        "Wild": bootstrap_results["wild"]["se"],
        "Block": bootstrap_results["block"]["se"],
        "Residual": bootstrap_results["residual"]["se"],
    }
)

print(se_comparison.to_string())

print("\nInterpretation:")
print("  - Pairs bootstrap is most robust (recommended)")
print("  - Wild bootstrap adjusts for heteroskedasticity")
print("  - Block bootstrap accounts for temporal dependence")
print("  - Residual bootstrap assumes i.i.d. (least robust)")

# ============================================================================
# 5. Compare Confidence Intervals
# ============================================================================

print("\n5. Comparing 95% Confidence Intervals:")
print("=" * 80)

# Asymptotic CI
ci_asymp = results.conf_int(alpha=0.05)

print("\nAsymptotic (Robust SE):")
print(ci_asymp.to_string())

for method in methods:
    ci = bootstrap_results[method]["ci"]
    print(f"\n{method.upper()} Bootstrap:")
    print(ci.to_string())

# ============================================================================
# 6. CI Width Comparison
# ============================================================================

print("\n6. Confidence Interval Widths:")
print("=" * 80)

widths = pd.DataFrame({"Asymptotic": ci_asymp["upper"] - ci_asymp["lower"]})

for method in methods:
    ci = bootstrap_results[method]["ci"]
    widths[method.capitalize()] = ci["upper"] - ci["lower"]

print(widths.to_string())

print("\nInterpretation:")
print("  - Wider CIs are more conservative")
print("  - Narrower CIs may underestimate uncertainty")
print("  - Compare with asymptotic to assess adequacy")

# ============================================================================
# 7. Bootstrap Bias Comparison
# ============================================================================

print("\n7. Bootstrap Bias Estimates:")
print("=" * 80)

bias_comparison = pd.DataFrame({"Estimate": results.params})

for method in methods:
    summary = bootstrap_results[method]["summary"]
    bias_comparison[f"{method.capitalize()} Bias"] = summary["Bootstrap Bias"]

print(bias_comparison.to_string())

print("\nInterpretation:")
print("  - Small bias indicates unbiased estimator")
print("  - Consistent bias across methods is reassuring")
print("  - Large bias may indicate model misspecification")

# ============================================================================
# 8. SE Ratios (Bootstrap / Asymptotic)
# ============================================================================

print("\n8. Standard Error Ratios (Bootstrap SE / Asymptotic SE):")
print("=" * 80)

se_ratios = pd.DataFrame(index=results.params.index)

for method in methods:
    se_ratios[method.capitalize()] = bootstrap_results[method]["se"] / results.std_errors.values

print(se_ratios.to_string())

print("\nInterpretation:")
print("  - Ratio ≈ 1.0: Bootstrap confirms asymptotic SE")
print("  - Ratio > 1.0: Asymptotic SE may be too small (anti-conservative)")
print("  - Ratio < 1.0: Asymptotic SE may be too large (conservative)")
print("  - Large deviations suggest using bootstrap for inference")

# ============================================================================
# 9. Method Recommendations
# ============================================================================

print("\n9. Method-Specific Recommendations:")
print("=" * 80)

print("\n📌 PAIRS BOOTSTRAP (Recommended Default)")
print("   When to use:")
print("   ✓ General purpose, works in most scenarios")
print("   ✓ Robust to heteroskedasticity")
print("   ✓ Robust to serial correlation within entities")
print("   ✓ Minimal assumptions")
print("   When NOT to use:")
print("   ✗ When entities are not independent (spatial dependence)")

print("\n📌 WILD BOOTSTRAP")
print("   When to use:")
print("   ✓ Heteroskedastic errors (primary concern)")
print("   ✓ Fixed regressors (X not random)")
print("   ✓ Short panels or cross-sectional data")
print("   When NOT to use:")
print("   ✗ Serial correlation is important")
print("   ✗ Clustered errors across time")

print("\n📌 BLOCK BOOTSTRAP")
print("   When to use:")
print("   ✓ Strong temporal dependence")
print("   ✓ Macro panels (time-series behavior)")
print("   ✓ Long panels (T ≥ 10)")
print("   When NOT to use:")
print("   ✗ Short panels (T < 5)")
print("   ✗ Weak temporal dependence")

print("\n📌 RESIDUAL BOOTSTRAP")
print("   When to use:")
print("   ✓ Benchmark comparison")
print("   ✓ Confident errors are i.i.d.")
print("   ✓ Speed is critical")
print("   When NOT to use:")
print("   ✗ Heteroskedastic errors")
print("   ✗ Serial correlation")
print("   ✗ Primary inference (use as supplement)")

# ============================================================================
# 10. Overall Assessment
# ============================================================================

print("\n10. Overall Assessment for This Data:")
print("=" * 80)

# Calculate average SE ratio
avg_ratios = se_ratios.mean()
print("\nAverage SE Ratios:")
for method, ratio in avg_ratios.items():
    print(f"  {method:12s}: {ratio:.3f}")

# Determine recommendation
max_ratio = avg_ratios.max()
min_ratio = avg_ratios.min()

if max_ratio > 1.15:
    print("\n⚠️  Bootstrap SEs notably larger than asymptotic")
    print("   → Recommend using bootstrap for inference")
    print("   → Asymptotic theory may be inadequate")
elif min_ratio < 0.85:
    print("\n⚠️  Bootstrap SEs notably smaller than asymptotic")
    print("   → Investigate potential issues")
    print("   → Check model specification")
else:
    print("\n✓  Bootstrap and asymptotic SEs are consistent")
    print("   → Asymptotic theory appears adequate")
    print("   → Either approach is defensible")

# Check consistency across bootstrap methods
bootstrap_se_std = se_comparison[["Pairs", "Wild", "Block", "Residual"]].std(axis=1).mean()
if bootstrap_se_std > 0.1:
    print("\n⚠️  Bootstrap methods show substantial variation")
    print("   → Data may violate some method's assumptions")
    print("   → Prefer pairs bootstrap (most robust)")
else:
    print("\n✓  Bootstrap methods show consistent results")
    print("   → Robustness to method choice")

# ============================================================================
# 11. Summary and Recommendations
# ============================================================================

print("\n" + "=" * 80)
print("Summary and Recommendations")
print("=" * 80)

print("\n✅ All four bootstrap methods completed successfully")
print(f"\n📊 Results Summary:")
print(
    f"   - Asymptotic SE (robust): {results.std_errors['x1']:.4f}, {results.std_errors['x2']:.4f}"
)
print(
    f"   - Pairs Bootstrap SE:     {bootstrap_results['pairs']['se'][0]:.4f}, {bootstrap_results['pairs']['se'][1]:.4f}"
)
print(
    f"   - Wild Bootstrap SE:      {bootstrap_results['wild']['se'][0]:.4f}, {bootstrap_results['wild']['se'][1]:.4f}"
)
print(
    f"   - Block Bootstrap SE:     {bootstrap_results['block']['se'][0]:.4f}, {bootstrap_results['block']['se'][1]:.4f}"
)
print(
    f"   - Residual Bootstrap SE:  {bootstrap_results['residual']['se'][0]:.4f}, {bootstrap_results['residual']['se'][1]:.4f}"
)

print("\n💡 General Guidance:")
print("   1. Start with pairs bootstrap (most robust)")
print("   2. Use wild bootstrap if heteroskedasticity is primary concern")
print("   3. Use block bootstrap for strong temporal dependence")
print("   4. Use residual bootstrap as benchmark only")
print("   5. If methods agree, any choice is defensible")
print("   6. If methods disagree, prefer pairs bootstrap")

print("\n📚 Further Reading:")
print("   - Efron & Tibshirani (1994): An Introduction to the Bootstrap")
print("   - Cameron & Trivedi (2005): Microeconometrics, Chapter 11")
print("   - Cameron et al. (2008): Bootstrap-based improvements for inference")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

# ============================================================================
# Optional: Save results
# ============================================================================

save_results = False  # Set to True to save

if save_results:
    # Save comparison tables
    se_comparison.to_csv("bootstrap_se_comparison.csv")
    widths.to_csv("bootstrap_ci_widths.csv")
    se_ratios.to_csv("bootstrap_se_ratios.csv")

    print("\n💾 Results saved to CSV files")
