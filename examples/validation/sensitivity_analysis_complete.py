"""
Comprehensive Sensitivity Analysis for Panel Data Models

This script demonstrates all sensitivity analysis methods available in PanelBox:
1. Leave-One-Out Entities - Assess influence of individual entities
2. Leave-One-Out Periods - Assess influence of individual time periods
3. Subset Sensitivity - Random subsample stability analysis
4. Visualization and interpretation

Author: PanelBox Development Team
Date: 2026-01-22
"""

import numpy as np
import pandas as pd

import panelbox as pb

# Set random seed
np.random.seed(42)

print("=" * 80)
print("Comprehensive Sensitivity Analysis for Panel Data")
print("=" * 80)

# ============================================================================
# 1. Generate Panel Data with Potential Outliers
# ============================================================================

print("\n1. Generating panel data with potential outliers...")

n_entities = 30
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

# Add influential outlier to entity 15
outlier_entity = 15
outlier_mask = entities == outlier_entity
entity_effects[outlier_mask] += 10  # Large outlier effect

# Errors
errors = np.random.randn(n_obs)

y = beta_1 * x1 + beta_2 * x2 + entity_effects + errors

data = pd.DataFrame({"entity": entities, "time": times, "y": y, "x1": x1, "x2": x2})

print(f"   Panel: {n_entities} entities × {n_periods} periods = {n_obs} observations")
print(f"   True parameters: β₁={beta_1}, β₂={beta_2}")
print(f"   Added outlier to entity {outlier_entity}")

# ============================================================================
# 2. Fit Fixed Effects Model
# ============================================================================

print("\n2. Fitting Fixed Effects model...")

model = pb.FixedEffects("y ~ x1 + x2", data, "entity", "time")
results = model.fit(cov_type="robust")

print("\nOriginal estimates:")
print(f"   x1: {results.params['x1']:.4f} (SE: {results.std_errors['x1']:.4f})")
print(f"   x2: {results.params['x2']:.4f} (SE: {results.std_errors['x2']:.4f})")

# ============================================================================
# 3. Initialize Sensitivity Analysis
# ============================================================================

print("\n3. Initializing sensitivity analysis...")

sensitivity = pb.SensitivityAnalysis(results, show_progress=False)

print(f"   Loaded {sensitivity.n_entities} entities")
print(f"   Loaded {sensitivity.n_periods} time periods")

# ============================================================================
# 4. Leave-One-Out Entities Analysis
# ============================================================================

print("\n4. Running Leave-One-Out Entities analysis...")
print("   (Removing one entity at a time and re-estimating)")

loo_entities = sensitivity.leave_one_out_entities(influence_threshold=2.0)

print(f"\n   Completed: {len(loo_entities.estimates)} estimations")
print(f"   Identified {len(loo_entities.influential_units)} influential entities")

if loo_entities.influential_units:
    print(f"   Influential entities: {loo_entities.influential_units}")

# Summary statistics
summary_entities = sensitivity.summary(loo_entities)
print("\n   Summary of LOO Entities:")
print(summary_entities.to_string(index=False))

# ============================================================================
# 5. Leave-One-Out Periods Analysis
# ============================================================================

print("\n5. Running Leave-One-Out Periods analysis...")
print("   (Removing one time period at a time and re-estimating)")

loo_periods = sensitivity.leave_one_out_periods(influence_threshold=2.0)

print(f"\n   Completed: {len(loo_periods.estimates)} estimations")
print(f"   Identified {len(loo_periods.influential_units)} influential periods")

if loo_periods.influential_units:
    print(f"   Influential periods: {loo_periods.influential_units}")

# Summary statistics
summary_periods = sensitivity.summary(loo_periods)
print("\n   Summary of LOO Periods:")
print(summary_periods.to_string(index=False))

# ============================================================================
# 6. Subset Sensitivity Analysis
# ============================================================================

print("\n6. Running Subset Sensitivity analysis...")
print("   (Random subsamples with 80% of entities)")

subset_results = sensitivity.subset_sensitivity(
    n_subsamples=30, subsample_size=0.8, random_state=42
)

print(f"\n   Completed: {len(subset_results.estimates)} subsamples")
successful = subset_results.subsample_info["converged"].sum()
print(f"   Successful estimations: {successful}/{len(subset_results.estimates)}")

# Summary statistics
summary_subset = sensitivity.summary(subset_results)
print("\n   Summary of Subset Sensitivity:")
print(summary_subset.to_string(index=False))

# ============================================================================
# 7. Compare Sensitivity Across Methods
# ============================================================================

print("\n7. Comparing Sensitivity Across Methods:")
print("=" * 80)

comparison = pd.DataFrame(
    {
        "Method": ["LOO Entities", "LOO Periods", "Subset (80%)"],
        "x1_Mean": [
            loo_entities.estimates["x1"].mean(),
            loo_periods.estimates["x1"].mean(),
            subset_results.estimates["x1"].mean(),
        ],
        "x1_Std": [
            loo_entities.estimates["x1"].std(),
            loo_periods.estimates["x1"].std(),
            subset_results.estimates["x1"].std(),
        ],
        "x2_Mean": [
            loo_entities.estimates["x2"].mean(),
            loo_periods.estimates["x2"].mean(),
            subset_results.estimates["x2"].mean(),
        ],
        "x2_Std": [
            loo_entities.estimates["x2"].std(),
            loo_periods.estimates["x2"].std(),
            subset_results.estimates["x2"].std(),
        ],
    }
)

print("\n" + comparison.to_string(index=False))

# ============================================================================
# 8. Detailed Analysis of Influential Entities
# ============================================================================

print("\n8. Detailed Analysis of Influential Entities:")
print("=" * 80)

# Check if entity 15 (our planted outlier) was detected
if loo_entities.influential_units:
    print(f"\n✓ Detected {len(loo_entities.influential_units)} influential entities")

    # Extract entity IDs from 'excl_X' format
    influential_ids = [int(unit.replace("excl_", "")) for unit in loo_entities.influential_units]

    if outlier_entity in influential_ids:
        print(f"✓ Successfully identified planted outlier (entity {outlier_entity})")
    else:
        print(f"⚠ Did not identify planted outlier (entity {outlier_entity})")

    print("\nParameter changes when excluding influential entities:")
    for unit in loo_entities.influential_units:
        entity_id = int(unit.replace("excl_", ""))
        x1_est = loo_entities.estimates.loc[unit, "x1"]
        x2_est = loo_entities.estimates.loc[unit, "x2"]

        x1_change = ((x1_est - results.params["x1"]) / results.params["x1"]) * 100
        x2_change = ((x2_est - results.params["x2"]) / results.params["x2"]) * 100

        print(f"   Entity {entity_id}:")
        print(f"      x1: {x1_est:.4f} ({x1_change:+.2f}% change)")
        print(f"      x2: {x2_est:.4f} ({x2_change:+.2f}% change)")
else:
    print("\n✓ No influential entities detected (estimates are stable)")

# ============================================================================
# 9. Stability Assessment
# ============================================================================

print("\n9. Stability Assessment:")
print("=" * 80)

# Calculate coefficient of variation (CV) for each parameter
cv_entities_x1 = (
    loo_entities.estimates["x1"].std() / abs(loo_entities.estimates["x1"].mean())
) * 100
cv_entities_x2 = (
    loo_entities.estimates["x2"].std() / abs(loo_entities.estimates["x2"].mean())
) * 100

cv_periods_x1 = (loo_periods.estimates["x1"].std() / abs(loo_periods.estimates["x1"].mean())) * 100
cv_periods_x2 = (loo_periods.estimates["x2"].std() / abs(loo_periods.estimates["x2"].mean())) * 100

cv_subset_x1 = (
    subset_results.estimates["x1"].std() / abs(subset_results.estimates["x1"].mean())
) * 100
cv_subset_x2 = (
    subset_results.estimates["x2"].std() / abs(subset_results.estimates["x2"].mean())
) * 100

print("\nCoefficient of Variation (lower is more stable):")
print(f"   LOO Entities - x1: {cv_entities_x1:.2f}%, x2: {cv_entities_x2:.2f}%")
print(f"   LOO Periods  - x1: {cv_periods_x1:.2f}%, x2: {cv_periods_x2:.2f}%")
print(f"   Subset (80%) - x1: {cv_subset_x1:.2f}%, x2: {cv_subset_x2:.2f}%")

# Interpretation
print("\nStability Interpretation:")
if cv_entities_x1 < 5 and cv_entities_x2 < 5:
    print("   ✓ Highly stable: CV < 5% for all parameters")
elif cv_entities_x1 < 10 and cv_entities_x2 < 10:
    print("   ✓ Moderately stable: CV < 10% for all parameters")
else:
    print("   ⚠ Potentially unstable: CV > 10% for some parameters")
    print("   → Consider investigating influential observations")

# ============================================================================
# 10. Estimate Ranges
# ============================================================================

print("\n10. Parameter Estimate Ranges:")
print("=" * 80)

print("\nOriginal estimates:")
print(f"   x1: {results.params['x1']:.4f}")
print(f"   x2: {results.params['x2']:.4f}")

print("\nLOO Entities ranges:")
print(
    f"   x1: [{loo_entities.estimates['x1'].min():.4f}, "
    f"{loo_entities.estimates['x1'].max():.4f}] "
    f"(range: {loo_entities.estimates['x1'].max() - loo_entities.estimates['x1'].min():.4f})"
)
print(
    f"   x2: [{loo_entities.estimates['x2'].min():.4f}, "
    f"{loo_entities.estimates['x2'].max():.4f}] "
    f"(range: {loo_entities.estimates['x2'].max() - loo_entities.estimates['x2'].min():.4f})"
)

print("\nLOO Periods ranges:")
print(
    f"   x1: [{loo_periods.estimates['x1'].min():.4f}, "
    f"{loo_periods.estimates['x1'].max():.4f}] "
    f"(range: {loo_periods.estimates['x1'].max() - loo_periods.estimates['x1'].min():.4f})"
)
print(
    f"   x2: [{loo_periods.estimates['x2'].min():.4f}, "
    f"{loo_periods.estimates['x2'].max():.4f}] "
    f"(range: {loo_periods.estimates['x2'].max() - loo_periods.estimates['x2'].min():.4f})"
)

print("\nSubset ranges:")
print(
    f"   x1: [{subset_results.estimates['x1'].min():.4f}, "
    f"{subset_results.estimates['x1'].max():.4f}] "
    f"(range: {subset_results.estimates['x1'].max() - subset_results.estimates['x1'].min():.4f})"
)
print(
    f"   x2: [{subset_results.estimates['x2'].min():.4f}, "
    f"{subset_results.estimates['x2'].max():.4f}] "
    f"(range: {subset_results.estimates['x2'].max() - subset_results.estimates['x2'].min():.4f})"
)

# ============================================================================
# 11. Recommendations for Practice
# ============================================================================

print("\n11. Practical Recommendations:")
print("=" * 80)

print("\n📌 WHEN TO USE EACH METHOD:")

print("\n1. Leave-One-Out Entities")
print("   Use when:")
print("   ✓ You have concerns about specific entities being outliers")
print("   ✓ You want to identify influential cross-sectional units")
print("   ✓ Panel is cross-sectionally focused (N > T)")
print("   Warning:")
print("   ⚠ Computationally intensive for large N")
print("   ⚠ May not converge if N is small")

print("\n2. Leave-One-Out Periods")
print("   Use when:")
print("   ✓ You have concerns about specific time periods (e.g., crisis years)")
print("   ✓ You want to identify influential temporal observations")
print("   ✓ Panel is time-series focused (T > N)")
print("   Warning:")
print("   ⚠ May not converge if T is small")
print("   ⚠ Loses information about within-entity variation")

print("\n3. Subset Sensitivity")
print("   Use when:")
print("   ✓ You want general stability assessment")
print("   ✓ You want to understand sampling variability")
print("   ✓ Computational resources are limited")
print("   ✓ You want reproducible results (set random_state)")
print("   Best practices:")
print("   • Use 20-50 subsamples for reliable assessment")
print("   • Try different subsample sizes (70%, 80%, 90%)")
print("   • Compare results across different random seeds")

# ============================================================================
# 12. Interpretation Guidelines
# ============================================================================

print("\n12. Interpretation Guidelines:")
print("=" * 80)

print("\n📊 ASSESSING SENSITIVITY RESULTS:")

print("\n1. Coefficient of Variation (CV)")
print("   CV < 5%:  Highly stable estimates")
print("   CV 5-10%: Moderately stable (acceptable in most cases)")
print("   CV > 10%: Potentially problematic instability")

print("\n2. Influential Units")
print("   • Units with deviation > 2 SE are considered influential")
print("   • Investigate why these units are influential")
print("   • Consider robustness checks with/without influential units")
print("   • Document findings in research papers")

print("\n3. Range of Estimates")
print("   • Narrow ranges indicate stable relationships")
print("   • Wide ranges suggest sensitivity to sample composition")
print("   • Compare range to standard errors for context")

print("\n4. Convergence Issues")
print("   • Track convergence rate across subsamples")
print("   • Low convergence rate may indicate model specification issues")
print("   • Consider simpler model or different estimator")

# ============================================================================
# 13. Integration with Bootstrap
# ============================================================================

print("\n13. Combining with Bootstrap Analysis:")
print("=" * 80)

print("\n💡 COMPLEMENTARY APPROACHES:")

print("\nBootstrap:")
print("   → Assesses sampling uncertainty")
print("   → Provides confidence intervals")
print("   → Tests distributional assumptions")

print("\nSensitivity Analysis:")
print("   → Assesses influence of specific observations")
print("   → Identifies outliers and leverage points")
print("   → Tests robustness to sample composition")

print("\nRecommended Workflow:")
print("   1. Fit model and examine diagnostics")
print("   2. Run sensitivity analysis to identify influential units")
print("   3. Run bootstrap for inference")
print("   4. Compare bootstrap and sensitivity results")
print("   5. Report both in final analysis")

# ============================================================================
# 14. Visualization (Optional)
# ============================================================================

print("\n14. Visualization:")
print("=" * 80)

try:
    import matplotlib.pyplot as plt

    print("\n   Creating sensitivity plots...")

    # Plot LOO entities
    fig1 = sensitivity.plot_sensitivity(loo_entities, figsize=(14, 5))
    plt.savefig("sensitivity_loo_entities.png", dpi=300, bbox_inches="tight")
    print("   ✓ Saved: sensitivity_loo_entities.png")
    plt.close(fig1)

    # Plot LOO periods
    fig2 = sensitivity.plot_sensitivity(loo_periods, figsize=(14, 5))
    plt.savefig("sensitivity_loo_periods.png", dpi=300, bbox_inches="tight")
    print("   ✓ Saved: sensitivity_loo_periods.png")
    plt.close(fig2)

    # Plot subset sensitivity
    fig3 = sensitivity.plot_sensitivity(subset_results, figsize=(14, 5))
    plt.savefig("sensitivity_subset.png", dpi=300, bbox_inches="tight")
    print("   ✓ Saved: sensitivity_subset.png")
    plt.close(fig3)

    print("\n   📊 All plots saved successfully!")

except ImportError:
    print("\n   ⚠ Matplotlib not available - skipping plots")
    print("   Install with: pip install matplotlib")

# ============================================================================
# 15. Summary and Recommendations
# ============================================================================

print("\n" + "=" * 80)
print("Summary and Recommendations")
print("=" * 80)

print(f"\n✅ Sensitivity analysis completed successfully")
print(f"\n📊 Results Summary:")
print(f"   - Original x1 estimate: {results.params['x1']:.4f}")
print(
    f"   - LOO entities mean:    {loo_entities.estimates['x1'].mean():.4f} "
    f"(CV: {cv_entities_x1:.2f}%)"
)
print(
    f"   - LOO periods mean:     {loo_periods.estimates['x1'].mean():.4f} "
    f"(CV: {cv_periods_x1:.2f}%)"
)
print(
    f"   - Subset mean:          {subset_results.estimates['x1'].mean():.4f} "
    f"(CV: {cv_subset_x1:.2f}%)"
)

print(f"\n   - Original x2 estimate: {results.params['x2']:.4f}")
print(
    f"   - LOO entities mean:    {loo_entities.estimates['x2'].mean():.4f} "
    f"(CV: {cv_entities_x2:.2f}%)"
)
print(
    f"   - LOO periods mean:     {loo_periods.estimates['x2'].mean():.4f} "
    f"(CV: {cv_periods_x2:.2f}%)"
)
print(
    f"   - Subset mean:          {subset_results.estimates['x2'].mean():.4f} "
    f"(CV: {cv_subset_x2:.2f}%)"
)

print("\n💡 Key Findings:")
if loo_entities.influential_units:
    print(f"   ⚠ Identified {len(loo_entities.influential_units)} influential entities")
    print("   → Review these entities for outliers or data quality issues")
else:
    print("   ✓ No influential entities detected")

if max(cv_entities_x1, cv_entities_x2, cv_periods_x1, cv_periods_x2) < 10:
    print("   ✓ Estimates are stable across all sensitivity tests")
    print("   → Results are robust to sample composition")
else:
    print("   ⚠ Some instability detected in estimates")
    print("   → Consider robustness checks or alternative specifications")

print("\n📚 Next Steps:")
print("   1. Investigate influential units (if any)")
print("   2. Run bootstrap analysis for confidence intervals")
print("   3. Consider robustness checks with different specifications")
print("   4. Document sensitivity findings in research paper")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

# ============================================================================
# Optional: Save detailed results
# ============================================================================

save_results = False  # Set to True to save

if save_results:
    # Save all estimates
    loo_entities.estimates.to_csv("sensitivity_loo_entities_estimates.csv")
    loo_periods.estimates.to_csv("sensitivity_loo_periods_estimates.csv")
    subset_results.estimates.to_csv("sensitivity_subset_estimates.csv")

    # Save summaries
    summary_entities.to_csv("sensitivity_loo_entities_summary.csv", index=False)
    summary_periods.to_csv("sensitivity_loo_periods_summary.csv", index=False)
    summary_subset.to_csv("sensitivity_subset_summary.csv", index=False)

    print("\n💾 Results saved to CSV files")
