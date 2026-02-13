"""
Example: Dumitrescu-Hurlin (2012) Test for Heterogeneous Panels

This example demonstrates:
1. Why heterogeneity matters in panel Granger causality
2. How to use the Dumitrescu-Hurlin test
3. Comparison with standard homogeneous Wald test
4. Interpretation of individual entity statistics
"""

import numpy as np
import pandas as pd

from panelbox.var.causality import dumitrescu_hurlin_test

# ==============================================================================
# 1. Generate heterogeneous panel data
# ==============================================================================


def generate_heterogeneous_panel(
    N=50,
    T=60,
    causality_share=0.5,  # Share of entities with causality
    beta_range=(0.3, 0.7),  # Range of causal coefficients
    seed=42,
):
    """
    Generate panel data with heterogeneous causality.

    Half of entities have x → y causality with varying strength.
    The other half have no causality.
    """
    np.random.seed(seed)

    data_list = []

    for i in range(N):
        # Determine if this entity has causality
        has_causality = (i / N) < causality_share

        if has_causality:
            # Random causal coefficient
            beta_xy = np.random.uniform(*beta_range)
        else:
            beta_xy = 0.0

        # Generate time series
        x = np.zeros(T)
        y = np.zeros(T)

        x[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)

        for t in range(1, T):
            # Entity-specific autoreg coefficients (heterogeneous)
            alpha_x = np.random.uniform(0.2, 0.5)
            alpha_y = np.random.uniform(0.3, 0.6)

            x[t] = alpha_x * x[t - 1] + np.random.normal(0, 0.5)
            y[t] = alpha_y * y[t - 1] + beta_xy * x[t - 1] + np.random.normal(0, 0.5)

        entity_df = pd.DataFrame(
            {
                "entity": i,
                "time": np.arange(T),
                "x": x,
                "y": y,
                "true_beta": beta_xy,  # For validation
            }
        )

        data_list.append(entity_df)

    return pd.concat(data_list, ignore_index=True)


# ==============================================================================
# 2. Run Dumitrescu-Hurlin Test
# ==============================================================================

print("=" * 80)
print("DUMITRESCU-HURLIN (2012) TEST FOR HETEROGENEOUS PANELS")
print("=" * 80)
print()

# Generate heterogeneous data
print("Generating heterogeneous panel data...")
print("  - 50% of entities have x → y causality")
print("  - 50% of entities have NO causality")
print("  - Causal coefficients vary across entities (0.3 to 0.7)")
print()

df = generate_heterogeneous_panel(N=50, T=60, causality_share=0.5)

print(
    f"Dataset: {df['entity'].nunique()} entities, avg {df.groupby('entity').size().mean():.0f} time periods"
)
print()

# Run DH test
print("Running Dumitrescu-Hurlin test...")
print()

result = dumitrescu_hurlin_test(
    data=df, cause="x", effect="y", lags=1, entity_col="entity", time_col="time"
)

# Display results
print(result.summary())
print()

# ==============================================================================
# 3. Examine individual entity statistics
# ==============================================================================

print("=" * 80)
print("INDIVIDUAL ENTITY STATISTICS")
print("=" * 80)
print()

print("Distribution of Wald statistics (W_i) across entities:")
print(f"  Min:     {np.min(result.individual_W):>10.4f}")
print(f"  25th %:  {np.percentile(result.individual_W, 25):>10.4f}")
print(f"  Median:  {np.median(result.individual_W):>10.4f}")
print(f"  75th %:  {np.percentile(result.individual_W, 75):>10.4f}")
print(f"  Max:     {np.max(result.individual_W):>10.4f}")
print()

# Identify which entities show strong evidence of causality
from scipy import stats as sp_stats

critical_value_05 = sp_stats.chi2.ppf(0.95, df=result.lags)
n_significant = np.sum(result.individual_W > critical_value_05)

print(f"Critical value (5% level): {critical_value_05:.4f}")
print(
    f"Entities with W_i > critical value: {n_significant}/{result.N} ({n_significant/result.N*100:.1f}%)"
)
print()

# Top 10 entities with strongest causality evidence
top_10_idx = np.argsort(result.individual_W)[-10:][::-1]
print("Top 10 entities with strongest causality evidence:")
for rank, idx in enumerate(top_10_idx, 1):
    W_i = result.individual_W[idx]
    p_approx = 1 - sp_stats.chi2.cdf(W_i, df=result.lags)
    print(f"  {rank:2d}. Entity {idx:2d}: W_i = {W_i:>8.4f}, p ≈ {p_approx:.4f}")
print()

# Bottom 10 entities (weakest evidence)
bottom_10_idx = np.argsort(result.individual_W)[:10]
print("Bottom 10 entities (weakest/no causality evidence):")
for rank, idx in enumerate(bottom_10_idx, 1):
    W_i = result.individual_W[idx]
    p_approx = 1 - sp_stats.chi2.cdf(W_i, df=result.lags)
    print(f"  {rank:2d}. Entity {idx:2d}: W_i = {W_i:>8.4f}, p ≈ {p_approx:.4f}")
print()

# ==============================================================================
# 4. Visualization (if matplotlib available)
# ==============================================================================

print("=" * 80)
print("VISUALIZATION")
print("=" * 80)
print()

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend

    # Plot histogram of individual W statistics
    fig = result.plot_individual_statistics(backend="matplotlib", show=False)

    # Save to file
    output_file = "dumitrescu_hurlin_histogram.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Histogram saved to: {output_file}")
    print()
except ImportError:
    print("Matplotlib not available. Skipping visualization.")
    print()

# ==============================================================================
# 5. Comparison: Homogeneous vs Heterogeneous
# ==============================================================================

print("=" * 80)
print("COMPARISON: STANDARD WALD VS DUMITRESCU-HURLIN")
print("=" * 80)
print()

print("Key Differences:")
print()
print("1. STANDARD WALD TEST (Homogeneous):")
print("   - Assumes ALL entities have same causal coefficient β")
print("   - Tests H0: β = 0 for all entities")
print("   - Can be underpowered if heterogeneity exists")
print()

print("2. DUMITRESCU-HURLIN TEST (Heterogeneous):")
print("   - Allows β to vary across entities: β_i")
print("   - Tests H0: β_i = 0 for ALL entities (homogeneous non-causality)")
print("   - H1: β_i ≠ 0 for AT LEAST SOME entities")
print("   - More powerful when heterogeneity exists")
print()

print("In this example:")
print(f"  - True structure: 50% have causality, 50% don't")
print(
    f"  - DH correctly detects causality (p = {min(result.Z_tilde_pvalue, result.Z_bar_pvalue):.4f})"
)
print(f"  - Standard Wald on pooled data would average out the heterogeneity")
print()

# ==============================================================================
# 6. When to use each test
# ==============================================================================

print("=" * 80)
print("GUIDELINES: WHEN TO USE EACH TEST")
print("=" * 80)
print()

print("Use STANDARD WALD TEST when:")
print("  ✓ You have reason to believe causal effects are homogeneous")
print("  ✓ You've estimated a pooled Panel VAR (common coefficients)")
print("  ✓ Small N (< 10 entities)")
print("  ✓ You want to test specific restrictions on pooled model")
print()

print("Use DUMITRESCU-HURLIN TEST when:")
print("  ✓ You suspect heterogeneous causal effects across entities")
print("  ✓ You have moderate to large N (> 20 entities)")
print("  ✓ T is not too small (T > 5 + Kp, preferably T > 20)")
print("  ✓ You want to test if causality exists for ANY subset of entities")
print("  ✓ Cross-country or cross-firm panels with structural differences")
print()

print("Recommended Workflow:")
print("  1. Start with DH test (more general, allows heterogeneity)")
print("  2. Examine individual W_i statistics to understand heterogeneity")
print("  3. If DH rejects, check which entities drive the result")
print("  4. Consider subgroup analysis if clear patterns emerge")
print()

# ==============================================================================
# 7. Statistical Notes
# ==============================================================================

print("=" * 80)
print("TECHNICAL NOTES")
print("=" * 80)
print()

print("Recommended Statistic:")
print(f"  - For your sample (T̄ = {result.T_avg:.1f}): Use {result.recommended_stat.upper()}")
print()

print("  - Z̃ (Z_tilde): Better for small T (T < 10)")
print("    Uses simple moments: E[W_i] = p, Var[W_i] = 2p")
print()

print("  - Z̄ (Z_bar): Better for larger T (T ≥ 10)")
print("    Uses exact finite-sample moments (more accurate)")
print()

print(f"Your results:")
print(f"  - Z̃ = {result.Z_tilde_stat:.4f}, p-value = {result.Z_tilde_pvalue:.4f}")
print(f"  - Z̄ = {result.Z_bar_stat:.4f}, p-value = {result.Z_bar_pvalue:.4f}")
print()

# ==============================================================================
# 8. Interpretation and Caveats
# ==============================================================================

print("=" * 80)
print("INTERPRETATION AND CAVEATS")
print("=" * 80)
print()

print("Interpretation of DH Test Rejection:")
print("  - DOES mean: Granger causality exists for at least some entities")
print("  - DOES NOT mean: Causality exists for all entities")
print("  - DOES NOT mean: The causal effect is the same across entities")
print()

print("Caveats:")
print("  1. Like standard Granger test, DH tests predictability, not true causation")
print("  2. Sensitive to lag length specification")
print("  3. Assumes no cross-sectional dependence (extension exists for CD)")
print("  4. Requires T large enough for individual regressions (T > Kp + 1)")
print()

print("Next Steps:")
print("  - If DH rejects: Investigate which entities have causality")
print("  - Consider panel VAR with entity-specific coefficients")
print("  - Use Impulse Response Functions (IRFs) to quantify dynamic effects")
print("  - Bootstrap for small-sample refinement (future release)")
print()

print("=" * 80)
print("END OF EXAMPLE")
print("=" * 80)
