"""
Standard Error Comparison Example
==================================

This example demonstrates how to use the StandardErrorComparison class
to compare different types of standard errors for the same model specification.

This is useful for:
1. Assessing robustness of inference to SE assumptions
2. Identifying which coefficients have sensitive inference
3. Understanding the impact of different data structures (clustering, heteroskedasticity)
"""

import numpy as np
import pandas as pd
from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.standard_errors import StandardErrorComparison

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("STANDARD ERROR COMPARISON EXAMPLE")
print("=" * 80)
print()

# ===========================
# 1. Generate Panel Data
# ===========================

print("1. Generating panel data...")
print("-" * 80)

n_entities = 30
n_periods = 10
n = n_entities * n_periods

# Create panel structure
data = pd.DataFrame({
    'entity': np.repeat(range(n_entities), n_periods),
    'time': np.tile(range(n_periods), n_entities),
})

# Generate y with heteroskedasticity and entity-level effects
entity_effects = np.random.randn(n_entities)
data['y'] = (
    entity_effects[data['entity']] +  # Fixed effect
    0.5 * np.random.randn(n) +         # Heteroskedastic error
    0.3 * np.random.randn(n) * (data['entity'] % 5)  # Group-level heterosked
)

# Generate x1 and x2
data['x1'] = np.random.randn(n)
data['x2'] = np.random.randn(n)

# Add x1 and x2 effects to y
data['y'] += 0.8 * data['x1'] + 1.2 * data['x2']

print(f"Panel structure: {n_entities} entities × {n_periods} periods = {n} observations")
print()

# ===========================
# 2. Estimate Models
# ===========================

print("2. Estimating models...")
print("-" * 80)

# Fixed Effects
print("Fitting Fixed Effects model...")
fe = FixedEffects('y ~ x1 + x2', data, 'entity', 'time')
fe_results = fe.fit()

print(f"Fixed Effects coefficients:")
print(fe_results.params)
print()

# Pooled OLS (for comparison)
print("Fitting Pooled OLS model...")
pooled = PooledOLS('y ~ x1 + x2', data, 'entity', 'time')
pooled_results = pooled.fit()

print(f"Pooled OLS coefficients:")
print(pooled_results.params)
print()

# ===========================
# 3. Fixed Effects - Standard Error Comparison
# ===========================

print("3. Comparing Standard Errors for Fixed Effects Model")
print("=" * 80)
print()

# Method 1: Manual comparison by refitting
print("Method 1: Manual Comparison by Refitting")
print("-" * 80)

se_types_to_compare = ['nonrobust', 'robust', 'hc3', 'clustered', 'twoway']
se_results = {}

for se_type in se_types_to_compare:
    result = fe.fit(cov_type=se_type)
    se_results[se_type] = result.std_errors

# Create comparison DataFrame
se_comparison_df = pd.DataFrame(se_results)
print("Standard Errors:")
print(se_comparison_df)
print()

# Compute ratios relative to nonrobust
se_ratios = se_comparison_df.div(se_comparison_df['nonrobust'], axis=0)
print("SE Ratios (relative to nonrobust):")
print(se_ratios.round(3))
print()

# ===========================
# 4. Pooled OLS - Standard Error Comparison
# ===========================

print("4. Comparing Standard Errors for Pooled OLS Model")
print("=" * 80)
print()

# Pooled OLS comparison
pooled_se_types = ['nonrobust', 'robust', 'hc3', 'clustered', 'twoway', 'driscoll_kraay']
pooled_se_results = {}

for se_type in pooled_se_types:
    try:
        if se_type == 'driscoll_kraay':
            result = pooled.fit(cov_type=se_type, max_lags=2)
        else:
            result = pooled.fit(cov_type=se_type)
        pooled_se_results[se_type] = result.std_errors
    except Exception as e:
        print(f"Warning: Could not compute {se_type}: {e}")

# Create comparison DataFrame
pooled_se_df = pd.DataFrame(pooled_se_results)
print("Standard Errors:")
print(pooled_se_df)
print()

# Compute ratios
pooled_se_ratios = pooled_se_df.div(pooled_se_df['nonrobust'], axis=0)
print("SE Ratios (relative to nonrobust):")
print(pooled_se_ratios.round(3))
print()

# ===========================
# 5. Inference Sensitivity Analysis
# ===========================

print("5. Inference Sensitivity Analysis")
print("=" * 80)
print()

# Compute t-statistics and p-values for each SE type
from scipy import stats

alpha = 0.05
df = fe_results.df_resid

print("Fixed Effects - Significance at 5% level:")
print("-" * 80)

for se_type in se_types_to_compare:
    result = fe.fit(cov_type=se_type)
    t_stats = result.params / result.std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))
    significant = (p_values < alpha).values if hasattr(p_values, 'values') else (p_values < alpha)

    print(f"\n{se_type}:")
    for i, coef_name in enumerate(result.params.index):
        sig_marker = "***" if significant[i] else ""
        pval = p_values.iloc[i] if hasattr(p_values, 'iloc') else p_values[i]
        print(f"  {coef_name:10s}: coef={result.params.iloc[i]:7.4f}, "
              f"se={result.std_errors.iloc[i]:7.4f}, "
              f"p={pval:6.4f} {sig_marker}")

print()

# ===========================
# 6. Summary and Recommendations
# ===========================

print("6. Summary and Recommendations")
print("=" * 80)
print()

print("Key Findings:")
print("-" * 80)

# Compare robust vs nonrobust
robust_ratio_mean = se_ratios['robust'].mean()
clustered_ratio_mean = se_ratios['clustered'].mean()

print(f"1. Heteroskedasticity Impact:")
print(f"   - Average robust SE is {robust_ratio_mean:.2f}× nonrobust SE")
if robust_ratio_mean > 1.2:
    print(f"   ⚠️  Substantial heteroskedasticity detected")
    print(f"   → Recommendation: Use robust or clustered SEs")
else:
    print(f"   ✓ Modest heteroskedasticity")

print()
print(f"2. Clustering Impact:")
print(f"   - Average clustered SE is {clustered_ratio_mean:.2f}× nonrobust SE")
if clustered_ratio_mean > 1.5:
    print(f"   ⚠️  Strong within-entity correlation")
    print(f"   → Recommendation: Use clustered SEs")
else:
    print(f"   ✓ Moderate clustering effect")

print()
print(f"3. Inference Robustness:")
# Check if inference is consistent across SE types
consistent_inference = True
for i, coef_name in enumerate(fe_results.params.index):
    sig_counts = []
    for se_type in se_types_to_compare:
        result = fe.fit(cov_type=se_type)
        t_stat = result.params.iloc[i] / result.std_errors.iloc[i]
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
        sig_counts.append(p_value < alpha)

    # If not all agree or all disagree, inference is inconsistent
    if sum(sig_counts) not in [0, len(sig_counts)]:
        print(f"   ⚠️  Inconsistent inference for {coef_name}")
        print(f"       Significant in {sum(sig_counts)}/{len(sig_counts)} SE types")
        consistent_inference = False

if consistent_inference:
    print(f"   ✓ Inference is consistent across all SE types")

print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()
print("For panel data with potential entity-level correlation:")
print("→ Use clustered standard errors (clustered by entity)")
print()
print("For macro panels with spatial/temporal correlation:")
print("→ Use Driscoll-Kraay standard errors")
print()
print("For robustness checks:")
print("→ Report results with multiple SE types to assess sensitivity")
print()

# ===========================
# 7. Optional: Visual Comparison (if matplotlib available)
# ===========================

try:
    import matplotlib.pyplot as plt

    print("7. Visual Comparison")
    print("=" * 80)
    print()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: SE comparison
    se_comparison_df.plot(kind='bar', ax=ax1)
    ax1.set_title('Standard Error Comparison (Fixed Effects)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Coefficient')
    ax1.set_ylabel('Standard Error')
    ax1.legend(title='SE Type')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: SE ratios
    se_ratios.drop('nonrobust', axis=1).plot(kind='bar', ax=ax2)
    ax2.set_title('SE Ratios (relative to nonrobust)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Coefficient')
    ax2.set_ylabel('Ratio')
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline')
    ax2.legend(title='SE Type')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('se_comparison_example.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved as 'se_comparison_example.png'")
    print()

except ImportError:
    print("(Matplotlib not available - skipping plots)")
    print()

print("=" * 80)
print("EXAMPLE COMPLETE")
print("=" * 80)
