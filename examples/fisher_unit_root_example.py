"""
Fisher-type Panel Unit Root Test - Complete Example

This example demonstrates the use of Fisher-type tests for panel data,
which combine p-values from individual unit root tests.

The Fisher test:
- Combines p-values from individual ADF or PP tests
- Allows for heterogeneity across entities
- Handles unbalanced panels
- Simple to implement and interpret

Examples cover:
1. Basic usage with Grunfeld data
2. Comparison with LLC and IPS tests
3. Fisher-ADF vs Fisher-PP
4. Different trend specifications
5. Interpretation guidelines
"""

import sys

sys.path.insert(0, "/home/guhaase/projetos/panelbox")

import numpy as np
import pandas as pd

import panelbox as pb

print("=" * 80)
print("FISHER-TYPE PANEL UNIT ROOT TEST - EXAMPLES")
print("=" * 80)

# ============================================================================
# Example 1: Basic Usage with Grunfeld Data
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 1: Basic Usage with Grunfeld Data")
print("=" * 80)

# Load data
data = pb.load_grunfeld()
print(f"\nDataset: {len(data)} observations, {data['firm'].nunique()} firms")

# Test 'invest' variable for unit root
print("\n1.1 Fisher-ADF Test on 'invest':")
print("-" * 80)

fisher_adf = pb.FisherTest(data, "invest", "firm", "year", test_type="adf", trend="c")
result_adf = fisher_adf.run()

print(result_adf)

print("\nIndividual p-values (first 5 firms):")
for entity, pval in list(result_adf.individual_pvalues.items())[:5]:
    print(f"  Firm {entity}: {pval:.4f}")

# ============================================================================
# Example 2: Comparison of Unit Root Tests (LLC, IPS, Fisher)
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 2: Comparison of Unit Root Tests")
print("=" * 80)

variables = ["invest", "value", "capital"]

print("\nTesting all three variables with multiple tests:")
print("-" * 80)

results_comparison = []

for var in variables:
    print(f"\nVariable: {var}")
    print("  " + "-" * 76)

    # LLC Test (assume homogeneity)
    llc = pb.LLCTest(data, var, "firm", "year", lags=1, trend="c")
    llc_result = llc.run()

    # IPS Test (allows heterogeneity)
    ips = pb.IPSTest(data, var, "firm", "year", lags=1, trend="c")
    ips_result = ips.run()

    # Fisher-ADF Test
    fisher = pb.FisherTest(data, var, "firm", "year", test_type="adf", trend="c")
    fisher_result = fisher.run()

    print(f"  LLC:        stat={llc_result.statistic:8.4f}, p={llc_result.pvalue:.4f}")
    print(f"  IPS:        stat={ips_result.statistic:8.4f}, p={ips_result.pvalue:.4f}")
    print(f"  Fisher-ADF: stat={fisher_result.statistic:8.4f}, p={fisher_result.pvalue:.4f}")

    results_comparison.append(
        {
            "Variable": var,
            "LLC_stat": llc_result.statistic,
            "LLC_pval": llc_result.pvalue,
            "IPS_stat": ips_result.statistic,
            "IPS_pval": ips_result.pvalue,
            "Fisher_stat": fisher_result.statistic,
            "Fisher_pval": fisher_result.pvalue,
        }
    )

print("\n💡 Interpretation:")
print("   - LLC assumes homogeneity (same ρ for all entities)")
print("   - IPS and Fisher allow heterogeneity (different ρ_i)")
print("   - Fisher combines p-values from individual tests")
print("   - All three tests complement each other")

# ============================================================================
# Example 3: Fisher-ADF vs Fisher-PP
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 3: Fisher-ADF vs Fisher-PP")
print("=" * 80)

print("\nComparing ADF-based and PP-based Fisher tests:")
print("-" * 80)

# Fisher-ADF
fisher_adf = pb.FisherTest(data, "value", "firm", "year", test_type="adf", trend="c")
result_adf = fisher_adf.run()

# Fisher-PP
fisher_pp = pb.FisherTest(data, "value", "firm", "year", test_type="pp", trend="c")
result_pp = fisher_pp.run()

print(f"\nFisher-ADF:")
print(f"  Statistic: {result_adf.statistic:.4f}")
print(f"  P-value:   {result_adf.pvalue:.4f}")
print(f"  Conclusion: {result_adf.conclusion}")

print(f"\nFisher-PP:")
print(f"  Statistic: {result_pp.statistic:.4f}")
print(f"  P-value:   {result_pp.pvalue:.4f}")
print(f"  Conclusion: {result_pp.conclusion}")

print("\n💡 Interpretation:")
print("   - ADF: parametric test (estimates AR lags)")
print("   - PP: non-parametric (uses Newey-West correction)")
print("   - ADF usually preferred for Fisher test")
print("   - PP can be useful for comparison")

# ============================================================================
# Example 4: Trend Specifications
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 4: Different Trend Specifications")
print("=" * 80)

print("\nTesting 'capital' with different trend specifications:")
print("-" * 80)

trends = {"n": "No constant, no trend", "c": "Constant only", "ct": "Constant and trend"}

for trend_spec, description in trends.items():
    fisher = pb.FisherTest(data, "capital", "firm", "year", test_type="adf", trend=trend_spec)
    result = fisher.run()

    print(f"\nTrend '{trend_spec}' ({description}):")
    print(f"  Statistic: {result.statistic:8.4f}")
    print(f"  P-value:   {result.pvalue:8.4f}")

print("\n💡 Interpretation:")
print("   - 'n': For series without drift or trend")
print("   - 'c': Most common (allows for different means)")
print("   - 'ct': For series with deterministic trend")
print("   - Choice affects critical values and power")

# ============================================================================
# Example 5: Simulated Data - Stationary vs Unit Root
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 5: Simulated Data - Comparing Stationary vs Unit Root")
print("=" * 80)

np.random.seed(42)
n_entities = 10
n_time = 50

# 5.1 Stationary data (AR(1) with ρ = 0.5)
print("\n5.1 Stationary Data (AR(1) with ρ = 0.5):")
print("-" * 80)

data_stationary = []
for i in range(n_entities):
    y = np.zeros(n_time)
    y[0] = np.random.randn()
    for t in range(1, n_time):
        y[t] = 0.5 * y[t - 1] + np.random.randn()

    df_i = pd.DataFrame({"entity": i, "time": range(n_time), "y": y})
    data_stationary.append(df_i)

data_stat = pd.concat(data_stationary, ignore_index=True)

fisher_stat = pb.FisherTest(data_stat, "y", "entity", "time", test_type="adf", trend="c")
result_stat = fisher_stat.run()

print(f"Fisher statistic: {result_stat.statistic:.4f}")
print(f"P-value: {result_stat.pvalue:.4f}")
print(f"Conclusion: {result_stat.conclusion}")

# 5.2 Unit root data (random walk)
print("\n5.2 Unit Root Data (Random Walk):")
print("-" * 80)

data_unit_root = []
for i in range(n_entities):
    y = np.cumsum(np.random.randn(n_time))  # Random walk

    df_i = pd.DataFrame({"entity": i, "time": range(n_time), "y": y})
    data_unit_root.append(df_i)

data_ur = pd.concat(data_unit_root, ignore_index=True)

fisher_ur = pb.FisherTest(data_ur, "y", "entity", "time", test_type="adf", trend="c")
result_ur = fisher_ur.run()

print(f"Fisher statistic: {result_ur.statistic:.4f}")
print(f"P-value: {result_ur.pvalue:.4f}")
print(f"Conclusion: {result_ur.conclusion}")

print("\n💡 Interpretation:")
print("   - Stationary data → low p-value (reject H0)")
print("   - Unit root data → high p-value (fail to reject H0)")
print("   - Fisher test correctly distinguishes the two cases")

# ============================================================================
# Example 6: Unbalanced Panel
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 6: Unbalanced Panel")
print("=" * 80)

print("\nFisher test can handle unbalanced panels:")
print("-" * 80)

# Create unbalanced panel
np.random.seed(123)
data_unbalanced = []

# Different lengths for different entities
lengths = [50, 30, 40, 35, 45]

for i, length in enumerate(lengths):
    y = np.random.randn(length).cumsum() * 0.3
    df_i = pd.DataFrame({"entity": i, "time": range(length), "y": y})
    data_unbalanced.append(df_i)

data_unbal = pd.concat(data_unbalanced, ignore_index=True)

print(f"Panel structure:")
for i, length in enumerate(lengths):
    print(f"  Entity {i}: {length} observations")

fisher_unbal = pb.FisherTest(data_unbal, "y", "entity", "time", test_type="adf", trend="c")
result_unbal = fisher_unbal.run()

print(f"\nFisher statistic: {result_unbal.statistic:.4f}")
print(f"P-value: {result_unbal.pvalue:.4f}")
print(f"Entities tested: {result_unbal.n_entities}")

print("\n💡 Key Advantage:")
print("   - Fisher test naturally handles unbalanced panels")
print("   - Each entity tested independently")
print("   - P-values combined via inverse chi-square")

# ============================================================================
# SUMMARY AND GUIDELINES
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: When to Use Fisher Test")
print("=" * 80)

print(
    """
Fisher-type test is ideal when:

✅ Advantages:
  • You have unbalanced panels
  • You want to allow heterogeneity across entities
  • You need a simple, intuitive test
  • You want to see individual entity p-values
  • T doesn't need to be large (unlike LLC)

⚠️  Considerations:
  • Assumes cross-sectional independence
  • Individual tests must be valid (need sufficient T per entity)
  • P-value combination is conservative
  • Less powerful than IPS in some cases

📊 Workflow:
  1. Check for unit roots with Fisher test
  2. Compare with LLC and IPS for robustness
  3. Examine individual p-values for outliers
  4. If H0 rejected: at least one series is stationary
  5. If H0 not rejected: proceed with differences or GMM

🔬 Test Variants:
  • Fisher-ADF: Most common (parametric)
  • Fisher-PP: Alternative (non-parametric)
  • Both use inverse chi-square transformation

📖 References:
  • Maddala & Wu (1999) - Original Fisher test paper
  • Choi (2001) - Extensions and modifications
"""
)

print("=" * 80)
print("✅ FISHER UNIT ROOT TEST EXAMPLES COMPLETED!")
print("=" * 80)
print("\n💡 Try running: python examples/fisher_unit_root_example.py")
