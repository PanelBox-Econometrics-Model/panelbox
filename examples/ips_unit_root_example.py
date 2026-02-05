"""
Example: IPS (Im-Pesaran-Shin) Panel Unit Root Test

This example demonstrates how to use the IPS test to check for unit roots
in panel data, allowing for heterogeneity across panels.
"""

import sys

sys.path.insert(0, "/home/guhaase/projetos/panelbox")

import numpy as np
import pandas as pd

import panelbox as pb

print("=" * 70)
print("IPS Panel Unit Root Test - Examples")
print("=" * 70)

# Example 1: Testing with Grunfeld data
print("\n" + "=" * 70)
print("Example 1: Grunfeld Dataset")
print("=" * 70)

data = pb.load_grunfeld()
print(
    f"\nDataset: {len(data)} observations, {data['firm'].nunique()} firms, {data['year'].nunique()} years"
)

# Test each variable
variables = ["invest", "value", "capital"]

for var in variables:
    print(f"\n{'-'*70}")
    print(f"Testing variable: {var.upper()}")
    print(f"{'-'*70}")

    ips = pb.IPSTest(data, var, "firm", "year", lags=1, trend="c")
    result = ips.run()

    print(result)


# Example 2: Heterogeneous stationary data
print("\n" + "=" * 70)
print("Example 2: Heterogeneous Stationary Data")
print("=" * 70)
print("AR(1) processes with different ρ_i for each firm (0.3 to 0.7)")

np.random.seed(42)
n_firms = 12
n_years = 40

data_list = []
for i in range(n_firms):
    # Different AR coefficient for each firm
    rho = 0.3 + 0.4 * (i / n_firms)  # ρ from 0.3 to 0.7
    y = np.zeros(n_years)
    y[0] = np.random.normal(0, 1)
    for t in range(1, n_years):
        y[t] = rho * y[t - 1] + np.random.normal(0, 1)

    data_list.append(pd.DataFrame({"firm": i, "year": range(n_years), "y": y}))

heterog_data = pd.concat(data_list, ignore_index=True)

ips = pb.IPSTest(heterog_data, "y", "firm", "year", lags=1, trend="c")
result = ips.run()

print(result)
print(f"\nInterpretation: P-value = {result.pvalue:.4f} < 0.05")
print("IPS correctly detects stationarity even with heterogeneous ρ_i")


# Example 3: Mixed panel (some stationary, some unit root)
print("\n" + "=" * 70)
print("Example 3: Mixed Panel (Half Stationary, Half Unit Root)")
print("=" * 70)

np.random.seed(123)
n_firms = 10
n_years = 50

data_list = []
for i in range(n_firms):
    if i < 5:
        # Stationary (AR(1) with ρ=0.6)
        rho = 0.6
        y = np.zeros(n_years)
        y[0] = np.random.normal(0, 1)
        for t in range(1, n_years):
            y[t] = rho * y[t - 1] + np.random.normal(0, 1)
    else:
        # Unit root (random walk)
        y = np.cumsum(np.random.normal(0, 1, n_years))

    data_list.append(pd.DataFrame({"firm": i, "year": range(n_years), "y": y}))

mixed_data = pd.concat(data_list, ignore_index=True)

ips = pb.IPSTest(mixed_data, "y", "firm", "year", lags=1, trend="c")
result = ips.run()

print(result)
print(f"\nInterpretation: P-value = {result.pvalue:.4f}")
print("IPS alternative hypothesis is 'SOME panels are stationary'")
print("So it rejects H0 because at least some panels (firms 0-4) are stationary")

# Show individual statistics
print("\nIndividual t-statistics:")
for firm_id, t_stat in result.individual_stats.items():
    panel_type = "Stationary" if firm_id < 5 else "Unit root"
    print(f"  Firm {firm_id}: t={t_stat:7.3f}  ({panel_type})")


# Example 4: Comparing LLC vs IPS
print("\n" + "=" * 70)
print("Example 4: Comparing LLC vs IPS on Heterogeneous Data")
print("=" * 70)

# Use the heterogeneous stationary data from Example 2
print("\nData: Heterogeneous AR(1) with ρ_i from 0.3 to 0.7")

# LLC test (assumes homogeneity)
print("\nLLC Test (assumes common ρ):")
llc = pb.LLCTest(heterog_data, "y", "firm", "year", lags=1, trend="c")
llc_result = llc.run()
print(f"  Statistic: {llc_result.statistic:.4f}")
print(f"  P-value:   {llc_result.pvalue:.4f}")
print(f"  Conclusion: {llc_result.conclusion}")

# IPS test (allows heterogeneity)
print("\nIPS Test (allows heterogeneous ρ_i):")
ips = pb.IPSTest(heterog_data, "y", "firm", "year", lags=1, trend="c")
ips_result = ips.run()
print(f"  W-stat:    {ips_result.statistic:.4f}")
print(f"  P-value:   {ips_result.pvalue:.4f}")
print(f"  Conclusion: {ips_result.conclusion}")

print("\nBoth tests reject H0, but IPS is more appropriate for")
print("heterogeneous panels.")


# Example 5: Automatic lag selection per entity
print("\n" + "=" * 70)
print("Example 5: Automatic Lag Selection (Entity-Specific)")
print("=" * 70)

data = pb.load_grunfeld()

print("\nIPS allows different lag lengths for different entities:")
ips = pb.IPSTest(data, "invest", "firm", "year", lags=None, trend="c")
result = ips.run()

if isinstance(result.lags, list):
    print(f"\nLags selected per firm: {result.lags}")
    print(f"Mean lags: {np.mean(result.lags):.1f}")
else:
    print(f"\nSame lags for all: {result.lags}")

print(f"\nW-statistic: {result.statistic:.4f}")
print(f"P-value: {result.pvalue:.4f}")


# Example 6: Different trend specifications
print("\n" + "=" * 70)
print("Example 6: Effect of Trend Specification")
print("=" * 70)

data = pb.load_grunfeld()

print("\nTesting 'value' with different specifications:")
print(f"{'-'*70}")

trends = [("n", "No deterministic terms"), ("c", "Constant only"), ("ct", "Constant and trend")]

for trend_code, trend_desc in trends:
    ips = pb.IPSTest(data, "value", "firm", "year", lags=1, trend=trend_code)
    result = ips.run()

    print(f"\n{trend_desc}:")
    print(f"  W-statistic: {result.statistic:8.4f}")
    print(f"  t-bar:       {result.t_bar:8.4f}")
    print(f"  P-value:     {result.pvalue:8.4f}")
    print(f"  Conclusion:  {result.conclusion}")


print("\n" + "=" * 70)
print("Summary: When to Use IPS vs LLC")
print("=" * 70)
print(
    """
**Use IPS when:**
- You suspect heterogeneity in autoregressive coefficients across panels
- You want a more general test (IPS nests LLC as special case)
- You have unbalanced panels (IPS handles them naturally)
- Alternative hypothesis is "some panels are stationary" (not all)

**Use LLC when:**
- You believe all panels follow the same unit root process
- You need a more powerful test under homogeneity
- Alternative hypothesis is "all panels are stationary"

**Key Differences:**
- LLC: H0: ρ_i = 0 for all i  vs  H1: ρ_i = ρ < 0 for all i
- IPS: H0: ρ_i = 0 for all i  vs  H1: ρ_i < 0 for some i

IPS is generally preferred as it's more flexible and robust to
heterogeneity, which is common in panel data.
"""
)

print("\n✓ Examples completed successfully!")
