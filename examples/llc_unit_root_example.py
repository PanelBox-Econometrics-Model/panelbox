"""
Example: LLC (Levin-Lin-Chu) Panel Unit Root Test

This example demonstrates how to use the LLC test to check for unit roots
in panel data.
"""

import sys

sys.path.insert(0, "/home/guhaase/projetos/panelbox")

import numpy as np
import pandas as pd

import panelbox as pb

print("=" * 70)
print("LLC Panel Unit Root Test - Examples")
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

    llc = pb.LLCTest(data, var, "firm", "year", lags=1, trend="c")
    result = llc.run()

    print(result)


# Example 2: Simulated stationary data
print("\n" + "=" * 70)
print("Example 2: Simulated Stationary Data (AR(1) with ρ=0.6)")
print("=" * 70)

np.random.seed(42)
n_firms = 15
n_years = 40

data_list = []
for i in range(n_firms):
    # AR(1) process with ρ = 0.6 (stationary)
    rho = 0.6
    y = np.zeros(n_years)
    y[0] = np.random.normal(0, 1)
    for t in range(1, n_years):
        y[t] = rho * y[t - 1] + np.random.normal(0, 1)

    data_list.append(pd.DataFrame({"firm": i, "year": range(n_years), "y": y}))

stationary_data = pd.concat(data_list, ignore_index=True)

llc = pb.LLCTest(stationary_data, "y", "firm", "year", lags=1, trend="c")
result = llc.run()

print(result)
print(f"\nInterpretation: Since p-value = {result.pvalue:.4f} < 0.05,")
print("we reject the null hypothesis of unit root.")
print("The data appears to be stationary.")


# Example 3: Simulated unit root data
print("\n" + "=" * 70)
print("Example 3: Simulated Unit Root Data (Random Walk)")
print("=" * 70)

np.random.seed(123)
data_list = []
for i in range(n_firms):
    # Random walk (unit root)
    y = np.cumsum(np.random.normal(0, 1, n_years))

    data_list.append(pd.DataFrame({"firm": i, "year": range(n_years), "y": y}))

unit_root_data = pd.concat(data_list, ignore_index=True)

llc = pb.LLCTest(unit_root_data, "y", "firm", "year", lags=1, trend="c")
result = llc.run()

print(result)
print(f"\nInterpretation: P-value = {result.pvalue:.4f}.")
print("If p-value > 0.05, we fail to reject the null hypothesis.")
print("The data appears to have a unit root (non-stationary).")


# Example 4: Comparing different trend specifications
print("\n" + "=" * 70)
print("Example 4: Effect of Trend Specification")
print("=" * 70)

data = pb.load_grunfeld()

print("\nTesting 'invest' with different specifications:")
print(f"{'-'*70}")

trends = [("n", "No deterministic terms"), ("c", "Constant only"), ("ct", "Constant and trend")]

for trend_code, trend_desc in trends:
    llc = pb.LLCTest(data, "invest", "firm", "year", lags=1, trend=trend_code)
    result = llc.run()

    print(f"\n{trend_desc}:")
    print(f"  Statistic: {result.statistic:8.4f}")
    print(f"  P-value:   {result.pvalue:8.4f}")
    print(f"  Conclusion: {result.conclusion}")


# Example 5: Automatic lag selection
print("\n" + "=" * 70)
print("Example 5: Automatic Lag Selection")
print("=" * 70)

data = pb.load_grunfeld()

print("\nLet the test automatically select the number of lags:")

llc = pb.LLCTest(data, "value", "firm", "year", lags=None, trend="c")
result = llc.run()

print(f"\nAutomatically selected {result.lags} lag(s)")
print(f"Statistic: {result.statistic:.4f}")
print(f"P-value: {result.pvalue:.4f}")


print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(
    """
The LLC (Levin-Lin-Chu) test is a panel unit root test that assumes a common
unit root process across all panels.

Key points:
- H0: All panels have unit roots (non-stationary)
- H1: All panels are stationary
- Test statistic follows standard normal distribution
- Reject H0 if p-value < α (typically 0.05)
- Requires balanced panel (or will warn)
- Choose trend specification based on data characteristics:
  * 'n': No trend or constant (rarely used)
  * 'c': Constant (most common)
  * 'ct': Constant and trend (if data has clear trend)

For heterogeneous panels (different unit root processes across panels),
consider using the IPS (Im-Pesaran-Shin) test instead.
"""
)

print("\n✓ Examples completed successfully!")
