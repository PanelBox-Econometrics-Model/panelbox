"""
Example: Granger Causality Analysis in Panel VAR

This example demonstrates how to:
1. Estimate a Panel VAR model
2. Perform Granger causality tests (Wald)
3. Construct Granger causality matrix
4. Test instantaneous causality
5. Interpret results
"""

import numpy as np
import pandas as pd

from panelbox.var import PanelVAR, PanelVARData

# ==============================================================================
# 1. Generate simulated data with known causality structure
# ==============================================================================


def generate_var_data_with_causality(N=40, T=100, seed=42):
    """
    Generate Panel VAR(1) data with known Granger causality structure.

    Structure:
    - GDP → Inflation (strong causal effect: 0.4)
    - Inflation → Interest Rate (moderate effect: 0.3)
    - Interest Rate → GDP (weak effect: -0.15)
    - No causality: Interest Rate ↛ Inflation
    """
    np.random.seed(seed)

    data_list = []

    for i in range(N):
        # Initialize
        gdp = np.zeros(T)
        inf = np.zeros(T)
        int_rate = np.zeros(T)

        gdp[0] = np.random.normal(100, 5)
        inf[0] = np.random.normal(3, 1)
        int_rate[0] = np.random.normal(4, 0.5)

        # VAR(1) dynamics with specific causality structure
        for t in range(1, T):
            # GDP equation: autoreg + interest rate effect (int → gdp)
            gdp[t] = (
                0.5 * gdp[t - 1]
                - 0.15 * int_rate[t - 1]  # CAUSALITY: int → gdp
                + np.random.normal(0, 1)
            )

            # Inflation equation: autoreg + gdp effect (gdp → inf)
            inf[t] = (
                0.4 * inf[t - 1]
                + 0.4 * gdp[t - 1]  # CAUSALITY: gdp → inf
                + np.random.normal(0, 0.3)
            )

            # Interest rate equation: autoreg + inflation effect (inf → int)
            int_rate[t] = (
                0.6 * int_rate[t - 1]
                + 0.3 * inf[t - 1]  # CAUSALITY: inf → int
                + np.random.normal(0, 0.2)
            )

        # Create DataFrame
        entity_df = pd.DataFrame(
            {
                "country": i,
                "year": np.arange(T),
                "gdp": gdp,
                "inflation": inf,
                "interest_rate": int_rate,
            }
        )

        data_list.append(entity_df)

    return pd.concat(data_list, ignore_index=True)


# ==============================================================================
# 2. Estimate Panel VAR
# ==============================================================================

print("=" * 80)
print("GRANGER CAUSALITY ANALYSIS IN PANEL VAR")
print("=" * 80)
print()

# Generate data
df = generate_var_data_with_causality(N=40, T=100)

print(
    f"Dataset: {df['country'].nunique()} countries, {df.groupby('country').size().iloc[0]} time periods"
)
print(f"Variables: {['gdp', 'inflation', 'interest_rate']}")
print()

# Prepare data
data = PanelVARData(
    df,
    endog_vars=["gdp", "inflation", "interest_rate"],
    entity_col="country",
    time_col="year",
    lags=1,
)

# Estimate model
model = PanelVAR(data)
result = model.fit(method="fe", cov_type="robust")

print("Model estimated successfully!")
print(f"Stable: {result.is_stable()}")
print()

# ==============================================================================
# 3. Individual Granger Causality Tests
# ==============================================================================

print("=" * 80)
print("INDIVIDUAL GRANGER CAUSALITY TESTS")
print("=" * 80)
print()

# Test: GDP → Inflation
print("Test 1: Does GDP Granger-cause Inflation?")
print("-" * 80)
gc_gdp_inf = result.granger_causality("gdp", "inflation")
print(gc_gdp_inf.summary())
print()

# Test: Inflation → Interest Rate
print("Test 2: Does Inflation Granger-cause Interest Rate?")
print("-" * 80)
gc_inf_int = result.granger_causality("inflation", "interest_rate")
print(gc_inf_int.summary())
print()

# Test: Interest Rate → GDP
print("Test 3: Does Interest Rate Granger-cause GDP?")
print("-" * 80)
gc_int_gdp = result.granger_causality("interest_rate", "gdp")
print(gc_int_gdp.summary())
print()

# Test: Interest Rate → Inflation (should NOT find causality)
print("Test 4: Does Interest Rate Granger-cause Inflation? (Expect: No)")
print("-" * 80)
gc_int_inf = result.granger_causality("interest_rate", "inflation")
print(gc_int_inf.summary())
print()

# ==============================================================================
# 4. Granger Causality Matrix
# ==============================================================================

print("=" * 80)
print("GRANGER CAUSALITY MATRIX (P-VALUES)")
print("=" * 80)
print()
print("Row variable Granger-causes Column variable")
print("(Lower p-value = stronger evidence of causality)")
print()

gc_matrix = result.granger_causality_matrix()
print(gc_matrix.round(4))
print()


# Interpretation helper
def interpret_causality_matrix(matrix, alpha=0.05):
    """Interpret causality matrix with significance marking."""
    print("Significant causalities (p < 0.05):")
    for i, cause in enumerate(matrix.index):
        for j, effect in enumerate(matrix.columns):
            if i != j:  # Skip diagonal
                p_value = matrix.iloc[i, j]
                if not pd.isna(p_value) and p_value < alpha:
                    stars = "***" if p_value < 0.01 else ("**" if p_value < 0.05 else "*")
                    print(f"  {cause} → {effect}: p = {p_value:.4f} {stars}")


interpret_causality_matrix(gc_matrix)
print()

# ==============================================================================
# 5. Instantaneous Causality Tests
# ==============================================================================

print("=" * 80)
print("INSTANTANEOUS CAUSALITY TESTS")
print("=" * 80)
print()
print("Tests for contemporaneous correlation between residuals")
print()

# Test GDP ↔ Inflation
print("Test: GDP ↔ Inflation (contemporaneous correlation)")
print("-" * 80)
ic_gdp_inf = result.instantaneous_causality("gdp", "inflation")
print(ic_gdp_inf.summary())
print()

# Instantaneous causality matrix
print("Instantaneous Causality: Correlation Matrix")
print("-" * 80)
corr_matrix, pvalue_matrix = result.instantaneous_causality_matrix()

print("\nCorrelation coefficients:")
print(corr_matrix.round(4))
print()

print("P-values:")
print(pvalue_matrix.round(4))
print()

# ==============================================================================
# 6. Summary and Recommendations
# ==============================================================================

print("=" * 80)
print("SUMMARY AND INTERPRETATION")
print("=" * 80)
print()

print("Key Findings:")
print()

# Expected causalities
expected = [
    ("gdp", "inflation", "strong"),
    ("inflation", "interest_rate", "moderate"),
    ("interest_rate", "gdp", "weak"),
]

for cause, effect, strength in expected:
    p_val = gc_matrix.loc[cause, effect]
    detected = "✓" if p_val < 0.05 else "✗"
    print(f"  {detected} {cause} → {effect} ({strength} effect expected)")
    print(f"      p-value = {p_val:.4f}")
    print()

print("Notes:")
print("  - Granger causality ≠ true causation")
print("  - Measures predictive power, not structural relationships")
print("  - Instantaneous causality may indicate omitted variables or simultaneity")
print()

# ==============================================================================
# 7. Visualization Recommendations
# ==============================================================================

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()

print("To visualize these results, you can:")
print("  1. Create a network graph of significant causalities")
print("  2. Plot heatmap of p-values")
print("  3. Examine impulse response functions (IRFs) to quantify dynamic effects")
print("  4. Perform forecast error variance decomposition (FEVD)")
print()

print("For more robust inference with small samples:")
print("  - Use bootstrap Granger causality tests (coming in next phase)")
print("  - Try Dumitrescu-Hurlin test for heterogeneous panels (next phase)")
print()

# ==============================================================================
# 8. Advanced: Conditional Causality (manual example)
# ==============================================================================

print("=" * 80)
print("ADVANCED: Testing Conditional Causality")
print("=" * 80)
print()

print("Question: Does GDP cause Interest Rate, controlling for Inflation?")
print()
print("Approach:")
print("  1. Estimate reduced model: Interest Rate ~ lags(Interest Rate, Inflation)")
print("  2. Estimate full model: Interest Rate ~ lags(Interest Rate, Inflation, GDP)")
print("  3. F-test for joint significance of GDP lags")
print()
print("This is equivalent to Granger test in the full system.")
print("Result from full system:")

p_val = gc_matrix.loc["gdp", "interest_rate"]
print(f"  GDP → Interest Rate: p = {p_val:.4f}")

if p_val < 0.05:
    print("  Conclusion: GDP Granger-causes Interest Rate, even controlling for Inflation")
else:
    print("  Conclusion: No evidence of GDP → Interest Rate causality")

print()
print("=" * 80)
print("END OF ANALYSIS")
print("=" * 80)
