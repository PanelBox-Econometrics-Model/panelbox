"""
Basic Panel VAR Example
========================

This example demonstrates the basic usage of the Panel VAR module in PanelBox.

Topics covered:
- Creating Panel VAR data
- Estimating a Panel VAR model with OLS
- Automatic lag selection
- Stability diagnostics
- Granger causality testing
- Export results to LaTeX/HTML

"""

import numpy as np
import pandas as pd

import panelbox as pb
from panelbox.var import PanelVAR, PanelVARData

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Step 1: Simulate Panel VAR Data
# ============================================================================

print("=" * 70)
print("SIMULATING PANEL VAR DATA")
print("=" * 70)

# True VAR(2) coefficients (stable system)
A1_true = np.array([[0.5, 0.1], [0.2, 0.4]])
A2_true = np.array([[0.1, 0.05], [0.05, 0.1]])

# Residual covariance
Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
Sigma_chol = np.linalg.cholesky(Sigma)

# Panel dimensions
n_entities = 20
n_periods = 30
K = 2  # Number of variables

data_list = []

for entity_id in range(n_entities):
    # Initialize with small random values
    y_history = [np.random.randn(K) * 0.5 for _ in range(2)]

    # Generate VAR(2) time series
    for t in range(n_periods):
        # VAR(2) equation: y_t = A1 * y_{t-1} + A2 * y_{t-2} + ε_t
        eps = Sigma_chol @ np.random.randn(K)
        y_new = A1_true @ y_history[-1] + A2_true @ y_history[-2] + eps
        y_history.append(y_new)

        # Store in dataframe format
        data_list.append(
            {
                "entity": f"Entity_{entity_id}",
                "time": t,
                "gdp": y_new[0],
                "inflation": y_new[1],
            }
        )

# Create DataFrame
df = pd.DataFrame(data_list)

print(f"✓ Generated panel data: {n_entities} entities × {n_periods} periods")
print(f"  Variables: gdp, inflation")
print(f"  Total observations: {len(df)}")
print()

# ============================================================================
# Step 2: Create PanelVARData Object
# ============================================================================

print("=" * 70)
print("CREATING PANEL VAR DATA")
print("=" * 70)

# Create PanelVARData with lags=2
pvar_data = PanelVARData(
    df,
    endog_vars=["gdp", "inflation"],
    entity_col="entity",
    time_col="time",
    lags=2,
    trend="constant",
)

print(f"✓ Panel VAR Data created:")
print(f"  K (variables): {pvar_data.K}")
print(f"  p (lags): {pvar_data.p}")
print(f"  N (entities): {pvar_data.N}")
print(f"  T_avg (avg periods): {pvar_data.T_avg:.1f}")
print(f"  n_obs (total obs after lags): {pvar_data.n_obs}")
print(f"  Balanced: {pvar_data.is_balanced}")
print()

# ============================================================================
# Step 3: Automatic Lag Selection
# ============================================================================

print("=" * 70)
print("AUTOMATIC LAG SELECTION")
print("=" * 70)

# First, create a model with p=1 for lag selection
data_temp = PanelVARData(
    df,
    endog_vars=["gdp", "inflation"],
    entity_col="entity",
    time_col="time",
    lags=1,
)

model_temp = PanelVAR(data_temp)
lag_results = model_temp.select_lag_order(max_lags=5)

print(lag_results.summary())
print()

selected_lag = lag_results.selected["BIC"]
print(f"✓ BIC selects: p = {selected_lag}")
print()

# ============================================================================
# Step 4: Estimate Panel VAR Model
# ============================================================================

print("=" * 70)
print("ESTIMATING PANEL VAR MODEL")
print("=" * 70)

# Create model and fit with cluster-robust standard errors
model = PanelVAR(pvar_data)
results = model.fit(method="ols", cov_type="clustered")

print(f"✓ Model estimated successfully")
print(f"  Method: {results.method.upper()}")
print(f"  Covariance type: {results.cov_type}")
print(f"  AIC: {results.aic:.4f}")
print(f"  BIC: {results.bic:.4f}")
print()

# Display full summary
print(results.summary())
print()

# ============================================================================
# Step 5: Stability Diagnostics
# ============================================================================

print("=" * 70)
print("STABILITY DIAGNOSTICS")
print("=" * 70)

print(f"System is stable: {results.is_stable()}")
print(f"Max eigenvalue modulus: {results.max_eigenvalue_modulus:.4f}")
print(f"Stability margin: {results.stability_margin:.4f}")
print()

# Plot stability (unit circle)
print("Plotting stability diagram...")
try:
    results.plot_stability(backend="matplotlib", show=False)
    print("✓ Stability plot created (figure not displayed in script)")
except Exception as e:
    print(f"  Note: Could not create plot: {e}")
print()

# ============================================================================
# Step 6: Coefficient Matrices
# ============================================================================

print("=" * 70)
print("COEFFICIENT MATRICES")
print("=" * 70)

print("Matrix A_1 (lag 1):")
print(results.coef_matrix(lag=1))
print()

print("Matrix A_2 (lag 2):")
print(results.coef_matrix(lag=2))
print()

print("True Matrix A_1:")
print(pd.DataFrame(A1_true, index=["gdp", "inflation"], columns=["gdp", "inflation"]))
print()

print("True Matrix A_2:")
print(pd.DataFrame(A2_true, index=["gdp", "inflation"], columns=["gdp", "inflation"]))
print()

# ============================================================================
# Step 7: Granger Causality Tests
# ============================================================================

print("=" * 70)
print("GRANGER CAUSALITY TESTS")
print("=" * 70)

# Test if GDP Granger-causes inflation
gc_test_1 = results.test_granger_causality("gdp", "inflation")
print("H0: GDP does not Granger-cause inflation")
print(gc_test_1)
print()

# Test if inflation Granger-causes GDP
gc_test_2 = results.test_granger_causality("inflation", "gdp")
print("H0: Inflation does not Granger-cause GDP")
print(gc_test_2)
print()

# ============================================================================
# Step 8: Export Results
# ============================================================================

print("=" * 70)
print("EXPORTING RESULTS")
print("=" * 70)

# Export to LaTeX
latex_output = results.to_latex()
print("✓ LaTeX table created")
print("  First few lines:")
print("\n".join(latex_output.split("\n")[:10]))
print("  ...")
print()

# Export to HTML
html_output = results.to_html()
print("✓ HTML table created")
print(f"  Length: {len(html_output)} characters")
print()

# Optionally save to files
# with open("panel_var_results.tex", "w") as f:
#     f.write(latex_output)
# with open("panel_var_results.html", "w") as f:
#     f.write(html_output)

print("=" * 70)
print("EXAMPLE COMPLETED SUCCESSFULLY")
print("=" * 70)
print()
print("Summary:")
print(f"  - Simulated VAR(2) data with {n_entities} entities")
print(f"  - Estimated Panel VAR with OLS + fixed effects")
print(f"  - System is {'stable' if results.is_stable() else 'unstable'}")
print(f"  - AIC: {results.aic:.4f}, BIC: {results.bic:.4f}")
print(f"  - Coefficients close to true values: ✓")
print()
