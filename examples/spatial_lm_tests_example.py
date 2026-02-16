"""
Example: Using LM Tests for Spatial Dependence in Panel Data

This example demonstrates how to use the LM (Lagrange Multiplier) tests
to diagnose spatial dependence in panel data models.
"""

import numpy as np
import pandas as pd
import patsy
from statsmodels.regression.linear_model import OLS

from panelbox.diagnostics.spatial_tests import run_lm_tests

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("Example: LM Tests for Spatial Dependence")
print("=" * 70)

# ==============================================================================
# Step 1: Generate Spatial Panel Data with Spatial Lag Structure
# ==============================================================================

print("\n1. Generating spatial panel data...")

n_entities = 20  # Number of spatial units
n_time = 5  # Number of time periods
n_obs = n_entities * n_time

# Create a simple spatial weight matrix (rook contiguity in a 4x5 grid)
nrows, ncols = 4, 5
W = np.zeros((n_entities, n_entities))

for i in range(n_entities):
    row_i = i // ncols
    col_i = i % ncols

    # Add neighbors (up, down, left, right)
    neighbors = []
    if row_i > 0:  # Up
        neighbors.append((row_i - 1) * ncols + col_i)
    if row_i < nrows - 1:  # Down
        neighbors.append((row_i + 1) * ncols + col_i)
    if col_i > 0:  # Left
        neighbors.append(row_i * ncols + (col_i - 1))
    if col_i < ncols - 1:  # Right
        neighbors.append(row_i * ncols + (col_i + 1))

    for neighbor in neighbors:
        W[i, neighbor] = 1

# Row-normalize W
row_sums = W.sum(axis=1)
W = W / row_sums[:, np.newaxis]

print(f"   - Number of spatial units: {n_entities}")
print(f"   - Number of time periods: {n_time}")
print(f"   - Total observations: {n_obs}")
print(f"   - Spatial weight matrix shape: {W.shape}")

# Generate data with spatial lag dependence
# y = rho * W * y + X * beta + epsilon

# True parameters
rho_true = 0.6  # Spatial lag parameter (strong spatial dependence)
beta_true = np.array([2.0, 1.5, -0.8])  # Intercept, x1, x2

# Generate X variables
entity_ids = np.repeat(np.arange(1, n_entities + 1), n_time)
time_ids = np.tile(np.arange(1, n_time + 1), n_entities)

x1 = np.random.normal(5, 2, n_obs)
x2 = np.random.normal(10, 3, n_obs)

X_full = np.column_stack([np.ones(n_obs), x1, x2])

# Generate y with spatial lag for each time period
y = np.zeros(n_obs)

for t in range(n_time):
    idx_start = t * n_entities
    idx_end = (t + 1) * n_entities

    X_t = X_full[idx_start:idx_end, :]
    epsilon_t = np.random.normal(0, 1, n_entities)

    # Reduced form: y = (I - rho*W)^{-1} * (X*beta + epsilon)
    I_minus_rhoW = np.eye(n_entities) - rho_true * W
    X_beta = X_t @ beta_true
    y[idx_start:idx_end] = np.linalg.solve(I_minus_rhoW, X_beta + epsilon_t)

# Create DataFrame
df = pd.DataFrame({"entity": entity_ids, "time": time_ids, "y": y, "x1": x1, "x2": x2})

print(f"\n   Data generated with true rho = {rho_true}")
print(f"   Data summary:\n{df.describe()}")

# ==============================================================================
# Step 2: Fit Non-Spatial Model (Pooled OLS)
# ==============================================================================

print("\n2. Fitting pooled OLS (ignoring spatial structure)...")

y_vec, X_mat = patsy.dmatrices("y ~ x1 + x2", data=df, return_type="dataframe")
ols_result = OLS(y_vec.values.flatten(), X_mat.values).fit()

print("\nOLS Results:")
print(ols_result.summary())

# ==============================================================================
# Step 3: Run LM Tests for Spatial Dependence
# ==============================================================================

print("\n3. Running LM tests for spatial dependence...")

lm_results = run_lm_tests(ols_result, W, alpha=0.05)

# Display individual test results
print("\n" + "=" * 70)
print("LM Test Results")
print("=" * 70)

for test_name in ["lm_lag", "lm_error", "robust_lm_lag", "robust_lm_error"]:
    result = lm_results[test_name]
    print(f"\n{result.summary()}")

# Display summary table
print("\n" + "=" * 70)
print("Summary Table")
print("=" * 70)
print(lm_results["summary"].to_string(index=False))

# Display recommendation
print("\n" + "=" * 70)
print("Model Recommendation")
print("=" * 70)
print(f"Recommended Model: {lm_results['recommendation']}")
print(f"Reason: {lm_results['reason']}")

# ==============================================================================
# Step 4: Interpretation
# ==============================================================================

print("\n" + "=" * 70)
print("Interpretation")
print("=" * 70)

print(
    """
Since we generated the data with a spatial lag structure (rho = 0.6),
we expect the LM-Lag test to be significant, indicating spatial lag dependence.

The decision tree works as follows:

1. If only LM-Lag is significant → Spatial Lag (SAR) model recommended
2. If only LM-Error is significant → Spatial Error (SEM) model recommended
3. If both are significant → Use robust tests to discriminate:
   - If Robust LM-Lag is significant → SAR
   - If Robust LM-Error is significant → SEM
   - If both robust tests are significant → SDM or GNS model
4. If neither is significant → No spatial dependence detected

In this case, the tests should detect the spatial lag dependence and
recommend a Spatial Lag (SAR) model.
"""
)

# ==============================================================================
# Example 2: Data with No Spatial Dependence
# ==============================================================================

print("\n" + "=" * 70)
print("Example 2: Data WITHOUT Spatial Dependence")
print("=" * 70)

# Generate data without spatial dependence
y_nospatial = X_full @ beta_true + np.random.normal(0, 1, n_obs)

df_nospatial = pd.DataFrame(
    {"entity": entity_ids, "time": time_ids, "y": y_nospatial, "x1": x1, "x2": x2}
)

# Fit OLS
y_vec2, X_mat2 = patsy.dmatrices("y ~ x1 + x2", data=df_nospatial, return_type="dataframe")
ols_result2 = OLS(y_vec2.values.flatten(), X_mat2.values).fit()

# Run LM tests
lm_results2 = run_lm_tests(ols_result2, W, alpha=0.05)

print("\nLM Test Results (No Spatial Dependence):")
print(lm_results2["summary"].to_string(index=False))

print(f"\nRecommended Model: {lm_results2['recommendation']}")
print(f"Reason: {lm_results2['reason']}")

print(
    """
Since this data has no spatial structure, the LM tests should NOT reject
the null hypothesis, and no spatial model is recommended.
"""
)

print("\n" + "=" * 70)
print("Example Complete!")
print("=" * 70)
