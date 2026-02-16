"""
Generate synthetic spatial panel data for testing LM tests.

This script creates a simple spatial panel dataset with known spatial structure
to validate the LM test implementations against R's splm package.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Parameters
n_entities = 30  # Number of spatial units
n_time = 10  # Number of time periods
n_obs = n_entities * n_time

# Create spatial weight matrix (row-normalized)
# Using a simple contiguity structure for reproducibility
W = np.zeros((n_entities, n_entities))

# Create a simple spatial structure (rook contiguity in a 5x6 grid)
nrows, ncols = 5, 6
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

# Generate data with spatial lag dependence
# y = rho * W * y + X * beta + epsilon
# For simplicity, we'll use the reduced form approach

# True parameters
rho_true = 0.5  # Spatial lag parameter
beta_true = np.array([1.0, 2.0, -1.5, 0.8])  # Intercept, x1, x2, x3

# Generate X variables
entity_ids = np.repeat(np.arange(1, n_entities + 1), n_time)
time_ids = np.tile(np.arange(1, n_time + 1), n_entities)

x1 = np.random.normal(5, 2, n_obs)
x2 = np.random.normal(10, 3, n_obs)
x3 = np.random.normal(0, 1, n_obs)

X_full = np.column_stack([np.ones(n_obs), x1, x2, x3])

# Generate y with spatial lag for each time period separately
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
df = pd.DataFrame({"entity": entity_ids, "time": time_ids, "y": y, "x1": x1, "x2": x2, "x3": x3})

# Save data
df.to_csv("spatial_test_data.csv", index=False)
print("Saved spatial_test_data.csv")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nSummary:\n{df.describe()}")

# Save spatial weights matrix (without row/column names for R compatibility)
np.savetxt("spatial_weights.csv", W, delimiter=",")
print("\nSaved spatial_weights.csv")
print(f"W shape: {W.shape}")
print(f"W row sums (should be 1): {W.sum(axis=1)[:5]}")

# Also save W with proper labels for debugging
W_df = pd.DataFrame(W)
W_df.to_csv("spatial_weights_labeled.csv", index=False)
print("Saved spatial_weights_labeled.csv (for debugging)")
