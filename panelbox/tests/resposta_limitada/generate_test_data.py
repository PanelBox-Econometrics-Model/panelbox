"""
Generate synthetic test data for R validation.

Creates three datasets:
1. binary_panel_test.csv - For Logit/Probit models
2. censored_panel_test.csv - For Tobit models
3. count_panel_test.csv - For Poisson/NegBin models
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Create data directory if it doesn't exist
data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

print("Generating test datasets for R validation...")

# ============================================
# 1. Binary Panel Data (Logit/Probit)
# ============================================
print("\n1. Generating binary panel data...")

n_entities = 100
n_time = 10
n_obs = n_entities * n_time

# Create panel structure
data_binary = pd.DataFrame(
    {
        "entity": np.repeat(np.arange(1, n_entities + 1), n_time),
        "time": np.tile(np.arange(1, n_time + 1), n_entities),
    }
)

# Generate covariates
data_binary["x1"] = np.random.randn(n_obs)
data_binary["x2"] = np.random.randn(n_obs)

# Generate binary outcome with Logit link
beta_0 = -0.5
beta_1 = 0.8
beta_2 = -0.6

X = data_binary[["x1", "x2"]].values
xb = beta_0 + X[:, 0] * beta_1 + X[:, 1] * beta_2
prob = 1 / (1 + np.exp(-xb))
data_binary["y"] = np.random.binomial(1, prob)

# Add some entity-level variation for FE models
# Make sure some entities have variation in y
entity_means = data_binary.groupby("entity")["y"].transform("mean")
# Ensure we have entities with variation (not all 0 or all 1)
data_binary.loc[entity_means == 0, "y"] = np.random.binomial(1, 0.3, size=(entity_means == 0).sum())
data_binary.loc[entity_means == 1, "y"] = np.random.binomial(1, 0.7, size=(entity_means == 1).sum())

output_path = data_dir / "binary_panel_test.csv"
data_binary.to_csv(output_path, index=False)
print(f"   Saved to: {output_path}")
print(f"   Shape: {data_binary.shape}")
print(f"   Mean of y: {data_binary['y'].mean():.3f}")
print(f"   Entities with variation: {(data_binary.groupby('entity')['y'].std() > 0).sum()}")

# ============================================
# 2. Censored Panel Data (Tobit)
# ============================================
print("\n2. Generating censored panel data...")

data_censored = pd.DataFrame(
    {
        "entity": np.repeat(np.arange(1, n_entities + 1), n_time),
        "time": np.tile(np.arange(1, n_time + 1), n_entities),
    }
)

# Generate covariates
data_censored["x1"] = np.random.randn(n_obs)
data_censored["x2"] = np.random.randn(n_obs)

# Generate latent variable
beta_0 = 2.0
beta_1 = 1.5
beta_2 = -1.0
sigma = 2.0

X = data_censored[["x1", "x2"]].values
xb = beta_0 + X[:, 0] * beta_1 + X[:, 1] * beta_2
y_star = xb + np.random.randn(n_obs) * sigma

# Apply left-censoring at 0
data_censored["y"] = np.maximum(0, y_star)

output_path = data_dir / "censored_panel_test.csv"
data_censored.to_csv(output_path, index=False)
print(f"   Saved to: {output_path}")
print(f"   Shape: {data_censored.shape}")
print(f"   Mean of y: {data_censored['y'].mean():.3f}")
print(
    f"   Censored observations: {(data_censored['y'] == 0).sum()} ({(data_censored['y'] == 0).mean()*100:.1f}%)"
)

# ============================================
# 3. Count Panel Data (Poisson/NegBin)
# ============================================
print("\n3. Generating count panel data...")

data_count = pd.DataFrame(
    {
        "entity": np.repeat(np.arange(1, n_entities + 1), n_time),
        "time": np.tile(np.arange(1, n_time + 1), n_entities),
    }
)

# Generate covariates
data_count["x1"] = np.random.randn(n_obs)
data_count["x2"] = np.random.randn(n_obs)

# Generate count outcome with Poisson
beta_0 = 1.0
beta_1 = 0.5
beta_2 = -0.3

X = data_count[["x1", "x2"]].values
xb = beta_0 + X[:, 0] * beta_1 + X[:, 1] * beta_2
lambda_poisson = np.exp(xb)

# For Negative Binomial, we'll add overdispersion
# Using NegBin parametrization: Var = mu + alpha * mu^2
alpha = 0.5  # Overdispersion parameter
# Generate from NegBin using Gamma-Poisson mixture
r = 1 / alpha  # Shape parameter
gamma_variates = np.random.gamma(r, lambda_poisson / r)
data_count["y"] = np.random.poisson(gamma_variates)

output_path = data_dir / "count_panel_test.csv"
data_count.to_csv(output_path, index=False)
print(f"   Saved to: {output_path}")
print(f"   Shape: {data_count.shape}")
print(f"   Mean of y: {data_count['y'].mean():.3f}")
print(f"   Variance of y: {data_count['y'].var():.3f}")
print(f"   Variance/Mean ratio: {data_count['y'].var() / data_count['y'].mean():.3f}")
print(f"   (>1 indicates overdispersion, suitable for NegBin)")

# ============================================
# Summary Statistics
# ============================================
print("\n" + "=" * 60)
print("DATA GENERATION COMPLETE")
print("=" * 60)

print("\nSummary:")
print(f"  - Binary data: {data_dir / 'binary_panel_test.csv'}")
print(f"  - Censored data: {data_dir / 'censored_panel_test.csv'}")
print(f"  - Count data: {data_dir / 'count_panel_test.csv'}")

print("\nPanel structure:")
print(f"  - Entities (N): {n_entities}")
print(f"  - Time periods (T): {n_time}")
print(f"  - Total observations: {n_obs}")

print("\nNext steps:")
print("  1. Run R benchmarks:")
print("     cd r && Rscript benchmark_discrete.R")
print("     cd r && Rscript benchmark_tobit.R")
print("     cd r && Rscript benchmark_count.R")
print("  2. Run Python validation:")
print("     pytest test_r_validation.py -v")
