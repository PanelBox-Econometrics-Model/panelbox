#!/usr/bin/env python
"""Simple test for Poisson Fixed Effects."""

import sys
from pathlib import Path

import numpy as np

# Add panelbox to path
sys.path.insert(0, str(Path(__file__).parent))

from panelbox.models.count.poisson import PoissonFixedEffects

# Generate simple test data
np.random.seed(42)

# Panel structure
n_entities = 10
n_periods = 5
n_obs = n_entities * n_periods

# Generate data
entity_id = np.repeat(np.arange(n_entities), n_periods)
time_id = np.tile(np.arange(n_periods), n_entities)

# Covariates (no intercept for FE)
X = np.random.randn(n_obs, 2)

# True parameters
beta_true = np.array([0.5, -0.3])

# Generate count outcomes
eta = X @ beta_true
lambda_true = np.exp(eta)
y = np.random.poisson(lambda_true)

print("Data generated:")
print(f"  Entities: {n_entities}")
print(f"  Periods: {n_periods}")
print(f"  Observations: {n_obs}")
print(f"  True beta: {beta_true}")

# Create model
print("\nCreating model...")
model = PoissonFixedEffects(y, X, entity_id, time_id)

print(f"  Kept entities: {len(model.kept_entities)}")
print(f"  Dropped entities: {len(model.dropped_entities)}")

# Fit model
print("\nFitting model...")
try:
    result = model.fit()
    print(f"  Estimated beta: {result.params}")
    print(f"  Standard errors: {result.se}")
    print(f"  Log-likelihood: {model.llf:.4f}")
    print("\n✓ SUCCESS: Model fitted successfully!")
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
