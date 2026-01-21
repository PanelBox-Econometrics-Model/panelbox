"""
Debug the valid_mask logic step by step
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from panelbox.gmm import DifferenceGMM

# Load data
df = pd.read_csv('../data/abdata.csv')
df = df.drop(columns=['c1'], errors='ignore')
df['id'] = df['id'].astype(int)
df['year'] = df['year'].astype(int)
df = df.sort_values(['id', 'year'])

for col in df.columns:
    if col not in ['id', 'year']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("Creating GMM object...")
gmm = DifferenceGMM(
    data=df,
    dep_var='n',
    lags=1,
    id_var='id',
    time_var='year',
    exog_vars=['w', 'k', 'ys'],
    time_dummies=True,
    collapse=False,
    two_step=False,
    robust=True,
    gmm_type='one_step'
)

# Manually transform data to see what happens
print("\nManually transforming data...")
y_diff, X_diff, ids, times = gmm._transform_data()

print(f"After transformation:")
print(f"  y_diff shape: {y_diff.shape}")
print(f"  X_diff shape: {X_diff.shape}")
print(f"  y_diff non-NaN: {(~np.isnan(y_diff)).sum()}")
print(f"  X_diff non-NaN rows: {(~np.isnan(X_diff).any(axis=1)).sum()}")

# Check instruments
print("\nGenerating instruments...")
Z = gmm._generate_instruments()

print(f"  Z shape: {Z.Z.shape}")
print(f"  Z NaN count: {np.isnan(Z.Z).sum()} / {Z.Z.size} ({100*np.isnan(Z.Z).sum()/Z.Z.size:.1f}%)")
print(f"  Z rows with no NaN: {(~np.isnan(Z.Z).any(axis=1)).sum()}")
print(f"  Z rows with at least 50% valid: {((~np.isnan(Z.Z)).sum(axis=1) >= Z.Z.shape[1]//2).sum()}")

# Check y, X, Z alignment
print("\nAlignment check:")
print(f"  y_diff valid: {(~np.isnan(y_diff)).sum()}")
print(f"  X_diff rows valid: {(~np.isnan(X_diff).any(axis=1)).sum()}")
print(f"  y AND X valid: {((~np.isnan(y_diff)) & (~np.isnan(X_diff).any(axis=1))).sum()}")

# Now simulate the valid_mask logic
print("\nSimulating valid_mask logic:")
y_valid = ~np.isnan(y_diff)
X_valid = ~np.isnan(X_diff).any(axis=1)

print(f"  y_valid: {y_valid.sum()}")
print(f"  X_valid: {X_valid.sum()}")

# Check Z per observation
Z_notnan = ~np.isnan(Z.Z)
n_valid_instruments = Z_notnan.sum(axis=1)

print(f"  Instruments per observation (min/mean/max): {n_valid_instruments.min()}/{n_valid_instruments.mean():.1f}/{n_valid_instruments.max()}")

k = X_diff.shape[1]
n_instruments_total = Z.Z.shape[1]
min_instruments = max(k + 1, n_instruments_total // 2)

print(f"  Required minimum instruments: {min_instruments} (k={k}, total={n_instruments_total})")

Z_valid = n_valid_instruments >= min_instruments
print(f"  Z_valid (enough instruments): {Z_valid.sum()}")

final_valid = y_valid & X_valid & Z_valid
print(f"  Final valid mask: {final_valid.sum()}")

if final_valid.sum() > 0:
    print("\n✓ We have valid observations!")
else:
    print("\n✗ No valid observations - debugging further...")
    print(f"\nReasons observations are invalid:")
    print(f"  Missing y: {(~y_valid).sum()}")
    print(f"  Missing X: {(~X_valid).sum()}")
    print(f"  Insufficient instruments: {(~Z_valid).sum()}")
    print(f"    (need >= {min_instruments} but distribution is:)")

    for threshold in range(0, n_instruments_total + 1, 5):
        n_obs = (n_valid_instruments >= threshold).sum()
        print(f"      >= {threshold} instruments: {n_obs} obs")
