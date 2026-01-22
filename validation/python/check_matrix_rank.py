"""
Check rank of instrument matrix to diagnose singular matrix issues
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from panelbox.gmm import DifferenceGMM

# Load data
DATA_PATH = Path(__file__).parent.parent / "data" / "abdata.csv"
df = pd.read_csv(DATA_PATH)

# Create model
model = DifferenceGMM(
    data=df,
    dep_var='n',
    lags=1,
    id_var='id',
    time_var='year',
    exog_vars=['w', 'k'],
    collapse=False,
    two_step=True,
    robust=True,
    time_dummies=False
)

# Transform data
y_diff, X_diff, ids, times = model._transform_data()
model.instrument_builder = model.instrument_builder

# Generate instruments
Z = model._generate_instruments()
Z_matrix = Z.Z.copy()

# Remove all-NaN columns
not_all_nan = ~np.isnan(Z_matrix).all(axis=0)
Z_matrix = Z_matrix[:, not_all_nan]

# Filter by GMM instrument availability (≥2 valid GMM instruments)
instrument_names = [name for i, name in enumerate(Z.instrument_names) if not_all_nan[i]]
gmm_cols = [i for i, name in enumerate(instrument_names) if name.startswith('n_t')]
Z_gmm = Z_matrix[:, gmm_cols]
n_valid_gmm = (~np.isnan(Z_gmm)).sum(axis=1)
obs_valid_mask = n_valid_gmm >= 2

# Filter
y_diff = y_diff[obs_valid_mask]
X_diff = X_diff[obs_valid_mask]
Z_matrix = Z_matrix[obs_valid_mask]

print("="*80)
print("MATRIX RANK ANALYSIS")
print("="*80)
print(f"Z shape after filtering: {Z_matrix.shape}")
print(f"Number of observations: {Z_matrix.shape[0]}")
print(f"Number of instruments: {Z_matrix.shape[1]}")
print()

# Check NaN distribution
nan_fraction = np.isnan(Z_matrix).mean()
print(f"NaN fraction in Z: {nan_fraction*100:.2f}%")
print(f"Non-NaN entries: {(~np.isnan(Z_matrix)).sum()}/{Z_matrix.size}")
print()

# Replace NaN with 0
Z_clean = np.nan_to_num(Z_matrix, nan=0.0)

# Check rank
rank_Z = np.linalg.matrix_rank(Z_clean)
print(f"Rank of Z (with NaN→0): {rank_Z}")
print(f"Expected max rank: min(n_obs, n_instruments) = {min(Z_clean.shape)}")
print(f"Rank deficiency: {min(Z_clean.shape) - rank_Z}")
print()

# Check Z'Z
ZtZ = Z_clean.T @ Z_clean
rank_ZtZ = np.linalg.matrix_rank(ZtZ)
print(f"Rank of Z'Z: {rank_ZtZ}")
print(f"Expected rank: {Z_clean.shape[1]}")
print(f"Rank deficiency in Z'Z: {Z_clean.shape[1] - rank_ZtZ}")
print()

# Check eigenvalues of Z'Z to see condition number
eigvals = np.linalg.eigvalsh(ZtZ)
eigvals = eigvals[eigvals > 1e-10]  # Filter near-zero
if len(eigvals) > 0:
    condition_number = eigvals.max() / eigvals.min()
    print(f"Condition number of Z'Z: {condition_number:.2e}")
    print(f"Min eigenvalue: {eigvals.min():.2e}")
    print(f"Max eigenvalue: {eigvals.max():.2e}")
    print()

# Check sparsity
zero_fraction = (Z_clean == 0).mean()
print(f"Fraction of zeros in Z: {zero_fraction*100:.2f}%")
print(f"Sparsity: {'VERY SPARSE' if zero_fraction > 0.9 else 'SPARSE' if zero_fraction > 0.5 else 'DENSE'}")
print()

# Check column correlation (sample of GMM columns)
if len(gmm_cols) > 0:
    print("="*80)
    print("CHECKING FOR COLLINEARITY IN GMM INSTRUMENTS")
    print("="*80)
    Z_gmm_clean = Z_clean[:, gmm_cols]

    # Compute correlation matrix
    corr = np.corrcoef(Z_gmm_clean.T)

    # Find highly correlated pairs
    high_corr = []
    for i in range(len(gmm_cols)):
        for j in range(i+1, len(gmm_cols)):
            if abs(corr[i, j]) > 0.99:
                high_corr.append((i, j, corr[i, j]))

    if high_corr:
        print(f"Found {len(high_corr)} pairs of highly correlated GMM instruments (|r| > 0.99):")
        for i, j, r in high_corr[:10]:  # Show first 10
            print(f"  {instrument_names[gmm_cols[i]]} <-> {instrument_names[gmm_cols[j]]}: r={r:.4f}")
    else:
        print("No highly correlated GMM instrument pairs found (|r| > 0.99)")
