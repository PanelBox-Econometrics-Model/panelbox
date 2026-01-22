"""
Check which observations have valid GMM instruments
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

print("="*80)
print("INSTRUMENT VALIDITY CHECK")
print("="*80)

# Remove all-NaN columns
not_all_nan = ~np.isnan(Z_matrix).all(axis=0)
Z_matrix = Z_matrix[:, not_all_nan]
print(f"Instruments after removing all-NaN columns: {Z_matrix.shape[1]}")
print()

# Identify GMM vs IV instruments
instrument_names = [name for i, name in enumerate(Z.instrument_names) if not_all_nan[i]]
gmm_cols = [i for i, name in enumerate(instrument_names) if name.startswith('n_t')]
iv_cols = [i for i, name in enumerate(instrument_names) if not name.startswith('n_t')]

print(f"GMM instruments (L.n): {len(gmm_cols)}")
print(f"IV instruments (w, k): {len(iv_cols)}")
print()

# Check how many valid GMM instruments each observation has
Z_gmm = Z_matrix[:, gmm_cols]
n_valid_gmm = (~np.isnan(Z_gmm)).sum(axis=1)

print("Valid GMM instruments per observation:")
print(f"  Min: {n_valid_gmm.min()}")
print(f"  Max: {n_valid_gmm.max()}")
print(f"  Mean: {n_valid_gmm.mean():.2f}")
print()

# Count observations by number of valid GMM instruments
print("Distribution of valid GMM instruments:")
for n in range(int(n_valid_gmm.max()) + 1):
    count = (n_valid_gmm == n).sum()
    if count > 0:
        print(f"  {n} valid GMM instruments: {count} observations")
print()

# How many observations have 0 valid GMM instruments?
n_zero_gmm = (n_valid_gmm == 0).sum()
print(f"Observations with ZERO valid GMM instruments: {n_zero_gmm}")
print(f"Observations with at least 1 valid GMM instrument: {(n_valid_gmm > 0).sum()}")
print()

# If we drop observations with 0 valid GMM instruments, what do we get?
valid_mask_gmm = n_valid_gmm > 0
valid_mask_y = ~np.isnan(y_diff.flatten())
valid_mask_x = ~np.isnan(X_diff).any(axis=1)

combined_mask = valid_mask_gmm & valid_mask_y & valid_mask_x
print("="*80)
print("IF WE REQUIRE AT LEAST 1 VALID GMM INSTRUMENT")
print("="*80)
print(f"Total observations: {combined_mask.sum()}")
print()

# Check by year
years_kept = times[combined_mask]
year_counts = pd.Series(years_kept).value_counts().sort_index()
print("Observations per year:")
total = 0
for year, count in year_counts.items():
    print(f"  {year:.0f}: {count}")
    total += count
print(f"Total: {total}")
print()
print(f"Stata expected: 611")
print(f"Difference: {total - 611}")
