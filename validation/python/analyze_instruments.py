"""
Analyze which instruments have >90% NaN and understand the pattern
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

# Transform and generate instruments
y_diff, X_diff, ids, times = model._transform_data()
model.instrument_builder = model.instrument_builder
Z = model._generate_instruments()

print("="*80)
print("INSTRUMENT NaN ANALYSIS")
print("="*80)
print(f"Total instruments: {Z.n_instruments}")
print(f"Total observations: {Z.Z.shape[0]}")
print()

# Analyze each instrument
Z_matrix = Z.Z.copy()
nan_fractions = np.isnan(Z_matrix).mean(axis=0)

print(f"{'Index':<6} {'Name':<25} {'NaN %':<10} {'Valid':<10} {'Status':<10}")
print("-"*80)

removed_count = 0
kept_count = 0

for i, (name, nan_frac) in enumerate(zip(Z.instrument_names, nan_fractions)):
    n_valid = (~np.isnan(Z_matrix[:, i])).sum()
    status = "REMOVED" if nan_frac >= 0.9 else "KEPT"

    if status == "REMOVED":
        removed_count += 1
    else:
        kept_count += 1

    print(f"{i:<6} {name:<25} {nan_frac*100:>7.2f}% {n_valid:>9} {status:<10}")

print("-"*80)
print(f"Kept: {kept_count}")
print(f"Removed: {removed_count}")
print()

# Analyze by instrument type
print("="*80)
print("BY INSTRUMENT TYPE")
print("="*80)

gmm_instruments = [i for i, name in enumerate(Z.instrument_names) if name.startswith('GMM')]
iv_instruments = [i for i, name in enumerate(Z.instrument_names) if not name.startswith('GMM')]

print(f"GMM instruments (L.n):")
print(f"  Total: {len(gmm_instruments)}")
gmm_removed = sum(1 for i in gmm_instruments if nan_fractions[i] >= 0.9)
print(f"  Removed: {gmm_removed}")
print(f"  Kept: {len(gmm_instruments) - gmm_removed}")
print()

print(f"IV instruments (w, k):")
print(f"  Total: {len(iv_instruments)}")
iv_removed = sum(1 for i in iv_instruments if nan_fractions[i] >= 0.9)
print(f"  Removed: {iv_removed}")
print(f"  Kept: {len(iv_instruments) - iv_removed}")
print()

# Show which GMM instruments are kept vs removed
print("="*80)
print("GMM INSTRUMENT DETAIL")
print("="*80)
for i in gmm_instruments:
    name = Z.instrument_names[i]
    nan_frac = nan_fractions[i]
    n_valid = (~np.isnan(Z_matrix[:, i])).sum()
    status = "REMOVED" if nan_frac >= 0.9 else "KEPT"
    print(f"{name:<25} {nan_frac*100:>7.2f}% NaN, {n_valid:>4} valid - {status}")
