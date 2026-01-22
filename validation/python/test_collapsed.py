"""
Test Difference GMM with collapsed instruments to check if this resolves
the numerical issues with singular matrices.
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

print("="*80)
print("TESTING WITH COLLAPSED INSTRUMENTS")
print("="*80)
print()

# Create model with collapse=True
model = DifferenceGMM(
    data=df,
    dep_var='n',
    lags=1,
    id_var='id',
    time_var='year',
    exog_vars=['w', 'k'],
    collapse=True,  # ← COLLAPSED
    two_step=True,
    robust=True,
    time_dummies=False
)

print("Fitting model with collapse=True...")
results = model.fit()

print("\n" + results.summary())

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"Collapsed instruments: {results.n_instruments}")
print(f"Observations: {results.nobs}")
print(f"Groups: {results.n_groups}")
print()
print("Coefficients:")
print(f"  L.n: {results.params['L1.n']:.6f}")
print(f"  w:   {results.params['w']:.6f}")
print(f"  k:   {results.params['k']:.6f}")
print()
print("Hansen J:")
print(f"  Statistic: {results.hansen_j.statistic:.3f}")
print(f"  P-value: {results.hansen_j.pvalue:.4f}")
