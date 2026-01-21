"""Quick test with collapse=False"""

import pandas as pd
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

print("Testing WITHOUT collapse:")
print()

gmm_no_collapse = DifferenceGMM(
    data=df,
    dep_var='n',
    lags=1,
    id_var='id',
    time_var='year',
    exog_vars=['w', 'k', 'ys'],
    time_dummies=True,
    collapse=False,  # NO COLLAPSE
    two_step=True,
    robust=True,
    gmm_type='two_step'
)

try:
    results = gmm_no_collapse.fit()
    print("Results:")
    print(f"  Observations: {results.nobs}")
    print(f"  Groups: {results.n_groups}")
    print(f"  Instruments: {results.n_instruments}")
    print(f"  n(-1) coefficient: {results.params['L1.n']:.3f}")
    print()
    print("This should match Arellano-Bond (1991): ~0.629")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
