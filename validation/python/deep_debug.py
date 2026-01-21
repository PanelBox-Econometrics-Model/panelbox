"""
Deep debugging: trace exactly where we lose observations
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from panelbox.gmm.instruments import InstrumentBuilder

# Load data
print("=" * 70)
print("Deep Debug: Tracing Data Loss")
print("=" * 70)
print()

df = pd.read_csv('../data/abdata.csv')
df = df.drop(columns=['c1'], errors='ignore')
df['id'] = df['id'].astype(int)
df['year'] = df['year'].astype(int)
df = df.sort_values(['id', 'year'])

for col in df.columns:
    if col not in ['id', 'year']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"Step 0: Original data")
print(f"  Observations: {len(df)}")
print(f"  Firms: {df['id'].nunique()}")
print()

# Step 1: Check dep var and lags
print(f"Step 1: Create lagged dependent variable")
df_sorted = df.sort_values(['id', 'year']).copy()
df_sorted['n_L1'] = df_sorted.groupby('id')['n'].shift(1)

n_missing = df_sorted['n_L1'].isnull().sum()
print(f"  n_L1 missing: {n_missing} / {len(df_sorted)} ({100*n_missing/len(df_sorted):.1f}%)")
print(f"  Complete cases: {(~df_sorted[['n', 'n_L1']].isnull().any(axis=1)).sum()}")
print()

# Step 2: First differences
print(f"Step 2: First differences")
df_sorted['n_diff'] = df_sorted.groupby('id')['n'].diff()
df_sorted['n_L1_diff'] = df_sorted.groupby('id')['n_L1'].diff()

n_diff_missing = df_sorted['n_diff'].isnull().sum()
print(f"  n_diff missing: {n_diff_missing} / {len(df_sorted)} ({100*n_diff_missing/len(df_sorted):.1f}%)")

complete_after_diff = ~df_sorted[['n_diff', 'n_L1_diff']].isnull().any(axis=1)
print(f"  Complete cases after diff: {complete_after_diff.sum()}")
print()

# Step 3: Add exogenous variables
print(f"Step 3: Add exogenous variables")
exog_vars = ['w', 'k', 'ys']

for var in exog_vars:
    df_sorted[f'{var}_diff'] = df_sorted.groupby('id')[var].diff()

exog_diff_vars = [f'{v}_diff' for v in exog_vars]
all_diff_vars = ['n_diff', 'n_L1_diff'] + exog_diff_vars

complete_with_exog = ~df_sorted[all_diff_vars].isnull().any(axis=1)
print(f"  Complete cases with exog: {complete_with_exog.sum()}")
print()

# Step 4: Simulate instrument generation
print(f"Step 4: Instrument generation (GMM-style collapsed)")
print()

# Create InstrumentBuilder
builder = InstrumentBuilder(df, 'id', 'year')

print(f"  Builder initialized:")
print(f"    n_groups: {builder.n_groups}")
print(f"    n_periods: {builder.n_periods}")
print(f"    time_periods: {list(builder.time_periods)}")
print()

# Try to create instruments for dep var
print(f"  Creating GMM instruments for 'n' (lags 2-99, collapsed):")
try:
    Z_n = builder.create_gmm_style_instruments(
        var='n',
        min_lag=2,
        max_lag=99,
        equation='diff',
        collapse=True
    )
    print(f"    Instruments created: {Z_n.n_instruments}")
    print(f"    Observations: {Z_n.n_obs}")
    print(f"    Instrument names: {Z_n.instrument_names}")

    # Check how many NaNs in instruments
    n_nans = np.isnan(Z_n.Z).sum()
    total_elements = Z_n.Z.size
    print(f"    NaN elements: {n_nans} / {total_elements} ({100*n_nans/total_elements:.1f}%)")

    # Check valid observations
    valid_Z = ~np.isnan(Z_n.Z).any(axis=1)
    print(f"    Valid observations (no NaN in any instrument): {valid_Z.sum()}")

except Exception as e:
    print(f"    Error: {e}")

print()

# Try with exog variables
print(f"  Creating IV instruments for 'w' (lag 0):")
try:
    Z_w = builder.create_iv_style_instruments(
        var='w',
        min_lag=0,
        max_lag=0,
        equation='diff'
    )
    print(f"    Instruments created: {Z_w.n_instruments}")
    valid_Z_w = ~np.isnan(Z_w.Z).any(axis=1)
    print(f"    Valid observations: {valid_Z_w.sum()}")
except Exception as e:
    print(f"    Error: {e}")

print()

# Step 5: Check what Stata would retain
print(f"Step 5: What should be valid for Arellano-Bond specification?")
print()

# Arellano-Bond uses:
# - dep var: n, lag=1
# - exog: w, k, ys (contemporaneous)
# - instruments: lags 2+ of n, contemporaneous w, k, ys

# After first-difference, we need:
# 1. Δn_t and Δn_{t-1} (non-missing)
# 2. Δw_t, Δk_t, Δys_t (non-missing)
# 3. n_{t-2}, n_{t-3}, ... (instruments, some can be missing)

print(f"  Minimum requirements:")
print(f"    - At least 3 periods per firm (to have t, t-1, t-2)")
print(f"    - Non-missing n, w, k, ys for at least 2 consecutive periods")
print()

# Count firms by number of consecutive non-missing observations
firms_summary = []
for firm_id in df['id'].unique():
    firm_data = df[df['id'] == firm_id].sort_values('year')
    n_periods = len(firm_data)

    # Check longest consecutive non-missing sequence
    key_vars = ['n', 'w', 'k', 'ys']
    complete = ~firm_data[key_vars].isnull().any(axis=1)

    # Find longest consecutive True sequence
    max_consecutive = 0
    current_consecutive = 0
    for val in complete:
        if val:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    firms_summary.append({
        'id': firm_id,
        'n_periods': n_periods,
        'max_consecutive_complete': max_consecutive
    })

firms_df = pd.DataFrame(firms_summary)

print(f"  Firms by max consecutive complete periods:")
for min_consec in [2, 3, 4, 5]:
    n_firms = (firms_df['max_consecutive_complete'] >= min_consec).sum()
    print(f"    >= {min_consec} periods: {n_firms} firms")

print()
print(f"  After differencing (lose 1 period per firm):")
for min_consec in [2, 3, 4, 5]:
    # After diff, need min_consec+1 original periods
    n_firms = (firms_df['max_consecutive_complete'] >= min_consec+1).sum()
    max_obs = firms_df[firms_df['max_consecutive_complete'] >= min_consec+1]['max_consecutive_complete'].sum()
    # Each firm loses 1 obs to differencing
    actual_obs = max_obs - n_firms
    print(f"    >= {min_consec} diff periods: {n_firms} firms, ~{actual_obs} observations")

print()
print("=" * 70)
print("Analysis complete")
print("=" * 70)
