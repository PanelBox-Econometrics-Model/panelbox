"""
Analyze which observations are kept vs dropped
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
print("OBSERVATION ANALYSIS")
print("="*80)
print(f"Total observations in dataset: {len(df)}")
print(f"Firms: {df['id'].nunique()}")
print(f"Years: {sorted(df['year'].unique())}")
print(f"Year range: {df['year'].min():.0f} - {df['year'].max():.0f}")
print()

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

print("="*80)
print("AFTER DIFFERENCING")
print("="*80)
print(f"Total observations: {len(y_diff)}")
print(f"Valid (non-NaN y): {(~np.isnan(y_diff.flatten())).sum()}")
print()

# Check which years are in the valid observations
valid_mask_y = ~np.isnan(y_diff.flatten())
years_in_diff = times[valid_mask_y]
print(f"Years in differenced data: {sorted(set(years_in_diff))}")
print(f"Year range: {years_in_diff.min():.0f} - {years_in_diff.max():.0f}")
print()

# Count observations per year
year_counts = pd.Series(years_in_diff).value_counts().sort_index()
print("Observations per year after differencing:")
for year, count in year_counts.items():
    print(f"  {year:.0f}: {count}")
print()

# Now let's see what _get_valid_mask does
model.instrument_builder = model.instrument_builder
Z = model._generate_instruments()
Z_matrix = Z.Z.copy()

# Remove all-NaN columns
not_all_nan = ~np.isnan(Z_matrix).all(axis=0)
Z_matrix = Z_matrix[:, not_all_nan]

# Replace NaN with 0
Z_matrix_clean = np.nan_to_num(Z_matrix, nan=0.0)

# Now call _get_valid_mask as the estimator does
from panelbox.gmm.estimator import GMMEstimator
estimator = GMMEstimator()

valid_mask = estimator._get_valid_mask(y_diff, X_diff, Z_matrix_clean)

print("="*80)
print("AFTER _get_valid_mask()")
print("="*80)
print(f"Total observations after mask: {valid_mask.sum()}/{len(valid_mask)}")
print()

# Which years are kept?
years_kept = times[valid_mask]
year_counts_kept = pd.Series(years_kept).value_counts().sort_index()
print("Observations per year after _get_valid_mask:")
for year, count in year_counts_kept.items():
    print(f"  {year:.0f}: {count}")
print()

# Calculate which years Stata should keep
print("="*80)
print("EXPECTED FOR STATA")
print("="*80)
print("Stata command: gmm(L.n, lag(2 .))")
print("  - For each time t, uses lags 2 and beyond")
print("  - Minimum t = 1978 (first year we can use lag 2)")
print()
print("Expected years in Stata:")
stata_years = [1978, 1979, 1980, 1981, 1982, 1983, 1984]
print(f"  {stata_years}")
print()

# Count expected observations
expected_counts = []
for year in stata_years:
    # Count firms that have data for this year
    count = len(df[df['year'] == year])
    expected_counts.append(count)
    print(f"  {year}: {count} firms")
print()
print(f"Expected total: {sum(expected_counts)}")
print(f"Stata actual: 611")
print(f"PanelBox actual: {valid_mask.sum()}")
print()

# What's the difference?
print("="*80)
print("DIAGNOSIS")
print("="*80)
print(f"Difference: {valid_mask.sum() - 611} observations")
print(f"Number of firms: {df['id'].nunique()}")
print()

if valid_mask.sum() - 611 == 140:
    print("PATTERN: Exactly 1 extra observation per firm")
    print("HYPOTHESIS: PanelBox includes year 1977, Stata excludes it")
    print()
    print("Year 1977 check:")
    n_1977 = (years_kept == 1977).sum()
    print(f"  Observations in 1977: {n_1977}")
    print(f"  Expected if hypothesis correct: 140")
