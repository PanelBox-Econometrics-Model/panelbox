"""
Convert Arellano-Bond dataset from Stata to CSV
and explore its structure.

Dataset: abdata.dta
Source: Arellano & Bond (1991), Review of Economic Studies
"""

import pandas as pd
import numpy as np
import sys
import os

# Add panelbox to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def main():
    print("=" * 70)
    print("Arellano-Bond Dataset Conversion")
    print("=" * 70)
    print()

    # Read Stata file
    print("Reading abdata.dta...")
    df = pd.read_stata('../data/abdata.dta')

    print(f"✓ Dataset loaded successfully")
    print()

    # Basic info
    print("=" * 70)
    print("Dataset Structure")
    print("=" * 70)
    print()
    print(f"Shape: {df.shape[0]} observations × {df.shape[1]} variables")
    print()

    print("Variables:")
    print(df.dtypes)
    print()

    # Panel structure
    print("=" * 70)
    print("Panel Structure")
    print("=" * 70)
    print()

    n_firms = df['id'].nunique()
    n_years = df['year'].nunique()

    print(f"Number of firms (N): {n_firms}")
    print(f"Number of years (T): {n_years}")
    print(f"Years range: {df['year'].min():.0f} - {df['year'].max():.0f}")
    print()

    # Check balance
    obs_per_firm = df.groupby('id').size()
    print(f"Observations per firm:")
    print(f"  Min: {obs_per_firm.min()}")
    print(f"  Max: {obs_per_firm.max()}")
    print(f"  Mean: {obs_per_firm.mean():.1f}")
    print(f"  Median: {obs_per_firm.median():.1f}")
    print()

    if obs_per_firm.min() != obs_per_firm.max():
        print("⚠ Panel is UNBALANCED")
    else:
        print("✓ Panel is balanced")
    print()

    # Summary statistics
    print("=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print()

    # Key variables from Arellano-Bond (1991)
    key_vars = ['n', 'w', 'k', 'ys']

    summary = df[key_vars].describe()
    print(summary.to_string())
    print()

    # Missing values
    print("=" * 70)
    print("Missing Values")
    print("=" * 70)
    print()

    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)

    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percent': missing_pct
    })

    print(missing_df[missing_df['Missing'] > 0].to_string())

    if missing.sum() == 0:
        print("✓ No missing values")
    print()

    # First few rows
    print("=" * 70)
    print("First 10 Observations")
    print("=" * 70)
    print()
    print(df.head(10).to_string())
    print()

    # Save to CSV
    csv_path = '../data/abdata.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Dataset saved to: {csv_path}")
    print()

    print("=" * 70)
    print("✓ Conversion completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
