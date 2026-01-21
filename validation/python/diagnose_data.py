"""
Diagnose why we're getting so few observations in the Arellano-Bond validation.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from panelbox.gmm import DifferenceGMM


def main():
    print("=" * 70)
    print("Diagnosing Arellano-Bond Dataset")
    print("=" * 70)
    print()

    # Load data
    df = pd.read_csv('../data/abdata.csv')
    df = df.drop(columns=['c1'], errors='ignore')
    df['id'] = df['id'].astype(int)
    df['year'] = df['year'].astype(int)
    df = df.sort_values(['id', 'year'])

    for col in df.columns:
        if col not in ['id', 'year']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Original dataset: {len(df)} observations, {df['id'].nunique()} firms")
    print()

    # Check missing values in key variables
    key_vars = ['n', 'w', 'wL1', 'k', 'kL1', 'ys', 'ysL1']
    print("Missing values in key variables:")
    for var in key_vars:
        missing = df[var].isnull().sum()
        pct = 100 * missing / len(df)
        print(f"  {var:10s}: {missing:4d} / {len(df)} ({pct:.1f}%)")
    print()

    # Check complete cases
    complete_cases = df[key_vars + ['id', 'year']].dropna()
    print(f"Complete cases (no missing): {len(complete_cases)} observations")
    print(f"  Firms with complete data: {complete_cases['id'].nunique()}")
    print()

    # Check what happens after differencing
    print("Testing with simpler specification (no time dummies):")
    print()

    gmm_simple = DifferenceGMM(
        data=df,
        dep_var='n',
        lags=1,
        id_var='id',
        time_var='year',
        exog_vars=['w', 'k', 'ys'],  # Only contemporary values
        time_dummies=False,  # No time dummies
        collapse=True,
        two_step=False,
        robust=True,
        gmm_type='one_step'
    )

    try:
        results_simple = gmm_simple.fit()
        print("Simple specification results:")
        print(f"  Observations: {results_simple.nobs}")
        print(f"  Groups: {results_simple.n_groups}")
        print(f"  Instruments: {results_simple.n_instruments}")
        print(f"  n(-1) coefficient: {results_simple.params['L1.n']:.3f}")
        print()
    except Exception as e:
        print(f"Error with simple specification: {e}")
        print()

    # Test with lagged variables included
    print("Testing with lagged exog variables:")
    print()

    gmm_with_lags = DifferenceGMM(
        data=df,
        dep_var='n',
        lags=1,
        id_var='id',
        time_var='year',
        exog_vars=['w', 'wL1', 'k', 'kL1', 'ys', 'ysL1'],
        time_dummies=False,  # No time dummies
        collapse=True,
        two_step=False,
        robust=True,
        gmm_type='one_step'
    )

    try:
        results_lags = gmm_with_lags.fit()
        print("With lagged variables:")
        print(f"  Observations: {results_lags.nobs}")
        print(f"  Groups: {results_lags.n_groups}")
        print(f"  Instruments: {results_lags.n_instruments}")
        print(f"  n(-1) coefficient: {results_lags.params['L1.n']:.3f}")
        print()
    except Exception as e:
        print(f"Error with lagged variables: {e}")
        print()

    # Test with time dummies
    print("Testing with time dummies:")
    print()

    gmm_dummies = DifferenceGMM(
        data=df,
        dep_var='n',
        lags=1,
        id_var='id',
        time_var='year',
        exog_vars=['w', 'k', 'ys'],
        time_dummies=True,  # WITH time dummies
        collapse=True,
        two_step=False,
        robust=True,
        gmm_type='one_step'
    )

    try:
        results_dummies = gmm_dummies.fit()
        print("With time dummies:")
        print(f"  Observations: {results_dummies.nobs}")
        print(f"  Groups: {results_dummies.n_groups}")
        print(f"  Instruments: {results_dummies.n_instruments}")
        print(f"  n(-1) coefficient: {results_dummies.params['L1.n']:.3f}")
        print()
    except Exception as e:
        print(f"Error with time dummies: {e}")
        print()

    print("=" * 70)
    print("Diagnosis complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
