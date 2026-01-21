"""
Validation: Arellano-Bond (1991) Employment Equation
====================================================

Replicate the classic employment equation from Arellano & Bond (1991)
using PanelBox and compare with published Stata xtabond2 results.

Model:
    n_it = γ n_{i,t-1} + β_w w_it + β_wL w_{i,t-1} +
           β_k k_it + β_kL k_{i,t-1} +
           β_ys ys_it + β_ysL ys_{i,t-1} +
           time_dummies + η_i + ε_it

Dataset: abdata (140 UK firms, 1976-1984)
Reference: Review of Economic Studies, 58(2), 277-297

Expected Results (from literature):
-----------------------------------
Arellano-Bond (1991), Table 4, Column a1:
- One-step n(-1): ~0.686
- Two-step n(-1): ~0.629

Range check: Should be between 0.733 (LSDV) and 1.045 (OLS)
"""

import pandas as pd
import numpy as np
import sys
import os

# Add panelbox to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from panelbox.gmm import DifferenceGMM


# Reference results from Arellano-Bond (1991) Table 4, Column a1
REFERENCE_RESULTS = {
    'one_step': {
        'n_L1': 0.686,
        'note': 'Arellano-Bond (1991), Table 4, Column a1'
    },
    'two_step': {
        'n_L1': 0.629,
        'note': 'Arellano-Bond (1991), Table 4, Column a1'
    },
    'range': {
        'lower': 0.733,  # LSDV estimate
        'upper': 1.045,  # OLS estimate
        'note': 'Credible range from OLS and LSDV'
    }
}


def load_data():
    """Load Arellano-Bond dataset and create lagged variables."""
    print("Loading Arellano-Bond dataset...")
    df = pd.read_csv('../data/abdata.csv')

    # Drop non-numeric column (c1 is object type)
    df = df.drop(columns=['c1'], errors='ignore')

    # Ensure id and year are numeric
    df['id'] = df['id'].astype(int)
    df['year'] = df['year'].astype(int)

    # Sort by id and year
    df = df.sort_values(['id', 'year'])

    # Convert all columns to float where possible
    for col in df.columns:
        if col not in ['id', 'year']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"✓ Loaded {len(df)} observations, {df['id'].nunique()} firms")
    print(f"  Key variables: n, w, k, ys (and lagged versions)")
    print()

    return df


def estimate_one_step(data):
    """Estimate one-step Difference GMM."""
    print("=" * 70)
    print("One-Step Difference GMM (Arellano-Bond 1991)")
    print("=" * 70)
    print()

    # Specification: n_it = γ n_{i,t-1} + β_w w_it + β_wL w_{i,t-1} + ...
    # The dataset already has lagged variables: wL1, kL1, ysL1
    gmm = DifferenceGMM(
        data=data,
        dep_var='n',
        lags=1,
        id_var='id',
        time_var='year',
        exog_vars=['w', 'wL1', 'k', 'kL1', 'ys', 'ysL1'],
        time_dummies=True,
        collapse=True,
        two_step=False,  # One-step
        robust=True,
        gmm_type='one_step'
    )

    results = gmm.fit()
    print(results.summary())
    print()

    return results


def estimate_two_step(data):
    """Estimate two-step Difference GMM with Windmeijer correction."""
    print("=" * 70)
    print("Two-Step Difference GMM (Arellano-Bond 1991) with Windmeijer Correction")
    print("=" * 70)
    print()

    gmm = DifferenceGMM(
        data=data,
        dep_var='n',
        lags=1,
        id_var='id',
        time_var='year',
        exog_vars=['w', 'wL1', 'k', 'kL1', 'ys', 'ysL1'],
        time_dummies=True,
        collapse=True,
        two_step=True,  # Two-step
        robust=True,
        gmm_type='two_step'
    )

    results = gmm.fit()
    print(results.summary())
    print()

    return results


def compare_results(one_step, two_step):
    """Compare PanelBox results with published values."""
    print("=" * 70)
    print("Comparison with Arellano-Bond (1991)")
    print("=" * 70)
    print()

    # Extract coefficient on n(-1)
    coef_one_step = one_step.params['L1.n']
    coef_two_step = two_step.params['L1.n']

    ref_one_step = REFERENCE_RESULTS['one_step']['n_L1']
    ref_two_step = REFERENCE_RESULTS['two_step']['n_L1']

    # Compute differences
    diff_one_step = coef_one_step - ref_one_step
    diff_two_step = coef_two_step - ref_two_step

    pct_diff_one_step = 100 * diff_one_step / ref_one_step
    pct_diff_two_step = 100 * diff_two_step / ref_two_step

    # Create comparison table
    comparison = pd.DataFrame({
        'PanelBox': [coef_one_step, coef_two_step],
        'Arellano-Bond (1991)': [ref_one_step, ref_two_step],
        'Difference': [diff_one_step, diff_two_step],
        '% Difference': [pct_diff_one_step, pct_diff_two_step]
    }, index=['One-Step n(-1)', 'Two-Step n(-1)'])

    print("Coefficient Comparison: n(-1)")
    print(comparison.to_string())
    print()

    # Check if within acceptable range
    lower_bound = REFERENCE_RESULTS['range']['lower']
    upper_bound = REFERENCE_RESULTS['range']['upper']

    print("Credible Range Check:")
    print(f"  Range: [{lower_bound}, {upper_bound}] (LSDV to OLS)")
    print()

    for name, coef in [('One-Step', coef_one_step), ('Two-Step', coef_two_step)]:
        in_range = lower_bound <= coef <= upper_bound
        status = "✓ PASS" if in_range else "⚠ OUT OF RANGE"
        print(f"  {name}: {coef:.3f} - {status}")

    print()

    # Tolerance check
    print("Tolerance Check (±5% acceptable for replication):")
    print()

    for name, pct in [('One-Step', pct_diff_one_step), ('Two-Step', pct_diff_two_step)]:
        within_tol = abs(pct) <= 5.0
        status = "✓ PASS" if within_tol else "⚠ WARNING"
        sign = "+" if pct >= 0 else ""
        print(f"  {name}: {sign}{pct:.2f}% - {status}")

    print()

    # Specification tests comparison
    print("=" * 70)
    print("Specification Tests")
    print("=" * 70)
    print()

    spec_tests = pd.DataFrame({
        'One-Step': [
            one_step.n_instruments,
            one_step.instrument_ratio,
            one_step.hansen_j.statistic,
            one_step.hansen_j.pvalue,
            one_step.sargan.statistic,
            one_step.sargan.pvalue,
            one_step.ar1_test.statistic if not pd.isna(one_step.ar1_test.statistic) else 'N/A',
            one_step.ar1_test.pvalue if not pd.isna(one_step.ar1_test.pvalue) else 'N/A',
            one_step.ar2_test.statistic if not pd.isna(one_step.ar2_test.statistic) else 'N/A',
            one_step.ar2_test.pvalue if not pd.isna(one_step.ar2_test.pvalue) else 'N/A',
        ],
        'Two-Step': [
            two_step.n_instruments,
            two_step.instrument_ratio,
            two_step.hansen_j.statistic,
            two_step.hansen_j.pvalue,
            two_step.sargan.statistic,
            two_step.sargan.pvalue,
            two_step.ar1_test.statistic if not pd.isna(two_step.ar1_test.statistic) else 'N/A',
            two_step.ar1_test.pvalue if not pd.isna(two_step.ar1_test.pvalue) else 'N/A',
            two_step.ar2_test.statistic if not pd.isna(two_step.ar2_test.statistic) else 'N/A',
            two_step.ar2_test.pvalue if not pd.isna(two_step.ar2_test.pvalue) else 'N/A',
        ]
    }, index=['Instruments', 'Instrument Ratio', 'Hansen J', 'Hansen p-value',
              'Sargan', 'Sargan p-value', 'AR(1)', 'AR(1) p-value',
              'AR(2)', 'AR(2) p-value'])

    print(spec_tests.to_string())
    print()

    # Validation summary
    print("=" * 70)
    print("Validation Summary")
    print("=" * 70)
    print()

    one_step_pass = abs(pct_diff_one_step) <= 5.0
    two_step_pass = abs(pct_diff_two_step) <= 5.0

    if one_step_pass and two_step_pass:
        print("✓ VALIDATION SUCCESSFUL")
        print()
        print("PanelBox Difference GMM matches Arellano-Bond (1991) results")
        print("within acceptable tolerance (±5%).")
    else:
        print("⚠ VALIDATION WARNING")
        print()
        print("Some coefficients differ by more than 5% from published values.")
        print("This may be due to:")
        print("  - Different instrument specifications")
        print("  - Different handling of unbalanced panels")
        print("  - Numerical precision differences")

    print()


def main():
    """Run complete validation."""
    print("=" * 70)
    print("PanelBox Validation: Arellano-Bond (1991) Employment Equation")
    print("=" * 70)
    print()

    # Load data
    data = load_data()

    # Estimate models
    one_step_results = estimate_one_step(data)
    two_step_results = estimate_two_step(data)

    # Compare with published results
    compare_results(one_step_results, two_step_results)

    print("=" * 70)
    print("✓ Validation completed")
    print("=" * 70)


if __name__ == '__main__':
    main()
