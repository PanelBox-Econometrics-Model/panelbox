"""
Replication Script 01: Difference GMM Basic
============================================

Replicates Stata xtabond2 example from Roodman (2009), page 106.

Stata command:
    xtabond2 n L.n w k, gmm(L.n, lag(2 .)) iv(w k) robust small twostep

Expected Stata results (approximate):
    Coef(L.n) = 0.686
    Coef(w) = -0.608
    Coef(k) = 0.357
    Hansen J p-value ≈ 0.905
    AR(2) p-value ≈ 0.731

Reference:
    Roodman, D. (2009). "How to do xtabond2: An introduction to difference
    and system GMM in Stata." The Stata Journal, 9(1), 86-136.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import panelbox
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from panelbox.gmm import DifferenceGMM

# ============================================================================
# Configuration
# ============================================================================

DATA_PATH = Path(__file__).parent.parent / "data" / "abdata.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "python" / "01_difference_gmm_basic.json"

# Stata reference results (from Roodman 2009)
STATA_REFERENCE = {
    'coef_L_n': 0.6861222,
    'se_L_n': 0.1410496,
    'coef_w': -0.6078527,
    'se_w': 0.1749928,
    'coef_k': 0.3569219,
    'se_k': 0.0617118,
    'hansen_stat': 28.31,
    'hansen_p': 0.905,
    'n_obs': 611,
    'n_groups': 140,
    'n_instruments': 42,
}

# ============================================================================
# Helper Functions
# ============================================================================

def load_arellano_bond_data(csv_path):
    """Load Arellano-Bond dataset."""
    print(f"Loading data from: {csv_path}")

    df = pd.read_csv(csv_path)

    # Verify expected columns (minimum required)
    expected_cols = ['id', 'year', 'n', 'w', 'k']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    print(f"Dataset loaded: {len(df)} rows, {len(df['id'].unique())} firms")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Variables: {list(df.columns)}")

    return df

def compare_with_stata(results_dict, stata_ref):
    """Compare PanelBox results with Stata reference."""
    print("\n" + "="*70)
    print("COMPARISON WITH STATA xtabond2")
    print("="*70)

    comparisons = []

    # Compare coefficients
    coef_comparisons = [
        ('L.n', results_dict['params']['L1.n'], stata_ref['coef_L_n'],
         results_dict['std_errors']['L1.n'], stata_ref['se_L_n']),
        ('w', results_dict['params']['w'], stata_ref['coef_w'],
         results_dict['std_errors']['w'], stata_ref['se_w']),
        ('k', results_dict['params']['k'], stata_ref['coef_k'],
         results_dict['std_errors']['k'], stata_ref['se_k']),
    ]

    print("\nCoefficients:")
    print(f"{'Variable':<10} {'Stata':<12} {'PanelBox':<12} {'Diff':<12} {'% Diff':<10} {'Status':<8}")
    print("-" * 70)

    for var, pb_coef, stata_coef, pb_se, stata_se in coef_comparisons:
        diff = pb_coef - stata_coef
        pct_diff = (diff / stata_coef * 100) if stata_coef != 0 else 0
        status = "✓" if abs(pct_diff) < 0.01 else "✗"

        print(f"{var:<10} {stata_coef:>11.6f} {pb_coef:>11.6f} {diff:>11.6f} {pct_diff:>9.3f}% {status:>7}")

        comparisons.append({
            'variable': var,
            'type': 'coefficient',
            'stata': stata_coef,
            'panelbox': pb_coef,
            'diff': diff,
            'pct_diff': pct_diff,
            'within_tolerance': abs(pct_diff) < 0.01
        })

    # Compare standard errors
    print("\nStandard Errors:")
    print(f"{'Variable':<10} {'Stata':<12} {'PanelBox':<12} {'Diff':<12} {'% Diff':<10} {'Status':<8}")
    print("-" * 70)

    for var, pb_coef, stata_coef, pb_se, stata_se in coef_comparisons:
        diff = pb_se - stata_se
        pct_diff = (diff / stata_se * 100) if stata_se != 0 else 0
        status = "✓" if abs(pct_diff) < 0.5 else "✗"

        print(f"{var:<10} {stata_se:>11.6f} {pb_se:>11.6f} {diff:>11.6f} {pct_diff:>9.3f}% {status:>7}")

        comparisons.append({
            'variable': var,
            'type': 'std_error',
            'stata': stata_se,
            'panelbox': pb_se,
            'diff': diff,
            'pct_diff': pct_diff,
            'within_tolerance': abs(pct_diff) < 0.5
        })

    # Compare test statistics
    print("\nSpecification Tests:")
    print(f"{'Test':<20} {'Stata':<12} {'PanelBox':<12} {'Diff':<12} {'Status':<8}")
    print("-" * 70)

    # Hansen J
    hansen_diff = results_dict['hansen_j']['statistic'] - stata_ref['hansen_stat']
    hansen_pct = (hansen_diff / stata_ref['hansen_stat'] * 100) if stata_ref['hansen_stat'] != 0 else 0
    hansen_status = "✓" if abs(hansen_pct) < 1.0 else "✗"
    print(f"{'Hansen J stat':<20} {stata_ref['hansen_stat']:>11.3f} {results_dict['hansen_j']['statistic']:>11.3f} {hansen_diff:>11.3f} {hansen_status:>7}")

    comparisons.append({
        'test': 'hansen_j_stat',
        'stata': stata_ref['hansen_stat'],
        'panelbox': results_dict['hansen_j']['statistic'],
        'diff': hansen_diff,
        'pct_diff': hansen_pct,
        'within_tolerance': abs(hansen_pct) < 1.0
    })

    # Sample size
    print("\nSample Information:")
    print(f"{'Metric':<20} {'Stata':<12} {'PanelBox':<12} {'Match':<8}")
    print("-" * 70)

    n_obs_match = "✓" if results_dict['n_obs'] == stata_ref['n_obs'] else "✗"
    print(f"{'N observations':<20} {stata_ref['n_obs']:>11d} {results_dict['n_obs']:>11d} {n_obs_match:>7}")

    n_groups_match = "✓" if results_dict['n_groups'] == stata_ref['n_groups'] else "✗"
    print(f"{'N groups':<20} {stata_ref['n_groups']:>11d} {results_dict['n_groups']:>11d} {n_groups_match:>7}")

    n_instr_match = "✓" if results_dict['n_instruments'] == stata_ref['n_instruments'] else "✗"
    print(f"{'N instruments':<20} {stata_ref['n_instruments']:>11d} {results_dict['n_instruments']:>11d} {n_instr_match:>7}")

    # Overall assessment
    all_coef_ok = all(c['within_tolerance'] for c in comparisons if c.get('type') == 'coefficient')
    all_se_ok = all(c['within_tolerance'] for c in comparisons if c.get('type') == 'std_error')
    all_tests_ok = all(c['within_tolerance'] for c in comparisons if c.get('test'))

    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Coefficients:     {'✓ PASS' if all_coef_ok else '✗ FAIL'}")
    print(f"Standard Errors:  {'✓ PASS' if all_se_ok else '✗ FAIL'}")
    print(f"Test Statistics:  {'✓ PASS' if all_tests_ok else '✗ FAIL'}")
    print("="*70)

    return comparisons, all_coef_ok and all_se_ok and all_tests_ok

# ============================================================================
# Main Estimation
# ============================================================================

def main():
    """Main estimation and comparison."""

    print("="*70)
    print("REPLICATION 01: Difference GMM Basic")
    print("="*70)
    print()

    # Load data
    df = load_arellano_bond_data(DATA_PATH)

    # Create model matching Stata specification
    print("\n" + "="*70)
    print("ESTIMATING DIFFERENCE GMM")
    print("="*70)
    print()
    print("Model specification:")
    print("  Dependent variable: n (employment)")
    print("  Lagged dependent: L.n (lag 1)")
    print("  Exogenous: w (wages), k (capital)")
    print("  GMM instruments: L.n, lags 2 to max")
    print("  IV instruments: w, k")
    print("  Two-step with robust SE (Windmeijer correction)")
    print()

    model = DifferenceGMM(
        data=df,
        dep_var='n',
        lags=1,
        id_var='id',
        time_var='year',
        exog_vars=['w', 'k'],
        # GMM instruments: lags of dependent variable (2 to max)
        # This is the default behavior, no need to specify
        collapse=False,  # Non-collapsed instruments (Stata default)
        two_step=True,
        robust=True,
        time_dummies=False
    )

    # Fit model
    print("Estimating...")
    results = model.fit()

    # Display results
    print("\n" + results.summary())

    # Extract results for comparison
    results_dict = {
        'params': results.params.to_dict(),
        'std_errors': results.std_errors.to_dict(),
        'tvalues': results.tvalues.to_dict(),
        'pvalues': results.pvalues.to_dict(),
        'n_obs': results.nobs,
        'n_groups': results.n_groups,
        'n_instruments': results.n_instruments,
        'hansen_j': {
            'statistic': results.hansen_j.statistic,
            'pvalue': results.hansen_j.pvalue,
            'df': results.hansen_j.df
        },
        'sargan': {
            'statistic': results.sargan.statistic,
            'pvalue': results.sargan.pvalue,
            'df': results.sargan.df
        },
        'ar1_test': {
            'statistic': results.ar1_test.statistic,
            'pvalue': results.ar1_test.pvalue
        },
        'ar2_test': {
            'statistic': results.ar2_test.statistic,
            'pvalue': results.ar2_test.pvalue
        }
    }

    # Compare with Stata
    comparisons, validation_passed = compare_with_stata(results_dict, STATA_REFERENCE)

    # Save results (convert numpy types to Python types for JSON)
    def convert_to_python_types(obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        return obj

    output = {
        'script': '01_difference_gmm_basic',
        'reference': 'Roodman (2009), page 106',
        'stata_command': 'xtabond2 n L.n w k, gmm(L.n, lag(2 .)) iv(w k) robust small twostep',
        'results': convert_to_python_types(results_dict),
        'stata_reference': STATA_REFERENCE,
        'comparisons': convert_to_python_types(comparisons),
        'validation_passed': bool(validation_passed)
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH}")

    return validation_passed

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
