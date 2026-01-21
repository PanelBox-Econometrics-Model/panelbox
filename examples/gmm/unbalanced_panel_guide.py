"""
Unbalanced Panel Data Guide
============================

Practical guide for handling unbalanced panels with GMM estimation.

This example demonstrates:
- Challenges with unbalanced panels (missing observations)
- Why complex specifications fail
- Practical workarounds and best practices
- Example with Arellano-Bond employment data

Key insights from Subfase 4.2 development.

Author: PanelBox Development Team
Date: January 2026
"""

import numpy as np
import pandas as pd
import sys
import os

# Add panelbox to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from panelbox.gmm import DifferenceGMM


def create_unbalanced_panel(n_firms=200, max_periods=10, missing_rate=0.20, seed=42):
    """
    Create synthetic unbalanced panel data.

    Parameters
    ----------
    n_firms : int
        Number of firms
    max_periods : int
        Maximum number of time periods
    missing_rate : float
        Proportion of observations to drop randomly (unbalanced structure)
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Unbalanced panel data
    """
    np.random.seed(seed)

    data = []

    for i in range(1, n_firms + 1):
        # Firm fixed effect
        eta_i = np.random.normal(0, 1.0)

        # Generate complete time series
        for t in range(max_periods):
            # Skip some observations randomly (create unbalanced structure)
            if np.random.rand() < missing_rate:
                continue

            # Lagged value (if exists)
            if len(data) > 0 and data[-1]['firm_id'] == i and data[-1]['year'] == 1999 + t:
                y_lag = data[-1]['y']
            else:
                y_lag = eta_i + np.random.normal(0, 0.5)

            # Exogenous variables
            x1 = np.random.normal(2, 0.5)
            x2 = np.random.normal(1, 0.3)

            # Dependent variable (dynamic panel)
            epsilon = np.random.normal(0, 0.4)
            y = 0.6 * y_lag + 0.3 * x1 + 0.2 * x2 + eta_i + epsilon

            data.append({
                'firm_id': i,
                'year': 2000 + t,
                'y': y,
                'x1': x1,
                'x2': x2
            })

    df = pd.DataFrame(data)

    print("="*70)
    print("UNBALANCED PANEL DATA CREATED")
    print("="*70)
    print()
    print(f"Firms: {df['firm_id'].nunique()}")
    print(f"Time periods: {df['year'].nunique()} (2000-{df['year'].max()})")
    print(f"Total observations: {len(df)}")
    print(f"Missing rate: {missing_rate*100:.0f}%")
    print()

    # Panel structure analysis
    obs_per_firm = df.groupby('firm_id').size()
    print("Panel structure:")
    print(f"  Min observations per firm: {obs_per_firm.min()}")
    print(f"  Max observations per firm: {obs_per_firm.max()}")
    print(f"  Mean observations per firm: {obs_per_firm.mean():.1f}")
    print(f"  Firms with all {max_periods} periods: {(obs_per_firm == max_periods).sum()}")
    print()

    return df


def example_1_simple_specification(df):
    """
    Example 1: Simple specification (works well).
    """
    print("="*70)
    print("EXAMPLE 1: Simple Specification (WORKS)")
    print("="*70)
    print()

    print("Specification:")
    print("  - 1 lag of y")
    print("  - 2 exogenous variables (x1, x2)")
    print("  - NO time dummies")
    print("  - collapse=True")
    print()

    gmm = DifferenceGMM(
        data=df,
        dep_var='y',
        lags=1,
        id_var='firm_id',
        time_var='year',
        exog_vars=['x1', 'x2'],
        time_dummies=False,  # KEY: No time dummies
        collapse=True,       # KEY: Always collapse
        two_step=True,
        robust=True
    )

    results = gmm.fit()
    print(results.summary())
    print()

    print("Result: ✓ SUCCESS")
    print(f"  Observations used: {results.nobs} (retained {results.nobs/len(df)*100:.1f}%)")
    print(f"  Instruments: {results.n_instruments}")
    print(f"  Instrument ratio: {results.instrument_ratio:.3f}")
    print()

    print("Why it works:")
    print("  - Few parameters (3 total: y_lag, x1, x2)")
    print("  - Few instruments (collapsed GMM-style)")
    print("  - No time dummies (would add 8+ parameters)")
    print("  - Most observations have sufficient instruments")
    print()

    return results


def example_2_with_time_dummies(df):
    """
    Example 2: With time dummies (MAY FAIL).
    """
    print("="*70)
    print("EXAMPLE 2: With Time Dummies (PROBLEMATIC)")
    print("="*70)
    print()

    print("Specification:")
    print("  - 1 lag of y")
    print("  - 2 exogenous variables")
    print("  - TIME DUMMIES (8 dummies for 9 periods)")
    print("  - collapse=True")
    print()

    print("⚠ WARNING: This often fails with unbalanced panels!")
    print()

    try:
        gmm = DifferenceGMM(
            data=df,
            dep_var='y',
            lags=1,
            id_var='firm_id',
            time_var='year',
            exog_vars=['x1', 'x2'],
            time_dummies=True,  # ← Problem!
            collapse=True,
            two_step=True,
            robust=True
        )

        results = gmm.fit()

        if results.nobs < len(df) * 0.10:
            print("Result: ⚠ VERY LOW OBSERVATIONS")
            print(f"  Only {results.nobs} observations retained ({results.nobs/len(df)*100:.1f}%)")
            print(f"  Parameters: {len(results.params)}")
            print(f"  Instruments: {results.n_instruments}")
        else:
            print("Result: ✓ SUCCESS (unusual for unbalanced panels)")
            print(f"  Observations: {results.nobs}")

    except Exception as e:
        print(f"Result: ✗ FAILED")
        print(f"  Error: {str(e)[:100]}")

    print()
    print("Why it often fails:")
    print("  - Time dummies add many parameters (8+ dummies)")
    print("  - Total parameters: 3 variables + 8 dummies = 11")
    print("  - Need 12+ valid instruments per observation")
    print("  - Unbalanced structure = sparse instrument matrix")
    print("  - Many observations dropped due to insufficient instruments")
    print()


def example_3_workaround_trend(df):
    """
    Example 3: Workaround - use trend instead of dummies.
    """
    print("="*70)
    print("EXAMPLE 3: Workaround - Linear Trend (RECOMMENDED)")
    print("="*70)
    print()

    # Add linear trend
    df_trend = df.copy()
    df_trend['trend'] = df_trend['year'] - df_trend['year'].min()

    print("Specification:")
    print("  - 1 lag of y")
    print("  - 2 exogenous variables")
    print("  - LINEAR TREND (instead of time dummies)")
    print("  - collapse=True")
    print()

    gmm = DifferenceGMM(
        data=df_trend,
        dep_var='y',
        lags=1,
        id_var='firm_id',
        time_var='year',
        exog_vars=['x1', 'x2', 'trend'],  # Include trend as exog
        time_dummies=False,  # No dummies
        collapse=True,
        two_step=True,
        robust=True
    )

    results = gmm.fit()
    print(results.summary())
    print()

    print("Result: ✓ SUCCESS")
    print(f"  Observations: {results.nobs} ({results.nobs/len(df)*100:.1f}%)")
    print(f"  Parameters: {len(results.params)} (much fewer than with dummies)")
    print()

    print("Advantages:")
    print("  ✓ Captures time trend")
    print("  ✓ Only 1 additional parameter (vs 8+ dummies)")
    print("  ✓ More observations retained")
    print("  ✓ More efficient estimation")
    print()

    print("When to use:")
    print("  - Unbalanced panels")
    print("  - Smooth time trends expected")
    print("  - Alternative: Include only key time dummies (e.g., crisis years)")
    print()

    return results


def example_4_subset_dummies(df):
    """
    Example 4: Use subset of time dummies.
    """
    print("="*70)
    print("EXAMPLE 4: Subset of Time Dummies")
    print("="*70)
    print()

    # Create dummies for specific years only
    df_subset = df.copy()
    df_subset['crisis_2008'] = (df_subset['year'] == 2008).astype(int)
    df_subset['crisis_2009'] = (df_subset['year'] == 2009).astype(int)

    print("Specification:")
    print("  - 1 lag of y")
    print("  - 2 exogenous variables")
    print("  - SUBSET OF DUMMIES (only 2 specific years)")
    print("  - collapse=True")
    print()

    gmm = DifferenceGMM(
        data=df_subset,
        dep_var='y',
        lags=1,
        id_var='firm_id',
        time_var='year',
        exog_vars=['x1', 'x2', 'crisis_2008', 'crisis_2009'],
        time_dummies=False,  # Not using automatic dummies
        collapse=True,
        two_step=True,
        robust=True
    )

    results = gmm.fit()
    print(results.summary())
    print()

    print("Result: ✓ SUCCESS")
    print(f"  Observations: {results.nobs}")
    print(f"  Only 2 dummies (vs 8+ with full time_dummies=True)")
    print()

    print("When to use:")
    print("  - Need to control for specific events (crisis, policy changes)")
    print("  - Don't need full time fixed effects")
    print("  - Balance between flexibility and parsimony")
    print()

    return results


def best_practices_summary():
    """
    Summary of best practices for unbalanced panels.
    """
    print("="*70)
    print("BEST PRACTICES FOR UNBALANCED PANELS")
    print("="*70)
    print()

    print("1. ALWAYS use collapse=True")
    print("   → Reduces instruments significantly")
    print("   → Essential for unbalanced panels")
    print()

    print("2. Avoid full time dummies")
    print("   → Each dummy adds a parameter")
    print("   → Alternatives:")
    print("       • Linear trend")
    print("       • Subset of key dummies")
    print("       • No time controls (if appropriate)")
    print()

    print("3. Keep specifications parsimonious")
    print("   → Fewer parameters = easier to overidentify")
    print("   → Focus on key variables")
    print("   → Avoid unnecessary lags")
    print()

    print("4. Check observations retained")
    print("   → If < 30% retained, simplify specification")
    print("   → Compare input observations vs results.nobs")
    print()

    print("5. Instrument diagnostics")
    print("   → Check instrument ratio (target: < 1.0)")
    print("   → Hansen J test (target: 0.10 < p < 0.25)")
    print("   → AR(2) test (target: p > 0.10)")
    print()

    print("6. When everything fails")
    print("   → Consider balanced subset")
    print("   → Use only firms with many observations")
    print("   → Try Difference GMM (more robust than System GMM)")
    print()

    print("7. Documentation")
    print("   → Always report observations used vs total")
    print("   → Explain instrument choices")
    print("   → Justify time control approach")
    print()


def diagnostic_checklist(results, df_original):
    """
    Checklist for unbalanced panel GMM results.
    """
    print("="*70)
    print("DIAGNOSTIC CHECKLIST FOR UNBALANCED PANELS")
    print("="*70)
    print()

    retention_rate = results.nobs / len(df_original)

    # 1. Observation retention
    print(f"1. Observation retention: {results.nobs}/{len(df_original)} ({retention_rate*100:.1f}%)")
    if retention_rate > 0.70:
        print("   ✓ Good retention")
    elif retention_rate > 0.50:
        print("   ✓ Acceptable retention")
    elif retention_rate > 0.30:
        print("   ⚠ Low retention - consider simplifying")
    else:
        print("   ✗ Very low retention - specification too complex")
    print()

    # 2. Instrument ratio
    print(f"2. Instrument ratio: {results.instrument_ratio:.3f}")
    if results.instrument_ratio < 0.5:
        print("   ✓ Good")
    elif results.instrument_ratio < 1.0:
        print("   ✓ Acceptable")
    else:
        print("   ✗ Too many instruments - use collapse=True")
    print()

    # 3. Hansen J
    hansen_p = results.hansen_j.pvalue
    print(f"3. Hansen J-test: p = {hansen_p:.3f}")
    if 0.10 < hansen_p < 0.25:
        print("   ✓ Ideal range")
    elif hansen_p < 0.10:
        print("   ✗ Instruments rejected")
    else:
        print("   ⚠ Very high - possible weak instruments")
    print()

    # 4. AR(2)
    ar2_p = results.ar2_test.pvalue
    print(f"4. AR(2) test: p = {ar2_p:.3f}")
    if ar2_p > 0.10:
        print("   ✓ Moment conditions valid")
    else:
        print("   ✗ Moment conditions invalid")
    print()

    # Overall assessment
    print("Overall Assessment:")
    all_good = (retention_rate > 0.50 and
                results.instrument_ratio < 1.0 and
                0.10 < hansen_p < 0.50 and
                ar2_p > 0.10)

    if all_good:
        print("  ✓ Results are reliable")
    else:
        print("  ⚠ Some diagnostics problematic - interpret with caution")
    print()


def main():
    """
    Run complete unbalanced panel guide.
    """
    print()
    print("="*70)
    print("UNBALANCED PANEL DATA: PRACTICAL GUIDE")
    print("="*70)
    print()
    print("This guide demonstrates:")
    print("  - Challenges with unbalanced panels")
    print("  - Practical workarounds")
    print("  - Best practices")
    print()

    # Create unbalanced panel
    df = create_unbalanced_panel(n_firms=200, max_periods=10, missing_rate=0.20)

    # Example 1: Simple specification (works)
    results_simple = example_1_simple_specification(df)

    # Example 2: With time dummies (problematic)
    example_2_with_time_dummies(df)

    # Example 3: Trend instead of dummies (recommended)
    results_trend = example_3_workaround_trend(df)

    # Example 4: Subset of dummies
    results_subset = example_4_subset_dummies(df)

    # Best practices
    best_practices_summary()

    # Diagnostic checklist (using simple specification)
    diagnostic_checklist(results_simple, df)

    print("="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print()
    print("1. Unbalanced panels are COMMON in real data")
    print("   → Don't try to \"fix\" by dropping observations")
    print("   → GMM can handle unbalanced structure")
    print()
    print("2. The main challenge: sparse instrument matrices")
    print("   → Many obs lack sufficient valid instruments")
    print("   → Especially problematic with many parameters")
    print()
    print("3. ALWAYS use collapse=True")
    print("   → Non-negotiable for unbalanced panels")
    print("   → Reduces instruments dramatically")
    print()
    print("4. Avoid full time dummies with unbalanced panels")
    print("   → Use linear trend instead")
    print("   → Or include only key time periods")
    print()
    print("5. Monitor observation retention rate")
    print("   → Target: > 50% of original observations")
    print("   → If much lower, simplify specification")
    print()
    print("6. System GMM more sensitive than Difference GMM")
    print("   → Level instruments more affected by unbalanced structure")
    print("   → Difference GMM often more robust")
    print()
    print("="*70)
    print("✓ Guide completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
