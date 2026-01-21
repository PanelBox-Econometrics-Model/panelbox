"""
OLS vs Fixed Effects vs GMM Comparison
=======================================

Educational example demonstrating bias in different estimators for
dynamic panel data with fixed effects.

This example shows:
1. OLS bias (upward) - ignores fixed effects
2. Fixed Effects bias (downward) - Nickell bias
3. GMM (unbiased) - correct estimator

Data Generating Process:
    y_it = γ y_{i,t-1} + β x_it + η_i + ε_it

Where:
    - γ = 0.5 (AR coefficient - TRUE VALUE)
    - β = 0.3 (effect of x - TRUE VALUE)
    - η_i ~ N(0, 1) (individual fixed effect)
    - ε_it ~ N(0, 0.5) (idiosyncratic error)

Expected Results:
    - OLS: γ̂ > 0.5 (upward bias)
    - FE:  γ̂ < 0.5 (downward bias, Nickell 1981)
    - GMM: γ̂ ≈ 0.5 (consistent)

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
from sklearn.linear_model import LinearRegression
from scipy import stats


def generate_dynamic_panel(n_groups=200, n_periods=10, gamma=0.5, beta=0.3, seed=42):
    """
    Generate synthetic dynamic panel data.

    Parameters
    ----------
    n_groups : int
        Number of cross-sectional units (individuals/firms)
    n_periods : int
        Number of time periods
    gamma : float
        True AR(1) coefficient (lagged y coefficient)
    beta : float
        True effect of exogenous variable x
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Panel data with known true parameters
    """
    np.random.seed(seed)

    data = []

    for i in range(1, n_groups + 1):
        # Individual fixed effect
        eta_i = np.random.normal(0, 1)

        # Exogenous variable (independent of fixed effect and error)
        x_it = np.random.normal(2, 0.5, n_periods)

        # Initialize y
        y_it = np.zeros(n_periods)
        y_it[0] = eta_i + np.random.normal(0, 0.5)

        # Generate y recursively
        for t in range(1, n_periods):
            epsilon_it = np.random.normal(0, 0.5)
            y_it[t] = gamma * y_it[t-1] + beta * x_it[t] + eta_i + epsilon_it

        # Store data
        for t in range(n_periods):
            data.append({
                'id': i,
                'year': t + 1,
                'y': y_it[t],
                'x': x_it[t]
            })

    df = pd.DataFrame(data)

    print(f"Generated dynamic panel data:")
    print(f"  N = {n_groups} individuals")
    print(f"  T = {n_periods} periods")
    print(f"  Total observations = {len(df)}")
    print(f"  TRUE parameters: γ = {gamma}, β = {beta}")
    print()

    return df, gamma, beta


def estimate_ols(df):
    """
    Pooled OLS estimation (BIASED).

    Ignores fixed effects, leading to upward bias in γ̂.
    """
    print("=" * 70)
    print("Method 1: Pooled OLS (Biased)")
    print("=" * 70)
    print()

    # Create lagged y
    df_sorted = df.sort_values(['id', 'year']).copy()
    df_sorted['y_lag'] = df_sorted.groupby('id')['y'].shift(1)

    # Drop missing (first observation per individual)
    df_ols = df_sorted.dropna()

    # OLS regression: y = γ y_{t-1} + β x + ε
    X = df_ols[['y_lag', 'x']].values
    y = df_ols['y'].values

    model = LinearRegression()
    model.fit(X, y)

    gamma_ols = model.coef_[0]
    beta_ols = model.coef_[1]

    # Compute standard errors (simple formula)
    y_pred = model.predict(X)
    residuals = y - y_pred
    n = len(y)
    k = 2
    sigma_sq = np.sum(residuals**2) / (n - k)

    # Var(β̂) = σ² (X'X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)
    var_beta = sigma_sq * np.diag(XtX_inv)
    se_gamma = np.sqrt(var_beta[0])
    se_beta = np.sqrt(var_beta[1])

    print(f"Pooled OLS Results:")
    print(f"  γ̂ (y_lag): {gamma_ols:.4f} (SE: {se_gamma:.4f})")
    print(f"  β̂ (x):     {beta_ols:.4f} (SE: {se_beta:.4f})")
    print(f"  R²:        {model.score(X, y):.4f}")
    print(f"  N:         {n}")
    print()

    print("Note: OLS ignores fixed effects η_i, causing UPWARD BIAS in γ̂")
    print("      (correlation between y_{t-1} and η_i)")
    print()

    return gamma_ols, beta_ols, se_gamma, se_beta


def estimate_fixed_effects(df):
    """
    Within (Fixed Effects) estimation (BIASED for dynamic panels).

    Demeaning causes Nickell bias (downward) when T is small.
    """
    print("=" * 70)
    print("Method 2: Fixed Effects / Within Estimator (Nickell Biased)")
    print("=" * 70)
    print()

    # Create lagged y
    df_sorted = df.sort_values(['id', 'year']).copy()
    df_sorted['y_lag'] = df_sorted.groupby('id')['y'].shift(1)
    df_fe = df_sorted.dropna()

    # Within transformation (demean by individual)
    for var in ['y', 'y_lag', 'x']:
        group_means = df_fe.groupby('id')[var].transform('mean')
        df_fe[f'{var}_dm'] = df_fe[var] - group_means

    # FE regression on demeaned data
    X_dm = df_fe[['y_lag_dm', 'x_dm']].values
    y_dm = df_fe['y_dm'].values

    model = LinearRegression(fit_intercept=False)  # Already demeaned
    model.fit(X_dm, y_dm)

    gamma_fe = model.coef_[0]
    beta_fe = model.coef_[1]

    # Standard errors
    y_pred = model.predict(X_dm)
    residuals = y_dm - y_pred
    n = len(y_dm)
    k = 2
    sigma_sq = np.sum(residuals**2) / (n - k)

    XtX_inv = np.linalg.inv(X_dm.T @ X_dm)
    var_beta = sigma_sq * np.diag(XtX_inv)
    se_gamma = np.sqrt(var_beta[0])
    se_beta = np.sqrt(var_beta[1])

    print(f"Fixed Effects Results:")
    print(f"  γ̂ (y_lag): {gamma_fe:.4f} (SE: {se_gamma:.4f})")
    print(f"  β̂ (x):     {beta_fe:.4f} (SE: {se_beta:.4f})")
    print(f"  R²:        {model.score(X_dm, y_dm):.4f}")
    print(f"  N:         {n}")
    print()

    print("Note: FE causes DOWNWARD BIAS (Nickell 1981) when T is small")
    print("      (correlation between demeaned y_{t-1} and demeaned ε_t)")
    print()

    return gamma_fe, beta_fe, se_gamma, se_beta


def estimate_gmm(df):
    """
    Difference GMM (CONSISTENT).

    Uses lagged levels as instruments for first-differenced equation.
    """
    print("=" * 70)
    print("Method 3: Difference GMM (Arellano-Bond 1991) - CONSISTENT")
    print("=" * 70)
    print()

    gmm = DifferenceGMM(
        data=df,
        dep_var='y',
        lags=1,
        id_var='id',
        time_var='year',
        exog_vars=['x'],
        time_dummies=False,  # Not needed for synthetic data
        collapse=True,
        two_step=True,
        robust=True
    )

    results = gmm.fit()

    gamma_gmm = results.params['L1.y']
    beta_gmm = results.params['x']
    se_gamma = results.std_errors['L1.y']
    se_beta = results.std_errors['x']

    print(results.summary())
    print()

    return gamma_gmm, beta_gmm, se_gamma, se_beta, results


def compare_results(gamma_true, beta_true,
                   gamma_ols, beta_ols, se_ols_g, se_ols_b,
                   gamma_fe, beta_fe, se_fe_g, se_fe_b,
                   gamma_gmm, beta_gmm, se_gmm_g, se_gmm_b,
                   gmm_results):
    """
    Create comparison table and visualizations.
    """
    print("=" * 70)
    print("COMPARISON OF ESTIMATORS")
    print("=" * 70)
    print()

    # Create comparison table
    comparison = pd.DataFrame({
        'True Value': [gamma_true, beta_true],
        'OLS': [gamma_ols, beta_ols],
        'FE': [gamma_fe, beta_fe],
        'GMM': [gamma_gmm, beta_gmm],
        'OLS SE': [se_ols_g, se_ols_b],
        'FE SE': [se_fe_g, se_fe_b],
        'GMM SE': [se_gmm_g, se_gmm_b]
    }, index=['γ (y_lag)', 'β (x)'])

    print("Coefficient Estimates:")
    print(comparison[['True Value', 'OLS', 'FE', 'GMM']].to_string())
    print()

    # Compute biases
    bias_ols_g = gamma_ols - gamma_true
    bias_fe_g = gamma_fe - gamma_true
    bias_gmm_g = gamma_gmm - gamma_true

    bias_pct_ols = 100 * bias_ols_g / gamma_true
    bias_pct_fe = 100 * bias_fe_g / gamma_true
    bias_pct_gmm = 100 * bias_gmm_g / gamma_true

    print("Bias Analysis (γ coefficient):")
    print(f"  True γ:     {gamma_true:.4f}")
    print()
    print(f"  OLS:        {gamma_ols:.4f}  (bias: {bias_ols_g:+.4f}, {bias_pct_ols:+.1f}%)")
    print(f"  FE:         {gamma_fe:.4f}  (bias: {bias_fe_g:+.4f}, {bias_pct_fe:+.1f}%)")
    print(f"  GMM:        {gamma_gmm:.4f}  (bias: {bias_gmm_g:+.4f}, {bias_pct_gmm:+.1f}%)")
    print()

    # Verify expected pattern
    print("Expected Pattern: OLS > True > GMM > FE")
    print(f"Actual:           {gamma_ols:.3f} > {gamma_true:.3f} {'>'if gamma_gmm < gamma_true else '<?'} {gamma_gmm:.3f} {'>' if gamma_gmm > gamma_fe else '<?'} {gamma_fe:.3f}")
    print()

    if gamma_ols > gamma_true > gamma_gmm > gamma_fe:
        print("✓ Pattern matches theory!")
    else:
        print("⚠ Pattern differs from theory (may happen with small samples)")
    print()

    # Statistical significance
    print("Is bias statistically significant? (2 SE rule)")
    print(f"  OLS:  {abs(bias_ols_g) > 2*se_ols_g} (|bias|={abs(bias_ols_g):.4f}, 2*SE={2*se_ols_g:.4f})")
    print(f"  FE:   {abs(bias_fe_g) > 2*se_fe_g} (|bias|={abs(bias_fe_g):.4f}, 2*SE={2*se_fe_g:.4f})")
    print(f"  GMM:  {abs(bias_gmm_g) > 2*se_gmm_g} (|bias|={abs(bias_gmm_g):.4f}, 2*SE={2*se_gmm_g:.4f})")
    print()

    # GMM diagnostics
    print("=" * 70)
    print("GMM Diagnostic Tests")
    print("=" * 70)
    print()

    print(f"Hansen J-test: p-value = {gmm_results.hansen_j.pvalue:.4f}")
    if 0.10 < gmm_results.hansen_j.pvalue < 0.25:
        print("  ✓ Instruments appear valid")
    elif gmm_results.hansen_j.pvalue < 0.10:
        print("  ✗ Instruments rejected")
    else:
        print("  ⚠ Possible weak instruments")
    print()

    print(f"AR(2) test: p-value = {gmm_results.ar2_test.pvalue if not pd.isna(gmm_results.ar2_test.pvalue) else 'N/A'}")
    if not pd.isna(gmm_results.ar2_test.pvalue):
        if gmm_results.ar2_test.pvalue > 0.10:
            print("  ✓ Moment conditions valid")
        else:
            print("  ✗ Moment conditions rejected")
    else:
        print("  (N/A - insufficient data)")
    print()

    print(f"Instrument ratio: {gmm_results.instrument_ratio:.3f}")
    if gmm_results.instrument_ratio < 0.5:
        print("  ✓ Good instrument count")
    elif gmm_results.instrument_ratio < 1.0:
        print("  ⚠ Moderate instrument count")
    else:
        print("  ✗ Too many instruments")
    print()


def main():
    """Run complete comparison."""
    print("=" * 70)
    print("DYNAMIC PANEL DATA: OLS vs FE vs GMM COMPARISON")
    print("=" * 70)
    print()
    print("This example demonstrates bias in different estimators")
    print("for dynamic panel data with individual fixed effects.")
    print()

    # Generate data
    df, gamma_true, beta_true = generate_dynamic_panel(
        n_groups=200,
        n_periods=10,
        gamma=0.5,
        beta=0.3
    )

    # Estimate with different methods
    gamma_ols, beta_ols, se_ols_g, se_ols_b = estimate_ols(df)
    gamma_fe, beta_fe, se_fe_g, se_fe_b = estimate_fixed_effects(df)
    gamma_gmm, beta_gmm, se_gmm_g, se_gmm_b, gmm_results = estimate_gmm(df)

    # Compare results
    compare_results(
        gamma_true, beta_true,
        gamma_ols, beta_ols, se_ols_g, se_ols_b,
        gamma_fe, beta_fe, se_fe_g, se_fe_b,
        gamma_gmm, beta_gmm, se_gmm_g, se_gmm_b,
        gmm_results
    )

    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. OLS is BIASED UPWARD (ignores fixed effects)")
    print("2. FE is BIASED DOWNWARD (Nickell bias with small T)")
    print("3. GMM is CONSISTENT (uses valid instruments)")
    print()
    print("4. For dynamic panels with fixed effects:")
    print("   → Use GMM (Difference or System)")
    print("   → Do NOT use OLS or standard FE")
    print()
    print("5. Always check GMM diagnostics:")
    print("   → AR(2) test (should not reject)")
    print("   → Hansen J test (0.10 < p < 0.25)")
    print("   → Instrument ratio (< 1.0)")
    print()
    print("=" * 70)
    print("✓ Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
