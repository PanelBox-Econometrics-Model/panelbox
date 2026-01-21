# Complete GMM Tutorial for Dynamic Panel Data

**A Hands-On Guide to Generalized Method of Moments with PanelBox**

---

## Table of Contents

1. [Understanding the Problem](#part-1-understanding-the-problem)
2. [The GMM Solution](#part-2-the-gmm-solution)
3. [Hands-On Example](#part-3-hands-on-example)
4. [Diagnostic Checklist](#part-4-diagnostic-checklist)
5. [Advanced Topics](#part-5-advanced-topics)

---

## Part 1: Understanding the Problem

### The Challenge: Dynamic Panel Data with Fixed Effects

Consider a typical dynamic panel data model:

```
y_it = γ y_{i,t-1} + β x_it + η_i + ε_it
```

Where:
- `y_it`: Dependent variable (e.g., log employment)
- `y_{i,t-1}`: Lagged dependent variable (persistence)
- `x_it`: Exogenous variables (e.g., wages, capital)
- `η_i`: Individual fixed effect (unobserved heterogeneity)
- `ε_it`: Idiosyncratic error term

**The problem:** Standard estimators fail!

### Why OLS Fails (Upward Bias)

Pooled OLS ignores the fixed effect `η_i`, leading to **omitted variable bias**.

**The correlation:**
```
Cov(y_{i,t-1}, η_i) ≠ 0
```

Because `η_i` affects both current and past values of `y`, the lagged dependent variable is correlated with the unobserved effect.

**Result:** OLS coefficient on `y_{i,t-1}` is **biased upward** (too high).

### Why Fixed Effects Fails (Nickell Bias)

The within (FE) estimator removes `η_i` by demeaning:

```
(y_it - ȳ_i) = γ(y_{i,t-1} - ȳ_i) + β(x_it - x̄_i) + (ε_it - ε̄_i)
```

**The problem:**
```
Cov(y_{i,t-1} - ȳ_i, ε_it - ε̄_i) ≠ 0
```

Because ȳ_i includes y_{i,t-1} and ε̄_i includes ε_{i,t-1}, the demeaned lagged variable is correlated with the demeaned error.

**Result:** FE coefficient is **biased downward** (Nickell 1981).

**Magnitude:** Bias ≈ -(1+γ)/(T-1) → Severe when T is small!

### Why Random Effects Fails

RE assumes **strict exogeneity**:
```
E[ε_it | x_i1, ..., x_iT, η_i] = 0 for all t, s
```

With a lagged dependent variable, this is violated because y_{i,t-1} depends on past shocks.

**Result:** RE is also **inconsistent**.

### Summary: The Estimator Ranking

For a dynamic panel with true coefficient γ:

```
OLS > γ_true > System GMM ≈ γ_true > Difference GMM ≈ γ_true > FE
```

**All standard methods fail!** We need GMM.

---

## Part 2: The GMM Solution

### The Key Insight: Moment Conditions

GMM exploits the fact that **past values of y are valid instruments** for the first-differenced equation.

### Difference GMM (Arellano-Bond 1991)

**Step 1: First-Difference to Remove Fixed Effects**

```
Δy_it = γ Δy_{i,t-1} + β Δx_it + Δε_it
```

where `Δy_it = y_it - y_{i,t-1}`.

This eliminates `η_i` without introducing bias (unlike FE demeaning).

**Step 2: Instrument with Lags**

The key moment conditions:
```
E[y_{i,t-s} · Δε_it] = 0  for s ≥ 2
```

**Why?**
- y_{i,t-2} is determined before ε_it occurs
- First-differencing creates MA(1) error structure
- y_{i,t-2} is orthogonal to Δε_it

**Valid instruments:**
- For Δy_{i3}: y_{i1}
- For Δy_{i4}: y_{i1}, y_{i2}
- For Δy_{i5}: y_{i1}, y_{i2}, y_{i3}
- etc.

**GMM-style:** Separate column per instrument (many instruments)
**GMM-style collapsed:** One column per lag depth (fewer instruments, recommended)

### System GMM (Blundell-Bond 1998)

**Problem with Difference GMM:** If y is highly persistent (γ ≈ 1), lagged levels are weak instruments for differences.

**Solution:** Combine two equations:

1. **Difference equation:**
   ```
   Δy_it = γ Δy_{i,t-1} + β Δx_it + Δε_it
   ```
   Instruments: y_{i,t-2}, y_{i,t-3}, ...

2. **Level equation:**
   ```
   y_it = γ y_{i,t-1} + β x_it + η_i + ε_it
   ```
   Instruments: Δy_{i,t-1}, Δx_{i,t-1}

**Additional moment condition:**
```
E[Δy_{i,t-1} · (η_i + ε_it)] = 0
```

Requires: Initial conditions uncorrelated with fixed effects (stationarity).

**Result:** More instruments, more efficient, better with persistent series.

### Choosing Difference vs System GMM

| Criterion | Difference GMM | System GMM |
|-----------|----------------|------------|
| Persistence (γ) | γ < 0.8 | γ ≥ 0.8 |
| Stationarity | Not required | Required |
| Weak instruments | Risk if persistent | Stronger instruments |
| Efficiency | Less efficient | More efficient |
| Robustness | More robust | Relies on stationarity |

**Rule of thumb:** Start with Difference GMM. If instruments are weak (large SEs), try System GMM.

---

## Part 3: Hands-On Example

### Step 1: Load and Prepare Data

```python
import pandas as pd
import numpy as np
from panelbox.gmm import DifferenceGMM, SystemGMM

# Load your panel data
# Required: panel identifier (id), time variable (year), outcome (y), regressors (x1, x2, ...)
df = pd.read_csv('panel_data.csv')

# Check panel structure
print(f"Panels: {df['id'].nunique()}")
print(f"Time periods: {df['year'].nunique()}")
print(f"Observations: {len(df)}")

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic descriptive statistics
print("\nDescriptive statistics:")
print(df[['y', 'x1', 'x2']].describe())
```

### Step 2: Exploratory Analysis

Before GMM, check if you need it:

```python
from sklearn.linear_model import LinearRegression

# Create lagged y
df_sorted = df.sort_values(['id', 'year']).copy()
df_sorted['y_lag'] = df_sorted.groupby('id')['y'].shift(1)
df_clean = df_sorted.dropna()

# Compare OLS and FE estimates (bounds for γ)
# OLS (upper bound)
X_ols = df_clean[['y_lag', 'x1', 'x2']].values
y_ols = df_clean['y'].values
ols = LinearRegression().fit(X_ols, y_ols)
gamma_ols = ols.coef_[0]

# FE (lower bound)
for var in ['y', 'y_lag', 'x1', 'x2']:
    df_clean[f'{var}_dm'] = df_clean[var] - df_clean.groupby('id')[var].transform('mean')

X_fe = df_clean[['y_lag_dm', 'x1_dm', 'x2_dm']].values
y_fe = df_clean['y_dm'].values
fe = LinearRegression(fit_intercept=False).fit(X_fe, y_fe)
gamma_fe = fe.coef_[0]

print(f"Credible range for γ: [{gamma_fe:.3f}, {gamma_ols:.3f}]")
print(f"  FE (lower bound): {gamma_fe:.3f}")
print(f"  OLS (upper bound): {gamma_ols:.3f}")
print(f"  Range width: {gamma_ols - gamma_fe:.3f}")

if gamma_ols - gamma_fe > 0.1:
    print("\n✓ Large gap suggests fixed effects matter - GMM recommended")
else:
    print("\n⚠ Small gap - GMM may not be necessary")
```

### Step 3: Estimate Difference GMM

```python
# Basic Difference GMM specification
gmm_diff = DifferenceGMM(
    data=df,
    dep_var='y',              # Dependent variable
    lags=1,                   # Include y_{t-1}
    id_var='id',              # Panel identifier
    time_var='year',          # Time variable
    exog_vars=['x1', 'x2'],   # Strictly exogenous variables
    time_dummies=True,        # Include time fixed effects
    collapse=True,            # Use collapsed instruments (recommended)
    two_step=True,            # Two-step GMM with Windmeijer correction
    robust=True               # Robust standard errors
)

# Fit the model
results_diff = gmm_diff.fit()

# Print results
print(results_diff.summary())
```

### Step 4: Interpret Results

```python
# Extract key coefficient
gamma_diff = results_diff.params['L1.y']
se_diff = results_diff.std_errors['L1.y']

print(f"\nDifference GMM Results:")
print(f"  γ̂ (lagged y): {gamma_diff:.4f} (SE: {se_diff:.4f})")
print(f"  95% CI: [{gamma_diff - 1.96*se_diff:.4f}, {gamma_diff + 1.96*se_diff:.4f}]")

# Check if in credible range
in_range = gamma_fe < gamma_diff < gamma_ols
print(f"\nCredibility check: {gamma_diff:.3f} in [{gamma_fe:.3f}, {gamma_ols:.3f}]? {in_range}")

if not in_range:
    print("⚠ Warning: Estimate outside credible bounds - check specification")
```

### Step 5: Run Diagnostic Tests

```python
print("\n" + "="*70)
print("DIAGNOSTIC TESTS")
print("="*70)

# 1. Hansen J-test (overidentification)
hansen_j = results_diff.hansen_j
print(f"\n1. Hansen J-test (H0: instruments valid)")
print(f"   Statistic: {hansen_j.statistic:.3f}")
print(f"   p-value: {hansen_j.pvalue:.3f}")

if 0.10 < hansen_j.pvalue < 0.25:
    print("   ✓ PASS: Instruments appear valid")
elif hansen_j.pvalue < 0.10:
    print("   ✗ FAIL: Instruments rejected - check model specification")
else:
    print("   ⚠ WARNING: p-value very high - possible weak instruments")

# 2. AR(2) test (critical!)
ar2 = results_diff.ar2_test
print(f"\n2. AR(2) test (H0: no 2nd-order autocorrelation)")
print(f"   Statistic: {ar2.statistic:.3f}")
print(f"   p-value: {ar2.pvalue:.3f}")

if ar2.pvalue > 0.10:
    print("   ✓ PASS: Moment conditions appear valid")
else:
    print("   ✗ FAIL: Moment conditions rejected - GMM invalid!")

# 3. AR(1) test (should reject)
ar1 = results_diff.ar1_test
print(f"\n3. AR(1) test (H0: no 1st-order autocorrelation)")
print(f"   Statistic: {ar1.statistic:.3f}")
print(f"   p-value: {ar1.pvalue:.3f}")

if ar1.pvalue < 0.10:
    print("   ✓ Expected: First-differencing induces MA(1)")
else:
    print("   ⚠ Unexpected: AR(1) not rejected")

# 4. Instrument count
print(f"\n4. Instrument diagnostics")
print(f"   Observations: {results_diff.nobs}")
print(f"   Groups: {results_diff.n_groups}")
print(f"   Instruments: {results_diff.n_instruments}")
print(f"   Instrument ratio: {results_diff.instrument_ratio:.3f}")

if results_diff.instrument_ratio < 0.5:
    print("   ✓ Good instrument count")
elif results_diff.instrument_ratio < 1.0:
    print("   ⚠ Moderate instrument count - acceptable")
else:
    print("   ✗ Too many instruments - risk of overfitting")
```

### Step 6: Try System GMM

```python
print("\n" + "="*70)
print("SYSTEM GMM (for comparison)")
print("="*70)

gmm_sys = SystemGMM(
    data=df,
    dep_var='y',
    lags=1,
    id_var='id',
    time_var='year',
    exog_vars=['x1', 'x2'],
    time_dummies=True,
    collapse=True,
    two_step=True,
    robust=True,
    level_instruments={'max_lags': 1}  # Use Δy_{t-1} as instrument for levels
)

results_sys = gmm_sys.fit()
print(results_sys.summary())

gamma_sys = results_sys.params['L1.y']
se_sys = results_sys.std_errors['L1.y']

print(f"\nComparison of Estimates:")
print(f"  OLS:            {gamma_ols:.4f} (upper bound)")
print(f"  System GMM:     {gamma_sys:.4f} (SE: {se_sys:.4f})")
print(f"  Difference GMM: {gamma_diff:.4f} (SE: {se_diff:.4f})")
print(f"  FE:             {gamma_fe:.4f} (lower bound)")

# System GMM should be more efficient (smaller SE)
efficiency_gain = (se_diff - se_sys) / se_diff * 100
print(f"\nEfficiency gain: {efficiency_gain:.1f}% reduction in SE")

# Check System GMM diagnostics
print(f"\nSystem GMM Diagnostics:")
print(f"  Hansen J: {results_sys.hansen_j.pvalue:.3f}")
print(f"  AR(2): {results_sys.ar2_test.pvalue:.3f}")
print(f"  Instruments: {results_sys.n_instruments}")
```

### Step 7: Choose Final Model

```python
print("\n" + "="*70)
print("MODEL SELECTION")
print("="*70)

# Decision criteria
diff_valid = (results_diff.ar2_test.pvalue > 0.10 and
              0.10 < results_diff.hansen_j.pvalue < 0.25)
sys_valid = (results_sys.ar2_test.pvalue > 0.10 and
             0.10 < results_sys.hansen_j.pvalue < 0.25)

print("\nDifference GMM:")
print(f"  Valid diagnostics: {diff_valid}")
print(f"  Coefficient: {gamma_diff:.4f} ({se_diff:.4f})")

print("\nSystem GMM:")
print(f"  Valid diagnostics: {sys_valid}")
print(f"  Coefficient: {gamma_sys:.4f} ({se_sys:.4f})")

if diff_valid and sys_valid:
    if se_sys < se_diff * 0.9:
        print("\n✓ RECOMMENDATION: Use System GMM (more efficient)")
        final_results = results_sys
    else:
        print("\n✓ RECOMMENDATION: Use Difference GMM (more robust)")
        final_results = results_diff
elif diff_valid:
    print("\n✓ RECOMMENDATION: Use Difference GMM (System GMM fails diagnostics)")
    final_results = results_diff
elif sys_valid:
    print("\n✓ RECOMMENDATION: Use System GMM (Difference GMM fails diagnostics)")
    final_results = results_sys
else:
    print("\n✗ WARNING: Both models fail diagnostics - check specification")
    final_results = None
```

### Step 8: Report Results

```python
if final_results is not None:
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    # Coefficient table
    coef_table = pd.DataFrame({
        'Coefficient': final_results.params,
        'Std. Error': final_results.std_errors,
        't-stat': final_results.tvalues,
        'p-value': final_results.pvalues
    })
    print("\n", coef_table.to_string())

    # LaTeX export (for papers)
    print("\n" + "="*70)
    print("LATEX OUTPUT")
    print("="*70)
    print(final_results.to_latex())
```

---

## Part 4: Diagnostic Checklist

Use this checklist to validate your GMM results:

### Essential Tests (Must Pass)

- [ ] **AR(2) test p-value > 0.10**
  Critical! If rejected, moment conditions are invalid.

- [ ] **Hansen J-test: 0.10 < p < 0.25**
  If p < 0.10: instruments rejected (try different spec)
  If p > 0.25: possible weak instruments (check relevance)

- [ ] **Instrument ratio < 1.0**
  If > 1.0: too many instruments, use `collapse=True`

- [ ] **Coefficient in credible range**
  Should be between FE (lower) and OLS (upper) bounds

### Recommended Checks

- [ ] **AR(1) test rejected (p < 0.10)**
  Expected due to first-differencing

- [ ] **Standard errors reasonable**
  Very large SEs suggest weak instruments

- [ ] **Number of observations reasonable**
  Large drop from input suggests specification issue

- [ ] **Compare Difference and System GMM**
  System should be more efficient if valid

### Red Flags

- 🚩 AR(2) p-value < 0.05 → Serious problem, model invalid
- 🚩 Hansen J p-value < 0.01 → Model misspecified
- 🚩 Instrument ratio > 2.0 → Severe overfitting risk
- 🚩 Coefficient outside [FE, OLS] range → Check specification
- 🚩 Very few observations retained → Simplify specification

---

## Part 5: Advanced Topics

### Predetermined vs Endogenous Variables

Not all regressors are strictly exogenous. GMM can handle different types:

1. **Strictly exogenous:** E[x_it ε_is] = 0 for all s, t
   Example: weather, policy changes
   Instruments: All lags valid

2. **Predetermined:** E[x_it ε_is] = 0 for s ≥ t
   Example: lagged inputs
   Instruments: t-1 and earlier

3. **Endogenous:** E[x_it ε_is] ≠ 0
   Example: contemporaneous inputs
   Instruments: t-2 and earlier

**Example:**

```python
gmm = DifferenceGMM(
    data=df,
    dep_var='y',
    lags=1,
    id_var='id',
    time_var='year',
    exog_vars=['policy'],           # Strictly exogenous
    predetermined_vars=['capital'],  # Predetermined (t-1 valid)
    endogenous_vars=['labor'],      # Endogenous (need t-2)
    collapse=True,
    two_step=True
)
```

### Handling Unbalanced Panels

Unbalanced panels (missing observations) require special care:

**Recommendations:**

1. **Always use `collapse=True`**
   Reduces instruments and handles missing better

2. **Avoid many time dummies**
   Each dummy increases parameter count
   Use trend or subset of dummies instead

3. **Keep specifications parsimonious**
   More parameters = harder to overidentify with missing data

4. **Check observations retained**
   If very low, simplify specification

**Example for unbalanced panels:**

```python
# Good: Simple specification
gmm = DifferenceGMM(
    data=df_unbalanced,
    dep_var='y',
    lags=1,
    id_var='id',
    time_var='year',
    exog_vars=['x1', 'x2'],
    time_dummies=False,  # Use trend instead
    collapse=True,       # Essential!
    two_step=True
)

# Add linear trend if needed
df_unbalanced['trend'] = df_unbalanced['year'] - df_unbalanced['year'].min()
```

### Instrument Selection Strategies

**Problem:** Too many instruments → overfitting
**Solution:** Limit instrument lags

```python
# Limit maximum lags used as instruments
# Not directly supported yet, but use collapse=True to reduce
```

**Rule of thumb:** Keep `n_instruments / n_groups < 1.0`

### Weak Instruments

**Symptoms:**
- Very large standard errors
- Hansen J p-value > 0.50
- Implausible coefficient estimates

**Solutions:**
1. Try System GMM (additional instruments)
2. Use fewer but more relevant instruments
3. Increase sample size (more groups or periods)
4. Check instrument relevance (first-stage F-test concept)

### Small Sample Corrections

For small N or T:

1. **Use Windmeijer (2005) correction**
   Automatically applied with `two_step=True`

2. **Prefer one-step GMM**
   Less efficient but more robust in small samples
   Set `two_step=False`

3. **Be conservative with instruments**
   Use `collapse=True` always

---

## Summary: Your GMM Workflow

1. **Check if you need GMM**
   Lagged dependent variable + fixed effects + small T

2. **Explore OLS and FE bounds**
   Establishes credible range for coefficients

3. **Start with Difference GMM**
   `collapse=True`, `two_step=True`, `robust=True`

4. **Run full diagnostics**
   AR(2) > 0.10, Hansen J in [0.10, 0.25], ratio < 1.0

5. **Try System GMM if persistent**
   Compare efficiency and diagnostics

6. **Choose best model**
   Valid diagnostics + smaller SE

7. **Report results clearly**
   Include all diagnostic tests

---

## Further Reading

**Essential Papers:**
- Arellano & Bond (1991) - Review of Economic Studies
- Blundell & Bond (1998) - Journal of Econometrics
- Windmeijer (2005) - Journal of Econometrics
- Roodman (2009) - Stata Journal (excellent practical guide)

**Textbooks:**
- Baltagi (2021) - Econometric Analysis of Panel Data
- Wooldridge (2010) - Econometric Analysis of Cross Section and Panel Data

**Software Documentation:**
- Stata xtabond2
- PanelBox README and examples

---

**Tutorial Version:** 1.0
**Last Updated:** January 2026
**Author:** PanelBox Development Team
