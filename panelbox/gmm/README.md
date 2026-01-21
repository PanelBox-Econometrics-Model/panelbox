# PanelBox GMM: Generalized Method of Moments for Dynamic Panel Data

Robust implementation of GMM estimators for dynamic panel data models, including Arellano-Bond (1991) Difference GMM and Blundell-Bond (1998) System GMM.

## Table of Contents

- [Introduction](#introduction)
- [When to Use GMM](#when-to-use-gmm)
- [Quick Start](#quick-start)
- [Features](#features)
- [Usage Examples](#usage-examples)
- [Diagnostic Tests](#diagnostic-tests)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Introduction

**Generalized Method of Moments (GMM)** is the standard approach for estimating dynamic panel data models with:
- **Lagged dependent variables** (y_{i,t-1})
- **Fixed effects** (individual heterogeneity)
- **Short time periods** (small T, large N)
- **Potential endogeneity** in regressors

### The Problem

Standard estimators fail with dynamic panels:
- **OLS:** Biased upward (ignores fixed effects)
- **Fixed Effects:** Biased downward (Nickell bias)
- **Random Effects:** Assumes strict exogeneity

### The Solution

GMM uses **moment conditions** with lagged values as instruments to obtain consistent estimates even with:
- Unobserved fixed effects
- Lagged dependent variables
- Endogenous regressors

---

## When to Use GMM

### Use Difference GMM when:
- ✅ You have a **lagged dependent variable**
- ✅ Panel has **large N, small T** (e.g., N > 100, T < 20)
- ✅ **Fixed effects** are present
- ✅ No autocorrelation in **idiosyncratic errors**
- ✅ Some regressors are **not strictly exogenous**

### Use System GMM when:
- ✅ All conditions for Difference GMM, plus:
- ✅ Variables are **highly persistent** (AR coefficient near 1)
- ✅ You have **stationarity** or mean-stationarity
- ✅ Initial conditions are **not correlated with fixed effects**

### Do NOT use GMM when:
- ❌ T is large relative to N (use Fixed Effects instead)
- ❌ No lagged dependent variable (use FE or RE)
- ❌ Strong autocorrelation in errors (moment conditions invalid)

---

## Quick Start

### Installation

```python
# PanelBox should be installed
from panelbox.gmm import DifferenceGMM, SystemGMM
```

### 30-Second Example

```python
import pandas as pd
from panelbox.gmm import DifferenceGMM

# Your panel data
data = pd.DataFrame({
    'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'year': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'y': [2.1, 2.5, 2.8, 3.0, 3.2, 3.5, 1.8, 2.0, 2.3],
    'x': [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 0.9, 1.0, 1.1]
})

# Estimate Difference GMM
gmm = DifferenceGMM(
    data=data,
    dep_var='y',
    lags=1,
    id_var='id',
    time_var='year',
    exog_vars=['x'],
    collapse=True,
    two_step=True,
    robust=True
)

results = gmm.fit()
print(results.summary())
```

---

## Features

### Estimation Methods

- **One-step GMM**
  - Weight matrix: W = (Z'Z)^(-1)
  - Efficient under homoskedasticity
  - Faster computation

- **Two-step GMM**
  - Optimal weight matrix
  - Robust to heteroskedasticity
  - Windmeijer (2005) finite-sample correction

- **Iterative GMM** (Continuously Updated Estimator)
  - Convergent weight matrix
  - Maximum efficiency

### Instrument Generation

- **GMM-style instruments**
  - Separate column per time period
  - Full instrument matrix

- **GMM-style collapsed**
  - One column per lag
  - Avoids instrument proliferation (Roodman 2009)
  - **Recommended for most applications**

- **IV-style instruments**
  - For strictly exogenous variables
  - One column per lag

### Specification Tests

- **Hansen J-test:** Overidentification test (robust)
- **Sargan test:** Overidentification test (non-robust)
- **Arellano-Bond AR(1) test:** Expected autocorrelation
- **Arellano-Bond AR(2) test:** Critical validity test
- **Difference-in-Hansen test:** Test instrument subsets

### Standard Errors

- **Robust:** Heteroskedasticity-consistent (default)
- **Windmeijer correction:** Finite-sample adjustment for two-step
- **Clustered:** By individual (automatic)

---

## Usage Examples

### Basic Difference GMM

```python
from panelbox.gmm import DifferenceGMM

gmm = DifferenceGMM(
    data=df,
    dep_var='y',           # Dependent variable
    lags=1,                # Lags of y to include
    id_var='id',           # Individual identifier
    time_var='year',       # Time identifier
    exog_vars=['x1', 'x2'], # Exogenous variables
    collapse=True,         # Use collapsed instruments (recommended)
    two_step=True,         # Two-step estimation
    robust=True            # Robust standard errors
)

results = gmm.fit()
print(results.summary())
```

### System GMM

```python
from panelbox.gmm import SystemGMM

sys_gmm = SystemGMM(
    data=df,
    dep_var='y',
    lags=1,
    id_var='id',
    time_var='year',
    exog_vars=['x1', 'x2'],
    collapse=True,
    two_step=True,
    robust=True,
    level_instruments={'max_lags': 1}  # Instruments for level equation
)

results = sys_gmm.fit()
print(results.summary())
```

### With Predetermined Variables

```python
gmm = DifferenceGMM(
    data=df,
    dep_var='y',
    lags=1,
    id_var='id',
    time_var='year',
    exog_vars=['x1'],           # Strictly exogenous
    predetermined_vars=['x2'],  # Predetermined (instruments: t-1 and earlier)
    endogenous_vars=['x3'],     # Endogenous (instruments: t-2 and earlier)
    collapse=True,
    two_step=True
)
```

### Accessing Results

```python
results = gmm.fit()

# Coefficients
print(results.params)
print(results.params['L1.y'])  # Lagged dependent variable coefficient

# Standard errors
print(results.std_errors)

# T-statistics and p-values
print(results.tvalues)
print(results.pvalues)

# Specification tests
print(f"Hansen J: {results.hansen_j.statistic:.3f} (p={results.hansen_j.pvalue:.3f})")
print(f"AR(2): {results.ar2_test.statistic:.3f} (p={results.ar2_test.pvalue:.3f})")

# Diagnostics
print(f"Observations: {results.nobs}")
print(f"Instruments: {results.n_instruments}")
print(f"Instrument ratio: {results.instrument_ratio:.3f}")

# LaTeX table
print(results.to_latex())
```

---

## Diagnostic Tests

### Hansen J-test (Overidentification)

**H₀:** Instruments are valid

**Interpretation:**
- **p-value < 0.10:** ❌ Reject instruments (model likely misspecified)
- **0.10 < p-value < 0.25:** ✅ Good (instruments appear valid)
- **p-value > 0.25:** ⚠️ Warning (possible weak instruments)

```python
if 0.10 < results.hansen_j.pvalue < 0.25:
    print("✓ Instruments appear valid")
elif results.hansen_j.pvalue < 0.10:
    print("✗ Instruments rejected - check model specification")
else:
    print("⚠ Possible weak instruments - check instrument relevance")
```

### AR(2) test (Autocorrelation)

**H₀:** No second-order autocorrelation in differenced errors

**Interpretation:**
- **p-value > 0.10:** ✅ Good (moment conditions valid)
- **p-value < 0.10:** ❌ Reject (moment conditions invalid)

**Critical test:** If AR(2) is rejected, moment conditions are invalid and estimates are inconsistent.

```python
if results.ar2_test.pvalue > 0.10:
    print("✓ Moment conditions valid")
else:
    print("✗ AR(2) rejected - moment conditions invalid!")
```

### AR(1) test

**Expected:** Should be **rejected** (p < 0.10)

**Why:** First-differencing induces MA(1) autocorrelation by construction.

### Instrument Ratio

**Rule of Thumb:** `n_instruments / n_groups < 1.0`

- **< 0.5:** ✅ Good
- **0.5 - 1.0:** ⚠️ Acceptable
- **> 1.0:** ❌ Too many instruments (risk of overfitting)

**Solution:** Use `collapse=True` to reduce instruments.

```python
if results.instrument_ratio < 0.5:
    print("✓ Instrument count appropriate")
elif results.instrument_ratio < 1.0:
    print("⚠ Moderate instrument count")
else:
    print("✗ Too many instruments - use collapse=True")
```

---

## Best Practices

### 1. Always Use `collapse=True`

Collapsed instruments avoid proliferation and overfitting:

```python
gmm = DifferenceGMM(..., collapse=True)  # Recommended
```

### 2. Check Diagnostic Tests

Critical checklist:
- [ ] AR(2) p-value > 0.10 ✅
- [ ] Hansen J: 0.10 < p < 0.25 ✅
- [ ] Instrument ratio < 1.0 ✅
- [ ] Coefficient in plausible range ✅

### 3. Compare Difference and System GMM

System GMM should be:
- More efficient (smaller standard errors)
- Coefficient between Difference GMM and OLS

```python
# Estimate both
diff_results = DifferenceGMM(...).fit()
sys_results = SystemGMM(...).fit()

# Compare
print(f"Difference: {diff_results.params['L1.y']:.3f} ({diff_results.std_errors['L1.y']:.3f})")
print(f"System:     {sys_results.params['L1.y']:.3f} ({sys_results.std_errors['L1.y']:.3f})")
```

### 4. Check Coefficient Bounds

**Plausible range:** OLS > System GMM > Difference GMM > FE

If ordering is violated, something is wrong.

### 5. Unbalanced Panels

For unbalanced panels:
- ✅ Use `collapse=True` (essential)
- ⚠️ Avoid many time dummies (prefer trend)
- ⚠️ Keep specifications parsimonious

---

## API Reference

### DifferenceGMM

```python
DifferenceGMM(
    data: pd.DataFrame,
    dep_var: str,
    lags: Union[int, List[int]],
    id_var: str = 'id',
    time_var: str = 'year',
    exog_vars: Optional[List[str]] = None,
    endogenous_vars: Optional[List[str]] = None,
    predetermined_vars: Optional[List[str]] = None,
    time_dummies: bool = True,
    collapse: bool = False,
    two_step: bool = True,
    robust: bool = True,
    gmm_type: str = 'two_step'
)
```

**Parameters:**
- `data`: Panel data (must have id and time variables)
- `dep_var`: Dependent variable name
- `lags`: Lags of dependent variable to include
- `exog_vars`: Strictly exogenous variables
- `predetermined_vars`: Predetermined variables (E[x_it ε_is] = 0 for s ≥ t)
- `endogenous_vars`: Endogenous variables (E[x_it ε_is] ≠ 0)
- `collapse`: Use collapsed instruments (recommended: True)
- `two_step`: Use two-step estimation with Windmeijer correction
- `robust`: Robust standard errors

### SystemGMM

Inherits from `DifferenceGMM` with additional parameter:

```python
level_instruments: Optional[dict] = None
```

Dictionary with:
- `max_lags`: Maximum lags for level instruments (default: 1)

### GMMResults

Returned by `.fit()`, contains:

**Attributes:**
- `params`: Estimated coefficients (pd.Series)
- `std_errors`: Standard errors (pd.Series)
- `tvalues`: T-statistics (pd.Series)
- `pvalues`: P-values (pd.Series)
- `nobs`: Number of observations
- `n_groups`: Number of individuals
- `n_instruments`: Number of instruments
- `instrument_ratio`: n_instruments / n_groups
- `hansen_j`: Hansen J-test result
- `sargan`: Sargan test result
- `ar1_test`: AR(1) test result
- `ar2_test`: AR(2) test result

**Methods:**
- `summary()`: Print formatted results
- `to_latex()`: Export to LaTeX table
- `conf_int(alpha=0.05)`: Confidence intervals

---

## Troubleshooting

### Low Number of Observations

**Problem:** "Number of observations: 14" (expected hundreds)

**Causes:**
1. Too many time dummies with unbalanced panel
2. Missing data in key variables
3. Instrument requirements too strict

**Solutions:**
- Use `time_dummies=False` or trend instead
- Check for missing data: `df.isnull().sum()`
- Use simpler specification
- Ensure `collapse=True`

### Weak Instruments

**Symptoms:**
- Hansen J p-value > 0.50
- Very large standard errors
- Implausible coefficients

**Solutions:**
- Use more lags as instruments
- Check instrument relevance
- Consider System GMM (more instruments)

### AR(2) Test Rejected

**Problem:** AR(2) p-value < 0.10

**Implications:** Moment conditions invalid, estimates inconsistent

**Solutions:**
- Check for autocorrelation in original errors
- Try different lag structure
- Consider different model specification
- Check data quality

### Too Many Instruments

**Problem:** Instrument ratio > 1.0

**Solutions:**
1. Set `collapse=True`
2. Reduce `max_lags` in instrument specification
3. Use fewer time dummies
4. Simplify model

---

## References

### Key Papers

- **Arellano, M., & Bond, S. (1991).** "Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations." *Review of Economic Studies*, 58(2), 277-297.

- **Blundell, R., & Bond, S. (1998).** "Initial Conditions and Moment Restrictions in Dynamic Panel Data Models." *Journal of Econometrics*, 87(1), 115-143.

- **Windmeijer, F. (2005).** "A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators." *Journal of Econometrics*, 126(1), 25-51.

- **Roodman, D. (2009).** "How to Do xtabond2: An Introduction to Difference and System GMM in Stata." *Stata Journal*, 9(1), 86-136.

### Software

- **Stata xtabond2:** https://www.stata.com/
- **pydynpd (Python):** https://github.com/dazhwu/pydynpd
- **plm (R):** https://cran.r-project.org/package=plm

### Further Reading

- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.

---

## Citation

If you use PanelBox GMM in your research, please cite:

```bibtex
@software{panelbox_gmm,
  title = {PanelBox GMM: Generalized Method of Moments for Dynamic Panel Data},
  author = {PanelBox Development Team},
  year = {2026},
  url = {https://github.com/your-repo/panelbox}
}
```

---

## Support

- **Documentation:** See `docs/gmm/` for detailed guides
- **Examples:** See `examples/gmm/` for practical examples
- **Issues:** Report bugs at GitHub Issues
- **Questions:** Stack Overflow with tag `panelbox`

---

**Version:** 0.1.0
**Last Updated:** January 2026
**License:** MIT
