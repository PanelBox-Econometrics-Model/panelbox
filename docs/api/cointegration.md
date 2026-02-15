# Panel Cointegration Tests API Reference

Panel cointegration tests for detecting long-run equilibrium relationships in panel data.

## Overview

PanelBox provides three families of panel cointegration tests:

- **Kao (1999)** - DF and ADF-based tests assuming homogeneous cointegrating vector
- **Pedroni (1999)** - 7 residual-based statistics allowing heterogeneity
- **Westerlund (2007)** - 4 ECM-based statistics with bootstrap support

All tests share a common null hypothesis: **no cointegration**.

---

## Kao Tests

```python
from panelbox.diagnostics.cointegration import kao_test
```

### `kao_test()`

Kao (1999) DF and ADF-based panel cointegration tests.

**Parameters:**

- `data` : `pd.DataFrame`
  Panel data in long format

- `entity_col` : `str`
  Name of entity identifier column

- `time_col` : `str`
  Name of time identifier column

- `y_var` : `str`
  Name of dependent variable

- `x_vars` : `str` or `list of str`
  Name(s) of independent variable(s)

- `method` : `{'df', 'adf', 'all'}`, default `'adf'`
  Test method to use:
  - `'df'` - Dickey-Fuller test
  - `'adf'` - Augmented Dickey-Fuller test
  - `'all'` - Both tests

- `trend` : `{'n', 'c', 'ct'}`, default `'c'`
  Deterministic trend specification:
  - `'n'` - No deterministic terms
  - `'c'` - Constant only
  - `'ct'` - Constant and trend

- `lags` : `int`, default `1`
  Number of lags for ADF test

**Returns:**

- `KaoResult` object with attributes:
  - `statistic` : dict - Test statistics
  - `pvalue` : dict - P-values
  - `critical_values` : dict - Critical values at 1%, 5%, 10%
  - `method` : str - Method used
  - `trend` : str - Trend specification
  - `n_entities`, `n_time` : int - Sample dimensions

**Example:**

```python
import pandas as pd
from panelbox.diagnostics.cointegration import kao_test

# Load panel data
data = pd.read_csv('panel_data.csv')

# Run Kao test
result = kao_test(
    data=data,
    entity_col='country',
    time_col='year',
    y_var='gdp',
    x_vars=['investment', 'labor'],
    method='adf',
    trend='c',
    lags=2
)

# View results
print(result.summary())
```

**Notes:**

- Assumes homogeneous cointegrating vector across all entities
- Tends to over-reject in finite samples (use with caution for small T)
- Pooled regression residuals tested for unit root

---

## Pedroni Tests

```python
from panelbox.diagnostics.cointegration import pedroni_test
```

### `pedroni_test()`

Pedroni (1999) residual-based panel cointegration tests.

**Parameters:**

- `data` : `pd.DataFrame`
  Panel data in long format

- `entity_col` : `str`
  Name of entity identifier column

- `time_col` : `str`
  Name of time identifier column

- `y_var` : `str`
  Name of dependent variable

- `x_vars` : `str` or `list of str`
  Name(s) of independent variable(s)

- `method` : `{'all', 'panel_v', 'panel_rho', 'panel_PP', 'panel_ADF', 'group_rho', 'group_PP', 'group_ADF'}`, default `'all'`
  Which statistic(s) to compute:
  - `'all'` - All 7 statistics
  - Individual statistics as listed

- `trend` : `{'n', 'c', 'ct'}`, default `'c'`
  Deterministic trend specification

**Returns:**

- `PedroniResult` object with attributes:
  - `statistic` : dict - All test statistics
  - `pvalue` : dict - P-values for each statistic
  - `critical_values` : dict - Critical values
  - `method` : str - Statistics computed
  - `trend` : str - Trend specification

**Example:**

```python
from panelbox.diagnostics.cointegration import pedroni_test

# Run all Pedroni tests
result = pedroni_test(
    data=data,
    entity_col='country',
    time_col='year',
    y_var='consumption',
    x_vars='income',
    method='all',
    trend='c'
)

# Check which tests reject
for stat, pval in result.pvalue.items():
    reject = "Yes" if pval < 0.05 else "No"
    print(f"{stat}: p={pval:.4f}, Reject={reject}")

# Summary table
print(result.summary())
```

**Statistics Description:**

**Within-dimension (Panel):**
- `panel_v` - Variance ratio (⚠️ may over-reject in finite samples)
- `panel_rho` - Rho-statistic (pooled)
- `panel_PP` - Phillips-Perron (non-parametric)
- `panel_ADF` - Augmented Dickey-Fuller (parametric)

**Between-dimension (Group):**
- `group_rho` - Group mean rho-statistic
- `group_PP` - Group mean Phillips-Perron
- `group_ADF` - Group mean ADF

**Notes:**

- Allows heterogeneous cointegrating vectors (β_i varies across entities)
- `panel_v` has poor finite-sample properties - use with caution
- ADF and PP statistics generally more reliable
- Panel and Group statistics test different alternatives

---

## Westerlund Tests

```python
from panelbox.diagnostics.cointegration import westerlund_test
```

### `westerlund_test()`

Westerlund (2007) error correction-based panel cointegration tests.

**Parameters:**

- `data` : `pd.DataFrame`
  Panel data in long format

- `entity_col` : `str`
  Name of entity identifier column

- `time_col` : `str`
  Name of time identifier column

- `y_var` : `str`
  Name of dependent variable

- `x_vars` : `str` or `list of str`
  Name(s) of independent variable(s)

- `method` : `{'all', 'Gt', 'Ga', 'Pt', 'Pa'}`, default `'all'`
  Which statistic(s) to compute:
  - `'all'` - All 4 statistics
  - `'Gt'` - Group-mean t-statistic
  - `'Ga'` - Group-mean alpha ratio
  - `'Pt'` - Panel pooled t-statistic
  - `'Pa'` - Panel pooled alpha ratio

- `trend` : `{'n', 'c', 'ct'}`, default `'c'`
  Deterministic trend specification

- `lags` : `int` or `{'auto', 'aic', 'bic'}`, default `'auto'`
  Lag selection:
  - Integer - Fixed number of lags
  - `'auto'` or `'aic'` - Automatic selection via AIC
  - `'bic'` - Automatic selection via BIC

- `n_bootstrap` : `int`, default `0`
  Number of bootstrap replications for p-values:
  - `0` - Use tabulated asymptotic critical values
  - `>0` - Bootstrap p-values (more accurate, slower)

- `max_lags` : `int`, optional
  Maximum lags to consider for automatic selection

**Returns:**

- `WesterlundResult` object with attributes:
  - `statistic` : dict - Test statistics
  - `pvalue` : dict - P-values
  - `critical_values` : dict - Critical values
  - `method` : str - Statistics computed
  - `lags` : int or dict - Lags used
  - `n_bootstrap` : int - Bootstrap replications

**Example:**

```python
from panelbox.diagnostics.cointegration import westerlund_test

# Run with automatic lag selection and bootstrap
result = westerlund_test(
    data=data,
    entity_col='country',
    time_col='year',
    y_var='exchange_rate',
    x_vars=['price_domestic', 'price_foreign'],
    method='all',
    trend='c',
    lags='aic',
    n_bootstrap=1000,  # Bootstrap p-values
    max_lags=4
)

# View results
print(result.summary())

# Check individual statistics
for stat_name in ['Gt', 'Ga', 'Pt', 'Pa']:
    print(f"{stat_name}: {result.statistic[stat_name]:.3f} (p={result.pvalue[stat_name]:.4f})")
```

**Statistics Description:**

**Group-mean statistics:**
- `Gt` - Average t-statistic for error correction parameter
- `Ga` - Average alpha ratio statistic
- Tests heterogeneous error correction across entities

**Panel statistics:**
- `Pt` - Pooled t-statistic for error correction
- `Pa` - Pooled alpha ratio
- Tests pooled error correction

**Notes:**

- Based on error correction model (ECM) representation
- More powerful than residual-based tests in many cases
- Bootstrap recommended for better finite-sample properties
- Automatic lag selection can improve power
- `Ga` may have low power in some configurations

---

## Result Objects

All test functions return result objects with common methods:

### `.summary()`

Print formatted summary table of results.

```python
result.summary()
```

### `.reject_at(alpha)`

Check if null hypothesis is rejected at given significance level.

**Parameters:**
- `alpha` : `float`, default `0.05`

**Returns:**
- `bool` or `dict of bool` - Whether to reject for each statistic

```python
# Check rejection at 5%
result.reject_at(0.05)

# Check at 1%
result.reject_at(0.01)
```

---

## Typical Workflow

### Step 1: Check for Unit Roots

Before testing cointegration, verify variables are I(1):

```python
from panelbox.validation.unit_root import ips_test

# Test y for unit root
ips_y = ips_test(data, entity_col='id', time_col='t', var='y')
print(f"Y has unit root: {ips_y.pvalue > 0.05}")

# Test x for unit root
ips_x = ips_test(data, entity_col='id', time_col='t', var='x')
print(f"X has unit root: {ips_x.pvalue > 0.05}")
```

### Step 2: Test for Cointegration

Run multiple tests to triangulate evidence:

```python
from panelbox.diagnostics.cointegration import kao_test, pedroni_test, westerlund_test

# Kao test
kao_result = kao_test(data, 'id', 't', 'y', 'x')

# Pedroni tests
pedroni_result = pedroni_test(data, 'id', 't', 'y', 'x', method='all')

# Westerlund tests
westerlund_result = westerlund_test(
    data, 'id', 't', 'y', 'x',
    lags='aic',
    n_bootstrap=500  # If time permits
)
```

### Step 3: Interpret Results

```python
# Count how many tests reject H0 of no cointegration
tests_rejecting = 0

if list(kao_result.pvalue.values())[0] < 0.05:
    tests_rejecting += 1

pedroni_rejections = sum(1 for p in pedroni_result.pvalue.values() if p < 0.05)
tests_rejecting += (pedroni_rejections >= 4)  # Majority rule

westerlund_rejections = sum(1 for p in westerlund_result.pvalue.values() if p < 0.05)
tests_rejecting += (westerlund_rejections >= 2)

print(f"Evidence for cointegration: {tests_rejecting}/3 test families reject H0")

if tests_rejecting >= 2:
    print("Strong evidence of cointegration")
elif tests_rejecting == 1:
    print("Weak evidence of cointegration")
else:
    print("No evidence of cointegration")
```

### Step 4: Estimate Cointegrating Relationship

If cointegration detected:

```python
from panelbox.var import PVECM

# Estimate panel VECM
vecm = PVECM(data, lags=2, deterministic='ci')
vecm.fit()
print(vecm.summary())
```

---

## Advanced Usage

### Custom Bootstrap for Westerlund

```python
# High-precision bootstrap (slow but accurate)
result = westerlund_test(
    data, 'id', 't', 'y', ['x1', 'x2'],
    n_bootstrap=2000,
    lags=2
)
```

### Multiple Regressors

```python
# Cointegration with multiple I(1) regressors
result = pedroni_test(
    data, 'country', 'year',
    y_var='gdp',
    x_vars=['capital', 'labor', 'technology'],
    trend='ct'  # Include time trend
)
```

### Comparing Trend Specifications

```python
# Test sensitivity to trend specification
for trend_spec in ['n', 'c', 'ct']:
    result = kao_test(data, 'id', 't', 'y', 'x', trend=trend_spec)
    print(f"Trend={trend_spec}: p-value={list(result.pvalue.values())[0]:.4f}")
```

---

## References

1. **Kao, C. (1999).** "Spurious Regression and Residual-Based Tests for Cointegration in Panel Data." *Journal of Econometrics*, 90(1), 1-44.

2. **Pedroni, P. (1999).** "Critical Values for Cointegration Tests in Heterogeneous Panels with Multiple Regressors." *Oxford Bulletin of Economics and Statistics*, 61(S1), 653-670.

3. **Pedroni, P. (2004).** "Panel Cointegration: Asymptotic and Finite Sample Properties of Pooled Time Series Tests with an Application to the PPP Hypothesis." *Econometric Theory*, 20(3), 597-625.

4. **Westerlund, J. (2007).** "Testing for Error Correction in Panel Data." *Oxford Bulletin of Economics and Statistics*, 69(6), 709-748.

---

## See Also

- [Panel Unit Root Tests](unit_root.md) - Test for integration order
- [Panel VECM](var.md#pvecm) - Vector error correction models
- [Validation Report](../../tests/validation/VALIDATION_COINTEGRATION.md) - Monte Carlo validation

---

## Notes and Warnings

⚠️ **Finite Sample Issues:**
- Kao and Pedroni panel_v tend to over-reject in small samples (T < 50)
- Consider bootstrap for Westerlund with N < 30 or T < 50
- Use multiple tests and look for consensus

⚠️ **Cross-Sectional Dependence:**
- Current tests assume cross-sectional independence
- Test for dependence first (e.g., Pesaran CD test)
- If strong dependence, results may be unreliable
- CS-augmented versions planned for future release

⚠️ **Lag Selection:**
- Too few lags: Size distortion
- Too many lags: Power loss
- Use information criteria (AIC/BIC) when uncertain
- Verify robustness to lag choice

✅ **Best Practices:**
1. Always check for unit roots first
2. Run multiple cointegration tests
3. Check sensitivity to trend specification
4. Consider bootstrap for small samples
5. Interpret with economic theory in mind
