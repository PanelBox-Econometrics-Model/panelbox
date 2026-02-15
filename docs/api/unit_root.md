# Panel Unit Root Tests API Reference

This document describes the panel unit root testing functionality in PanelBox.

## Overview

Panel unit root tests are used to determine whether panel data series are stationary or contain unit roots. PanelBox implements advanced tests that complement traditional methods like IPS and LLC.

## Quick Start

```python
from panelbox.diagnostics.unit_root import panel_unit_root_test
import pandas as pd

# Load your panel data
data = pd.read_csv('panel_data.csv')

# Run all available tests
result = panel_unit_root_test(
    data,
    variable='gdp',
    entity_col='country',
    time_col='year',
    test='all',
    trend='c'
)

# View results
print(result.summary_table())
```

## Main Functions

### `panel_unit_root_test()`

Run multiple panel unit root tests and compare results.

**Signature:**
```python
panel_unit_root_test(
    data: pd.DataFrame,
    variable: str,
    entity_col: str = 'entity',
    time_col: str = 'time',
    test: Union[str, list] = 'all',
    trend: Literal['c', 'ct'] = 'c',
    alpha: float = 0.05,
    **kwargs
) -> PanelUnitRootResult
```

**Parameters:**

- **data** (*pd.DataFrame*): Panel data in long format
- **variable** (*str*): Name of the variable to test
- **entity_col** (*str*, default='entity'): Column identifying cross-sectional units
- **time_col** (*str*, default='time'): Column identifying time periods
- **test** (*str or list*, default='all'): Which test(s) to run
  - `'all'`: Run all available tests
  - `'hadri'`: Hadri (2000) LM test
  - `'breitung'`: Breitung (2000) test
  - `'ips'`: Im-Pesaran-Shin (2003) test (if available)
  - `'llc'`: Levin-Lin-Chu (2002) test (if available)
  - List of test names: e.g., `['hadri', 'breitung']`
- **trend** (*{'c', 'ct'}*, default='c'): Deterministic specification
  - `'c'`: Constant only
  - `'ct'`: Constant and linear trend
- **alpha** (*float*, default=0.05): Significance level
- **kwargs**: Additional arguments passed to individual test functions

**Returns:**

- **PanelUnitRootResult**: Object containing results from all tests

**Example:**

```python
# Run all tests
result = panel_unit_root_test(df, 'gdp', test='all', trend='c')
print(result.summary_table())

# Run specific tests
result = panel_unit_root_test(
    df, 'gdp',
    test=['hadri', 'breitung'],
    trend='ct'
)
print(result.interpretation())
```

---

### `hadri_test()`

Hadri (2000) LM test for stationarity in panel data.

**Signature:**
```python
hadri_test(
    data: pd.DataFrame,
    variable: str,
    entity_col: str = 'entity',
    time_col: str = 'time',
    trend: Literal['c', 'ct'] = 'c',
    robust: bool = True,
    alpha: float = 0.05
) -> HadriResult
```

**Parameters:**

- **data** (*pd.DataFrame*): Panel data in long format
- **variable** (*str*): Name of the variable to test
- **entity_col** (*str*, default='entity'): Entity identifier column
- **time_col** (*str*, default='time'): Time identifier column
- **trend** (*{'c', 'ct'}*, default='c'): Deterministic specification
- **robust** (*bool*, default=True): Use heteroskedasticity-robust version
- **alpha** (*float*, default=0.05): Significance level

**Returns:**

- **HadriResult**: Test results

**Null Hypothesis:** All series are stationary

**Alternative Hypothesis:** At least one series has a unit root

**Notes:**

This test reverses the typical null hypothesis. A rejection suggests presence of unit roots, while failure to reject supports stationarity.

**Example:**

```python
from panelbox.diagnostics.unit_root import hadri_test

# Test with constant only
result = hadri_test(df, 'gdp', trend='c', robust=True)
print(result.summary())

# Test with constant and trend
result_ct = hadri_test(df, 'gdp', trend='ct', robust=True)
print(f"Statistic: {result_ct.statistic:.4f}")
print(f"P-value: {result_ct.pvalue:.4f}")
```

---

### `breitung_test()`

Breitung (2000) unit root test for panel data.

**Signature:**
```python
breitung_test(
    data: pd.DataFrame,
    variable: str,
    entity_col: str = 'entity',
    time_col: str = 'time',
    trend: Literal['c', 'ct'] = 'ct',
    alpha: float = 0.05
) -> BreitungResult
```

**Parameters:**

- **data** (*pd.DataFrame*): Panel data in long format
- **variable** (*str*): Name of the variable to test
- **entity_col** (*str*, default='entity'): Entity identifier column
- **time_col** (*str*, default='time'): Time identifier column
- **trend** (*{'c', 'ct'}*, default='ct'): Deterministic specification
- **alpha** (*float*, default=0.05): Significance level

**Returns:**

- **BreitungResult**: Test results

**Null Hypothesis:** All series have a unit root

**Alternative Hypothesis:** All series are stationary

**Notes:**

The test is robust to heterogeneity in intercepts and trends. Recommended when heterogeneity across entities is a concern.

**Example:**

```python
from panelbox.diagnostics.unit_root import breitung_test

# Test with constant and trend (recommended)
result = breitung_test(df, 'gdp', trend='ct')
print(result.summary())

# Test with constant only
result_c = breitung_test(df, 'gdp', trend='c')
```

---

## Result Classes

### `PanelUnitRootResult`

Combined results from multiple panel unit root tests.

**Attributes:**

- **results** (*dict*): Dictionary mapping test names to result objects
- **variable** (*str*): Name of the tested variable
- **n_entities** (*int*): Number of cross-sectional units
- **n_time** (*int*): Number of time periods
- **tests_run** (*list*): List of tests that were executed

**Methods:**

#### `summary_table()`

Generate formatted comparison table of all test results.

**Returns:** *str* - Formatted table

**Example:**
```python
result = panel_unit_root_test(df, 'gdp', test='all')
print(result.summary_table())
```

#### `interpretation()`

Provide interpretation of combined test results.

**Returns:** *str* - Interpretation text

**Example:**
```python
result = panel_unit_root_test(df, 'gdp', test='all')
print(result.interpretation())
```

---

### `HadriResult`

Results from Hadri (2000) LM test.

**Attributes:**

- **statistic** (*float*): Z-statistic (standardized LM statistic)
- **pvalue** (*float*): P-value from standard normal distribution
- **reject** (*bool*): Whether to reject H0 at specified alpha level
- **lm_statistic** (*float*): Raw LM statistic before standardization
- **individual_lm** (*np.ndarray*): LM statistic for each entity
- **n_entities** (*int*): Number of entities
- **n_time** (*int*): Number of time periods
- **trend** (*str*): Deterministic specification used
- **robust** (*bool*): Whether robust version was used

**Methods:**

#### `summary()`

Generate formatted summary of test results.

**Returns:** *str* - Formatted summary

**Example:**
```python
result = hadri_test(df, 'gdp')
print(result.summary())
```

---

### `BreitungResult`

Results from Breitung (2000) test.

**Attributes:**

- **statistic** (*float*): Standardized test statistic
- **pvalue** (*float*): P-value from standard normal distribution
- **reject** (*bool*): Whether to reject H0 at specified alpha level
- **raw_statistic** (*float*): Raw test statistic before standardization
- **n_entities** (*int*): Number of entities
- **n_time** (*int*): Number of time periods
- **trend** (*str*): Deterministic specification used

**Methods:**

#### `summary()`

Generate formatted summary of test results.

**Returns:** *str* - Formatted summary

**Example:**
```python
result = breitung_test(df, 'gdp')
print(result.summary())
```

---

## Understanding Null Hypotheses

Different tests have different null hypotheses:

| Test | H0 | H1 | Reject H0 means |
|------|----|----|-----------------|
| **Hadri** | Stationarity | Unit root | Evidence of unit root |
| **Breitung** | Unit root | Stationarity | Evidence of stationarity |
| **IPS** | Unit root | Stationarity | Evidence of stationarity |
| **LLC** | Unit root | Stationarity | Evidence of stationarity |

## Decision Guide

### Interpreting Combined Results

| Unit Root Tests | Hadri Test | Conclusion |
|----------------|------------|------------|
| Reject H0 | Don't reject H0 | **Stationary** - Safe to use levels |
| Don't reject H0 | Reject H0 | **Unit Root** - Consider differencing |
| Mixed results | Mixed results | **Inconclusive** - Additional analysis needed |

### Recommended Workflow

1. **Run all tests:** Use `panel_unit_root_test(test='all')`
2. **Check majority:** Look at the overall recommendation
3. **Verify robustness:** Try different trend specifications
4. **Consider theory:** Combine statistical evidence with economic theory
5. **Make decision:**
   - If stationary: Proceed with level regressions
   - If unit root: Consider differencing or cointegration
   - If mixed: Investigate further or use robust methods

## Common Use Cases

### Testing PPP (Purchasing Power Parity)

```python
# PPP suggests real exchange rates should be stationary
result = panel_unit_root_test(
    data=exchange_rate_data,
    variable='real_exchange_rate',
    entity_col='country',
    time_col='quarter',
    test='all',
    trend='c'
)

print(result.summary_table())

# If stationary: PPP holds in long run
# If unit root: PPP fails (permanent deviations)
```

### Testing Interest Rate Parity

```python
# Interest rate differentials should be stationary
result = panel_unit_root_test(
    data=interest_rate_data,
    variable='interest_differential',
    test=['hadri', 'breitung'],
    trend='c'
)

if not result.results['hadri'].reject:
    print("Evidence of stationarity - Interest rate parity supported")
```

### Pre-test for Panel Regression

```python
# Before running panel regression, test for unit roots
for var in ['gdp', 'investment', 'consumption']:
    result = panel_unit_root_test(df, var, test='all', trend='ct')
    print(f"\n{var}:")
    print(result.interpretation())

# If all stationary: Use levels
# If all unit root: Check for cointegration
# If mixed: Proceed with caution
```

## Troubleshooting

### Unbalanced Panel Error

Both Hadri and Breitung tests require balanced panels.

**Error:**
```
ValueError: Hadri test requires balanced panel (same T for all entities)
```

**Solution:**
```python
# Balance the panel first
df_balanced = df.groupby('entity').filter(lambda x: len(x) == df.groupby('entity').size().mode()[0])

# Then run the test
result = hadri_test(df_balanced, 'y')
```

### Tests Disagree

When tests give conflicting results:

```python
# Try different specifications
result_c = panel_unit_root_test(df, 'y', trend='c')
result_ct = panel_unit_root_test(df, 'y', trend='ct')

# Compare
print("With constant only:")
print(result_c.interpretation())
print("\nWith constant and trend:")
print(result_ct.interpretation())
```

### Low Power in Small Samples

If T or N is small (T < 50 or N < 10), tests may have low power:

```python
# Use multiple tests and check robustness
result = panel_unit_root_test(df, 'y', test='all')

# Look for consistent evidence across tests
print(result.summary_table())
```

## References

1. Hadri, K. (2000). "Testing for Stationarity in Heterogeneous Panel Data." *Econometrics Journal*, 3(2), 148-161.

2. Breitung, J. (2000). "The Local Power of Some Unit Root Tests for Panel Data." In *Advances in Econometrics*, Vol. 15, 161-177.

3. Im, K. S., Pesaran, M. H., & Shin, Y. (2003). "Testing for Unit Roots in Heterogeneous Panels." *Journal of Econometrics*, 115(1), 53-74.

4. Levin, A., Lin, C. F., & Chu, C. S. J. (2002). "Unit Root Tests in Panel Data: Asymptotic and Finite-Sample Properties." *Journal of Econometrics*, 108(1), 1-24.

## See Also

- [Panel Unit Root Tutorial](../tutorials/panel_unit_root.ipynb)
- [Panel Cointegration Tests](cointegration.md)
- [Panel VECM](../tutorials/panel_vecm.ipynb)
