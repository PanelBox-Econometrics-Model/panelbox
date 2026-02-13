# Granger Causality Tests for Panel VAR

This module provides comprehensive Granger causality testing for Panel VAR models, including standard Wald tests, the Dumitrescu-Hurlin (2012) test for heterogeneous panels, instantaneous causality tests, and bootstrap inference.

## Overview

Granger causality tests whether past values of one variable help predict another variable. In Panel VAR models, we have several testing approaches:

1. **Standard Wald Test**: Assumes homogeneous coefficients across entities
2. **Dumitrescu-Hurlin Test**: Allows heterogeneous coefficients across entities
3. **Instantaneous Causality**: Tests for contemporaneous correlation
4. **Bootstrap Tests**: Small-sample refinement (coming soon)

## Quick Start

```python
from panelbox.var import PanelVARData, PanelVAR

# Prepare data
data = PanelVARData(
    df,
    endog_vars=['gdp', 'inflation', 'interest_rate'],
    entity_col='country',
    time_col='year',
    lags=2
)

# Estimate Panel VAR
model = PanelVAR(data)
result = model.fit(method='fe', cov_type='robust')

# Test individual causality
gc = result.granger_causality('gdp', 'inflation')
print(gc.summary())

# Full causality matrix
gc_matrix = result.granger_causality_matrix()
print(gc_matrix)

# Instantaneous causality
ic = result.instantaneous_causality('gdp', 'inflation')
print(ic.summary())
```

## Main Functions and Classes

### Panel VAR Result Methods

#### `result.granger_causality(cause, effect)`

Test if `cause` Granger-causes `effect` using Wald test.

**Parameters:**
- `cause` (str): Name of the causing variable
- `effect` (str): Name of the effect variable

**Returns:**
- `GrangerCausalityResult`: Enhanced test result with formatted summary

**Example:**
```python
gc = result.granger_causality('gdp', 'inflation')
print(gc.summary())
# Shows: Wald statistic, F-statistic, p-value, conclusion
```

#### `result.granger_causality_matrix(significance_level=0.05)`

Compute Granger causality matrix for all variable pairs.

**Parameters:**
- `significance_level` (float): Significance level (default: 0.05)

**Returns:**
- `pd.DataFrame`: Matrix of p-values (K × K) where element (i,j) is the p-value for "variable i Granger-causes variable j"

**Example:**
```python
matrix = result.granger_causality_matrix()
print(matrix)
#              gdp    inflation    interest
# gdp          NaN       0.003       0.142
# inflation  0.087       NaN         0.001
# interest   0.234     0.456         NaN
```

#### `result.instantaneous_causality(var1, var2)`

Test for instantaneous (contemporaneous) causality between two variables.

**Parameters:**
- `var1` (str): First variable name
- `var2` (str): Second variable name

**Returns:**
- `InstantaneousCausalityResult`: Test result with correlation and LR statistic

**Example:**
```python
ic = result.instantaneous_causality('gdp', 'inflation')
print(f"Correlation: {ic.correlation:.3f}")
print(f"P-value: {ic.p_value:.4f}")
```

#### `result.instantaneous_causality_matrix()`

Compute instantaneous causality matrix for all variable pairs.

**Returns:**
- `corr_matrix` (pd.DataFrame): Correlation matrix of residuals (K × K)
- `pvalue_matrix` (pd.DataFrame): P-value matrix for LR tests (K × K)

**Example:**
```python
corr, pvals = result.instantaneous_causality_matrix()
print("Correlations:")
print(corr)
print("\nP-values:")
print(pvals)
```

### Standalone Functions

#### `dumitrescu_hurlin_test(data, cause, effect, lags, entity_col='entity', time_col='time')`

Perform Dumitrescu-Hurlin (2012) panel Granger causality test for heterogeneous panels.

**Parameters:**
- `data` (pd.DataFrame): Panel data with entity and time identifiers
- `cause` (str): Name of the causing variable
- `effect` (str): Name of the effect variable
- `lags` (int): Number of lags to test
- `entity_col` (str): Name of entity identifier column (default: 'entity')
- `time_col` (str): Name of time identifier column (default: 'time')

**Returns:**
- `DumitrescuHurlinResult`: Test results including Z̃, Z̄ statistics and individual W_i

**Example:**
```python
from panelbox.var.causality import dumitrescu_hurlin_test

result = dumitrescu_hurlin_test(
    data=df,
    cause='gdp',
    effect='inflation',
    lags=2
)

print(result.summary())
print(f"\nIndividual W statistics: {result.individual_W}")
```

#### `granger_causality_wald(params, cov_params, exog_names, causing_var, caused_var, lags, n_obs=None)`

Low-level function for Granger causality via Wald test.

**Parameters:**
- `params` (np.ndarray): Coefficient vector for the equation
- `cov_params` (np.ndarray): Covariance matrix of parameters
- `exog_names` (List[str]): Names of exogenous variables
- `causing_var` (str): Name of the causing variable
- `caused_var` (str): Name of the caused variable
- `lags` (int): Number of lags in the VAR
- `n_obs` (int, optional): Number of observations (for F-statistic)

**Returns:**
- `GrangerCausalityResult`: Test results

**Notes:**
This is a low-level function. Most users should use `result.granger_causality()` instead.

#### `instantaneous_causality(resid1, resid2, var1, var2)`

Test for instantaneous causality using residuals.

**Parameters:**
- `resid1` (np.ndarray): Residuals from first equation
- `resid2` (np.ndarray): Residuals from second equation
- `var1` (str): Name of first variable
- `var2` (str): Name of second variable

**Returns:**
- `InstantaneousCausalityResult`: Test result

**Notes:**
The test statistic is: LR = -n·log(1 - r²) ~ χ²(1)

### Result Classes

#### `GrangerCausalityResult`

Result container for Granger causality test.

**Attributes:**
- `cause` (str): Name of the causing variable
- `effect` (str): Name of the effect variable
- `wald_stat` (float): Wald test statistic
- `f_stat` (float): F-statistic (Wald/df)
- `df` (int): Degrees of freedom
- `p_value` (float): P-value from chi-squared distribution
- `p_value_f` (float, optional): P-value from F distribution
- `conclusion` (str): Statistical conclusion
- `lags_tested` (int): Number of lags tested

**Methods:**
- `summary()`: Generate formatted summary string

#### `DumitrescuHurlinResult`

Result container for Dumitrescu-Hurlin test.

**Attributes:**
- `cause` (str): Name of the causing variable
- `effect` (str): Name of the effect variable
- `W_bar` (float): Average Wald statistic across entities
- `Z_tilde_stat` (float): Z̃ statistic (for T fixed, N→∞)
- `Z_tilde_pvalue` (float): P-value for Z̃
- `Z_bar_stat` (float): Z̄ statistic (for T→∞, N→∞)
- `Z_bar_pvalue` (float): P-value for Z̄
- `individual_W` (np.ndarray): Individual Wald statistics by entity
- `recommended_stat` (str): Which statistic to use ('Z_tilde' or 'Z_bar')
- `N` (int): Number of entities
- `T_avg` (float): Average time periods per entity
- `lags` (int): Number of lags tested

**Methods:**
- `summary()`: Generate formatted summary
- `plot_individual_statistics(backend='matplotlib', show=True)`: Plot histogram of individual W_i

**Example:**
```python
result = dumitrescu_hurlin_test(df, 'x', 'y', lags=1)

print(result.summary())

# Plot distribution of individual statistics
fig = result.plot_individual_statistics()
```

#### `InstantaneousCausalityResult`

Result container for instantaneous causality test.

**Attributes:**
- `var1` (str): First variable
- `var2` (str): Second variable
- `correlation` (float): Correlation between residuals
- `lr_stat` (float): Likelihood ratio statistic
- `p_value` (float): P-value from χ²(1)
- `n_obs` (int): Number of observations

**Methods:**
- `summary()`: Generate formatted summary

## Theoretical Background

### Granger Causality

Variable x "Granger-causes" variable y if past values of x help predict y beyond what past values of y alone can predict.

**Test Procedure:**

1. Estimate full model: y_t = α + Σ β_l·y_{t-l} + Σ γ_l·x_{t-l} + ε_t
2. Test H₀: γ_1 = γ_2 = ... = γ_p = 0 (all lags of x are zero)
3. Use Wald test: W = (Rβ̂)' [R·V̂(β̂)·R']⁻¹ (Rβ̂) ~ χ²(p)

### Dumitrescu-Hurlin Test

Allows for heterogeneous coefficients across entities.

**Hypotheses:**
- H₀: x does not Granger-cause y for ANY entity (homogeneous non-causality)
- H₁: x Granger-causes y for AT LEAST SOME entities

**Procedure:**

1. For each entity i: estimate individual regression and compute W_i
2. Compute W̄ = (1/N) Σ_i W_i
3. Standardize to get Z statistics:
   - Z̃ = √(N/(2p)) × (W̄ - p) ~ N(0,1) [for T fixed, N→∞]
   - Z̄ = √N × (W̄ - E[W̄]) / √Var[W̄] ~ N(0,1) [for T→∞, N→∞]

**When to use:**
- Use Z̃ for small T (T < 10)
- Use Z̄ for larger T (T ≥ 10)

### Instantaneous Causality

Tests for contemporaneous correlation between variables.

**Test Statistic:**
LR = -n·log(1 - r²) ~ χ²(1)

where r is the correlation between residuals.

**Interpretation:**
- Significant instantaneous causality may indicate:
  - Omitted variables affecting both
  - Simultaneity/reverse causation
  - Need for structural identification

## Interpretation Guidelines

### P-value Interpretation

- **p < 0.01**: Strong evidence of Granger causality (***))
- **p < 0.05**: Moderate evidence of Granger causality (**)
- **p < 0.10**: Weak evidence of Granger causality (*)
- **p ≥ 0.10**: No evidence of Granger causality

### Important Caveats

1. **Granger Causality ≠ True Causation**
   - Tests predictability, not structural causation
   - "x Granger-causes y" means "x helps predict y"
   - Does not imply x actually causes y in the real world

2. **Lag Length Sensitivity**
   - Results can be sensitive to lag length choice
   - Use information criteria (AIC, BIC) to select lags
   - Consider testing multiple lag lengths

3. **Stationarity**
   - Variables should be stationary
   - Use unit root tests before Granger causality
   - Consider cointegration if variables are I(1)

4. **Sample Size**
   - Standard Wald: requires sufficient T for asymptotic properties
   - Dumitrescu-Hurlin: requires N ≥ 10 and T > Kp + 1
   - Bootstrap recommended for small samples (coming soon)

## Comparison: Standard Wald vs Dumitrescu-Hurlin

| Feature | Standard Wald | Dumitrescu-Hurlin |
|---------|---------------|-------------------|
| **Coefficients** | Homogeneous (same for all entities) | Heterogeneous (vary across entities) |
| **H₀** | β = 0 for all entities | β_i = 0 for ALL entities |
| **H₁** | β ≠ 0 for all entities | β_i ≠ 0 for SOME entities |
| **Use when** | Coefficients believed homogeneous | Heterogeneity expected |
| **Sample** | Any N, T | N ≥ 10, T > Kp + 1 |
| **Power** | High if truly homogeneous | Higher if heterogeneous |
| **Output** | Single test statistic | Individual W_i for each entity |

**Recommendation:** Start with Dumitrescu-Hurlin (more general), then examine individual statistics to understand heterogeneity.

## Worked Example

```python
import pandas as pd
from panelbox.var import PanelVARData, PanelVAR

# Load data
df = pd.read_csv('macro_panel.csv')

# Prepare Panel VAR data
data = PanelVARData(
    df,
    endog_vars=['gdp', 'inflation', 'interest'],
    entity_col='country',
    time_col='year',
    lags=2
)

# Estimate
model = PanelVAR(data)
result = model.fit(method='fe', cov_type='robust')

# Test: Does GDP Granger-cause Inflation?
gc = result.granger_causality('gdp', 'inflation')
print(gc.summary())

# Full causality matrix
gc_matrix = result.granger_causality_matrix()

# Visualize significant causalities
significant = gc_matrix < 0.05
print("\nSignificant causalities (p < 0.05):")
for i in gc_matrix.index:
    for j in gc_matrix.columns:
        if i != j and significant.loc[i, j]:
            print(f"  {i} → {j}: p = {gc_matrix.loc[i, j]:.4f}")

# Dumitrescu-Hurlin for heterogeneous panels
from panelbox.var.causality import dumitrescu_hurlin_test

dh = dumitrescu_hurlin_test(
    data=df,
    cause='gdp',
    effect='inflation',
    lags=2
)

print(dh.summary())

# Check which entities have strong causality
import numpy as np
from scipy import stats as sp_stats

critical_value = sp_stats.chi2.ppf(0.95, df=2)
strong_causality = dh.individual_W > critical_value
n_strong = np.sum(strong_causality)

print(f"\n{n_strong} out of {dh.N} entities show significant causality (p < 0.05)")
```

## References

1. Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric Models and Cross-spectral Methods". *Econometrica*, 37(3), 424-438.

2. Dumitrescu, E. I., & Hurlin, C. (2012). "Testing for Granger non-causality in heterogeneous panels". *Economic Modelling*, 29(4), 1450-1460.

3. Holtz-Eakin, D., Newey, W., & Rosen, H. S. (1988). "Estimating vector autoregressions with panel data". *Econometrica*, 1371-1395.

4. Abrigo, M. R., & Love, I. (2016). "Estimation of panel vector autoregression in Stata". *The Stata Journal*, 16(3), 778-804.

## See Also

- [Panel VAR Estimation](var.md)
- [Impulse Response Functions](var_irf.md) (coming soon)
- [Forecast Error Variance Decomposition](var_fevd.md) (coming soon)
- [Bootstrap Inference](var_bootstrap.md) (coming soon)
