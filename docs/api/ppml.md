# PPML (Poisson Pseudo-Maximum Likelihood) API Reference

## Overview

The `panelbox.models.count.PPML` class implements Poisson Pseudo-Maximum Likelihood estimation, particularly useful for gravity models in international trade and other applications with heteroskedasticity and zero values.

## Why PPML?

### Problems with Log-Linear Models

Traditional log-linear gravity models:
```
log(y) = X'β + ε
```

Have serious issues:

1. **Cannot handle zeros**: log(0) undefined
2. **Jensen's inequality**: E[log y] ≠ log E[y]
3. **Heteroskedasticity bias**: E[y|X] ≠ exp(X'β) when errors are heteroskedastic

### PPML Solution

PPML estimates:
```
E[y | X] = exp(X'β + αᵢ)
```

using Poisson MLE.

**Advantages:**
- ✓ Handles zeros naturally
- ✓ Consistent under heteroskedasticity (QML property)
- ✓ No log-transformation bias
- ✓ Correctly specifies E[y|X]
- ✓ Robust standard errors (cluster-robust mandatory)

**Reference:** Santos Silva & Tenreyro (2006), "The Log of Gravity"

## PPML Class

### `PPML(formula, data, entity_col, time_col, fixed_effects=True)`

Poisson Pseudo-Maximum Likelihood estimator for panel data.

**Parameters:**

- `formula` : str
  - Model formula in Wilkinson notation (e.g., "y ~ x1 + x2")
- `data` : pd.DataFrame
  - Panel dataset
- `entity_col` : str
  - Column name for entity identifier
- `time_col` : str
  - Column name for time identifier
- `fixed_effects` : bool, default=True
  - Whether to include entity fixed effects

**Example:**

```python
import pandas as pd
import panelbox as pb

# Load bilateral trade data
df = pd.read_csv('trade_data.csv')
# Columns: exporter, importer, year, trade_flow, distance, gdp_exp, gdp_imp

# Estimate PPML with fixed effects
ppml = pb.PPML(
    "trade_flow ~ log_distance + log_gdp_exp + log_gdp_imp",
    data=df,
    entity_col='pair_id',  # Country-pair identifier
    time_col='year',
    fixed_effects=True
)

result = ppml.fit()
print(result.summary())
```

## PPMLResult

Result class from PPML estimation.

**Attributes:**

- `params` : np.ndarray - Estimated coefficients
- `se` : np.ndarray - Cluster-robust standard errors
- `tvalues` : np.ndarray - t-statistics
- `pvalues` : np.ndarray - p-values
- `llf` : float - Log-likelihood
- `aic` : float - Akaike Information Criterion
- `deviance` : float - Model deviance

**Methods:**

### `elasticity(variable)`

Compute elasticity for a variable.

For PPML with log-transformed variables:
- Elasticity = coefficient (direct interpretation)

For level variables:
- Semi-elasticity = coefficient

**Parameters:**
- `variable` : str or int
  - Variable name or index

**Returns:**
- `elasticity` : float

**Example:**

```python
# For log-transformed variable: elasticity is the coefficient
distance_elasticity = result.elasticity('log_distance')
print(f"Distance elasticity: {distance_elasticity:.3f}")
# Interpretation: 1% increase in distance → distance_elasticity % change in trade

# For level variable: semi-elasticity
if 'border' in result.params:
    border_effect = result.elasticity('border')
    # exp(border_effect) gives multiplicative effect
    print(f"Border effect: {np.exp(border_effect):.3f}x")
```

### `semi_elasticity(variable)`

Compute semi-elasticity for level variables.

**Parameters:**
- `variable` : str or int

**Returns:**
- `semi_elasticity` : float

### `compare_with_ols(ols_result)`

Compare PPML estimates with OLS on log-transformed dependent variable.

**Parameters:**
- `ols_result` : OLS estimation result

**Returns:**
- `comparison` : pd.DataFrame

**Example:**

```python
# Estimate OLS for comparison
ols = pb.PooledOLS(
    "log_trade_flow ~ log_distance + log_gdp_exp + log_gdp_imp",
    data=df[df['trade_flow'] > 0],  # Must drop zeros for OLS
    entity_col='pair_id',
    time_col='year'
)
ols_result = ols.fit(cov_type='clustered')

# Compare
comparison = ppml_result.compare_with_ols(ols_result)
print(comparison)
```

## Gravity Model Application

### Standard Gravity Equation

Trade between countries i and j:

```
Trade_ij = (GDP_i × GDP_j) / Distance_ij^γ × exp(δ Border_ij + ε)
```

Log-linearized:
```
log(Trade_ij) = β₀ + β₁ log(GDP_i) + β₂ log(GDP_j) - γ log(Distance_ij) + δ Border_ij + ε
```

### PPML Specification

```python
# Without zeros problem:
ppml = pb.PPML(
    "trade_flow ~ log_distance + log_gdp_exp + log_gdp_imp + border + language",
    data=df,
    entity_col='pair_id',
    time_col='year',
    fixed_effects=True  # Country-pair FE
)

result = ppml.fit()
```

### Interpreting Coefficients

**Distance elasticity:**
```python
gamma = result.params['log_distance']
print(f"Distance elasticity: {gamma:.3f}")
# E.g., γ = -1.2 → doubling distance reduces trade by 2^(-1.2) ≈ 43%
```

**Binary variable effects:**
```python
border_coef = result.params['border']
border_effect = np.exp(border_coef)
print(f"Border effect: {border_effect:.2f}x")
# E.g., border_effect = 0.7 → border reduces trade by 30%
```

**GDP elasticities:**
```python
gdp_exp_elasticity = result.params['log_gdp_exp']
print(f"Exporter GDP elasticity: {gdp_exp_elasticity:.3f}")
# E.g., 1.1 → 1% increase in exporter GDP → 1.1% increase in trade
```

## Handling Zeros

A major advantage of PPML is natural handling of zeros.

### Example with Zeros

```python
import numpy as np

# Count zeros
n_zeros = (df['trade_flow'] == 0).sum()
pct_zeros = 100 * n_zeros / len(df)
print(f"Zeros in data: {n_zeros} ({pct_zeros:.1f}%)")

# PPML includes zeros
ppml_result = ppml.fit()
print(f"PPML observations: {ppml_result.nobs}")

# OLS must drop zeros
df_positive = df[df['trade_flow'] > 0]
ols = pb.PooledOLS("log_trade_flow ~ log_distance + log_gdp_exp", data=df_positive)
ols_result = ols.fit()
print(f"OLS observations: {ols_result.nobs}")
print(f"Observations lost: {ppml_result.nobs - ols_result.nobs}")
```

### Why Zeros Matter

1. **Selection bias**: Dropping zeros is non-random
2. **Information loss**: Zeros carry information (no trade is informative)
3. **Inconsistent estimates**: OLS estimates biased when zeros dropped

## Heteroskedasticity

PPML is robust to heteroskedasticity (QML consistency).

### Illustration

```python
# PPML: Consistent even with heteroskedasticity
ppml_result = ppml.fit()

# OLS: Inconsistent if heteroskedasticity present
ols_result = ols.fit()

# Compare estimates
comparison = ppml_result.compare_with_ols(ols_result)
print("\nCoefficient Comparison:")
print(comparison)
```

### Monte Carlo Evidence

Santos Silva & Tenreyro (2006) show:
- PPML: Unbiased under heteroskedasticity
- OLS on logs: Severely biased

## Fixed Effects

### Country-Pair Fixed Effects

```python
# Bilateral FE absorb all time-invariant pair characteristics
ppml_fe = pb.PPML(
    "trade_flow ~ rta + currency_union",  # Only time-varying
    data=df,
    entity_col='pair_id',
    time_col='year',
    fixed_effects=True
)
result_fe = ppml_fe.fit()
```

**What FE controls for:**
- Distance (time-invariant)
- Language
- Colonial ties
- Border
- Culture
- Any other time-invariant pair characteristics

### Exporter-Year and Importer-Year FE

For structural gravity (Anderson & van Wincoop):

```python
# Create exporter-year and importer-year identifiers
df['exp_year'] = df['exporter'].astype(str) + '_' + df['year'].astype(str)
df['imp_year'] = df['importer'].astype(str) + '_' + df['year'].astype(str)

# This requires custom implementation with multiple FE
# (not yet fully supported - coming soon)
```

## Standard Errors

PPML automatically uses **cluster-robust standard errors** (mandatory).

### Why Cluster-Robust?

1. **Within-cluster correlation**: Trade flows for same pair correlated over time
2. **Heteroskedasticity**: Variance depends on E[y]
3. **Consistency**: Cluster-robust SEs are consistent

### Clustering

```python
# Clustering by entity (default)
result = ppml.fit()

# Cluster-robust SEs computed automatically
print(result.se)
```

## Diagnostics

### Model Fit

```python
# Deviance
print(f"Deviance: {result.deviance:.2f}")

# AIC
print(f"AIC: {result.aic:.2f}")

# Pseudo R²
print(f"Pseudo R²: {result.pseudo_r2:.4f}")
```

### Residual Analysis

```python
# Pearson residuals
residuals = result.resid_pearson

# Plot residuals
import matplotlib.pyplot as plt
plt.scatter(result.fittedvalues, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Pearson Residuals')
plt.title('PPML Residuals vs. Fitted')
plt.show()
```

### Predicted vs. Actual

```python
# Predictions
y_pred = result.fittedvalues
y_actual = df['trade_flow']

# Correlation
correlation = np.corrcoef(y_pred, y_actual)[0, 1]
print(f"Correlation (predicted vs. actual): {correlation:.4f}")

# Plot
plt.scatter(y_actual, y_pred, alpha=0.3, s=10)
plt.plot([0, y_actual.max()], [0, y_actual.max()], 'r--', lw=2)
plt.xlabel('Actual Trade Flow')
plt.ylabel('Predicted Trade Flow')
plt.title('PPML: Predicted vs. Actual')
plt.show()
```

## Complete Example

```python
import pandas as pd
import numpy as np
import panelbox as pb
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('gravity_data.csv')

# Data preparation
df['log_distance'] = np.log(df['distance'])
df['log_gdp_exp'] = np.log(df['gdp_exporter'])
df['log_gdp_imp'] = np.log(df['gdp_importer'])
df['pair_id'] = df['exporter'] + '_' + df['importer']

# PPML estimation
ppml = pb.PPML(
    "trade ~ log_distance + log_gdp_exp + log_gdp_imp + rta + border",
    data=df,
    entity_col='pair_id',
    time_col='year',
    fixed_effects=True
)

result = ppml.fit()

# Results
print(result.summary())

# Elasticities
print("\nKey Elasticities:")
print(f"  Distance: {result.elasticity('log_distance'):.3f}")
print(f"  Exporter GDP: {result.elasticity('log_gdp_exp'):.3f}")
print(f"  Importer GDP: {result.elasticity('log_gdp_imp'):.3f}")

# RTA effect
rta_coef = result.params['rta']
rta_effect = np.exp(rta_coef) - 1
print(f"  RTA increases trade by: {100*rta_effect:.1f}%")

# Compare with OLS
df_positive = df[df['trade'] > 0]
ols = pb.PooledOLS(
    "log_trade ~ log_distance + log_gdp_exp + log_gdp_imp + rta + border",
    data=df_positive,
    entity_col='pair_id',
    time_col='year'
)
ols_result = ols.fit(cov_type='clustered')

comparison = result.compare_with_ols(ols_result)
print("\nPPML vs. OLS Comparison:")
print(comparison)
```

## When to Use PPML

### Use PPML when:
- ✓ Zeros present in dependent variable
- ✓ Heteroskedasticity suspected
- ✓ Gravity models (trade, FDI, migration)
- ✓ Count data with continuous interpretation
- ✓ Want robust estimates

### Use OLS when:
- All observations positive
- Homoskedasticity (rare in practice)
- Purely log-log relationship

### General Recommendation
**For gravity models: always use PPML** (Santos Silva & Tenreyro, 2006)

## References

**Key Papers:**
1. Santos Silva, J.M.C., & Tenreyro, S. (2006). "The Log of Gravity." *Review of Economics and Statistics*, 88(4), 641-658.
2. Santos Silva, J.M.C., & Tenreyro, S. (2010). "On the Existence of the Maximum Likelihood Estimates in Poisson Regression." *Economics Letters*, 107(2), 310-312.
3. Head, K., & Mayer, T. (2014). "Gravity Equations: Workhorse, Toolkit, and Cookbook." *Handbook of International Economics*, 4, 131-195.

## See Also

- [Tutorial: PPML Gravity Model](../tutorials/ppml_gravity.ipynb)
- [Gravity Models Guide](../guides/gravity_models.md)
- [Count Models](count_models.md)
