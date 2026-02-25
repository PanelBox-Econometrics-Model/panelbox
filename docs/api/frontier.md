---
title: "Frontier API"
description: "API reference for panelbox.frontier — Stochastic Frontier Analysis, Four-Component SFA, True FE/RE"
---

# Frontier (SFA) API Reference

!!! info "Module"
    **Import**: `from panelbox.frontier import StochasticFrontier, FourComponentSFA, SFResult`
    **Source**: `panelbox/frontier/`

## Overview

The frontier module implements Stochastic Frontier Analysis (SFA) for estimating production and cost frontiers with maximum likelihood estimation:

| Class | Description | Reference |
|-------|-------------|-----------|
| `StochasticFrontier` | Main SFA model with multiple distributions | Aigner et al. (1977) |
| `FourComponentSFA` | Persistent + transient inefficiency decomposition | Kumbhakar et al. (2014) |
| True FE/RE functions | Separate heterogeneity from inefficiency | Greene (2005) |

### Model Structure

**Production frontier**: ln(y_i) = x_i' * beta + v_i - u_i

**Cost frontier**: ln(y_i) = x_i' * beta + v_i + u_i

Where v_i ~ N(0, sigma_v^2) is noise and u_i >= 0 is inefficiency.

## Enumerations

### FrontierType

```python
from panelbox.frontier import FrontierType

FrontierType.PRODUCTION  # Inefficiency reduces output
FrontierType.COST        # Inefficiency increases cost
```

### DistributionType

```python
from panelbox.frontier import DistributionType

DistributionType.HALF_NORMAL       # Half-normal (Aigner et al. 1977)
DistributionType.EXPONENTIAL       # Exponential (Meeusen & van den Broeck 1977)
DistributionType.TRUNCATED_NORMAL  # Truncated normal with location parameter
DistributionType.GAMMA             # Gamma distribution (Greene 1990)
```

### ModelType

```python
from panelbox.frontier import ModelType

ModelType.CROSS_SECTION     # No panel structure
ModelType.POOLED            # Pooled panel
ModelType.PITT_LEE          # Time-invariant inefficiency (1981)
ModelType.BATTESE_COELLI_92 # Time-varying inefficiency (1992)
ModelType.BATTESE_COELLI_95 # With heterogeneity variables (1995)
```

## Classes

### StochasticFrontier

Main model class for Stochastic Frontier Analysis.

#### Constructor

```python
StochasticFrontier(
    data: pd.DataFrame,
    depvar: str,
    exog: list[str],
    entity: str | None = None,
    time: str | None = None,
    frontier: str | FrontierType = "production",
    dist: str | DistributionType = "half_normal",
    inefficiency_vars: list[str] | None = None,
    het_vars: list[str] | None = None,
    model_type: str | ModelType | None = None,
    css_time_trend: str | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | *required* | Panel or cross-section data |
| `depvar` | `str` | *required* | Dependent variable (typically log of output/cost) |
| `exog` | `list[str]` | *required* | Exogenous regressors (inputs) |
| `entity` | `str \| None` | `None` | Entity column (required for panel models) |
| `time` | `str \| None` | `None` | Time column (required for panel models) |
| `frontier` | `str \| FrontierType` | `"production"` | Frontier type: `"production"` or `"cost"` |
| `dist` | `str \| DistributionType` | `"half_normal"` | Inefficiency distribution |
| `inefficiency_vars` | `list[str] \| None` | `None` | Variables affecting mean inefficiency |
| `het_vars` | `list[str] \| None` | `None` | Variables affecting error variance |
| `model_type` | `str \| ModelType \| None` | `None` | Panel model type (auto-detected if None) |
| `css_time_trend` | `str \| None` | `None` | Time trend specification for CSS model |

#### Methods

##### `.fit()`

Estimate the frontier model via maximum likelihood.

```python
result = model.fit()
```

**Returns**: [`SFResult`](#sfresult)

#### Example

```python
from panelbox.frontier import StochasticFrontier

sf = StochasticFrontier(
    data=df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    dist="half_normal",
)
result = sf.fit()
print(result.summary())

# Get efficiency scores
eff = result.efficiency(estimator="bc")
print(f"Mean efficiency: {eff.mean():.4f}")
```

---

### FourComponentSFA

Four-Component SFA model separating persistent and transient inefficiency. Decomposes the error into:

$$y_{it} = x_{it}'\beta + (\mu_i + v_{it}) - (\eta_i + u_{it})$$

Where $\eta_i$ is persistent and $u_{it}$ is transient inefficiency.

#### Constructor

```python
FourComponentSFA(
    data: pd.DataFrame,
    depvar: str,
    exog: list[str],
    entity: str,
    time: str,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | *required* | Panel data |
| `depvar` | `str` | *required* | Dependent variable |
| `exog` | `list[str]` | *required* | Exogenous regressors |
| `entity` | `str` | *required* | Entity column |
| `time` | `str` | *required* | Time column |

#### Example

```python
from panelbox.frontier import FourComponentSFA

fc = FourComponentSFA(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
)
result = fc.fit()

# Decomposed efficiency
persistent = result.persistent_efficiency()
transient = result.transient_efficiency()
overall = result.overall_efficiency()
```

---

## Result Classes

### SFResult

Result container for `StochasticFrontier` estimation.

#### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `np.ndarray` | Estimated coefficients |
| `param_names` | `list` | Parameter names |
| `se` | `np.ndarray` | Standard errors |
| `tvalues` | `np.ndarray` | t-statistics |
| `pvalues` | `np.ndarray` | p-values |
| `loglik` | `float` | Log-likelihood |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |
| `sigma_v` | `float` | Noise standard deviation |
| `sigma_u` | `float` | Inefficiency standard deviation |
| `lambda_param` | `float` | lambda = sigma_u / sigma_v |
| `gamma` | `float` | gamma = sigma_u^2 / (sigma_v^2 + sigma_u^2) |
| `converged` | `bool` | Convergence status |

#### Methods

##### `.efficiency(estimator="bc", ci_level=0.95)`

Compute technical efficiency scores.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | `str` | `"bc"` | Estimator: `"bc"` (Battese-Coelli), `"jlms"` (Jondrow et al.), `"mode"` |
| `ci_level` | `float` | `0.95` | Confidence interval level |

**Returns**: Efficiency scores in (0, 1] for production, [1, inf) for cost.

##### `.summary()`

Print formatted estimation results.

##### `.compare_distributions()`

Compare different distributional assumptions.

### FourComponentResult

Result container for `FourComponentSFA`.

#### Methods

- `.persistent_efficiency()` — Persistent (time-invariant) efficiency
- `.transient_efficiency()` — Transient (time-varying) efficiency
- `.overall_efficiency()` — Overall efficiency (persistent x transient)

---

## True Model Functions

Functions for True Fixed Effects (TFE) and True Random Effects (TRE) models that separate unobserved heterogeneity from inefficiency (Greene 2005).

### loglik_true_fixed_effects

```python
from panelbox.frontier import loglik_true_fixed_effects

ll = loglik_true_fixed_effects(params, y, X, groups, dist="half_normal")
```

### loglik_true_random_effects

```python
from panelbox.frontier import loglik_true_random_effects

ll = loglik_true_random_effects(params, y, X, groups, dist="half_normal")
```

### Bias Correction

```python
from panelbox.frontier import bias_correct_tfe_analytical, bias_correct_tfe_jackknife

# Analytical correction
corrected = bias_correct_tfe_analytical(params, data)

# Jackknife correction
corrected = bias_correct_tfe_jackknife(params, data)
```

### Variance Decomposition

```python
from panelbox.frontier import variance_decomposition_tre

decomp = variance_decomposition_tre(result)
```

---

## Statistical Tests

### hausman_test_tfe_tre

Hausman test for choosing between TFE and TRE models.

```python
from panelbox.frontier import hausman_test_tfe_tre

h_result = hausman_test_tfe_tre(tfe_result, tre_result)
```

### lr_test

Likelihood ratio test for nested frontier models.

```python
from panelbox.frontier import lr_test

lr_result = lr_test(restricted_result, unrestricted_result)
```

### wald_test

Wald test for parameter restrictions.

```python
from panelbox.frontier import wald_test

w_result = wald_test(result, R, r)  # Test R * beta = r
```

### inefficiency_presence_test

Test whether inefficiency is statistically significant (sigma_u > 0).

```python
from panelbox.frontier import inefficiency_presence_test

test = inefficiency_presence_test(result)
```

### skewness_test

Test for skewness in OLS residuals (necessary condition for SFA).

```python
from panelbox.frontier import skewness_test

test = skewness_test(result)
```

### vuong_test

Vuong (1989) test for comparing non-nested frontier models.

```python
from panelbox.frontier import vuong_test

v_result = vuong_test(result1, result2)
```

---

## Utility Functions

### add_translog

Generate translog terms (squares and interactions) for frontier estimation.

```python
from panelbox.frontier import add_translog

df_translog = add_translog(df, variables=["log_labor", "log_capital"])
```

### prepare_panel_index

Set up panel index (entity, time) for frontier data.

```python
from panelbox.frontier import prepare_panel_index

df = prepare_panel_index(df, entity="firm_id", time="year")
```

### validate_frontier_data

Validate data meets SFA requirements.

```python
from panelbox.frontier import validate_frontier_data

validate_frontier_data(df, depvar="log_output", exog=["log_labor", "log_capital"])
```

## See Also

- [Tutorials: Frontier](../tutorials/frontier.md) — Step-by-step SFA guide
- [Theory: Frontier](../theory/sfa-theory.md) — SFA theory and derivations
