---
title: "Pooled OLS"
description: "Pooled OLS estimator for panel data — baseline regression ignoring entity and time structure."
---

# Pooled OLS

!!! info "Quick Reference"
    **Class:** `panelbox.models.static.pooled_ols.PooledOLS`
    **Import:** `from panelbox import PooledOLS`
    **Stata equivalent:** `reg y x1 x2`
    **R equivalent:** `plm(y ~ x1 + x2, data, model = "pooling")`

## Overview

Pooled OLS is the simplest panel data estimator. It stacks all observations from all entities and time periods into a single dataset and estimates a standard OLS regression, completely ignoring the panel structure. The model is:

$$y_{it} = X_{it} \beta + \varepsilon_{it}$$

where $i$ indexes entities and $t$ indexes time periods. No entity-specific or time-specific effects are included.

Pooled OLS serves primarily as a **baseline model** for comparison with panel-specific estimators like Fixed Effects or Random Effects. If unobserved entity-specific heterogeneity exists and is correlated with the regressors, Pooled OLS produces biased and inconsistent estimates due to omitted variable bias.

In practice, researchers estimate Pooled OLS first, then test whether the panel structure matters by comparing it with Fixed Effects (using an F-test) or Random Effects (using the Breusch-Pagan LM test).

## Quick Example

```python
from panelbox import PooledOLS
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = PooledOLS("invest ~ value + capital", data, "firm", "year")
results = model.fit(cov_type="robust")
print(results.summary())
```

## When to Use

- As a **baseline** or **benchmark** before estimating panel models
- When you believe no unobserved entity-specific heterogeneity exists
- When the panel structure is irrelevant (e.g., pooled cross-sections)
- To compare coefficient magnitudes with FE/RE and detect potential omitted variable bias
- When computing bounds on the true coefficients (Pooled OLS vs FE)

!!! warning "Key Assumptions"
    - **No unobserved heterogeneity**: $E[\varepsilon_{it} | X_{it}] = 0$ (no omitted entity effects)
    - **Linearity**: The conditional expectation of $y$ is linear in $X$
    - **No perfect multicollinearity**: Regressors are not perfectly correlated
    - **If using classical SEs**: Homoskedasticity and no serial correlation within entities

    If unobserved entity effects $\alpha_i$ exist and are correlated with $X_{it}$, Pooled OLS is **biased and inconsistent**. Use [Fixed Effects](fixed-effects.md) instead.

## Detailed Guide

### Data Preparation

PanelBox expects data in **long format** (one row per entity-time observation):

```python
import pandas as pd
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
print(data.head())
#    firm  year  invest  value  capital
# 0     1  1935   317.6  3078.5    2.8
# 1     1  1936   391.8  4661.7   52.6
# ...
```

The data must contain:

- A **dependent variable** and one or more **independent variables**
- An **entity identifier** column (e.g., `"firm"`)
- A **time identifier** column (e.g., `"year"`)

### Estimation

```python
from panelbox import PooledOLS

# Basic estimation with classical standard errors
model = PooledOLS("invest ~ value + capital", data, "firm", "year")
results = model.fit()

# With robust standard errors (recommended)
results_robust = model.fit(cov_type="robust")

# With clustered standard errors by entity
results_cluster = model.fit(cov_type="clustered")

# With Driscoll-Kraay standard errors
results_dk = model.fit(cov_type="driscoll_kraay", max_lags=3)
```

### Interpreting Results

```python
print(results.summary())
```

Key output attributes:

| Attribute | Description |
|-----------|-------------|
| `results.params` | Estimated coefficients (pd.Series) |
| `results.std_errors` | Standard errors (pd.Series) |
| `results.tvalues` | t-statistics (pd.Series) |
| `results.pvalues` | Two-sided p-values (pd.Series) |
| `results.rsquared` | R-squared |
| `results.rsquared_adj` | Adjusted R-squared |
| `results.nobs` | Number of observations |
| `results.conf_int()` | 95% confidence intervals (DataFrame) |
| `results.resid` | Residuals (np.ndarray) |
| `results.fittedvalues` | Fitted values (np.ndarray) |

```python
# Access individual results
print(f"R-squared: {results.rsquared:.4f}")
print(f"Coefficients:\n{results.params}")
print(f"Confidence intervals:\n{results.conf_int()}")
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | str | required | R-style formula (e.g., `"y ~ x1 + x2"`) |
| `data` | DataFrame | required | Panel data in long format |
| `entity_col` | str | required | Entity identifier column name |
| `time_col` | str | required | Time identifier column name |
| `weights` | np.ndarray | `None` | Observation weights for WLS estimation |

**`fit()` method:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cov_type` | str | `"nonrobust"` | Standard error type (see table below) |
| `max_lags` | int | auto | Maximum lags for Driscoll-Kraay / Newey-West |
| `kernel` | str | `"bartlett"` | Kernel for HAC estimators |

## Standard Errors

| `cov_type` | Method | When to Use |
|------------|--------|-------------|
| `"nonrobust"` | Classical OLS | Homoskedastic errors, no autocorrelation |
| `"robust"` / `"hc1"` | White HC1 | Heteroskedasticity of unknown form |
| `"hc0"` | White HC0 | Heteroskedasticity (no small-sample correction) |
| `"hc2"` | HC2 | Heteroskedasticity (leverage-based) |
| `"hc3"` | HC3 | Heteroskedasticity (jackknife-like, conservative) |
| `"clustered"` | Cluster-robust | Within-entity correlation over time |
| `"twoway"` | Two-way clustered | Correlation within entities **and** time periods |
| `"driscoll_kraay"` | Driscoll-Kraay | Cross-sectional dependence + serial correlation |
| `"newey_west"` | Newey-West HAC | Serial correlation in time series |
| `"pcse"` | Panel-corrected (Beck-Katz) | Cross-sectional dependence, T > N |

!!! tip "Recommendation"
    For panel data, always use at least `cov_type="clustered"` to account for within-entity correlation. Classical standard errors are almost always too small for panel data.

## Diagnostics

After estimating Pooled OLS, run these tests to check whether the panel structure matters:

```python
from panelbox import FixedEffects, RandomEffects

# Compare with Fixed Effects
fe = FixedEffects("invest ~ value + capital", data, "firm", "year")
fe_results = fe.fit()

# F-test for entity effects (FE vs Pooled OLS)
# Reported automatically in FE results
print(f"F-statistic: {fe_results.f_statistic:.4f}")
print(f"F-test p-value: {fe_results.f_pvalue:.4f}")
# If p < 0.05 -> entity effects exist -> Pooled OLS is inadequate

# Breusch-Pagan LM test for random effects
from panelbox.validation import BreuschPaganTest
bp = BreuschPaganTest(results)
bp_result = bp.run()
print(bp_result.summary())
```

## Tutorials

| Tutorial | Level | Colab |
|----------|-------|-------|
| Pooled OLS Introduction | Beginner | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/fundamentals/01_pooled_ols_introduction.ipynb) |
| Comparison of All Estimators | Advanced | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/advanced/06_comparison_estimators.ipynb) |

## See Also

- [Fixed Effects](fixed-effects.md) -- Controls for time-invariant unobserved heterogeneity
- [Random Effects](random-effects.md) -- Efficient estimation when effects are uncorrelated with regressors
- [FE vs RE Decision Guide](fe-vs-re.md) -- How to choose between Fixed and Random Effects
- [Between Estimator](between.md) -- Regression on entity means

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapter 10.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 2.
- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press. Chapter 21.
