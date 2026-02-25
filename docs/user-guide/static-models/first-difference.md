---
title: "First Difference Estimator"
description: "First Difference estimator for panel data — eliminates fixed effects by differencing consecutive observations."
---

# First Difference Estimator

!!! info "Quick Reference"
    **Class:** `panelbox.models.static.first_difference.FirstDifferenceEstimator`
    **Import:** `from panelbox import FirstDifferenceEstimator`
    **Stata equivalent:** `reg D.y D.x1 D.x2`
    **R equivalent:** `plm(y ~ x1 + x2, data, model = "fd")`

## Overview

The First Difference (FD) estimator eliminates unobserved entity-specific fixed effects by taking differences between consecutive observations rather than demeaning (as in Fixed Effects). The transformation is:

$$\Delta y_{it} = y_{it} - y_{i,t-1} = \Delta X_{it} \beta + \Delta \varepsilon_{it}$$

The entity fixed effect $\alpha_i$ cancels out because it is time-invariant: $\Delta \alpha_i = \alpha_i - \alpha_i = 0$. This provides an alternative to the within transformation used by [Fixed Effects](fixed-effects.md).

When T = 2, FD and FE are **numerically identical**. When T > 2, they generally differ because they weight time periods differently. FD places equal weight on each consecutive pair, while FE weights by the distance from entity means. Under homoskedastic, serially uncorrelated errors, FE is more efficient. However, FD is more robust to serial correlation and is preferred when errors follow a random walk process.

## Quick Example

```python
from panelbox import FirstDifferenceEstimator
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = FirstDifferenceEstimator("invest ~ value + capital", data, "firm", "year")
results = model.fit(cov_type="clustered")
print(results.summary())
```

## When to Use

- As an **alternative to Fixed Effects** when you suspect serial correlation in errors
- When errors follow a **random walk** or AR(1) process (FD is more efficient than FE in this case)
- When T = 2 (FD and FE are equivalent, but FD is simpler)
- When the dependent variable may have a **unit root** (non-stationary in levels)
- When you want to verify FE results: similar coefficients increase confidence; large differences suggest model misspecification

!!! warning "Key Assumptions"
    - **Sequential exogeneity**: $E[\Delta \varepsilon_{it} | \Delta X_{it}] = 0$
    - **No perfect multicollinearity** among differenced regressors
    - **At least T = 2** observations per entity (first period is lost)
    - Time-invariant variables cannot be estimated (absorbed by differencing)

    Differencing induces MA(1) serial correlation in errors even if original errors are i.i.d.: $\text{Cov}(\Delta \varepsilon_{it}, \Delta \varepsilon_{i,t-1}) = -\sigma^2_\varepsilon$. Use `cov_type="clustered"` or `"driscoll_kraay"` to account for this.

## Detailed Guide

### Data Preparation

Data must be in long format. PanelBox handles sorting and differencing internally:

```python
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
```

### Estimation

```python
from panelbox import FirstDifferenceEstimator

model = FirstDifferenceEstimator("invest ~ value + capital", data, "firm", "year")

# Clustered standard errors (recommended)
results = model.fit(cov_type="clustered")

# Driscoll-Kraay (for serial correlation + heteroskedasticity)
results_dk = model.fit(cov_type="driscoll_kraay", max_lags=2)
```

### Interpreting Results

Key attributes specific to First Difference:

| Attribute | Description |
|-----------|-------------|
| `model.n_obs_original` | Number of observations before differencing |
| `model.n_obs_differenced` | Number of observations after differencing |
| `results.nobs` | Same as `n_obs_differenced` |
| `results.n_obs_original` | Original observation count |
| `results.n_obs_dropped` | Number of observations lost to differencing |
| `results.rsquared` | R-squared of the differenced model |

```python
print(f"Original observations: {model.n_obs_original}")
print(f"After differencing: {model.n_obs_differenced}")
print(f"Observations lost: {model.n_obs_original - model.n_obs_differenced}")
print(f"R-squared (differenced): {results.rsquared:.4f}")
```

!!! note "No Intercept"
    The FD estimator does not include an intercept by default. The intercept from the original model is eliminated by differencing (it becomes a constant difference, which is zero). If a trend existed in the original model, it would appear as an intercept in the differenced model.

**Comparing with Fixed Effects:**

```python
from panelbox import FixedEffects

fe = FixedEffects("invest ~ value + capital", data, "firm", "year")
fe_results = fe.fit(cov_type="clustered")

import pandas as pd
comparison = pd.DataFrame({
    "First Difference": results.params,
    "Fixed Effects": fe_results.params
})
print(comparison)
# Similar coefficients -> consistent results
# Different coefficients -> investigate serial correlation / misspecification
```

| Aspect | First Difference | Fixed Effects |
|--------|------------------|---------------|
| Transformation | $y_{it} - y_{i,t-1}$ | $y_{it} - \bar{y}_i$ |
| Observations lost | First period per entity (N) | None |
| Serial correlation | More robust | Problematic with MA(1) in $\Delta \varepsilon$ |
| Efficiency | Less efficient under i.i.d. errors | More efficient under i.i.d. errors |
| Unit roots | Handles well | May be inconsistent |
| T = 2 | Numerically identical to FE | Numerically identical to FD |

## Configuration Options

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | str | required | R-style formula (e.g., `"y ~ x1 + x2"`) |
| `data` | DataFrame | required | Panel data in long format |
| `entity_col` | str | required | Entity identifier column name |
| `time_col` | str | required | Time identifier column name |
| `weights` | np.ndarray | `None` | Observation weights (applied to differenced data) |

**`fit()` method:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cov_type` | str | `"nonrobust"` | Standard error type |
| `max_lags` | int | auto | Maximum lags for HAC estimators |
| `kernel` | str | `"bartlett"` | Kernel for HAC estimators |

## Standard Errors

| `cov_type` | Method | When to Use |
|------------|--------|-------------|
| `"nonrobust"` | Classical OLS | Only if differenced errors are i.i.d. (rare) |
| `"robust"` / `"hc1"` | White HC1 | Heteroskedasticity in differenced errors |
| `"hc0"`, `"hc2"`, `"hc3"` | HC variants | Heteroskedasticity with varying corrections |
| `"clustered"` | Cluster-robust | Within-entity serial correlation (recommended) |
| `"twoway"` | Two-way clustered | Entity + time correlation |
| `"driscoll_kraay"` | Driscoll-Kraay | Serial correlation + cross-sectional dependence |
| `"newey_west"` | Newey-West HAC | Serial correlation |
| `"pcse"` | Panel-corrected | Cross-sectional dependence |

!!! tip "Recommendation"
    Always use `cov_type="clustered"` with the First Difference estimator. Differencing induces negative serial correlation in errors (MA(1) structure), making classical standard errors invalid even if the original errors are i.i.d.

## Diagnostics

```python
# Compare FD and FE coefficients
from panelbox import FirstDifferenceEstimator, FixedEffects

fd_results = FirstDifferenceEstimator("invest ~ value + capital", data, "firm", "year").fit(cov_type="clustered")
fe_results = FixedEffects("invest ~ value + capital", data, "firm", "year").fit(cov_type="clustered")

# If coefficients are similar, both methods are likely valid
# Large differences suggest serial correlation issues

# Test for serial correlation in FD residuals
from panelbox.validation import WooldridgeTest
wooldridge = WooldridgeTest(fd_results)
result = wooldridge.run()
print(result.summary())
```

## Tutorials

| Tutorial | Level | Colab |
|----------|-------|-------|
| First Difference and Between Estimators | Advanced | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/advanced/04_first_difference_between.ipynb) |
| Comparison of All Estimators | Advanced | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/advanced/06_comparison_estimators.ipynb) |

## See Also

- [Fixed Effects](fixed-effects.md) -- Alternative within transformation (demeaning)
- [Between Estimator](between.md) -- Complementary estimator using entity means
- [Pooled OLS](pooled-ols.md) -- Baseline model without differencing

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Section 10.5.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 3.
- Hsiao, C. (2014). *Analysis of Panel Data* (3rd ed.). Cambridge University Press. Chapter 4.
- Anderson, T. W., & Hsiao, C. (1981). "Estimation of Dynamic Models with Error Components." *Journal of the American Statistical Association*, 76(375), 598--606.
