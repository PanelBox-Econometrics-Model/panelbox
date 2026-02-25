---
title: "Random Effects"
description: "Random Effects (GLS) estimator for panel data — efficient estimation when entity effects are uncorrelated with regressors."
---

# Random Effects (GLS Estimator)

!!! info "Quick Reference"
    **Class:** `panelbox.models.static.random_effects.RandomEffects`
    **Import:** `from panelbox import RandomEffects`
    **Stata equivalent:** `xtreg y x1 x2, re`
    **R equivalent:** `plm(y ~ x1 + x2, data, model = "random")`

## Overview

The Random Effects (RE) estimator uses Generalized Least Squares (GLS) to efficiently estimate panel models by exploiting the variance component structure. The model is:

$$y_{it} = X_{it} \beta + u_i + \varepsilon_{it}$$

where $u_i \sim \text{i.i.d.}(0, \sigma^2_u)$ is the entity-specific random effect and $\varepsilon_{it} \sim \text{i.i.d.}(0, \sigma^2_\varepsilon)$ is the idiosyncratic error. The GLS transformation applies a **partial demeaning** (quasi-demeaning):

$$y^*_{it} = y_{it} - \theta \bar{y}_i, \quad X^*_{it} = X_{it} - \theta \bar{X}_i$$

where $\theta = 1 - \sqrt{\sigma^2_\varepsilon / (\sigma^2_\varepsilon + T \sigma^2_u)}$ depends on the estimated variance components. When $\theta = 0$, RE reduces to Pooled OLS; when $\theta = 1$, it becomes equivalent to Fixed Effects.

The critical assumption is that entity effects $u_i$ are **uncorrelated** with the regressors: $E[u_i | X_{it}] = 0$. If this assumption holds, RE is more efficient than FE (smaller standard errors). If it fails, RE is **biased and inconsistent** -- use [Fixed Effects](fixed-effects.md) instead. The [Hausman test](fe-vs-re.md) helps decide between the two.

## Quick Example

```python
from panelbox import RandomEffects
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = RandomEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit()
print(results.summary())

# Examine variance components
print(f"Entity variance (sigma2_u): {model.sigma2_u:.4f}")
print(f"Idiosyncratic variance (sigma2_e): {model.sigma2_e:.4f}")
print(f"Theta: {model.theta:.4f}")
```

## When to Use

- Entity-specific effects are **uncorrelated** with regressors: $E[u_i | X_{it}] = 0$
- You need to **estimate time-invariant variables** (e.g., gender, country, industry)
- The sample is a **random draw** from a large population
- You want **more efficient estimates** (smaller standard errors) than Fixed Effects
- The Hausman test does **not reject** the RE specification

!!! warning "Key Assumptions"
    - **Orthogonality**: $E[u_i | X_{it}] = 0$ -- random effects uncorrelated with regressors
    - **Strict exogeneity**: $E[\varepsilon_{it} | X_{i1}, \ldots, X_{iT}, u_i] = 0$
    - **Homoskedastic random effects**: $\text{Var}(u_i) = \sigma^2_u$ (constant across entities)
    - **Independence**: $u_i$ and $\varepsilon_{it}$ are independent

    If $E[u_i | X_{it}] \neq 0$, RE produces **biased estimates**. Always run the [Hausman test](fe-vs-re.md) to verify.

## Detailed Guide

### Data Preparation

Same as other static models -- data in long format with entity and time identifiers:

```python
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
```

### Estimation

```python
from panelbox import RandomEffects

# Default: Swamy-Arora variance estimator
model = RandomEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit()

# With different variance estimator
model_amemiya = RandomEffects(
    "invest ~ value + capital", data, "firm", "year",
    variance_estimator="amemiya"
)
results_amemiya = model_amemiya.fit()

# With robust standard errors
results_robust = model.fit(cov_type="robust")

# With clustered standard errors
results_clustered = model.fit(cov_type="clustered")
```

### Variance Estimators

PanelBox supports four methods for estimating the variance components $\sigma^2_u$ and $\sigma^2_\varepsilon$:

| Estimator | Description |
|-----------|-------------|
| `"swamy-arora"` | Most commonly used (default). Based on within and between residuals. |
| `"walhus"` | Wallace-Hussain estimator |
| `"amemiya"` | Amemiya's alternative estimator |
| `"nerlove"` | Nerlove's estimator |

All produce consistent estimates; differences are typically small in practice.

### Interpreting Results

```python
print(results.summary())
```

Key output attributes specific to Random Effects:

| Attribute | Description |
|-----------|-------------|
| `model.sigma2_u` | Estimated variance of entity effects ($\sigma^2_u$) |
| `model.sigma2_e` | Estimated variance of idiosyncratic errors ($\sigma^2_\varepsilon$) |
| `model.theta` | GLS transformation parameter ($\theta$) |
| `results.rsquared_within` | Within R-squared |
| `results.rsquared_between` | Between R-squared |
| `results.rsquared_overall` | Overall R-squared (primary for RE) |
| `results.params` | Estimated coefficients (includes intercept) |

**Variance components interpretation:**

```python
# Proportion of variance due to entity effects
rho = model.sigma2_u / (model.sigma2_u + model.sigma2_e)
print(f"Proportion due to entity effects (rho): {rho:.2%}")

# Theta close to 1 -> RE behaves like FE
# Theta close to 0 -> RE behaves like Pooled OLS
print(f"Theta: {model.theta:.4f}")
```

**Key differences from FE output:**

- RE estimates include an **intercept** (FE absorbs it into entity effects)
- RE can estimate coefficients on **time-invariant variables**
- RE reports **overall R-squared** as its primary goodness-of-fit measure

## Configuration Options

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | str | required | R-style formula (e.g., `"y ~ x1 + x2"`) |
| `data` | DataFrame | required | Panel data in long format |
| `entity_col` | str | required | Entity identifier column name |
| `time_col` | str | required | Time identifier column name |
| `variance_estimator` | str | `"swamy-arora"` | Method for variance components |
| `weights` | np.ndarray | `None` | Observation weights for WLS |

**`fit()` method:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cov_type` | str | `"nonrobust"` | Standard error type (see table below) |
| `max_lags` | int | auto | Maximum lags for Driscoll-Kraay / Newey-West |
| `kernel` | str | `"bartlett"` | Kernel for HAC estimators |

## Standard Errors

| `cov_type` | Method | When to Use |
|------------|--------|-------------|
| `"nonrobust"` | Classical GLS | Correctly specified model, no heteroskedasticity |
| `"robust"` / `"hc1"` | White HC1 | Heteroskedasticity |
| `"hc0"`, `"hc2"`, `"hc3"` | HC variants | Heteroskedasticity with varying corrections |
| `"clustered"` | Cluster-robust | Within-entity correlation |
| `"twoway"` | Two-way clustered | Correlation within entities and time periods |
| `"driscoll_kraay"` | Driscoll-Kraay | Cross-sectional dependence + serial correlation |
| `"newey_west"` | Newey-West HAC | Serial correlation |

## Diagnostics

After estimating Random Effects, verify the specification:

```python
from panelbox import FixedEffects, RandomEffects
from panelbox.validation import HausmanTest, MundlakTest

# Estimate both models
fe_results = FixedEffects("invest ~ value + capital", data, "firm", "year").fit()
re_results = RandomEffects("invest ~ value + capital", data, "firm", "year").fit()

# 1. Hausman test: FE vs RE
hausman = HausmanTest(fe_results, re_results)
print(f"Hausman statistic: {hausman.statistic:.4f}")
print(f"P-value: {hausman.pvalue:.4f}")
print(f"Recommendation: {hausman.recommendation}")

# 2. Mundlak test: add entity means to RE
mundlak = MundlakTest(re_results)
result = mundlak.run()
print(result.summary())
```

See the [FE vs RE Decision Guide](fe-vs-re.md) for a comprehensive workflow.

## Tutorials

| Tutorial | Level | Colab |
|----------|-------|-------|
| Random Effects and Hausman Test | Beginner | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/fundamentals/03_random_effects_hausman.ipynb) |
| Comparison of All Estimators | Advanced | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/advanced/06_comparison_estimators.ipynb) |

## See Also

- [Fixed Effects](fixed-effects.md) -- Consistent alternative when effects are correlated with regressors
- [FE vs RE Decision Guide](fe-vs-re.md) -- Hausman test and decision workflow
- [Pooled OLS](pooled-ols.md) -- Special case when $\theta = 0$ (no entity effects)
- [Between Estimator](between.md) -- Uses only between-entity variation

## References

- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 2.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapter 10.
- Swamy, P. A. V. B., & Arora, S. S. (1972). "The Exact Finite Sample Properties of the Estimators of Coefficients in the Error Components Regression Models." *Econometrica*, 40(2), 261--275.
- Hausman, J. A. (1978). "Specification Tests in Econometrics." *Econometrica*, 46(6), 1251--1271.
