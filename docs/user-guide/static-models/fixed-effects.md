---
title: "Fixed Effects"
description: "Fixed Effects (Within) estimator for panel data — controls for time-invariant unobserved heterogeneity through demeaning."
---

# Fixed Effects (Within Estimator)

!!! info "Quick Reference"
    **Class:** `panelbox.models.static.fixed_effects.FixedEffects`
    **Import:** `from panelbox import FixedEffects`
    **Stata equivalent:** `xtreg y x1 x2, fe`
    **R equivalent:** `plm(y ~ x1 + x2, data, model = "within")`

## Overview

The Fixed Effects (FE) estimator, also known as the **Within estimator**, is the workhorse model of panel data econometrics. It controls for all time-invariant unobserved heterogeneity by demeaning each variable within each entity. The model is:

$$y_{it} = \alpha_i + \gamma_t + X_{it} \beta + \varepsilon_{it}$$

where $\alpha_i$ are entity fixed effects and $\gamma_t$ are (optional) time fixed effects. The **within transformation** removes entity effects by subtracting entity means:

$$(y_{it} - \bar{y}_i) = (X_{it} - \bar{X}_i) \beta + (\varepsilon_{it} - \bar{\varepsilon}_i)$$

This is equivalent to including a dummy variable for each entity, but computationally much more efficient. The key advantage is that $\alpha_i$ can be freely correlated with the regressors $X_{it}$, making FE consistent under weaker assumptions than Random Effects.

The trade-off is that Fixed Effects **cannot estimate coefficients on time-invariant variables** (e.g., gender, country, industry) because they are perfectly absorbed by the entity effects. If you need time-invariant coefficients, consider [Random Effects](random-effects.md) instead.

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit(cov_type="clustered")
print(results.summary())
```

## When to Use

- Unobserved entity-specific heterogeneity exists and is **correlated** with regressors
- You want to control for **all time-invariant confounders** without specifying them
- Your entities are not a random sample from a larger population (e.g., specific firms, specific countries)
- You do not need to estimate coefficients on time-invariant variables
- You have at least T = 2 observations per entity

!!! warning "Key Assumptions"
    - **Strict exogeneity**: $E[\varepsilon_{it} | X_{i1}, \ldots, X_{iT}, \alpha_i] = 0$ for all $t$
    - **No perfect multicollinearity** among time-varying regressors
    - **Sufficient within-entity variation**: At least T = 2 per entity
    - Time-invariant variables are **not** of interest (they will be dropped)

    **Strict exogeneity fails** when the model includes lagged dependent variables ($y_{i,t-1}$), leading to Nickell bias. Use [GMM estimators](../gmm/difference-gmm.md) in that case.

## Detailed Guide

### Data Preparation

PanelBox requires data in long format with entity and time identifiers:

```python
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Verify panel structure
print(f"Entities: {data['firm'].nunique()}")
print(f"Time periods: {data['year'].nunique()}")
print(f"Total observations: {len(data)}")
```

!!! note "Time-Invariant Variables"
    If your formula includes variables that do not change over time within an entity, they will be automatically absorbed by the fixed effects and dropped from the estimation. This is by design, not an error.

### Estimation

#### One-Way Fixed Effects (Entity Only)

```python
from panelbox import FixedEffects

# Entity fixed effects (default)
model = FixedEffects(
    "invest ~ value + capital",
    data, "firm", "year",
    entity_effects=True,   # default
    time_effects=False     # default
)
results = model.fit(cov_type="clustered")
```

#### Two-Way Fixed Effects (Entity + Time)

```python
# Two-way fixed effects
model_twoway = FixedEffects(
    "invest ~ value + capital",
    data, "firm", "year",
    entity_effects=True,
    time_effects=True
)
results_twoway = model_twoway.fit(cov_type="twoway")
```

Use two-way FE when there are common time shocks (recessions, policy changes) affecting all entities simultaneously.

#### Time Fixed Effects Only

```python
# Time fixed effects only (removes common time trends)
model_time = FixedEffects(
    "invest ~ value + capital",
    data, "firm", "year",
    entity_effects=False,
    time_effects=True
)
results_time = model_time.fit()
```

### Interpreting Results

```python
print(results.summary())
```

Key output attributes specific to Fixed Effects:

| Attribute | Description |
|-----------|-------------|
| `results.rsquared_within` | R-squared from the within (demeaned) model |
| `results.rsquared_between` | R-squared for entity means |
| `results.rsquared_overall` | Overall R-squared including fixed effects |
| `results.f_statistic` | F-test statistic: FE vs Pooled OLS |
| `results.f_pvalue` | p-value for the F-test |
| `model.entity_fe` | Estimated entity fixed effects (pd.Series) |
| `model.time_fe` | Estimated time fixed effects (pd.Series, if applicable) |

**R-squared interpretation:**

- **Within R-squared** is the primary goodness-of-fit measure for FE. It measures how well the model explains variation *within* each entity over time.
- **Between R-squared** measures how well entity means of fitted values match entity means of the dependent variable.
- **Overall R-squared** combines both sources of variation.

```python
print(f"Within R-squared:  {results.rsquared_within:.4f}")
print(f"Between R-squared: {results.rsquared_between:.4f}")
print(f"Overall R-squared: {results.rsquared_overall:.4f}")
```

**Accessing estimated fixed effects:**

```python
# Entity fixed effects
print(model.entity_fe)
# firm
# 1    -70.297
# 2    101.906
# ...

# Time fixed effects (if time_effects=True)
if model.time_fe is not None:
    print(model.time_fe)
```

**F-test for entity effects (FE vs Pooled OLS):**

The F-test evaluates whether entity fixed effects are jointly significant. A significant F-test (p < 0.05) means Pooled OLS is inadequate and FE is needed.

$$F = \frac{(SSR_{Pooled} - SSR_{FE}) / (N - 1)}{SSR_{FE} / (NT - N - K)}$$

```python
print(f"F-statistic: {results.f_statistic:.4f}")
print(f"F-test p-value: {results.f_pvalue:.4f}")
# p < 0.05 -> reject Pooled OLS in favor of FE
```

## Configuration Options

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | str | required | R-style formula (e.g., `"y ~ x1 + x2"`) |
| `data` | DataFrame | required | Panel data in long format |
| `entity_col` | str | required | Entity identifier column name |
| `time_col` | str | required | Time identifier column name |
| `entity_effects` | bool | `True` | Include entity fixed effects |
| `time_effects` | bool | `False` | Include time fixed effects |
| `weights` | np.ndarray | `None` | Observation weights for WLS |

!!! note
    At least one of `entity_effects` or `time_effects` must be `True`. If you want no fixed effects, use [Pooled OLS](pooled-ols.md) instead.

**`fit()` method:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cov_type` | str | `"nonrobust"` | Standard error type (see table below) |
| `max_lags` | int | auto | Maximum lags for Driscoll-Kraay / Newey-West |
| `kernel` | str | `"bartlett"` | Kernel for HAC estimators |

## Standard Errors

| `cov_type` | Method | When to Use |
|------------|--------|-------------|
| `"nonrobust"` | Classical | Homoskedastic errors, no serial correlation |
| `"robust"` / `"hc1"` | White HC1 | Heteroskedasticity |
| `"hc0"`, `"hc2"`, `"hc3"` | HC variants | Heteroskedasticity with varying corrections |
| `"clustered"` | Cluster-robust | Within-entity correlation (recommended) |
| `"twoway"` | Two-way clustered | Correlation within entities **and** time periods |
| `"driscoll_kraay"` | Driscoll-Kraay | Cross-sectional dependence + serial correlation |
| `"newey_west"` | Newey-West HAC | Serial correlation |
| `"pcse"` | Panel-corrected | Cross-sectional dependence, requires T > N |

!!! tip "Recommendation"
    Always use `cov_type="clustered"` for entity FE models. After demeaning, residuals within the same entity are still correlated, and classical standard errors will be too small. For two-way FE, use `cov_type="twoway"`.

## Diagnostics

After estimating Fixed Effects, consider the following tests:

```python
from panelbox import FixedEffects, RandomEffects

# 1. F-test: FE vs Pooled OLS (automatic in FE results)
fe = FixedEffects("invest ~ value + capital", data, "firm", "year")
fe_results = fe.fit(cov_type="clustered")
print(f"F-test p-value: {fe_results.f_pvalue:.4f}")

# 2. Hausman test: FE vs RE
re = RandomEffects("invest ~ value + capital", data, "firm", "year")
re_results = re.fit()

from panelbox.validation import HausmanTest
hausman = HausmanTest(fe_results, re_results)
print(f"Hausman p-value: {hausman.pvalue:.4f}")
print(f"Recommendation: {hausman.recommendation}")

# 3. Serial correlation test (Wooldridge)
from panelbox.validation import WooldridgeTest
wooldridge = WooldridgeTest(fe_results)
result = wooldridge.run()
print(result.summary())
```

See the [FE vs RE Decision Guide](fe-vs-re.md) for a comprehensive workflow on choosing between Fixed and Random Effects.

## Tutorials

| Tutorial | Level | Colab |
|----------|-------|-------|
| Fixed Effects | Beginner | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/fundamentals/02_fixed_effects.ipynb) |
| Random Effects and Hausman Test | Beginner | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/fundamentals/03_random_effects_hausman.ipynb) |
| Comparison of All Estimators | Advanced | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/advanced/06_comparison_estimators.ipynb) |

## See Also

- [Random Effects](random-effects.md) -- Efficient alternative when effects are uncorrelated with regressors
- [FE vs RE Decision Guide](fe-vs-re.md) -- How to choose between Fixed and Random Effects
- [First Difference](first-difference.md) -- Alternative transformation that eliminates entity effects
- [Between Estimator](between.md) -- Complementary estimator using entity means
- [Pooled OLS](pooled-ols.md) -- Baseline model without fixed effects

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapter 10.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 2.
- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press. Chapter 21.
- Nickell, S. (1981). "Biases in Dynamic Models with Fixed Effects." *Econometrica*, 49(6), 1417--1426.
