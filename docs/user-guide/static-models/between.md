---
title: "Between Estimator"
description: "Between estimator for panel data — regression on entity means exploiting cross-sectional variation only."
---

# Between Estimator

!!! info "Quick Reference"
    **Class:** `panelbox.models.static.between.BetweenEstimator`
    **Import:** `from panelbox import BetweenEstimator`
    **Stata equivalent:** `xtreg y x1 x2, be`
    **R equivalent:** `plm(y ~ x1 + x2, data, model = "between")`

## Overview

The Between estimator regresses entity-level means of the dependent variable on entity-level means of the regressors. It captures only the **cross-sectional (between-entity)** variation, discarding all time-series (within-entity) variation. The model is:

$$\bar{y}_i = \alpha + \bar{X}_i \beta + \bar{u}_i$$

where bars denote averages over time for each entity $i$. Instead of working with $NT$ panel observations, the Between estimator reduces the data to $N$ entity-level observations.

The Between estimator is the complement of the [Fixed Effects](fixed-effects.md) (Within) estimator. While FE asks "when a firm increases $X$, does $Y$ also change?", the Between estimator asks "do firms with higher average $X$ also have higher average $Y$?" This distinction is important because cross-sectional and within-entity relationships can differ substantially.

The Between estimator is also a building block of the [Random Effects](random-effects.md) estimator, which is a weighted average of the Within and Between estimators.

## Quick Example

```python
from panelbox import BetweenEstimator
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = BetweenEstimator("invest ~ value + capital", data, "firm", "year")
results = model.fit(cov_type="robust")
print(results.summary())
```

## When to Use

- Interest is in **cross-sectional variation** (differences across entities, not changes within)
- As a **diagnostic tool** alongside FE to understand sources of variation
- When **time-invariant regressors** are of primary interest
- As a **component analysis** of the Random Effects estimator
- When T is small relative to N (many entities, few periods)

!!! warning "Key Assumptions"
    - **Between-entity exogeneity**: $E[\bar{u}_i | \bar{X}_i] = 0$
    - Time-invariant unobserved heterogeneity must be uncorrelated with entity-mean regressors
    - Sufficient cross-sectional variation (N > K)

    The Between estimator is **biased** if unobserved entity characteristics are correlated with entity-mean regressors, which is common in practice. Use with caution for causal inference.

## Detailed Guide

### Data Preparation

Same long-format panel data as other estimators:

```python
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
```

### Estimation

```python
from panelbox import BetweenEstimator

model = BetweenEstimator("invest ~ value + capital", data, "firm", "year")
results = model.fit()

# Access entity-level means used in estimation
print(model.entity_means)
```

### Interpreting Results

Key output attributes:

| Attribute | Description |
|-----------|-------------|
| `model.entity_means` | DataFrame of entity-level means for all variables |
| `results.rsquared` | Between R-squared (primary measure) |
| `results.nobs` | Number of entities (N, not NT) |
| `results.df_resid` | N - K degrees of freedom |

```python
print(f"Between R-squared: {results.rsquared:.4f}")
print(f"Number of entities (observations): {results.nobs}")
```

**Comparing with Fixed Effects:**

```python
from panelbox import FixedEffects

fe = FixedEffects("invest ~ value + capital", data, "firm", "year")
fe_results = fe.fit()

print(f"Between coefs: {results.params.to_dict()}")
print(f"Within coefs:  {fe_results.params.to_dict()}")
# Different coefficients suggest different relationships
# across entities vs within entities over time
```

| Estimator | Variation Used | Effective Sample Size | Time-Invariant X |
|-----------|---------------|----------------------|------------------|
| **Between** | Across entities | N | Allowed |
| **Fixed Effects** | Within entities | NT | Dropped |
| **Random Effects** | Both (weighted) | NT | Allowed |
| **Pooled OLS** | Both (unweighted) | NT | Allowed |

## Configuration Options

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | str | required | R-style formula (e.g., `"y ~ x1 + x2"`) |
| `data` | DataFrame | required | Panel data in long format |
| `entity_col` | str | required | Entity identifier column name |
| `time_col` | str | required | Time identifier column name |
| `weights` | np.ndarray | `None` | Observation weights (applied to entity means) |

**`fit()` method:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cov_type` | str | `"nonrobust"` | Standard error type |
| `max_lags` | int | auto | Maximum lags for HAC estimators |
| `kernel` | str | `"bartlett"` | Kernel for HAC estimators |

## Standard Errors

| `cov_type` | Method | When to Use |
|------------|--------|-------------|
| `"nonrobust"` | Classical OLS | Homoskedastic entity-mean errors |
| `"robust"` / `"hc1"` | White HC1 | Heteroskedasticity across entities |
| `"hc0"`, `"hc2"`, `"hc3"` | HC variants | Heteroskedasticity with varying corrections |
| `"clustered"` | Cluster-robust | Custom clustering variable |
| `"driscoll_kraay"` | Driscoll-Kraay | Spatial dependence across entities |
| `"newey_west"` | Newey-West HAC | Serial dependence in entity ordering |

!!! note
    Since the Between estimator operates on N entity-level observations (not NT), the effective sample size is much smaller. Standard errors are naturally larger, and clustered SEs by entity are equivalent to robust SEs.

## Diagnostics

The Between estimator is primarily used as a diagnostic alongside other estimators:

```python
from panelbox import PooledOLS, FixedEffects, RandomEffects, BetweenEstimator

# Estimate all four
pooled_r = PooledOLS("invest ~ value + capital", data, "firm", "year").fit()
fe_r = FixedEffects("invest ~ value + capital", data, "firm", "year").fit()
re_r = RandomEffects("invest ~ value + capital", data, "firm", "year").fit()
be_r = BetweenEstimator("invest ~ value + capital", data, "firm", "year").fit()

# Compare coefficients
import pandas as pd
comparison = pd.DataFrame({
    "Pooled": pooled_r.params,
    "FE (Within)": fe_r.params,
    "RE": re_r.params,
    "Between": be_r.params
})
print(comparison)
```

If Between and Within coefficients diverge substantially, it suggests that the cross-sectional and time-series relationships are different -- a potential signal of omitted variable bias in one or both dimensions.

## Tutorials

| Tutorial | Level | Colab |
|----------|-------|-------|
| First Difference and Between Estimators | Advanced | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/advanced/04_first_difference_between.ipynb) |
| Comparison of All Estimators | Advanced | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/advanced/06_comparison_estimators.ipynb) |

## See Also

- [Fixed Effects](fixed-effects.md) -- Within estimator (uses within-entity variation)
- [Random Effects](random-effects.md) -- GLS estimator (weighted between + within)
- [Pooled OLS](pooled-ols.md) -- Ignores panel structure entirely
- [First Difference](first-difference.md) -- Alternative transformation for removing entity effects

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Section 10.2.2.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 2.
- Hsiao, C. (2014). *Analysis of Panel Data* (3rd ed.). Cambridge University Press. Chapter 3.
