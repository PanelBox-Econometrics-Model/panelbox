---
title: "Static Models API"
description: "API reference for panelbox.models.static — PooledOLS, FixedEffects, RandomEffects, Between, FirstDifference"
---

# Static Models API Reference

!!! info "Module"
    **Import**: `from panelbox.models.static import PooledOLS, FixedEffects, RandomEffects, BetweenEstimator, FirstDifferenceEstimator`
    **Source**: `panelbox/models/static/`

## Overview

Static panel models are the workhorses of panel data econometrics. All five estimators share a consistent interface: construct with a formula and data, then call `.fit()` to obtain `PanelResults`.

| Estimator | Description | Use Case |
|-----------|-------------|----------|
| `PooledOLS` | Ordinary least squares ignoring panel structure | Baseline comparison |
| `FixedEffects` | Within estimator eliminating entity-specific intercepts | Time-invariant unobserved heterogeneity |
| `RandomEffects` | GLS with random entity effects | Uncorrelated unobserved effects |
| `BetweenEstimator` | OLS on entity means | Cross-sectional variation |
| `FirstDifferenceEstimator` | OLS on first-differenced data | Alternative to FE for T=2 |

## Common Constructor Pattern

All static models share the same constructor signature:

```python
ModelClass(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    weights: np.ndarray | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | `str` | *required* | R-style formula, e.g. `"y ~ x1 + x2"` |
| `data` | `pd.DataFrame` | *required* | Panel data DataFrame |
| `entity_col` | `str` | *required* | Column identifying entities |
| `time_col` | `str` | *required* | Column identifying time periods |
| `weights` | `np.ndarray \| None` | `None` | Observation weights for WLS |

## Common `.fit()` Method

```python
model.fit(cov_type: str = "nonrobust", **cov_kwds) -> PanelResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cov_type` | `str` | `"nonrobust"` | Covariance estimator type |
| `**cov_kwds` | `dict` | — | Additional keyword arguments for the covariance estimator |

### Available `cov_type` Options

| Value | Description |
|-------|-------------|
| `"nonrobust"` | Classical OLS/GLS standard errors |
| `"robust"` | Heteroskedasticity-robust (HC1) |
| `"hc0"` -- `"hc3"` | White heteroskedasticity-consistent variants |
| `"clustered"` | Cluster-robust by entity (default clustering) |
| `"twoway"` | Two-way clustering by entity and time |
| `"driscoll_kraay"` | Driscoll-Kraay (cross-sectionally robust) |
| `"newey_west"` | Newey-West HAC |
| `"pcse"` | Panel-corrected standard errors (Beck-Katz) |

**Returns**: [`PanelResults`](core.md#panelresults)

---

## Classes

### PooledOLS

Pooled Ordinary Least Squares. Treats all observations as independent, ignoring the panel structure. Useful as a baseline for comparison with panel estimators.

```python
PooledOLS(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    weights: np.ndarray | None = None,
)
```

#### Example

```python
from panelbox import PooledOLS, load_grunfeld

data = load_grunfeld()
model = PooledOLS("invest ~ value + capital", data, "firm", "year")
result = model.fit(cov_type="clustered")
result.summary()
```

---

### FixedEffects

Within estimator that eliminates entity-specific (and optionally time-specific) fixed effects by demeaning.

```python
FixedEffects(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    entity_effects: bool = True,
    time_effects: bool = False,
    weights: np.ndarray | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity_effects` | `bool` | `True` | Include entity fixed effects |
| `time_effects` | `bool` | `False` | Include time fixed effects (two-way FE) |

!!! tip "When to use Fixed Effects"
    Use FE when you suspect unobserved entity-level heterogeneity is **correlated** with the regressors. The Hausman test can help decide between FE and RE.

#### Example

```python
from panelbox import FixedEffects, load_grunfeld

data = load_grunfeld()

# One-way entity FE
fe = FixedEffects("invest ~ value + capital", data, "firm", "year")
result = fe.fit(cov_type="robust")

# Two-way FE (entity + time)
fe2 = FixedEffects(
    "invest ~ value + capital", data, "firm", "year",
    entity_effects=True, time_effects=True
)
result2 = fe2.fit(cov_type="clustered")
```

---

### RandomEffects

GLS estimator with random entity effects. Uses the Swamy-Arora variance decomposition by default.

```python
RandomEffects(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    variance_estimator: str = "swamy-arora",
    weights: np.ndarray | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variance_estimator` | `str` | `"swamy-arora"` | Method for estimating variance components |

!!! tip "When to use Random Effects"
    Use RE when unobserved heterogeneity is **uncorrelated** with the regressors. RE is more efficient than FE under this assumption. Verify with the Hausman test.

#### Example

```python
from panelbox import RandomEffects, load_grunfeld

data = load_grunfeld()
model = RandomEffects("invest ~ value + capital", data, "firm", "year")
result = model.fit()
result.summary()
```

---

### BetweenEstimator

OLS regression on entity means (cross-sectional variation only). Estimates the relationship using between-entity variation by averaging all observations within each entity.

```python
BetweenEstimator(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    weights: np.ndarray | None = None,
)
```

#### Example

```python
from panelbox import BetweenEstimator, load_grunfeld

data = load_grunfeld()
model = BetweenEstimator("invest ~ value + capital", data, "firm", "year")
result = model.fit()
result.summary()
```

---

### FirstDifferenceEstimator

OLS on first-differenced data. Eliminates entity fixed effects by differencing consecutive observations. Equivalent to Fixed Effects when T=2.

```python
FirstDifferenceEstimator(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    weights: np.ndarray | None = None,
)
```

!!! tip "FD vs FE"
    First Difference uses only adjacent-period variation, while FE uses all within-entity variation. FD is more robust to serial correlation in errors but less efficient when errors are not a random walk.

#### Example

```python
from panelbox import FirstDifferenceEstimator, load_grunfeld

data = load_grunfeld()
model = FirstDifferenceEstimator("invest ~ value + capital", data, "firm", "year")
result = model.fit(cov_type="robust")
result.summary()
```

---

## Comparison Example

```python
from panelbox import (
    PooledOLS, FixedEffects, RandomEffects,
    BetweenEstimator, FirstDifferenceEstimator,
    load_grunfeld,
)

data = load_grunfeld()
formula = "invest ~ value + capital"

models = {
    "Pooled OLS": PooledOLS(formula, data, "firm", "year"),
    "Fixed Effects": FixedEffects(formula, data, "firm", "year"),
    "Random Effects": RandomEffects(formula, data, "firm", "year"),
    "Between": BetweenEstimator(formula, data, "firm", "year"),
    "First Difference": FirstDifferenceEstimator(formula, data, "firm", "year"),
}

for name, model in models.items():
    result = model.fit()
    print(f"{name:20s}  R-sq={result.rsquared:.4f}  N={result.nobs}")
```

## See Also

- [Core API](core.md) — `PanelResults` attributes and methods
- [IV API](iv.md) — Instrumental Variables extension
- [Standard Errors](standard-errors.md) — All covariance estimator types
- [Tutorials: Static Models](../tutorials/static-models.md) — Step-by-step guide
- [Validation API](validation.md) — Hausman test for FE vs RE
