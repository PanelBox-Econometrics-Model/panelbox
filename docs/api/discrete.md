---
title: "Discrete Choice API"
description: "API reference for panelbox.models.discrete — Logit, Probit, Multinomial, Ordered, Conditional"
---

# Discrete Choice API Reference

!!! info "Module"
    **Import**: `from panelbox.models.discrete import PooledLogit, PooledProbit, FixedEffectsLogit, MultinomialLogit, OrderedLogit`
    **Source**: `panelbox/models/discrete/`

## Overview

The discrete module provides nonlinear panel models for categorical outcomes:

| Model | Outcome Type | Estimation |
|-------|-------------|------------|
| `PooledLogit` | Binary (0/1) | MLE with cluster-robust SE |
| `PooledProbit` | Binary (0/1) | MLE with cluster-robust SE |
| `FixedEffectsLogit` | Binary (0/1) | Conditional MLE (Chamberlain 1980) |
| `RandomEffectsProbit` | Binary (0/1) | Gauss-Hermite quadrature |
| `MultinomialLogit` | Unordered categorical (J > 2) | MLE |
| `ConditionalLogit` | Choice among alternatives | Conditional MLE (McFadden 1974) |
| `OrderedLogit` | Ordered categorical | MLE |
| `OrderedProbit` | Ordered categorical | MLE |
| `RandomEffectsOrderedLogit` | Ordered categorical | RE via quadrature |

## Classes

### PooledLogit

Pooled logistic regression with cluster-robust standard errors.

#### Constructor

```python
PooledLogit(
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
| `data` | `pd.DataFrame` | *required* | Panel data |
| `entity_col` | `str` | *required* | Entity column |
| `time_col` | `str` | *required* | Time column |
| `weights` | `np.ndarray \| None` | `None` | Observation weights |

#### Methods

##### `.fit()`

```python
def fit(
    self,
    cov_type: Literal["nonrobust", "robust", "cluster"] = "cluster",
    **kwargs,
) -> PanelResults
```

#### Example

```python
from panelbox import PooledLogit

logit = PooledLogit("lfp ~ age + educ + kids", data, "id", "year")
result = logit.fit(cov_type="cluster")
result.summary()
```

---

### PooledProbit

Pooled probit regression. Same interface as `PooledLogit` but uses the normal CDF link function.

```python
PooledProbit(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    weights: np.ndarray | None = None,
)
```

---

### FixedEffectsLogit

Fixed Effects Logit using Chamberlain (1980) conditional maximum likelihood. Eliminates entity-specific effects by conditioning on the sufficient statistic.

```python
FixedEffectsLogit(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    weights: np.ndarray | None = None,
)
```

!!! warning "Dropped entities"
    Entities with no variation in the dependent variable (all 0s or all 1s) are automatically dropped, as they contribute no information to the conditional likelihood.

#### Example

```python
from panelbox import FixedEffectsLogit

fe_logit = FixedEffectsLogit("lfp ~ exper + kidslt6", data, "id", "year")
result = fe_logit.fit()
result.summary()
```

---

### RandomEffectsProbit

Random Effects probit using Gauss-Hermite quadrature for integration over the random effects distribution.

```python
RandomEffectsProbit(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    weights: np.ndarray | None = None,
)
```

---

### MultinomialLogit

Multinomial logit for unordered categorical outcomes with J > 2 alternatives.

#### Constructor

```python
MultinomialLogit(
    endog: np.ndarray | pd.Series,
    exog: np.ndarray | pd.DataFrame,
    n_alternatives: int | None = None,
    base_alternative: int = 0,
    method: str = "pooled",
    entity_col: str | None = None,
    time_col: str | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `np.ndarray \| pd.Series` | *required* | Categorical outcome (integer-coded) |
| `exog` | `np.ndarray \| pd.DataFrame` | *required* | Independent variables |
| `n_alternatives` | `int \| None` | `None` | Number of alternatives (auto-detected) |
| `base_alternative` | `int` | `0` | Base (reference) category |
| `method` | `str` | `"pooled"` | Estimation: `"pooled"`, `"fe"`, or `"re"` |
| `entity_col` | `str \| None` | `None` | Entity column (for panel methods) |
| `time_col` | `str \| None` | `None` | Time column |

#### Methods

- `.fit()` -> `MultinomialLogitResult`
- `.predict_proba()` — Predicted probabilities for all alternatives
- `.predict()` — Most likely alternative
- `.marginal_effects(at="mean", variable=None)` — Average marginal effects

!!! tip "Interpreting coefficients"
    Coefficients represent **log-odds relative to the base category**. Use `.marginal_effects()` for more interpretable results, as marginal effects give the change in probability for each alternative.

#### Example

```python
from panelbox.models.discrete import MultinomialLogit

mlogit = MultinomialLogit(
    endog=df["occupation"],
    exog=df[["education", "experience", "age"]],
    base_alternative=0,
    method="pooled",
)
result = mlogit.fit()
result.summary()

# Marginal effects
me = result.marginal_effects(at="mean")
```

---

### ConditionalLogit

McFadden (1974) conditional logit for discrete choice among alternatives with alternative-varying attributes.

```python
ConditionalLogit(
    endog: np.ndarray | pd.Series,
    exog: np.ndarray | pd.DataFrame,
    n_alternatives: int | None = None,
    base_alternative: int = 0,
    method: str = "pooled",
    entity_col: str | None = None,
    time_col: str | None = None,
)
```

**Returns**: `ConditionalLogitResult`

---

### OrderedLogit

Ordered logit for ordinal outcomes (e.g., survey responses, credit ratings).

#### Constructor

```python
OrderedLogit(
    endog: np.ndarray,
    exog: np.ndarray,
    groups: np.ndarray,
    time: np.ndarray | None = None,
    n_categories: int | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `np.ndarray` | *required* | Ordered categorical outcome |
| `exog` | `np.ndarray` | *required* | Independent variables |
| `groups` | `np.ndarray` | *required* | Entity identifiers |
| `time` | `np.ndarray \| None` | `None` | Time identifiers |
| `n_categories` | `int \| None` | `None` | Number of ordered categories (auto-detected) |

#### Example

```python
from panelbox.models.discrete import OrderedLogit

ologit = OrderedLogit(
    endog=df["satisfaction"].values,
    exog=df[["income", "education"]].values,
    groups=df["person_id"].values,
)
result = ologit.fit()
result.summary()
```

---

### OrderedProbit

Ordered probit model. Same interface as `OrderedLogit` but uses the normal CDF link.

```python
OrderedProbit(endog, exog, groups, time=None, n_categories=None)
```

---

### RandomEffectsOrderedLogit

Random effects ordered logit with Gauss-Hermite quadrature.

---

### NonlinearPanelModel

Base class for all nonlinear panel models. Provides common infrastructure for MLE estimation, gradient computation, and result formatting.

---

## Result Classes

### MultinomialLogitResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `np.ndarray` | Coefficients (J-1 x K matrix) |
| `std_errors` | `np.ndarray` | Standard errors |
| `loglik` | `float` | Log-likelihood |
| `pseudo_r2` | `float` | McFadden pseudo R-squared |

Methods: `.predict_proba()`, `.predict()`, `.marginal_effects()`, `.summary()`

### ConditionalLogitResult

Same structure as `MultinomialLogitResult`.

## See Also

- [Count Models API](count.md) — Poisson, NB for count outcomes
- [Censored & Selection API](censored.md) — Tobit, Heckman
- [Marginal Effects API](marginal-effects.md) — AME, MEM, MER computation
- [Tutorials: Discrete](../tutorials/discrete.md) — Step-by-step guide
