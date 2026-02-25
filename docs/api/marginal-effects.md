---
title: "Marginal Effects API"
description: "API reference for panelbox.marginal_effects — AME, MEM, MER, ordered choice, and interaction effects"
---

# Marginal Effects API Reference

!!! info "Module"
    **Import**: `from panelbox.marginal_effects import ...`
    **Source**: `panelbox/marginal_effects/`

## Overview

PanelBox computes marginal effects for nonlinear panel models (logit, probit, Poisson, ordered choice) using analytical derivatives and the delta method for standard errors. Three types of marginal effects are supported:

| Type | Function | Description |
|------|----------|-------------|
| AME | `compute_ame()` | **Average Marginal Effects** — average over all observations |
| MEM | `compute_mem()` | **Marginal Effects at Means** — evaluate at sample means |
| MER | `compute_mer()` | **Marginal Effects at Representative** — evaluate at user-specified values |

For **ordered choice** models (ordered logit/probit), dedicated functions compute outcome-specific marginal effects.

---

## Standard Marginal Effects

### `compute_ame()`

Average Marginal Effects — computes the marginal effect for each observation and averages across the sample. Most commonly reported in applied work.

```python
def compute_ame(
    result,
    varlist: list[str] | None = None,
    dummy_method: str = "diff",
) -> MarginalEffectsResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | model result | — | Fitted model result (logit, probit, Poisson, etc.) |
| `varlist` | `list[str] \| None` | `None` | Variables to compute ME for (all if None) |
| `dummy_method` | `str` | `"diff"` | Method for binary variables: `"diff"` (discrete change) or `"continuous"` |

**Returns:** `MarginalEffectsResult`

### `compute_mem()`

Marginal Effects at Means — evaluates marginal effects at the sample means of all covariates.

```python
def compute_mem(
    result,
    varlist: list[str] | None = None,
) -> MarginalEffectsResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | model result | — | Fitted model result |
| `varlist` | `list[str] \| None` | `None` | Variables to compute ME for (all if None) |

**Returns:** `MarginalEffectsResult`

### `compute_mer()`

Marginal Effects at Representative values — evaluates at user-specified covariate values.

```python
def compute_mer(
    result,
    at: dict,
    varlist: list[str] | None = None,
) -> MarginalEffectsResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | model result | — | Fitted model result |
| `at` | `dict` | — | Covariate values, e.g. `{"age": 35, "education": 16}` |
| `varlist` | `list[str] \| None` | `None` | Variables to compute ME for (all if None) |

**Returns:** `MarginalEffectsResult`

### `MarginalEffectsResult`

Container for marginal effects results with inference.

```python
class MarginalEffectsResult:
    def __init__(
        self,
        marginal_effects: dict | pd.Series,
        std_errors: dict | pd.Series,
        parent_result,
        me_type: str = "ame",
        at_values: dict | None = None,
    )
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `z_stats` | `pd.Series` | z-statistics (ME / SE) |
| `pvalues` | `pd.Series` | Two-sided p-values |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `conf_int(alpha=0.05)` | `pd.DataFrame` | Confidence intervals |
| `summary(alpha=0.05)` | `pd.DataFrame` | Full summary table with ME, SE, z, p, CI |

**Example:**

```python
from panelbox.models.discrete import PooledLogit
from panelbox.marginal_effects import compute_ame, compute_mem, compute_mer

model = PooledLogit(data, formula="y ~ x1 + x2 + x3")
result = model.fit()

# Average Marginal Effects
ame = compute_ame(result)
print(ame.summary())

# Marginal Effects at Means
mem = compute_mem(result)
print(mem.summary())

# Marginal Effects at Representative values
mer = compute_mer(result, at={"x1": 0.5, "x2": 1.0})
print(mer.summary())
```

---

## Ordered Choice Marginal Effects

For ordered logit and ordered probit models, marginal effects are outcome-specific: one marginal effect per variable per outcome category.

### `compute_ordered_ame()`

```python
def compute_ordered_ame(
    result,
    varlist: list[str] | None = None,
) -> OrderedMarginalEffectsResult
```

### `compute_ordered_mem()`

```python
def compute_ordered_mem(
    result,
    varlist: list[str] | None = None,
) -> OrderedMarginalEffectsResult
```

### `OrderedMarginalEffectsResult`

```python
class OrderedMarginalEffectsResult:
    def __init__(
        self,
        marginal_effects: pd.DataFrame,    # Variables x Outcomes
        std_errors: pd.DataFrame,           # Variables x Outcomes
        parent_result,
        me_type: str = "ame",
        at_values: dict | None = None,
    )
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `z_stats` | `pd.DataFrame` | z-statistics for each variable-outcome pair |
| `pvalues` | `pd.DataFrame` | p-values for each variable-outcome pair |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `verify_sum_to_zero(tol=1e-10)` | `bool` | Verify MEs sum to zero across outcomes |
| `plot(var, ax=None, **kwargs)` | — | Plot MEs across outcomes for a variable |
| `summary(alpha=0.05)` | — | Print formatted summary |

**Example:**

```python
from panelbox.models.discrete import OrderedLogit
from panelbox.marginal_effects import compute_ordered_ame

model = OrderedLogit(data, formula="rating ~ x1 + x2")
result = model.fit()

ame = compute_ordered_ame(result)
ame.summary()

# MEs must sum to zero across outcomes (probability conservation)
assert ame.verify_sum_to_zero()

# Plot marginal effects for x1 across all outcome categories
ame.plot("x1")
```

---

## Interaction Effects

Compute and test cross-partial derivatives (interaction effects) in nonlinear models. In nonlinear models, the interaction effect is **not** simply the coefficient on the interaction term (Ai & Norton, 2003).

### `compute_interaction_effects()`

```python
def compute_interaction_effects(
    model_result,
    var1: str | int,
    var2: str | int,
    interaction_term: str | int | None = None,
    method: str = "delta",
    n_bootstrap: int = 100,
) -> InteractionEffectsResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_result` | model result | — | Fitted nonlinear model |
| `var1` | `str \| int` | — | First interacting variable |
| `var2` | `str \| int` | — | Second interacting variable |
| `interaction_term` | `str \| int \| None` | `None` | Explicit interaction term (auto-detected if None) |
| `method` | `str` | `"delta"` | SE method: `"delta"` or `"bootstrap"` |
| `n_bootstrap` | `int` | `100` | Number of bootstrap replications |

**Returns:** `InteractionEffectsResult`

### `test_interaction_significance()`

Test whether the interaction effect is statistically significant by comparing models with and without the interaction.

```python
def test_interaction_significance(
    model_with_interaction,
    model_without_interaction,
    var1: str | int,
    var2: str | int,
) -> dict[str, Any]
```

**Returns:** Dictionary with `statistic`, `pvalue`, `df`, and `conclusion`.

### `InteractionEffectsResult`

```python
class InteractionEffectsResult:
    def __init__(
        self,
        cross_partial: np.ndarray,
        standard_errors: np.ndarray | None = None,
        predicted_prob: np.ndarray | None = None,
        var1_name: str = "X1",
        var2_name: str = "X2",
    )
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Formatted summary with mean, median, range of interaction effects |
| `plot(figsize=(12, 8))` | `Figure` | Plot interaction effects against predicted probability |

---

## Delta Method Utilities

Low-level functions for computing standard errors via the delta method and numerical differentiation.

### `delta_method_se()`

Compute standard errors using the delta method.

```python
def delta_method_se(
    gradient: np.ndarray,
    cov_matrix: np.ndarray,
    alpha: float = 0.05,
) -> dict
```

**Returns:** Dictionary with keys `se`, `ci_lower`, `ci_upper`, `z_stat`, `pvalue`.

### `numerical_gradient()`

Compute numerical gradient of a function using central differences.

```python
def numerical_gradient(
    func: Callable,
    params: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray
```

---

## See Also

- [Discrete Choice API](discrete.md) — logit, probit, and ordered models
- [Count Models API](count.md) — Poisson and negative binomial models
- [Tutorials: Marginal Effects](../tutorials/marginal-effects.md) — practical guide
