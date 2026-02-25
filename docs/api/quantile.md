---
title: "Quantile Regression API"
description: "API reference for panelbox.models.quantile — Pooled, Fixed Effects, Canay, Location-Scale, Dynamic quantile regression"
---

# Quantile Regression API Reference

!!! info "Module"
    **Import**: `from panelbox.models.quantile import PooledQuantile, FixedEffectsQuantile, CanayTwoStep, LocationScale`
    **Source**: `panelbox/models/quantile/`

## Overview

The quantile module provides panel data quantile regression methods that go beyond mean regression to estimate the full conditional distribution:

| Estimator | Description | Reference |
|-----------|-------------|-----------|
| `PooledQuantile` | Pooled QR with cluster-robust inference | Koenker & Bassett (1978) |
| `FixedEffectsQuantile` | Penalized FE quantile regression | Koenker (2004) |
| `CanayTwoStep` | Two-step estimator for panel QR | Canay (2011) |
| `LocationScale` | Location-scale model | Machado & Santos Silva (2019) |
| `DynamicQuantile` | Dynamic panel quantile regression | — |
| `QuantileTreatmentEffects` | Quantile treatment effects | — |

!!! note "Optional dependencies"
    Some quantile classes (`FixedEffectsQuantile`, `CanayTwoStep`, `LocationScale`, `DynamicQuantile`) use optional imports and may be `None` if dependencies are missing.

## Classes

### PooledQuantile

Pooled quantile regression with cluster-robust standard errors. Estimates conditional quantiles ignoring the panel structure.

#### Constructor

```python
PooledQuantile(
    endog: np.ndarray | pd.Series,
    exog: np.ndarray | pd.DataFrame,
    entity_id: np.ndarray | pd.Series | None = None,
    time_id: np.ndarray | pd.Series | None = None,
    quantiles: float | np.ndarray = 0.5,
    weights: np.ndarray | pd.Series | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `np.ndarray \| pd.Series` | *required* | Dependent variable |
| `exog` | `np.ndarray \| pd.DataFrame` | *required* | Independent variables |
| `entity_id` | `np.ndarray \| pd.Series \| None` | `None` | Entity identifiers (for clustering) |
| `time_id` | `np.ndarray \| pd.Series \| None` | `None` | Time identifiers |
| `quantiles` | `float \| np.ndarray` | `0.5` | Quantile(s) to estimate (0 < tau < 1) |
| `weights` | `np.ndarray \| pd.Series \| None` | `None` | Observation weights |

#### Methods

##### `.fit()`

```python
def fit(
    self,
    method: str = "interior_point",
    maxiter: int = 1000,
    tol: float = 1e-6,
    se_type: str = "cluster",
    alpha: float = 0.05,
    **kwargs,
) -> PooledQuantileResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"interior_point"` | Optimization method |
| `maxiter` | `int` | `1000` | Maximum iterations |
| `tol` | `float` | `1e-6` | Convergence tolerance |
| `se_type` | `str` | `"cluster"` | SE type: `"cluster"`, `"bootstrap"`, `"iid"` |
| `alpha` | `float` | `0.05` | Significance level for confidence intervals |

#### Example

```python
from panelbox.models.quantile import PooledQuantile
import numpy as np

model = PooledQuantile(
    endog=df["wage"],
    exog=df[["education", "experience"]],
    entity_id=df["worker_id"],
    quantiles=np.array([0.25, 0.50, 0.75]),
)
result = model.fit(se_type="cluster")
print(result.summary())
```

---

### FixedEffectsQuantile

Fixed Effects quantile regression using the penalized approach of Koenker (2004). Controls for entity-specific heterogeneity.

#### Constructor

```python
FixedEffectsQuantile(
    endog, exog, entity_id, time_id=None,
    quantiles=0.5, weights=None,
)
```

!!! tip "Koenker (2004) approach"
    Uses an L1-penalty on entity effects to achieve consistent estimation. The penalty parameter controls the bias-variance tradeoff.

---

### CanayTwoStep

Canay (2011) two-step estimator for panel quantile regression. Step 1 estimates entity effects using mean regression, Step 2 estimates quantile regression on the adjusted data.

#### Constructor

```python
CanayTwoStep(
    endog, exog, entity_id, time_id=None,
    quantiles=0.5,
)
```

!!! tip "When to use Canay"
    Canay's estimator is computationally simpler than Koenker (2004) but requires large T. Best suited for panels with many time periods per entity.

---

### LocationScale

Location-Scale model of Machado & Santos Silva (2019). Models both the location (conditional mean) and scale (conditional dispersion) of the distribution.

#### Constructor

```python
LocationScale(
    endog, exog, entity_id, time_id=None,
    quantiles=0.5,
)
```

---

### DynamicQuantile

Dynamic panel quantile regression with lagged dependent variable. Handles the incidental parameters problem in dynamic quantile settings.

---

### QuantileTreatmentEffects

Quantile Treatment Effects (QTE) estimation for panel data. Estimates heterogeneous treatment effects across the distribution.

---

### QuantileMonotonicity

Non-crossing constraints for multiple quantile regressions. Ensures that estimated quantile curves do not cross (monotonicity constraint).

---

### FEQuantileComparison

Compare Fixed Effects quantile regression methods (Koenker 2004, Canay 2011, Machado-Santos Silva 2019) on the same dataset.

---

## Base Classes

### QuantilePanelModel

Base class for all panel quantile regression models.

### QuantilePanelResult

Base result class for panel quantile regression.

---

## Inference

### QuantileBootstrap

```python
from panelbox.inference.quantile import QuantileBootstrap

boot = QuantileBootstrap(model, n_bootstrap=999, method="pairs")
boot_result = boot.run()
```

Bootstrap inference for quantile regression models, accounting for panel structure.

### BootstrapResult

Result container with bootstrap confidence intervals and standard errors.

---

## Diagnostics

### QuantileRegressionDiagnostics

```python
from panelbox.diagnostics.quantile import QuantileRegressionDiagnostics

diag = QuantileRegressionDiagnostics(result)
```

Quantile-specific diagnostics including goodness-of-fit, specification tests, and graphical diagnostics.

## Example: Quantile Process

```python
import numpy as np
from panelbox.models.quantile import PooledQuantile

# Estimate across the distribution
quantiles = np.arange(0.1, 1.0, 0.1)
model = PooledQuantile(
    endog=df["income"],
    exog=df[["education", "experience", "age"]],
    entity_id=df["person_id"],
    quantiles=quantiles,
)
result = model.fit(se_type="cluster")

# Compare coefficients across quantiles
for q in quantiles:
    print(f"tau={q:.1f}: education={result.params[q]['education']:.3f}")
```

## See Also

- [Tutorials: Quantile](../tutorials/quantile.md) — Step-by-step quantile regression guide
- [Theory: Quantile](../theory/quantile-theory.md) — Quantile regression theory
- [Inference API](standard-errors.md) — Standard error types
