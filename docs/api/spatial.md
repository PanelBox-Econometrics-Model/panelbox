---
title: "Spatial Models API"
description: "API reference for panelbox.models.spatial — SAR, SEM, SDM, Dynamic Spatial, GNS"
---

# Spatial Models API Reference

!!! info "Module"
    **Import**: `from panelbox.models.spatial import SpatialLag, SpatialError, SpatialDurbin, DynamicSpatialPanel, GeneralNestingSpatial`
    **Source**: `panelbox/models/spatial/`

## Overview

The spatial module provides five spatial econometric models for panel data, each incorporating different types of spatial interaction through a weight matrix W:

| Model | Abbreviation | Spatial Interaction |
|-------|-------------|---------------------|
| `SpatialLag` | SAR | Spatial lag of dependent variable (Wy) |
| `SpatialError` | SEM | Spatial autocorrelation in errors |
| `SpatialDurbin` | SDM | Spatial lag of dependent AND independent variables |
| `DynamicSpatialPanel` | DSP | Space-time lags (dynamic + spatial) |
| `GeneralNestingSpatial` | GNS | Most general specification (SAR + SEM + SDM) |

All models require a spatial weight matrix `W` (N x N) defining spatial relationships between entities.

## Classes

### SpatialLag (SAR)

Spatial Autoregressive model. Includes a spatially-lagged dependent variable:

y = rho * W * y + X * beta + epsilon

#### Constructor

```python
SpatialLag(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    W: np.ndarray | SpatialWeights,
    weights: np.ndarray | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | `str` | *required* | R-style formula |
| `data` | `pd.DataFrame` | *required* | Panel data |
| `entity_col` | `str` | *required* | Entity column |
| `time_col` | `str` | *required* | Time column |
| `W` | `np.ndarray \| SpatialWeights` | *required* | Spatial weight matrix (N x N) |
| `weights` | `np.ndarray \| None` | `None` | Observation weights |

#### Methods

##### `.fit()`

```python
def fit(
    self,
    effects: str = "fixed",
    method: str = "qml",
    rho_grid_size: int = 20,
    optimizer: str = "brent",
    maxiter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
    **kwargs,
) -> SpatialPanelResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `effects` | `str` | `"fixed"` | `"fixed"` or `"random"` effects |
| `method` | `str` | `"qml"` | Estimation method: `"qml"` (quasi-ML) or `"ml"` |
| `rho_grid_size` | `int` | `20` | Grid size for initial rho search |
| `optimizer` | `str` | `"brent"` | Optimizer for rho |
| `maxiter` | `int` | `1000` | Maximum iterations |
| `tol` | `float` | `1e-6` | Convergence tolerance |
| `verbose` | `bool` | `False` | Print progress |

#### Example

```python
import numpy as np
from panelbox.models.spatial import SpatialLag

# Create contiguity weight matrix
W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
W = W / W.sum(axis=1, keepdims=True)  # Row-normalize

model = SpatialLag("y ~ x1 + x2", data, "entity", "year", W=W)
result = model.fit(effects="fixed", method="qml")
print(f"Spatial rho: {result.rho:.4f}")
result.summary()
```

---

### SpatialError (SEM)

Spatial Error Model. Spatial autocorrelation in the error term:

y = X * beta + u, where u = lambda * W * u + epsilon

#### Constructor

```python
SpatialError(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    W: np.ndarray | SpatialWeights,
    weights: np.ndarray | None = None,
)
```

#### Methods

##### `.fit()`

```python
def fit(
    self,
    effects: str = "fixed",
    method: str = "gmm",
    n_lags: int = 2,
    maxiter: int = 1000,
    verbose: bool = False,
    **kwargs,
) -> SpatialPanelResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `effects` | `str` | `"fixed"` | `"fixed"` or `"random"` effects |
| `method` | `str` | `"gmm"` | Estimation method: `"gmm"` or `"ml"` |
| `n_lags` | `int` | `2` | Number of spatial lags in GMM |

---

### SpatialDurbin (SDM)

Spatial Durbin Model. Includes spatial lags of both dependent and independent variables:

y = rho * W * y + X * beta + W * X * theta + epsilon

#### Constructor

```python
SpatialDurbin(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    W: np.ndarray | SpatialWeights,
    effects: Literal["fixed", "random"] = "fixed",
    weights: np.ndarray | None = None,
)
```

#### Methods

##### `.fit()`

```python
def fit(
    self,
    method: Literal["qml", "ml"] = "qml",
    effects: str | None = None,
    initial_values: dict[str, float] | None = None,
    maxiter: int = 1000,
    **kwargs,
) -> SpatialPanelResults
```

!!! tip "SDM as a general specification"
    SDM nests both SAR (theta=0) and SEM (theta = -rho*beta). Use LR tests or Wald tests to determine which restriction holds.

---

### DynamicSpatialPanel

Dynamic Spatial Panel model with space-time lags:

y_t = rho * W * y_t + gamma * y_{t-1} + X * beta + epsilon

#### Constructor

```python
DynamicSpatialPanel(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    W: np.ndarray,
    weights: np.ndarray | None = None,
)
```

#### Methods

##### `.fit()`

```python
def fit(
    self,
    effects: Literal["fixed", "random"] = "fixed",
    method: Literal["gmm", "qml"] = "gmm",
    lags: int = 1,
    spatial_lags: int = 1,
    time_lags: int = 2,
    initial_values: dict | None = None,
    maxiter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
    **kwargs,
) -> SpatialPanelResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `effects` | `str` | `"fixed"` | Panel effects type |
| `method` | `str` | `"gmm"` | Estimation method |
| `lags` | `int` | `1` | Temporal lag order |
| `spatial_lags` | `int` | `1` | Spatial lag order |
| `time_lags` | `int` | `2` | Number of time lags for GMM instruments |

---

### GeneralNestingSpatial (GNS)

General Nesting Spatial model. The most general spatial specification:

y = rho * W1 * y + X * beta + W2 * X * theta + u, where u = lambda * W3 * u + epsilon

Supports different weight matrices for each spatial component.

#### Constructor

```python
GeneralNestingSpatial(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    W1: np.ndarray | None = None,
    W2: np.ndarray | None = None,
    W3: np.ndarray | None = None,
    weights: np.ndarray | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `W1` | `np.ndarray \| None` | `None` | Weight matrix for spatial lag (y) |
| `W2` | `np.ndarray \| None` | `None` | Weight matrix for spatial lag (X) |
| `W3` | `np.ndarray \| None` | `None` | Weight matrix for spatial error |

#### Methods

##### `.fit()`

```python
def fit(
    self,
    effects: Literal["fixed", "random", "pooled"] = "fixed",
    method: Literal["ml", "gmm"] = "ml",
    rho_init: float = 0.0,
    lambda_init: float = 0.0,
    include_wx: bool = True,
    maxiter: int = 1000,
    optim_method: str = "L-BFGS-B",
    **kwargs,
) -> SpatialPanelResults
```

---

## Result Classes

### SpatialPanelResults

Result container for spatial panel models with spatial-specific attributes.

#### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `pd.Series` | Estimated coefficients |
| `rho` | `float` | Spatial autoregressive parameter (SAR/SDM/GNS) |
| `lambda_` | `float` | Spatial error parameter (SEM/GNS) |
| `sigma2` | `float` | Error variance |
| `llf` | `float` | Log-likelihood |
| `nobs` | `int` | Number of observations |
| `resid` | `np.ndarray` | Residuals |

#### Spatial Effects Decomposition

| Attribute | Type | Description |
|-----------|------|-------------|
| `direct_effects` | `pd.Series` | Direct (own-region) effects |
| `indirect_effects` | `pd.Series` | Indirect (spillover) effects |
| `total_effects` | `pd.Series` | Total effects (direct + indirect) |

#### Methods

- `.summary()` — Formatted results with spatial diagnostics
- `.predict(new_data=None, W=None)` — Spatial predictions

## Spatial Effects Functions

### compute_spatial_effects

```python
from panelbox.effects import compute_spatial_effects

effects = compute_spatial_effects(result)
```

Decompose coefficients into direct, indirect, and total effects for SAR/SDM/GNS models.

### spatial_impact_matrix

```python
from panelbox.effects import spatial_impact_matrix

S = spatial_impact_matrix(result, variable="x1")
```

Compute the full N x N impact matrix for a given variable.

## Example: Complete Spatial Workflow

```python
import numpy as np
from panelbox.models.spatial import SpatialLag, SpatialDurbin
from panelbox.effects import compute_spatial_effects

# Create weight matrix (e.g., from contiguity)
W = ...  # Your N x N weight matrix

# Estimate SAR model
sar = SpatialLag("gdp ~ investment + labor", data, "country", "year", W=W)
sar_result = sar.fit(effects="fixed")

# Estimate SDM model
sdm = SpatialDurbin("gdp ~ investment + labor", data, "country", "year", W=W)
sdm_result = sdm.fit(effects="fixed")

# Compute spatial effects
effects = compute_spatial_effects(sdm_result)
print("Direct effects:", effects.direct_effects)
print("Indirect effects:", effects.indirect_effects)
print("Total effects:", effects.total_effects)
```

## See Also

- [Standard Errors: Spatial HAC](standard-errors.md) — Spatial HAC covariance estimation
- [Diagnostics](diagnostics.md) — Moran's I, LM tests for spatial dependence
- [Tutorials: Spatial](../tutorials/spatial.md) — Step-by-step spatial guide
- [Theory: Spatial](../theory/spatial-theory.md) — Spatial econometric theory
