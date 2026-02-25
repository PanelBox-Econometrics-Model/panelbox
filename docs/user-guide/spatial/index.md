---
title: Spatial Econometrics
description: Guide to spatial panel data models in PanelBox - SAR, SEM, SDM, Dynamic Spatial, and General Nesting Spatial models.
---

# Spatial Econometrics

Spatial econometrics addresses situations where observations are not independent but connected through geographic proximity, trade links, or social networks. When outcomes in one region affect outcomes in neighboring regions, standard panel estimators produce biased and inconsistent estimates. Spatial panel models explicitly model these interdependencies.

PanelBox provides five spatial estimators covering the main model types in the literature, along with spatial diagnostics (Moran's I, LM tests) and effects decomposition (direct, indirect, and total effects).

## Spatial Dependence: Three Channels

Spatial dependence can arise through different channels, and each model captures a different mechanism:

```text
SAR (Spatial Lag):     y = rho * W * y + X * beta + e
                       Outcome in i depends on outcomes in neighbors

SEM (Spatial Error):   y = X * beta + u,  u = lambda * W * u + e
                       Shocks spill over to neighbors through errors

SDM (Spatial Durbin):  y = rho * W * y + X * beta + W * X * theta + e
                       Both outcome and covariate spillovers
```

## Available Models

| Model | Class | Abbreviation | Captures |
|-------|-------|-------------|----------|
| Spatial Lag | `SpatialLag` | SAR | Endogenous interaction (outcome spillovers) |
| Spatial Error | `SpatialError` | SEM | Error dependence (correlated shocks) |
| Spatial Durbin | `SpatialDurbin` | SDM | Outcome + covariate spillovers |
| Dynamic Spatial | `DynamicSpatialPanel` | DSAR | Temporal + spatial dynamics |
| General Nesting | `GeneralNestingSpatial` | GNS | All channels simultaneously |

## Quick Example

```python
from panelbox.models.spatial import SpatialLag
from panelbox.datasets import load_grunfeld
import numpy as np

data = load_grunfeld()
n_entities = data["firm"].nunique()

# Create a simple contiguity weight matrix (example)
W = np.ones((n_entities, n_entities)) - np.eye(n_entities)
W = W / W.sum(axis=1, keepdims=True)  # Row-normalize

model = SpatialLag("invest ~ value + capital", data, "firm", "year", W=W)
results = model.fit()
print(results.summary())
```

## Key Concepts

### Weight Matrix

The spatial weight matrix $W$ defines the neighborhood structure. Common specifications:

| Type | Description | Use Case |
|------|-------------|----------|
| Contiguity | $W_{ij} = 1$ if regions share a border | Geographic neighbors |
| Inverse distance | $W_{ij} = 1/d_{ij}$ | Distance decay |
| k-nearest neighbors | $W_{ij} = 1$ if $j$ is among $k$ nearest | Fixed number of neighbors |
| Economic | Trade flows, migration | Non-geographic links |

!!! warning "Always row-normalize"
    Row-normalize $W$ so each row sums to 1. This ensures the spatial lag $Wy$ is a weighted average of neighbors' values.

### Effects Decomposition

In spatial models, a change in $X_i$ affects $y_i$ (direct effect) and $y_j$ for neighbors (indirect/spillover effect):

```python
# After fitting a spatial model
effects = results.effects_decomposition()
print(effects.direct)      # Direct effects
print(effects.indirect)    # Spillover effects
print(effects.total)       # Direct + Indirect
```

### Model Selection with LM Tests

Use LM tests to choose between SAR and SEM:

| Test | H0 | Rejects -> |
|------|----|-----------|
| LM-Lag | No spatial lag (rho = 0) | Use SAR |
| LM-Error | No spatial error (lambda = 0) | Use SEM |
| Robust LM-Lag | No spatial lag (controlling for error) | Use SAR over SEM |
| Robust LM-Error | No spatial error (controlling for lag) | Use SEM over SAR |

!!! tip "Decision rule"
    If both robust tests reject, consider SDM. If only one rejects, use the corresponding model. If neither rejects, spatial dependence may be weak.

## Detailed Guides

- [Spatial Lag (SAR)](spatial-lag.md) -- Endogenous spatial interaction *(detailed guide coming soon)*
- [Spatial Error (SEM)](spatial-error.md) -- Spatially correlated errors *(detailed guide coming soon)*
- [Spatial Durbin (SDM)](spatial-durbin.md) -- Full spatial model *(detailed guide coming soon)*
- [Dynamic Spatial](dynamic-spatial.md) -- Spatial-temporal dynamics *(detailed guide coming soon)*
- [Weight Matrices](spatial-weights.md) -- Constructing and validating W *(detailed guide coming soon)*

## Tutorials

See [Spatial Econometrics Tutorial](../../tutorials/spatial.md) for interactive notebooks with Google Colab.

## API Reference

See [Spatial Models API](../../api/spatial.md) for complete technical reference.

## References

- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic Publishers.
- LeSage, J. P., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
