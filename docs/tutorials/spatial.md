---
title: "Spatial Econometrics Tutorials"
description: "Interactive tutorials for spatial panel models including SAR, SEM, SDM, and dynamic spatial panels with PanelBox"
---

# Spatial Econometrics Tutorials

!!! info "Learning Path"
    **Prerequisites**: [Static Models](static-models.md) tutorials, understanding of spatial concepts (proximity, contiguity)
    **Time**: 4--7 hours
    **Level**: Intermediate -- Advanced

## Overview

Spatial econometrics extends panel models to account for geographic or network dependence between cross-sectional units. When outcomes in one region are influenced by outcomes or characteristics in neighboring regions, standard panel estimators produce biased and inconsistent results.

These tutorials cover the full spectrum of spatial panel models: the Spatial Autoregressive model (SAR), Spatial Error model (SEM), Spatial Durbin model (SDM), and dynamic spatial panels. You will learn how to construct spatial weight matrices, estimate direct and indirect (spillover) effects, and test for the appropriate spatial specification.

The [Spatial Econometrics notebook tutorial](spatial_econometrics_complete.ipynb) provides a self-contained end-to-end example that complements these tutorials.

## Notebooks

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Introduction to Spatial Econometrics](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/01_intro_spatial_econometrics.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/01_intro_spatial_econometrics.ipynb) |
| 2 | [Spatial Weight Matrices](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/02_spatial_weights_matrices.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/02_spatial_weights_matrices.ipynb) |
| 3 | [Spatial Lag Model (SAR)](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/03_spatial_lag_model.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/03_spatial_lag_model.ipynb) |
| 4 | [Spatial Error Model (SEM)](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/04_spatial_error_model.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/04_spatial_error_model.ipynb) |
| 5 | [Spatial Durbin Model (SDM)](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/05_spatial_durbin_model.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/05_spatial_durbin_model.ipynb) |
| 6 | [Spatial Marginal Effects](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/06_spatial_marginal_effects.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/06_spatial_marginal_effects.ipynb) |
| 7 | [Dynamic Spatial Panels](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/07_dynamic_spatial_panels.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/07_dynamic_spatial_panels.ipynb) |
| 8 | [Specification Tests](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/08_specification_tests.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/08_specification_tests.ipynb) |

## Learning Paths

### :material-lightning-bolt: Core (4 hours)

Essential spatial models for applied research:

**Notebooks**: 1, 2, 3, 4, 5

Covers weight matrix construction, the three canonical spatial models (SAR, SEM, SDM), and how to choose between them.

### :material-trophy: Advanced (7 hours)

Complete spatial econometrics coverage:

**Notebooks**: 1--8

Adds direct/indirect effects decomposition, dynamic spatial panels, and formal specification testing.

## Key Concepts Covered

- **Spatial dependence**: Why ignoring spatial correlation leads to bias
- **Weight matrices**: Contiguity, distance-based, k-nearest neighbors
- **SAR (Spatial Lag)**: Endogenous spatial lag of the dependent variable
- **SEM (Spatial Error)**: Spatial correlation in the error term
- **SDM (Spatial Durbin)**: Both spatially lagged dependent variable and regressors
- **Direct vs indirect effects**: Own-region vs spillover impacts
- **Dynamic spatial**: Combining temporal dynamics with spatial dependence
- **LM tests**: Lagrange Multiplier tests for spatial specification

## Quick Example

```python
from panelbox.models.spatial import SpatialLag
import numpy as np

# Create a spatial weight matrix (row-standardized)
W = ...  # your NxN weight matrix

# Estimate Spatial Lag model
sar = SpatialLag(
    data=data,
    formula="y ~ x1 + x2",
    entity_col="id",
    time_col="year",
    w=W
).fit()

print(sar.summary())
print(f"Spatial rho: {sar.rho:.4f}")
```

## Related Documentation

- [Spatial Econometrics Tutorial](spatial_econometrics_complete.ipynb) -- Complete self-contained notebook
- [Theory: Spatial Models Comparison](../user-guide/spatial/choosing-model.md) -- SAR vs SEM vs SDM
- [Theory: Spatial Autocorrelation](../user-guide/spatial/spatial-effects.md) -- Moran's I and diagnostics
- [User Guide](../user-guide/index.md) -- API reference
- [Validation & Diagnostics](validation.md) -- Spatial diagnostic tests
