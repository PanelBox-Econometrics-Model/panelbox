---
title: "General Nesting Spatial Model (GNS)"
description: "General Nesting Spatial model that encompasses all spatial specifications as special cases in PanelBox."
---

# General Nesting Spatial Model (GNS)

!!! info "Quick Reference"
    **Class:** `panelbox.models.spatial.GeneralNestingSpatial`
    **Import:** `from panelbox.models.spatial import GeneralNestingSpatial`
    **Stata equivalent:** No direct equivalent
    **R equivalent:** `spatialreg::sacsarlm()` (partial)

## Overview

The General Nesting Spatial (GNS) model is the **encompassing spatial specification** that nests all other static spatial models as special cases. It simultaneously includes a spatial lag of the dependent variable ($\rho W_1 y$), spatially lagged covariates ($W_2 X\theta$), and spatially autocorrelated errors ($\lambda W_3 u$).

The model is specified as:

$$y = \rho W_1 y + X\beta + W_2 X\theta + u, \quad u = \lambda W_3 u + \varepsilon$$

where:

- $W_1$, $W_2$, $W_3$ are spatial weight matrices (can be different or the same)
- $\rho$ is the spatial lag parameter
- $\theta$ is the vector of spatially lagged covariate effects
- $\lambda$ is the spatial error parameter
- $\varepsilon \sim \text{iid}(0, \sigma^2)$

The primary use of the GNS is **model selection**: by estimating the full model and testing parameter restrictions, you can determine which simpler spatial specification is supported by the data.

### Nested Models

| Restriction | Resulting Model | Description |
|-------------|----------------|-------------|
| $\lambda = 0, \theta = 0$ | **SAR** | Spatial lag only |
| $\rho = 0, \theta = 0$ | **SEM** | Spatial error only |
| $\lambda = 0$ | **SDM** | Spatial Durbin |
| $\rho = 0, \lambda = 0$ | **SLX** | Spatial lag of X only |
| $\theta = 0$ | **SAC/SARAR** | Spatial lag + error |
| $\rho = 0$ | **SDEM** | Spatial Durbin error |
| $\rho = 0, \theta = 0, \lambda = 0$ | **OLS** | No spatial dependence |
| None | **GNS** | Full model |

## Quick Example

```python
import numpy as np
from panelbox.models.spatial import GeneralNestingSpatial, SpatialWeights

# Create weight matrix (use same W for all three)
W = SpatialWeights.from_contiguity(gdf, criterion='queen')

# Fit GNS model
model = GeneralNestingSpatial(
    "y ~ x1 + x2", data, "region", "year",
    W1=W.matrix, W2=W.matrix, W3=W.matrix
)
results = model.fit(effects='fixed', method='ml')

# View all spatial parameters
print(results.summary())

# Test restrictions to select the best nested model
# Test: is SAR adequate? (theta = 0, lambda = 0)
test_sar = model.test_restrictions(restrictions={'theta': 0, 'lambda': 0})
print(f"GNS vs SAR: LR = {test_sar['lr_statistic']:.2f}, p = {test_sar['p_value']:.4f}")

# Test: is SEM adequate? (rho = 0, theta = 0)
test_sem = model.test_restrictions(restrictions={'rho': 0, 'theta': 0})
print(f"GNS vs SEM: LR = {test_sem['lr_statistic']:.2f}, p = {test_sem['p_value']:.4f}")

# Automatic model identification
model_type = model.identify_model_type(results)
print(f"Identified model type: {model_type}")  # e.g., 'SDM', 'SAR', etc.
```

## When to Use

- **Model selection**: determine which spatial specification is supported by the data
- **Robustness check**: verify that your chosen model is not rejected by the data
- **Exploratory analysis**: when you have no strong prior about the form of spatial dependence
- **Publication**: report the GNS alongside your preferred model to show robustness
- **Different weight matrices**: when the spatial structure differs for lag, Durbin, and error terms

!!! warning "Key Assumptions"
    - **Large $N$ recommended**: the GNS has many parameters and requires sufficient cross-sectional variation for identification
    - **Balanced panel**: required for spatial structure
    - **Stationarity**: $|\rho| < 1$ and $|\lambda| < 1$
    - **Identification concern**: with a single $W$ for all three terms, the model may be weakly identified (Gibbons & Overman, 2012)

## Detailed Guide

### Using Different Weight Matrices

A key feature of the GNS is the ability to use **different weight matrices** for each spatial component. This is theoretically motivated when the mechanisms of spatial interaction differ:

```python
# Example: Regional economics
# W1: contiguity (outcome spillovers via shared borders)
W1 = SpatialWeights.from_contiguity(gdf, criterion='queen').matrix

# W2: trade flows (covariate spillovers via economic linkages)
W2 = trade_weight_matrix  # Custom N x N matrix

# W3: distance (error correlation via proximity)
W3 = SpatialWeights.from_distance(coords, threshold=500).matrix

model = GeneralNestingSpatial(
    "y ~ x1 + x2", data, "region", "year",
    W1=W1, W2=W2, W3=W3
)
results = model.fit(effects='fixed', method='ml')
```

!!! tip
    Using different weight matrices improves identification. When $W_1 = W_2 = W_3$, the GNS can suffer from weak identification — different matrices provide the variation needed to separately identify $\rho$, $\theta$, and $\lambda$.

### Estimation

The GNS is estimated by **maximum likelihood** with numerical optimization over $\rho$ and $\lambda$:

```python
results = model.fit(
    effects='fixed',       # 'fixed', 'random', or 'pooled'
    method='ml',           # Only ML is available for GNS
    rho_init=0.0,          # Starting value for rho
    lambda_init=0.0,       # Starting value for lambda
    include_wx=True,       # Include W2*X terms
    maxiter=1000,          # Maximum iterations
    optim_method='L-BFGS-B'  # Optimization algorithm
)
```

The log-likelihood involves two spatial determinant terms:

$$\ell = -\frac{NT}{2}\ln(2\pi\sigma^2) + T\ln|I - \rho W_1| + T\ln|I - \lambda W_3| - \frac{1}{2\sigma^2}\varepsilon'\varepsilon$$

### Testing Restrictions

The `test_restrictions()` method performs **Likelihood Ratio (LR) tests** for nested models:

```python
# LR test: H0: restricted model is adequate
# Under H0: LR = 2(llf_unrestricted - llf_restricted) ~ chi2(df)

# Test theta = 0 (GNS → SAC)
test_sac = model.test_restrictions(restrictions={'theta': 0})

# Test lambda = 0 (GNS → SDM)
test_sdm = model.test_restrictions(restrictions={'lambda': 0})

# Test rho = 0 (GNS → SDEM)
test_sdem = model.test_restrictions(restrictions={'rho': 0})

# Test all spatial parameters = 0 (GNS → OLS)
test_ols = model.test_restrictions(restrictions={'rho': 0, 'theta': 0, 'lambda': 0})

# Print results
for name, test in [('SAC', test_sac), ('SDM', test_sdm),
                    ('SDEM', test_sdem), ('OLS', test_ols)]:
    print(f"GNS vs {name}: LR={test['lr_statistic']:.2f}, "
          f"p={test['p_value']:.4f}, {test['conclusion']}")
```

### Automatic Model Identification

The `identify_model_type()` method classifies the estimated GNS based on which parameters are statistically significant:

```python
model_type = model.identify_model_type(results)
print(f"Data supports: {model_type}")
# Returns one of: 'SAR', 'SEM', 'SDM', 'SAC', 'SDEM', 'SDEM-SEM', 'GNS', 'OLS'
```

The identification uses significance of $\rho$, $\theta$, and $\lambda$ at the 5% level:

| $\rho$ sig. | $\theta$ sig. | $\lambda$ sig. | Model |
|-------------|---------------|----------------|-------|
| Yes | No | No | SAR |
| No | No | Yes | SEM |
| Yes | Yes | No | SDM |
| Yes | No | Yes | SAC |
| No | Yes | No | SLX |
| No | Yes | Yes | SDEM |
| Yes | Yes | Yes | GNS |
| No | No | No | OLS |

## Configuration Options

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | `str` | Required | Wilkinson formula |
| `data` | `pd.DataFrame` | Required | Panel dataset |
| `entity_col` | `str` | Required | Entity identifier |
| `time_col` | `str` | Required | Time identifier |
| `W1` | `np.ndarray` | `None` | Weight matrix for spatial lag ($Wy$) |
| `W2` | `np.ndarray` | `None` | Weight matrix for spatial Durbin ($WX$) |
| `W3` | `np.ndarray` | `None` | Weight matrix for spatial error ($Wu$) |
| `weights` | `np.ndarray` | `None` | Observation weights |

### `fit()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `effects` | `str` | `'fixed'` | `'fixed'`, `'random'`, or `'pooled'` |
| `method` | `str` | `'ml'` | `'ml'` (only available method) |
| `rho_init` | `float` | `0.0` | Initial value for $\rho$ |
| `lambda_init` | `float` | `0.0` | Initial value for $\lambda$ |
| `include_wx` | `bool` | `True` | Include $W_2 X$ terms |
| `maxiter` | `int` | `1000` | Maximum iterations |
| `optim_method` | `str` | `'L-BFGS-B'` | Scipy optimizer |

!!! note "Caution"
    The GNS is a computationally expensive model. For large $N$, estimation may take several minutes due to the need to compute two log-determinant terms ($|I - \rho W_1|$ and $|I - \lambda W_3|$) at each iteration. Consider starting with simpler models (SAR, SEM, SDM) and using the GNS only for formal model selection.

## Tutorials

| Tutorial | Description | Links |
|----------|-------------|-------|
| Spatial Econometrics | Includes GNS model selection | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/01_intro_spatial_econometrics.ipynb) |

## See Also

- [Spatial Lag (SAR)](spatial-lag.md) — Nested model with $\theta = 0, \lambda = 0$
- [Spatial Error (SEM)](spatial-error.md) — Nested model with $\rho = 0, \theta = 0$
- [Spatial Durbin (SDM)](spatial-durbin.md) — Nested model with $\lambda = 0$
- [Choosing a Spatial Model](choosing-model.md) — Using GNS for model selection
- [Spatial Diagnostics](diagnostics.md) — LR tests and model comparison

## References

1. Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
2. LeSage, J. and Pace, R.K. (2009). *Introduction to Spatial Econometrics*. Chapman & Hall/CRC.
3. Manski, C.F. (1993). Identification of endogenous social effects: The reflection problem. *Review of Economic Studies*, 60(3), 531-542.
4. Gibbons, S. and Overman, H.G. (2012). Mostly pointless spatial econometrics? *Journal of Regional Science*, 52(2), 172-191.
