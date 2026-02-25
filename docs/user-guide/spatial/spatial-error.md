---
title: "Spatial Error Model (SEM)"
description: "Spatial Error Model for panel data with spatially correlated unobservables in PanelBox."
---

# Spatial Error Model (SEM)

!!! info "Quick Reference"
    **Class:** `panelbox.models.spatial.SpatialError`
    **Import:** `from panelbox.models.spatial import SpatialError`
    **Stata equivalent:** `xsmle y x1 x2, wmat(W) model(sem) fe`
    **R equivalent:** `splm::spml(..., spatial.error=TRUE)`

## Overview

The Spatial Error Model (SEM) captures **spatial correlation in unobservable factors**. Unlike the SAR model, the SEM does not model direct outcome spillovers between units. Instead, it accounts for the fact that neighboring units may share common unobserved shocks — such as weather patterns, regional policy effects, or unmeasured local amenities — that create spatial dependence in the error term.

The model is specified as:

$$y = X\beta + \alpha + u, \quad u = \lambda W u + \varepsilon$$

where:

- $y$ is the $NT \times 1$ dependent variable vector
- $X$ is the $NT \times K$ matrix of explanatory variables
- $\beta$ is the $K \times 1$ coefficient vector
- $\alpha$ captures individual effects
- $u$ is the spatially autocorrelated error
- $\lambda$ is the spatial error parameter
- $W$ is the $N \times N$ spatial weight matrix
- $\varepsilon \sim \text{iid}(0, \sigma^2)$ is the idiosyncratic error

The key distinction from SAR: in the SEM, coefficients $\beta$ are directly interpretable as marginal effects. There are **no indirect (spillover) effects** because the spatial dependence operates through the error term, not through the outcome.

## Quick Example

```python
import numpy as np
from panelbox.models.spatial import SpatialError, SpatialWeights

# Create weight matrix
W = SpatialWeights.from_contiguity(gdf, criterion='queen')

# Fit SEM with fixed effects using GMM
model = SpatialError("y ~ x1 + x2", data, "region", "year", W=W.matrix)
results = model.fit(effects='fixed', method='gmm', n_lags=2)

# View results
print(results.summary())

# Lambda (spatial error parameter)
print(f"lambda = {results.params['lambda']:.4f}")

# Coefficients are directly interpretable as marginal effects
print(f"Effect of x1: {results.params['x1']:.4f}")
```

## When to Use

- **Shared unobserved shocks**: neighboring units are affected by common factors not captured in $X$
    - Weather shocks affecting agricultural yields across adjacent counties
    - Unobserved regional amenities influencing housing prices
    - Correlated measurement errors across units
- **Moran's I is significant but LM-lag is not**: spatial dependence is in errors, not outcomes
- **Efficiency matters**: ignoring spatial error correlation leads to inefficient OLS estimates (correct point estimates but wrong standard errors)
- **No theoretical reason for outcome spillovers**: when there is no mechanism for $y_j$ to directly affect $y_i$

!!! warning "Key Assumptions"
    - **Balanced panel**: all entities must have observations for all time periods
    - **Stationarity**: $|\lambda| < 1$ (spatial error process is stable)
    - **Exogeneity of $X$**: all regressors are strictly exogenous
    - **No outcome spillovers**: if outcomes do spillover, use SAR or SDM instead
    - **Correct $W$ specification**: the weight matrix captures the true spatial structure of the errors

## Detailed Guide

### Data Preparation

As with all spatial models, the panel must be balanced:

```python
# Verify balance
entity_counts = data.groupby("region").size()
assert entity_counts.nunique() == 1, "Panel must be balanced for spatial models"
```

### Estimation Methods

=== "GMM with Fixed Effects (Default)"

    The default estimation approach uses two-step GMM with spatial instruments. Fixed effects are removed via within-transformation.

    ```python
    results = model.fit(effects='fixed', method='gmm', n_lags=2)
    ```

    **How it works:**

    1. Within-transform $y$ and $X$ to remove entity effects
    2. Construct spatial instruments $Z = [X, WX, W^2X, \ldots, W^{n\_lags}X]$
    3. First stage: initial 2SLS estimates
    4. Second stage: optimal GMM with efficient weighting matrix

    The `n_lags` parameter controls how many spatial lags of $X$ are used as instruments. More lags = more instruments = potentially more efficient but also more risk of weak instruments.

=== "GMM Pooled"

    GMM estimation without entity effects:

    ```python
    results = model.fit(effects='pooled', method='gmm', n_lags=2)
    ```

=== "Maximum Likelihood"

    Full ML estimation of $\lambda$ and $\beta$ jointly:

    ```python
    results = model.fit(effects='fixed', method='ml', maxiter=1000)
    ```

### Interpreting Results

#### The Spatial Error Parameter $\lambda$

- $\lambda > 0$: positive spatial correlation in unobservables (most common)
- $\lambda < 0$: negative spatial correlation in unobservables
- $\lambda = 0$: no spatial error correlation (reduces to standard panel model)

Unlike $\rho$ in the SAR model, $\lambda$ does not have a direct economic interpretation in terms of spillover magnitude. It indicates the presence of spatially correlated omitted variables.

#### Coefficients as Marginal Effects

A major advantage of the SEM: $\beta$ coefficients are **directly interpretable** as marginal effects:

$$\frac{\partial E[y_i]}{\partial x_{ik}} = \beta_k$$

There are no indirect effects to compute, no spatial multiplier to worry about. This simplicity makes SEM attractive when the substantive interest is in the $\beta$ coefficients rather than in spatial spillovers.

!!! tip "SEM vs OLS"
    If $\lambda \neq 0$ but you use OLS:

    - Point estimates $\hat{\beta}_{OLS}$ are **consistent but inefficient**
    - Standard errors are **biased** (typically too small), leading to incorrect inference
    - The SEM corrects both problems

### GMM Instruments

The GMM estimator uses spatial lags of $X$ as instruments:

$$Z = [X, \; WX, \; W^2X, \; \ldots, \; W^{n\_lags}X]$$

| `n_lags` | Instruments | Trade-off |
|----------|-------------|-----------|
| 1 | $[X, WX]$ | Fewer instruments, may be less efficient |
| 2 | $[X, WX, W^2X]$ | Default, good balance |
| 3 | $[X, WX, W^2X, W^3X]$ | More instruments, better efficiency if valid |

!!! note
    Higher `n_lags` increases the number of instruments. With too many instruments relative to $N$, the GMM estimator can be biased toward OLS. A rule of thumb: keep the instrument count well below $N$.

## Configuration Options

### `fit()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `effects` | `str` | `'fixed'` | `'fixed'`, `'random'`, or `'pooled'` |
| `method` | `str` | `'gmm'` | `'gmm'` (default) or `'ml'` |
| `n_lags` | `int` | `2` | Number of spatial lag instruments (GMM only) |
| `maxiter` | `int` | `1000` | Maximum iterations |
| `verbose` | `bool` | `False` | Print optimization progress |

### Results Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `pd.Series` | All estimated parameters (including $\lambda$) |
| `bse` | `pd.Series` | Standard errors |
| `tvalues` | `pd.Series` | t-statistics |
| `pvalues` | `pd.Series` | Two-tailed p-values |
| `llf` | `float` | Log-likelihood (ML) or quasi-log-likelihood |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |
| `rsquared_pseudo` | `float` | Pseudo R-squared |
| `resid` | `np.ndarray` | Residuals |
| `fitted_values` | `np.ndarray` | Fitted values |
| `method` | `str` | Estimation method (`'GMM'` or `'ML'`) |
| `effects` | `str` | Effects specification |

## Predictions

For the SEM, prediction is straightforward because there is no spatial multiplier in the conditional mean:

```python
# In-sample prediction
y_hat = results.predict()  # y_hat = X * beta
```

Since $E[u] = 0$, the predicted value is simply $\hat{y} = X\hat{\beta}$ (plus fixed effects if applicable). The spatial structure only affects the error covariance, not the prediction.

## Diagnostics

### Motivation: Moran's I on OLS Residuals

The SEM is typically motivated by finding significant spatial autocorrelation in OLS residuals:

```python
from panelbox import FixedEffects
from panelbox.diagnostics.spatial import MoranIPanelTest

# Step 1: Fit standard FE model
fe_model = FixedEffects("y ~ x1 + x2", data, "region", "year")
fe_results = fe_model.fit()

# Step 2: Test residuals for spatial autocorrelation
moran = MoranIPanelTest(fe_results.resid, W.matrix)
moran_result = moran.run()
print(f"Moran's I: {moran_result.statistic:.4f} (p = {moran_result.pvalue:.4f})")
# If significant: spatial model needed
```

### Post-Estimation Checks

After fitting the SEM, verify that spatial autocorrelation in residuals has been eliminated:

```python
# Moran's I on SEM residuals
moran_post = MoranIPanelTest(results.resid, W.matrix)
post_result = moran_post.run()
print(f"Post-SEM Moran's I: {post_result.statistic:.4f} (p = {post_result.pvalue:.4f})")
# Should be non-significant
```

See [Spatial Diagnostics](diagnostics.md) for the full diagnostic workflow.

## Tutorials

| Tutorial | Description | Links |
|----------|-------------|-------|
| Spatial Econometrics | Complete SAR, SEM, SDM workflow | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/01_intro_spatial_econometrics.ipynb) |

## See Also

- [Spatial Weight Matrices](spatial-weights.md) — How to construct $W$
- [Spatial Lag (SAR)](spatial-lag.md) — When dependence is in outcomes, not errors
- [Spatial Durbin (SDM)](spatial-durbin.md) — Nests both SAR and SEM as special cases
- [Direct, Indirect, and Total Effects](spatial-effects.md) — SEM has no indirect effects
- [Choosing a Spatial Model](choosing-model.md) — When to use SEM vs SAR vs SDM
- [Spatial Diagnostics](diagnostics.md) — LM tests to distinguish SAR from SEM

## References

1. Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic.
2. Kelejian, H.H. and Prucha, I.R. (1998). A generalized spatial two-stage least squares procedure for estimating a spatial autoregressive model with autoregressive disturbances. *Journal of Real Estate Finance and Economics*, 17(1), 99-121.
3. Kapoor, M., Kelejian, H.H., and Prucha, I.R. (2007). Panel data models with spatially correlated error components. *Journal of Econometrics*, 140(1), 97-130.
4. Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
