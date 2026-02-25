---
title: "Spatial Durbin Model (SDM)"
description: "Spatial Durbin Model with endogenous and exogenous spatial lags for panel data in PanelBox."
---

# Spatial Durbin Model (SDM)

!!! info "Quick Reference"
    **Class:** `panelbox.models.spatial.SpatialDurbin`
    **Import:** `from panelbox.models.spatial import SpatialDurbin`
    **Stata equivalent:** `xsmle y x1 x2, wmat(W) model(sdm) fe`
    **R equivalent:** `splm::spml(..., lag=TRUE, durbin=TRUE)`

## Overview

The Spatial Durbin Model (SDM) is the most general static spatial lag specification. It includes both the spatially lagged dependent variable ($Wy$) and spatially lagged explanatory variables ($WX$), capturing both **outcome spillovers** and **covariate spillovers** simultaneously.

The model is specified as:

$$y = \rho W y + X\beta + WX\theta + \alpha + \varepsilon$$

where:

- $\rho$ is the spatial autoregressive parameter (spatial lag of $y$)
- $\beta$ is the $K \times 1$ vector of direct covariate effects
- $\theta$ is the $K \times 1$ vector of spatially lagged covariate effects
- $W$ is the $N \times N$ row-standardized spatial weight matrix
- $\alpha$ captures individual effects
- $\varepsilon \sim \text{iid}(0, \sigma^2)$

The SDM nests several other spatial models as special cases:

| Restriction | Resulting Model |
|-------------|----------------|
| $\theta = 0$ | SAR (Spatial Lag) |
| $\rho = 0$ | SLX (Spatial Lag of X) |
| $\theta = -\rho\beta$ | SEM (Spatial Error) |

This nesting property makes the SDM a natural **starting point** for spatial analysis. LeSage and Pace (2009) recommend beginning with the SDM and testing down to simpler models.

## Quick Example

```python
from panelbox.models.spatial import SpatialDurbin, SpatialWeights

# Create weight matrix
W = SpatialWeights.from_contiguity(gdf, criterion='queen')

# Fit SDM with fixed effects
model = SpatialDurbin("y ~ x1 + x2", data, "region", "year", W=W.matrix)
results = model.fit(method='qml', effects='fixed')

# View results
print(results.summary())

# Spatial parameter
print(f"rho = {results.rho:.4f}")

# Spillover effects decomposition
effects = results.spillover_effects
for var in ['x1', 'x2']:
    print(f"\n{var}:")
    print(f"  Direct:   {effects['direct'][var]:.4f}")
    print(f"  Indirect: {effects['indirect'][var]:.4f}")
    print(f"  Total:    {effects['total'][var]:.4f}")
```

## When to Use

- **Default choice** when spatial dependence is suspected (LeSage & Pace 2009 recommendation)
- **Both outcome and covariate spillovers**: neighbors' outcomes AND their characteristics affect your outcome
    - Housing prices: neighbor house values ($Wy$) and neighbor amenities ($WX$) both matter
    - Regional growth: neighbor GDP ($Wy$) and neighbor infrastructure ($WX$) both influence local growth
    - Education: peer achievement ($Wy$) and peer background ($WX$) both affect student outcomes
- **Uncertain spillover mechanism**: when you do not know whether dependence is in outcomes, covariates, or errors
- **Both robust LM tests are significant**: indicates both lag and error dependence

!!! warning "Key Assumptions"
    - **Balanced panel**: all entities observed for all time periods
    - **Stationarity**: $|\rho| < 1$ for the spatial process to be stable
    - **Exogeneity of $W$**: the weight matrix is predetermined
    - **No spatial error correlation**: errors are i.i.d. across units (if not, consider GNS)

## Detailed Guide

### Model Specification

The SDM augments the design matrix with spatially lagged covariates. For each variable $x_k$ in the formula, the model automatically computes $Wx_k$ and includes it alongside the original variable.

```python
# Formula specifies the direct covariates; WX terms are added automatically
model = SpatialDurbin("y ~ x1 + x2", data, "region", "year", W=W.matrix)
```

The resulting parameter vector contains:

- $\rho$: spatial lag parameter
- $\beta_1, \beta_2$: coefficients on $x_1, x_2$
- $\theta_1, \theta_2$: coefficients on $Wx_1, Wx_2$

### Estimation Methods

=== "QML Fixed Effects (Default)"

    Within-transformation removes entity effects, then concentrated quasi-maximum likelihood is used:

    ```python
    results = model.fit(method='qml', effects='fixed')
    ```

    The concentrated log-likelihood over $\rho$ is:

    $$\ell_c(\rho) = -\frac{NT}{2}\ln(\hat{\sigma}^2(\rho)) + T \ln|I_N - \rho W|$$

    where $\hat{\sigma}^2(\rho)$ comes from OLS of $(\tilde{y} - \rho W\tilde{y})$ on $[\tilde{X}, W\tilde{X}]$.

=== "ML Random Effects"

    Full maximum likelihood with GLS quasi-demeaning for random effects:

    ```python
    results = model.fit(method='ml', effects='random')
    ```

    Estimates variance components $\sigma_\alpha^2$ and $\sigma_\varepsilon^2$ alongside $\rho$, $\beta$, and $\theta$.

=== "QML Pooled"

    Ignores individual effects:

    ```python
    results = model.fit(method='qml', effects='pooled')
    ```

### Interpreting Results

!!! warning "Common Mistake"
    In the SDM, $\beta$ is **not** the direct effect and $\theta$ is **not** the indirect effect. The actual direct and indirect effects are computed from the spatial multiplier matrix $(I - \rho W)^{-1}$.

#### Effect Decomposition

The proper effects in the SDM are derived from the reduced form:

$$y = (I - \rho W)^{-1}(X\beta + WX\theta) + (I - \rho W)^{-1}(\alpha + \varepsilon)$$

The partial derivative of $y$ with respect to the $k$-th variable is:

$$\frac{\partial y}{\partial x_k'} = (I - \rho W)^{-1}(I_N \beta_k + W \theta_k)$$

From this $N \times N$ matrix:

- **Direct effect**: average of the diagonal elements
- **Indirect effect**: average of the off-diagonal elements (= Total - Direct)
- **Total effect**: average row sum

```python
# Access the spillover effects
effects = results.spillover_effects

# Direct effects (not equal to beta!)
print("Direct effects:", effects['direct'])

# Indirect effects (spillovers to/from neighbors)
print("Indirect effects:", effects['indirect'])

# Total effects
print("Total effects:", effects['total'])
```

See [Spatial Effects](spatial-effects.md) for a thorough explanation with formulas and examples.

#### Testing Nested Models

The SDM allows you to test whether simpler models are adequate:

```python
from panelbox.models.spatial import GeneralNestingSpatial

# Fit GNS for formal LR tests
gns = GeneralNestingSpatial("y ~ x1 + x2", data, "region", "year",
                             W1=W.matrix, W2=W.matrix, W3=W.matrix)
gns_results = gns.fit(effects='fixed', method='ml')

# Test theta = 0 (SDM → SAR)
lr_sar = gns.test_restrictions(restrictions={'theta': 0})
print(f"SDM vs SAR: LR = {lr_sar['lr_statistic']:.2f}, p = {lr_sar['p_value']:.4f}")

# Test rho = 0, theta = 0 (SDM → OLS)
lr_ols = gns.test_restrictions(restrictions={'rho': 0, 'theta': 0})
print(f"SDM vs OLS: LR = {lr_ols['lr_statistic']:.2f}, p = {lr_ols['p_value']:.4f}")
```

### Prediction

The SDM supports two prediction modes:

```python
# Direct effects only (X*beta + WX*theta)
y_direct = results.predict(effects_type='direct')

# Total effects including spatial multiplier: (I - rho*W)^{-1}(X*beta + WX*theta)
y_total = results.predict(effects_type='total')
```

## Configuration Options

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | `str` | Required | Wilkinson-style formula (e.g., `"y ~ x1 + x2"`) |
| `data` | `pd.DataFrame` | Required | Panel dataset |
| `entity_col` | `str` | Required | Entity identifier column |
| `time_col` | `str` | Required | Time identifier column |
| `W` | `np.ndarray` or `SpatialWeights` | Required | Spatial weight matrix |
| `effects` | `str` | `'fixed'` | `'fixed'` or `'random'` |
| `weights` | `np.ndarray` | `None` | Observation weights |

### `fit()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `'qml'` | `'qml'` (fixed/pooled) or `'ml'` (random) |
| `effects` | `str` | `None` | Override constructor effects |
| `initial_values` | `np.ndarray` | `None` | Starting values for optimization |
| `maxiter` | `int` | `1000` | Maximum iterations |

### Results Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `pd.Series` | All parameters ($\rho$, $\beta$, $\theta$) |
| `rho` | `float` | Spatial autoregressive parameter |
| `bse` | `pd.Series` | Standard errors |
| `tvalues` | `pd.Series` | t-statistics |
| `pvalues` | `pd.Series` | p-values |
| `llf` | `float` | Log-likelihood |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |
| `rsquared_pseudo` | `float` | Pseudo R-squared |
| `spillover_effects` | `dict` | `{'direct': {...}, 'indirect': {...}, 'total': {...}}` |
| `resid` | `np.ndarray` | Residuals |
| `fitted_values` | `np.ndarray` | Fitted values |

## Diagnostics

After fitting the SDM:

1. **Check $\rho$ significance**: if not significant, consider SLX (no spatial lag of $y$)
2. **Check $\theta$ significance**: if no $\theta$ is significant, SAR may suffice
3. **Test the common factor restriction** ($\theta = -\rho\beta$): if not rejected, SEM is adequate
4. **Residual Moran's I**: should be non-significant

```python
# Compare AIC/BIC across specifications
print(f"SDM: AIC={sdm_results.aic:.1f}, BIC={sdm_results.bic:.1f}")
print(f"SAR: AIC={sar_results.aic:.1f}, BIC={sar_results.bic:.1f}")
print(f"SEM: AIC={sem_results.aic:.1f}, BIC={sem_results.bic:.1f}")
# Lower AIC/BIC indicates better fit
```

See [Spatial Diagnostics](diagnostics.md) for the complete workflow.

## Tutorials

| Tutorial | Description | Links |
|----------|-------------|-------|
| Spatial Econometrics | SDM estimation and effect decomposition | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/01_intro_spatial_econometrics.ipynb) |

## See Also

- [Spatial Weight Matrices](spatial-weights.md) — How to construct $W$
- [Spatial Lag (SAR)](spatial-lag.md) — Special case with $\theta = 0$
- [Spatial Error (SEM)](spatial-error.md) — Special case with $\theta = -\rho\beta$
- [Direct, Indirect, and Total Effects](spatial-effects.md) — Essential for SDM interpretation
- [General Nesting Spatial (GNS)](gns.md) — Encompasses SDM with spatial error term
- [Choosing a Spatial Model](choosing-model.md) — Why SDM is the recommended starting point

## References

1. LeSage, J. and Pace, R.K. (2009). *Introduction to Spatial Econometrics*. Chapman & Hall/CRC.
2. Elhorst, J.P. (2010). Applied spatial econometrics: raising the bar. *Spatial Economic Analysis*, 5(1), 9-28.
3. Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
4. Lee, L.F. and Yu, J. (2010). Estimation of spatial autoregressive panel data models with fixed effects. *Journal of Econometrics*, 154(2), 165-185.
