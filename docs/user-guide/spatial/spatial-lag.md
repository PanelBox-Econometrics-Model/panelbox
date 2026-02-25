---
title: "Spatial Lag Model (SAR)"
description: "Spatial Autoregressive (SAR) panel model for outcome spillovers in PanelBox."
---

# Spatial Lag Model (SAR)

!!! info "Quick Reference"
    **Class:** `panelbox.models.spatial.SpatialLag`
    **Import:** `from panelbox.models.spatial import SpatialLag`
    **Stata equivalent:** `xsmle y x1 x2, wmat(W) model(sar) fe`
    **R equivalent:** `splm::spml(..., model="within", lag=TRUE)`

## Overview

The Spatial Lag model (also called the Spatial Autoregressive model, SAR) captures **outcome spillovers** — situations where one unit's outcome directly affects its neighbors' outcomes. The key feature is the spatially lagged dependent variable $Wy$ on the right-hand side of the equation.

The model is specified as:

$$y = \rho W y + X\beta + \alpha + \varepsilon$$

where:

- $y$ is the $NT \times 1$ vector of dependent variable observations
- $W$ is the $N \times N$ spatial weight matrix (row-standardized)
- $\rho$ is the spatial autoregressive parameter measuring the strength of spatial spillovers
- $X$ is the $NT \times K$ matrix of explanatory variables
- $\beta$ is the $K \times 1$ vector of coefficients
- $\alpha$ captures individual (entity) effects
- $\varepsilon$ is the error term

The parameter $\rho$ has a direct economic interpretation: it captures how much a neighbor's outcome affects your own. A positive $\rho$ indicates spatial clustering of similar values, while a negative $\rho$ indicates spatial dispersion.

## Quick Example

```python
import numpy as np
from panelbox.models.spatial import SpatialLag, SpatialWeights

# Load your panel data (must be balanced)
# data has columns: y, x1, x2, region, year

# Create weight matrix (e.g., from contiguity)
W = SpatialWeights.from_contiguity(gdf, criterion='queen')

# Fit SAR model with fixed effects
model = SpatialLag("y ~ x1 + x2", data, "region", "year", W=W.matrix)
results = model.fit(effects='fixed', method='qml')

# View results
print(results.summary())

# Spatial parameter
print(f"rho = {results.rho:.4f}")  # e.g., 0.35

# Spillover effects
effects = results.spillover_effects
print(f"Direct effect of x1:   {effects['direct']['x1']:.4f}")
print(f"Indirect effect of x1: {effects['indirect']['x1']:.4f}")
print(f"Total effect of x1:    {effects['total']['x1']:.4f}")
```

## When to Use

- **Outcome-based interactions**: neighbors' outcomes directly influence your own outcome
    - Regional unemployment spillovers (labor mobility)
    - Technology adoption cascading across firms
    - Crime contagion between neighborhoods
    - Housing price effects between neighboring areas
- **Trade and migration**: flows between regions create interdependence in economic outcomes
- **Policy diffusion**: policy adoption in one jurisdiction influences neighbors
- **Epidemiology**: disease prevalence in one area affects neighboring areas

!!! warning "Key Assumptions"
    - **Balanced panel**: all entities must have observations for all time periods
    - **Stationarity**: $|\rho| < 1$ (spatial process is stable)
    - **Exogeneity of $W$**: the weight matrix is exogenous and correctly specified
    - **No spatial error correlation**: errors are independent across units (otherwise use SDM or SEM)
    - **Panel structure**: $W$ matches the cross-sectional dimension ($N \times N$)

## Detailed Guide

### Data Preparation

Spatial models in PanelBox require **balanced panels**. Ensure every entity has observations for every time period.

```python
import pandas as pd

# Check balance
counts = data.groupby("region").size()
print(f"Entities: {counts.nunique()}, Periods per entity: {counts.unique()}")

# If unbalanced, balance it
complete_periods = data.groupby("year")["region"].nunique()
valid_periods = complete_periods[complete_periods == complete_periods.max()].index
data_balanced = data[data["year"].isin(valid_periods)]
```

### Estimation Methods

The SAR model offers three estimation approaches depending on the effects specification:

=== "Fixed Effects (QML)"

    The default and most common approach. Uses within-transformation to eliminate entity effects, then maximizes a concentrated quasi-log-likelihood over $\rho$.

    ```python
    results = model.fit(effects='fixed', method='qml')
    ```

    The concentrated log-likelihood is:

    $$\ell_c(\rho) = -\frac{NT}{2}\ln(2\pi) - \frac{NT}{2}\ln(\hat{\sigma}^2(\rho)) + T \ln|I_N - \rho W|$$

    where $\hat{\sigma}^2(\rho)$ is the concentrated variance for a given $\rho$, obtained by OLS of $\tilde{y} - \rho W\tilde{y}$ on $\tilde{X}$ (tilde denotes within-transformed).

=== "Random Effects (ML)"

    Full maximum likelihood estimation with random individual effects.

    ```python
    results = model.fit(effects='random', method='ml')
    ```

    Estimates the variance components $\sigma_\alpha^2$ (between) and $\sigma_\varepsilon^2$ (within) alongside $\rho$ and $\beta$.

=== "Pooled (QML)"

    Ignores individual effects entirely. Useful as a baseline or when entity effects are not relevant.

    ```python
    results = model.fit(effects='pooled', method='qml')
    ```

### Interpreting Results

The SAR model requires careful interpretation because of the spatial multiplier effect.

#### The Spatial Parameter $\rho$

- $\rho > 0$: positive spatial dependence (common in most applications)
- $\rho < 0$: negative spatial dependence (competition effects)
- $\rho = 0$: no spatial dependence (reduces to standard panel model)
- The **spatial multiplier** is $\frac{1}{1 - \rho}$. For $\rho = 0.3$, the multiplier is $\approx 1.43$

#### Why Coefficients Are Not Direct Effects

In a standard OLS model, $\beta_k$ is the marginal effect of $x_k$ on $y$. In the SAR model, this is **not** the case because of spatial feedback:

$$y = (I - \rho W)^{-1} X\beta + (I - \rho W)^{-1} \varepsilon$$

The matrix $(I - \rho W)^{-1}$ creates a **spatial multiplier** that distributes the effect of $x_k$ across all units. See the [Spatial Effects](spatial-effects.md) page for a full treatment of direct, indirect, and total effects.

#### Summary Output

```python
print(results.summary())
```

The summary includes:

- **Coefficient table**: estimates, standard errors, t-statistics, p-values
- **Spatial parameter** $\rho$ with standard error and significance
- **Model fit**: log-likelihood, AIC, BIC, pseudo R-squared
- **Spatial information**: number of units, weight matrix density

## Configuration Options

### `fit()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `effects` | `str` | `'fixed'` | `'fixed'`, `'random'`, or `'pooled'` |
| `method` | `str` | `'qml'` | `'qml'` (fixed/pooled) or `'ml'` (random) |
| `rho_grid_size` | `int` | `20` | Grid search resolution for $\rho$ (QML) |
| `optimizer` | `str` | `'brent'` | `'brent'` (1D) or `'l-bfgs-b'` |
| `maxiter` | `int` | `1000` | Maximum optimization iterations |
| `tol` | `float` | `1e-6` | Convergence tolerance |
| `verbose` | `bool` | `False` | Print optimization progress |

### Results Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `pd.Series` | All estimated parameters (including $\rho$) |
| `bse` | `pd.Series` | Standard errors |
| `tvalues` | `pd.Series` | t-statistics |
| `pvalues` | `pd.Series` | Two-tailed p-values |
| `rho` | `float` | Spatial autoregressive parameter |
| `llf` | `float` | Log-likelihood at convergence |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |
| `rsquared_pseudo` | `float` | Pseudo R-squared |
| `resid` | `np.ndarray` | Residuals |
| `fitted_values` | `np.ndarray` | Fitted values |
| `method` | `str` | Estimation method used |
| `effects` | `str` | Effects specification |
| `spillover_effects` | `dict` | Direct, indirect, total effects |

## Predictions

```python
# In-sample prediction
y_hat = results.predict()

# Out-of-sample prediction (requires W for new spatial structure)
y_new = results.predict(new_data=new_df, W=W_new)
```

For SAR models, prediction uses the spatial multiplier:

$$\hat{y} = (I - \hat{\rho} W)^{-1} (X\hat{\beta} + \hat{\alpha})$$

This is computed separately for each time period.

## Diagnostics

After fitting a SAR model, check:

1. **Significance of $\rho$**: if not significant, a standard panel model may suffice
2. **Residual spatial autocorrelation**: Moran's I on residuals should be insignificant
3. **Model comparison**: compare AIC/BIC with SEM and SDM alternatives

```python
# Check if spatial dependence is captured
from panelbox.diagnostics.spatial import MoranIPanelTest

moran = MoranIPanelTest(results.resid, W.matrix)
moran_result = moran.run()
print(f"Moran's I on residuals: {moran_result.statistic:.4f}")
print(f"p-value: {moran_result.pvalue:.4f}")
# Should be non-significant if SAR adequately captures spatial dependence
```

See [Spatial Diagnostics](diagnostics.md) for the full diagnostic workflow.

## Tutorials

| Tutorial | Description | Links |
|----------|-------------|-------|
| Spatial Econometrics | Complete SAR, SEM, SDM workflow | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/01_intro_spatial_econometrics.ipynb) |

## See Also

- [Spatial Weight Matrices](spatial-weights.md) — How to construct $W$
- [Spatial Error (SEM)](spatial-error.md) — When dependence is in errors, not outcomes
- [Spatial Durbin (SDM)](spatial-durbin.md) — Generalization with spatially lagged covariates
- [Direct, Indirect, and Total Effects](spatial-effects.md) — Proper effect interpretation
- [Choosing a Spatial Model](choosing-model.md) — When to use SAR vs SEM vs SDM
- [Spatial Diagnostics](diagnostics.md) — Tests for spatial dependence

## References

1. Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic.
2. Lee, L.F. and Yu, J. (2010). Estimation of spatial autoregressive panel data models with fixed effects. *Journal of Econometrics*, 154(2), 165-185.
3. LeSage, J. and Pace, R.K. (2009). *Introduction to Spatial Econometrics*. Chapman & Hall/CRC.
4. Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
