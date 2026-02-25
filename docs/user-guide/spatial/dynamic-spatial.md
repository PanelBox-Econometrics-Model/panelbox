---
title: "Dynamic Spatial Panel"
description: "Dynamic Spatial Panel model combining temporal dynamics with spatial dependence for panel data in PanelBox."
---

# Dynamic Spatial Panel

!!! info "Quick Reference"
    **Class:** `panelbox.models.spatial.DynamicSpatialPanel`
    **Import:** `from panelbox.models.spatial import DynamicSpatialPanel`
    **Stata equivalent:** No direct equivalent (custom GMM)
    **R equivalent:** `SDPDmod::SDPDm()`

## Overview

The Dynamic Spatial Panel model combines **temporal dynamics** (a lagged dependent variable) with **spatial dependence** (spatial lag of the dependent variable). This is the most realistic specification for many applied settings where both time persistence and spatial spillovers are present.

The model is specified as:

$$y_{it} = \gamma y_{i,t-1} + \rho W y_{it} + X_{it}\beta + \alpha_i + \varepsilon_{it}$$

where:

- $\gamma$ is the temporal autoregressive parameter (persistence)
- $\rho$ is the spatial autoregressive parameter (spillovers across units)
- $y_{i,t-1}$ is the lagged dependent variable
- $Wy_{it}$ is the contemporaneous spatial lag
- $X_{it}$ are exogenous covariates
- $\alpha_i$ are individual fixed effects
- $\varepsilon_{it}$ is the idiosyncratic error

Both $y_{i,t-1}$ and $Wy_{it}$ are endogenous: the lagged dependent variable is correlated with $\alpha_i$, and the spatial lag creates simultaneous feedback. This double endogeneity requires instrumental variable or GMM estimation.

## Quick Example

```python
from panelbox.models.spatial import DynamicSpatialPanel, SpatialWeights

# Create weight matrix
W = SpatialWeights.from_contiguity(gdf, criterion='queen')

# Fit dynamic spatial panel with GMM
model = DynamicSpatialPanel("y ~ x1 + x2", data, "region", "year", W=W.matrix)
results = model.fit(effects='fixed', method='gmm', lags=1, spatial_lags=1)

# View results
print(results.summary())
print(f"Temporal persistence (gamma): {results.params['y_lag1']:.4f}")
print(f"Spatial spillover (rho):      {results.rho:.4f}")

# Impulse response: how a shock to entity 0 propagates through space and time
irf = results.compute_impulse_response(shock_entity=0, periods=10)
```

## When to Use

- **Regional economics**: GDP growth persists over time AND spills across regions
- **Epidemiology**: disease prevalence has temporal persistence AND geographic contagion
- **Housing markets**: prices have momentum AND are influenced by neighboring areas
- **Technology diffusion**: adoption has path dependence AND network effects
- **Fiscal policy**: public spending has inertia AND fiscal competition between jurisdictions

In general, use the dynamic spatial model when:

- You suspect both **temporal autocorrelation** (persistence, momentum)
- AND **spatial autocorrelation** (spillovers, contagion, competition)
- Standard spatial models (SAR, SEM, SDM) ignore the lagged dependent variable
- Standard dynamic models (Arellano-Bond) ignore spatial dependence

!!! warning "Key Assumptions"
    - **Balanced panel**: required for the spatial structure
    - **Stationarity**: $|\gamma| < 1$ and $|\rho| < 1$ for a stable process
    - **Sufficient time periods**: at least $T \geq 4$ to construct valid instruments (after differencing and lagging)
    - **Exogeneity of $X$**: covariates must be strictly exogenous
    - **Fixed $W$**: the spatial weight matrix is time-invariant

## Detailed Guide

### Data Requirements

The dynamic spatial model uses first-differencing and lagging, which consumes time periods:

```python
# Minimum T depends on lag structure
# With lags=1, spatial_lags=1, time_lags=2:
# Need at least T >= 4 (1 for lag + 2 for instruments + 1 for differencing)

T_actual = data.groupby("region").size().iloc[0]
print(f"Available time periods: {T_actual}")
print(f"Minimum required: 4")
```

### Estimation

The model is estimated via two-step efficient GMM:

**Step 1 (First Stage)**: 2SLS with instruments consisting of:

- Deeper lags of $y$: $y_{i,t-2}, y_{i,t-3}, \ldots$
- Spatial lags of $X$: $WX_{it}, W^2X_{it}$
- Interactions: $Wy_{i,t-2}$, etc.

**Step 2 (Second Stage)**: Optimal GMM with the efficient weighting matrix computed from first-stage residuals.

```python
results = model.fit(
    effects='fixed',
    method='gmm',
    lags=1,           # Number of temporal lags of y
    spatial_lags=1,   # Number of spatial lags in instruments
    time_lags=2,      # Depth of temporal instruments
    maxiter=1000,
    tol=1e-6,
    verbose=False
)
```

### Interpreting Results

#### Parameters

| Parameter | Symbol | Interpretation |
|-----------|--------|---------------|
| `y_lag1` | $\gamma$ | Temporal persistence: how much of last period's outcome carries forward |
| `rho` | $\rho$ | Spatial spillover: how much neighbors' current outcomes affect own outcome |
| `x1`, `x2`, ... | $\beta$ | Direct effects of covariates |

#### Short-Run vs Long-Run Effects

The model produces different multiplier effects over time:

- **Short-run (contemporaneous)**: effect within the same period, through spatial multiplier $(I - \rho W)^{-1}$
- **Long-run (steady state)**: cumulative effect including temporal persistence, through $(I - \gamma - \rho W)^{-1}$ (when it converges)

The long-run spatial multiplier is larger because it includes the compounding of temporal persistence with spatial feedback.

### Impulse Response Function

The impulse response function (IRF) traces how a shock to one entity propagates through space and time:

```python
# Shock to entity 0, tracked for 10 periods
irf = results.compute_impulse_response(shock_entity=0, periods=10)
# Returns array of shape (periods, N)
# irf[t, i] = response of entity i at time t to unit shock at entity 0

# Plot the response
import matplotlib.pyplot as plt

# Response of the shocked entity over time
plt.plot(range(10), irf[:, 0], label='Shocked entity')

# Response of neighbors
neighbor_indices = [1, 2, 3]  # indices of neighboring entities
for idx in neighbor_indices:
    plt.plot(range(10), irf[:, idx], alpha=0.5)

plt.xlabel('Periods after shock')
plt.ylabel('Response')
plt.title('Spatio-Temporal Impulse Response')
plt.legend()
plt.show()
```

The IRF combines two propagation channels:

1. **Temporal propagation**: $\gamma$ causes the shock to persist over time (exponential decay if $|\gamma| < 1$)
2. **Spatial propagation**: $\rho$ causes the shock to spread to neighbors, who then spread it further

### Prediction

```python
# One-step-ahead prediction using last observed values
y_pred = results.predict(steps=1)

# Multi-step prediction with future covariates
y_pred_5 = results.predict(steps=5, X_future=X_future_df)
```

Multi-step prediction combines the spatial structure (applying $W$ at each step) with temporal dynamics (feeding predicted values back as lagged values).

## Configuration Options

### `fit()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `effects` | `str` | `'fixed'` | `'fixed'` or `'random'` |
| `method` | `str` | `'gmm'` | `'gmm'` (only available method) |
| `lags` | `int` | `1` | Number of temporal lags of $y$ |
| `spatial_lags` | `int` | `1` | Number of spatial lags in instruments |
| `time_lags` | `int` | `2` | Depth of temporal instruments |
| `initial_values` | `np.ndarray` | `None` | Starting values for optimization |
| `maxiter` | `int` | `1000` | Maximum GMM iterations |
| `tol` | `float` | `1e-6` | Convergence tolerance |
| `verbose` | `bool` | `False` | Print optimization progress |

### Results Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `pd.Series` | All parameters ($\gamma$, $\rho$, $\beta$) |
| `rho` | `float` | Spatial autoregressive parameter |
| `bse` | `pd.Series` | Standard errors (sandwich formula) |
| `tvalues` | `pd.Series` | t-statistics |
| `pvalues` | `pd.Series` | p-values |
| `llf` | `float` | Quasi-log-likelihood |
| `aic` | `float` | AIC |
| `bic` | `float` | BIC |
| `resid` | `np.ndarray` | GMM residuals |

### Results Methods

| Method | Parameters | Description |
|--------|-----------|-------------|
| `compute_impulse_response()` | `shock_entity`: int, `periods`: int (default 10) | Spatio-temporal IRF |
| `predict()` | `steps`: int (default 1), `X_future`: array | Multi-step prediction |
| `summary()` | — | Formatted results table |

## Diagnostics

### Hansen J-Test

The GMM estimator provides an overidentification test:

```python
# Hansen J-test is computed during estimation
# Access via the results (if available in the output)
print(results.summary())  # J-test reported in summary
```

The Hansen J statistic tests whether the instruments are valid (orthogonal to the error term). Under $H_0$ (instruments are valid), $J \sim \chi^2(L - K)$ where $L$ is the number of instruments and $K$ is the number of parameters.

### Stability Check

Verify that the estimated parameters imply a stable process:

```python
gamma = results.params['y_lag1']
rho = results.rho

print(f"|gamma| = {abs(gamma):.4f} < 1: {abs(gamma) < 1}")
print(f"|rho|   = {abs(rho):.4f} < 1: {abs(rho) < 1}")

# The characteristic roots of the system should be inside the unit circle
# A necessary (but not sufficient) condition: |gamma| + |rho| < 1
print(f"|gamma| + |rho| = {abs(gamma) + abs(rho):.4f}")
```

!!! note "Limitations"
    - QML estimation is **not yet implemented** for the dynamic spatial model; only GMM is available
    - The model requires a **balanced panel**
    - Computational cost grows quickly with $N$ and the instrument count
    - For very large $N$ with many instruments, consider reducing `time_lags` or `spatial_lags`

## Tutorials

| Tutorial | Description | Links |
|----------|-------------|-------|
| Spatial Econometrics | Includes dynamic spatial panel example | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/01_intro_spatial_econometrics.ipynb) |

## See Also

- [Spatial Lag (SAR)](spatial-lag.md) — Static version without temporal dynamics
- [Spatial Durbin (SDM)](spatial-durbin.md) — Static version with spatially lagged covariates
- [Spatial Weight Matrices](spatial-weights.md) — Constructing $W$
- [Direct, Indirect, and Total Effects](spatial-effects.md) — Effect interpretation
- [Choosing a Spatial Model](choosing-model.md) — When to add dynamics
- [Spatial Diagnostics](diagnostics.md) — Tests and validation

## References

1. Yu, J., de Jong, R., and Lee, L.F. (2008). Quasi-maximum likelihood estimators for spatial dynamic panel data with fixed effects when both $n$ and $T$ are large. *Journal of Econometrics*, 146(1), 118-134.
2. Lee, L.F. and Yu, J. (2010). Estimation of spatial autoregressive panel data models with fixed effects. *Journal of Econometrics*, 154(2), 165-185.
3. Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
4. Anselin, L., Le Gallo, J., and Jayet, H. (2008). Spatial panel econometrics. In Matyas, L. and Sevestre, P. (eds), *The Econometrics of Panel Data*, 625-660.
