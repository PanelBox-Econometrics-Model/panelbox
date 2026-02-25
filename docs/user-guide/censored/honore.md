---
title: "Honore Trimmed Estimator"
description: "Honore (1992) bias-corrected fixed effects estimator for censored panel data in PanelBox."
---

# Honore Trimmed Estimator

!!! info "Quick Reference"
    **Class:** `panelbox.models.censored.HonoreTrimmedEstimator`
    **Import:** `from panelbox.models.censored import HonoreTrimmedEstimator`
    **Stata equivalent:** `pantob` (community-contributed)
    **R equivalent:** Custom implementation required

## Overview

Standard Tobit models with fixed effects suffer from the **incidental parameters problem**: as $N \to \infty$ with fixed $T$, the maximum likelihood estimator of $\beta$ is **inconsistent** because the number of nuisance parameters ($\alpha_1, \ldots, \alpha_N$) grows with $N$. Unlike the linear FE model (where the within transformation perfectly eliminates fixed effects), the nonlinearity of Tobit prevents a simple demeaning solution.

Honore (1992) proposed a **semiparametric trimmed LAD (Least Absolute Deviations) estimator** that resolves this problem. The estimator uses pairwise differences across time periods within each entity to eliminate the fixed effects, and applies trimming to handle the asymmetric censoring structure. The key advantage is that it provides **consistent** slope estimates without distributional assumptions on the error terms or the fixed effects.

## Quick Example

```python
import numpy as np
from panelbox.models.censored import HonoreTrimmedEstimator

# Simulated censored panel data
np.random.seed(42)
N, T = 100, 5
entity = np.repeat(np.arange(N), T)
time = np.tile(np.arange(T), N)
X = np.random.randn(N * T, 2)
alpha = np.repeat(np.random.randn(N), T)  # Fixed effects
y_star = X @ np.array([0.5, -0.3]) + alpha + np.random.randn(N * T)
y = np.maximum(0, y_star)  # Left-censored at 0

# Fit Honore estimator
model = HonoreTrimmedEstimator(
    endog=y, exog=X, groups=entity, time=time,
    censoring_point=0.0,
)
results = model.fit(method="L-BFGS-B", verbose=True)
print(f"Coefficients: {results.params}")
print(f"Trimmed observations: {results.n_trimmed}")
```

## When to Use

- You have censored panel data and need **fixed effects** (entity-level heterogeneity correlated with regressors)
- You are suspicious of the Random Effects assumption ($\text{Cov}(\alpha_i, X_{it}) = 0$)
- You want consistent slope estimates **without distributional assumptions** on errors or fixed effects
- You have at least $T \geq 2$ time periods per entity

!!! warning "Key Assumptions"
    - **Left censoring only**: the current implementation supports left-censoring at a known point
    - **Panel structure**: requires at least 2 time periods per entity
    - **Stationarity**: the error distribution does not change over time (within entity)
    - **No distributional assumption** on $\varepsilon_{it}$ or $\alpha_i$ (semiparametric)

## The Fixed Effects Censoring Problem

### Why Standard FE Tobit Fails

In a linear panel model, the within transformation eliminates $\alpha_i$:

$$y_{it} - \bar{y}_i = (X_{it} - \bar{X}_i)'\beta + (\varepsilon_{it} - \bar{\varepsilon}_i)$$

But the Tobit model is nonlinear: $y_{it} = \max(c, X_{it}'\beta + \alpha_i + \varepsilon_{it})$. The $\max$ operator prevents the fixed effects from canceling via demeaning. Estimating $N$ individual effects alongside $\beta$ produces inconsistent estimates when $T$ is small.

### Honore's Solution: Pairwise Differencing + Trimming

Honore (1992) takes **pairs of observations** $(y_{it_1}, y_{it_2})$ within each entity and applies a trimming rule:

1. For each entity $i$, form all pairs of time periods $(t, s)$ where $t < s$
2. Compute pairwise differences: $\Delta y = y_{it} - y_{is}$ and $\Delta X = X_{it} - X_{is}$
3. **Trim** pairs where both observations are censored (these are uninformative)
4. Minimize the trimmed LAD objective:

$$\hat{\beta} = \arg\min_\beta \sum_{i} \sum_{t < s} w_{its} \cdot |\Delta y_{its} - \Delta X_{its}'\beta|$$

where $w_{its}$ is the trimming indicator (1 if at least one observation in the pair is uncensored, 0 otherwise).

The differencing eliminates $\alpha_i$, and the trimming handles the censoring structure.

## Detailed Guide

### Model Specification

```python
from panelbox.models.censored import HonoreTrimmedEstimator

model = HonoreTrimmedEstimator(
    endog=y,                   # Dependent variable (censored)
    exog=X,                    # Regressors (n x k), no intercept
    groups=entity,             # Entity IDs
    time=time,                 # Time IDs
    censoring_point=0.0,       # Left-censoring threshold
)
```

!!! note "No intercept"
    The Honore estimator only identifies **slope** coefficients. Fixed effects (and thus the intercept) are differenced out. Do not include a constant column in `exog`.

### Estimation

```python
results = model.fit(
    method="L-BFGS-B",        # Optimization method
    maxiter=500,               # Maximum iterations
    tol=1e-6,                  # Convergence tolerance
    verbose=True,              # Print progress
)
```

!!! warning "Experimental"
    The Honore estimator is marked as experimental in PanelBox. It is computationally intensive for large datasets because it forms all pairwise differences within each entity ($O(N \cdot T^2)$ pairs). Use with caution on datasets with many time periods.

### Result Attributes

The `fit()` method returns a `HonoreResults` object:

| Attribute | Description |
|-----------|-------------|
| `results.params` | Estimated slope coefficients $\hat{\beta}$ |
| `results.converged` | Whether optimization converged |
| `results.n_iter` | Number of iterations |
| `results.n_obs` | Total number of observations |
| `results.n_entities` | Number of entities |
| `results.n_trimmed` | Number of trimmed (both-censored) pairs |

### Predictions

```python
# Linear predictions X'beta (without fixed effects)
y_pred = results.predict(exog=X_new)
```

Predictions return $X'\hat{\beta}$ only -- fixed effects are not recovered.

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | array-like | required | Dependent variable (censored) |
| `exog` | array-like | required | Regressors (no intercept) |
| `groups` | array-like | required | Entity identifiers |
| `time` | array-like | required | Time identifiers |
| `censoring_point` | float | `0.0` | Left-censoring threshold |

### fit() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_params` | array-like | `None` | Starting values (OLS on untrimmed diffs if `None`) |
| `method` | str | `"L-BFGS-B"` | Optimization method |
| `maxiter` | int | `500` | Maximum iterations |
| `tol` | float | `1e-6` | Convergence tolerance |
| `verbose` | bool | `True` | Print progress |

## Advantages and Limitations

### Advantages

- **Consistent** FE estimation with censored data
- **Semiparametric**: no distributional assumptions on $\varepsilon_{it}$ or $\alpha_i$
- Eliminates fixed effects via differencing (no incidental parameters problem)

### Limitations

- Only **slopes** are identified (no intercept or fixed effect recovery)
- Only **left censoring** is supported
- Requires $T \geq 2$ time periods per entity
- **No standard errors** are computed (the LAD objective is non-smooth; bootstrap is recommended for inference)
- Computationally intensive for large $T$ (pairwise differences grow as $T^2$)
- Less efficient than RE Tobit when the RE assumption holds

### When to Choose Honore vs. RE Tobit

| Criterion | Honore (FE) | RE Tobit |
|-----------|-------------|----------|
| $\text{Cov}(\alpha_i, X_{it}) \neq 0$ | Consistent | Inconsistent |
| Distributional assumptions | None | Normal errors + RE |
| Standard errors | Not provided (bootstrap) | From Hessian |
| Efficiency (if RE holds) | Lower | Higher |
| Intercept / FE recovery | No | Yes |

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Censored Models | Complete censored model comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/03_honore_estimator.ipynb) |

## See Also

- [Tobit Models](tobit.md) -- Pooled and RE Tobit for censored data
- [Panel Heckman](heckman.md) -- Sample selection correction
- [Marginal Effects for Censored Models](marginal-effects.md) -- Interpreting nonlinear effects

## References

- Honore, B. E. (1992). Trimmed LAD and least squares estimation of truncated and censored regression models with fixed effects. *Econometrica*, 60(3), 533-565.
- Honore, B. E. (2002). Nonlinear models with panel data. *Portuguese Economic Journal*, 1(2), 163-179.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapter 17.
