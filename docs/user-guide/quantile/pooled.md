---
title: "Pooled Quantile Regression"
description: "Pooled quantile regression for panel data with cluster-robust standard errors in PanelBox"
---

# Pooled Quantile Regression

!!! info "Quick Reference"
    **Class:** `panelbox.models.quantile.pooled.PooledQuantile`
    **Import:** `from panelbox.models.quantile import PooledQuantile`
    **Stata equivalent:** `qreg y x1 x2, vce(cluster id)`
    **R equivalent:** `quantreg::rq(y ~ x1 + x2, tau = 0.5, data = df)`

## Overview

Pooled Quantile Regression estimates conditional quantile functions by pooling all observations across entities and time periods. While standard OLS estimates the conditional **mean** $E[y|X]$, quantile regression estimates the conditional **quantile** $Q_\tau(y|X)$ for any quantile level $\tau \in (0,1)$.

The model was introduced by Koenker and Bassett (1978) and solves:

$$\min_{\beta} \sum_{i=1}^{N} \sum_{t=1}^{T} \rho_\tau(y_{it} - X_{it}'\beta_\tau)$$

where $\rho_\tau(u) = u(\tau - \mathbb{1}\{u < 0\})$ is the **check loss function** (also called the pinball loss). This asymmetric loss function penalizes positive and negative residuals differently depending on $\tau$, producing estimates of the $\tau$-th conditional quantile.

Pooled quantile regression ignores panel structure in estimation (no fixed effects), but PanelBox provides cluster-robust standard errors by entity to account for within-entity correlation.

## Quick Example

```python
import numpy as np
from panelbox.models.quantile import PooledQuantile

# Generate panel data
np.random.seed(42)
n_entities, n_time = 50, 10
n_obs = n_entities * n_time
entity_id = np.repeat(np.arange(n_entities), n_time)
time_id = np.tile(np.arange(n_time), n_entities)

X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, 2)])
y = X @ np.array([1.0, 0.5, -0.3]) + np.random.randn(n_obs)

# Estimate median regression
model = PooledQuantile(endog=y, exog=X, entity_id=entity_id,
                       time_id=time_id, quantiles=0.5)
results = model.fit(se_type="cluster")
print(results.summary())
```

## When to Use

- **Baseline analysis**: start with pooled quantile regression before adding fixed effects
- **Heterogeneous effects**: examine how covariate effects vary across the conditional distribution
- **Robustness to outliers**: median regression ($\tau=0.5$) is robust to outliers, unlike OLS
- **Distributional analysis**: characterize the full conditional distribution, not just the mean
- **Inequality research**: study effects at tails (e.g., $\tau=0.10$ vs $\tau=0.90$)

!!! warning "Key Assumptions"
    - **Linear conditional quantile**: $Q_\tau(y|X) = X'\beta_\tau$
    - **i.i.d. across entities** (relaxed with cluster-robust SEs)
    - **No unobserved heterogeneity** — if entity-level confounders exist, use [Fixed Effects QR](fixed-effects.md) or [Canay Two-Step](canay.md)

## Detailed Guide

### Data Preparation

PooledQuantile accepts NumPy arrays or Pandas objects directly:

```python
import pandas as pd
from panelbox.models.quantile import PooledQuantile

# From arrays
model = PooledQuantile(
    endog=y,              # (n_obs,) dependent variable
    exog=X,               # (n_obs, k) independent variables
    entity_id=entity_id,  # entity identifiers (for clustering)
    time_id=time_id,      # time identifiers
    quantiles=0.5,        # quantile level(s)
    weights=None,          # observation weights (optional)
)

# From DataFrame (preserves variable names)
df = pd.DataFrame({"y": y, "x1": X[:, 1], "x2": X[:, 2]})
X_df = pd.DataFrame({"const": 1, "x1": df["x1"], "x2": df["x2"]})
model = PooledQuantile(endog=df["y"], exog=X_df,
                       entity_id=entity_id, quantiles=[0.25, 0.5, 0.75])
```

### Estimation

The `fit()` method uses the interior point algorithm (Frisch-Newton) for efficient estimation:

```python
results = model.fit(
    method="interior_point",  # optimization algorithm
    maxiter=1000,              # maximum iterations
    tol=1e-6,                  # convergence tolerance
    se_type="cluster",         # standard error type
    alpha=0.05,                # significance level for CIs
)
```

### Multiple Quantiles

Estimate several quantile levels simultaneously to trace out the conditional distribution:

```python
model = PooledQuantile(endog=y, exog=X, entity_id=entity_id,
                       quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
results = model.fit(se_type="cluster")

# Access results for each quantile
for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
    r = results.results[tau]
    print(f"tau={tau:.2f}: beta = {r.params}")
```

### Interpreting Results

```python
# Point estimates
results.results[0.5].params        # coefficients at median
results.results[0.5].std_errors    # standard errors
results.results[0.5].tvalues       # t-statistics
results.results[0.5].pvalues       # p-values
results.results[0.5].converged     # convergence flag

# Compare effects across quantiles
# A coefficient that increases with tau indicates
# larger effects in the upper tail of the distribution
```

**Interpretation**: $\hat{\beta}_\tau$ measures the marginal effect of $X$ on the $\tau$-th quantile of $y$. If $\hat{\beta}_{0.9} > \hat{\beta}_{0.1}$, the covariate has a larger effect at the top of the distribution, indicating heterogeneous effects.

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | array | *required* | Dependent variable $(n,)$ |
| `exog` | array | *required* | Independent variables $(n, k)$ |
| `entity_id` | array | `None` | Entity identifiers for clustering |
| `time_id` | array | `None` | Time identifiers |
| `quantiles` | float/array | `0.5` | Quantile level(s) in $(0, 1)$ |
| `weights` | array | `None` | Observation weights |

### Fit Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `"interior_point"` | Optimization: `"interior_point"`, `"gradient_descent"` |
| `maxiter` | int | `1000` | Maximum iterations |
| `tol` | float | `1e-6` | Convergence tolerance |
| `se_type` | str | `"cluster"` | SE type: `"cluster"`, `"robust"`, `"nonrobust"` |
| `alpha` | float | `0.05` | Significance level for confidence intervals |

## Standard Errors

| Type | Description | When to Use |
|------|-------------|-------------|
| `"cluster"` | Cluster-robust by entity | Default for panel data — accounts for within-entity correlation |
| `"robust"` | Heteroskedasticity-robust (sandwich) | Cross-sectional data or when clustering is unnecessary |
| `"nonrobust"` | Classical i.i.d. standard errors | Homoskedastic errors assumed |

## Diagnostics

After fitting, compare quantile regression with OLS to assess heterogeneity:

```python
# Compare median regression with OLS
from panelbox.models import PooledOLS

ols_model = PooledOLS(endog=y, exog=X)
ols_results = ols_model.fit()

# If coefficients differ substantially across quantiles,
# there is evidence of distributional heterogeneity
print("OLS:     ", ols_results.params)
print("QR(0.25):", results.results[0.25].params)
print("QR(0.50):", results.results[0.50].params)
print("QR(0.75):", results.results[0.75].params)
```

Check for crossing quantiles when using multiple quantile levels:

```python
from panelbox.models.quantile import QuantileMonotonicity

report = QuantileMonotonicity.detect_crossing(results.results, X)
report.summary()
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Quantile Regression Basics | Introduction to panel quantile regression | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/01_quantile_regression_fundamentals.ipynb) |
| Comparing QR Methods | Pooled vs FE vs Canay comparison | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/02_multiple_quantiles_process.ipynb) |

## See Also

- [Fixed Effects Quantile Regression](fixed-effects.md) — control for entity-level heterogeneity
- [Canay Two-Step](canay.md) — computationally efficient FE quantile regression
- [Location-Scale Model](location-scale.md) — non-crossing quantile curves by construction
- [Non-Crossing Constraints](monotonicity.md) — detect and fix crossing quantile curves
- [Diagnostics](diagnostics.md) — quantile regression diagnostic tests

## References

- Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Parente, P. M. D. C., & Santos Silva, J. M. C. (2016). Quantile regression with clustered data. *Journal of Econometric Methods*, 5(1), 1-15.
