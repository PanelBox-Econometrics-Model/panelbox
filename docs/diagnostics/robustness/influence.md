---
title: "Influence Diagnostics"
description: "Identify observations with outsized impact on panel regression results using Cook's distance, DFFITS, and DFBETAS."
---

# Influence Diagnostics

!!! info "Quick Reference"
    **Class:** `panelbox.validation.robustness.InfluenceDiagnostics`
    **Import:** `from panelbox.validation.robustness import InfluenceDiagnostics`
    **Key method:** `influence.compute()` returns `InfluenceResults`
    **Stata equivalent:** `predict, cooksd dffits dfbeta(x1)`
    **R equivalent:** `stats::influence.measures()`, `car::influenceIndexPlot()`

## What It Measures

Influence diagnostics identify observations that have disproportionate impact on regression results. An influential observation is one whose removal would substantially change the estimated coefficients, fitted values, or predictions.

Three measures capture different aspects of influence:

| Measure | What It Measures | Default Threshold |
|---------|-----------------|:-----------------:|
| Cook's D | Overall influence on **all** coefficients jointly | $4/N$ |
| DFFITS | Influence on the observation's **own fitted value** | $2\sqrt{K/N}$ |
| DFBETAS | Influence on **each coefficient individually** | $2/\sqrt{N}$ |

## Mathematical Details

### Cook's Distance

Cook's distance combines residual magnitude and leverage into a single measure:

$$D_i = \frac{r_i^2}{K \cdot MSE} \cdot \frac{h_{ii}}{(1 - h_{ii})^2}$$

where:

- $r_i$ = residual for observation $i$
- $K$ = number of parameters
- $MSE$ = mean squared error
- $h_{ii}$ = leverage (hat value) for observation $i$

An observation can be influential through a large residual (outlier), high leverage (unusual predictor values), or both.

### DFFITS

DFFITS measures how much the fitted value $\hat{y}_i$ changes when observation $i$ is deleted:

$$DFFITS_i = r_i^{*} \sqrt{\frac{h_{ii}}{1 - h_{ii}}}$$

where $r_i^{*}$ is the standardized residual $r_i / \sqrt{MSE \cdot (1 - h_{ii})}$.

### DFBETAS

DFBETAS measures how much each individual coefficient $\hat\beta_j$ changes when observation $i$ is deleted:

$$DFBETAS_{ij} = \frac{\hat\beta_j - \hat\beta_j^{(-i)}}{SE(\hat\beta_j^{(-i)})}$$

This reveals which specific coefficients are most affected by each observation.

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.validation.robustness import InfluenceDiagnostics
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit()

# Compute all influence diagnostics
influence = InfluenceDiagnostics(results, verbose=True)
infl = influence.compute()

# Access individual measures
print(f"Max Cook's D: {infl.cooks_d.max():.6f}")
print(f"Max |DFFITS|: {infl.dffits.abs().max():.6f}")

# Find influential observations
influential = influence.influential_observations(method="cooks_d")
print(influential)

# Summary with top 10 most influential observations
print(influence.summary())

# 4-panel influence plot
influence.plot_influence()
```

## API Reference

### Constructor

```python
InfluenceDiagnostics(
    results=results,   # PanelResults from model.fit()
    verbose=True,      # Print progress
)
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `compute()` | `InfluenceResults` | Compute Cook's D, DFFITS, DFBETAS, leverage |
| `influential_observations(method, threshold)` | `pd.DataFrame` | Observations exceeding influence threshold |
| `summary(n_top=10)` | `str` | Formatted summary of top influential observations |
| `plot_influence(save_path=None)` | -- | 4-panel influence diagnostic plot |

### Properties (auto-compute on access)

| Property | Type | Description |
|----------|------|-------------|
| `leverage` | `pd.Series` | Approximate leverage (hat) values |
| `cooks_d` | `pd.Series` | Cook's distance |
| `dffits` | `pd.Series` | DFFITS values |
| `dfbetas` | `pd.DataFrame` | DFBETAS (observations $\times$ parameters) |

These properties call `compute()` automatically if it hasn't been called yet.

### InfluenceResults Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `cooks_d` | `pd.Series` | Cook's distance for each observation |
| `dffits` | `pd.Series` | DFFITS for each observation |
| `dfbetas` | `pd.DataFrame` | DFBETAS ($N \times K$) |
| `leverage` | `pd.Series` | Approximate leverage values |
| `standardized_residuals` | `pd.Series` | Standardized residuals |

## Identifying Influential Observations

```python
# By Cook's distance (default threshold: 4/N)
influential_cook = influence.influential_observations(method="cooks_d")

# By DFFITS (default threshold: 2*sqrt(K/N))
influential_dffits = influence.influential_observations(method="dffits")

# By DFBETAS (default threshold: 2/sqrt(N))
influential_dfbetas = influence.influential_observations(method="dfbetas")

# Custom threshold
influential_custom = influence.influential_observations(method="cooks_d", threshold=0.5)
```

### Default Thresholds

| Method | Default Threshold | Formula |
|--------|:-----------------:|---------|
| `cooks_d` | $4/N$ | Flags approximately 0.4% of observations in large samples |
| `dffits` | $2\sqrt{K/N}$ | Scales with model complexity and sample size |
| `dfbetas` | $2/\sqrt{N}$ | Per-coefficient threshold; flags if *any* coefficient is affected |

Where $N$ = number of observations and $K$ = number of parameters.

## Visualization

```python
influence.plot_influence()
```

Produces a 4-panel figure:

1. **Cook's Distance** -- Stem plot with $4/N$ threshold line
2. **DFFITS** -- Stem plot with $\pm 2\sqrt{K/N}$ threshold lines
3. **Leverage vs Residuals** -- Scatter of leverage against standardized residuals (high in both corners = influential)
4. **Leverage Values** -- Stem plot with $2 \times \bar{h}$ threshold line

## Interpretation Guide

### Cook's Distance

| Range | Interpretation |
|-------|---------------|
| $D_i < 4/N$ | Not influential |
| $4/N < D_i < 1$ | Moderately influential; investigate |
| $D_i > 1$ | Highly influential; strongly affects results |

### DFFITS

| Range | Interpretation |
|-------|---------------|
| $\|DFFITS_i\| < 2\sqrt{K/N}$ | Not influential on fitted value |
| $\|DFFITS_i\| > 2\sqrt{K/N}$ | Substantially changes own fitted value when removed |

### DFBETAS

| Range | Interpretation |
|-------|---------------|
| $\|DFBETAS_{ij}\| < 2/\sqrt{N}$ | Not influential on coefficient $j$ |
| $\|DFBETAS_{ij}\| > 2/\sqrt{N}$ | Substantially changes coefficient $j$ when removed |

!!! tip "Which Measure to Use?"
    - Start with **Cook's D** for an overall picture
    - Use **DFFITS** to assess prediction impact
    - Use **DFBETAS** when you need to know which specific coefficients are affected
    - Always examine observations flagged by any measure

## Panel Context

In panel data, influential observations may cluster within entities or time periods:

```python
import pandas as pd

# Compute influence
influence = InfluenceDiagnostics(results)
infl = influence.compute()

# Merge with panel identifiers
model = results._model
infl_data = pd.DataFrame({
    "entity": model.data.data[model.data.entity_col].values,
    "time": model.data.data[model.data.time_col].values,
    "cooks_d": infl.cooks_d.values,
})

# Check if influence clusters by entity
entity_influence = infl_data.groupby("entity")["cooks_d"].mean()
print("Mean Cook's D by entity:")
print(entity_influence.sort_values(ascending=False))

# Check if influence clusters by time period
time_influence = infl_data.groupby("time")["cooks_d"].mean()
print("\nMean Cook's D by time period:")
print(time_influence.sort_values(ascending=False))
```

If influence clusters within a specific entity or time period, consider whether that unit has unique characteristics that warrant separate treatment or additional controls.

## Comprehensive Example

```python
from panelbox import FixedEffects
from panelbox.validation.robustness import InfluenceDiagnostics
from panelbox.datasets import load_grunfeld
import numpy as np

data = load_grunfeld()
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit()

# Full influence analysis
influence = InfluenceDiagnostics(results, verbose=True)
infl = influence.compute()

# Thresholds
n = len(infl.cooks_d)
k = len(results.params)
print(f"N = {n}, K = {k}")
print(f"Cook's D threshold (4/N):          {4/n:.6f}")
print(f"DFFITS threshold (2*sqrt(K/N)):    {2*np.sqrt(k/n):.6f}")
print(f"DFBETAS threshold (2/sqrt(N)):     {2/np.sqrt(n):.6f}")

# Count influential observations by each measure
n_cook = (infl.cooks_d > 4/n).sum()
n_dffits = (infl.dffits.abs() > 2*np.sqrt(k/n)).sum()
n_dfbetas = (infl.dfbetas.abs() > 2/np.sqrt(n)).any(axis=1).sum()

print(f"\nInfluential by Cook's D: {n_cook}")
print(f"Influential by DFFITS:   {n_dffits}")
print(f"Influential by DFBETAS:  {n_dfbetas}")

# Detailed summary
print(influence.summary())

# Visualization
influence.plot_influence()
```

## Common Pitfalls

!!! warning "Watch Out"

    1. **Approximate leverage**: For panel models with entity fixed effects, PanelBox uses an approximate leverage based on the Mahalanobis distance of the design matrix. Exact leverage requires the full hat matrix, which may be too large for big panels.
    2. **Multiple flagged observations**: Influence is computed one observation at a time. If multiple influential observations are present, deleting one may reveal (or mask) others.
    3. **Don't remove without investigation**: Finding high Cook's D does not mean the observation should be deleted. It means it deserves scrutiny.
    4. **Small panels**: In small panels, a few observations can have high leverage by construction. Thresholds like $4/N$ may flag too many observations; consider using $1.0$ as an alternative Cook's D cutoff.

## See Also

- [Outlier Detection](outliers.md) -- Residual-based and multivariate outlier identification
- [Jackknife Analysis](jackknife.md) -- Entity-level leave-one-out influence
- [Sensitivity Analysis](sensitivity.md) -- Parameter stability across subsamples
- [Robustness Overview](index.md) -- Full robustness toolkit

## References

- Cook, R. D. (1977). Detection of influential observation in linear regression. *Technometrics*, 19(1), 15-18.
- Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity*. John Wiley & Sons.
- Cook, R. D., & Weisberg, S. (1982). *Residuals and Influence in Regression*. Chapman and Hall.
- Welsch, R. E., & Kuh, E. (1977). Linear regression diagnostics. *NBER Working Paper No. 173*.
