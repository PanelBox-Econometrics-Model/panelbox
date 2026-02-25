---
title: "Dynamic Panel Quantile Regression"
description: "Dynamic quantile regression with lagged dependent variables and IV methods in PanelBox"
---

# Dynamic Panel Quantile Regression

!!! info "Quick Reference"
    **Class:** `panelbox.models.quantile.dynamic.DynamicQuantile`
    **Import:** `from panelbox.models.quantile import DynamicQuantile`
    **Stata equivalent:** `ivqreg y x1 x2 (L.y = L2.y L3.y), quantile(0.5)`
    **R equivalent:** `ivqr::ivqr()` (for IV approach)

## Overview

Dynamic Panel Quantile Regression extends standard panel quantile models to include lagged dependent variables. The model is:

$$Q_{y_{it}}(\tau | y_{i,t-1}, X_{it}, \alpha_i) = \rho(\tau) y_{i,t-1} + X_{it}'\beta(\tau) + \alpha_i$$

The key challenge is that the lagged dependent variable $y_{i,t-1}$ is endogenous — it is correlated with the fixed effect $\alpha_i$ and potentially with the error term. This endogeneity is present in both mean and quantile regression, but quantile regression lacks the within-transformation that partially addresses it in OLS.

PanelBox implements three approaches to handle this endogeneity:

- **IV approach** (Galvao, 2011): uses deeper lags as instruments
- **Quantile Control Function** (Powell, 2016): two-step control function approach
- **GMM approach**: Arellano-Bond type moment conditions adapted for quantile regression

The persistence parameter $\rho(\tau)$ captures how past outcomes affect current outcomes at different parts of the distribution. If $\rho(0.9) > \rho(0.1)$, high-outcome observations exhibit more persistence than low-outcome observations.

## Quick Example

```python
from panelbox.core.panel_data import PanelData
from panelbox.models.quantile import DynamicQuantile

panel_data = PanelData(data=df, entity_col="firm_id", time_col="year")

model = DynamicQuantile(
    data=panel_data,
    formula="investment ~ value + capital",
    tau=[0.25, 0.5, 0.75],
    lags=1,
    method="iv",
)
results = model.fit(iv_lags=2, bootstrap=True, n_boot=100)
```

## When to Use

- **Dynamic processes**: outcomes persist over time (e.g., earnings, investment, GDP)
- **State dependence**: past outcomes causally affect current outcomes
- **Quantile-specific persistence**: $\rho(\tau)$ varies across the distribution
- **Short-run vs long-run effects**: separate transitory from permanent impacts
- **Heterogeneous dynamics**: adjustment speed differs at different quantiles

!!! warning "Key Assumptions"
    - **Balanced panel required**: observations must be available for all entities at all time periods
    - **Valid instruments**: deeper lags ($y_{i,t-2}, y_{i,t-3}, \ldots$) are uncorrelated with current errors
    - **Sufficient time periods**: $T$ must be large enough for the instruments to be relevant ($T \geq 4$ minimum)
    - **Sequential exogeneity**: $E[\rho_\tau'(\varepsilon_{it}) | y_{i,t-1}, X_{it}, \alpha_i] = 0$

## Detailed Guide

### The Endogeneity Problem

In a dynamic quantile model, $y_{i,t-1}$ is correlated with $\alpha_i$ by construction (past $y$ depends on the fixed effect). The standard within-transformation does not eliminate this correlation in quantile regression because the check loss function is not linear.

Three solutions are available:

### Method 1: Instrumental Variables (Galvao 2011)

Uses deeper lags $y_{i,t-2}, y_{i,t-3}, \ldots$ as instruments for $y_{i,t-1}$:

```python
model = DynamicQuantile(
    data=panel_data,
    formula="y ~ x1 + x2",
    tau=0.5,
    lags=1,
    method="iv",     # Galvao (2011) IV approach
)
results = model.fit(
    iv_lags=2,       # use y_{t-2} and y_{t-3} as instruments
    bootstrap=True,
    n_boot=100,
)
```

The IV procedure:

1. First-difference to remove $\alpha_i$: $\Delta y_{it} = \rho \Delta y_{i,t-1} + \Delta X_{it}'\beta + \Delta \varepsilon_{it}$
2. Use $y_{i,t-2}, y_{i,t-3}$ as instruments for $\Delta y_{i,t-1}$
3. Estimate by instrumental variable quantile regression

### Method 2: Quantile Control Function (Powell 2016)

A two-step approach that controls for endogeneity via a control function:

```python
model = DynamicQuantile(
    data=panel_data,
    formula="y ~ x1 + x2",
    tau=0.5,
    lags=1,
    method="qcf",     # Powell (2016) control function
)
results = model.fit(bootstrap=True, n_boot=100)
```

### Method 3: GMM Approach

Uses Arellano-Bond type moment conditions:

```python
model = DynamicQuantile(
    data=panel_data,
    formula="y ~ x1 + x2",
    tau=0.5,
    lags=1,
    method="gmm",
)
results = model.fit(iv_lags=2)
```

### Estimation with Bootstrap Inference

Bootstrap is recommended for dynamic quantile models because analytical standard errors are complex:

```python
results = model.fit(
    iv_lags=2,         # instrument depth
    bootstrap=True,    # bootstrap inference
    n_boot=100,        # bootstrap replications
    verbose=True,      # print progress
)
```

### Interpreting Results

```python
# Access results for each quantile
for tau in model.tau:
    r = results.results[tau]
    print(f"tau={tau:.2f}:")
    print(f"  Lag coefficient (rho): {r.params[0]:.4f}")
    print(f"  Other coefficients: {r.params[1:]}")
    if hasattr(r, "se_boot"):
        print(f"  Bootstrap SEs: {r.se_boot}")
```

**Interpreting $\rho(\tau)$**:

- $\rho(\tau) > 0$: positive persistence at quantile $\tau$
- $\rho(\tau) \approx 1$: near-unit root behavior (strong persistence)
- $\rho(\tau)$ increasing in $\tau$: high-outcome observations are more persistent
- $\rho(\tau)$ decreasing in $\tau$: mean reversion is stronger at the top

### Choosing the Number of Instrument Lags

More instrument lags increase efficiency but may weaken relevance:

```python
# Compare different instrument depths
for iv_depth in [1, 2, 3]:
    model = DynamicQuantile(data=panel_data, formula="y ~ x1 + x2",
                             tau=0.5, lags=1, method="iv")
    results = model.fit(iv_lags=iv_depth)
    print(f"iv_lags={iv_depth}: rho = {results.results[0.5].params[0]:.4f}")
```

!!! tip "Rule of Thumb"
    Start with `iv_lags=2` (instruments $y_{i,t-2}$ and $y_{i,t-3}$). If results are unstable, try `iv_lags=1` for fewer but stronger instruments.

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | PanelData | *required* | Panel data object |
| `formula` | str | `None` | Model formula `"y ~ x1 + x2"` |
| `tau` | float/array | `0.5` | Quantile level(s) in $(0, 1)$ |
| `lags` | int | `1` | Number of dependent variable lags |
| `method` | str | `"iv"` | Estimation: `"iv"`, `"qcf"`, `"gmm"` |

### Fit Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iv_lags` | int | `2` | Additional lags used as instruments |
| `bootstrap` | bool | `False` | Use bootstrap for inference |
| `n_boot` | int | `100` | Number of bootstrap replications |
| `verbose` | bool | `False` | Print estimation progress |

## Diagnostics

### Checking Instrument Validity

```python
# The lag coefficient should be stable across instrument specifications
for iv_depth in [1, 2, 3, 4]:
    model = DynamicQuantile(data=panel_data, formula="y ~ x1",
                             tau=0.5, lags=1, method="iv")
    r = model.fit(iv_lags=iv_depth)
    print(f"iv_lags={iv_depth}: rho={r.results[0.5].params[0]:.4f}")
# Large variation suggests instrument problems
```

### Quantile Process

```python
# Estimate across many quantiles to trace persistence
tau_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
model = DynamicQuantile(data=panel_data, formula="y ~ x1",
                         tau=tau_grid, lags=1, method="iv")
results = model.fit(iv_lags=2, bootstrap=True, n_boot=100)

# Examine how persistence varies across quantiles
rho_values = [results.results[tau].params[0] for tau in tau_grid]
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Dynamic Quantile | IV estimation with lagged dependent variables | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/10_dynamic_quantile_models.ipynb) |

## See Also

- [Pooled Quantile Regression](pooled.md) — static quantile model without lags
- [Fixed Effects Quantile Regression](fixed-effects.md) — static model with FE
- [Diagnostics](diagnostics.md) — bootstrap inference and model comparison
- [Difference GMM](../gmm/difference-gmm.md) — Arellano-Bond for mean regression

## References

- Galvao, A. F. (2011). Quantile regression for dynamic panel data with fixed effects. *Journal of Econometrics*, 164(1), 142-157.
- Powell, D. (2016). Quantile treatment effects in the presence of covariates. *RAND Working Paper*.
- Arellano, M., & Bonhomme, S. (2016). Nonlinear panel data estimation via quantile regressions. *The Econometrics Journal*, 19(3), C61-C94.
- Galvao, A. F., & Kato, K. (2016). Smoothed quantile regression for panel data. *Journal of Econometrics*, 193(1), 92-112.
