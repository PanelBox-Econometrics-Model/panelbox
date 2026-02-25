---
title: "PPML (Poisson Pseudo-Maximum Likelihood)"
description: "Guide to PPML estimation in PanelBox for gravity models — handles zeros, heteroskedasticity, and automatic elasticity computation for trade analysis."
---

# PPML (Poisson Pseudo-Maximum Likelihood)

!!! info "Quick Reference"
    **Class:** `panelbox.models.count.PPML`
    **Import:** `from panelbox.models.count import PPML`
    **Stata equivalent:** `ppmlhdfe`
    **R equivalent:** `fixest::fepois()`, `gravity::ppml()`

## Overview

The Poisson Pseudo-Maximum Likelihood (PPML) estimator is the standard approach for gravity models in international trade, migration, and FDI research. Santos Silva and Tenreyro (2006) showed that the traditional log-linear OLS gravity model:

$$\ln(\text{trade}_{ij}) = \beta_0 + \beta_1 \ln(\text{GDP}_i) + \beta_2 \ln(\text{GDP}_j) - \gamma \ln(\text{dist}_{ij}) + \varepsilon_{ij}$$

suffers from two fundamental problems: (1) it cannot handle zero trade flows since $\ln(0)$ is undefined, and (2) it produces inconsistent estimates under heteroskedasticity due to Jensen's inequality --- $E[\ln y] \neq \ln E[y]$.

PPML solves both problems by estimating the multiplicative model directly:

$$E[y_{ij} \mid X_{ij}] = \exp(X_{ij}'\beta)$$

PanelBox's `PPML` class wraps Poisson estimation with gravity-model-specific tools: automatic elasticity computation, OLS comparison, and cluster-robust standard errors.

## Quick Example

```python
from panelbox.models.count import PPML
import numpy as np

# Gravity model for bilateral trade
model = PPML(
    endog=data["trade_flow"],
    exog=data[["log_distance", "log_gdp_exp", "log_gdp_imp", "rta"]],
    entity_id=data["pair_id"],
    time_id=data["year"],
    fixed_effects=True,
    exog_names=["log_distance", "log_gdp_exp", "log_gdp_imp", "rta"]
)
results = model.fit()
print(results.summary())

# Elasticities
print(results.elasticities())
```

## When to Use

- **Gravity models**: international trade, migration, FDI flows
- **Zeros in the dependent variable**: trade flows, patent citations, any count with many zeros
- **Heteroskedasticity concerns**: PPML is consistent regardless of variance structure
- **Non-negative continuous outcomes**: PPML does not require integer-valued data
- **Multiplicative models**: whenever $E[y \mid X] = \exp(X'\beta)$ is the target

!!! warning "Key Assumptions"
    - **Correct mean specification**: $E[y \mid X] = \exp(X'\beta)$
    - **Non-negative outcome**: $y \geq 0$ (zeros are fine)
    - **No perfect separation**: some positive observations for each regressor pattern
    - Poisson distributional assumption is **not** required --- PPML is a QML estimator

## Detailed Guide

### The Gravity Model Problem

The standard gravity equation models trade between countries $i$ and $j$ as:

$$\text{Trade}_{ij} = A \cdot \frac{\text{GDP}_i^{\beta_1} \cdot \text{GDP}_j^{\beta_2}}{\text{Distance}_{ij}^{\gamma}} \cdot \exp(\delta \cdot \text{RTA}_{ij} + \varepsilon_{ij})$$

Log-linearizing gives OLS-estimable form, but this approach has three problems:

| Problem | Consequence |
|---------|-------------|
| **Zero trade flows** | $\ln(0)$ is undefined; dropping zeros causes selection bias |
| **Jensen's inequality** | $E[\ln y] \neq \ln E[y]$; OLS on logs estimates the wrong quantity |
| **Heteroskedasticity** | Log-linear OLS is inconsistent when $\text{Var}(\varepsilon \mid X)$ depends on $X$ |

### PPML Solution

PPML estimates $E[y \mid X] = \exp(X'\beta)$ directly using Poisson MLE:

- Handles zeros naturally (no log transformation)
- Consistent under heteroskedasticity (QML property)
- No Jensen's inequality bias
- No retransformation problem

### Data Preparation

```python
import pandas as pd
import numpy as np

# Standard gravity data preparation
df["log_distance"] = np.log(df["distance"])
df["log_gdp_exp"] = np.log(df["gdp_exporter"])
df["log_gdp_imp"] = np.log(df["gdp_importer"])
df["pair_id"] = df["exporter"] + "_" + df["importer"]

# Check zeros
n_zeros = (df["trade_flow"] == 0).sum()
pct_zeros = 100 * n_zeros / len(df)
print(f"Zero trade flows: {n_zeros} ({pct_zeros:.1f}%)")
print(f"PPML uses all {len(df)} observations (OLS would drop {n_zeros})")
```

### Estimation

```python
from panelbox.models.count import PPML

model = PPML(
    endog=df["trade_flow"],
    exog=df[["log_distance", "log_gdp_exp", "log_gdp_imp", "rta", "border"]],
    entity_id=df["pair_id"],
    time_id=df["year"],
    fixed_effects=True,
    exog_names=["log_distance", "log_gdp_exp", "log_gdp_imp", "rta", "border"]
)
results = model.fit(se_type="cluster")
```

!!! note "Cluster-Robust SE"
    PPML uses cluster-robust standard errors by default (and enforces them). This accounts for within-entity correlation and heteroskedasticity.

### Interpreting Results

#### Elasticities (Log-Transformed Variables)

For variables entered in logs (e.g., `log_distance`, `log_gdp`), the coefficient is the elasticity directly:

$$\frac{\partial \ln E[y]}{\partial \ln x} = \beta$$

```python
# Distance elasticity
dist_elast = results.elasticity("log_distance")
print(f"Distance elasticity: {dist_elast['elasticity']:.3f}")
# E.g., -1.2 means doubling distance reduces trade by 2^(-1.2) ~ 43%

# All elasticities as a table
print(results.elasticities())
```

The `elasticity()` method returns a dictionary with:

| Key | Description |
|-----|-------------|
| `coefficient` | Parameter estimate ($\beta$) |
| `se` | Standard error |
| `elasticity` | Elasticity value |
| `elasticity_se` | Standard error of elasticity (delta method) |
| `is_log_transformed` | Whether variable is log-transformed |

The `elasticities()` method returns a DataFrame with all variables.

#### Semi-Elasticities (Level Variables)

For variables in levels (e.g., binary indicators like `rta`, `border`), the coefficient is a semi-elasticity. The percentage effect on trade is:

$$\text{Percentage change} = 100 \times (\exp(\beta) - 1)\%$$

```python
# RTA effect
rta_coef = results.params[results.exog_names.index("rta")]
rta_pct = 100 * (np.exp(rta_coef) - 1)
print(f"RTA increases trade by {rta_pct:.1f}%")
```

### Comparing PPML with OLS

```python
from panelbox.models.static import PooledOLS

# OLS on log(trade) --- must drop zeros
df_positive = df[df["trade_flow"] > 0].copy()
df_positive["log_trade"] = np.log(df_positive["trade_flow"])

ols = PooledOLS(
    endog=df_positive["log_trade"],
    exog=df_positive[["log_distance", "log_gdp_exp", "log_gdp_imp", "rta"]],
    entity_id=df_positive["pair_id"],
    time_id=df_positive["year"]
)
ols_result = ols.fit(cov_type="clustered")

# Side-by-side comparison
comparison = results.compare_with_ols(ols_result)
print(comparison)
```

The comparison DataFrame includes coefficients, standard errors, and the difference between PPML and OLS estimates with t-statistics.

### Fixed Effects

With `fixed_effects=True`, PPML includes entity fixed effects that absorb all time-invariant characteristics:

```python
# Bilateral FE absorb time-invariant pair characteristics
model_fe = PPML(
    endog=df["trade_flow"],
    exog=df[["rta", "currency_union"]],  # Only time-varying regressors
    entity_id=df["pair_id"],
    time_id=df["year"],
    fixed_effects=True,
    exog_names=["rta", "currency_union"]
)
results_fe = model_fe.fit()
```

When using pair FE, time-invariant variables (distance, language, colonial ties, border) are absorbed and cannot be estimated directly.

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | array-like | *required* | Dependent variable (non-negative) |
| `exog` | array-like | *required* | Independent variables |
| `entity_id` | array-like | `None` | Entity identifiers |
| `time_id` | array-like | `None` | Time identifiers |
| `fixed_effects` | bool | `True` | Include entity fixed effects |
| `exog_names` | list | `None` | Variable names (for elasticity labeling) |

### fit() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `se_type` | str | `"cluster"` | Standard error type (enforced cluster-robust) |

## Diagnostics

### Model Fit

```python
# Log-likelihood and information criteria
print(f"Log-likelihood: {results.llf:.2f}")
print(f"AIC: {results.aic:.2f}")
print(f"BIC: {results.bic:.2f}")
```

### Predictions

```python
# Predicted trade flows
y_hat = results.predict()

# Compare with actual
correlation = np.corrcoef(df["trade_flow"], y_hat)[0, 1]
print(f"Correlation (predicted vs actual): {correlation:.4f}")
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Count Data Models | Complete guide including PPML gravity models | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/04_ppml_gravity.ipynb) |

## See Also

- [Count Data Overview](index.md) --- Introduction and model selection guide
- [Poisson Models](poisson.md) --- Pooled, FE, RE, and QML Poisson
- [Negative Binomial](negative-binomial.md) --- Handling overdispersion
- [Marginal Effects for Count Data](marginal-effects.md) --- Elasticities and interpretation

## References

- Santos Silva, J. M. C., & Tenreyro, S. (2006). The Log of Gravity. *Review of Economics and Statistics*, 88(4), 641--658.
- Santos Silva, J. M. C., & Tenreyro, S. (2010). On the Existence of the Maximum Likelihood Estimates in Poisson Regression. *Economics Letters*, 107(2), 310--312.
- Head, K., & Mayer, T. (2014). Gravity Equations: Workhorse, Toolkit, and Cookbook. *Handbook of International Economics*, 4, 131--195.
- Anderson, J. E., & van Wincoop, E. (2003). Gravity with Gravitas: A Solution to the Border Puzzle. *American Economic Review*, 93(1), 170--192.
