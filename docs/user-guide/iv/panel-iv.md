---
title: "Panel IV / 2SLS"
description: "Panel Instrumental Variables (Two-Stage Least Squares) estimation for endogenous regressors in PanelBox."
---

# Panel IV / Two-Stage Least Squares

!!! info "Quick Reference"
    **Class:** `panelbox.models.iv.PanelIV`
    **Import:** `from panelbox.models.iv import PanelIV`
    **Stata equivalent:** `xtivreg y x1 (endog = z1 z2), fe`
    **R equivalent:** `plm::plm(..., model="within", inst.method="bvk")`

## Overview

Instrumental Variables (IV) estimation addresses the problem of **endogenous regressors** -- explanatory variables that are correlated with the error term. This can arise from omitted variables, measurement error, simultaneity, or reverse causality. When endogeneity is present, OLS produces **biased and inconsistent** estimates.

The IV approach uses **instruments** $Z$ -- variables that are correlated with the endogenous regressor but uncorrelated with the error term -- to isolate the exogenous variation in $X$ and obtain consistent estimates. The most common implementation is **Two-Stage Least Squares (2SLS)**.

PanelBox's `PanelIV` supports pooled, fixed effects, and random effects specifications with a convenient formula syntax and comprehensive first-stage diagnostics.

## Quick Example

```python
from panelbox.models.iv import PanelIV

# Formula: y ~ exogenous + endogenous | instruments
model = PanelIV(
    formula="invest ~ capital + value | capital + lag_value + lag2_value",
    data=df,
    entity_col="firm",
    time_col="year",
    model_type="fe",
)
results = model.fit(cov_type="clustered")
print(results.summary())

# Check instrument strength
for var, fs in results.first_stage_results.items():
    print(f"{var}: F-stat = {fs['f_statistic']:.2f}")
```

## When to Use

- One or more regressors are **endogenous** ($\text{Cov}(X_{it}, \varepsilon_{it}) \neq 0$)
- You have **external instruments** that are correlated with the endogenous variable but uncorrelated with the error
- The first-stage relationship is **strong** (F-statistic > 10)
- Examples: returns to education (ability bias), demand estimation (simultaneity), peer effects (reflection problem)

!!! warning "Key Assumptions"
    - **Relevance**: $\text{Cov}(Z_{it}, X_{it}) \neq 0$ -- instruments must predict the endogenous variable
    - **Exogeneity**: $\text{Cov}(Z_{it}, \varepsilon_{it}) = 0$ -- instruments must be uncorrelated with the error
    - **Exclusion restriction**: instruments affect $y$ **only** through $X$, not directly

## The Endogeneity Problem

### Sources of Endogeneity

| Source | Example | Why OLS fails |
|--------|---------|---------------|
| Omitted variables | Ability affects both education and wages | $\text{Cov}(education, ability) \neq 0$ |
| Simultaneity | Price and quantity determined jointly | Supply and demand interact |
| Measurement error | True $X^*$ measured with noise $X = X^* + u$ | $\text{Cov}(X, \varepsilon) \neq 0$ by construction |
| Reverse causality | Does trade cause growth or growth cause trade? | Direction of causation unclear |

### The IV Solution

Find instruments $Z$ such that:

1. **Stage 1**: $X = Z'\pi + \nu$ (instruments predict the endogenous variable)
2. **Stage 2**: $y = \hat{X}'\beta + \varepsilon$ (use predicted $\hat{X}$ in place of $X$)

The predicted values $\hat{X}$ contain only the variation in $X$ that is driven by $Z$, which by assumption is uncorrelated with $\varepsilon$.

## Detailed Guide

### Formula Syntax

PanelIV uses a formula with `|` to separate the outcome equation from the instrument list:

```text
"y ~ exog_vars + endog_vars | all_instruments"
```

- **Before `|`**: all variables in the outcome equation (both exogenous and endogenous)
- **After `|`**: all instruments (exogenous controls appear on both sides; excluded instruments appear only after `|`)

**Identification rule**: Variables that appear before `|` but **not** after `|` are treated as endogenous. Variables that appear after `|` but **not** before `|` are the excluded instruments.

```python
# Example: 'value' is endogenous, instrumented by lag_value and lag2_value
# 'capital' is exogenous (appears on both sides)
model = PanelIV(
    formula="invest ~ capital + value | capital + lag_value + lag2_value",
    data=df,
    entity_col="firm",
    time_col="year",
)
```

In this example:

- `invest`: dependent variable
- `capital`: exogenous regressor
- `value`: endogenous regressor (in outcome but not in instruments)
- `lag_value`, `lag2_value`: excluded instruments (in instruments but not in outcome)

### Panel Specifications

```python
# Pooled IV (no entity effects)
model = PanelIV(formula, data, "id", "year", model_type="pooled")

# Fixed Effects IV (within transformation)
model = PanelIV(formula, data, "id", "year", model_type="fe")

# Random Effects IV
model = PanelIV(formula, data, "id", "year", model_type="re")
```

| `model_type` | Transformation | Intercept | Use when |
|--------------|---------------|-----------|----------|
| `"pooled"` | None | Yes | No entity heterogeneity |
| `"fe"` | Within (demean) | No | $\text{Cov}(\alpha_i, X_{it}) \neq 0$ |
| `"re"` | None | Yes | $\text{Cov}(\alpha_i, X_{it}) = 0$ |

### Estimation

```python
results = model.fit(cov_type="clustered")
```

### Result Attributes

| Attribute | Description |
|-----------|-------------|
| `results.params` | 2SLS coefficient estimates |
| `results.std_errors` | Standard errors |
| `results.cov_params` | Variance-covariance matrix |
| `results.resid` | Second-stage residuals |
| `results.first_stage_results` | Dict of first-stage results per endogenous var |
| `results.weak_instruments` | `True` if any first-stage F < 10 |
| `results.n_instruments` | Number of instruments |
| `results.n_endogenous` | Number of endogenous variables |
| `results.endogenous_vars` | List of endogenous variable names |
| `results.instruments` | List of instrument names |

### First-Stage Results

The first-stage results are stored as a dictionary keyed by endogenous variable name:

```python
for endog_var, fs in results.first_stage_results.items():
    print(f"\nFirst stage for '{endog_var}':")
    print(f"  F-statistic:  {fs['f_statistic']:.2f}")
    print(f"  R-squared:    {fs['rsquared']:.4f}")
```

| Key | Description |
|-----|-------------|
| `"fitted"` | Fitted values from first stage |
| `"gamma"` | First-stage coefficients |
| `"rsquared"` | First-stage $R^2$ |
| `"f_statistic"` | First-stage F-statistic |
| `"residuals"` | First-stage residuals |

## Standard Errors

PanelIV supports multiple covariance estimators:

```python
# Classical 2SLS (homoskedastic)
results = model.fit(cov_type="nonrobust")

# Heteroskedasticity-robust (HC1)
results = model.fit(cov_type="robust")

# Cluster-robust (by entity)
results = model.fit(cov_type="clustered")

# Two-way clustering (entity and time)
results = model.fit(cov_type="twoway")

# Driscoll-Kraay HAC
results = model.fit(cov_type="driscoll_kraay")
```

| `cov_type` | Description | When to use |
|------------|-------------|-------------|
| `"nonrobust"` | Classical 2SLS | Homoskedastic errors (textbook) |
| `"robust"` / `"hc1"` | HC1 robust | Heteroskedasticity suspected |
| `"clustered"` | Cluster by entity | Within-entity correlation (recommended) |
| `"twoway"` | Two-way clustering | Entity and time correlation |
| `"driscoll_kraay"` | DK HAC | Cross-sectional dependence |

!!! tip "Default recommendation"
    For panel data, `cov_type="clustered"` is the safest default. It accounts for arbitrary within-entity correlation and heteroskedasticity.

## Configuration Options

### PanelIV Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | str | required | `"y ~ exog + endog \| instruments"` |
| `data` | DataFrame | required | Panel data in long format |
| `entity_col` | str | required | Entity identifier column |
| `time_col` | str | required | Time identifier column |
| `model_type` | str | `"pooled"` | `"pooled"`, `"fe"`, or `"re"` |
| `weights` | array-like | `None` | Observation weights |

### fit() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cov_type` | str | `"nonrobust"` | Covariance type |
| `**cov_kwds` | dict | -- | Additional SE arguments (e.g., `cluster`, `maxlags`) |

## Identification

### Order Condition

The model requires at least as many **excluded instruments** as endogenous variables:

$$k_{excluded} \geq k_{endogenous}$$

where $k_{excluded}$ = number of instruments not in the outcome equation.

| Case | $k_{excluded}$ vs $k_{endog}$ | Name | Implication |
|------|-------------------------------|------|-------------|
| $k_{excluded} < k_{endog}$ | Under-identified | Cannot estimate | Need more instruments |
| $k_{excluded} = k_{endog}$ | Exactly identified | Unique solution | Cannot test overidentification |
| $k_{excluded} > k_{endog}$ | Over-identified | Can test validity | Sargan/Hansen J test available |

PanelBox raises a `ValueError` if the model is under-identified.

### Instrument Strength

A weak first-stage relationship leads to:

- Biased 2SLS estimates (toward OLS)
- Unreliable standard errors and test statistics
- Confidence intervals with poor coverage

The **first-stage F-statistic** is the key diagnostic:

```python
if results.weak_instruments:
    print("WARNING: Weak instruments detected!")
    print("Consider finding stronger instruments or using LIML.")
```

See [IV Diagnostics](diagnostics.md) for detailed guidance.

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Panel IV | 2SLS estimation with diagnostics | [Static Models Tutorial](../../tutorials/static-models.md) |

## See Also

- [IV Diagnostics](diagnostics.md) -- Instrument validity, weak instruments, and overidentification tests
- [GMM (Arellano-Bond)](../gmm/difference-gmm.md) -- Dynamic panels with internal instruments
- [Static Models](../static-models/index.md) -- OLS, FE, RE without endogeneity

## References

- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV regression. In D. W. K. Andrews & J. H. Stock (Eds.), *Identification and Inference for Econometric Models* (pp. 80-108). Cambridge University Press.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapters 5-6.
- Baltagi, B. H. (2013). *Econometric Analysis of Panel Data* (5th ed.). John Wiley & Sons.
