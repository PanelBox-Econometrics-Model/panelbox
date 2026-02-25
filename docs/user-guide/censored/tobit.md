---
title: "Tobit Models"
description: "Pooled and Random Effects Tobit models for censored panel data in PanelBox."
---

# Tobit Models

!!! info "Quick Reference"
    **Classes:** `panelbox.models.censored.PooledTobit`, `panelbox.models.censored.RandomEffectsTobit`
    **Import:** `from panelbox.models.censored import PooledTobit, RandomEffectsTobit`
    **Stata equivalent:** `tobit` (pooled), `xttobit` (RE)
    **R equivalent:** `censReg::censReg()`, `pglm::pglm()`

## Overview

The Tobit model (Tobin, 1958) is the standard approach for regression with **censored dependent variables** -- outcomes that are observed at a boundary value rather than their true latent value. Common examples include hours worked (censored at zero), expenditure on durable goods, and insurance claims.

The key insight is that the observed outcome is a censored version of a latent variable:

$$y_{it}^* = X_{it}'\beta + \varepsilon_{it}$$

$$y_{it} = \max(c, y_{it}^*)$$

where $c$ is the censoring point. Standard OLS ignores the pile-up of observations at the boundary, producing **biased and inconsistent** estimates. The Tobit model accounts for censoring by modeling the likelihood of being at the boundary versus being uncensored.

PanelBox provides two Tobit specifications: **PooledTobit** (ignores panel structure) and **RandomEffectsTobit** (accounts for entity-level heterogeneity via random effects with Gauss-Hermite quadrature integration).

## Quick Example

```python
import numpy as np
from panelbox.models.censored import PooledTobit

# Simulate censored data
np.random.seed(42)
n = 500
X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
y_star = X @ np.array([1.0, 0.5, -0.3]) + np.random.randn(n)
y = np.maximum(0, y_star)  # Left-censored at 0
groups = np.repeat(np.arange(50), 10)

# Fit Pooled Tobit
model = PooledTobit(endog=y, exog=X, groups=groups, censoring_point=0.0)
result = model.fit(method="BFGS")
print(result.summary())
```

## When to Use

- Your dependent variable is **censored** at a known boundary (e.g., 0, 100)
- You observe the boundary value for censored observations (not missing)
- You want to estimate effects on the **latent** (uncensored) outcome
- The censoring mechanism is **exogenous** (not related to unobservables)

!!! warning "Key Assumptions"
    - **Normality**: errors $\varepsilon_{it} \sim N(0, \sigma^2)$ (pooled) or $\varepsilon_{it} \sim N(0, \sigma_\varepsilon^2)$ (RE)
    - **Exogenous censoring**: the censoring point is fixed and known
    - **Linearity**: the latent variable is linear in $X$
    - **RE Tobit additionally assumes**: $\alpha_i \sim N(0, \sigma_\alpha^2)$, independent of $X$

## Censoring vs. Truncation

It is important to distinguish censoring from truncation and sample selection:

| Problem | What happens | Observed data | Model |
|---------|-------------|---------------|-------|
| **Censoring** | $y^*$ is limited to $[c, \infty)$; observe $y = \max(c, y^*)$ | All observations, some at boundary | Tobit |
| **Truncation** | Observations with $y^* \leq c$ are dropped entirely | Only uncensored observations | Truncated regression |
| **Selection** | $y$ observed only if a separate condition holds | Outcome missing for some | Heckman |

With censoring, you still **observe** the censored value (e.g., zero hours worked). With truncation, those observations are completely absent from the data.

## Detailed Guide

### Censoring Types

PanelBox supports three types of censoring via the `censoring_type` parameter:

| Type | Formula | Example |
|------|---------|---------|
| `"left"` (default) | $y = \max(c, y^*)$ | Hours worked $\geq 0$ |
| `"right"` | $y = \min(c, y^*)$ | Test scores $\leq 100$ |
| `"both"` | $y = \max(l, \min(u, y^*))$ | Satisfaction score in $[1, 5]$ |

### PooledTobit

The Pooled Tobit ignores the panel structure, treating all observations as independent. It is suitable when entity-level heterogeneity is not a concern or as a baseline model.

```python
from panelbox.models.censored import PooledTobit

model = PooledTobit(
    endog=y,                    # Dependent variable (censored)
    exog=X,                     # Regressors (n x k)
    groups=entity,              # Entity IDs (for clustered SEs)
    censoring_point=0.0,        # Censoring threshold
    censoring_type="left",      # 'left', 'right', or 'both'
)
result = model.fit(method="BFGS", maxiter=1000)
```

**Key attributes after fitting:**

| Attribute | Description |
|-----------|-------------|
| `result.beta` | Coefficient vector $\hat{\beta}$ |
| `result.sigma` | Error standard deviation $\hat{\sigma}$ |
| `result.llf` | Log-likelihood value |
| `result.bse` | Standard errors |
| `result.converged` | Whether optimization converged |

### RandomEffectsTobit

The RE Tobit adds entity-specific random effects to account for unobserved heterogeneity:

$$y_{it}^* = X_{it}'\beta + \alpha_i + \varepsilon_{it}$$

where $\alpha_i \sim N(0, \sigma_\alpha^2)$ and $\varepsilon_{it} \sim N(0, \sigma_\varepsilon^2)$. The random effect is integrated out using **Gauss-Hermite quadrature**.

```python
from panelbox.models.censored import RandomEffectsTobit

model = RandomEffectsTobit(
    endog=y,                    # Dependent variable
    exog=X,                     # Regressors
    groups=entity,              # Entity IDs
    time=time,                  # Time IDs
    censoring_point=0.0,        # Censoring threshold
    censoring_type="left",      # Censoring type
    quadrature_points=12,       # Integration accuracy
)
result = model.fit(method="BFGS", maxiter=1000)
```

**Additional attributes for RE Tobit:**

| Attribute | Description |
|-----------|-------------|
| `result.sigma_eps` | Idiosyncratic error SD $\hat{\sigma}_\varepsilon$ |
| `result.sigma_alpha` | Random effect SD $\hat{\sigma}_\alpha$ |

!!! tip "Quadrature points"
    The `quadrature_points` parameter controls the accuracy of the numerical integration over the random effect distribution. Higher values (e.g., 20) increase accuracy but slow computation. The default of 12 is adequate for most applications.

### Double Censoring

For outcomes censored at both ends, use `censoring_type="both"` with explicit limits:

```python
model = PooledTobit(
    endog=y,
    exog=X,
    groups=entity,
    censoring_type="both",
    lower_limit=1.0,        # Lower bound
    upper_limit=5.0,        # Upper bound
)
result = model.fit()
```

### Predictions

Tobit models offer three types of predictions, each with a different interpretation:

=== "Latent"

    ```python
    # E[y*|X] = X'beta (ignores censoring)
    y_latent = result.predict(pred_type="latent")
    ```

    The expected value of the latent variable, as if there were no censoring. This can produce values below the censoring point.

=== "Censored"

    ```python
    # E[y|X] accounting for censoring
    y_censored = result.predict(pred_type="censored")
    ```

    The expected value of the observed (censored) outcome. For left censoring at $c$:

    $$E[y|X] = X'\beta \cdot \Phi\!\left(\frac{X'\beta - c}{\sigma}\right) + c \cdot \left[1 - \Phi\!\left(\frac{X'\beta - c}{\sigma}\right)\right] + \sigma \cdot \phi\!\left(\frac{X'\beta - c}{\sigma}\right)$$

=== "Probability"

    ```python
    # P(y > c | X) — probability of being uncensored (PooledTobit only)
    p_uncensored = result.predict(pred_type="probability")
    ```

    The probability that the observation is uncensored. Only available for `PooledTobit`.

### Marginal Effects

In nonlinear models, coefficients $\beta$ do not directly represent marginal effects. PanelBox computes three types of marginal effects:

```python
# Average Marginal Effects (AME) on conditional mean
ame = result.marginal_effects(at="overall", which="conditional")

# Marginal Effects at Means (MEM) on probability
mem = result.marginal_effects(at="mean", which="probability")
```

See [Marginal Effects for Censored Models](marginal-effects.md) for details.

## Configuration Options

### PooledTobit Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | array-like | required | Dependent variable |
| `exog` | array-like | required | Regressors |
| `groups` | array-like | `None` | Entity IDs |
| `censoring_point` | float | `0.0` | Censoring threshold |
| `censoring_type` | str | `"left"` | `"left"`, `"right"`, or `"both"` |
| `lower_limit` | float | `None` | Lower bound (for `"both"`) |
| `upper_limit` | float | `None` | Upper bound (for `"both"`) |

### RandomEffectsTobit Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | array-like | required | Dependent variable |
| `exog` | array-like | required | Regressors |
| `groups` | array-like | required | Entity IDs |
| `time` | array-like | `None` | Time IDs |
| `censoring_point` | float | `0.0` | Censoring threshold |
| `censoring_type` | str | `"left"` | `"left"`, `"right"`, or `"both"` |
| `lower_limit` | float | `None` | Lower bound (for `"both"`) |
| `upper_limit` | float | `None` | Upper bound (for `"both"`) |
| `quadrature_points` | int | `12` | Gauss-Hermite quadrature points |

### fit() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_params` | array-like | `None` | Starting values (auto-computed from OLS if `None`) |
| `method` | str | `"BFGS"` | Optimization method |
| `maxiter` | int | `1000` | Maximum iterations |

## Diagnostics

### Percentage Censored

```python
n_censored = np.sum(np.abs(y - 0.0) < 1e-10)
pct_censored = n_censored / len(y) * 100
print(f"Censored observations: {n_censored} ({pct_censored:.1f}%)")
```

!!! note "Censoring rate"
    If the censoring rate is very high (>80%) or very low (<5%), the Tobit model may be poorly identified. Very high censoring means little information about the latent variable, and very low censoring means OLS may be adequate.

### Comparing Pooled vs. RE Tobit

```python
# Compare log-likelihoods
print(f"Pooled Tobit LL: {pooled_result.llf:.2f}")
print(f"RE Tobit LL:     {re_result.llf:.2f}")

# LR test for random effects
lr_stat = 2 * (re_result.llf - pooled_result.llf)
print(f"LR statistic: {lr_stat:.2f}")
# Compare with chi-squared(1) critical value
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Censored Models | Full walkthrough of Tobit estimation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/01_tobit_introduction.ipynb) |

## See Also

- [Honore Trimmed Estimator](honore.md) -- Fixed effects estimation for censored data
- [Panel Heckman](heckman.md) -- Sample selection models
- [Marginal Effects for Censored Models](marginal-effects.md) -- Interpreting nonlinear effects
- [Murphy-Topel Correction](murphy-topel.md) -- Correcting SEs in two-step estimators

## References

- Tobin, J. (1958). Estimation of relationships for limited dependent variables. *Econometrica*, 26(1), 24-36.
- Amemiya, T. (1984). Tobit models: A survey. *Journal of Econometrics*, 24(1-2), 3-61.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapter 17.
