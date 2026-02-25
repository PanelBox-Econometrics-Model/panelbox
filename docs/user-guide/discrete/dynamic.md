---
title: "Dynamic Binary Panel"
description: "Dynamic discrete choice models with state dependence and initial conditions for panel data in PanelBox"
---

# Dynamic Binary Panel

!!! info "Quick Reference"
    **Class:** `DynamicBinaryPanel`
    **Import:** `from panelbox.models.discrete.dynamic import DynamicBinaryPanel`
    **Stata equivalent:** `xtprobit y L.y x1, re` with Wooldridge initial conditions
    **R equivalent:** Custom implementation (no standard package)

## Overview

Dynamic binary panel models address a fundamental question in applied economics: **does past behavior causally affect current behavior?** When we observe persistence in binary outcomes -- individuals who were employed last period tend to be employed this period, firms that exported last year tend to export this year -- this persistence may arise from two distinct sources:

1. **True state dependence**: past outcomes $y_{i,t-1}$ have a genuine causal effect on current outcomes $y_{it}$ (e.g., work experience builds human capital, making future employment more likely)
2. **Spurious state dependence**: unobserved individual heterogeneity $\alpha_i$ creates persistence without any causal effect of past behavior (e.g., inherently motivated individuals are always more likely to be employed)

The dynamic binary panel model disentangles these two channels:

$$P(y_{it} = 1 \mid y_{i,t-1}, X_{it}, \alpha_i) = F(\gamma \, y_{i,t-1} + X_{it}'\beta + \alpha_i)$$

where $\gamma$ captures true state dependence and $\alpha_i$ captures unobserved heterogeneity. A significant $\gamma > 0$ provides evidence of genuine state dependence beyond what heterogeneity alone can explain.

## Quick Example

```python
import numpy as np
from panelbox.models.discrete.dynamic import DynamicBinaryPanel

model = DynamicBinaryPanel(
    endog=y, exog=X,
    entity=entity, time=time,
    initial_conditions="wooldridge",
    effects="random"
)
result = model.fit()

print(f"State dependence (gamma): {result.gamma:.4f}")
print(f"RE std deviation:         {result.sigma_u:.4f}")
print(result.summary())
```

## When to Use

- **Persistence in binary outcomes**: employment status, export participation, technology adoption, brand loyalty, poverty traps
- **Disentangling causes of persistence**: is past behavior truly causal, or is it driven by permanent traits?
- **Policy evaluation**: if $\gamma \approx 0$, a temporary intervention has no lasting effect; if $\gamma > 0$, temporary subsidies can shift long-run behavior
- **Dynamic structural models**: when the lagged dependent variable is part of the economic model

!!! warning "Key Assumptions"
    - **Correct initial conditions specification**: The initial observation $y_{i0}$ is endogenous to $\alpha_i$. Ignoring this leads to upward-biased estimates of $\gamma$.
    - **Random effects**: $\alpha_i$ is modeled parametrically (normal distribution). If $\alpha_i$ is correlated with $X_{it}$, estimates are biased.
    - **Strict exogeneity of $X$**: Regressors must not be affected by past values of $y$.
    - **Binary outcomes only**: This model is designed for $y \in \{0, 1\}$.

## The Initial Conditions Problem

The key econometric challenge in dynamic binary panels is the **initial conditions problem**. Since the model includes $\alpha_i$ as an unobserved effect and $y_{i,t-1}$ as a regressor, the initial observation $y_{i0}$ is correlated with $\alpha_i$. Simply conditioning on $y_{i0}$ and treating it as exogenous leads to biased estimates.

PanelBox offers three approaches:

### Wooldridge (2005) -- Recommended

Models the distribution of $\alpha_i$ conditional on the initial observation and time-averages of covariates:

$$\alpha_i = \delta_0 + \delta_1 \, y_{i0} + \delta_2' \bar{X}_i + a_i, \quad a_i \sim N(0, \sigma^2_a)$$

This is the most practical approach and is widely used in applied work.

```python
model = DynamicBinaryPanel(
    endog=y, exog=X, entity=entity, time=time,
    initial_conditions="wooldridge",
    effects="random"
)
result = model.fit()

# Structural parameters
print(f"gamma (lag effect):    {result.gamma:.4f}")
print(f"delta_y0 (initial):    {result.delta_y0:.4f}")
print(f"delta_xbar (X means):  {result.delta_xbar}")
```

### Heckman (1981)

Models the joint distribution of $(y_{i0}, \alpha_i)$ by specifying a separate reduced-form equation for the initial period:

$$P(y_{i0} = 1 \mid Z_i, \alpha_i) = F(Z_i'\pi + \alpha_i)$$

```python
model = DynamicBinaryPanel(
    endog=y, exog=X, entity=entity, time=time,
    initial_conditions="heckman",
    effects="random"
)
result = model.fit()
```

### Simple (Exogenous)

Treats $y_{i0}$ as exogenous -- drops the first period for each entity and uses $y_{i1}$ as the initial condition. This is biased when $\alpha_i$ matters but can serve as a quick baseline.

```python
model = DynamicBinaryPanel(
    endog=y, exog=X, entity=entity, time=time,
    initial_conditions="simple",
    effects="random"
)
result = model.fit()
```

!!! tip "Which Approach?"
    The **Wooldridge** approach is recommended for most applications. It is computationally simpler than Heckman and produces similar estimates in most cases. Use `"simple"` only as a diagnostic baseline -- if $\gamma$ changes substantially between `"simple"` and `"wooldridge"`, the initial conditions matter.

## Detailed Guide

### Data Preparation

The model requires balanced or unbalanced panel data. Entity and time identifiers are used to construct the lagged dependent variable internally.

```python
import numpy as np
import pandas as pd

# Simulated panel data
n_entities = 500
n_periods = 10
N = n_entities * n_periods

entity = np.repeat(range(n_entities), n_periods)
time = np.tile(range(n_periods), n_entities)
x1 = np.random.normal(0, 1, N)
x2 = np.random.normal(0, 1, N)
X = np.column_stack([x1, x2])

# Binary outcome with state dependence
# (in practice, y comes from your data)
y = np.random.binomial(1, 0.5, N)
```

### Estimation

```python
from panelbox.models.discrete.dynamic import DynamicBinaryPanel

model = DynamicBinaryPanel(
    endog=y,
    exog=X,
    entity=entity,
    time=time,
    initial_conditions="wooldridge",
    effects="random"
)
result = model.fit()
print(result.summary())
```

### Interpreting Results

The key parameters are:

| Parameter | Attribute | Interpretation |
|-----------|-----------|---------------|
| $\gamma$ | `result.gamma` | State dependence: effect of $y_{i,t-1}$ on $P(y_{it}=1)$ |
| $\beta$ | `result.beta` | Covariate effects |
| $\sigma_u$ | `result.sigma_u` | Heterogeneity: standard deviation of $\alpha_i$ |
| $\delta_{y0}$ | `result.delta_y0` | Initial value coefficient (Wooldridge) |
| $\delta_{\bar{x}}$ | `result.delta_xbar` | Time-average coefficients (Wooldridge) |

**Interpreting $\gamma$**:

- $\gamma > 0$ and significant: **true state dependence** -- past behavior causally affects current behavior
- $\gamma \approx 0$: persistence is entirely due to unobserved heterogeneity
- Large $\sigma_u$ with small $\gamma$: most persistence comes from permanent individual traits

### Predictions

```python
# Predicted probabilities
probs = result.predict()

# Marginal effects at the mean
me = result.marginal_effects()
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `ndarray` | required | Binary dependent variable (0/1) |
| `exog` | `ndarray` | required | Exogenous covariates |
| `entity` | `ndarray` | required | Entity identifiers |
| `time` | `ndarray` | required | Time identifiers |
| `initial_conditions` | `str` | `"wooldridge"` | `"wooldridge"`, `"heckman"`, or `"simple"` |
| `effects` | `str` | `"random"` | `"random"` or `"pooled"` |

**fit() parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_params` | `ndarray` | `None` | Starting values (auto-computed if `None`) |

## Result Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `ndarray` | Full parameter vector |
| `beta` | `ndarray` | Covariate coefficients |
| `gamma` | `float` | Lag coefficient (state dependence) |
| `sigma_u` | `float` | Random effects standard deviation |
| `delta_y0` | `float` | Initial value coefficient (Wooldridge only) |
| `delta_xbar` | `ndarray` | Time-average coefficients (Wooldridge only) |
| `llf` | `float` | Log-likelihood |
| `converged` | `bool` | Convergence flag |
| `n_iter` | `int` | Number of iterations |

## Diagnostics

### Testing for State Dependence

The primary diagnostic is whether $\gamma$ is significantly different from zero:

```python
result = model.fit()

# Check state dependence
print(f"gamma = {result.gamma:.4f}")
print(result.summary())  # Includes z-statistics and p-values
```

### Sensitivity to Initial Conditions

Compare estimates across different initial conditions specifications:

```python
from panelbox.models.discrete.dynamic import DynamicBinaryPanel

# Wooldridge (preferred)
m1 = DynamicBinaryPanel(endog=y, exog=X, entity=entity, time=time,
                          initial_conditions="wooldridge", effects="random")
r1 = m1.fit()

# Heckman
m2 = DynamicBinaryPanel(endog=y, exog=X, entity=entity, time=time,
                          initial_conditions="heckman", effects="random")
r2 = m2.fit()

# Simple (biased baseline)
m3 = DynamicBinaryPanel(endog=y, exog=X, entity=entity, time=time,
                          initial_conditions="simple", effects="random")
r3 = m3.fit()

print(f"Wooldridge gamma: {r1.gamma:.4f}")
print(f"Heckman gamma:    {r2.gamma:.4f}")
print(f"Simple gamma:     {r3.gamma:.4f} (upward biased)")
```

!!! note "Expected Bias Pattern"
    The `"simple"` approach typically overestimates $\gamma$ because it attributes some of the heterogeneity effect to state dependence. If `"simple"` and `"wooldridge"` produce similar $\gamma$, the initial conditions problem is not severe in your data.

### Comparing with Static Models

```python
from panelbox.models.discrete.binary import RandomEffectsProbit

# Static RE Probit (no lag)
static = RandomEffectsProbit("y ~ x1 + x2", data, "id", "year")
static_res = static.fit()

# Dynamic (with lag)
dynamic = DynamicBinaryPanel(endog=y, exog=X, entity=entity, time=time,
                              initial_conditions="wooldridge", effects="random")
dynamic_res = dynamic.fit()

print(f"Static sigma_alpha:  {static.sigma_alpha:.4f}")
print(f"Dynamic sigma_u:     {dynamic_res.sigma_u:.4f}")
print(f"Dynamic gamma:       {dynamic_res.gamma:.4f}")
# sigma_u should be smaller than sigma_alpha if state dependence is real
```

## Common Applications

| Application | Outcome | State Dependence Interpretation |
|------------|---------|-------------------------------|
| Labor economics | Employment status | Job experience makes future employment more likely |
| International trade | Export participation | Sunk costs of entering export markets |
| Technology adoption | Use of technology | Learning-by-doing, switching costs |
| Marketing | Brand loyalty | Habit formation, satisfaction feedback |
| Poverty | Poverty status | Poverty traps, asset depletion |

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Dynamic Discrete Choice | State dependence analysis with initial conditions | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/08_dynamic_discrete.ipynb) |

## See Also

- [Binary Choice Models](binary.md) -- Static Logit and Probit models
- [Ordered Choice Models](ordered.md) -- Extension to ordinal outcomes
- [Marginal Effects](marginal-effects.md) -- Computing and interpreting effects
- [Dynamic GMM](../gmm/difference-gmm.md) -- Arellano-Bond for continuous dynamic panels

## References

- Wooldridge, J. M. (2005). "Simple Solutions to the Initial Conditions Problem in Dynamic, Nonlinear Panel Data Models with Unobserved Heterogeneity." *Journal of Applied Econometrics*, 20(1), 39-54.
- Heckman, J. J. (1981). "The Incidental Parameters Problem and the Problem of Initial Conditions in Estimating a Discrete Time-Discrete Data Stochastic Process." In *Structural Analysis of Discrete Data*, ed. C. Manski and D. McFadden. MIT Press.
- Arulampalam, W. and Stewart, M. B. (2009). "Simplified Implementation of the Heckman Estimator of the Dynamic Probit Model and a Comparison with Alternative Estimators." *Oxford Bulletin of Economics and Statistics*, 71(5), 659-681.
- Stewart, M. B. (2007). "The Interrelated Dynamics of Unemployment and Low-Wage Employment." *Journal of Applied Econometrics*, 22(3), 511-531.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press. Chapter 15.
