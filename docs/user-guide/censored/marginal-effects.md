---
title: "Marginal Effects for Censored Models"
description: "Computing and interpreting marginal effects in Tobit and Heckman selection models with PanelBox."
---

# Marginal Effects for Censored Models

!!! info "Quick Reference"
    **Method:** `model.marginal_effects(at="overall", which="conditional")`
    **Applies to:** `PooledTobit`, `RandomEffectsTobit`, `PanelHeckman`
    **Stata equivalent:** `margins, predict(ystar(0,.))` or `margins, dydx(*)`
    **R equivalent:** `margins::margins()`

## Overview

In nonlinear models like Tobit and Heckman, the estimated coefficients $\hat{\beta}$ do **not** directly represent marginal effects. Unlike linear regression where $\partial E[y|X] / \partial x_j = \beta_j$, censored and selection models have non-constant marginal effects that depend on where in the distribution you evaluate them.

For the Tobit model, there are **three distinct types** of marginal effects, each answering a different economic question. Understanding which one to report is critical for correct interpretation of results.

## Three Types of Marginal Effects in Tobit

### 1. Conditional Marginal Effect

$$\frac{\partial E[y \mid y > c, X]}{\partial x_j}$$

**Question**: Among observations that are **uncensored** (above the boundary), how does a change in $x_j$ affect the expected outcome?

**Formula** (left censoring at $c$):

$$\text{ME}^{cond}_j = \beta_j \left[ 1 - \lambda(z) \cdot (z + \lambda(z)) \right]$$

where $z = (X'\beta - c)/\sigma$ and $\lambda(z) = \phi(z)/\Phi(z)$ is the Inverse Mills Ratio.

**Use when**: You care about the intensive margin -- how the outcome changes for those already above the boundary.

### 2. Unconditional Marginal Effect

$$\frac{\partial E[y \mid X]}{\partial x_j}$$

**Question**: How does a change in $x_j$ affect the **overall** expected outcome, including the probability of being at the boundary?

**Formula** (left censoring at $c$):

$$\text{ME}^{uncond}_j = \beta_j \cdot \Phi(z)$$

where $\Phi(z)$ is the probability of being uncensored. This is the most commonly reported marginal effect.

**Use when**: You want the total effect combining both the probability of being uncensored and the conditional effect.

### 3. Probability Marginal Effect

$$\frac{\partial P(y > c \mid X)}{\partial x_j}$$

**Question**: How does a change in $x_j$ affect the **probability** of being uncensored?

**Formula** (left censoring at $c$):

$$\text{ME}^{prob}_j = \frac{\beta_j}{\sigma} \cdot \phi(z)$$

**Use when**: You care about the extensive margin -- how the probability of crossing the boundary changes.

### McDonald & Moffitt (1980) Decomposition

The total (unconditional) marginal effect decomposes into two components:

$$\frac{\partial E[y|X]}{\partial x_j} = \underbrace{P(y > c) \cdot \frac{\partial E[y|y>c, X]}{\partial x_j}}_{\text{intensive margin}} + \underbrace{E[y|y>c, X] \cdot \frac{\partial P(y>c)}{\partial x_j}}_{\text{extensive margin}}$$

This decomposition separates the effect into:

1. **Intensive margin**: among the uncensored, how does the level change?
2. **Extensive margin**: how does the probability of being uncensored change, weighted by the expected level?

## Computing Marginal Effects in PanelBox

### Tobit Marginal Effects

Both `PooledTobit` and `RandomEffectsTobit` support the `marginal_effects()` method:

```python
from panelbox.models.censored import PooledTobit

# Fit model
model = PooledTobit(endog=y, exog=X, groups=entity, censoring_point=0.0)
result = model.fit()

# Average Marginal Effects (AME) — conditional
ame_cond = result.marginal_effects(at="overall", which="conditional")

# Average Marginal Effects — unconditional
ame_uncond = result.marginal_effects(at="overall", which="unconditional")

# Average Marginal Effects — probability
ame_prob = result.marginal_effects(at="overall", which="probability")

# Marginal Effects at Means (MEM)
mem = result.marginal_effects(at="mean", which="conditional")
```

### Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `at` | `"overall"` | **AME**: Average over all observations |
| | `"mean"` | **MEM**: Evaluate at sample means |
| `which` | `"conditional"` | $\partial E[y \mid y > c, X] / \partial x$ |
| | `"unconditional"` | $\partial E[y \mid X] / \partial x$ |
| | `"probability"` | $\partial P(y > c \mid X) / \partial x$ |
| `varlist` | list of str | Compute for specific variables only |

### AME vs. MEM

| Method | Computation | When to use |
|--------|-------------|-------------|
| **AME** (`at="overall"`) | Compute ME for each obs, then average | Preferred for policy analysis; robust to nonlinearity |
| **MEM** (`at="mean"`) | Evaluate ME at $\bar{X}$ | Simpler to compute; may not represent any actual observation |

AME is generally preferred because it averages over the actual distribution of covariates rather than evaluating at a potentially unrealistic "average" observation.

## Marginal Effects in the Heckman Model

For the Panel Heckman model, marginal effects must account for the selection correction. The prediction types provide the building blocks:

```python
from panelbox.models.selection import PanelHeckman

model = PanelHeckman(
    endog=y, exog=X, selection=d, exog_selection=Z,
    method="two_step",
)
results = model.fit()

# Unconditional prediction: E[y*] = X'beta
y_uncond = results.predict(type="unconditional")

# Conditional prediction: E[y|selected] = X'beta + rho*sigma*lambda
y_cond = results.predict(type="conditional")
```

### Unconditional Marginal Effect (Heckman)

$$\frac{\partial E[y^*]}{\partial x_j} = \beta_j$$

The unconditional marginal effect equals the coefficient -- the same as in a linear model -- because the latent outcome is linear in $X$.

### Conditional Marginal Effect (Heckman)

$$\frac{\partial E[y \mid d=1, X, W]}{\partial x_j} = \beta_j + \rho \sigma_\varepsilon \frac{\partial \lambda}{\partial x_j}$$

If $x_j$ appears only in the outcome equation (not in the selection equation), the conditional effect simplifies to $\beta_j$. If $x_j$ also appears in the selection equation, there is an additional indirect effect through the IMR.

## Interpreting Results

### Coefficients vs. Marginal Effects

A common mistake is to interpret Tobit coefficients as marginal effects:

| | Linear (OLS) | Tobit |
|--|-------------|-------|
| $\beta_j$ represents | $\partial E[y]/\partial x_j$ directly | Effect on **latent** $y^*$, not observed $y$ |
| Marginal effect | $= \beta_j$ (constant) | $\neq \beta_j$ (varies with $X$) |
| Scale | One unit of $y$ per unit of $x$ | Dampened by $\Phi(z)$ |

### The Dampening Factor

For the unconditional ME, the coefficient is dampened by $\Phi(z)$, the probability of being uncensored:

$$\text{ME}^{uncond}_j = \beta_j \cdot \Phi\!\left(\frac{X'\beta - c}{\sigma}\right)$$

Since $\Phi(z) \in (0, 1)$, the marginal effect is always **smaller in magnitude** than the coefficient $\beta_j$. The closer to the censoring point, the stronger the dampening.

### Practical Comparison

```python
# Compare coefficients vs marginal effects
print("Variable | Coefficient | AME (uncond) | Ratio")
print("-" * 55)

ame = result.marginal_effects(at="overall", which="unconditional")
for j, (coef, me) in enumerate(zip(result.beta, ame.effects)):
    ratio = me / coef if abs(coef) > 1e-10 else float('nan')
    print(f"x_{j:<6d} | {coef:>11.4f} | {me:>12.4f} | {ratio:.3f}")
```

The ratio $\text{ME}/\beta$ tells you how much the censoring dampens the raw coefficient. A ratio near 1 means little censoring; a ratio near 0 means heavy censoring.

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Censored Models | Marginal effects computation and interpretation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/07_marginal_effects.ipynb) |

## See Also

- [Tobit Models](tobit.md) -- Pooled and RE Tobit estimation
- [Panel Heckman](heckman.md) -- Sample selection models
- [Murphy-Topel Correction](murphy-topel.md) -- Correct SEs for marginal effect inference

## References

- McDonald, J. F., & Moffitt, R. A. (1980). The uses of Tobit analysis. *Review of Economics and Statistics*, 62(2), 318-321.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapter 17.
- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press.
- Greene, W. H. (2012). *Econometric Analysis* (7th ed.). Prentice Hall. Chapter 19.
