---
title: "Ordered Choice Models"
description: "Ordered Logit and Probit models for ordinal panel data outcomes with random effects in PanelBox"
---

# Ordered Choice Models

!!! info "Quick Reference"
    **Classes:** `OrderedLogit`, `OrderedProbit`, `RandomEffectsOrderedLogit`
    **Import:** `from panelbox.models.discrete.ordered import OrderedLogit, OrderedProbit, RandomEffectsOrderedLogit`
    **Stata equivalent:** `ologit`, `oprobit`, `xtologit`
    **R equivalent:** `MASS::polr()`, `ordinal::clmm()`

## Overview

Ordered choice models are designed for ordinal dependent variables where $y_{it} \in \{0, 1, \ldots, J-1\}$ with a natural ordering but no meaningful numeric scale. Examples include survey responses (strongly disagree to strongly agree), credit ratings (AAA to D), health status (poor, fair, good, excellent), or education levels.

The model posits a latent continuous variable:

$$y^*_{it} = X_{it}'\beta + \varepsilon_{it}$$

The observed ordinal outcome is determined by cutpoints (thresholds) $\kappa_0 < \kappa_1 < \cdots < \kappa_{J-2}$:

$$y_{it} = j \quad \text{if} \quad \kappa_{j-1} < y^*_{it} \leq \kappa_j$$

with $\kappa_{-1} = -\infty$ and $\kappa_{J-1} = +\infty$. The probability of observing category $j$ is:

$$P(y_{it} = j \mid X_{it}) = F(\kappa_j - X_{it}'\beta) - F(\kappa_{j-1} - X_{it}'\beta)$$

where $F(\cdot)$ is the logistic CDF for Ordered Logit or the standard normal CDF for Ordered Probit.

## Quick Example

```python
import numpy as np
from panelbox.models.discrete.ordered import OrderedLogit

# Ordinal outcome: 0=low, 1=medium, 2=high
model = OrderedLogit(endog=y, exog=X, groups=entity, time=time)
model.fit(method="BFGS")

# Predicted probabilities for each category
probs = model.predict_proba()  # Shape: (N, J)

# Predicted most-likely category
categories = model.predict(type="category")

print(model.summary())
```

## When to Use

- **Ordinal dependent variable** with a natural ordering (survey scales, ratings, grades)
- **Proportional odds assumption** is plausible -- the effect of $X$ is the same across all cutpoints
- **OrderedLogit** -- Default choice; logistic errors yield proportional odds interpretation
- **OrderedProbit** -- When normality of errors is preferred; results are typically similar to logit
- **RandomEffectsOrderedLogit** -- When panel data has individual heterogeneity uncorrelated with regressors

!!! warning "Key Assumptions"
    - **Proportional odds (parallel regression)**: The slope coefficients $\beta$ are the same for all cutpoints. If this fails, consider generalized ordered logit models.
    - **Correct category ordering**: Categories must have a meaningful natural order.
    - **No constant term**: The cutpoints absorb the intercept; do not include a constant in $X$.

## Detailed Guide

### Data Preparation

The dependent variable should be integer-coded starting from 0. PanelBox automatically remaps categories to $\{0, 1, \ldots, J-1\}$ if they are not already in this format.

```python
import numpy as np
import pandas as pd

# Example: satisfaction ratings (1-5 scale)
n_entities = 200
n_periods = 4
N = n_entities * n_periods

entity = np.repeat(range(n_entities), n_periods)
time = np.tile(range(n_periods), n_entities)
x1 = np.random.normal(0, 1, N)
x2 = np.random.normal(0, 1, N)

# Exogenous variables (no constant -- cutpoints serve as intercepts)
X = np.column_stack([x1, x2])
```

### OrderedLogit

Uses the logistic CDF: $F(z) = \Lambda(z) = \frac{e^z}{1 + e^z}$

```python
from panelbox.models.discrete.ordered import OrderedLogit

model = OrderedLogit(endog=y, exog=X, groups=entity, time=time)
model.fit(method="BFGS", maxiter=1000)

# Estimated parameters
print("Coefficients:", model.beta)
print("Cutpoints:", model.cutpoints)

# Predicted category probabilities
probs = model.predict_proba()  # (N, J) array
print(f"Probability of category 0: {probs[:, 0].mean():.3f}")
print(f"Probability of category 1: {probs[:, 1].mean():.3f}")

# Most likely category
predicted = model.predict(type="category")
```

### OrderedProbit

Uses the standard normal CDF: $F(z) = \Phi(z)$

```python
from panelbox.models.discrete.ordered import OrderedProbit

model = OrderedProbit(endog=y, exog=X, groups=entity, time=time)
model.fit(method="BFGS")

print("Coefficients:", model.beta)
print("Cutpoints:", model.cutpoints)
print(model.summary())
```

### RandomEffectsOrderedLogit

Extends the ordered logit with individual random effects $\alpha_i \sim N(0, \sigma^2_\alpha)$:

$$y^*_{it} = X_{it}'\beta + \alpha_i + \varepsilon_{it}$$

The marginal likelihood integrates out $\alpha_i$ using Gauss-Hermite quadrature:

$$L_i = \int \prod_{t=1}^{T_i} P(y_{it} \mid X_{it}, \alpha_i) \, \phi(\alpha_i / \sigma_\alpha) \, d\alpha_i$$

```python
from panelbox.models.discrete.ordered import RandomEffectsOrderedLogit

model = RandomEffectsOrderedLogit(
    endog=y, exog=X, groups=entity, time=time,
    quadrature_points=12
)
model.fit(method="BFGS", maxiter=1000)

print("Coefficients:", model.beta)
print("Cutpoints:", model.cutpoints)
print(f"sigma_alpha: {model.sigma_alpha:.4f}")
print(model.summary())
```

### Interpreting Results

Coefficients in ordered choice models indicate the **direction** of the effect on the latent variable $y^*$, but not directly the magnitude of the effect on category probabilities:

- **Positive $\beta_k$**: increases $X_{it}'\beta$, shifting probability mass toward higher categories
- **Negative $\beta_k$**: shifts probability mass toward lower categories
- **Cutpoints** define the boundaries between categories on the latent scale

!!! note "Marginal Effects Are Essential"
    A positive coefficient shifts mass to higher categories but can **decrease** the probability of intermediate categories. Always compute marginal effects for proper interpretation. See [Marginal Effects](marginal-effects.md) for details.

### Cutpoint Parameterization

PanelBox uses an exponential parameterization to enforce $\kappa_0 < \kappa_1 < \cdots < \kappa_{J-2}$:

$$\kappa_0 = \gamma_0, \quad \kappa_j = \kappa_{j-1} + \exp(\gamma_j) \quad \text{for } j > 0$$

This ensures strictly ordered cutpoints without constrained optimization. The parameters $\gamma_j$ are unconstrained and estimated via MLE.

## Configuration Options

### OrderedLogit / OrderedProbit

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `ndarray` | required | Ordinal dependent variable |
| `exog` | `ndarray` | required | Exogenous variables (no constant) |
| `groups` | `ndarray` | required | Entity identifiers |
| `time` | `ndarray` | `None` | Time period identifiers |
| `n_categories` | `int` | `None` | Number of categories (inferred if `None`) |

**fit() parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_params` | `ndarray` | `None` | Starting values (auto-computed if `None`) |
| `method` | `str` | `"BFGS"` | Optimization method |
| `maxiter` | `int` | `1000` | Maximum iterations |

### RandomEffectsOrderedLogit

All parameters from `OrderedLogit` plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quadrature_points` | `int` | `12` | Gauss-Hermite quadrature nodes |

## Result Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `ndarray` | Full parameter vector $[\beta; \gamma]$ |
| `beta` | `ndarray` | Slope coefficients |
| `cutpoints` | `ndarray` | Ordered threshold values $\kappa_0 < \kappa_1 < \cdots$ |
| `llf` | `float` | Log-likelihood at maximum |
| `converged` | `bool` | Convergence flag |
| `n_iter` | `int` | Number of iterations |
| `bse` | `ndarray` | Standard errors |
| `cov_params` | `ndarray` | Variance-covariance matrix |

Additional for **RandomEffectsOrderedLogit**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `sigma_alpha` | `float` | Random effects standard deviation |

## Diagnostics

### Goodness of Fit

```python
# Log-likelihood comparison
print(f"Log-likelihood: {model.llf:.3f}")

# Predicted vs actual categories
predicted = model.predict(type="category")
accuracy = np.mean(predicted == y)
print(f"Classification accuracy: {accuracy:.3f}")
```

### Comparing Logit and Probit

Results from ordered logit and ordered probit are typically similar after rescaling. The logistic distribution has variance $\pi^2/3 \approx 3.29$, while the standard normal has variance 1. Therefore, probit coefficients should be approximately $\beta_{logit} / 1.81$ compared to logit coefficients.

```python
from panelbox.models.discrete.ordered import OrderedLogit, OrderedProbit

ologit = OrderedLogit(endog=y, exog=X, groups=entity, time=time)
ologit.fit()

oprobit = OrderedProbit(endog=y, exog=X, groups=entity, time=time)
oprobit.fit()

# Approximate rescaling
print("Logit coefficients:", ologit.beta)
print("Probit coefficients:", oprobit.beta)
print("Logit / 1.81:", ologit.beta / 1.81)  # Should be close to probit
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Discrete Choice Models | Full guide including ordered models | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/07_ordered_models.ipynb) |

## See Also

- [Binary Choice Models](binary.md) -- Logit and Probit for binary outcomes
- [Multinomial and Conditional Logit](multinomial.md) -- Unordered multi-category outcomes
- [Dynamic Binary Panel](dynamic.md) -- State dependence models
- [Marginal Effects](marginal-effects.md) -- Essential for interpreting ordered model coefficients

## References

- McKelvey, R. D. and Zavoina, W. (1975). "A Statistical Model for the Analysis of Ordinal Level Dependent Variables." *Journal of Mathematical Sociology*, 4(1), 103-120.
- Greene, W. H. and Hensher, D. A. (2010). *Modeling Ordered Choices: A Primer*. Cambridge University Press.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press. Chapter 15.
- Brant, R. (1990). "Assessing Proportionality in the Proportional Odds Model for Ordinal Logistic Regression." *Biometrics*, 46(4), 1171-1178.
