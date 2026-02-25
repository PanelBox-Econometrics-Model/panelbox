---
title: "Marginal Effects for Discrete Choice"
description: "Computing and interpreting marginal effects for binary, ordered, and multinomial discrete choice models in PanelBox"
---

# Marginal Effects for Discrete Choice Models

!!! info "Quick Reference"
    **Method:** `results.marginal_effects(at="overall", method="dydx")`
    **Stata equivalent:** `margins, dydx(*)`
    **R equivalent:** `margins::margins()`

## Overview

In nonlinear models such as logit, probit, and multinomial logit, the estimated coefficients $\beta$ do **not** directly represent the effect of a one-unit change in $x_k$ on the probability of the outcome. This is because the probability function is nonlinear:

$$P(y = 1 \mid X) = F(X'\beta)$$

The marginal effect of variable $x_k$ on the probability is:

$$\frac{\partial P(y = 1)}{\partial x_k} = f(X'\beta) \cdot \beta_k$$

where $f(\cdot) = F'(\cdot)$ is the density function. This depends on the values of **all** covariates through $f(X'\beta)$, so the marginal effect varies across observations.

PanelBox provides three approaches to summarize marginal effects, along with standard errors computed via the delta method.

## Types of Marginal Effects

### Average Marginal Effect (AME)

Compute the marginal effect for each observation, then average:

$$\text{AME}_k = \frac{1}{N} \sum_{i=1}^{N} f(X_i'\hat{\beta}) \cdot \hat{\beta}_k$$

This is the most commonly reported measure and reflects the average effect across the actual sample distribution.

```python
me = model.marginal_effects(at="overall", method="dydx")
```

### Marginal Effect at the Mean (MEM)

Evaluate the marginal effect at the sample mean of all covariates:

$$\text{MEM}_k = f(\bar{X}'\hat{\beta}) \cdot \hat{\beta}_k$$

This gives the effect for a "representative" individual at the mean of all variables.

```python
me = model.marginal_effects(at="mean", method="dydx")
```

### Marginal Effect at Representative Values (MER)

Evaluate at user-specified covariate values:

```python
# For PooledLogit / PooledProbit
me = model.marginal_effects(
    at="overall",
    method="dydx",
    representative={"education": 16, "experience": 10}
)
```

### AME vs MEM: Which to Use?

| Feature | AME (`at="overall"`) | MEM (`at="mean"`) |
|---------|---------------------|-------------------|
| Computation | Average over all observations | Evaluate at $\bar{X}$ |
| Interpretation | Population-level average effect | Effect for "average" person |
| Robustness | More robust to distribution shape | Sensitive to where $\bar{X}$ falls |
| Recommended | Preferred in most applications | Quick summary measure |

!!! tip "Prefer AME"
    The AME is generally preferred because (1) the "average person" (at $\bar{X}$) may not actually exist in the data, (2) AME accounts for the full distribution of covariates, and (3) AME is more robust when the link function is highly nonlinear.

## Marginal Effects by Model Type

### Binary Models (Logit / Probit)

For **PooledLogit**, the marginal effect of $x_k$ is:

$$\frac{\partial P}{\partial x_k} = \Lambda(X'\beta) \cdot [1 - \Lambda(X'\beta)] \cdot \beta_k$$

For **PooledProbit**:

$$\frac{\partial P}{\partial x_k} = \phi(X'\beta) \cdot \beta_k$$

```python
from panelbox.models.discrete.binary import PooledLogit, PooledProbit

# Logit
logit = PooledLogit("employed ~ education + experience + age", data, "id", "year")
logit_res = logit.fit(cov_type="cluster")
me_logit = logit.marginal_effects(at="overall", method="dydx")
print(me_logit)

# Probit
probit = PooledProbit("employed ~ education + experience + age", data, "id", "year")
probit_res = probit.fit(cov_type="cluster")
me_probit = probit.marginal_effects(at="overall", method="dydx")
print(me_probit)
```

!!! note "Logit vs Probit Marginal Effects"
    Even though logit and probit coefficients differ in scale, AME values are typically very similar between the two models. This makes AME a useful basis for comparing model specifications.

### Fixed Effects Logit

The FE logit estimates **conditional** log-odds ratios. Because entity fixed effects $\alpha_i$ are not estimated, standard marginal effects on $P(y = 1)$ are not computable. However, the coefficients have a conditional odds ratio interpretation:

$$\exp(\beta_k) = \text{odds ratio for a one-unit change in } x_k$$

```python
from panelbox.models.discrete.binary import FixedEffectsLogit

model = FixedEffectsLogit("employed ~ education + experience", data, "id", "year")
results = model.fit()

# Odds ratios (exponentiated coefficients)
import numpy as np
odds_ratios = np.exp(results.params)
print("Odds ratios:", odds_ratios)
```

!!! warning "No Marginal Effects for FE Logit"
    Fixed Effects Logit does not support `marginal_effects()` because individual effects are not estimated. Use odds ratios or consider Random Effects Probit if marginal effects are essential.

### Random Effects Probit

The RE probit integrates over the distribution of $\alpha_i$:

$$P(y_{it} = 1 \mid X_{it}) = \int \Phi(X_{it}'\beta + \alpha_i) \, \phi(\alpha_i / \sigma_\alpha) \, d\alpha_i$$

Marginal effects can be computed as population-averaged effects (integrating out $\alpha_i$):

```python
from panelbox.models.discrete.binary import RandomEffectsProbit

model = RandomEffectsProbit("employed ~ education + experience + female",
                             data, "id", "year", quadrature_points=12)
results = model.fit()

# Marginal effects (integrating over RE distribution)
me = model.marginal_effects(at="mean", method="dydx")
print(me)
```

### Ordered Models

For ordered logit/probit, marginal effects are computed **per category**. Increasing $x_k$ can:

- Increase the probability of some categories
- Decrease the probability of others
- The effects across all categories sum to zero

For an Ordered Logit with $J$ categories:

$$\frac{\partial P(y = j)}{\partial x_k} = [f(\kappa_{j-1} - X'\beta) - f(\kappa_j - X'\beta)] \cdot \beta_k$$

where $f(\cdot)$ is the logistic PDF.

```python
from panelbox.models.discrete.ordered import OrderedLogit
import numpy as np

model = OrderedLogit(endog=y, exog=X, groups=entity, time=time)
model.fit()

# Get probabilities at the mean
probs = model.predict_proba()

# Manual marginal effects for ordered models:
# The sign of beta determines the direction of shift
# Positive beta: increases probability of higher categories,
#                decreases probability of lower categories
print("Coefficients:", model.beta)
print("Average predicted probabilities:", probs.mean(axis=0))
```

!!! note "Intermediate Categories"
    For intermediate categories, the sign of the marginal effect can be **ambiguous** -- a positive coefficient does not always increase the probability of a middle category. Always compute the full set of category-specific marginal effects.

### Multinomial Logit

In multinomial logit, marginal effects form a $(J, K)$ matrix. The effect of $x_k$ on $P(y = j)$ is:

$$\frac{\partial P(y = j)}{\partial x_k} = P(y = j) \left[ \beta_{jk} - \sum_{m=0}^{J-1} P(y = m) \, \beta_{mk} \right]$$

where $\beta_{0k} = 0$ for the base alternative. Marginal effects sum to zero across alternatives for each variable.

```python
from panelbox.models.discrete.multinomial import MultinomialLogit

model = MultinomialLogit(endog=y, exog=X, n_alternatives=3, base_alternative=0)
result = model.fit()

# Marginal effects: (J, K) matrix
me = result.marginal_effects(at="mean")  # At sample means
print("Marginal effects at mean:")
print(me)

# Average marginal effects
me_overall = result.marginal_effects(at="overall")

# For a specific variable
me_var = result.marginal_effects(at="mean", variable=0)

# Standard errors for marginal effects
me_se = result.marginal_effects_se(at="mean")
```

!!! tip "Visualization"
    Use the built-in plotting for multinomial marginal effects:
    ```python
    result.plot_marginal_effects(variable=0, at="mean")
    ```

### Dynamic Binary Panel

Marginal effects in dynamic models have two components:

1. **Short-run effect**: the immediate effect of $x_k$ on $P(y_{it} = 1)$
2. **Long-run effect**: the accumulated effect through the feedback loop $y_{it} \to y_{i,t+1} \to \cdots$

The short-run marginal effect is $f(\gamma y_{i,t-1} + X'\beta + \alpha_i) \cdot \beta_k$.

```python
from panelbox.models.discrete.dynamic import DynamicBinaryPanel

model = DynamicBinaryPanel(endog=y, exog=X, entity=entity, time=time,
                            initial_conditions="wooldridge", effects="random")
result = model.fit()

# Marginal effects at the mean
me = result.marginal_effects()
print(me)
```

## Discrete vs Continuous Variables

For **continuous** variables, the marginal effect is the partial derivative $\partial P / \partial x_k$.

For **discrete** (binary/categorical) variables, the marginal effect is the **discrete change** in probability:

$$\Delta P = P(y = 1 \mid x_k = 1) - P(y = 1 \mid x_k = 0)$$

PanelBox computes this automatically when using `method="dydx"` based on variable type detection:

```python
# For continuous variables: partial derivative
# For discrete/binary variables: discrete change
me = model.marginal_effects(at="overall", method="dydx")
```

## Interpretation Guide

### Comparing Marginal Effects Across Models

| Model | ME Computation | Notes |
|-------|---------------|-------|
| **PooledLogit** | $\Lambda' \cdot \beta_k$ | Standard AME or MEM |
| **PooledProbit** | $\phi \cdot \beta_k$ | AME similar to logit AME |
| **FE Logit** | Not available | Use odds ratios: $\exp(\beta_k)$ |
| **RE Probit** | Integrated over $\alpha_i$ | Population-averaged effects |
| **Ordered** | Per-category, sums to zero | $J$ effects per variable |
| **Multinomial** | $(J, K)$ matrix, sums to zero | Per-alternative effects |
| **Dynamic** | Short-run effect | Long-run requires simulation |

### Common Pitfalls

!!! warning "Interpretation Mistakes to Avoid"

    1. **Comparing raw coefficients across models**: Logit and probit coefficients have different scales. Use marginal effects for comparison.
    2. **Ignoring the evaluation point**: MEM depends on $\bar{X}$, which may not be representative. AME is safer.
    3. **Sign of intermediate categories**: In ordered models, the marginal effect on middle categories can have the opposite sign of $\beta$.
    4. **Base alternative confusion**: In multinomial logit, coefficients are relative to the base. Marginal effects are absolute (summing to zero).
    5. **FE Logit marginal effects**: These do not exist in the standard sense because $\alpha_i$ is not estimated.

## Summary Table

| Model | `at="overall"` | `at="mean"` | SE Method | Command |
|-------|:---:|:---:|-----------|---------|
| PooledLogit | AME | MEM | Delta method | `model.marginal_effects(at=..., method="dydx")` |
| PooledProbit | AME | MEM | Delta method | `model.marginal_effects(at=..., method="dydx")` |
| FE Logit | N/A | N/A | N/A | Use `np.exp(results.params)` for odds ratios |
| RE Probit | Integrated | Integrated | Delta method | `model.marginal_effects(at=..., method="dydx")` |
| Ordered Logit/Probit | Per-category | Per-category | Numerical | Via `predict_proba()` |
| Multinomial Logit | $(J,K)$ matrix | $(J,K)$ matrix | Numerical Hessian | `result.marginal_effects(at=...)` |
| Dynamic Binary | Short-run | Short-run | Numerical | `result.marginal_effects()` |

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Marginal Effects Guide | Computing and comparing MEs across models | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/04_marginal_effects.ipynb) |

## See Also

- [Binary Choice Models](binary.md) -- Logit and Probit models
- [Ordered Choice Models](ordered.md) -- Marginal effects for ordinal outcomes
- [Multinomial and Conditional Logit](multinomial.md) -- Marginal effects in $(J, K)$ form
- [Dynamic Binary Panel](dynamic.md) -- Short-run vs long-run effects

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press.
- Cameron, A. C. and Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press. Section 14.3.
- Greene, W. H. (2018). *Econometric Analysis*. 8th ed. Pearson. Chapter 17.
- Bartus, T. (2005). "Estimation of Marginal Effects Using margeff." *The Stata Journal*, 5(3), 309-329.
- Long, J. S. and Freese, J. (2014). *Regression Models for Categorical Dependent Variables Using Stata*. 3rd ed. Stata Press.
