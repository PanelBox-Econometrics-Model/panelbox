---
title: Discrete Choice Models
description: Guide to panel discrete choice models in PanelBox - Logit, Probit, Fixed Effects, Random Effects, Ordered, Multinomial, and Dynamic models.
---

# Discrete Choice Models

Discrete choice models are designed for outcomes that take a finite set of values rather than a continuous range. When the dependent variable is binary (0/1), ordered (low/medium/high), or multinomial (bus/car/train), standard linear models are inappropriate -- they can predict values outside the valid range and mischaracterize the data generating process. Panel discrete choice models handle these outcomes while accounting for unobserved heterogeneity across entities.

PanelBox provides a comprehensive suite of 10 discrete choice estimators covering binary, ordered, and multinomial outcomes with pooled, fixed effects, random effects, and dynamic specifications.

## Available Models

### Binary Choice

| Model | Class | Estimation | When to Use |
|-------|-------|-----------|-------------|
| Pooled Logit | `PooledLogit` | MLE | Baseline; no entity effects |
| Pooled Probit | `PooledProbit` | MLE | Normal distribution preferred |
| FE Logit | `FixedEffectsLogit` | Conditional MLE | Correlated entity effects |
| RE Probit | `RandomEffectsProbit` | Simulated MLE | Uncorrelated entity effects |
| Dynamic Binary | `DynamicBinaryPanel` | MLE with initial conditions | State dependence |

### Ordered Choice

| Model | Class | Estimation | When to Use |
|-------|-------|-----------|-------------|
| Ordered Logit | `OrderedLogit` | MLE | Ordered categories (logistic) |
| Ordered Probit | `OrderedProbit` | MLE | Ordered categories (normal) |
| RE Ordered Logit | `RandomEffectsOrderedLogit` | Simulated MLE | Ordered + entity effects |

### Multinomial Choice

| Model | Class | Estimation | When to Use |
|-------|-------|-----------|-------------|
| Multinomial Logit | `MultinomialLogit` | MLE | Unordered alternatives |
| Conditional Logit | `ConditionalLogit` | MLE | Alternative-specific attributes |

## Quick Example

```python
from panelbox.models.discrete import PooledLogit

# Binary outcome model
model = PooledLogit("union ~ age + grade + hours", data, "id", "year")
results = model.fit(cov_type="cluster")
print(results.summary())

# Marginal effects at the mean
mfx = results.marginal_effects(at="mean")
print(mfx.summary())
```

## Key Concepts

### Fixed Effects Logit: The Incidental Parameters Problem

In nonlinear models, fixed effects estimation with many entity dummies leads to the **incidental parameters problem**: entity-specific intercepts are inconsistently estimated when $T$ is small, biasing slope coefficients. The Fixed Effects Logit avoids this by using Chamberlain's conditional likelihood, which conditions out the fixed effects entirely.

!!! warning "FE Logit drops entities"
    Entities whose outcome never varies (always 0 or always 1) are dropped because they contribute no information to the conditional likelihood.

### Marginal Effects

In nonlinear models, coefficients do not directly measure the effect of a one-unit change in $X$ on $P(Y=1)$. Compute marginal effects for interpretable quantities:

```python
# Average marginal effects (AME)
mfx = results.marginal_effects(at="overall")

# Marginal effects at the mean (MEM)
mfx = results.marginal_effects(at="mean")

# Marginal effects at specific values
mfx = results.marginal_effects(at={"age": 30, "grade": 12})
```

### Dynamic Discrete Choice

When past outcomes affect current choices (state dependence), use the dynamic specification:

```python
from panelbox.models.discrete import DynamicBinaryPanel

model = DynamicBinaryPanel(
    "union ~ L.union + age + grade",
    data, "id", "year"
)
results = model.fit()
```

!!! tip "State dependence vs. heterogeneity"
    Observed persistence in binary outcomes can arise from true state dependence (past $y$ causally affects current $y$) or unobserved heterogeneity (some entities are always more likely to have $y=1$). The dynamic model with RE separates these channels.

## Detailed Guides

- [Pooled Logit / Probit](binary.md) -- Baseline binary models *(detailed guide coming soon)*
- [Fixed Effects Logit](binary.md) -- Conditional MLE approach *(detailed guide coming soon)*
- [Random Effects Probit](binary.md) -- Simulated MLE with RE *(detailed guide coming soon)*
- [Ordered Models](ordered.md) -- Ordered logit and probit *(detailed guide coming soon)*
- [Multinomial Models](multinomial.md) -- Multinomial and conditional logit *(detailed guide coming soon)*
- [Dynamic Binary](dynamic.md) -- State dependence models *(detailed guide coming soon)*

## Tutorials

See [Discrete Choice Tutorial](../../tutorials/discrete.md) for interactive notebooks with Google Colab.

## API Reference

See [Discrete Choice API](../../api/discrete.md) for complete technical reference.

## References

- Chamberlain, G. (1980). Analysis of covariance with qualitative data. *Review of Economic Studies*, 47(1), 225-238.
- Wooldridge, J. M. (2005). Simple solutions to the initial conditions problem in dynamic, nonlinear panel data models with unobserved heterogeneity. *Journal of Applied Econometrics*, 20(1), 39-54.
- Train, K. E. (2009). *Discrete Choice Methods with Simulation* (2nd ed.). Cambridge University Press.
- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press.
