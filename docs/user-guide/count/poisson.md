---
title: "Poisson Regression"
description: "Guide to panel Poisson regression models in PanelBox — Pooled, Fixed Effects, Random Effects, and Quasi-ML estimators for count data."
---

# Poisson Regression

!!! info "Quick Reference"
    **Classes:** `PooledPoisson`, `PoissonFixedEffects`, `RandomEffectsPoisson`, `PoissonQML`
    **Import:** `from panelbox.models.count import PooledPoisson, PoissonFixedEffects, RandomEffectsPoisson, PoissonQML`
    **Stata equivalent:** `poisson`, `xtpoisson, fe`, `xtpoisson, re`
    **R equivalent:** `pglm::pglm()`, `fixest::fepois()`

## Overview

Poisson regression is the baseline model for count data --- outcomes that take non-negative integer values such as $y_{it} \in \{0, 1, 2, \ldots\}$. The model specifies the conditional mean as an exponential function of regressors:

$$E[y_{it} \mid X_{it}] = \exp(X_{it}'\beta)$$

The exponential link ensures predicted counts are always non-negative. Under the Poisson distribution, the variance equals the mean (**equidispersion**):

$$\text{Var}(y_{it} \mid X_{it}) = E[y_{it} \mid X_{it}] = \exp(X_{it}'\beta)$$

PanelBox provides four Poisson variants for panel data, each handling unobserved entity heterogeneity and overdispersion differently.

## Quick Example

```python
from panelbox.models.count import PoissonFixedEffects
import numpy as np

# Poisson FE for patent counts
model = PoissonFixedEffects(
    endog=data["patents"],
    exog=data[["rd_spending", "employees", "capital"]],
    entity_id=data["firm"],
    time_id=data["year"]
)
results = model.fit()
print(results.summary())
```

## When to Use

- Count outcomes: patents, hospital visits, accidents, publications, trade events
- Modeling rates or event frequencies in panel data
- When $E[y \mid X] = \exp(X'\beta)$ is a reasonable specification
- As a baseline before testing for overdispersion or excess zeros

!!! warning "Key Assumptions"
    - **Poisson distribution**: $\text{Var}(y) = E[y]$ (equidispersion) --- violated in most applications
    - **Correct mean specification**: $E[y_{it} \mid X_{it}] = \exp(X_{it}'\beta)$
    - **Independence** (Pooled): observations are independent conditional on $X$
    - **Strict exogeneity** (FE/RE): $E[y_{it} \mid X_{i1}, \ldots, X_{iT}, \alpha_i] = \exp(X_{it}'\beta + \alpha_i)$

## Detailed Guide

### Model Variants

PanelBox provides four Poisson estimators, each suited to different situations:

| Model | Class | Heterogeneity | Overdispersion-Robust | Entities Dropped |
|-------|-------|--------------|----------------------|-----------------|
| Pooled Poisson | `PooledPoisson` | Ignored | Via cluster SE | None |
| Poisson FE | `PoissonFixedEffects` | Fixed effects (conditional ML) | By construction | Zero-count entities |
| Poisson RE | `RandomEffectsPoisson` | Random effects (gamma/lognormal) | Partially | None |
| Poisson QML | `PoissonQML` | Via robust SE | Fully robust | None |

### Data Preparation

Count data must satisfy:

- **Non-negative integers**: $y_{it} \geq 0$
- **Panel structure**: entity and time identifiers
- **No missing values** in outcome or regressors

```python
import pandas as pd
import numpy as np

# Check data requirements
assert (data["patents"] >= 0).all(), "Outcome must be non-negative"
assert data["patents"].dtype in [int, np.int64, np.int32] or \
       (data["patents"] == data["patents"].astype(int)).all(), \
       "Outcome should be integer-valued"

# Summary statistics for count data
print(f"Mean:     {data['patents'].mean():.2f}")
print(f"Variance: {data['patents'].var():.2f}")
print(f"Zeros:    {(data['patents'] == 0).sum()} ({100*(data['patents'] == 0).mean():.1f}%)")
print(f"Max:      {data['patents'].max()}")
```

### Estimation

#### Pooled Poisson

The baseline model ignores entity heterogeneity. Use cluster-robust standard errors to account for within-entity correlation:

```python
from panelbox.models.count import PooledPoisson

model = PooledPoisson(
    endog=data["patents"],
    exog=data[["rd_spending", "employees"]],
    entity_id=data["firm"],
    time_id=data["year"]
)
results = model.fit(se_type="cluster")
```

#### Poisson Fixed Effects

Conditional maximum likelihood (Hausman, Hall, and Griliches, 1984) eliminates entity-specific effects by conditioning on $\sum_t y_{it}$. This avoids the incidental parameters problem:

$$P(y_{i1}, \ldots, y_{iT} \mid \sum_t y_{it}) = \frac{(\sum_t y_{it})!}{\prod_t y_{it}!} \prod_t \left(\frac{\exp(X_{it}'\beta)}{\sum_s \exp(X_{is}'\beta)}\right)^{y_{it}}$$

```python
from panelbox.models.count import PoissonFixedEffects

model = PoissonFixedEffects(
    endog=data["patents"],
    exog=data[["rd_spending", "employees"]],
    entity_id=data["firm"],
    time_id=data["year"]
)
results = model.fit()

# Check dropped entities (those with all-zero counts)
print(f"Entities dropped: {results.n_dropped}")
print(f"Dropped IDs: {results.dropped_entities}")
```

!!! note "Dropped Entities"
    Entities where $\sum_t y_{it} = 0$ (all-zero counts) carry no information for conditional ML and are automatically dropped. The attributes `n_dropped` and `dropped_entities` track which entities were removed.

#### Random Effects Poisson

Assumes entity heterogeneity follows a parametric distribution. Two mixing distributions are available:

=== "Gamma RE (default)"

    ```python
    from panelbox.models.count import RandomEffectsPoisson

    model = RandomEffectsPoisson(
        endog=data["patents"],
        exog=data[["rd_spending", "employees"]],
        entity_id=data["firm"],
        time_id=data["year"]
    )
    results = model.fit(distribution="gamma")

    # Overdispersion parameter
    print(f"theta: {results.theta:.4f}")
    ```

    With gamma RE, $\alpha_i \sim \text{Gamma}(1/\theta, \theta)$, which leads to a Negative Binomial marginal distribution. The overdispersion is $1 + \theta$.

=== "Lognormal RE"

    ```python
    results = model.fit(distribution="normal")

    # Variance of random intercept
    print(f"theta: {results.theta:.4f}")
    ```

    With lognormal RE, $\alpha_i = \exp(u_i)$ where $u_i \sim N(0, \theta)$. Uses Gauss-Hermite quadrature for likelihood integration.

#### Poisson QML (Wooldridge 1999)

The quasi-maximum likelihood estimator requires only correct specification of the conditional mean --- no distributional assumption:

$$E[y_{it} \mid X_{it}] = \exp(X_{it}'\beta)$$

This makes it consistent even when the Poisson distribution is misspecified (e.g., under overdispersion):

```python
from panelbox.models.count import PoissonQML

model = PoissonQML(
    endog=data["patents"],
    exog=data[["rd_spending", "employees"]],
    entity_id=data["firm"],
    time_id=data["year"]
)
results = model.fit(se_type="robust")
```

!!! tip "When to use QML"
    Poisson QML is the safest choice when you are unsure about the distributional assumptions. It uses robust (sandwich) standard errors that are valid regardless of the true data-generating process, as long as the mean specification is correct.

### Interpreting Results

Poisson coefficients have a **semi-elasticity** interpretation:

$$\frac{\partial \ln E[y \mid X]}{\partial x_k} = \beta_k$$

This means: a one-unit increase in $x_k$ changes $E[y]$ by approximately $100 \times \beta_k$ percent.

```python
# Print coefficients as semi-elasticities
for name, coef, se in zip(results.exog_names, results.params, results.se):
    pct_change = 100 * coef
    print(f"{name}: beta = {coef:.4f} (SE = {se:.4f})")
    print(f"  -> 1-unit increase in {name} changes E[y] by {pct_change:.1f}%")
```

For **incidence rate ratios** (IRR), exponentiate the coefficients:

$$\text{IRR}_k = \exp(\beta_k)$$

A unit increase in $x_k$ multiplies the expected count by $\exp(\beta_k)$.

```python
# Incidence Rate Ratios
for name, coef in zip(results.exog_names, results.params):
    irr = np.exp(coef)
    print(f"{name}: IRR = {irr:.4f}")
    print(f"  -> 1-unit increase in {name} multiplies E[y] by {irr:.3f}")
```

### Predictions

All Poisson models support predictions in two forms:

```python
# Predicted counts: E[y|X] = exp(X'beta)
y_hat = results.predict(type="response")

# Linear predictor: X'beta
xb = results.predict(type="linear")
```

## Configuration Options

### PooledPoisson / PoissonQML

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | array-like | *required* | Dependent variable (non-negative counts) |
| `exog` | array-like | *required* | Independent variables |
| `entity_id` | array-like | `None` | Entity identifiers (for clustering) |
| `time_id` | array-like | `None` | Time identifiers |

### fit() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_params` | array | `None` | Starting values for optimization |
| `method` | str | `"BFGS"` | Optimization method |
| `maxiter` | int | `1000` | Maximum iterations |
| `se_type` | str | `"cluster"` | Standard error type: `"cluster"` or `"robust"` |

### PoissonFixedEffects

Same constructor as Pooled, but `entity_id` is required. The `fit()` method does not take `se_type` (standard errors come from the conditional likelihood).

### RandomEffectsPoisson

Same constructor as Pooled, but `entity_id` is required. The `fit()` method accepts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distribution` | str | `"gamma"` | RE distribution: `"gamma"` or `"normal"` |
| `start_params` | array | `None` | Starting values |

## Standard Errors

| SE Type | Models | Description |
|---------|--------|-------------|
| `"cluster"` | PooledPoisson, PoissonQML | Clustered by entity (default) |
| `"robust"` | PooledPoisson, PoissonQML | Heteroskedasticity-robust (sandwich) |
| Conditional ML | PoissonFixedEffects | From conditional log-likelihood Hessian |
| Model-based | RandomEffectsPoisson | From integrated log-likelihood Hessian |

## Diagnostics

### Overdispersion Test

The Poisson assumption $\text{Var}(y) = E[y]$ rarely holds in practice. Use `check_overdispersion()` to test:

```python
from panelbox.models.count import PooledPoisson

model = PooledPoisson(
    endog=data["patents"],
    exog=data[["rd_spending", "employees"]],
    entity_id=data["firm"],
    time_id=data["year"]
)
results = model.fit(se_type="cluster")

# Cameron-Trivedi test for overdispersion
od_test = results.check_overdispersion()
print(od_test)
```

If overdispersion is detected:

| Approach | When to Use |
|----------|-------------|
| Cluster-robust SE | Mild overdispersion; coefficients unbiased |
| Poisson QML | Moderate overdispersion; robust inference |
| Negative Binomial | Strong overdispersion; models variance directly |

### Comparing Models

```python
from panelbox.models.count import PooledPoisson, PoissonFixedEffects, PoissonQML

# Fit multiple specifications
pooled = PooledPoisson(endog=y, exog=X, entity_id=entity, time_id=time)
res_pooled = pooled.fit(se_type="cluster")

fe = PoissonFixedEffects(endog=y, exog=X, entity_id=entity, time_id=time)
res_fe = fe.fit()

qml = PoissonQML(endog=y, exog=X, entity_id=entity, time_id=time)
res_qml = qml.fit(se_type="robust")

# Compare log-likelihoods and AIC
print(f"Pooled: LLF = {res_pooled.llf:.2f}, AIC = {res_pooled.aic:.2f}")
print(f"FE:     LLF = {res_fe.llf:.2f}")
print(f"QML:    LLF = {res_qml.llf:.2f}, AIC = {res_qml.aic:.2f}")
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Count Data Models | Complete guide with Poisson, NB, and ZI models | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/01_poisson_introduction.ipynb) |

## See Also

- [Count Data Overview](index.md) --- Introduction and model selection guide
- [PPML](ppml.md) --- Poisson Pseudo-Maximum Likelihood for gravity models
- [Negative Binomial](negative-binomial.md) --- Handling overdispersion
- [Zero-Inflated Models](zero-inflated.md) --- Excess zeros
- [Marginal Effects for Count Data](marginal-effects.md) --- Interpreting nonlinear models

## References

- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.
- Hausman, J. A., Hall, B. H., & Griliches, Z. (1984). Econometric Models for Count Data with an Application to the Patents-R&D Relationship. *Econometrica*, 52(4), 909--938.
- Wooldridge, J. M. (1999). Distribution-Free Estimation of Some Nonlinear Panel Data Models. *Journal of Econometrics*, 90(1), 77--97.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press, Chapter 18.
