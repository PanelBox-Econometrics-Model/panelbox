---
title: "Binary Choice Models"
description: "Logit and Probit models for binary panel data outcomes with fixed and random effects in PanelBox"
---

# Binary Choice Models

!!! info "Quick Reference"
    **Classes:** `PooledLogit`, `PooledProbit`, `FixedEffectsLogit`, `RandomEffectsProbit`
    **Import:** `from panelbox.models.discrete.binary import PooledLogit, PooledProbit, FixedEffectsLogit, RandomEffectsProbit`
    **Stata equivalent:** `logit`, `probit`, `xtlogit, fe`, `xtprobit, re`
    **R equivalent:** `glm(family=binomial)`, `bife::bife()`, `pglm::pglm()`

## Overview

Binary choice models are used when the dependent variable takes only two values, typically coded as $y_{it} \in \{0, 1\}$. The goal is to model the probability of the event occurring conditional on covariates:

$$P(y_{it} = 1 \mid X_{it}) = F(X_{it}'\beta)$$

where $F(\cdot)$ is a link function. The **Logit** model uses the logistic CDF $\Lambda(\cdot)$, while the **Probit** model uses the standard normal CDF $\Phi(\cdot)$.

Panel data introduces individual-specific unobserved heterogeneity $\alpha_i$ that, if correlated with the regressors, biases pooled estimates. PanelBox provides four estimation strategies to handle this: pooled models that ignore heterogeneity (useful as baselines), fixed effects that control for it through conditional likelihood, and random effects that model it parametrically.

## Quick Example

```python
import pandas as pd
from panelbox.models.discrete.binary import PooledLogit

# Load panel data with binary outcome
data = pd.read_csv("panel_data.csv")

# Fit Pooled Logit with cluster-robust SEs (default)
model = PooledLogit("employed ~ education + experience + age", data, "id", "year")
results = model.fit(cov_type="cluster")
print(results.summary())

# Marginal effects
me = model.marginal_effects(at="overall", method="dydx")
print(me)

# Classification quality
metrics = model.classification_metrics(threshold=0.5)
print(f"Accuracy: {metrics['accuracy']:.3f}, AUC: {metrics['auc_roc']:.3f}")
```

## When to Use

- **PooledLogit / PooledProbit** -- Baseline model; use when unobserved heterogeneity is not a concern or as a starting point for comparison
- **FixedEffectsLogit** -- When unobserved individual effects $\alpha_i$ are correlated with regressors; eliminates bias from time-invariant omitted variables
- **RandomEffectsProbit** -- When $\alpha_i$ is uncorrelated with regressors; more efficient than FE, identifies time-invariant covariates

!!! warning "Key Assumptions"
    - **Logit**: errors follow logistic distribution; odds ratios have proportional interpretation
    - **Probit**: errors follow normal distribution; coefficients scale with $\sigma$
    - **FE Logit**: strict exogeneity conditional on $\alpha_i$; only within-entity variation in $y$ identifies parameters
    - **RE Probit**: $\alpha_i \sim N(0, \sigma^2_\alpha)$ independent of regressors; Gauss-Hermite quadrature for integration

## Model Comparison

| Model | Heterogeneity | Time-Invariant Vars | Marginal Effects | Estimation |
|-------|--------------|---------------------|-----------------|------------|
| `PooledLogit` | Ignored | Identified | AME, MEM | MLE (statsmodels) |
| `PooledProbit` | Ignored | Identified | AME, MEM | MLE (statsmodels) |
| `FixedEffectsLogit` | Controlled (FE) | Not identified | Conditional only | Conditional MLE |
| `RandomEffectsProbit` | Modeled (RE) | Identified | Integrated AME | MLE + quadrature |

## Detailed Guide

### Data Preparation

Binary choice models require the dependent variable to be coded as 0 or 1. Ensure your panel data is in long format with entity and time identifiers.

```python
import pandas as pd
import numpy as np

# Example panel data
data = pd.DataFrame({
    "id": np.repeat(range(1, 101), 5),
    "year": np.tile(range(2015, 2020), 100),
    "employed": np.random.binomial(1, 0.7, 500),
    "education": np.random.normal(12, 3, 500),
    "experience": np.random.exponential(5, 500),
    "female": np.repeat(np.random.binomial(1, 0.5, 100), 5),
})
```

### Pooled Logit

The pooled logit model ignores the panel structure and estimates:

$$P(y_{it} = 1 \mid X_{it}) = \Lambda(X_{it}'\beta) = \frac{\exp(X_{it}'\beta)}{1 + \exp(X_{it}'\beta)}$$

```python
from panelbox.models.discrete.binary import PooledLogit

model = PooledLogit("employed ~ education + experience", data, "id", "year")
results = model.fit(cov_type="cluster")  # Cluster-robust SEs by entity

# Coefficients (log-odds ratios)
print(results.params)

# Predicted probabilities
probs = model.predict(type="prob")

# Pseudo R-squared variants
print(model.pseudo_r2(kind="mcfadden"))
print(model.pseudo_r2(kind="cox_snell"))
print(model.pseudo_r2(kind="nagelkerke"))
```

### Pooled Probit

The pooled probit uses the standard normal CDF:

$$P(y_{it} = 1 \mid X_{it}) = \Phi(X_{it}'\beta)$$

```python
from panelbox.models.discrete.binary import PooledProbit

model = PooledProbit("employed ~ education + experience", data, "id", "year")
results = model.fit(cov_type="robust")  # Heteroskedasticity-robust SEs
print(results.summary())
```

### Fixed Effects Logit (Chamberlain 1980)

The FE logit avoids the incidental parameters problem by conditioning on the sufficient statistic $\sum_t y_{it}$:

$$L_i^{cond}(\beta) = \frac{\exp\left(\sum_t y_{it} X_{it}'\beta\right)}{\sum_{d \in \mathcal{D}_i} \exp\left(\sum_t d_t X_{it}'\beta\right)}$$

where $\mathcal{D}_i$ is the set of all binary sequences with the same sum as the observed sequence.

```python
from panelbox.models.discrete.binary import FixedEffectsLogit

model = FixedEffectsLogit("employed ~ education + experience", data, "id", "year")
results = model.fit(method="bfgs")

# Entities dropped (no variation in y)
print(f"Entities used: {model.n_used_entities}")
print(f"Entities dropped: {model.n_dropped_entities}")

# Coefficients (only time-varying variables)
print(results.params)
```

!!! warning "Dropped Entities"
    Entities where $y_{it}$ is always 0 or always 1 provide no information for the conditional likelihood and are automatically dropped. Check `n_dropped_entities` to understand how many observations are lost. If a large fraction is dropped, consider the RE Probit instead.

### Random Effects Probit

The RE probit models individual heterogeneity as $\alpha_i \sim N(0, \sigma^2_\alpha)$ and integrates it out using Gauss-Hermite quadrature:

$$P(y_{it} = 1 \mid X_{it}, \alpha_i) = \Phi(X_{it}'\beta + \alpha_i)$$

$$L_i(\beta, \sigma_\alpha) = \int \prod_t \Phi(X_{it}'\beta + \alpha_i)^{y_{it}} [1 - \Phi(X_{it}'\beta + \alpha_i)]^{1-y_{it}} \, \phi(\alpha_i / \sigma_\alpha) \, d\alpha_i$$

```python
from panelbox.models.discrete.binary import RandomEffectsProbit

model = RandomEffectsProbit(
    "employed ~ education + experience + female",
    data, "id", "year",
    quadrature_points=12  # Number of Gauss-Hermite nodes
)
results = model.fit(method="bfgs", maxiter=500, tol=1e-8)

# Random effects parameters
print(f"sigma_alpha: {model.sigma_alpha:.4f}")
print(f"rho (ICC): {model.rho:.4f}")

# rho = sigma_alpha^2 / (1 + sigma_alpha^2)
# High rho indicates substantial unobserved heterogeneity
```

!!! tip "Choosing Quadrature Points"
    The default of 12 quadrature points provides good accuracy for most applications. Increase to 20-24 if $\sigma_\alpha$ is large (> 2) or if you observe sensitivity of estimates to the number of points. Fewer points (8) may suffice for small $\sigma_\alpha$.

## Configuration Options

### PooledLogit / PooledProbit

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | `str` | required | R-style formula (e.g., `"y ~ x1 + x2"`) |
| `data` | `DataFrame` | required | Panel data in long format |
| `entity_col` | `str` | required | Entity identifier column |
| `time_col` | `str` | required | Time identifier column |
| `weights` | `ndarray` | `None` | Observation weights |

**fit() parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cov_type` | `str` | `"cluster"` | SE type: `"nonrobust"`, `"robust"`, `"cluster"` |

### FixedEffectsLogit

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | `str` | required | Formula with time-varying variables only |
| `data` | `DataFrame` | required | Panel data in long format |
| `entity_col` | `str` | required | Entity identifier column |
| `time_col` | `str` | required | Time identifier column |

**fit() parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"bfgs"` | Optimization: `"bfgs"` or `"newton"` |

### RandomEffectsProbit

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | `str` | required | R-style formula |
| `data` | `DataFrame` | required | Panel data in long format |
| `entity_col` | `str` | required | Entity identifier column |
| `time_col` | `str` | required | Time identifier column |
| `quadrature_points` | `int` | `12` | Gauss-Hermite quadrature nodes |
| `weights` | `ndarray` | `None` | Observation weights |

**fit() parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"bfgs"` | Optimization: `"bfgs"`, `"l-bfgs-b"` |
| `maxiter` | `int` | `500` | Maximum iterations |
| `tol` | `float` | `1e-8` | Convergence tolerance |

## Standard Errors

| Type | `cov_type` | Available In | Description |
|------|-----------|-------------|-------------|
| Classical | `"nonrobust"` | Pooled models | Assumes i.i.d. errors; biased with clustering |
| Robust | `"robust"` | Pooled models | Heteroskedasticity-robust (sandwich) |
| Cluster-robust | `"cluster"` | Pooled models | Robust to within-entity correlation (default) |

!!! note "FE Logit and RE Probit Standard Errors"
    For `FixedEffectsLogit`, standard errors are derived from the Hessian of the conditional log-likelihood. For `RandomEffectsProbit`, they are computed from the Hessian of the marginal (integrated) log-likelihood.

## Diagnostics

### Classification Metrics

```python
# Available for Pooled Logit and Pooled Probit
metrics = model.classification_metrics(threshold=0.5)
print(f"Accuracy:  {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall:    {metrics['recall']:.3f}")
print(f"F1 Score:  {metrics['f1']:.3f}")
print(f"AUC-ROC:   {metrics['auc_roc']:.3f}")
```

### Hosmer-Lemeshow Test

Tests goodness-of-fit by grouping observations into deciles of predicted probabilities:

```python
hl_test = model.hosmer_lemeshow_test(n_groups=10)
print(f"Statistic: {hl_test['statistic']:.3f}")
print(f"p-value:   {hl_test['p_value']:.3f}")
# Null hypothesis: model fits well. Reject if p < 0.05
```

### Information Matrix Test

Tests for model misspecification:

```python
im_test = model.information_matrix_test()
print(f"Statistic: {im_test['statistic']:.3f}")
print(f"p-value:   {im_test['p_value']:.3f}")
```

### Link Test

Tests whether the functional form (logit/probit link) is correctly specified:

```python
link = model.link_test()
print(f"hat coefficient:  {link['hat']:.3f}")
print(f"hatsq coefficient: {link['hatsq']:.3f}")
print(f"hatsq p-value:    {link['hatsq_pvalue']:.3f}")
# Significant hatsq suggests misspecification
```

### Pseudo R-squared

```python
# McFadden (default)
r2_mcf = model.pseudo_r2(kind="mcfadden")

# Cox-Snell
r2_cs = model.pseudo_r2(kind="cox_snell")

# Nagelkerke
r2_nag = model.pseudo_r2(kind="nagelkerke")
```

### Model Selection

```python
# Compare models using information criteria
print(f"Log-likelihood: {results.llf:.3f}")
print(f"AIC: {results.aic:.3f}")
print(f"BIC: {results.bic:.3f}")
```

## Result Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `Series` | Coefficient estimates |
| `std_errors` | `Series` | Standard errors |
| `cov_params` | `DataFrame` | Variance-covariance matrix |
| `resid` | `ndarray` | Response residuals ($y - \hat{p}$) |
| `fittedvalues` | `ndarray` | Fitted probabilities |
| `llf` | `float` | Log-likelihood at maximum |
| `ll_null` | `float` | Null model log-likelihood |
| `pseudo_r2_mcfadden` | `float` | McFadden pseudo $R^2$ |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |
| `converged` | `bool` | Convergence flag |

Additional attributes for **RandomEffectsProbit**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `rho` | `float` | Intra-class correlation $\rho = \sigma^2_\alpha / (1 + \sigma^2_\alpha)$ |
| `sigma_alpha` | `float` | Random effects standard deviation |
| `quadrature_points` | `int` | Number of quadrature points used |
| `n_iter` | `int` | Number of optimization iterations |

Additional attributes for **FixedEffectsLogit**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_used_entities` | `int` | Entities contributing to the likelihood |
| `n_dropped_entities` | `int` | Entities dropped (no variation) |

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Discrete Choice Models | Comprehensive guide to all binary models | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/01_binary_choice_introduction.ipynb) |

## See Also

- [Ordered Choice Models](ordered.md) -- Extension to ordinal outcomes
- [Multinomial and Conditional Logit](multinomial.md) -- Unordered multi-category outcomes
- [Dynamic Binary Panel](dynamic.md) -- State dependence with lagged dependent variable
- [Marginal Effects](marginal-effects.md) -- Computing and interpreting marginal effects
- [Standard Errors](../../inference/index.md) -- Robust and clustered standard errors

## References

- Chamberlain, G. (1980). "Analysis of Covariance with Qualitative Data." *Review of Economic Studies*, 47(1), 225-238.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press. Chapters 15-16.
- Cameron, A. C. and Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press.
- Hosmer, D. W. and Lemeshow, S. (2000). *Applied Logistic Regression*. 2nd ed. Wiley.
- Greene, W. H. (2018). *Econometric Analysis*. 8th ed. Pearson. Chapter 17.
