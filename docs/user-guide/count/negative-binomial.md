---
title: "Negative Binomial Regression"
description: "Guide to Negative Binomial panel regression in PanelBox - NB2 parameterization for overdispersed count data with pooled and fixed effects estimators."
---

# Negative Binomial Regression

!!! info "Quick Reference"
    **Classes:** `NegativeBinomial`, `FixedEffectsNegativeBinomial`
    **Import:** `from panelbox.models.count import NegativeBinomial, FixedEffectsNegativeBinomial`
    **Stata equivalent:** `nbreg`, `xtnbreg, fe`
    **R equivalent:** `pglm::pglm(family="negbin")`, `MASS::glm.nb()`

## Overview

The Negative Binomial (NB) model extends Poisson regression to handle **overdispersion** --- the common situation where the variance of count data exceeds its mean ($\text{Var}(y) > E[y]$). While Poisson regression assumes equidispersion ($\text{Var}(y) = E[y]$), real-world count data almost always violates this assumption.

PanelBox implements the NB2 parameterization (Cameron and Trivedi, 2013), where variance is a quadratic function of the mean:

$$\text{Var}(y_{it} \mid X_{it}) = \mu_{it} + \alpha \mu_{it}^2$$

where $\mu_{it} = \exp(X_{it}'\beta)$ and $\alpha \geq 0$ is the overdispersion parameter. When $\alpha = 0$, the model reduces to standard Poisson.

## Quick Example

```python
from panelbox.models.count import NegativeBinomial

model = NegativeBinomial(
    endog=data["claims"],
    exog=data[["age", "income", "risk_score"]],
    entity_id=data["policy_id"],
    time_id=data["year"]
)
results = model.fit()

# Overdispersion parameter
print(f"alpha = {results.alpha:.4f}")

# LR test: Poisson vs NB
lr_test = results.lr_test_poisson()
print(f"LR statistic: {lr_test['statistic']:.2f}, p-value: {lr_test['pvalue']:.4f}")
print(lr_test["conclusion"])
```

## When to Use

- Count data with $\text{Var}(y) > E[y]$ (overdispersion)
- Insurance claims, hospital visits, accident counts
- Patent data, publication counts
- Any count outcome where Poisson standard errors are too small

!!! warning "Key Assumptions"
    - **NB2 variance**: $\text{Var}(y) = \mu + \alpha \mu^2$ with $\alpha \geq 0$
    - **Correct mean specification**: $E[y \mid X] = \exp(X'\beta)$
    - **Independence across entities**: observations from different entities are independent
    - **No underdispersion**: NB cannot handle $\text{Var}(y) < E[y]$

## Detailed Guide

### When Poisson Fails

The Poisson model assumes $\text{Var}(y) = E[y]$, but in practice variance typically exceeds the mean. Overdispersion does **not** bias Poisson coefficient estimates, but it causes:

- **Standard errors that are too small** --- leading to inflated t-statistics
- **Confidence intervals that are too narrow** --- producing false rejections
- **Incorrect model selection** --- AIC/BIC comparisons are invalid

```python
from panelbox.models.count import PooledPoisson

# First, fit Poisson to check overdispersion
poisson = PooledPoisson(
    endog=data["claims"],
    exog=data[["age", "income"]],
    entity_id=data["policy_id"],
    time_id=data["year"]
)
pois_results = poisson.fit(se_type="cluster")

# Check variance-to-mean ratio
od_test = pois_results.check_overdispersion()
print(od_test)
```

### NB2 Parameterization

The NB2 model introduces one additional parameter $\alpha$ to capture overdispersion:

| Quantity | Formula | Poisson ($\alpha = 0$) |
|----------|---------|----------------------|
| Mean | $\mu = \exp(X'\beta)$ | Same |
| Variance | $\mu + \alpha \mu^2$ | $\mu$ |
| Prob($y = k$) | $\frac{\Gamma(k + 1/\alpha)}{\Gamma(k+1)\Gamma(1/\alpha)} \left(\frac{1/\alpha}{1/\alpha + \mu}\right)^{1/\alpha} \left(\frac{\mu}{1/\alpha + \mu}\right)^k$ | $e^{-\mu} \mu^k / k!$ |

The NB2 model can be derived as a **Poisson-Gamma mixture**: $y \mid \lambda \sim \text{Poisson}(\lambda)$ with $\lambda \sim \text{Gamma}(\mu, \alpha)$.

### Estimation

#### Pooled Negative Binomial

```python
from panelbox.models.count import NegativeBinomial

model = NegativeBinomial(
    endog=data["claims"],
    exog=data[["age", "income", "risk_score"]],
    entity_id=data["policy_id"],
    time_id=data["year"]
)
results = model.fit(method="BFGS", maxiter=1000)

print(results.summary())
```

#### Fixed Effects Negative Binomial

The FE NB model (Allison and Waterman, 2002) includes entity dummies in the NB model:

```python
from panelbox.models.count import FixedEffectsNegativeBinomial

model = FixedEffectsNegativeBinomial(
    endog=data["claims"],
    exog=data[["age", "income", "risk_score"]],
    entity_id=data["policy_id"],
    time_id=data["year"]
)
results = model.fit()
```

!!! note "FE NB Caveat"
    The Allison-Waterman FE NB estimator uses a dummy variable approach (LSDV) rather than true conditional ML. With many entities, this can be computationally intensive, and PanelBox will warn if there are more than 100 entities. For large panels, consider Poisson FE with cluster-robust SE as an alternative.

### Interpreting Results

#### Coefficients

As in Poisson, NB coefficients are **semi-elasticities**:

$$\frac{\partial \ln E[y \mid X]}{\partial x_k} = \beta_k$$

A one-unit increase in $x_k$ changes $E[y]$ by approximately $100 \times \beta_k$ percent. Exponentiated coefficients give incidence rate ratios (IRR):

```python
import numpy as np

# Coefficients and IRR
for name, coef, se in zip(results.exog_names, results.params_exog, results.se):
    irr = np.exp(coef)
    print(f"{name}: beta = {coef:.4f} (SE = {se:.4f}), IRR = {irr:.4f}")
```

#### Overdispersion Parameter

The estimated $\alpha$ quantifies the degree of overdispersion:

```python
print(f"Overdispersion (alpha): {results.alpha:.4f}")

# Interpretation
if results.alpha < 0.01:
    print("Minimal overdispersion - Poisson may suffice")
elif results.alpha < 1.0:
    print("Moderate overdispersion - NB preferred")
else:
    print("Strong overdispersion - NB strongly preferred")
```

### Testing Poisson vs Negative Binomial

#### Likelihood Ratio Test

The LR test compares Poisson ($\alpha = 0$) against NB ($\alpha > 0$):

$$LR = 2(\ell_{NB} - \ell_{\text{Poisson}}) \sim \bar{\chi}^2(1)$$

The distribution is a mixture of $\chi^2(0)$ and $\chi^2(1)$ since $\alpha = 0$ is on the boundary.

```python
# Built-in LR test
lr_test = results.lr_test_poisson()
print(f"LR statistic: {lr_test['statistic']:.2f}")
print(f"p-value: {lr_test['pvalue']:.4f}")
print(f"Conclusion: {lr_test['conclusion']}")
```

#### Informal Check: Variance-to-Mean Ratio

```python
var_mean_ratio = data["claims"].var() / data["claims"].mean()
print(f"Var/Mean ratio: {var_mean_ratio:.2f}")
# Poisson expects ~1.0; values >> 1 suggest overdispersion
```

### When NOT to Use

- **Underdispersion** ($\text{Var}(y) < E[y]$): NB cannot handle this; consider generalized Poisson
- **Excess zeros**: if overdispersion is driven by too many zeros, consider [Zero-Inflated models](zero-inflated.md)
- **Gravity models**: use [PPML](ppml.md) instead, which provides elasticity tools and handles heteroskedasticity

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | array-like | *required* | Dependent variable (non-negative counts) |
| `exog` | array-like | *required* | Independent variables |
| `entity_id` | array-like | `None` | Entity identifiers |
| `time_id` | array-like | `None` | Time identifiers |
| `weights` | array-like | `None` | Observation weights |

### fit() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_params` | array | `None` | Starting values (Poisson estimates + $\alpha = 0.1$ if `None`) |
| `method` | str | `"BFGS"` | Optimization method |
| `maxiter` | int | `1000` | Maximum iterations |

## Diagnostics

### Model Comparison

```python
from panelbox.models.count import PooledPoisson, NegativeBinomial

# Fit both models
poisson = PooledPoisson(endog=y, exog=X, entity_id=entity, time_id=time)
pois_res = poisson.fit(se_type="cluster")

nb = NegativeBinomial(endog=y, exog=X, entity_id=entity, time_id=time)
nb_res = nb.fit()

# Compare
print(f"Poisson LLF: {pois_res.llf:.2f}, AIC: {pois_res.aic:.2f}")
print(f"NB LLF:      {nb_res.llf:.2f}, AIC: {nb_res.aic:.2f}")
print(f"Alpha:       {nb_res.alpha:.4f}")

# LR test
lr_test = nb_res.lr_test_poisson()
print(f"LR test p-value: {lr_test['pvalue']:.4f}")
```

### Predictions

```python
# Predicted counts
y_hat = nb_res.predict(which="mean")

# Linear predictor
xb = nb_res.predict(which="linear")
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Count Data Models | Poisson vs NB comparison with overdispersion testing | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/02_negative_binomial.ipynb) |

## See Also

- [Count Data Overview](index.md) --- Introduction and model selection guide
- [Poisson Models](poisson.md) --- Baseline count model (equidispersion assumed)
- [Zero-Inflated Models](zero-inflated.md) --- When excess zeros drive overdispersion
- [Marginal Effects for Count Data](marginal-effects.md) --- AME and IRR computation

## References

- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.
- Allison, P. D., & Waterman, R. P. (2002). Fixed-Effects Negative Binomial Regression Models. *Sociological Methodology*, 32(1), 247--265.
- Hilbe, J. M. (2011). *Negative Binomial Regression* (2nd ed.). Cambridge University Press.
- Cameron, A. C., & Trivedi, P. K. (1986). Econometric Models Based on Count Data: Comparisons and Applications of Some Estimators and Tests. *Journal of Applied Econometrics*, 1(1), 29--53.
