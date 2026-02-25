---
title: "Marginal Effects for Count Data"
description: "Guide to computing and interpreting marginal effects, elasticities, and incidence rate ratios for count data models in PanelBox."
---

# Marginal Effects for Count Data

!!! info "Quick Reference"
    **Method:** `results.marginal_effects(at="overall")`
    **PPML:** `results.elasticity(variable)`, `results.elasticities()`
    **Stata equivalent:** `margins, dydx(*) predict(n)`
    **R equivalent:** `margins::margins()`, `marginaleffects::marginaleffects()`

## Overview

Count data models use a nonlinear link function --- $E[y \mid X] = \exp(X'\beta)$ --- which means the raw coefficients ($\beta$) do not directly measure the effect of a unit change in $x$ on the expected count. Instead, $\beta$ measures the semi-elasticity: the proportional change in $E[y]$.

To communicate results in terms of **actual count changes**, researchers compute marginal effects. This page covers marginal effects for all PanelBox count models: Poisson, Negative Binomial, PPML, and Zero-Inflated models.

## Quick Example

```python
from panelbox.models.count import PooledPoisson
import numpy as np

model = PooledPoisson(
    endog=data["patents"],
    exog=data[["rd_spending", "employees"]],
    entity_id=data["firm"],
    time_id=data["year"]
)
results = model.fit(se_type="cluster")

# Average Marginal Effects
ame = results.marginal_effects(at="overall")
print(ame)
```

## Three Ways to Read Coefficients

Count model coefficients can be interpreted in three equivalent ways:

### 1. Semi-Elasticity (Direct Coefficient)

The coefficient $\beta_k$ is the semi-elasticity:

$$\frac{\partial \ln E[y \mid X]}{\partial x_k} = \beta_k$$

A one-unit increase in $x_k$ changes $E[y]$ by approximately $100 \times \beta_k$ percent.

```python
# Direct reading from coefficients
for name, coef in zip(results.exog_names, results.params):
    print(f"{name}: 1-unit increase -> {100*coef:.1f}% change in E[y]")
```

### 2. Incidence Rate Ratio (Exponentiated Coefficient)

The IRR is $\exp(\beta_k)$, giving a multiplicative interpretation:

$$\frac{E[y \mid x_k + 1]}{E[y \mid x_k]} = \exp(\beta_k)$$

A unit increase in $x_k$ multiplies the expected count by $\exp(\beta_k)$.

```python
# Incidence Rate Ratios
for name, coef in zip(results.exog_names, results.params):
    irr = np.exp(coef)
    print(f"{name}: IRR = {irr:.4f}")
    if irr > 1:
        print(f"  -> multiplies E[y] by {irr:.3f} ({100*(irr-1):.1f}% increase)")
    else:
        print(f"  -> multiplies E[y] by {irr:.3f} ({100*(1-irr):.1f}% decrease)")
```

### 3. Marginal Effect (Actual Count Change)

The marginal effect gives the change in the expected count for a unit change in $x_k$:

$$\frac{\partial E[y \mid X]}{\partial x_k} = \beta_k \cdot \exp(X'\beta) = \beta_k \cdot E[y \mid X]$$

Because this depends on $X$, it varies across observations. Two summary measures are standard:

| Measure | Formula | Description |
|---------|---------|-------------|
| **AME** (Average Marginal Effect) | $\frac{1}{N}\sum_i \beta_k \cdot \exp(X_i'\beta)$ | Average over all observations |
| **MEM** (Marginal Effect at Means) | $\beta_k \cdot \exp(\bar{X}'\beta)$ | Evaluate at sample means |

## Computing Marginal Effects

### Poisson and Negative Binomial

All Poisson and NB models support the `marginal_effects()` method:

```python
# Average Marginal Effects (AME)
ame = results.marginal_effects(at="overall")

# Marginal Effects at Means (MEM)
mem = results.marginal_effects(at="means")

# Subset of variables
ame_subset = results.marginal_effects(at="overall", varlist=["rd_spending"])
```

The `at` parameter controls the evaluation point:

| Value | Type | Description |
|-------|------|-------------|
| `"overall"` or `"mean"` | AME | Average over all observations |
| `"means"` or `"mem"` | MEM | Evaluate at sample means |

### PPML Elasticities

PPML provides specialized elasticity methods for gravity models:

```python
from panelbox.models.count import PPML

model = PPML(
    endog=df["trade_flow"],
    exog=df[["log_distance", "log_gdp_exp", "log_gdp_imp", "rta"]],
    entity_id=df["pair_id"],
    time_id=df["year"],
    fixed_effects=True,
    exog_names=["log_distance", "log_gdp_exp", "log_gdp_imp", "rta"]
)
results = model.fit()

# Elasticity for a specific variable
dist_elast = results.elasticity("log_distance")
print(f"Distance elasticity: {dist_elast['elasticity']:.3f}")
print(f"SE: {dist_elast['elasticity_se']:.3f}")

# All elasticities as a DataFrame
print(results.elasticities())
```

**For log-transformed variables**: the coefficient is the elasticity directly:

$$\frac{\partial \ln E[y]}{\partial \ln x} = \beta$$

**For level variables**: the coefficient is a semi-elasticity. The percentage effect is:

$$\text{Percentage change} = 100 \times (\exp(\beta) - 1)\%$$

```python
# Binary variable interpretation (e.g., RTA)
rta_coef = results.params[3]  # assuming RTA is 4th variable
pct_effect = 100 * (np.exp(rta_coef) - 1)
print(f"RTA increases trade by {pct_effect:.1f}%")
```

### Zero-Inflated Marginal Effects

For ZI models, the overall marginal effect combines contributions from both the inflation and count components. The expected value is:

$$E[y \mid X, Z] = (1 - \pi) \cdot \lambda$$

where $\pi = \Lambda(Z'\gamma)$ and $\lambda = \exp(X'\beta)$. If a variable $x_k$ appears in both parts, the total marginal effect is:

$$\frac{\partial E[y]}{\partial x_k} = \underbrace{-\frac{\partial \pi}{\partial x_k} \cdot \lambda}_{\text{inflation effect}} + \underbrace{(1 - \pi) \cdot \beta_k \lambda}_{\text{count effect}}$$

The first term captures how changes in $x_k$ affect the probability of being a structural zero. The second term captures how changes in $x_k$ affect the count among non-structural-zero observations.

```python
from panelbox.models.count import ZeroInflatedPoisson

model = ZeroInflatedPoisson(
    endog=data["patents"],
    exog_count=data[["rd_spending", "employees"]],
    exog_inflate=data[["small_firm", "new_entrant"]]
)
results = model.fit()

# Predictions for decomposition
overall_mean = results.predict(which="mean")
count_mean = results.predict(which="count-mean")
pi_hat = results.predict(which="prob-zero-structural")

# Manual AME for count variable (rd_spending)
beta_rd = results.params_count[0]
ame_rd = np.mean((1 - pi_hat) * beta_rd * count_mean)
print(f"AME of rd_spending: {ame_rd:.4f}")

# Manual AME for inflation variable (small_firm)
gamma_small = results.params_inflate[0]
# Logit derivative: pi * (1-pi) * gamma
ame_inflate = np.mean(-pi_hat * (1 - pi_hat) * gamma_small * count_mean)
print(f"AME of small_firm (via inflation): {ame_inflate:.4f}")
```

## Comparison Across Models

The table below shows how marginal effects differ across count model families:

| Model | ME Formula | Notes |
|-------|------------|-------|
| **Poisson** | $\beta_k \cdot \exp(X'\beta)$ | Simple exponential |
| **NB** | $\beta_k \cdot \exp(X'\beta)$ | Same formula, different estimates |
| **PPML** | Same as Poisson | Use `elasticity()` for trade variables |
| **ZIP** | $(1-\pi) \beta_k \lambda - \frac{\partial\pi}{\partial x_k} \lambda$ | Two-component decomposition |
| **ZINB** | Same as ZIP | With NB count component |

## Reporting Best Practices

!!! tip "What to Report"
    1. **Always report IRR** ($\exp(\beta)$) alongside raw coefficients for accessibility
    2. **AME** for policy analysis (in natural units: "1 more year of education increases expected patents by 0.3")
    3. **Elasticities** for PPML / gravity models ("1% increase in GDP increases trade by 0.8%")
    4. **Both parts** for ZI models (inflation effects and count effects separately)

```python
import pandas as pd

# Create a comprehensive results table
table = pd.DataFrame({
    "Variable": results.exog_names,
    "Coefficient": results.params,
    "SE": results.se,
    "IRR": np.exp(results.params),
    "pct_change": 100 * (np.exp(results.params) - 1),
})
print(table.to_string(index=False, float_format="%.4f"))
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Count Data Models | Marginal effects across all count specifications | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/06_marginal_effects_count.ipynb) |

## See Also

- [Poisson Models](poisson.md) --- Coefficient interpretation as semi-elasticities
- [PPML](ppml.md) --- Elasticity computation for gravity models
- [Negative Binomial](negative-binomial.md) --- IRR with overdispersion
- [Zero-Inflated Models](zero-inflated.md) --- Two-component marginal effects

## References

- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press, Chapter 2.6.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press, Chapter 18.
- Long, J. S., & Freese, J. (2014). *Regression Models for Categorical Dependent Variables Using Stata* (3rd ed.). Stata Press.
- Santos Silva, J. M. C., & Tenreyro, S. (2006). The Log of Gravity. *Review of Economics and Statistics*, 88(4), 641--658.
