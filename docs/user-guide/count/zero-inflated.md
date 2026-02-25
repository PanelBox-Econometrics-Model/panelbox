---
title: "Zero-Inflated Models"
description: "Guide to Zero-Inflated Poisson (ZIP) and Zero-Inflated Negative Binomial (ZINB) models in PanelBox for count data with excess zeros."
---

# Zero-Inflated Models

!!! info "Quick Reference"
    **Classes:** `ZeroInflatedPoisson`, `ZeroInflatedNegativeBinomial`
    **Import:** `from panelbox.models.count import ZeroInflatedPoisson, ZeroInflatedNegativeBinomial`
    **Stata equivalent:** `zip`, `zinb`
    **R equivalent:** `pscl::zeroinfl()`

## Overview

Zero-inflated (ZI) models address count data with **excess zeros** --- more zeros than standard Poisson or Negative Binomial distributions can explain. The key insight is that zeros can arise from two distinct processes:

1. **Structural zeros**: the event can never occur for some observations (e.g., a firm without an R&D department will never file patents)
2. **Sampling zeros**: the event could occur but did not in the observation period (e.g., an R&D-active firm that happened to file zero patents this year)

ZI models combine two components in a **mixture model**:

$$P(y_{it} = 0) = \pi_{it} + (1 - \pi_{it}) \cdot f(0; X_{it}'\beta)$$

$$P(y_{it} = k) = (1 - \pi_{it}) \cdot f(k; X_{it}'\beta), \quad k = 1, 2, \ldots$$

where $\pi_{it} = \Lambda(Z_{it}'\gamma)$ is the probability of a structural zero (modeled by a logit), and $f(\cdot)$ is the count distribution (Poisson or NB).

## Quick Example

```python
from panelbox.models.count import ZeroInflatedPoisson

model = ZeroInflatedPoisson(
    endog=data["patents"],
    exog_count=data[["rd_spending", "employees", "capital"]],
    exog_inflate=data[["small_firm", "new_entrant"]]
)
results = model.fit()
print(results.summary())

# Zero proportions
print(f"Actual zeros:    {results.actual_zeros:.1%}")
print(f"Predicted zeros: {results.predicted_zeros:.1%}")

# Vuong test: ZIP vs standard Poisson
print(f"Vuong statistic: {results.vuong_stat:.2f} (p = {results.vuong_pvalue:.4f})")
```

## When to Use

- Count data with more zeros than Poisson/NB can explain
- Healthcare utilization: many people never visit the doctor (structural zeros)
- Patent counts: some firms lack R&D capacity (never-patenters)
- Insurance claims: some policies cover non-claimable events
- When the zero-generating process differs from the count process

!!! warning "Key Assumptions"
    - **Two-process model**: zeros arise from both a structural and a sampling process
    - **Logit inflation**: $P(\text{structural zero}) = \Lambda(Z'\gamma)$
    - **Count distribution**: Poisson (ZIP) or Negative Binomial (ZINB)
    - **Correct specification of both parts**: wrong regressors in either part biases both

## Detailed Guide

### The Excess Zeros Problem

Standard count models often underpredict the number of zeros:

```python
import numpy as np
from scipy.stats import poisson

# Check if Poisson predicts too few zeros
y = data["patents"].values
mu_hat = y.mean()
expected_zeros_poisson = len(y) * poisson.pmf(0, mu_hat)
actual_zeros = (y == 0).sum()

print(f"Actual zeros:           {actual_zeros}")
print(f"Poisson-expected zeros: {expected_zeros_poisson:.0f}")
print(f"Excess zeros:           {actual_zeros - expected_zeros_poisson:.0f}")
```

If actual zeros greatly exceed Poisson-expected zeros, a ZI model may be appropriate.

### Two-Part Model Structure

The ZI model splits the data-generating process:

**Part 1 --- Inflation Model (Logit)**:

$$\pi_{it} = P(\text{structural zero}_i) = \frac{\exp(Z_{it}'\gamma)}{1 + \exp(Z_{it}'\gamma)}$$

This determines *who* generates structural zeros. The regressors $Z$ can differ from $X$.

**Part 2 --- Count Model (Poisson or NB)**:

$$P(y_{it} = k \mid \text{not structural zero}) = f(k; \mu_{it})$$

where $\mu_{it} = \exp(X_{it}'\beta)$. This determines the *intensity* of events among potential generators.

**Combined likelihood**:

$$\ell = \sum_{y_i = 0} \ln[\pi_i + (1-\pi_i) f(0)] + \sum_{y_i > 0} \ln[(1-\pi_i) f(y_i)]$$

### Estimation

#### Zero-Inflated Poisson (ZIP)

```python
from panelbox.models.count import ZeroInflatedPoisson

model = ZeroInflatedPoisson(
    endog=data["patents"],
    exog_count=data[["rd_spending", "employees", "capital"]],
    exog_inflate=data[["small_firm", "new_entrant"]],
    exog_count_names=["rd_spending", "employees", "capital"],
    exog_inflate_names=["small_firm", "new_entrant"]
)
results = model.fit(method="BFGS", maxiter=1000)
```

=== "Different Regressors"

    The count and inflation components can use different regressors. This is conceptually motivated: variables that determine *whether* an event can occur may differ from those that determine *how many* events occur.

    ```python
    model = ZeroInflatedPoisson(
        endog=data["patents"],
        exog_count=data[["rd_spending", "employees", "capital"]],
        exog_inflate=data[["small_firm", "new_entrant", "no_rd_dept"]]
    )
    ```

=== "Same Regressors"

    If `exog_inflate` is not specified, it defaults to the same regressors as the count model:

    ```python
    model = ZeroInflatedPoisson(
        endog=data["patents"],
        exog_count=data[["rd_spending", "employees", "capital"]]
        # exog_inflate defaults to exog_count
    )
    ```

#### Zero-Inflated Negative Binomial (ZINB)

ZINB adds an overdispersion parameter $\alpha$ to the count component, handling both excess zeros and overdispersion:

```python
from panelbox.models.count import ZeroInflatedNegativeBinomial

model = ZeroInflatedNegativeBinomial(
    endog=data["claims"],
    exog_count=data[["age", "income", "risk_score"]],
    exog_inflate=data[["new_policy", "low_coverage"]],
    exog_count_names=["age", "income", "risk_score"],
    exog_inflate_names=["new_policy", "low_coverage"]
)
results = model.fit(method="L-BFGS-B", maxiter=1000)

# Overdispersion parameter
print(f"alpha = {results.alpha:.4f}")
```

!!! tip "ZIP vs ZINB"
    Use **ZIP** when the count process follows Poisson (equidispersion among non-structural zeros). Use **ZINB** when there is both excess zeros *and* overdispersion in the count process. ZINB is more flexible but requires estimating one additional parameter.

### Interpreting Results

#### Parameter Estimates

Results contain separate coefficients for each component:

```python
# Count model coefficients (beta)
print("Count model:")
for name, coef, se in zip(
    results.exog_count_names, results.params_count, results.bse_count
):
    print(f"  {name}: {coef:.4f} (SE = {se:.4f})")

# Inflation model coefficients (gamma)
print("\nInflation model:")
for name, coef, se in zip(
    results.exog_inflate_names, results.params_inflate, results.bse_inflate
):
    print(f"  {name}: {coef:.4f} (SE = {se:.4f})")
```

**Count coefficients** ($\beta$): semi-elasticities of the expected count *among potential generators*.

**Inflation coefficients** ($\gamma$): log-odds of being a structural zero. Positive $\gamma$ means higher probability of being a "never-taker."

#### Predictions

ZI models support multiple prediction types:

```python
# Overall expected count: E[y] = (1-pi) * lambda
y_hat = results.predict(which="mean")

# Probability of zero (total)
p_zero = results.predict(which="prob-zero")

# Structural zero probability (pi)
p_structural = results.predict(which="prob-zero-structural")

# Sampling zero probability: (1-pi) * f(0)
p_sampling = results.predict(which="prob-zero-sampling")

# Count mean among potential generators (lambda)
count_mean = results.predict(which="count-mean")
```

### Model Selection: Vuong Test

The Vuong test (1989) compares the ZI model against its non-inflated counterpart:

- $H_0$: Standard model and ZI model are equivalent
- $H_1$: ZI model fits better (Vuong > 1.96) or standard model fits better (Vuong < -1.96)

```python
# ZIP results include automatic Vuong test
print(f"Vuong statistic: {results.vuong_stat:.2f}")
print(f"Vuong p-value:   {results.vuong_pvalue:.4f}")

if results.vuong_stat > 1.96:
    print("ZIP preferred over Poisson")
elif results.vuong_stat < -1.96:
    print("Poisson preferred over ZIP")
else:
    print("No significant difference")
```

### Choosing Between ZIP and ZINB

| Feature | ZIP | ZINB |
|---------|-----|------|
| Excess zeros | Yes | Yes |
| Overdispersion | No | Yes |
| Parameters | $\beta, \gamma$ | $\beta, \gamma, \alpha$ |
| Count process | Poisson | Negative Binomial |
| Use when | Equidispersion among "users" | Overdispersion among "users" |

## Configuration Options

### ZeroInflatedPoisson

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | array-like | *required* | Dependent variable (non-negative integers) |
| `exog_count` | array-like | *required* | Regressors for count process |
| `exog_inflate` | array-like | `None` | Regressors for inflation process (defaults to `exog_count`) |
| `exog_count_names` | list | `None` | Variable names for count model |
| `exog_inflate_names` | list | `None` | Variable names for inflation model |

### ZeroInflatedNegativeBinomial

Same parameters as ZIP. The `fit()` method defaults to `method="L-BFGS-B"` (with bounds) for numerical stability.

### fit() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_params` | array | `None` | Starting values (auto-generated if `None`) |
| `method` | str | `"BFGS"` / `"L-BFGS-B"` | Optimization method |
| `maxiter` | int | `1000` | Maximum iterations |

### Result Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | array | Full parameter vector $[\beta, \gamma]$ or $[\beta, \gamma, \ln\alpha]$ |
| `params_count` | array | Count model coefficients ($\beta$) |
| `params_inflate` | array | Inflation model coefficients ($\gamma$) |
| `bse` | array | Standard errors (all parameters) |
| `bse_count` | array | Standard errors for count model |
| `bse_inflate` | array | Standard errors for inflation model |
| `llf` | float | Log-likelihood |
| `aic` | float | Akaike Information Criterion |
| `bic` | float | Bayesian Information Criterion |
| `actual_zeros` | float | Proportion of actual zeros |
| `predicted_zeros` | float | Proportion of predicted zeros |
| `vuong_stat` | float | Vuong test statistic (ZIP only) |
| `vuong_pvalue` | float | Vuong test p-value (ZIP only) |
| `alpha` | float | Overdispersion parameter (ZINB only) |

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Count Data Models | Complete guide including ZIP and ZINB | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/05_zero_inflated.ipynb) |

## See Also

- [Count Data Overview](index.md) --- Introduction and model selection guide
- [Poisson Models](poisson.md) --- Baseline count models (no excess zeros)
- [Negative Binomial](negative-binomial.md) --- Overdispersion without excess zeros
- [Marginal Effects for Count Data](marginal-effects.md) --- ZI marginal effects decomposition

## References

- Lambert, D. (1992). Zero-Inflated Poisson Regression, with an Application to Defects in Manufacturing. *Technometrics*, 34(1), 1--14.
- Vuong, Q. H. (1989). Likelihood Ratio Tests for Model Selection and Non-Nested Hypotheses. *Econometrica*, 57(2), 307--333.
- Hall, D. B. (2000). Zero-Inflated Poisson and Binomial Regression with Random Effects: A Case Study. *Biometrics*, 56(4), 1030--1039.
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press, Chapter 4.
- Greene, W. H. (1994). Accounting for Excess Zeros and Sample Selection in Poisson and Negative Binomial Regression Models. Working Paper EC-94-10, NYU.
