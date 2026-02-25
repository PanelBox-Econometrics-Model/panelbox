---
title: "Cox Test & Encompassing Tests"
description: "Cox test, Wald encompassing test, and likelihood ratio test for model comparison in PanelBox."
---

# Cox Test & Encompassing Tests

!!! info "Quick Reference"
    **Functions:**
    `panelbox.diagnostics.specification.encompassing.cox_test()`
    `panelbox.diagnostics.specification.encompassing.wald_encompassing_test()`
    `panelbox.diagnostics.specification.encompassing.likelihood_ratio_test()`
    **Result:** `EncompassingResult`
    **H₀ (Cox):** Both models fit equally well
    **H₀ (Wald/LR):** Restricted model is adequate (restrictions valid)
    **Stata equivalent:** `lrtest` (for LR test)
    **R equivalent:** `lmtest::encomptest()`, `lmtest::lrtest()`

## What They Test

This module provides three tests for comparing model adequacy:

| Test | Models | Compares | Based On |
|------|--------|----------|----------|
| **Cox test** | Non-nested | Whether models fit equally well | Log-likelihood difference |
| **Wald encompassing** | Nested | Whether additional parameters are needed | RSS or likelihood |
| **Likelihood ratio** | Nested | Whether restrictions are valid | Log-likelihood ratio |

### The Encompassing Principle

A model **encompasses** another if it can explain everything the other model explains plus more. If Model 1 encompasses Model 2, there is no reason to prefer Model 2.

## Cox Test

### What It Tests

The Cox test (1961) compares two **non-nested** models using their log-likelihoods. Unlike the [J-test](j-test.md), which uses fitted values in augmented regressions, the Cox test directly compares how well each model predicts the data in a likelihood framework.

### Quick Example

```python
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.diagnostics.specification.encompassing import cox_test
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Two competing specifications
model1 = PooledOLS("invest ~ value", data, "firm", "year")
result1 = model1.fit()

model2 = PooledOLS("invest ~ capital", data, "firm", "year")
result2 = model2.fit()

# Cox test
cox_result = cox_test(
    result1, result2,
    model1_name="Value Model",
    model2_name="Capital Model"
)

print(cox_result.interpretation())
```

### Mathematical Details

The Cox test statistic is:

$$T_{Cox} = \frac{\ell_1 - \ell_2}{\text{SE}(\ell_1 - \ell_2)}$$

where $\ell_1$ and $\ell_2$ are the maximized log-likelihoods of Models 1 and 2.

Under H₀ (both models fit equally well), $T_{Cox} \sim N(0, 1)$. A two-tailed p-value is computed.

The standard error is estimated from the variance of the log-likelihood difference, using the squared residual ratio:

$$\text{Var} = \text{Var}\left[\log(e_{1t}^2) - \log(e_{2t}^2)\right] / n$$

### Interpretation

| P-value | Decision | Interpretation |
|---------|----------|----------------|
| p < 0.05 | Reject H₀ | Models fit significantly differently |
| p $\geq$ 0.05 | Fail to reject | No significant difference in fit |

When rejected, the sign of the statistic indicates which model fits better:

- $T_{Cox} > 0$: Model 1 has higher log-likelihood (better fit)
- $T_{Cox} < 0$: Model 2 has higher log-likelihood (better fit)

### Parameters

```python
cox_test(result1, result2, model1_name=None, model2_name=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result1` | fitted result | required | First model (must have `.llf`, `.resid`) |
| `result2` | fitted result | required | Second model |
| `model1_name` | `str` or `None` | `"Model 1"` | Display name for Model 1 |
| `model2_name` | `str` or `None` | `"Model 2"` | Display name for Model 2 |

### Cox Test Metadata

| Key | Type | Description |
|-----|------|-------------|
| `llf1` | `float` | Log-likelihood of Model 1 |
| `llf2` | `float` | Log-likelihood of Model 2 |
| `llf_diff` | `float` | Difference in log-likelihoods |
| `se_diff` | `float` | Standard error of the difference |

## Wald Encompassing Test

### What It Tests

The Wald encompassing test compares **nested** models by testing whether the additional parameters in the unrestricted model are jointly significant. It answers: does the fuller model significantly improve upon the restricted one?

### Quick Example

```python
from panelbox.diagnostics.specification.encompassing import wald_encompassing_test

# Restricted model: fewer variables
model_r = PooledOLS("invest ~ value", data, "firm", "year")
result_r = model_r.fit()

# Unrestricted model: more variables
model_u = PooledOLS("invest ~ value + capital", data, "firm", "year")
result_u = model_u.fit()

# Wald encompassing test
wald_result = wald_encompassing_test(
    result_r, result_u,
    model_restricted_name="Value Only",
    model_unrestricted_name="Value + Capital"
)

print(wald_result.interpretation())
```

### Mathematical Details

The Wald statistic is based on the improvement in residual sum of squares:

$$W = \frac{(RSS_r - RSS_u) / \hat{\sigma}_u^2}{1}$$

where $\hat{\sigma}_u^2 = RSS_u / (n - k_u)$ is the unrestricted variance estimate.

Under H₀ (restrictions valid), $W \sim \chi^2(q)$ where $q = k_u - k_r$ is the number of restrictions.

When RSS is not available but log-likelihoods are, the test uses the likelihood-ratio approach: $W = 2(\ell_u - \ell_r)$.

### Parameters

```python
wald_encompassing_test(
    result_restricted, result_unrestricted,
    model_restricted_name=None, model_unrestricted_name=None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result_restricted` | fitted result | required | Restricted (nested) model |
| `result_unrestricted` | fitted result | required | Unrestricted (full) model |
| `model_restricted_name` | `str` or `None` | `"Restricted Model"` | Display name |
| `model_unrestricted_name` | `str` or `None` | `"Unrestricted Model"` | Display name |

!!! note "Model Ordering"
    The unrestricted model **must** have more parameters than the restricted model. PanelBox raises a `ValueError` if this condition is not met.

## Likelihood Ratio Test

### What It Tests

The likelihood ratio (LR) test compares **nested** models estimated by maximum likelihood. It tests whether the restricted model's constraints are supported by the data.

### Quick Example

```python
from panelbox.diagnostics.specification.encompassing import likelihood_ratio_test

# Restricted and unrestricted models (must have .llf attribute)
lr_result = likelihood_ratio_test(
    result_r, result_u,
    model_restricted_name="Base Model",
    model_unrestricted_name="Full Model"
)

print(lr_result.interpretation())
print(f"LR statistic: {lr_result.statistic:.4f}")
print(f"P-value: {lr_result.pvalue:.4f}")
print(f"Degrees of freedom: {lr_result.df}")
```

### Mathematical Details

The LR statistic is:

$$LR = -2(\ell_r - \ell_u) = 2(\ell_u - \ell_r)$$

Under H₀ (restrictions valid), $LR \sim \chi^2(q)$ where $q$ is the number of restrictions (difference in parameter counts).

### Parameters

```python
likelihood_ratio_test(
    result_restricted, result_unrestricted,
    model_restricted_name=None, model_unrestricted_name=None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result_restricted` | fitted result | required | Restricted model (must have `.llf`) |
| `result_unrestricted` | fitted result | required | Unrestricted model (must have `.llf`) |
| `model_restricted_name` | `str` or `None` | `"Restricted Model"` | Display name |
| `model_unrestricted_name` | `str` or `None` | `"Unrestricted Model"` | Display name |

### LR Test Metadata

| Key | Type | Description |
|-----|------|-------------|
| `llf_restricted` | `float` | Log-likelihood of restricted model |
| `llf_unrestricted` | `float` | Log-likelihood of unrestricted model |
| `llf_diff` | `float` | Difference ($\ell_u - \ell_r$) |
| `k_restricted` | `int` | Parameters in restricted model |
| `k_unrestricted` | `int` | Parameters in unrestricted model |
| `num_restrictions` | `int` | Number of restrictions ($q$) |

## EncompassingResult Interface

All three tests return `EncompassingResult` objects:

```python
result.test_name          # str   -- Name of the test
result.statistic          # float -- Test statistic
result.pvalue             # float -- P-value
result.df                 # float or None -- Degrees of freedom
result.null_hypothesis    # str   -- H₀ description
result.alternative        # str   -- H₁ description
result.model1_name        # str   -- Name of first model
result.model2_name        # str   -- Name of second model
result.additional_info    # dict  -- Test-specific metadata

# Methods
result.interpretation()   # str   -- Human-readable interpretation
result.summary()          # pd.DataFrame -- Summary table
```

## Choosing the Right Test

| Situation | Test | Reason |
|-----------|------|--------|
| Non-nested models, have residuals | [J-Test](j-test.md) | Most common for regression models |
| Non-nested models, have log-likelihoods | **Cox test** | Likelihood-based comparison |
| Nested models, OLS | **Wald encompassing** or F-test | Tests joint significance of extra variables |
| Nested models, MLE | **Likelihood ratio** | Standard for likelihood-based models |

## Complete Example: Comprehensive Model Comparison

```python
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.diagnostics.specification.davidson_mackinnon import j_test
from panelbox.diagnostics.specification.encompassing import (
    cox_test, wald_encompassing_test, likelihood_ratio_test
)
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Three specifications
model_a = PooledOLS("invest ~ value", data, "firm", "year")
result_a = model_a.fit()

model_b = PooledOLS("invest ~ capital", data, "firm", "year")
result_b = model_b.fit()

model_c = PooledOLS("invest ~ value + capital", data, "firm", "year")
result_c = model_c.fit()

# --- Non-nested comparison: Model A vs Model B ---
print("=== Non-Nested: Value vs Capital ===")

# J-test
jtest = j_test(result_a, result_b,
               model1_name="Value", model2_name="Capital")
print(jtest.interpretation())

# Cox test
cox = cox_test(result_a, result_b,
               model1_name="Value", model2_name="Capital")
print(cox.interpretation())

# --- Nested comparison: Model A vs Model C ---
print("\n=== Nested: Value vs Value+Capital ===")

# Wald encompassing
wald = wald_encompassing_test(
    result_a, result_c,
    model_restricted_name="Value Only",
    model_unrestricted_name="Value + Capital"
)
print(wald.interpretation())

# Likelihood ratio
lr = likelihood_ratio_test(
    result_a, result_c,
    model_restricted_name="Value Only",
    model_unrestricted_name="Value + Capital"
)
print(lr.interpretation())
```

## Common Pitfalls

### Cox Test Requires Log-Likelihood

The Cox test requires `.llf` (log-likelihood) on both model results. If your model does not compute a log-likelihood (e.g., some robust estimators), use the [J-test](j-test.md) instead.

### LR Test Requires Nested Models

The likelihood ratio test is only valid for **nested** models estimated by maximum likelihood on the **same sample**. Using it with non-nested models or models estimated on different data produces meaningless results.

### Unrestricted Must Have More Parameters

For Wald and LR tests, the unrestricted model must have strictly more parameters. If the models have the same number of parameters, they are not nested and you should use the Cox or J-test instead.

### Multiple Testing

When comparing multiple model pairs, p-values should be adjusted for multiple comparisons (e.g., Bonferroni correction). Running many pairwise tests without correction inflates the overall false positive rate.

### Difference from J-Test

The Cox test and J-test both handle non-nested models but use different approaches:

| Feature | J-Test | Cox Test |
|---------|--------|----------|
| Approach | Augmented regressions | Log-likelihood comparison |
| Statistic | t-statistic | z-statistic (Normal) |
| Two-directional | Yes (forward + reverse) | One comparison |
| Requires | Fitted values + regressors | Log-likelihood |
| Interpretation | Which model encompasses | Which model fits better |

## See Also

- [J-Test](j-test.md) -- Regression-based non-nested model comparison
- [RESET Test](reset.md) -- Functional form specification
- [Specification Tests Overview](index.md) -- All specification tests
- [Diagnostics Overview](../index.md) -- Complete diagnostic workflow

## References

- Cox, D. R. (1961). Tests of separate families of hypotheses. *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 105-123.
- Mizon, G. E., & Richard, J. F. (1986). The encompassing principle and its application to testing non-nested hypotheses. *Econometrica*, 54(3), 657-678.
- Vuong, Q. H. (1989). Likelihood ratio tests for model selection and non-nested hypotheses. *Econometrica*, 57(2), 307-333.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapter 18.
- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press. Chapter 8.
