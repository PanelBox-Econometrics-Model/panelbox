---
title: "J-Test (Davidson-MacKinnon)"
description: "Davidson-MacKinnon J-test for comparing non-nested panel data models."
---

# J-Test (Davidson-MacKinnon)

!!! info "Quick Reference"
    **Function:** `panelbox.diagnostics.specification.davidson_mackinnon.j_test()`
    **Result:** `JTestResult`
    **H₀ (forward):** Model 1 is correctly specified
    **H₁ (forward):** Model 2's fitted values add explanatory power to Model 1
    **Stata equivalent:** Manual (augmented regressions)
    **R equivalent:** `lmtest::jtest()`

## What It Tests

The J-test (Davidson & MacKinnon, 1981) compares two **non-nested** models -- models where neither is a special case of the other. Standard F-tests and likelihood ratio tests cannot handle this situation because there is no natural restricted/unrestricted nesting.

The J-test resolves this by asking: **do the fitted values from one model have explanatory power when added to the other?**

### Non-Nested Models

Two models are non-nested when you cannot obtain one from the other by imposing parameter restrictions:

- **Model 1**: $y = X_1'\beta + \varepsilon_1$ (e.g., labor + capital)
- **Model 2**: $y = X_2'\gamma + \varepsilon_2$ (e.g., R&D + market share)

These models use entirely different regressors, so an F-test comparing them is not defined.

## Quick Example

```python
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.diagnostics.specification.davidson_mackinnon import j_test
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Two competing specifications
model1 = PooledOLS("invest ~ value", data, "firm", "year")
result1 = model1.fit()

model2 = PooledOLS("invest ~ capital", data, "firm", "year")
result2 = model2.fit()

# Run J-test (both directions)
jtest = j_test(
    result1, result2,
    direction="both",
    model1_name="Value Model",
    model2_name="Capital Model"
)

# Full interpretation
print(jtest.interpretation())

# Summary table
print(jtest.summary())
```

## Interpretation

The J-test is performed in two directions, and the combination of results determines the conclusion:

| Forward (H₀: Model 1) | Reverse (H₀: Model 2) | Conclusion |
|------------------------|------------------------|------------|
| Reject | Fail to reject | **Prefer Model 2** -- Model 2 encompasses Model 1 |
| Fail to reject | Reject | **Prefer Model 1** -- Model 1 encompasses Model 2 |
| Fail to reject | Fail to reject | **Both acceptable** -- cannot discriminate |
| Reject | Reject | **Neither adequate** -- both models miss important features |

!!! tip "Reading the Results"
    - **Forward test**: adds $\hat{y}_2$ (from Model 2) to Model 1. Rejection means Model 2 captures something Model 1 misses.
    - **Reverse test**: adds $\hat{y}_1$ (from Model 1) to Model 2. Rejection means Model 1 captures something Model 2 misses.

### When Both Models Are Rejected

This is the most informative outcome: neither specification adequately describes the data. Consider:

- A model that combines elements of both specifications
- Additional regressors or different functional forms
- A more general nesting model

### When Neither Is Rejected

The data cannot discriminate between the models. Use other criteria:

- Economic theory and plausibility
- Information criteria (AIC, BIC)
- Out-of-sample prediction performance
- Parsimony (prefer simpler models)

## Mathematical Details

### Forward Test

1. Estimate Model 2 to get fitted values $\hat{y}_2 = X_2\hat{\gamma}$
2. Estimate the augmented Model 1: $y = X_1'\beta + \alpha \hat{y}_2 + u$
3. Test $H_0: \alpha = 0$ using a t-test

Under $H_0$ (Model 1 is correct), $\hat{y}_2$ should add no information beyond $X_1$, so $\alpha = 0$. Under $H_1$, Model 2 captures some aspect of the DGP that Model 1 misses, so $\alpha \neq 0$.

### Reverse Test

The same procedure in the opposite direction:

1. Estimate Model 1 to get $\hat{y}_1 = X_1\hat{\beta}$
2. Estimate augmented Model 2: $y = X_2'\gamma + \delta \hat{y}_1 + u$
3. Test $H_0: \delta = 0$

### Test Statistic

The test statistic is a standard t-statistic on the coefficient of the added fitted values:

$$t = \frac{\hat{\alpha}}{\text{SE}(\hat{\alpha})}$$

For panel data, PanelBox uses heteroskedasticity-robust (HC1) or cluster-robust standard errors when entity information is available.

### Asymptotic Properties

- Under H₀, $t \xrightarrow{d} N(0, 1)$ as $N \to \infty$
- The test is consistent: power approaches 1 as the sample grows
- Cluster-robust inference ensures validity with panel structure

## Configuration Options

```python
from panelbox.diagnostics.specification.davidson_mackinnon import j_test

# Both directions (default)
result = j_test(result1, result2, direction="both")

# Forward only: does Model 2 improve Model 1?
result = j_test(result1, result2, direction="forward")

# Reverse only: does Model 1 improve Model 2?
result = j_test(result1, result2, direction="reverse")

# Custom model names for clearer output
result = j_test(
    result1, result2,
    direction="both",
    model1_name="Cobb-Douglas",
    model2_name="Translog"
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result1` | fitted result | required | First model's results (must have `.fittedvalues` and `.model`) |
| `result2` | fitted result | required | Second model's results |
| `direction` | `str` | `"both"` | `"forward"`, `"reverse"`, or `"both"` |
| `model1_name` | `str` or `None` | `None` | Display name for Model 1 (default: "Model 1") |
| `model2_name` | `str` or `None` | `None` | Display name for Model 2 (default: "Model 2") |

### JTestResult Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.forward` | `dict` or `None` | Forward test: `statistic`, `pvalue`, `alpha_coef`, `alpha_se` |
| `.reverse` | `dict` or `None` | Reverse test: `statistic`, `pvalue`, `gamma_coef`, `gamma_se` |
| `.model1_name` | `str` | Name of Model 1 |
| `.model2_name` | `str` | Name of Model 2 |
| `.direction` | `str` | Direction tested |

### JTestResult Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `.interpretation()` | `str` | Human-readable interpretation of results |
| `.summary()` | `pd.DataFrame` | Summary table with statistics and p-values |

## Complete Example: Comparing Production Functions

```python
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.diagnostics.specification.davidson_mackinnon import j_test
from panelbox.datasets import load_grunfeld
import numpy as np

data = load_grunfeld()

# Model 1: value-driven investment
model1 = PooledOLS("invest ~ value", data, "firm", "year")
result1 = model1.fit()

# Model 2: capital-driven investment
model2 = PooledOLS("invest ~ capital", data, "firm", "year")
result2 = model2.fit()

# J-test
jtest = j_test(
    result1, result2,
    direction="both",
    model1_name="Value Model",
    model2_name="Capital Model"
)

# Detailed results
print(jtest.interpretation())

# Individual test statistics
if jtest.forward:
    print(f"\nForward test (H0: Value Model is correct):")
    print(f"  t-stat = {jtest.forward['statistic']:.4f}")
    print(f"  p-value = {jtest.forward['pvalue']:.4f}")
    print(f"  alpha = {jtest.forward['alpha_coef']:.4f} "
          f"(SE = {jtest.forward['alpha_se']:.4f})")

if jtest.reverse:
    print(f"\nReverse test (H0: Capital Model is correct):")
    print(f"  t-stat = {jtest.reverse['statistic']:.4f}")
    print(f"  p-value = {jtest.reverse['pvalue']:.4f}")
    print(f"  gamma = {jtest.reverse['gamma_coef']:.4f} "
          f"(SE = {jtest.reverse['gamma_se']:.4f})")
```

## Common Pitfalls

### Same Dependent Variable Required

Both models must have the **same dependent variable**. The J-test compares how well different regressors explain the same outcome. PanelBox issues a warning if the dependent variables appear to differ.

### Same Sample Required

Both models must be estimated on the **same observations**. Different samples invalidate the test because fitted values would not be comparable. PanelBox raises a `ValueError` if sample sizes differ.

### Ambiguous Results

The J-test can produce inconclusive outcomes:

- **Both reject**: The data rejects both specifications. This is frustrating but informative -- it tells you to look for a better model.
- **Neither reject**: Low power or genuinely equivalent models. The test provides no guidance; use economic theory or other criteria.

### Multicollinearity with Fitted Values

If one model's fitted values are highly correlated with the other model's regressors, the augmented regression may suffer from multicollinearity, reducing test power. This is more common when models share some regressors.

### Not for Nested Models

If models are nested (one is a special case of the other), use an F-test or likelihood ratio test instead. The J-test is designed for genuinely non-nested alternatives.

## See Also

- [Cox & Encompassing Tests](cox-encompassing.md) -- Likelihood-based model comparison
- [RESET Test](reset.md) -- Functional form specification
- [Specification Tests Overview](index.md) -- All specification tests
- [Diagnostics Overview](../index.md) -- Complete diagnostic workflow

## References

- Davidson, R., & MacKinnon, J. G. (1981). Several tests for model specification in the presence of alternative hypotheses. *Econometrica*, 49(3), 781-793.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapter 18.
- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press. Chapter 8.
- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson. Chapter 5.4.
