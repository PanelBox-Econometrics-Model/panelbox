---
title: "RESET Test"
description: "Ramsey RESET test for functional form misspecification in panel data models."
---

# RESET Test

!!! info "Quick Reference"
    **Class:** `panelbox.validation.specification.reset.RESETTest`
    **Result:** `ValidationTestResult`
    **H₀:** Model is correctly specified (linear functional form is appropriate)
    **H₁:** Nonlinear terms are needed (functional form misspecification)
    **Stata equivalent:** `estat ovtest`
    **R equivalent:** `lmtest::resettest()`

## What It Tests

The Ramsey (1969) Regression Equation Specification Error Test (RESET) checks whether the functional form of a linear model is correct. It detects omitted nonlinearities by testing whether powers of the fitted values have explanatory power beyond the original regressors.

If the true relationship is nonlinear -- for example, quadratic, logarithmic, or involving interactions -- the fitted values from the linear model will partially capture the nonlinearity through their powers. A significant test indicates the linear specification is inadequate.

!!! warning "What RESET Does Not Tell You"
    RESET detects that a nonlinearity exists but does **not** identify which specific nonlinearity is missing. Rejection tells you the model is wrong, not how to fix it.

## Quick Example

```python
from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.validation.specification.reset import RESETTest
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Estimate model
fe = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = fe.fit()

# Run RESET test with default powers [2, 3]
reset = RESETTest(results)
result = reset.run(alpha=0.05)
print(result.summary())

# Access results
print(f"F-statistic: {result.statistic:.4f}")
print(f"P-value: {result.pvalue:.4f}")
print(f"Degrees of freedom: {result.df}")
```

## Interpretation

| P-value | Decision | Interpretation | Action |
|---------|----------|----------------|--------|
| p < 0.01 | Strong rejection | Strong evidence of misspecification | Transform variables or add nonlinear terms |
| 0.01 $\leq$ p < 0.05 | Rejection | Moderate evidence of misspecification | Consider alternative functional forms |
| 0.05 $\leq$ p < 0.10 | Borderline | Weak evidence | Investigate further |
| p $\geq$ 0.10 | Fail to reject | No evidence of misspecification | Linear form appears adequate |

### What to Do When RESET Rejects

When the test indicates misspecification, consider these remedies:

=== "Add Polynomials"
    ```python
    # Add squared terms
    data["value_sq"] = data["value"] ** 2
    fe = FixedEffects("invest ~ value + value_sq + capital",
                      data, "firm", "year")
    ```

=== "Log Transform"
    ```python
    import numpy as np
    data["log_value"] = np.log(data["value"])
    data["log_capital"] = np.log(data["capital"])
    fe = FixedEffects("invest ~ log_value + log_capital",
                      data, "firm", "year")
    ```

=== "Add Interactions"
    ```python
    data["value_capital"] = data["value"] * data["capital"]
    fe = FixedEffects("invest ~ value + capital + value_capital",
                      data, "firm", "year")
    ```

## Mathematical Details

### Test Procedure

1. **Estimate the original model**: $y = X\beta + \varepsilon$
2. **Compute fitted values**: $\hat{y} = X\hat{\beta}$
3. **Estimate augmented model**: $y = X\beta + \delta_2 \hat{y}^2 + \delta_3 \hat{y}^3 + u$
4. **Test**: $H_0: \delta_2 = \delta_3 = 0$ using an F-test

### F-Statistic

The test uses a Wald/F-statistic for joint significance of the power terms:

$$F = \frac{W / q}{1} \sim F(q, n - k - q)$$

where:

- $W$ = Wald statistic from joint test on $\delta$ coefficients
- $q$ = number of added power terms (e.g., 2 for powers [2, 3])
- $n$ = number of observations
- $k$ = number of original parameters

### Why Powers of Fitted Values?

The fitted values $\hat{y} = X\hat{\beta}$ are linear combinations of the regressors. Their powers $\hat{y}^2, \hat{y}^3$ approximate general nonlinear functions of $X$. If these powers are significant, the linear model misses some nonlinear relationship.

## Configuration Options

```python
from panelbox.validation.specification.reset import RESETTest

reset = RESETTest(results)

# Default: powers [2, 3] (quadratic and cubic)
result = reset.run(alpha=0.05)

# Only quadratic
result = reset.run(alpha=0.05, powers=[2])

# Quadratic, cubic, and quartic
result = reset.run(alpha=0.05, powers=[2, 3, 4])
```

### Parameters

The `.run()` method accepts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.05` | Significance level |
| `powers` | `list[int]` | `[2, 3]` | Powers of fitted values to include (must be $\geq 2$) |

### Result Metadata

| Key | Type | Description |
|-----|------|-------------|
| `powers` | `list[int]` | Powers used in the test |
| `gamma_coefficients` | `dict` | Coefficients on power terms |
| `standard_errors` | `dict` | Standard errors for power terms |
| `wald_statistic` | `float` | Wald chi-squared statistic |
| `F_statistic` | `float` | F-statistic (primary test statistic) |
| `df_numerator` | `int` | Numerator degrees of freedom |
| `df_denominator` | `int` | Denominator degrees of freedom |
| `pvalue_chi2` | `float` | P-value from chi-squared approximation |
| `augmented_formula` | `str` | Augmented regression formula |

## Choosing Powers

The choice of powers affects test sensitivity:

| Powers | Detects | Trade-off |
|--------|---------|-----------|
| `[2]` | Quadratic misspecification | Low power against cubic |
| `[2, 3]` | Quadratic and cubic (default) | Good balance |
| `[2, 3, 4]` | Higher-order nonlinearity | May lose power (too many terms) |

!!! tip "Recommendation"
    Use the default `powers=[2, 3]` unless you have strong reasons to expect a specific form of nonlinearity. Adding too many powers reduces test power.

## Complete Example

```python
from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.validation.specification.reset import RESETTest
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Linear specification
fe = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = fe.fit()

# Test with different power specifications
for powers in [[2], [2, 3], [2, 3, 4]]:
    reset = RESETTest(results)
    result = reset.run(powers=powers)
    print(f"Powers {powers}: F={result.statistic:.4f}, p={result.pvalue:.4f}")

# If RESET rejects, try log specification
import numpy as np
data["log_invest"] = np.log(data["invest"])
data["log_value"] = np.log(data["value"])
data["log_capital"] = np.log(data["capital"])

fe_log = FixedEffects("log_invest ~ log_value + log_capital",
                      data, "firm", "year")
results_log = fe_log.fit()

reset_log = RESETTest(results_log)
result_log = reset_log.run()
print(f"\nLog specification: F={result_log.statistic:.4f}, "
      f"p={result_log.pvalue:.4f}")
```

## Common Pitfalls

### Rejection Is Common

Many economic relationships are inherently nonlinear. RESET rejection does not mean the model is useless -- it means a nonlinear specification might fit better. Always consider whether the improvement in fit is economically meaningful.

### Not a Test for Omitted Variables

RESET detects nonlinear misspecification, not omitted linear variables. A model that passes RESET may still suffer from omitted variable bias if a relevant linear regressor is missing.

### Panel Data Considerations

PanelBox implements RESET using Pooled OLS with cluster-robust standard errors on the augmented model. This accounts for within-entity correlation when testing the significance of power terms. The test is applicable to any panel model type (FE, RE, Pooled).

### Multicollinearity in Augmented Model

Powers of fitted values can be highly correlated with the original regressors. If multicollinearity is severe, the test may be unreliable. Consider using a single power (e.g., `powers=[2]`) instead.

## See Also

- [Hausman Test](hausman.md) -- FE vs RE specification
- [Chow Test](chow.md) -- Structural break detection
- [Specification Tests Overview](index.md) -- All specification tests
- [Diagnostics Overview](../index.md) -- Complete diagnostic workflow

## References

- Ramsey, J. B. (1969). Tests for specification errors in classical linear least squares regression analysis. *Journal of the Royal Statistical Society, Series B*, 31(2), 350-371.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press. Chapter 8.
