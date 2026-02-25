---
title: "Mundlak Test"
description: "Mundlak test for correlated random effects in panel data models, a robust alternative to the Hausman test."
---

# Mundlak Test

!!! info "Quick Reference"
    **Class:** `panelbox.validation.specification.mundlak.MundlakTest`
    **Result:** `ValidationTestResult`
    **H₀:** RE is consistent (entity effects uncorrelated with regressors, $\gamma = 0$)
    **H₁:** RE is inconsistent (use Fixed Effects, $\gamma \neq 0$)
    **Stata equivalent:** Manually add group means to RE regression
    **R equivalent:** `plm::phtest(model, method = "aux")`

## What It Tests

The Mundlak test is an alternative to the [Hausman test](hausman.md) for choosing between Fixed Effects and Random Effects. Instead of comparing two sets of estimates, it augments the RE model with entity-level means of the time-varying regressors and tests whether those means are jointly significant.

The intuition is straightforward: if $\alpha_i$ is correlated with $X_{it}$, then the entity means $\bar{X}_i$ will capture this correlation. If $\bar{X}_i$ is significant in the augmented model, the RE assumption is violated.

### The Mundlak Device

The standard RE model is:

$$y_{it} = X_{it}'\beta + \alpha_i + \varepsilon_{it}$$

The Mundlak augmented model adds entity means:

$$y_{it} = X_{it}'\beta + \bar{X}_i'\gamma + \alpha_i + \varepsilon_{it}$$

where $\bar{X}_i = \frac{1}{T_i} \sum_{t=1}^{T_i} X_{it}$ is the time average of regressors for entity $i$.

**If $\gamma = 0$**: the group means add no information, confirming RE is appropriate.

**If $\gamma \neq 0$**: the regressors are correlated with the individual effects, and FE should be used.

## Quick Example

```python
from panelbox.models.static.random_effects import RandomEffects
from panelbox.validation.specification.mundlak import MundlakTest
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Estimate Random Effects model
re = RandomEffects("invest ~ value + capital", data, "firm", "year")
re_results = re.fit()

# Run Mundlak test
mundlak = MundlakTest(re_results)
result = mundlak.run(alpha=0.05)
print(result.summary())

# Programmatic access
print(f"Wald statistic: {result.statistic:.4f}")
print(f"P-value: {result.pvalue:.4f}")
print(f"Degrees of freedom: {result.df}")
print(f"Reject H0: {result.reject_null}")
```

## Interpretation

| P-value | Decision | Interpretation | Action |
|---------|----------|----------------|--------|
| p < 0.01 | Strong rejection | Strong evidence of correlated effects | Use Fixed Effects |
| 0.01 $\leq$ p < 0.05 | Rejection | Moderate evidence of correlation | Use Fixed Effects |
| 0.05 $\leq$ p < 0.10 | Borderline | Weak evidence | Report both; lean toward FE |
| p $\geq$ 0.10 | Fail to reject | No evidence of correlation | Use Random Effects |

### Examining Individual Group Means

The test metadata provides the coefficients on individual group-mean variables, revealing which regressors drive the correlation:

```python
result = mundlak.run(alpha=0.05)

# Coefficients on group means
for var, coef in result.metadata["delta_coefficients"].items():
    se = result.metadata["standard_errors"][var]
    t_stat = coef / se if se > 0 else 0
    print(f"  {var}: coef={coef:.4f}, se={se:.4f}, t={t_stat:.2f}")
```

## Mathematical Details

### Wald Test Statistic

The test statistic is a Wald test for the joint significance of $\gamma$:

$$W = \hat{\gamma}' \left[\widehat{\text{Var}}(\hat{\gamma})\right]^{-1} \hat{\gamma} \sim \chi^2(K)$$

where $K$ is the number of time-varying regressors.

### Equivalence to Hausman

Mundlak (1978) showed that the Correlated Random Effects (CRE) model:

$$y_{it} = X_{it}'\beta + \bar{X}_i'\gamma + \alpha_i^* + \varepsilon_{it}$$

yields $\hat{\beta}$ identical to the FE estimator when $\gamma \neq 0$. The test on $\gamma$ is asymptotically equivalent to the Hausman test but is computed differently, offering practical advantages.

### Implementation Details

PanelBox implements the Mundlak test using Pooled OLS with entity-clustered standard errors on the augmented model. This approach:

1. Adds entity-mean variables for all time-varying regressors
2. Estimates the augmented model with cluster-robust standard errors
3. Performs a Wald test on the joint significance of the mean variables

!!! note "Implementation Note"
    PanelBox uses Pooled OLS with clustered SEs (rather than RE estimation on the augmented model) to avoid numerical issues with variables that are constant within entities. This produces results consistent with the auxiliary regression approach used in R's `plm` package.

## Configuration Options

```python
from panelbox.validation.specification.mundlak import MundlakTest

# Basic usage
mundlak = MundlakTest(re_results)
result = mundlak.run(alpha=0.05)

# Stricter significance level
result = mundlak.run(alpha=0.01)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `PanelResults` | required | Results from Random Effects estimation |

The `.run()` method accepts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.05` | Significance level |

### Result Metadata

| Key | Type | Description |
|-----|------|-------------|
| `n_time_varying_vars` | `int` | Number of time-varying regressors |
| `delta_coefficients` | `dict` | Coefficients on group-mean variables |
| `standard_errors` | `dict` | Standard errors for group-mean coefficients |
| `F_statistic` | `float` | F-statistic (Wald / df) |
| `augmented_formula` | `str` | Formula used for augmented regression |

## Advantages Over Hausman

The Mundlak test offers several practical advantages:

| Feature | Hausman | Mundlak |
|---------|---------|---------|
| Models required | FE **and** RE | RE only |
| Positive test statistic | Not guaranteed | Always (Wald test) |
| Robust standard errors | Problematic | Fully compatible |
| Identifies which variable | No | Yes (individual $\hat{\gamma}_k$) |
| Time-invariant regressors | Excluded from test | Handled naturally |
| Unbalanced panels | Potential issues | Straightforward |

!!! tip "When to Prefer Mundlak"
    Use the Mundlak test when:

    - You want to identify **which** regressors are correlated with entity effects
    - The Hausman test produces a negative statistic
    - You need robust/clustered standard errors for the test
    - You only have RE results available

## Comparing Hausman and Mundlak

```python
from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.random_effects import RandomEffects
from panelbox.validation.specification.hausman import HausmanTest
from panelbox.validation.specification.mundlak import MundlakTest
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Estimate both models
fe = FixedEffects("invest ~ value + capital", data, "firm", "year")
fe_results = fe.fit()

re = RandomEffects("invest ~ value + capital", data, "firm", "year")
re_results = re.fit()

# Hausman test
hausman = HausmanTest(fe_results, re_results)
print(f"Hausman: chi2={hausman.statistic:.4f}, p={hausman.pvalue:.4f}")
print(f"  Recommendation: {hausman.recommendation}")

# Mundlak test
mundlak = MundlakTest(re_results)
mundlak_result = mundlak.run(alpha=0.05)
print(f"Mundlak: Wald={mundlak_result.statistic:.4f}, p={mundlak_result.pvalue:.4f}")
print(f"  Conclusion: {mundlak_result.conclusion}")

# Both tests should agree in most cases
```

## Common Pitfalls

### RE Model Required

The Mundlak test is only applicable to Random Effects models. Passing FE or Pooled OLS results will raise a `ValueError`:

```python
# This will raise ValueError
mundlak = MundlakTest(fe_results)  # Error: only for RE models
```

### Time-Invariant Regressors

If all regressors are time-invariant (no within-entity variation), the test cannot be computed because there are no group means to add. Ensure at least one regressor varies over time.

### Small Samples

With few entities or short time series, the Wald test may have limited power. Consider:

- Using a higher significance level (e.g., $\alpha = 0.10$)
- Reporting both FE and RE estimates regardless of test outcome
- Supplementing with economic reasoning about likely endogeneity

## See Also

- [Hausman Test](hausman.md) -- Classical FE vs RE test
- [RESET Test](reset.md) -- Functional form specification
- [Specification Tests Overview](index.md) -- All specification tests
- [Diagnostics Overview](../index.md) -- Complete diagnostic workflow

## References

- Mundlak, Y. (1978). On the pooling of time series and cross section data. *Econometrica*, 46(1), 69-85.
- Chamberlain, G. (1982). Multivariate regression models for panel data. *Journal of Econometrics*, 18(1), 5-46.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapter 10.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 4.
