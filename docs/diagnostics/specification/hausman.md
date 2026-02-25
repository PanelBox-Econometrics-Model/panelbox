---
title: "Hausman Test"
description: "Hausman specification test for choosing between Fixed Effects and Random Effects panel data models in PanelBox."
---

# Hausman Test

!!! info "Quick Reference"
    **Class:** `panelbox.validation.specification.hausman.HausmanTest`
    **Result:** `HausmanTestResult`
    **H₀:** Random Effects is consistent and efficient (no correlation between regressors and individual effects)
    **H₁:** Random Effects is inconsistent (use Fixed Effects)
    **Stata equivalent:** `hausman fe re`
    **R equivalent:** `plm::phtest(fe_model, re_model)`

## What It Tests

The Hausman test addresses the most fundamental question in panel data analysis: **should you use Fixed Effects or Random Effects?**

The answer depends on whether the unobserved individual effects $\alpha_i$ are correlated with the regressors $X_{it}$:

- If $\text{Cov}(\alpha_i, X_{it}) = 0$: Both FE and RE are consistent, but **RE is more efficient** (smaller standard errors)
- If $\text{Cov}(\alpha_i, X_{it}) \neq 0$: **FE remains consistent**, but RE becomes inconsistent (biased)

The Hausman test exploits this asymmetry: under H₀ (no correlation), both estimators converge to the same coefficients; under H₁, they diverge.

## Quick Example

```python
from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.random_effects import RandomEffects
from panelbox.validation.specification.hausman import HausmanTest

# Load data
from panelbox.datasets import load_grunfeld
data = load_grunfeld()

# Estimate both models
fe = FixedEffects("invest ~ value + capital", data, "firm", "year")
fe_results = fe.fit()

re = RandomEffects("invest ~ value + capital", data, "firm", "year")
re_results = re.fit()

# Run Hausman test
hausman = HausmanTest(fe_results, re_results, alpha=0.05)
print(hausman.summary())

# Access results directly
print(f"Chi2 statistic: {hausman.statistic:.4f}")
print(f"P-value: {hausman.pvalue:.4f}")
print(f"Degrees of freedom: {hausman.df}")
print(f"Recommendation: {hausman.recommendation}")
```

!!! tip "Auto-Run on Initialization"
    `HausmanTest` runs the test automatically when created. You can access `.statistic`, `.pvalue`, `.recommendation`, and `.reject_null` directly on the test object without calling `.run()`.

## Interpretation

| P-value | Decision | Interpretation | Action |
|---------|----------|----------------|--------|
| p < 0.01 | Strong rejection | Strong evidence that RE is inconsistent | Use Fixed Effects |
| 0.01 $\leq$ p < 0.05 | Rejection | Moderate evidence against RE | Use Fixed Effects |
| 0.05 $\leq$ p < 0.10 | Borderline | Weak evidence against RE | Report both; consider Mundlak test |
| p $\geq$ 0.10 | Fail to reject | No evidence against RE consistency | Use Random Effects |

The `HausmanTestResult` provides a `.recommendation` attribute that returns `"Fixed Effects"` or `"Random Effects"` based on the test outcome.

### Reading the Coefficient Comparison

The test summary includes a coefficient comparison table showing how FE and RE estimates differ for each variable. Large differences indicate the source of the test rejection:

```text
COEFFICIENT COMPARISON
======================================================================
Variable        Fixed Effects   Random Effects  Difference
----------------------------------------------------------------------
value                  0.1101         0.1048         0.0053
capital                0.3101         0.3249        -0.0148
======================================================================
```

## Mathematical Details

### Test Statistic

The Hausman statistic is:

$$
H = (\hat{\beta}_{FE} - \hat{\beta}_{RE})' \left[\widehat{\text{Var}}(\hat{\beta}_{FE}) - \widehat{\text{Var}}(\hat{\beta}_{RE})\right]^{-1} (\hat{\beta}_{FE} - \hat{\beta}_{RE})
$$

Under H₀, $H \sim \chi^2(K)$ where $K$ is the number of common coefficients tested (excluding the intercept, which FE does not estimate).

### Key Insight

Under H₀ (RE is consistent):

- $\text{plim}(\hat{\beta}_{FE}) = \text{plim}(\hat{\beta}_{RE}) = \beta$
- The variance difference $\text{Var}(\hat{\beta}_{FE}) - \text{Var}(\hat{\beta}_{RE})$ is positive semi-definite
- RE is efficient (has smaller variance), so $\text{Var}(\hat{\beta}_{FE}) - \text{Var}(\hat{\beta}_{RE}) \geq 0$

Under H₁ (RE is inconsistent):

- $\text{plim}(\hat{\beta}_{FE}) = \beta$ but $\text{plim}(\hat{\beta}_{RE}) \neq \beta$
- The difference $\hat{\beta}_{FE} - \hat{\beta}_{RE}$ is large, producing a large test statistic

## Configuration Options

```python
# Standard usage
hausman = HausmanTest(fe_results, re_results, alpha=0.05)

# Different significance level
hausman = HausmanTest(fe_results, re_results, alpha=0.10)

# Re-run with different alpha (test already computed on init)
result = hausman.run(alpha=0.01)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fe_results` | `PanelResults` | required | Results from Fixed Effects estimation |
| `re_results` | `PanelResults` | required | Results from Random Effects estimation |
| `alpha` | `float` | `0.05` | Significance level |

### Result Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.statistic` | `float` | Chi-squared test statistic |
| `.pvalue` | `float` | P-value from $\chi^2$ distribution |
| `.df` | `int` | Degrees of freedom (number of common coefficients) |
| `.recommendation` | `str` | `"Fixed Effects"` or `"Random Effects"` |
| `.reject_null` | `bool` | `True` if p < alpha |
| `.conclusion` | `str` | Human-readable conclusion |
| `.fe_params` | `pd.Series` | FE coefficients for common variables |
| `.re_params` | `pd.Series` | RE coefficients for common variables |
| `.diff` | `pd.Series` | Coefficient differences (FE - RE) |

## Complete Example with Interpretation

```python
from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.random_effects import RandomEffects
from panelbox.validation.specification.hausman import HausmanTest
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Estimate both models
fe = FixedEffects("invest ~ value + capital", data, "firm", "year")
fe_results = fe.fit()

re = RandomEffects("invest ~ value + capital", data, "firm", "year")
re_results = re.fit()

# Hausman test
hausman = HausmanTest(fe_results, re_results)

# Full summary
print(hausman.summary())

# Programmatic decision
if hausman.reject_null:
    print(f"Use Fixed Effects (p = {hausman.pvalue:.4f})")
    chosen = fe_results
else:
    print(f"Use Random Effects (p = {hausman.pvalue:.4f})")
    chosen = re_results

# Examine coefficient differences
result = hausman.run()
for var in result.diff.index:
    print(f"  {var}: FE={result.fe_params[var]:.4f}, "
          f"RE={result.re_params[var]:.4f}, "
          f"diff={result.diff[var]:.4f}")
```

## Common Pitfalls

### Negative Test Statistic

The variance difference matrix $\widehat{\text{Var}}(\hat{\beta}_{FE}) - \widehat{\text{Var}}(\hat{\beta}_{RE})$ may not be positive definite, leading to a negative test statistic. PanelBox handles this automatically using a generalized (pseudo) inverse.

**Causes:**

- Small sample sizes
- Highly correlated regressors
- Numerical precision issues

**Solution:** Consider the [Mundlak test](mundlak.md) as a more robust alternative.

### Large Sample Sizes

With very large panels, the Hausman test has high power and will reject H₀ even when the FE-RE differences are economically negligible. In such cases:

- Examine the coefficient differences in the comparison table
- Consider whether differences are **economically significant**, not just statistically significant
- Report both models for robustness

### Time-Invariant Variables

Fixed Effects cannot estimate coefficients on time-invariant variables (e.g., gender, region). These variables are absorbed into the entity fixed effect. If time-invariant variables are central to your analysis, and the Hausman test fails to reject H₀, Random Effects preserves those estimates.

### Comparison with Mundlak Test

The [Mundlak test](mundlak.md) tests the same hypothesis as Hausman but offers practical advantages:

| Feature | Hausman | Mundlak |
|---------|---------|---------|
| Requires both FE and RE | Yes | No (RE only) |
| Test statistic always positive | No | Yes |
| Compatible with robust SEs | Limited | Yes |
| Identifies source of endogeneity | No | Yes (via group mean coefficients) |

## See Also

- [Mundlak Test](mundlak.md) -- Robust alternative to Hausman
- [RESET Test](reset.md) -- Test for functional form misspecification
- [Specification Tests Overview](index.md) -- All specification tests
- [Diagnostics Overview](../index.md) -- Complete diagnostic workflow

## References

- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6), 1251-1271.
- Mundlak, Y. (1978). On the pooling of time series and cross section data. *Econometrica*, 46(1), 69-85.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapter 10.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 4.
