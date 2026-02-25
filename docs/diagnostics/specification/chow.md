---
title: "Chow Test"
description: "Chow test for structural breaks and parameter stability in panel data models."
---

# Chow Test

!!! info "Quick Reference"
    **Class:** `panelbox.validation.specification.chow.ChowTest`
    **Result:** `ValidationTestResult`
    **H₀:** No structural break (parameters are stable across subperiods, $\beta_1 = \beta_2$)
    **H₁:** Structural break exists (parameters differ, $\beta_1 \neq \beta_2$)
    **Stata equivalent:** Manual (split sample + F-test)
    **R equivalent:** `strucchange::sctest()`

## What It Tests

The Chow test (1960) examines whether the regression parameters change at a known point in time. In panel data, this tests whether a structural event -- such as a policy change, financial crisis, or regulatory reform -- fundamentally altered the relationship between variables.

The test splits the sample at a specified break point and compares the fit of:

- **Restricted model**: single set of parameters for the entire sample
- **Unrestricted model**: separate parameters for each subperiod

If the unrestricted model fits significantly better, parameters are unstable.

## Quick Example

```python
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.validation.specification.chow import ChowTest
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Estimate model
model = PooledOLS("invest ~ value + capital", data, "firm", "year")
results = model.fit()

# Test for structural break at a specific year
chow = ChowTest(results)
result = chow.run(alpha=0.05, break_point=1945)
print(result.summary())

# Access results
print(f"F-statistic: {result.statistic:.4f}")
print(f"P-value: {result.pvalue:.4f}")
print(f"Break point: {result.metadata['break_point']}")
```

## Interpretation

| P-value | Decision | Interpretation | Action |
|---------|----------|----------------|--------|
| p < 0.01 | Strong rejection | Strong evidence of structural break | Split sample or add time interactions |
| 0.01 $\leq$ p < 0.05 | Rejection | Moderate evidence of break | Consider regime-specific models |
| 0.05 $\leq$ p < 0.10 | Borderline | Weak evidence | Investigate further |
| p $\geq$ 0.10 | Fail to reject | No evidence of break | Parameters appear stable |

### Comparing Subperiod Coefficients

When the test rejects, examine how coefficients differ across subperiods:

```python
result = chow.run(break_point=1945)

# Coefficients before and after break
coefs_before = result.metadata["coefficients_period1"]
coefs_after = result.metadata["coefficients_period2"]

print("Variable         Before      After       Change")
print("-" * 55)
for var in coefs_before:
    before = coefs_before[var]
    after = coefs_after[var]
    print(f"{var:<16} {before:>10.4f} {after:>10.4f} {after - before:>10.4f}")
```

## Mathematical Details

### F-Statistic

The Chow test statistic is:

$$F = \frac{(RSS_r - RSS_1 - RSS_2) / K}{(RSS_1 + RSS_2) / (N - 2K)} \sim F(K, N - 2K)$$

where:

- $RSS_r$ = residual sum of squares from the pooled (restricted) model
- $RSS_1$ = residual sum of squares from subperiod 1 (before break)
- $RSS_2$ = residual sum of squares from subperiod 2 (after break)
- $K$ = number of parameters (including intercept)
- $N$ = total number of observations

### Intuition

Under H₀ ($\beta_1 = \beta_2$), pooling the data should not significantly worsen the fit: $RSS_r \approx RSS_1 + RSS_2$.

Under H₁ ($\beta_1 \neq \beta_2$), the restricted model is forced to average across two different relationships, producing $RSS_r \gg RSS_1 + RSS_2$.

## Configuration Options

```python
from panelbox.validation.specification.chow import ChowTest

chow = ChowTest(results)

# Specific time period as break point
result = chow.run(break_point=2008)

# Fraction of sample (e.g., 50% = midpoint)
result = chow.run(break_point=0.5)

# Auto-detect: uses median time period
result = chow.run(break_point=None)

# Different significance level
result = chow.run(alpha=0.01, break_point=2008)
```

### Parameters

The `.run()` method accepts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.05` | Significance level |
| `break_point` | `int`, `float`, or `None` | `None` | Time period to split at (see below) |

**Break point specification:**

| Type | Interpretation | Example |
|------|----------------|---------|
| `int` | Exact time period value | `break_point=2008` |
| `float` (0-1) | Fraction of sample | `break_point=0.5` (midpoint) |
| `None` | Median time period | Auto-detected |

### Result Metadata

| Key | Type | Description |
|-----|------|-------------|
| `break_point` | value | The break point used |
| `n_obs_period1` | `int` | Observations before break |
| `n_obs_period2` | `int` | Observations after break |
| `n_obs_total` | `int` | Total observations |
| `ssr_restricted` | `float` | RSS from pooled model |
| `ssr_unrestricted` | `float` | RSS from unrestricted ($RSS_1 + RSS_2$) |
| `ssr_period1` | `float` | RSS from subperiod 1 |
| `ssr_period2` | `float` | RSS from subperiod 2 |
| `k_parameters` | `int` | Number of parameters |
| `coefficients_period1` | `dict` | Coefficients before break |
| `coefficients_period2` | `dict` | Coefficients after break |

## Complete Example: Testing for Policy Change Effect

```python
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.validation.specification.chow import ChowTest
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Estimate model
model = PooledOLS("invest ~ value + capital", data, "firm", "year")
results = model.fit()

# Test for structural break at different points
print("Scanning for structural breaks:")
print(f"{'Year':<8} {'F-stat':<12} {'P-value':<12} {'Result'}")
print("-" * 50)

years = sorted(data["year"].unique())
# Test middle years (need sufficient observations on each side)
for year in years[3:-3]:
    chow = ChowTest(results)
    try:
        result = chow.run(break_point=year, alpha=0.05)
        status = "BREAK" if result.reject_null else "Stable"
        print(f"{year:<8} {result.statistic:<12.4f} "
              f"{result.pvalue:<12.4f} {status}")
    except ValueError:
        continue
```

## Common Pitfalls

### Known Break Point Required

The Chow test requires specifying the break point **a priori**. It tests whether parameters change at a **known** time, not where a break occurs. Testing multiple break points without correction inflates the Type I error rate.

!!! warning "Data Mining Risk"
    Scanning many possible break points and reporting only the most significant result is invalid. If you need to find an unknown break point, use methods designed for that purpose (e.g., Bai-Perron tests for unknown structural breaks).

### Minimum Sample Size

Each subperiod must have at least $2K$ observations (twice the number of parameters). With short panels or many regressors, valid break points are limited. The test raises a `ValueError` if a subperiod has insufficient observations.

### Homoskedasticity Assumption

The standard Chow test assumes equal error variances across subperiods. If heteroskedasticity is present, consider:

- Using robust standard errors on the subperiod models
- Supplementing with a formal heteroskedasticity test

### Panel Structure

For panel data, the Chow test splits by **time period**, not by entity. It tests whether parameters are stable over time, with all entities pooled together. To test for entity-specific parameter differences, use entity interaction terms instead.

### Interpreting Non-Rejection

Failing to reject does not prove stability -- it means the data lacks sufficient evidence of a break. With small samples or gradual parameter changes, the test may have low power.

## See Also

- [RESET Test](reset.md) -- Functional form misspecification
- [Hausman Test](hausman.md) -- FE vs RE specification
- [J-Test](j-test.md) -- Non-nested model comparison
- [Specification Tests Overview](index.md) -- All specification tests
- [Diagnostics Overview](../index.md) -- Complete diagnostic workflow

## References

- Chow, G. C. (1960). Tests of equality between sets of coefficients in two linear regressions. *Econometrica*, 28(3), 591-605.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 4.
- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson. Chapter 6.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
