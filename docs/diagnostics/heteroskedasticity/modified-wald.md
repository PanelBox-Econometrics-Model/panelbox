---
title: "Modified Wald Test"
description: "Modified Wald test for groupwise heteroskedasticity in fixed effects panel models using PanelBox."
---

# Modified Wald Test

!!! info "Quick Reference"
    **Class:** `panelbox.validation.heteroskedasticity.modified_wald.ModifiedWaldTest`
    **H₀:** $\sigma_i^2 = \sigma^2$ for all $i = 1, \ldots, N$ (homoskedastic across entities)
    **H₁:** $\sigma_i^2 \neq \sigma_j^2$ for at least one pair (groupwise heteroskedasticity)
    **Statistic:** Wald ~ $\chi^2(N)$
    **Stata equivalent:** `xttest3`
    **R equivalent:** Custom implementation

## What It Tests

The Modified Wald test detects **groupwise heteroskedasticity** in fixed effects panel models. It tests whether the error variance is the same across all cross-sectional entities or varies by group.

This is the most relevant heteroskedasticity test for FE models because it directly addresses the question: "Do different entities have different error variances?"

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.datasets import load_grunfeld
from panelbox.validation.heteroskedasticity.modified_wald import ModifiedWaldTest

# Estimate Fixed Effects model
data = load_grunfeld()
fe = FixedEffects(data, "invest", ["value", "capital"], "firm", "year")
results = fe.fit()

# Run Modified Wald test
test = ModifiedWaldTest(results)
result = test.run(alpha=0.05)

print(f"Wald statistic: {result.statistic:.3f}")
print(f"P-value:        {result.pvalue:.4f}")
print(f"Degrees of freedom: {result.df}")
print(result.conclusion)

# Examine variance heterogeneity
meta = result.metadata
print(f"N entities:       {meta['n_entities']}")
print(f"Pooled variance:  {meta['pooled_variance']:.4f}")
print(f"Min entity var:   {meta['min_entity_var']:.4f}")
print(f"Max entity var:   {meta['max_entity_var']:.4f}")
print(f"Variance ratio:   {meta['variance_ratio']:.2f}")
```

## Interpretation

### Jointly Using the p-value and Variance Ratio

The test provides both a formal statistical test and a descriptive measure of heterogeneity:

| p-value | Variance Ratio | Interpretation |
|---------|---------------|----------------|
| < 0.01 | > 10 | Strong heteroskedasticity -- use robust SE or FGLS |
| 0.01 -- 0.05 | 5 -- 10 | Moderate heteroskedasticity -- use robust SE |
| 0.05 -- 0.10 | 2 -- 5 | Weak heteroskedasticity -- consider robust SE |
| > 0.10 | < 2 | No evidence -- standard SE adequate |

!!! tip "Variance Ratio Guidelines"
    The **variance ratio** (`max_entity_var / min_entity_var`) provides practical context:

    - **Ratio < 2**: Variances are relatively homogeneous
    - **Ratio 2--5**: Moderate heterogeneity
    - **Ratio 5--10**: Strong heterogeneity; robust SE recommended
    - **Ratio > 10**: Very strong heterogeneity; consider FGLS

    A significant p-value with a small variance ratio (< 2) may indicate the test is detecting statistically significant but practically negligible differences due to large sample size.

## Mathematical Details

### Entity-Level Variances

For each entity $i$, the entity-specific variance is estimated as:

$$\hat{\sigma}_i^2 = \frac{1}{T_i - 1} \sum_{t=1}^{T_i} \hat{e}_{it}^2$$

where $\hat{e}_{it}$ are the fixed effects residuals.

### Pooled Variance

The pooled variance is:

$$\hat{\sigma}^2 = \frac{\sum_{i=1}^N \sum_{t=1}^{T_i} \hat{e}_{it}^2}{nT - N - k}$$

where $nT$ is the total number of observations, $N$ is the number of entities, and $k$ is the number of estimated parameters.

### Wald Statistic

The test statistic is:

$$W = \sum_{i=1}^{N} T_i \ln\left(\frac{\hat{\sigma}^2}{\hat{\sigma}_i^2}\right) \sim \chi^2(N)$$

Under $H_0$, all entity variances equal the pooled variance and $W$ is approximately chi-squared distributed with $N$ degrees of freedom.

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.05` | Significance level |

### Result Metadata

| Key | Type | Description |
|-----|------|-------------|
| `n_entities` | `int` | Number of entities (N) |
| `pooled_variance` | `float` | Pooled error variance $\hat{\sigma}^2$ |
| `min_entity_var` | `float` | Minimum entity-level variance |
| `max_entity_var` | `float` | Maximum entity-level variance |
| `variance_ratio` | `float` | Ratio of max to min entity variance |

## Diagnostics

### Comparing Standard Error Corrections

When the test rejects homoskedasticity, compare how different SE corrections change inference:

```python
if result.reject_null:
    # Re-estimate with different SE
    results_std = fe.fit()
    results_robust = fe.fit(cov_type="robust")
    results_cluster = fe.fit(cov_type="clustered")

    print("Variable   | SE(std)  | SE(robust) | SE(cluster) | Ratio(r/s)")
    print("-----------|----------|------------|-------------|----------")
    for var in ["value", "capital"]:
        se_s = results_std.std_errors[var]
        se_r = results_robust.std_errors[var]
        se_c = results_cluster.std_errors[var]
        print(f"{var:10s} | {se_s:.4f}  | {se_r:.4f}    | "
              f"{se_c:.4f}     | {se_r/se_s:.2f}")
```

!!! example "Reading the Comparison"
    - **Ratio > 1**: Robust SE are larger than standard SE, suggesting standard SE are downward biased (the common case with heteroskedasticity)
    - **Ratio < 1**: Standard SE are already conservative; heteroskedasticity inflates standard SE for some variables
    - **Large differences** (ratio > 1.5 or < 0.7): Heteroskedasticity has substantial impact on inference

## Common Pitfalls

!!! warning "Common Pitfalls"
    1. **FE models only**: The test is designed for Fixed Effects models. Using it with other model types triggers a warning and may produce unreliable results.
    2. **Entities with zero variance**: If any entity has zero residual variance (perfect fit within entity), it is skipped in the computation. This can happen with entities having very few observations.
    3. **Large N**: With many entities, the chi-squared approximation with $N$ degrees of freedom works well, but the test becomes very powerful and may detect trivially small differences.
    4. **Statistical vs. practical significance**: With large panels, the test will almost always reject. Always check the **variance ratio** alongside the p-value to assess practical significance.
    5. **Unbalanced panels**: The test handles unbalanced panels through entity-specific $T_i$ weights, but entities with very few periods will have imprecisely estimated variances.

## See Also

- [Heteroskedasticity Tests Overview](index.md) -- comparison of all tests
- [Breusch-Pagan Test](breusch-pagan.md) -- identifies which regressors drive heteroskedasticity
- [White Test](white.md) -- model-free heteroskedasticity test
- [Robust Standard Errors](../../inference/robust.md) -- HC0--HC3 corrections
- [Clustered Standard Errors](../../inference/clustered.md) -- cluster-robust inference

## References

- Greene, W. H. (2000). *Econometric Analysis* (4th ed.). Prentice Hall, Chapter 14.
- Baum, C. F. (2001). "Residual diagnostics for cross-section time series regression models." *Stata Journal*, 1(1), 101-104.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press, Chapter 10.
