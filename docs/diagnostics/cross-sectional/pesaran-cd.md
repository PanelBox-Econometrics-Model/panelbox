---
title: "Pesaran CD Test"
description: "Pesaran CD test for cross-sectional dependence in large panel data using PanelBox."
---

# Pesaran CD Test

!!! info "Quick Reference"
    **Class:** `panelbox.validation.cross_sectional_dependence.pesaran_cd.PesaranCDTest`
    **H₀:** No cross-sectional dependence ($E(\varepsilon_{it} \varepsilon_{jt}) = 0$ for all $i \neq j$)
    **H₁:** Cross-sectional dependence present
    **Statistic:** CD ~ N(0, 1) under H₀
    **Stata equivalent:** `xtcsd, pesaran abs`
    **R equivalent:** `plm::pcdtest(, test="cd")`

## What It Tests

The Pesaran (2004) CD test detects **cross-sectional dependence** in panel data by examining the pairwise correlations of residuals across entities. It is the recommended default test for cross-sectional dependence because it:

- Works well for **large N** panels (even with small T)
- Has a simple standard normal distribution under H₀
- Is computationally efficient
- Does not require normality assumptions

The test statistic is based on the **sum of pairwise residual correlations**, not their squares. This makes it powerful against alternatives where cross-sectional correlations have a consistent sign (all positive or all negative), but less powerful when positive and negative correlations cancel out.

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.datasets import load_grunfeld
from panelbox.validation.cross_sectional_dependence.pesaran_cd import PesaranCDTest

# Estimate model
data = load_grunfeld()
fe = FixedEffects(data, "invest", ["value", "capital"], "firm", "year")
results = fe.fit()

# Run Pesaran CD test
test = PesaranCDTest(results)
result = test.run(alpha=0.05)

print(f"CD statistic: {result.statistic:.3f}")
print(f"P-value:      {result.pvalue:.4f}")
print(f"Reject H₀:    {result.reject_null}")
print(result.conclusion)

# Examine correlation structure
meta = result.metadata
print(f"\nCorrelation Analysis:")
print(f"  N entities:         {meta['n_entities']}")
print(f"  N entity pairs:     {meta['n_pairs']}")
print(f"  Avg. correlation:   {meta['avg_correlation']:.3f}")
print(f"  Avg. |correlation|: {meta['avg_abs_correlation']:.3f}")
print(f"  Max |correlation|:  {meta['max_abs_correlation']:.3f}")
print(f"  Range:              [{meta['min_correlation']:.3f}, {meta['max_correlation']:.3f}]")
```

## Interpretation

### CD Statistic

Since CD ~ N(0, 1) under H₀, standard normal critical values apply:

| |CD| | p-value | Interpretation |
|------|---------|----------------|
| < 1.645 | > 0.10 | No evidence of cross-sectional dependence |
| 1.645 -- 1.96 | 0.05 -- 0.10 | Weak evidence of dependence |
| 1.96 -- 2.576 | 0.01 -- 0.05 | Moderate cross-sectional dependence |
| > 2.576 | < 0.01 | Strong cross-sectional dependence |

### Average Correlation Strength

The metadata provides the average absolute pairwise correlation, which quantifies the practical significance:

| $|\bar{\rho}|$ | Strength | Recommended Action |
|----------------|----------|---------------------|
| < 0.1 | Negligible | Standard or entity-clustered SE sufficient |
| 0.1 -- 0.3 | Moderate | Use Driscoll-Kraay SE |
| 0.3 -- 0.5 | Strong | Use PCSE or spatial models |
| > 0.5 | Very strong | Likely model misspecification; add common factors |

!!! tip "Sign of CD Statistic"
    - **CD > 0**: Predominance of positive pairwise correlations (common positive shocks)
    - **CD < 0**: Predominance of negative pairwise correlations (competitive/substitution effects)
    - **|CD| large but $|\bar{\rho}|$ small**: Many entity pairs; even small average correlations sum to a large statistic

## Mathematical Details

### Pairwise Correlations

For each pair of entities $(i, j)$, the sample correlation of residuals is:

$$\hat{\rho}_{ij} = \frac{\sum_{t=1}^{T_{ij}} \hat{e}_{it} \hat{e}_{jt}}{\sqrt{\sum_{t=1}^{T_{ij}} \hat{e}_{it}^2} \sqrt{\sum_{t=1}^{T_{ij}} \hat{e}_{jt}^2}}$$

where $T_{ij}$ is the number of common time periods for entities $i$ and $j$.

### CD Statistic

$$CD = \sqrt{\frac{2\bar{T}}{N(N-1)}} \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} \hat{\rho}_{ij}$$

where $\bar{T}$ is the average number of common time periods across all pairs.

### Distribution

Under $H_0$:

$$CD \xrightarrow{d} N(0, 1) \quad \text{as } N \to \infty$$

The p-value is computed from the two-sided standard normal distribution:

$$p = 2 \times (1 - \Phi(|CD|))$$

### Why Raw (Not Squared) Correlations

The CD statistic uses raw correlations $\hat{\rho}_{ij}$, not squared correlations $\hat{\rho}_{ij}^2$. This means:

- **Advantage**: Powerful when dependence has a consistent direction (all positive or all negative)
- **Limitation**: Positive and negative correlations can cancel, reducing power when dependence patterns are mixed

The [Breusch-Pagan LM test](bp-lm.md) uses squared correlations and does not suffer from this cancellation, but it is only appropriate for small N.

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.05` | Significance level |

### Result Metadata

| Key | Type | Description |
|-----|------|-------------|
| `n_entities` | `int` | Number of entities (N) |
| `n_time_periods` | `int` | Number of time periods (T) |
| `n_pairs` | `int` | Number of entity pairs with valid correlations |
| `avg_correlation` | `float` | Mean of pairwise correlations $\bar{\rho}$ |
| `avg_abs_correlation` | `float` | Mean of $|\hat{\rho}_{ij}|$ |
| `max_abs_correlation` | `float` | Maximum $|\hat{\rho}_{ij}|$ |
| `min_correlation` | `float` | Minimum $\hat{\rho}_{ij}$ |
| `max_correlation` | `float` | Maximum $\hat{\rho}_{ij}$ |

## Diagnostics

### Before and After Time Effects

A common strategy is to compare CD before and after including time fixed effects, which absorb common shocks:

```python
# Without time effects
fe = FixedEffects(data, "invest", ["value", "capital"], "firm", "year")
results = fe.fit()
cd_before = PesaranCDTest(results).run()

# With time effects
fe_tw = FixedEffects(
    data, "invest", ["value", "capital"], "firm", "year",
    time_effects=True
)
results_tw = fe_tw.fit()
cd_after = PesaranCDTest(results_tw).run()

print(f"CD without time effects: {cd_before.statistic:.3f} "
      f"(avg |rho| = {cd_before.metadata['avg_abs_correlation']:.3f})")
print(f"CD with time effects:    {cd_after.statistic:.3f} "
      f"(avg |rho| = {cd_after.metadata['avg_abs_correlation']:.3f})")

reduction = (1 - cd_after.metadata['avg_abs_correlation'] /
             cd_before.metadata['avg_abs_correlation']) * 100
print(f"Reduction in avg |correlation|: {reduction:.1f}%")
```

## Common Pitfalls

!!! warning "Common Pitfalls"
    1. **Minimum T**: Requires at least **T >= 3** time periods to compute meaningful correlations. Raises `ValueError` otherwise.
    2. **Cancellation effect**: When some entity pairs have positive correlations and others negative, the CD statistic can be close to zero even with strong dependence. In such cases, check the `avg_abs_correlation` in the metadata or use the [Breusch-Pagan LM test](bp-lm.md).
    3. **Unbalanced panels**: For unbalanced panels, the test uses the average effective T across pairs ($\bar{T}$) and computes correlations using pairwise complete observations. Pairs with fewer than 3 common periods are skipped.
    4. **Large N**: The test is designed for large N and has excellent properties in this setting. For very small N (< 5), the asymptotic normal approximation may not hold well.
    5. **Time effects**: If cross-sectional dependence is driven by common time shocks, including time fixed effects in the model may eliminate it. Always test with and without time effects.

## See Also

- [Cross-Sectional Dependence Tests Overview](index.md) -- comparison of all tests
- [Breusch-Pagan LM Test](bp-lm.md) -- for small N panels (uses squared correlations)
- [Driscoll-Kraay Standard Errors](../../inference/driscoll-kraay.md) -- SE robust to cross-sectional dependence
- [Panel-Corrected Standard Errors](../../inference/pcse.md) -- Beck-Katz PCSE

## References

- Pesaran, M. H. (2004). "General diagnostic tests for cross section dependence in panels." *University of Cambridge Working Paper*, No. 0435.
- Pesaran, M. H. (2015). "Testing weak cross-sectional dependence in large panels." *Econometric Reviews*, 34(6-10), 1089-1117.
- De Hoyos, R. E., & Sarafidis, V. (2006). "Testing for cross-sectional dependence in panel-data models." *Stata Journal*, 6(4), 482-496.
