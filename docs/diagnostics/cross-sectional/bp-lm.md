---
title: "Breusch-Pagan LM Test (Cross-Sectional Dependence)"
description: "Breusch-Pagan LM test for cross-sectional dependence in small panel data using PanelBox."
---

# Breusch-Pagan LM Test for Cross-Sectional Dependence

!!! info "Quick Reference"
    **Class:** `panelbox.validation.cross_sectional_dependence.breusch_pagan_lm.BreuschPaganLMTest`
    **H₀:** No cross-sectional dependence ($\text{Corr}(\varepsilon_{it}, \varepsilon_{jt}) = 0$ for all $i \neq j$)
    **H₁:** At least one pair of entities has correlated errors
    **Statistic:** LM ~ $\chi^2(N(N-1)/2)$
    **Stata equivalent:** `xtcsd, xttest2`
    **R equivalent:** `plm::pcdtest(, test="lm")`

## What It Tests

The Breusch-Pagan (1980) LM test for cross-sectional dependence examines whether the residuals from different entities are correlated contemporaneously. It uses the **sum of squared pairwise correlations**, making it powerful for detecting any pattern of cross-sectional dependence -- both positive and negative correlations contribute to the test statistic.

!!! note "Not to Be Confused"
    This is the Breusch-Pagan LM test for **cross-sectional dependence**, not the [Breusch-Pagan test for heteroskedasticity](../heteroskedasticity/breusch-pagan.md). Despite sharing the same authors, these tests address different diagnostic questions.

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.datasets import load_grunfeld
from panelbox.validation.cross_sectional_dependence.breusch_pagan_lm import (
    BreuschPaganLMTest,
)

# Estimate model
data = load_grunfeld()
fe = FixedEffects(data, "invest", ["value", "capital"], "firm", "year")
results = fe.fit()

# Run Breusch-Pagan LM test
test = BreuschPaganLMTest(results)
result = test.run(alpha=0.05)

print(f"LM statistic: {result.statistic:.3f}")
print(f"P-value:      {result.pvalue:.4f}")
print(f"Degrees of freedom: {result.df}")
print(result.conclusion)

# Examine correlation details
meta = result.metadata
print(f"\nN entities:         {meta['n_entities']}")
print(f"N time periods:     {meta['n_time_periods']}")
print(f"N pairs:            {meta['n_pairs']} (expected: {meta['n_pairs_expected']})")
print(f"Mean |correlation|: {meta['mean_abs_correlation']:.3f}")
print(f"Max |correlation|:  {meta['max_abs_correlation']:.3f}")
print(f"Positive corr.:     {meta['n_positive_correlations']}")
print(f"Negative corr.:     {meta['n_negative_correlations']}")
```

## Interpretation

| p-value | Decision | Interpretation |
|---------|----------|----------------|
| < 0.01 | Strong rejection | Strong evidence of cross-sectional dependence |
| 0.01 -- 0.05 | Rejection | Cross-sectional dependence present |
| 0.05 -- 0.10 | Borderline | Weak evidence; consider Driscoll-Kraay SE |
| > 0.10 | Fail to reject | No evidence of cross-sectional dependence |

!!! warning "Over-Rejection with Large N"
    The BP LM test is known to be **over-sized** (rejects H₀ too frequently) when N is large. For panels with **N > 30**, use the [Pesaran CD test](pesaran-cd.md) instead. The implementation automatically includes a warning in the metadata when N > 30.

## Mathematical Details

### Test Statistic

$$LM = T \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} \hat{\rho}_{ij}^2$$

where $\hat{\rho}_{ij}$ is the sample correlation between the residuals of entities $i$ and $j$.

### Distribution

Under $H_0$:

$$LM \sim \chi^2\left(\frac{N(N-1)}{2}\right)$$

The degrees of freedom equal the number of unique entity pairs.

### Key Difference from Pesaran CD

| Feature | Breusch-Pagan LM | Pesaran CD |
|---------|-------------------|------------|
| Uses | $\hat{\rho}_{ij}^2$ (squared) | $\hat{\rho}_{ij}$ (raw) |
| Cancellation | No (squares are always positive) | Yes (opposite signs cancel) |
| Distribution | $\chi^2(N(N-1)/2)$ | N(0, 1) |
| Large N | Over-sized (too many rejections) | Correct size |
| Small N | Appropriate | May lack power |
| Power against mixed CD | High | Low (cancellation) |

### Why Squared Correlations Matter

The LM statistic uses $\hat{\rho}_{ij}^2$, which means:

- Both positive and negative correlations contribute positively to the statistic
- The test is powerful against **any** pattern of cross-sectional dependence
- It does not suffer from the cancellation problem of the Pesaran CD test

However, with $N(N-1)/2$ degrees of freedom, the chi-squared distribution becomes a poor approximation for large N, leading to over-rejection.

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.05` | Significance level |

### Result Metadata

| Key | Type | Description |
|-----|------|-------------|
| `n_entities` | `int` | Number of entities (N) |
| `n_time_periods` | `int` | Number of time periods (T) |
| `n_pairs` | `int` | Actual number of valid entity pairs |
| `n_pairs_expected` | `int` | Expected pairs: $N(N-1)/2$ |
| `mean_abs_correlation` | `float` | Mean of $|\hat{\rho}_{ij}|$ |
| `max_abs_correlation` | `float` | Maximum $|\hat{\rho}_{ij}|$ |
| `n_positive_correlations` | `int` | Count of positive pairwise correlations |
| `n_negative_correlations` | `int` | Count of negative pairwise correlations |
| `warning` | `str` or `None` | Warning message if N > 30 |

## Diagnostics

### Comparing with Pesaran CD

Running both tests provides complementary information:

```python
from panelbox.validation.cross_sectional_dependence.pesaran_cd import PesaranCDTest
from panelbox.validation.cross_sectional_dependence.breusch_pagan_lm import (
    BreuschPaganLMTest,
)

pesaran = PesaranCDTest(results).run()
bp = BreuschPaganLMTest(results).run()

print(f"Pesaran CD: stat={pesaran.statistic:.3f}, p={pesaran.pvalue:.4f}")
print(f"BP LM:      stat={bp.statistic:.3f}, p={bp.pvalue:.4f}")
```

!!! example "Reading the Comparison"
    - **Both reject**: Strong evidence of cross-sectional dependence with a consistent directional pattern
    - **BP rejects, Pesaran does not**: Mixed dependence -- some entity pairs positively correlated, others negatively. The squared correlations in BP detect this, but the raw correlations in Pesaran cancel out
    - **Pesaran rejects, BP does not**: Unusual; may occur with borderline significance
    - **Neither rejects**: No evidence of cross-sectional dependence

### When to Prefer BP LM Over Pesaran CD

The BP LM test is the better choice when:

1. **Small N** (< 20 entities): The chi-squared approximation is accurate
2. **Large T**: The test requires $T > N$ for well-estimated correlations
3. **Mixed dependence patterns**: When you suspect both positive and negative correlations across different entity pairs
4. **Full power needed**: The squared correlations ensure no cancellation

## Common Pitfalls

!!! warning "Common Pitfalls"
    1. **Large N panels**: The test becomes increasingly over-sized as N grows. For N > 30, the Pesaran CD test is strongly preferred. Check the `warning` field in metadata.
    2. **Requires T >= 3**: Each entity pair needs at least 3 common time periods to compute a meaningful correlation. Pairs with fewer periods are excluded.
    3. **Computational cost**: The test computes $N(N-1)/2$ pairwise correlations. For very large N (> 500), this can be slow ($O(N^2 T)$).
    4. **Constant residuals**: If any entity has constant residuals (zero variance), the correlation is undefined and that pair is skipped. Check `n_pairs` vs `n_pairs_expected` in metadata.
    5. **Unbalanced panels**: For unbalanced panels, the test uses pairwise complete observations. The effective degrees of freedom may differ from the theoretical $N(N-1)/2$.
    6. **Not for heteroskedasticity**: Despite the shared name, this is a test for **cross-sectional dependence**, not heteroskedasticity. For heteroskedasticity, see the [Breusch-Pagan heteroskedasticity test](../heteroskedasticity/breusch-pagan.md).

## See Also

- [Cross-Sectional Dependence Tests Overview](index.md) -- comparison of all tests
- [Pesaran CD Test](pesaran-cd.md) -- preferred for large N panels
- [Breusch-Pagan Heteroskedasticity Test](../heteroskedasticity/breusch-pagan.md) -- the other Breusch-Pagan test
- [Driscoll-Kraay Standard Errors](../../inference/driscoll-kraay.md) -- SE robust to CD
- [Panel-Corrected Standard Errors](../../inference/pcse.md) -- Beck-Katz PCSE

## References

- Breusch, T. S., & Pagan, A. R. (1980). "The Lagrange Multiplier test and its applications to model specification in econometrics." *Review of Economic Studies*, 47(1), 239-253.
- Pesaran, M. H. (2004). "General diagnostic tests for cross section dependence in panels." *University of Cambridge Working Paper*, No. 0435.
- De Hoyos, R. E., & Sarafidis, V. (2006). "Testing for cross-sectional dependence in panel-data models." *Stata Journal*, 6(4), 482-496.
- Frees, E. W. (1995). "Assessing cross-sectional correlation in panel data." *Journal of Econometrics*, 69(2), 393-414.
