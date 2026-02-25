---
title: "Baltagi-Wu LBI Test"
description: "Baltagi-Wu locally best invariant test for serial correlation in unbalanced panel data using PanelBox."
---

# Baltagi-Wu LBI Test

!!! info "Quick Reference"
    **Class:** `panelbox.validation.serial_correlation.baltagi_wu.BaltagiWuTest`
    **H₀:** No first-order serial correlation ($\rho = 0$)
    **H₁:** AR(1) serial correlation present ($\rho \neq 0$)
    **Statistic:** z-statistic ~ N(0, 1) asymptotically
    **Stata equivalent:** `xtserial` (variant)
    **R equivalent:** `plm::pbltest()`

## What It Tests

The Baltagi-Wu (1999) locally best invariant (LBI) test detects **first-order autocorrelation** in panel data, with specific strengths for **unbalanced panels**. The test is based on a modified Durbin-Watson statistic that accounts for heterogeneous time series lengths across entities.

Unlike the standard Durbin-Watson test, the Baltagi-Wu LBI statistic:

- Works with unbalanced panels where entities have different numbers of time periods
- Accounts for gaps in the time series
- Uses an asymptotic normal distribution for inference

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.datasets import load_grunfeld
from panelbox.validation.serial_correlation.baltagi_wu import BaltagiWuTest

# Estimate model
data = load_grunfeld()
fe = FixedEffects(data, "invest", ["value", "capital"], "firm", "year")
results = fe.fit()

# Run Baltagi-Wu test
test = BaltagiWuTest(results)
result = test.run(alpha=0.05)

print(f"z-statistic:   {result.statistic:.3f}")
print(f"P-value:       {result.pvalue:.4f}")
print(f"Reject H₀:     {result.reject_null}")
print(result.conclusion)

# Access detailed metadata
meta = result.metadata
print(f"LBI statistic: {meta['lbi_statistic']:.4f}")
print(f"Estimated rho: {meta['rho_estimate']:.4f}")
print(f"N entities:    {meta['n_entities']}")
print(f"Avg T:         {meta['avg_time_periods']:.1f}")
print(f"T range:       [{meta['min_time_periods']}, {meta['max_time_periods']}]")
```

## Interpretation

### LBI Statistic

The LBI statistic behaves like a Durbin-Watson statistic:

| LBI Value | Interpretation |
|-----------|----------------|
| LBI < 2 | Positive autocorrelation ($\rho > 0$) |
| LBI $\approx$ 2 | No autocorrelation ($\rho \approx 0$) |
| LBI > 2 | Negative autocorrelation ($\rho < 0$) |

### z-Statistic (Standardized)

| p-value | Decision | Interpretation |
|---------|----------|----------------|
| < 0.01 | Strong rejection | Strong evidence of AR(1) serial correlation |
| 0.01 -- 0.05 | Rejection | AR(1) autocorrelation present |
| 0.05 -- 0.10 | Borderline | Weak evidence; consider robust SE |
| > 0.10 | Fail to reject | No evidence of serial correlation |

### Estimated AR(1) Coefficient

The metadata includes an estimate of $\rho$, the AR(1) coefficient:

$$\hat{\rho} \approx 1 - \frac{LBI}{2}$$

| $\hat{\rho}$ | Autocorrelation Strength |
|---------------|--------------------------|
| $|\hat{\rho}| < 0.1$ | Negligible |
| $0.1 \leq |\hat{\rho}| < 0.3$ | Weak |
| $0.3 \leq |\hat{\rho}| < 0.6$ | Moderate |
| $|\hat{\rho}| \geq 0.6$ | Strong |

## Mathematical Details

### LBI Statistic

The locally best invariant test statistic is defined as:

$$LBI = \frac{\sum_{i=1}^{N} \sum_{t=2}^{T_i} (\hat{e}_{it} - \hat{e}_{i,t-1})^2}{\sum_{i=1}^{N} \sum_{t=1}^{T_i} \hat{e}_{it}^2}$$

where $\hat{e}_{it}$ are the model residuals and $T_i$ is the number of time periods for entity $i$.

### Asymptotic Distribution

Under $H_0: \rho = 0$:

- $E[LBI] \approx 2$
- $\text{Var}(LBI) \approx \frac{4 \sum_{i=1}^N (1/T_i)}{N}$

The standardized test statistic is:

$$z = \frac{LBI - 2}{\sqrt{\text{Var}(LBI)}} \xrightarrow{d} N(0, 1)$$

The variance formula accounts for the unbalanced structure through the entity-specific $T_i$ values.

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.05` | Significance level |

### Result Metadata

| Key | Type | Description |
|-----|------|-------------|
| `lbi_statistic` | `float` | Raw LBI statistic (Durbin-Watson-like) |
| `z_statistic` | `float` | Standardized z-statistic |
| `rho_estimate` | `float` | Estimated AR(1) coefficient |
| `n_entities` | `int` | Number of entities |
| `n_obs_total` | `int` | Total observations |
| `n_obs_used` | `int` | Observations used (after differencing) |
| `avg_time_periods` | `float` | Average T across entities |
| `min_time_periods` | `int` | Minimum T across entities |
| `max_time_periods` | `int` | Maximum T across entities |
| `variance_lbi` | `float` | Estimated variance of LBI |
| `se_lbi` | `float` | Standard error of LBI |

## Common Pitfalls

!!! warning "Common Pitfalls"
    1. **Minimum T**: Each entity needs at least 2 time periods. The test raises a `ValueError` if any entity has fewer.
    2. **Two-sided test**: The test is two-sided, detecting both positive and negative autocorrelation. Check the sign of $\hat{\rho}$ or the LBI value to determine the direction.
    3. **Asymptotic approximation**: For very small panels (few entities and short T), the normal approximation may be imprecise. The test is most reliable with larger panels.
    4. **Comparison with Wooldridge**: For balanced panels, the Wooldridge test is generally preferred. The Baltagi-Wu test adds value specifically for **unbalanced** panels.

## See Also

- [Serial Correlation Tests Overview](index.md) -- comparison of all tests
- [Wooldridge AR(1) Test](wooldridge.md) -- recommended for balanced panels
- [Breusch-Godfrey Test](breusch-godfrey.md) -- for higher-order serial correlation
- [Clustered Standard Errors](../../inference/clustered.md) -- correcting for autocorrelation

## References

- Baltagi, B. H., & Wu, P. X. (1999). "Unequally spaced panel data regressions with AR(1) disturbances." *Econometric Theory*, 15(6), 814-823.
- Baltagi, B. H., & Li, Q. (1995). "Testing AR(1) against MA(1) disturbances in an error component model." *Journal of Econometrics*, 68(1), 133-151.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer.
