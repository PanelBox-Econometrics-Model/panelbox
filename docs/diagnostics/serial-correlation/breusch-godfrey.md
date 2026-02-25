---
title: "Breusch-Godfrey Test"
description: "Breusch-Godfrey LM test for higher-order serial correlation in panel data models using PanelBox."
---

# Breusch-Godfrey Test

!!! info "Quick Reference"
    **Class:** `panelbox.validation.serial_correlation.breusch_godfrey.BreuschGodfreyTest`
    **H₀:** No serial correlation up to lag $p$
    **H₁:** At least one autoregressive coefficient $\rho_j \neq 0$
    **Statistic:** LM = $N \times R^2$ ~ $\chi^2(p)$
    **Stata equivalent:** `xttest1` (variant)
    **R equivalent:** `plm::pbgtest()`

## What It Tests

The Breusch-Godfrey (BG) test detects **serial correlation up to an arbitrary order** $p$ in the error terms of a panel regression. Unlike the Wooldridge test (which only detects AR(1)), the BG test can identify AR(2), AR(4), or any higher-order autoregressive pattern.

The test is a **Lagrange Multiplier (LM) test** based on an auxiliary regression of residuals on their own lags and the original regressors.

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.datasets import load_grunfeld
from panelbox.validation.serial_correlation.breusch_godfrey import BreuschGodfreyTest

# Estimate model
data = load_grunfeld()
fe = FixedEffects(data, "invest", ["value", "capital"], "firm", "year")
results = fe.fit()

# Test for AR(1) serial correlation
test = BreuschGodfreyTest(results)
result = test.run(alpha=0.05, lags=1)

print(f"LM statistic: {result.statistic:.3f}")
print(f"P-value:      {result.pvalue:.4f}")
print(f"Degrees of freedom: {result.df}")
print(result.conclusion)

# Test for higher-order serial correlation
for lag in [1, 2, 3, 4]:
    r = test.run(lags=lag)
    print(f"AR({lag}): LM={r.statistic:.3f}, p={r.pvalue:.4f}, "
          f"R²_aux={r.metadata['R2_auxiliary']:.4f}")
```

## Interpretation

| p-value | Decision | Interpretation |
|---------|----------|----------------|
| < 0.01 | Strong rejection | Strong evidence of serial correlation up to lag $p$ |
| 0.01 -- 0.05 | Rejection | Serial correlation present at tested lag order |
| 0.05 -- 0.10 | Borderline | Weak evidence; consider robust SE |
| > 0.10 | Fail to reject | No evidence of serial correlation up to lag $p$ |

!!! tip "Choosing the Number of Lags"
    - **Annual data**: test lags 1--2
    - **Quarterly data**: test up to lag 4 (annual cycle)
    - **Monthly data**: test up to lag 12 (annual cycle)
    - **General rule**: start with lag 1, increase if rejected

## Mathematical Details

### Auxiliary Regression

Given the original model residuals $\hat{e}_{it}$, the BG test estimates the auxiliary regression:

$$\hat{e}_{it} = X_{it}\gamma + \sum_{j=1}^{p} \rho_j \hat{e}_{i,t-j} + v_{it}$$

where $X_{it}$ are the original regressors and $p$ is the number of lags being tested.

### Hypotheses

$$H_0: \rho_1 = \rho_2 = \cdots = \rho_p = 0 \quad \text{(no serial correlation)}$$

$$H_1: \text{At least one } \rho_j \neq 0$$

### LM Statistic

For the panel data version, the test statistic is:

$$LM = N \times R^2_{\text{aux}} \sim \chi^2(p)$$

where $N$ is the number of cross-sectional entities and $R^2_{\text{aux}}$ is the R-squared from the auxiliary regression.

!!! note "Panel vs. Time-Series BG Test"
    The panel version uses $N$ (number of entities) instead of $n$ (total observations) in the LM formula. This follows the approach of Baltagi & Li (1995) and the `plm` package in R.

### Why It Works

By including both the lagged residuals and the original regressors $X$ in the auxiliary regression, the test:

1. Accounts for any correlation between residuals and regressors (important when $X$ includes lagged dependent variables)
2. Tests whether past residuals have **additional** explanatory power beyond what $X$ already captures

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.05` | Significance level |
| `lags` | `int` | `1` | Number of lags to test (order of AR process) |

### Result Metadata

| Key | Type | Description |
|-----|------|-------------|
| `lags` | `int` | Number of lags tested |
| `R2_auxiliary` | `float` | R-squared from the auxiliary regression |
| `n_obs_auxiliary` | `int` | Observations used in auxiliary regression |
| `n_entities` | `int` | Number of cross-sectional entities |

## Diagnostics

### Testing Multiple Lag Orders

A common diagnostic approach is to test progressively higher lag orders to identify the nature of serial correlation:

```python
test = BreuschGodfreyTest(results)

print("Lag | LM Stat  | p-value  | R² (aux) | Reject?")
print("----|----------|----------|----------|--------")
for lag in range(1, 5):
    r = test.run(lags=lag)
    print(f"  {lag} | {r.statistic:8.3f} | {r.pvalue:8.4f} | "
          f"{r.metadata['R2_auxiliary']:8.4f} | {'Yes' if r.reject_null else 'No'}")
```

!!! example "Reading the Results"
    - If only AR(1) is rejected: likely a simple AR(1) process -- use clustered SE
    - If AR(1) and AR(2) are rejected but not AR(3): AR(2) process present
    - If all tested lags are rejected: strong autocorrelation -- use HAC SE (Newey-West)

### Advantages Over Durbin-Watson

| Feature | Durbin-Watson | Breusch-Godfrey |
|---------|---------------|-----------------|
| Lagged dependent variables | Invalid | Valid |
| Higher-order AR | AR(1) only | AR(p) for any p |
| Panel data | Not designed | Panel-adapted |
| Distribution | Bounds test (inconclusive zone) | Chi-squared (exact critical values) |

## Common Pitfalls

!!! warning "Common Pitfalls"
    1. **Design matrix required**: The test needs access to the original design matrix $X$. If the model does not store it internally, a `ValueError` is raised.
    2. **Lags must be >= 1**: Passing `lags=0` or negative values raises a `ValueError`.
    3. **Observation loss**: Each additional lag drops one observation per entity. With short panels, testing high lag orders can leave too few observations.
    4. **Low power with many lags**: Testing too many lags (relative to T) reduces statistical power. The chi-squared distribution has $p$ degrees of freedom, so more lags require stronger signals.
    5. **Panel-specific LM formula**: The PanelBox implementation uses $LM = N \times R^2$ (not $nT \times R^2$), following the panel econometrics convention. Direct comparison with time-series BG statistics requires this adjustment.

## See Also

- [Serial Correlation Tests Overview](index.md) -- comparison of all tests
- [Wooldridge AR(1) Test](wooldridge.md) -- simpler first-order test
- [Baltagi-Wu LBI Test](baltagi-wu.md) -- for unbalanced panels
- [Newey-West Standard Errors](../../inference/newey-west.md) -- HAC standard errors for higher-order autocorrelation

## References

- Breusch, T. S. (1978). "Testing for autocorrelation in dynamic linear models." *Australian Economic Papers*, 17(31), 334-355.
- Godfrey, L. G. (1978). "Testing against general autoregressive and moving average error models when the regressors include lagged dependent variables." *Econometrica*, 46(6), 1293-1301.
- Baltagi, B. H., & Li, Q. (1995). "Testing AR(1) against MA(1) disturbances in an error component model." *Journal of Econometrics*, 68(1), 133-151.
