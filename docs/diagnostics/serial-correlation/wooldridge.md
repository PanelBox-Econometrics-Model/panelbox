---
title: "Wooldridge AR(1) Test"
description: "Wooldridge test for first-order autocorrelation in panel data fixed effects models using PanelBox."
---

# Wooldridge AR(1) Test

!!! info "Quick Reference"
    **Class:** `panelbox.validation.serial_correlation.wooldridge_ar.WooldridgeARTest`
    **H₀:** No first-order autocorrelation in idiosyncratic errors
    **H₁:** AR(1) autocorrelation present
    **Statistic:** F-statistic ~ F(1, N-1)
    **Stata equivalent:** `xtserial y x1 x2`
    **R equivalent:** `plm::pwartest()`

## What It Tests

The Wooldridge test (2002) detects **first-order autocorrelation (AR(1))** in the idiosyncratic errors of fixed effects panel models. It is the recommended default test for serial correlation in panel data because of its simplicity and robustness.

The test exploits a key property of first-differenced residuals: under the null hypothesis of no serial correlation, the coefficient from regressing $\Delta \hat{\varepsilon}_{it}$ on $\Delta \hat{\varepsilon}_{i,t-1}$ equals $-0.5$.

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.datasets import load_grunfeld
from panelbox.validation.serial_correlation.wooldridge_ar import WooldridgeARTest

# Estimate Fixed Effects model
data = load_grunfeld()
fe = FixedEffects(data, "invest", ["value", "capital"], "firm", "year")
results = fe.fit()

# Run Wooldridge test
test = WooldridgeARTest(results)
result = test.run(alpha=0.05)

print(f"F-statistic: {result.statistic:.3f}")
print(f"P-value:     {result.pvalue:.4f}")
print(f"Reject H₀:   {result.reject_null}")
print(result.conclusion)

# Access additional metadata
print(f"Coefficient:  {result.metadata['coefficient']:.4f} (expected -0.5 under H₀)")
print(f"Std. Error:   {result.metadata['std_error']:.4f}")
print(f"N entities:   {result.metadata['n_entities']}")
print(f"Obs. used:    {result.metadata['n_obs_used']}")
```

## Interpretation

| p-value | Decision | Interpretation |
|---------|----------|----------------|
| < 0.01 | Strong rejection | Strong evidence of AR(1) autocorrelation |
| 0.01 -- 0.05 | Rejection | Moderate evidence of autocorrelation |
| 0.05 -- 0.10 | Borderline | Weak evidence; consider robust SE as precaution |
| > 0.10 | Fail to reject | No evidence of first-order autocorrelation |

!!! tip "Practical Guidance"
    If the test rejects H₀, re-estimate the model with **clustered standard errors** (`cov_type="clustered"`) or **Newey-West standard errors** (`cov_type="newey_west"`). Coefficient estimates remain consistent but standard errors from the original model are unreliable.

## Mathematical Details

### Test Procedure

**Step 1.** Estimate the fixed effects model and obtain residuals $\hat{\varepsilon}_{it}$.

**Step 2.** Compute first differences of the residuals:

$$\Delta \hat{\varepsilon}_{it} = \hat{\varepsilon}_{it} - \hat{\varepsilon}_{i,t-1}$$

**Step 3.** Run the auxiliary regression:

$$\Delta \hat{\varepsilon}_{it} = \beta \, \Delta \hat{\varepsilon}_{i,t-1} + v_{it}$$

**Step 4.** Test $H_0: \beta = -0.5$ using an F-test.

### Why $\beta = -0.5$ Under H₀

Under the null hypothesis of no serial correlation ($\text{Cov}(\varepsilon_{it}, \varepsilon_{is}) = 0$ for $t \neq s$):

$$\text{Cov}(\Delta \varepsilon_{it}, \Delta \varepsilon_{i,t-1}) = E[(\varepsilon_{it} - \varepsilon_{i,t-1})(\varepsilon_{i,t-1} - \varepsilon_{i,t-2})] = -\sigma^2_\varepsilon$$

$$\text{Var}(\Delta \varepsilon_{it}) = 2\sigma^2_\varepsilon$$

Therefore:

$$\beta = \frac{\text{Cov}(\Delta \varepsilon_{it}, \Delta \varepsilon_{i,t-1})}{\text{Var}(\Delta \varepsilon_{i,t-1})} = \frac{-\sigma^2_\varepsilon}{2\sigma^2_\varepsilon} = -0.5$$

### F-Statistic

The test statistic is:

$$F = \left(\frac{\hat{\beta} - (-0.5)}{\text{SE}(\hat{\beta})}\right)^2 \sim F(1, N-1)$$

where $N$ is the number of cross-sectional entities.

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.05` | Significance level for the test |

### Result Metadata

| Key | Type | Description |
|-----|------|-------------|
| `coefficient` | `float` | Estimated $\hat{\beta}$ from auxiliary regression |
| `std_error` | `float` | Standard error of $\hat{\beta}$ |
| `t_statistic` | `float` | t-statistic for $H_0: \beta = -0.5$ |
| `n_entities` | `int` | Number of cross-sectional entities |
| `n_obs_used` | `int` | Observations used in auxiliary regression |

## Diagnostics

### Responding to Rejection

```python
# If Wooldridge test rejects, compare SE correction methods
if result.reject_null:
    # Standard (biased) SE
    results_std = fe.fit()

    # Cluster-robust SE
    results_cluster = fe.fit(cov_type="clustered")

    # Compare for a key variable
    for var in ["value", "capital"]:
        se_std = results_std.std_errors[var]
        se_cl = results_cluster.std_errors[var]
        print(f"{var}: SE(standard)={se_std:.4f}, SE(clustered)={se_cl:.4f}, "
              f"ratio={se_cl/se_std:.2f}")
```

### Checking the Estimated Coefficient

```python
# The coefficient should be close to -0.5 under H₀
beta_hat = result.metadata['coefficient']
deviation = beta_hat - (-0.5)
print(f"Estimated beta: {beta_hat:.4f}")
print(f"Deviation from -0.5: {deviation:.4f}")

# Large deviations indicate strong autocorrelation
# beta > -0.5 suggests positive AR(1) (rho > 0)
# beta < -0.5 suggests negative AR(1) (rho < 0)
```

## Common Pitfalls

!!! warning "Common Pitfalls"
    1. **Model type**: The test is designed for Fixed Effects models. Using it with Pooled OLS or Random Effects produces a warning and may give unreliable results.
    2. **Minimum periods**: Requires at least **T >= 3** time periods per entity (two periods are lost to differencing and lagging). Raises `ValueError` if this condition is not met.
    3. **Small N**: The F(1, N-1) distribution approximation may be poor with very few entities (N < 10). Consider supplementing with Baltagi-Wu or Breusch-Godfrey.
    4. **Only tests AR(1)**: This test does not detect higher-order serial correlation. If you suspect AR(2) or higher, use the [Breusch-Godfrey test](breusch-godfrey.md).
    5. **Dynamic models**: When the model includes a lagged dependent variable, the test may have reduced power. Consider testing the residuals from a GMM estimator instead.

## See Also

- [Serial Correlation Tests Overview](index.md) -- comparison of all available tests
- [Breusch-Godfrey Test](breusch-godfrey.md) -- for higher-order serial correlation
- [Baltagi-Wu LBI Test](baltagi-wu.md) -- for unbalanced panels
- [Clustered Standard Errors](../../inference/clustered.md) -- correcting for serial correlation
- [Newey-West Standard Errors](../../inference/newey-west.md) -- HAC standard errors

## References

- Wooldridge, J. M. (2002). *Econometric Analysis of Cross Section and Panel Data*. MIT Press, Section 10.4.1.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press, Chapter 10.
- Drukker, D. M. (2003). "Testing for serial correlation in linear panel-data models." *Stata Journal*, 3(2), 168-177.
