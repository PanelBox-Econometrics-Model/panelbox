---
title: "Breitung Test"
description: "Breitung debiased panel unit root test in PanelBox — improved finite-sample properties and reduced cross-sectional dependence sensitivity."
---

# Breitung Test (Debiased Panel Unit Root)

!!! info "Quick Reference"
    **Function:** `panelbox.diagnostics.unit_root.breitung.breitung_test()`
    **H₀:** All panels have a unit root (common $\rho = 0$)
    **H₁:** All panels are stationary (common $\rho < 0$)
    **Stata equivalent:** `xtunitroot breitung variable`
    **R equivalent:** `plm::purtest(x, test="Breitung")`

## What It Tests

The Breitung (2000) test is a **common unit root** test (like LLC) that uses a **debiasing transformation** to improve finite-sample properties. It is less sensitive to cross-sectional dependence and nuisance parameter bias than the LLC test.

The test assumes a common autoregressive parameter across all entities and tests whether it equals zero (unit root) or is negative (stationarity).

## Quick Example

```python
from panelbox.datasets import load_grunfeld
from panelbox.diagnostics.unit_root.breitung import breitung_test

data = load_grunfeld()

# Breitung test with constant and trend
result = breitung_test(data, "invest", "firm", "year", trend="ct")
print(result.summary())
```

Output:

```text
======================================================================
Breitung (2000) Unit Root Test
======================================================================
H0: All series have a unit root
H1: All series are stationary

Specification: Constant and trend
Number of entities (N): 10
Number of periods (T): 20

Test statistic: -2.4521
P-value: 0.0071

Decision at 5% level: REJECT H0

Evidence of stationarity (reject unit root)
======================================================================
```

## Interpretation

| p-value | Decision | Meaning |
|:--------|:---------|:--------|
| $p < 0.01$ | Strong rejection of H₀ | Strong evidence of stationarity |
| $0.01 \leq p < 0.05$ | Rejection of H₀ | Evidence of stationarity |
| $0.05 \leq p < 0.10$ | Borderline | Weak evidence; consider additional tests |
| $p \geq 0.10$ | Fail to reject H₀ | Consistent with unit root |

## Mathematical Details

### Detrending Transformation

The Breitung test removes deterministic components using a robust transformation that avoids nuisance parameter bias:

=== "Constant only (`trend='c'`)"

    $$\tilde{y}_{it} = y_{it} - \bar{y}_i$$

    Simple demeaning by entity mean.

=== "Constant + trend (`trend='ct'`)"

    $$\tilde{y}_{it} = y_{it} - \bar{y}_i - (t - \bar{T}) \frac{y_{iT} - y_{i1}}{T - 1}$$

    where $\bar{T} = (T+1)/2$. This transformation removes both the mean and the linear trend in a way that is robust under the unit root null.

### Pooled Regression

After detrending, the pooled regression is:

$$\Delta \tilde{y}_{it} = \rho \, \tilde{y}_{i,t-1} + \varepsilon_{it}$$

The bias-corrected estimator adjusts for the finite-sample bias of approximately $-3.5/T$.

### Test Statistic

The bias-corrected t-statistic:

$$t_{\rho}^{bc} = \frac{\hat{\rho} - \text{bias}}{SE(\hat{\rho})} \xrightarrow{d} N(0, 1)$$

The p-value is computed from the standard normal CDF (left-tailed: reject for negative values).

## Configuration Options

```python
breitung_test(
    data,                   # pd.DataFrame: Panel data in long format
    variable,               # str: Variable to test
    entity_col='entity',    # str: Entity identifier column
    time_col='time',        # str: Time identifier column
    trend='ct',             # str: 'c' (constant) or 'ct' (constant + trend, default)
    alpha=0.05,             # float: Significance level
)
```

!!! note "Default trend is `'ct'`"
    Unlike LLC which defaults to `trend='c'`, the Breitung test defaults to `trend='ct'` (constant and trend). This is because the Breitung detrending is specifically designed to handle trending data effectively.

### Result Object: `BreitungResult`

| Attribute | Type | Description |
|:----------|:-----|:-----------|
| `statistic` | `float` | Bias-corrected test statistic |
| `pvalue` | `float` | P-value from standard normal (left-tailed) |
| `reject` | `bool` | Whether to reject H₀ at the specified $\alpha$ |
| `raw_statistic` | `float` | Raw $\hat{\rho}$ before bias correction |
| `n_entities` | `int` | Number of cross-sectional units |
| `n_time` | `int` | Number of time periods |
| `trend` | `str` | Trend specification |

## When to Use

**Use Breitung when:**

- You suspect **cross-sectional dependence** may bias the LLC test
- You want a **more conservative** but more reliable common unit root test
- The panel has **moderate T** where LLC may have finite-sample bias
- Data contains **deterministic trends** (`trend='ct'` is the default)

## Comparing LLC and Breitung

```python
from panelbox.validation.unit_root.llc import LLCTest
from panelbox.diagnostics.unit_root.breitung import breitung_test

data = load_grunfeld()

# LLC test
llc = LLCTest(data, "invest", "firm", "year", trend="ct")
llc_result = llc.run()

# Breitung test
breit_result = breitung_test(data, "invest", "firm", "year", trend="ct")

print(f"LLC:      stat={llc_result.statistic:.4f}, p={llc_result.pvalue:.4f}")
print(f"Breitung: stat={breit_result.statistic:.4f}, p={breit_result.pvalue:.4f}")
```

| Aspect | LLC | Breitung |
|:-------|:----|:---------|
| Root type | Common $\rho$ | Common $\rho$ |
| Bias correction | Mean/variance adjustment | Explicit debiasing |
| Cross-sectional dependence | Sensitive | More robust |
| Finite-sample properties | May over-reject | More conservative |
| Trend handling | Standard demeaning | Robust transformation |
| Default trend | `'c'` | `'ct'` |

## Common Pitfalls

!!! warning "Balanced Panel Required"
    The Breitung test requires a **balanced panel** (same $T$ for all entities). If the panel is unbalanced, a `ValueError` is raised. Use the [Fisher test](fisher.md) for unbalanced panels.

!!! warning "Homogeneity Assumption"
    Like LLC, the Breitung test assumes a **common** autoregressive parameter $\rho$ across entities. If entities have heterogeneous dynamics, consider using the [IPS test](ips.md) or [Fisher test](fisher.md).

!!! warning "Conservative Test"
    The Breitung test is more conservative than LLC (less likely to reject H₀). In very small samples, it may fail to detect stationarity even when it exists. Always complement with other tests.

## See Also

- [Unit Root Tests Overview](index.md) -- Comparison of all five tests
- [LLC Test](llc.md) -- Standard common unit root test
- [IPS Test](ips.md) -- Heterogeneous unit root test
- [Hadri Test](hadri.md) -- Confirmation test with reversed null
- [Cointegration Tests](../cointegration/index.md) -- Next step if variables are I(1)

## References

- Breitung, J. (2000). "The local power of some unit root tests for panel data." In *Advances in Econometrics*, Vol. 15, 161-177.
- Breitung, J., & Das, S. (2005). "Panel unit root tests under cross-sectional dependence." *Statistica Neerlandica*, 59(4), 414-433.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 12.
