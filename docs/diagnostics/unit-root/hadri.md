---
title: "Hadri Test"
description: "Hadri panel stationarity test in PanelBox — KPSS-based LM test with reversed null hypothesis where H0 is stationarity."
---

# Hadri Test (Panel Stationarity)

!!! info "Quick Reference"
    **Function:** `panelbox.diagnostics.unit_root.hadri.hadri_test()`
    **H₀:** All panels are **stationary** ($\sigma^2_{u_i} = 0$ for all $i$)
    **H₁:** At least one panel has a unit root ($\sigma^2_{u_i} > 0$ for some $i$)
    **Stata equivalent:** `xtunitroot hadri variable, robust`
    **R equivalent:** `plm::purtest(x, test="hadri")`

!!! warning "Reversed Null Hypothesis"
    The Hadri test has the **opposite** null hypothesis from LLC, IPS, Fisher, and Breitung. Here, H₀ is **stationarity**. Rejecting H₀ means evidence of **unit roots**, not stationarity. This reversal is the defining feature of the Hadri test.

## What It Tests

The Hadri (2000) test is a panel extension of the KPSS framework. It tests whether all series in the panel are stationary against the alternative that at least some series contain a unit root.

Because it reverses the null hypothesis, the Hadri test is primarily used as a **confirmation test**: if LLC/IPS/Fisher reject H₀ (unit root) and Hadri does not reject H₀ (stationarity), there is strong evidence for stationarity.

## Quick Example

```python
from panelbox.datasets import load_grunfeld
from panelbox.diagnostics.unit_root.hadri import hadri_test

data = load_grunfeld()

# Hadri test with robust standard errors
result = hadri_test(data, "invest", "firm", "year",
                    trend="c", robust=True)
print(result.summary())
```

Output:

```text
======================================================================
Hadri (2000) LM Test for Stationarity
======================================================================
H0: All series are stationary
H1: At least one series has a unit root

Specification: Constant only
Robust version: Yes
Number of entities (N): 10
Number of periods (T): 20

LM statistic: 0.4521
Z-statistic: 5.2847
P-value: 0.0000

Decision at 5% level: REJECT H0

Evidence against stationarity (at least one series has unit root)
======================================================================
```

## Interpretation

!!! warning "Reversed Logic"
    Remember: in the Hadri test, **rejecting** H₀ means the data is **not stationary** (unit root evidence). This is the **opposite** of LLC/IPS/Fisher interpretation.

| p-value | Decision | Meaning |
|:--------|:---------|:--------|
| $p < 0.01$ | Strong rejection of H₀ | Strong evidence against stationarity (unit roots present) |
| $0.01 \leq p < 0.05$ | Rejection of H₀ | Evidence against stationarity |
| $0.05 \leq p < 0.10$ | Borderline | Weak evidence against stationarity |
| $p \geq 0.10$ | Fail to reject H₀ | Consistent with stationarity in all panels |

### Confirmation Strategy

Use the Hadri test alongside LLC/IPS/Fisher to reach a robust conclusion:

| LLC/IPS/Fisher | Hadri | Conclusion |
|:--------------|:------|:-----------|
| Not reject H₀ (unit root) | Reject H₀ (not stationary) | **Unit root confirmed** -- both tests agree |
| Reject H₀ (stationary) | Not reject H₀ (stationary) | **Stationarity confirmed** -- both tests agree |
| Reject H₀ (stationary) | Reject H₀ (not stationary) | **Mixed evidence** -- investigate further |
| Not reject H₀ (unit root) | Not reject H₀ (stationary) | **Mixed evidence** -- investigate further |

## Mathematical Details

### Model Decomposition

The Hadri test decomposes the data generating process:

$$y_{it} = r_{it} + \beta_i t + \varepsilon_{it}$$

$$r_{it} = r_{i,t-1} + u_{it}$$

where:

- $r_{it}$ is a random walk component
- $u_{it}$ are random walk innovations with variance $\sigma^2_{u_i}$
- $\varepsilon_{it}$ is a stationary error

Under H₀: $\sigma^2_{u_i} = 0$ for all $i$ (no random walk component, series is stationary).

### LM Statistic

The test statistic is an LM (Lagrange Multiplier) statistic based on the KPSS framework:

1. For each entity $i$, regress $y_{it}$ on deterministic terms and compute residuals $\hat{\varepsilon}_{it}$
2. Compute partial sums: $S_{it} = \sum_{s=1}^{t} \hat{\varepsilon}_{is}$
3. Entity-specific LM statistic:

$$LM_i = \frac{1}{T^2} \sum_{t=1}^{T} \frac{S_{it}^2}{\hat{\sigma}^2_{\varepsilon_i}}$$

4. Average across entities: $\overline{LM} = \frac{1}{N} \sum_{i=1}^{N} LM_i$

### Standardization

The Z-statistic is:

$$Z = \frac{\sqrt{N}(\overline{LM} - \mu)}{\sigma} \xrightarrow{d} N(0, 1)$$

where $\mu$ and $\sigma$ are asymptotic moments that depend on the trend specification:

| Specification | $\mu$ | $\sigma^2$ |
|:-------------|:------|:-----------|
| Constant only (`trend='c'`) | $1/6$ | $1/45$ |
| Constant + trend (`trend='ct'`) | $1/15$ | $1/6300$ |

The p-value is computed as $1 - \Phi(Z)$ (right-tailed test: reject for large values).

## Configuration Options

```python
hadri_test(
    data,                   # pd.DataFrame: Panel data in long format
    variable,               # str: Variable to test
    entity_col='entity',    # str: Entity identifier column
    time_col='time',        # str: Time identifier column
    trend='c',              # str: 'c' (constant) or 'ct' (constant + trend)
    robust=True,            # bool: Use heteroskedasticity-robust variance
    alpha=0.05,             # float: Significance level
)
```

### Robust vs. Non-Robust

=== "Robust (recommended)"

    ```python
    # Heteroskedasticity-robust version using Newey-West estimator
    result = hadri_test(data, "invest", "firm", "year",
                        trend="c", robust=True)
    ```

    Uses a Bartlett kernel with automatic bandwidth selection for the long-run variance estimate.

=== "Non-robust"

    ```python
    # Standard (homoskedastic) version
    result = hadri_test(data, "invest", "firm", "year",
                        trend="c", robust=False)
    ```

    Assumes homoskedastic errors within each entity.

### Result Object: `HadriResult`

| Attribute | Type | Description |
|:----------|:-----|:-----------|
| `statistic` | `float` | Z-statistic (standardized LM) |
| `pvalue` | `float` | P-value from standard normal (right-tailed) |
| `reject` | `bool` | Whether to reject H₀ at the specified $\alpha$ |
| `lm_statistic` | `float` | Raw LM statistic before standardization |
| `individual_lm` | `np.ndarray` | LM statistics for each entity |
| `n_entities` | `int` | Number of cross-sectional units |
| `n_time` | `int` | Number of time periods |
| `trend` | `str` | Trend specification (`'c'` or `'ct'`) |
| `robust` | `bool` | Whether robust version was used |

## When to Use

**Use Hadri when:**

- You want to **confirm** stationarity findings from other tests
- The economic theory suggests variables should be stationary, and you want to test this as H₀
- You want a complete battery covering both unit root and stationarity null hypotheses

## Complete Battery Example

```python
from panelbox.datasets import load_grunfeld
from panelbox.validation.unit_root.llc import LLCTest
from panelbox.validation.unit_root.ips import IPSTest
from panelbox.diagnostics.unit_root.hadri import hadri_test

data = load_grunfeld()

# H0: unit root tests
llc = LLCTest(data, "invest", "firm", "year", trend="c")
llc_result = llc.run()

ips = IPSTest(data, "invest", "firm", "year", trend="c")
ips_result = ips.run()

# H0: stationarity test (Hadri)
hadri_result = hadri_test(data, "invest", "firm", "year",
                           trend="c", robust=True)

print("=== Unit Root Battery ===")
print(f"LLC (H0: unit root):      p={llc_result.pvalue:.4f}")
print(f"IPS (H0: unit root):      p={ips_result.pvalue:.4f}")
print(f"Hadri (H0: stationarity): p={hadri_result.pvalue:.4f}")

# Interpret
if llc_result.pvalue < 0.05 and not hadri_result.reject:
    print("Conclusion: Stationarity confirmed")
elif llc_result.pvalue >= 0.05 and hadri_result.reject:
    print("Conclusion: Unit root confirmed")
else:
    print("Conclusion: Mixed evidence -- investigate further")
```

## Common Pitfalls

!!! warning "Balanced Panel Required"
    The Hadri test requires a **balanced panel** (same $T$ for all entities). If the panel is unbalanced, the function raises a `ValueError`.

!!! warning "Sensitivity to Cross-Sectional Dependence"
    The Hadri test is **sensitive to cross-sectional dependence**. When entities are correlated (common shocks), the test tends to **over-reject** the stationarity null, falsely indicating unit roots. Always test for cross-sectional dependence first and interpret with caution.

!!! warning "Trend Misspecification"
    Choosing the wrong trend specification can distort results. If the data has a deterministic trend but you use `trend='c'`, the test may falsely reject stationarity. Conversely, including an unnecessary trend (`trend='ct'` for trendless data) reduces power.

## See Also

- [Unit Root Tests Overview](index.md) -- Comparison of all five tests and testing strategy
- [LLC Test](llc.md) -- Common unit root test (H₀ = unit root)
- [IPS Test](ips.md) -- Heterogeneous unit root test (H₀ = unit root)
- [Fisher Test](fisher.md) -- P-value combination test (H₀ = unit root)
- [Breitung Test](breitung.md) -- Debiased common unit root test
- [Cointegration Tests](../cointegration/index.md) -- Next step if variables are I(1)

## References

- Hadri, K. (2000). "Testing for stationarity in heterogeneous panel data." *Econometrics Journal*, 3(2), 148-161.
- Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). "Testing the null hypothesis of stationarity against the alternative of a unit root." *Journal of Econometrics*, 54(1-3), 159-178.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 12.
