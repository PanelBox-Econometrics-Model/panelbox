---
title: "LLC Test"
description: "Levin-Lin-Chu common unit root test for panel data in PanelBox — testing for a common autoregressive parameter across all panel entities."
---

# LLC Test (Levin-Lin-Chu)

!!! info "Quick Reference"
    **Class:** `panelbox.validation.unit_root.llc.LLCTest`
    **H₀:** All panels contain unit roots (common $\rho = 0$)
    **H₁:** All panels are stationary (common $\rho < 0$)
    **Stata equivalent:** `xtunitroot llc variable, lags(p) trend(none|demean|trend)`
    **R equivalent:** `plm::purtest(x, test="levinlin")`

## What It Tests

The Levin-Lin-Chu (LLC) test examines whether all panels in the dataset share a **common unit root**. It assumes that the autoregressive parameter $\rho$ is the same across all entities, making it a **homogeneous** panel unit root test.

If the test rejects H₀, there is evidence that **all** panels are stationary. If it fails to reject, the data is consistent with a unit root process in all panels.

## Quick Example

```python
from panelbox.datasets import load_grunfeld
from panelbox.validation.unit_root.llc import LLCTest

data = load_grunfeld()

# Basic LLC test with constant
llc = LLCTest(data, "invest", "firm", "year", trend="c")
result = llc.run()
print(result)
```

Output:

```text
======================================================================
Levin-Lin-Chu Panel Unit Root Test
======================================================================
Test statistic:    -3.2145
P-value:           0.0007
Lags:              1
Observations:      180
Cross-sections:    10
Deterministics:    Constant

H0: All panels contain unit roots
H1: All panels are stationary

Conclusion: Reject H0: Evidence against unit root (panels are stationary)
======================================================================
```

## Interpretation

| p-value | Decision | Meaning |
|:--------|:---------|:--------|
| $p < 0.01$ | Strong rejection of H₀ | Strong evidence of stationarity across all panels |
| $0.01 \leq p < 0.05$ | Rejection of H₀ | Evidence of stationarity |
| $0.05 \leq p < 0.10$ | Borderline | Weak evidence; consider additional tests |
| $p \geq 0.10$ | Fail to reject H₀ | Consistent with unit root in all panels |

## Mathematical Details

### Model

The LLC test is based on the augmented Dickey-Fuller regression applied to each panel:

$$\Delta y_{it} = \rho \, y_{i,t-1} + \sum_{j=1}^{p_i} \theta_{ij} \Delta y_{i,t-j} + \alpha_i + \delta_i t + \varepsilon_{it}$$

where:

- $\rho$ is the **common** autoregressive parameter (same for all $i$)
- $\alpha_i$ are entity-specific intercepts
- $\delta_i t$ are entity-specific time trends (when `trend='ct'`)
- $p_i$ is the lag order (can be entity-specific via AIC)

### Procedure

The test follows a three-step procedure:

1. **Orthogonalize**: For each entity $i$, regress $\Delta y_{it}$ and $y_{i,t-1}$ on the augmented lags and deterministic terms. Collect the residuals $\tilde{e}_{it}$ and $\tilde{v}_{it}$.

2. **Normalize**: Divide $\tilde{e}_{it}$ and $\tilde{v}_{it}$ by each entity's long-run standard deviation $\hat{\sigma}_i$ to remove heteroskedasticity across entities.

3. **Pool and test**: Run the pooled regression $\tilde{e}_{it} / \hat{\sigma}_i = \rho \, \tilde{v}_{it} / (\hat{\sigma}_i \sqrt{T_i}) + \text{error}$ and compute the adjusted t-statistic.

### Test Statistic

The adjusted t-statistic converges to a standard normal distribution under H₀:

$$t_{\rho}^{*} \xrightarrow{d} N(0, 1) \quad \text{as } N, T \to \infty$$

The p-value is computed from the standard normal CDF (left-tailed test).

## Configuration Options

```python
LLCTest(
    data,                   # pd.DataFrame: Panel data in long format
    variable,               # str: Variable to test for unit root
    entity_col,             # str: Entity identifier column
    time_col,               # str: Time identifier column
    lags=None,              # int or None: Number of augmented lags (None = auto via AIC)
    trend='c',              # str: 'n' (none), 'c' (constant), 'ct' (constant + trend)
)
```

### Trend Specifications

=== "No deterministic terms (`trend='n'`)"

    ```python
    # For data without intercept or trend (e.g., returns)
    llc = LLCTest(data, "returns", "firm", "year", trend="n")
    result = llc.run()
    ```

=== "Constant only (`trend='c'`)"

    ```python
    # Default — entity-specific intercept
    llc = LLCTest(data, "invest", "firm", "year", trend="c")
    result = llc.run()
    ```

=== "Constant + trend (`trend='ct'`)"

    ```python
    # For variables with deterministic trends (e.g., log GDP)
    llc = LLCTest(data, "log_gdp", "country", "year", trend="ct")
    result = llc.run()
    ```

### Result Object: `LLCTestResult`

| Attribute | Type | Description |
|:----------|:-----|:-----------|
| `statistic` | `float` | Adjusted t-statistic |
| `pvalue` | `float` | P-value from standard normal |
| `lags` | `int` | Number of lags used |
| `n_obs` | `int` | Number of observations |
| `n_entities` | `int` | Number of cross-sectional units |
| `test_type` | `str` | `"LLC"` |
| `deterministics` | `str` | Description of trend specification |
| `null_hypothesis` | `str` | `"All panels contain unit roots"` |
| `alternative_hypothesis` | `str` | `"All panels are stationary"` |
| `conclusion` | `str` | Test conclusion at 5% level |

## When to Use

**Use LLC when:**

- Entities share similar dynamics (common $\rho$ is plausible)
- The panel is balanced (same $T$ for all entities)
- You need a simple, well-known test as a baseline

**Examples of appropriate settings:**

- Exchange rates for countries in the same monetary zone
- Stock returns in the same market
- Inflation rates for similar economies

## Common Pitfalls

!!! warning "Balanced Panel Required"
    The LLC test requires a balanced panel. If your panel is unbalanced, a warning is issued. Consider using the [Fisher test](fisher.md) instead, which handles unbalanced panels naturally.

!!! warning "Homogeneity Assumption"
    The LLC test assumes all panels share the same $\rho$. If entities have heterogeneous adjustment speeds, the test may have **low power** or produce misleading results. Use the [IPS test](ips.md) for heterogeneous panels.

!!! warning "Cross-Sectional Independence"
    The LLC test assumes cross-sectional independence. Common shocks (e.g., global financial crises) violate this assumption and can lead to **size distortions** (over-rejection). The [Breitung test](breitung.md) is more robust to cross-sectional dependence.

## See Also

- [Unit Root Tests Overview](index.md) -- Comparison of all five tests
- [IPS Test](ips.md) -- Heterogeneous alternative to LLC
- [Breitung Test](breitung.md) -- More robust common unit root test
- [Hadri Test](hadri.md) -- Confirmation test with reversed null
- [Cointegration Tests](../cointegration/index.md) -- Next step if variables are I(1)

## References

- Levin, A., Lin, C. F., & Chu, C. S. J. (2002). "Unit root tests in panel data: asymptotic and finite-sample properties." *Journal of Econometrics*, 108(1), 1-24.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 12.
