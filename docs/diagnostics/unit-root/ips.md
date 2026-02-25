---
title: "IPS Test"
description: "Im-Pesaran-Shin heterogeneous unit root test for panel data in PanelBox — allowing different autoregressive parameters across panel entities."
---

# IPS Test (Im-Pesaran-Shin)

!!! info "Quick Reference"
    **Class:** `panelbox.validation.unit_root.ips.IPSTest`
    **H₀:** All panels contain unit roots ($\rho_i = 0$ for all $i$)
    **H₁:** Some panels are stationary ($\rho_i < 0$ for some $i$)
    **Stata equivalent:** `xtunitroot ips variable, lags(aic p)`
    **R equivalent:** `plm::purtest(x, test="ips")`

## What It Tests

The Im-Pesaran-Shin (IPS) test examines whether panels contain unit roots, allowing each entity to have a **different autoregressive parameter** $\rho_i$. Unlike the [LLC test](llc.md) which assumes a common $\rho$, the IPS test permits heterogeneous adjustment speeds across entities.

If the test rejects H₀, there is evidence that **at least some** panels are stationary (not necessarily all). This makes the IPS test more flexible and realistic for panels with heterogeneous entities.

## Quick Example

```python
from panelbox.datasets import load_grunfeld
from panelbox.validation.unit_root.ips import IPSTest

data = load_grunfeld()

# IPS test with constant
ips = IPSTest(data, "invest", "firm", "year", trend="c")
result = ips.run()
print(result)
```

Output:

```text
======================================================================
Im-Pesaran-Shin Panel Unit Root Test
======================================================================
W-statistic:       -2.8534
t-bar statistic:   -2.1240
P-value:           0.0022
Lags:              1
Observations:      180
Cross-sections:    10
Deterministics:    Constant

H0: All panels contain unit roots
H1: Some panels are stationary

Conclusion: Reject H0: Evidence that some panels are stationary
======================================================================
```

## Interpretation

| p-value | Decision | Meaning |
|:--------|:---------|:--------|
| $p < 0.01$ | Strong rejection of H₀ | Strong evidence that some panels are stationary |
| $0.01 \leq p < 0.05$ | Rejection of H₀ | Evidence of stationarity in some panels |
| $0.05 \leq p < 0.10$ | Borderline | Weak evidence; investigate individual entities |
| $p \geq 0.10$ | Fail to reject H₀ | Consistent with unit root in all panels |

!!! tip "Examine Individual Statistics"
    After rejecting H₀, inspect `result.individual_stats` to see which entities drive the rejection. Entities with more negative t-statistics are more likely to be stationary.

    ```python
    for entity, t_stat in result.individual_stats.items():
        status = "likely stationary" if t_stat < -2.0 else "likely unit root"
        print(f"Entity {entity}: t = {t_stat:.3f} ({status})")
    ```

## Mathematical Details

### Model

Each entity has its own ADF regression with a potentially different $\rho_i$:

$$\Delta y_{it} = \rho_i y_{i,t-1} + \sum_{j=1}^{p_i} \theta_{ij} \Delta y_{i,t-j} + \alpha_i + \delta_i t + \varepsilon_{it}$$

where $\rho_i$ is **heterogeneous** across entities.

### Procedure

1. **Individual ADF tests**: For each entity $i$, run an ADF test and obtain the individual t-statistic $t_i$ for testing $\rho_i = 0$.

2. **Average t-statistic**: Compute the cross-sectional average:

    $$\bar{t} = \frac{1}{N} \sum_{i=1}^{N} t_i$$

3. **Standardize**: The W-statistic is:

    $$W = \frac{\sqrt{N}\left(\bar{t} - E[\bar{t}]\right)}{\sqrt{\text{Var}[\bar{t}]}} \xrightarrow{d} N(0,1)$$

    where $E[\bar{t}]$ and $\text{Var}[\bar{t}]$ are the expected value and variance of the individual ADF t-statistics under H₀, tabulated in Im, Pesaran & Shin (2003).

### Test Statistic

The W-statistic follows a standard normal distribution under H₀:

$$W \xrightarrow{d} N(0, 1) \quad \text{as } N, T \to \infty$$

## Configuration Options

```python
IPSTest(
    data,                   # pd.DataFrame: Panel data in long format
    variable,               # str: Variable to test for unit root
    entity_col,             # str: Entity identifier column
    time_col,               # str: Time identifier column
    lags=None,              # int, dict, or None: Lags (None = auto per entity)
    trend='c',              # str: 'n' (none), 'c' (constant), 'ct' (constant + trend)
)
```

### Lag Specification

The IPS test supports flexible lag selection:

=== "Automatic (recommended)"

    ```python
    # AIC-based lag selection per entity
    ips = IPSTest(data, "invest", "firm", "year", lags=None)
    result = ips.run()
    ```

=== "Fixed lags"

    ```python
    # Same lag length for all entities
    ips = IPSTest(data, "invest", "firm", "year", lags=2)
    result = ips.run()
    ```

=== "Per-entity lags"

    ```python
    # Different lags per entity via dictionary
    lags_dict = {"firm_1": 1, "firm_2": 2, "firm_3": 1}
    ips = IPSTest(data, "invest", "firm", "year", lags=lags_dict)
    result = ips.run()
    ```

### Result Object: `IPSTestResult`

| Attribute | Type | Description |
|:----------|:-----|:-----------|
| `statistic` | `float` | W-statistic (standardized $\bar{t}$) |
| `t_bar` | `float` | Average of individual ADF t-statistics |
| `pvalue` | `float` | P-value from standard normal |
| `lags` | `int` or `list` | Lags used (int if uniform, list if varying) |
| `n_obs` | `int` | Total number of observations |
| `n_entities` | `int` | Number of cross-sectional units |
| `individual_stats` | `dict` | Individual ADF t-statistics per entity |
| `test_type` | `str` | `"IPS"` |
| `deterministics` | `str` | Description of trend specification |
| `conclusion` | `str` | Test conclusion at 5% level |

## When to Use

**Use IPS when:**

- Entities have heterogeneous dynamics (different $\rho_i$ are plausible)
- You want to know whether **some** (not necessarily all) panels are stationary
- The panel is (approximately) balanced

**Examples of appropriate settings:**

- GDP across countries with different economic structures
- Stock prices of firms in different industries
- Any panel where entities may have different adjustment speeds

## Comparing LLC and IPS

```python
from panelbox.validation.unit_root.llc import LLCTest
from panelbox.validation.unit_root.ips import IPSTest

data = load_grunfeld()

# LLC: common rho, H1 = ALL stationary
llc = LLCTest(data, "invest", "firm", "year", trend="c")
llc_result = llc.run()

# IPS: heterogeneous rho_i, H1 = SOME stationary
ips = IPSTest(data, "invest", "firm", "year", trend="c")
ips_result = ips.run()

print(f"LLC:  stat={llc_result.statistic:.4f}, p={llc_result.pvalue:.4f}")
print(f"IPS:  W={ips_result.statistic:.4f}, t_bar={ips_result.t_bar:.4f}, p={ips_result.pvalue:.4f}")
```

| Aspect | LLC | IPS |
|:-------|:----|:----|
| $\rho$ | Common | Heterogeneous $\rho_i$ |
| H₁ | All stationary | Some stationary |
| Power | Higher if common $\rho$ holds | Higher if $\rho_i$ vary |
| Balance | Required | Preferred |

## Common Pitfalls

!!! warning "Cross-Sectional Independence"
    Like LLC, the IPS test assumes cross-sectional independence. Correlated errors across entities (e.g., due to common shocks) can cause **size distortions**. Consider using the [Fisher test](fisher.md) or checking for cross-sectional dependence with the Pesaran CD test.

!!! warning "Small N"
    The IPS test requires a moderate number of entities for the central limit theorem approximation to work. With very few entities (N < 5), the standardized W-statistic may not be well-approximated by the normal distribution.

!!! warning "Partial Stationarity"
    When IPS rejects H₀, it only indicates that **some** panels are stationary, not which ones or how many. Examine `result.individual_stats` to identify which entities drive the rejection.

## See Also

- [Unit Root Tests Overview](index.md) -- Comparison of all five tests
- [LLC Test](llc.md) -- Common unit root test (homogeneous $\rho$)
- [Fisher Test](fisher.md) -- Alternative heterogeneous test via p-value combination
- [Hadri Test](hadri.md) -- Confirmation test with reversed null
- [Cointegration Tests](../cointegration/index.md) -- Next step if variables are I(1)

## References

- Im, K. S., Pesaran, M. H., & Shin, Y. (2003). "Testing for unit roots in heterogeneous panels." *Journal of Econometrics*, 115(1), 53-74.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 12.
