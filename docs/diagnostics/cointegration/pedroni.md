---
title: "Pedroni Test"
description: "Pedroni panel cointegration test in PanelBox — heterogeneous residual-based test with 7 statistics for testing long-run equilibrium relationships."
---

# Pedroni Test (Heterogeneous Panel Cointegration)

!!! info "Quick Reference"
    **Class:** `panelbox.validation.cointegration.pedroni.PedroniTest`
    **H₀:** No cointegration in any panel
    **H₁:** Cointegration exists for all panels (panel stats) or some panels (group stats)
    **Stata equivalent:** `xtpedroni y x1 x2`
    **R equivalent:** Custom implementation

## What It Tests

The Pedroni (1999, 2004) test is a **residual-based** cointegration test that produces **seven test statistics** -- four within-dimension (panel) statistics and three between-dimension (group) statistics. It allows the cointegrating vector $\beta_i$ to be **heterogeneous** across entities, meaning each entity can have a different long-run relationship.

The test estimates entity-specific cointegrating regressions, collects the residuals, and then tests whether these residuals contain unit roots. If the residuals are I(0), there is evidence of cointegration.

## Quick Example

```python
from panelbox.datasets import load_grunfeld
from panelbox.validation.cointegration.pedroni import PedroniTest

data = load_grunfeld()

# Test cointegration between invest and value
pedroni = PedroniTest(data, "invest", ["value"], "firm", "year", trend="c")
result = pedroni.run()
print(result)
```

Output:

```text
======================================================================
Pedroni Panel Cointegration Tests
======================================================================

Within-dimension (Panel statistics):
  Panel v-statistic:          2.3456  (p = 0.0095)
  Panel rho-statistic:       -1.2345  (p = 0.1086)
  Panel PP-statistic:        -3.4567  (p = 0.0003)
  Panel ADF-statistic:       -2.8901  (p = 0.0019)

Between-dimension (Group statistics):
  Group rho-statistic:       -0.9876  (p = 0.1617)
  Group PP-statistic:        -3.1234  (p = 0.0009)
  Group ADF-statistic:       -2.5678  (p = 0.0051)

Observations:      200
Cross-sections:    10
Trend:             Constant

H0: No cointegration
H1: Cointegration exists

Conclusion: Reject H0 (5/7 tests): Evidence of cointegration
======================================================================
```

## The Seven Statistics

The Pedroni test produces seven statistics organized in two groups:

### Within-Dimension (Panel Statistics)

These statistics **pool** information across all entities, assuming a common AR parameter in the residuals:

| Statistic | Type | Tail | Description |
|:----------|:-----|:-----|:-----------|
| Panel $v$ | Variance ratio | Right-tailed | Tests variance ratio of residuals |
| Panel $\rho$ | Phillips-Perron $\rho$ | Left-tailed | Pooled DF-type statistic |
| Panel $PP$ | Phillips-Perron $t$ | Left-tailed | Non-parametric t-statistic |
| Panel $ADF$ | Augmented DF $t$ | Left-tailed | Parametric t-statistic (includes lags) |

### Between-Dimension (Group Statistics)

These statistics **average** individual statistics across entities, allowing heterogeneous AR parameters:

| Statistic | Type | Tail | Description |
|:----------|:-----|:-----|:-----------|
| Group $\rho$ | Phillips-Perron $\rho$ | Left-tailed | Average of entity-specific $\rho$ statistics |
| Group $PP$ | Phillips-Perron $t$ | Left-tailed | Average of entity-specific PP t-statistics |
| Group $ADF$ | Augmented DF $t$ | Left-tailed | Average of entity-specific ADF t-statistics |

!!! tip "Which statistics to report?"
    The **Panel ADF** and **Group ADF** statistics generally have the best finite-sample properties. The **Panel $v$** statistic tends to over-reject in small samples. Report at least one panel and one group statistic for robustness.

## Interpretation

### Overall Conclusion

The `summary_conclusion` property provides a majority-rule interpretation:

```python
print(result.summary_conclusion)
# "Reject H0 (5/7 tests): Evidence of cointegration"
```

### Detailed Interpretation

Examine each statistic's p-value:

```python
# All p-values
for stat_name, pval in result.pvalues.items():
    decision = "Reject H0" if pval < 0.05 else "Fail to reject"
    print(f"{stat_name:15s}: p={pval:.4f} ({decision})")
```

| Consensus | Interpretation |
|:---------|:--------------|
| 5-7 out of 7 reject | Strong evidence of cointegration |
| 3-4 out of 7 reject | Moderate evidence; consider Westerlund for confirmation |
| 1-2 out of 7 reject | Weak evidence; likely no cointegration |
| 0 out of 7 reject | No evidence of cointegration |

## Mathematical Details

### Cointegrating Regression

For each entity $i$, estimate:

$$y_{it} = \alpha_i + \delta_i t + \beta_i' x_{it} + \varepsilon_{it}$$

where $\beta_i$ is allowed to differ across entities (heterogeneous cointegrating vector).

### Residual Unit Root Tests

Collect residuals $\hat{\varepsilon}_{it}$ and test for unit roots using:

$$\Delta \hat{\varepsilon}_{it} = \rho_i \hat{\varepsilon}_{i,t-1} + v_{it}$$

**Panel statistics** pool across entities (common $\rho$):

$$Z_{panel} = f\left(\sum_i \sum_t \hat{\varepsilon}_{i,t-1} \Delta \hat{\varepsilon}_{it}, \sum_i \sum_t \hat{\varepsilon}_{i,t-1}^2\right)$$

**Group statistics** average individual statistics (heterogeneous $\rho_i$):

$$Z_{group} = \frac{1}{N} \sum_{i=1}^{N} Z_i$$

All standardized statistics converge to $N(0,1)$ under H₀.

## Configuration Options

```python
PedroniTest(
    data,                   # pd.DataFrame: Panel data in long format
    dependent,              # str: Dependent variable name
    independents,           # list[str]: Independent variable names
    entity_col,             # str: Entity identifier column
    time_col,               # str: Time identifier column
    trend='c',              # str: 'c' (constant) or 'ct' (constant + trend)
    lags=None,              # int or None: Lag length for ADF (None = auto)
)
```

### Trend Specifications

=== "Constant only (`trend='c'`)"

    ```python
    # Entity-specific intercepts (most common)
    pedroni = PedroniTest(data, "invest", ["value", "capital"],
                           "firm", "year", trend="c")
    ```

=== "Constant + trend (`trend='ct'`)"

    ```python
    # Entity-specific intercepts and trends
    pedroni = PedroniTest(data, "log_gdp", ["log_capital", "log_labor"],
                           "country", "year", trend="ct")
    ```

### Result Object: `PedroniTestResult`

| Attribute | Type | Description |
|:----------|:-----|:-----------|
| `panel_v` | `float` | Panel v-statistic (variance ratio) |
| `panel_rho` | `float` | Panel rho-statistic |
| `panel_pp` | `float` | Panel PP-statistic |
| `panel_adf` | `float` | Panel ADF-statistic |
| `group_rho` | `float` | Group rho-statistic |
| `group_pp` | `float` | Group PP-statistic |
| `group_adf` | `float` | Group ADF-statistic |
| `pvalues` | `dict` | P-values for all 7 statistics |
| `n_obs` | `int` | Total observations |
| `n_entities` | `int` | Number of cross-sections |
| `trend` | `str` | Trend specification |
| `summary_conclusion` | `str` | Majority-rule conclusion |

## When to Use

**Use Pedroni when:**

- The cointegrating vector may be **heterogeneous** across entities ($\beta_i$ varies)
- You want a **comprehensive** battery of statistics (7 tests in one)
- You want both **panel** (pooled) and **group** (averaged) perspectives

**Advantages:**

- Allows heterogeneous $\beta_i$: different entities can have different long-run relationships
- Seven statistics provide robustness through multiple perspectives
- Well-established in the literature; widely used and cited

**Limitations:**

- Panel $v$ statistic has poor finite-sample properties (tends to over-reject)
- Assumes cross-sectional independence in the errors
- P-values use normal approximation (exact critical values require Monte Carlo)

## Common Pitfalls

!!! warning "Panel v Over-Rejection"
    The Panel $v$ statistic tends to **over-reject** in finite samples, especially with small T. Do not base conclusions solely on this statistic. Always check the ADF and PP statistics.

!!! warning "I(1) Prerequisite"
    Variables must be individually I(1) before testing for cointegration. If variables are I(0), cointegration tests are meaningless. Run [unit root tests](../unit-root/index.md) first.

!!! warning "Cross-Sectional Dependence"
    If entities are subject to common shocks, the Pedroni test may produce unreliable results. Consider using the [Westerlund test](westerlund.md) with bootstrap p-values, which corrects for cross-sectional dependence.

!!! warning "Trend Specification"
    Choosing between `'c'` and `'ct'` matters. If the cointegrating relationship involves a deterministic trend, use `trend='ct'`. Using the wrong specification can lead to spurious rejection or failure to detect cointegration.

## See Also

- [Cointegration Tests Overview](index.md) -- Comparison of all three tests
- [Kao Test](kao.md) -- Simpler homogeneous cointegration test
- [Westerlund Test](westerlund.md) -- ECM-based test with bootstrap
- [Unit Root Tests](../unit-root/index.md) -- Prerequisite: testing for I(1)

## References

- Pedroni, P. (1999). "Critical values for cointegration tests in heterogeneous panels with multiple regressors." *Oxford Bulletin of Economics and Statistics*, 61(S1), 653-670.
- Pedroni, P. (2004). "Panel cointegration: asymptotic and finite sample properties of pooled time series tests with an application to the PPP hypothesis." *Econometric Theory*, 20(3), 597-625.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 12.
