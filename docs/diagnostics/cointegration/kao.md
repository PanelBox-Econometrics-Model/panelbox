---
title: "Kao Test"
description: "Kao panel cointegration test in PanelBox — ADF-type residual-based test assuming homogeneous cointegrating vectors across panel entities."
---

# Kao Test (Homogeneous Panel Cointegration)

!!! info "Quick Reference"
    **Class:** `panelbox.validation.cointegration.kao.KaoTest`
    **H₀:** No cointegration (residuals are I(1))
    **H₁:** Cointegration exists (residuals are I(0), homogeneous $\beta$)
    **Stata equivalent:** `xtcointtest kao y x1 x2`
    **R equivalent:** Custom implementation

## What It Tests

The Kao (1999) test is a **residual-based** cointegration test that assumes a **homogeneous cointegrating vector** across all entities. It estimates a pooled cointegrating regression, collects the residuals, and applies an ADF-type test to check whether the residuals are stationary.

Unlike the [Pedroni test](pedroni.md) which allows different $\beta_i$ per entity, the Kao test restricts $\beta$ to be the same for all entities. This makes it simpler and more powerful when the homogeneity assumption holds, but too restrictive when entities have genuinely different long-run relationships.

## Quick Example

```python
from panelbox.datasets import load_grunfeld
from panelbox.validation.cointegration.kao import KaoTest

data = load_grunfeld()

# Test cointegration between invest and value
kao = KaoTest(data, "invest", ["value"], "firm", "year", trend="c")
result = kao.run()
print(result)
```

Output:

```text
======================================================================
Kao Panel Cointegration Test
======================================================================
ADF statistic:     -3.2145
P-value:           0.0007
Observations:      200
Cross-sections:    10
Trend:             Constant

H0: No cointegration
H1: Cointegration exists

Conclusion: Reject H0: Evidence of cointegration
======================================================================
```

## Interpretation

| p-value | Decision | Meaning |
|:--------|:---------|:--------|
| $p < 0.01$ | Strong rejection of H₀ | Strong evidence of cointegration |
| $0.01 \leq p < 0.05$ | Rejection of H₀ | Evidence of cointegration |
| $0.05 \leq p < 0.10$ | Borderline | Weak evidence; run additional tests |
| $p \geq 0.10$ | Fail to reject H₀ | No evidence of cointegration |

## Mathematical Details

### Cointegrating Regression

The Kao test estimates a **pooled** cointegrating regression with entity-specific intercepts:

$$y_{it} = \alpha_i + \beta' x_{it} + \varepsilon_{it}$$

where $\beta$ is **common** across all entities (homogeneity).

### ADF Test on Residuals

The test then examines whether the pooled residuals have a unit root:

$$\Delta \hat{\varepsilon}_{it} = \rho \hat{\varepsilon}_{i,t-1} + v_{it}$$

The t-statistic for $\rho$ is adjusted for panel structure using Kao's correction factors:

$$t_{Kao}^{*} = \frac{t_\rho - \sqrt{NT_{avg}} \cdot \mu}{\sigma \cdot \sqrt{N}} \xrightarrow{d} N(0, 1)$$

where $\mu$ and $\sigma$ are adjustment parameters from Kao (1999) that depend on the trend specification.

### Adjustment Parameters

| Trend | $\mu$ (approx.) | $\sigma$ (approx.) |
|:------|:----------------|:-------------------|
| Constant (`'c'`) | $-1.25$ | $1.00$ |
| Constant + trend (`'ct'`) | $-1.75$ | $1.10$ |

## Configuration Options

```python
KaoTest(
    data,                   # pd.DataFrame: Panel data in long format
    dependent,              # str: Dependent variable name
    independents,           # list[str]: Independent variable names
    entity_col,             # str: Entity identifier column
    time_col,               # str: Time identifier column
    trend='c',              # str: 'c' (constant) or 'ct' (constant + trend)
)
```

### Result Object: `KaoTestResult`

| Attribute | Type | Description |
|:----------|:-----|:-----------|
| `statistic` | `float` | Kao ADF test statistic (adjusted) |
| `pvalue` | `float` | P-value from standard normal (left-tailed) |
| `n_obs` | `int` | Total observations |
| `n_entities` | `int` | Number of cross-sections |
| `trend` | `str` | Trend specification |
| `null_hypothesis` | `str` | `"No cointegration"` |
| `alternative_hypothesis` | `str` | `"Cointegration exists"` |
| `conclusion` | `str` | Test conclusion at 5% level |

## When to Use

**Use Kao when:**

- The cointegrating vector is likely **homogeneous** ($\beta$ is the same for all entities)
- You want a **quick, simple** check with a single test statistic
- The panel is relatively **homogeneous** (similar entities)

**Examples of appropriate settings:**

- Firms in the same industry with similar cost structures
- Countries in the same economic bloc with harmonized policies
- Regions within a country with similar institutions

## Comparing Kao and Pedroni

```python
from panelbox.validation.cointegration.pedroni import PedroniTest
from panelbox.validation.cointegration.kao import KaoTest

data = load_grunfeld()

# Kao: homogeneous beta
kao = KaoTest(data, "invest", ["value", "capital"],
               "firm", "year", trend="c")
kao_result = kao.run()

# Pedroni: heterogeneous beta_i
pedroni = PedroniTest(data, "invest", ["value", "capital"],
                       "firm", "year", trend="c")
pedroni_result = pedroni.run()

print(f"Kao:     stat={kao_result.statistic:.4f}, p={kao_result.pvalue:.4f}")
print(f"Pedroni: {pedroni_result.summary_conclusion}")
```

| Aspect | Kao | Pedroni |
|:-------|:----|:--------|
| $\beta$ | Common (homogeneous) | Heterogeneous $\beta_i$ |
| Statistics | 1 (ADF-type) | 7 (panel + group) |
| Power | Higher if homogeneity holds | Lower but more flexible |
| Complexity | Simple | Comprehensive |
| Sensitivity | More sensitive to misspecification | More robust to heterogeneity |

## Common Pitfalls

!!! warning "Homogeneity Assumption"
    The Kao test assumes the cointegrating vector $\beta$ is the **same** for all entities. If entities have genuinely different long-run relationships, the test may produce misleading results. Use the [Pedroni test](pedroni.md) or [Westerlund test](westerlund.md) for heterogeneous panels.

!!! warning "Over-Rejection in Finite Samples"
    The Kao test tends to **over-reject** in finite samples, especially with small T. Cross-check with other tests before concluding.

!!! warning "I(1) Prerequisite"
    All variables must be individually I(1). Run [unit root tests](../unit-root/index.md) before applying the Kao test. Cointegration between I(0) variables is meaningless.

!!! warning "Cross-Sectional Dependence"
    Like the Pedroni test, the Kao test assumes cross-sectional independence. If entities are subject to common shocks, consider using the [Westerlund test](westerlund.md) with bootstrap p-values.

## See Also

- [Cointegration Tests Overview](index.md) -- Comparison of all three tests
- [Pedroni Test](pedroni.md) -- Heterogeneous cointegration test (7 statistics)
- [Westerlund Test](westerlund.md) -- ECM-based test with bootstrap
- [Unit Root Tests](../unit-root/index.md) -- Prerequisite: testing for I(1)

## References

- Kao, C. (1999). "Spurious regression and residual-based tests for cointegration in panel data." *Journal of Econometrics*, 90(1), 1-44.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 12.
