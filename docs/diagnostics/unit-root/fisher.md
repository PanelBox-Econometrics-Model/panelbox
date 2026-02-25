---
title: "Fisher Test"
description: "Fisher-type panel unit root test combining individual p-values in PanelBox — Maddala-Wu test supporting ADF and Phillips-Perron variants for unbalanced panels."
---

# Fisher Test (Maddala-Wu)

!!! info "Quick Reference"
    **Class:** `panelbox.validation.unit_root.fisher.FisherTest`
    **H₀:** All panels contain unit roots
    **H₁:** At least one panel is stationary
    **Stata equivalent:** `xtunitroot fisher variable, dfuller lags(p)` or `xtunitroot fisher variable, pperron`
    **R equivalent:** `plm::purtest(x, test="madwu")`

## What It Tests

The Fisher test combines p-values from **individual** unit root tests (ADF or Phillips-Perron) across entities using the inverse chi-square transformation. It tests whether all panels have unit roots against the alternative that at least one panel is stationary.

The test is highly flexible: it works with **unbalanced panels**, allows **heterogeneous lag structures**, and imposes no restriction on the cross-sectional structure of the autoregressive parameters.

## Quick Example

```python
from panelbox.datasets import load_grunfeld
from panelbox.validation.unit_root.fisher import FisherTest

data = load_grunfeld()

# Fisher-ADF test
fisher = FisherTest(data, "invest", "firm", "year", test_type="adf", trend="c")
result = fisher.run()
print(result)
```

Output:

```text
======================================================================
Fisher-type Panel Unit Root Test
======================================================================
Test type:         ADF
Fisher statistic:     45.2381
P-value:              0.0012

Cross-sections:    10
Trend:             c

H0: All series have unit roots
H1: At least one series is stationary

Conclusion: Reject H0 at 5.0% level: Evidence against unit root
======================================================================
```

## Interpretation

| p-value | Decision | Meaning |
|:--------|:---------|:--------|
| $p < 0.01$ | Strong rejection of H₀ | Strong evidence that at least one panel is stationary |
| $0.01 \leq p < 0.05$ | Rejection of H₀ | Evidence against universal unit root |
| $0.05 \leq p < 0.10$ | Borderline | Weak evidence; consider additional tests |
| $p \geq 0.10$ | Fail to reject H₀ | Consistent with unit root in all panels |

### Examining Individual Results

After running the test, inspect which entities drive the rejection:

```python
# Check individual p-values
for entity, pval in result.individual_pvalues.items():
    status = "stationary" if pval < 0.05 else "unit root"
    print(f"Entity {entity}: p={pval:.4f} ({status})")
```

- If **most** $p_i < 0.05$: Strong panel-wide stationarity
- If **few** $p_i < 0.05$: Only some entities are stationary
- If **all** $p_i > 0.10$: Strong evidence for universal unit root

## Mathematical Details

### Test Statistic

The Fisher test statistic combines individual p-values using the inverse chi-square transformation:

$$P = -2 \sum_{i=1}^{N} \ln(p_i) \sim \chi^2(2N) \quad \text{under } H_0$$

where $p_i$ is the p-value from the individual unit root test (ADF or Phillips-Perron) for entity $i$.

### Intuition

- Under H₀, each $p_i$ is uniformly distributed on $[0, 1]$
- Therefore, $-2 \ln(p_i)$ follows a $\chi^2(2)$ distribution
- Summing over $N$ independent entities gives $\chi^2(2N)$
- Large values of $P$ (many small $p_i$) indicate rejection of the unit root

### Individual Tests

=== "ADF (`test_type='adf'`)"

    The Augmented Dickey-Fuller test for each entity $i$:

    $$\Delta y_{it} = \rho_i y_{i,t-1} + \sum_{j=1}^{p_i} \theta_{ij} \Delta y_{i,t-j} + \alpha_i + \varepsilon_{it}$$

    Uses `statsmodels.tsa.stattools.adfuller` internally with AIC-based lag selection.

=== "Phillips-Perron (`test_type='pp'`)"

    The Phillips-Perron test uses a nonparametric correction for serial correlation:

    $$y_t = \rho y_{t-1} + \varepsilon_t$$

    with Newey-West adjusted standard errors.

## Configuration Options

```python
FisherTest(
    data,                   # pd.DataFrame: Panel data in long format
    variable,               # str: Variable to test for unit root
    entity_col,             # str: Entity identifier column
    time_col,               # str: Time identifier column
    test_type='adf',        # str: 'adf' (Augmented Dickey-Fuller) or 'pp' (Phillips-Perron)
    lags=None,              # int or None: Lag length (None = AIC selection for ADF)
    trend='c',              # str: 'n' (none), 'c' (constant), 'ct' (constant + trend)
)
```

### ADF vs. Phillips-Perron

```python
# ADF-based Fisher test
fisher_adf = FisherTest(data, "invest", "firm", "year",
                         test_type="adf", trend="c")
result_adf = fisher_adf.run()
print(f"Fisher-ADF: stat={result_adf.statistic:.4f}, p={result_adf.pvalue:.4f}")

# PP-based Fisher test
fisher_pp = FisherTest(data, "invest", "firm", "year",
                        test_type="pp", trend="c")
result_pp = fisher_pp.run()
print(f"Fisher-PP:  stat={result_pp.statistic:.4f}, p={result_pp.pvalue:.4f}")
```

| Feature | ADF | Phillips-Perron |
|:--------|:----|:----------------|
| Serial correlation | Parametric (lags) | Nonparametric (kernel) |
| Lag selection | AIC (automatic) | Not needed |
| Power | Good for AR processes | Good for MA processes |
| Recommended | Default choice | When serial correlation structure is unknown |

### Result Object: `FisherTestResult`

| Attribute | Type | Description |
|:----------|:-----|:-----------|
| `statistic` | `float` | Fisher chi-square statistic |
| `pvalue` | `float` | P-value from $\chi^2(2N)$ distribution |
| `individual_pvalues` | `dict` | P-values from individual unit root tests per entity |
| `n_entities` | `int` | Number of cross-sectional units |
| `test_type` | `str` | `"adf"` or `"pp"` |
| `trend` | `str` | Trend specification used |
| `conclusion` | `str` | Test conclusion |

## When to Use

**Use Fisher when:**

- The panel is **unbalanced** (different $T$ per entity)
- You want maximum **flexibility** (heterogeneous lags, dynamics)
- You want to **inspect individual entity results**
- You want to compare ADF vs. PP approaches

**Advantages:**

- Works with **unbalanced panels** naturally
- No restriction on cross-sectional structure
- Heterogeneous lag selection per entity
- Simple and distribution-free under H₀

**Disadvantages:**

- **Less powerful** than LLC/IPS when common $\rho$ or homogeneity holds
- Assumes **cross-sectional independence**
- Chi-square approximation may be poor with very few entities

## Testing First Differences

A common pattern is to test both levels and first differences:

```python
# Test levels (expect unit root)
fisher_levels = FisherTest(data, "invest", "firm", "year",
                            test_type="adf", trend="c")
result_levels = fisher_levels.run()

# Test first differences (expect stationary)
data["d_invest"] = data.groupby("firm")["invest"].diff()
data_diff = data.dropna(subset=["d_invest"])

fisher_diff = FisherTest(data_diff, "d_invest", "firm", "year",
                          test_type="adf", trend="c")
result_diff = fisher_diff.run()

print(f"Levels:  p={result_levels.pvalue:.4f}  (expect high)")
print(f"Diffs:   p={result_diff.pvalue:.4f}  (expect low)")
```

If levels show unit root ($p \geq 0.05$) and first differences are stationary ($p < 0.05$), the variable is I(1).

## Common Pitfalls

!!! warning "Cross-Sectional Independence"
    The Fisher test assumes independence across entities. Common factors or cross-sectional dependence inflate the test statistic, leading to **over-rejection**. Test for cross-sectional dependence first using the Pesaran CD test.

!!! warning "Small Entity Count"
    With very few entities (N < 5), the chi-square approximation $P \sim \chi^2(2N)$ may be inaccurate. The Choi (2001) Z-statistic (normal approximation) is an alternative for small N.

!!! warning "Conservative P-values"
    When an individual ADF test fails for an entity (e.g., too few observations), the Fisher test assigns $p_i = 1.0$ (conservative). Check `result.individual_pvalues` to ensure no entity is driving results through default values.

## See Also

- [Unit Root Tests Overview](index.md) -- Comparison of all five tests
- [LLC Test](llc.md) -- Common unit root test (more powerful if homogeneity holds)
- [IPS Test](ips.md) -- Alternative heterogeneous test via t-bar averaging
- [Hadri Test](hadri.md) -- Confirmation test with reversed null
- [Cointegration Tests](../cointegration/index.md) -- Next step if variables are I(1)

## References

- Maddala, G. S., & Wu, S. (1999). "A comparative study of unit root tests with panel data and a new simple test." *Oxford Bulletin of Economics and Statistics*, 61(S1), 631-652.
- Choi, I. (2001). "Unit root tests for panel data." *Journal of International Money and Finance*, 20(2), 249-272.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 12.
