---
title: "Granger Causality"
description: "Guide to Granger causality testing in Panel VAR models with PanelBox: standard Wald tests, Dumitrescu-Hurlin heterogeneous test, and causality networks."
---

# Granger Causality

!!! info "Quick Reference"
    **Classes:** `panelbox.var.causality.GrangerCausalityResult`, `panelbox.var.causality.DumitrescuHurlinResult`
    **Methods:** `PanelVARResult.granger_causality()`, `PanelVARResult.dumitrescu_hurlin()`
    **Import:** `from panelbox.var import PanelVAR`
    **Stata equivalent:** `pvargranger`
    **R equivalent:** `panelvar::pvargmm()` + Granger tests

## Overview

Granger causality tests whether the past values of one variable help predict another variable beyond what the other variable's own past values can predict. In a Panel VAR context, this provides a formal statistical framework for testing **predictive relationships** between variables.

Given the Panel VAR equation for variable $Y$:

$$
Y_{it} = \sum_{l=1}^{p} \gamma_l Y_{i,t-l} + \sum_{l=1}^{p} \beta_l X_{i,t-l} + \alpha_i + \varepsilon_{it}
$$

**Granger causality** from $X$ to $Y$ tests:

$$
H_0: \beta_1 = \beta_2 = \ldots = \beta_p = 0 \quad \text{(X does not Granger-cause Y)}
$$

PanelBox provides two complementary approaches:

1. **Standard Panel Wald test**: Assumes homogeneous coefficients across entities (pooled test)
2. **Dumitrescu-Hurlin (2012) test**: Allows for **heterogeneous** causality across entities -- causality may exist for some entities but not others

## Quick Example

```python
from panelbox.var import PanelVARData, PanelVAR

# Estimate model
var_data = PanelVARData(df, endog_vars=["gdp", "inflation", "rate"],
                         entity_col="country", time_col="year", lags=2)
model = PanelVAR(data=var_data)
results = model.fit(cov_type="clustered")

# Standard Granger causality test
gc = results.granger_causality(cause="inflation", effect="gdp")
print(gc.summary())

# Dumitrescu-Hurlin heterogeneous test
dh = results.dumitrescu_hurlin(cause="inflation", effect="gdp")
print(dh.summary())

# Full causality matrix (all pairs)
gc_matrix = results.granger_causality_matrix()
print(gc_matrix)
```

## When to Use

- **Predictive relationships**: Does knowing past inflation help predict future GDP growth?
- **Policy analysis**: Does monetary policy (interest rate) Granger-cause output?
- **Identifying variable ordering**: Granger causality can inform the ordering for Cholesky IRFs
- **Model specification**: Variables with no causal relationship may be excluded from the system
- **Heterogeneity**: Use Dumitrescu-Hurlin when causality may differ across entities

!!! warning "Key Assumptions"
    - **Granger causality is NOT structural causality**: It measures predictive content, not true causal mechanisms
    - **Sensitive to lag length**: Results may change with different lag orders
    - **Requires stationarity**: Variables must be stationary (or use VECM for cointegrated variables)
    - **Omitted variables**: Missing relevant variables can create spurious Granger causality

## Detailed Guide

### Standard Panel Granger Causality (Wald Test)

The standard test assumes **homogeneous** coefficients across all entities and tests joint significance of all lags of the causing variable in the equation of the effect variable.

```python
gc = results.granger_causality(cause="inflation", effect="gdp")
print(gc.summary())
```

**Test statistics:**

- **Wald statistic**: $W = (R\hat{\beta})' [R \cdot \text{Var}(\hat{\beta}) \cdot R']^{-1} (R\hat{\beta}) \sim \chi^2(p)$
- **F-statistic**: $F = W / p$

The restriction matrix $R$ selects all $p$ lag coefficients of the causing variable in the effect variable's equation.

**GrangerCausalityResult attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `cause` | `str` | Causing variable name |
| `effect` | `str` | Effect variable name |
| `wald_stat` | `float` | Wald test statistic |
| `f_stat` | `float` | F-statistic ($W/p$) |
| `df` | `int` | Degrees of freedom (number of lags) |
| `p_value` | `float` | P-value from $\chi^2$ distribution |
| `p_value_f` | `float` | P-value from F distribution |
| `conclusion` | `str` | Statistical conclusion |
| `lags_tested` | `int` | Number of lags tested |

```python
# Access individual results
print(f"Wald stat: {gc.wald_stat:.4f}")
print(f"F-stat: {gc.f_stat:.4f}")
print(f"P-value: {gc.p_value:.4f}")
print(f"Conclusion: {gc.conclusion}")
```

### All-Pairs Causality Matrix

Test Granger causality for all variable pairs simultaneously:

```python
# P-value matrix (K x K)
gc_matrix = results.granger_causality_matrix(significance_level=0.05)
print(gc_matrix)
```

The resulting DataFrame has p-values where element $(i, j)$ is the p-value for testing whether variable $i$ Granger-causes variable $j$. Diagonal elements are `NaN`.

### Dumitrescu-Hurlin (2012) Heterogeneous Test

The standard Wald test assumes that all entities share the same causal relationship. The **Dumitrescu-Hurlin (DH)** test relaxes this assumption by allowing for **heterogeneous** coefficients across entities.

The DH procedure:

1. For each entity $i$, estimate an individual bivariate regression and compute entity-specific Wald statistic $W_i$
2. Compute the average: $\bar{W} = \frac{1}{N} \sum_{i=1}^{N} W_i$
3. Standardize using two statistics:
   - $\tilde{Z} = \sqrt{\frac{N}{2p}} (\bar{W} - p)$ -- for fixed $T$, $N \to \infty$
   - $\bar{Z} = \sqrt{N} \frac{\bar{W} - E[W_i]}{Var[W_i]}$ -- for $T \to \infty$, $N \to \infty$

Both test statistics are asymptotically standard normal under $H_0$.

```python
dh = results.dumitrescu_hurlin(cause="inflation", effect="gdp")
print(dh.summary())
```

**DumitrescuHurlinResult attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `cause` | `str` | Causing variable |
| `effect` | `str` | Effect variable |
| `W_bar` | `float` | Average Wald statistic across entities |
| `Z_tilde_stat` | `float` | $\tilde{Z}$ statistic (for fixed $T$, $N \to \infty$) |
| `Z_tilde_pvalue` | `float` | P-value for $\tilde{Z}$ |
| `Z_bar_stat` | `float` | $\bar{Z}$ statistic (for $T \to \infty$, $N \to \infty$) |
| `Z_bar_pvalue` | `float` | P-value for $\bar{Z}$ |
| `individual_W` | `np.ndarray` | Per-entity Wald statistics |
| `recommended_stat` | `str` | `"Z_tilde"` or `"Z_bar"` (automatic selection) |
| `N` | `int` | Number of entities |
| `T_avg` | `float` | Average time periods |
| `lags` | `int` | Number of lags tested |

!!! tip "Which Statistic to Use?"
    PanelBox automatically recommends the appropriate statistic based on the sample:

    - **$\tilde{Z}$ (`Z_tilde`)**: Use when $T$ is small (< 10) relative to $N$
    - **$\bar{Z}$ (`Z_bar`)**: Use when both $T$ and $N$ are large

    The `recommended_stat` attribute tells you which to use.

### Visualizing Individual Heterogeneity

The DH test provides per-entity Wald statistics, revealing the distribution of causality strength across entities:

```python
# Plot histogram of individual Wald statistics
dh.plot_individual_statistics(backend="matplotlib")

# Access individual statistics
for i, w in enumerate(dh.individual_W):
    print(f"Entity {i}: W = {w:.4f}")
```

The plot shows the distribution of $W_i$ values with the 5% critical value and the average $\bar{W}$ marked. Entities above the critical value exhibit individual Granger causality.

### Instantaneous Causality

Test for contemporaneous (same-period) correlation between variables:

```python
# Single pair
ic = results.instantaneous_causality(var1="gdp", var2="inflation")
print(ic.summary())

# Full matrix
corr_matrix, pvalue_matrix = results.instantaneous_causality_matrix()
print("Correlation matrix:")
print(corr_matrix)
print("\nP-value matrix:")
print(pvalue_matrix)
```

The test uses the likelihood ratio statistic:

$$
LR = -NT \ln(1 - r^2) \sim \chi^2(1)
$$

where $r$ is the correlation between residuals of the two equations.

### Causality Network Visualization

Visualize all significant Granger causality relationships as a directed network graph:

```python
# Interactive network plot
results.plot_causality_network(
    threshold=0.05,           # Significance threshold
    layout="circular",        # "circular", "spring", "kamada_kawai", "shell"
    backend="plotly",         # "plotly" or "matplotlib"
)
```

The network shows:

- **Nodes**: Variables
- **Directed edges**: Significant Granger causality relationships ($p < \text{threshold}$)
- **Edge thickness**: Inversely proportional to p-value (stronger significance = thicker edge)
- **Edge color**: Dark green ($p < 0.01$), green ($p < 0.05$), orange ($p < 0.10$)

!!! note "Requirement"
    Network visualization requires `networkx`: `pip install networkx`

### Comparison: Standard vs Dumitrescu-Hurlin

| Feature | Standard Wald | Dumitrescu-Hurlin |
|---------|--------------|-------------------|
| Coefficients | Homogeneous (pooled) | Heterogeneous (per-entity) |
| Null hypothesis | $\beta_l = 0$ for all lags | $\beta_{il} = 0$ for all entities and lags |
| Alternative | Causality for all entities | Causality for **at least some** entities |
| Requires | Fitted Panel VAR | Raw panel data (re-estimates per entity) |
| Small-sample | More powerful (if homogeneous) | More reliable under heterogeneity |
| Entity-level results | No | Yes (`individual_W`) |

## Common Pitfalls

!!! warning "Granger Causality is NOT True Causality"
    Granger causality is a statistical concept about **predictive content**, not about structural or mechanistic causation. Variable $X$ Granger-causing $Y$ means that past values of $X$ contain information useful for predicting $Y$ beyond $Y$'s own history. This could be due to:

    - True causal effect
    - Common unobserved factors
    - Third variable driving both

!!! warning "Sensitivity to Lag Length"
    Results can change substantially with different lag orders. Always:

    1. Use information criteria (BIC) to select the lag order first
    2. Test robustness with $p \pm 1$ lags
    3. Report the lag length alongside results

## Complete Workflow Example

```python
import pandas as pd
from panelbox.var import PanelVARData, PanelVAR

# Load data
df = pd.read_csv("macro_panel.csv")

# Estimate Panel VAR
var_data = PanelVARData(
    data=df,
    endog_vars=["gdp_growth", "inflation", "interest_rate"],
    entity_col="country",
    time_col="year",
    lags=2,
)
model = PanelVAR(data=var_data)
results = model.fit(cov_type="clustered")

# 1. Standard Granger causality -- all pairs
print("=== Standard Granger Causality (Wald) ===\n")
gc_matrix = results.granger_causality_matrix()
print(gc_matrix)

# 2. Dumitrescu-Hurlin -- allows heterogeneous causality
print("\n=== Dumitrescu-Hurlin Heterogeneous Test ===\n")
pairs = [
    ("inflation", "gdp_growth"),
    ("interest_rate", "gdp_growth"),
    ("gdp_growth", "inflation"),
    ("interest_rate", "inflation"),
]

for cause, effect in pairs:
    dh = results.dumitrescu_hurlin(cause=cause, effect=effect)
    rec_stat = dh.recommended_stat
    rec_pval = dh.Z_tilde_pvalue if rec_stat == "Z_tilde" else dh.Z_bar_pvalue
    sig = "***" if rec_pval < 0.01 else "**" if rec_pval < 0.05 else "*" if rec_pval < 0.10 else ""
    print(f"{cause} -> {effect}: W_bar={dh.W_bar:.3f}, "
          f"{rec_stat} p={rec_pval:.4f} {sig}")

# 3. Instantaneous causality
print("\n=== Instantaneous Causality ===\n")
corr, pvals = results.instantaneous_causality_matrix()
print("Correlations:")
print(corr)
print("\nP-values:")
print(pvals)

# 4. Network visualization
results.plot_causality_network(threshold=0.05, layout="circular")
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Granger Causality | Standard and DH tests with visualization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/04_granger_causality.ipynb) |

## See Also

- [Panel VAR Estimation](estimation.md) -- Model setup and estimation
- [Impulse Response Functions](irf.md) -- Dynamic effects of shocks (Granger causality can inform ordering)
- [FEVD](fevd.md) -- Relative importance of shocks
- [VECM](vecm.md) -- Causality in cointegrated systems
- [Forecasting](forecasting.md) -- Multi-step predictions

## References

- Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424-438.
- Dumitrescu, E. I., & Hurlin, C. (2012). Testing for Granger non-causality in heterogeneous panels. *Economic Modelling*, 29(4), 1450-1460.
- Holtz-Eakin, D., Newey, W., & Rosen, H. S. (1988). Estimating vector autoregressions with panel data. *Econometrica*, 56(6), 1371-1395.
- Lopez, L., & Weber, S. (2017). Testing for Granger causality in panel data. *The Stata Journal*, 17(4), 972-984.
