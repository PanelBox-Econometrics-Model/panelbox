---
title: "Forecast Error Variance Decomposition"
description: "Guide to computing and interpreting FEVD in Panel VAR models with PanelBox: relative importance of shocks across horizons."
---

# Forecast Error Variance Decomposition

!!! info "Quick Reference"
    **Class:** `panelbox.var.fevd.FEVDResult`
    **Method:** `PanelVARResult.fevd()`
    **Import:** `from panelbox.var import PanelVAR` (FEVD accessed via results)
    **Stata equivalent:** `irf table fevd`
    **R equivalent:** `vars::fevd()`

## Overview

Forecast Error Variance Decomposition (FEVD) measures the **proportion of the forecast error variance** of each variable that is attributable to shocks from each variable in the system. While IRFs trace the effect of a single shock over time, FEVD answers: *What fraction of the uncertainty in forecasting variable $i$ at horizon $h$ is due to shocks in variable $j$?*

At each horizon $h$ and for each variable $i$, the FEVD sums to 100%:

$$
\sum_{j=1}^{K} \omega_{ij}(h) = 1
$$

where $\omega_{ij}(h)$ is the share of forecast error variance of variable $i$ at horizon $h$ explained by shocks to variable $j$.

For Cholesky-identified FEVD:

$$
\omega_{ij}(h) = \frac{\sum_{s=0}^{h} (\Phi_s^{\text{orth}}[i,j])^2}{\sum_{s=0}^{h} \sum_{k=1}^{K} (\Phi_s^{\text{orth}}[i,k])^2}
$$

FEVD provides a complementary perspective to IRFs: IRFs show the direction and magnitude of effects, while FEVD shows their relative importance.

## Quick Example

```python
from panelbox.var import PanelVARData, PanelVAR

# Estimate model
var_data = PanelVARData(df, endog_vars=["gdp", "inflation", "rate"],
                         entity_col="country", time_col="year", lags=2)
model = PanelVAR(data=var_data)
results = model.fit(cov_type="clustered")

# Compute FEVD
fevd = results.fevd(periods=20, method="cholesky")

# View decomposition for GDP
print(fevd["gdp"])    # Returns (periods+1, K) array

# Summary at selected horizons
print(fevd.summary(horizons=[1, 5, 10, 20]))

# Plot
fevd.plot()
```

## When to Use

- **Relative importance**: Which variables drive the most uncertainty in your variable of interest?
- **Exogeneity assessment**: If a variable's forecast error is dominated by own shocks at all horizons, it behaves as (nearly) exogenous
- **Shock dominance**: At what horizon do external shocks start to dominate own shocks?
- **Model validation**: FEVD patterns should be consistent with economic theory and IRF findings

!!! warning "Key Assumptions"
    - Same identification assumptions as IRFs (Cholesky ordering or Generalized)
    - VAR must be stable for well-defined FEVD
    - Cholesky FEVD depends on variable ordering (same as Cholesky IRFs)
    - Generalized FEVD is order-invariant but requires normalization (rows may not sum to 1 before normalization)

## Detailed Guide

### Computing FEVD

```python
fevd = results.fevd(
    periods=20,               # Number of horizons
    method="cholesky",        # "cholesky" or "generalized"
    order=None,               # Custom ordering (Cholesky only)
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `periods` | `int` | `20` | Number of horizons to compute |
| `method` | `str` | `"cholesky"` | `"cholesky"` or `"generalized"` |
| `order` | `list[str]` | `None` | Variable ordering for Cholesky |

### Identification Methods

=== "Cholesky FEVD"

    Based on orthogonalized IRFs from Cholesky decomposition. Shares sum exactly to 100% at every horizon.

    ```python
    fevd = results.fevd(periods=20, method="cholesky")

    # Custom ordering
    fevd = results.fevd(
        periods=20,
        method="cholesky",
        order=["rate", "inflation", "gdp"],
    )
    ```

=== "Generalized FEVD"

    Based on Generalized IRFs (Pesaran-Shin). Order-invariant but raw shares may not sum to 100%. PanelBox automatically **normalizes** the shares so each row sums to 1.

    ```python
    fevd = results.fevd(periods=20, method="generalized")
    ```

### Accessing Results

```python
# Access FEVD for a specific variable
gdp_fevd = fevd["gdp"]           # np.ndarray (periods+1, K)
# gdp_fevd[h, j] = share of GDP variance at horizon h from shock j

# Convert to DataFrame
df_gdp = fevd.to_dataframe(variable="gdp")
df_gdp_horizons = fevd.to_dataframe(variable="gdp", horizons=[1, 5, 10, 20])
df_all = fevd.to_dataframe(horizons=[1, 5, 10, 20])

# Summary table
print(fevd.summary(horizons=[1, 5, 10, 20]))
```

**FEVDResult Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `decomposition` | `np.ndarray` | Shape `(periods+1, K, K)`. `decomposition[h, i, j]` = share of var $i$ variance at $h$ from shock $j$ |
| `var_names` | `list[str]` | Variable names |
| `periods` | `int` | Number of horizons |
| `method` | `str` | Identification method |
| `ordering` | `list[str]` or `None` | Variable ordering used |

### Visualization

```python
# Stacked area chart (default) -- shows evolution of shares over horizons
fevd.plot(kind="area", backend="matplotlib")

# Stacked bar chart at selected horizons
fevd.plot(kind="bar", horizons=[1, 5, 10, 20])

# Plot specific variables only
fevd.plot(variables=["gdp", "inflation"])

# Interactive Plotly chart
fevd.plot(backend="plotly", theme="professional")
```

### Interpreting FEVD

**Typical patterns:**

| Horizon | Pattern | Interpretation |
|---------|---------|----------------|
| $h = 1$ | Own shock dominates (~80-100%) | At short horizons, variables are mostly driven by their own shocks |
| $h = 5$ | Cross-variable shares increase | External shocks start accumulating effect |
| $h = 20$ | Shares stabilize | Long-run decomposition reveals structural importance |

**Reading the decomposition:**

```text
Variable: gdp
Horizon    Shock gdp    Shock inflation    Shock rate
h=1           95.2%            3.1%           1.7%
h=5           72.4%           18.3%           9.3%
h=10          58.1%           25.6%          16.3%
h=20          52.3%           28.4%          19.3%
```

This tells us:
- At $h=1$, 95% of GDP forecast uncertainty comes from GDP's own shocks
- By $h=20$, inflation shocks explain 28% and interest rate shocks explain 19% of GDP forecast uncertainty
- Inflation is a more important driver of GDP uncertainty than the interest rate

!!! tip "Consistency Check"
    FEVD results should be **consistent with IRFs**. If the IRF shows that inflation has a large effect on GDP, then inflation shocks should explain a meaningful share of GDP's FEVD. If they don't match, investigate the identification assumptions.

### Comparing with IRFs

```python
# Compute both IRF and FEVD with same identification
irf = results.irf(periods=20, method="cholesky", order=["rate", "inflation", "gdp"])
fevd = results.fevd(periods=20, method="cholesky", order=["rate", "inflation", "gdp"])

# IRF shows direction and magnitude
print("IRF: GDP response to inflation shock")
print(irf["gdp", "inflation"])

# FEVD shows relative importance
print("\nFEVD: GDP variance decomposition")
print(fevd.to_dataframe(variable="gdp", horizons=[1, 5, 10, 20]))
```

## Configuration Options

### Method Comparison

| Feature | Cholesky | Generalized |
|---------|----------|-------------|
| Ordering-dependent | Yes | No |
| Shares sum to 100% | Exactly | After normalization |
| Consistent with IRF method | Yes (use same method) | Yes |
| Best for | Theory-guided analysis | Exploratory / robustness |

## Complete Example

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

# FEVD with economic ordering: interest rate -> inflation -> GDP
fevd = results.fevd(
    periods=20,
    method="cholesky",
    order=["interest_rate", "inflation", "gdp_growth"],
)

# Detailed summary
print(fevd.summary())

# Export to DataFrame for further analysis
df_fevd = fevd.to_dataframe(horizons=[1, 5, 10, 20])
print(df_fevd)

# Visual analysis
fevd.plot(kind="area", backend="matplotlib", theme="academic")
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| FEVD Analysis | Variance decomposition workflow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/03_fevd_decomposition.ipynb) |

## See Also

- [Impulse Response Functions](irf.md) -- Dynamic effects (complementary to FEVD)
- [Panel VAR Estimation](estimation.md) -- Model setup and estimation
- [Granger Causality](granger.md) -- Statistical causality tests
- [VECM](vecm.md) -- FEVD for cointegrated systems
- [Forecasting](forecasting.md) -- Out-of-sample predictions

## References

- Luetkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer-Verlag, Chapter 2.
- Pesaran, H. H., & Shin, Y. (1998). Generalized impulse response analysis in linear multivariate models. *Economics Letters*, 58(1), 17-29.
- Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1-48.
