---
title: "Panel VAR Estimation"
description: "Complete guide to Panel VAR estimation in PanelBox: model setup, lag selection, covariance types, and stability analysis."
---

# Panel VAR Estimation

!!! info "Quick Reference"
    **Class:** `panelbox.var.model.PanelVAR`
    **Data:** `panelbox.var.data.PanelVARData`
    **Import:** `from panelbox.var import PanelVARData, PanelVAR`
    **Stata equivalent:** `pvar` (community-contributed)
    **R equivalent:** `panelvar::pvargmm()`

## Overview

Panel Vector Autoregression (Panel VAR) extends the classical VAR framework to panel data settings where multiple entities (countries, firms, individuals) are observed over time. Each endogenous variable is modeled as a function of its own lagged values and the lagged values of all other endogenous variables in the system, while controlling for entity-specific fixed effects.

The Panel VAR(p) model with $K$ endogenous variables, $N$ entities, and $p$ lags is:

$$
Y_{it} = A_1 Y_{i,t-1} + A_2 Y_{i,t-2} + \ldots + A_p Y_{i,t-p} + \alpha_i + \varepsilon_{it}
$$

where $Y_{it}$ is the $K \times 1$ vector of endogenous variables for entity $i$ at time $t$, $A_1, \ldots, A_p$ are $K \times K$ coefficient matrices, $\alpha_i$ are entity-specific fixed effects, and $\varepsilon_{it} \sim (0, \Sigma)$ is the error term.

PanelBox estimates Panel VAR using OLS equation-by-equation with within transformation (entity demeaning) to remove fixed effects. This approach follows the methodology of Holtz-Eakin, Newey, and Rosen (1988) and the implementation strategy of Abrigo and Love (2016).

## Quick Example

```python
import pandas as pd
from panelbox.var import PanelVARData, PanelVAR

# Step 1: Prepare data container
var_data = PanelVARData(
    data=df,
    endog_vars=["gdp", "inflation", "unemployment"],
    entity_col="country",
    time_col="year",
    lags=2,
    trend="constant",
)

# Step 2: Create and estimate model
model = PanelVAR(data=var_data)
results = model.fit(method="ols", cov_type="clustered")

# Step 3: View results
print(results.summary())
print(f"Stable: {results.is_stable()}")
print(f"AIC: {results.aic:.4f}, BIC: {results.bic:.4f}")
```

## When to Use

- **Macroeconomic dynamics**: Studying interactions between GDP, inflation, interest rates, and unemployment across countries
- **Financial contagion**: Analyzing how shocks propagate across markets or firms
- **Policy analysis**: Evaluating the dynamic effects of monetary or fiscal policy
- **Firm-level dynamics**: Modeling investment, employment, and output interactions across firms
- **No strong prior on causality**: When the causal ordering among variables is unknown

!!! warning "Key Assumptions"
    - **Stationarity**: All endogenous variables must be stationary (or cointegrated -- see [VECM](vecm.md))
    - **No cross-entity contamination**: Lags are constructed within each entity separately
    - **Homogeneous slope coefficients**: The $A_l$ matrices are assumed identical across entities
    - **Continuous time series**: No internal gaps allowed within any entity's time series
    - **Sufficient time periods**: $T > K \times p + 1$ for each entity after lag construction

## Detailed Guide

### Data Preparation

The `PanelVARData` class handles all data preparation, including lag construction, gap detection, and missing value handling.

```python
from panelbox.var import PanelVARData

var_data = PanelVARData(
    data=df,                                    # pandas DataFrame in long format
    endog_vars=["gdp", "inflation", "rate"],    # Endogenous variables (K)
    entity_col="country",                       # Entity identifier column
    time_col="year",                            # Time identifier column
    exog_vars=None,                             # Optional exogenous variables
    lags=2,                                     # Number of lags (p)
    trend="constant",                           # Deterministic terms
    dropna="any",                               # Missing value strategy
)

# Inspect data properties
print(f"Variables (K): {var_data.K}")
print(f"Lags (p): {var_data.p}")
print(f"Entities (N): {var_data.N}")
print(f"Observations: {var_data.n_obs}")
print(f"Balanced: {var_data.is_balanced}")
print(f"T range: [{var_data.T_min}, {var_data.T_max}]")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | required | Panel data in long format |
| `endog_vars` | `list[str]` | required | Names of endogenous variables |
| `entity_col` | `str` | required | Entity identifier column name |
| `time_col` | `str` | required | Time identifier column name |
| `exog_vars` | `list[str]` | `None` | Optional exogenous variable names |
| `lags` | `int` | `1` | Number of lags ($p$) |
| `trend` | `str` | `"constant"` | `"none"`, `"constant"`, `"trend"`, or `"both"` |
| `dropna` | `str` | `"any"` | `"any"` (drop if any variable missing) or `"equation"` |

!!! note "Critical Safety Feature"
    Lags are constructed using `.groupby(entity).shift()` to ensure that lag $t$ of entity A never contains an observation from entity B. A cross-contamination verification check runs automatically after lag construction.

### Estimation

The `PanelVAR` class performs OLS equation-by-equation estimation with within transformation (entity demeaning) to remove fixed effects.

```python
from panelbox.var import PanelVAR

model = PanelVAR(data=var_data)
results = model.fit(
    method="ols",              # Estimation method
    cov_type="clustered",      # Covariance estimator
)
```

**Within Transformation**: Before OLS estimation, each variable is demeaned within each entity:

$$
\tilde{y}_{it} = y_{it} - \bar{y}_i
$$

This removes the entity-specific fixed effects $\alpha_i$ and avoids the incidental parameters problem.

**Estimation proceeds equation by equation**: For each equation $k = 1, \ldots, K$:

1. Apply within transformation to $y_k$ and $X$
2. Estimate $\hat{\beta}_k = (X'X)^{-1} X' y_k$
3. Compute residuals $\hat{\varepsilon}_k = y_k - X \hat{\beta}_k$
4. Compute covariance matrix using the specified method

### Covariance Types

| `cov_type` | Description | When to Use |
|-----------|-------------|-------------|
| `"clustered"` | Cluster-robust by entity | **Default (recommended)**. Accounts for within-entity correlation |
| `"driscoll_kraay"` | Driscoll-Kraay HAC | Cross-sectional dependence suspected |
| `"hc1"` | Heteroskedasticity-robust (HC1) | Heteroskedastic errors, no clustering |
| `"nonrobust"` | Classical OLS | Homoskedastic, no clustering (rarely appropriate) |
| `"sur"` | Seemingly Unrelated Regressions | Exploit cross-equation correlation |

```python
# Cluster-robust standard errors (recommended)
results = model.fit(cov_type="clustered")

# Driscoll-Kraay for cross-sectional dependence
results = model.fit(cov_type="driscoll_kraay", max_lags=3)

# SUR covariance (exploits cross-equation correlation)
results = model.fit(cov_type="sur")
```

### Lag Selection

Choosing the optimal lag order is crucial. Too few lags can lead to omitted variable bias; too many waste degrees of freedom and reduce efficiency.

```python
# Automatic lag selection
lag_result = model.select_lag_order(max_lags=8, cov_type="clustered")
print(lag_result.summary())

# Access optimal lag by criterion
optimal_bic = lag_result.selected["BIC"]
optimal_aic = lag_result.selected["AIC"]

# Visualize information criteria
fig = lag_result.plot(backend="plotly")
```

The `select_lag_order` method tests $p = 1, 2, \ldots, \text{max\_lags}$ and computes four information criteria:

| Criterion | Formula | Properties |
|-----------|---------|------------|
| AIC | $\log|\hat{\Sigma}| + \frac{2K^2p}{NT}$ | Tends to overfit |
| BIC | $\log|\hat{\Sigma}| + \frac{K^2p \log(NT)}{NT}$ | **Recommended**: consistent |
| HQIC | $\log|\hat{\Sigma}| + \frac{2K^2p \log(\log(NT))}{NT}$ | Compromise between AIC and BIC |
| MBIC | $\log|\hat{\Sigma}| + \frac{K^2p \log(NT) \log(\log(NT))}{NT}$ | Modified BIC (Andrews & Lu, 2001) |

The `LagOrderResult` object contains:

- `criteria_df`: DataFrame with all criteria values for each lag
- `selected`: Dictionary mapping criterion name to optimal lag
- `summary()`: Formatted summary table
- `plot()`: Visual comparison of criteria

!!! tip "Practical Advice"
    BIC is generally recommended for Panel VAR because it penalizes complexity more heavily and is consistent (selects the true lag order as $N, T \to \infty$). Start with `max_lags=8` and reduce if you get warnings about insufficient observations.

### Interpreting Results

The `PanelVARResult` object provides comprehensive access to estimation results.

```python
# Coefficient matrices A_1, A_2, ..., A_p
for lag in range(1, results.p + 1):
    print(f"\nA_{lag}:")
    print(results.coef_matrix(lag))  # Returns labeled DataFrame

# Residual covariance matrix
print(f"\nSigma (residual covariance):\n{results.Sigma}")

# Information criteria
print(f"AIC: {results.aic:.6f}")
print(f"BIC: {results.bic:.6f}")
print(f"HQIC: {results.hqic:.6f}")
print(f"Log-likelihood: {results.loglik:.2f}")

# Per-equation summary
for k in range(results.K):
    print(results.equation_summary(k))

# System-level summary (compact)
print(results.summary_system())

# Full summary with coefficient tables
print(results.summary())
```

**Key Result Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `params_by_eq` | `list[np.ndarray]` | Coefficient vectors per equation |
| `std_errors_by_eq` | `list[np.ndarray]` | Standard errors per equation |
| `A_matrices` | `list[np.ndarray]` | $K \times K$ coefficient matrices $[A_1, \ldots, A_p]$ |
| `Sigma` | `np.ndarray` | Residual covariance matrix ($K \times K$) |
| `aic`, `bic`, `hqic` | `float` | Information criteria |
| `loglik` | `float` | Log-likelihood |
| `K`, `p`, `N`, `n_obs` | `int` | Dimensions |

### Stability Analysis

A Panel VAR is **stable** (stationary) if all eigenvalues of the companion matrix have modulus strictly less than 1. Stability is essential for meaningful impulse response functions and forecasts.

```python
# Check stability
print(f"Stable: {results.is_stable()}")
print(f"Max eigenvalue modulus: {results.max_eigenvalue_modulus:.6f}")
print(f"Stability margin: {results.stability_margin:.6f}")

# Eigenvalues of companion matrix
eigenvalues = results.eigenvalues
print(f"Eigenvalues: {eigenvalues}")

# Companion matrix (Kp x Kp)
F = results.companion_matrix()

# Visual stability check
results.plot_stability(backend="matplotlib")
```

The **companion matrix** $\mathbf{F}$ reformulates VAR(p) as VAR(1):

$$
\mathbf{F} = \begin{bmatrix} A_1 & A_2 & \cdots & A_p \\ I_K & 0 & \cdots & 0 \\ 0 & I_K & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & I_K & 0 \end{bmatrix}
$$

The `plot_stability()` method visualizes eigenvalues on the complex plane with the unit circle, making it easy to identify whether any eigenvalues lie outside the circle (unstable system).

!!! warning "Unstable VAR"
    If `is_stable()` returns `False`, the system is explosive. IRFs will diverge rather than converge to zero, and forecasts will blow up. Consider:

    - Differencing non-stationary variables
    - Using VECM for cointegrated variables (see [VECM](vecm.md))
    - Reducing the lag order
    - Checking for data issues

## Export Results

```python
# LaTeX table
latex = results.to_latex()
with open("var_results.tex", "w") as f:
    f.write(latex)

# HTML table
html = results.to_html()

# Single equation export
latex_eq1 = results.to_latex(equation=0)
```

## Configuration Options

### PanelVAR.fit() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"ols"` | Estimation method (currently `"ols"` only) |
| `cov_type` | `str` | `"clustered"` | Covariance estimator type |
| `**cov_kwds` | | | Additional keyword arguments for covariance (e.g., `max_lags` for Driscoll-Kraay) |

### PanelVAR.select_lag_order() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_lags` | `int` | `8` | Maximum number of lags to test |
| `criteria` | `list[str]` | `["AIC", "BIC", "HQIC", "MBIC"]` | Information criteria to compute |
| `cov_type` | `str` | `"clustered"` | Covariance type for each estimation |

## Complete Workflow Example

```python
import pandas as pd
from panelbox.var import PanelVARData, PanelVAR

# Load macroeconomic panel data
# df has columns: country, year, gdp_growth, inflation, unemployment
df = pd.read_csv("macro_panel.csv")

# Step 1: Create data container
var_data = PanelVARData(
    data=df,
    endog_vars=["gdp_growth", "inflation", "unemployment"],
    entity_col="country",
    time_col="year",
    lags=2,  # Start with initial guess
    trend="constant",
)

print(f"Panel: N={var_data.N}, T_avg={var_data.T_avg:.1f}, "
      f"K={var_data.K}, balanced={var_data.is_balanced}")

# Step 2: Select optimal lag order
model = PanelVAR(data=var_data)
lag_result = model.select_lag_order(max_lags=6)
print(lag_result.summary())
optimal_p = lag_result.selected["BIC"]
print(f"\nOptimal lag (BIC): {optimal_p}")

# Step 3: Re-estimate with optimal lag
var_data_opt = PanelVARData(
    data=df,
    endog_vars=["gdp_growth", "inflation", "unemployment"],
    entity_col="country",
    time_col="year",
    lags=optimal_p,
)
model_opt = PanelVAR(data=var_data_opt)
results = model_opt.fit(method="ols", cov_type="clustered")

# Step 4: Check stability
print(f"\nStable: {results.is_stable()}")
print(f"Max eigenvalue modulus: {results.max_eigenvalue_modulus:.4f}")

# Step 5: Full summary
print(results.summary())

# Step 6: Proceed to analysis
# IRF, FEVD, Granger causality, forecasting
irf = results.irf(periods=10, method="cholesky")
gc = results.granger_causality(cause="inflation", effect="gdp_growth")
print(gc.summary())
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Panel VAR Notebook | Full estimation workflow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/01_var_introduction.ipynb) |

## See Also

- [Impulse Response Functions](irf.md) -- Dynamic effects of shocks
- [Forecast Error Variance Decomposition](fevd.md) -- Relative importance of shocks
- [Granger Causality](granger.md) -- Testing predictive relationships
- [VECM](vecm.md) -- For cointegrated (non-stationary) variables
- [Forecasting](forecasting.md) -- Multi-step ahead predictions

## References

- Holtz-Eakin, D., Newey, W., & Rosen, H. S. (1988). Estimating vector autoregressions with panel data. *Econometrica*, 56(6), 1371-1395.
- Abrigo, M. R., & Love, I. (2016). Estimation of panel vector autoregression in Stata. *The Stata Journal*, 16(3), 778-804.
- Andrews, D. W. K., & Lu, B. (2001). Consistent model and moment selection procedures for GMM estimation with application to dynamic panel data models. *Journal of Econometrics*, 101(1), 123-164.
- Luetkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer-Verlag.
