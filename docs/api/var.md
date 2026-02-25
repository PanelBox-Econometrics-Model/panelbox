---
title: "Panel VAR API"
description: "API reference for panelbox.var — PanelVAR, PanelVECM, impulse responses, Granger causality, forecasting"
---

# Panel VAR API Reference

!!! info "Module"
    **Import**: `from panelbox.var import PanelVAR, PanelVECM, PanelVARData`
    **Source**: `panelbox/var/`

## Overview

The VAR module implements Panel Vector Autoregression models for analyzing dynamic interdependencies among multiple variables across entities:

| Class | Description | Reference |
|-------|-------------|-----------|
| `PanelVARData` | Data container with lag structure | — |
| `PanelVAR` | Panel VAR estimation (OLS/GMM) | Holtz-Eakin, Newey & Rosen (1988) |
| `PanelVECM` | Panel Vector Error Correction Model | Johansen (1991) |
| `CointegrationRankTest` | Cointegration rank selection | — |

Key features: lag order selection, impulse response functions (IRF), forecast error variance decomposition (FEVD), Granger causality tests, and forecasting.

## Classes

### PanelVARData

Data container for Panel VAR models. Handles lag construction, transformation, and variable management.

#### Constructor

```python
PanelVARData(
    data: pd.DataFrame,
    endog_vars: list[str],
    entity_col: str,
    time_col: str,
    exog_vars: list[str] | None = None,
    lags: int = 1,
    trend: Literal["none", "constant", "trend", "both"] = "constant",
    dropna: Literal["any", "equation"] = "any",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | *required* | Panel data |
| `endog_vars` | `list[str]` | *required* | Endogenous variables (VAR system) |
| `entity_col` | `str` | *required* | Entity column |
| `time_col` | `str` | *required* | Time column |
| `exog_vars` | `list[str] \| None` | `None` | Exogenous variables |
| `lags` | `int` | `1` | Number of lags |
| `trend` | `str` | `"constant"` | Trend specification: `"none"`, `"constant"`, `"trend"`, `"both"` |
| `dropna` | `str` | `"any"` | How to handle missing values |

#### Example

```python
from panelbox.var import PanelVARData

var_data = PanelVARData(
    data=df,
    endog_vars=["gdp", "inflation", "interest_rate"],
    entity_col="country",
    time_col="year",
    lags=2,
    trend="constant",
)
```

---

### PanelVAR

Panel Vector Autoregression model. Supports OLS and GMM estimation with comprehensive analysis tools.

#### Constructor

```python
PanelVAR(data: PanelVARData)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `PanelVARData` | Prepared VAR data container |

#### Methods

##### `.fit()`

```python
def fit(
    self,
    method: str = "ols",
    cov_type: str = "clustered",
    **cov_kwds,
) -> PanelVARResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"ols"` | Estimation method: `"ols"` or `"gmm"` |
| `cov_type` | `str` | `"clustered"` | Covariance type |

##### `.select_lag_order()`

```python
def select_lag_order(
    self,
    max_lags: int,
    criteria: list[str] = ["aic", "bic", "hqic"],
) -> LagOrderResult
```

Select optimal lag order using information criteria.

##### `.impulse_response()`

Compute impulse response functions (orthogonalized or generalized).

##### `.forecast_error_variance_decomposition()`

Compute FEVD showing each variable's contribution to forecast error variance.

##### `.granger_causality()`

Test Granger causality between variable pairs.

##### `.forecast()`

Generate multi-step-ahead forecasts with confidence intervals.

#### Example

```python
from panelbox.var import PanelVARData, PanelVAR

# Prepare data
var_data = PanelVARData(
    df, endog_vars=["gdp", "inflation", "interest_rate"],
    entity_col="country", time_col="year", lags=2,
)

# Estimate
model = PanelVAR(var_data)
result = model.fit(method="ols", cov_type="clustered")
print(result.summary())

# Lag order selection
lag_result = model.select_lag_order(max_lags=8)
print(lag_result.summary())

# Granger causality
gc = result.granger_causality("gdp", "inflation")
print(f"GDP -> Inflation: F={gc.statistic:.3f}, p={gc.pvalue:.3f}")

# Impulse response
irf = result.impulse_response(periods=20, method="cholesky")

# Forecast
forecast = result.forecast(periods=5, confidence=0.95)
```

---

### PanelVECM

Panel Vector Error Correction Model for cointegrated panel data.

#### Constructor

```python
PanelVECM(data: PanelVARData)
```

#### Methods

##### `.fit()`

```python
def fit(
    self,
    coint_rank: int,
    lags: int | None = None,
) -> PanelVECMResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `coint_rank` | `int` | *required* | Cointegration rank (number of cointegrating vectors) |
| `lags` | `int \| None` | `None` | Lag order (uses data's lags if None) |

#### Example

```python
from panelbox.var import PanelVARData, PanelVECM, CointegrationRankTest

var_data = PanelVARData(
    df, endog_vars=["consumption", "income", "wealth"],
    entity_col="household", time_col="quarter", lags=2,
)

# Test for cointegration rank
rank_test = CointegrationRankTest(var_data)
rank_result = rank_test.test()
print(rank_result.summary())

# Estimate VECM with selected rank
vecm = PanelVECM(var_data)
result = vecm.fit(coint_rank=rank_result.selected_rank)
print(result.summary())
```

---

### CointegrationRankTest

Test for cointegration rank in Panel VECM.

#### Constructor

```python
CointegrationRankTest(
    data: PanelVARData,
    max_rank: int | None = None,
    deterministic: str = "c",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `PanelVARData` | *required* | VAR data container |
| `max_rank` | `int \| None` | `None` | Maximum rank to test |
| `deterministic` | `str` | `"c"` | Deterministic specification |

---

## Result Classes

### PanelVARResult

Result container for Panel VAR estimation.

#### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `dict` | Coefficient matrices per equation |
| `cov_params` | `dict` | Covariance matrices |
| `resid` | `pd.DataFrame` | Residuals |
| `sigma_u` | `np.ndarray` | Residual covariance matrix |
| `nobs` | `int` | Number of observations |

#### Methods

- `.summary()` — Formatted results for all equations
- `.granger_causality(cause, effect)` — Test Granger causality
- `.granger_causality_matrix()` — All pairwise causality tests
- `.instantaneous_causality(var1, var2)` — Contemporaneous causality test
- `.impulse_response(periods, method)` — IRF computation
- `.forecast_error_variance_decomposition(periods)` — FEVD
- `.forecast(periods, confidence)` — Multi-step forecasts

### LagOrderResult

Lag order selection results with information criteria.

#### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `aic` | `dict` | AIC by lag order |
| `bic` | `dict` | BIC by lag order |
| `hqic` | `dict` | HQIC by lag order |
| `selected_lag` | `int` | Recommended lag order |

### ForecastResult

Multi-step forecast results with confidence intervals.

### PanelVECMResult

VECM estimation results with cointegrating vectors and adjustment coefficients.

### RankSelectionResult

Cointegration rank test results.

### RankTestResult

Individual rank test result (dataclass).

| Attribute | Type | Description |
|-----------|------|-------------|
| `rank` | `int` | Tested rank |
| `test_stat` | `float` | Test statistic |
| `z_stat` | `float` | Standardized statistic |
| `p_value` | `float` | p-value |
| `test_type` | `str` | Test type |
| `critical_values` | `dict \| None` | Critical values at standard levels |

---

## Visualization

### plot_causality_network

```python
from panelbox.var import plot_causality_network

fig = plot_causality_network(var_result)
```

Generate a network visualization of Granger causality relationships.

## References

- Holtz-Eakin, D., Newey, W. & Rosen, H. (1988). "Estimating vector autoregressions with panel data." *Econometrica*, 56(6), 1371-1395.
- Love, I. & Zicchino, L. (2006). "Financial development and dynamic investment behavior." *QREE*, 46(2), 190-210.
- Abrigo, M. & Love, I. (2016). "Estimation of panel vector autoregression in Stata." *Stata Journal*, 16(3), 778-804.

## See Also

- [GMM API](gmm.md) — GMM estimation methods
- [Diagnostics](diagnostics.md) — Cointegration tests (Kao, Pedroni, Westerlund)
- [Tutorials: VAR](../tutorials/var.md) — Step-by-step Panel VAR guide
