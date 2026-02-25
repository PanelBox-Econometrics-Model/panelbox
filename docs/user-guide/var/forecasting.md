---
title: "Panel VAR Forecasting"
description: "Guide to multi-step ahead forecasting with Panel VAR models in PanelBox: iterative forecasts, confidence intervals, and accuracy evaluation."
---

# Panel VAR Forecasting

!!! info "Quick Reference"
    **Class:** `panelbox.var.forecast.ForecastResult`
    **Method:** `PanelVARResult.forecast()`
    **Import:** `from panelbox.var import PanelVAR` (forecasts accessed via results)
    **Stata equivalent:** `fcast compute`
    **R equivalent:** `vars::predict()`

## Overview

Panel VAR forecasting generates multi-step-ahead predictions for all endogenous variables and all entities simultaneously. Forecasts are produced **iteratively**: the one-step-ahead forecast uses observed data, while longer horizons use previously generated forecasts as inputs.

For a VAR(p) system, the $h$-step-ahead forecast is:

$$
\hat{Y}_{i,T+h} = \sum_{l=1}^{p} A_l \hat{Y}_{i,T+h-l}
$$

where $\hat{Y}_{i,T+h-l} = Y_{i,T+h-l}$ for $T+h-l \le T$ (observed data) and $\hat{Y}_{i,T+h-l}$ is the forecast for $T+h-l > T$.

PanelBox supports both **bootstrap** and **analytical** confidence intervals to quantify forecast uncertainty.

## Quick Example

```python
from panelbox.var import PanelVARData, PanelVAR

# Estimate model
var_data = PanelVARData(df, endog_vars=["gdp", "inflation", "rate"],
                         entity_col="country", time_col="year", lags=2)
model = PanelVAR(data=var_data)
results = model.fit(cov_type="clustered")

# Generate 5-step ahead forecast with bootstrap CIs
forecast = results.forecast(
    steps=5,
    ci_method="bootstrap",
    n_bootstrap=500,
    ci_level=0.95,
)

# View summary
print(forecast.summary())

# Convert to DataFrame for a specific entity
df_fcst = forecast.to_dataframe(entity=0)
print(df_fcst)

# Plot forecast for one entity/variable
forecast.plot(entity=0, variable="gdp")
```

## When to Use

- **Economic forecasting**: Predicting macroeconomic variables (GDP, inflation, unemployment) across countries
- **Policy scenario analysis**: Projecting the effects of current conditions forward
- **Out-of-sample evaluation**: Testing the predictive ability of the estimated model
- **Business planning**: Forecasting firm-level metrics across multiple business units

!!! warning "Key Assumptions"
    - **Stability**: The VAR must be stable (`results.is_stable() == True`) for forecasts to remain bounded
    - **Structural stability**: The estimated relationships are assumed to hold in the forecast period
    - **No future shocks**: Point forecasts assume $\varepsilon_{T+h} = 0$; CIs account for shock uncertainty
    - **Accuracy degrades with horizon**: Forecast uncertainty grows with $h$

## Detailed Guide

### Computing Forecasts

```python
forecast = results.forecast(
    steps=10,                     # Number of steps ahead
    exog_future=None,             # Future exogenous values (if applicable)
    ci_method="bootstrap",        # None, "bootstrap", or "analytical"
    ci_level=0.95,                # Confidence level
    n_bootstrap=500,              # Bootstrap replications
    seed=42,                      # Random seed for reproducibility
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps` | `int` | `10` | Number of forecast horizons |
| `exog_future` | `np.ndarray` | `None` | Future exogenous values, shape `(steps, N, n_exog)` |
| `ci_method` | `str` | `None` | `None`, `"bootstrap"`, or `"analytical"` |
| `ci_level` | `float` | `0.95` | Confidence level for intervals |
| `n_bootstrap` | `int` | `500` | Number of bootstrap replications |
| `seed` | `int` | `None` | Random seed for reproducibility |

### Confidence Interval Methods

=== "Bootstrap CIs"

    The bootstrap approach resamples residuals to generate forecast paths that account for both parameter uncertainty and shock uncertainty.

    ```python
    forecast = results.forecast(
        steps=10,
        ci_method="bootstrap",
        n_bootstrap=1000,
        ci_level=0.95,
        seed=42,
    )
    ```

    **Procedure:**

    1. Resample residuals with replacement
    2. Add resampled shocks to the iterative forecast path
    3. Repeat $B$ times
    4. Compute percentile intervals from the distribution of forecast paths

=== "Analytical CIs"

    Analytical CIs use the MA representation to compute forecast error variance:

    $$
    \text{Var}(\hat{Y}_{T+h}) = \sum_{s=0}^{h-1} \Phi_s \hat{\Sigma} \Phi_s'
    $$

    ```python
    forecast = results.forecast(
        steps=10,
        ci_method="analytical",
        ci_level=0.95,
    )
    ```

    !!! note
        Analytical CIs only account for shock uncertainty (assuming known parameters). They are faster but may understate total uncertainty in small samples. Requires a stable VAR.

=== "No CIs"

    ```python
    forecast = results.forecast(steps=10, ci_method=None)
    ```

### Accessing Results

The `ForecastResult` object stores forecasts for all entities and variables.

```python
# Point forecasts: shape (steps, N, K)
print(forecast.forecasts.shape)

# Convert to DataFrame
df_all = forecast.to_dataframe()                    # All entities (long format)
df_usa = forecast.to_dataframe(entity="USA")        # Single entity (wide format)
df_gdp = forecast.to_dataframe(entity=0, variable="gdp")  # Single entity + variable

# Summary
print(forecast.summary())
```

**ForecastResult Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `forecasts` | `np.ndarray` | Point forecasts, shape `(steps, N, K)` |
| `ci_lower` | `np.ndarray` or `None` | Lower CI bounds, shape `(steps, N, K)` |
| `ci_upper` | `np.ndarray` or `None` | Upper CI bounds, shape `(steps, N, K)` |
| `horizon` | `int` | Forecast horizon |
| `K` | `int` | Number of variables |
| `N` | `int` | Number of entities |
| `endog_names` | `list[str]` | Variable names |
| `entity_names` | `list[str]` | Entity names |
| `ci_level` | `float` | Confidence level |
| `method` | `str` | Forecast method |
| `ci_method` | `str` | CI method used |

### Visualization

```python
# Basic forecast plot with CIs
forecast.plot(entity=0, variable="gdp", backend="plotly")

# With actual values for comparison (out-of-sample evaluation)
forecast.plot(entity=0, variable="gdp", actual=y_actual, backend="matplotlib")
```

The plot shows:

- Point forecast (dashed blue line with markers)
- Confidence interval band (shaded area)
- Actual values (black line, if provided)

### Forecast Evaluation

When actual data is available, evaluate forecast accuracy:

```python
# Evaluate against actual values
# actual should have shape (steps, N, K) or (steps, K) for single entity
accuracy = forecast.evaluate(actual=y_actual, entity=0)
print(accuracy)
```

**Accuracy Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| RMSE | $\sqrt{\frac{1}{H}\sum_{h=1}^{H}(y_{T+h} - \hat{y}_{T+h})^2}$ | Scale-dependent; lower is better |
| MAE | $\frac{1}{H}\sum_{h=1}^{H}|y_{T+h} - \hat{y}_{T+h}|$ | Less sensitive to outliers than RMSE |
| MAPE | $\frac{100}{H}\sum_{h=1}^{H}\frac{|y_{T+h} - \hat{y}_{T+h}|}{|y_{T+h}|}$ | Scale-free percentage error |

### Practical Considerations

!!! tip "Forecast Horizon"
    Forecast accuracy degrades rapidly with the horizon. A useful rule of thumb:

    - **Short-term** ($h \le p$): Forecasts use mostly observed data; relatively reliable
    - **Medium-term** ($p < h \le 2p$): Forecasts increasingly rely on own predictions; uncertainty grows
    - **Long-term** ($h > 2p$): Dominated by the VAR dynamics; confidence intervals widen substantially

!!! warning "Unstable Systems"
    If `results.is_stable() == False`, forecasts will **diverge** (grow without bound). This is not a numerical error -- it reflects the explosive dynamics of the estimated system. Solutions:

    - Difference non-stationary variables
    - Use VECM for cointegrated variables
    - Reduce lag order
    - Check for data errors

## Complete Workflow Example

```python
import numpy as np
import pandas as pd
from panelbox.var import PanelVARData, PanelVAR

# Load data
df = pd.read_csv("macro_panel.csv")

# Split into training and test sets (hold out last 5 periods)
years = sorted(df["year"].unique())
cutoff = years[-5]
df_train = df[df["year"] <= cutoff]
df_test = df[df["year"] > cutoff]

# Estimate Panel VAR on training data
var_data = PanelVARData(
    data=df_train,
    endog_vars=["gdp_growth", "inflation", "interest_rate"],
    entity_col="country",
    time_col="year",
    lags=2,
)
model = PanelVAR(data=var_data)
results = model.fit(cov_type="clustered")

# Check stability
print(f"Stable: {results.is_stable()}")

# Generate forecasts with bootstrap CIs
forecast = results.forecast(
    steps=5,
    ci_method="bootstrap",
    n_bootstrap=500,
    ci_level=0.95,
    seed=42,
)

print(forecast.summary())

# View forecast for a specific entity
df_fcst = forecast.to_dataframe(entity=0)
print(df_fcst)

# Plot forecast with uncertainty bands
forecast.plot(entity=0, variable="gdp_growth", backend="matplotlib")
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| VAR Forecasting | Multi-step forecasting and evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/07_case_study.ipynb) |

## See Also

- [Panel VAR Estimation](estimation.md) -- Model setup and estimation
- [Impulse Response Functions](irf.md) -- Dynamic effects of shocks
- [FEVD](fevd.md) -- Variance decomposition
- [Granger Causality](granger.md) -- Predictive relationships
- [VECM](vecm.md) -- Forecasting with cointegrated systems

## References

- Luetkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer-Verlag.
- Abrigo, M. R., & Love, I. (2016). Estimation of panel vector autoregression in Stata. *The Stata Journal*, 16(3), 778-804.
- Holtz-Eakin, D., Newey, W., & Rosen, H. S. (1988). Estimating vector autoregressions with panel data. *Econometrica*, 56(6), 1371-1395.
