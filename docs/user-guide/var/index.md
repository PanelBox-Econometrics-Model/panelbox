---
title: Panel VAR
description: Guide to Panel Vector Autoregression in PanelBox - VAR, VECM, Impulse Response Functions, Forecast Error Variance Decomposition, and Granger Causality.
---

# Panel VAR

Panel Vector Autoregression (Panel VAR) models capture the dynamic interdependencies among multiple variables across entities and time. Unlike single-equation models, VAR treats all variables as potentially endogenous, allowing each variable to depend on its own lags and the lags of all other variables. Combined with panel data, this reveals dynamic relationships while controlling for unobserved heterogeneity.

PanelBox provides a complete Panel VAR toolkit: estimation, impulse response functions (IRF), forecast error variance decomposition (FEVD), Granger causality tests, and forecasting.

## Available Features

| Feature | Class / Method | Description |
|---------|---------------|-------------|
| Panel VAR | `PanelVAR` | VAR estimation with entity fixed effects |
| Panel VECM | `PanelVECM` | Vector Error Correction for cointegrated panels |
| Impulse Response | `PanelVAR.irf()` | Orthogonalized and structural IRFs |
| FEVD | `PanelVAR.fevd()` | Forecast error variance decomposition |
| Granger Causality | `PanelVAR.granger_causality()` | Bivariate and multivariate causality tests |
| Forecast | `PanelVAR.forecast()` | Out-of-sample forecasting |
| Lag Selection | `PanelVAR.select_lag_order()` | AIC, BIC, HQIC lag order selection |

## Quick Example

```python
from panelbox.var import PanelVAR
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

model = PanelVAR(
    variables=["invest", "value", "capital"],
    data=data,
    entity_col="firm",
    time_col="year",
    lags=2
)
results = model.fit()
print(results.summary())

# Impulse Response Functions
irf = results.irf(periods=10)
irf.plot()

# Granger Causality
gc = results.granger_causality(caused="invest", causing="value")
print(gc.summary())
```

## Key Concepts

### Panel VAR Specification

The Panel VAR(p) model for $K$ variables and $p$ lags:

$$
Y_{it} = \alpha_i + A_1 Y_{i,t-1} + A_2 Y_{i,t-2} + \ldots + A_p Y_{i,t-p} + \epsilon_{it}
$$

where $Y_{it}$ is a $K \times 1$ vector, $\alpha_i$ captures entity fixed effects, and $A_1, \ldots, A_p$ are $K \times K$ coefficient matrices.

### Lag Order Selection

```python
lag_result = model.select_lag_order(max_lags=8)
print(lag_result.summary())
# Shows AIC, BIC, HQIC for each lag order
```

### Impulse Response Functions

IRFs trace the dynamic effect of a one-standard-deviation shock to variable $j$ on variable $k$ over time:

```python
irf = results.irf(periods=20, method="cholesky")
irf.plot(impulse="value", response="invest")
```

### Forecast Error Variance Decomposition

FEVD shows what fraction of the forecast error variance of each variable is attributable to shocks in each other variable:

```python
fevd = results.fevd(periods=10)
fevd.plot()
```

### Granger Causality

Tests whether past values of variable $X$ help predict variable $Y$ beyond the information in past values of $Y$ alone:

```python
# Bivariate Granger causality
gc = results.granger_causality(caused="invest", causing="value")
print(f"p-value: {gc.pvalue:.4f}")
print(gc.conclusion)
```

## Detailed Guides

- [Panel VAR Estimation](estimation.md) -- Model specification and estimation *(detailed guide coming soon)*
- [Panel VECM](vecm.md) -- Error correction for cointegrated panels *(detailed guide coming soon)*
- [IRF Analysis](irf.md) -- Impulse response functions *(detailed guide coming soon)*
- [Granger Causality](granger.md) -- Causality testing *(detailed guide coming soon)*
- [Forecasting](forecasting.md) -- Out-of-sample prediction *(detailed guide coming soon)*

## Tutorials

See [Panel VAR Tutorial](../../tutorials/var.md) for interactive notebooks with Google Colab.

## API Reference

See [VAR API](../../api/var.md) for complete technical reference.

## References

- Holtz-Eakin, D., Newey, W., & Rosen, H. S. (1988). Estimating vector autoregressions with panel data. *Econometrica*, 56(6), 1371-1395.
- Love, I., & Zicchino, L. (2006). Financial development and dynamic investment behavior: Evidence from panel VAR. *Quarterly Review of Economics and Finance*, 46(2), 190-210.
- Abrigo, M. R. M., & Love, I. (2016). Estimation of panel vector autoregression in Stata. *Stata Journal*, 16(3), 778-804.
