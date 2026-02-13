# Panel VAR: Complete Guide

## Introduction

This tutorial provides a comprehensive guide to estimating Panel Vector Autoregressions (Panel VAR) using PanelBox. Panel VAR models are used to analyze dynamic interactions between multiple variables in a panel data setting.

### What is Panel VAR?

Panel VAR extends traditional VAR models to panel data, allowing for:

- Analysis of dynamic relationships between variables across entities
- Impulse Response Functions (IRFs) showing how shocks propagate
- Granger causality testing in panel settings
- Variance decomposition (FEVD)

### When to Use Panel VAR?

**Use Panel VAR when:**
- You have panel data (multiple entities observed over time)
- Multiple endogenous variables with potential feedback effects
- Interest in dynamic interactions and shock propagation
- Variables are stationary (or first-differenced)

**Don't use Panel VAR when:**
- Single equation is sufficient (use dynamic panel models instead)
- Variables are non-stationary and cointegrated (use Panel VECM)
- Static relationships are of interest (use fixed effects)

---

## Part 1: Data Preparation and Exploration

### Step 1: Load Data

```python
import pandas as pd
import numpy as np
from panelbox.var import PanelVAR, PanelVARData

# Load example data (synthetic macro panel)
# Variables: GDP growth, inflation, policy rate
# 30 countries, 25 years (1995-2020)
data = pd.read_csv("macro_panel.csv")

# Inspect data
print(data.head())
```

**Expected output:**
```
   country  year  gdp_growth  inflation  policy_rate
0  USA      1995   3.2         2.8        5.5
1  USA      1996   3.7         2.9        5.3
...
```

### Step 2: Visualize Data

```python
import matplotlib.pyplot as plt

# Plot time series for selected countries
countries = ['USA', 'Germany', 'Japan']
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

for i, var in enumerate(['gdp_growth', 'inflation', 'policy_rate']):
    for country in countries:
        country_data = data[data['country'] == country]
        axes[i].plot(country_data['year'], country_data[var], label=country)

    axes[i].set_ylabel(var)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('Year')
plt.tight_layout()
plt.show()
```

---

## Part 2: Stationarity Tests (Pre-requisite)

Before estimating VAR, **always test for stationarity**. Panel VAR requires stationary variables.

### Step 3: Panel Unit Root Tests

```python
from panelbox.validation.unit_root import llc_test, ips_test

# Test each variable for unit roots
variables = ['gdp_growth', 'inflation', 'policy_rate']

for var in variables:
    # Levin-Lin-Chu test
    llc_result = llc_test(data, var, entity_col='country', time_col='year')
    print(f"\n{var} - LLC Test:")
    print(f"  Statistic: {llc_result.statistic:.4f}")
    print(f"  P-value: {llc_result.p_value:.4f}")
    print(f"  Result: {'Stationary ✓' if llc_result.reject else 'Non-stationary ✗'}")

    # Im-Pesaran-Shin test
    ips_result = ips_test(data, var, entity_col='country', time_col='year')
    print(f"\n{var} - IPS Test:")
    print(f"  Statistic: {ips_result.statistic:.4f}")
    print(f"  P-value: {ips_result.p_value:.4f}")
    print(f"  Result: {'Stationary ✓' if ips_result.reject else 'Non-stationary ✗'}")
```

**Interpretation:**
- If p-value < 0.05: Variable is stationary → Can proceed with VAR
- If p-value > 0.05: Variable has unit root → Need to difference or use VECM

**Common pitfall:** Using Panel VAR with I(1) variables leads to spurious regressions!

---

## Part 3: Lag Order Selection

### Step 4: Select Optimal Lag Order

```python
from panelbox.var import lag_order_selection

# Prepare data
pvar_data = PanelVARData(
    data,
    endog_vars=['gdp_growth', 'inflation', 'policy_rate'],
    entity_col='country',
    time_col='year'
)

# Test lags 1-4
lag_results = lag_order_selection(pvar_data, max_lags=4)
print(lag_results)
```

**Expected output:**
```
   Lag  AIC     BIC     HQIC
0  1    1250.3  1280.5  1262.1
1  2    1245.8  1290.3  1263.4  ← Minimum AIC
2  3    1248.2  1307.1  1271.9
3  4    1252.7  1326.0  1282.5
```

**Recommendation:** Choose lag order that minimizes information criterion (usually AIC or BIC).

---

## Part 4: Estimate Panel VAR (OLS)

### Step 5: Estimate with OLS

```python
# Create PanelVARData object
pvar_data = PanelVARData(
    data,
    endog_vars=['gdp_growth', 'inflation', 'policy_rate'],
    entity_col='country',
    time_col='year',
    lags=2  # From lag selection
)

# Estimate Panel VAR with OLS
model = PanelVAR(pvar_data)
result = model.fit(method='ols', cov_type='clustered')

# Summary
print(result.summary())
```

**Output includes:**
- Coefficient estimates (A₁, A₂ matrices)
- Standard errors (cluster-robust)
- T-statistics and p-values
- R² for each equation
- System information criteria (AIC, BIC, HQIC)

---

## Part 5: Model Diagnostics

### Step 6: Check Stability

```python
# Stability test (eigenvalues of companion matrix)
is_stable = result.stability_test()
print(f"VAR is stable: {is_stable}")

if not is_stable:
    print("⚠ Warning: VAR is unstable. IRFs may diverge!")
    print("Consider:")
    print("  - Reducing lag order")
    print("  - Differencing variables")
    print("  - Checking for outliers")
```

**Stability condition:** All eigenvalues must have modulus < 1.

### Step 7: Visualize Eigenvalues

```python
# Plot stability diagram
fig = result.plot_stability()
plt.show()
```

---

## Part 6: GMM Estimation (Advanced)

For potentially endogenous regressors, use GMM:

### Step 8: Estimate with GMM

```python
# GMM estimation with first-orthogonal deviations
result_gmm = model.fit(
    method='gmm',
    transform='fod',  # First-orthogonal deviations (recommended)
    instruments='collapsed',  # Avoid too many instruments
    gmm_step='two-step'  # Two-step GMM
)

print(result_gmm.summary())
```

### Step 9: GMM Diagnostics

```python
# Hansen J test (overidentifying restrictions)
print(f"\nHansen J test:")
print(f"  Statistic: {result_gmm.hansen_j:.4f}")
print(f"  P-value: {result_gmm.hansen_j_pvalue:.4f}")
print(f"  Result: {'Valid instruments ✓' if result_gmm.hansen_j_pvalue > 0.10 else 'Reject instruments ✗'}")

# AR(1) and AR(2) tests (serial correlation)
print(f"\nAR(1) test:")
print(f"  Statistic: {result_gmm.ar1_stat:.4f}")
print(f"  P-value: {result_gmm.ar1_pvalue:.4f}")
print(f"  Expected: Reject (negative correlation expected)")

print(f"\nAR(2) test:")
print(f"  Statistic: {result_gmm.ar2_stat:.4f}")
print(f"  P-value: {result_gmm.ar2_pvalue:.4f}")
print(f"  Expected: Cannot reject (no AR(2))")
```

---

## Part 7: Granger Causality Tests

### Step 10: Pairwise Granger Causality

```python
# Test if inflation Granger-causes GDP growth
granger_result = result.granger_causality('inflation', 'gdp_growth')
print(f"\nInflation → GDP growth:")
print(f"  Wald statistic: {granger_result.statistic:.4f}")
print(f"  P-value: {granger_result.p_value:.4f}")
print(f"  Result: {'Granger-causes ✓' if granger_result.reject else 'No Granger causality'}")

# Test reverse direction
granger_result2 = result.granger_causality('gdp_growth', 'inflation')
print(f"\nGDP growth → Inflation:")
print(f"  P-value: {granger_result2.p_value:.4f}")
```

### Step 11: Granger Causality Matrix

```python
# Get full matrix of p-values
granger_matrix = result.granger_causality_matrix()
print("\nGranger Causality Matrix (p-values):")
print(granger_matrix)
```

**Output:**
```
              gdp_growth  inflation  policy_rate
gdp_growth    1.000       0.023      0.450
inflation     0.012       1.000      0.001
policy_rate   0.340       0.008      1.000
```

**Interpretation:**
- inflation → gdp_growth (p=0.023): Significant at 5%
- gdp_growth → inflation (p=0.012): Significant at 5%
- Bidirectional causality between GDP growth and inflation

### Step 12: Causality Network Visualization

```python
from panelbox.var.causality_network import plot_causality_network

# Interactive network graph
fig = plot_causality_network(
    granger_matrix,
    threshold=0.05,
    layout='circular',
    backend='plotly'
)
fig.show()
```

---

## Part 8: Impulse Response Functions (IRFs)

IRFs show how the system responds to a shock in one variable.

### Step 13: Compute IRFs (Cholesky)

```python
# IRF: How does a shock to policy rate affect other variables?
irf_result = result.irf(
    periods=20,
    impulse='policy_rate',
    response=None,  # All variables
    method='cholesky',
    ci_method='bootstrap',
    n_bootstrap=500,
    ci_level=0.95
)

# Plot
fig = irf_result.plot()
plt.show()
```

**Interpretation example:**
- 1 pp increase in policy rate
- GDP growth declines by 0.3 pp after 2 years
- Inflation declines by 0.5 pp after 3 years
- Effects persist for ~5 years

### Step 14: Generalized IRFs

Generalized IRFs don't depend on variable ordering:

```python
# Generalized IRF (order-invariant)
irf_gen = result.irf(
    periods=20,
    method='generalized',
    ci_method='bootstrap',
    n_bootstrap=500
)

fig = irf_gen.plot()
plt.show()
```

### Step 15: Cumulative IRFs

```python
# Cumulative effect
irf_cum = result.irf(periods=20, cumulative=True)
fig = irf_cum.plot()
plt.show()
```

---

## Part 9: Forecast Error Variance Decomposition (FEVD)

FEVD shows what fraction of forecast error variance is due to each shock.

### Step 16: Compute FEVD

```python
# FEVD at horizon 20
fevd_result = result.fevd(
    periods=20,
    method='cholesky'
)

# Plot
fig = fevd_result.plot()
plt.show()
```

**Interpretation:**
At horizon 10:
- 60% of GDP growth forecast error due to own shocks
- 30% due to inflation shocks
- 10% due to policy rate shocks

---

## Part 10: Forecasting

### Step 17: Generate Forecasts

```python
# Forecast 10 periods ahead
forecast_result = result.forecast(
    steps=10,
    ci_method='bootstrap',
    n_bootstrap=500,
    ci_level=0.95
)

# Plot forecast for USA
fig = forecast_result.plot(
    entity='USA',
    variable='gdp_growth',
    show=True
)
```

### Step 18: Evaluate Forecast Accuracy

```python
# If test data is available
test_data = pd.read_csv("macro_panel_test.csv")

metrics = forecast_result.evaluate(test_data)
print(f"\nForecast Evaluation:")
print(f"  RMSE: {metrics['RMSE']:.4f}")
print(f"  MAE: {metrics['MAE']:.4f}")
print(f"  MAPE: {metrics['MAPE']:.2f}%")
```

---

## Part 11: Reporting Results

### Step 19: Generate HTML Report

```python
from panelbox.report import HTMLReport

# Create report
report = HTMLReport(title="Panel VAR Analysis: Monetary Policy Transmission")

# Add sections
report.add_section("Model Summary", result.summary())
report.add_section("Stability Test", f"VAR is stable: {result.stability_test()}")
report.add_figure("Impulse Responses", irf_result.plot())
report.add_figure("FEVD", fevd_result.plot())
report.add_section("Granger Causality", granger_matrix.to_html())

# Save
report.save("panel_var_report.html")
print("Report saved to panel_var_report.html")
```

---

## Part 12: When to Use Panel VECM Instead

If variables are I(1) and cointegrated, use Panel VECM:

### Step 20: Test for Cointegration

```python
from panelbox.validation.cointegration import pedroni_test

# Test for cointegration
coint_result = pedroni_test(
    data,
    endog_vars=['gdp_level', 'consumption_level', 'investment_level'],
    entity_col='country',
    time_col='year'
)

if coint_result.reject:
    print("Variables are cointegrated → Use Panel VECM")

    # Estimate VECM instead
    from panelbox.var import PanelVECM

    vecm_data = PanelVARData(
        data,
        endog_vars=['gdp_level', 'consumption_level', 'investment_level'],
        entity_col='country',
        time_col='year',
        lags=2
    )

    vecm = PanelVECM(vecm_data)
    vecm_result = vecm.fit(rank=1)
    print(vecm_result.summary())
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Using Panel VAR with I(1) Variables

**Problem:** Spurious regression if variables have unit roots

**Solution:**
1. Test for unit roots (LLC, IPS tests)
2. If I(1): Difference variables OR use Panel VECM (if cointegrated)

### Pitfall 2: Too Many GMM Instruments

**Problem:** Hansen J test loses power, bias toward OLS

**Solution:**
- Use `instruments='collapsed'`
- Check `result_gmm.n_instruments` (should be < N)
- Verify Hansen J p-value > 0.10

### Pitfall 3: Ignoring Variable Ordering

**Problem:** Cholesky IRFs depend on arbitrary variable order

**Solution:**
- Justify ordering theoretically (e.g., policy→inflation→output)
- Use Generalized IRFs (order-invariant)
- Compare IRFs under different orderings

### Pitfall 4: Unstable VAR

**Problem:** Eigenvalues > 1 → IRFs explode

**Solution:**
- Reduce lag order
- Difference variables
- Check for outliers/structural breaks

---

## Decision Tree: Which Model to Use?

```
Are variables stationary?
├─ YES → Continue
└─ NO → Are they cointegrated?
    ├─ YES → Use Panel VECM
    └─ NO → Difference variables, then Panel VAR

Is there endogeneity concern?
├─ YES → Use GMM
└─ NO → Use OLS (faster)

Multiple endogenous variables?
├─ YES → Panel VAR
└─ NO → Single equation (Arellano-Bond, Difference GMM)

Need shock propagation analysis?
├─ YES → Panel VAR (IRFs, FEVD)
└─ NO → Static panel models may suffice
```

---

## Summary

This tutorial covered:

1. ✓ Data preparation and visualization
2. ✓ Stationarity testing (pre-requisite)
3. ✓ Lag order selection
4. ✓ OLS estimation
5. ✓ Stability diagnostics
6. ✓ GMM estimation (advanced)
7. ✓ Granger causality (tests + visualization)
8. ✓ Impulse Response Functions
9. ✓ Variance decomposition (FEVD)
10. ✓ Forecasting
11. ✓ Reporting
12. ✓ When to use VECM instead

**Key takeaways:**
- Always test for stationarity first
- Check stability before interpreting IRFs
- GMM diagnostics are critical (Hansen J, AR tests)
- Generalized IRFs avoid ordering issues
- Panel VECM for cointegrated I(1) variables

**Next steps:**
- Apply to your own data
- Experiment with different lag orders
- Compare OLS vs GMM
- Explore sensitivity to variable ordering

---

## References

- Love, I., & Zicchino, L. (2006). "Financial development and dynamic investment behavior: Evidence from panel VAR." *The Quarterly Review of Economics and Finance*, 46(2), 190-210.

- Abrigo, M. R., & Love, I. (2016). "Estimation of panel vector autoregression in Stata." *The Stata Journal*, 16(3), 778-804.

- Holtz-Eakin, D., Newey, W., & Rosen, H. S. (1988). "Estimating vector autoregressions with panel data." *Econometrica*, 56(6), 1371-1395.

---

**For more information:**
- API documentation: `help(PanelVAR)`
- Theory guide: `docs/theory/panel_var_theory.md`
- FAQ: `docs/how-to/var_faq.md`
