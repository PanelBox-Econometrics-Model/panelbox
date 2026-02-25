---
title: "Cointegration Tests"
description: "Panel cointegration tests in PanelBox — Pedroni, Kao, and Westerlund tests for long-run equilibrium relationships between non-stationary panel variables."
---

# Cointegration Tests

!!! info "Quick Reference"
    **Module:** `panelbox.validation.cointegration` (Pedroni, Kao) and `panelbox.diagnostics.cointegration` (Westerlund)
    **Purpose:** Test whether non-stationary I(1) variables share a long-run equilibrium relationship
    **H₀:** No cointegration (all tests)
    **Stata equivalent:** `xtpedroni`, `xtcointtest kao`, `xtwest`
    **R equivalent:** `plm::cipstest()`, `urca::ca.jo()` (adapted)

## What Is Cointegration?

**Cointegration** describes a long-run equilibrium relationship between non-stationary variables. While individual series may each be I(1) -- containing a unit root and drifting over time -- a linear combination of them may be stationary (I(0)):

$$y_{it} \sim I(1), \quad x_{it} \sim I(1), \quad \text{but} \quad y_{it} - \beta x_{it} \sim I(0)$$

This means the variables may deviate from equilibrium in the short run, but are tied together in the long run by economic forces.

**Economic examples:**

- **Purchasing Power Parity (PPP)**: Exchange rates and relative price levels
- **Consumption-Income**: Long-run propensity to consume
- **Interest Rate Parity**: Domestic and foreign interest rates
- **Growth convergence**: GDP levels across countries

## Why Test for Cointegration?

Testing for cointegration is critical because:

1. **Spurious regression**: Regressing one I(1) variable on another I(1) variable without cointegration produces spurious results -- significant t-statistics and high $R^2$ even with unrelated variables
2. **Model selection**: If variables are cointegrated, you should estimate a VECM (Vector Error Correction Model) rather than a VAR in differences
3. **Long-run relationships**: Cointegration validates the existence of meaningful long-run equilibrium relationships between economic variables
4. **Error correction**: The speed of adjustment to equilibrium ($\alpha$) has important economic interpretation

## Three Tests Compared

PanelBox implements three panel cointegration tests using different approaches:

| Test | Approach | H₀ | Statistics | Heterogeneous $\beta$? | Bootstrap? |
|:-----|:---------|:---|:-----------|:---------------------:|:----------:|
| [Pedroni](pedroni.md) | Residual-based | No cointegration | 7 (4 panel + 3 group) | Yes | No |
| [Kao](kao.md) | Residual-based | No cointegration | 1 (ADF-type) | No | No |
| [Westerlund](westerlund.md) | ECM-based | No cointegration | 4 ($G_\tau$, $G_\alpha$, $P_\tau$, $P_\alpha$) | Yes | Yes |

### Key Distinctions

**Residual-based vs. ECM-based:**

- **Pedroni and Kao** are residual-based: they estimate a cointegrating regression, then test whether the residuals have a unit root. If residuals are I(0), cointegration exists.
- **Westerlund** is ECM-based: it directly tests whether the error correction mechanism is active ($\alpha_i < 0$). This is more powerful because it tests the adjustment process directly.

**Homogeneous vs. heterogeneous cointegrating vector:**

- **Kao** assumes a **common** $\beta$ across all entities -- the same long-run relationship everywhere
- **Pedroni** and **Westerlund** allow **heterogeneous** $\beta_i$ -- different long-run relationships per entity

## Testing Workflow

### Step 1: Confirm variables are I(1)

Before testing for cointegration, verify that each variable individually has a unit root:

```python
from panelbox.datasets import load_grunfeld
from panelbox.validation.unit_root.ips import IPSTest

data = load_grunfeld()

# Test each variable for unit root
for var in ["invest", "value", "capital"]:
    ips = IPSTest(data, var, "firm", "year", trend="c")
    result = ips.run()
    print(f"{var:10s}: W={result.statistic:7.3f}, p={result.pvalue:.4f} "
          f"-- {'I(1)' if result.pvalue >= 0.05 else 'I(0)'}")
```

!!! warning "Prerequisites"
    Cointegration tests are only meaningful when variables are individually non-stationary (I(1)). If variables are already I(0), use standard regression methods instead.

### Step 2: Run cointegration test battery

Run Pedroni + Kao + Westerlund for robustness:

```python
from panelbox.validation.cointegration.pedroni import PedroniTest
from panelbox.validation.cointegration.kao import KaoTest
from panelbox.diagnostics.cointegration.westerlund import westerlund_test

# Pedroni (7 statistics)
pedroni = PedroniTest(data, "invest", ["value", "capital"],
                       "firm", "year", trend="c")
pedroni_result = pedroni.run()
print(pedroni_result)

# Kao (ADF-type)
kao = KaoTest(data, "invest", ["value", "capital"],
               "firm", "year", trend="c")
kao_result = kao.run()
print(kao_result)

# Westerlund (ECM-based with bootstrap)
west_result = westerlund_test(data, "firm", "year",
                               "invest", ["value", "capital"],
                               n_bootstrap=200)
print(west_result)
```

### Step 3: Interpret results

**Consensus approach** -- look for agreement across tests:

| Evidence Level | Criteria | Action |
|:--------------|:---------|:-------|
| **Strong** | 2+ test families reject H₀ | Cointegration exists; estimate long-run model |
| **Moderate** | 1 test family rejects consistently | Likely cointegration; verify with robustness checks |
| **Weak** | Mixed results across tests | Inconclusive; consider alternative specifications |
| **None** | No tests reject H₀ | No cointegration; use differenced variables |

### Step 4: Next steps

If cointegration is confirmed:

- Estimate the long-run relationship using FMOLS or DOLS
- Use Panel VECM to model both short-run dynamics and long-run equilibrium
- Interpret the error correction speed of adjustment $\alpha$

If no cointegration is found:

- First-difference all variables
- Estimate models in differences (e.g., VAR in differences)
- Re-examine variable selection

## Comparison with Time-Series Tests

Panel cointegration tests have **more power** than single-equation tests because they exploit the cross-sectional dimension:

| Feature | Time-Series (e.g., Johansen) | Panel (Pedroni/Kao/Westerlund) |
|:--------|:----------------------------|:-------------------------------|
| Power | Limited by T | Enhanced by N and T |
| Heterogeneity | Single entity | Can accommodate N entities |
| Sample size | T must be large | N compensates for small T |
| Asymptotics | T only | N and T jointly |

## Choosing the Right Test

| Scenario | Recommended Test | Reason |
|:---------|:----------------|:-------|
| General purpose, heterogeneous panels | Pedroni | 7 statistics, flexible $\beta_i$ |
| Quick check, homogeneous panels | Kao | Simple, single statistic |
| Cross-sectional dependence suspected | Westerlund (with bootstrap) | Bootstrap corrects for CD |
| Small N, large T | Westerlund | ECM-based, more powerful |
| Comprehensive robustness | Pedroni + Kao + Westerlund | Different approaches |

## Software Comparison

| Test | PanelBox | Stata | R |
|:-----|:---------|:------|:--|
| Pedroni | `PedroniTest` | `xtpedroni` | Custom |
| Kao | `KaoTest` | `xtcointtest kao` | Custom |
| Westerlund | `westerlund_test()` | `xtwest` | Custom |

## See Also

- [Pedroni Test](pedroni.md) -- Heterogeneous residual-based cointegration (7 statistics)
- [Kao Test](kao.md) -- Homogeneous residual-based cointegration (ADF-type)
- [Westerlund Test](westerlund.md) -- ECM-based cointegration with bootstrap
- [Unit Root Tests](../unit-root/index.md) -- Prerequisite: testing for I(1)

## References

- Pedroni, P. (1999). "Critical values for cointegration tests in heterogeneous panels with multiple regressors." *Oxford Bulletin of Economics and Statistics*, 61(S1), 653-670.
- Pedroni, P. (2004). "Panel cointegration: asymptotic and finite sample properties of pooled time series tests with an application to the PPP hypothesis." *Econometric Theory*, 20(3), 597-625.
- Kao, C. (1999). "Spurious regression and residual-based tests for cointegration in panel data." *Journal of Econometrics*, 90(1), 1-44.
- Westerlund, J. (2007). "Testing for error correction in panel data." *Oxford Bulletin of Economics and Statistics*, 69(6), 709-748.
- Engle, R. F., & Granger, C. W. J. (1987). "Co-integration and error correction: representation, estimation, and testing." *Econometrica*, 55(2), 251-276.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 12.
