---
title: "Unit Root Tests"
description: "Panel unit root tests for stationarity analysis in PanelBox — LLC, IPS, Fisher, Hadri, and Breitung tests for detecting non-stationarity in panel data."
---

# Unit Root Tests

!!! info "Quick Reference"
    **Module:** `panelbox.validation.unit_root` (LLC, IPS, Fisher) and `panelbox.diagnostics.unit_root` (Hadri, Breitung)
    **Purpose:** Test whether panel data series are stationary or contain unit roots
    **Stata equivalent:** `xtunitroot llc`, `xtunitroot ips`, `xtunitroot fisher`, `xtunitroot hadri`, `xtunitroot breitung`
    **R equivalent:** `plm::purtest()`

## Why Stationarity Matters

Non-stationary data creates serious problems for panel data analysis:

- **Spurious regression**: Regressions between unrelated I(1) variables produce significant-looking but meaningless results (Granger & Newbold, 1974)
- **Invalid inference**: Standard t-statistics and F-tests have non-standard distributions when variables contain unit roots
- **Inconsistent estimators**: OLS estimates may not converge to true parameter values
- **Model misspecification**: Ignoring unit roots leads to incorrect dynamic models

Testing for unit roots is a **prerequisite** before estimating panel models. If variables are I(1), you should either first-difference the data or test for cointegration.

## Unit Root in Panel Data

Panel unit root tests extend the univariate ADF/PP framework to panel settings. The panel dimension provides **more power** because the cross-sectional variation supplements the time-series variation.

The general model is:

$$\Delta y_{it} = \rho_i y_{i,t-1} + \sum_{j=1}^{p_i} \phi_{ij} \Delta y_{i,t-j} + \alpha_i + \delta_i t + \varepsilon_{it}$$

where:

- $y_{it}$ is the variable of interest for entity $i$ at time $t$
- $\rho_i$ is the autoregressive parameter (unit root if $\rho_i = 0$)
- $\alpha_i$ is an entity-specific intercept (fixed effect)
- $\delta_i t$ is an entity-specific linear trend
- $p_i$ is the number of augmented lags

## Five Tests Compared

PanelBox implements five panel unit root tests, each with different assumptions:

| Test | H₀ | H₁ | Root Type | Heterogeneous? | Unbalanced? |
|:-----|:----|:----|:----------|:--------------:|:-----------:|
| [LLC](llc.md) | Unit root | All stationary | Common $\rho$ | No | No |
| [IPS](ips.md) | Unit root | Some stationary | Individual $\rho_i$ | Yes | Limited |
| [Fisher](fisher.md) | Unit root | Some stationary | Individual $\rho_i$ | Yes | Yes |
| [Hadri](hadri.md) | **Stationary** | Some unit roots | -- | No | No |
| [Breitung](breitung.md) | Unit root | All stationary | Common $\rho$ | No | No |

### Key Distinctions

**Common vs. heterogeneous autoregressive parameter:**

- **LLC and Breitung** assume a **common** $\rho$ across all entities. This is appropriate when entities share similar dynamics (e.g., exchange rates under a common monetary regime).
- **IPS and Fisher** allow **heterogeneous** $\rho_i$ per entity. This is more realistic when entities have different adjustment speeds (e.g., GDP across countries with different economic structures).

!!! warning "Hadri: Opposite Null Hypothesis"
    The Hadri test has the **opposite** null hypothesis from all other tests. Its H₀ is **stationarity**, not unit root. Rejecting H₀ in the Hadri test means evidence **against** stationarity -- the opposite of what rejecting H₀ means in LLC/IPS/Fisher/Breitung.

## Trend Specification

All tests support deterministic trend specifications that control for entity-specific intercepts and trends:

| Specification | Code | Description | When to Use |
|:-------------|:-----|:-----------|:------------|
| No deterministic terms | `trend='n'` | No intercept or trend | Returns, growth rates, already-demeaned data |
| Constant only | `trend='c'` | Entity-specific intercept | Default; most common for levels |
| Constant + trend | `trend='ct'` | Intercept and linear trend | Variables with deterministic trends (e.g., log GDP) |

!!! note
    Hadri and Breitung only support `trend='c'` and `trend='ct'` (no `'n'` option).

## Lag Selection

Lag selection controls the number of augmented terms $\Delta y_{i,t-j}$ included to account for serial correlation in the errors:

- **Automatic** (`lags=None`): Selected by AIC. Recommended for most applications.
- **Manual** (`lags=k`): User-specified fixed lag length for all entities.
- **Per-entity** (IPS only, `lags=dict`): Different lag orders per entity via a dictionary mapping entity to lag count.

A common rule of thumb for maximum lag length: $p_{max} = \lfloor 12 (T/100)^{1/3} \rfloor$.

## Quick Example: Battery of Tests

The recommended approach is to run multiple tests as a battery for robustness:

```python
from panelbox.datasets import load_grunfeld
from panelbox.validation.unit_root.llc import LLCTest
from panelbox.validation.unit_root.ips import IPSTest
from panelbox.validation.unit_root.fisher import FisherTest
from panelbox.diagnostics.unit_root.hadri import hadri_test
from panelbox.diagnostics.unit_root.breitung import breitung_test

# Load data
data = load_grunfeld()

# --- Test battery on 'invest' variable ---

# 1. LLC: Common unit root (H0: unit root)
llc = LLCTest(data, "invest", "firm", "year", trend="c")
llc_result = llc.run()
print(f"LLC:      stat={llc_result.statistic:.4f}, p={llc_result.pvalue:.4f}")

# 2. IPS: Heterogeneous unit root (H0: unit root)
ips = IPSTest(data, "invest", "firm", "year", trend="c")
ips_result = ips.run()
print(f"IPS:      stat={ips_result.statistic:.4f}, p={ips_result.pvalue:.4f}")

# 3. Fisher-ADF: Combining p-values (H0: unit root)
fisher = FisherTest(data, "invest", "firm", "year", test_type="adf", trend="c")
fisher_result = fisher.run()
print(f"Fisher:   stat={fisher_result.statistic:.4f}, p={fisher_result.pvalue:.4f}")

# 4. Breitung: Debiased common root (H0: unit root)
breit_result = breitung_test(data, "invest", "firm", "year", trend="ct")
print(f"Breitung: stat={breit_result.statistic:.4f}, p={breit_result.pvalue:.4f}")

# 5. Hadri: Stationarity null (H0: stationary -- REVERSED!)
hadri_result = hadri_test(data, "invest", "firm", "year", trend="c")
print(f"Hadri:    stat={hadri_result.statistic:.4f}, p={hadri_result.pvalue:.4f}")
```

## Testing Strategy

### Step 1: Run the main battery

Run LLC + IPS + Fisher on the variable in levels. These all test H₀: unit root.

### Step 2: Confirm with Hadri

Run the Hadri test (H₀: stationarity) to confirm the findings from Step 1:

| LLC/IPS/Fisher | Hadri | Conclusion |
|:--------------|:------|:-----------|
| Not reject H₀ (unit root) | Reject H₀ (not stationary) | **Unit root confirmed** |
| Reject H₀ (stationary) | Not reject H₀ (stationary) | **Stationarity confirmed** |
| Reject H₀ | Reject H₀ | Mixed evidence -- investigate further |
| Not reject H₀ | Not reject H₀ | Mixed evidence -- investigate further |

### Step 3: Test first differences

If variables are I(1) in levels, verify that first differences are I(0) to confirm integration order:

```python
# First difference the variable
data["d_invest"] = data.groupby("firm")["invest"].diff()
data_diff = data.dropna(subset=["d_invest"])

# Test first differences (should be stationary)
llc_diff = LLCTest(data_diff, "d_invest", "firm", "year", trend="c")
result_diff = llc_diff.run()
print(f"LLC on first differences: p={result_diff.pvalue:.4f}")
# Expect: small p-value (reject unit root in differences)
```

### Step 4: Next steps

- If variables are **I(0)**: Estimate models in levels (Fixed Effects, Random Effects, etc.)
- If variables are **I(1)**: Either first-difference the data **or** test for [cointegration](../cointegration/index.md)
- If variables are **cointegrated**: Use Panel VECM or estimate long-run relationships

## What If a Unit Root Is Detected?

1. **First-difference the data** to achieve stationarity:

    ```python
    data["dy"] = data.groupby("entity")["y"].diff()
    ```

2. **Use GMM estimators** designed for dynamic panel data:

    ```python
    from panelbox.gmm import DifferenceGMM
    ```

3. **Test for cointegration** if multiple I(1) variables may share a long-run equilibrium:

    ```python
    from panelbox.validation.cointegration.pedroni import PedroniTest
    from panelbox.validation.cointegration.kao import KaoTest
    ```

4. **Include time effects** to control for common trends across entities:

    ```python
    from panelbox.models import FixedEffects
    ```

## Software Comparison

| Test | PanelBox | Stata | R |
|:-----|:---------|:------|:--|
| LLC | `LLCTest` | `xtunitroot llc` | `plm::purtest(test="levinlin")` |
| IPS | `IPSTest` | `xtunitroot ips` | `plm::purtest(test="ips")` |
| Fisher | `FisherTest` | `xtunitroot fisher` | `plm::purtest(test="madwu")` |
| Breitung | `breitung_test()` | `xtunitroot breitung` | `plm::purtest(test="Breitung")` |
| Hadri | `hadri_test()` | `xtunitroot hadri` | `plm::purtest(test="hadri")` |

## See Also

- [LLC Test](llc.md) -- Common unit root test (Levin-Lin-Chu)
- [IPS Test](ips.md) -- Heterogeneous unit root test (Im-Pesaran-Shin)
- [Fisher Test](fisher.md) -- Combining individual p-values (Maddala-Wu)
- [Hadri Test](hadri.md) -- Stationarity null test (reversed hypothesis)
- [Breitung Test](breitung.md) -- Debiased common unit root test
- [Cointegration Tests](../cointegration/index.md) -- Testing for long-run relationships between I(1) variables

## References

- Levin, A., Lin, C. F., & Chu, C. S. J. (2002). "Unit root tests in panel data: asymptotic and finite-sample properties." *Journal of Econometrics*, 108(1), 1-24.
- Im, K. S., Pesaran, M. H., & Shin, Y. (2003). "Testing for unit roots in heterogeneous panels." *Journal of Econometrics*, 115(1), 53-74.
- Maddala, G. S., & Wu, S. (1999). "A comparative study of unit root tests with panel data and a new simple test." *Oxford Bulletin of Economics and Statistics*, 61(S1), 631-652.
- Hadri, K. (2000). "Testing for stationarity in heterogeneous panel data." *Econometrics Journal*, 3(2), 148-161.
- Breitung, J. (2000). "The local power of some unit root tests for panel data." In *Advances in Econometrics*, Vol. 15, 161-177.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 12.
