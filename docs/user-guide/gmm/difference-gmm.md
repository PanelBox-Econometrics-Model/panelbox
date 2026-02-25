---
title: "Difference GMM (Arellano-Bond)"
description: "Arellano-Bond (1991) Difference GMM estimator for dynamic panel data with fixed effects, using first-differencing and lagged levels as instruments."
---

# Difference GMM (Arellano-Bond)

!!! info "Quick Reference"
    **Class:** `panelbox.gmm.DifferenceGMM`
    **Import:** `from panelbox.gmm import DifferenceGMM`
    **Stata equivalent:** `xtabond2 y L.y x1 x2, gmm(y, lag(2 .)) iv(x1 x2) noleveleq`
    **R equivalent:** `pgmm(y ~ lag(y, 1) + x1 + x2 | lag(y, 2:99), transformation = "d")`

## Overview

Difference GMM, introduced by Arellano and Bond (1991), is the foundational estimator for **dynamic panel data** models where a lagged dependent variable appears as a regressor alongside individual fixed effects. Standard estimators (OLS, Fixed Effects, Random Effects) all produce biased and inconsistent estimates in this setting.

The key insight is to **first-difference** the equation to eliminate fixed effects, then use **lagged levels** of the dependent variable as instruments for the differenced equation. This yields consistent estimates under the assumption that the original errors are serially uncorrelated.

Difference GMM is the workhorse estimator for dynamic panels with **short T** (few time periods) and **large N** (many cross-sectional units). PanelBox's implementation provides feature parity with Stata's `xtabond2`, including collapsed instruments (Roodman 2009) and the Windmeijer (2005) finite-sample correction for two-step standard errors.

## Quick Example

```python
from panelbox.gmm import DifferenceGMM
from panelbox.datasets import load_abdata

# Load Arellano-Bond employment dataset
data = load_abdata()

# Estimate Difference GMM
model = DifferenceGMM(
    data=data,
    dep_var="n",
    lags=1,
    id_var="id",
    time_var="year",
    exog_vars=["w", "k"],
    collapse=True,
    two_step=True,
    robust=True,
)
results = model.fit()
print(results.summary())
```

## When to Use

- **Dynamic panel models** where $y_{it}$ depends on $y_{i,t-1}$
- **Short panels** with small T (typically T < 20) and large N
- **Fixed effects** are correlated with regressors (rules out RE)
- **Strict exogeneity fails** (some regressors are endogenous or predetermined)
- **Moderate persistence**: AR coefficient $\gamma < 0.8$ (for persistent series, consider [System GMM](system-gmm.md))

!!! warning "Key Assumptions"
    1. **No serial correlation** in idiosyncratic errors: $E[\varepsilon_{it} \varepsilon_{is}] = 0$ for $t \neq s$
    2. **Sequential exogeneity**: $E[y_{i,t-s} \varepsilon_{it}] = 0$ for $s \geq 1$ (lagged levels are valid instruments)
    3. **Large N asymptotics**: Consistency relies on $N \to \infty$ with T fixed
    4. **Initial conditions**: $E[y_{i1} \varepsilon_{it}] = 0$ for $t \geq 2$

## Detailed Guide

### The Dynamic Panel Problem

Consider the standard dynamic panel model:

$$y_{it} = \gamma y_{i,t-1} + X_{it}'\beta + \alpha_i + \varepsilon_{it}$$

where $\alpha_i$ is an unobserved individual fixed effect and $\varepsilon_{it}$ is the idiosyncratic error.

**Why OLS is biased upward:** The lagged dependent variable $y_{i,t-1}$ is correlated with $\alpha_i$ (since $\alpha_i$ affects all periods), producing omitted variable bias. OLS overestimates $\gamma$.

**Why Fixed Effects is biased (Nickell bias):** The within transformation creates correlation between the demeaned lagged variable $(y_{i,t-1} - \bar{y}_i)$ and the demeaned error $(\varepsilon_{it} - \bar{\varepsilon}_i)$, because $\bar{y}_i$ contains $y_{i,t-1}$ and $\bar{\varepsilon}_i$ contains $\varepsilon_{i,t-1}$. The bias is approximately $-(1 + \gamma)/(T - 1)$, which is severe for small T.

**Result:** For a true coefficient $\gamma$, we expect:

$$\hat{\gamma}_{OLS} > \gamma > \hat{\gamma}_{FE}$$

This provides **bounds** for validating GMM estimates.

### First-Differencing and Instruments

**Step 1:** First-difference to eliminate $\alpha_i$:

$$\Delta y_{it} = \gamma \Delta y_{i,t-1} + \Delta X_{it}'\beta + \Delta \varepsilon_{it}$$

**Step 2:** Use lagged levels as instruments. The key moment conditions are:

$$E[y_{i,t-s} \cdot \Delta \varepsilon_{it}] = 0 \quad \text{for } s \geq 2$$

This works because $y_{i,t-2}$ is predetermined (determined before $\varepsilon_{it}$) and uncorrelated with $\Delta \varepsilon_{it} = \varepsilon_{it} - \varepsilon_{i,t-1}$ under the assumption of no serial correlation in levels.

### One-Step vs Two-Step Estimation

**One-step GMM** uses a simple weighting matrix $W_1 = (Z'HZ)^{-1}$ where $H$ is a block-diagonal matrix based on the first-difference structure. It is consistent but not efficient.

**Two-step GMM** constructs an optimal weighting matrix from one-step residuals:

$$W_2 = \left(\frac{1}{N} \sum_i Z_i' \hat{u}_i \hat{u}_i' Z_i\right)^{-1}$$

Two-step is asymptotically efficient but has **downward-biased standard errors** in finite samples.

### Windmeijer (2005) Correction

The Windmeijer correction adjusts two-step standard errors for the estimation error in the weighting matrix. This correction is critical in practice and is **automatically applied** when `robust=True` (the default).

!!! tip "Best Practice"
    Always use `two_step=True` with `robust=True` to get efficient estimates with properly corrected standard errors.

### Data Preparation

```python
import pandas as pd
from panelbox.datasets import load_abdata

data = load_abdata()

# Check panel structure
print(f"Panels (N): {data['id'].nunique()}")
print(f"Time periods (T): {data['year'].nunique()}")
print(f"Observations: {len(data)}")
print(f"Balance: {'Balanced' if data.groupby('id').size().nunique() == 1 else 'Unbalanced'}")
```

### Estimation

```python
from panelbox.gmm import DifferenceGMM

model = DifferenceGMM(
    data=data,
    dep_var="n",              # Log employment
    lags=1,                   # AR(1) specification
    id_var="id",
    time_var="year",
    exog_vars=["w", "k"],     # Wages and capital (strictly exogenous)
    time_dummies=True,         # Include time fixed effects
    collapse=True,             # Collapse instruments (Roodman 2009)
    two_step=True,             # Two-step estimation
    robust=True,               # Windmeijer-corrected SEs
)

results = model.fit()
```

**With predetermined and endogenous variables:**

```python
model = DifferenceGMM(
    data=data,
    dep_var="n",
    lags=1,
    id_var="id",
    time_var="year",
    exog_vars=["policy"],            # Strictly exogenous
    predetermined_vars=["capital"],  # Instruments: t-2 and earlier
    endogenous_vars=["labor"],       # Instruments: t-3 and earlier
    collapse=True,
    two_step=True,
)
results = model.fit()
```

### Interpreting Results

```python
# Coefficient on lagged dependent variable
gamma = results.params["L1.n"]
se = results.std_errors["L1.n"]
print(f"Persistence: {gamma:.4f} (SE: {se:.4f})")

# 95% confidence interval
ci = results.conf_int()
print(f"95% CI: [{ci.loc['L1.n', 'lower']:.4f}, {ci.loc['L1.n', 'upper']:.4f}]")

# Diagnostic tests
print(f"AR(2) p-value: {results.ar2_test.pvalue:.4f}")
print(f"Hansen J p-value: {results.hansen_j.pvalue:.4f}")
print(f"Instrument ratio: {results.instrument_ratio:.3f}")
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | required | Panel data in long format |
| `dep_var` | `str` | required | Dependent variable name |
| `lags` | `int` or `list[int]` | required | Lags of dependent variable (e.g., `1` or `[1, 2]`) |
| `id_var` | `str` | `"id"` | Cross-sectional identifier |
| `time_var` | `str` | `"year"` | Time variable |
| `exog_vars` | `list[str]` | `None` | Strictly exogenous variables |
| `endogenous_vars` | `list[str]` | `None` | Endogenous variables (instrumented with t-3+) |
| `predetermined_vars` | `list[str]` | `None` | Predetermined variables (instrumented with t-2+) |
| `time_dummies` | `bool` | `True` | Include time fixed effects |
| `collapse` | `bool` | `False` | Collapse instruments (Roodman 2009) |
| `two_step` | `bool` | `True` | Use two-step estimation |
| `robust` | `bool` | `True` | Windmeijer-corrected standard errors |
| `gmm_type` | `str` | `"two_step"` | `"one_step"`, `"two_step"`, or `"iterative"` |
| `gmm_max_lag` | `int` or `None` | `None` | Maximum lag for GMM instruments (None = all) |
| `iv_max_lag` | `int` | `0` | Maximum lag for IV instruments of exogenous vars |

!!! tip "Recommended Settings"
    For most applications, use `collapse=True`, `two_step=True`, `robust=True`. Only change these for specific robustness checks.

## Diagnostics

### Essential Diagnostic Tests

```python
# 1. AR(2) test - CRITICAL: must NOT reject
ar2 = results.ar2_test
print(f"AR(2): z={ar2.statistic:.3f}, p={ar2.pvalue:.4f} [{ar2.conclusion}]")

# 2. Hansen J test - instruments validity
hansen = results.hansen_j
print(f"Hansen J: stat={hansen.statistic:.3f}, p={hansen.pvalue:.4f} [{hansen.conclusion}]")

# 3. Instrument ratio - overfitting check
print(f"Instruments: {results.n_instruments}, Groups: {results.n_groups}")
print(f"Instrument ratio: {results.instrument_ratio:.3f}")

# 4. AR(1) test - expected to reject
ar1 = results.ar1_test
print(f"AR(1): z={ar1.statistic:.3f}, p={ar1.pvalue:.4f} [{ar1.conclusion}]")
```

### Diagnostic Checklist

| Test | Criterion | Interpretation |
|------|-----------|----------------|
| AR(2) | p > 0.10 | Moment conditions valid |
| Hansen J | 0.10 < p < 0.25 | Instruments appear valid |
| Hansen J | p > 0.25 | Possible weak instruments |
| Hansen J | p < 0.10 | Instruments rejected |
| Instrument ratio | < 1.0 | No proliferation |
| AR(1) | p < 0.10 | Expected (informational) |

### Overfitting Diagnostics

```python
from panelbox.gmm import GMMOverfitDiagnostic

diag = GMMOverfitDiagnostic(model, results)
print(diag.summary())
```

For detailed diagnostic interpretation, see [GMM Diagnostics](diagnostics.md).

## Common Issues

| Problem | Symptom | Solution |
|---------|---------|----------|
| Too many instruments | Instrument ratio > 1.0, Hansen p near 1.0 | Use `collapse=True` |
| Weak instruments | Very large SEs, Hansen p > 0.50 | Try [System GMM](system-gmm.md) |
| Serial correlation | AR(2) p < 0.05 | Add more lags: `lags=[1, 2]` |
| Low observation retention | Warning about < 30% retention | Set `time_dummies=False`, use `collapse=True` |
| Coefficient outside bounds | Estimate > OLS or < FE | Check specification, reduce instruments |

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Complete GMM Guide | Step-by-step applied tutorial | [Complete Guide](complete-guide.md) |
| GMM Instruments | Instrument selection and management | [Instruments](instruments.md) |
| GMM Diagnostics | Interpreting all diagnostic tests | [Diagnostics](diagnostics.md) |

## See Also

- [System GMM](system-gmm.md) -- Blundell-Bond System GMM for persistent series
- [CUE-GMM](cue-gmm.md) -- Continuous Updating Estimator for robustness checks
- [Bias-Corrected GMM](bias-corrected.md) -- Analytical bias correction for moderate N, T
- [Instruments](instruments.md) -- Instrument selection and the proliferation problem
- [Diagnostics](diagnostics.md) -- Complete guide to GMM diagnostic tests
- [Complete Guide](complete-guide.md) -- Step-by-step applied GMM tutorial

## References

1. Arellano, M., & Bond, S. (1991). "Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations." *Review of Economic Studies*, 58(2), 277-297.
2. Roodman, D. (2009). "How to do xtabond2: An Introduction to Difference and System GMM in Stata." *The Stata Journal*, 9(1), 86-136.
3. Windmeijer, F. (2005). "A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators." *Journal of Econometrics*, 126(1), 25-51.
4. Nickell, S. (1981). "Biases in Dynamic Models with Fixed Effects." *Econometrica*, 49(6), 1417-1426.
5. Bond, S. R. (2002). "Dynamic Panel Data Models: A Guide to Micro Data Methods and Practice." *Portuguese Economic Journal*, 1(2), 141-162.
