---
title: "GMM Instruments"
description: "Guide to instrument selection, the proliferation problem, GMM-style vs IV-style instruments, and the InstrumentBuilder API in PanelBox."
---

# GMM Instruments

!!! info "Quick Reference"
    **Class:** `panelbox.gmm.instruments.InstrumentBuilder`
    **Import:** `from panelbox.gmm.instruments import InstrumentBuilder`
    **Overfit diagnostics:** `from panelbox.gmm import GMMOverfitDiagnostic`

## Overview

Instrument selection is one of the most critical decisions in GMM estimation. Too few instruments lead to imprecise estimates; too many cause **instrument proliferation**, which biases coefficients toward OLS/FE values and weakens specification tests. This page covers the theory and practice of instrument management in PanelBox.

The central rule of thumb, established by Roodman (2009), is simple: **the number of instruments should not exceed the number of groups (N)**. PanelBox provides tools to monitor and control instrument counts, including the `collapse` option (which reduces instruments from $O(T^2)$ to $O(T)$) and the `GMMOverfitDiagnostic` class for comprehensive overfitting checks.

## The Instrument Proliferation Problem

### Why It Matters

In GMM estimation, the instrument matrix grows with the number of time periods. Without collapsing, the number of instruments is:

$$\text{instruments} = \frac{(T-2)(T-1)}{2} \quad \text{(for one variable)}$$

For $T = 10$: 36 instruments. For $T = 20$: 171 instruments. This quadratic growth causes:

1. **Overfitting**: Too many instruments fit the endogenous part of the regressors too well
2. **Biased coefficients**: Estimates converge toward OLS or FE instead of the true value
3. **Weak specification tests**: Hansen J test loses power and always "passes"
4. **Numerical instability**: Weight matrix inversion becomes unreliable

### The Roodman Rule

!!! tip "Rule of Thumb"
    Keep the **instrument ratio** (instruments / groups) below 1.0:

    $$\frac{L}{N} < 1.0$$

    where $L$ is the number of instruments and $N$ is the number of groups.

| Ratio | Assessment | Action |
|-------|------------|--------|
| < 0.5 | Good | Proceed with confidence |
| 0.5 -- 1.0 | Acceptable | Monitor diagnostics |
| 1.0 -- 2.0 | Warning | Use `collapse=True`, reduce lags |
| > 2.0 | Problematic | Severe overfitting risk |

## GMM-Style vs IV-Style Instruments

PanelBox supports two instrument generation strategies, matching Stata's `xtabond2`:

### GMM-Style Instruments

GMM-style instruments create a **separate column for each available lag at each time period**, producing a sparse, block-diagonal instrument matrix. This is the standard approach for the lagged dependent variable and other endogenous/predetermined regressors.

```python
from panelbox.gmm.instruments import InstrumentBuilder

builder = InstrumentBuilder(data, id_var="id", time_var="year")

# GMM-style instruments for the dependent variable
# Uses y_{t-2}, y_{t-3}, ... as instruments for differenced equation
Z_gmm = builder.create_gmm_style_instruments(
    var="y",
    min_lag=2,
    max_lag=99,      # Use all available lags
    equation="diff",
    collapse=False,   # Full instrument set
)
print(f"Instruments (full): {Z_gmm.n_instruments}")
```

### GMM-Style Collapsed

Collapsed instruments combine all available lags into **one column per lag depth**, reducing the instrument count from $O(T^2)$ to $O(T)$:

```python
Z_collapsed = builder.create_gmm_style_instruments(
    var="y",
    min_lag=2,
    max_lag=99,
    equation="diff",
    collapse=True,   # Collapsed instruments
)
print(f"Instruments (collapsed): {Z_collapsed.n_instruments}")
```

!!! tip "Always Use Collapse"
    Roodman (2009) recommends `collapse=True` as best practice. Collapsed instruments provide better finite-sample properties, avoid overfitting, and maintain numerical stability.

### IV-Style Instruments

IV-style instruments create **one column per lag**, with observations placed at each time period. This is the standard approach for strictly exogenous variables.

```python
# IV-style instruments for exogenous variables
# In differenced equation: uses first-differences of the variable
Z_iv = builder.create_iv_style_instruments(
    var="x1",
    min_lag=0,       # Current value
    max_lag=0,       # Only current value (default for exogenous)
    equation="diff",
)
print(f"IV instruments: {Z_iv.n_instruments}")
```

### Combining Instruments

```python
# Combine GMM and IV instruments
Z_combined = builder.combine_instruments(Z_collapsed, Z_iv)

# Analyze instrument count
analysis = builder.instrument_count_analysis(Z_combined)
print(analysis)
```

## Variable Classification and Instrument Rules

How a variable is classified determines which lags are valid instruments:

| Variable Type | Parameter | Instrument Lags | Rationale |
|--------------|-----------|-----------------|-----------|
| Strictly exogenous | `exog_vars` | All lags and leads (IV-style) | Uncorrelated with all errors |
| Predetermined | `predetermined_vars` | $t-2$ and earlier (GMM-style) | Correlated with current but not future errors |
| Endogenous | `endogenous_vars` | $t-3$ and earlier (GMM-style) | Correlated with current and past errors |
| Lagged dependent | `lags` | $t-2$ and earlier (GMM-style) | Same as endogenous by construction |

```python
from panelbox.gmm import DifferenceGMM

model = DifferenceGMM(
    data=data,
    dep_var="y",
    lags=1,
    id_var="id",
    time_var="year",
    exog_vars=["policy"],           # Strictly exogenous: IV-style, lag 0
    predetermined_vars=["capital"],  # Predetermined: GMM-style, lag 2+
    endogenous_vars=["labor"],       # Endogenous: GMM-style, lag 3+
    collapse=True,
    two_step=True,
)
```

## Controlling Instrument Count

### Using `collapse=True`

The most effective way to reduce instrument count:

```python
# Without collapse: O(T^2) instruments
model_full = DifferenceGMM(
    data=data, dep_var="y", lags=1,
    exog_vars=["x1"], collapse=False,
    two_step=True,
)
results_full = model_full.fit()
print(f"Full: {results_full.n_instruments} instruments")

# With collapse: O(T) instruments
model_collapsed = DifferenceGMM(
    data=data, dep_var="y", lags=1,
    exog_vars=["x1"], collapse=True,
    two_step=True,
)
results_collapsed = model_collapsed.fit()
print(f"Collapsed: {results_collapsed.n_instruments} instruments")
```

### Limiting Maximum Lag Depth

The `gmm_max_lag` parameter limits how deep GMM instruments go:

```python
# Use only lags 2 and 3 (instead of all available)
model = DifferenceGMM(
    data=data,
    dep_var="y",
    lags=1,
    exog_vars=["x1"],
    collapse=True,
    gmm_max_lag=3,  # Only use y_{t-2} and y_{t-3}
    two_step=True,
)
```

### Controlling IV Instrument Lags

The `iv_max_lag` parameter controls exogenous variable instrument depth:

```python
# Default: iv_max_lag=0 (current value only, matches pydynpd)
model = DifferenceGMM(..., iv_max_lag=0)

# Stata xtabond2 style: iv_max_lag=6 (lags 0-6)
model = DifferenceGMM(..., iv_max_lag=6)
```

### Removing Time Dummies

Time dummies add parameters without adding instruments, potentially causing under-identification:

```python
# If instrument count is low relative to parameters:
model = DifferenceGMM(
    data=data, dep_var="y", lags=1,
    exog_vars=["x1"],
    time_dummies=False,  # Reduce parameter count
    collapse=True,
    two_step=True,
)
```

## Instrument Count Impact Example

```python
from panelbox.gmm import DifferenceGMM

# Compare specifications with varying instrument counts
configs = [
    {"collapse": False, "gmm_max_lag": None, "label": "Full (no collapse)"},
    {"collapse": True, "gmm_max_lag": None, "label": "Collapsed (all lags)"},
    {"collapse": True, "gmm_max_lag": 4, "label": "Collapsed (max_lag=4)"},
    {"collapse": True, "gmm_max_lag": 3, "label": "Collapsed (max_lag=3)"},
]

for cfg in configs:
    model = DifferenceGMM(
        data=data, dep_var="n", lags=1,
        id_var="id", time_var="year",
        exog_vars=["w", "k"],
        collapse=cfg["collapse"],
        gmm_max_lag=cfg.get("gmm_max_lag"),
        two_step=True, robust=True,
        time_dummies=False,
    )
    results = model.fit()
    print(
        f"{cfg['label']:30s} | "
        f"L={results.n_instruments:3d} | "
        f"ratio={results.instrument_ratio:.3f} | "
        f"AR coef={results.params['L1.n']:.4f} | "
        f"Hansen p={results.hansen_j.pvalue:.4f}"
    )
```

## GMMOverfitDiagnostic

PanelBox provides comprehensive overfitting diagnostics through the `GMMOverfitDiagnostic` class:

```python
from panelbox.gmm import DifferenceGMM, GMMOverfitDiagnostic

model = DifferenceGMM(
    data=data, dep_var="n", lags=1,
    id_var="id", time_var="year",
    exog_vars=["w", "k"], collapse=True,
    two_step=True, robust=True,
)
results = model.fit()

diag = GMMOverfitDiagnostic(model, results)
print(diag.summary())
```

The diagnostic report includes five checks:

| Check | What It Tests | Signal |
|-------|--------------|--------|
| **Feasibility** | Instrument ratio vs Roodman rule | GREEN/YELLOW/RED |
| **Sensitivity** | Coefficient stability across `gmm_max_lag` values | GREEN/YELLOW/RED |
| **Bounds** | GMM coefficient between OLS and FE | GREEN/YELLOW/RED |
| **Jackknife** | Leave-one-group-out stability | GREEN/YELLOW/RED |
| **Step comparison** | One-step vs two-step consistency | GREEN/YELLOW/RED |

### Individual Diagnostic Checks

```python
# 1. Assess feasibility (Roodman rule)
feas = diag.assess_feasibility()
print(f"Ratio: {feas['instrument_ratio']:.3f} [{feas['signal']}]")

# 2. Instrument sensitivity (varying max_lag)
sens = diag.instrument_sensitivity(max_lag_range=[2, 3, 4, 5])
print(sens)

# 3. Coefficient bounds test (Nickell)
bounds = diag.coefficient_bounds_test()
print(f"OLS: {bounds['ols_coef']:.4f}, GMM: {bounds['gmm_coef']:.4f}, FE: {bounds['fe_coef']:.4f}")

# 4. Step comparison (one-step vs two-step)
step = diag.step_comparison()
print(f"One-step: {step['one_step_coef']:.4f}, Two-step: {step['two_step_coef']:.4f}")
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Complete GMM Guide | End-to-end GMM workflow | [Complete Guide](complete-guide.md) |
| GMM Diagnostics | All diagnostic tests | [Diagnostics](diagnostics.md) |

## See Also

- [Difference GMM](difference-gmm.md) -- Where instruments are used
- [System GMM](system-gmm.md) -- Additional level instruments
- [Diagnostics](diagnostics.md) -- Testing instrument validity
- [Complete Guide](complete-guide.md) -- Applied workflow

## References

1. Roodman, D. (2009). "How to do xtabond2: An Introduction to Difference and System GMM in Stata." *The Stata Journal*, 9(1), 86-136.
2. Arellano, M., & Bond, S. (1991). "Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations." *Review of Economic Studies*, 58(2), 277-297.
3. Blundell, R., & Bond, S. (1998). "Initial Conditions and Moment Restrictions in Dynamic Panel Data Models." *Journal of Econometrics*, 87(1), 115-143.
4. Windmeijer, F. (2005). "A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators." *Journal of Econometrics*, 126(1), 25-51.
