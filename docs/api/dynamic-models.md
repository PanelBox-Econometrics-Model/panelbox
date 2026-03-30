---
title: "Dynamic Models API"
description: "API reference for panelbox.models.dynamic — LSDVC, LSDVCResults, AndersonHsiao"
---

# Dynamic Models API Reference

!!! info "Module"
    **Import**: `from panelbox.models.dynamic import LSDVC, LSDVCResults, AndersonHsiao`
    **Source**: `panelbox/models/dynamic/`

## Overview

Dynamic panel models include lagged dependent variables as regressors. The LSDVC estimator corrects the Nickell (1981) bias in Fixed Effects estimation using analytical bias formulas from Kiviet (1995), while Anderson-Hsiao provides a consistent IV estimator often used as an initial step.

| Estimator | Class | Reference | Use Case |
|-----------|-------|-----------|----------|
| LSDVC | `LSDVC` | Kiviet (1995), Bruno (2005) | Bias-corrected FE for dynamic panels |
| Anderson-Hsiao | `AndersonHsiao` | Anderson & Hsiao (1982) | Consistent IV estimator; initial estimator for LSDVC |

!!! tip "LSDVC vs GMM"
    LSDVC is preferred over GMM when **N is small to moderate** (N < 100) and T is not too large. GMM requires large N for its asymptotic properties, while LSDVC works well even with small N. See the [decision guide](../user-guide/dynamic-panels.md) for details.

---

## Classes

### LSDVC

Least Squares Dummy Variable Corrected estimator. Estimates a dynamic panel model with fixed effects and corrects for Nickell bias using analytical bias formulas.

The model is:

$$
y_{it} = \rho \, y_{i,t-1} + X_{it}'\beta + \alpha_i + \varepsilon_{it}
$$

```python
LSDVC(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    lags: int = 1,
    initial_estimator: str = "ab",
    bias_order: int = 2,
    max_iter: int = 50,
    tol: float = 1e-6,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | `str` | *required* | R-style formula, e.g. `"y ~ x1 + x2"`. The lagged dependent variable is added automatically. |
| `data` | `pd.DataFrame` | *required* | Panel data in long format |
| `entity_col` | `str` | *required* | Column identifying entities (individuals/firms) |
| `time_col` | `str` | *required* | Column identifying time periods |
| `lags` | `int` | `1` | Number of AR lags (currently only 1 supported) |
| `initial_estimator` | `str` | `"ab"` | Initial consistent estimator: `"ah"` (Anderson-Hsiao), `"ab"` (Arellano-Bond), `"bb"` (Blundell-Bond) |
| `bias_order` | `int` | `2` | Order of Kiviet bias approximation (1, 2, or 3) |
| `max_iter` | `int` | `50` | Maximum iterations for iterated bias correction |
| `tol` | `float` | `1e-6` | Convergence tolerance for iterated bias correction |

#### Initial Estimator Options

| Value | Method | When to Use |
|-------|--------|-------------|
| `"ah"` | Anderson-Hsiao IV | Fast; good default for small panels |
| `"ab"` | Arellano-Bond (one-step) | More efficient; requires N > T |
| `"bb"` | Blundell-Bond (one-step) | Best for persistent processes (high $\rho$) |

!!! warning "Bias Order"
    Higher orders provide more accurate bias correction but may increase variance. Order 2 (default) is recommended for most applications. Order 3 adds the Bun & Kiviet (2003) refinement but gains are typically small.

#### `.fit()` Method

```python
model.fit(
    n_bootstrap: int = 500,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> LSDVCResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bootstrap` | `int` | `500` | Number of bootstrap replications for inference. Set to 0 to skip bootstrap. |
| `ci_level` | `float` | `0.95` | Confidence level for bootstrap percentile intervals |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |

**Returns**: [`LSDVCResults`](#lsdvcresults)

**Pipeline**:

1. Build lag matrices from panel data
2. LSDV (Fixed Effects with lagged dependent variable) estimation
3. Initial consistent estimate of $\rho$ via the chosen initial estimator
4. Iterative Kiviet bias correction
5. Parametric bootstrap for standard errors and confidence intervals

#### Example

```python
from panelbox.models.dynamic import LSDVC
from panelbox import load_grunfeld

data = load_grunfeld()

# Fit LSDVC with Arellano-Bond initial estimator, order-2 bias correction
model = LSDVC(
    "invest ~ value + capital", data, "firm", "year",
    initial_estimator="ab", bias_order=2,
)
results = model.fit(n_bootstrap=500, seed=42)
print(results.summary())
```

```python
# Compare initial estimators
for init in ["ah", "ab", "bb"]:
    model = LSDVC("invest ~ value + capital", data, "firm", "year",
                   initial_estimator=init)
    res = model.fit(n_bootstrap=200, seed=42)
    print(f"{init.upper()}: rho = {res.params.iloc[0]:.4f}")
```

---

### LSDVCResults

Results container for LSDVC estimation. Inherits from [`PanelResults`](core.md#panelresults) and adds LSDVC-specific attributes for bias correction diagnostics and bootstrap inference.

#### Attributes

All standard `PanelResults` attributes (`params`, `std_errors`, `cov_params`, `resid`, `fittedvalues`, `nobs`, `rsquared`, etc.) plus:

| Attribute | Type | Description |
|-----------|------|-------------|
| `bias_correction` | `pd.Series` | Bias correction vector applied to each parameter |
| `lsdv_params` | `pd.Series` | Uncorrected LSDV parameters (before bias correction) |
| `initial_estimator` | `str` | Method used for initial consistent estimate (`'ah'`, `'ab'`, `'bb'`) |
| `initial_rho` | `float` | Initial consistent estimate of $\rho$ |
| `n_iterations` | `int` | Number of iterations until convergence |
| `converged` | `bool` | Whether the iterated bias correction converged |
| `bias_order` | `int` | Order of the Kiviet bias approximation used |
| `sigma2` | `float` | Error variance estimate from LSDV |
| `bootstrap_params` | `np.ndarray \| None` | Bootstrap parameter distribution (B × K matrix), or None |
| `bootstrap_ci` | `tuple \| None` | `(ci_lower, ci_upper)` bootstrap percentile confidence intervals |

#### Methods

##### `.summary(title=None)`

Generate formatted summary of LSDVC estimation results. Includes:

- Model information (formula, initial estimator, bias order, convergence)
- Sample size details
- Coefficient table with bootstrap standard errors, z-statistics, p-values, and confidence intervals
- Bias correction comparison table (LSDV vs LSDVC estimates)

```python
results = model.fit(n_bootstrap=500)
print(results.summary())
```

##### `.bias_summary()`

Focused table showing the magnitude of bias correction per parameter, including the initial estimator details and convergence information.

```python
print(results.bias_summary())
```

##### `.plot_bootstrap_distribution(variable=None)`

Plot histogram of the bootstrap parameter distribution for a given variable.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variable` | `str \| None` | `None` | Variable name to plot. If None, plots the first parameter (lagged dependent variable). |

**Returns**: `matplotlib.figure.Figure`

```python
fig = results.plot_bootstrap_distribution("L.invest")
fig.savefig("bootstrap_rho.png")
```

!!! tip "Inspecting bootstrap results"
    Access the raw bootstrap distribution via `results.bootstrap_params` (a B × K numpy array) for custom analysis beyond the built-in plots.

---

### AndersonHsiao

Anderson-Hsiao (1982) IV estimator for dynamic panel models. Provides consistent (but inefficient) estimation using instrumental variables on the first-differenced equation.

The model estimated is:

$$
\Delta y_{it} = \rho \, \Delta y_{i,t-1} + \Delta X_{it}'\beta + \Delta \varepsilon_{it}
$$

using $y_{i,t-2}$ (levels) or $\Delta y_{i,t-2}$ (differences) as an instrument for the endogenous $\Delta y_{i,t-1}$.

```python
AndersonHsiao(
    data: pd.DataFrame,
    dep_var: str,
    entity_col: str,
    time_col: str,
    lags: int = 1,
    exog_vars: list[str] | None = None,
    instrument: str = "levels",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | *required* | Panel data in long format |
| `dep_var` | `str` | *required* | Name of the dependent variable |
| `entity_col` | `str` | *required* | Column identifying entities |
| `time_col` | `str` | *required* | Column identifying time periods |
| `lags` | `int` | `1` | Number of AR lags (currently only 1 supported) |
| `exog_vars` | `list[str] \| None` | `None` | Names of exogenous regressors |
| `instrument` | `str` | `"levels"` | Instrument type: `"levels"` ($y_{t-2}$, more efficient) or `"differences"` ($\Delta y_{t-2}$) |

#### `.fit()` Method

```python
ah.fit() -> dict
```

**Returns** a dictionary with:

| Key | Type | Description |
|-----|------|-------------|
| `"rho"` | `float` | AR(1) coefficient estimate |
| `"beta"` | `np.ndarray \| None` | Exogenous variable coefficients |
| `"params"` | `np.ndarray` | All parameters combined |
| `"resid"` | `np.ndarray` | Residuals |
| `"param_names"` | `list[str]` | Parameter labels |
| `"nobs"` | `int` | Number of observations used |

#### Example

```python
from panelbox.models.dynamic import AndersonHsiao
from panelbox import load_grunfeld

data = load_grunfeld()
ah = AndersonHsiao(data, "invest", "firm", "year",
                   exog_vars=["value", "capital"])
result = ah.fit()
print(f"rho = {result['rho']:.4f}")
print(f"N = {result['nobs']}")
```

!!! note "Primary use case"
    Anderson-Hsiao is primarily used as an initial consistent estimator for LSDVC (`initial_estimator="ah"`). For standalone dynamic panel estimation, consider LSDVC or [GMM estimators](gmm.md).

---

## Comparison Example

```python
from panelbox.models.dynamic import LSDVC
from panelbox import FixedEffects, load_grunfeld

data = load_grunfeld()

# Standard FE (biased with lagged dependent variable)
fe = FixedEffects("invest ~ value + capital", data, "firm", "year")
fe_result = fe.fit()

# LSDVC (bias-corrected) with different initial estimators
for init in ["ah", "ab", "bb"]:
    model = LSDVC("invest ~ value + capital", data, "firm", "year",
                   initial_estimator=init)
    result = model.fit(n_bootstrap=200, seed=42)
    rho = result.params.iloc[0]
    rho_lsdv = result.lsdv_params.iloc[0]
    print(f"LSDVC ({init.upper()}): rho={rho:.4f}  (LSDV={rho_lsdv:.4f}, "
          f"bias correction={result.bias_correction.iloc[0]:.4f})")
```

## See Also

- [Core API](core.md) — `PanelResults` attributes and methods
- [Static Models](static-models.md) — Standard panel estimators (PooledOLS, FE, RE)
- [GMM API](gmm.md) — Arellano-Bond, Blundell-Bond GMM estimators
- [Nickell Bias & LSDVC Theory](../theory/nickell-bias-lsdvc.md) — Mathematical foundations
- [Dynamic Panels User Guide](../user-guide/dynamic-panels.md) — Decision guide and practical advice
