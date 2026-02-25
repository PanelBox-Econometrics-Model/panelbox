---
title: "Censored & Selection API"
description: "API reference for panelbox.models.censored and panelbox.models.selection — Tobit, Honore, PanelHeckman"
---

# Censored & Selection API Reference

!!! info "Module"
    **Import**: `from panelbox.models.censored import PooledTobit, RandomEffectsTobit, HonoreTrimmedEstimator`
    `from panelbox.models.selection import PanelHeckman, compute_imr`
    **Source**: `panelbox/models/censored/`, `panelbox/models/selection/`

## Overview

These modules handle two related problems in panel data:

- **Censored data**: The dependent variable is observed only within a range (e.g., wages censored at zero)
- **Sample selection**: Observations are missing non-randomly (e.g., wages observed only for workers)

| Model | Problem | Reference |
|-------|---------|-----------|
| `PooledTobit` | Left/right/two-sided censoring | Tobin (1958) |
| `RandomEffectsTobit` | Censoring with random effects | — |
| `HonoreTrimmedEstimator` | Semiparametric FE Tobit | Honore (1992) |
| `PanelHeckman` | Sample selection bias correction | Heckman (1979), Wooldridge (1995) |

---

## Censored Models

### PooledTobit

Tobit model for censored dependent variables. Handles left censoring (at zero), right censoring, or both.

#### Constructor

```python
PooledTobit(
    endog: np.ndarray,
    exog: np.ndarray,
    groups: np.ndarray,
    time: np.ndarray | None = None,
    censoring_point: float = 0.0,
    censoring_type: Literal["left", "right", "both"] = "left",
    lower_limit: float | None = None,
    upper_limit: float | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `np.ndarray` | *required* | Censored dependent variable |
| `exog` | `np.ndarray` | *required* | Independent variables |
| `groups` | `np.ndarray` | *required* | Entity identifiers |
| `time` | `np.ndarray \| None` | `None` | Time identifiers |
| `censoring_point` | `float` | `0.0` | Censoring threshold |
| `censoring_type` | `str` | `"left"` | Type: `"left"`, `"right"`, or `"both"` |
| `lower_limit` | `float \| None` | `None` | Lower censoring limit (for `"both"`) |
| `upper_limit` | `float \| None` | `None` | Upper censoring limit (for `"both"`) |

#### Example

```python
from panelbox.models.censored import PooledTobit

tobit = PooledTobit(
    endog=df["hours_worked"].values,
    exog=df[["wage", "education", "age"]].values,
    groups=df["person_id"].values,
    censoring_point=0.0,
    censoring_type="left",
)
result = tobit.fit()
result.summary()
```

---

### RandomEffectsTobit

Random effects Tobit model using Gauss-Hermite quadrature.

#### Constructor

```python
RandomEffectsTobit(
    endog: np.ndarray,
    exog: np.ndarray,
    groups: np.ndarray,
    time: np.ndarray | None = None,
    censoring_point: float = 0.0,
    censoring_type: Literal["left", "right", "both"] = "left",
    lower_limit: float | None = None,
    upper_limit: float | None = None,
    quadrature_points: int = 12,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quadrature_points` | `int` | `12` | Number of quadrature points for integration |

All other parameters are the same as `PooledTobit`.

---

### HonoreTrimmedEstimator

Semiparametric fixed effects estimator for censored panel data (Honore 1992). Does not require distributional assumptions on the error term or the fixed effects.

#### Constructor

```python
HonoreTrimmedEstimator(
    endog: np.ndarray,
    exog: np.ndarray,
    groups: np.ndarray,
    time: np.ndarray,
    censoring_point: float = 0.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `np.ndarray` | *required* | Censored dependent variable |
| `exog` | `np.ndarray` | *required* | Independent variables |
| `groups` | `np.ndarray` | *required* | Entity identifiers |
| `time` | `np.ndarray` | *required* | Time identifiers |
| `censoring_point` | `float` | `0.0` | Censoring threshold |

!!! tip "When to use Honore"
    Use when you need FE with censoring but want to avoid the incidental parameters problem. Requires at least T=2 periods per entity and uses **pairwise comparisons** across time periods.

---

## Selection Models

### PanelHeckman

Panel Heckman selection model correcting for non-random sample selection bias. Supports two-step (Wooldridge 1995) and maximum likelihood estimation.

#### Constructor

```python
PanelHeckman(
    endog: np.ndarray,
    exog: np.ndarray,
    selection: np.ndarray,
    exog_selection: np.ndarray,
    entity: np.ndarray | None = None,
    time: np.ndarray | None = None,
    method: Literal["two_step", "mle"] = "two_step",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `np.ndarray` | *required* | Outcome variable (observed only when selected) |
| `exog` | `np.ndarray` | *required* | Outcome equation regressors |
| `selection` | `np.ndarray` | *required* | Selection indicator (1=selected, 0=not) |
| `exog_selection` | `np.ndarray` | *required* | Selection equation regressors (should include exclusion restriction) |
| `entity` | `np.ndarray \| None` | `None` | Entity identifiers |
| `time` | `np.ndarray \| None` | `None` | Time identifiers |
| `method` | `str` | `"two_step"` | `"two_step"` (Heckman 1979) or `"mle"` (full information) |

!!! warning "Exclusion restriction"
    The selection equation (`exog_selection`) should include at least one variable **not** in the outcome equation (`exog`). Without an exclusion restriction, identification relies solely on functional form.

#### Example

```python
from panelbox.models.selection import PanelHeckman

heckman = PanelHeckman(
    endog=df["wage"].values,
    exog=df[["education", "experience"]].values,
    selection=df["employed"].values,
    exog_selection=df[["education", "experience", "children", "spouse_income"]].values,
    entity=df["person_id"].values,
    time=df["year"].values,
    method="two_step",
)
result = heckman.fit()
result.summary()
```

---

### PanelHeckmanResult

Result container for Heckman selection models.

#### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `outcome_params` | `pd.Series` | Outcome equation coefficients |
| `probit_params` | `pd.Series` | Selection equation (probit) coefficients |
| `sigma` | `float` | Standard deviation of outcome error |
| `rho` | `float` | Correlation between errors (selection and outcome) |
| `lambda_imr` | `float` | Coefficient on Inverse Mills Ratio |

#### Methods

- `.summary()` — Full results with both equations
- `.predict(type="unconditional")` — Predictions (`"unconditional"` or `"conditional"`)
- `.selection_effect()` — Test for selection bias (H0: rho = 0)
- `.imr_diagnostics()` — Inverse Mills Ratio diagnostics
- `.compare_ols_heckman()` — Compare with naive OLS (quantify selection bias)
- `.plot_imr()` — Diagnostic plots for IMR

---

## Utility Functions

### compute_imr

Compute the Inverse Mills Ratio (IMR).

```python
from panelbox.models.selection import compute_imr

imr = compute_imr(x, params)
```

### imr_derivative

Compute the derivative of the IMR.

```python
from panelbox.models.selection import imr_derivative

d_imr = imr_derivative(x, params)
```

### imr_diagnostics

Run diagnostic tests on the IMR.

```python
from panelbox.models.selection import imr_diagnostics

diag = imr_diagnostics(data, selection_results)
```

### test_selection_effect

Test whether selection bias is statistically significant (H0: rho = 0).

```python
from panelbox.models.selection import test_selection_effect

test = test_selection_effect(results)
print(f"rho = {test.rho:.4f}, p-value = {test.pvalue:.4f}")
```

---

## Complete Heckman Workflow

```python
from panelbox.models.selection import PanelHeckman, test_selection_effect

# Step 1: Estimate Heckman model
heckman = PanelHeckman(
    endog=df["wage"].values,
    exog=df[["education", "experience"]].values,
    selection=df["employed"].values,
    exog_selection=df[["education", "experience", "children"]].values,
    entity=df["person_id"].values,
    method="two_step",
)
result = heckman.fit()

# Step 2: Check for selection bias
test = test_selection_effect(result)
if test.pvalue < 0.05:
    print("Significant selection bias detected!")
    print(f"rho = {result.rho:.4f}")

# Step 3: Compare with naive OLS
bias = result.compare_ols_heckman()

# Step 4: View full results
result.summary()
```

## See Also

- [Discrete Choice API](discrete.md) — Binary models for the selection equation
- [Marginal Effects API](marginal-effects.md) — Marginal effects with selection correction
- [Tutorials: Censored](../tutorials/censored.md) — Step-by-step Tobit and Heckman guide
