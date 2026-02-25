---
title: "Canay Two-Step Estimator"
description: "Canay (2011) two-step quantile regression for panel data with fixed effects in PanelBox"
---

# Canay Two-Step Estimator

!!! info "Quick Reference"
    **Class:** `panelbox.models.quantile.canay.CanayTwoStep`
    **Import:** `from panelbox.models.quantile import CanayTwoStep`
    **Stata equivalent:** `xtqreg y x1 x2, fe method(canay)`
    **R equivalent:** `qrpanel::qregpd(y ~ x1 + x2, method = "cre")`

## Overview

The Canay (2011) Two-Step estimator provides a simple and computationally efficient approach to fixed effects quantile regression for panel data. It avoids the incidental parameters problem by estimating fixed effects in a first step and then removing them before running pooled quantile regression.

The two-step procedure is:

1. **Step 1** (Within-transformation OLS): Estimate fixed effects $\hat{\alpha}_i$ via standard FE-OLS regression
2. **Step 2** (Pooled QR on transformed data): Run pooled quantile regression on $\tilde{y}_{it} = y_{it} - \hat{\alpha}_i$

The estimator relies on a **key assumption**: fixed effects are **pure location shifters** — they shift the entire conditional distribution by the same amount across all quantiles. Formally:

$$Q_\tau(y_{it} | X_{it}, \alpha_i) = X_{it}'\beta_\tau + \alpha_i \quad \forall \tau$$

where $\alpha_i$ does not depend on $\tau$. This means individual heterogeneity affects only the level, not the shape, of the conditional distribution.

PanelBox provides a formal test of this assumption via the `test_location_shift()` method.

## Quick Example

```python
from panelbox.core.panel_data import PanelData
from panelbox.models.quantile import CanayTwoStep

panel_data = PanelData(data=df, entity_col="firm_id", time_col="year")

model = CanayTwoStep(
    data=panel_data,
    formula="investment ~ value + capital",
    tau=[0.25, 0.5, 0.75],
)
results = model.fit(se_adjustment="two-step")

# Test the location-shift assumption
test = model.test_location_shift(method="wald")
print(f"Location shift test: stat={test.statistic:.3f}, p={test.pvalue:.3f}")
```

## When to Use

- **Location-shift is plausible**: entity heterogeneity affects levels but not distributional shape
- **Computational speed**: much faster than the Koenker penalty method
- **Large panels**: works well with large $N$ and moderate $T$
- **Quick analysis**: useful for initial exploration before using more complex methods

!!! warning "Key Assumptions"
    - **Location shift**: fixed effects are pure location shifters (same $\alpha_i$ for all $\tau$)
    - **Large $T$**: the first-step FE-OLS estimator of $\hat{\alpha}_i$ must be consistent, requiring $T$ to grow
    - **Strict exogeneity**: $E[\varepsilon_{it}|X_i, \alpha_i] = 0$
    - **Testable**: use `test_location_shift()` to check the key assumption

!!! tip "When NOT to Use"
    If the treatment or covariates affect different parts of the distribution differently (e.g., a policy helps low-income workers more than high-income workers), the location-shift assumption is violated. Use the [Koenker penalty method](fixed-effects.md) or [Location-Scale model](location-scale.md) instead.

## Detailed Guide

### The Two-Step Procedure

**Step 1: FE-OLS to estimate $\hat{\alpha}_i$**

Standard within-transformation OLS regression:

$$y_{it} = X_{it}'\beta + \alpha_i + \varepsilon_{it}$$

The within estimator demeans by entity: $\ddot{y}_{it} = y_{it} - \bar{y}_i$, then recovers $\hat{\alpha}_i = \bar{y}_i - \bar{X}_i'\hat{\beta}$.

**Step 2: Pooled QR on $\tilde{y}_{it} = y_{it} - \hat{\alpha}_i$**

$$\min_{\beta_\tau} \sum_{i=1}^{N}\sum_{t=1}^{T} \rho_\tau(\tilde{y}_{it} - X_{it}'\beta_\tau)$$

This is simply a pooled quantile regression on the "de-fixed-effected" data.

### Data Preparation

```python
from panelbox.core.panel_data import PanelData
from panelbox.models.quantile import CanayTwoStep

panel_data = PanelData(data=df, entity_col="id", time_col="year")

model = CanayTwoStep(
    data=panel_data,
    formula="y ~ x1 + x2",
    tau=[0.1, 0.25, 0.5, 0.75, 0.9],
)
```

### Estimation

```python
results = model.fit(
    se_adjustment="two-step",  # account for first-step uncertainty
    verbose=False,
)
```

### Standard Error Adjustment

The two-step nature of the estimator affects inference. Three options are available:

| SE Adjustment | Description | Recommended |
|---------------|-------------|-------------|
| `"two-step"` | Accounts for estimation error in $\hat{\alpha}_i$ from step 1 | Yes (default) |
| `"naive"` | Ignores first-step uncertainty; treats $\hat{\alpha}_i$ as known | No (understates SEs) |
| `"bootstrap"` | Block bootstrap over both steps | For robustness checks |

```python
# Recommended: account for two-step uncertainty
results = model.fit(se_adjustment="two-step")

# Bootstrap for robustness
results_boot = model.fit(se_adjustment="bootstrap")
```

### Interpreting Results

```python
# Coefficients at each quantile
for tau in model.tau:
    r = results.results[tau]
    print(f"tau={tau:.2f}: beta = {r.params}, se = {r.std_errors}")

# First-step results
print(f"FE-OLS R²: {model.fe_ols_result_.rsquared:.4f}")
print(f"Fixed effects (first 5): {model.fixed_effects_[:5]}")

# Transformed dependent variable
print(f"y_tilde stats: mean={model.y_transformed_.mean():.4f}")
```

### Testing the Location-Shift Assumption

This is the most important diagnostic for the Canay estimator:

```python
# Wald test
test_wald = model.test_location_shift(method="wald")
print(f"Wald test: stat={test_wald.statistic:.3f}, p={test_wald.pvalue:.3f}")

# Kolmogorov-Smirnov test
test_ks = model.test_location_shift(method="ks")
print(f"KS test: stat={test_ks.statistic:.3f}, p={test_ks.pvalue:.3f}")
```

- **$H_0$**: $\beta(\tau)$ is constant across $\tau$ (location shift holds)
- **$H_1$**: $\beta(\tau)$ varies with $\tau$ (location shift violated)
- **Decision**: if $p < 0.05$, the location-shift assumption is rejected

!!! note "Interpreting the Test"
    Rejection means the Canay estimator may be inconsistent. Consider using the [Koenker penalty method](fixed-effects.md) or [Location-Scale model](location-scale.md) instead. Non-rejection does not prove the assumption holds — it may simply lack power.

### Comparison with Penalty Method

```python
# Direct comparison
comparison = model.compare_with_penalty_method(tau=0.5, lambda_fe="auto")

# The comparison dict contains:
# - Coefficient estimates from both methods
# - Computation times
# - Maximum absolute difference in coefficients
```

If results differ substantially, the location-shift assumption is likely violated.

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | PanelData | *required* | Panel data object |
| `formula` | str | `None` | Model formula `"y ~ x1 + x2"` |
| `tau` | float/list | `0.5` | Quantile level(s) in $(0, 1)$ |

### Fit Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `se_adjustment` | str | `"two-step"` | SE method: `"two-step"`, `"naive"`, `"bootstrap"` |
| `verbose` | bool | `False` | Print estimation progress |

### Result Attributes

| Attribute | Description |
|-----------|-------------|
| `results` | Dict mapping $\tau \to$ result objects |
| `fixed_effects_` | Estimated entity fixed effects from step 1 |
| `fe_ols_result_` | Full step-1 FE-OLS results |
| `y_transformed_` | Transformed dependent variable $\tilde{y}_{it}$ |

### Test Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `test_location_shift()` | `(tau_grid=None, method="wald")` | Test $H_0$: location shift holds |
| `compare_with_penalty_method()` | `(tau=0.5, lambda_fe="auto")` | Compare Canay vs Koenker |

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Canay Two-Step | Step-by-step estimation and assumption testing | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/03_fixed_effects_canay.ipynb) |
| FE QR Comparison | Canay vs Koenker penalty method | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/04_fixed_effects_penalty.ipynb) |

## See Also

- [Pooled Quantile Regression](pooled.md) — baseline without fixed effects (step 2 of Canay)
- [Fixed Effects Quantile Regression](fixed-effects.md) — Koenker penalty method (no location-shift assumption)
- [Location-Scale Model](location-scale.md) — alternative FE approach with non-crossing guarantee
- [Diagnostics](diagnostics.md) — additional diagnostic tests

## References

- Canay, I. A. (2011). A simple approach to quantile regression for panel data. *The Econometrics Journal*, 14(3), 368-386.
- Koenker, R. (2004). Quantile regression for longitudinal data. *Journal of Multivariate Analysis*, 91(1), 74-89.
- Abrevaya, J., & Dahl, C. M. (2008). The effects of birth inputs on birthweight: Evidence from quantile estimation on panel data. *Journal of Business & Economic Statistics*, 26(4), 379-397.
