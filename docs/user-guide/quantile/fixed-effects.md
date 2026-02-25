---
title: "Fixed Effects Quantile Regression"
description: "Koenker (2004) penalized fixed effects quantile regression for panel data in PanelBox"
---

# Fixed Effects Quantile Regression

!!! info "Quick Reference"
    **Class:** `panelbox.models.quantile.fixed_effects.FixedEffectsQuantile`
    **Import:** `from panelbox.models.quantile import FixedEffectsQuantile`
    **Stata equivalent:** `xtqreg y x1 x2, fe`
    **R equivalent:** `qrpanel::qregpd(y ~ x1 + x2, panel = "fd")`

## Overview

Fixed Effects Quantile Regression addresses a fundamental challenge: estimating quantile functions while controlling for unobserved entity-level heterogeneity. In standard panel models, fixed effects can be consistently estimated as $N \to \infty$ with fixed $T$. However, in quantile regression, the **incidental parameters problem** prevents consistent estimation of slope coefficients when fixed effects are included as separate parameters with short panels.

Koenker (2004) proposed an elegant solution: add an $\ell_1$ (LASSO-type) penalty on the fixed effects to shrink them toward zero. The penalized objective function is:

$$\min_{\beta, \alpha_1, \ldots, \alpha_N} \sum_{\tau \in \mathcal{T}} \sum_{i=1}^{N} \sum_{t=1}^{T} \rho_\tau(y_{it} - X_{it}'\beta_\tau - \alpha_i) + \lambda \sum_{i=1}^{N} |\alpha_i|$$

The penalty parameter $\lambda$ controls the shrinkage of fixed effects. PanelBox selects $\lambda$ automatically via cross-validation (`lambda_fe="auto"`) or accepts a user-specified value.

This approach produces consistent estimates of the slope coefficients $\beta_\tau$ while preventing the fixed effects from dominating the estimation in short panels.

## Quick Example

```python
from panelbox.core.panel_data import PanelData
from panelbox.models.quantile import FixedEffectsQuantile

# Create PanelData object
panel_data = PanelData(data=df, entity_col="firm_id", time_col="year")

# Estimate at multiple quantiles with automatic lambda selection
model = FixedEffectsQuantile(
    data=panel_data,
    formula="investment ~ value + capital",
    tau=[0.25, 0.5, 0.75],
    lambda_fe="auto",
)
results = model.fit(method="L-BFGS-B", cv_folds=5)
```

## When to Use

- **Entity heterogeneity**: unobserved entity characteristics correlated with regressors
- **Short panels**: $T$ is small relative to $N$ (the penalty handles the incidental parameters problem)
- **Slope coefficient focus**: primary interest is in $\beta_\tau$, not the fixed effects themselves
- **Distributional analysis with controls**: study how covariate effects vary across quantiles while controlling for entities
- **Comparison with Canay**: when the location-shift assumption is suspect

!!! warning "Key Assumptions"
    - **Linear conditional quantile**: $Q_\tau(y_{it}|X_{it}, \alpha_i) = X_{it}'\beta_\tau + \alpha_i$
    - **Strict exogeneity**: $E[\rho_\tau'(y_{it} - X_{it}'\beta_\tau - \alpha_i) | X_i] = 0$
    - **Additive fixed effects**: entity effects enter additively
    - **Appropriate penalty**: $\lambda$ must balance bias and variance

## Detailed Guide

### The Incidental Parameters Problem

In quantile regression, the objective function is not differentiable, and the fixed effects $\alpha_i$ are estimated with error of order $O(1/T)$. Unlike OLS where the within-transformation eliminates fixed effects, no such transformation exists for quantile regression. The bias in $\hat{\alpha}_i$ contaminates the slope estimates $\hat{\beta}_\tau$.

The Koenker penalty addresses this by shrinking fixed effects:

- Small $\lambda$: estimates close to unpenalized QR with individual dummies (biased slopes)
- Large $\lambda$: all fixed effects shrunk to zero (equivalent to pooled QR)
- Optimal $\lambda$: balance between controlling entity heterogeneity and minimizing bias

### Data Preparation

FixedEffectsQuantile requires a `PanelData` object:

```python
from panelbox.core.panel_data import PanelData

# From DataFrame
panel_data = PanelData(data=df, entity_col="id", time_col="year")

# The formula specifies the model
model = FixedEffectsQuantile(
    data=panel_data,
    formula="y ~ x1 + x2",
    tau=[0.1, 0.25, 0.5, 0.75, 0.9],
    lambda_fe="auto",        # automatic cross-validation
)
```

### Lambda Selection

The penalty parameter $\lambda$ is critical. PanelBox provides two approaches:

=== "Automatic (Cross-Validation)"

    ```python
    # Entity-based K-fold CV preserves panel structure
    model = FixedEffectsQuantile(
        data=panel_data,
        formula="y ~ x1 + x2",
        tau=0.5,
        lambda_fe="auto",
    )
    results = model.fit(cv_folds=5, verbose=True)

    # Check selected lambda
    print(f"Optimal lambda: {results.results[0.5].lambda_fe:.4f}")
    ```

=== "Manual"

    ```python
    # Specify lambda directly (useful for sensitivity analysis)
    model = FixedEffectsQuantile(
        data=panel_data,
        formula="y ~ x1 + x2",
        tau=0.5,
        lambda_fe=1.5,
    )
    results = model.fit()
    ```

The cross-validation procedure (`_select_lambda_cv`) works as follows:

1. Compute $\lambda_\text{max}$: the smallest $\lambda$ that sets all fixed effects to zero
2. Create a log-spaced grid from $0.001 \cdot \lambda_\text{max}$ to $\lambda_\text{max}$ (20 values)
3. Perform entity-based K-fold CV, splitting by entities to preserve panel structure
4. Select $\lambda$ minimizing average out-of-fold check loss

### Estimation

```python
results = model.fit(
    method="L-BFGS-B",   # optimization method (quasi-Newton)
    cv_folds=5,            # folds for lambda CV
    verbose=False,         # print progress
)
```

### Interpreting Results

```python
# Slope coefficients for each quantile
for tau in model.tau:
    r = results.results[tau]
    print(f"tau={tau:.2f}:")
    print(f"  Coefficients: {r.params}")
    print(f"  Std Errors:   {r.std_errors}")
    print(f"  Lambda used:  {r.lambda_fe}")

# Estimated fixed effects
fe = results.results[0.5].fixed_effects  # entity-level effects

# Many will be shrunk to zero (sparse solution)
n_nonzero = np.sum(np.abs(fe) > 1e-6)
print(f"Non-zero fixed effects: {n_nonzero}/{len(fe)}")
```

### Shrinkage Path

Examine how coefficients change across $\lambda$ values:

```python
# Estimate at multiple lambda values
import numpy as np

lambdas = np.logspace(-2, 2, 20)
coefs_path = []

for lam in lambdas:
    model_temp = FixedEffectsQuantile(
        data=panel_data, formula="y ~ x1 + x2",
        tau=0.5, lambda_fe=lam
    )
    res = model_temp.fit()
    coefs_path.append(res.results[0.5].params)

# Plot shrinkage path
# Stable coefficients across lambda indicate robustness
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | PanelData | *required* | Panel data object |
| `formula` | str | `None` | Model formula `"y ~ x1 + x2"` |
| `tau` | float/list | `0.5` | Quantile level(s) in $(0, 1)$ |
| `lambda_fe` | float/str | `"auto"` | Penalty: `"auto"` for CV or a positive float |

### Fit Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `"L-BFGS-B"` | Optimization method |
| `cv_folds` | int | `5` | Number of CV folds for $\lambda$ selection |
| `verbose` | bool | `False` | Print estimation progress |

### Result Attributes

| Attribute | Description |
|-----------|-------------|
| `params` | Estimated slope coefficients $\hat{\beta}_\tau$ |
| `std_errors` | Standard errors |
| `fixed_effects` | Estimated entity fixed effects $\hat{\alpha}_i$ |
| `lambda_fe` | Penalty parameter used |
| `converged` | Optimization convergence flag |

## Diagnostics

### Sensitivity to Lambda

```python
# Compare estimates across lambda values to assess robustness
for lam in [0.1, 1.0, 10.0]:
    m = FixedEffectsQuantile(data=panel_data, formula="y ~ x1 + x2",
                              tau=0.5, lambda_fe=lam)
    r = m.fit()
    print(f"lambda={lam:5.1f}: beta = {r.results[0.5].params}")
```

### Comparison with Canay Two-Step

```python
from panelbox.models.quantile import CanayTwoStep

canay = CanayTwoStep(data=panel_data, formula="y ~ x1 + x2",
                      tau=[0.25, 0.5, 0.75])
canay_results = canay.fit()

# Large differences suggest location-shift assumption is violated
comparison = canay.compare_with_penalty_method(tau=0.5)
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| FE Quantile Regression | Penalty method with lambda selection | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/03_fixed_effects_canay.ipynb) |
| Lambda Selection | Cross-validation and shrinkage analysis | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/04_fixed_effects_penalty.ipynb) |

## See Also

- [Pooled Quantile Regression](pooled.md) — baseline without fixed effects
- [Canay Two-Step](canay.md) — faster alternative under location-shift assumption
- [Location-Scale Model](location-scale.md) — non-crossing quantiles with FE support
- [Non-Crossing Constraints](monotonicity.md) — detect and fix quantile crossing
- [Diagnostics](diagnostics.md) — diagnostic tests for quantile models

## References

- Koenker, R. (2004). Quantile regression for longitudinal data. *Journal of Multivariate Analysis*, 91(1), 74-89.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Lamarche, C. (2010). Robust penalized quantile regression estimation for panel data. *Journal of Econometrics*, 157(2), 396-408.
- Galvao, A. F., & Montes-Rojas, G. V. (2010). Penalized quantile regression for dynamic panel data. *Journal of Statistical Planning and Inference*, 140(11), 3476-3497.
