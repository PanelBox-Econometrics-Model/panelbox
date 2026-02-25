---
title: "Quantile Regression Diagnostics"
description: "Diagnostic tests, bootstrap inference, and visualization for panel quantile regression in PanelBox"
---

# Quantile Regression Diagnostics

!!! info "Quick Reference"
    **Modules:** `panelbox.models.quantile.monotonicity`, `panelbox.models.quantile.canay`, `panelbox.models.quantile.location_scale`
    **Key tools:** `QuantileMonotonicity.detect_crossing()`, `CanayTwoStep.test_location_shift()`, `LocationScale.test_normality()`

## Overview

Quantile regression diagnostics go beyond standard regression diagnostics because each quantile level produces a separate model. Key concerns include: whether quantile curves cross (monotonicity), whether simplifying assumptions hold (location shift, distributional form), and whether inference is valid (bootstrap methods, standard error choices).

This page provides a comprehensive diagnostic workflow for panel quantile regression models.

## Diagnostic Checklist

Use this checklist after fitting any quantile regression model:

| Check | Tool | When |
|-------|------|------|
| Do quantile curves cross? | `QuantileMonotonicity.detect_crossing()` | Always (multiple quantiles) |
| Is location-shift assumption valid? | `CanayTwoStep.test_location_shift()` | When using Canay |
| Is the reference distribution correct? | `LocationScale.test_normality()` | When using Location-Scale |
| Are bootstrap CIs reasonable? | `fit(bootstrap=True)` | For robust inference |
| Does the quantile process reveal heterogeneity? | Visual inspection | Always |
| Are results stable across methods? | `compare_with_penalty_method()` | For sensitivity |

## Detailed Diagnostics

### 1. Quantile Process Analysis

The quantile process plots coefficients as a function of $\tau$, revealing how covariate effects change across the conditional distribution:

```python
import numpy as np
from panelbox.models.quantile import PooledQuantile

# Estimate across a fine grid of quantiles
tau_grid = np.arange(0.05, 1.0, 0.05)
model = PooledQuantile(endog=y, exog=X, entity_id=entity,
                       quantiles=tau_grid)
results = model.fit(se_type="cluster")

# Extract coefficient paths
coef_paths = {}
se_paths = {}
for tau in tau_grid:
    r = results.results[tau]
    coef_paths[tau] = r.params
    se_paths[tau] = r.std_errors

# Plot quantile process for each variable
import matplotlib.pyplot as plt

n_vars = X.shape[1]
fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4))
if n_vars == 1:
    axes = [axes]

for j, ax in enumerate(axes):
    coefs = [coef_paths[tau][j] for tau in tau_grid]
    ses = [se_paths[tau][j] for tau in tau_grid]

    ax.plot(tau_grid, coefs, "b-", linewidth=2)
    ax.fill_between(tau_grid,
                     [c - 1.96 * s for c, s in zip(coefs, ses)],
                     [c + 1.96 * s for c, s in zip(coefs, ses)],
                     alpha=0.2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Quantile (tau)")
    ax.set_ylabel(f"Coefficient {j}")
    ax.set_title(f"Variable {j}")

plt.tight_layout()
```

**Interpretation**:

- **Flat line**: homogeneous effect (OLS is sufficient)
- **Upward slope**: larger effect at higher quantiles
- **Downward slope**: larger effect at lower quantiles
- **Non-monotone**: complex heterogeneity

### 2. Crossing Detection

Always check for crossing when estimating multiple quantiles:

```python
from panelbox.models.quantile import QuantileMonotonicity

report = QuantileMonotonicity.detect_crossing(results.results, X)
report.summary()

# Detailed violation report as DataFrame
if report.has_crossing:
    crossing_df = report.to_dataframe()
    print(crossing_df)

    # Visualize violations
    report.plot_violations(X, results.results)
```

If crossing is detected, choose a correction method:

```python
if report.has_crossing:
    # Option 1: Rearrangement (fastest)
    fixed = QuantileMonotonicity.rearrangement(results.results, X)

    # Option 2: Switch to Location-Scale (prevents crossing)
    from panelbox.models.quantile import LocationScale
    ls_model = LocationScale(data=panel_data, formula="y ~ x1 + x2",
                              tau=tau_grid, distribution="normal")
    ls_results = ls_model.fit()
```

See [Non-Crossing Constraints](monotonicity.md) for a full comparison of methods.

### 3. Location-Shift Test (for Canay)

The Canay two-step estimator requires that fixed effects are pure location shifters. Test this:

```python
from panelbox.models.quantile import CanayTwoStep

canay = CanayTwoStep(data=panel_data, formula="y ~ x1 + x2",
                      tau=[0.25, 0.5, 0.75])
canay_results = canay.fit(se_adjustment="two-step")

# Wald test: H0: beta(tau) is constant across tau
test_wald = canay.test_location_shift(
    tau_grid=[0.1, 0.25, 0.5, 0.75, 0.9],
    method="wald",
)
print(f"Wald test: stat={test_wald.statistic:.3f}, p={test_wald.pvalue:.3f}")

# Kolmogorov-Smirnov test
test_ks = canay.test_location_shift(method="ks")
print(f"KS test: stat={test_ks.statistic:.3f}, p={test_ks.pvalue:.3f}")
```

| Result | Action |
|--------|--------|
| $p > 0.05$ | Location shift not rejected; Canay is appropriate |
| $p < 0.05$ | Location shift rejected; use [Koenker penalty](fixed-effects.md) or [Location-Scale](location-scale.md) |

### 4. Normality Test (for Location-Scale)

The Location-Scale model assumes a reference distribution. Test whether the choice is appropriate:

```python
from panelbox.models.quantile import LocationScale

ls_model = LocationScale(data=panel_data, formula="y ~ x1 + x2",
                          tau=[0.1, 0.5, 0.9], distribution="normal")
ls_results = ls_model.fit()

# Test normality of standardized residuals
normality = ls_results.test_normality()
print(f"Normality test: stat={normality.statistic:.3f}, p={normality.pvalue:.3f}")
```

If rejected, try alternative distributions:

```python
for dist in ["normal", "logistic", "t", "laplace"]:
    m = LocationScale(data=panel_data, formula="y ~ x1 + x2",
                       tau=0.5, distribution=dist)
    r = m.fit()
    test = r.test_normality()
    print(f"{dist:10s}: stat={test.statistic:.3f}, p={test.pvalue:.3f}")
```

### 5. Bootstrap Inference

Bootstrap is essential for valid inference in quantile regression, especially with:

- Cluster-dependent data
- Two-step estimators (Canay)
- Dynamic models with instruments

```python
from panelbox.models.quantile.base import QuantilePanelModel

# Base class supports bootstrap
model = PooledQuantile(endog=y, exog=X, entity_id=entity, quantiles=0.5)

# The base class fit() supports bootstrap
# Alternatively, use Canay with bootstrap SE adjustment:
canay = CanayTwoStep(data=panel_data, formula="y ~ x1 + x2", tau=0.5)
results_boot = canay.fit(se_adjustment="bootstrap")
```

Bootstrap types for panel data:

| Type | Description | Use When |
|------|-------------|----------|
| Cluster bootstrap | Resample entities with replacement | Standard for panel data |
| Pairs bootstrap | Resample (y, X) pairs | Cross-sectional data |
| Wild bootstrap | Perturb residuals with random signs | Heteroskedastic errors |
| Block bootstrap | Resample time blocks within entities | Time-dependent errors |

### 6. Method Comparison

Compare results across estimation methods to assess sensitivity:

```python
from panelbox.models.quantile import (
    PooledQuantile, FixedEffectsQuantile,
    CanayTwoStep, LocationScale,
)

tau = 0.5

# Pooled QR
pooled = PooledQuantile(endog=y, exog=X, entity_id=entity, quantiles=tau)
r_pooled = pooled.fit()

# Canay
canay = CanayTwoStep(data=panel_data, formula="y ~ x1 + x2", tau=tau)
r_canay = canay.fit()

# Koenker penalty
fe_qr = FixedEffectsQuantile(data=panel_data, formula="y ~ x1 + x2",
                               tau=tau, lambda_fe="auto")
r_fe = fe_qr.fit()

# Location-Scale
ls = LocationScale(data=panel_data, formula="y ~ x1 + x2",
                    tau=tau, distribution="normal")
r_ls = ls.fit()

# Compare
print("Method Comparison (tau=0.5):")
print(f"  Pooled QR:    {r_pooled.results[tau].params}")
print(f"  Canay:        {r_canay.results[tau].params}")
print(f"  Koenker FE:   {r_fe.results[tau].params}")
print(f"  Location-Scale: {r_ls.results[tau].params}")
```

**Interpretation**:

- Pooled vs FE methods: large difference suggests entity heterogeneity matters
- Canay vs Koenker: large difference suggests location shift is violated
- All methods agree: results are robust

### 7. Canay vs Koenker Comparison

A built-in tool for comparing the two FE approaches:

```python
canay = CanayTwoStep(data=panel_data, formula="y ~ x1 + x2",
                      tau=[0.25, 0.5, 0.75])
canay.fit()

comparison = canay.compare_with_penalty_method(tau=0.5, lambda_fe="auto")
# Returns dict with:
# - Coefficients from both methods
# - Computation times
# - Maximum absolute difference
```

### 8. Goodness of Fit

Quantile regression uses the check loss rather than $R^2$. A pseudo-$R^2$ can be computed:

```python
# Pseudo R-squared: 1 - (check loss / null check loss)
def pseudo_r2(results, y, tau):
    """Compute pseudo R-squared for quantile regression."""
    r = results.results[tau]
    fitted = X @ r.params
    residuals = y - fitted
    check_loss = np.sum(residuals * (tau - (residuals < 0).astype(float)))

    # Null model: intercept only (unconditional quantile)
    null_quantile = np.quantile(y, tau)
    null_resid = y - null_quantile
    null_loss = np.sum(null_resid * (tau - (null_resid < 0).astype(float)))

    return 1 - check_loss / null_loss

for tau in [0.25, 0.5, 0.75]:
    pr2 = pseudo_r2(results, y, tau)
    print(f"Pseudo R² (tau={tau}): {pr2:.4f}")
```

### 9. Model Selection

Compare models using information criteria adapted for quantile regression:

```python
# Compare specifications via check loss
specs = {
    "Simple": "y ~ x1",
    "Full": "y ~ x1 + x2",
    "Interaction": "y ~ x1 + x2 + x1:x2",
}

for name, formula in specs.items():
    m = CanayTwoStep(data=panel_data, formula=formula, tau=0.5)
    r = m.fit()
    # Lower check loss = better fit
    fitted = m.X @ r.results[0.5].params
    resid = m.y_transformed_ - fitted
    loss = np.sum(resid * (0.5 - (resid < 0).astype(float)))
    print(f"{name:15s}: check loss = {loss:.4f}")
```

## Complete Diagnostic Workflow

```python
import numpy as np
from panelbox.core.panel_data import PanelData
from panelbox.models.quantile import (
    PooledQuantile, CanayTwoStep, LocationScale, QuantileMonotonicity,
)

# Step 1: Estimate at multiple quantiles
tau_grid = [0.1, 0.25, 0.5, 0.75, 0.9]
model = PooledQuantile(endog=y, exog=X, entity_id=entity,
                       quantiles=tau_grid)
results = model.fit(se_type="cluster")

# Step 2: Check for crossing
report = QuantileMonotonicity.detect_crossing(results.results, X)
print(f"Crossing detected: {report.has_crossing}")
if report.has_crossing:
    print(f"  Inversions: {report.total_inversions}")
    print(f"  Affected: {report.pct_affected:.1f}%")

# Step 3: If using Canay, test location shift
panel_data = PanelData(data=df, entity_col="id", time_col="year")
canay = CanayTwoStep(data=panel_data, formula="y ~ x1 + x2",
                      tau=tau_grid)
canay.fit()
loc_test = canay.test_location_shift(method="wald")
print(f"Location shift test p-value: {loc_test.pvalue:.4f}")

# Step 4: If using Location-Scale, test distribution
ls = LocationScale(data=panel_data, formula="y ~ x1 + x2",
                    tau=tau_grid, distribution="normal")
ls_results = ls.fit()
norm_test = ls_results.test_normality()
print(f"Normality test p-value: {norm_test.pvalue:.4f}")

# Step 5: Compare methods for robustness
print("\nMethod comparison at median:")
print(f"  Pooled: {results.results[0.5].params}")
print(f"  Canay:  {canay.fit().results[0.5].params}")
print(f"  L-S:    {ls_results.results[0.5].params}")
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| QR Diagnostics | Complete diagnostic workflow | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/06_advanced_diagnostics.ipynb) |
| Method Comparison | Comparing all QR methods | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/07_bootstrap_inference.ipynb) |

## See Also

- [Pooled Quantile Regression](pooled.md) — baseline model
- [Fixed Effects Quantile Regression](fixed-effects.md) — Koenker penalty method
- [Canay Two-Step](canay.md) — location-shift test details
- [Location-Scale Model](location-scale.md) — normality test details
- [Non-Crossing Constraints](monotonicity.md) — crossing detection and correction

## References

- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Chernozhukov, V., Fernandez-Val, I., & Galichon, A. (2010). Quantile and probability curves without crossing. *Econometrica*, 78(3), 1093-1125.
- Canay, I. A. (2011). A simple approach to quantile regression for panel data. *The Econometrics Journal*, 14(3), 368-386.
- Machado, J. A., & Santos Silva, J. M. C. (2019). Quantiles via moments. *Journal of Econometrics*, 213(1), 145-173.
- Koenker, R., & Machado, J. A. (1999). Goodness of fit and related inference processes for quantile regression. *Journal of the American Statistical Association*, 94(448), 1296-1310.
