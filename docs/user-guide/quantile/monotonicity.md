---
title: "Non-Crossing Quantile Constraints"
description: "Detect and fix crossing quantile curves using rearrangement, isotonic regression, and constrained optimization in PanelBox"
---

# Non-Crossing Quantile Constraints

!!! info "Quick Reference"
    **Class:** `panelbox.models.quantile.monotonicity.QuantileMonotonicity`
    **Import:** `from panelbox.models.quantile import QuantileMonotonicity`
    **Related classes:** `CrossingReport`, `MonotonicityComparison`

## Overview

When quantile regression models are estimated independently at multiple quantile levels, the resulting quantile curves can **cross** — meaning that the predicted 25th percentile exceeds the predicted 75th percentile for some observations. This is a logical contradiction: by definition, $Q_\tau(y|X)$ must be monotonically non-decreasing in $\tau$.

The crossing problem occurs because each quantile regression is estimated separately without imposing the ordering constraint $Q_{\tau_1}(y|X) \leq Q_{\tau_2}(y|X)$ for $\tau_1 < \tau_2$. Crossing is more common when:

- The sample size is small
- Many quantile levels are estimated
- Covariates have complex effects
- The data exhibits heteroskedasticity

PanelBox provides tools to **detect** crossing, **fix** it via several methods, and **prevent** it by using the [Location-Scale model](location-scale.md) which guarantees non-crossing by construction.

## Quick Example

```python
from panelbox.models.quantile import PooledQuantile, QuantileMonotonicity
import numpy as np

# Estimate at multiple quantiles
model = PooledQuantile(endog=y, exog=X, entity_id=entity,
                       quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
results = model.fit()

# Detect crossing
report = QuantileMonotonicity.detect_crossing(results.results, X)
report.summary()

# Fix via rearrangement
if report.has_crossing:
    fixed = QuantileMonotonicity.rearrangement(results.results, X)
```

## When to Use

- **After estimating multiple quantiles**: always check for crossing when fitting quantile regression at more than one level
- **Before reporting results**: crossing invalidates the interpretation as a conditional distribution
- **Risk management**: crossed quantiles produce nonsensical risk measures (e.g., VaR)
- **Density estimation**: quantile-based density estimates require monotonicity

!!! warning "When Crossing is a Problem"
    - Predicted distributions can have negative density regions
    - Prediction intervals can be inverted (lower bound > upper bound)
    - Interpolated quantile functions are not well-defined
    - Results cannot be interpreted as a valid conditional distribution

## Detailed Guide

### Detection

Use `QuantileMonotonicity.detect_crossing()` to check for violations:

```python
from panelbox.models.quantile import QuantileMonotonicity

# results.results is a dict {tau: result_object}
report = QuantileMonotonicity.detect_crossing(results.results, X)

# Report attributes
print(f"Has crossing: {report.has_crossing}")
print(f"Total inversions: {report.total_inversions}")
print(f"Percent affected: {report.pct_affected:.2f}%")
```

The `CrossingReport` object provides:

| Attribute | Description |
|-----------|-------------|
| `has_crossing` | `True` if any crossing detected |
| `total_inversions` | Total number of inversions across all quantile pairs |
| `pct_affected` | Percentage of observations with at least one inversion |
| `crossings` | List of dicts with details for each quantile pair |

Each entry in `crossings` contains:

```python
for c in report.crossings:
    tau1, tau2 = c["tau_pair"]
    print(f"tau={tau1:.2f} vs tau={tau2:.2f}:")
    print(f"  Inversions: {c['n_inversions']} ({c['pct_inversions']:.1f}%)")
    print(f"  Max violation: {c['max_violation']:.4f}")
    print(f"  Mean violation: {c['mean_violation']:.4f}")
```

Convert to DataFrame for further analysis:

```python
crossing_df = report.to_dataframe()
print(crossing_df)
```

### Fix Method 1: Rearrangement (Chernozhukov et al. 2010)

The simplest and most widely used post-hoc correction. For each observation, sort the predicted quantiles to restore monotonicity:

```python
fixed_results = QuantileMonotonicity.rearrangement(results.results, X)

# Verify crossing is fixed
report_fixed = QuantileMonotonicity.detect_crossing(fixed_results, X)
print(f"Crossing after rearrangement: {report_fixed.has_crossing}")
```

**How it works**:

1. Compute predictions $\hat{Q}_{\tau_j}(y|X_i) = X_i'\hat{\beta}_{\tau_j}$ for each observation $i$ and quantile $j$
2. For each observation, sort the predictions: $\hat{Q}^*_{\tau_1} \leq \hat{Q}^*_{\tau_2} \leq \ldots$
3. Recover new coefficients via least-squares: $\hat{\beta}^*_{\tau_j} = (X'X)^{-1}X'\hat{Q}^*_{\tau_j}$

| Pros | Cons |
|------|------|
| Simple and fast | Post-hoc (not embedded in estimation) |
| Always eliminates crossing | May distort coefficient interpretation |
| Widely accepted in the literature | Approximation — new coefficients are not exact QR solutions |

### Fix Method 2: Isotonic Regression

Applies monotone smoothing to the coefficient paths across quantiles:

```python
import numpy as np

# Get coefficient matrix (n_tau x n_coef)
tau_list = np.array(sorted(results.results.keys()))
coef_matrix = np.array([results.results[tau].params for tau in tau_list])

# Apply isotonic regression to each coefficient
monotone_coefs = QuantileMonotonicity.isotonic_regression(coef_matrix, tau_list)
```

**How it works**: for each coefficient $\beta_j$, fit an isotonic regression to the path $\{\beta_j(\tau_1), \ldots, \beta_j(\tau_K)\}$, ensuring monotonicity in $\tau$.

!!! note
    Isotonic regression on individual coefficients does not guarantee non-crossing of the predicted quantile curves — it only ensures each coefficient is monotone in $\tau$. For guaranteed non-crossing, use rearrangement or constrained optimization.

### Fix Method 3: Constrained Optimization

The most principled approach: estimate all quantiles jointly subject to non-crossing constraints:

```python
tau_list = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

fixed_results = QuantileMonotonicity.constrained_qr(
    X=X,
    y=y,
    tau_list=tau_list,
    method="trust-constr",  # optimization method
    max_iter=1000,
    verbose=True,
)

# Result is {tau: beta_array}
for tau in tau_list:
    print(f"tau={tau:.2f}: beta = {fixed_results[tau]}")
```

The optimization problem:

$$\min \sum_{\tau \in \mathcal{T}} \sum_{i=1}^{n} \rho_\tau(y_i - X_i'\beta(\tau)) \quad \text{s.t.} \quad X_i'\beta(\tau_1) \leq X_i'\beta(\tau_2) \quad \forall i, \; \tau_1 < \tau_2$$

| Pros | Cons |
|------|------|
| Most principled (optimizes true objective) | Slowest method |
| Guarantees non-crossing by construction | High-dimensional constraint set |
| Estimates all quantiles jointly | May not converge for large problems |

### Fix Method 4: Simultaneous QR with Soft Penalty

An alternative that adds a penalty for crossing rather than hard constraints:

```python
fixed_results = QuantileMonotonicity.simultaneous_qr(
    X=X,
    y=y,
    tau_list=tau_list,
    lambda_nc=1.0,     # penalty strength
    max_iter=100,
    tol=1e-6,
    verbose=True,
)
```

The penalized objective:

$$\min \sum_\tau \sum_i \rho_\tau(y_i - X_i'\beta(\tau)) + \lambda \sum_\tau \sum_i [\max(0, X_i'\beta(\tau) - X_i'\beta(\tau^+))]^2$$

### Fix Method 5: Projection

Project predictions to the monotone space:

```python
import numpy as np

# Compute predictions matrix (n_obs x n_tau)
predictions = np.column_stack([X @ results.results[tau].params
                                for tau in sorted(results.results.keys())])

# Project to monotone space
projected = QuantileMonotonicity.project_to_monotone(
    predictions,
    method="averaging",   # or "isotonic"
)
```

### Comparison of Methods

| Method | Speed | Quality | Non-crossing Guarantee | Implementation |
|--------|-------|---------|----------------------|----------------|
| Rearrangement | Fast | Good | Yes (predictions) | Post-hoc |
| Isotonic regression | Fast | Moderate | No (coefficients only) | Post-hoc |
| Constrained QR | Slow | Best | Yes (by construction) | Joint estimation |
| Simultaneous QR | Moderate | Good | Soft (penalty-based) | Joint estimation |
| Location-Scale | Fast | Good | Yes (by construction) | Different model |

Use `MonotonicityComparison` for systematic comparison:

```python
from panelbox.models.quantile.monotonicity import MonotonicityComparison

comp = MonotonicityComparison(X=X, y=y, tau_list=tau_list)
comparison_df = comp.compare_methods(
    methods=["unconstrained", "rearrangement", "isotonic", "constrained"],
    verbose=True,
)
print(comparison_df)
```

The comparison DataFrame shows:

| Column | Description |
|--------|-------------|
| `method` | Correction method name |
| `has_crossing` | Whether crossing remains |
| `total_inversions` | Number of inversions |
| `pct_affected` | Percentage of observations affected |
| `total_loss` | Total check loss (objective value) |
| `avg_loss` | Average check loss per observation |

### Visualization

```python
# Visualize crossing violations
report.plot_violations(X, results.results)

# Compare coefficient paths across methods
comp.plot_comparison(var_idx=0)  # plot for first variable
```

### The Location-Scale Alternative

Instead of fixing crossing after estimation, avoid it entirely with the [Location-Scale model](location-scale.md):

```python
from panelbox.models.quantile import LocationScale

# Non-crossing by construction
ls_model = LocationScale(
    data=panel_data,
    formula="y ~ x1 + x2",
    tau=[0.1, 0.25, 0.5, 0.75, 0.9],
    distribution="normal",
)
ls_results = ls_model.fit()
# Guaranteed: Q_0.1 <= Q_0.25 <= Q_0.5 <= Q_0.75 <= Q_0.9
```

## Configuration Options

### `detect_crossing(results, X)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `results` | dict | `{tau: result}` with `.params` attribute |
| `X` | ndarray | Design matrix for predictions |

Returns: `CrossingReport`

### `rearrangement(results, X)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `results` | dict | `{tau: result}` |
| `X` | ndarray | Design matrix |

Returns: `dict` — new results with rearranged coefficients

### `constrained_qr(X, y, tau_list, method, max_iter, verbose)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | ndarray | *required* | Design matrix $(n, p)$ |
| `y` | ndarray | *required* | Response $(n,)$ |
| `tau_list` | ndarray | *required* | Quantile levels |
| `method` | str | `"trust-constr"` | Optimization method |
| `max_iter` | int | `1000` | Maximum iterations |
| `verbose` | bool | `False` | Print progress |

Returns: `dict` — `{tau: beta_array}` with non-crossing guarantee

### `simultaneous_qr(X, y, tau_list, lambda_nc, max_iter, tol, verbose)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lambda_nc` | float | `1.0` | Non-crossing penalty strength |
| `max_iter` | int | `100` | Maximum iterations |
| `tol` | float | `1e-6` | Convergence tolerance |

Returns: `dict` — `{tau: beta_array}`

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Non-Crossing Quantiles | Detection, correction, and comparison | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/08_monotonicity_non_crossing.ipynb) |

## See Also

- [Pooled Quantile Regression](pooled.md) — standard QR (may produce crossing)
- [Fixed Effects Quantile Regression](fixed-effects.md) — FE QR (may produce crossing)
- [Location-Scale Model](location-scale.md) — non-crossing by construction
- [Diagnostics](diagnostics.md) — crossing detection as part of diagnostic workflow

## References

- Chernozhukov, V., Fernandez-Val, I., & Galichon, A. (2010). Quantile and probability curves without crossing. *Econometrica*, 78(3), 1093-1125.
- Bondell, H. D., Reich, B. J., & Wang, H. (2010). Noncrossing quantile regression curve estimation. *Biometrika*, 97(4), 825-838.
- Dette, H., & Volgushev, S. (2008). Non-crossing non-parametric estimates of quantile curves. *Journal of the Royal Statistical Society B*, 70(3), 609-627.
- Machado, J. A., & Santos Silva, J. M. C. (2019). Quantiles via moments. *Journal of Econometrics*, 213(1), 145-173.
