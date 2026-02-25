---
title: "Location-Scale Model"
description: "Machado-Santos Silva (2019) location-scale quantile regression with non-crossing guarantee in PanelBox"
---

# Location-Scale Model

!!! info "Quick Reference"
    **Class:** `panelbox.models.quantile.location_scale.LocationScale`
    **Import:** `from panelbox.models.quantile import LocationScale`
    **Stata equivalent:** No direct equivalent
    **R equivalent:** Partially via `Qtools::qlss()`

## Overview

The Location-Scale (LS) quantile regression model, introduced by Machado and Santos Silva (2019), provides a fundamentally different approach to estimating conditional quantiles. Instead of estimating each quantile independently, the LS model represents the entire conditional distribution through two components — location (mean) and scale (variance) — combined with a reference distribution's quantile function.

The conditional quantile is modeled as:

$$Q_y(\tau | X) = X'\alpha + \sqrt{\exp(X'\gamma)} \cdot q(\tau)$$

where:

- $X'\alpha$ is the **location** (conditional mean)
- $\sqrt{\exp(X'\gamma)}$ is the **scale** (conditional standard deviation)
- $q(\tau)$ is the **quantile function** of a chosen reference distribution

This decomposition provides three major advantages:

1. **Non-crossing guarantee**: quantile curves cannot cross by construction, since $q(\tau)$ is monotonically increasing and shared across all observations
2. **Computational efficiency**: only two regressions (location and scale) are needed, regardless of how many quantiles are requested
3. **Full density estimation**: the location-scale structure allows prediction of the complete conditional density, not just specific quantiles

## Quick Example

```python
from panelbox.core.panel_data import PanelData
from panelbox.models.quantile import LocationScale

panel_data = PanelData(data=df, entity_col="firm_id", time_col="year")

model = LocationScale(
    data=panel_data,
    formula="investment ~ value + capital",
    tau=[0.1, 0.25, 0.5, 0.75, 0.9],
    distribution="normal",
    fixed_effects=True,
)
results = model.fit(robust_scale=True)

# Predict conditional quantiles (guaranteed non-crossing)
quantile_preds = results.predict_quantiles(tau=[0.1, 0.5, 0.9])

# Predict conditional density
y_grid, density = results.predict_density(n_points=100)
```

## When to Use

- **Non-crossing is critical**: applications where quantile crossing would be problematic (e.g., risk management, forecasting)
- **Full distribution needed**: interest in the complete conditional distribution, not just specific quantiles
- **Computational speed**: estimating many quantiles (e.g., 99 percentiles for density estimation)
- **Panel fixed effects**: natural handling of entity heterogeneity
- **Extrapolation**: need quantiles beyond the range observed in the data (e.g., extreme tails)

!!! warning "Key Assumptions"
    - **Location-scale structure**: the conditional distribution is fully characterized by its mean and variance
    - **Correct reference distribution**: the shape of $q(\tau)$ matches the true conditional distribution
    - **Homogeneous distributional shape**: all observations share the same distributional form (up to location and scale shifts)
    - **Testable**: use `test_normality()` to check the reference distribution choice

## Detailed Guide

### Theoretical Foundation

The LS model decomposes the quantile estimation into two moment conditions:

**Step 1: Location estimation (conditional mean)**

$$\hat{\alpha} = \arg\min_\alpha \sum_{i,t} (y_{it} - X_{it}'\alpha)^2$$

This is standard OLS (or FE-OLS when `fixed_effects=True`).

**Step 2: Scale estimation (conditional variance)**

Using residuals $\hat{\varepsilon}_{it} = y_{it} - X_{it}'\hat{\alpha}$, the scale parameters are estimated from:

$$\log|\hat{\varepsilon}_{it}| = \frac{X_{it}'\gamma}{2} + \text{adjustment} + v_{it}$$

The robust approach (default) uses log-absolute residuals, which is more stable than regressing on squared residuals.

**Step 3: Quantile coefficients**

For any quantile $\tau$:

$$\hat{\beta}(\tau) = \hat{\alpha} + \sqrt{\exp(\hat{\gamma})} \odot q(\tau)$$

where $q(\tau)$ is the quantile function of the reference distribution.

### Reference Distributions

PanelBox supports four built-in reference distributions:

| Distribution | $q(\tau)$ | Tails | Best For |
|-------------|-----------|-------|----------|
| `"normal"` | $\Phi^{-1}(\tau)$ | Light | General purpose (default) |
| `"logistic"` | $\log[\tau/(1-\tau)]$ | Medium | Heavier-tailed data |
| `"t"` | $t_\nu^{-1}(\tau)$ | Heavy (adjustable) | Financial data, outliers |
| `"laplace"` | $-\text{sign}(\tau-0.5)\log(1-2|\tau-0.5|)$ | Medium-heavy | Peaked distributions |

You can also provide a custom callable:

```python
# Custom quantile function
import scipy.stats as stats

model = LocationScale(
    data=panel_data,
    formula="y ~ x1 + x2",
    tau=[0.1, 0.5, 0.9],
    distribution=lambda tau: stats.gennorm.ppf(tau, beta=1.5),
)
```

### Data Preparation

```python
from panelbox.core.panel_data import PanelData
from panelbox.models.quantile import LocationScale

panel_data = PanelData(data=df, entity_col="id", time_col="year")

model = LocationScale(
    data=panel_data,
    formula="y ~ x1 + x2",
    tau=[0.1, 0.25, 0.5, 0.75, 0.9],
    distribution="normal",      # reference distribution
    fixed_effects=True,          # include entity FE
    df_t=5,                      # degrees of freedom (for 't' distribution)
)
```

### Estimation

```python
results = model.fit(
    robust_scale=True,   # use log-transformation for scale (recommended)
    verbose=True,        # print step-by-step progress
)
```

The fit returns a `LocationScaleResult` containing:

- Location and scale regression results
- Quantile coefficients for all requested $\tau$
- Methods for prediction and diagnostics

### Interpreting Results

```python
# Location parameters (conditional mean effects)
print("Location params:", results.model.location_params_)

# Scale parameters (conditional variance effects)
print("Scale params:", results.model.scale_params_)

# Quantile-specific coefficients
for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
    r = results.results[tau]
    print(f"tau={tau:.2f}: beta = {r.params}")

# Location and scale from individual results
r_50 = results.results[0.5]
print(f"Location: {r_50.location_params}")
print(f"Scale: {r_50.scale_params}")
```

**Key insight**: if the scale parameter $\gamma_j$ for covariate $x_j$ is significantly different from zero, then $x_j$ affects not just the level but also the spread of $y$. This is a direct test for heteroskedasticity and distributional effects.

### Predicting Conditional Quantiles

```python
# Predict quantiles (guaranteed non-crossing)
quantile_df = results.predict_quantiles(
    X=None,                  # use estimation sample (or provide new data)
    tau=[0.1, 0.25, 0.5, 0.75, 0.9],
    ci=True,                  # include confidence intervals
    alpha=0.05,               # significance level
)
print(quantile_df.head())
```

### Predicting Conditional Density

A unique feature of the LS model: estimate the full conditional density $f(y|X)$.

```python
y_grid, density = results.predict_density(
    X=None,          # mean of covariates (default)
    y_grid=None,     # automatic grid
    n_points=100,    # grid resolution
)

# Plot the density
import matplotlib.pyplot as plt
plt.plot(y_grid, density)
plt.xlabel("y")
plt.ylabel("f(y|X)")
plt.title("Conditional Density Estimate")
```

### Testing the Reference Distribution

```python
# Test if the normal distribution is appropriate
normality = results.test_normality()
print(f"Normality test: stat={normality.statistic:.3f}, p={normality.pvalue:.3f}")

# If rejected, try alternative distributions
for dist in ["normal", "logistic", "t", "laplace"]:
    model_d = LocationScale(data=panel_data, formula="y ~ x1 + x2",
                             tau=0.5, distribution=dist)
    res_d = model_d.fit()
    norm_test = res_d.test_normality()
    print(f"{dist:10s}: test stat = {norm_test.statistic:.3f}, p = {norm_test.pvalue:.3f}")
```

### Fixed Effects

The LS model naturally accommodates entity fixed effects:

```python
# With fixed effects
model_fe = LocationScale(
    data=panel_data,
    formula="y ~ x1 + x2",
    tau=[0.1, 0.5, 0.9],
    distribution="normal",
    fixed_effects=True,    # within-transformation for location and scale
)
results_fe = model_fe.fit()
```

With `fixed_effects=True`:

1. **Location step**: uses within-transformation (FE-OLS)
2. **Scale step**: applies the same within-transformation to log-residuals
3. **Non-crossing property is preserved** because the transformation is linear

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | PanelData | *required* | Panel data object |
| `formula` | str | `None` | Model formula `"y ~ x1 + x2"` |
| `tau` | float/array | `0.5` | Quantile level(s) in $(0, 1)$ |
| `distribution` | str/callable | `"normal"` | Reference distribution: `"normal"`, `"logistic"`, `"t"`, `"laplace"` |
| `fixed_effects` | bool | `False` | Include entity fixed effects |
| `df_t` | float | `5` | Degrees of freedom for Student's $t$ distribution |

### Fit Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `robust_scale` | bool | `True` | Use robust (log-transformation) scale estimation |
| `verbose` | bool | `False` | Print estimation progress |

### Result Attributes

| Attribute | Description |
|-----------|-------------|
| `results` | Dict mapping $\tau \to$ `LocationScaleQuantileResult` |
| `location_result` | Step 1 (location) regression results |
| `scale_result` | Step 2 (scale) regression results |
| `model.location_params_` | Location parameter estimates $\hat{\alpha}$ |
| `model.scale_params_` | Scale parameter estimates $\hat{\gamma}$ |

### Result Methods

| Method | Description |
|--------|-------------|
| `predict_quantiles(X, tau, ci, alpha)` | Predict conditional quantiles |
| `predict_density(X, y_grid, n_points)` | Estimate conditional density |
| `test_normality(tau_grid)` | Test reference distribution fit |

## Comparison with Standard Quantile Regression

| Feature | Standard QR | Location-Scale |
|---------|------------|----------------|
| Non-crossing | Not guaranteed | Guaranteed by construction |
| Computation | One optimization per $\tau$ | Two regressions total |
| Flexibility | Fully nonparametric in $\tau$ | Constrained by reference distribution |
| Density estimation | Not directly available | Available via `predict_density()` |
| Fixed effects | Incidental parameters problem | Natural via within-transformation |
| Extreme quantiles | Unreliable near boundaries | Extrapolation via $q(\tau)$ |

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Location-Scale Model | Complete LS workflow with density estimation | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/05_location_scale_models.ipynb) |
| Distribution Selection | Comparing reference distributions | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/06_advanced_diagnostics.ipynb) |

## See Also

- [Pooled Quantile Regression](pooled.md) — standard quantile regression without distributional assumptions
- [Fixed Effects Quantile Regression](fixed-effects.md) — Koenker penalty FE approach
- [Canay Two-Step](canay.md) — also assumes location-shift structure
- [Non-Crossing Constraints](monotonicity.md) — post-hoc methods for standard QR
- [Diagnostics](diagnostics.md) — diagnostic tests including normality

## References

- Machado, J. A., & Santos Silva, J. M. C. (2019). Quantiles via moments. *Journal of Econometrics*, 213(1), 145-173.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Chernozhukov, V., Fernandez-Val, I., & Melly, B. (2013). Inference on counterfactual distributions. *Econometrica*, 81(6), 2205-2268.
- He, X. (1997). Quantile curves without crossing. *The American Statistician*, 51(2), 186-192.
