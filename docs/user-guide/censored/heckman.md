---
title: "Panel Heckman"
description: "Panel Heckman sample selection model (Wooldridge 1995) for correcting selection bias in panel data."
---

# Panel Heckman

!!! info "Quick Reference"
    **Class:** `panelbox.models.selection.PanelHeckman`
    **Import:** `from panelbox.models.selection import PanelHeckman`
    **Stata equivalent:** `heckman y x1 x2, select(d = z1 z2)`
    **R equivalent:** `sampleSelection::selection()`

## Overview

The Panel Heckman model corrects for **sample selection bias** -- the problem that arises when the outcome of interest is observed only for a non-random subset of the population. The classic example is wage estimation: wages are observed only for employed individuals, and the employment decision is likely correlated with potential wages. Ignoring this produces biased estimates.

The model consists of two equations:

1. **Selection equation**: determines whether the outcome is observed
$$d_{it} = \mathbb{1}\{W_{it}'\gamma + v_{it} > 0\}$$

2. **Outcome equation**: the model of interest, observed only when $d_{it} = 1$
$$y_{it} = X_{it}'\beta + \varepsilon_{it}$$

The errors $(v_{it}, \varepsilon_{it})$ follow a bivariate normal distribution with correlation $\rho$. When $\rho \neq 0$, OLS on the selected subsample is biased. The Heckman correction adds the **Inverse Mills Ratio** (IMR) to the outcome equation, removing the selection bias.

PanelBox implements the Wooldridge (1995) panel extension with both **two-step** and **maximum likelihood** estimation, plus comprehensive diagnostics for assessing selection bias.

## Quick Example

```python
import numpy as np
from panelbox.models.selection import PanelHeckman

# Prepare data
y = wages              # Outcome (observed for employed only)
X = X_outcome          # Outcome regressors: [education, experience]
d = employed           # Selection indicator: 1=employed, 0=not
Z = X_selection        # Selection regressors: [education, experience, children]
entity = person_id
time = year

# Fit two-step Heckman model
model = PanelHeckman(
    endog=y,
    exog=X,
    selection=d,
    exog_selection=Z,
    entity=entity,
    time=time,
    method="two_step",
)
results = model.fit()
print(results.summary())

# Test for selection bias
test = results.selection_effect()
print(test["interpretation"])
```

## When to Use

- Your outcome is observed only for a **non-random subsample**
- You suspect the selection mechanism is **correlated** with the outcome
- You have a valid **exclusion restriction** (a variable affecting selection but not the outcome)
- Examples: wages (employed only), medical costs (insured only), firm profits (surviving firms only)

!!! warning "Key Assumptions"
    - **Bivariate normality**: $(v_{it}, \varepsilon_{it}) \sim N(0, \Sigma)$
    - **Exclusion restriction**: at least one variable in $W$ should not appear in $X$
    - **Exogenous regressors**: $X$ and $W$ are uncorrelated with the errors (conditional on selection)
    - The selection indicator is **binary** (0/1)

## The Selection Problem

### Why OLS is Biased

When we estimate $y_{it} = X_{it}'\beta + \varepsilon_{it}$ using only selected observations ($d_{it} = 1$), the conditional expectation is:

$$E[y_{it} \mid d_{it} = 1, X_{it}] = X_{it}'\beta + \rho \sigma_\varepsilon \lambda(W_{it}'\gamma)$$

where $\lambda(z) = \phi(z) / \Phi(z)$ is the Inverse Mills Ratio. The term $\rho \sigma_\varepsilon \lambda(\cdot)$ is the **omitted variable** that causes bias in OLS.

### Direction of Bias

| Sign of $\rho$ | Interpretation | OLS bias |
|----------------|---------------|----------|
| $\rho > 0$ | Positive selection: high-outcome individuals more likely selected | Upward bias |
| $\rho < 0$ | Negative selection: low-outcome individuals more likely selected | Downward bias |
| $\rho = 0$ | No selection bias | OLS is unbiased |

## Detailed Guide

### Model Specification

```python
from panelbox.models.selection import PanelHeckman

model = PanelHeckman(
    endog=y,                   # Outcome variable
    exog=X,                    # Outcome regressors (n x k_outcome)
    selection=d,               # Binary selection: 1=observed, 0=not
    exog_selection=Z,          # Selection regressors (n x k_selection)
    entity=entity,             # Entity IDs (optional)
    time=time,                 # Time IDs (optional)
    method="two_step",         # 'two_step' or 'mle'
)
```

!!! tip "Exclusion restriction"
    The selection regressors `exog_selection` should include at least one variable that is **not** in `exog`. This exclusion restriction is critical for identification. Without it, the model relies solely on the nonlinearity of the IMR, which is fragile.

    **Good exclusion restrictions** affect selection but not the outcome directly:

    | Application | Selection variable | Exclusion restriction |
    |-------------|-------------------|----------------------|
    | Wages | Employment | Number of children, non-labor income |
    | Training effects | Participation | Program availability, distance |
    | Insurance | Purchase decision | State regulations, subsidies |

### Two-Step Estimation

The default and recommended approach:

```python
results = model.fit()  # Uses two_step by default
```

**Step 1**: Estimate the selection equation via probit:

$$\hat{\gamma} = \arg\max \sum_{i,t} \left[ d_{it} \log \Phi(W_{it}'\gamma) + (1-d_{it}) \log(1-\Phi(W_{it}'\gamma)) \right]$$

**Step 2**: Compute the IMR and run augmented OLS on the selected sample:

$$y_{it} = X_{it}'\beta + \theta \hat{\lambda}_{it} + \text{error}_{it} \quad (d_{it}=1 \text{ only})$$

where $\hat{\lambda}_{it} = \phi(W_{it}'\hat{\gamma}) / \Phi(W_{it}'\hat{\gamma})$ and $\theta = \rho \sigma_\varepsilon$.

### MLE Estimation

Full information maximum likelihood jointly estimates all parameters:

```python
results_mle = model.fit(method="mle")
```

MLE is asymptotically more efficient but computationally expensive. It uses Fisher's z-transformation for $\rho$ and log-transformation for $\sigma$ to ensure parameters stay in valid ranges.

!!! note "Performance"
    MLE with $N > 500$ observations may take several minutes. Use two-step estimation for large samples or exploratory analysis.

### Result Attributes

| Attribute | Description |
|-----------|-------------|
| `results.outcome_params` | Outcome equation coefficients $\hat{\beta}$ |
| `results.probit_params` | Selection equation coefficients $\hat{\gamma}$ |
| `results.sigma` | Outcome error SD $\hat{\sigma}_\varepsilon$ |
| `results.rho` | Error correlation $\hat{\rho}$ |
| `results.lambda_imr` | Inverse Mills Ratio for each observation |
| `results.method` | Estimation method used |
| `results.llf` | Log-likelihood (MLE only) |
| `results.converged` | Convergence status |
| `results.n_selected` | Number of selected observations |
| `results.n_total` | Total number of observations |

### Predictions

```python
# Unconditional: E[y*] = X'beta (latent outcome)
y_unconditional = results.predict(type="unconditional")

# Conditional: E[y|selected] = X'beta + rho*sigma*lambda
y_conditional = results.predict(type="conditional")

# Out-of-sample prediction
y_new = results.predict(
    exog=X_new,
    exog_selection=Z_new,
    type="conditional",
)
```

## Diagnostics

### Testing for Selection Bias

The fundamental question: is selection bias actually present? Test $H_0: \rho = 0$.

```python
test = results.selection_effect()
print(f"Test statistic: {test['statistic']:.3f}")
print(f"P-value: {test['pvalue']:.4f}")
print(test["interpretation"])
```

If you fail to reject $H_0$, OLS on the selected sample may be adequate.

### IMR Diagnostics

Examine the distribution of the Inverse Mills Ratio:

```python
diag = results.imr_diagnostics()
print(f"Mean IMR: {diag['imr_mean']:.3f}")
print(f"Std IMR: {diag['imr_std']:.3f}")
print(f"Range: [{diag['imr_min']:.3f}, {diag['imr_max']:.3f}]")
print(f"High IMR (>2): {diag['high_imr_count']}")
print(f"Selection rate: {diag['selection_rate']:.1%}")
```

High IMR values (> 2) indicate observations with very low selection probabilities where the correction is large. Many such observations can make the model unstable.

### IMR Visualization

```python
fig = results.plot_imr(figsize=(12, 5))
# Creates scatter (IMR vs selection prob) and histogram
```

### Comparing OLS vs. Heckman

Assess the magnitude of selection bias:

```python
comparison = results.compare_ols_heckman()
print(comparison["interpretation"])

# Coefficient-by-coefficient comparison
for i in range(len(comparison["beta_ols"])):
    print(f"  Variable {i}: OLS={comparison['beta_ols'][i]:.4f}, "
          f"Heckman={comparison['beta_heckman'][i]:.4f}, "
          f"Diff={comparison['difference'][i]:.4f}")
```

Large differences indicate substantial selection bias. If the differences are small and $\rho \approx 0$, OLS is adequate.

## Configuration Options

### PanelHeckman Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | array-like | required | Outcome variable |
| `exog` | array-like | required | Outcome equation regressors |
| `selection` | array-like | required | Binary selection indicator (0/1) |
| `exog_selection` | array-like | required | Selection equation regressors |
| `entity` | array-like | `None` | Entity identifiers |
| `time` | array-like | `None` | Time identifiers |
| `method` | str | `"two_step"` | `"two_step"` or `"mle"` |

### fit() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `None` | Override default method |

## Common Issues

### Collinearity with IMR

If the exclusion restriction is weak ($X$ and $W$ are very similar), the IMR $\hat{\lambda}$ becomes highly collinear with $X$. This leads to unstable estimates and large standard errors. **Solution**: find a better exclusion restriction.

### Extreme Selection Probabilities

When $\Phi(W'\gamma)$ is close to 0 or 1, the IMR can become very large, causing numerical instability. PanelBox clips probabilities to $[10^{-10}, 1 - 10^{-10}]$ to prevent division by zero, but extreme values still produce unreliable corrections.

### MLE Convergence

If MLE fails to converge:

1. Check that two-step estimates are reasonable (used as starting values)
2. Reduce the number of parameters (simplify the model)
3. Check for near-boundary $\rho$ ($|\rho| \approx 1$)
4. Fall back to two-step estimation

### Extreme Selection Rates

```text
Selection rate < 5% or > 95%
```

Very low or very high selection rates make the IMR unstable. PanelBox warns when this occurs. Consider whether the Heckman model is appropriate for your data.

## Standard Errors

!!! note "Murphy-Topel correction"
    The two-step estimator produces **naive** standard errors that understate uncertainty because they treat the first-step probit estimates as known. The [Murphy-Topel correction](murphy-topel.md) adjusts for the estimation error in $\hat{\gamma}$, producing correct (larger) standard errors. See the dedicated page for details.

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Selection Models | Two-step and MLE Heckman estimation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/04_heckman_selection.ipynb) |

## See Also

- [Murphy-Topel Correction](murphy-topel.md) -- Correcting standard errors in two-step estimators
- [Tobit Models](tobit.md) -- Censored regression (outcome at boundary, not missing)
- [Marginal Effects for Censored Models](marginal-effects.md) -- Interpreting nonlinear effects
- [Honore Trimmed Estimator](honore.md) -- FE estimation for censored data

## References

- Heckman, J. J. (1979). Sample selection bias as a specification error. *Econometrica*, 47(1), 153-161.
- Wooldridge, J. M. (1995). Selection corrections for panel data models under conditional mean independence assumptions. *Journal of Econometrics*, 68(1), 115-132.
- Murphy, K. M., & Topel, R. H. (1985). Estimation and inference in two-step econometric models. *Journal of Business & Economic Statistics*, 3(4), 370-379.
- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press. Chapter 16.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapter 19.
