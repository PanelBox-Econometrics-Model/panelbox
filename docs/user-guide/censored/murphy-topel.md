---
title: "Murphy-Topel Variance Correction"
description: "Murphy-Topel correction for standard errors in two-step estimators like the Heckman selection model."
---

# Murphy-Topel Variance Correction

!!! info "Quick Reference"
    **Functions:** `murphy_topel_variance`, `heckman_two_step_variance`
    **Import:** `from panelbox.models.selection.murphy_topel import murphy_topel_variance, heckman_two_step_variance`
    **Applies to:** Any two-step estimator (Heckman two-step, generated regressors)

## Overview

Two-step estimation procedures -- such as the Heckman two-step -- produce standard errors that are **too small** when computed naively. The problem is that Step 2 treats the Step 1 estimates as **known constants**, ignoring the sampling uncertainty from Step 1. This leads to understated standard errors, overstated t-statistics, and incorrect inference.

Murphy & Topel (1985) developed a general correction that accounts for the estimation error propagated from Step 1 to Step 2. In the context of the Heckman model, Step 1 is the probit estimation of the selection equation, and Step 2 is the augmented OLS with the Inverse Mills Ratio (IMR) -- which depends on the estimated probit coefficients $\hat{\gamma}$.

The corrected variance is always **larger** than the naive variance, reflecting the additional uncertainty from the first step.

## The Problem

### Naive Two-Step Standard Errors

Consider a two-step procedure:

1. **Step 1**: Estimate $\hat{\theta}_1$ (e.g., probit coefficients $\hat{\gamma}$)
2. **Step 2**: Estimate $\hat{\theta}_2$ using $\hat{\theta}_1$ as if known (e.g., OLS with IMR)

The naive variance from Step 2 is:

$$\hat{V}_2^{naive} = \hat{\sigma}^2 (X_{aug}'X_{aug})^{-1}$$

This **ignores** that $\hat{\theta}_1$ was estimated, not known. The true asymptotic variance is larger.

### Impact on Inference

| Measure | Naive (incorrect) | Murphy-Topel (correct) |
|---------|-------------------|----------------------|
| Standard errors | Too small | Correct (larger) |
| t-statistics | Too large | Correct (smaller) |
| Confidence intervals | Too narrow | Correct (wider) |
| p-values | Too small | Correct (larger) |
| Rejection rates | Too high (over-rejection) | Correct |

## The Murphy-Topel Correction

### General Formula

The corrected variance-covariance matrix is:

$$\hat{V}^{MT}_2 = \hat{V}_2 + C \hat{V}_1 C'$$

where:

- $\hat{V}_1$ is the variance-covariance from Step 1 (probit)
- $\hat{V}_2$ is the uncorrected variance from Step 2 (OLS)
- $C = \frac{\partial^2 Q}{\partial \theta_2 \partial \theta_1'}$ is the **cross-derivative** matrix that captures how Step 2 depends on Step 1

The correction term $C \hat{V}_1 C'$ is always positive semi-definite, so corrected standard errors are always at least as large as naive ones.

### For Heckman Specifically

In the Heckman two-step:

- Step 1 estimates $\hat{\gamma}$ (probit on selection equation)
- Step 2 estimates $(\hat{\beta}, \hat{\theta})$ where $\theta = \rho \sigma_\varepsilon$ is the IMR coefficient
- The IMR $\hat{\lambda}_{it} = \phi(W_{it}'\hat{\gamma}) / \Phi(W_{it}'\hat{\gamma})$ depends on $\hat{\gamma}$
- The cross-derivative captures how the IMR changes with $\gamma$:

$$\frac{\partial \hat{\lambda}}{\partial \gamma} = \frac{d\lambda}{dz} \cdot W_{it} = -\lambda(\lambda + z) \cdot W_{it}$$

## API Reference

### General Murphy-Topel Correction

```python
from panelbox.models.selection.murphy_topel import murphy_topel_variance

corrected_vcov = murphy_topel_variance(
    vcov_step1=vcov_probit,            # (k1 x k1) from Step 1
    vcov_step2_uncorrected=vcov_ols,   # (k2 x k2) naive from Step 2
    cross_derivative=C,                 # (k2 x k1) cross-derivative
)
```

**Parameters:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `vcov_step1` | $(k_1, k_1)$ | Step 1 variance-covariance matrix |
| `vcov_step2_uncorrected` | $(k_2, k_2)$ | Naive Step 2 variance-covariance |
| `cross_derivative` | $(k_2, k_1)$ | Cross-derivative $\partial^2 Q / \partial \theta_2 \partial \theta_1'$ |

**Returns:** Corrected $(k_2, k_2)$ variance-covariance matrix.

### Heckman-Specific Convenience Function

```python
from panelbox.models.selection.murphy_topel import heckman_two_step_variance

vcov_corrected, se_corrected = heckman_two_step_variance(
    X=X,                   # Outcome regressors (full sample)
    W=W,                   # Selection regressors (full sample)
    y=y,                   # Outcome variable (NaN for non-selected)
    beta=beta_hat,         # Outcome coefficients
    gamma=gamma_hat,       # Probit coefficients
    theta=theta_hat,       # IMR coefficient (rho * sigma)
    sigma=sigma_hat,       # Outcome error SD
    selected=d,            # Selection indicator
    vcov_probit=V_probit,  # Probit variance-covariance
)
```

This function handles all intermediate computations:

1. Computes the IMR and its derivative
2. Computes the cross-derivative matrix
3. Computes the uncorrected OLS variance
4. Applies the Murphy-Topel correction

**Returns:**

- `vcov_corrected`: Corrected $(k_{outcome}+1, k_{outcome}+1)$ matrix (for $\beta$ and $\theta$)
- `se_corrected`: Corrected standard errors (square root of diagonal)

### Cross-Derivative Computation

For advanced users, the cross-derivative can be computed separately:

```python
from panelbox.models.selection.murphy_topel import compute_cross_derivative_heckman

C = compute_cross_derivative_heckman(
    X=X_selected,          # Outcome regressors (selected only)
    W=W_selected,          # Selection regressors (selected only)
    imr=imr_selected,      # IMR values (selected only)
    imr_derivative=dimr,   # dλ/dz (selected only)
    beta=beta_hat,         # Outcome coefficients
    theta=theta_hat,       # IMR coefficient
    selected=selected,     # Selection indicator
)
```

## Practical Example

### Manual Correction

```python
import numpy as np
from panelbox.models.selection import PanelHeckman
from panelbox.models.selection.murphy_topel import murphy_topel_variance

# Fit the Heckman model
model = PanelHeckman(
    endog=y, exog=X, selection=d, exog_selection=Z,
    method="two_step",
)
results = model.fit()

# The PanelHeckman two-step automatically applies Murphy-Topel
# internally. But you can also apply it manually:

# Step 1: Get probit variance (from probit fit)
# Step 2: Get uncorrected OLS variance
# Step 3: Compute cross-derivative
# Step 4: Apply correction
# corrected_vcov = murphy_topel_variance(V1, V2, C)
```

### Comparing Naive vs. Corrected SEs

```python
# The correction typically increases SEs by 5-30%
# depending on the strength of the first-stage estimation
print("Naive SEs are always <= Murphy-Topel SEs")
print("The difference reflects Step 1 estimation uncertainty")
```

## When the Correction Matters

The Murphy-Topel correction is most important when:

1. **Strong selection**: $|\rho|$ is far from zero, so the IMR plays a large role
2. **Imprecise first stage**: the probit has few observations or weak predictors
3. **Many observations at the boundary**: a large fraction of the sample is non-selected

The correction is less important when:

1. **Weak selection**: $\rho \approx 0$ (IMR coefficient is near zero)
2. **Precise first stage**: many observations, strong exclusion restriction
3. **MLE is used**: MLE automatically produces correct standard errors

## Alternatives

### Bootstrap

An alternative to the analytical Murphy-Topel correction is to **bootstrap the entire two-step procedure**:

1. Resample entities (with replacement) to preserve panel structure
2. Re-estimate both steps on the bootstrap sample
3. Repeat $B$ times
4. Compute variance across bootstrap estimates

This is computationally expensive ($B \times$ estimation time) but does not require analytical derivatives and is robust to model misspecification.

!!! note "Bootstrap implementation"
    PanelBox provides the `bootstrap_two_step_variance` function signature, but the full implementation is not yet available. For now, use the analytical Murphy-Topel correction or implement panel bootstrap manually.

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Selection Models | Includes Murphy-Topel correction examples | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/04_heckman_selection.ipynb) |

## See Also

- [Panel Heckman](heckman.md) -- The primary user of Murphy-Topel correction
- [Tobit Models](tobit.md) -- Censored regression (uses MLE, not two-step)
- [Marginal Effects for Censored Models](marginal-effects.md) -- Interpreting effects in nonlinear models

## References

- Murphy, K. M., & Topel, R. H. (1985). Estimation and inference in two-step econometric models. *Journal of Business & Economic Statistics*, 3(4), 370-379.
- Wooldridge, J. M. (1995). Selection corrections for panel data models under conditional mean independence assumptions. *Journal of Econometrics*, 68(1), 115-132.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Section 12.5.
- Pagan, A. (1984). Econometric issues in the analysis of regressions with generated regressors. *International Economic Review*, 25(1), 221-247.
