---
title: "CUE-GMM (Continuous Updating Estimator)"
description: "Hansen-Heaton-Yaron (1996) Continuous Updating GMM estimator with better finite-sample properties and invariance to moment normalization."
---

# CUE-GMM (Continuous Updating Estimator)

!!! info "Quick Reference"
    **Class:** `panelbox.gmm.ContinuousUpdatedGMM`
    **Import:** `from panelbox.gmm import ContinuousUpdatedGMM`
    **Stata equivalent:** `gmm (eq), instruments(...) winitial(unadjusted) wmatrix(hac)`
    **R equivalent:** `gmm(..., type = "cue")`

## Overview

The Continuous Updating Estimator (CUE), developed by Hansen, Heaton, and Yaron (1996), is an alternative to two-step GMM that **jointly optimizes** the coefficient vector $\beta$ and the weighting matrix $W(\beta)$. Unlike two-step GMM, which fixes $W$ at first-step estimates, CUE continuously recomputes $W$ as a function of the parameters being estimated.

CUE-GMM minimizes:

$$\hat{\beta}^{CUE} = \arg\min_\beta \; g(\beta)' \, W(\beta)^{-1} \, g(\beta)$$

where $g(\beta) = \frac{1}{N} \sum_i Z_i' \varepsilon_i(\beta)$ are the sample moment conditions and $W(\beta) = \frac{1}{N} \sum_i g_i(\beta) g_i(\beta)'$.

This approach offers **lower finite-sample bias** and **invariance to linear transformations** of the moment conditions, making it a valuable robustness check for standard GMM results.

## Quick Example

```python
from panelbox.gmm import ContinuousUpdatedGMM

model = ContinuousUpdatedGMM(
    data=panel_data,
    dep_var="y",
    exog_vars=["x1", "x2"],
    instruments=["z1", "z2", "z3"],
    weighting="hac",
    bandwidth="auto",
)
results = model.fit()
print(results.summary())
```

## When to Use

- **Robustness check** for two-step GMM results (do estimates change substantially?)
- **Small to moderate samples** (N = 100-1000) where finite-sample bias matters
- **Sensitivity concerns** about moment normalization
- **Cross-sectional or pooled GMM** settings (not dynamic panel-specific)

!!! warning "Key Considerations"
    1. CUE-GMM is **computationally expensive** -- the weighting matrix is recomputed at each optimization iteration
    2. **Good starting values** are critical -- defaults to two-step GMM estimates
    3. **Convergence is not guaranteed** with weak instruments
    4. Unlike `DifferenceGMM`/`SystemGMM`, CUE-GMM works with **explicit instrument variables** (not GMM-style dynamic instruments)

## Detailed Guide

### Motivation: Why Not Two-Step?

Two-step GMM estimates $\beta$ in two stages:

1. Estimate $\hat{\beta}_1$ with $W = I$ (identity)
2. Construct $\hat{W}$ from step-1 residuals, re-estimate $\hat{\beta}_2$

This creates a dependence on the first-step estimates through $\hat{W}$, which can cause:

- **Finite-sample bias** in moderate samples
- **Sensitivity to normalization** of moment conditions
- **Suboptimal efficiency** when $\hat{W}$ is imprecise

CUE eliminates this by treating $W$ as a function of $\beta$ in a single optimization.

### Theoretical Properties

Under standard regularity conditions (Hansen, Heaton, and Yaron 1996):

- **Consistent**: $\hat{\beta}^{CUE} \xrightarrow{p} \beta_0$
- **Asymptotically normal**: same limiting distribution as efficient two-step GMM
- **Invariant**: results do not depend on how moment conditions are scaled
- **Lower finite-sample bias**: better coverage of confidence intervals

### Weighting Options

CUE-GMM supports three weighting matrix types:

=== "HAC (Default)"
    ```python
    model = ContinuousUpdatedGMM(
        data=data, dep_var="y",
        exog_vars=["x1", "x2"],
        instruments=["z1", "z2", "z3"],
        weighting="hac",
        bandwidth="auto",  # Newey-West automatic bandwidth
    )
    ```
    Newey-West HAC kernel accounts for heteroskedasticity and autocorrelation. Bandwidth is set automatically using $L = \lfloor 4(T/100)^{2/9} \rfloor$.

=== "Cluster"
    ```python
    model = ContinuousUpdatedGMM(
        data=data,  # Must have MultiIndex (entity, time)
        dep_var="y",
        exog_vars=["x1", "x2"],
        instruments=["z1", "z2", "z3"],
        weighting="cluster",
    )
    ```
    Cluster-robust weighting by the first level of a MultiIndex (entity).

=== "Homoskedastic"
    ```python
    model = ContinuousUpdatedGMM(
        data=data, dep_var="y",
        exog_vars=["x1", "x2"],
        instruments=["z1", "z2", "z3"],
        weighting="homoskedastic",
    )
    ```
    Simple moment variance. Use only when errors are homoskedastic.

### Standard Error Options

=== "Analytical (Default)"
    ```python
    model = ContinuousUpdatedGMM(
        ..., se_type="analytical"
    )
    ```
    Uses the sandwich formula $\hat{V} = (\bar{G}' \hat{W}^{-1} \bar{G})^{-1}$.

=== "Bootstrap"
    ```python
    model = ContinuousUpdatedGMM(
        ...,
        se_type="bootstrap",
        n_bootstrap=999,
        bootstrap_method="residual",  # or "pairs"
    )
    results = model.fit()
    ci = model.conf_int(alpha=0.05, method="percentile")
    ```
    Residual or pairs bootstrap for inference robust to distributional assumptions.

### Comparing CUE with Two-Step GMM

```python
from panelbox.gmm import ContinuousUpdatedGMM

# Estimate CUE-GMM
cue_model = ContinuousUpdatedGMM(
    data=data, dep_var="y",
    exog_vars=["x1", "x2"],
    instruments=["z1", "z2", "z3"],
)
cue_results = cue_model.fit()

# Compare with two-step (using two_step results as baseline)
# CUE automatically uses two-step as starting values
j_test = cue_model.j_statistic()
print(f"J-statistic: {j_test['statistic']:.4f}")
print(f"J p-value: {j_test['pvalue']:.4f}")
print(f"Overidentification df: {j_test['df']}")
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | required | Data (MultiIndex for cluster) |
| `dep_var` | `str` | required | Dependent variable name |
| `exog_vars` | `list[str]` | required | Exogenous regressors |
| `instruments` | `list[str]` | required | Instrumental variables |
| `weighting` | `str` | `"hac"` | `"hac"`, `"cluster"`, or `"homoskedastic"` |
| `bandwidth` | `str` or `int` | `"auto"` | HAC bandwidth (`"auto"` or integer) |
| `se_type` | `str` | `"analytical"` | `"analytical"` or `"bootstrap"` |
| `n_bootstrap` | `int` | `999` | Bootstrap replications |
| `bootstrap_method` | `str` | `"residual"` | `"residual"` or `"pairs"` |
| `max_iter` | `int` | `100` | Maximum optimization iterations |
| `tol` | `float` | `1e-6` | Convergence tolerance |
| `regularize` | `bool` | `True` | Ridge regularization for near-singular W |

## Diagnostics

### Hansen J-Test

```python
j_test = cue_model.j_statistic()
print(f"J-stat: {j_test['statistic']:.4f}, p-value: {j_test['pvalue']:.4f}")
print(f"Degrees of freedom: {j_test['df']}")
print(f"Interpretation: {j_test['interpretation']}")
```

### Convergence Check

```python
results = cue_model.fit()
print(f"Converged: {cue_model.converged_}")
print(f"Iterations: {cue_model.niter_}")
print(f"Criterion value: {cue_model.criterion_value_:.6f}")
```

!!! note "If CUE Does Not Converge"
    Try providing better starting values, increasing `max_iter`, or relaxing `tol`. Convergence failures often indicate weak instruments or near-singular weighting matrices.

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Complete GMM Guide | Overview of all GMM estimators | [Complete Guide](complete-guide.md) |
| GMM Diagnostics | Interpreting J-test and other diagnostics | [Diagnostics](diagnostics.md) |

## See Also

- [Difference GMM](difference-gmm.md) -- Standard dynamic panel GMM
- [System GMM](system-gmm.md) -- System GMM for persistent series
- [Bias-Corrected GMM](bias-corrected.md) -- Analytical bias correction
- [Diagnostics](diagnostics.md) -- GMM diagnostic tests

## References

1. Hansen, L. P., Heaton, J., & Yaron, A. (1996). "Finite-Sample Properties of Some Alternative GMM Estimators." *Journal of Business & Economic Statistics*, 14(3), 262-280.
2. Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703-708.
3. Hansen, L. P. (1982). "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica*, 50(4), 1029-1054.
4. Hall, A. R. (2005). *Generalized Method of Moments*. Oxford University Press.
