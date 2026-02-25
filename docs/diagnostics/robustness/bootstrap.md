---
title: "Bootstrap Inference"
description: "Resampling-based confidence intervals for panel data models using pairs, wild, block, and residual bootstrap methods."
---

# Bootstrap Inference

!!! info "Quick Reference"
    **Class:** `panelbox.validation.robustness.PanelBootstrap`
    **Import:** `from panelbox.validation.robustness import PanelBootstrap`
    **Key method:** `bootstrap.run()` then `bootstrap.conf_int()`
    **Stata equivalent:** `bootstrap` prefix
    **R equivalent:** `boot::boot()`

## Why Bootstrap?

Asymptotic inference relies on assumptions -- normality, correct variance specification, large samples -- that may not hold in practice. Bootstrap inference replaces these assumptions with computation: resample the data, re-estimate the model many times, and let the empirical distribution speak for itself.

Bootstrap is especially valuable when:

- The number of clusters (entities) is small ($N < 50$), making clustered SEs unreliable
- The distribution of the test statistic is non-standard
- You want distribution-free confidence intervals
- You suspect heteroskedasticity or serial correlation patterns that analytical SEs may not fully capture

## Four Bootstrap Methods

PanelBox implements four bootstrap methods, each suited to different data structures:

| Method | Resampling Unit | Preserves | Best For |
|--------|----------------|-----------|----------|
| `pairs` | Entire entities | Panel structure, within-entity correlation | General purpose (default) |
| `wild` | Residuals (Rademacher weights) | Heteroskedasticity pattern | Heteroskedastic errors |
| `block` | Blocks of time periods | Temporal dependence | Autocorrelated data |
| `residual` | i.i.d. residuals | Nothing special | Homoskedastic i.i.d. errors |

### Pairs Bootstrap (Default)

Resamples entire entities with replacement. If the original panel has $N$ entities, draw $N$ entities randomly (with replacement) and stack their complete time series. This preserves within-entity correlation and is robust to both heteroskedasticity and serial correlation.

```python
bootstrap = PanelBootstrap(results, n_bootstrap=1000, method="pairs", random_state=42)
```

### Wild Bootstrap

Keeps the design matrix $X$ fixed and perturbs residuals using Rademacher weights $w_i \in \{-1, +1\}$ with equal probability. The bootstrap outcome is $y^* = \hat{y} + w \cdot \hat{e}$. Specifically designed for heteroskedasticity but does not preserve serial correlation.

```python
bootstrap = PanelBootstrap(results, n_bootstrap=1000, method="wild", random_state=42)
```

### Block Bootstrap

Resamples blocks of consecutive time periods (moving block bootstrap). Block size defaults to $T^{1/3}$ or can be set manually. Preserves temporal dependence within blocks while breaking dependence between blocks.

```python
bootstrap = PanelBootstrap(
    results, n_bootstrap=1000, method="block", block_size=3, random_state=42
)
```

### Residual Bootstrap

Resamples centered residuals assuming i.i.d. errors. The algorithm: (1) center residuals $\tilde{e} = e - \bar{e}$, (2) resample $\tilde{e}^*$ with replacement, (3) reconstruct $y^* = \hat{y} + \tilde{e}^*$. Most restrictive assumptions -- use only when confident errors are i.i.d.

```python
bootstrap = PanelBootstrap(results, n_bootstrap=1000, method="residual", random_state=42)
```

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.validation.robustness import PanelBootstrap
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit()

# Pairs bootstrap with BCA intervals
bootstrap = PanelBootstrap(
    results=results,
    n_bootstrap=1000,
    method="pairs",
    random_state=42,
    show_progress=True,
)
bootstrap.run()

# Confidence intervals
ci = bootstrap.conf_int(alpha=0.05, method="percentile")
print(ci)

# Compare bootstrap SEs with asymptotic SEs
summary = bootstrap.summary()
print(summary)

# Visualize bootstrap distribution
bootstrap.plot_distribution(param="value")
```

## API Reference

### Constructor

```python
PanelBootstrap(
    results=results,         # PanelResults from model.fit()
    n_bootstrap=1000,        # Number of replications
    method="pairs",          # 'pairs', 'wild', 'block', 'residual'
    block_size=None,         # For block bootstrap (default: T^(1/3))
    random_state=42,         # Reproducibility seed
    show_progress=True,      # Display progress bar
    parallel=False,          # Parallel computation (not yet implemented)
)
```

!!! note "Backward Compatibility"
    The `model` parameter is accepted as an alias for `results`, and `seed` as an alias for `random_state`. Use the preferred names `results` and `random_state` for new code.

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `run()` | `PanelBootstrap` | Execute bootstrap (returns self for chaining) |
| `conf_int(alpha, method)` | `pd.DataFrame` | Confidence intervals (lower/upper columns) |
| `summary()` | `pd.DataFrame` | Comparison of original and bootstrap estimates |
| `plot_distribution(param)` | -- | Histogram of bootstrap distribution with CI bands |

### Result Attributes (after `run()`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `bootstrap_estimates_` | `np.ndarray` | Bootstrap coefficient estimates ($B \times K$) |
| `bootstrap_se_` | `np.ndarray` | Bootstrap standard errors |
| `bootstrap_t_stats_` | `np.ndarray` | Studentized bootstrap t-statistics |
| `n_failed_` | `int` | Number of failed replications |

## Confidence Interval Methods

```python
# Percentile method (simplest, recommended)
ci = bootstrap.conf_int(alpha=0.05, method="percentile")

# Basic (reflection) method
ci = bootstrap.conf_int(alpha=0.05, method="basic")

# Bias-corrected accelerated (most accurate)
ci = bootstrap.conf_int(alpha=0.05, method="bca")

# Studentized (requires nested bootstrap)
ci = bootstrap.conf_int(alpha=0.05, method="studentized")
```

| Method | Formula | Properties |
|--------|---------|------------|
| `percentile` | $[\theta^*_{\alpha/2}, \theta^*_{1-\alpha/2}]$ | Simple, range-preserving |
| `basic` | $[2\hat\theta - \theta^*_{1-\alpha/2}, 2\hat\theta - \theta^*_{\alpha/2}]$ | Bias-corrected by reflection |
| `bca` | Bias-corrected and accelerated | Most accurate; adjusts for bias and skewness |
| `studentized` | Uses bootstrap t-distribution | Asymptotically optimal; computationally intensive |

!!! warning "BCA and Studentized"
    The `bca` and `studentized` methods currently fall back to the `percentile` method with a warning. The `percentile` method is adequate for most applications.

## Rules of Thumb

| Goal | Minimum `n_bootstrap` |
|------|----------------------|
| Standard errors | 500 |
| Confidence intervals | 1,000 |
| Hypothesis testing | 2,000 |
| Percentile precision | Use odd numbers (e.g., 999, 1999) |

!!! tip "Handling Failures"
    If more than 10% of replications fail, PanelBox issues a warning. If more than 50% fail, an error is raised. Many failures indicate problems with the model specification or insufficient data within resampled subsets. Try a different bootstrap method or simplify the model.

## Choosing a Bootstrap Method

```text
Is serial correlation a concern?
├── Yes → Is heteroskedasticity also present?
│        ├── Yes → pairs (preserves both)
│        └── No  → block (preserves time dependence)
└── No  → Is heteroskedasticity present?
         ├── Yes → wild (specifically designed for het.)
         └── No  → residual (most efficient under i.i.d.)

When in doubt → pairs (safest default)
```

## Comparing Bootstrap and Asymptotic Inference

```python
import pandas as pd

# Run bootstrap
bootstrap = PanelBootstrap(results, n_bootstrap=1000, method="pairs", random_state=42)
bootstrap.run()

# Summary table: Original vs Bootstrap
summary = bootstrap.summary()
print(summary)
# Columns: Original, Bootstrap Mean, Bootstrap Bias, Original SE, Bootstrap SE, SE Ratio

# If SE Ratio >> 1: asymptotic SEs are too small (liberal inference)
# If SE Ratio << 1: asymptotic SEs are too large (conservative inference)
# If SE Ratio ≈ 1: asymptotic inference is reliable
```

## Visualization

```python
# Plot distribution for a single parameter
bootstrap.plot_distribution(param="value")

# Plot all parameters
bootstrap.plot_distribution()
```

The plot shows a histogram of bootstrap estimates with a red dashed line at the original point estimate.

## Common Pitfalls

!!! warning "Common Issues"

    1. **Too few replications**: Using $B < 500$ gives noisy SE estimates. Always use at least 1,000 for CIs.
    2. **Ignoring failures**: Check `bootstrap.n_failed_` after `run()`. High failure rates invalidate results.
    3. **Wrong method for the DGP**: Using `residual` bootstrap when errors are heteroskedastic will produce incorrect CIs.
    4. **Block size too large**: For block bootstrap, a block size near $T$ effectively resamples the entire time series, defeating the purpose.

## See Also

- [Jackknife](jackknife.md) -- Deterministic leave-one-out alternative to bootstrap
- [Sensitivity Analysis](sensitivity.md) -- Subsample stability assessment
- [Robustness Overview](index.md) -- Full robustness toolkit

## References

- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press, Chapter 11.
- Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414-427.
- Liu, R. Y. (1988). Bootstrap procedures under some non-i.i.d. models. *The Annals of Statistics*, 16(4), 1696-1708.
- Kunsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217-1241.
