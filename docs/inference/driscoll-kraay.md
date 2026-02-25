---
title: "Driscoll-Kraay Standard Errors"
description: "Standard errors robust to cross-sectional dependence, heteroskedasticity, and autocorrelation for panel data in PanelBox."
---

# Driscoll-Kraay Standard Errors

!!! info "Quick Reference"
    **Class:** `panelbox.standard_errors.DriscollKraayStandardErrors`
    **Convenience:** `panelbox.standard_errors.driscoll_kraay()`
    **Model integration:** `model.fit(cov_type="driscoll_kraay")`
    **Stata equivalent:** `xtscc` (Hoechle 2007)
    **R equivalent:** `plm::vcovSCC()`

## Overview

In many panel datasets, entities are subject to **common shocks** --- macroeconomic fluctuations, regulatory changes, or weather events that affect all entities simultaneously. This creates **cross-sectional dependence**: errors for different entities in the same time period are correlated.

Standard clustered SE (by entity) do **not** account for this. Driscoll & Kraay (1998) proposed a nonparametric covariance estimator that is simultaneously robust to:

- Heteroskedasticity
- Autocorrelation (serial correlation)
- Cross-sectional dependence (spatial correlation)

This makes it ideal for **macro panels** and **industry panels** where common factors drive residual correlation.

## When to Use

- Panel data with **common shocks** (macroeconomic, policy, weather)
- Large $N$, moderately large $T$ (need $T$ large enough for HAC)
- When the Pesaran CD test rejects the null of cross-sectional independence
- Regional or industry panels where cross-sectional dependence is expected

!!! note "When NOT to use"
    - **Single time series**: Use [Newey-West](newey-west.md) instead
    - **Very short panels** ($T < 15$): DK may not perform well
    - **Spatial correlation with distance decay**: Use [Spatial HAC](spatial-hac.md) for explicit geographic weighting
    - **Macro panels with $T > N$**: Consider [PCSE](pcse.md) as an alternative

## Quick Example

```python
from panelbox.standard_errors import DriscollKraayStandardErrors, driscoll_kraay

# Convenience function
result = driscoll_kraay(X, resid, time_ids, max_lags=3)
print(f"SE: {result.std_errors}")
print(f"Kernel: {result.kernel}")
print(f"Max lags: {result.max_lags}")
print(f"Time periods: {result.n_periods}")

# Class-based (more control)
dk = DriscollKraayStandardErrors(
    X=X, resid=resid, time_ids=time_ids,
    max_lags=5,
    kernel="bartlett",
)
result = dk.compute()

# Via model.fit()
from panelbox.models import FixedEffects
model = FixedEffects("y ~ x1 + x2", data, entity="country", time="year")
results = model.fit(cov_type="driscoll_kraay")
print(results.summary())
```

## Mathematical Details

### The Driscoll-Kraay Estimator

The covariance matrix is:

$$
V_{DK} = (X'X)^{-1} S_{DK} (X'X)^{-1}
$$

where the **meat** $S_{DK}$ is a HAC estimator applied to **time-averaged cross-sectional moments**:

$$
S_{DK} = \hat{\Gamma}_0 + \sum_{j=1}^{m} w(j) \left( \hat{\Gamma}_j + \hat{\Gamma}_j' \right)
$$

The autocovariance matrix at lag $j$ is:

$$
\hat{\Gamma}_j = \sum_{t=j+1}^{T} \left( \sum_i X_{it} \hat{e}_{it} \right) \left( \sum_i X_{i,t-j} \hat{e}_{i,t-j} \right)'
$$

The key insight is that the cross-sectional sums $\sum_i X_{it} \hat{e}_{it}$ aggregate information across entities at each time period, capturing both within-entity and cross-entity correlation.

### Kernel Functions

The kernel weights $w(j)$ ensure positive semi-definiteness of $S_{DK}$:

=== "Bartlett (default)"

    $$w(j) = 1 - \frac{j}{m+1}$$

    Linear decay. Most commonly used. Equivalent to the Newey-West kernel.

=== "Parzen"

    $$
    w(j) = \begin{cases}
    1 - 6z^2 + 6z^3 & \text{if } z \leq 0.5 \\
    2(1-z)^3 & \text{if } 0.5 < z \leq 1
    \end{cases}
    $$

    where $z = j/(m+1)$. Smoother than Bartlett, with higher-order bias reduction.

=== "Quadratic Spectral"

    $$w(j) = \frac{3}{x^2}\left(\frac{\sin x}{x} - \cos x\right)$$

    where $x = 6\pi j / (5(m+1))$. Optimal in a mean-squared error sense (Andrews, 1991).

### Bandwidth Selection

The default bandwidth (maximum lags) follows the Newey-West rule:

$$
m = \left\lfloor 4 \left( \frac{T}{100} \right)^{2/9} \right\rfloor
$$

| $T$ | Default $m$ |
|-----|-------------|
| 10  | 1 |
| 20  | 2 |
| 30  | 2 |
| 50  | 3 |
| 100 | 4 |
| 200 | 5 |

## Configuration Options

### DriscollKraayStandardErrors Class

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | --- | Design matrix $(n \times k)$ |
| `resid` | `np.ndarray` | --- | Residuals $(n,)$ |
| `time_ids` | `np.ndarray` | --- | Time period identifiers $(n,)$ |
| `max_lags` | `int` or `None` | `None` | Maximum lags; if `None`, uses $\lfloor 4(T/100)^{2/9} \rfloor$ |
| `kernel` | `str` | `"bartlett"` | Kernel: `"bartlett"`, `"parzen"`, `"quadratic_spectral"` |

### DriscollKraayResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `cov_matrix` | `np.ndarray` | DK covariance matrix $(k \times k)$ |
| `std_errors` | `np.ndarray` | DK standard errors $(k,)$ |
| `max_lags` | `int` | Lags used |
| `kernel` | `str` | Kernel function used |
| `n_obs` | `int` | Number of observations |
| `n_params` | `int` | Number of parameters |
| `n_periods` | `int` | Number of time periods |
| `bandwidth` | `float` or `None` | Bandwidth parameter |

## Kernel Comparison Example

```python
from panelbox.standard_errors import DriscollKraayStandardErrors

kernels = ["bartlett", "parzen", "quadratic_spectral"]

for kernel in kernels:
    dk = DriscollKraayStandardErrors(
        X=X, resid=resid, time_ids=time_ids,
        max_lags=4, kernel=kernel,
    )
    result = dk.compute()
    print(f"{kernel:25s}: SE = {result.std_errors}")
```

## Diagnostics

### Diagnostic Summary

```python
dk = DriscollKraayStandardErrors(X, resid, time_ids)
print(dk.diagnostic_summary())
```

Reports the number of observations, time periods, maximum lags, kernel function, and warnings about potential issues (e.g., few time periods, large max_lags relative to $T$).

### Testing for Cross-Sectional Dependence

Before using DK, test whether cross-sectional dependence is actually present:

```python
from panelbox.validation.cross_sectional_dependence.pesaran_cd import PesaranCDTest

cd_result = PesaranCDTest(results).run(alpha=0.05)
print(f"Pesaran CD: statistic={cd_result.statistic:.3f}, p={cd_result.pvalue:.4f}")
print(cd_result.conclusion)
```

## Driscoll-Kraay vs Other SE Types

| Feature | Clustered (entity) | Driscoll-Kraay | Newey-West |
|---------|-------------------|----------------|------------|
| Heteroskedasticity | Yes | Yes | Yes |
| Within-entity autocorrelation | Yes | Yes | Yes |
| Cross-sectional dependence | **No** | **Yes** | **No** |
| Requires cluster structure | Yes | No | No |
| Asymptotics | $G \to \infty$ | $T \to \infty$ | $n \to \infty$ |
| Best for | Micro panels | Macro panels | Single time series |

## Common Pitfalls

!!! warning "Pitfall 1: Too few time periods"
    DK requires $T$ to be moderately large for the HAC estimator to work well. With $T < 15$, DK standard errors may be unreliable. The diagnostic summary warns when $T < 20$.

!!! warning "Pitfall 2: Bandwidth too large"
    Setting `max_lags` too large relative to $T$ reduces the effective sample size for each autocovariance estimate. A rule of thumb is $m < T/4$.

!!! warning "Pitfall 3: Highly unbalanced panels"
    DK works best with balanced or near-balanced panels. Severely unbalanced panels can distort the time-averaged moments.

## See Also

- [Newey-West HAC](newey-west.md) --- For autocorrelation without cross-sectional dependence
- [Clustered](clustered.md) --- When cross-sectional dependence is absent
- [Spatial HAC](spatial-hac.md) --- When spatial correlation follows geographic distance
- [PCSE](pcse.md) --- Alternative for macro panels with $T > N$
- [Comparison](comparison.md) --- Compare DK with other SE types
- [Inference Overview](index.md) --- Choosing the right SE type

## References

- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics*, 80(4), 549-560.
- Hoechle, D. (2007). Robust standard errors for panel regressions with cross-sectional dependence. *The Stata Journal*, 7(3), 281-312.
- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica*, 59(3), 817-858.
- Vogelsang, T. J. (2012). Heteroskedasticity, autocorrelation, and spatial correlation robust inference in linear panel models with fixed-effects. *Journal of Econometrics*, 166(2), 303-319.
