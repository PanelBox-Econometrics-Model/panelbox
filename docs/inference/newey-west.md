---
title: "Newey-West HAC Standard Errors"
description: "Heteroskedasticity and autocorrelation consistent (HAC) standard errors for time series and panel data in PanelBox."
---

# Newey-West HAC Standard Errors

!!! info "Quick Reference"
    **Class:** `panelbox.standard_errors.NeweyWestStandardErrors`
    **Convenience:** `panelbox.standard_errors.newey_west()`
    **Model integration:** `model.fit(cov_type="newey_west")`
    **Stata equivalent:** `newey, lag(m)`
    **R equivalent:** `sandwich::NeweyWest()`

## Overview

When errors are **serially correlated** --- that is, $\text{Cov}(\varepsilon_t, \varepsilon_{t-j}) \neq 0$ for some $j > 0$ --- both classical and heteroskedasticity-robust standard errors are invalid. The Newey-West (1987) estimator provides standard errors that are consistent in the presence of both heteroskedasticity and autocorrelation of unknown form.

The estimator uses **kernel-weighted autocovariances** to capture serial dependence up to a specified maximum lag, while ensuring the resulting covariance matrix is positive semi-definite.

## When to Use

- Time series or longitudinal data with **serial correlation**
- Single entity observed over many time periods
- Panel data treated as pooled time series (before considering cross-sectional dependence)

!!! note "When NOT to use"
    - For **panel data with cross-sectional dependence**, use [Driscoll-Kraay](driscoll-kraay.md) (which extends Newey-West to panels).
    - For **panel data with within-entity correlation**, [clustered SE](clustered.md) is simpler and often sufficient.
    - Newey-West does **not** handle cross-sectional dependence.

## Quick Example

```python
from panelbox.standard_errors import NeweyWestStandardErrors, newey_west

# Convenience function
result = newey_west(X, resid, max_lags=4)
print(f"SE: {result.std_errors}")
print(f"Max lags: {result.max_lags}")
print(f"Kernel: {result.kernel}")
print(f"Prewhitening: {result.prewhitening}")

# Class-based (more control)
nw = NeweyWestStandardErrors(
    X=X, resid=resid,
    max_lags=6,
    kernel="bartlett",
    prewhitening=False,
)
result = nw.compute()

# Via model.fit()
from panelbox.models import PooledOLS
model = PooledOLS("y ~ x1 + x2", data, entity="firm", time="year")
results = model.fit(cov_type="newey_west")
print(results.summary())
```

## Mathematical Details

### The Newey-West Estimator

The HAC covariance matrix is:

$$
V_{NW} = (X'X)^{-1} \hat{\Omega}_{NW} (X'X)^{-1}
$$

where the **meat** is:

$$
\hat{\Omega}_{NW} = \hat{\Gamma}_0 + \sum_{j=1}^{m} w(j) \left( \hat{\Gamma}_j + \hat{\Gamma}_j' \right)
$$

### Autocovariance Matrices

The lag-$j$ autocovariance matrix captures serial dependence at lag $j$:

$$
\hat{\Gamma}_j = \frac{1}{n} \sum_{t=j+1}^{n} \left( x_t \hat{e}_t \right) \left( x_{t-j} \hat{e}_{t-j} \right)'
$$

At lag 0, this reduces to the heteroskedasticity-robust meat:

$$
\hat{\Gamma}_0 = \frac{1}{n} \sum_{t=1}^{n} \hat{e}_t^2 x_t x_t'
$$

### Kernel Weighting

The kernel weights $w(j)$ serve two purposes: (1) downweight distant autocovariances (which are noisily estimated), and (2) ensure positive semi-definiteness.

=== "Bartlett (default)"

    $$w(j) = 1 - \frac{j}{m+1}$$

    Linear decay to zero. Simple and widely used.

=== "Parzen"

    $$
    w(j) = \begin{cases}
    1 - 6z^2 + 6z^3 & \text{if } z \leq 0.5 \\
    2(1-z)^3 & \text{if } 0.5 < z \leq 1
    \end{cases}
    $$

    where $z = j/(m+1)$. Smoother decay with better bias properties.

=== "Quadratic Spectral"

    $$w(j) = \frac{3}{x^2}\left(\frac{\sin x}{x} - \cos x\right)$$

    where $x = 6\pi j / (5(m+1))$. Optimal in MSE sense (Andrews, 1991).

### Bandwidth Selection

The default bandwidth follows the Newey-West (1994) rule:

$$
m = \left\lfloor 4 \left( \frac{n}{100} \right)^{2/9} \right\rfloor
$$

where $n$ is the number of observations. This is conservative and works well in practice.

### Prewhitening

The optional AR(1) prewhitening step (Andrews & Monahan, 1992):

1. Fit $\hat{e}_t = \rho \hat{e}_{t-1} + v_t$
2. Apply HAC to the prewhitened residuals $v_t$
3. Transform back to get the covariance of the original estimator

Prewhitening can reduce finite-sample bias when autocorrelation is strong.

## Configuration Options

### NeweyWestStandardErrors Class

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | --- | Design matrix $(n \times k)$ |
| `resid` | `np.ndarray` | --- | Residuals $(n,)$ |
| `max_lags` | `int` or `None` | `None` | Maximum lags; if `None`, uses $\lfloor 4(n/100)^{2/9} \rfloor$ |
| `kernel` | `str` | `"bartlett"` | Kernel: `"bartlett"`, `"parzen"`, `"quadratic_spectral"` |
| `prewhitening` | `bool` | `False` | Apply AR(1) prewhitening |

### NeweyWestResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `cov_matrix` | `np.ndarray` | NW covariance matrix $(k \times k)$ |
| `std_errors` | `np.ndarray` | NW standard errors $(k,)$ |
| `max_lags` | `int` | Lags used |
| `kernel` | `str` | Kernel function used |
| `n_obs` | `int` | Number of observations |
| `n_params` | `int` | Number of parameters |
| `prewhitening` | `bool` | Whether prewhitening was applied |

## Prewhitening Comparison

```python
from panelbox.standard_errors import NeweyWestStandardErrors

# Without prewhitening
nw = NeweyWestStandardErrors(X, resid, max_lags=4, prewhitening=False)
result_no_pw = nw.compute()

# With prewhitening
nw_pw = NeweyWestStandardErrors(X, resid, max_lags=4, prewhitening=True)
result_pw = nw_pw.compute()

print("Without prewhitening:", result_no_pw.std_errors)
print("With prewhitening:   ", result_pw.std_errors)
```

## Newey-West vs Driscoll-Kraay

| Feature | Newey-West | Driscoll-Kraay |
|---------|-----------|----------------|
| Heteroskedasticity | Yes | Yes |
| Autocorrelation | Yes | Yes |
| Cross-sectional dependence | **No** | **Yes** |
| Designed for | Single time series | Panel data |
| Asymptotics | $n \to \infty$ | $T \to \infty$ |
| Bandwidth default | $\lfloor 4(n/100)^{2/9} \rfloor$ | $\lfloor 4(T/100)^{2/9} \rfloor$ |

In panel data, Driscoll-Kraay is generally preferred because it additionally handles cross-sectional dependence. Newey-West is appropriate when working with a single long time series or when cross-sectional independence holds.

## Diagnostics

### Diagnostic Summary

```python
nw = NeweyWestStandardErrors(X, resid, max_lags=4)
print(nw.diagnostic_summary())
```

### Testing for Serial Correlation

```python
from panelbox.validation.serial_correlation.wooldridge_ar import WooldridgeARTest

ar_result = WooldridgeARTest(results).run(alpha=0.05)
print(f"Wooldridge AR(1): statistic={ar_result.statistic:.3f}, p={ar_result.pvalue:.4f}")
print(ar_result.conclusion)
```

## Common Pitfalls

!!! warning "Pitfall 1: Too many lags"
    Setting `max_lags` too high relative to sample size wastes degrees of freedom and increases variance. Use the default or set $m < n/3$.

!!! warning "Pitfall 2: Using NW for panel data"
    Newey-West does not account for cross-sectional dependence. In panel data, errors at the same time period across different entities may be correlated. Use [Driscoll-Kraay](driscoll-kraay.md) instead.

!!! warning "Pitfall 3: Small samples"
    NW standard errors require a moderately large sample ($n > 50$) for the HAC estimator to perform well. The diagnostic summary warns when $n < 50$.

## See Also

- [Driscoll-Kraay](driscoll-kraay.md) --- Extends NW to panel data with cross-sectional dependence
- [Robust (HC0-HC3)](robust.md) --- When only heteroskedasticity is present (no autocorrelation)
- [Clustered](clustered.md) --- Alternative for panel data with within-entity correlation
- [Comparison](comparison.md) --- Compare NW with other SE types
- [Inference Overview](index.md) --- Choosing the right SE type

## References

- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
- Newey, W. K., & West, K. D. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies*, 61(4), 631-653.
- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica*, 59(3), 817-858.
- Andrews, D. W. K., & Monahan, J. C. (1992). An improved heteroskedasticity and autocorrelation consistent covariance matrix estimator. *Econometrica*, 60(4), 953-966.
