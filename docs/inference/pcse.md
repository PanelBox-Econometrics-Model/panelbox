---
title: "Panel-Corrected Standard Errors (PCSE)"
description: "Beck-Katz panel-corrected standard errors for time-series cross-section data with contemporaneous correlation in PanelBox."
---

# Panel-Corrected Standard Errors (PCSE)

!!! info "Quick Reference"
    **Class:** `panelbox.standard_errors.PanelCorrectedStandardErrors`
    **Convenience:** `panelbox.standard_errors.pcse()`
    **Model integration:** `model.fit(cov_type="pcse")`
    **Stata equivalent:** `xtpcse`
    **R equivalent:** `pcse::pcse()`

## Overview

Panel-Corrected Standard Errors (PCSE), proposed by Beck & Katz (1995), are designed for **time-series cross-section (TSCS)** data --- datasets with a small number of entities ($N$) observed over a long time period ($T$). Typical examples include panels of countries, states, or industries over decades.

PCSE estimates the full $N \times N$ **contemporaneous cross-sectional covariance matrix** $\hat{\Sigma}$ from the residuals, then uses it to compute corrected standard errors. This accounts for cross-entity error correlation (e.g., all countries being affected by a global recession in the same year).

!!! warning "Key Requirement: $T > N$"
    PCSE requires more time periods than entities. If $T \leq N$, the estimated $\hat{\Sigma}$ matrix is singular or poorly conditioned. For micro panels where $N \gg T$, use [clustered SE](clustered.md) instead.

## When to Use

- **Macro panels**: Countries, states, or regions observed over decades ($N = 20$, $T = 40$)
- **Political science TSCS data**: International relations, comparative politics
- **Industry panels**: Small number of industries over many quarters/years
- When contemporaneous cross-sectional correlation is expected

!!! note "When NOT to use"
    - **Micro panels** ($N \gg T$): Use [clustered SE](clustered.md)
    - **Spatial correlation with distance decay**: Use [Spatial HAC](spatial-hac.md)
    - **Large $N$, moderate $T$**: Use [Driscoll-Kraay](driscoll-kraay.md) (does not require $T > N$)

## Quick Example

```python
from panelbox.standard_errors import PanelCorrectedStandardErrors, pcse

# Convenience function
result = pcse(X, resid, entity_ids, time_ids)
print(f"SE: {result.std_errors}")
print(f"Entities (N): {result.n_entities}")
print(f"Periods (T): {result.n_periods}")
print(f"Sigma matrix shape: {result.sigma_matrix.shape}")

# Class-based (with diagnostics)
pcse_calc = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)
result = pcse_calc.compute()
print(pcse_calc.diagnostic_summary())

# Via model.fit()
from panelbox.models import PooledOLS
model = PooledOLS("y ~ x1 + x2", data, entity="country", time="year")
results = model.fit(cov_type="pcse")
print(results.summary())
```

## Mathematical Details

### The PCSE Estimator

PCSE uses FGLS with the estimated contemporaneous covariance matrix:

$$
V_{PCSE} = (X' \hat{\Omega}^{-1} X)^{-1}
$$

where:

$$
\hat{\Omega} = \hat{\Sigma} \otimes I_T
$$

and $\otimes$ denotes the Kronecker product.

### Estimating $\hat{\Sigma}$

The contemporaneous covariance matrix $\hat{\Sigma}$ is estimated from the OLS residuals:

$$
\hat{\Sigma}_{ij} = \frac{1}{T} \sum_{t=1}^{T} \hat{e}_{it} \hat{e}_{jt}
$$

In matrix form, if $E$ is the $N \times T$ matrix of residuals (entities as rows, time as columns):

$$
\hat{\Sigma} = \frac{1}{T} E E'
$$

### Why $T > N$?

The matrix $\hat{\Sigma}$ is $N \times N$, but its rank is at most $\min(N, T)$. When $T \leq N$:

- $\hat{\Sigma}$ is singular (rank $T < N$)
- $\hat{\Sigma}^{-1}$ does not exist (PanelBox falls back to pseudo-inverse with a warning)
- Standard errors are unreliable

A safe rule of thumb is $T > 2N$ for well-conditioned estimation.

### PCSE vs FGLS

Beck & Katz (1995) showed that Parks-Kmenta FGLS standard errors are **severely anti-conservative** in typical TSCS settings, rejecting at 50-60% when the true rejection rate should be 5%. PCSE provides much better coverage properties:

| Method | True rejection rate ($\alpha = 0.05$) | Coverage |
|--------|--------------------------------------|----------|
| OLS with classical SE | Varies | May under/over-cover |
| FGLS (Parks-Kmenta) | 50-60% | Severe under-coverage |
| OLS with PCSE | 5-8% | Near-nominal coverage |

## Configuration Options

### PanelCorrectedStandardErrors Class

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `np.ndarray` | Design matrix $(n \times k)$ |
| `resid` | `np.ndarray` | Residuals $(n,)$ |
| `entity_ids` | `np.ndarray` | Entity identifiers $(n,)$ |
| `time_ids` | `np.ndarray` | Time period identifiers $(n,)$ |

### PCSEResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `cov_matrix` | `np.ndarray` | PCSE covariance matrix $(k \times k)$ |
| `std_errors` | `np.ndarray` | PCSE standard errors $(k,)$ |
| `sigma_matrix` | `np.ndarray` | Estimated cross-sectional covariance $\hat{\Sigma}$ $(N \times N)$ |
| `n_obs` | `int` | Number of observations |
| `n_params` | `int` | Number of parameters |
| `n_entities` | `int` | Number of entities ($N$) |
| `n_periods` | `int` | Number of time periods ($T$) |

## Diagnostics

### Diagnostic Summary

```python
pcse_calc = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)
print(pcse_calc.diagnostic_summary())
```

The summary reports:

- Number of observations, entities ($N$), and time periods ($T$)
- $T/N$ ratio and whether it is sufficient
- Warnings if $T \leq N$ or $T < 2N$

### Examining the Cross-Sectional Covariance

```python
result = pcse_calc.compute()

# Inspect Sigma matrix
import numpy as np
print(f"Sigma shape: {result.sigma_matrix.shape}")
print(f"Sigma diagonal (variances): {np.diag(result.sigma_matrix)}")
print(f"Sigma condition number: {np.linalg.cond(result.sigma_matrix):.2f}")

# Correlation matrix
D_inv = np.diag(1.0 / np.sqrt(np.diag(result.sigma_matrix)))
corr = D_inv @ result.sigma_matrix @ D_inv
print(f"Cross-sectional correlation range: [{corr.min():.3f}, {corr.max():.3f}]")
```

## Common Pitfalls

!!! warning "Pitfall 1: $T \leq N$"
    The most critical issue. With 30 countries and 20 years, $\hat{\Sigma}$ is rank-deficient. PanelBox warns and uses pseudo-inverse, but results are unreliable. Use clustered SE or Driscoll-Kraay instead.

!!! warning "Pitfall 2: Unbalanced panels"
    PCSE works best with balanced panels. Missing observations reduce the effective $T$ for pairwise covariance estimation, which can degrade $\hat{\Sigma}$.

!!! warning "Pitfall 3: Ignoring autocorrelation"
    Standard PCSE accounts for contemporaneous cross-sectional correlation but not autocorrelation. If residuals are serially correlated, consider combining PCSE with a Prais-Winsten transformation or using [Driscoll-Kraay](driscoll-kraay.md).

!!! warning "Pitfall 4: Using PCSE coefficients instead of OLS"
    Beck & Katz (1995) recommend using **OLS coefficients** with PCSE standard errors, not FGLS coefficients. The OLS estimator is consistent and PCSE corrects only the standard errors.

## See Also

- [Clustered](clustered.md) --- For micro panels with $N \gg T$
- [Driscoll-Kraay](driscoll-kraay.md) --- Alternative that does not require $T > N$
- [Comparison](comparison.md) --- Compare PCSE with other SE types
- [Inference Overview](index.md) --- Choosing the right SE type

## References

- Beck, N., & Katz, J. N. (1995). What to do (and not to do) with time-series cross-section data. *American Political Science Review*, 89(3), 634-647.
- Beck, N., & Katz, J. N. (1996). Nuisance vs. substance: Specifying and estimating time-series-cross-section models. *Political Analysis*, 6, 1-36.
- Bailey, D., & Katz, J. N. (2011). Implementing panel corrected standard errors in R: The pcse package. *Journal of Statistical Software*, 42(CS1), 1-11.
- Parks, R. W. (1967). Efficient estimation of a system of regression equations when disturbances are both serially and contemporaneously correlated. *Journal of the American Statistical Association*, 62(318), 500-509.
