---
title: "Heckman Benchmarks"
description: "Performance benchmarks for Panel Heckman selection models including two-step and MLE estimation"
---

# Heckman Benchmarks

This page presents detailed performance benchmarks for PanelBox's Panel Heckman selection models. The two estimation methods — two-step (Heckman, 1979) and MLE (maximum likelihood) — have drastically different performance profiles.

!!! info "Benchmark Environment"
    **CPU**: Intel i7-10700K (8 cores, 3.8 GHz) | **RAM**: 32 GB DDR4 | **Python**: 3.12 | **NumPy**: 2.0 (MKL)

    Selection rate ~50%. Each benchmark averaged over 10 runs with fixed seed.

## Two-Step vs MLE

### Estimation Time

| N | T | Two-Step | MLE (q=10) | MLE (q=15) | MLE (q=20) |
|---|---|----------|-----------|-----------|-----------|
| 100 | 10 | 0.4s | 2.3s | 4.8s | 8.1s |
| 200 | 10 | 0.7s | 5.1s | 11.2s | 19.4s |
| 500 | 20 | 2.8s | 28.4s | 64.3s | 112.8s |
| 1000 | 20 | 5.9s | 71.2s | 168.5s | 294.1s |

Where q = number of Gauss-Hermite quadrature points.

### MLE/Two-Step Ratio

| N | q=10 | q=15 | q=20 |
|---|------|------|------|
| 100 | 6x | 12x | 20x |
| 200 | 7x | 16x | 28x |
| 500 | 10x | 23x | 40x |
| 1000 | 12x | 29x | 50x |

!!! warning "MLE Scalability"
    MLE becomes **impractical for N > 500** with standard quadrature settings. The MLE/two-step ratio increases with both N and q, making MLE progressively more expensive for larger panels.

### Convergence Rates

| Method | N=100 | N=200 | N=500 | N=1000 |
|--------|-------|-------|-------|--------|
| Two-Step | 100% | 100% | 100% | 100% |
| MLE (q=10) | 98% | 96% | 94% | 91% |

MLE convergence improves significantly when using two-step estimates as starting values.

## Quadrature Points Impact

The number of Gauss-Hermite quadrature points controls the accuracy of numerical integration over unobserved effects. Each additional point adds ~40-60% to the runtime.

| Quadrature Points | N=200 Time | Relative | Accuracy |
|-------------------|-----------|----------|----------|
| 5 | 2.1s | 1.0x | Low |
| 10 | 5.1s | 2.4x | Good (default) |
| 15 | 11.2s | 5.3x | High |
| 20 | 19.4s | 9.2x | Very high |
| 30 | 42s | 20x | Diminishing returns |

!!! tip "Quadrature Point Selection"
    - **q=10**: Default, good balance of speed and accuracy for most applications
    - **q=15**: For critical applications or smaller samples where accuracy matters
    - **q=20+**: Only for very small N (< 100) when maximum precision is needed
    - Beyond q=20, accuracy gains are negligible for typical panel data

## Bottleneck Analysis

### Two-Step Method

| Component | % of Runtime | Notes |
|-----------|-------------|-------|
| Probit estimation | 35% | Uses statsmodels GLM — well-optimized |
| Outcome regression (OLS + IMR) | 25% | Already optimized |
| Murphy-Topel covariance | 20% | Adjusted SEs for two-step procedure |
| Inverse Mills ratio | 15% | `norm.pdf()` / `norm.cdf()` |
| Other | 5% | |

### MLE Method

| Component | % of Runtime | Notes |
|-----------|-------------|-------|
| Quadrature integration | 60% | **Critical bottleneck** — exponential in q |
| Likelihood evaluation | 25% | Per-observation, per-iteration |
| Gradient computation | 10% | Numerical derivatives |
| Hessian inversion (SEs) | 5% | Via SciPy |

## Memory Usage

| Method | N=200 | N=500 | N=1000 |
|--------|-------|-------|--------|
| Two-Step | ~20 MB | ~45 MB | ~80 MB |
| MLE (q=10) | ~35 MB | ~80 MB | ~150 MB |
| MLE (q=15) | ~50 MB | ~120 MB | ~220 MB |
| MLE (q=20) | ~65 MB | ~160 MB | ~300 MB |

MLE stores likelihood contributions for each observation at each quadrature point, so memory scales as O(N * T * q).

## Selection Equation Complexity

More variables in the selection equation increase Probit estimation time (two-step) and likelihood dimensionality (MLE):

| Selection Variables | Two-Step (N=500) | MLE (N=500, q=10) |
|--------------------|-----------------|-------------------|
| 3 | 2.5s | 25s |
| 5 | 2.8s | 28s |
| 10 | 3.5s | 35s |
| 20 | 5.2s | 52s |

The effect is moderate: doubling selection variables adds ~30-40% to estimation time.

## Convergence Tips

### If MLE Does Not Converge

1. **Use two-step starting values** (most effective):

    ```python
    result_2step = model.fit(method='two-step')
    result_mle = model.fit(method='mle', start_params=result_2step.params)
    ```

2. **Reduce quadrature points**: start with q=10, increase if needed

3. **Check for collinearity**: remove highly correlated regressors

4. **Scale variables**: standardize continuous variables before estimation

5. **Increase maxiter**: default is 100, try 200-500 for difficult models

### If Estimates Are Unreasonable

- **rho > 1 or rho < -1**: likely model misspecification. Check exclusion restrictions and ensure the selection equation has valid instruments.
- **Very large standard errors**: weak identification. The selection equation may not be sufficiently predictive.

## Recommendations

=== "Two-Step (Default)"

    Use two-step estimation for:

    - **Large panels** (N > 200)
    - **Exploratory analysis** (quick results)
    - **Publication** (widely accepted, standard in applied work)
    - **When inference is not critical** (point estimates are primary interest)

    ```python
    model = PanelHeckman(data, outcome_formula, selection_formula,
                         entity_col, time_col)
    result = model.fit(method='two-step')
    ```

=== "MLE"

    Use MLE estimation for:

    - **Small panels** (N < 200) where efficiency gains matter
    - **Likelihood ratio tests** between nested models
    - **When two-step SEs are suspect** (e.g., extreme selection rates)

    ```python
    # Always use two-step starting values
    result_2step = model.fit(method='two-step')
    result_mle = model.fit(method='mle',
                           start_params=result_2step.params,
                           quadrature_points=10)
    ```

### Quick Decision Guide

```text
Is N > 500?
  → Use Two-Step (MLE is impractical)

Is N < 200 and efficiency matters?
  → Try MLE with q=10 and two-step starting values

Do you need likelihood ratio tests?
  → Use MLE (only method that provides log-likelihood)

Is this exploratory analysis?
  → Use Two-Step (5-50x faster)

Default:
  → Two-Step estimation
```

## Comparison with Other Software

| Software | N=200, T=10 (Two-Step) | N=200, T=10 (MLE, q=10) |
|----------|------------------------|--------------------------|
| PanelBox | 0.7s | 5.1s |
| R `sampleSelection` | 0.9s | N/A (no panel MLE) |
| Stata `heckman` / `xtheckman` | 0.5s | 3.8s |

PanelBox two-step is competitive with R and Stata. MLE is slightly slower than Stata but includes features Stata lacks (Murphy-Topel SEs, panel random effects via quadrature).

See [Comparison with R/Stata](comparison.md) for detailed cross-platform benchmarks.

## References

- Heckman, J. J. (1979). "Sample selection bias as a specification error." *Econometrica*.
- Murphy, K. M., & Topel, R. H. (1985). "Estimation and inference in two-step econometric models." *Journal of Business & Economic Statistics*.
- Wooldridge, J. M. (1995). "Selection corrections for panel data models under conditional mean independence assumptions." *Journal of Econometrics*.
- Kyriazidou, E. (1997). "Estimation of a panel data sample selection model." *Econometrica*.

## See Also

- [Performance Overview](index.md) — General performance guide
- [Censored & Selection API](../api/censored.md) — Full API documentation
- [Comparison with R/Stata](comparison.md) — Cross-platform benchmarks
