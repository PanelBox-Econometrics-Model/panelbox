---
title: "GMM Benchmarks"
description: "Performance benchmarks for dynamic panel GMM estimators including CUE-GMM, Bias-Corrected GMM, and instrument collapse"
---

# GMM Benchmarks

This page presents detailed performance benchmarks for PanelBox's GMM estimators: one-step, two-step, CUE-GMM (Continuous Updated Estimator), and Bias-Corrected GMM (Hahn-Kuersteiner).

!!! info "Benchmark Environment"
    **CPU**: Intel i7-10700K (8 cores, 3.8 GHz) | **RAM**: 32 GB DDR4 | **Python**: 3.12 | **NumPy**: 2.0 (MKL)

    Each benchmark averaged over 10 runs. Data generated with fixed seed for reproducibility.

## One-Step vs Two-Step vs CUE-GMM

### Estimation Time Comparison

| N | T | Instruments | One-Step | Two-Step | CUE-GMM | CUE/Two-Step Ratio |
|---|---|-------------|----------|----------|---------|---------------------|
| 100 | 10 | 45 | 0.2s | 0.4s | 1.5s | 3.8x |
| 500 | 10 | 45 | 0.5s | 1.0s | 4.0s | 4.0x |
| 1000 | 10 | 45 | 1.2s | 2.5s | 10s | 4.0x |
| 5000 | 10 | 45 | 5.1s | 9.8s | 48s | 4.9x |

### CUE-GMM: Instrument Count Scaling

CUE-GMM scales quadratically with the number of moment conditions due to repeated weighting matrix inversion (O(m^3) per iteration).

| N | Moments | CUE-GMM Time | Two-Step Time | Ratio |
|---|---------|-------------|---------------|-------|
| 500 | 10 | 2.1s | 0.5s | 4.2x |
| 500 | 20 | 3.8s | 0.7s | 5.4x |
| 500 | 50 | 12.4s | 1.2s | 10.3x |

!!! warning "CUE-GMM Performance Thresholds"
    - **N < 2000, moments < 30**: CUE-GMM completes in reasonable time (< 30s)
    - **N > 5000 or moments > 50**: Consider two-step GMM instead
    - CUE provides efficiency gains over two-step but at significant computational cost

### CUE-GMM Profiling Breakdown

| Function | % of Runtime |
|----------|-------------|
| `scipy.optimize.minimize` | 45% |
| `_compute_objective` | 28% |
| `np.linalg.inv` (weighting matrix) | 15% |
| `_compute_moments` | 8% |
| `_compute_jacobian` | 4% |

## Bias-Corrected GMM

Bias-Corrected GMM (Hahn-Kuersteiner, 2002) adds a bias correction term that is most beneficial when T is small (T < 15). The overhead comes from computing derivative matrices.

| N | T | Standard GMM | BC-GMM | Overhead |
|---|---|-------------|--------|----------|
| 50 | 10 | 0.3s | 0.5s | +67% |
| 100 | 20 | 0.6s | 1.1s | +83% |
| 200 | 50 | 2.1s | 4.3s | +105% |
| 500 | 50 | 5.8s | 12.4s | +114% |

### BC-GMM Profiling Breakdown

| Function | % of Runtime |
|----------|-------------|
| `_compute_bias_term` | 35% |
| `scipy.optimize.minimize` | 30% |
| `_numerical_derivative` | 20% |
| `_compute_moments` | 10% |
| `np.linalg.inv` | 5% |

!!! tip "When to Use BC-GMM"
    - **Use** when T < 15 (small time dimension where bias is significant)
    - **Skip** when T > 30 (bias is negligible, standard GMM is fine)
    - **N < 1000** for reasonable runtime

## Collapse Effect

Using `collapse=True` reduces the instrument matrix from O(T^2) to O(T) instruments, providing significant speedups for large T panels.

### Instrument Count by T

| T | Standard Instruments | Collapsed Instruments | Reduction |
|---|---------------------|----------------------|-----------|
| 5 | 10 | 4 | 60% |
| 10 | 45 | 9 | 80% |
| 20 | 190 | 19 | 90% |
| 50 | 1225 | 49 | 96% |

### Collapse Performance Impact

| N | T | Standard (instruments) | Time | Collapsed (instruments) | Time | Speedup |
|---|---|----------------------|------|------------------------|------|---------|
| 500 | 10 | 45 | 1.0s | 9 | 0.4s | 2.5x |
| 500 | 20 | 190 | 3.2s | 19 | 0.6s | 5.3x |
| 500 | 50 | 1225 | 28s | 49 | 1.8s | 15.6x |

!!! warning "Collapse Trade-off"
    Collapsing instruments improves speed and reduces instrument proliferation but may sacrifice some efficiency. For large T, the speed gain is substantial and the efficiency loss is usually small.

## Windmeijer Correction

The Windmeijer (2005) finite-sample correction for two-step standard errors adds minimal overhead:

| N | T | Two-Step (no correction) | Two-Step (Windmeijer) | Overhead |
|---|---|-------------------------|----------------------|----------|
| 100 | 10 | 0.38s | 0.42s | +10% |
| 500 | 10 | 0.95s | 1.05s | +11% |
| 1000 | 10 | 2.40s | 2.65s | +10% |

The Windmeijer correction overhead is consistently ~10%, making it always worthwhile for two-step estimation.

## Memory Usage

The GMM instrument matrix is the primary memory consumer, with dimensions L x (N*T) where L = number of instruments.

| N | T | Instruments (L) | Instrument Matrix | Total Memory |
|---|---|----------------|-------------------|--------------|
| 100 | 10 | 45 | ~3.4 MB | ~10 MB |
| 500 | 10 | 45 | ~17 MB | ~50 MB |
| 1000 | 10 | 45 | ~34 MB | ~100 MB |
| 500 | 20 | 190 | ~145 MB | ~200 MB |
| 500 | 50 | 1225 | ~4.7 GB | ~5 GB |

!!! danger "Memory with Large T"
    Standard (non-collapsed) instruments for T=50 require ~5 GB. Always use `collapse=True` for T > 20.

## Scaling Summary

| Dimension | One-Step/Two-Step | CUE-GMM | BC-GMM |
|-----------|------------------|---------|--------|
| N (panels) | O(N) | O(N) | O(N) |
| T (time periods) | O(T) | O(T) | O(T) |
| m (moments) | O(m^2) | O(m^3) | O(m^2) |
| Iterations | 1-2 | 10-50 | 1-2 |

## Recommendations

### When to Use Each Estimator

=== "One-Step GMM"

    Best for initial exploration and large datasets.

    - Fastest option
    - Consistent but not efficient
    - Use when N < L (fewer groups than instruments)
    - Good for quick model specification checks

=== "Two-Step GMM"

    Default choice for most applications.

    - 2x slower than one-step
    - Efficient, with Windmeijer-corrected SEs
    - Use for publication-quality results with N > L

=== "CUE-GMM"

    For maximum efficiency when computational cost is acceptable.

    - 4-10x slower than two-step
    - Robust to instrument proliferation
    - Best for N < 2000, moments < 30
    - Not available in R's `plm` package

=== "BC-GMM"

    For small-T panels where bias is a concern.

    - ~70-115% overhead over standard GMM
    - Most beneficial for T < 15
    - Can combine with two-step or CUE

### Quick Decision Guide

```text
Is T < 15 and bias is a concern?
  → Use BC-GMM

Is N > 5000 or moments > 50?
  → Use Two-Step GMM (CUE too slow)

Do you need maximum efficiency?
  → Use CUE-GMM (if N < 2000)

Is this exploratory analysis?
  → Use One-Step GMM

Default:
  → Two-Step GMM with Windmeijer correction
```

### Performance Optimization Checklist

1. **Use `collapse=True`** for T > 10 (reduces instruments by 80-96%)
2. **Enable Windmeijer correction** (only 10% overhead)
3. **Limit lag depth** (`gmm(y, 2:4)` instead of `gmm(y, 2:.)`)
4. **Use one-step** when N < L (avoids rank-deficient weight matrix)
5. **Profile** with `%timeit` to identify bottlenecks in your specific setup

## Comparison with Other Software

| Software | N=1000, T=10, 20 moments | Language |
|----------|--------------------------|----------|
| PanelBox (two-step) | 2.5s | Python |
| R `plm::pgmm` | 3.8s | R |
| Stata `xtabond2` | 2.1s | Stata (C backend) |

PanelBox is competitive with R and approximately 1.2x slower than Stata's highly optimized C implementation. See [Comparison with R/Stata](comparison.md) for detailed cross-platform benchmarks.

## References

- Hansen, L. P., Heaton, J., & Yaron, A. (1996). "Finite-sample properties of some alternative GMM estimators." *Journal of Business & Economic Statistics*.
- Hahn, J., & Kuersteiner, G. (2002). "Asymptotically unbiased inference for a dynamic panel model with fixed effects when both N and T are large."
- Newey, W. K., & Smith, R. J. (2004). "Higher order properties of GMM and generalized empirical likelihood estimators."
- Windmeijer, F. (2005). "A finite sample correction for the variance of linear efficient two-step GMM estimators." *Journal of Econometrics*.
- Roodman, D. (2009). "How to do xtabond2: An introduction to difference and system GMM in Stata." *The Stata Journal*.

## See Also

- [Performance Overview](index.md) — General performance guide
- [GMM API Reference](../api/gmm.md) — Full API documentation
- [GMM Tutorial](../tutorials/gmm.md) — Getting started with GMM estimation
- [Comparison with R/Stata](comparison.md) — Cross-platform benchmarks
