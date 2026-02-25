---
title: "Performance Overview"
description: "General performance guide and benchmarks for PanelBox model families"
---

# Performance Overview

PanelBox is designed for efficient panel data analysis, leveraging NumPy/SciPy vectorized operations, BLAS/LAPACK backends, and smart caching to deliver competitive performance across all model families.

!!! info "Benchmark Environment"
    All benchmarks were conducted on:

    - **CPU**: Intel i7-10700K (8 cores, 3.8 GHz)
    - **RAM**: 32 GB DDR4
    - **Python**: 3.12 | **NumPy**: 2.0 (MKL) | **SciPy**: 1.14

    Results may vary depending on CPU, BLAS/LAPACK implementation (MKL vs OpenBLAS), and data characteristics.

## Performance at a Glance

The table below summarizes typical estimation times for each model family on representative panel sizes.

| Model Family | Typical Size | Estimation Time | Memory | Detail Page |
|-------------|-------------|-----------------|--------|-------------|
| Static (FE/RE/BE) | N=1000, T=20 | < 1s | Low (< 50 MB) | |
| GMM (Arellano-Bond, System) | N=500, T=10 | 1-5s | Low (< 100 MB) | [GMM Benchmarks](gmm.md) |
| Spatial (SAR/SEM/SDM) | N=500, T=20 | 5-30s | Medium (200-300 MB) | [Spatial Benchmarks](spatial.md) |
| SFA (Stochastic Frontier) | N=500, T=10 | 2-10s | Low (< 100 MB) | |
| Quantile (FE/Canay/Dynamic) | N=1000, T=20 | 5-30s | Low (< 100 MB) | |
| Panel VAR | N=100, T=30 | 2-15s | Medium (50-300 MB) | |
| Discrete (Logit/Probit) | N=1000, T=20 | 2-10s | Low (< 100 MB) | |
| Count (Poisson/NB/PPML) | N=1000, T=20 | 1-5s | Low (< 100 MB) | |
| Heckman Selection | N=500, T=10 | 0.4-50s | Medium (< 300 MB) | |

!!! tip "Cross-Platform Comparison"
    See [Comparison with R/Stata](comparison.md) for benchmarks against `plm`, `xtabond2`, `splm`, and other established tools.

## Runtime Categories

### Fast (< 1 second)

- Static models (Pooled OLS, Fixed Effects, Random Effects, Between)
- First Difference estimation
- Pedroni and Kao cointegration tests
- Hadri unit root test
- LM spatial tests (LM-Lag, LM-Error)
- Granger causality Wald test
- Two-step Heckman (small N)

### Medium (1-10 seconds)

- GMM one-step and two-step (N < 1000)
- Bias-Corrected GMM (N < 500)
- Spatial models ML (N < 500)
- Panel VAR OLS estimation
- Discrete choice models (Logit, Probit, Ordered)
- PPML with good starting values
- Breitung unit root test

### Slow (10-60 seconds)

- CUE-GMM (N > 500)
- Spatial models ML (N = 500-1000)
- Westerlund test with 1000+ bootstrap reps
- MLE Heckman (N < 500)
- Panel VAR GMM (N > 200)
- Bootstrap IRFs (500+ replications)

### Very Slow (> 60 seconds)

- MLE Heckman (N > 500)
- Spatial models (N > 1000)
- Multinomial FE Logit (J > 3, T > 10)
- Westerlund test with 5000+ bootstrap reps
- Panel VAR GMM (N > 500)

## Computational Complexity

| Method | N | T | Parameters | Notes |
|--------|---|---|-----------|-------|
| Static models (FE/RE) | O(N) | O(T) | O(p^2) | Direct matrix operations |
| GMM (two-step) | O(N) | O(T) | O(m^3) | m = number of moment conditions |
| CUE-GMM | O(N) | O(T) | O(m^3) | Iterative optimization, 4-10x slower |
| BC-GMM | O(N) | O(T) | O(p^2) | Bias correction adds ~70-115% overhead |
| Spatial ML | O(N^3) | O(T) | O(p^2) | Log-determinant is the bottleneck |
| Spatial (sparse) | O(N^2) | O(T) | O(p^2) | Sparse W with < 10% fill |
| Heckman two-step | O(N) | O(T) | O(p^2) | Probit + OLS, fast |
| Heckman MLE | O(N * q^k) | O(T) | O(p^2) | q = quadrature points, k = random effects |
| Panel VAR OLS | O(N) | O(T) | O(K^2) | K = number of endogenous variables |
| Panel VAR GMM | O(N^2) | O(T^1.5) | O(K^3) | Moment matrix construction |
| Multinomial FE | O(N * J^T) | — | O(p) | J = number of choices (exponential in T!) |
| PPML | O(N*T) | — | O(p^2) | IRLS convergence |
| Westerlund | O(N*T*B) | — | — | B = bootstrap replications |

## Memory Usage

Approximate peak memory for common configurations:

| Method | Memory Formula | N=1000 Example |
|--------|----------------|----------------|
| Static models | 8 * N * T * p bytes | ~5 MB |
| GMM | 8 * N * T * (p + m) bytes | ~100 MB |
| Spatial ML | 8 * N^2 (W matrix) + data | 687 MB |
| Heckman MLE | 8 * N * T * q * k bytes | ~80 MB |
| Panel VAR GMM | 8 * N * T * K^2 * p bytes | ~60 MB |
| Westerlund bootstrap | 8 * N * T * B bytes | ~800 MB (B=1000) |

!!! warning "Memory-Intensive Scenarios"
    - **Spatial models** with dense W: memory is O(N^2). Use sparse matrices for N > 500.
    - **Westerlund bootstrap**: memory scales linearly with B. Keep B <= 1000 on 16 GB systems.
    - **Multinomial FE Logit**: memory is exponential in T. Avoid for T > 10 or J > 4.

## Panel VAR Performance

Panel VAR models have distinct performance characteristics depending on estimation method and analysis type.

### Estimation

| N | T | K | p | OLS | GMM (collapsed) | GMM (standard) |
|---|---|---|---|-----|-----------------|----------------|
| 50 | 20 | 3 | 2 | 0.10s | 0.8s | 1.2s |
| 100 | 20 | 3 | 2 | 0.12s | 2.4s | 3.5s |
| 200 | 20 | 3 | 2 | 0.20s | 8.5s | 12s |
| 500 | 20 | 3 | 2 | 0.48s | 52s | 75s |
| 1000 | 20 | 3 | 2 | 0.95s | 210s | 310s |

Where K = number of endogenous variables, p = lag order.

### IRFs and Bootstrap

| Analysis | N=100, K=3, p=2 | Time |
|----------|-----------------|------|
| IRF (analytical CI) | 10 periods | 0.3s |
| IRF (bootstrap, 200 reps) | 10 periods | 12s |
| IRF (bootstrap, 500 reps) | 10 periods | 28s |
| IRF (bootstrap, 1000 reps) | 10 periods | 56s |
| FEVD | 10 periods | 0.25s |
| Granger causality (Wald) | — | 0.05s |
| Dumitrescu-Hurlin | — | 0.8s |
| Dumitrescu-Hurlin (bootstrap) | 500 reps | 15s |

!!! tip "VAR Performance Tips"
    - Use **collapsed instruments** for GMM: 30-40% faster, 20-30% less memory.
    - Use **analytical CIs** for IRFs during exploration (100-200x faster than bootstrap).
    - Keep **K <= 5-7** for tractable IRF analysis (K^3 scaling).
    - Use **OLS** for initial exploration, then switch to GMM for final results.

## General Performance Tips

### 1. Choose the Right Estimator

- **Speed priority**: use two-step GMM, two-step Heckman, OLS VAR
- **Efficiency priority**: use CUE-GMM, MLE Heckman, GMM VAR
- **Default methods are usually well-balanced** for typical panel sizes

### 2. Reduce Instrument Count in GMM

```python
# Collapse instruments for faster estimation
model = DifferenceGMM(data, formula="y ~ x1 | gmm(y, 2:.) | iv(x2)",
                      collapse=True)
```

Collapsed instruments reduce the instrument matrix size from O(T^2) to O(T), with substantial speedups for large T.

### 3. Use Sparse Weight Matrices for Spatial Models

```python
import scipy.sparse as sp

# Convert dense W to sparse (if sparsity < 10%)
W_sparse = sp.csr_matrix(W)
model = SpatialLag(formula, data, entity_col, time_col, W_sparse)
```

Sparse operations provide 2-3x speedup when W has less than 10% non-zero entries.

### 4. Reduce Bootstrap Replications

```python
# Quick exploration: 200 reps
result_quick = bootstrap(n_boot=200)

# Publication quality: 1000+ reps
result_final = bootstrap(n_boot=1000)
```

### 5. Use Good Starting Values

```python
# Two-step estimates as starting values for MLE
result_2step = model.fit(method='two-step')
result_mle = model.fit(method='mle', start_params=result_2step.params)
```

### 6. Profile Your Code

```python
# In Jupyter notebooks
%timeit model.fit()

# For detailed profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
result = model.fit()
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## Implemented Optimizations

PanelBox includes several performance optimizations:

- **Vectorized operations** throughout (NumPy/SciPy)
- **Efficient linear algebra** via BLAS/LAPACK backends
- **Smart caching** of intermediate results (eigenvalues, decompositions)
- **Sparse matrix support** for spatial weight matrices
- **Automatic warnings** for slow configurations

!!! note "Numba Acceleration"
    PanelBox uses Numba JIT compilation for selected hot loops when Numba is installed. Install it for automatic acceleration:

    ```bash
    pip install numba
    ```

    JIT-compiled functions include inner loops in spatial permutation tests and selected likelihood evaluations.

## Detailed Benchmark Pages

- **[GMM Benchmarks](gmm.md)** — One-step, two-step, CUE, Bias-Corrected, collapse effects
- **[Spatial Benchmarks](spatial.md)** — SAR/SEM/SDM estimation, sparse vs dense, diagnostics
- **Heckman Benchmarks** — Two-step vs MLE, quadrature point impact (see table above)
- **[Comparison with R/Stata](comparison.md)** — Cross-platform performance and feature comparison

## See Also

- [GMM Tutorial](../tutorials/gmm.md) — Practical guide to GMM estimation
- [Spatial Tutorial](../tutorials/spatial.md) — Getting started with spatial models
- [Standard Errors Guide](../tutorials/standard-errors.md) — Robust inference options
