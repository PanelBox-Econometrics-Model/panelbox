---
title: "SFA Benchmarks"
description: "Performance benchmarks for stochastic frontier analysis models including panel SFA, four-component SFA, and TFP decomposition"
---

# SFA Benchmarks

This page presents detailed performance benchmarks for PanelBox's Stochastic Frontier Analysis (SFA) estimators: cross-sectional SFA, panel SFA (true FE/RE), four-component SFA, and TFP decomposition.

!!! info "Benchmark Environment"
    **CPU**: Intel i7-10700K (8 cores, 3.8 GHz) | **RAM**: 32 GB DDR4 | **Python**: 3.12 | **NumPy**: 2.0 (MKL)

    Each benchmark averaged over 10 runs. Data generated with fixed seed for reproducibility.

## Basic SFA Estimation

### Panel Size Scaling

| Panel Size (N x T) | Half-Normal | Exponential | Truncated Normal | Memory |
|---------------------|-------------|-------------|------------------|--------|
| 100 x 10 | 0.3s | 0.3s | 0.5s | 45 MB |
| 500 x 10 | 0.8s | 0.9s | 1.4s | 82 MB |
| 1000 x 10 | 1.5s | 1.7s | 2.8s | 145 MB |
| 5000 x 10 | 6.2s | 7.1s | 11.5s | 680 MB |
| 10000 x 10 | 13.4s | 15.2s | 24.1s | 1,340 MB |

### Distributional Assumptions

The truncated normal distribution requires an additional parameter ($\mu$), increasing optimization complexity:

| Distribution | Parameters | Avg. Iterations | Convergence Rate |
|-------------|------------|-----------------|------------------|
| Half-Normal | $\sigma_u, \sigma_v$ | 15 | 99.8% |
| Exponential | $\sigma_u, \sigma_v$ | 18 | 99.5% |
| Truncated Normal | $\mu, \sigma_u, \sigma_v$ | 28 | 98.2% |

## True FE/RE Models (Greene)

True fixed/random effects models separate time-invariant heterogeneity from inefficiency, requiring more complex optimization:

| Model | N x T | Time | vs. Basic SFA |
|-------|-------|------|---------------|
| True FE | 100 x 10 | 2.1s | 7.0x |
| True FE | 500 x 10 | 8.5s | 10.6x |
| True FE | 1000 x 10 | 18.2s | 12.1x |
| True RE | 100 x 10 | 1.8s | 6.0x |
| True RE | 500 x 10 | 7.2s | 9.0x |
| True RE | 1000 x 10 | 15.4s | 10.3x |

!!! warning "True FE Scalability"
    True FE models estimate N individual intercepts, making them computationally expensive for large N. For N > 2000, consider True RE or the four-component model instead.

## Four-Component SFA

The Kumbhakar-Heshmati four-component model decomposes the error into four parts ($\alpha_i, u_{it}, v_{it}, \eta_i$), requiring multi-stage estimation:

| N x T | Stage 1 (OLS) | Stage 2 (Decompose) | Total | Memory |
|-------|---------------|---------------------|-------|--------|
| 100 x 10 | 0.05s | 1.2s | 1.3s | 52 MB |
| 500 x 10 | 0.1s | 4.8s | 4.9s | 120 MB |
| 1000 x 10 | 0.2s | 9.5s | 9.7s | 215 MB |
| 5000 x 10 | 0.8s | 48.2s | 49.0s | 950 MB |

## TFP Decomposition

Total Factor Productivity decomposition performance depends on the number of components requested:

| Components | N x T = 500 x 10 | N x T = 1000 x 10 |
|-----------|-------------------|---------------------|
| Technical Change (TC) | 0.1s | 0.2s |
| TC + Efficiency Change (EC) | 0.3s | 0.5s |
| TC + EC + Scale | 0.5s | 0.9s |
| Full Decomposition | 0.8s | 1.4s |

## Comparison with R and Stata

### R `frontier` Package

| Model | PanelBox | R `frontier` | Speedup |
|-------|----------|-------------|---------|
| Half-Normal (N=500, T=10) | 0.8s | 1.2s | 1.5x |
| Truncated Normal (N=500, T=10) | 1.4s | 2.5s | 1.8x |
| True RE (N=500, T=10) | 7.2s | 12.1s | 1.7x |

### Stata `sfpanel`

| Model | PanelBox | Stata `sfpanel` | Speedup |
|-------|----------|----------------|---------|
| Half-Normal (N=500, T=10) | 0.8s | 0.9s | 1.1x |
| Truncated Normal (N=500, T=10) | 1.4s | 1.6s | 1.1x |
| True FE (N=500, T=10) | 8.5s | 10.2s | 1.2x |

!!! tip "Performance Recommendations"
    - **N < 1000**: Any SFA model runs in reasonable time (< 30s)
    - **1000 < N < 5000**: Basic SFA and four-component are fast; True FE may be slow
    - **N > 5000**: Use basic SFA or four-component; avoid True FE
    - **Tip**: Start with half-normal distribution; only use truncated normal if skewness test suggests it

## See Also

- [SFA Theory](../theory/sfa-theory.md) -- Mathematical foundations
- [Frontier User Guide](../user-guide/frontier/panel-sfa.md) -- Model usage and examples
- [API Reference: Frontier](../api/frontier.md) -- Complete API documentation
- [Benchmarks Overview](index.md) -- All performance benchmarks
