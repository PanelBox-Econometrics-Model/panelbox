# PanelBox Performance Benchmarks

## Overview

This directory contains detailed performance benchmarks for all advanced methods implemented in PanelBox. Benchmarks were conducted on a standard development machine and provide guidance on expected runtimes and computational complexity.

## Available Benchmarks

### 1. [GMM Performance](gmm_performance.md)
- **CUE-GMM**: Continuous Updated Estimator
- **Bias-Corrected GMM**: Hahn-Kuersteiner estimator
- **Key finding**: CUE-GMM is 4-10x slower than two-step but scales well
- **Recommendation**: Use CUE for N < 2000, moments < 30

### 2. [Heckman Performance](heckman_performance.md)
- **Two-Step**: Fast, 0.4-6s for typical panels
- **MLE**: 5-50x slower, depends on quadrature points
- **Key finding**: MLE impractical for N > 500
- **Recommendation**: Use two-step for large N, MLE for efficiency when N < 200

### 3. Cointegration Tests Performance
- **Westerlund**: Bootstrap is dominant cost (linear in # reps)
- **Pedroni**: Fast, <1s for typical panels
- **Kao**: Very fast, <0.5s
- **Recommendation**: Use tabulated critical values for Westerlund if bootstrap too slow

### 4. Unit Root Tests Performance
- **Hadri**: Fast, <0.5s
- **Breitung**: Medium, 1-3s depending on N
- **Recommendation**: All tests are fast enough for routine use

### 5. Multinomial Logit Performance
- **Fixed Effects**: Exponential in J (# choices) and T
- **Random Effects**: Linear in J and T
- **Key finding**: FE impractical for J > 4 or T > 10
- **Recommendation**: Use RE or conditional logit for large J

### 6. PPML Performance
- **Fixed Effects**: Similar to Poisson FE
- **Robust**: 1-5s for typical gravity models
- **Key finding**: Convergence depends on good starting values
- **Recommendation**: Use Poisson estimates as starting values

## General Performance Guidelines

### Computational Complexity

| Method | Complexity in N | Complexity in T | Complexity in p |
|--------|-----------------|-----------------|-----------------|
| CUE-GMM | O(N) | O(T) | O(m³) |
| BC-GMM | O(N) | O(T) | O(p²) |
| Heckman Two-Step | O(N) | O(T) | O(p²) |
| Heckman MLE | O(N·q^k) | O(T) | O(p²) |
| Westerlund | O(N·T·B) | - | - |
| Pedroni | O(N·T) | - | O(p) |
| Multinomial FE | O(N·J^T) | - | O(p) |
| PPML | O(N·T) | - | O(p²) |

Where:
- N = # panels
- T = # time periods
- p = # parameters
- m = # moments (GMM)
- q = # quadrature points (MLE)
- k = # random effects
- B = # bootstrap replications
- J = # choices (multinomial)

### Runtime Categories

**Fast** (< 1 second):
- Pedroni, Kao tests
- Hadri test
- PPML with good starting values
- Two-step Heckman (small N)

**Medium** (1-10 seconds):
- CUE-GMM (N < 1000)
- BC-GMM (N < 500)
- Two-step Heckman (large N)
- Breitung test

**Slow** (10-60 seconds):
- CUE-GMM (N > 1000)
- Westerlund with 1000+ bootstrap reps
- MLE Heckman (N < 500, q=10)

**Very Slow** (> 60 seconds):
- MLE Heckman (N > 500 or q > 15)
- Multinomial FE (J > 3, T > 10)
- Westerlund with 5000+ bootstrap reps

### Memory Usage

Approximate peak memory usage:

| Method | Memory Formula | N=1000 Example |
|--------|----------------|----------------|
| CUE-GMM | 8·N·T·(p+m) bytes | ~100 MB |
| Heckman MLE | 8·N·T·q·k bytes | ~80 MB |
| Westerlund | 8·N·T·B bytes | ~800 MB (B=1000) |
| Multinomial FE | 8·N·T·J^T bytes | Exponential! |

**Note**: For very large datasets (N > 100,000), consider:
1. Chunking data
2. Using out-of-core computation
3. Distributed computing

## Performance Tips

### 1. Choose the Right Estimator

- **Speed priority**: Use faster alternatives (two-step, tabulated values)
- **Efficiency priority**: Use slower but efficient methods (CUE, MLE)
- **Balance**: Default methods are usually well-chosen

### 2. Use Good Starting Values

Many iterative methods benefit from good starting values:

```python
# Example: MLE Heckman
result_2step = model.fit(method='two-step')
result_mle = model.fit(method='mle', start_params=result_2step.params)
```

### 3. Reduce Bootstrap Replications

For exploratory analysis:
- Use 200-500 bootstrap reps instead of 1000+
- Use tabulated critical values when available

### 4. Leverage Parallelization

Some methods support parallel computation:

```python
# Westerlund test with parallel bootstrap
result = westerlund_test(..., n_jobs=-1)  # Use all cores
```

### 5. Monitor Convergence

Check convergence warnings:
- Increase `maxiter` if needed
- Tighten or loosen `tol` as appropriate
- Use `verbose=True` to monitor progress

## Profiling Your Own Code

To identify bottlenecks in your specific use case:

```python
import cProfile
import pstats

# Profile your code
profiler = cProfile.Profile()
profiler.enable()

# Your model fitting code here
result = model.fit(...)

profiler.disable()

# Print results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

## Optimization Recommendations

### Implemented Optimizations

- ✓ Vectorized operations throughout
- ✓ Efficient linear algebra (BLAS/LAPACK via NumPy)
- ✓ Smart caching of intermediate results
- ✓ Warnings for slow operations

### Future Optimizations

- [ ] Numba JIT for hot loops
- [ ] Parallel bootstrap (joblib)
- [ ] GPU acceleration for very large N
- [ ] Analytical derivatives where possible
- [ ] Sparse matrix operations

## Benchmark Methodology

All benchmarks were conducted on:
- **CPU**: Intel i7-10700K (8 cores, 3.8 GHz)
- **RAM**: 32 GB DDR4
- **Python**: 3.12.3
- **NumPy**: 2.0.2 (with MKL)
- **SciPy**: 1.14.1

Results may vary based on:
- CPU speed and # cores
- BLAS/LAPACK implementation (MKL vs OpenBLAS)
- Data characteristics (sparsity, correlation structure)

## Running Benchmarks

To reproduce benchmarks:

```bash
cd /home/guhaase/projetos/panelbox

# Run all benchmarks
python tests/performance/benchmark_all.py

# Run specific method
python tests/performance/benchmark_all.py --method gmm

# Run with profiling
python tests/performance/profile_gmm.py
python tests/performance/profile_heckman.py
python tests/performance/profile_cointegration.py
```

## Questions?

If you have performance questions or encounter unexpectedly slow runtimes:

1. Check if your data size exceeds recommended limits (see warnings)
2. Verify you're using appropriate estimator for your use case
3. Try reducing bootstrap reps or quadrature points for testing
4. Profile your specific use case to identify bottlenecks
5. Open an issue on GitHub with benchmark details

## References

- Hansen et al. (1996) - GMM finite sample properties
- Wooldridge (2010) - Econometric Analysis of Cross Section and Panel Data
- Cameron & Trivedi (2005) - Microeconometrics: Methods and Applications

---

*Last updated: 2026-02-15*
