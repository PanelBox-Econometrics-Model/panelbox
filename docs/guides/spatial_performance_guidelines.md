# Spatial Panel Models - Performance Guidelines

## Overview

This guide provides performance benchmarks and optimization strategies for spatial panel models in PanelBox. Understanding these guidelines will help you choose appropriate methods for your data size and optimize computation time.

## Performance Benchmarks

### Model Estimation Times

Based on extensive benchmarking, here are typical estimation times for different panel sizes:

| Panel Size (N × T) | SAR-FE | SEM-FE | SDM-FE | Memory Usage |
|-------------------|---------|---------|---------|--------------|
| 100 × 10         | < 1s    | < 1s    | < 2s    | < 100 MB     |
| 500 × 10         | 5-10s   | 5-10s   | 10-15s  | 200-300 MB   |
| 1000 × 10        | 20-30s  | 20-30s  | 30-45s  | 500-800 MB   |
| 5000 × 5         | 2-5min  | 2-5min  | 5-10min | 2-4 GB       |
| 10000 × 5        | 10-20min| 10-20min| 20-40min| 8-16 GB      |

**Note**: Times are for a standard desktop (8-core CPU, 16GB RAM). Your results may vary based on:
- Sparsity of the weight matrix W
- Number of covariates
- Convergence tolerance
- Hardware specifications

### Scaling Behavior

Spatial models exhibit different scaling patterns:

- **Direct methods**: O(N³) time complexity
  - Suitable for N < 1000
  - Exact log-determinant calculation
  - Most accurate results

- **Sparse methods**: O(N²) to O(N² log N)
  - Effective when W is sparse (< 10% non-zero)
  - Significant speedup for N > 500
  - Minimal accuracy loss

- **Approximation methods**: O(N log N) to O(N²)
  - Necessary for N > 5000
  - Chebyshev polynomial approximation
  - Trade-off between speed and accuracy

## Optimization Strategies

### 1. Choosing the Right Weight Matrix

**Sparse vs Dense**

```python
from panelbox.core.spatial_weights import SpatialWeights

# For sparse connectivity (recommended for large N)
W = SpatialWeights.from_contiguity(gdf, method='queen')
W_sparse = W.to_sparse()  # Convert to sparse format

# Check sparsity
sparsity = W.nnz / W.size
print(f"Sparsity: {sparsity:.2%}")

# If sparsity < 10%, sparse operations will be faster
if sparsity < 0.1:
    model = SpatialLag(..., W=W_sparse)
```

**K-Nearest Neighbors for Large N**

```python
# Limit connectivity for large panels
W = SpatialWeights.from_knn(coords, k=10)  # Only 10 nearest neighbors
# Results in O(kN) = O(N) non-zero elements
```

### 2. Enabling Performance Optimizations

**Eigenvalue Caching**

```python
# Eigenvalues are automatically cached
model = SpatialLag(...)
result1 = model.fit()  # Computes eigenvalues
result2 = model.fit(starting_values=result1.params)  # Reuses cached eigenvalues
```

**Parallel Processing**

```python
# Enable parallel permutation tests
from panelbox.validation.spatial import MoranIPanelTest

moran_test = MoranIPanelTest(
    residuals, W, entity_ids, time_ids,
    n_permutations=999,
    n_jobs=-1  # Use all CPU cores
)
```

**Numba JIT Compilation**

```python
# Install numba for automatic JIT compilation
# pip install numba

# JIT compilation is automatic when available
import panelbox  # Will use numba if installed
```

### 3. Large N Strategies (N > 1000)

**Use Chebyshev Approximation**

```python
# For N > 5000, automatic Chebyshev approximation
model = SpatialLag(..., approximation='chebyshev', chebyshev_order=50)
```

**Subset Your Data**

```python
# If full estimation is too slow, consider:

# 1. Random sampling for initial estimates
sample_idx = np.random.choice(N, size=1000, replace=False)
model_sample = SpatialLag(data_sample, W_sample, ...)
initial_params = model_sample.fit().params

# 2. Use initial params for full model
model_full = SpatialLag(data, W, ...)
result = model_full.fit(starting_values=initial_params)
```

**Block Estimation**

```python
# For very large panels, estimate by blocks
from panelbox.optimization import block_estimation

# Split by geographic regions
regions = data.groupby('region')
results = {}

for region_id, region_data in regions:
    W_region = W[region_mask][:, region_mask]
    model = SpatialLag(region_data, W_region, ...)
    results[region_id] = model.fit()
```

### 4. Memory Management

**Monitor Memory Usage**

```python
import tracemalloc

tracemalloc.start()
model = SpatialLag(...)
result = model.fit()
current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

**Reduce Memory Footprint**

```python
# 1. Use float32 instead of float64
W = W.astype(np.float32)
data[numeric_cols] = data[numeric_cols].astype(np.float32)

# 2. Delete intermediate results
del model._cache  # Clear cached computations

# 3. Use out-of-core computation for very large data
# Consider using dask for distributed computation
```

## Diagnostic Performance

### Moran's I Test

| Data Size | Global Moran | Local Moran (LISA) | Permutation (999) |
|-----------|--------------|-------------------|-------------------|
| N=100     | < 0.1s      | < 0.5s            | < 2s             |
| N=500     | < 0.5s      | < 2s              | < 10s            |
| N=1000    | < 1s        | < 5s              | < 30s            |
| N=5000    | < 10s       | < 30s             | 2-5min           |

### LM Tests

LM tests are generally fast as they only require OLS residuals:

```python
# Fast diagnostic workflow
ols = FixedEffects(formula, data, entity_col, time_col)
ols_result = ols.fit()  # Fast

# LM tests use only OLS residuals
lm_results = run_lm_tests(ols_result, W)  # < 1s for N=1000
```

## Best Practices

### 1. Workflow for Large Panels

```python
# Step 1: Start with diagnostics on a sample
sample = data.sample(min(1000, len(data)))
W_sample = create_sample_W(sample)

# Step 2: Run quick diagnostics
moran_quick = MoranIPanelTest(sample, W_sample).run()

if moran_quick.pvalue < 0.05:
    # Step 3: Estimate on sample to get starting values
    model_sample = SpatialLag(sample, W_sample, ...)
    initial_params = model_sample.fit().params

    # Step 4: Full estimation with starting values
    model_full = SpatialLag(data, W, ...)
    result = model_full.fit(starting_values=initial_params)
```

### 2. Choosing Estimation Method

```python
# For different N sizes
if N < 500:
    method = 'ml'  # Full maximum likelihood
elif N < 2000:
    method = 'ml'
    W = W.to_sparse()  # Use sparse operations
elif N < 5000:
    method = 'gm'  # Generalized moments (faster)
else:
    method = 'chebyshev'  # Approximation methods
```

### 3. Parallel Bootstrap

```python
from panelbox.optimization import ParallelBootstrap

# Use all CPU cores for bootstrap
bootstrap = ParallelBootstrap(
    estimator=spatial_estimator,
    n_bootstrap=1000,
    n_jobs=-1,  # All cores
    bootstrap_type='pairs'
)

# Much faster than sequential
boot_results = bootstrap.run(data, W, formula, entity_col, time_col)
```

## Hardware Recommendations

### Minimum Requirements

- **Small panels (N < 500)**: 4GB RAM, dual-core CPU
- **Medium panels (N < 2000)**: 8GB RAM, quad-core CPU
- **Large panels (N < 5000)**: 16GB RAM, 6+ core CPU
- **Very large panels (N > 5000)**: 32GB+ RAM, 8+ core CPU

### Optimal Setup

For production use with large spatial panels:

- **CPU**: 8+ cores (for parallel processing)
- **RAM**: 32GB or more
- **Storage**: SSD (faster I/O for large matrices)
- **GPU**: Optional (future versions may support GPU acceleration)

## Troubleshooting Performance Issues

### Problem: Estimation takes too long

**Solutions**:
1. Check weight matrix sparsity
2. Enable parallel processing
3. Use approximation methods
4. Provide starting values
5. Reduce convergence tolerance

### Problem: Out of memory errors

**Solutions**:
1. Use sparse matrices
2. Reduce data precision (float32)
3. Process in blocks
4. Increase system RAM
5. Use out-of-core methods

### Problem: No convergence

**Solutions**:
1. Check data scaling
2. Verify W is properly normalized
3. Provide better starting values
4. Increase max iterations
5. Try different optimization method

## Performance Monitoring

```python
# Complete performance monitoring setup
import time
import tracemalloc
from panelbox.utils import PerformanceMonitor

# Initialize monitor
monitor = PerformanceMonitor()

# Track estimation
with monitor.track("model_estimation"):
    model = SpatialLag(...)
    result = model.fit()

# Get report
report = monitor.get_report()
print(f"Total time: {report['model_estimation']['time']:.2f}s")
print(f"Peak memory: {report['model_estimation']['memory_mb']:.1f} MB")
```

## Future Optimizations

Planned performance improvements for future releases:

- **GPU acceleration** for matrix operations (CUDA/OpenCL)
- **Distributed computing** support (Dask/Ray)
- **Improved approximations** for very large N
- **Automatic optimization** selection based on data size
- **Incremental estimation** for streaming data

## References

1. LeSage, J. and Pace, R.K. (2009). *Introduction to Spatial Econometrics*. Chapman & Hall/CRC.
2. Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
3. Bivand, R. and Piras, G. (2015). Comparing implementations of estimation methods for spatial econometrics. *Journal of Statistical Software*, 63(18).
