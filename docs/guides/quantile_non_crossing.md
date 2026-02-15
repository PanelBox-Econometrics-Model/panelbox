# Guide to Non-Crossing Quantile Constraints

## Introduction

One fundamental requirement for quantile regression is that conditional quantile functions should not cross - that is, for any $\tau_1 < \tau_2$, we must have $Q_Y(\tau_1|X) \leq Q_Y(\tau_2|X)$ for all $X$. This guide covers methods to detect, prevent, and correct quantile crossing in panel data models.

## The Crossing Problem

### Why Crossing Occurs

Quantile crossing can happen when:
- Each quantile is estimated independently
- Model misspecification
- Finite sample bias
- High-dimensional covariates
- Extreme quantiles (near 0 or 1)

### Implications

- **Logical inconsistency**: Violates definition of quantiles
- **Interpretation issues**: Confusing policy implications
- **Prediction problems**: Invalid density estimates
- **Statistical invalidity**: Affects inference

## Detection Methods

### Visual Inspection

```python
from panelbox.models.quantile import PooledQuantile
from panelbox.models.quantile.monotonicity import QuantileMonotonicity

# Estimate multiple quantiles
tau_grid = np.arange(0.1, 1.0, 0.1)
model = PooledQuantile(data, formula, tau=tau_grid)
results = model.fit()

# Check for crossing
crossing_report = QuantileMonotonicity.detect_crossing(results)
crossing_report.summary()
```

### Formal Testing

```python
# Test at specific points
X_test = np.random.randn(100, p)  # Test points
report = QuantileMonotonicity.detect_crossing(results, X_test)

print(f"Crossing detected: {report.has_crossing}")
print(f"Percentage affected: {report.pct_affected:.2f}%")
```

### Visualization

```python
# Plot quantile curves for sample observations
fig = report.plot_violations(X_test, results)
```

## Prevention Methods

### 1. Location-Scale Models (Recommended)

The Machado-Santos Silva approach guarantees non-crossing by construction:

```python
from panelbox.models.quantile import LocationScale

model = LocationScale(
    data=panel_data,
    formula='y ~ x1 + x2',
    tau=[0.25, 0.5, 0.75],
    distribution='normal'
)
result = model.fit()

# Quantiles cannot cross by construction!
```

**Advantages:**
- Guaranteed non-crossing
- Computationally efficient
- Natural for panel data

**Limitations:**
- Assumes location-scale structure
- Less flexible than nonparametric methods

### 2. Constrained Optimization

Directly impose monotonicity constraints during estimation:

```python
# Estimate with non-crossing constraints
results_constrained = QuantileMonotonicity.constrained_qr(
    X, y,
    tau_list=[0.25, 0.5, 0.75],
    method='trust-constr'
)
```

**Mathematical formulation:**
$$\min \sum_{\tau} \sum_i \rho_\tau(y_i - X_i'\beta(\tau))$$
$$\text{s.t. } X_i'\beta(\tau_1) \leq X_i'\beta(\tau_2) \text{ for all } i, \tau_1 < \tau_2$$

**Advantages:**
- Direct enforcement of constraints
- Flexible specification

**Limitations:**
- Computationally intensive
- May not converge for high dimensions

### 3. Penalized Methods

Add penalty for crossing violations:

```python
# Penalized QR with soft constraints
from panelbox.optimization.quantile import penalized_qr

results = penalized_qr(
    X, y, tau_list,
    penalty='crossing',
    lambda_pen=0.1
)
```

## Correction Methods

### 1. Rearrangement (Post-Processing)

The Chernozhukov et al. (2010) rearrangement method:

```python
# Rearrange after estimation
results_rearranged = QuantileMonotonicity.rearrangement(
    results,
    X=X_sample
)
```

**Algorithm:**
1. Estimate quantiles independently
2. For each observation, sort predicted quantiles
3. Maintain monotonicity while minimizing distance to original

**Advantages:**
- Simple post-processing
- Preserves most of original estimates
- Computationally fast

**Limitations:**
- Ad-hoc correction
- May affect statistical properties

### 2. Isotonic Regression

Apply isotonic regression to coefficient paths:

```python
# Monotonize coefficient paths
coef_matrix = np.array([results[tau].params for tau in tau_list])
coef_monotonic = QuantileMonotonicity.isotonic_regression(
    coef_matrix,
    tau_list
)
```

**Properties:**
- Preserves order while minimizing squared distance
- Optimal in L2 sense
- Fast computation via pool-adjacent-violators algorithm

### 3. Smoothing Methods

Apply kernel smoothing with monotonicity constraints:

```python
from panelbox.methods.quantile import monotone_smooth

# Smooth quantile curves
results_smooth = monotone_smooth(
    results,
    bandwidth=0.05,
    kernel='gaussian'
)
```

## Comparison of Methods

| Method | Non-Crossing | Efficiency | Flexibility | Computation |
|--------|-------------|------------|-------------|-------------|
| Location-Scale | ✅ Guaranteed | High | Medium | Fast |
| Constrained Opt | ✅ Guaranteed | Medium | High | Slow |
| Rearrangement | ✅ Post-hoc | High | High | Fast |
| Isotonic | ✅ Post-hoc | Medium | Medium | Fast |
| Penalized | ⚠️ Approximate | Medium | High | Medium |

## Best Practices

### For Research Papers

1. **Primary Analysis**: Use Location-Scale for main results
2. **Robustness Check**: Compare with rearrangement method
3. **Report Crossing**: Always test and report crossing issues
4. **Sensitivity Analysis**: Try different reference distributions

### For Policy Analysis

1. **Prioritize Interpretability**: Use methods with guaranteed non-crossing
2. **Focus on Key Quantiles**: Median, quartiles rather than extreme quantiles
3. **Bootstrap Inference**: Account for correction in standard errors

### For Prediction

1. **Use Location-Scale**: Best for density estimation
2. **Cross-Validation**: Select method based on out-of-sample performance
3. **Ensemble Methods**: Combine multiple approaches

## Implementation Example

Complete workflow for non-crossing quantile regression:

```python
import numpy as np
import pandas as pd
from panelbox.models.quantile import (
    LocationScale, PooledQuantile, QuantileMonotonicity
)

# 1. Load and prepare data
data = load_panel_data()

# 2. Define quantiles of interest
tau_grid = [0.1, 0.25, 0.5, 0.75, 0.9]

# 3. Method 1: Location-Scale (guaranteed non-crossing)
ls_model = LocationScale(
    data=data,
    formula='y ~ x1 + x2 + x3',
    tau=tau_grid,
    distribution='normal',
    fixed_effects=True
)
ls_result = ls_model.fit()

# 4. Method 2: Traditional QR with rearrangement
qr_model = PooledQuantile(data, formula, tau=tau_grid)
qr_results = qr_model.fit()

# Check for crossing
crossing = QuantileMonotonicity.detect_crossing(qr_results)
if crossing.has_crossing:
    print("Crossing detected, applying rearrangement...")
    qr_results = QuantileMonotonicity.rearrangement(qr_results)

# 5. Compare methods
from panelbox.models.quantile.comparison import compare_methods

comparison = compare_methods(
    data, formula, tau_grid,
    methods=['location_scale', 'qr_rearranged', 'constrained']
)
comparison.plot_comparison()

# 6. Bootstrap inference accounting for correction
ls_result_boot = ls_model.fit(bootstrap=True, n_boot=999)
```

## Troubleshooting

### Issue: Severe crossing in traditional QR

**Solutions:**
1. Check for multicollinearity
2. Reduce number of quantiles
3. Use regularization
4. Switch to Location-Scale

### Issue: Location-Scale too restrictive

**Solutions:**
1. Try different reference distributions
2. Add interaction terms for flexibility
3. Use semi-parametric extensions
4. Consider local polynomial QR

### Issue: Computational constraints

**Solutions:**
1. Use Location-Scale (fastest)
2. Rearrangement (post-processing)
3. Reduce grid of quantiles
4. Parallel computation for bootstrap

## References

- Chernozhukov, V., Fernández‐Val, I., & Galichon, A. (2010). Quantile and probability curves without crossing. *Econometrica*, 78(3), 1093-1125.

- Machado, J. A., & Santos Silva, J. M. C. (2019). Quantiles via moments. *Journal of Econometrics*, 213(1), 145-173.

- Dette, H., & Volgushev, S. (2008). Non-crossing non-parametric estimates of quantile curves. *Journal of the Royal Statistical Society B*, 70(3), 609-627.

- Bondell, H. D., Reich, B. J., & Wang, H. (2010). Noncrossing quantile regression curve estimation. *Biometrika*, 97(4), 825-838.
