# Quantile Regression Quick Start Guide

## Installation & Imports

```python
import panelbox as pb
from panelbox import (
    PooledQuantile,
    BootstrapInference,
    QuantileRegressionDiagnostics,
    quantile_process_plot,
)
import numpy as np
import pandas as pd
```

## Basic Usage

### 1. Single Quantile Regression

```python
# Create sample panel data
n_obs = 500
y = np.random.randn(n_obs)
X = np.random.randn(n_obs, 3)
entity_id = np.repeat(np.arange(50), 10)

# Fit pooled quantile regression at median
model = pb.PooledQuantile(y, X, entity_id=entity_id, quantiles=0.5)
results = model.fit()

# View summary
print(results.summary())

# Access parameters
params = results.params
std_errors = results.std_errors
pvalues = results.pvalues
```

### 2. Multiple Quantiles

```python
# Estimate at 25th, 50th, and 75th percentiles
model = pb.PooledQuantile(y, X, entity_id=entity_id,
                          quantiles=[0.25, 0.5, 0.75])
results = model.fit()

# Parameters shape: (n_vars, 3) - one set per quantile
print(results.params.shape)  # (3, 3)
```

### 3. Different Standard Errors

```python
# Cluster-robust (default for panel data)
results_cluster = model.fit(se_type='cluster')

# Heteroskedasticity-robust
results_robust = model.fit(se_type='robust')

# Classical standard errors
results_classical = model.fit(se_type='nonrobust')
```

## Bootstrap Inference

```python
# Initialize bootstrap
boot = BootstrapInference(model, n_bootstrap=1000, n_jobs=-1)

# Cluster bootstrap (recommended for panel data)
boot_se = boot.cluster_bootstrap(results.params.ravel(), tau=0.5)

# Alternative: Pairs bootstrap
boot_se_pairs = boot.pairs_bootstrap(results.params.ravel(), tau=0.5)

# Wild bootstrap (robust to heteroskedasticity)
boot_se_wild = boot.wild_bootstrap(results.params.ravel(), tau=0.5,
                                   dist='rademacher')
```

## Diagnostics

```python
# Create diagnostics object
diag = QuantileRegressionDiagnostics(model, results.params.ravel(), tau=0.5)

# Pseudo R² (0 to 1 scale)
r2 = diag.pseudo_r2()

# Goodness of fit tests
gof = diag.goodness_of_fit()
print(gof.keys())  # ['pseudo_r2', 'mean_residual', 'median_residual', ...]

# Symmetry test (H0: correct specification)
z_stat, pval = diag.symmetry_test()

# Specification test
chi2_stat, pval = diag.goodness_of_fit_test()

# Print diagnostics summary
print(diag.summary())
```

## Visualization

```python
import matplotlib.pyplot as plt

# Quantile process plot
fig, ax = plt.subplots(figsize=(10, 6))
quantile_process_plot(results.quantiles, results.params,
                     results.std_errors, ax=ax)
plt.tight_layout()
plt.show()

# Residual plot
diag.residuals  # Access residuals
residual_plot(diag.residuals)

# Q-Q plot
qq_plot(diag.residuals)
```

## DataFrame Input (with Formula)

```python
# Using pandas DataFrame
df = pd.DataFrame({
    'y': y,
    'x1': X[:, 0],
    'x2': X[:, 1],
    'x3': X[:, 2],
    'entity_id': entity_id,
    'time_id': np.tile(np.arange(10), 50)
})

# Create model from arrays
X_df = df[['x1', 'x2', 'x3']].values
y_df = df['y'].values

model = pb.PooledQuantile(y_df, X_df,
                         entity_id=df['entity_id'],
                         quantiles=[0.25, 0.5, 0.75])
results = model.fit()
```

## Predictions

```python
# Predictions on training data
pred_train = results.predict()

# Predictions on new data
X_new = np.random.randn(10, 3)
pred_new = results.predict(exog=X_new)

# For specific quantile (multiple quantiles case)
pred_q25 = results.predict(quantile_idx=0)  # 25th percentile
pred_q50 = results.predict(quantile_idx=1)  # 50th percentile
pred_q75 = results.predict(quantile_idx=2)  # 75th percentile
```

## Confidence Intervals

```python
# 95% confidence intervals
lower, upper = results.conf_int(alpha=0.05)

# Access for specific parameter
param_idx = 0
print(f"95% CI: [{lower[param_idx, 0]:.4f}, {upper[param_idx, 0]:.4f}]")
```

## Key Methods

### PooledQuantile
```
.fit(method='interior_point', maxiter=1000, tol=1e-6,
     se_type='cluster', alpha=0.05)
.predict(params=None, exog=None, quantile_idx=0)
```

### PooledQuantileResults
```
.params              # Coefficient estimates
.std_errors          # Standard errors
.tvalues             # t-statistics
.pvalues             # p-values
.quantiles           # Quantile levels
.summary()           # Formatted summary
.predict(exog=None)  # Predictions
.conf_int(alpha=0.05) # Confidence intervals
```

### BootstrapInference
```
.cluster_bootstrap(params, tau=0.5)
.pairs_bootstrap(params, tau=0.5)
.wild_bootstrap(params, tau=0.5, dist='rademacher')
.subsampling_bootstrap(params, tau=0.5, subsample_size=None)
```

### QuantileRegressionDiagnostics
```
.pseudo_r2()                      # Pseudo R² measure
.goodness_of_fit()                # GOF statistics
.symmetry_test()                  # Specification test
.goodness_of_fit_test()           # Chi-square test
.residual_quantiles(quantiles)    # Residual quantiles
.summary()                        # Print diagnostics
```

## Common Workflows

### 1. Comprehensive Analysis

```python
# Fit model with multiple quantiles
model = pb.PooledQuantile(y, X, entity_id=entity_id,
                         quantiles=[0.25, 0.5, 0.75])
results = model.fit()

# Bootstrap standard errors
boot = BootstrapInference(model, n_bootstrap=500)
boot_se = boot.cluster_bootstrap(results.params.ravel(), tau=0.5)

# Diagnostics
diag = QuantileRegressionDiagnostics(model, results.params[:, 1])

# Visualization
quantile_process_plot(results.quantiles, results.params)

# Summary
print(results.summary())
print(diag.summary())
```

### 2. Robustness Checks

```python
# Different SE types
results_cluster = model.fit(se_type='cluster')
results_robust = model.fit(se_type='robust')
results_classical = model.fit(se_type='nonrobust')

# Check consistency
print("SE Comparison:")
print(f"Cluster:  {results_cluster.std_errors}")
print(f"Robust:   {results_robust.std_errors}")
print(f"Classical: {results_classical.std_errors}")
```

### 3. Quantile Process Analysis

```python
# Estimate at many quantiles
quantiles = np.arange(0.05, 1.0, 0.05)
model_process = pb.PooledQuantile(y, X, entity_id=entity_id,
                                 quantiles=quantiles)
results_process = model_process.fit()

# Plot coefficient evolution
fig, ax = plt.subplots()
for i in range(X.shape[1]):
    ax.plot(results_process.quantiles, results_process.params[i, :],
           label=f'X{i}')
ax.legend()
ax.set_xlabel('Quantile')
ax.set_ylabel('Coefficient')
plt.show()
```

## Performance Tips

1. **Use cluster bootstrap for panel data**: Respects clustering structure
2. **Parallel processing**: Set `n_jobs=-1` in BootstrapInference
3. **Multiple quantiles**: Estimate simultaneously, not sequentially
4. **Large samples**: Interior point converges faster than alternatives
5. **Memory**: Use float32 for very large datasets if needed

## Troubleshooting

### Convergence Issues
- Increase `maxiter` in `.fit()`
- Reduce tolerance with `tol=1e-8`
- Check for perfect multicollinearity

### SE Estimation
- Default is cluster-robust, appropriate for panel data
- Use `se_type='nonrobust'` for classical SEs
- Bootstrap provides robust alternative

### NaN Results
- Check for missing values in y or X
- Verify entity_id matches data length
- Ensure quantiles are in (0, 1)

## References

Koenker, R. (2005). Quantile Regression. Cambridge University Press.

Angrist, J. D., & Pischke, J. S. (2009). Mostly Harmless Econometrics.
Princeton University Press.

---

**Last Updated:** February 15, 2026
**PanelBox Version:** 1.0+
