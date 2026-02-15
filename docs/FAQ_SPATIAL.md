# FAQ - Spatial Econometrics with PanelBox

## General Questions

### Q: When should I use spatial models instead of standard panel models?

**A:** You should consider spatial models when:
- Your data has a geographic or network structure
- Moran's I test shows significant spatial autocorrelation (p < 0.05)
- Economic theory suggests spillovers or interactions between units
- You observe clustering patterns in your residuals
- Cross-sectional dependence tests indicate spatial patterns

Run diagnostics first:
```python
diagnostics = experiment.run_spatial_diagnostics(W)
if diagnostics['morans_i']['pvalue'] < 0.05:
    print("Spatial models needed!")
```

### Q: How do I choose between SAR, SEM, and SDM models?

**A:** Use the Lagrange Multiplier (LM) tests to guide model selection:

| LM Test Results | Recommended Model | Interpretation |
|----------------|-------------------|----------------|
| Only LM-lag significant | SAR | Spatial dependence in dependent variable |
| Only LM-error significant | SEM | Spatial dependence in error term |
| Both LM tests significant | Check robust LM tests | Need more specific tests |
| Robust LM-lag significant | SAR | Spatial lag dominates |
| Robust LM-error significant | SEM | Spatial error dominates |
| Both robust tests significant | SDM or GNS | Both types of spatial dependence |

PanelBox automates this:
```python
diagnostics = experiment.run_spatial_diagnostics(W)
print(f"Recommended: {diagnostics['recommendation']}")
```

### Q: What is the spatial lag parameter ρ (rho) and how do I interpret it?

**A:** The spatial lag parameter ρ captures the strength of spatial spillovers:
- **ρ > 0**: Positive spatial dependence (clustering of similar values)
- **ρ < 0**: Negative spatial dependence (dissimilar neighbors)
- **ρ = 0**: No spatial dependence
- **|ρ| < 1**: Required for model stability

Interpretation example:
- ρ = 0.3 means a 10% increase in neighbors' y leads to a 3% increase in own y
- The spatial multiplier is 1/(1-ρ), so ρ = 0.3 gives a multiplier of 1.43

### Q: What are direct vs indirect effects in spatial models?

**A:** In spatial models, especially SDM:

- **Direct Effect**: Impact of a change in X_i on y_i (own-unit effect)
- **Indirect Effect**: Impact of a change in X_i on y_j for j≠i (spillover to neighbors)
- **Total Effect**: Direct + Indirect effects

Example interpretation:
```python
effects = sdm_result.effects_decomposition()
# If direct effect of education = 0.5 and indirect = 0.2:
# - 1% increase in own education → 0.5% increase in own outcome
# - 1% increase in own education → 0.2% increase in neighbors' outcomes
# - Total impact = 0.7%
```

### Q: How do I interpret spatial error parameter λ (lambda)?

**A:** The spatial error parameter λ in SEM models captures spatial correlation in unobserved factors:
- **λ > 0**: Positive spatial correlation in errors (omitted variables with spatial pattern)
- **λ < 0**: Negative spatial correlation in errors
- **|λ| < 1**: Required for model stability

Unlike ρ, λ doesn't have a direct economic interpretation but indicates:
- Presence of spatially correlated omitted variables
- Need to account for spatial patterns in error structure
- Potential for biased standard errors if ignored

---

## Spatial Weight Matrix Questions

### Q: How do I create a spatial weight matrix?

**A:** PanelBox supports multiple methods:

1. **Contiguity-based** (shared borders):
```python
import geopandas as gpd
gdf = gpd.read_file('shapefile.shp')
W = SpatialWeights.from_contiguity(gdf, criterion='queen')
```

2. **Distance-based** (within threshold):
```python
coords = data[['longitude', 'latitude']].values
W = SpatialWeights.from_distance(coords, threshold=500)  # 500km
```

3. **k-Nearest Neighbors**:
```python
W = SpatialWeights.from_knn(coords, k=5)  # 5 nearest neighbors
```

4. **Custom matrix**:
```python
import numpy as np
W_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
W = SpatialWeights(W_matrix)
```

### Q: Should I row-standardize my weight matrix?

**A:** Yes, in most cases you should row-standardize:

```python
W_std = W.standardize('row')  # Each row sums to 1
```

Benefits of row-standardization:
- Facilitates interpretation (weighted average of neighbors)
- Ensures ρ is bounded between -1 and 1
- Makes results comparable across different W specifications
- Required for most spatial models

When NOT to row-standardize:
- When using inverse distance weights with specific interpretation
- Some network models with meaningful edge weights
- When preserving absolute distance matters

### Q: How do I handle islands (units with no neighbors)?

**A:** Islands (isolated units) require special handling:

```python
# Check for islands
W_array = W.W
islands = np.where(W_array.sum(axis=1) == 0)[0]
if len(islands) > 0:
    print(f"Warning: {len(islands)} islands found")

# Options:
# 1. Remove islands from analysis
data_no_islands = data[~data.index.isin(islands)]

# 2. Connect to nearest neighbor
W_fixed = W.handle_islands(method='nearest')

# 3. Use distance-based with larger threshold
W_distance = SpatialWeights.from_distance(coords, threshold=1000)
```

---

## Model Estimation Questions

### Q: Why does spatial model estimation take longer than OLS?

**A:** Spatial models require:
1. Computation of log-determinant of (I - ρW) for each ρ value
2. Iterative optimization (not closed-form like OLS)
3. Larger matrices for spatial transformations

Speed tips:
- Use sparse matrices for large N: `W_sparse = W.to_sparse()`
- Pre-compute eigenvalues for repeated estimation
- Consider sampling for very large datasets (N > 10,000)
- Use Chebyshev approximation for log-det: `method='chebyshev'`

### Q: What if my spatial model doesn't converge?

**A:** Try these solutions:

1. **Check weight matrix**:
```python
# Ensure W is properly normalized
assert np.allclose(W.W.sum(axis=1), 1.0)
# Check for extreme values
assert W.W.max() < 1.0
```

2. **Simplify model**:
```python
# Start with SAR or SEM before trying SDM
sar_result = experiment.add_spatial_model('SAR', W, 'sar')
```

3. **Adjust optimizer settings**:
```python
result = model.fit(
    method='ml',
    optim_method='l-bfgs-b',
    maxiter=1000,
    tol=1e-6
)
```

4. **Check for multicollinearity**:
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
```

### Q: How do I test if the spatial model improved over OLS?

**A:** Use multiple criteria:

1. **Likelihood Ratio Test**:
```python
lr_stat = 2 * (spatial_model.llf - ols_model.llf)
p_value = 1 - stats.chi2.cdf(lr_stat, df=1)  # df = number of spatial parameters
```

2. **Information Criteria**:
```python
comparison = experiment.compare_spatial_models()
# Lower AIC/BIC is better
best_model = comparison.loc[comparison['AIC'].idxmin(), 'Model']
```

3. **Residual Spatial Autocorrelation**:
```python
# Should be non-significant after spatial model
moran_test_residuals = MoranIPanelTest(spatial_model.resid, W)
result = moran_test_residuals.run()
assert result.pvalue > 0.05  # No remaining autocorrelation
```

---

## Inference and Standard Errors

### Q: What is Spatial HAC and when should I use it?

**A:** Spatial HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors, following Conley (1999), are robust to both spatial and temporal correlation:

```python
result = model.fit(
    se_type='spatial_hac',
    spatial_cutoff=500,   # km for spatial correlation
    temporal_cutoff=2     # years for temporal correlation
)
```

Use Spatial HAC when:
- You have both spatial and temporal dependence
- Standard errors seem too small (overly optimistic)
- You want conservative inference
- Cross-sectional dependence persists in residuals

### Q: How do I perform hypothesis tests on spatial parameters?

**A:** Standard hypothesis tests apply:

```python
# Test H0: ρ = 0 (no spatial dependence)
z_stat = sar_result.rho / sar_result.rho_se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Test H0: ρ = 0.5 (specific value)
z_stat = (sar_result.rho - 0.5) / sar_result.rho_se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Joint test for SDM: H0: θ = 0 (no spatial lag of X)
# Use Wald test
wald_stat = sdm_result.wald_test_spatial_x()
```

### Q: What about Driscoll-Kraay standard errors?

**A:** Driscoll-Kraay is an alternative to Spatial HAC that doesn't require specifying W:

```python
result = model.fit(se_type='driscoll_kraay', lag_cutoff=2)
```

Comparison:
- **Spatial HAC**: Uses explicit distance/contiguity structure
- **Driscoll-Kraay**: Assumes general cross-sectional dependence
- Both are robust to heteroskedasticity

Choose Driscoll-Kraay when:
- Spatial structure is unknown or complex
- You want to be agnostic about spatial patterns
- Cross-sectional dependence is present but not necessarily geographic

---

## Practical Issues

### Q: My data has missing values. How do I handle them?

**A:** Spatial models require balanced panels in the cross-section:

```python
# Option 1: Drop entities with any missing values
data_complete = data.groupby('entity').filter(lambda x: x['y'].notna().all())

# Option 2: Impute using spatial information
from panelbox.utils import spatial_impute
data_imputed = spatial_impute(data, W, variables=['y', 'x1'])

# Option 3: Use only complete time periods
complete_periods = data.groupby('time')['y'].apply(lambda x: x.notna().all())
data_balanced = data[data['time'].isin(complete_periods[complete_periods].index)]
```

### Q: How do I handle time-varying spatial weights?

**A:** PanelBox supports time-varying W matrices:

```python
# Create different W for each period
W_list = []
for t in periods:
    W_t = create_weight_matrix_for_period(t)
    W_list.append(W_t)

# Use in estimation
model = SpatialLag(formula, data, entity_col, time_col, W=W_list)
```

Common applications:
- Transportation networks that change over time
- Trade relationships that evolve
- Social networks with changing connections

### Q: Can I use spatial models with unbalanced panels?

**A:** Yes, but with limitations:

```python
# Check panel balance
from panelbox.utils import check_panel_balance
balance_report = check_panel_balance(data, 'entity', 'time')

# For slightly unbalanced panels
if balance_report['percent_complete'] > 0.9:
    # Can use standard spatial models with caution
    result = model.fit(handle_missing='drop')

# For heavily unbalanced panels
else:
    # Consider:
    # 1. Focus on balanced subset
    # 2. Use methods that handle unbalanced panels
    # 3. Imputation strategies
```

### Q: How large can N be for spatial models?

**A:** Performance guidelines:

| N (entities) | T (periods) | Estimation Time | Recommended Method |
|--------------|-------------|-----------------|-------------------|
| < 100 | Any | < 1 second | Standard ML |
| 100-500 | Any | 1-10 seconds | Standard ML |
| 500-1000 | Any | 10-60 seconds | Standard ML |
| 1000-5000 | < 20 | 1-5 minutes | Sparse matrices |
| 5000-10000 | < 10 | 5-30 minutes | Chebyshev approximation |
| > 10000 | < 10 | > 30 minutes | Consider alternatives |

For very large N:
```python
# Use sparse matrices
W_sparse = W.to_sparse()

# Use approximations
result = model.fit(method='chebyshev', order=50)

# Or consider GMM/2SLS
result = model.fit(method='gmm')
```

---

## Visualization and Reporting

### Q: How do I visualize spatial patterns?

**A:** PanelBox provides several visualization tools:

```python
# 1. Moran's I scatterplot
morans_test.plot_scatterplot()

# 2. LISA cluster map (requires geopandas)
lisa_results = LocalMoranI(data, W).run()
lisa_results.plot_clusters(gdf)

# 3. Spatial lag plot
import matplotlib.pyplot as plt
y = data['y'].values
Wy = W @ y
plt.scatter(y, Wy)
plt.xlabel('y')
plt.ylabel('Spatial lag of y')

# 4. Weight matrix visualization
W.plot_weights()
```

### Q: How do I export results for publication?

**A:** Multiple export options:

```python
# LaTeX table
latex_table = result.to_latex(
    caption="Spatial Model Results",
    label="tab:spatial",
    stars=True
)

# HTML report
experiment.generate_spatial_report('report.html')

# CSV for further processing
comparison_df = experiment.compare_spatial_models()
comparison_df.to_csv('model_comparison.csv')

# Markdown summary
summary = result.summary_markdown()
```

### Q: What should I report in my paper?

**A:** Essential elements for spatial econometric results:

1. **Diagnostics**: Moran's I test, LM tests
2. **Model comparison**: AIC, BIC, Log-likelihood
3. **Spatial parameters**: ρ, λ with standard errors
4. **Effects decomposition**: Direct, indirect, total (for SDM)
5. **Post-estimation tests**: Residual Moran's I
6. **Weight matrix**: Description and summary statistics

Example table structure:
```
Model:          OLS     SAR     SEM     SDM
------------------------------------------------
β_1            0.50    0.45    0.48    0.43
              (0.10)  (0.09)  (0.09)  (0.08)
ρ                      0.30            0.25
                      (0.05)          (0.04)
λ                              0.35
                              (0.06)
Direct (β_1)                           0.44
Indirect (β_1)                         0.15
Total (β_1)                            0.59
------------------------------------------------
Log-lik        -500    -480    -485    -475
AIC            1010     972     982     966
Moran's I      5.2**   0.8     0.5     0.3
```

---

## Troubleshooting

### Q: I get "Matrix is singular" error. What's wrong?

**A:** Common causes and solutions:

1. **Perfect multicollinearity**:
```python
# Check correlation matrix
corr_matrix = data[['x1', 'x2', 'x3']].corr()
# Remove highly correlated variables (|r| > 0.9)
```

2. **Weight matrix issues**:
```python
# Check if W has islands
assert W.W.sum(axis=1).min() > 0
# Check eigenvalues
eigenvalues = np.linalg.eigvals(W.W)
assert eigenvalues.max() < 1
```

3. **Insufficient variation**:
```python
# Check within-entity variation for FE models
within_var = data.groupby('entity')['y'].var()
assert within_var.min() > 0
```

### Q: The spatial lag coefficient ρ is exactly 1 or -1. Is this wrong?

**A:** Yes, |ρ| should be strictly less than 1 for model stability. If you get |ρ| = 1:

- Check weight matrix normalization
- Consider different W specification
- May indicate model misspecification
- Try simpler model (SAR instead of SDM)
- Check for perfect spatial correlation in data

### Q: How do I debug slow estimation?

**A:** Performance diagnostics:

```python
import time

# Profile different steps
start = time.time()
W_eigen = np.linalg.eigvals(W.W)
print(f"Eigenvalues: {time.time() - start:.2f}s")

start = time.time()
result = model.fit(verbose=True)  # Shows optimization progress
print(f"Estimation: {time.time() - start:.2f}s")

# Use sparse matrices if W is sparse
sparsity = (W.W == 0).sum() / W.W.size
if sparsity > 0.9:
    W_sparse = W.to_sparse()
    print(f"Using sparse matrix (sparsity: {sparsity:.1%})")
```

---

## Advanced Topics

### Q: Can I use spatial models for prediction?

**A:** Yes, but with considerations for spatial dependence:

```python
# In-sample prediction
y_pred = sar_result.predict()

# Out-of-sample requires handling spatial structure
# For new location i with known neighbors:
X_new = np.array([[1, 2, 3]])  # New observation features
neighbors_y = np.array([10, 12, 11])  # Neighbors' y values

# SAR prediction: y_i = ρ * W_i * y_neighbors + X_i * β + ε
y_pred_new = (
    sar_result.rho * neighbors_y.mean() +
    X_new @ sar_result.params[1:]
)
```

### Q: How do I test for spatial structural breaks?

**A:** Test if spatial dependence changes over time:

```python
# Split sample
early = data[data['year'] <= 2010]
late = data[data['year'] > 2010]

# Estimate separate models
sar_early = SpatialLag(formula, early, entity_col, time_col, W).fit()
sar_late = SpatialLag(formula, late, entity_col, time_col, W).fit()

# Chow-type test for spatial parameter
z_stat = (sar_early.rho - sar_late.rho) / np.sqrt(
    sar_early.rho_se**2 + sar_late.rho_se**2
)
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
```

### Q: Can I combine spatial models with other panel methods?

**A:** Yes, PanelBox supports various combinations:

```python
# Spatial + IV
spatial_iv = SpatialLag(
    formula='y ~ x1 + x2 | z1 + z2',  # z1, z2 are instruments
    data=data,
    entity_col='entity',
    time_col='time',
    W=W
)

# Spatial + Dynamic (spatial panel VAR)
from panelbox.var import SpatialPanelVAR
spvar = SpatialPanelVAR(data, lags=2, W=W)

# Spatial + Nonlinear (spatial probit/logit)
spatial_probit = SpatialProbit(formula, data, W=W)
```

---

## References and Further Reading

### Key Papers:
- **Anselin (1988)**: Spatial Econometrics: Methods and Models
- **LeSage & Pace (2009)**: Introduction to Spatial Econometrics
- **Elhorst (2014)**: Spatial Econometrics: From Cross-Sectional Data to Spatial Panels
- **Conley (1999)**: GMM estimation with cross sectional dependence
- **Lee & Yu (2010)**: Estimation of spatial autoregressive panel data models

### PanelBox Resources:
- [Documentation](https://panelbox.readthedocs.io/spatial/)
- [Examples](https://github.com/panelbox/examples/spatial/)
- [API Reference](https://panelbox.readthedocs.io/api/spatial/)
- [Tutorials](https://panelbox.readthedocs.io/tutorials/spatial/)

### R Equivalents:
- `splm` package: `spml()`, `spreml()`
- `spdep` package: `lagsarlm()`, `errorsarlm()`
- `sphet` package: `spreg()`

---

**Have more questions?** Open an issue on [GitHub](https://github.com/panelbox/panelbox/issues) or check the [documentation](https://panelbox.readthedocs.io).
