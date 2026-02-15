# Quantile Regression Integration Guide

## Overview

This guide explains how to integrate quantile regression models with the broader PanelBox ecosystem, including data handling, model selection, visualization, and reporting.

## Quick Start

```python
from panelbox import PanelExperiment
from panelbox.models.quantile import LocationScale, PooledQuantile

# Load your panel data
exp = PanelExperiment(data='your_dataset.csv')

# Estimate quantile regression
qr_result = exp.estimate_quantile(
    tau=[0.25, 0.5, 0.75],
    method='location_scale'
)

# Compare with other models
comparison = exp.compare_quantile_methods()
```

## Integration with PanelData

### Data Preparation

```python
from panelbox.data import PanelData

# Create panel data object
panel = PanelData(
    data=df,
    entity_col='firm_id',
    time_col='year'
)

# Check balance for quantile regression
if not panel.is_balanced:
    print("Warning: Unbalanced panel may affect quantile estimates")
    panel = panel.make_balanced()

# Handle missing values appropriately
panel = panel.dropna(subset=['outcome', 'treatment'])
```

### Variable Transformations

```python
# Create variables useful for quantile regression
panel['log_outcome'] = panel.transform(
    lambda x: np.log(x + 1),
    columns=['outcome']
)

# Winsorize to handle outliers
panel['outcome_wins'] = panel.winsorize(
    columns=['outcome'],
    limits=(0.01, 0.99)
)

# Standardize for interpretation
panel['outcome_std'] = panel.standardize(
    columns=['outcome'],
    method='robust'  # Uses median and MAD
)
```

## Integration with Model Selection

### Automated Model Comparison

```python
from panelbox.model_selection import ModelSelector

selector = ModelSelector(panel, formula='y ~ x1 + x2 + x3')

# Include quantile regression in selection
best_model = selector.select_best_model(
    include_quantile=True,
    quantile_methods=['pooled', 'canay', 'location_scale'],
    criteria='aic'
)

print(f"Best model: {best_model['name']}")
print(f"AIC: {best_model['aic']:.2f}")
```

### Cross-Validation for Quantile Regression

```python
from panelbox.model_selection import panel_cross_validate

# Define quantile models to compare
models = {
    'qr_median': PooledQuantile(panel, formula, tau=0.5),
    'qr_ls': LocationScale(panel, formula, tau=0.5),
    'qr_fe': FixedEffectsQuantile(panel, formula, tau=0.5)
}

# Perform cross-validation
cv_results = panel_cross_validate(
    models=models,
    cv_method='time_series',  # or 'entity_based'
    n_splits=5,
    metric='check_loss'
)

best_qr_model = cv_results['best_model']
```

## Integration with Fixed Effects Models

### Combining FE with Quantile Regression

```python
from panelbox.models.linear import FixedEffectsOLS
from panelbox.models.quantile import CanayTwoStep, LocationScale

# Method 1: Canay two-step
canay_model = CanayTwoStep(
    data=panel,
    formula='y ~ x1 + x2',
    tau=[0.25, 0.5, 0.75]
)
canay_result = canay_model.fit()

# Method 2: Location-Scale with FE
ls_fe_model = LocationScale(
    data=panel,
    formula='y ~ x1 + x2',
    tau=[0.25, 0.5, 0.75],
    fixed_effects=True
)
ls_fe_result = ls_fe_model.fit()

# Compare with standard FE
fe_model = FixedEffectsOLS(panel, formula='y ~ x1 + x2')
fe_result = fe_model.fit()

# Create comparison table
from panelbox.reporting import create_comparison_table

table = create_comparison_table({
    'FE-OLS (Mean)': fe_result,
    'Canay QR (Median)': canay_result.results[0.5],
    'LS-QR (Median)': ls_fe_result.results[0.5]
})
print(table)
```

## Integration with Inference Methods

### Bootstrap and Clustering

```python
from panelbox.inference.quantile import ClusteredQuantileBootstrap

# Setup clustered bootstrap for panel QR
bootstrap = ClusteredQuantileBootstrap(
    cluster_var='entity_id',
    n_boot=999,
    method='wild'  # or 'pairs'
)

# Apply to any quantile model
qr_model = PooledQuantile(panel, formula, tau=0.5)
result_with_ci = bootstrap.apply(qr_model)

print(f"Coefficient: {result_with_ci.params[1]:.4f}")
print(f"95% CI: [{result_with_ci.ci_lower[1]:.4f}, {result_with_ci.ci_upper[1]:.4f}]")
```

### Multiple Testing Correction

```python
from panelbox.inference import multiple_testing_correction

# Testing effects at multiple quantiles
tau_grid = np.arange(0.1, 1.0, 0.1)
p_values = []

for tau in tau_grid:
    model = PooledQuantile(panel, formula, tau)
    result = model.fit()
    p_values.append(result.pvalues[1])  # p-value for x1

# Apply correction
p_adjusted = multiple_testing_correction(
    p_values,
    method='fdr_bh'  # Benjamini-Hochberg
)

# Which quantiles show significant effects?
significant = [tau for tau, p in zip(tau_grid, p_adjusted) if p < 0.05]
print(f"Significant effects at quantiles: {significant}")
```

## Integration with Visualization

### Standard Visualizations

```python
from panelbox.visualization.quantile import QuantilePlotter

plotter = QuantilePlotter(style='academic')

# Coefficient plots across quantiles
fig = plotter.plot_coefficients(
    results=qr_results,
    var_name='treatment',
    show_ols=True,  # Add OLS for comparison
    confidence_level=0.95
)

# Quantile process plots
fig = plotter.plot_process(
    results=qr_results,
    variables=['x1', 'x2'],
    normalize=True
)

# Distribution comparison
fig = plotter.plot_distributions(
    y_treated=panel[panel.treated==1].outcome,
    y_control=panel[panel.treated==0].outcome,
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
)
```

### Advanced Visualizations

```python
# Interactive quantile regression plots
from panelbox.visualization.interactive import InteractiveQR

interactive_qr = InteractiveQR(qr_results)

# Create dashboard
dashboard = interactive_qr.create_dashboard(
    include_diagnostics=True,
    include_predictions=True
)
dashboard.show()

# 3D surface plot for two covariates
fig = plotter.plot_3d_surface(
    model=ls_model,
    x1_range=(-2, 2),
    x2_range=(-2, 2),
    tau=0.5
)
```

## Integration with Reporting

### Automated Report Generation

```python
from panelbox.reporting import QuantileReport

# Create comprehensive report
report = QuantileReport(
    title="Quantile Regression Analysis",
    author="Your Name"
)

# Add models
report.add_model(qr_results, name="Main Specification")
report.add_model(ls_results, name="Location-Scale")

# Add diagnostics
report.add_crossing_test()
report.add_specification_tests()
report.add_goodness_of_fit()

# Generate LaTeX/HTML/Markdown
report.to_latex('quantile_report.tex')
report.to_html('quantile_report.html')
report.to_markdown('quantile_report.md')
```

### Publication-Ready Tables

```python
from panelbox.reporting.tables import make_qr_table

# Create regression table for paper
table = make_qr_table(
    models={
        '(1)': qr_results[0.25],
        '(2)': qr_results[0.5],
        '(3)': qr_results[0.75]
    },
    stars=True,
    se_below=True,
    notes="Clustered standard errors in parentheses."
)

# Export to LaTeX
with open('table_qr.tex', 'w') as f:
    f.write(table.to_latex())
```

## Integration with Workflow Pipeline

### Complete Analysis Pipeline

```python
from panelbox import Pipeline

# Define analysis pipeline
pipeline = Pipeline()

# Step 1: Data preparation
pipeline.add_step('load_data', PanelData, {'file': 'data.csv'})
pipeline.add_step('clean', lambda x: x.dropna().winsorize())

# Step 2: Descriptive statistics
pipeline.add_step('describe', lambda x: x.describe_panel())

# Step 3: Model estimation
pipeline.add_step('qr_pooled', PooledQuantile, {
    'formula': 'y ~ x1 + x2',
    'tau': [0.25, 0.5, 0.75]
})

pipeline.add_step('qr_fe', CanayTwoStep, {
    'formula': 'y ~ x1 + x2',
    'tau': [0.25, 0.5, 0.75]
})

# Step 4: Diagnostics
pipeline.add_step('test_crossing', QuantileMonotonicity.detect_crossing)
pipeline.add_step('test_specification', specification_test)

# Step 5: Visualization
pipeline.add_step('plot_results', create_figures)

# Step 6: Report
pipeline.add_step('generate_report', create_report)

# Execute pipeline
results = pipeline.run()
```

## Integration with External Tools

### Export to R

```python
# Export for R analysis
from panelbox.io import export_for_R

export_for_R(
    data=panel,
    results=qr_results,
    file='quantile_results.RData',
    include_model_objects=True
)

# Generate R code for replication
r_code = generate_r_replication_code(qr_model)
with open('replicate_qr.R', 'w') as f:
    f.write(r_code)
```

### Export to Stata

```python
# Export for Stata
from panelbox.io import export_for_stata

export_for_stata(
    data=panel,
    file='panel_data.dta',
    include_labels=True
)

# Generate Stata do-file
stata_code = f"""
* Quantile Regression Analysis
use panel_data.dta, clear

* Pooled quantile regression
qreg y x1 x2, q(50)

* Panel quantile regression
xtqreg y x1 x2, i(entity_id) q(25 50 75)
"""

with open('quantile_analysis.do', 'w') as f:
    f.write(stata_code)
```

## Performance Optimization

### Parallel Computing

```python
from panelbox.optimization import ParallelConfig

# Configure parallel processing
ParallelConfig.set_backend('joblib')
ParallelConfig.n_jobs = -1  # Use all cores

# Parallel estimation of multiple quantiles
from joblib import Parallel, delayed

def estimate_single_tau(tau):
    model = PooledQuantile(panel, formula, tau)
    return tau, model.fit()

tau_grid = np.arange(0.05, 1.0, 0.05)
results = dict(Parallel(n_jobs=-1)(
    delayed(estimate_single_tau)(tau) for tau in tau_grid
))
```

### Memory Management

```python
# For large datasets
from panelbox.utils import chunked_estimation

# Estimate in chunks
results = chunked_estimation(
    model_class=PooledQuantile,
    data=large_panel,
    chunk_size=10000,
    formula=formula,
    tau=[0.25, 0.5, 0.75]
)
```

## Best Practices Summary

1. **Data Preparation**
   - Check panel balance
   - Handle outliers appropriately
   - Consider transformations for skewed outcomes

2. **Model Selection**
   - Start with pooled QR for baseline
   - Use Canay or Location-Scale for FE
   - Always check for crossing

3. **Inference**
   - Use clustered bootstrap for panels
   - Correct for multiple testing
   - Report both point estimates and CI

4. **Visualization**
   - Plot coefficients across quantiles
   - Compare with OLS baseline
   - Show confidence bands

5. **Reporting**
   - Include specification tests
   - Document method choices
   - Provide replication code

## Troubleshooting

### Common Integration Issues

```python
# Issue: Model won't converge
# Solution: Adjust optimization settings
model.fit(
    method='interior-point',
    max_iter=1000,
    tol=1e-6
)

# Issue: Memory errors with large panels
# Solution: Use sparse matrices
from scipy.sparse import csr_matrix
X_sparse = csr_matrix(X)
model.fit(X=X_sparse)

# Issue: Slow bootstrap
# Solution: Use parallel processing
result = model.fit(
    bootstrap=True,
    n_boot=999,
    n_jobs=-1
)
```

## Further Resources

- [PanelBox Documentation](https://panelbox.readthedocs.io)
- [Quantile Regression Examples](examples/quantile/)
- [API Reference](api/quantile/)
- [Contributing Guide](contributing.md)
