# PanelBox Validation & Experiment Quick Reference

## Core Classes at a Glance

### Validation Classes
```python
# Test a single model
from panelbox.validation import ValidationSuite
suite = ValidationSuite(results)
report = suite.run(tests='all')  # or 'default', 'serial', 'het', 'cd'
print(report.summary())

# Individual tests
from panelbox.validation import (
    HausmanTest, ModifiedWaldTest, WooldridgeARTest,
    PesaranCDTest, BreuschPaganTest, WhiteTest
)
hausman = HausmanTest(fe_results, re_results)
result = hausman.run()
print(result.summary())
```

### Robustness Analysis Classes
```python
from panelbox.validation.robustness import (
    PanelBootstrap, TimeSeriesCV, PanelJackknife,
    OutlierDetector, InfluenceDiagnostics
)

# Bootstrap
boot = PanelBootstrap(results, n_bootstrap=1000, method='pairs')
boot_results = boot.run()
print(boot_results.summary())

# Cross-validation
cv = TimeSeriesCV(results, n_folds=10, method='expanding')
cv_results = cv.run()
print(cv_results.summary())

# Outliers
outlier = OutlierDetector(results)
outlier_results = outlier.run(method='iqr')
print(outlier_results.summary())

# Influence
influence = InfluenceDiagnostics(results)
influence_results = influence.run()
print(influence_results.summary())
```

### Experiment Classes
```python
from panelbox.experiment import PanelExperiment

# Fit multiple models
exp = PanelExperiment(
    data=df,
    formula="y ~ x1 + x2",
    entity_col="firm_id",
    time_col="year"
)

# Fit models (supports ~20 types)
exp.fit_model('pooled_ols', name='ols')
exp.fit_model('fixed_effects', name='fe', cov_type='clustered')
exp.fit_model('random_effects', name='re')

# Retrieve models
models = exp.list_models()  # ['ols', 'fe', 're']
fe_model = exp.get_model('fe')

# Compare models
comparison = exp.compare_models(['ols', 'fe', 're'])
print(comparison.summary())
comparison.save_html('comparison.html')
comparison.save_json('comparison.json')

# Get best model
best = comparison.best_model('aic')  # or 'bic', 'rsquared'
```

---

## Data & Visualization Utilities

### Load Datasets
```python
from validation.utils import load_dataset

# Available: 'firmdata', 'macro_panel', 'small_panel', 'sales_panel',
#            'macro_ts_panel', 'panel_with_outliers', 'real_firms',
#            'panel_comprehensive', 'panel_unbalanced'
data = load_dataset('firmdata')
```

### Generate Synthetic Data
```python
from validation.utils import (
    generate_firmdata, generate_macro_panel, generate_small_panel,
    generate_sales_panel, generate_macro_ts_panel,
    generate_panel_with_outliers, generate_real_firms,
    generate_panel_comprehensive, generate_panel_unbalanced
)

# Each accepts random_state, return pandas DataFrame
df = generate_firmdata(n_firms=100, n_years=10, random_state=42)
```

### Plotting Utilities
```python
from validation.utils import (
    plot_residuals_by_entity,
    plot_acf_panel,
    plot_correlation_heatmap,
    plot_bootstrap_distribution,
    plot_cv_predictions,
    plot_influence_index,
    plot_forest_plot,
)

# All return matplotlib.figure.Figure for .savefig() or plt.show()
fig = plot_residuals_by_entity(results.resid, entity_col, max_entities=20)
fig = plot_acf_panel(results.resid, entity_col, lags=10, n_sample=6)
fig = plot_correlation_heatmap(results.resid, entity_col, max_entities=20)
fig = plot_bootstrap_distribution(boot_estimates, param_name='β₁')
fig = plot_cv_predictions(cv_predictions, actual)
fig = plot_influence_index(influence_results.cooks_d)
fig = plot_forest_plot(params, ci_lower, ci_upper, names)
```

---

## Common Workflows

### Workflow 1: Validate a Single Model
```python
from panelbox.models.static import FixedEffects
from panelbox.validation import ValidationSuite

# Fit model
fe = FixedEffects("y ~ x1 + x2", data, "entity", "time")
results = fe.fit()

# Run comprehensive validation
suite = ValidationSuite(results)
report = suite.run(tests='all')
print(report.summary())

# Save report
report.save_html('validation_report.html')
report.save_json('validation_results.json')
```

### Workflow 2: Compare Multiple Models
```python
from panelbox.experiment import PanelExperiment

exp = PanelExperiment(data, "y ~ x1 + x2", "entity", "time")
exp.fit_model('pooled_ols', name='ols')
exp.fit_model('fixed_effects', name='fe')
exp.fit_model('random_effects', name='re')

comparison = exp.compare_models(['ols', 'fe', 're'])
print(comparison.summary())

# Find best by different criteria
best_aic = comparison.best_model('aic')
best_r2 = comparison.best_model('rsquared')

comparison.save_html('model_comparison.html')
```

### Workflow 3: Bootstrap & Cross-Validation
```python
from panelbox.validation.robustness import PanelBootstrap, TimeSeriesCV

# Bootstrap
boot = PanelBootstrap(results, n_bootstrap=1000, method='pairs')
boot_results = boot.run()
print(f"Bootstrap SE: {boot_results.bootstrap_se_}")

# Cross-validation
cv = TimeSeriesCV(results, n_folds=10, method='expanding')
cv_results = cv.run()
print(f"CV RMSE: {cv_results.metrics['rmse']}")
```

### Workflow 4: Detect Outliers & Influential Points
```python
from panelbox.validation.robustness import OutlierDetector, InfluenceDiagnostics

# Outliers
outlier = OutlierDetector(results)
outlier_results = outlier.run(method='iqr')
print(outlier_results.summary())
outlier_df = outlier_results.outlier_table  # DataFrame of flagged rows

# Influence
influence = InfluenceDiagnostics(results)
influence_results = influence.run()
print(influence_results.summary())
```

---

## Supported Model Types in PanelExperiment

### Linear Models
- `'pooled_ols'` or `'pooled'`
- `'fixed_effects'` or `'fe'`
- `'random_effects'` or `'re'`

### Discrete Choice
- `'pooled_logit'`
- `'pooled_probit'`
- `'fe_logit'` or `'fixed_effects_logit'`
- `'re_probit'` or `'random_effects_probit'`

### Count Models
- `'poisson'` or `'pooled_poisson'`
- `'fe_poisson'` or `'poisson_fixed_effects'`
- `'re_poisson'` or `'random_effects_poisson'`
- `'negbin'` or `'negative_binomial'`

### Censored Models
- `'tobit'` or `'re_tobit'` or `'random_effects_tobit'`

### Ordered Models
- `'ologit'` or `'ordered_logit'`
- `'oprobit'` or `'ordered_probit'`

---

## ValidationSuite Test Categories

### Available Tests by Category

```python
suite.run(tests='all')  # Run everything below

# Specification tests (compare models)
suite.run(tests=['hausman', 'mundlak', 'reset', 'chow'])

# Serial correlation tests
suite.run(tests=['wooldridge_ar', 'breusch_godfrey', 'baltagi_wu'])

# Heteroskedasticity tests
suite.run(tests=['modified_wald', 'breusch_pagan', 'white'])

# Cross-sectional dependence tests
suite.run(tests=['pesaran_cd', 'breusch_pagan_lm', 'frees'])
```

---

## ValidationReport Structure

```
ValidationReport
├── model_info              # Model type, formula, N, T
├── specification_tests     # {test_name: ValidationTestResult}
├── serial_tests           # {test_name: ValidationTestResult}
├── het_tests              # {test_name: ValidationTestResult}
└── cd_tests               # {test_name: ValidationTestResult}

ValidationTestResult
├── test_name              # str
├── statistic              # float
├── pvalue                 # float
├── df                     # int or tuple
├── conclusion             # str
└── summary()              # formatted output
```

---

## Tips & Common Patterns

1. **Always set random_state** for reproducibility:
   ```python
   PanelBootstrap(results, n_bootstrap=1000, random_state=42)
   generate_firmdata(random_state=42)
   ```

2. **Use named models** in PanelExperiment for clarity:
   ```python
   exp.fit_model('fixed_effects', name='fe_basic')
   exp.fit_model('fixed_effects', name='fe_clustered', cov_type='clustered')
   ```

3. **Export results** for presentations:
   ```python
   report.save_html('report.html', theme='professional')
   comparison.save_json('models.json')
   ```

4. **Combine validation approaches**:
   ```python
   # Run suite + bootstrap + outlier detection
   suite = ValidationSuite(results).run()
   boot = PanelBootstrap(results).run()
   outliers = OutlierDetector(results).run()
   ```

5. **Create custom model comparison**:
   ```python
   exp = PanelExperiment(data, formula, entity_col, time_col)
   for model_type in ['ols', 'fe', 're']:
       exp.fit_model(model_type, name=model_type)
   best = exp.compare_models(exp.list_models()).best_model('aic')
   ```

---

## Files & Locations

```
panelbox/
├── experiment/
│   ├── panel_experiment.py       (PanelExperiment class)
│   └── results/
│       └── comparison_result.py   (ComparisonResult class)
└── validation/
    ├── validation_suite.py        (ValidationSuite class)
    ├── validation_report.py       (ValidationReport class)
    └── robustness/
        ├── bootstrap.py           (PanelBootstrap)
        ├── cross_validation.py    (TimeSeriesCV)
        ├── outliers.py            (OutlierDetector)
        └── influence.py           (InfluenceDiagnostics)

examples/validation/
├── notebooks/
│   ├── 01_assumption_tests.ipynb
│   ├── 02_bootstrap_cross_validation.ipynb
│   ├── 03_outliers_influence.ipynb
│   └── 04_experiments_model_comparison.ipynb
├── data/                         (10 CSV datasets)
└── utils/
    ├── data_generators.py        (9+ generators)
    ├── plot_helpers.py           (7 plotting functions)
    └── __init__.py
```

---

## Dataset Sizes & Characteristics

| Dataset | Rows | Cols | Purpose |
|---------|------|------|---------|
| firmdata | 1,000 | 6 | Heteroskedasticity |
| macro_panel | 600 | 6 | Cross-sectional dependence + AR(1) |
| small_panel | 200 | 5 | Simple i.i.d. |
| sales_panel | 1,200 | 6 | Seasonal component |
| macro_ts_panel | 600 | 5 | Structural break (2008) |
| panel_with_outliers | 640 | 6 | ~5% outliers injected |
| real_firms | 600 | 6 | Natural heterogeneity |
| panel_comprehensive | 1,200 | 12 | Rich variable set |
| panel_unbalanced | ~1,050 | 6 | Unbalanced (attrition) |
