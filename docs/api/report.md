# Report and Export API

API documentation for reporting and export utilities.

**NEW in v0.8.0**: PanelExperiment, ValidationTest, ComparisonTest, HTML reports, Master reports

---

## PanelExperiment (NEW in v0.8.0)

High-level API for panel data analysis with integrated report generation.

### Overview

`PanelExperiment` provides a unified interface for:
- Fitting multiple panel models
- Running validation tests
- Comparing models
- Analyzing residuals
- Generating HTML reports

### Usage

```python
import panelbox as pb

# Create experiment
experiment = pb.PanelExperiment(
    data=data,
    formula="y ~ x1 + x2",
    entity_col="entity",
    time_col="time"
)

# Fit models
experiment.fit_model('pooled_ols', name='ols')
experiment.fit_model('fixed_effects', name='fe')
experiment.fit_model('random_effects', name='re')

# List fitted models
print(experiment.list_models())  # ['ols', 'fe', 're']

# Get model
results = experiment.get_model('fe')
```

### Methods

#### fit_model(model_type, name, **kwargs)

Fit a panel model and store it in the experiment.

**Parameters:**
- `model_type` (str): Model type ('pooled_ols', 'fixed_effects', 'random_effects', 'between', 'first_differences')
- `name` (str): Name for the fitted model
- `**kwargs`: Additional arguments passed to model constructor

**Returns:** PanelResults object

#### validate_model(name, config='basic', tests=None)

Run validation tests on a fitted model.

**Parameters:**
- `name` (str): Name of fitted model
- `config` (str): Test configuration ('quick', 'basic', 'full')
- `tests` (list, optional): Specific tests to run

**Returns:** ValidationResult object

**Example:**
```python
# Quick validation (2 tests)
val_quick = experiment.validate_model('fe', config='quick')

# Full validation (all tests)
val_full = experiment.validate_model('fe', config='full')

# Custom tests
val_custom = experiment.validate_model('fe', tests=['heteroskedasticity', 'normality'])
```

#### compare_models(model_names, include_coefficients=True, include_statistics=True)

Compare multiple fitted models.

**Parameters:**
- `model_names` (list): List of model names to compare
- `include_coefficients` (bool): Include coefficient comparison
- `include_statistics` (bool): Include fit statistics

**Returns:** ComparisonResult object

**Example:**
```python
comparison = experiment.compare_models(['ols', 'fe', 're'])
best = comparison.best_model('rsquared_adj', prefer_lower=False)
print(f"Best model: {best}")
```

#### analyze_residuals(name)

Analyze residuals from a fitted model.

**Parameters:**
- `name` (str): Name of fitted model

**Returns:** ResidualResult object

#### save_master_report(file_path, theme='professional', title=None, reports=None)

Generate master HTML report with navigation to all sub-reports.

**Parameters:**
- `file_path` (str): Output file path
- `theme` (str): Visual theme ('professional', 'academic', 'presentation')
- `title` (str, optional): Custom report title
- `reports` (list, optional): List of sub-report configurations

**Returns:** str (file path)

**Example:**
```python
experiment.save_master_report('master.html', theme='professional', reports=[
    {'type': 'validation', 'title': 'Validation', 'description': 'Tests', 'file_path': 'val.html'},
    {'type': 'comparison', 'title': 'Comparison', 'description': 'Models', 'file_path': 'comp.html'},
    {'type': 'residuals', 'title': 'Residuals', 'description': 'Diagnostics', 'file_path': 'res.html'}
])
```

---

## ValidationTest (NEW in v0.8.0)

Test runner for model validation with configurable presets.

### Overview

`ValidationTest` provides three preset configurations:
- **quick**: Fast validation (heteroskedasticity, autocorrelation)
- **basic**: Standard validation (adds normality test)
- **full**: Comprehensive validation (adds Hausman test)

### Usage

```python
from panelbox.experiment.tests import ValidationTest

runner = ValidationTest()

# Run with preset
validation_result = runner.run(results, config='full')

# Run with custom tests
validation_result = runner.run(results, tests=['heteroskedasticity', 'normality'])
```

### Available Configs

```python
runner.CONFIGS = {
    'quick': ['heteroskedasticity', 'autocorrelation'],
    'basic': ['heteroskedasticity', 'autocorrelation', 'normality'],
    'full': ['heteroskedasticity', 'autocorrelation', 'normality', 'hausman']
}
```

---

## ComparisonTest (NEW in v0.8.0)

Test runner for comparing multiple models.

### Usage

```python
from panelbox.experiment.tests import ComparisonTest

runner = ComparisonTest()

# Compare models
models = {
    'ols': ols_results,
    'fe': fe_results,
    're': re_results
}

comparison_result = runner.run(models)
```

---

## Result Containers

### ValidationResult (NEW in v0.8.0)

Container for validation test results.

**Methods:**
- `save_html(file_path, test_type='validation', theme='professional')`: Generate HTML report
- `save_json(file_path)`: Export to JSON
- `summary()`: Get text summary

**Example:**
```python
validation = experiment.validate_model('fe')
validation.save_html('validation.html', test_type='validation', theme='professional')
print(validation.summary())
```

### ComparisonResult (NEW in v0.8.0)

Container for model comparison results.

**Methods:**
- `save_html(file_path, test_type='comparison', theme='professional')`: Generate HTML report
- `save_json(file_path)`: Export to JSON
- `summary()`: Get text summary
- `best_model(metric, prefer_lower=True)`: Identify best model

**Example:**
```python
comparison = experiment.compare_models(['ols', 'fe', 're'])
best = comparison.best_model('aic', prefer_lower=True)
comparison.save_html('comparison.html', test_type='comparison')
```

### ResidualResult (NEW in v0.7.0)

Container for residual diagnostics.

**Methods:**
- `save_html(file_path, test_type='residuals', theme='professional')`: Generate HTML report
- `save_json(file_path)`: Export to JSON
- `summary()`: Get text summary

**Properties:**
- `shapiro_test`: Shapiro-Wilk normality test
- `durbin_watson`: Durbin-Watson autocorrelation test
- `jarque_bera`: Jarque-Bera normality test
- `ljung_box`: Ljung-Box serial correlation test

---

## Themes

PanelBox provides three professional themes for HTML reports:

### Professional (Default)
- **Color**: Blue (#2563eb)
- **Use Case**: Corporate reports, general analysis
- **Style**: Clean, modern, professional

### Academic
- **Color**: Gray (#4b5563)
- **Use Case**: Research papers, academic publications
- **Style**: Conservative, publication-ready

### Presentation
- **Color**: Purple (#7c3aed)
- **Use Case**: Presentations, slides, demos
- **Style**: Bold, eye-catching

**Example:**
```python
# Try different themes
validation.save_html('val_pro.html', theme='professional')
validation.save_html('val_academic.html', theme='academic')
validation.save_html('val_presentation.html', theme='presentation')
```

---

## Summary Tables

### summary Method

Generate formatted summary table of estimation results.

**Usage:**

```python
results = model.fit()

# Print summary
print(results.summary())

# Get as string
summary_str = str(results.summary())
```

**Available via:**

::: panelbox.core.results.PanelResults.summary
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

---

## Model Comparison

### Compare Multiple Models

Compare results from multiple models side-by-side.

**Usage:**

```python
import pandas as pd

# Estimate models
pooled = pb.PooledOLS(...).fit()
fe = pb.FixedEffects(...).fit()
re = pb.RandomEffects(...).fit()

# Compare coefficients
comparison = pd.DataFrame({
    'Pooled OLS': pooled.params,
    'Fixed Effects': fe.params,
    'Random Effects': re.params
})

print(comparison)

# Compare standard errors
se_comparison = pd.DataFrame({
    'Pooled OLS': pooled.std_errors,
    'Fixed Effects': fe.std_errors,
    'Random Effects': re.std_errors
})

print(se_comparison)
```

---

## Export Examples

### Example: Side-by-Side Model Comparison

```python
# Estimate multiple models
models = {
    'Difference GMM': pb.DifferenceGMM(...).fit(),
    'System GMM': pb.SystemGMM(...).fit()
}

# Create comparison table
comparison = pd.DataFrame({
    name: res.params for name, res in models.items()
})

# Add standard errors row
for name, res in models.items():
    comparison[f"{name} (SE)"] = res.std_errors

# Export to LaTeX manually
with open("comparison.tex", "w") as f:
    f.write(comparison.to_latex(float_format="%.3f"))
```
