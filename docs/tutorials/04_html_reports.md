# HTML Report System Tutorial

> Learn how to generate professional HTML reports for your panel data analysis.

**NEW in v0.8.0**

## What You'll Learn

In this tutorial, you will:

- Create a PanelExperiment for analysis
- Generate validation reports with diagnostic tests
- Compare multiple models side-by-side
- Analyze residuals with interactive plots
- Generate a master report with navigation
- Customize reports with different themes

## Prerequisites

- Completed [Getting Started](01_getting_started.md)
- Completed [Static Panel Models](02_static_models.md)
- PanelBox ≥ 0.8.0

## What is the Report System?

PanelBox v0.8.0 introduces a comprehensive HTML report system that generates:

- **Validation Reports**: Diagnostic tests with pass/fail indicators
- **Comparison Reports**: Side-by-side model comparison
- **Residual Reports**: Interactive diagnostic plots
- **Master Reports**: Overview dashboard with navigation

All reports are:
- Self-contained (work offline)
- Interactive (Plotly charts, sortable tables)
- Professional (three themes available)
- Exportable (JSON format for programmatic analysis)

## Step 1: Create PanelExperiment

The `PanelExperiment` class is the high-level interface for analysis:

```python
import panelbox as pb

# Load data
data = pb.load_grunfeld()

# Create experiment
experiment = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

print(experiment)
```

**Output:**
```
PanelExperiment(
  formula='invest ~ value + capital',
  n_obs=200,
  n_models=0,
  models=[none]
)
```

## Step 2: Fit Multiple Models

Fit different panel models to compare:

```python
# Fit three models
experiment.fit_model('pooled_ols', name='ols')
experiment.fit_model('fixed_effects', name='fe')
experiment.fit_model('random_effects', name='re')

# List fitted models
print(f"Fitted models: {experiment.list_models()}")
```

**Output:**
```
Fitting pooled_ols model 'ols'...
✅ Model 'ols' fitted successfully

Fitting fixed_effects model 'fe'...
✅ Model 'fe' fitted successfully

Fitting random_effects model 're'...
✅ Model 're' fitted successfully

Fitted models: ['ols', 'fe', 're']
```

## Step 3: Generate Validation Report

Run diagnostic tests and generate an interactive HTML report:

```python
# Validate Fixed Effects model
validation = experiment.validate_model('fe', config='full')

# Generate HTML report
validation.save_html(
    'validation_report.html',
    test_type='validation',
    theme='professional'
)

print(validation.summary())
```

**What's in the validation report:**
- Heteroskedasticity test (Breusch-Pagan)
- Autocorrelation test (Wooldridge)
- Normality test (Jarque-Bera)
- Hausman test (FE vs RE)
- Pass/fail indicators with color coding
- Test statistics and p-values
- Recommendations

### Validation Configs

Three preset configurations are available:

```python
# Quick: 2 tests (heteroskedasticity, autocorrelation)
val_quick = experiment.validate_model('fe', config='quick')

# Basic: 3 tests (adds normality)
val_basic = experiment.validate_model('fe', config='basic')

# Full: 4+ tests (adds Hausman)
val_full = experiment.validate_model('fe', config='full')
```

## Step 4: Generate Comparison Report

Compare multiple models side-by-side:

```python
# Compare all three models
comparison = experiment.compare_models(['ols', 'fe', 're'])

# Generate HTML report
comparison.save_html(
    'comparison_report.html',
    test_type='comparison',
    theme='professional'
)

# Identify best model
best_aic = comparison.best_model('aic', prefer_lower=True)
best_r2 = comparison.best_model('rsquared_adj', prefer_lower=False)

print(f"Best by AIC: {best_aic}")
print(f"Best by R²: {best_r2}")

print(comparison.summary())
```

**What's in the comparison report:**
- Side-by-side coefficient comparison
- Standard errors and significance stars
- Fit statistics (R², AIC, BIC, F-statistic)
- Interactive table (sortable, searchable)
- Best model highlighted

## Step 5: Generate Residual Diagnostics

Analyze residuals with interactive plots:

```python
# Analyze residuals from Fixed Effects model
residuals = experiment.analyze_residuals('fe')

# Generate HTML report
residuals.save_html(
    'residuals_report.html',
    test_type='residuals',
    theme='professional'
)

print(residuals.summary())
```

**What's in the residuals report:**
- Residuals vs Fitted plot
- QQ plot (normality check)
- Scale-Location plot (homoskedasticity)
- Residuals vs Leverage
- ACF/PACF plots (autocorrelation)
- Diagnostic test statistics:
  - Shapiro-Wilk test (normality)
  - Durbin-Watson test (autocorrelation)
  - Jarque-Bera test (normality)
  - Ljung-Box test (serial correlation)

## Step 6: Generate Master Report

Create a master report with navigation to all sub-reports:

```python
# Generate master report
experiment.save_master_report(
    'master_report.html',
    theme='professional',
    title='Panel Data Analysis - Complete Report',
    reports=[
        {
            'type': 'validation',
            'title': 'Model Validation',
            'description': 'Specification tests for Fixed Effects model',
            'file_path': 'validation_report.html'
        },
        {
            'type': 'comparison',
            'title': 'Model Comparison',
            'description': 'Compare Pooled OLS, FE, and RE models',
            'file_path': 'comparison_report.html'
        },
        {
            'type': 'residuals',
            'title': 'Residual Diagnostics',
            'description': 'Diagnostic plots and tests',
            'file_path': 'residuals_report.html'
        }
    ]
)

print("✅ Master report generated: master_report.html")
```

**What's in the master report:**
- Experiment overview (formula, observations, entities)
- Summary of all fitted models
- Quick start guide
- Navigation cards to all sub-reports
- Responsive design (works on mobile)

**Open `master_report.html` in your browser to explore!**

## Step 7: Try Different Themes

PanelBox provides three professional themes:

### Professional Theme (Default)
```python
validation.save_html('val_professional.html', theme='professional')
```
- **Color**: Blue (#2563eb)
- **Use Case**: Corporate reports, general analysis
- **Style**: Clean, modern, professional

### Academic Theme
```python
validation.save_html('val_academic.html', theme='academic')
```
- **Color**: Gray (#4b5563)
- **Use Case**: Research papers, publications
- **Style**: Conservative, publication-ready

### Presentation Theme
```python
validation.save_html('val_presentation.html', theme='presentation')
```
- **Color**: Purple (#7c3aed)
- **Use Case**: Presentations, slides, demos
- **Style**: Bold, eye-catching

## Step 8: Export to JSON

All results can be exported to JSON for programmatic analysis:

```python
# Export to JSON
validation.save_json('validation_results.json')
comparison.save_json('comparison_results.json')
residuals.save_json('residuals_results.json')

print("✅ All results exported to JSON")
```

JSON files contain:
- All test results and statistics
- Model metadata
- Timestamps
- Configuration settings

Use JSON export for:
- Custom analysis pipelines
- Integration with other tools
- Archiving results
- Reproducibility

## Complete Workflow

Here's the complete workflow in one script:

```python
import panelbox as pb

# 1. Load data and create experiment
data = pb.load_grunfeld()
experiment = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

# 2. Fit multiple models
experiment.fit_model('pooled_ols', name='ols')
experiment.fit_model('fixed_effects', name='fe')
experiment.fit_model('random_effects', name='re')

# 3. Generate all reports
validation = experiment.validate_model('fe', config='full')
validation.save_html('validation.html', test_type='validation')

comparison = experiment.compare_models(['ols', 'fe', 're'])
comparison.save_html('comparison.html', test_type='comparison')

residuals = experiment.analyze_residuals('fe')
residuals.save_html('residuals.html', test_type='residuals')

# 4. Generate master report
experiment.save_master_report('master.html', reports=[
    {'type': 'validation', 'title': 'Validation',
     'description': 'Specification tests', 'file_path': 'validation.html'},
    {'type': 'comparison', 'title': 'Comparison',
     'description': 'Model comparison', 'file_path': 'comparison.html'},
    {'type': 'residuals', 'title': 'Residuals',
     'description': 'Diagnostic plots', 'file_path': 'residuals.html'}
])

print("✅ Complete analysis finished! Open master.html")
```

## Best Practices

### 1. Always Validate Models

Before interpreting results, run validation tests:

```python
validation = experiment.validate_model('fe', config='full')
if validation.all_passed:
    print("✅ Model passes all tests")
else:
    print("⚠️ Some tests failed - review validation report")
```

### 2. Compare Multiple Models

Never rely on a single model:

```python
# Fit multiple specifications
experiment.fit_model('pooled_ols', name='ols')
experiment.fit_model('fixed_effects', name='fe')
experiment.fit_model('random_effects', name='re')

# Compare and choose best
comparison = experiment.compare_models(['ols', 'fe', 're'])
best = comparison.best_model('rsquared_adj', prefer_lower=False)
```

### 3. Check Residuals

Always inspect residuals for violations:

```python
residuals = experiment.analyze_residuals('fe')

# Check diagnostic tests
if residuals.shapiro_test[1] < 0.05:
    print("⚠️ Residuals not normal")

if residuals.durbin_watson < 1.5 or residuals.durbin_watson > 2.5:
    print("⚠️ Autocorrelation detected")
```

### 4. Use Master Reports

Generate master reports for comprehensive documentation:

```python
experiment.save_master_report(
    'analysis_complete.html',
    title=f'{data_name} Panel Analysis',
    reports=[...]  # Include all sub-reports
)
```

### 5. Archive Results

Export to JSON for reproducibility:

```python
validation.save_json(f'validation_{timestamp}.json')
comparison.save_json(f'comparison_{timestamp}.json')
```

## Tips and Tricks

### Custom Test Selection

Run specific tests only:

```python
validation = experiment.validate_model('fe', tests=[
    'heteroskedasticity',
    'autocorrelation'
])
```

### Theme Customization

Try all themes and choose the best for your use case:

```python
themes = ['professional', 'academic', 'presentation']
for theme in themes:
    validation.save_html(f'report_{theme}.html', theme=theme)
```

### Batch Processing

Analyze multiple datasets:

```python
datasets = ['data1.csv', 'data2.csv', 'data3.csv']

for data_file in datasets:
    data = pd.read_csv(data_file)
    experiment = pb.PanelExperiment(data, formula, entity_col, time_col)
    experiment.fit_model('fixed_effects', name='fe')
    validation = experiment.validate_model('fe')
    validation.save_html(f'validation_{data_file}.html')
```

## Next Steps

Now that you've mastered the HTML report system:

- Explore [GMM models](03_gmm_intro.md) for dynamic panels
- Check [API Reference](../api/report.md) for detailed documentation
- Review the examples directory for complete workflow examples

## Summary

You learned how to:

- ✅ Create PanelExperiment for analysis
- ✅ Generate validation reports with diagnostic tests
- ✅ Compare multiple models side-by-side
- ✅ Analyze residuals with interactive plots
- ✅ Generate master reports with navigation
- ✅ Customize reports with themes
- ✅ Export results to JSON

**The HTML report system makes panel data analysis professional, reproducible, and easy to share!**

---

**Tutorial complete!** 🎉

For more information:
- [API Reference](../api/report.md)
- Check the `examples/` directory for complete workflow examples
- View the project CHANGELOG for latest updates
