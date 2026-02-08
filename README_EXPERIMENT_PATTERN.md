# PanelBox Experiment Pattern - Quick Start

**Status**: ✅ Production Ready | **Version**: 1.0 | **Date**: 2026-02-08

---

## What is the Experiment Pattern?

The Experiment Pattern provides a high-level, streamlined API for panel data analysis in PanelBox. Instead of manually creating and managing multiple models, the pattern offers:

- **Factory-based model creation** - Create models by name
- **Automatic model storage** - All models tracked automatically
- **One-liner workflows** - Validate and compare with single commands
- **Professional reports** - Generate HTML reports instantly
- **Best model selection** - Find optimal models automatically

---

## Quick Start (30 seconds)

```python
import panelbox as pb

# Create experiment
experiment = pb.PanelExperiment(data, "y ~ x1 + x2", "firm", "year")

# Fit all standard models
experiment.fit_all_models(names=['pooled', 'fe', 're'])

# Validate fixed effects model
val_result = experiment.validate_model('fe')
val_result.save_html('validation.html', test_type='validation')

# Compare all models
comp_result = experiment.compare_models()
print(f"Best model: {comp_result.best_model('rsquared')}")
```

That's it! You now have:
- 3 fitted models (Pooled OLS, Fixed Effects, Random Effects)
- Validation report with 9+ diagnostic tests
- Comparison report with fit metrics
- HTML reports with interactive Plotly charts

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/panelbox
cd panelbox

# Install with poetry
poetry install

# Or with pip
pip install -e .
```

---

## Core Components

### 1. PanelExperiment

**Purpose**: Orchestrate panel data experiments

**Key Methods**:
- `fit_model(model_type, name, **kwargs)` - Fit a single model
- `fit_all_models(names, **kwargs)` - Fit multiple models at once
- `validate_model(name, tests, alpha)` - Validate and get ValidationResult
- `compare_models(model_names)` - Compare and get ComparisonResult
- `list_models()` - List all fitted models
- `get_model(name)` - Retrieve a fitted model

**Example**:
```python
experiment = pb.PanelExperiment(data, "y ~ x1 + x2", "firm", "year")
experiment.fit_model('fixed_effects', name='fe', cov_type='clustered')
fe_model = experiment.get_model('fe')
```

### 2. ValidationResult

**Purpose**: Container for validation test results

**Key Features**:
- Properties: `total_tests`, `passed_tests`, `failed_tests`, `pass_rate`
- Method: `summary()` - Text summary
- Method: `save_html()` - Generate HTML report
- Method: `save_json()` - Save as JSON

**Example**:
```python
val_result = experiment.validate_model('fe')
print(f"Pass rate: {val_result.pass_rate:.1%}")
print(f"Failed: {val_result.failed_tests}")
val_result.save_html('validation.html', test_type='validation')
```

### 3. ComparisonResult

**Purpose**: Container for model comparison

**Key Features**:
- Properties: `n_models`, `model_names`
- Method: `best_model(metric, prefer_lower)` - Find best model
- Method: `summary()` - Text summary
- Method: `save_html()` - Generate HTML report

**Example**:
```python
comp_result = experiment.compare_models()
best = comp_result.best_model('rsquared')
worst_aic = comp_result.best_model('aic', prefer_lower=True)
comp_result.save_html('comparison.html', test_type='comparison')
```

---

## Complete Example

See `examples/complete_workflow_example.py` for a full working example.

Run it:
```bash
poetry run python examples/complete_workflow_example.py
```

Output:
```
✅ STEP 1: CREATE PANEL DATA
✅ STEP 2: CREATE PANELEXPERIMENT
✅ STEP 3: FIT MULTIPLE MODELS
✅ STEP 4: VALIDATE MODEL (9 tests, 100.0% pass rate)
✅ STEP 5: SAVE VALIDATION REPORT (77.5 KB HTML)
✅ STEP 6: COMPARE MODELS (best: fe by R²)
✅ STEP 7: SAVE COMPARISON REPORT (53.3 KB HTML)
```

---

## Supported Model Types

Use any of these in `fit_model()`:

| Full Name | Alias | Description |
|-----------|-------|-------------|
| `'pooled_ols'` | `'pooled'` | Pooled OLS (no effects) |
| `'fixed_effects'` | `'fe'` | Fixed Effects (within estimator) |
| `'random_effects'` | `'re'` | Random Effects (GLS estimator) |

Example with aliases:
```python
experiment.fit_model('fe', name='model1')  # Same as 'fixed_effects'
experiment.fit_model('re', name='model2')  # Same as 'random_effects'
```

---

## HTML Reports

All reports are **self-contained** HTML files with:
- ✅ Embedded CSS (no external stylesheets)
- ✅ Embedded JavaScript (works offline)
- ✅ Interactive Plotly charts
- ✅ Responsive design (mobile-friendly)
- ✅ Professional themes

**Themes available**:
- `'professional'` (default) - Clean, business-ready
- `'academic'` - Publication-ready
- `'presentation'` - Slide-ready

```python
val_result.save_html(
    'report.html',
    test_type='validation',
    theme='professional',
    title='My Validation Report'
)
```

---

## Factory Methods

Both result containers have factory methods for convenience:

### ValidationResult.from_model_results()

```python
# Instead of:
validation = fe_results.validate()
val_result = pb.ValidationResult(validation, fe_results)

# Do this:
val_result = pb.ValidationResult.from_model_results(
    fe_results,
    alpha=0.05,
    tests='default'
)
```

### ComparisonResult.from_experiment()

```python
# Instead of:
models = {name: experiment.get_model(name) for name in experiment.list_models()}
comp_result = pb.ComparisonResult(models)

# Do this:
comp_result = pb.ComparisonResult.from_experiment(experiment)
```

---

## Best Practices

### 1. Use Meaningful Names
```python
# Good
experiment.fit_model('fixed_effects', name='baseline')
experiment.fit_model('fixed_effects', name='with_controls', cov_type='robust')

# Avoid
experiment.fit_model('fixed_effects', name='model1')
experiment.fit_model('fixed_effects', name='model2')
```

### 2. Validate Before Comparing
```python
# Always validate your preferred model first
val_result = experiment.validate_model('fe')

# Then compare if validation looks good
if val_result.pass_rate > 0.7:
    comp_result = experiment.compare_models()
```

### 3. Check Failed Tests
```python
val_result = experiment.validate_model('fe')
if val_result.failed_tests:
    print("Issues detected:")
    for test in val_result.failed_tests:
        print(f"  - {test}")
```

### 4. Use best_model() Appropriately
```python
# Maximize R²
best_fit = comp_result.best_model('rsquared')

# Minimize AIC (penalizes complexity)
best_parsimony = comp_result.best_model('aic', prefer_lower=True)
```

---

## Architecture

```
PanelExperiment (Factory + Storage)
├── fit_model() → creates and stores model
├── fit_all_models() → fits multiple at once
├── validate_model() → creates ValidationResult
└── compare_models() → creates ComparisonResult

ValidationResult (Result Container)
├── Wraps ValidationReport
├── Properties: total_tests, pass_rate, etc.
├── save_html() → HTML report
└── save_json() → JSON export

ComparisonResult (Result Container)
├── Stores multiple models
├── best_model() → find optimal model
├── save_html() → HTML report
└── save_json() → JSON export
```

---

## Documentation

- **Complete Overview**: `COMPLETE_PROJECT_SUMMARY.md`
- **Quick Reference**: `PROJECT_STATUS.md`
- **Integration Details**: `INTEGRATION_COMPLETE.md`
- **Sprint Reviews**: `sprint*_review.md`
- **Working Example**: `examples/complete_workflow_example.py`

---

## Support

**Questions?** Check the documentation files above.

**Issues?** All tests are passing - if you encounter problems, ensure:
1. PanelBox is installed: `poetry install`
2. Dependencies are up to date
3. You're using the public API: `import panelbox as pb`

**Examples?** See:
- `examples/complete_workflow_example.py` - Complete workflow
- `test_sprint4_complete_workflow.py` - Test with assertions
- All `test_*.py` files serve as usage examples

---

**Status**: ✅ **Production Ready**

Developed across 4 sprints | 53 story points | 113% velocity
