# API Reference

Complete API documentation for PanelBox.

## Overview

PanelBox provides a comprehensive API for panel data econometrics organized into the following modules:

### 📊 [Static Models](models.md)

Panel models without dynamics:

- **PooledOLS**: Pooled Ordinary Least Squares
- **FixedEffects**: Fixed Effects (Within) estimator
- **RandomEffects**: Random Effects (GLS) estimator
- **Between**: Between estimator (entity means)
- **FirstDifferences**: First Differences estimator

### 🔄 [GMM Models](gmm.md)

Dynamic panel GMM estimators:

- **DifferenceGMM**: Arellano-Bond Difference GMM (1991)
- **SystemGMM**: Blundell-Bond System GMM (1998)

### 📈 [Results](results.md)

Results container class:

- **PanelResults**: Estimation results with summary, tests, export

### ✅ [Validation](validation.md)

Diagnostic and specification tests:

- **HausmanTest**: Fixed Effects vs Random Effects
- **BreuschPaganLM**: Random effects test
- **BreuschPaganTest**: Heteroskedasticity test
- **WooldridgeTest**: Serial correlation test
- **Hansen J, Sargan**: GMM overidentification tests
- **AR tests**: GMM serial correlation tests

### 📦 [Datasets](datasets.md)

Example datasets for learning and testing:

- **load_grunfeld()**: Grunfeld investment data
- **load_abdata()**: Arellano-Bond employment data
- **list_datasets()**: List available datasets
- **get_dataset_info()**: Dataset information

### 📋 [Report](report.md)

Reporting and export utilities:

- **PanelExperiment**: High-level API for panel data analysis (NEW in v0.8.0)
- **ValidationResult**: Container for validation test results with HTML export (NEW in v0.8.0)
- **ComparisonResult**: Container for model comparison with HTML export (NEW in v0.8.0)
- **ResidualResult**: Container for residual diagnostics with HTML export (NEW in v0.7.0)
- **ValidationTest**: Test runner with configurable presets (NEW in v0.8.0)
- **ComparisonTest**: Multi-model comparison runner (NEW in v0.8.0)
- **to_latex()**: Export to LaTeX tables
- **summary()**: Formatted summary tables
- **save_html()**: Generate interactive HTML reports (NEW in v0.8.0)
- **save_master_report()**: Generate master report with navigation (NEW in v0.8.0)

## Quick Links

| Topic | API Documentation |
|-------|-------------------|
| Estimate Fixed Effects | [FixedEffects](models.md#fixedeffects) |
| Estimate Random Effects | [RandomEffects](models.md#randomeffects) |
| Run Hausman Test | [HausmanTest](validation.md#hausmantest) |
| Estimate Difference GMM | [DifferenceGMM](gmm.md#differencegmm) |
| Estimate System GMM | [SystemGMM](gmm.md#systemgmm) |
| Check Hansen J Test | [Results](results.md#hansen-j-test) |
| Load Example Data | [load_grunfeld](datasets.md#load_grunfeld) |
| Export to LaTeX | [to_latex](report.md#to_latex) |
| Create Experiment | [PanelExperiment](report.md#panelexperiment) |
| Validate Model | [ValidationTest](report.md#validationtest) |
| Compare Models | [ComparisonTest](report.md#comparisontest) |
| Generate HTML Report | [save_html](report.md#save_html) |
| Master Report | [save_master_report](report.md#save_master_report) |

## Usage Patterns

### Basic Workflow

```python
import panelbox as pb

# 1. Load data
data = pb.load_grunfeld()

# 2. Create model
model = pb.FixedEffects(
    formula="invest ~ value + capital",
    data=data,
    entity_col="firm",
    time_col="year"
)

# 3. Fit model
results = model.fit(cov_type='clustered')

# 4. View results
print(results.summary())

# 5. Export
results.to_latex("table1.tex")
```

### Advanced Workflow (GMM)

```python
# 1. Load data
data = pb.load_grunfeld()

# 2. Create GMM model
gmm = pb.SystemGMM(
    data=data,
    dep_var='invest',
    lags=1,
    exog_vars=['value', 'capital'],
    id_var='firm',
    time_var='year',
    collapse=True,
    robust=True
)

# 3. Fit
results = gmm.fit()

# 4. Check diagnostics
print(f"Hansen J: {results.hansen_j.pvalue:.3f}")
print(f"AR(2): {results.ar2_test.pvalue:.3f}")

# 5. If tests pass, view results
if results.hansen_j.pvalue > 0.10 and results.ar2_test.pvalue > 0.10:
    print(results.summary())
```

### Complete Workflow with Reports (NEW in v0.8.0)

```python
import panelbox as pb

# 1. Load data
data = pb.load_grunfeld()

# 2. Create experiment
experiment = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

# 3. Fit multiple models
experiment.fit_model('pooled_ols', name='ols')
experiment.fit_model('fixed_effects', name='fe')
experiment.fit_model('random_effects', name='re')

# 4. Generate validation report
validation = experiment.validate_model('fe', config='full')
validation.save_html('validation.html', test_type='validation', theme='professional')

# 5. Generate comparison report
comparison = experiment.compare_models(['ols', 'fe', 're'])
comparison.save_html('comparison.html', test_type='comparison', theme='professional')

# 6. Generate residual diagnostics
residuals = experiment.analyze_residuals('fe')
residuals.save_html('residuals.html', test_type='residuals', theme='professional')

# 7. Generate master report
experiment.save_master_report('master.html', theme='professional', reports=[
    {'type': 'validation', 'title': 'Model Validation',
     'description': 'Specification tests', 'file_path': 'validation.html'},
    {'type': 'comparison', 'title': 'Model Comparison',
     'description': 'Compare OLS, FE, RE', 'file_path': 'comparison.html'},
    {'type': 'residuals', 'title': 'Residual Diagnostics',
     'description': 'Diagnostic plots', 'file_path': 'residuals.html'}
])

# Open master.html in your browser!
```

## Module Organization

```
panelbox/
├── models/
│   ├── static/
│   │   ├── pooled_ols.py      → PooledOLS
│   │   ├── fixed_effects.py   → FixedEffects
│   │   ├── random_effects.py  → RandomEffects
│   │   ├── between.py         → Between
│   │   └── first_differences.py → FirstDifferences
│   └── base.py                → PanelModel (base class)
├── gmm/
│   ├── difference_gmm.py      → DifferenceGMM
│   └── system_gmm.py          → SystemGMM
├── core/
│   └── results.py             → PanelResults
├── validation/
│   ├── specification/
│   │   ├── hausman.py         → HausmanTest
│   │   └── breusch_pagan_lm.py → BreuschPaganLM
│   └── ...
├── datasets/
│   └── load.py                → load_grunfeld, etc.
└── report/
    └── latex.py               → LaTeX export
```

## Next Steps

- Browse specific API documentation in the navigation
- See [Tutorials](../tutorials/01_getting_started.md) for hands-on examples
- Check [How-To Guides](../how-to/choose_model.md) for task-oriented help
