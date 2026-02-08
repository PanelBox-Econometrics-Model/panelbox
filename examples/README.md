# PanelBox Examples

This directory contains comprehensive examples demonstrating PanelBox's capabilities.

## 🎯 Recommended Starting Point

### **Complete Workflow (v0.7.0)**
**File**: `complete_workflow_v07.py`

**What it demonstrates**:
- ✅ Complete analysis pipeline with PanelExperiment
- ✅ All three result containers:
  - ValidationResult - Model specification tests
  - ComparisonResult - Model comparison and selection
  - ResidualResult - Residual diagnostics (NEW in v0.7.0!)
- ✅ HTML report generation with interactive charts
- ✅ JSON export for programmatic analysis

**Run it**:
```bash
poetry run python examples/complete_workflow_v07.py
```

**Output files**:
- `validation_report_v07.html` - Comprehensive validation diagnostics
- `comparison_report_v07.html` - Model comparison with metrics
- `residuals_report_v07.html` - Residual diagnostics with 4 tests
- JSON files for programmatic analysis

---

## 📚 Examples by Category

### 1. Getting Started

#### `basic_usage.py`
Basic usage of static panel models (Pooled OLS, Fixed Effects, Random Effects).

**Run it**:
```bash
poetry run python examples/basic_usage.py
```

---

### 2. Jupyter Notebooks

Interactive tutorials in `jupyter/` directory:

#### `00_getting_started.ipynb`
Introduction to PanelBox with step-by-step walkthrough.

#### `01_static_models_complete.ipynb`
Comprehensive guide to static panel models.

#### `02_dynamic_gmm_complete.ipynb`
Dynamic GMM estimation (Difference GMM and System GMM).

#### `03_validation_complete.ipynb`
Model validation and specification tests.

#### `04_robust_inference.ipynb`
Robust standard errors and bootstrap inference.

#### `05_report_generation.ipynb`
HTML and LaTeX report generation.

#### `06_visualization_reports.ipynb`
Interactive visualizations and chart customization (NEW in v0.5.0!).

#### `07_real_world_case_study.ipynb`
Real-world application with complete analysis.

#### `08_html_reports_complete_guide.ipynb`
Comprehensive guide to HTML report generation.

#### `09_residual_diagnostics_v07.ipynb` ⭐ **NEW in v0.7.0!**
**Complete guide to residual diagnostics**:
- ResidualResult container usage
- 4 diagnostic tests (Shapiro-Wilk, Jarque-Bera, Durbin-Watson, Ljung-Box)
- Summary statistics and interpretation
- HTML report generation
- Outlier detection with standardized residuals

**Run notebooks**:
```bash
jupyter lab examples/jupyter/
```

---

### 3. Workflow Examples

#### `complete_workflow_v07.py` ⭐ **RECOMMENDED**
**Complete workflow with v0.7.0 features**:
- PanelExperiment for model management
- ValidationResult, ComparisonResult, ResidualResult
- HTML reports with interactive charts
- JSON export

#### `complete_workflow_example.py`
Alternative complete workflow example.

---

### 4. Report Generation

#### `report_generation_example.py`
Advanced HTML report generation with custom templates.

#### `simple_report_example.py`
Minimal example of report generation.

#### `minimal_report_example.py`
Bare-bones report generation.

---

### 5. Validation & Testing

#### `validation_example.py`
Comprehensive model validation with diagnostic tests.

#### Unit Root Tests:
- `llc_unit_root_example.py` - LLC test for unit roots
- `ips_unit_root_example.py` - IPS test for unit roots
- `fisher_unit_root_example.py` - Fisher test for unit roots

---

### 6. Utilities

#### `serialization_example.py`
Save and load model results.

#### `gallery_generator.py`
Generate visualization galleries.

---

## 📖 Feature-Specific Examples

### Experiment Pattern (v0.6.0+)

The **Experiment Pattern** is the recommended way to use PanelBox:

```python
import panelbox as pb

# Create experiment
experiment = pb.PanelExperiment(
    data=data,
    formula="y ~ x1 + x2",
    entity_col="firm",
    time_col="year"
)

# Fit multiple models at once
experiment.fit_all_models(names=['pooled', 'fe', 're'])

# Validate model
validation_result = experiment.validate_model('fe')

# Compare models
comparison_result = experiment.compare_models(['pooled', 'fe', 're'])

# Analyze residuals (NEW in v0.7.0!)
residual_result = experiment.analyze_residuals('fe')

# Generate HTML reports
validation_result.save_html('validation.html', test_type='validation')
comparison_result.save_html('comparison.html', test_type='comparison')
residual_result.save_html('residuals.html', test_type='residuals')
```

**Example**: `complete_workflow_v07.py`

---

### Residual Diagnostics (v0.7.0)

**NEW** in v0.7.0: Comprehensive residual diagnostics with 4 tests:

```python
# Analyze residuals
residual_result = experiment.analyze_residuals('fe')

# Print summary
print(residual_result.summary())

# Access individual tests
stat, pvalue = residual_result.shapiro_test  # Normality
stat, pvalue = residual_result.jarque_bera   # Normality
dw = residual_result.durbin_watson            # Autocorrelation
stat, pvalue = residual_result.ljung_box      # Serial correlation

# Summary statistics
mean = residual_result.mean
std = residual_result.std
skewness = residual_result.skewness
kurtosis = residual_result.kurtosis

# Generate HTML report
residual_result.save_html('residuals.html', test_type='residuals')
```

**Example**: `complete_workflow_v07.py` (Step 6)

---

### Visualization System (v0.5.0+)

35+ interactive Plotly charts with 3 professional themes:

```python
from panelbox.visualization import create_validation_charts

# Create validation charts
charts = create_validation_charts(validation_result.to_dict())

# Export to HTML
export_chart(charts['test_overview'], 'test_overview.html')
```

**Example**: `jupyter/06_visualization_reports.ipynb`

---

### Model Validation

Comprehensive specification tests:

```python
validation_result = experiment.validate_model('fe')

# Access test results
for test_name, test_result in validation_result.tests.items():
    print(f"{test_name}: p-value = {test_result.pvalue:.4f}")
```

**Example**: `validation_example.py`, `jupyter/03_validation_complete.ipynb`

---

### Model Comparison

Compare multiple models and select the best:

```python
comparison_result = experiment.compare_models(['pooled', 'fe', 're'])

# Get best model
best = comparison_result.best_model('aic', prefer_lower=True)
print(f"Best model by AIC: {best}")

# Compare all models
print(comparison_result.summary())
```

**Example**: `complete_workflow_v07.py` (Step 5)

---

## 🚀 Quick Start

1. **Install PanelBox**:
   ```bash
   pip install panelbox
   ```

2. **Run the recommended example**:
   ```bash
   poetry run python examples/complete_workflow_v07.py
   ```

3. **Explore the generated HTML reports** in your browser

4. **Try the Jupyter notebooks**:
   ```bash
   jupyter lab examples/jupyter/
   ```

---

## 📊 Datasets Used

### Grunfeld Investment Data
Classic panel dataset used in most examples. Included in PanelBox:

```python
data = pb.load_grunfeld()
```

**Variables**:
- `invest`: Gross investment
- `value`: Market value
- `capital`: Capital stock
- `firm`: Firm identifier (10 firms)
- `year`: Year (1935-1954, 20 years)

---

## 🎯 Learning Path

### Beginner
1. `basic_usage.py` - Understand static models
2. `jupyter/00_getting_started.ipynb` - Interactive tutorial
3. `complete_workflow_v07.py` - Complete analysis pipeline

### Intermediate
4. `jupyter/01_static_models_complete.ipynb` - Deep dive into static models
5. `jupyter/03_validation_complete.ipynb` - Model validation
6. `jupyter/04_robust_inference.ipynb` - Robust standard errors

### Advanced
7. `jupyter/02_dynamic_gmm_complete.ipynb` - GMM estimation
8. `jupyter/06_visualization_reports.ipynb` - Custom visualizations
9. `jupyter/07_real_world_case_study.ipynb` - Real applications

---

## 📝 Example Output

When you run `complete_workflow_v07.py`, you'll see:

```
================================================================================
PanelBox v0.7.0 - Complete Workflow Example
================================================================================

Step 1: Loading Grunfeld investment dataset...
Dataset shape: (200, 5)
Panel structure: 10 firms, 20 years

Step 2: Creating PanelExperiment...
✓ Experiment created successfully

Step 3: Fitting multiple panel models...
✓ Models fitted:
  - pooled: R² = 0.8090
  - fe: R² = 0.7899
  - re: R² = 0.8003

Step 4: Validating Fixed Effects model...
✓ ValidationResult created

Step 5: Comparing all models...
✓ ComparisonResult created
Best model by AIC: fe
Best model by Adjusted R²: pooled

Step 6: Analyzing residuals (NEW in v0.7.0!)...
✓ ResidualResult created

Shapiro-Wilk Test: p-value = 0.0234
Durbin-Watson Statistic: 1.2456
...

Workflow Complete!
================================================================================
```

---

## 🆘 Getting Help

- **Documentation**: https://github.com/PanelBox-Econometrics-Model/panelbox
- **Issues**: https://github.com/PanelBox-Econometrics-Model/panelbox/issues
- **Discussions**: https://github.com/PanelBox-Econometrics-Model/panelbox/discussions

---

## 🎉 What's New

### v0.7.0 (2026-02-08)
- ✨ **ResidualResult** - Complete residual diagnostics with 4 tests
- ✨ `analyze_residuals()` method in PanelExperiment
- ✨ All 35 charts now correctly registered
- ✨ Zero console warnings
- ✨ HTML reports with embedded interactive charts

### v0.6.0 (2026-02-08)
- ✨ **PanelExperiment** - Factory-based model management
- ✨ **ValidationResult** & **ComparisonResult** containers
- ✨ One-liner workflows: `validate_model()`, `compare_models()`

### v0.5.0 (2026-02-08)
- ✨ 35+ interactive Plotly charts
- ✨ 3 professional themes
- ✨ HTML report generation system

---

**Made with ❤️ for econometricians and researchers**
