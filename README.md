<div align="center">
  <img src="https://raw.githubusercontent.com/PanelBox-Econometrics-Model/panelbox/main/docs/assets/images/logo.svg" alt="PanelBox Logo" width="400">

  <h1>PanelBox</h1>

  <p><strong>Panel Data Econometrics in Python</strong></p>

[![CI](https://github.com/PanelBox-Econometrics-Model/panelbox/actions/workflows/tests.yml/badge.svg)](https://github.com/PanelBox-Econometrics-Model/panelbox/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/PanelBox-Econometrics-Model/panelbox/branch/main/graph/badge.svg)](https://codecov.io/gh/PanelBox-Econometrics-Model/panelbox)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI version](https://badge.fury.io/py/panelbox.svg)](https://badge.fury.io/py/panelbox)
[![Python versions](https://img.shields.io/pypi/pyversions/panelbox)](https://pypi.org/project/panelbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Development Status](https://img.shields.io/badge/development%20status-beta-orange)

</div>

---

PanelBox provides comprehensive tools for panel data econometrics, bringing Stata's `xtabond2` and R's `plm` capabilities to Python with modern, user-friendly APIs.

## Features

### ✅ Static Panel Models
- **Pooled OLS**: Standard OLS with panel data
- **Fixed Effects**: Control for time-invariant heterogeneity
- **Random Effects**: GLS estimation with random effects
- **Hausman Test**: Test for endogeneity of random effects

### ✅ Dynamic Panel GMM (v0.2.0)
- **Difference GMM**: Arellano-Bond (1991) estimator
- **System GMM**: Blundell-Bond (1998) estimator
- **Robust to unbalanced panels**: Smart instrument selection
- **Windmeijer correction**: Finite-sample standard error correction
- **Comprehensive diagnostics**:
  - Hansen J-test for overidentification
  - Sargan test
  - Arellano-Bond AR tests
  - Instrument ratio monitoring

### 🔧 Panel-Specific Features
- **Unbalanced panel support**: Handles missing observations gracefully
- **Time effects**: Time dummies, linear trends, or custom time controls
- **Clustered standard errors**: Robust inference
- **Instrument generation**: Automatic GMM-style and IV-style instruments
- **Collapse option**: Avoids instrument proliferation (Roodman 2009)

### 📊 Publication-Ready Output
- **Summary tables**: Professional regression output
- **Diagnostic tests**: Comprehensive specification testing
- **LaTeX export**: Ready for academic papers
- **Warnings system**: Guides users to correct specifications

## Installation

```bash
pip install panelbox
```

Or install from source:

```bash
git clone https://github.com/PanelBox-Econometrics-Model/panelbox.git
cd panelbox
pip install -e .
```

## Quick Start

### 🎯 Experiment Pattern (Recommended - v0.6.0+)

```python
import panelbox as pb
import pandas as pd

# Load your panel data
data = pd.read_csv('panel_data.csv')

# Create experiment
experiment = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

# Fit multiple models at once
experiment.fit_all_models(names=['pooled', 'fe', 're'])

# Validate model specification
validation_result = experiment.validate_model('fe')
print(validation_result.summary())
validation_result.save_html('validation_report.html', test_type='validation')

# Compare models and select best one
comparison_result = experiment.compare_models(['pooled', 'fe', 're'])
print(f"Best model: {comparison_result.best_model}")
comparison_result.save_html('comparison_report.html', test_type='comparison')

# Analyze residuals (v0.7.0)
residual_result = experiment.analyze_residuals('fe')
print(residual_result.summary())

# Check diagnostic tests
stat, pvalue = residual_result.shapiro_test
print(f"Shapiro-Wilk normality test: p={pvalue:.4f}")

dw = residual_result.durbin_watson
print(f"Durbin-Watson statistic: {dw:.4f}")

residual_result.save_html('residuals_report.html', test_type='residuals')

# Generate master report with all sub-reports (NEW in v0.8.0!)
experiment.save_master_report(
    'master_report.html',
    theme='professional',
    reports=[
        {'type': 'validation', 'title': 'Model Validation',
         'description': 'Specification tests', 'file_path': 'validation_report.html'},
        {'type': 'comparison', 'title': 'Model Comparison',
         'description': 'Compare pooled, FE, RE', 'file_path': 'comparison_report.html'},
        {'type': 'residuals', 'title': 'Residual Diagnostics',
         'description': 'Diagnostic tests', 'file_path': 'residuals_report.html'}
    ]
)
```

### Static Panel Models (Traditional API)

```python
import panelbox as pb
import pandas as pd

# Load your panel data
data = pd.read_csv('panel_data.csv')

# Fixed Effects model
fe = pb.FixedEffects(
    formula="invest ~ value + capital",
    data=data,
    entity_col="firm",
    time_col="year"
)
results = fe.fit(cov_type='clustered')
print(results.summary())

# Hausman test
hausman = pb.HausmanTest(fe_results, re_results)
print(hausman)
```

### Dynamic Panel GMM

```python
from panelbox import DifferenceGMM

# Arellano-Bond employment equation
gmm = DifferenceGMM(
    data=data,
    dep_var='employment',
    lags=1,
    id_var='firm',
    time_var='year',
    exog_vars=['wages', 'capital', 'output'],
    time_dummies=False,
    collapse=True,
    two_step=True,
    robust=True
)

results = gmm.fit()
print(results.summary())

# Check specification tests
print(f"Hansen J p-value: {results.hansen_j.pvalue:.3f}")
print(f"AR(2) p-value: {results.ar2_test.pvalue:.3f}")
```

### System GMM (Blundell-Bond)

```python
from panelbox import SystemGMM

# System GMM for persistent series
sys_gmm = SystemGMM(
    data=data,
    dep_var='y',
    lags=1,
    id_var='id',
    time_var='year',
    exog_vars=['x1', 'x2'],
    collapse=True,
    two_step=True,
    robust=True
)

results = sys_gmm.fit()
print(results.summary())

# Compare efficiency with Difference GMM
print(f"Instrument count: {results.n_instruments}")
print(f"Instrument ratio: {results.instrument_ratio:.3f}")
```

## 📖 Best Practices for GMM

### Recommended: Use `collapse=True`

Following Roodman (2009), we **strongly recommend** using collapsed instruments:

```python
# ✅ RECOMMENDED
gmm = DifferenceGMM(..., collapse=True)
```

**Why collapse instruments?**
- ✅ **Better numerical stability** - Avoids ill-conditioned matrices
- ✅ **Reduces overfitting** - Fewer instruments mean less overfitting risk
- ✅ **Improves finite-sample properties** - Better performance with limited data
- ✅ **Grows as O(T) not O(T²)** - Scales better with time periods

**When you use `collapse=False`:**
- ⚠️ You'll see a detailed warning message
- ⚠️ May encounter numerical instability warnings
- ⚠️ Works but requires careful interpretation

See `examples/gmm/unbalanced_panel_guide.py` for detailed guidance.

**Reference:** Roodman, D. (2009). "How to do xtabond2: An introduction to difference and system GMM in Stata." *The Stata Journal*, 9(1), 86-136.

## Key Advantages

### 1. Handles Unbalanced Panels Gracefully

Unlike some implementations, PanelBox:
- ✅ Automatically detects unbalanced panel structure
- ✅ Warns about problematic specifications
- ✅ Intelligently selects instruments based on data availability
- ✅ Provides clear guidance when specifications fail

```python
# Smart warnings for unbalanced panels
gmm = DifferenceGMM(data=unbalanced_data, ...)
# UserWarning: Unbalanced panel detected (20% balanced) with 8 time dummies.
# This may result in very few observations being retained.
#
# Recommendations:
#   1. Set time_dummies=False and add a linear trend
#   2. Use only subset of key time dummies
#   3. Ensure collapse=True
```

### 2. Comprehensive Specification Tests

All GMM models include:
- **Hansen J-test**: Overidentification test with interpretation
- **Sargan test**: Alternative overidentification test
- **AR(1) and AR(2) tests**: Serial correlation in first-differenced errors
- **Instrument ratio**: n_instruments / n_groups (should be < 1.0)

### 3. Follows Best Practices

Based on Roodman (2009) "How to do xtabond2":
- Collapse option to avoid instrument proliferation
- Windmeijer (2005) standard error correction
- Automatic lag selection based on data availability
- Clear warnings for problematic specifications

### 4. Rich Documentation

- 📚 Comprehensive [tutorial](https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/docs/gmm/tutorial.md)
- 📖 [Interpretation guide](https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/docs/gmm/interpretation_guide.md) with decision tables
- 💡 [Example scripts](https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/examples/gmm/) for common use cases
- 🔬 [Unbalanced panel guide](https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/examples/gmm/unbalanced_panel_guide.py)

## Learning Resources

### 📚 Interactive Tutorials (NEW!)

We've created comprehensive Jupyter notebook tutorials to help you master panel data econometrics:

**[Getting Started Guide](examples/GETTING_STARTED.md)** - Your roadmap to learning PanelBox

#### Module 1: Fundamentals (3.5-4.5 hours)
Perfect for beginners! Learn the core concepts:
- [01 - Introduction to Panel Data](examples/tutorials/01_fundamentals/01_introduction_panel_data.ipynb) - Loading and transforming panel data
- [02 - Model Specification with Formulas](examples/tutorials/01_fundamentals/02_formulas_specification.ipynb) - R-style formula syntax
- [03 - Estimation and Results Interpretation](examples/tutorials/01_fundamentals/03_estimation_interpretation.ipynb) - Fitting models and understanding output
- [04 - Spatial Fundamentals](examples/tutorials/01_fundamentals/04_spatial_fundamentals.ipynb) - Creating spatial weight matrices

**More modules coming soon:**
- Module 2: Classical Estimators (Fixed Effects, Random Effects)
- Module 3: Dynamic GMM (Arellano-Bond)
- Module 4: Spatial Panel Models

See the [tutorials directory](examples/tutorials/) for the complete learning path.

### 💡 Example Scripts

See the [examples directory](https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/examples) for:

- **OLS vs FE vs GMM comparison**: Demonstrating bias in each estimator
- **Firm growth model**: Intermediate example with error handling
- **Production function estimation**: Advanced example with simultaneity bias
- **Unbalanced panel guide**: Practical solutions for unbalanced data

## Comparison with Other Packages

| Feature | PanelBox | linearmodels | pyfixest | statsmodels |
|---------|----------|--------------|----------|-------------|
| Difference GMM | ✅ | ❌ | ❌ | ❌ |
| System GMM | ✅ | ❌ | ❌ | ❌ |
| Unbalanced panels | ✅ Smart | ⚠️ Basic | ⚠️ Basic | ⚠️ Basic |
| Collapse option | ✅ | ❌ | ❌ | ❌ |
| Windmeijer correction | ✅ | ❌ | ❌ | ❌ |
| User warnings | ✅ Proactive | ⚠️ Reactive | ⚠️ Reactive | ⚠️ Reactive |
| Documentation | ✅ Rich | ✅ Good | ✅ Good | ✅ Good |

## Requirements

- Python >= 3.9
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- SciPy >= 1.10.0
- statsmodels >= 0.14.0
- patsy >= 0.5.3

## Validation

PanelBox has been validated against:
- ✅ Arellano-Bond (1991) employment equation
- ✅ Stata xtabond2 (with appropriate specifications)
- ✅ Multiple synthetic datasets with known DGP

See [validation directory](https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/validation) for details.

## Citation

If you use PanelBox in your research, please cite:

```bibtex
@software{panelbox2026,
  author = {Haase, Gustavo and Dourado, Paulo},
  title = {PanelBox: Panel Data Econometrics in Python},
  year = {2026},
  version = {0.7.0},
  url = {https://github.com/PanelBox-Econometrics-Model/panelbox}
}
```

## References

### Implemented Methods

- **Arellano, M., & Bond, S. (1991)**. "Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations." *Review of Economic Studies*, 58(2), 277-297.

- **Blundell, R., & Bond, S. (1998)**. "Initial Conditions and Moment Restrictions in Dynamic Panel Data Models." *Journal of Econometrics*, 87(1), 115-143.

- **Windmeijer, F. (2005)**. "A Finite Sample Correction for the Variance of Linear Efficient Two-step GMM Estimators." *Journal of Econometrics*, 126(1), 25-51.

- **Roodman, D. (2009)**. "How to do xtabond2: An Introduction to Difference and System GMM in Stata." *Stata Journal*, 9(1), 86-136.

### Textbooks

- **Baltagi, B. H. (2021)**. *Econometric Analysis of Panel Data* (6th ed.). Springer.
- **Wooldridge, J. M. (2010)**. *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/LICENSE) file for details.

## Support

- 📫 Issues: [GitHub Issues](https://github.com/PanelBox-Econometrics-Model/panelbox/issues)
- 📖 Documentation: [GitHub Wiki](https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/docs)
- 💬 Discussions: [GitHub Discussions](https://github.com/PanelBox-Econometrics-Model/panelbox/discussions)

## Changelog

See [CHANGELOG.md](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/CHANGELOG.md) for complete version history.

### Latest Release: v0.8.0 (2026-02-08)

**🎯 Test Runners & Master Report**

**Test Runners (NEW in v0.8.0):**
- ✨ **ValidationTest** - Configurable test runner with 3 presets (quick, basic, full)
- ✨ **ComparisonTest** - Multi-model comparison with automatic metrics extraction
- ✨ Clean one-liner APIs for running tests on any fitted model
- ✨ Integrates seamlessly with PanelExperiment workflow

**Master Report System (NEW in v0.8.0):**
- ✨ **Master HTML Report** - Comprehensive overview of entire experiment
- ✨ **Experiment Overview** - Formula, observations, entities, time periods
- ✨ **Models Summary** - Grid with key metrics (R², AIC, BIC) for all fitted models
- ✨ **Reports Navigation** - Click through to validation, comparison, and residuals reports
- ✨ **Quick Start Guide** - Embedded code examples in the report
- ✨ **Responsive Design** - Professional layouts for all screen sizes

### Previous Release: v0.7.0 (2026-02-08)

**🎯 Advanced Features & Production Polish**

**Experiment Pattern & Result Containers:**
- ✨ **PanelExperiment** - Factory-based model management with automatic storage
- ✨ **ValidationResult** - Container for validation test results with HTML/JSON export
- ✨ **ComparisonResult** - Container for model comparison with best model selection
- ✨ **ResidualResult** (NEW!) - Container for residual diagnostics with 4 tests
- ✨ One-liner workflows: `validate_model()`, `compare_models()`, `analyze_residuals()`

**Comprehensive Visualization System:**
- ✨ 35+ interactive Plotly charts for panel data analysis
- ✨ 3 professional themes (Professional, Academic, Presentation)
- ✨ Interactive HTML reports with embedded charts
- ✨ Multiple export formats (HTML, JSON, PNG, SVG, PDF)
- ✨ High-level convenience APIs for common visualizations

**Residual Diagnostics (NEW in v0.7.0):**
- ✨ **Shapiro-Wilk test** - Test for normality of residuals
- ✨ **Jarque-Bera test** - Alternative normality test
- ✨ **Durbin-Watson statistic** - Autocorrelation detection
- ✨ **Ljung-Box test** - Serial correlation up to 10 lags
- ✨ Summary statistics (mean, std, skewness, kurtosis)
- ✨ Professional summary output with interpretation guidelines

**Static Panel Models:**
- ✨ Pooled OLS, Fixed Effects, Random Effects, Between, First Differences
- ✨ 8 types of robust standard errors (HC0-HC3, clustered, Driscoll-Kraay, Newey-West, PCSE)
- ✨ Comprehensive specification tests

**Dynamic Panel GMM:**
- ✨ Difference GMM (Arellano-Bond 1991)
- ✨ System GMM (Blundell-Bond 1998)
- ✨ Smart instrument selection for unbalanced panels
- ✨ Windmeijer finite-sample correction

**Advanced Features:**
- ✨ Bootstrap inference (4 methods: pairs, wild, block, residual)
- ✨ Sensitivity analysis (leave-one-out, subset stability)
- ✨ 20+ validation tests (unit root, cointegration, diagnostics)
- ✨ Professional report generation (HTML, Markdown, LaTeX)

**Quality & Performance:**
- 🔧 Complete result container trilogy (Validation, Comparison, Residual)
- 🔧 Zero console warnings
- 🔧 16 new tests for ResidualResult (85% coverage)
- 🔧 HTML reports with embedded interactive charts
- ✅ Production-ready package

---

**Made with ❤️ for econometricians and researchers**
