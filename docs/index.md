<div align="center" style="background: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
  <img src="assets/images/logo.svg" alt="PanelBox Logo" width="600" style="max-width: 100%; height: auto;">
</div>

<div align="center">
  <h1>PanelBox</h1>
  <p style="font-size: 1.2em;"><strong>Python library for panel data econometrics</strong></p>

  <p>
    <a href="https://pypi.org/project/panelbox/"><img src="https://img.shields.io/pypi/v/panelbox.svg" alt="PyPI version"></a>
    <a href="https://pypi.org/project/panelbox/"><img src="https://img.shields.io/pypi/pyversions/panelbox.svg" alt="Python versions"></a>
    <img src="https://img.shields.io/badge/status-stable-brightgreen.svg" alt="Development Status">
    <a href="https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  </p>
</div>

---

## Overview

PanelBox is a comprehensive Python library for **panel data econometrics**, providing implementations of:

- **Static panel models**: Pooled OLS, Fixed Effects, Random Effects, Between, First Differences
- **Dynamic panel GMM**: Difference GMM (Arellano-Bond 1991), System GMM (Blundell-Bond 1998)
- **Robust inference**: 8+ types of standard errors (clustered, HAC, heteroskedasticity-robust)
- **Diagnostic tests**: Hansen J, Sargan, AR tests, Hausman, Wooldridge, Breusch-Pagan
- **Validation**: Cross-validated against Stata's `xtabond2` and R's `plm`

**Design Philosophy:**

- 🎯 **Ease of use**: R-style formulas, pandas-friendly API
- 🔬 **Academic rigor**: Implementations match published econometrics papers
- ⚡ **Performance**: Numba-optimized critical paths (up to 348x speedup)
- 📊 **Publication-ready**: LaTeX export, formatted output tables

---

## Quick Example

```python
import panelbox as pb

# Load example data
data = pb.load_grunfeld()

# Estimate System GMM
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

results = gmm.fit()
print(results.summary())

# Check diagnostics
print(f"Hansen J p-value: {results.hansen_j.pvalue:.3f}")  # 0.185
print(f"AR(2) p-value: {results.ar2_test.pvalue:.3f}")      # 0.412
```

**Output:**
```
================================================================================
                       System GMM Estimation Results
================================================================================
Dependent Variable:              invest        Hansen J p-value:           0.185
Model:                       System GMM        AR(2) p-value:              0.412
No. Observations:                   180        Instrument ratio:             2.5
No. Entities:                        10
================================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
invest_L1         0.512      0.098      5.224      0.000       0.320       0.704
value             0.088      0.029      3.034      0.002       0.031       0.145
capital           0.185      0.067      2.761      0.006       0.054       0.316
================================================================================
```

---

## Installation

Install PanelBox via pip:

```bash
pip install panelbox
```

**Requirements:**
- Python ≥ 3.9
- NumPy ≥ 1.24.0
- Pandas ≥ 2.0.0
- SciPy ≥ 1.10.0

**Optional dependencies:**
```bash
pip install panelbox[plots]        # Matplotlib for plotting
pip install panelbox[performance]  # Numba for speed
pip install panelbox[all]          # Everything
```

See [Installation Guide](how-to/install.md) for detailed instructions.

---

## Features

### Static Panel Models

**Estimators:**
- **Pooled OLS**: Baseline model ignoring panel structure
- **Fixed Effects (Within)**: Controls for time-invariant entity heterogeneity
- **Random Effects (GLS)**: Efficient if effects uncorrelated with regressors
- **Between**: Cross-sectional regression of entity means
- **First Differences**: Simple differencing to eliminate fixed effects

**Standard Errors (8 types):**
- Heteroskedasticity-robust: HC0, HC1, HC2, HC3
- Cluster-robust: One-way and two-way clustering
- HAC: Driscoll-Kraay, Newey-West
- Panel-corrected (PCSE)

**Specification Tests:**
- Hausman test (FE vs RE)
- Breusch-Pagan LM test (random effects)
- Wooldridge test (serial correlation)
- F-test for fixed effects

### Dynamic Panel GMM

**Estimators:**
- **Difference GMM** (Arellano-Bond 1991)
  - First-difference transformation
  - Lagged levels as instruments
  - Handles short panels (small T, large N)

- **System GMM** (Blundell-Bond 1998)
  - Combines difference and level equations
  - More efficient for persistent series
  - Additional moment conditions

**Features:**
- One-step and two-step estimation
- Windmeijer finite-sample correction (2005)
- Instrument collapse (Roodman 2009) to avoid proliferation
- Robust to unbalanced panels

**Diagnostic Tests:**
- Hansen J test (overidentification)
- Sargan test (alternative)
- AR(1) and AR(2) tests (serial correlation)
- Difference-in-Hansen test (System GMM levels)

### Data and Reporting

**Datasets:**
- Grunfeld investment data (10 firms, 20 years)
- Arellano-Bond employment data (optional)

**Output Formats:**
- Console-friendly summary tables
- LaTeX export for publications
- Pandas DataFrames for further analysis
- Diagnostic test reports

---

## Documentation

### 📘 Getting Started

- **[Installation](how-to/install.md)**: Install PanelBox on your system
- **[Quick Start Tutorial](tutorials/01_getting_started.md)**: Your first panel model in 15 minutes
- **[Choose a Model](how-to/choose_model.md)**: Decision guide for selecting the right estimator

### 📚 Tutorials (Learning-Oriented)

1. **[Getting Started](tutorials/01_getting_started.md)**: Load data, estimate Pooled OLS, interpret results
2. **[Static Panel Models](tutorials/02_static_models.md)**: Fixed Effects, Random Effects, Hausman test
3. **[GMM Introduction](tutorials/03_gmm_intro.md)**: Difference GMM, System GMM, diagnostics

### 🛠️ How-To Guides (Task-Oriented)

- **[Install PanelBox](how-to/install.md)**: Installation on Windows, macOS, Linux
- **[Load Your Data](how-to/load_data.md)**: Prepare panel data from various sources
- **[Choose a Model](how-to/choose_model.md)**: Decision trees and workflows
- **[Interpret Tests](how-to/interpret_tests.md)**: Understand diagnostic test output

### 📖 Explanation Guides (Understanding-Oriented)

- **[Panel Data Introduction](guides/panel_data_intro.md)**: What is panel data and when to use it
- **[Fixed vs Random Effects](guides/fixed_vs_random.md)**: Deep dive into FE and RE
- **[GMM Explained](guides/gmm_explained.md)**: Theory and mechanics of GMM estimation

### 🔍 API Reference

- **[Static Models API](api/models.md)**: PooledOLS, FixedEffects, RandomEffects, Between, FirstDifferences
- **[GMM API](api/gmm.md)**: DifferenceGMM, SystemGMM
- **[Results API](api/results.md)**: PanelResults class
- **[Validation API](api/validation.md)**: Diagnostic tests
- **[Datasets API](api/datasets.md)**: load_grunfeld, load_abdata

---

## Why PanelBox?

### 🎯 Designed for Researchers

**PanelBox brings Stata and R panel econometrics to Python:**

| Feature | PanelBox | Stata | R (plm) | linearmodels |
|---------|----------|-------|---------|--------------|
| Difference GMM | ✅ | ✅ (xtabond2) | ✅ | ❌ |
| System GMM | ✅ | ✅ (xtabond2) | ✅ | ❌ |
| Instrument collapse | ✅ | ✅ | ✅ | ❌ |
| Windmeijer correction | ✅ | ✅ | ✅ | ❌ |
| Unbalanced panels | ✅ | ✅ | ✅ | ⚠️ (limited) |
| Hansen J test | ✅ | ✅ | ✅ | ❌ |
| AR(1)/AR(2) tests | ✅ | ✅ | ✅ | ❌ |
| Hausman test | ✅ | ✅ | ✅ | ✅ |

**Validated against academic standards:**
- Stata `xtabond2` (Roodman 2009) for GMM
- R `plm` package for static models
- 600+ unit tests with 93% passing
- Reproduction of published results from seminal papers

### ⚡ Performance

**Numba-optimized critical paths:**

| Operation | Pure Python | Numba | Speedup |
|-----------|-------------|-------|---------|
| GMM weighting matrix | 12.5s | 0.036s | **348x** |
| Within transformation | 2.1s | 0.15s | **14x** |
| Instrument construction | 5.3s | 0.41s | **13x** |

**Benchmark:** N=5000, T=10, 2 lags (typical dynamic panel)

### 📊 Publication-Ready

**Export to LaTeX:**
```python
results.to_latex("table1.tex", caption="Investment Regression Results")
```

**Formatted tables:**
- Coefficient estimates with stars (*, **, ***)
- Standard errors in parentheses
- R-squared, diagnostics footer
- Customizable formatting

---

## Examples

### Fixed Effects

```python
import panelbox as pb

data = pb.load_grunfeld()

# Two-way fixed effects
fe = pb.FixedEffects(
    formula="invest ~ value + capital",
    data=data,
    entity_col="firm",
    time_col="year",
    entity_effects=True,
    time_effects=True
)

results = fe.fit(cov_type='clustered')
print(results.summary())
```

### Hausman Test (FE vs RE)

```python
# Estimate both models
fe = pb.FixedEffects("invest ~ value + capital", data, "firm", "year").fit()
re = pb.RandomEffects("invest ~ value + capital", data, "firm", "year").fit()

# Test
from panelbox.validation import HausmanTest
hausman = HausmanTest(fe, re)
print(hausman)

# Output: p-value = 0.3113 → Use Random Effects (more efficient)
```

### System GMM with Diagnostics

```python
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

results = gmm.fit()

# Check validity
assert results.hansen_j.pvalue > 0.10, "Hansen J test failed"
assert results.ar2_test.pvalue > 0.10, "AR(2) test failed"
assert results.instrument_ratio < 2.0, "Too many instruments"

print(results.summary())
results.to_latex("gmm_results.tex")
```

---

## Roadmap

**Completed (v1.0.0):**
- ✅ Static panel models (Pooled, FE, RE, Between, FD)
- ✅ Dynamic GMM (Difference and System)
- ✅ Comprehensive diagnostic tests
- ✅ Robust standard errors (8 types)
- ✅ Validation against Stata and R
- ✅ Complete documentation

**Planned (v1.1.0+):**
- 🔜 Panel cointegration tests (Pedroni, Kao, Westerlund)
- 🔜 Panel unit root tests (Im-Pesaran-Shin, Levin-Lin-Chu)
- 🔜 Panel VAR models
- 🔜 Quantile regression for panels
- 🔜 Spatial panel models

---

## Citation

If you use PanelBox in academic research, please cite:

```bibtex
@software{panelbox2024,
  author = {Haase, Gustavo and Dourado, Paulo},
  title = {PanelBox: Panel Data Econometrics for Python},
  year = {2024},
  version = {1.0.0},
  url = {https://github.com/PanelBox-Econometrics-Model/panelbox}
}
```

**Key references implemented:**
- Arellano, M., & Bond, S. (1991). "Some Tests of Specification for Panel Data", *Review of Economic Studies*, 58(2), 277-297.
- Blundell, R., & Bond, S. (1998). "Initial Conditions and Moment Restrictions", *Journal of Econometrics*, 87(1), 115-143.
- Roodman, D. (2009). "How to do xtabond2", *The Stata Journal*, 9(1), 86-136.
- Windmeijer, F. (2005). "A Finite Sample Correction", *Journal of Econometrics*, 126(1), 25-51.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/CONTRIBUTING.md) for guidelines.

**Ways to contribute:**
- 🐛 Report bugs via [GitHub Issues](https://github.com/PanelBox-Econometrics-Model/panelbox/issues)
- 💡 Suggest features or enhancements
- 📝 Improve documentation
- 🧪 Add tests or examples
- 🔧 Submit pull requests

---

## License

PanelBox is released under the [MIT License](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/LICENSE).

---

## Support

- **Documentation**: [https://panelbox-econometrics-model.github.io/panelbox](https://panelbox-econometrics-model.github.io/panelbox)
- **Issues**: [GitHub Issues](https://github.com/PanelBox-Econometrics-Model/panelbox/issues)
- **Discussions**: [GitHub Discussions](https://github.com/PanelBox-Econometrics-Model/panelbox/discussions)
- **PyPI**: [https://pypi.org/project/panelbox/](https://pypi.org/project/panelbox/)

---

**Built with ❤️ for econometricians and data scientists**
