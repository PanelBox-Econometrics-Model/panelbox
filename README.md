# PanelBox

**Panel Data Econometrics in Python**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Development Status](https://img.shields.io/badge/status-beta-green.svg)]()
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/panelbox?period=total&units=international_system&left_color=black&right_color=green&left_text=downloads)](https://pepy.tech/projects/panelbox)

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

### Static Panel Models

```python
import panelbox as pb
import pandas as pd

# Load your panel data
data = pd.read_csv('panel_data.csv')

# Fixed Effects model
fe = pb.FixedEffects(
    formula="invest ~ value + capital",
    data=data,
    id_var="firm",
    time_var="year"
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

## Examples

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
  version = {0.2.0},
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

See [CHANGELOG.md](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/CHANGELOG.md) for version history.

### Latest Release: v0.2.0 (2026-01-21)

**Major Features:**
- ✨ Difference GMM (Arellano-Bond 1991)
- ✨ System GMM (Blundell-Bond 1998)
- ✨ Smart instrument selection for unbalanced panels
- ✨ Comprehensive warning system
- ✨ Rich documentation and examples

**Improvements:**
- 🔧 Robust to unbalanced panels (72% retention vs 0% before)
- 🔧 Windmeijer standard error correction
- 🔧 Automatic weak instrument filtering
- 📚 Tutorial, interpretation guide, and examples

---

**Made with ❤️ for econometricians and researchers**
