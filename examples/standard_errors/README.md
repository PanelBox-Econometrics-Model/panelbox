# Standard Errors and Robust Inference in Panel Data

**Tutorial Series for PanelBox**

A comprehensive tutorial series covering robust standard errors, clustering, HAC estimators, spatial correlation, and advanced inference methods for panel data econometrics.

---

## Overview

This tutorial series provides hands-on guidance for implementing and understanding various standard error corrections and inference methods in panel data analysis. Each notebook builds progressively from fundamental concepts to advanced applications.

### What You'll Learn

- **Robust Standard Errors**: Heteroskedasticity-consistent estimators (HC0-HC3)
- **Clustered Standard Errors**: One-way and two-way clustering for panel data
- **HAC Estimators**: Newey-West and Driscoll-Kraay for autocorrelated errors
- **Spatial Correlation**: Spatial HAC with distance-based kernels
- **MLE Inference**: Sandwich estimators and delta method
- **Bootstrap Methods**: Quantile regression and asymptotic refinement
- **Method Comparison**: Comprehensive comparison and selection guidelines

---

## Tutorial Structure

### Learning Paths

**Basic Path** (beginners):
- 01 → 02 → 07

**Intermediate Path** (econometrics background):
- 01 → 02 → 03 → 07

**Advanced Path** (research applications):
- 01 → 02 → 03 → 05 → 04 → 06 → 07

### Notebooks

| # | Notebook | Topics | Duration | Difficulty |
|---|----------|--------|----------|------------|
| 01 | `robust_fundamentals.ipynb` | HC0-HC3, White robust SEs | 45 min | Beginner |
| 02 | `clustering_panels.ipynb` | One-way, two-way clustering | 60 min | Intermediate |
| 03 | `hac_autocorrelation.ipynb` | Newey-West, Driscoll-Kraay | 60 min | Intermediate |
| 04 | `spatial_errors.ipynb` | Spatial HAC, distance matrices | 75 min | Advanced |
| 05 | `mle_inference.ipynb` | Sandwich estimator, delta method | 60 min | Advanced |
| 06 | `bootstrap_quantile.ipynb` | Bootstrap for quantile regression | 60 min | Advanced |
| 07 | `methods_comparison.ipynb` | Comprehensive comparison | 90 min | Intermediate |

---

## Prerequisites

### Knowledge Requirements

- **Python**: Basic proficiency with Python, pandas, and Jupyter notebooks
- **Statistics**: Understanding of regression, hypothesis testing, and confidence intervals
- **Econometrics**: Familiarity with panel data concepts (entities, time periods)

### Software Requirements

- **Python**: 3.9 or higher
- **PanelBox**: 0.8.0 or higher
- **Required packages**: pandas, numpy, matplotlib, seaborn, scipy

### Installation

Install PanelBox and dependencies:

```bash
pip install panelbox>=0.8.0
```

Or install from source:

```bash
git clone https://github.com/yourusername/panelbox.git
cd panelbox
pip install -e .
```

---

## Getting Started

### Quick Start

1. Clone or download this repository
2. Navigate to the `standard_errors/notebooks/` directory
3. Launch Jupyter:
   ```bash
   cd /path/to/panelbox/examples/standard_errors/notebooks
   jupyter notebook
   ```
4. Start with `01_robust_fundamentals.ipynb`

### Data Access

All datasets are located in the `../data/` directory. Notebooks use relative paths for portability.

### Utility Functions

Helper functions for plotting and diagnostics are available in `../utils/`:
- `plotting.py`: Visualization helpers
- `diagnostics.py`: Diagnostic tests
- `data_generators.py`: Synthetic data generation

Import utilities in notebooks:
```python
import sys
sys.path.append('../utils')
from plotting import plot_se_comparison, plot_residuals
from diagnostics import test_heteroskedasticity
```

---

## Datasets

### Included Datasets

| Dataset | Description | N | T | Application |
|---------|-------------|---|---|-------------|
| `grunfeld.csv` | Corporate investment | 10 firms | 20 years | Basic robust SEs |
| `macro_growth.csv` | Economic growth | 30 countries | 40 years | HAC estimators |
| `financial_panel.csv` | Stock returns | 50 firms | 120 months | Two-way clustering |
| `agricultural_panel.csv` | Agricultural productivity | 200 counties | 10 years | Spatial correlation |
| `wage_panel.csv` | Wages and education | 2000 individuals | 5 years | Quantile regression |
| `credit_approval.csv` | Credit approval (binary) | 5000 obs | 1 period | MLE inference |
| `health_insurance.csv` | Health plan choice | 1000 individuals | 5 years | Discrete choice |
| `gdp_quarterly.csv` | Quarterly GDP | 1 country | 100 quarters | Time series HAC |
| `policy_reform.csv` | Labor reform impact | 30 countries | varies | Unbalanced panels |
| `real_estate.csv` | Real estate prices | 500 properties | 5 years | Spatial spillovers |
| `income_inequality.csv` | Family income | 5000 obs | 1 period | Quantile methods |

---

## Key Concepts

### When to Use Each Method

**Heteroskedasticity-Robust (HC)**:
- Cross-sectional heteroskedasticity
- No clustering or autocorrelation
- Example: Investment decisions vary by firm size

**Clustered Standard Errors**:
- Within-group correlation (e.g., students in schools)
- Multiple observations per cluster
- Example: Firms within industries, states within countries

**HAC (Newey-West)**:
- Time series autocorrelation
- Panel or time series data
- Example: Quarterly GDP, monthly returns

**Driscoll-Kraay**:
- Cross-sectional dependence in panels
- Long time dimension (T)
- Example: Country-level economic variables

**Spatial HAC**:
- Correlation based on geographic distance
- Spatial spillovers
- Example: Regional unemployment, agricultural yields

**Bootstrap**:
- Small samples or non-standard estimators
- Quantile regression
- Example: Wage inequality at different quantiles

---

## Directory Structure

```
standard_errors/
├── data/                     # Datasets
├── notebooks/                # Tutorial notebooks (01-07)
├── outputs/
│   ├── figures/             # Generated plots (by notebook)
│   └── reports/html/        # HTML reports
├── utils/                    # Utility functions
│   ├── plotting.py
│   ├── diagnostics.py
│   └── data_generators.py
└── README.md                 # This file
```

---

## Tips for Success

### Best Practices

1. **Work through notebooks in order**: Each builds on previous concepts
2. **Run all cells**: Don't skip code cells, even if output seems obvious
3. **Modify and experiment**: Change parameters and observe effects
4. **Complete exercises**: Reinforce learning with hands-on practice
5. **Consult references**: Follow citations for deeper understanding

### Troubleshooting

**Import errors**: Ensure PanelBox is installed (`pip install panelbox`)

**File not found**: Check that you're running notebooks from the `notebooks/` directory

**Plot styling issues**: Update matplotlib/seaborn to latest versions

**Slow execution**: Some bootstrap methods can be computationally intensive

---

## Contributing

Found a bug or have a suggestion? We welcome contributions!

- **Issues**: Report bugs or suggest improvements
- **Pull requests**: Fix errors, add examples, improve clarity
- **Discussions**: Share applications and ask questions

---

## References

### Key Papers

**Robust Standard Errors**:
- White, H. (1980). A Heteroskedasticity-Consistent Covariance Matrix Estimator. *Econometrica*, 48(4), 817-838.
- MacKinnon, J. G., & White, H. (1985). Some Heteroskedasticity-Consistent Covariance Matrix Estimators. *JASA*, 80(391), 580-586.

**Clustered Standard Errors**:
- Arellano, M. (1987). Computing Robust Standard Errors for Within-Groups Estimators. *Oxford Bulletin of Economics and Statistics*, 49(4), 431-434.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust Inference With Multiway Clustering. *Journal of Business & Economic Statistics*, 29(2), 238-249.

**HAC Estimators**:
- Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703-708.
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data. *Review of Economics and Statistics*, 80(4), 549-560.

**Spatial Methods**:
- Conley, T. G. (1999). GMM Estimation with Cross Sectional Dependence. *Journal of Econometrics*, 92(1), 1-45.

**Bootstrap Methods**:
- Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. CRC Press.
- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press.

### Documentation

- **PanelBox Documentation**: [https://panelbox.readthedocs.io](https://panelbox.readthedocs.io)
- **API Reference**: Standard errors module
- **Examples**: Additional examples in main PanelBox repository

---

## Citation

If you use these tutorials in your research or teaching, please cite:

```bibtex
@misc{panelbox_se_tutorials,
  title={Standard Errors and Robust Inference in Panel Data: Tutorial Series},
  author={PanelBox Development Team},
  year={2026},
  howpublished={\url{https://github.com/yourusername/panelbox}},
  note={PanelBox v0.8.0+}
}
```

---

## License

These tutorials are distributed under the same license as PanelBox. See LICENSE file for details.

---

## Version Information

- **Version**: 1.0.0
- **Last Updated**: 2026-02-16
- **PanelBox Version**: 0.8.0+
- **Python Version**: 3.9+

---

## Contact

For questions, suggestions, or collaboration:

- **Issues**: GitHub issue tracker
- **Discussions**: GitHub discussions
- **Email**: [your-email@example.com]

---

**Ready to get started?** Open `notebooks/01_robust_fundamentals.ipynb` and begin your journey into robust inference for panel data!
