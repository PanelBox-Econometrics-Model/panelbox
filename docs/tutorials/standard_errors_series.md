# Standard Errors and Robust Inference Tutorial Series

**Complete Guide to Standard Errors in Panel Data Econometrics**

---

## Overview

This comprehensive tutorial series covers robust inference methods for panel data models, from heteroskedasticity-robust standard errors to advanced spatial HAC estimators. Each notebook builds on the previous one, providing a progressive learning path from fundamentals to advanced topics.

**Target Audience**: Graduate students, researchers, and practitioners working with panel data

**Prerequisites**:
- Basic econometrics (OLS, hypothesis testing)
- Python fundamentals (pandas, numpy)
- Understanding of panel data structure (entities, time periods)

**Estimated Total Duration**: 8-10 hours

---

## Learning Path

The series offers three learning paths based on your needs:

### 🟢 Basic Path (3-4 hours)
**For**: Practitioners who need robust standard errors for applied work

1. [01 - Robust Fundamentals](#notebook-01-robust-fundamentals) → Understanding heteroskedasticity-robust standard errors
2. [02 - Clustering in Panels](#notebook-02-clustering-panels) → One-way and two-way clustering
3. [07 - Methods Comparison](#notebook-07-methods-comparison) → Choosing the right method

### 🟡 Intermediate Path (5-6 hours)
**For**: Researchers working with time series panels or autocorrelated errors

1. [01 - Robust Fundamentals](#notebook-01-robust-fundamentals)
2. [02 - Clustering in Panels](#notebook-02-clustering-panels)
3. [03 - HAC and Autocorrelation](#notebook-03-hac-autocorrelation) → Newey-West and Driscoll-Kraay
4. [07 - Methods Comparison](#notebook-07-methods-comparison)

### 🔴 Advanced Path (8-10 hours)
**For**: Researchers dealing with spatial data, MLE models, or quantile regression

1. [01 - Robust Fundamentals](#notebook-01-robust-fundamentals)
2. [02 - Clustering in Panels](#notebook-02-clustering-panels)
3. [03 - HAC and Autocorrelation](#notebook-03-hac-autocorrelation)
4. [05 - MLE Inference](#notebook-05-mle-inference) → Sandwich estimator and delta method
5. [04 - Spatial Errors](#notebook-04-spatial-errors) → Spatial HAC with distance matrices
6. [06 - Bootstrap for Quantiles](#notebook-06-bootstrap-quantile) → Bootstrap inference
7. [07 - Methods Comparison](#notebook-07-methods-comparison)

---

## Notebooks Overview

### Notebook 01: Robust Fundamentals
**File**: `01_robust_fundamentals.ipynb`
**Duration**: 60-75 minutes
**Difficulty**: 🟢 Beginner

**Topics**:
- Heteroskedasticity and its consequences
- HC0, HC1, HC2, HC3 standard errors
- White's robust standard errors
- When to use each HC variant
- Practical implementation in PanelBox

**Key Learning Outcomes**:
- Diagnose heteroskedasticity in panel data
- Choose appropriate HC estimator
- Interpret robust standard errors correctly
- Understand finite-sample corrections

**Dataset**: Grunfeld corporate investment data (10 firms, 20 years)

---

### Notebook 02: Clustering Panels
**File**: `02_clustering_panels.ipynb`
**Duration**: 75-90 minutes
**Difficulty**: 🟢 Beginner-Intermediate

**Topics**:
- Within-cluster correlation
- One-way clustering (entity or time)
- Two-way clustering (entity and time)
- Cluster-robust variance estimation
- Minimum cluster requirements

**Key Learning Outcomes**:
- Understand when clustering is necessary
- Implement one-way and two-way clustering
- Diagnose cluster structure in your data
- Avoid common pitfalls (too few clusters, unbalanced panels)

**Datasets**:
- Grunfeld data (10 firms, 20 years)
- Financial panel (50 firms, 120 months)
- Wage panel (2000 individuals, 5 years)
- Policy reform data (30 countries, varying time)

---

### Notebook 03: HAC and Autocorrelation
**File**: `03_hac_autocorrelation.ipynb`
**Duration**: 90-120 minutes
**Difficulty**: 🟡 Intermediate

**Topics**:
- Serial correlation in panel data
- Newey-West HAC standard errors
- Driscoll-Kraay standard errors for panels
- Optimal lag selection
- Kernel choice (Bartlett, Parzen, Quadratic Spectral)

**Key Learning Outcomes**:
- Detect and test for serial correlation
- Choose between Newey-West and Driscoll-Kraay
- Select appropriate lag length
- Compare HAC with clustered standard errors

**Datasets**:
- Macro growth panel (30 countries, 40 years)
- GDP quarterly data (1 country, 100 quarters)

---

### Notebook 04: Spatial Errors
**File**: `04_spatial_errors.ipynb`
**Duration**: 90-120 minutes
**Difficulty**: 🔴 Advanced

**Topics**:
- Spatial correlation and spatial HAC
- Distance-based kernel weighting
- Conley (1999) spatial HAC estimator
- Spatial diagnostics (Moran's I)
- Geographic weight matrices

**Key Learning Outcomes**:
- Detect spatial correlation in panel data
- Construct distance-based weight matrices
- Implement spatial HAC standard errors
- Interpret spatial correlation patterns

**Datasets**:
- Agricultural panel (200 counties, 10 years) with coordinates
- Real estate prices (500 properties, 5 years) with spatial data

---

### Notebook 05: MLE Inference
**File**: `05_mle_inference.ipynb`
**Duration**: 75-90 minutes
**Difficulty**: 🟡 Intermediate-Advanced

**Topics**:
- Maximum likelihood estimation
- Sandwich estimator (Huber-White)
- Delta method for nonlinear functions
- Odds ratios and marginal effects
- Bootstrap for MLE models

**Key Learning Outcomes**:
- Understand MLE sandwich estimator
- Apply delta method for transformations
- Compute standard errors for marginal effects
- Use bootstrap for complex MLE models

**Datasets**:
- Credit approval (5000 obs, cross-section) - Binary logit
- Health insurance choice (1000 individuals, 5 years) - Panel multinomial

---

### Notebook 06: Bootstrap for Quantiles
**File**: `06_bootstrap_quantile.ipynb`
**Duration**: 90-120 minutes
**Difficulty**: 🔴 Advanced

**Topics**:
- Quantile regression basics
- Bootstrap methods for quantile regression
- Panel bootstrap (pairs, residual, wild)
- Quantile treatment effects
- Uniform confidence bands

**Key Learning Outcomes**:
- Understand why quantile regression needs bootstrap
- Implement panel bootstrap methods
- Construct uniform confidence bands
- Estimate quantile treatment effects

**Dataset**:
- Wage panel (2000 individuals, 5 years)
- Income inequality (5000 obs, cross-section)

---

### Notebook 07: Methods Comparison
**File**: `07_methods_comparison.ipynb`
**Duration**: 90-120 minutes
**Difficulty**: 🟡 Intermediate

**Topics**:
- Comprehensive comparison of all methods
- Decision tree for choosing standard errors
- Monte Carlo simulation studies
- Power and size analysis
- Best practices and recommendations

**Key Learning Outcomes**:
- Choose appropriate standard error method for your data
- Understand trade-offs between methods
- Interpret simulation evidence
- Follow best practices for robust inference

**Datasets**: Multiple datasets from previous notebooks + simulated data

---

## Installation

The tutorial series requires PanelBox v0.8.0 or higher.

### Install PanelBox

```bash
pip install panelbox>=0.8.0
```

### Install Additional Dependencies

```bash
pip install jupyter matplotlib seaborn statsmodels scipy
```

### Clone the Examples Repository

```bash
git clone https://github.com/PanelBox-Econometrics-Model/panelbox.git
cd panelbox/examples/standard_errors
```

---

## Repository Structure

```
examples/standard_errors/
│
├── data/                     # Datasets used in tutorials
│   ├── grunfeld.csv
│   ├── macro_growth.csv
│   ├── financial_panel.csv
│   └── ...
│
├── notebooks/                # Tutorial notebooks
│   ├── 01_robust_fundamentals.ipynb
│   ├── 02_clustering_panels.ipynb
│   ├── 03_hac_autocorrelation.ipynb
│   ├── 04_spatial_errors.ipynb
│   ├── 05_mle_inference.ipynb
│   ├── 06_bootstrap_quantile.ipynb
│   └── 07_methods_comparison.ipynb
│
├── utils/                    # Helper functions
│   ├── plotting.py           # Plotting utilities
│   ├── diagnostics.py        # Diagnostic tests
│   └── data_generators.py    # Synthetic data generation
│
├── outputs/                  # Generated figures and reports
│   ├── figures/
│   └── reports/
│
├── README.md                 # Series overview
├── CHANGELOG.md             # Version history
└── .gitignore
```

---

## Quick Start

### Option 1: Run in Jupyter

```bash
cd panelbox/examples/standard_errors/notebooks
jupyter notebook 01_robust_fundamentals.ipynb
```

### Option 2: Run in JupyterLab

```bash
cd panelbox/examples/standard_errors/notebooks
jupyter lab
```

### Option 3: Run in VS Code

Open the `notebooks` folder in VS Code with the Jupyter extension installed.

---

## Key Concepts Covered

### Standard Error Types

| Method | Use Case | Notebook |
|--------|----------|----------|
| **HC0-HC3** | Heteroskedasticity only | 01 |
| **Clustered (1-way)** | Within-entity or within-time correlation | 02 |
| **Clustered (2-way)** | Both entity and time correlation | 02 |
| **Newey-West** | Autocorrelation (single entity) | 03 |
| **Driscoll-Kraay** | Autocorrelation + cross-sectional dependence | 03 |
| **Spatial HAC** | Geographic spatial correlation | 04 |
| **MLE Sandwich** | Maximum likelihood models | 05 |
| **Bootstrap** | Quantile regression, complex models | 06 |

### Diagnostic Tests

| Test | Purpose | Notebook |
|------|---------|----------|
| **White Test** | Detect heteroskedasticity | 01 |
| **Breusch-Pagan** | Test for heteroskedasticity | 01 |
| **Durbin-Watson** | Detect first-order autocorrelation | 03 |
| **Breusch-Godfrey** | Test for higher-order autocorrelation | 03 |
| **Moran's I** | Test for spatial correlation | 04 |
| **ACF/PACF** | Autocorrelation diagnostics | 03 |

---

## Common Use Cases

### 🔍 "My data has heteroskedasticity"
**Solution**: Start with [Notebook 01](#notebook-01-robust-fundamentals)
**Method**: Use HC3 standard errors (default in PanelBox)

### 🔍 "I have panel data with many firms"
**Solution**: Go to [Notebook 02](#notebook-02-clustering-panels)
**Method**: Use entity-clustered standard errors

### 🔍 "My time series panel has autocorrelation"
**Solution**: Check [Notebook 03](#notebook-03-hac-autocorrelation)
**Method**: Use Driscoll-Kraay standard errors

### 🔍 "I'm working with spatial/geographic data"
**Solution**: See [Notebook 04](#notebook-04-spatial-errors)
**Method**: Use spatial HAC with distance-based weights

### 🔍 "I'm estimating a logit/probit model"
**Solution**: Learn from [Notebook 05](#notebook-05-mle-inference)
**Method**: Use MLE sandwich estimator or bootstrap

### 🔍 "I'm doing quantile regression"
**Solution**: Study [Notebook 06](#notebook-06-bootstrap-quantile)
**Method**: Use panel bootstrap methods

### 🔍 "I don't know which method to use"
**Solution**: Jump to [Notebook 07](#notebook-07-methods-comparison)
**Tool**: Follow the decision tree and best practices

---

## Best Practices

### 1. Always Test First
- Run diagnostic tests before choosing standard errors
- Don't apply robust methods blindly
- Document your testing process

### 2. Report Multiple Methods
- Show results with and without robust standard errors
- Report cluster-robust and Driscoll-Kraay in panels
- Use sensitivity analysis (Notebook 07)

### 3. Check Cluster Requirements
- Minimum 20-30 clusters for asymptotic validity
- Check cluster balance
- Consider wild cluster bootstrap for few clusters

### 4. Mind the Lags
- Use information criteria for lag selection
- Don't over-specify (use parsimony)
- Check sensitivity to lag choice

### 5. Document Assumptions
- State which standard errors you use
- Justify your choice based on diagnostics
- Report any deviations from best practices

---

## FAQ

### Q: Which standard error should I use?
**A**: It depends on your data structure:
- **Cross-section with heteroskedasticity**: HC3
- **Panel with entity clustering**: Cluster-robust (entity)
- **Long time series panel**: Driscoll-Kraay or two-way clustering
- **Spatial data**: Spatial HAC
- **MLE models**: Sandwich estimator or bootstrap
- **Quantile regression**: Bootstrap

See [Notebook 07](#notebook-07-methods-comparison) for a comprehensive decision tree.

### Q: Can I use clustered and HAC together?
**A**: Not directly. Driscoll-Kraay standard errors combine clustering and HAC, which is usually what you want for panels with both cross-sectional and time-series correlation.

### Q: How many clusters do I need?
**A**: General rule: 20-30 clusters minimum. With fewer clusters, consider wild cluster bootstrap.

### Q: Should I always use robust standard errors?
**A**: Robust standard errors are rarely worse than conventional ones (they're consistent even under homoskedasticity), but they can have lower power in small samples. Always report both in applied work.

### Q: What's the difference between Newey-West and Driscoll-Kraay?
**A**:
- **Newey-West**: For single time series (one entity)
- **Driscoll-Kraay**: For panel data (many entities), allows cross-sectional correlation

### Q: Can I use these methods with Fixed Effects or Random Effects?
**A**: Yes! All robust standard error methods work with any estimator (Pooled OLS, Fixed Effects, Random Effects, etc.).

---

## Citation

If you use these tutorials in your research or teaching, please cite:

```bibtex
@software{panelbox2026,
  title = {PanelBox: Python Library for Panel Data Econometrics},
  author = {PanelBox Development Team},
  year = {2026},
  url = {https://github.com/PanelBox-Econometrics-Model/panelbox},
  version = {0.8.0}
}
```

---

## Contributing

Found an error or have suggestions for improvement?

- Open an issue: [GitHub Issues](https://github.com/PanelBox-Econometrics-Model/panelbox/issues)
- Submit a pull request: [GitHub Pull Requests](https://github.com/PanelBox-Econometrics-Model/panelbox/pulls)
- Email us: panelbox@example.com

---

## Additional Resources

### Academic Papers

- **White (1980)**: "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity", *Econometrica*
- **Arellano (1987)**: "Computing Robust Standard Errors for Within-Groups Estimators", *Oxford Bulletin of Economics and Statistics*
- **Newey & West (1987)**: "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix", *Econometrica*
- **Driscoll & Kraay (1998)**: "Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data", *Review of Economics and Statistics*
- **Cameron, Gelbach & Miller (2011)**: "Robust Inference with Multiway Clustering", *Journal of Business & Economic Statistics*
- **Conley (1999)**: "GMM Estimation with Cross Sectional Dependence", *Journal of Econometrics*

### Books

- **Wooldridge (2010)**: *Econometric Analysis of Cross Section and Panel Data*, 2nd ed., MIT Press
- **Cameron & Trivedi (2005)**: *Microeconometrics: Methods and Applications*, Cambridge University Press
- **Angrist & Pischke (2009)**: *Mostly Harmless Econometrics*, Princeton University Press

### Software Documentation

- [PanelBox Documentation](https://panelbox-econometrics-model.github.io/panelbox)
- [PanelBox API Reference](https://panelbox-econometrics-model.github.io/panelbox/api/)
- [PanelBox GitHub Repository](https://github.com/PanelBox-Econometrics-Model/panelbox)

---

## License

This tutorial series is licensed under the MIT License.

---

**Last Updated**: 2026-02-16
**Version**: 1.0.0
**PanelBox Version**: 0.8.0+
