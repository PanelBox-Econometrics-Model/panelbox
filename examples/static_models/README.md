# Static Panel Models Tutorials

Welcome to the PanelBox static panel models tutorial series! This collection of Jupyter notebooks provides a comprehensive introduction to estimating and interpreting static panel data models, from basic pooled OLS to advanced instrumental variables methods.

## Learning Path

The tutorials are organized into three progressive levels. Follow the path sequentially for the best learning experience.

### Level 1: Fundamentals (3-4 hours)

**Start here if you're new to panel data.**

#### 1. [Pooled OLS Introduction](fundamentals/01_pooled_ols_introduction.ipynb)
- **Duration**: 45-60 minutes
- **Prerequisites**: Basic OLS regression, pandas
- **Topics**:
  - Panel data structure (entity-time format)
  - Pooled OLS estimation
  - When to use (and when not to use) pooled OLS
  - Visualizing panel data patterns
- **Dataset**: Grunfeld investment data

#### 2. [Fixed Effects](fundamentals/02_fixed_effects.ipynb)
- **Duration**: 60-75 minutes
- **Prerequisites**: Notebook 01
- **Topics**:
  - Entity-specific fixed effects
  - Within transformation
  - Time-invariant vs. time-varying regressors
  - Interpretation of coefficients
  - Comparison with pooled OLS
- **Datasets**: Grunfeld, wage panel

#### 3. [Random Effects and Hausman Test](fundamentals/03_random_effects_hausman.ipynb)
- **Duration**: 60-75 minutes
- **Prerequisites**: Notebooks 01-02
- **Topics**:
  - Random effects (RE) estimation
  - GLS vs. OLS
  - Fixed effects vs. random effects
  - Hausman specification test
  - Choosing between FE and RE
- **Datasets**: Wage panel, country growth

---

### Level 2: Advanced (4-6 hours)

**Proceed here after completing the fundamentals.**

#### 4. [First Difference and Between Estimators](advanced/04_first_difference_between.ipynb)
- **Duration**: 60-90 minutes
- **Prerequisites**: Notebooks 01-03
- **Topics**:
  - First difference (FD) estimator
  - Between estimator
  - Comparison of FE, FD, and Between
  - Efficiency considerations
  - Handling serial correlation
- **Datasets**: Grunfeld, wage panel

#### 5. [Panel Instrumental Variables](advanced/05_panel_iv.ipynb)
- **Duration**: 90-120 minutes
- **Prerequisites**: Notebooks 01-04, basic IV/2SLS knowledge
- **Topics**:
  - IV estimation with panel data
  - Fixed effects 2SLS (FE-IV)
  - First-stage diagnostics
  - Weak instruments in panel context
  - Overidentification tests
- **Datasets**: Wage panel, country growth

#### 6. [Comparison of Estimators](advanced/06_comparison_estimators.ipynb)
- **Duration**: 75-90 minutes
- **Prerequisites**: Notebooks 01-05
- **Topics**:
  - Side-by-side comparison of all estimators
  - Monte Carlo simulation study
  - Bias-variance tradeoffs
  - Practical guidance on estimator choice
  - Reporting best practices
- **Datasets**: All datasets, simulated data

---

### Level 3: Expert (1.5-2 hours)

**Advanced topics for experienced practitioners.**

#### 7. [Advanced IV Diagnostics](expert/07_iv_diagnostics_advanced.ipynb)
- **Duration**: 75-90 minutes
- **Prerequisites**: Notebooks 01-06, advanced IV theory
- **Topics**:
  - Instrument strength diagnostics
  - Testing for endogeneity
  - Heterogeneous treatment effects
  - Sensitivity analysis
  - Reporting IV results
- **Datasets**: Wage panel, country growth

---

## Quick Start

To get started immediately:

```python
# Install PanelBox (if not already installed)
pip install panelbox

# Navigate to the fundamentals folder
cd fundamentals

# Open the first notebook
jupyter notebook 01_pooled_ols_introduction.ipynb
```

---

## Datasets Used

All datasets are located in `../datasets/panel/` with documentation in `../datasets/metadata/`.

| Dataset | Entities | Periods | Total Obs | Type | Used In |
|---------|----------|---------|-----------|------|---------|
| [grunfeld.csv](../datasets/panel/grunfeld.csv) | 10 firms | 20 years | 200 | Balanced | 01, 02, 04, 06 |
| [wage_panel.csv](../datasets/panel/wage_panel.csv) | 500 individuals | 7 years | 3500 | Balanced | 02, 03, 05, 06 |
| [country_growth.csv](../datasets/panel/country_growth.csv) | 80 countries | 30 years | 2400 | Balanced | 03, 05, 06 |
| [firm_productivity.csv](../datasets/panel/firm_productivity.csv) | 300 firms | 12 years | 3200 | Unbalanced | 06, 07 |

**Codebooks**: See `../datasets/metadata/` for detailed variable descriptions and citations.

---

## Prerequisites

### Software Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab
- PanelBox (latest version)
- Standard scientific Python stack (NumPy, pandas, matplotlib, seaborn)

### Install Dependencies

```bash
pip install panelbox numpy pandas matplotlib seaborn jupyter
```

### Statistical Background

- **Fundamentals**: Basic understanding of linear regression (OLS), hypothesis testing, and confidence intervals
- **Advanced**: Familiarity with instrumental variables, 2SLS, and basic econometric theory
- **Expert**: Graduate-level econometrics (endogeneity, identification, causal inference)

---

## Learning Objectives

By completing this tutorial series, you will be able to:

1. ✅ **Recognize** when panel data methods are appropriate
2. ✅ **Estimate** pooled OLS, fixed effects, random effects, first difference, and IV models
3. ✅ **Interpret** coefficients in panel data regressions correctly
4. ✅ **Choose** the appropriate estimator based on model assumptions and data structure
5. ✅ **Diagnose** specification issues (e.g., using Hausman test, instrument diagnostics)
6. ✅ **Report** results following best practices in applied econometrics
7. ✅ **Visualize** panel data patterns and estimation results effectively

---

## Notebook Structure

Each notebook follows a consistent structure:

1. **Introduction**: Motivation and learning objectives
2. **Theory**: Brief theoretical background (with references for deeper study)
3. **Data Exploration**: Load and visualize the dataset
4. **Estimation**: Step-by-step estimation with PanelBox
5. **Interpretation**: How to read and interpret the results
6. **Diagnostics**: Model checks and specification tests
7. **Exercises**: Practice problems (solutions available separately)
8. **Summary**: Key takeaways and next steps

---

## Companion Materials

### Utility Functions

Reusable plotting and analysis functions are available in `../utils/`:

```python
import sys
sys.path.append('..')

from utils.visualization.panel_plots import spaghetti_plot, within_between_scatter
from utils.visualization.comparison_plots import coefficient_plot
from utils.helpers.model_comparison import compare_models
```

### Exercise Solutions

Solutions to in-notebook exercises are available in `../../solutions/static_models/`. We recommend attempting exercises independently before consulting solutions.

---

## Recommended References

### Textbooks

- **Wooldridge, J. M. (2010)**. *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
  - Comprehensive treatment of panel methods (intermediate to advanced)

- **Baltagi, B. H. (2021)**. *Econometric Analysis of Panel Data* (6th ed.). Springer.
  - Accessible introduction with many examples (intermediate)

- **Cameron, A. C., & Trivedi, P. K. (2005)**. *Microeconometrics: Methods and Applications*. Cambridge University Press.
  - Strong on interpretation and application (intermediate to advanced)

### Journal Articles

- **Hausman, J. A. (1978)**. "Specification Tests in Econometrics." *Econometrica*, 46(6), 1251-1271.
  - Classic paper on the FE vs. RE test

- **Angrist, J. D., & Pischke, J.-S. (2009)**. *Mostly Harmless Econometrics*. Princeton University Press.
  - Practical guide to applied work (very accessible)

### Online Resources

- [PanelBox Documentation](https://panelbox.readthedocs.io/)
- [PanelBox Examples Gallery](../README.md)

---

## Troubleshooting

### Common Issues

**Import errors**: Ensure PanelBox is installed and up to date
```bash
pip install --upgrade panelbox
```

**Data not found**: Verify you're running notebooks from the correct directory. Data paths are relative to notebook location.

**Kernel crashes**: Some datasets are large. Increase Jupyter memory limit if needed.

### Getting Help

- **Bug reports**: [GitHub Issues](https://github.com/yourorg/panelbox/issues)
- **Questions**: [Discussion Forum](https://github.com/yourorg/panelbox/discussions)
- **Documentation**: [Read the Docs](https://panelbox.readthedocs.io/)

---

## Contributing

Found a typo or have a suggestion? Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

---

## License

These tutorials are released under the MIT License. See [LICENSE](../../LICENSE) for details.

---

## Acknowledgments

- Datasets: Grunfeld (1958), NLSY, Penn World Tables
- Inspiration: Wooldridge, Baltagi, Cameron & Trivedi
- Funding: [Grant/Institution if applicable]

---

**Happy Learning!** 📊📈

For questions or feedback, please open an issue on GitHub or contact the maintainers.
