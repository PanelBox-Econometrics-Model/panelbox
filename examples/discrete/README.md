# Discrete Choice Models - Tutorial Series

**Author**: PanelBox Contributors
**Version**: 1.0
**Last Updated**: 2026-02-16

---

## Overview

This directory contains a comprehensive tutorial series on **discrete choice econometrics** using the PanelBox library. These tutorials cover binary, multinomial, ordered, and dynamic discrete choice models with panel data, including estimation, interpretation, marginal effects, and model validation.

### What You'll Learn

- Binary choice models (logit, probit, linear probability)
- Panel data methods (pooled, fixed effects, random effects)
- Marginal effects and interpretation
- Multinomial and conditional logit models
- Ordered choice models (ordered logit/probit)
- Dynamic discrete choice with state dependence
- Model validation and diagnostics
- Robust inference and standard errors

---

## Learning Path

The tutorials are designed to be completed sequentially, building from fundamental concepts to advanced applications:

### Phase 1: Binary Choice Fundamentals (Notebooks 01-04)

**01. Binary Choice Introduction** (`01_binary_choice_introduction.ipynb`)
- Logit, probit, and linear probability models
- Link functions and their properties
- Maximum likelihood estimation
- Model interpretation and goodness-of-fit

**02. Fixed Effects Logit** (`02_fixed_effects_logit.ipynb`)
- Conditional logit for panel data
- Handling unobserved heterogeneity
- Incidental parameters problem
- Chamberlain's conditional MLE

**03. Random Effects** (`03_random_effects.ipynb`)
- Random effects probit specification
- Comparison with pooled and fixed effects
- Hausman test for model selection
- Integration methods (GHK, quadrature)

**04. Marginal Effects** (`04_marginal_effects.ipynb`)
- Average marginal effects (AME)
- Marginal effects at means (MEM)
- Marginal effects at representative values (MER)
- Standard errors via delta method and bootstrap

### Phase 2: Multinomial Choice (Notebooks 05-06)

**05. Conditional Logit** (`05_conditional_logit.ipynb`)
- Choice-specific variables and attributes
- IIA assumption and testing
- Mixed logit and nested logit extensions
- Application: Transportation mode choice

**06. Multinomial Logit** (`06_multinomial_logit.ipynb`)
- Individual-specific covariates
- Interpretation of coefficients
- Predicted probabilities and classification
- Application: Career choice after graduation

### Phase 3: Ordered and Dynamic Models (Notebooks 07-08)

**07. Ordered Models** (`07_ordered_models.ipynb`)
- Ordered logit and ordered probit
- Threshold/cutpoint estimation
- Parallel regression assumption
- Application: Credit ratings

**08. Dynamic Discrete Choice** (`08_dynamic_discrete.ipynb`)
- State dependence vs. unobserved heterogeneity
- Initial conditions problem (Heckman, Wooldridge)
- Dynamic probit and logit
- Application: Labor market dynamics

### Phase 4: Complete Application (Notebook 09)

**09. Complete Case Study** (`09_complete_case_study.ipynb`)
- End-to-end workflow from data to publication
- Model selection and comparison
- Sensitivity analysis and robustness checks
- Professional report generation

---

## Prerequisites

### Statistical Background
- Probability and statistics fundamentals
- Maximum likelihood estimation
- Panel data econometrics (basic)
- Understanding of logit/probit models

**Recommended Resources**:
- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed.
- Greene, W.H. (2018). *Econometric Analysis*, 8th ed.
- Cameron, A.C. & Trivedi, P.K. (2005). *Microeconometrics: Methods and Applications*

### Programming Skills
- Python basics (variables, functions, loops)
- pandas for data manipulation
- Basic matplotlib/seaborn plotting
- Familiarity with Jupyter notebooks

**Recommended Preparation**:
- Complete PanelBox "Getting Started" tutorial
- Review pandas documentation: https://pandas.pydata.org/docs/
- Basic NumPy: https://numpy.org/doc/stable/user/quickstart.html

---

## Installation and Setup

### 1. Install PanelBox

```bash
pip install panelbox
```

Or for development version:
```bash
git clone https://github.com/panelbox/panelbox.git
cd panelbox
pip install -e .
```

### 2. Install Tutorial Dependencies

```bash
pip install jupyter pandas numpy matplotlib seaborn scipy scikit-learn
```

### 3. Launch Jupyter

```bash
cd examples/discrete/notebooks
jupyter notebook
```

---

## Directory Structure

```
discrete/
├── README.md                    # This file
├── notebooks/                   # Tutorial notebooks (01-09)
│   ├── 01_binary_choice_introduction.ipynb
│   ├── 02_fixed_effects_logit.ipynb
│   ├── 03_random_effects.ipynb
│   ├── 04_marginal_effects.ipynb
│   ├── 05_conditional_logit.ipynb
│   ├── 06_multinomial_logit.ipynb
│   ├── 07_ordered_models.ipynb
│   ├── 08_dynamic_discrete.ipynb
│   └── 09_complete_case_study.ipynb
├── data/                        # Datasets
│   ├── labor_participation.csv
│   ├── job_training.csv
│   ├── transportation_choice.csv
│   ├── credit_rating.csv
│   ├── career_choice.csv
│   └── README.md               # Data dictionary
├── solutions/                   # Solution notebooks
│   └── [corresponding solution notebooks]
├── outputs/                     # Generated outputs
│   ├── figures/                # Plots and visualizations
│   ├── reports/                # HTML/LaTeX reports
│   └── tables/                 # Exported tables
└── utils/                       # Utility functions
    ├── data_generators.py      # Synthetic data generation
    └── visualization_helpers.py # Plotting functions
```

---

## Datasets

All datasets are stored in the `data/` directory. See `data/README.md` for complete variable descriptions and data sources.

### Available Datasets

| Dataset | Observations | Purpose | Used In |
|---------|--------------|---------|---------|
| `labor_participation.csv` | Panel (N×T) | Labor force participation | 01-04 |
| `job_training.csv` | Panel (N×T) | Training program evaluation | 02, 08 |
| `transportation_choice.csv` | Cross-section | Multi-modal transport choice | 05, 06 |
| `credit_rating.csv` | Panel (N×T) | Ordered credit ratings | 07 |
| `career_choice.csv` | Cross-section | Career choice after graduation | 06, 09 |

### Loading Data in Notebooks

All notebooks use the following standardized pattern:

```python
from pathlib import Path
import pandas as pd

# Data path relative to notebook location
DATA_DIR = Path("..") / "data"

# Load data
data = pd.read_csv(DATA_DIR / "labor_participation.csv")
```

---

## Using the Tutorials

### Workflow

1. **Read the notebook**: Each notebook has introductory text explaining concepts
2. **Run code cells**: Execute cells sequentially to see results
3. **Complete exercises**: Exercises are marked with `### EXERCISE` comments
4. **Check solutions**: Compare your work with solution notebooks in `solutions/`
5. **Experiment**: Modify parameters and explore alternative specifications

### Code Conventions

All notebooks follow consistent import and plotting conventions:

```python
# Standard library
from pathlib import Path
import warnings

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PanelBox imports
from panelbox import PooledLogit, FixedEffectsLogit, RandomEffectsProbit
from panelbox.marginal_effects import MarginalEffects

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
warnings.filterwarnings('ignore', category=FutureWarning)
```

---

## Utility Functions

### Data Generators (`utils/data_generators.py`)

Generate synthetic data for experimentation:

```python
from discrete.utils.data_generators import generate_labor_data

# Generate synthetic labor participation data
data = generate_labor_data(n_individuals=1000, n_periods=5, seed=42)
```

Available generators:
- `generate_labor_data()`: Binary labor force participation
- `generate_multinomial_choice_data()`: Multinomial/conditional choice
- `generate_ordered_data()`: Ordered categorical outcomes
- `generate_dynamic_binary_data()`: Dynamic binary with state dependence

### Visualization Helpers (`utils/visualization_helpers.py`)

Standardized plotting functions:

```python
from discrete.utils.visualization_helpers import (
    plot_link_functions,
    plot_predicted_probabilities,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_marginal_effects
)

# Compare link functions
plot_link_functions(compare=['logit', 'probit', 'lpm'])

# Plot ROC curve
plot_roc_curve(results, actual_y=data['lfp'])
```

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'panelbox'`
- **Solution**: Install PanelBox: `pip install panelbox`

**Issue**: Notebooks cannot find data files
- **Solution**: Ensure you're running notebooks from `notebooks/` directory
- **Alternative**: Update `DATA_DIR` path in notebook

**Issue**: Plots not displaying in Jupyter
- **Solution**: Add `%matplotlib inline` at top of notebook

**Issue**: Convergence warnings in MLE estimation
- **Solution**: Check for multicollinearity, try different starting values, or rescale variables

**Issue**: Memory errors with large datasets
- **Solution**: Use smaller subsamples for initial exploration, then scale up

### Getting Help

- **PanelBox Documentation**: [https://panelbox.readthedocs.io](https://panelbox.readthedocs.io)
- **GitHub Issues**: [https://github.com/panelbox/panelbox/issues](https://github.com/panelbox/panelbox/issues)
- **Discussions**: [https://github.com/panelbox/panelbox/discussions](https://github.com/panelbox/panelbox/discussions)

---

## Key References

### Textbooks

1. **Wooldridge, J.M.** (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed. MIT Press.
   - Chapters 15-16: Binary and multinomial response models
   - Chapter 23: Dynamic models for panel data

2. **Greene, W.H.** (2018). *Econometric Analysis*, 8th ed. Pearson.
   - Chapters 17-18: Discrete choice models
   - Chapter 19: Panel data models

3. **Cameron, A.C. & Trivedi, P.K.** (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press.
   - Chapters 14-15: Multinomial and ordered models
   - Chapter 23: Count and limited dependent variable models

### Seminal Papers

1. **McFadden, D.** (1974). "Conditional logit analysis of qualitative choice behavior." In *Frontiers in Econometrics*, ed. P. Zarembka, 105-142.

2. **Chamberlain, G.** (1980). "Analysis of covariance with qualitative data." *Review of Economic Studies* 47(1): 225-238.

3. **Heckman, J.J.** (1981). "The incidental parameters problem and the problem of initial conditions in estimating a discrete time-discrete data stochastic process." In *Structural Analysis of Discrete Data with Econometric Applications*, eds. C. Manski and D. McFadden.

4. **Wooldridge, J.M.** (2005). "Simple solutions to the initial conditions problem in dynamic, nonlinear panel data models with unobserved heterogeneity." *Journal of Applied Econometrics* 20(1): 39-54.

### Software Documentation

- **PanelBox**: https://panelbox.readthedocs.io
- **statsmodels**: https://www.statsmodels.org/stable/discretemod.html
- **scikit-learn**: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

---

## Contributing

We welcome contributions to improve these tutorials! Please:

1. Report issues or suggest improvements via GitHub Issues
2. Submit pull requests with corrections or enhancements
3. Share your use cases and applications

---

## License

These tutorials are part of the PanelBox project and are released under the MIT License. You are free to use, modify, and distribute them with attribution.

---

## Citation

If you use these tutorials in your research or teaching, please cite:

```bibtex
@misc{panelbox_discrete_tutorials,
  title={Discrete Choice Models Tutorial Series},
  author={PanelBox Contributors},
  year={2026},
  howpublished={\url{https://github.com/panelbox/panelbox/examples/discrete}},
  note={Version 1.0}
}
```

---

## Acknowledgments

These tutorials were developed by the PanelBox community with contributions from econometricians, statisticians, and practitioners. Special thanks to all contributors and reviewers.

For questions or feedback, please open an issue on GitHub or contact the maintainers.

**Happy learning!** 📊
