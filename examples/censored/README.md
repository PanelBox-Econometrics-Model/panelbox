# Censored and Selection Models Tutorial Series

This tutorial series provides a comprehensive introduction to censored regression models (Tobit) and sample selection models (Heckman) using PanelBox.

## Learning Path

| # | Notebook | Topic | Level |
|---|----------|-------|-------|
| 01 | [Tobit Introduction](notebooks/01_tobit_introduction.ipynb) | Cross-sectional Tobit model basics | Beginner |
| 02 | [Tobit Panel](notebooks/02_tobit_panel.ipynb) | Random Effects Tobit for panel data | Intermediate |
| 03 | [Honoré Estimator](notebooks/03_honore_estimator.ipynb) | Semiparametric FE Tobit (trimmed LAD) | Advanced |
| 04 | [Heckman Selection](notebooks/04_heckman_selection.ipynb) | Two-step Heckman selection correction | Intermediate |
| 05 | [Heckman MLE](notebooks/05_heckman_mle.ipynb) | Maximum likelihood Heckman estimation | Intermediate |
| 06 | [Identification](notebooks/06_identification.ipynb) | Exclusion restrictions and identification | Intermediate |
| 07 | [Marginal Effects](notebooks/07_marginal_effects.ipynb) | Marginal effects in censored/selection models | Intermediate |
| 08 | [Complete Case Study](notebooks/08_complete_case_study.ipynb) | Applied analysis combining all methods | Advanced |

## Recommended Sequence

**Part I - Censored Models (Tobit)**
1. Start with `01_tobit_introduction.ipynb` for fundamentals
2. Move to `02_tobit_panel.ipynb` for panel data extensions
3. Optionally explore `03_honore_estimator.ipynb` for advanced semiparametric methods

**Part II - Selection Models (Heckman)**
4. Study `04_heckman_selection.ipynb` for the classic two-step approach
5. Compare with `05_heckman_mle.ipynb` for MLE estimation
6. Understand `06_identification.ipynb` for proper model specification

**Part III - Interpretation and Application**
7. Learn interpretation via `07_marginal_effects.ipynb`
8. Apply everything in `08_complete_case_study.ipynb`

## Prerequisites

- Familiarity with linear regression and panel data concepts
- Basic understanding of Maximum Likelihood Estimation (MLE)
- Working knowledge of Python, NumPy, and pandas
- PanelBox installed (see [main README](../../README.md))

## Key References

- Tobin, J. (1958). "Estimation of Relationships for Limited Dependent Variables." *Econometrica*, 26(1), 24-36.
- Heckman, J.J. (1979). "Sample Selection Bias as a Specification Error." *Econometrica*, 47(1), 153-161.
- Honoré, B.E. (1992). "Trimmed LAD and Least Squares Estimation of Truncated and Censored Regression Models with Fixed Effects." *Econometrica*, 60(3), 533-565.
- McDonald, J.F. & Moffitt, R.A. (1980). "The Uses of Tobit Analysis." *Review of Economics and Statistics*, 62(2), 318-321.
- Wooldridge, J.M. (1995). "Selection Corrections for Panel Data Models Under Conditional Mean Independence Assumptions." *Journal of Econometrics*, 68(1), 115-132.

## Directory Structure

```
censored/
├── README.md              # This file
├── __init__.py             # Python package init
├── data/                   # Datasets
├── notebooks/              # Tutorial notebooks (01-08)
├── outputs/                # Generated figures and tables
│   ├── figures/
│   └── tables/
├── solutions/              # Exercise solutions
└── utils/                  # Helper scripts
```

## Quick Start

```python
import numpy as np
from panelbox.models.censored import PooledTobit, RandomEffectsTobit
from panelbox.models.selection import PanelHeckman

# Tobit model
model = PooledTobit(y, X, censoring_point=0.0)
result = model.fit()
print(result.summary())

# Heckman selection model
model = PanelHeckman(y, X, selection, Z, method="two_step")
result = model.fit()
print(result.summary())
```
