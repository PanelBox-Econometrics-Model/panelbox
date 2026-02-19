# GMM Tutorial Series — PanelBox

## Overview

This tutorial series provides a comprehensive, hands-on guide to **Generalized Method of Moments (GMM)** estimation for dynamic panel data models using PanelBox. The series progresses from foundational concepts to advanced techniques, with real-world applications throughout.

## Learning Objectives

By completing this series you will be able to:

- Understand and implement Difference GMM and System GMM estimators
- Specify and validate instrument sets for dynamic panel models
- Interpret diagnostic tests (Hansen J, AR(1)/AR(2), Sargan, Diff-in-Hansen)
- Detect overfitting with `GMMOverfitDiagnostic` (Nickell bounds, jackknife, sensitivity)
- Apply CUE-GMM and bias-correction methods when appropriate
- Conduct a full empirical workflow from specification to economic interpretation

## Prerequisites

- **Statistical background**: Linear regression, panel data basics, instrumental variables
- **Python**: Intermediate level (NumPy, pandas, matplotlib)
- **PanelBox**: Basic usage (static panel models)

## Installation

```bash
pip install panelbox
# or from source
pip install -e /path/to/panelbox
```

**Additional dependencies:**

```bash
pip install matplotlib seaborn jupyter
```

## Tutorial Roadmap

| # | Notebook | Duration | Topics |
|---|----------|----------|--------|
| 01 | [Difference GMM Fundamentals](notebooks/01_difference_gmm_fundamentals.ipynb) | ~80 min | Nickell bias, basic Difference GMM, OLS/FE/GMM comparison |
| 02 | [System GMM & Efficiency](notebooks/02_system_gmm_efficiency.ipynb) | ~90 min | Weak instruments, System GMM, efficiency gains, Difference-in-Hansen |
| 03 | [Instrument Specification](notebooks/03_instrument_specification.ipynb) | ~110 min | GMM-style vs IV-style, proliferation, collapse, lag selection |
| 04 | [Tests & Diagnostics](notebooks/04_gmm_tests_diagnostics.ipynb) | ~110 min | Hansen J, AR tests, Sargan vs Hansen, Diff-in-Hansen, `GMMOverfitDiagnostic` |
| 05 | [CUE & Bias Correction](notebooks/05_cue_bias_correction.ipynb) | ~120 min | CUE-GMM, two-step vs CUE, bias correction, advanced techniques |
| 06 | [Complete Applied Case](notebooks/06_complete_applied_case.ipynb) | ~150 min | End-to-end workflow, firm investment dynamics, full validation |

**Total estimated time: ~11 hours**

## Directory Structure

```
gmm/
├── README.md                    # This file
├── __init__.py                  # Package initialization
├── data/                        # Datasets (real + simulated)
├── notebooks/                   # Tutorial notebooks (01–06)
├── solutions/                   # Exercise solutions
├── outputs/                     # Generated figures, tables, reports
│   ├── figures/
│   ├── tables/
│   └── reports/
└── utils/                       # Helper functions
    ├── data_generation.py
    ├── validation.py
    └── visualization.py
```

## Datasets

### Real Data
| File | Description | Source |
|------|-------------|--------|
| `abdata.csv` | Arellano-Bond employment data (140 firms, 7–9 years) | Arellano & Bond (1991) |
| `growth.csv` | Country growth panel (100 countries, 20 years) | Penn World Tables |
| `firm_investment.csv` | Corporate investment (500 firms, 10 years) | Compustat-style |

### Simulated Data
| File | Purpose | Used in |
|------|---------|---------|
| `dgp_nickell_bias.csv` | Demonstrates Nickell bias | Notebook 01 |
| `weak_instruments.csv` | Weak instruments (near unit root) | Notebook 02 |
| `bad_specification.csv` | Diagnostic pedagogy (omitted variable) | Notebook 04 |
| `medium_panel_bias.csv` | Bias correction demonstration | Notebook 05 |

See [`data/README.md`](data/README.md) for full documentation.

## Quick Start

```python
import pandas as pd
from panelbox.gmm import DifferenceGMM

# Load data
df = pd.read_csv("data/abdata.csv")

# Estimate Difference GMM
model = DifferenceGMM(
    df,
    depvar="n",
    entity="firm",
    time="year",
    endog=["n_lag1"],
    exog=["w", "k"],
    gmm_instruments=["n_lag1"],
    maxlags=4
)
result = model.fit()
print(result.summary())
```

## References

- Arellano, M., & Bond, S. (1991). Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations. *Review of Economic Studies*, 58(2), 277–297.
- Arellano, M., & Bover, O. (1995). Another look at the instrumental variable estimation of error-components models. *Journal of Econometrics*, 68(1), 29–51.
- Blundell, R., & Bond, S. (1998). Initial conditions and moment restrictions in dynamic panel data models. *Journal of Econometrics*, 87(1), 115–143.
- Hansen, L. P. (1982). Large sample properties of generalized method of moments estimators. *Econometrica*, 50(4), 1029–1054.
- Roodman, D. (2009). How to do xtabond2: An introduction to difference and system GMM in Stata. *Stata Journal*, 9(1), 86–136.
- Windmeijer, F. (2005). A finite sample correction for the variance of linear efficient two-step GMM estimators. *Journal of Econometrics*, 126(1), 25–51.

## License

Part of the PanelBox project. See the main repository for license information.
