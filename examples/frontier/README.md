# Stochastic Frontier Analysis Tutorial Series

Comprehensive tutorial series on Stochastic Frontier Analysis (SFA) and
technical efficiency measurement using the PanelBox library.

## Overview

| # | Notebook | Level | Duration | Topics |
|---|----------|-------|----------|--------|
| 1 | Introduction to SFA | Beginner | 90-120 min | Frontier concept, MLE, efficiency estimation |
| 2 | Panel SFA | Intermediate | 120-150 min | Pitt-Lee, BC92, BC95, CSS, Kumbhakar |
| 3 | Four-Component & TFP | Advanced | 90-120 min | Persistent/transient efficiency, TFP decomposition |
| 4 | Determinants & Heterogeneity | Inter-Advanced | 90-120 min | BC95 determinants, Wang 2002, marginal effects |
| 5 | Testing & Comparison | Inter-Advanced | 90-120 min | LR tests, Vuong, bootstrap, model selection |
| 6 | Complete Case Study | Capstone | 180-240 min | Brazilian manufacturing analysis |

## Prerequisites

- Python 3.8+
- panelbox >= 0.7.0
- pandas, numpy, scipy, matplotlib, seaborn

## Learning Path

**Recommended order**: 01 → 02 → 03 → 04 → 05 → 06

- **Beginners**: Start with 01 (Introduction) and 02 (Panel SFA) for core concepts
- **Applied Researchers**: Follow 01 → 02 → 04 → 06 for practical applications
- **Quick Start**: 01 → 02 → 05 for essential SFA with model selection

## Folder Structure

```
frontier/
├── README.md              # This file
├── __init__.py            # Python package init
├── data/                  # Datasets (synthetic production data)
├── notebooks/             # Tutorial notebooks (01-06)
├── outputs/               # Generated figures, tables, and reports
│   ├── figures/
│   ├── tables/
│   └── reports/
├── solutions/             # Exercise solutions
└── utils/                 # Helper scripts
```

## Datasets

| Dataset | Type | N | T | Used in | Description |
|---------|------|---|---|---------|-------------|
| `hospital_data.csv` | Cross-section | 200 | — | 01 | Hospital production efficiency |
| `farm_data.csv` | Cross-section | 300 | — | 01 | Agricultural production |
| `bank_panel.csv` | Panel | 50 | 15 | 02 | Banking efficiency |
| `airline_panel.csv` | Panel | 25 | 20 | 02 | Airline efficiency |
| `manufacturing_panel.csv` | Panel | 100 | 10 | 03 | Manufacturing TFP |
| `electricity_panel.csv` | Panel | 60 | 12 | 03 | Electricity generation |
| `hospital_panel.csv` | Panel | 80 | 10 | 04 | Hospital with determinants |
| `school_panel.csv` | Panel | 100 | 8 | 04 | School efficiency |
| `dairy_farm.csv` | Cross-section | 500 | — | 05 | Dairy farm comparison |
| `telecom_panel.csv` | Panel | 40 | 15 | 05 | Telecom efficiency |
| `brazilian_firms.csv` | Panel | 500 | 10 | 06 | Case study data |

## Quick Start

```python
import pandas as pd
from panelbox.frontier import StochasticFrontier

# Load data
data = pd.read_csv("data/hospital_data.csv")

# Estimate production frontier
sf = StochasticFrontier(
    data=data,
    depvar="log_output",
    exog=["log_labor", "log_capital", "log_supplies"],
    frontier="production",
    dist="half_normal",
)
result = sf.fit()
print(result.summary())

# Efficiency estimates
eff = result.efficiency(estimator="bc")
print(f"Mean efficiency: {result.mean_efficiency:.4f}")
```

## Key References

- Aigner, D., Lovell, C.A.K. & Schmidt, P. (1977). "Formulation and Estimation of Stochastic Frontier Production Function Models." *Journal of Econometrics*, 6(1), 21-37.
- Battese, G.E. & Coelli, T.J. (1992). "Frontier Production Functions, Technical Efficiency and Panel Data." *Journal of Productivity Analysis*, 3, 153-169.
- Battese, G.E. & Coelli, T.J. (1995). "A Model for Technical Inefficiency Effects in a Stochastic Frontier Production Function for Panel Data." *Empirical Economics*, 20, 325-332.
- Kumbhakar, S.C. & Lovell, C.A.K. (2000). *Stochastic Frontier Analysis*. Cambridge University Press.
- Kumbhakar, S.C., Lien, G. & Hardaker, J.B. (2014). "Technical Efficiency in Competing Panel Data Models." *Journal of Productivity Analysis*, 41, 321-337.
- Wang, H.J. (2002). "Heteroscedasticity and Non-Monotonic Efficiency Effects of a Stochastic Frontier Model." *Journal of Productivity Analysis*, 18, 241-253.
