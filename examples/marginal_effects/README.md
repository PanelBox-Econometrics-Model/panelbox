# Marginal Effects Tutorial Series

This series teaches how to compute, interpret, and communicate marginal
effects from nonlinear econometric models using PanelBox.

## Notebooks

| # | File | Topic | Level | Duration |
|---|------|-------|-------|----------|
| 01 | `01_me_fundamentals.ipynb` | Why coefficients ≠ marginal effects | Intermediate | 45-60 min |
| 02 | `02_discrete_me_complete.ipynb` | ME in binary, multinomial, ordered models | Int-Adv | 90-120 min |
| 03 | `03_count_me.ipynb` | AME and IRR in count models | Int-Adv | 60-75 min |
| 04 | `04_censored_me.ipynb` | Tobit and Heckman ME | Advanced | 75-90 min |
| 05 | `05_interaction_effects.ipynb` | Nonlinear interactions (Ai & Norton 2003) | Advanced | 60-75 min |
| 06 | `06_interpretation_guide.ipynb` | Reporting and communication best practices | Intermediate | 60 min |

## Datasets

| File | Used In | Description |
|------|---------|-------------|
| `data/mroz.csv` | 01, 02, 05, 06 | Mroz (1987) labor force participation |
| `data/mroz_hours.csv` | 04 | Extended Mroz with hours worked (censored) |
| `data/patents.csv` | 03 | Firm-level patent counts and R&D |
| `data/doctor_visits.csv` | 03 | German health data with visit counts |
| `data/job_satisfaction.csv` | 02 | Ordered satisfaction scale 1-5 |

> **Note:** If CSV files are not present the loaders in `utils/data_loaders.py`
> automatically generate synthetic equivalents with the same structure.

## Folder Structure

```
marginal_effects/
├── __init__.py
├── README.md
├── notebooks/          # Tutorial notebooks (run in order)
├── data/               # CSV datasets
├── outputs/
│   ├── plots/          # PNG/SVG figures produced by notebooks
│   └── tables/         # CSV and LaTeX table exports
├── solutions/          # Fully worked solution notebooks
└── utils/
    ├── __init__.py
    ├── data_loaders.py  # load_dataset(name) function
    └── me_helpers.py    # plot_forest(), format_me_table(), ...
```

## Setup

### 1. Install PanelBox (development version)

```bash
pip install -e /home/guhaase/projetos/panelbox
```

### 2. Import convention inside notebooks

```python
import sys
sys.path.insert(0, '/home/guhaase/projetos/panelbox')

import panelbox as pb
from panelbox.marginal_effects import (
    compute_ame,
    compute_mem,
    compute_mer,
    MarginalEffectsResult,
)
from panelbox.models.discrete.binary import PooledLogit, PooledProbit
from panelbox.models.count.poisson import PooledPoisson
from panelbox.models.count.negbin import NegativeBinomial
from panelbox.models.censored.tobit import PooledTobit
```

### 3. Import utilities

```python
import sys
sys.path.insert(0, '..')   # notebook is inside notebooks/
from utils.data_loaders import load_dataset
from utils.me_helpers import plot_forest, format_me_table
```

### 4. Run notebooks in order

Each notebook is self-contained but assumes knowledge from the previous ones.

## Key Concepts Covered

- **AME** — Average Marginal Effect
- **MEM** — Marginal Effect at the Mean
- **MER** — Marginal Effect at a Representative point
- **IRR** — Incidence Rate Ratio (count models)
- **Interaction effects** in nonlinear models (Ai & Norton, 2003)
- Confidence intervals via the delta method and bootstrap
- Publication-ready tables and forest plots

## References

- Mroz, T. A. (1987). *The sensitivity of an empirical model of married
  women's hours of work to economic and statistical assumptions.* Econometrica.
- Ai, C., & Norton, E. C. (2003). *Interaction terms in logit and probit
  models.* Economics Letters.
- Long, J. S., & Freese, J. (2014). *Regression Models for Categorical
  Dependent Variables Using Stata* (3rd ed.). Stata Press.
- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
