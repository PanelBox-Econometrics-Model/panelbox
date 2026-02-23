# Production & Deployment Tutorials

Take your PanelBox models from estimation to production deployment.

## Overview

| # | Notebook | Level | Duration | Topics |
|---|----------|-------|----------|--------|
| 1 | Predict Fundamentals | Beginner | 45-60 min | `predict(newdata)` for all model types |
| 2 | Save & Load Models | Beginner | 30-45 min | Persistence, serialization, portability |
| 3 | Production Pipeline | Intermediate | 60-90 min | PanelPipeline, end-to-end workflow |
| 4 | Model Validation | Intermediate | 45-60 min | Pre-deployment checks, diagnostics |
| 5 | Model Versioning | Intermediate | 45-60 min | ModelRegistry, drift detection |
| 6 | Case Study: Bank LGD | Advanced | 90-120 min | Complete banking production workflow |

## Prerequisites

- Familiarity with PanelBox estimation (Module 03 or 09)
- Basic Python knowledge

## Learning Pathways

- **Quick Start**: 01 &rarr; 02 &rarr; 03
- **Full Production**: 01 &rarr; 02 &rarr; 03 &rarr; 04 &rarr; 05
- **Banking Focus**: 01 &rarr; 03 &rarr; 06
- **Complete**: 01 &rarr; 02 &rarr; 03 &rarr; 04 &rarr; 05 &rarr; 06

## Quick Start

```python
from panelbox.production import PanelPipeline
from panelbox.gmm import DifferenceGMM

# Build and train pipeline
pipeline = PanelPipeline(model_class=DifferenceGMM, model_params={
    'dep_var': 'lgd_logit',
    'lags': 1,
    'exog_vars': ['saldo_real', 'pib_growth', 'selic'],
    'id_var': 'contract_id',
    'time_var': 'month',
})
pipeline.fit(training_data)
pipeline.save('model.pkl')

# In production:
pipeline = PanelPipeline.load('model.pkl')
predictions = pipeline.predict(new_data)
```

## Directory Structure

```
production/
├── README.md               # This file
├── GETTING_STARTED.md      # Setup guide
├── data/                   # Tutorial datasets
├── notebooks/              # 6 tutorial notebooks
├── solutions/              # Exercise solutions
├── outputs/                # Generated figures, models, tables
├── utils/                  # Helper functions
└── tests/                  # Verification tests
```

## Datasets

| Dataset | Description | Rows |
|---------|-------------|------|
| `firm_panel.csv` | Firm-level panel (N=100, T=20) | 2,000 |
| `bank_lgd.csv` | Banking LGD data (N=200, T=15) | 3,000 |
| `macro_quarterly.csv` | Macro indicators (N=30, T=40) | 1,200 |
| `new_firms.csv` | Out-of-sample firms | 100 |
| `new_bank_data.csv` | New bank observations | 150 |
| `future_macro.csv` | Future exogenous variables | 120 |

## Running Tests

```bash
pytest examples/production/tests/ -v
```
