---
title: "Panel VAR Tutorials"
description: "Interactive tutorials for panel vector autoregression, IRF, FEVD, Granger causality, and VECM with PanelBox"
---

# Panel VAR Tutorials

!!! info "Learning Path"
    **Prerequisites**: Time series basics, [GMM](gmm.md) concepts helpful
    **Time**: 4--7 hours
    **Level**: Intermediate -- Advanced

## Overview

Panel Vector Autoregression (Panel VAR) combines the multivariate structure of VAR models with the cross-sectional richness of panel data. This allows researchers to study dynamic interdependencies between multiple variables across many entities, while controlling for unobserved heterogeneity.

These tutorials cover the complete Panel VAR workflow: estimation via OLS and GMM, impulse response functions (IRFs), forecast error variance decomposition (FEVD), Granger causality testing, and the error-correction extension (VECM) for cointegrated panels. You will learn to select lag orders, assess model stability, and generate forecasts.

The existing [Panel VAR Complete Guide](var.md) provides additional theoretical context and a comprehensive decision tree for choosing between VAR and VECM.

## Notebooks

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Panel VAR Introduction](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/01_var_introduction.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/01_var_introduction.ipynb) |
| 2 | [IRF Analysis](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/02_irf_analysis.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/02_irf_analysis.ipynb) |
| 3 | [FEVD (Variance Decomposition)](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/03_fevd_decomposition.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/03_fevd_decomposition.ipynb) |
| 4 | [Granger Causality](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/04_granger_causality.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/04_granger_causality.ipynb) |
| 5 | [VECM (Cointegration)](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/05_vecm_cointegration.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/05_vecm_cointegration.ipynb) |
| 6 | [Dynamic GMM Estimation](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/06_dynamic_gmm.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/06_dynamic_gmm.ipynb) |
| 7 | [Case Study](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/07_case_study.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/07_case_study.ipynb) |

## Learning Paths

### :material-lightning-bolt: Core (4 hours)

Essential Panel VAR methods:

**Notebooks**: 1, 2, 3, 4

Covers estimation, IRFs, FEVD, and Granger causality. Sufficient for most empirical applications.

### :material-trophy: Advanced (7 hours)

Complete Panel VAR and VECM coverage:

**Notebooks**: 1--7

Adds VECM for cointegrated systems, GMM-based estimation, and a full applied case study.

## Key Concepts Covered

- **Panel VAR specification**: Lag selection (AIC, BIC, HQIC)
- **OLS and GMM estimation**: Fixed effects with Helmert transformation
- **Impulse Response Functions**: Orthogonalized and cumulative IRFs
- **FEVD**: Variance decomposition over forecast horizons
- **Granger causality**: Testing predictive relationships
- **Model stability**: Eigenvalue tests and companion matrix
- **VECM**: Error-correction models for cointegrated panels
- **Forecasting**: Out-of-sample prediction with Panel VAR

## Quick Example

```python
from panelbox.var import PanelVAR

# Estimate a Panel VAR
var = PanelVAR(
    data=data,
    variables=["gdp", "investment", "consumption"],
    entity_col="country",
    time_col="year",
    lags=2
).fit()

# Impulse Response Functions
irf = var.irf(periods=10)
irf.plot()

# Granger Causality
gc = var.granger_causality()
print(gc.summary())
```

## Solutions

| Tutorial | Solution |
|----------|----------|
| 01. VAR Introduction | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/solutions/01_var_introduction_solutions.ipynb) |
| 02. IRF Analysis | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/solutions/02_irf_analysis_solutions.ipynb) |
| 03. FEVD Decomposition | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/solutions/03_fevd_decomposition_solutions.ipynb) |
| 04. Granger Causality | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/solutions/04_granger_causality_solutions.ipynb) |
| 05. VECM Cointegration | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/solutions/05_vecm_cointegration_solutions.ipynb) |
| 06. Dynamic GMM | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/solutions/06_dynamic_gmm_solutions.ipynb) |
| 07. Case Study | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/solutions/07_case_study_solutions.ipynb) |

## Related Documentation

- [Panel VAR Complete Guide](var.md) -- Detailed theory and decision tree
- [Theory: Panel Cointegration](../diagnostics/cointegration/index.md) -- Cointegration testing foundations
- [User Guide](../user-guide/index.md) -- API reference
- [Validation & Diagnostics](validation.md) -- Unit root and cointegration tests
