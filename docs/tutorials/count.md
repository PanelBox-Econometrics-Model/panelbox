---
title: "Count Data Tutorials"
description: "Interactive tutorials for Poisson, negative binomial, PPML, and zero-inflated panel count models with PanelBox"
---

# Count Data Tutorials

!!! info "Learning Path"
    **Prerequisites**: [Static Models](static-models.md) tutorials, basic MLE concepts
    **Time**: 3--6 hours
    **Level**: Beginner -- Advanced

## Overview

Count data models apply when the dependent variable is a non-negative integer: patent counts, number of doctor visits, trade flows, accident frequencies. Standard linear regression is inappropriate because it can predict negative values and ignores the discrete, non-negative nature of count data.

These tutorials cover the Poisson model (pooled, fixed effects, random effects), quasi-maximum likelihood Poisson (QML), the Pseudo Poisson Maximum Likelihood estimator (PPML) for gravity models in trade, negative binomial models for overdispersion, and zero-inflated models for excess zeros.

The [PPML Gravity notebook](ppml_gravity.ipynb) provides a self-contained introduction to gravity models using PPML.

## Notebooks

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Poisson Introduction](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/01_poisson_introduction.ipynb) | Beginner | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/01_poisson_introduction.ipynb) |
| 2 | [Negative Binomial](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/02_negative_binomial.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/02_negative_binomial.ipynb) |
| 3 | [FE/RE Count Models](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/03_fe_re_count.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/03_fe_re_count.ipynb) |
| 4 | [PPML Gravity Models](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/04_ppml_gravity.ipynb) | Intermediate | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/04_ppml_gravity.ipynb) |
| 5 | [Zero-Inflated Models](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/05_zero_inflated.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/05_zero_inflated.ipynb) |
| 6 | [Marginal Effects for Count](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/06_marginal_effects_count.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/06_marginal_effects_count.ipynb) |
| 7 | [Innovation Case Study](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/07_innovation_case_study.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/notebooks/07_innovation_case_study.ipynb) |

## Learning Paths

### :material-lightning-bolt: Core (3 hours)

Essential count data methods:

**Notebooks**: 1, 2, 3, 4

Covers Poisson, negative binomial, FE/RE count models, and PPML for gravity equations.

### :material-trophy: Complete (6 hours)

Full count data analysis coverage:

**Notebooks**: 1--7

Adds zero-inflated models, marginal effects, overdispersion diagnostics, and a complete case study.

## Key Concepts Covered

- **Poisson regression**: Exponential mean function, equidispersion assumption
- **FE Poisson**: Conditional MLE for panel count data
- **RE Poisson**: Random effects with Gauss-Hermite quadrature
- **QML Poisson**: Robust to distributional misspecification
- **PPML**: Pseudo Poisson for gravity models (Santos Silva & Tenreyro, 2006)
- **Negative binomial**: Accommodating overdispersion
- **Zero-inflation**: Modeling excess zeros with a two-part model
- **Overdispersion tests**: Cameron-Trivedi, Dean's score test
- **Marginal effects**: Incidence rate ratios and semi-elasticities

## Quick Example

```python
from panelbox.models.count import PoissonFE, PPML

# FE Poisson
poisson = PoissonFE(
    data=data,
    formula="patents ~ rd_spending + firm_size",
    entity_col="firm",
    time_col="year"
).fit()

print(poisson.summary())

# PPML for gravity
ppml = PPML(
    data=trade_data,
    formula="trade_flow ~ log_gdp_i + log_gdp_j + log_distance",
    entity_col="pair",
    time_col="year"
).fit()
```

## Solutions

| Tutorial | Solution |
|----------|----------|
| 01. Poisson Introduction | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/solutions/01_poisson_introduction_solutions.ipynb) |
| 02. Negative Binomial | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/solutions/02_negative_binomial_solutions.ipynb) |
| 03. FE/RE Count | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/solutions/03_fe_re_count_solutions.ipynb) |
| 04. PPML Gravity | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/solutions/04_ppml_gravity_solutions.ipynb) |
| 05. Zero-Inflated | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/solutions/05_zero_inflated_solutions.ipynb) |
| 06. Marginal Effects | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/solutions/06_marginal_effects_count_solutions.ipynb) |
| 07. Case Study | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/count/solutions/07_innovation_case_study_solutions.ipynb) |

## Related Documentation

- [PPML Gravity Tutorial](ppml_gravity.ipynb) -- Self-contained gravity model notebook
- [Marginal Effects Tutorials](marginal-effects.md) -- AME for nonlinear models
- [User Guide](../user-guide/index.md) -- API reference
