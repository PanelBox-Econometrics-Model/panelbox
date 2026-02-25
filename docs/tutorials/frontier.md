---
title: "Stochastic Frontier Tutorials"
description: "Interactive tutorials for stochastic frontier analysis, efficiency measurement, and TFP decomposition with PanelBox"
---

# Stochastic Frontier Tutorials

!!! info "Learning Path"
    **Prerequisites**: [Static Models](static-models.md) tutorials, basic MLE concepts
    **Time**: 3--6 hours
    **Level**: Intermediate -- Advanced

## Overview

Stochastic Frontier Analysis (SFA) separates random noise from technical inefficiency, allowing researchers to measure how close firms, hospitals, banks, or other decision-making units operate relative to the efficient frontier. This is fundamental for productivity analysis, regulatory benchmarking, and performance evaluation.

These tutorials cover the progression from basic cross-sectional SFA to advanced panel models, including the four-component model (GTRE) that separates persistent from transient inefficiency, and Total Factor Productivity (TFP) decomposition that attributes output growth to technical change, efficiency change, and scale effects.

The [SFA Tutorial notebook](sfa_tutorial.ipynb) provides additional background on the fundamentals.

## Notebooks

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Introduction to SFA](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/01_introduction_sfa.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/01_introduction_sfa.ipynb) |
| 2 | [Panel SFA Models](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/02_panel_sfa.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/02_panel_sfa.ipynb) |
| 3 | [Four-Component Model & TFP](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/03_four_component_tfp.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/03_four_component_tfp.ipynb) |
| 4 | [Determinants & Heterogeneity](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/04_determinants_heterogeneity.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/04_determinants_heterogeneity.ipynb) |
| 5 | [Testing & Model Comparison](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/05_testing_comparison.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/05_testing_comparison.ipynb) |
| 6 | [Complete Case Study](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/06_complete_case_study.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/06_complete_case_study.ipynb) |

## Learning Paths

### :material-lightning-bolt: Basic (3 hours)

Essential SFA methods for efficiency measurement:

**Notebooks**: 1, 2, 5

Covers SFA fundamentals, panel extensions, and model comparison. Sufficient for basic efficiency analysis.

### :material-trophy: Complete (6 hours)

Full stochastic frontier analysis coverage:

**Notebooks**: 1--6

Adds the four-component model (GTRE), TFP decomposition, inefficiency determinants, and a comprehensive case study.

## Key Concepts Covered

- **Stochastic frontier**: Separating noise ($v_{it}$) from inefficiency ($u_{it}$)
- **Production vs cost frontiers**: Sign convention and interpretation
- **Panel SFA models**: Battese-Coelli, Pitt-Lee, True FE/RE
- **Four-component model (GTRE)**: Persistent + transient inefficiency + firm heterogeneity
- **TFP decomposition**: Technical change, efficiency change, scale effects
- **Inefficiency determinants**: Modeling inefficiency as a function of covariates
- **Model selection**: LR tests, Vuong test, information criteria

## Quick Example

```python
from panelbox.frontier import StochasticFrontier

# Estimate a panel SFA model
sfa = StochasticFrontier(
    data=data,
    formula="log_output ~ log_capital + log_labor",
    entity_col="firm",
    time_col="year",
    frontier_type="production"
).fit()

print(sfa.summary())
print(f"Mean efficiency: {sfa.efficiency.mean():.4f}")
```

## Solutions

| Tutorial | Solution |
|----------|----------|
| 01. Introduction to SFA | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/solutions/01_introduction_sfa_solution.ipynb) |
| 02. Panel SFA Models | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/solutions/02_panel_sfa_solution.ipynb) |
| 03. Four-Component & TFP | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/solutions/03_four_component_tfp_solution.ipynb) |
| 04. Determinants & Heterogeneity | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/solutions/04_determinants_heterogeneity_solution.ipynb) |
| 05. Testing & Comparison | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/solutions/05_testing_comparison_solution.ipynb) |
| 06. Complete Case Study | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/solutions/06_complete_case_study_solution.ipynb) |

## Related Documentation

- [SFA Tutorial Notebook](sfa_tutorial.ipynb) -- Self-contained SFA tutorial
- [Theory: SFA Sign Convention](../user-guide/frontier/sign-convention.md) -- Production vs cost frontiers
- [User Guide](../user-guide/index.md) -- API reference
- [Visualization](../visualization/charts/index.md) -- Efficiency plots and frontier charts
