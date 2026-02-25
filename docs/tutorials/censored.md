---
title: "Censored & Selection Tutorials"
description: "Interactive tutorials for Tobit, Honore estimator, and Heckman panel selection models with PanelBox"
---

# Censored & Selection Tutorials

!!! info "Learning Path"
    **Prerequisites**: Basic MLE concepts, understanding of sample selection bias
    **Time**: 3--7 hours
    **Level**: Beginner -- Advanced

## Overview

Censored and selection models address two distinct but related data problems. Censoring occurs when the dependent variable is observed only within a certain range (e.g., income capped at zero, expenditure truncated above a threshold). Sample selection arises when the decision to observe the outcome is correlated with the outcome itself (e.g., wages observed only for those who choose to work).

These tutorials cover Tobit models for censored panel data, the Honore bias-corrected estimator for fixed effects Tobit, and the Panel Heckman selection model in both two-step and FIML variants. You will learn to test for selection bias, construct valid exclusion restrictions, and interpret the inverse Mills ratio (IMR).

## Notebooks

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Censoring Fundamentals](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/01_tobit_introduction.ipynb) | Beginner | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/01_tobit_introduction.ipynb) |
| 2 | [Tobit Panel Models](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/02_tobit_panel.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/02_tobit_panel.ipynb) |
| 3 | [Honore Bias-Corrected Estimator](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/03_honore_estimator.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/03_honore_estimator.ipynb) |
| 4 | [Panel Heckman (Two-Step)](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/04_heckman_selection.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/04_heckman_selection.ipynb) |
| 5 | [Heckman MLE (FIML)](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/05_heckman_mle.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/05_heckman_mle.ipynb) |
| 6 | [Identification & Exclusion Restrictions](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/06_identification.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/06_identification.ipynb) |
| 7 | [Marginal Effects](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/07_marginal_effects.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/07_marginal_effects.ipynb) |
| 8 | [Complete Case Study](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/08_complete_case_study.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/notebooks/08_complete_case_study.ipynb) |

## Learning Paths

### :material-lightning-bolt: Tobit (3 hours)

Censored data models:

**Notebooks**: 1, 2, 3

Covers censoring fundamentals, panel Tobit, and the Honore bias-corrected estimator.

### :material-flask: Selection (5 hours)

Add sample selection correction:

**Notebooks**: 1, 2, 3, 4, 5, 6

Covers Tobit, Panel Heckman (two-step and FIML), and identification strategies.

### :material-trophy: Complete (7 hours)

Full censored and selection model coverage:

**Notebooks**: 1--8

Adds marginal effects for censored models and a comprehensive case study.

## Key Concepts Covered

- **Censoring vs truncation**: Different data problems requiring different models
- **Tobit model**: MLE for censored outcomes (Type I, Type II)
- **Honore estimator**: Bias-corrected FE Tobit using pairwise trimming
- **Panel Heckman (two-step)**: Selection equation + outcome equation
- **Heckman FIML**: Full Information Maximum Likelihood joint estimation
- **Exclusion restrictions**: Variables in selection but not outcome equation
- **Inverse Mills Ratio (IMR)**: Correction term for selection bias
- **Marginal effects**: Unconditional, conditional on being uncensored
- **Selection bias testing**: Significance of the IMR / correlation parameter

## Quick Example

```python
from panelbox.models.censored import PanelHeckman

# Panel Heckman selection model
heckman = PanelHeckman(
    data=data,
    outcome_formula="wage ~ education + experience",
    selection_formula="employed ~ education + experience + children",
    entity_col="id",
    time_col="year"
).fit()

print(heckman.summary())
print(f"Selection correlation (rho): {heckman.rho:.4f}")
print(f"IMR significant: {heckman.imr_pvalue < 0.05}")
```

## Solutions

| Tutorial | Solution |
|----------|----------|
| 01. Censoring Fundamentals | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/solutions/01_tobit_introduction_solution.ipynb) |
| 02. Tobit Panel | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/solutions/02_tobit_panel_solution.ipynb) |
| 03. Honore Estimator | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/solutions/03_honore_estimator_solution.ipynb) |
| 04. Heckman Selection | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/solutions/04_heckman_selection_solution.ipynb) |
| 05. Heckman MLE | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/solutions/05_heckman_mle_solution.ipynb) |
| 06. Identification | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/solutions/06_identification_solution.ipynb) |
| 07. Marginal Effects | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/solutions/07_marginal_effects_solution.ipynb) |
| 08. Case Study | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/censored/solutions/08_complete_case_study_solution.ipynb) |

## Related Documentation

- [Theory: Selection Models](../theory/selection-theory.md) -- Mathematical foundations
- [Marginal Effects Tutorials](marginal-effects.md) -- AME for censored/selection models
- [User Guide](../user-guide/index.md) -- API reference
