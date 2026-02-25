---
title: "Marginal Effects Tutorials"
description: "Interactive tutorials for computing and interpreting marginal effects in nonlinear panel models with PanelBox"
---

# Marginal Effects Tutorials

!!! info "Learning Path"
    **Prerequisites**: [Discrete Choice](discrete.md) or [Count Models](count.md) tutorials
    **Time**: 2--5 hours
    **Level**: Beginner -- Advanced

## Overview

In nonlinear models (logit, probit, Poisson, Tobit), raw coefficients do not have a direct marginal interpretation. A one-unit change in $x$ does not produce a constant change in the outcome -- the effect depends on the values of all covariates. Marginal effects bridge this gap by computing the actual change in the predicted outcome for a unit change in each covariate.

These tutorials cover the three main types of marginal effects -- Average Marginal Effects (AME), Marginal Effects at the Mean (MEM), and Marginal Effects at Representative values (MER) -- for discrete choice, count, and censored models. You will learn when to use each type, how to compute them with PanelBox, and how to handle interaction terms and categorical variables.

## Notebooks

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Marginal Effects Fundamentals](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/notebooks/01_me_fundamentals.ipynb) | Beginner | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/notebooks/01_me_fundamentals.ipynb) |
| 2 | [Discrete Model Effects](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/notebooks/02_discrete_me_complete.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/notebooks/02_discrete_me_complete.ipynb) |
| 3 | [Count Model Effects](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/notebooks/03_count_me.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/notebooks/03_count_me.ipynb) |
| 4 | [Censored Model Effects](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/notebooks/04_censored_me.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/notebooks/04_censored_me.ipynb) |
| 5 | [Interaction Effects](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/notebooks/05_interaction_effects.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/notebooks/05_interaction_effects.ipynb) |
| 6 | [Interpretation Guide](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/notebooks/06_interpretation_guide.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/notebooks/06_interpretation_guide.ipynb) |

## Learning Paths

### :material-lightning-bolt: Essential (2 hours)

Core marginal effects concepts:

**Notebooks**: 1, 2

Covers AME vs MEM vs MER and marginal effects for discrete choice models. Sufficient for most applied work.

### :material-trophy: Complete (5 hours)

Full marginal effects coverage:

**Notebooks**: 1--6

Adds count model effects, censored model effects, interaction terms, and a comprehensive interpretation guide.

## Key Concepts Covered

- **AME (Average Marginal Effects)**: Average across all observations
- **MEM (Marginal Effects at the Mean)**: Evaluated at sample means
- **MER (Marginal Effects at Representative values)**: Evaluated at user-specified values
- **Delta method**: Standard errors for nonlinear transformations
- **Discrete changes**: Effects of switching a binary variable from 0 to 1
- **Interaction effects**: Marginal effects with interaction terms (not just the coefficient)
- **Incidence rate ratios**: Exponentiated Poisson/NB coefficients
- **Elasticities**: Percentage change interpretation
- **Visualization**: Marginal effect plots across covariate ranges

## AME vs MEM vs MER

| Type | Computation | Best For |
|------|-------------|----------|
| **AME** | Compute ME for each observation, average | Population-level policy effects |
| **MEM** | Compute ME at sample means | Representative individual effect |
| **MER** | Compute ME at specified values | Scenario analysis, subgroup effects |

!!! tip "Rule of Thumb"
    AME is the most commonly reported in applied work because it represents the average policy effect across the population and is robust to the distribution of covariates.

## Quick Example

```python
from panelbox.models.discrete import PooledLogit

# Estimate a logit model
logit = PooledLogit(
    data=data,
    formula="outcome ~ x1 + x2 + x3",
    entity_col="id",
    time_col="year"
).fit()

# Average Marginal Effects
ame = logit.marginal_effects(method="ame")
print(ame.summary())

# Marginal Effects at the Mean
mem = logit.marginal_effects(method="mem")
print(mem.summary())
```

## Solutions

| Tutorial | Solution |
|----------|----------|
| 01. Fundamentals | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/solutions/01_me_fundamentals_solution.ipynb) |
| 02. Discrete Effects | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/solutions/02_discrete_me_complete_solution.ipynb) |
| 03. Count Effects | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/solutions/03_count_me_solution.ipynb) |
| 04. Censored Effects | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/solutions/04_censored_me_solution.ipynb) |
| 05. Interaction Effects | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/solutions/05_interaction_effects_solution.ipynb) |
| 06. Interpretation Guide | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/marginal_effects/solutions/06_interpretation_guide_solution.ipynb) |

## Related Documentation

- [Discrete Choice Tutorials](discrete.md) -- Binary, ordered, multinomial models
- [Count Models Tutorials](count.md) -- Poisson, NB, PPML
- [Censored & Selection Tutorials](censored.md) -- Tobit, Heckman
- [User Guide](../user-guide/index.md) -- API reference
