---
title: "Discrete Choice Tutorials"
description: "Interactive tutorials for binary, ordered, and multinomial panel discrete choice models with PanelBox"
---

# Discrete Choice Tutorials

!!! info "Learning Path"
    **Prerequisites**: [Static Models](static-models.md) tutorials, basic MLE concepts
    **Time**: 3--7 hours
    **Level**: Beginner -- Advanced

## Overview

Discrete choice models apply when the dependent variable is categorical: a binary outcome (yes/no), an ordered outcome (rating scales), or a multinomial choice (selecting from multiple alternatives). Standard linear estimators are inappropriate for these data types because they can produce predictions outside the valid range and violate distributional assumptions.

These tutorials cover the full spectrum of panel discrete choice models: binary logit and probit (pooled, fixed effects, random effects), ordered models, conditional and multinomial logit, and dynamic discrete choice. You will learn to estimate marginal effects, assess model fit, and test for the IIA assumption in multinomial models.

The [Multinomial Logit notebook](multinomial_tutorial.ipynb) provides a self-contained introduction to multinomial choice modeling.

## Notebooks

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Binary Choice Introduction](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/01_binary_choice_introduction.ipynb) | Beginner | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/01_binary_choice_introduction.ipynb) |
| 2 | [Fixed Effects Logit](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/02_fixed_effects_logit.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/02_fixed_effects_logit.ipynb) |
| 3 | [Random Effects Probit](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/03_random_effects.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/03_random_effects.ipynb) |
| 4 | [Marginal Effects](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/04_marginal_effects.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/04_marginal_effects.ipynb) |
| 5 | [Conditional Logit (McFadden)](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/05_conditional_logit.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/05_conditional_logit.ipynb) |
| 6 | [Multinomial Logit](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/06_multinomial_logit.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/06_multinomial_logit.ipynb) |
| 7 | [Ordered Models](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/07_ordered_models.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/07_ordered_models.ipynb) |
| 8 | [Dynamic Discrete Choice](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/08_dynamic_discrete.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/08_dynamic_discrete.ipynb) |
| 9 | [Complete Case Study](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/09_complete_case_study.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/09_complete_case_study.ipynb) |

## Learning Paths

### :material-lightning-bolt: Binary Models (3 hours)

Essential binary choice methods:

**Notebooks**: 1, 2, 3, 4

Covers logit/probit, FE logit, RE probit, and marginal effects. Sufficient for most binary outcome analyses.

### :material-trophy: Full (7 hours)

Complete discrete choice coverage:

**Notebooks**: 1--9

Adds conditional logit, multinomial logit, ordered models, dynamic discrete choice, and a case study.

## Key Concepts Covered

- **Logit vs Probit**: Link functions and interpretation
- **FE Logit (Conditional)**: Chamberlain's conditional logit for panel data
- **RE Probit**: Random effects with Gauss-Hermite quadrature
- **Marginal effects**: AME, MEM, and MER for nonlinear models
- **Conditional Logit (McFadden)**: Choice-specific attributes
- **Multinomial Logit**: Unordered multi-category outcomes
- **IIA assumption**: Independence of Irrelevant Alternatives
- **Ordered Logit/Probit**: Ordered categorical outcomes with thresholds
- **Dynamic models**: State dependence vs unobserved heterogeneity

## Quick Example

```python
from panelbox.models.discrete import FixedEffectsLogit

# Conditional FE Logit
fe_logit = FixedEffectsLogit(
    data=data,
    formula="outcome ~ x1 + x2 + x3",
    entity_col="id",
    time_col="year"
).fit()

print(fe_logit.summary())

# Marginal effects
me = fe_logit.marginal_effects()
print(me.summary())
```

## Solutions

| Tutorial | Solution |
|----------|----------|
| 01. Binary Choice | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/solutions/01_binary_choice_solutions.ipynb) |
| 02. Fixed Effects Logit | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/solutions/02_fixed_effects_solutions.ipynb) |
| 03. Random Effects | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/solutions/03_random_effects_solutions.ipynb) |
| 04. Marginal Effects | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/solutions/04_marginal_effects_solutions.ipynb) |
| 05. Conditional Logit | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/solutions/05_conditional_logit_solutions.ipynb) |
| 06. Multinomial Logit | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/solutions/06_multinomial_logit_solutions.ipynb) |
| 07. Ordered Models | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/solutions/07_ordered_models_solutions.ipynb) |
| 08. Dynamic Discrete | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/solutions/08_dynamic_discrete_solutions.ipynb) |
| 09. Case Study | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/solutions/09_complete_case_study_solutions.ipynb) |

## Related Documentation

- [Multinomial Logit Tutorial](multinomial_tutorial.ipynb) -- Self-contained notebook
- [Theory: Multinomial Logit](../user-guide/discrete/multinomial.md) -- Mathematical foundations
- [Marginal Effects Tutorials](marginal-effects.md) -- Detailed marginal effects guide
- [User Guide](../user-guide/index.md) -- API reference
