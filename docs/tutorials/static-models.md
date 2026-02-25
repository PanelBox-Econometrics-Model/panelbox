---
title: "Static Models Tutorials"
description: "Interactive tutorials for Pooled OLS, Fixed Effects, Random Effects, and IV estimation with PanelBox"
---

# Static Models Tutorials

!!! info "Learning Path"
    **Prerequisites**: [Fundamentals](fundamentals.md) tutorials completed
    **Time**: 5--8 hours
    **Level**: Beginner -- Advanced

## Overview

Static panel models are the workhorses of empirical research. These tutorials cover every major estimator -- from Pooled OLS through Fixed Effects, Random Effects, Between and First Difference estimators, to Instrumental Variables -- with hands-on code and real datasets.

You will learn when to use each estimator, how to choose between Fixed and Random Effects with the Hausman test, and how to apply robust standard errors and two-way fixed effects. The advanced notebooks tackle IV estimation and systematic model comparison workflows.

The existing [Static Models Tutorial](static-models.md) provides additional context on the theory behind these models, including the decision tree for choosing between estimators.

## Notebooks

### Fundamentals

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Pooled OLS Introduction](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/fundamentals/01_pooled_ols_introduction.ipynb) | Beginner | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/fundamentals/01_pooled_ols_introduction.ipynb) |
| 2 | [Fixed Effects](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/fundamentals/02_fixed_effects.ipynb) | Beginner | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/fundamentals/02_fixed_effects.ipynb) |
| 3 | [Random Effects & Hausman Test](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/fundamentals/03_random_effects_hausman.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/fundamentals/03_random_effects_hausman.ipynb) |

### Advanced

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 4 | [First Difference & Between](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/advanced/04_first_difference_between.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/advanced/04_first_difference_between.ipynb) |
| 5 | [Panel IV (2SLS)](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/advanced/05_panel_iv.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/advanced/05_panel_iv.ipynb) |
| 6 | [Comparison of Estimators](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/advanced/06_comparison_estimators.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/advanced/06_comparison_estimators.ipynb) |

### Expert

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 7 | [IV Diagnostics Advanced](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/expert/07_iv_diagnostics_advanced.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/expert/07_iv_diagnostics_advanced.ipynb) |

## Learning Paths

### :material-lightning-bolt: Basic (3 hours)

Essential models for any panel analysis:

**Notebooks**: 1, 2, 3

Covers Pooled OLS, Fixed Effects, Random Effects, and the Hausman test. Sufficient for most applied work.

### :material-flask: Advanced (5 hours)

Add alternative estimators and comparison workflows:

**Notebooks**: 1, 2, 3, 4, 6

Includes First Difference, Between estimator, and a systematic model comparison framework.

### :material-trophy: Complete (8 hours)

Master every static estimator including IV:

**Notebooks**: 1--7

Adds Instrumental Variables (2SLS), IV diagnostics, and advanced comparison techniques.

## Key Concepts Covered

- **Pooled OLS**: When pooling is appropriate and its limitations
- **Fixed Effects**: Within transformation, entity demeaning, time-invariant variables
- **Random Effects**: GLS estimation, RE assumptions, efficiency gains
- **Hausman test**: Systematic FE vs RE selection
- **Between estimator**: Cross-sectional variation only
- **First Difference**: Alternative to FE for persistent data
- **Two-way FE**: Entity and time fixed effects simultaneously
- **Panel IV**: Two-stage least squares for endogeneity
- **Robust SE**: Heteroskedasticity and cluster-robust standard errors

## Quick Example

```python
import panelbox as pb

data = pb.load_dataset("grunfeld")

# Fixed Effects
fe = pb.FixedEffects(
    data=data, formula="invest ~ value + capital",
    entity_col="firm", time_col="year"
).fit()

# Random Effects
re = pb.RandomEffects(
    data=data, formula="invest ~ value + capital",
    entity_col="firm", time_col="year"
).fit()

# Hausman test: FE vs RE
from panelbox.validation import hausman_test
hausman = hausman_test(fe, re)
print(f"Hausman p-value: {hausman.pvalue:.4f}")
```

## Related Documentation

- [Getting Started Tutorial](fundamentals.md) -- First steps with PanelBox
- [Static Models Tutorial](static-models.md) -- Detailed walkthrough with theory
- [Standard Errors Tutorials](standard-errors.md) -- Robust inference for panel models
- [Validation & Diagnostics](validation.md) -- Testing model assumptions
- [User Guide](../user-guide/index.md) -- API reference
