---
title: "Quantile Regression Tutorials"
description: "Interactive tutorials for panel quantile regression, Canay two-step, location-scale models, and QTE with PanelBox"
---

# Quantile Regression Tutorials

!!! info "Learning Path"
    **Prerequisites**: [Static Models](static-models.md) tutorials, basic understanding of quantiles
    **Time**: 4--8 hours
    **Level**: Intermediate -- Advanced

## Overview

Standard regression focuses on the conditional mean. Quantile regression extends this to model the entire conditional distribution, revealing how covariates affect different parts of the outcome distribution differently. For example, a policy might reduce income inequality not by changing the mean but by raising the lower quantiles relative to the upper ones.

These tutorials cover pooled quantile regression, fixed effects quantile methods (Canay two-step and penalized approaches), location-scale models, bootstrap inference, and quantile treatment effects (QTE). You will learn to estimate quantile processes, test for heterogeneous effects, and ensure monotonicity across quantiles.

The existing [Quantile Treatment Effects Tutorial](quantile.md) provides additional depth on QTE methods, and the [Panel Quantile Regression notebook](intro_panel_quantile_regression.ipynb) offers a self-contained introduction.

## Notebooks

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Quantile Regression Fundamentals](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/01_quantile_regression_fundamentals.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/01_quantile_regression_fundamentals.ipynb) |
| 2 | [Multiple Quantiles & Process Plots](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/02_multiple_quantiles_process.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/02_multiple_quantiles_process.ipynb) |
| 3 | [Fixed Effects (Canay Two-Step)](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/03_fixed_effects_canay.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/03_fixed_effects_canay.ipynb) |
| 4 | [Fixed Effects (Penalized)](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/04_fixed_effects_penalty.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/04_fixed_effects_penalty.ipynb) |
| 5 | [Location-Scale Models](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/05_location_scale_models.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/05_location_scale_models.ipynb) |
| 6 | [Advanced Diagnostics](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/06_advanced_diagnostics.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/06_advanced_diagnostics.ipynb) |
| 7 | [Bootstrap Inference](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/07_bootstrap_inference.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/07_bootstrap_inference.ipynb) |
| 8 | [Monotonicity & Non-Crossing](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/08_monotonicity_non_crossing.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/08_monotonicity_non_crossing.ipynb) |
| 9 | [Quantile Treatment Effects](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/09_quantile_treatment_effects.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/09_quantile_treatment_effects.ipynb) |
| 10 | [Dynamic Quantile Models](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/10_dynamic_quantile_models.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/10_dynamic_quantile_models.ipynb) |

## Learning Paths

### :material-lightning-bolt: Essential (4 hours)

Core quantile methods for applied research:

**Notebooks**: 1, 2, 3, 5

Covers fundamentals, quantile process estimation, fixed effects quantile (Canay), and location-scale models.

### :material-trophy: Complete (8 hours)

Master every quantile technique:

**Notebooks**: 1--10

Adds penalized FE, diagnostics, bootstrap inference, non-crossing constraints, QTE, and dynamic models.

## Key Concepts Covered

- **Quantile regression**: Modeling conditional quantiles instead of the mean
- **Quantile process**: Estimating a full range of quantiles (e.g., 0.10 to 0.90)
- **Canay two-step**: Fixed effects quantile regression via mean-demeaning
- **Penalized FE quantile**: Alternative approach with regularization
- **Location-scale models**: Joint modeling of location and scale
- **Bootstrap inference**: Resampling-based confidence intervals and tests
- **Monotonicity**: Ensuring quantile functions do not cross
- **QTE**: Quantile treatment effects for heterogeneous impacts
- **Dynamic quantile**: Quantile regression with lagged dependent variables

## Quick Example

```python
from panelbox.models.quantile import CanayTwoStep

# Canay two-step FE quantile regression
model = CanayTwoStep(
    data=data,
    formula="wage ~ education + experience",
    entity_col="id",
    time_col="year",
    quantiles=[0.10, 0.25, 0.50, 0.75, 0.90]
).fit()

print(model.summary())
```

## Solutions

| Tutorial | Solution |
|----------|----------|
| 01. Fundamentals | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/solutions/solutions_01.ipynb) |
| 02. Multiple Quantiles | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/solutions/solutions_02.ipynb) |
| 03. Canay Two-Step | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/solutions/solutions_03.ipynb) |
| 04. Penalized FE | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/solutions/solutions_04.ipynb) |
| 05. Location-Scale | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/solutions/solutions_05.ipynb) |
| 06. Advanced Diagnostics | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/solutions/solutions_06.ipynb) |
| 07. Bootstrap Inference | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/solutions/solutions_07.ipynb) |
| 08. Non-Crossing | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/solutions/solutions_08.ipynb) |
| 09. QTE | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/solutions/solutions_09.ipynb) |
| 10. Dynamic Quantile | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/solutions/solutions_10.ipynb) |

## Related Documentation

- [Quantile Treatment Effects Tutorial](quantile.md) -- In-depth QTE methods
- [Panel Quantile Regression](intro_panel_quantile_regression.ipynb) -- Self-contained notebook
- [Theory: Location-Scale](../user-guide/quantile/location-scale.md) -- Mathematical foundations
- [User Guide](../user-guide/index.md) -- API reference
