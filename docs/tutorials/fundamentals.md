---
title: "Fundamentals Tutorials"
description: "Interactive tutorials for getting started with panel data analysis using PanelBox"
---

# Fundamentals Tutorials

!!! info "Learning Path"
    **Prerequisites**: Basic Python, pandas, introductory statistics
    **Time**: 3--4 hours
    **Level**: Beginner -- Intermediate

## Overview

These tutorials introduce the core concepts of panel data analysis. You will learn what panel data is, how to structure it, and how to estimate and interpret the fundamental models that form the basis of all panel econometrics.

Panel data combines cross-sectional and time-series dimensions, enabling researchers to control for unobserved heterogeneity and study dynamic relationships. These tutorials build your intuition for within-entity vs between-entity variation, the role of fixed and random effects, and how to choose the right estimator for your research question.

By the end of this series, you will be comfortable loading data into PanelBox, specifying models with formulas, and interpreting regression output -- skills you will use throughout every other tutorial category.

## Notebooks

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Introduction to Panel Data](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/tutorials/01_fundamentals/01_introduction_panel_data.ipynb) | Beginner | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/tutorials/01_fundamentals/01_introduction_panel_data.ipynb) |
| 2 | [Formulas & Specification](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/tutorials/01_fundamentals/02_formulas_specification.ipynb) | Beginner | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/tutorials/01_fundamentals/02_formulas_specification.ipynb) |
| 3 | [Estimation & Interpretation](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/tutorials/01_fundamentals/03_estimation_interpretation.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/tutorials/01_fundamentals/03_estimation_interpretation.ipynb) |
| 4 | [Spatial Fundamentals](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/tutorials/01_fundamentals/04_spatial_fundamentals.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/tutorials/01_fundamentals/04_spatial_fundamentals.ipynb) |

## Learning Paths

### :material-lightning-bolt: Quick Start (1.5 hours)

Focus on the essentials to get productive quickly:

**Notebooks**: 1, 2, 3

You will learn how to load panel data, specify models using PanelBox formulas, and interpret estimation results. This is sufficient to move on to the [Static Models](static-models.md) tutorials.

### :material-book-open-variant: Comprehensive (3 hours)

Work through all four notebooks in order:

**Notebooks**: 1, 2, 3, 4

This path adds spatial data fundamentals, which is useful if you plan to work with geographic or network data later.

## Key Concepts Covered

- **Panel data structure**: Entity (cross-section) and time dimensions
- **Balanced vs unbalanced panels**: Handling missing observations
- **Within vs between variation**: How panel data decomposes total variation
- **Formula syntax**: PanelBox formula interface (`y ~ x1 + x2`)
- **Entity and time identifiers**: Setting up `entity_col` and `time_col`
- **Result interpretation**: Coefficients, standard errors, p-values, R-squared
- **Programmatic access**: Extracting results for further analysis

## Quick Example

```python
import panelbox as pb

# Load a built-in dataset
data = pb.load_dataset("grunfeld")

# Estimate a Fixed Effects model
results = pb.FixedEffects(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
).fit()

# View the summary
print(results.summary())
```

## Solutions

| Tutorial | Solution |
|----------|----------|
| 01. Introduction to Panel Data | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/solutions/01_fundamentals/01_introduction_solutions.ipynb) |
| 02. Formulas & Specification | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/solutions/01_fundamentals/02_formulas_solutions.ipynb) |
| 03. Estimation & Interpretation | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/solutions/01_fundamentals/03_estimation_solutions.ipynb) |
| 04. Spatial Fundamentals | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/solutions/01_fundamentals/04_spatial_solutions.ipynb) |

## What You Will Be Able To Do

After completing these tutorials, you will be able to:

- Load and explore panel datasets with PanelBox
- Understand the structure of balanced and unbalanced panels
- Specify regression models using the formula interface
- Estimate Pooled OLS, Fixed Effects, and Random Effects models
- Interpret coefficients, standard errors, and goodness-of-fit measures
- Access results programmatically for custom analysis

## Related Documentation

- [Getting Started Tutorial](fundamentals.md) -- Detailed first-steps walkthrough
- [Static Models Tutorials](static-models.md) -- Next step after fundamentals
- [User Guide](../user-guide/index.md) -- Comprehensive model reference
- [Standard Errors Tutorials](standard-errors.md) -- Robust inference methods
