---
title: "Standard Errors Tutorials"
description: "Interactive tutorials for robust, clustered, HAC, Driscoll-Kraay, and bootstrap standard errors with PanelBox"
---

# Standard Errors Tutorials

!!! info "Learning Path"
    **Prerequisites**: [Fundamentals](fundamentals.md) tutorials, basic econometrics
    **Time**: 3--8 hours
    **Level**: Beginner -- Advanced

## Overview

Correct standard errors are essential for valid inference. Panel data frequently violates the classical OLS assumptions: errors may be heteroskedastic, serially correlated within entities, or cross-sectionally dependent across entities. Using the wrong standard errors leads to inflated t-statistics, misleadingly small p-values, and incorrect confidence intervals.

These tutorials cover the complete range of robust inference methods available in PanelBox: heteroskedasticity-consistent (HC0--HC3) standard errors, cluster-robust errors, Newey-West HAC for serial correlation, Conley spatial HAC for geographic correlation, Driscoll-Kraay for cross-sectional dependence, Panel Corrected Standard Errors (PCSE), and bootstrap methods.

The existing [Standard Errors Series](standard-errors.md) provides additional depth including a decision tree for method selection.

## Which Method Should I Use?

| Data Characteristic | Recommended Method | Notebook |
|---------------------|-------------------|----------|
| Heteroskedasticity only | HC0--HC3 | 01 |
| Within-entity correlation | Cluster-robust | 02 |
| Serial correlation (known structure) | Newey-West HAC | 03 |
| Geographic/spatial correlation | Conley Spatial HAC | 04 |
| Cross-sectional dependence (large T) | Driscoll-Kraay | 06 |
| Small sample, unknown structure | Bootstrap | 07 |
| MLE models | Sandwich estimator | 05 |

## Notebooks

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Robust Fundamentals (HC0--HC3)](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/standard_errors/notebooks/01_robust_fundamentals.ipynb) | Beginner | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/standard_errors/notebooks/01_robust_fundamentals.ipynb) |
| 2 | [Clustering in Panels](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/standard_errors/notebooks/02_clustering_panels.ipynb) | Beginner | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/standard_errors/notebooks/02_clustering_panels.ipynb) |
| 3 | [HAC (Newey-West)](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/standard_errors/notebooks/03_hac_autocorrelation.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/standard_errors/notebooks/03_hac_autocorrelation.ipynb) |
| 4 | [Spatial HAC (Conley)](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/standard_errors/notebooks/04_spatial_errors.ipynb) | Intermediate | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/standard_errors/notebooks/04_spatial_errors.ipynb) |
| 5 | [MLE Sandwich Inference](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/standard_errors/notebooks/05_mle_inference.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/standard_errors/notebooks/05_mle_inference.ipynb) |
| 6 | [Driscoll-Kraay & PCSE](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/standard_errors/notebooks/06_bootstrap_quantile.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/standard_errors/notebooks/06_bootstrap_quantile.ipynb) |
| 7 | [Methods Comparison](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/standard_errors/notebooks/07_methods_comparison.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/standard_errors/notebooks/07_methods_comparison.ipynb) |

## Learning Paths

### :material-lightning-bolt: Basic (3 hours)

Essential inference for any panel analysis:

**Notebooks**: 1, 2

Covers robust SE basics and clustering. These two methods handle the vast majority of applied work.

### :material-flask: Intermediate (5 hours)

Add HAC and sandwich methods:

**Notebooks**: 1, 2, 3, 4, 5

Includes Newey-West, Conley spatial HAC, and sandwich inference for MLE models.

### :material-trophy: Advanced (8 hours)

Master every inference method:

**Notebooks**: 1--7

Adds Driscoll-Kraay, PCSE, bootstrap, and a systematic comparison of all methods.

## Key Concepts Covered

- **HC0--HC3**: Heteroskedasticity-consistent SE (White, MacKinnon-White)
- **Cluster-robust**: Accounting for within-cluster correlation
- **Two-way clustering**: Simultaneous entity and time clustering
- **HAC (Newey-West)**: Heteroskedasticity and autocorrelation consistent
- **Spatial HAC (Conley)**: SE for spatially correlated data
- **Driscoll-Kraay**: Cross-sectionally dependent panels (large T)
- **PCSE**: Panel Corrected SE (Beck & Katz)
- **MLE sandwich**: Robust SE for maximum likelihood estimators
- **Bootstrap**: Nonparametric, wild, and block bootstrap

## Quick Example

```python
import panelbox as pb

# FE with cluster-robust SE
fe = pb.FixedEffects(
    data=data,
    formula="y ~ x1 + x2",
    entity_col="id",
    time_col="year",
    cov_type="clustered"
).fit()

print(fe.summary())
```

## Related Documentation

- [Standard Errors Series](standard-errors.md) -- Detailed tutorial with decision tree
- [Inference](../inference/index.md) -- Standard error theory and methods
- [User Guide](../user-guide/index.md) -- API reference
