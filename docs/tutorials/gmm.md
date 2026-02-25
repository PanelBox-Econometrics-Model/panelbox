---
title: "GMM Tutorials"
description: "Interactive tutorials for dynamic panel models with Arellano-Bond and Blundell-Bond GMM in PanelBox"
---

# GMM Tutorials

!!! info "Learning Path"
    **Prerequisites**: [Static Models](static-models.md) tutorials, basic understanding of instrumental variables
    **Time**: 4--8 hours
    **Level**: Intermediate -- Advanced

## Overview

Dynamic panel models include lagged dependent variables as regressors, which creates endogeneity that standard estimators cannot handle. The Generalized Method of Moments (GMM) approach uses internal instruments -- lagged levels or lagged differences of the dependent variable -- to achieve consistent estimation.

These tutorials cover the two canonical dynamic panel estimators: Arellano-Bond (Difference GMM) and Blundell-Bond (System GMM), along with advanced variants including CUE-GMM and bias-corrected estimation. You will learn to diagnose instrument validity, manage instrument proliferation, and apply these methods to real-world datasets.

The existing [GMM Introduction Tutorial](gmm.md) provides additional theoretical background and step-by-step derivations.

!!! warning "Instrument Proliferation"
    A key practical challenge with GMM is having too many instruments relative to cross-sectional units. The `collapse` option is critical for maintaining test validity. See notebook 03 for details.

## Notebooks

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Difference GMM Fundamentals](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/notebooks/01_difference_gmm_fundamentals.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/notebooks/01_difference_gmm_fundamentals.ipynb) |
| 2 | [System GMM & Efficiency](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/notebooks/02_system_gmm_efficiency.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/notebooks/02_system_gmm_efficiency.ipynb) |
| 3 | [Instrument Specification & Collapse](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/notebooks/03_instrument_specification.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/notebooks/03_instrument_specification.ipynb) |
| 4 | [GMM Diagnostics (Hansen J, AR Tests)](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/notebooks/04_gmm_tests_diagnostics.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/notebooks/04_gmm_tests_diagnostics.ipynb) |
| 5 | [CUE-GMM & Bias Correction](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/notebooks/05_cue_bias_correction.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/notebooks/05_cue_bias_correction.ipynb) |
| 6 | [Complete Applied Case Study](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/notebooks/06_complete_applied_case.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/notebooks/06_complete_applied_case.ipynb) |

**Bonus**: [Validation: PanelBox vs pydynpd](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/notebooks/lgd_validacao_panelbox_vs_pydynpd.ipynb) -- Cross-validation of PanelBox GMM results against pydynpd.

## Learning Paths

### :material-lightning-bolt: Essential (4 hours)

Core dynamic panel methods for applied research:

**Notebooks**: 1, 2, 3, 4

Covers both Difference and System GMM, instrument specification, and all critical diagnostic tests (Hansen J, AR(1)/AR(2)).

### :material-trophy: Complete (8 hours)

Master every GMM variant including advanced estimators:

**Notebooks**: 1--6 + solutions

Adds CUE-GMM, bias correction, and a comprehensive applied case study.

## Key Concepts Covered

- **Dynamic panel bias**: Why FE fails with lagged dependent variables (Nickell bias)
- **Difference GMM**: Arellano-Bond estimator using lagged levels as instruments
- **System GMM**: Blundell-Bond extension using both levels and differences
- **One-step vs two-step**: Efficiency vs reliability trade-offs
- **Instrument proliferation**: Why too many instruments weaken tests
- **Collapse**: Reducing the instrument count while preserving validity
- **Hansen J-test**: Overidentification test for instrument validity
- **AR(1)/AR(2) tests**: Serial correlation diagnostics
- **CUE-GMM**: Continuously-updated estimator for robustness
- **Windmeijer correction**: Finite-sample correction for two-step SE

## Quick Example

```python
from panelbox.gmm import DifferenceGMM, SystemGMM

# Arellano-Bond Difference GMM
ab = DifferenceGMM(
    data=data,
    dep_var="y",
    predetermined=["x1"],
    exogenous=["x2"],
    entity_col="id",
    time_col="year",
    lags=1,
    collapse=True
).fit()

print(ab.summary())
print(f"Hansen J p-value: {ab.hansen_test.pvalue:.4f}")
print(f"AR(2) p-value: {ab.ar_tests[2].pvalue:.4f}")
```

## Solutions

| Tutorial | Solution |
|----------|----------|
| 01. Difference GMM Fundamentals | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/solutions/01_solutions.ipynb) |
| 02. System GMM & Efficiency | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/solutions/02_solutions.ipynb) |
| 03. Instrument Specification | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/solutions/03_solutions.ipynb) |
| 04. GMM Diagnostics | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/solutions/04_solutions.ipynb) |
| 05. CUE & Bias Correction | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/solutions/05_solutions.ipynb) |
| 06. Applied Case Study | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/gmm/solutions/06_solutions.ipynb) |

## Related Documentation

- [GMM Introduction Tutorial](gmm.md) -- Detailed theory and walkthrough
- [Theory: Advanced GMM](../theory/gmm-theory.md) -- Mathematical foundations
- [User Guide](../user-guide/index.md) -- API reference
- [Validation & Diagnostics](validation.md) -- General diagnostic testing
