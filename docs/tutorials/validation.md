---
title: "Validation & Diagnostics Tutorials"
description: "Interactive tutorials for model validation, diagnostic testing, unit roots, and cointegration with PanelBox"
---

# Validation & Diagnostics Tutorials

!!! info "Learning Path"
    **Prerequisites**: At least one model family completed
    **Time**: 3--6 hours
    **Level**: Intermediate -- Advanced

## Overview

Estimation is only half the story. Every panel model rests on assumptions -- exogeneity, no serial correlation, homoskedasticity, stationarity -- and failing to test these assumptions can lead to unreliable results. Diagnostics also help you choose between models (Hausman test, specification tests) and assess robustness (bootstrap, cross-validation, influence analysis).

These tutorials cover two complementary areas. The **validation** notebooks focus on assumption testing, bootstrap inference, outlier detection, and the PanelExperiment workflow for systematic model comparison. The **diagnostics** notebooks cover time-series diagnostics relevant for panels: unit root tests, cointegration tests, serial correlation, heteroskedasticity, and spatial diagnostics.

Additional self-contained notebook tutorials are available for specific topics:

- [Panel Unit Root](panel_unit_root.ipynb) -- LLC, IPS, Fisher tests
- [Panel Cointegration](panel_cointegration.ipynb) -- Pedroni, Kao, Westerlund tests
- [J-Test Specification](jtest_tutorial.ipynb) -- J-test for non-nested models

## Validation Notebooks

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Assumption Tests](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/validation/notebooks/01_assumption_tests.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/validation/notebooks/01_assumption_tests.ipynb) |
| 2 | [Bootstrap & Cross-Validation](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/validation/notebooks/02_bootstrap_cross_validation.ipynb) | Intermediate | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/validation/notebooks/02_bootstrap_cross_validation.ipynb) |
| 3 | [Outliers & Influence](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/validation/notebooks/03_outliers_influence.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/validation/notebooks/03_outliers_influence.ipynb) |
| 4 | [Experiments & Model Comparison](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/validation/notebooks/04_experiments_model_comparison.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/validation/notebooks/04_experiments_model_comparison.ipynb) |

## Diagnostics Notebooks

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Unit Root Tests](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/diagnostics/notebooks/01_unit_root_tests.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/diagnostics/notebooks/01_unit_root_tests.ipynb) |
| 2 | [Cointegration Tests](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/diagnostics/notebooks/02_cointegration_tests.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/diagnostics/notebooks/02_cointegration_tests.ipynb) |
| 3 | [Specification Tests](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/diagnostics/notebooks/03_specification_tests.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/diagnostics/notebooks/03_specification_tests.ipynb) |
| 4 | [Spatial Diagnostics](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/diagnostics/notebooks/04_spatial_tests.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/diagnostics/notebooks/04_spatial_tests.ipynb) |

## Learning Paths

### :material-lightning-bolt: Essential (3 hours)

Core diagnostics for any panel analysis:

**Notebooks**: Validation 1, 3 + Diagnostics 1, 3

Covers assumption testing, outlier detection, unit root tests, and specification tests.

### :material-trophy: Complete (6 hours)

Full validation and diagnostics coverage:

**Notebooks**: All 8 notebooks

Adds bootstrap/cross-validation, PanelExperiment workflows, cointegration tests, and spatial diagnostics.

## Key Concepts Covered

- **Assumption testing**: Heteroskedasticity (Modified Wald), serial correlation (Wooldridge), cross-sectional dependence (Pesaran CD)
- **Hausman test**: Fixed vs Random Effects specification
- **Mundlak test**: RE with correlated effects
- **Bootstrap**: Nonparametric and wild bootstrap for robust inference
- **Cross-validation**: Panel-aware k-fold and leave-one-entity-out
- **Outlier detection**: DFBETAS, Cook's distance, leverage
- **PanelExperiment**: Systematic multi-model comparison with automated reports
- **Unit root tests**: LLC, IPS, Fisher (ADF/PP), Pesaran CADF
- **Cointegration**: Pedroni, Kao, Westerlund tests
- **Spatial diagnostics**: Moran's I, LM tests for spatial specification

## Quick Example

```python
from panelbox import PanelExperiment

# Systematic model comparison
exp = PanelExperiment(data, formula="y ~ x1 + x2", entity_col="id", time_col="year")
exp.fit_all_models(["pooled_ols", "fixed_effects", "random_effects"])

# Validate the FE model
val = exp.validate_model("fixed_effects", tests="all")
print(f"Pass rate: {val.pass_rate:.1%}")

# Compare all models
comp = exp.compare_models()
best, _ = comp.best_model(criterion="aic")
print(f"Best model: {best}")
```

## Solutions

### Validation Solutions

| Tutorial | Solution |
|----------|----------|
| 01. Assumption Tests | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/validation/solutions/01_assumption_tests_solution.ipynb) |
| 02. Bootstrap & CV | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/validation/solutions/02_bootstrap_cv_solution.ipynb) |
| 03. Outliers & Influence | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/validation/solutions/03_outliers_solution.ipynb) |
| 04. Experiments | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/validation/solutions/04_experiments_solution.ipynb) |

### Diagnostics Solutions

| Tutorial | Solution |
|----------|----------|
| 01. Unit Root Tests | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/diagnostics/solutions/01_unit_root_solutions.ipynb) |
| 02. Cointegration | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/diagnostics/solutions/02_cointegration_solutions.ipynb) |
| 03. Specification Tests | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/diagnostics/solutions/03_specification_solutions.ipynb) |
| 04. Spatial Diagnostics | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/diagnostics/solutions/04_spatial_solutions.ipynb) |

## Related Documentation

- [Panel Unit Root Tutorial](panel_unit_root.ipynb) -- Self-contained unit root notebook
- [Panel Cointegration Tutorial](panel_cointegration.ipynb) -- Cointegration testing notebook
- [J-Test Tutorial](jtest_tutorial.ipynb) -- Non-nested model testing
- [Theory: Specification Tests](../diagnostics/specification/index.md) -- Test theory
- [Theory: Panel Cointegration](../diagnostics/cointegration/index.md) -- Cointegration theory
- [Diagnostics](../diagnostics/index.md) -- Full diagnostics reference
- [User Guide](../user-guide/index.md) -- API reference
