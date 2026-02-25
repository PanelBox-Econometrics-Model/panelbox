---
title: "Visualization & Reports Tutorials"
description: "Interactive tutorials for creating charts, visual diagnostics, and automated HTML reports with PanelBox"
---

# Visualization & Reports Tutorials

!!! info "Learning Path"
    **Prerequisites**: At least one model family completed
    **Time**: 2--5 hours
    **Level**: Beginner -- Intermediate

## Overview

PanelBox includes a comprehensive visualization system with 28+ chart types and an automated HTML report generator. These tutorials show you how to create publication-quality diagnostic plots, compare models visually, customize themes, and generate self-contained HTML reports that combine all your analysis in one interactive document.

The visualization system supports both Plotly (interactive) and Matplotlib (static) backends, with three built-in themes (professional, academic, presentation) and the ability to create custom themes. The report system generates standalone HTML files that can be shared with collaborators -- no Python installation needed to view them.

The existing [HTML Reports Tutorial](visualization.md) provides additional depth on the report system, including master reports and JSON export.

## Visualization Notebooks

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Visualization Introduction](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/visualization/notebooks/01_visualization_introduction.ipynb) | Beginner | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/visualization/notebooks/01_visualization_introduction.ipynb) |
| 2 | [Visual Diagnostics](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/visualization/notebooks/02_visual_diagnostics.ipynb) | Beginner | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/visualization/notebooks/02_visual_diagnostics.ipynb) |
| 3 | [Advanced Visualizations](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/visualization/notebooks/03_advanced_visualizations.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/visualization/notebooks/03_advanced_visualizations.ipynb) |
| 4 | [Automated Reports](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/visualization/notebooks/04_automated_reports.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/visualization/notebooks/04_automated_reports.ipynb) |

## Production Notebooks

For workflows that take models from development to deployment:

| # | Tutorial | Level | Time | Colab |
|---|---------|-------|------|-------|
| 1 | [Predict Fundamentals](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/01_predict_fundamentals.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/01_predict_fundamentals.ipynb) |
| 2 | [Save & Load Models](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/02_save_load_models.ipynb) | Intermediate | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/02_save_load_models.ipynb) |
| 3 | [Production Pipeline](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/03_production_pipeline.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/03_production_pipeline.ipynb) |
| 4 | [Model Validation](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/04_model_validation.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/04_model_validation.ipynb) |
| 5 | [Model Versioning](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/05_model_versioning.ipynb) | Advanced | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/05_model_versioning.ipynb) |
| 6 | [Bank LGD Case Study](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/06_case_study_bank_lgd.ipynb) | Advanced | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/06_case_study_bank_lgd.ipynb) |

## Learning Paths

### :material-lightning-bolt: Charts (2 hours)

Essential visualization skills:

**Notebooks**: Visualization 1, 2

Covers chart creation, residual diagnostics, and theme selection.

### :material-flask: Reports (3 hours)

Add automated reporting:

**Notebooks**: Visualization 1, 2, 4 + Production 1

Includes automated HTML report generation and prediction fundamentals.

### :material-trophy: Complete (5 hours)

Full visualization and production workflow:

**Notebooks**: Visualization 1--4 + Production 1--2

Adds advanced visualizations, custom themes, and model persistence.

## Key Concepts Covered

- **Chart Factory**: Create charts by name using `ChartFactory.create()`
- **Residual diagnostics**: QQ plot, residual vs fitted, scale-location, leverage
- **Model comparison**: Coefficient plots, forest plots, IC comparison
- **Panel charts**: Entity effects, time effects, between-within decomposition
- **Themes**: Professional, academic, presentation, custom themes
- **Export**: PNG, SVG, PDF, HTML formats
- **HTML reports**: Self-contained interactive reports
- **Master reports**: Combined validation + comparison + residual reports
- **PanelExperiment**: Automated multi-model analysis workflow
- **Model persistence**: Save and load fitted models

## Quick Example

```python
from panelbox.visualization import create_residual_diagnostics, create_comparison_charts
from panelbox import PanelExperiment

# Residual diagnostic charts
charts = create_residual_diagnostics(results, theme='professional')
charts['qq_plot'].to_html()

# Model comparison
exp = PanelExperiment(data, formula="y ~ x1 + x2", entity_col="id", time_col="year")
exp.fit_all_models(["pooled_ols", "fixed_effects", "random_effects"])

# Master report
exp.save_master_report("master_report.html", theme='professional')
```

## Solutions

### Visualization Solutions

| Tutorial | Solution |
|----------|----------|
| 01. Introduction | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/visualization/solutions/01_visualization_introduction_solution.ipynb) |
| 02. Visual Diagnostics | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/visualization/solutions/02_visual_diagnostics_solution.ipynb) |
| 03. Advanced Visualizations | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/visualization/solutions/03_advanced_visualizations_solution.ipynb) |
| 04. Automated Reports | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/visualization/solutions/04_automated_reports_solution.ipynb) |

### Production Solutions

| Tutorial | Solution |
|----------|----------|
| 01. Predict Fundamentals | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/solutions/01_predict_fundamentals_solutions.ipynb) |
| 02. Save & Load | [:material-notebook: Solution](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/solutions/02_save_load_solutions.ipynb) |

## Related Documentation

- [HTML Reports Tutorial](visualization.md) -- Detailed HTML report walkthrough
- [Visualization](../visualization/charts/index.md) -- Chart gallery and API reference
- [User Guide](../user-guide/index.md) -- Experiment and report API
