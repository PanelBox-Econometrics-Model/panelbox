---
title: Production Tutorials
description: Interactive production deployment tutorials for PanelBox with Google Colab
---

# Production Tutorials

Learn how to deploy PanelBox models in production environments. These tutorials cover prediction workflows, model serialization, production pipelines, validation strategies, model versioning, and a complete case study on bank Loss Given Default (LGD) modeling.

!!! tip "Prerequisites"
    Understanding of PanelBox model fitting and results interpretation.
    Familiarity with Python packaging and deployment concepts is helpful for advanced tutorials.

## Notebooks

| # | Tutorial | Level | Time | Colab |
|---|----------|-------|------|-------|
| 1 | Prediction Fundamentals | Intermediate | ~20 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/01_predict_fundamentals.ipynb) |
| 2 | Save & Load Models | Intermediate | ~20 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/02_save_load_models.ipynb) |
| 3 | Production Pipelines | Advanced | ~35 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/03_production_pipeline.ipynb) |
| 4 | Model Validation | Advanced | ~30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/04_model_validation.ipynb) |
| 5 | Model Versioning | Advanced | ~30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/05_model_versioning.ipynb) |
| 6 | Case Study: Bank LGD | Advanced | ~45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/notebooks/06_case_study_bank_lgd.ipynb) |

## Solutions

Solutions with complete code and explanations are available for all tutorials:

| Tutorial | Solution |
|----------|----------|
| Prediction Fundamentals | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/solutions/01_predict_fundamentals_solutions.ipynb) |
| Save & Load Models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/solutions/02_save_load_solutions.ipynb) |
| Production Pipelines | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/solutions/03_production_pipeline_solutions.ipynb) |
| Model Validation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/solutions/04_model_validation_solutions.ipynb) |
| Model Versioning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/solutions/05_model_versioning_solutions.ipynb) |
| Case Study: Bank LGD | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/production/solutions/06_case_study_bank_lgd_solutions.ipynb) |

## What You Will Learn

**Getting Started (Tutorials 1--2)**: Generate predictions from fitted models (in-sample and out-of-sample), and serialize/deserialize models for reuse without re-estimation.

**Advanced Deployment (Tutorials 3--5)**: Build end-to-end production pipelines with data validation, implement model validation frameworks for monitoring drift, and manage model versions across iterations.

**Case Study (Tutorial 6)**: Apply everything in a real-world bank LGD (Loss Given Default) modeling scenario, from data preparation through model deployment and monitoring.

## Related Documentation

- [Getting Started](../getting-started/quickstart.md)
