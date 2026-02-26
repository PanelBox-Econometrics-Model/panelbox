# Validation Tutorial Series

A hands-on tutorial series covering panel-data validation, robustness analysis,
and model comparison using the **panelbox** library.

---

## Learning Objectives

After completing these tutorials you will be able to:

1. Diagnose heteroskedasticity, serial correlation, and cross-sectional
   dependence in panel data.
2. Apply bootstrap resampling and time-series cross-validation to assess model
   stability.
3. Detect and handle outliers and influential observations.
4. Design systematic experiments to compare model specifications.

---

## Notebook Overview

| Notebook | Topic | Key panelbox APIs |
|----------|-------|-------------------|
| `01_assumption_tests.ipynb` | Diagnostic tests for panel assumptions | `ModifiedWaldTest`, `WooldridgeARTest`, `PesaranCDTest`, `HausmanTest` |
| `02_bootstrap_cross_validation.ipynb` | Bootstrap inference and CV | `PanelBootstrap`, `TimeSeriesCV`, `PanelJackknife` |
| `03_outliers_influence.ipynb` | Outlier detection and influence diagnostics | `OutlierDetector`, `InfluenceDiagnostics`, `SensitivityAnalysis` |
| `04_experiments_model_comparison.ipynb` | Systematic model comparison | `PanelExperiment`, `ComparisonResult`, `ValidationResult` |

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.9 |
| panelbox | current |
| numpy | ≥ 1.23 |
| pandas | ≥ 1.5 |
| scipy | ≥ 1.9 |
| matplotlib | ≥ 3.5 |

---

## Directory Structure

```
validation/
├── data/               Synthetic CSV datasets (generated offline)
├── notebooks/          Tutorial notebooks (01_–04_)
├── outputs/            Generated reports, plots, JSON exports
├── solutions/          Complete worked solutions for each notebook
└── utils/              Shared Python utilities
    ├── data_generators.py
    └── plot_helpers.py
```

---

## Quick Start

See `GETTING_STARTED.md` for step-by-step instructions.
