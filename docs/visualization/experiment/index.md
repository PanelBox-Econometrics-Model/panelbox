---
title: Experiment Pattern
description: Guide to the PanelExperiment workflow in PanelBox - fit, validate, compare, and report in a unified pipeline.
---

# Experiment Pattern

The `PanelExperiment` class provides a high-level workflow for systematic econometric analysis. Instead of manually fitting models, running diagnostics, and building comparison tables, the Experiment pattern automates the entire pipeline: **fit all models**, **validate assumptions**, **compare results**, and **generate a master report** -- all in a few lines of code.

## Workflow Overview

```text
PanelExperiment
    |
    +-- fit_all_models()      Estimate multiple model specifications
    |
    +-- validate_model()      Run diagnostic tests on chosen model
    |
    +-- compare_models()      Side-by-side coefficient and fit comparison
    |
    +-- save_master_report()  Generate comprehensive HTML report
```

## Quick Example

```python
from panelbox.experiment import PanelExperiment
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Create experiment
exp = PanelExperiment(data, "invest ~ value + capital", "firm", "year")

# Step 1: Fit all models
exp.fit_all_models(["pooled", "fe", "re"])

# Step 2: Validate the chosen model
validation = exp.validate_model("fe")
print(validation.summary())

# Step 3: Compare models
comparison = exp.compare_models(["pooled", "fe", "re"])
print(comparison.summary())

# Step 4: Generate master report
exp.save_master_report("grunfeld_analysis.html")
```

## Available Model Types

The `fit_all_models()` method accepts a list of model identifiers:

| Identifier | Model | Class |
|-----------|-------|-------|
| `"pooled"` | Pooled OLS | `PooledOLS` |
| `"fe"` | Fixed Effects | `FixedEffects` |
| `"re"` | Random Effects | `RandomEffects` |
| `"between"` | Between Estimator | `BetweenEstimator` |
| `"fd"` | First Difference | `FirstDifferenceEstimator` |

## Step-by-Step Guide

### Step 1: Fit Models

```python
exp = PanelExperiment(data, "invest ~ value + capital", "firm", "year")

# Fit specific models
exp.fit_all_models(["pooled", "fe", "re"])

# Access individual results
fe_results = exp.results["fe"]
print(fe_results.summary())
```

### Step 2: Validate Assumptions

The `validate_model()` method runs all applicable diagnostic tests:

```python
validation = exp.validate_model("fe")

# Check individual test results
for test_name, result in validation.tests.items():
    print(f"{test_name}: p={result.pvalue:.4f} -- {result.conclusion}")
```

Validation includes:

- Serial correlation tests (Wooldridge AR)
- Heteroskedasticity tests (Modified Wald, Breusch-Pagan)
- Cross-sectional dependence tests (Pesaran CD)
- Specification tests (RESET)

### Step 3: Compare Models

```python
comparison = exp.compare_models(["pooled", "fe", "re"])
print(comparison.summary())

# Includes:
# - Side-by-side coefficient table
# - Standard errors for each model
# - R-squared, AIC, BIC
# - Number of observations
```

### Step 4: Generate Report

```python
# HTML report with interactive charts
exp.save_master_report("analysis.html")
```

The master report includes:

- All model summaries
- Coefficient comparison table
- Diagnostic test results
- Residual diagnostic charts
- Entity and time effects plots
- Model fit statistics

## Spatial Experiment

For spatial models, use `SpatialPanelExperiment`:

```python
from panelbox.experiment import SpatialPanelExperiment

exp = SpatialPanelExperiment(
    data, "y ~ x1 + x2", "region", "year",
    W=weight_matrix
)
exp.fit_all_models(["sar", "sem", "sdm"])
comparison = exp.compare_models(["sar", "sem", "sdm"])
exp.save_master_report("spatial_analysis.html")
```

## Customizing the Experiment

### Standard Error Options

```python
exp = PanelExperiment(data, formula, entity, time)
exp.fit_all_models(
    ["pooled", "fe", "re"],
    cov_type="clustered"  # Applied to all models
)
```

### Adding Custom Models

```python
from panelbox import FixedEffects

# Fit a custom specification
custom = FixedEffects("invest ~ value + capital + L.invest", data, "firm", "year")
custom_results = custom.fit(cov_type="clustered")

# Add to experiment
exp.add_result("custom_fe", custom_results)
comparison = exp.compare_models(["fe", "custom_fe"])
```

## Detailed Guides

- [PanelExperiment API](fitting.md) -- Full API reference *(detailed guide coming soon)*
- [Spatial Experiment](spatial-extension.md) -- Spatial extensions *(detailed guide coming soon)*
- [Custom Workflows](fitting.md) -- Advanced experiment patterns *(detailed guide coming soon)*

## Tutorials

See [Visualization Tutorial](../../tutorials/visualization.md) for Experiment pattern examples with Google Colab.

## API Reference

See [Experiment API](../../api/experiment.md) for complete technical reference.
