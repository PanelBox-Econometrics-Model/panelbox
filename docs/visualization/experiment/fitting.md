---
title: "Experiment Workflow"
description: "Creating experiments, fitting models, and managing results with PanelExperiment"
---

# Experiment Workflow

## Overview

`PanelExperiment` is the central orchestrator for panel data analysis in PanelBox. It manages the lifecycle of an econometric study: from data ingestion and model fitting to validation, comparison, and report generation.

This page covers creating an experiment, fitting models, and managing stored results.

## Creating an Experiment

### With Entity and Time Columns

If your DataFrame has entity and time identifiers as regular columns:

```python
import pandas as pd
import panelbox as pb

# Load data
data = pb.load_grunfeld()

# Create experiment
exp = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

print(exp)
```

Output:

```text
PanelExperiment(
  formula='invest ~ value + capital',
  n_obs=200,
  n_models=0,
  models=[none]
)
```

### With MultiIndex

If your DataFrame already has a `(entity, time)` MultiIndex:

```python
# Data with MultiIndex
data_mi = data.set_index(["firm", "year"])

exp = pb.PanelExperiment(
    data=data_mi,
    formula="invest ~ value + capital"
)
```

!!! warning "Data Validation"
    `PanelExperiment` validates the input on creation:

    - Data must be a non-empty `pd.DataFrame`
    - If `entity_col` and `time_col` are `None`, the DataFrame must have a `MultiIndex`
    - If provided, `entity_col` and `time_col` must exist as columns in the DataFrame

### Formula Syntax

PanelBox uses Patsy-style formulas:

```python
# Simple regression
exp = pb.PanelExperiment(data, "y ~ x1 + x2", entity_col="id", time_col="t")

# Multiple regressors
exp = pb.PanelExperiment(data, "gdp ~ investment + labor + trade", entity_col="country", time_col="year")
```

## Fitting Models

### Single Model

Use `fit_model()` to fit one model at a time:

```python
# Fit with explicit name
results = exp.fit_model('fixed_effects', name='fe')

# Fit with auto-generated name (e.g., 'pooled_ols_1')
results = exp.fit_model('pooled_ols')
```

The `name` parameter identifies the model in the experiment. If omitted, PanelBox auto-generates a name like `fixed_effects_1`, `pooled_ols_1`, etc.

### Model-Specific Options

Pass keyword arguments to the underlying model's `fit()` method:

```python
# Fixed Effects with clustered standard errors
exp.fit_model('fixed_effects', name='fe_clustered', cov_type='clustered')

# Random Effects with specific options
exp.fit_model('random_effects', name='re_model')
```

### Multiple Models at Once

Use `fit_all_models()` to fit several models in one call:

```python
# Fit default trio (pooled_ols, fixed_effects, random_effects)
results = exp.fit_all_models()
print(exp.list_models())
# ['pooled_ols_1', 'fixed_effects_1', 'random_effects_1']

# Fit specific models with custom names
results = exp.fit_all_models(
    model_types=['pooled_ols', 'fixed_effects', 'random_effects'],
    names=['ols', 'fe', 're']
)
print(exp.list_models())
# ['ols', 'fe', 're']
```

!!! tip "Default Models"
    When `model_types` is `None`, `fit_all_models()` fits the three core linear models: `pooled_ols`, `fixed_effects`, and `random_effects`.

### Supported Model Types

| Category | Alias(es) | Resolved Type | Description |
|----------|-----------|---------------|-------------|
| **Linear** | `pooled_ols`, `pooled` | `pooled_ols` | Pooled OLS regression |
| | `fixed_effects`, `fe` | `fixed_effects` | Entity fixed effects |
| | `random_effects`, `re` | `random_effects` | Random effects (GLS) |
| **Discrete** | `pooled_logit` | `pooled_logit` | Pooled logistic regression |
| | `pooled_probit` | `pooled_probit` | Pooled probit regression |
| | `fe_logit`, `fixed_effects_logit` | `fixed_effects_logit` | Conditional logit (FE) |
| | `re_probit`, `random_effects_probit` | `random_effects_probit` | Random effects probit |
| **Count** | `pooled_poisson`, `poisson` | `pooled_poisson` | Pooled Poisson regression |
| | `fe_poisson`, `poisson_fe` | `poisson_fixed_effects` | Poisson with fixed effects |
| | `re_poisson` | `random_effects_poisson` | Random effects Poisson |
| | `negbin`, `negative_binomial` | `negative_binomial` | Negative binomial regression |
| **Censored** | `tobit`, `re_tobit` | `random_effects_tobit` | Random effects Tobit |
| **Ordered** | `ologit`, `ordered_logit` | `ordered_logit` | Ordered logistic regression |
| | `oprobit`, `ordered_probit` | `ordered_probit` | Ordered probit regression |

## Managing Models

### Listing Models

```python
exp.list_models()
# ['ols', 'fe', 're']
```

### Retrieving a Model

```python
fe_results = exp.get_model('fe')
print(fe_results.params)
```

If the model name does not exist, a `KeyError` is raised with a list of available models.

### Model Metadata

Each fitted model stores metadata about how it was created:

```python
meta = exp.get_model_metadata('fe')
print(meta)
# {
#     'model_type': 'fixed_effects',
#     'fitted_at': datetime(2026, 2, 25, 14, 30, 0),
#     'formula': 'invest ~ value + capital',
#     'kwargs': {}
# }
```

## Complete Example

```python
import panelbox as pb

# 1. Load data and create experiment
data = pb.load_grunfeld()
exp = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

# 2. Fit five models
exp.fit_model('pooled_ols', name='ols')
exp.fit_model('fixed_effects', name='fe')
exp.fit_model('random_effects', name='re')
exp.fit_model('pooled_poisson', name='poisson')
exp.fit_model('negbin', name='nb')

# 3. List all fitted models
print(f"Models: {exp.list_models()}")

# 4. Compare the linear models
comp = exp.compare_models(model_names=['ols', 'fe', 're'])
best = comp.best_model('aic')
print(f"Best model by AIC: {best}")

# 5. Validate the best model
val = exp.validate_model(best, tests='default')
print(f"Pass rate: {val.pass_rate:.0%}")

# 6. Analyze residuals
resid = exp.analyze_residuals(best)
print(resid.summary())

# 7. Generate reports
val.save_html("validation.html", test_type="validation", theme="professional")
comp.save_html("comparison.html", test_type="comparison", theme="professional")
resid.save_html("residuals.html", test_type="residuals", theme="professional")
exp.save_master_report("master.html", title="Grunfeld Analysis")
```

## Comparison with Other Software

| Task | PanelBox | Stata | R |
|------|----------|-------|---|
| Store multiple models | `exp.fit_model(name=...)` | `eststo: reg ...` | `models <- list()` |
| List stored models | `exp.list_models()` | `estimates dir` | `names(models)` |
| Retrieve a model | `exp.get_model('fe')` | `estimates restore fe` | `models[["fe"]]` |
| Fit all at once | `exp.fit_all_models()` | Manual loop | Manual loop |

## See Also

- [Experiment Overview](index.md) -- Pattern overview and quick start
- [Validation Reports](validation.md) -- Running diagnostic tests
- [Comparison Reports](comparison.md) -- Side-by-side model comparison
- [Residual Analysis](residuals.md) -- Residual diagnostics
- [Master Reports](master-reports.md) -- Combined report generation
- [Tutorials: HTML Reports](../../tutorials/visualization.md) -- Step-by-step tutorial
