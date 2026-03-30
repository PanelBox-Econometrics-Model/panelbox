---
title: "AutoExperiment Quickstart"
description: "From zero to automated model selection in 10 lines of code"
---

# AutoExperiment Quickstart

This tutorial takes you from zero to a fully validated panel data model using AutoExperiment -- PanelBox's automated model selection pipeline.

## What You'll Learn

- Run AutoExperiment with a single `run()` call
- Interpret the results summary and ranking
- Generate an HTML report
- Customize transformations and sign constraints

## Step 1: Load Data and Run

AutoExperiment needs panel data, a dependent variable, and entity/time identifiers. Everything else has sensible defaults.

```python
from panelbox.datasets import load_grunfeld
from panelbox.autoexperiment import AutoExperiment

data = load_grunfeld()

auto = AutoExperiment(
    data=data,
    depvar="invest",
    entity_col="firm",
    time_col="year",
)
results = auto.run()
```

That's it -- AutoExperiment will:

1. Generate variable transformations (lag-1, first difference by default)
2. Run forward stepwise selection by BIC for each model type
3. Estimate Pooled OLS, Fixed Effects, Random Effects, and First Difference
4. Validate each model with diagnostic tests
5. Run the Hausman test (FE vs RE)
6. Rank all models and select the best one

## Step 2: View Results

```python
print(results.summary())
```

```text
==================================================
AutoExperiment Results
==================================================
Best model: Fixed Effects
Formula:    invest ~ value + capital
BIC:        1502.3
R-squared:  0.767
Cov type:   robust
Status:     VALID (all tests passed)
--------------------------------------------------
Models tested:     4
Combinations:      24
Variables tested:  6
Transformations:   2 types
```

## Step 3: Compare Top Models

```python
print(results.compare_top(3))
```

```text
  model_type                  formula      bic      aic    rsq classification  cov_type   score
0         fe   invest ~ value + capital  1502.3  1495.1  0.767          VALID    robust  1488.3
1         re   invest ~ value + capital  1510.8  1503.6  0.761        WARNING  clustered  1548.8
2  pooled_ols  invest ~ value + capital  1580.2  1573.0  0.812          VALID  nonrobust  1566.2
```

## Step 4: Access the Best Model

```python
# Print the best model's full regression output
print(results.best_model.summary())

# Key attributes
print(f"Estimator: {results.best_estimator}")
print(f"Formula: {results.best_formula}")
print(f"Cov type: {results.best_cov_type}")
```

## Step 5: Generate an HTML Report

```python
results.report("autoexperiment_report.html")
```

This creates a comprehensive report with the ranking table, variable selection details, and diagnostic test results.

---

## Going Further: Custom Configuration

### Add Transformations

Generate lags, logs, growth rates, and more:

```python
auto = AutoExperiment(
    data=data,
    depvar="invest",
    entity_col="firm",
    time_col="year",
    transformations={
        "lag": [1, 2],     # L1_var, L2_var
        "diff": True,      # D_var
        "log": True,       # log_var
        "growth": True,    # growth_var
    },
)
results = auto.run()
```

### Add Sign Constraints

Encode economic theory to prevent spurious results:

```python
auto = AutoExperiment(
    data=data,
    depvar="invest",
    entity_col="firm",
    time_col="year",
    sign_constraints={
        "value": "+",      # Market value should increase investment
        "capital": "+",    # Capital stock should increase investment
    },
)
results = auto.run()
```

### Use AIC Instead of BIC

```python
auto = AutoExperiment(
    data=data,
    depvar="invest",
    entity_col="firm",
    time_col="year",
    criterion="aic",
)
results = auto.run()
```

### Visualize Results

```python
# BIC comparison across models
fig = results.plot_bic_comparison()

# Variable selection frequency
fig = results.plot_variable_importance()
```

---

!!! warning "Data Mining"
    AutoExperiment tests many combinations. Always validate on holdout data and use sign constraints to anchor results in economic theory. Check `results.datamining_warning` to see if the threshold was exceeded.

## Next Steps

<div class="grid cards" markdown>

- :material-api: **[API Reference](../api/autoexperiment.md)**

    Full parameter documentation for all classes

- :material-book-open-variant: **[User Guide](../user-guide/autoexperiment.md)**

    Limitations, interpreting results, and best practices

- :material-flask: **[Experiment Pattern](../visualization/experiment/index.md)**

    Manual model comparison for fine-grained control

</div>
