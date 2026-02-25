---
title: Quick Start
description: Run your first panel data model with PanelBox in 5 minutes
---

# Quick Start

This guide takes you from zero to a working panel data model with diagnostics and an HTML report -- all in under 5 minutes.

## What You'll Learn

- Estimate a Fixed Effects model with 3 lines of code
- Run the Hausman test to validate your model choice
- Generate an interactive HTML report using the Experiment pattern

## Step 1: Load Data

PanelBox ships with classic econometric datasets. We'll use the **Grunfeld** dataset: investment data for 10 US firms over 20 years (1935--1954).

```python
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
print(f"Shape: {data.shape}")
print(f"Firms: {data['firm'].nunique()}, Years: {data['year'].nunique()}")
print(data.head())
```

```text
Shape: (200, 5)
Firms: 10, Years: 20
   firm  year   invest     value   capital
0     1  1935   317.60   3078.50     2.80
1     1  1936   391.80   4661.70    52.60
2     1  1937   410.60   5387.10   156.90
3     1  1938   257.70   2792.20   209.20
4     1  1939   330.80   4313.20   203.40
```

| Variable | Description |
|----------|-------------|
| `firm` | Firm identifier (1--10) |
| `year` | Year (1935--1954) |
| `invest` | Gross investment |
| `value` | Market value of the firm |
| `capital` | Stock of plant and equipment |

## Step 2: Estimate a Model

Fit a **Fixed Effects** model with clustered standard errors in three lines:

```python
from panelbox import FixedEffects

model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit(cov_type="clustered")
print(results.summary())
```

```text
================================================================================
                      Fixed Effects Estimation Results
================================================================================
Dependent Variable:              invest        No. Observations:             200
Model:                    Fixed Effects        No. Entities:                  10
Method:                 Within (LSDV)          No. Time Periods:              20
Cov. Type:                  clustered          R-squared (within):         0.767
================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
value             0.1101      0.012      9.286      0.000       0.087       0.134
capital           0.3101      0.053      5.837      0.000       0.205       0.415
================================================================================
```

!!! info "Formula syntax"
    PanelBox uses R-style formulas: `"y ~ x1 + x2"`. The four positional arguments are always: **formula**, **data**, **entity column**, **time column**.

## Step 3: Interpret Results

The key outputs to check:

| Output | Value | Meaning |
|--------|-------|---------|
| **R-squared (within)** | 0.767 | Model explains 76.7% of within-firm variation |
| **value** | 0.1101 (p < 0.001) | A unit increase in firm value raises investment by 0.11 |
| **capital** | 0.3101 (p < 0.001) | A unit increase in capital stock raises investment by 0.31 |
| **Cov. Type** | clustered | Standard errors account for within-firm correlation |

Access results programmatically:

```python
# Coefficients as a pandas Series
print(results.params)

# Standard errors
print(results.std_errors)

# R-squared
print(f"R-squared (within): {results.rsquared_within:.4f}")
```

## Step 4: Run Diagnostics

Use the **Hausman test** to verify that Fixed Effects is preferred over Random Effects:

```python
from panelbox import RandomEffects
from panelbox.validation import HausmanTest

re_model = RandomEffects("invest ~ value + capital", data, "firm", "year")
re_results = re_model.fit()

hausman = HausmanTest(results, re_results)
print(hausman)
```

```text
Hausman Test
H0: Random Effects is consistent and efficient
statistic: 14.82, p-value: 0.0006
Decision: Reject H0 → Use Fixed Effects
```

!!! tip "Interpreting the Hausman test"
    **p < 0.05**: Reject the null -- use Fixed Effects (entity effects are correlated with regressors).
    **p >= 0.05**: Fail to reject -- Random Effects is more efficient.

## Step 5: Generate a Report

The **PanelExperiment** pattern automates model comparison, validation, and reporting:

```python
from panelbox.experiment import PanelExperiment

# Create experiment
exp = PanelExperiment(data, "invest ~ value + capital", "firm", "year")

# Fit multiple models at once
exp.fit_all_models(["pooled", "fe", "re"])

# Run validation on the preferred model
validation = exp.validate_model("fe")

# Compare all models side by side
comparison = exp.compare_models(["pooled", "fe", "re"])

# Generate an interactive HTML report
exp.save_master_report("grunfeld_analysis.html")
```

This produces a self-contained HTML file with:

- Summary tables for each model
- Side-by-side coefficient comparison
- Diagnostic test results
- Interactive Plotly charts

## Next Steps

You now have a working panel data analysis pipeline. Here's where to go next:

<div class="grid cards" markdown>

- :material-book-open-variant: **[Core Concepts](core-concepts.md)**

    Learn about panel data structure, formulas, and the PanelBox workflow

- :material-map-marker-path: **[Choosing a Model](choosing-model.md)**

    Decision guide covering all 13 model families

- :material-chart-line: **[Static Models](../user-guide/static-models/index.md)**

    Deep dive into Pooled OLS, Fixed Effects, and Random Effects

- :material-cog-sync: **[Dynamic GMM](../user-guide/gmm/index.md)**

    Handle dynamics and endogeneity with Arellano-Bond and Blundell-Bond

</div>
