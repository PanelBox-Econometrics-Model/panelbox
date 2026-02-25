---
title: Core Concepts
description: Understand panel data structure, formulas, results objects, and the PanelBox workflow
---

# Core Concepts

This page covers everything you need to understand before diving into specific models.

## What is Panel Data?

Panel data (also called **longitudinal data**) combines two dimensions:

- **Cross-sectional**: Multiple entities (firms, individuals, countries)
- **Time-series**: Each entity observed over multiple time periods

```text
Entity × Time = Panel
   10 firms × 20 years = 200 observations
```

This structure lets you control for unobserved entity-specific factors (like management quality or individual ability) that cross-sectional data cannot account for.

### Panel Data vs Other Structures

| Type | Entities | Time Periods | Example |
|------|----------|-------------|---------|
| **Cross-sectional** | Many | 1 | Survey of 1,000 firms in 2024 |
| **Time-series** | 1 | Many | US GDP from 1960--2024 |
| **Panel data** | Many | Many | 500 firms over 10 years |

## Panel Data Structure

### Balanced vs Unbalanced

A **balanced** panel has every entity observed in every time period:

```text
   firm  year  invest
      1  2020    100    ← Firm 1: all 3 years
      1  2021    120
      1  2022    135
      2  2020     80    ← Firm 2: all 3 years
      2  2021     85
      2  2022     90
```

An **unbalanced** panel has missing entity-period combinations:

```text
   firm  year  invest
      1  2020    100    ← Firm 1: 2 years only
      1  2021    120
      2  2020     80    ← Firm 2: all 3 years
      2  2021     85
      2  2022     90
```

PanelBox handles both balanced and unbalanced panels automatically.

### Short vs Long Panels

| Panel Type | Structure | Typical Use | PanelBox Models |
|-----------|-----------|-------------|-----------------|
| **Short panel** | Large N, small T | Micro data (firms, households) | FE, RE, GMM |
| **Long panel** | Small N, large T | Macro data (countries) | Panel VAR, cointegration tests |

## The PanelData Container

Every PanelBox model takes the same four core arguments:

```python
model = FixedEffects(
    formula="invest ~ value + capital",  # R-style formula
    data=data,                            # pandas DataFrame
    entity_col="firm",                    # Column identifying entities
    time_col="year"                       # Column identifying time periods
)
```

Internally, PanelBox validates your data:

- Checks that `entity_col` and `time_col` exist in the DataFrame
- Detects whether the panel is balanced or unbalanced
- Sorts by entity and time
- Handles missing values according to the model's requirements

!!! note "DataFrame format"
    PanelBox expects data in **long format** (one row per entity-time observation). If your data is in wide format (one row per entity, columns for each time period), reshape it first with `pandas.melt()`.

## Formula Syntax

PanelBox uses R-style formulas via the **patsy** library:

### Basic Formulas

```python
# Simple regression
"invest ~ value + capital"

# With interaction
"invest ~ value + capital + value:capital"

# With polynomial term
"invest ~ value + capital + I(value**2)"

# Log transformation
"invest ~ np.log(value) + capital"
```

### Formula Reference

| Syntax | Meaning | Example |
|--------|---------|---------|
| `y ~ x1 + x2` | y regressed on x1 and x2 | `"invest ~ value + capital"` |
| `x1:x2` | Interaction term | `"y ~ x1 + x2 + x1:x2"` |
| `x1*x2` | Main effects + interaction | `"y ~ x1*x2"` (same as above) |
| `I(expr)` | Evaluate Python expression | `"y ~ x1 + I(x1**2)"` |
| `np.log(x)` | NumPy function | `"y ~ np.log(x1) + x2"` |
| `C(x)` | Treat as categorical | `"y ~ C(region) + x1"` |

!!! warning "No intercept in Fixed Effects"
    Fixed Effects models absorb the intercept into entity-specific effects. PanelBox handles this automatically -- you do not need to remove the intercept from the formula.

### Lag Notation (GMM Models)

GMM models use a special lag notation:

```python
from panelbox.gmm import SystemGMM

model = SystemGMM(
    "n ~ L.n + w + k",        # L.n = first lag of n
    data, "id", "year",
    gmm_instruments=["L.n"],
    iv_instruments=["w", "k"]
)
```

| Syntax | Meaning |
|--------|---------|
| `L.x` | First lag of x (x_{t-1}) |
| `L2.x` | Second lag of x (x_{t-2}) |

## Results Objects

Calling `model.fit()` returns a results object with a consistent interface across all models:

```python
results = model.fit(cov_type="clustered")
```

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `results.params` | `pd.Series` | Estimated coefficients |
| `results.std_errors` | `pd.Series` | Standard errors |
| `results.tvalues` | `pd.Series` | t-statistics |
| `results.pvalues` | `pd.Series` | p-values |
| `results.conf_int()` | `pd.DataFrame` | Confidence intervals |
| `results.rsquared` | `float` | R-squared |
| `results.rsquared_within` | `float` | Within R-squared (FE models) |
| `results.nobs` | `int` | Number of observations |
| `results.resid` | `np.ndarray` | Residuals |

### Key Methods

```python
# Full summary table
print(results.summary())

# Confidence intervals (default 95%)
print(results.conf_int(alpha=0.05))

# Predictions
y_hat = results.predict(new_data)
```

### Standard Error Options

Control standard errors via the `cov_type` parameter:

```python
results = model.fit(cov_type="robust")      # HC1 robust
results = model.fit(cov_type="clustered")    # Clustered by entity
results = model.fit(cov_type="kernel")       # Driscoll-Kraay
```

| `cov_type` | Description | When to Use |
|------------|-------------|-------------|
| `"unadjusted"` | Classical OLS | Homoskedastic errors |
| `"robust"` | HC1 heteroskedasticity-robust | Default for most cases |
| `"clustered"` | Clustered by entity | Within-entity correlation |
| `"kernel"` | Driscoll-Kraay | Cross-sectional dependence |

## The Experiment Pattern

For larger analyses, the **PanelExperiment** class provides a structured workflow:

```text
fit_all_models → validate → compare → save_master_report
```

```python
from panelbox.experiment import PanelExperiment

# 1. Create experiment
exp = PanelExperiment(data, "invest ~ value + capital", "firm", "year")

# 2. Fit multiple models
exp.fit_all_models(["pooled", "fe", "re"])

# 3. Validate preferred model
validation = exp.validate_model("fe")

# 4. Compare models side by side
comparison = exp.compare_models(["pooled", "fe", "re"])

# 5. Generate report
exp.save_master_report("analysis.html")
```

The Experiment pattern handles:

- Fitting all requested models with a single call
- Running standard diagnostic tests (Hausman, Breusch-Pagan, etc.)
- Generating side-by-side comparison tables
- Producing a self-contained HTML report with interactive charts

!!! tip "When to use PanelExperiment"
    Use `PanelExperiment` when you want to compare multiple models or generate reports. For a single model estimation, direct model usage (`FixedEffects(...).fit()`) is simpler.

## Key Terminology

| Term | Definition |
|------|-----------|
| **Entity** | Cross-sectional unit (firm, individual, country). Also called *group* or *panel unit*. |
| **Time period** | Temporal observation point (year, quarter, month). |
| **Balanced panel** | All entities observed in all time periods. |
| **Unbalanced panel** | Some entities missing in some periods. |
| **Within variation** | How a variable changes over time *within* the same entity. |
| **Between variation** | How a variable differs *across* entities (entity averages). |
| **Fixed effects** | Entity-specific intercepts that capture unobserved time-invariant heterogeneity. |
| **Random effects** | Entity-specific effects modeled as random draws from a distribution. |
| **Strict exogeneity** | Regressors are uncorrelated with the error term in all time periods. |
| **Endogeneity** | A regressor is correlated with the error term (violates exogeneity). |
| **Instruments** | Variables correlated with the endogenous regressor but uncorrelated with the error. |
| **Nickell bias** | Downward bias in dynamic FE models when T is small. Solved by GMM. |
| **Clustered SE** | Standard errors that allow for correlation within entities. |

## Next Steps

- **[Choosing a Model](choosing-model.md)** -- Decision guide for all 13 model families
- **[Static Models](../user-guide/static-models/index.md)** -- Pooled OLS, Fixed Effects, Random Effects
- **[GMM Models](../user-guide/gmm/index.md)** -- Arellano-Bond and Blundell-Bond estimators
- **[API Reference](../api/index.md)** -- Complete class and method documentation
