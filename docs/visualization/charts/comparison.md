---
title: "Comparison Plots"
description: "Compare model coefficients, fit metrics, and information criteria across multiple panel data models"
---

# Comparison Plots

Comparison plots display coefficients, fit metrics, and information criteria side by side across multiple models. PanelBox provides 4 chart types for systematic model comparison.

## Quick Start

```python
from panelbox.visualization import create_comparison_charts

charts = create_comparison_charts(
    [pooled_results, fe_results, re_results],
    names=["Pooled OLS", "Fixed Effects", "Random Effects"],
    theme="professional",
)

# Access individual charts
charts["coefficients"].to_html()
charts["fit_comparison"].save_image("fit.png")
```

By default, `create_comparison_charts()` generates `coefficients`, `fit_comparison`, and `ic_comparison`. Select specific charts:

```python
charts = create_comparison_charts(
    results_list,
    names=model_names,
    charts=["coefficients", "forest_plot", "fit_comparison", "ic_comparison"],
)
```

!!! note
    The `forest_plot` is designed for a single model's coefficients. When passing multiple models, it is skipped with a warning.

## Coefficient Comparison

**Registry name**: `comparison_coefficients`

Grouped bar chart comparing coefficient estimates across models with standard error bars. A horizontal reference line at zero helps identify sign changes.

```python
from panelbox.visualization import ChartFactory

chart = ChartFactory.create(
    "comparison_coefficients",
    data={
        "models": ["Pooled OLS", "Fixed Effects", "Random Effects"],
        "coefficients": {
            "hours": [0.15, 0.12, 0.14],
            "education": [0.08, None, 0.07],   # FE drops time-invariant
            "experience": [0.03, 0.02, 0.025],
        },
        "std_errors": {                          # Optional
            "hours": [0.02, 0.03, 0.025],
            "education": [0.01, None, 0.012],
            "experience": [0.005, 0.006, 0.005],
        },
        "ci_level": 0.95,
    },
    theme="professional",
)
```

**Interpretation**:

- Similar coefficients across models: robust finding
- Sign changes: specification sensitivity, investigate further
- FE estimate differs from RE: Hausman test likely rejects, endogeneity present
- Wider error bars: less precise estimate, possibly fewer observations

## Forest Plot

**Registry name**: `comparison_forest_plot`

Publication-ready forest plot showing coefficient point estimates with horizontal confidence intervals. Colors indicate statistical significance.

```python
chart = ChartFactory.create(
    "comparison_forest_plot",
    data={
        "variables": ["hours", "education", "experience", "tenure"],
        "estimates": [0.12, 0.08, 0.025, 0.015],
        "ci_lower": [0.06, 0.05, 0.010, -0.002],
        "ci_upper": [0.18, 0.11, 0.040, 0.032],
        "pvalues": [0.001, 0.003, 0.02, 0.08],    # Optional
        "sort_by_size": True,                        # Optional
    },
    theme="academic",
)
```

Color coding by significance:

- Dark green: p < 0.001 (highly significant)
- Green: p < 0.01
- Orange: p < 0.05
- Gray: p >= 0.05 (not significant)

A vertical reference line at zero identifies which coefficients are statistically different from zero (CIs not crossing zero).

**Interpretation**:

- CI not crossing zero: statistically significant at the chosen level
- Narrow CI: precise estimate
- Wide CI: imprecise, consider more data or different specification
- Ordering by magnitude highlights the most important predictors

## Model Fit Comparison

**Registry name**: `comparison_model_fit`

Grouped bar chart comparing goodness-of-fit statistics across models.

```python
chart = ChartFactory.create(
    "comparison_model_fit",
    data={
        "models": ["Pooled OLS", "Fixed Effects", "Random Effects"],
        "metrics": {
            "R-squared": [0.25, 0.72, 0.65],
            "Adj. R-squared": [0.24, 0.68, 0.64],
            "F-statistic": [45.2, 120.5, 98.3],
        },
        "normalize": False,     # Optional: normalize metrics for comparison
    },
    theme="professional",
)
```

**Interpretation**:

- Higher R-squared in FE: entity effects capture substantial variation
- Large gap between R-squared and adjusted R-squared: possible overfitting
- Higher F-statistic: stronger overall model significance
- When comparing FE and RE, also consider the Hausman test (not just fit metrics)

## Information Criteria Chart

**Registry name**: `comparison_ic`

Grouped bar chart of AIC and BIC values across models. The model with the lowest IC value receives a gold border highlight. Shows delta values ($\Delta$) from the best model.

```python
chart = ChartFactory.create(
    "comparison_ic",
    data={
        "models": ["Pooled OLS", "Fixed Effects", "Random Effects"],
        "aic": [1250.3, 980.5, 1015.2],
        "bic": [1265.8, 1050.1, 1035.7],
        "hqic": [1255.0, 1005.2, 1020.3],   # Optional: Hannan-Quinn IC
        "show_delta": True,                    # Show delta from best model
    },
    theme="academic",
)
```

**Interpretation**:

- Lower AIC/BIC: better model (penalizes complexity)
- $\Delta$AIC < 2: models are essentially equivalent
- $\Delta$AIC 2--7: considerably less support for the higher-AIC model
- $\Delta$AIC > 10: virtually no support for the higher-AIC model
- AIC and BIC disagree: BIC penalizes complexity more, prefer BIC for large $N$

## Data Transformers

The `ComparisonDataTransformer` extracts comparison data from model results objects:

```python
from panelbox.visualization.transformers.comparison import ComparisonDataTransformer

transformer = ComparisonDataTransformer()

# Full transformation
data = transformer.transform(
    [pooled_results, fe_results, re_results],
    names=["Pooled", "FE", "RE"],
)
# Returns dict with: 'models', 'coefficients', 'std_errors',
#                     'pvalues', 'fit_metrics', 'ic_values'
```

## Complete Example

Compare three estimators for a wage equation:

```python
import panelbox as pb
from panelbox.visualization import create_comparison_charts, export_charts

# Estimate three models
pooled = pb.PooledOLS(data=data, formula="lwage ~ hours + exper + tenure").fit()
fe = pb.FixedEffects(data=data, formula="lwage ~ hours + exper + tenure + EntityEffects").fit()
re = pb.RandomEffects(data=data, formula="lwage ~ hours + exper + tenure").fit()

# Create all comparison charts
charts = create_comparison_charts(
    [pooled, fe, re],
    names=["Pooled OLS", "Fixed Effects", "Random Effects"],
    theme="professional",
)

# Forest plot for the preferred model
from panelbox.visualization import ChartFactory

forest = ChartFactory.create(
    "comparison_forest_plot",
    data={
        "variables": list(fe.params.index),
        "estimates": list(fe.params.values),
        "ci_lower": list(fe.conf_int().iloc[:, 0]),
        "ci_upper": list(fe.conf_int().iloc[:, 1]),
        "pvalues": list(fe.pvalues),
    },
    theme="academic",
)

# Export all
export_charts(charts, output_dir="./comparison", format="svg", prefix="model_")
forest.save_image("comparison/forest_plot.svg")
```

## Comparison with Other Software

| Chart | PanelBox | Stata | R |
|-------|----------|-------|---|
| Coefficient comparison | `comparison_coefficients` | `estimates table` | `modelsummary::modelplot()` |
| Forest plot | `comparison_forest_plot` | `coefplot` | `coefplot`, `forestplot` |
| Model fit | `comparison_model_fit` | `estimates stats` | `modelsummary()` |
| Information criteria | `comparison_ic` | `estimates stats` | `AIC()`, `BIC()` |

## See Also

- [Diagnostics Plots](model-diagnostics.md) -- Residual diagnostics for individual models
- [Test Plots](test-plots.md) -- Validation test comparison heatmap
- [Panel Plots](panel-structure.md) -- Entity and time effects from individual models
- [Themes & Customization](themes.md) -- Style your comparison charts
