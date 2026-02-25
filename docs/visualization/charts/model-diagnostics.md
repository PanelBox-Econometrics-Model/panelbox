---
title: "Diagnostics Plots"
description: "Residual analysis and model validation plots for panel data models with PanelBox"
---

# Diagnostics Plots

Residual diagnostic plots provide visual assessment of model assumptions: normality, homoskedasticity, independence, and influence. PanelBox generates 7 interactive Plotly charts from any model results object.

## Quick Start

```python
from panelbox.visualization import create_residual_diagnostics

# Create all 6 standard diagnostic charts
charts = create_residual_diagnostics(results, theme="professional")

# Access individual charts
charts["qq_plot"].to_html()
charts["residual_vs_fitted"].save_image("rvf.png", width=800, height=600)
```

The `create_residual_diagnostics()` function returns a dictionary with keys: `qq_plot`, `residual_vs_fitted`, `scale_location`, `residual_vs_leverage`, `residual_timeseries`, `residual_distribution`.

You can also select specific charts:

```python
charts = create_residual_diagnostics(
    results,
    theme="academic",
    charts=["qq_plot", "residual_vs_fitted"],
)
```

## Q-Q Plot

**Registry name**: `residual_qq_plot`

Compares the distribution of residuals against a theoretical normal distribution. Points falling along the diagonal indicate normality.

```python
from panelbox.visualization import ChartFactory

chart = ChartFactory.create(
    "residual_qq_plot",
    data={
        "residuals": results.resids,
        "standardized": True,         # Use standardized residuals
        "show_confidence": True,      # Show 95% confidence bands
        "confidence_level": 0.95,
    },
    theme="professional",
)
```

**Interpretation**:

- Points on the diagonal: residuals are normally distributed
- S-shaped curve: heavy tails (leptokurtic) or light tails (platykurtic)
- Upward curve at right end: right-skewed distribution
- Systematic departures: non-normality may affect inference

## Residuals vs Fitted

**Registry name**: `residual_vs_fitted`

Scatter plot of residuals against fitted values to detect heteroskedasticity and nonlinearity.

```python
chart = ChartFactory.create(
    "residual_vs_fitted",
    data={
        "fitted": results.fitted_values,
        "residuals": results.resids,
        "add_lowess": True,       # LOWESS smoothing line
        "add_reference": True,    # Horizontal line at y=0
    },
    theme="professional",
)
```

**Interpretation**:

- Random scatter around zero: assumptions satisfied
- Funnel shape (widening/narrowing): heteroskedasticity present
- Curved LOWESS line: nonlinear relationship missed by the model
- Clusters or patterns: possible omitted variable bias

## Scale-Location Plot

**Registry name**: `residual_scale_location`

Plots $\sqrt{|\text{standardized residuals}|}$ against fitted values. Useful for detecting heteroskedasticity independently from the residuals-vs-fitted plot.

```python
chart = ChartFactory.create(
    "residual_scale_location",
    data={
        "fitted": results.fitted_values,
        "residuals": results.resids,
        "add_lowess": True,
    },
    theme="academic",
)
```

**Interpretation**:

- Flat LOWESS line: constant variance (homoskedasticity)
- Upward slope: variance increases with fitted values
- Downward slope: variance decreases with fitted values

## Residuals vs Leverage

**Registry name**: `residual_vs_leverage`

Identifies influential observations using leverage values and Cook's distance contours.

```python
chart = ChartFactory.create(
    "residual_vs_leverage",
    data={
        "residuals": results.resids,
        "leverage": leverage_values,
        "cooks_d": cooks_distances,       # Optional
        "show_contours": True,            # Cook's distance contours at 0.5 and 1.0
        "labels": entity_labels,          # Optional point labels
    },
    theme="professional",
)
```

**Interpretation**:

- Points beyond Cook's distance = 0.5: moderately influential
- Points beyond Cook's distance = 1.0: highly influential, investigate these observations
- High leverage + large residual: observation may be distorting the regression

## Residual Time Series

**Registry name**: `residual_timeseries`

Plots residuals over time to visually detect serial correlation.

```python
chart = ChartFactory.create(
    "residual_timeseries",
    data={
        "residuals": results.resids,
        "time_index": time_periods,   # Optional, defaults to range index
        "add_bands": True,            # +/- 2 standard deviation bands
    },
    theme="professional",
)
```

**Interpretation**:

- Random scatter within bands: no serial correlation
- Runs of positive/negative residuals: positive autocorrelation
- Rapid alternation: negative autocorrelation
- Points outside bands: potential outliers

## Residual Distribution

**Registry name**: `residual_distribution`

Histogram of residuals with KDE and theoretical normal distribution overlay.

```python
chart = ChartFactory.create(
    "residual_distribution",
    data={
        "residuals": results.resids,
        "bins": "auto",           # Or integer for fixed bins
        "show_kde": True,         # Kernel density estimate
        "show_normal": True,      # Theoretical normal overlay
    },
    theme="academic",
)
```

**Interpretation**:

- KDE matching normal curve: residuals are approximately normal
- Heavy tails (wider KDE): leptokurtic distribution
- Skewed KDE: asymmetric residuals, check for outliers or misspecification

## Partial Regression Plot

**Registry name**: `residual_partial_regression`

Added-variable plot showing the partial effect of one predictor after controlling for all others.

```python
chart = ChartFactory.create(
    "residual_partial_regression",
    data={
        "y_resid": y_residuals,           # y residuals from auxiliary regression
        "x_resid": x_residuals,           # x residuals from auxiliary regression
        "variable_name": "education",
        "add_regression_line": True,
        "add_confidence": True,           # 95% confidence band
    },
    theme="academic",
)
```

**Interpretation**:

- Slope of the fitted line: partial regression coefficient for that variable
- Tight confidence band: precise estimate
- Nonlinear pattern: consider transformations or polynomial terms

## Complete Example

Full diagnostic workflow for a Fixed Effects model:

```python
import panelbox as pb
from panelbox.visualization import create_residual_diagnostics, export_charts

# Estimate model
model = pb.FixedEffects(
    data=panel_data,
    formula="lwage ~ hours + age + tenure + EntityEffects",
)
results = model.fit()

# Generate all diagnostic plots
charts = create_residual_diagnostics(results, theme="professional")

# Display Q-Q plot (in Jupyter)
charts["qq_plot"].figure.show()

# Export all charts as PNG
paths = export_charts(
    charts,
    output_dir="./diagnostics",
    format="png",
    width=800,
    height=600,
    scale=2.0,       # Retina resolution
    prefix="fe_",
)
# Output: fe_qq_plot.png, fe_residual_vs_fitted.png, ...

# Export as interactive HTML
for name, chart in charts.items():
    with open(f"diagnostics/{name}.html", "w") as f:
        f.write(chart.to_html(include_plotlyjs="cdn"))
```

## Data Transformers

The `ResidualDataTransformer` automatically extracts residual data from model results objects:

```python
from panelbox.visualization.transformers.residuals import ResidualDataTransformer

transformer = ResidualDataTransformer()

# Individual data preparation methods
qq_data = transformer.prepare_qq_data(results)
rvf_data = transformer.prepare_residual_fitted_data(results)
scale_data = transformer.prepare_scale_location_data(results)
leverage_data = transformer.prepare_leverage_data(results)
ts_data = transformer.prepare_timeseries_data(results)
dist_data = transformer.prepare_distribution_data(results)
```

## Comparison with Other Software

| Chart | PanelBox | Stata | R |
|-------|----------|-------|---|
| Q-Q plot | `residual_qq_plot` | `qnorm` | `qqnorm()`, `qqline()` |
| Residuals vs fitted | `residual_vs_fitted` | `rvfplot` | `plot(model, which=1)` |
| Scale-location | `residual_scale_location` | Manual | `plot(model, which=3)` |
| Residuals vs leverage | `residual_vs_leverage` | `lvr2plot` | `plot(model, which=5)` |
| Partial regression | `residual_partial_regression` | `avplot` | `car::avPlots()` |

## See Also

- [Panel Plots](panel-structure.md) -- Entity/time effects and panel structure
- [Test Plots](test-plots.md) -- ACF/PACF and other diagnostic test charts
- [Themes & Customization](themes.md) -- Styling your diagnostic charts
- [Comparison Plots](comparison.md) -- Comparing diagnostics across models
