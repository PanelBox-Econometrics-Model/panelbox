---
title: Charts & Visualization
description: Guide to the PanelBox interactive visualization system - 35+ chart types with Plotly, customizable themes, and ChartFactory API.
---

# Charts & Visualization

PanelBox includes a comprehensive visualization system with **35+ interactive chart types** built on Plotly. Charts are designed for panel data workflows: residual diagnostics, model comparison, coefficient plots, time series decomposition, and more. Three built-in themes (Professional, Academic, Presentation) ensure publication-ready output.

## Chart Categories

<div class="grid cards" markdown>

- :material-chart-scatter-plot: **Residual Diagnostics**

    ---

    Q-Q plots, residuals vs. fitted, scale-location, leverage plots, residual distribution, partial regression

- :material-compare-horizontal: **Model Comparison**

    ---

    Coefficient comparison, forest plots, model fit comparison, information criteria charts

- :material-chart-bar: **Distribution**

    ---

    Histograms, KDE plots, violin plots, box plots

- :material-chart-line: **Time Series**

    ---

    Panel time series, trend lines, faceted time series

- :material-view-dashboard: **Panel-Specific**

    ---

    Entity effects, time effects, between-within variation, panel structure

- :material-test-tube: **Econometric Tests**

    ---

    ACF/PACF, unit root visualization, cointegration heatmaps, cross-sectional dependence

- :material-check-circle: **Validation**

    ---

    Test overview, p-value distribution, test statistics, test comparison heatmap, validation dashboard

- :material-chart-bell-curve: **Quantile**

    ---

    Quantile process plots, 3D surfaces, interactive coefficient plots

</div>

## Available Charts

### Residual Diagnostics

| Chart | Class | Description |
|-------|-------|-------------|
| Q-Q Plot | `QQPlot` | Normality check for residuals |
| Residuals vs. Fitted | `ResidualVsFittedPlot` | Heteroskedasticity and nonlinearity |
| Scale-Location | `ScaleLocationPlot` | Variance stability |
| Residuals vs. Leverage | `ResidualVsLeveragePlot` | Influential observations |
| Residual Time Series | `ResidualTimeSeriesPlot` | Serial correlation patterns |
| Residual Distribution | `ResidualDistributionPlot` | Histogram + KDE of residuals |
| Partial Regression | `PartialRegressionPlot` | Added-variable plots |

### Model Comparison

| Chart | Class | Description |
|-------|-------|-------------|
| Coefficient Comparison | `CoefficientComparisonChart` | Side-by-side coefficient estimates |
| Forest Plot | `ForestPlotChart` | Confidence interval visualization |
| Model Fit Comparison | `ModelFitComparisonChart` | R-squared, AIC, BIC comparison |
| Information Criteria | `InformationCriteriaChart` | AIC/BIC across models |

### Distribution

| Chart | Class | Description |
|-------|-------|-------------|
| Histogram | `HistogramChart` | Frequency distribution |
| KDE | `KDEChart` | Kernel density estimation |
| Violin Plot | `ViolinPlotChart` | Distribution shape by group |
| Box Plot | `BoxPlotChart` | Summary statistics by group |

### Time Series

| Chart | Class | Description |
|-------|-------|-------------|
| Panel Time Series | `PanelTimeSeriesChart` | Multi-entity time series |
| Trend Line | `TrendLineChart` | Trend visualization |
| Faceted Time Series | `FacetedTimeSeriesChart` | Small multiples by entity |

### Panel-Specific

| Chart | Class | Description |
|-------|-------|-------------|
| Entity Effects | `EntityEffectsPlot` | Fixed/random effects by entity |
| Time Effects | `TimeEffectsPlot` | Time fixed effects |
| Between-Within | `BetweenWithinPlot` | Variance decomposition |
| Panel Structure | `PanelStructurePlot` | Balance and coverage |

### Econometric Tests

| Chart | Class | Description |
|-------|-------|-------------|
| ACF/PACF | `ACFPACFPlot` | Autocorrelation diagnostics |
| Unit Root Test | `UnitRootTestPlot` | Stationarity visualization |
| Cointegration Heatmap | `CointegrationHeatmap` | Pairwise cointegration tests |
| Cross-Sectional Dependence | `CrossSectionalDependencePlot` | CD test visualization |

### Validation

| Chart | Class | Description |
|-------|-------|-------------|
| Test Overview | `TestOverviewChart` | Summary of all diagnostic tests |
| P-Value Distribution | `PValueDistributionChart` | Distribution of test p-values |
| Test Statistics | `TestStatisticsChart` | Test statistic visualization |
| Test Comparison Heatmap | `TestComparisonHeatmap` | Cross-model test comparison |
| Validation Dashboard | `ValidationDashboard` | Comprehensive validation view |

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.visualization import ChartFactory
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit(cov_type="clustered")

# Create charts using ChartFactory
factory = ChartFactory(results)

# Residual diagnostics
factory.residual_vs_fitted().show()
factory.qq_plot().show()

# Entity effects
factory.entity_effects().show()
```

## Using ChartFactory

The `ChartFactory` is the recommended entry point for creating charts. It automatically extracts the necessary data from model results:

```python
from panelbox.visualization import ChartFactory

factory = ChartFactory(results)

# All chart types are available as methods
chart = factory.residual_vs_fitted()
chart = factory.coefficient_comparison([results1, results2])
chart = factory.forest_plot()
chart = factory.entity_effects()
```

### Direct Class Usage

For more control, instantiate chart classes directly:

```python
from panelbox.visualization import QQPlot

chart = QQPlot(residuals=results.resid)
fig = chart.create()
fig.show()
```

## Themes

PanelBox provides three built-in themes:

| Theme | Best For |
|-------|---------|
| Professional | Reports, client presentations |
| Academic | Journal papers, dissertations |
| Presentation | Slides, large-screen display |

```python
from panelbox.visualization import ChartFactory, Theme

factory = ChartFactory(results, theme=Theme.ACADEMIC)
chart = factory.forest_plot()
chart.show()
```

## Exporting Charts

```python
# Save as HTML (interactive)
chart.write_html("chart.html")

# Save as static image
chart.write_image("chart.png", width=800, height=600)
chart.write_image("chart.pdf")
chart.write_image("chart.svg")
```

## Detailed Guides

- [Residual Diagnostics](model-diagnostics.md) -- Residual analysis charts *(detailed guide coming soon)*
- [Model Comparison](comparison.md) -- Comparing model results visually *(detailed guide coming soon)*
- [Themes & Customization](themes.md) -- Customizing chart appearance *(detailed guide coming soon)*

## Tutorials

See [Visualization Tutorial](../../tutorials/visualization.md) for interactive notebooks with Google Colab.

## API Reference

See [Visualization API](../../api/visualization.md) for complete technical reference.
