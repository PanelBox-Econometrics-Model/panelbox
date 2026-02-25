---
title: "Visualization API"
description: "API reference for panelbox.visualization — chart factory, themes, 28+ chart types, and export utilities"
---

# Visualization API Reference

!!! info "Module"
    **Import**: `from panelbox.visualization import ...`
    **Source**: `panelbox/visualization/`

## Overview

PanelBox provides a comprehensive visualization system built on Plotly and Matplotlib, with a chart factory pattern, theme system, and 28+ registered chart types covering residual diagnostics, model comparison, panel structure, and econometric tests.

| Class / Function | Description |
|-----------------|-------------|
| `ChartFactory` | Static factory for creating charts by type name |
| `ChartRegistry` | Registry for managing and discovering chart types |
| `Theme` | Dataclass for consistent styling across all charts |
| `BaseChart` | Abstract base class for all chart types |
| `PlotlyChartBase` | Base class for Plotly-based interactive charts |
| `MatplotlibChartBase` | Base class for Matplotlib-based static charts |
| `register_chart` | Decorator to register custom chart types |
| `get_theme` / `register_theme` | Theme utility functions |

---

## ChartFactory

Static factory for creating charts by type name.

### `ChartFactory.create()`

```python
@staticmethod
def create(
    chart_type: str,
    data: dict[str, Any] | None = None,
    theme: str | Theme | None = None,
    config: dict[str, Any] | None = None,
    **kwargs,
) -> BaseChart
```

### `ChartFactory.create_multiple()`

```python
@staticmethod
def create_multiple(
    chart_specs: list[dict[str, Any]],
    common_theme: str | Theme | None = None,
) -> dict[str, BaseChart]
```

### `ChartFactory.list_available_charts()`

```python
@staticmethod
def list_available_charts() -> list[str]
```

### `ChartFactory.get_chart_info()`

```python
@staticmethod
def get_chart_info(chart_type: str) -> dict[str, str]
```

**Example:**

```python
from panelbox.visualization import ChartFactory

# List available chart types
print(ChartFactory.list_available_charts())

# Create a chart
chart = ChartFactory.create("qq_plot", data=residual_data, theme="academic")
html = chart.to_html()
```

---

## ChartRegistry

Registry for managing chart types. Used by `ChartFactory` internally, and for extending PanelBox with custom charts.

```python
class ChartRegistry:
    @classmethod
    def register(cls, name: str, chart_class: type[BaseChart]) -> None
    @classmethod
    def get(cls, name: str) -> type[BaseChart]
    @classmethod
    def list_charts(cls) -> list[str]
    @classmethod
    def is_registered(cls, name: str) -> bool
    @classmethod
    def unregister(cls, name: str) -> None
    @classmethod
    def clear(cls) -> None
    @classmethod
    def get_chart_info(cls, name: str) -> dict[str, str]
```

### `register_chart` Decorator

Register a custom chart type:

```python
from panelbox.visualization import register_chart, PlotlyChartBase

@register_chart("my_custom_chart")
class MyCustomChart(PlotlyChartBase):
    def _create_figure(self, data, **kwargs):
        # Create and return a Plotly figure
        ...
```

---

## Registered Chart Types

### Residual Diagnostics

| Chart Type | Class | Description |
|------------|-------|-------------|
| `qq_plot` | `QQPlot` | Quantile-quantile plot |
| `residual_vs_fitted` | `ResidualVsFittedPlot` | Residuals vs. fitted values |
| `scale_location` | `ScaleLocationPlot` | Scale-location (spread-level) plot |
| `residual_vs_leverage` | `ResidualVsLeveragePlot` | Residuals vs. leverage |
| `residual_time_series` | `ResidualTimeSeriesPlot` | Residuals over time |
| `residual_distribution` | `ResidualDistributionPlot` | Histogram/KDE of residuals |
| `partial_regression` | `PartialRegressionPlot` | Added-variable (partial regression) plots |

### Model Comparison

| Chart Type | Class | Description |
|------------|-------|-------------|
| `coefficient_comparison` | `CoefficientComparisonChart` | Coefficients across models |
| `forest_plot` | `ForestPlotChart` | Forest plot with confidence intervals |
| `model_fit_comparison` | `ModelFitComparisonChart` | R², AIC, BIC comparison |
| `information_criteria` | `InformationCriteriaChart` | AIC/BIC comparison bars |

### Distribution

| Chart Type | Class | Description |
|------------|-------|-------------|
| `histogram` | `HistogramChart` | Histogram |
| `kde` | `KDEChart` | Kernel density estimation |
| `violin` | `ViolinPlotChart` | Violin plots |
| `boxplot` | `BoxPlotChart` | Box plots |

### Correlation

| Chart Type | Class | Description |
|------------|-------|-------------|
| `correlation_heatmap` | `CorrelationHeatmapChart` | Correlation matrix heatmap |
| `pairwise_correlation` | `PairwiseCorrelationChart` | Pairwise scatter matrix |

### Panel Data

| Chart Type | Class | Description |
|------------|-------|-------------|
| `panel_time_series` | `PanelTimeSeriesChart` | Time series by entity |
| `trend_line` | `TrendLineChart` | Trend analysis |
| `faceted_time_series` | `FacetedTimeSeriesChart` | Small multiples by entity |
| `entity_effects` | `EntityEffectsPlot` | Fixed entity effects |
| `time_effects` | `TimeEffectsPlot` | Fixed time effects |
| `between_within` | `BetweenWithinPlot` | Between vs. within variation |
| `panel_structure` | `PanelStructurePlot` | Panel balance and structure |

### Econometric Tests

| Chart Type | Class | Description |
|------------|-------|-------------|
| `acf_pacf` | `ACFPACFPlot` | Autocorrelation and partial ACF |
| `unit_root_test` | `UnitRootTestPlot` | Unit root test visualization |
| `cointegration_heatmap` | `CointegrationHeatmap` | Cointegration results heatmap |
| `cross_sectional_dependence` | `CrossSectionalDependencePlot` | CD test visualization |

### Validation

| Chart Type | Class | Description |
|------------|-------|-------------|
| `test_overview` | `TestOverviewChart` | Validation test summary |
| `pvalue_distribution` | `PValueDistributionChart` | p-value distribution |
| `test_statistics` | `TestStatisticsChart` | Test statistics bar chart |
| `test_comparison_heatmap` | `TestComparisonHeatmap` | Multi-test comparison |
| `validation_dashboard` | `ValidationDashboard` | Full validation dashboard |

---

## Theme System

### `Theme`

```python
@dataclass
class Theme:
    name: str
    color_scheme: list[str]
    font_config: dict[str, Any]
    layout_config: dict[str, Any]
    plotly_template: str = "plotly_white"
    matplotlib_style: str = "seaborn-v0_8-whitegrid"
    success_color: str = "#28a745"
    warning_color: str = "#ffc107"
    danger_color: str = "#dc3545"
    info_color: str = "#17a2b8"
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_color(index)` | `str` | Get color from scheme (wraps around) |
| `to_dict()` | `dict` | Serialize theme |

### Built-in Themes

| Theme | Constant | Primary Color | Use Case |
|-------|----------|---------------|----------|
| Professional | `PROFESSIONAL_THEME` | Blue (#2563eb) | Corporate reports |
| Academic | `ACADEMIC_THEME` | Gray (#4b5563) | Research papers |
| Presentation | `PRESENTATION_THEME` | Purple (#7c3aed) | Slides, demos |

### Theme Utilities

```python
def get_theme(theme: str | Theme) -> Theme
def register_theme(theme: Theme) -> None
def list_themes() -> list[str]
```

**Example:**

```python
from panelbox.visualization import Theme, register_theme, get_theme

# Create custom theme
my_theme = Theme(
    name="corporate",
    color_scheme=["#003366", "#336699", "#6699CC", "#99CCFF"],
    font_config={"family": "Arial", "size": 14},
    layout_config={"margin": {"t": 60, "b": 60}},
)
register_theme(my_theme)

# Use custom theme
chart = ChartFactory.create("forest_plot", data=data, theme="corporate")
```

---

## Base Classes

### `BaseChart`

Abstract base class for all chart types.

```python
class BaseChart(theme: Theme | None = None, config: dict | None = None)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `create(data, **kwargs)` | `BaseChart` | Create chart from data |
| `to_json()` | `str` | Serialize to JSON (abstract) |
| `to_html(**kwargs)` | `str` | Render as HTML (abstract) |
| `to_dict()` | `dict` | Convert to dictionary |

### `PlotlyChartBase`

Base class for Plotly-based interactive charts.

```python
class PlotlyChartBase(theme: Theme | None = None, config: dict | None = None)
```

**Additional Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_html(include_plotlyjs="cdn", config=None)` | `str` | Self-contained HTML |
| `to_image(format="png", width=None, height=None, scale=1.0)` | `bytes` | Static image |
| `save_image(file_path, format=None, width=None, height=None, scale=1.0)` | — | Save to file |
| `to_png(width=None, height=None, scale=1.0)` | `bytes` | PNG bytes |
| `to_svg(width=None, height=None)` | `bytes` | SVG bytes |
| `to_pdf(width=None, height=None)` | `bytes` | PDF bytes |

### `MatplotlibChartBase`

Base class for Matplotlib-based static charts.

```python
class MatplotlibChartBase(theme: Theme | None = None, config: dict | None = None)
```

**Additional Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_base64(format="png", dpi=150)` | `str` | Base64-encoded image |

---

## High-Level Convenience Functions

### Residual Diagnostics

```python
def create_residual_diagnostics(
    results,
    theme: str = "academic",
    charts: list[str] | None = None,
) -> dict[str, BaseChart]
```

### Validation Charts

```python
def create_validation_charts(
    validation_report,
    theme: str = "professional",
    interactive: bool = True,
) -> dict[str, BaseChart]
```

### Model Comparison

```python
def create_comparison_charts(
    results_list,
    model_names: list[str] | None = None,
    theme: str = "professional",
) -> dict[str, BaseChart]
```

### Panel Structure

```python
def create_panel_charts(data, theme="professional", charts=None) -> dict[str, BaseChart]
def create_entity_effects_plot(data, entity_col, time_col, theme=None) -> BaseChart
def create_time_effects_plot(data, entity_col, time_col, theme=None) -> BaseChart
def create_between_within_plot(data, entity_col, time_col, theme=None) -> BaseChart
def create_panel_structure_plot(data, entity_col, time_col, theme=None) -> BaseChart
```

### Econometric Tests

```python
def create_acf_pacf_plot(data, lags=None, theme=None) -> BaseChart
def create_unit_root_test_plot(test_results, theme=None) -> BaseChart
def create_cointegration_heatmap(data, theme=None) -> BaseChart
def create_cross_sectional_dependence_plot(data, theme=None) -> BaseChart
```

---

## Export Functions

```python
def export_chart(chart: BaseChart, file_path: str, format: str = "html") -> None
def export_charts(charts: dict[str, BaseChart], output_dir: str, formats: list[str] = ["html"]) -> None
def export_charts_multiple_formats(chart: BaseChart, output_path: str, formats: list[str] = ["html", "png"]) -> None
```

**Example:**

```python
from panelbox.visualization import create_residual_diagnostics, export_charts

charts = create_residual_diagnostics(fe_result, theme="academic")
export_charts(charts, output_dir="./plots", formats=["html", "png"])
```

---

## Data Transformers

Internal transformers that convert model outputs to chart data schemas.

| Transformer | Description |
|-------------|-------------|
| `ValidationDataTransformer` | Transform validation results for charts |
| `ComparisonDataTransformer` | Transform comparison data for charts |
| `PanelDataTransformer` | Transform panel data for structure charts |
| `ResidualDataTransformer` | Transform residuals for diagnostic charts |

---

## See Also

- [Report API](report.md) — generate full HTML/LaTeX reports with charts
- [Experiment API](experiment.md) — integrated model fitting and visualization
- [Tutorials: Visualization](../tutorials/visualization.md) — practical chart creation guide
