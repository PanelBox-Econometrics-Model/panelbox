# PanelBox Visualization Module - Quick Reference Guide

## Location
`/home/guhaase/projetos/panelbox/panelbox/visualization/`

## Module Structure at a Glance

```
visualization/
├── Core Architecture
│   ├── api.py              (High-level convenience functions)
│   ├── base.py             (Abstract base classes)
│   ├── factory.py          (ChartFactory - main entry point)
│   ├── registry.py         (Chart registry system)
│   ├── themes.py           (Visual themes)
│   ├── exceptions.py       (Custom exceptions)
│
├── Chart Implementations
│   ├── plotly/             (35 interactive chart types)
│   ├── matplotlib/         (Static Matplotlib charts)
│   ├── quantile/           (Quantile regression visualizations)
│   ├── spatial_plots.py    (Spatial model charts)
│   └── var_plots.py        (VAR model charts)
│
├── Supporting Infrastructure
│   ├── config/             (Configuration classes)
│   ├── transformers/       (Data transformation layer)
│   └── utils/              (Utility functions)
```

## Quick Start Examples

### 1. Create Validation Charts
```python
from panelbox.visualization import create_validation_charts

charts = create_validation_charts(
    validation_data,
    theme='professional'
)
charts['test_overview'].to_html()
```

### 2. Create Residual Diagnostics
```python
from panelbox.visualization import create_residual_diagnostics

diagnostics = create_residual_diagnostics(
    model_results,
    theme='academic'
)
diagnostics['qq_plot'].save_image('qq_plot.png')
```

### 3. Use ChartFactory Directly
```python
from panelbox.visualization import ChartFactory

chart = ChartFactory.create(
    'residual_qq_plot',
    data={'residuals': residuals},
    theme='professional'
)
html = chart.to_html()
```

### 4. Export Multiple Charts
```python
from panelbox.visualization import export_charts_multiple_formats

paths = export_charts_multiple_formats(
    charts,
    output_dir='output/',
    formats=['png', 'svg', 'pdf']
)
```

## Available Themes
- `PROFESSIONAL_THEME` (default)
- `ACADEMIC_THEME`
- `PRESENTATION_THEME`

Or create custom: `Theme(name='custom', color_scheme=[...], font_config={...}, ...)`

## Chart Types (35 Total)

| Category | Types |
|----------|-------|
| Residual Diagnostics | qq_plot, vs_fitted, scale_location, vs_leverage, timeseries, distribution, partial_regression |
| Model Comparison | coefficients, forest_plot, model_fit, ic |
| Distributions | histogram, kde, violin, boxplot |
| Correlation | heatmap, pairwise |
| Time Series | panel, trend, faceted |
| Panel | entity_effects, time_effects, between_within, structure |
| Validation | test_overview, pvalue_distribution, test_statistics, comparison_heatmap, dashboard |
| Econometric Tests | acf_pacf, unit_root, cointegration, cross_sectional_dependence |
| Basic | bar, line |

## High-Level APIs (Main Entry Points)

```python
# Validation
create_validation_charts(validation_data, theme, interactive, charts)

# Residuals
create_residual_diagnostics(results, theme, charts)

# Comparison
create_comparison_charts(results_list, names, theme, charts)

# Panel Data
create_panel_charts(panel_results, chart_types, theme)
create_entity_effects_plot(panel_results, theme)
create_time_effects_plot(panel_results, theme)
create_between_within_plot(panel_data, variables, theme, style)
create_panel_structure_plot(panel_data, theme)

# Econometric Tests
create_acf_pacf_plot(residuals, max_lags, confidence_level, theme)
create_unit_root_test_plot(test_results, include_series, theme)
create_cointegration_heatmap(cointegration_results, theme)
create_cross_sectional_dependence_plot(cd_results, theme)

# Export
export_chart(chart, file_path, format, width, height, scale)
export_charts(charts, output_dir, format, prefix, width, height, scale)
export_charts_multiple_formats(charts, output_dir, formats, ...)
```

## ChartFactory Methods

```python
ChartFactory.create(chart_type, data, theme, config)
ChartFactory.create_multiple(chart_specs, common_theme)
ChartFactory.list_available_charts()
ChartFactory.get_chart_info(chart_type)
```

## Export Formats Supported

- **HTML**: Interactive or embedded
- **PNG**: Standard and high-resolution
- **SVG**: Vector graphics
- **PDF**: Print-ready
- **JPEG**: Compressed
- **WebP**: Modern web format
- **JSON**: Data interchange

## Key Classes

### BaseChart (Abstract)
All charts inherit from this. Main methods:
- `create(data, **kwargs)` - Create chart
- `to_html()` - Export as HTML
- `to_json()` - Export as JSON
- `to_dict()` - Export as dictionary

### PlotlyChartBase (extends BaseChart)
For interactive charts. Additional methods:
- `to_image(format, width, height, scale)` - Export as image bytes
- `save_image(file_path, ...)` - Save to file
- `to_png()`, `to_svg()`, `to_pdf()` - Convenience exports

### ChartFactory
Static factory class for creating charts
- Single entry point for all chart types
- Automatic theme resolution
- Configuration management

### ChartRegistry
Centralized chart management
- `register(name, chart_class)` - Register chart
- `get(name)` - Retrieve chart class
- `list_charts()` - List all registered
- Decorator: `@register_chart('name')`

### Theme
Dataclass for visual styling
- `color_scheme` - List of hex colors
- `font_config` - Font settings
- `layout_config` - Layout defaults
- `plotly_template` - Plotly template
- `matplotlib_style` - Matplotlib style

## Data Transformation Layer

Transformers convert model results to chart-friendly format:
- `ValidationDataTransformer` - Validation results
- `ResidualDataTransformer` - Residual data
- `ComparisonDataTransformer` - Model comparison data
- `PanelDataTransformer` - Panel data extraction

## Design Patterns

1. **Factory Pattern** - ChartFactory for centralized creation
2. **Registry Pattern** - @register_chart() decorator
3. **Strategy Pattern** - Multiple rendering backends
4. **Template Method** - BaseChart.create() orchestrates workflow
5. **Decorator Pattern** - Theme application

## Main Files

| File | Purpose | Lines |
|------|---------|-------|
| api.py | High-level APIs | 1,386 |
| base.py | Base classes | 854 |
| factory.py | Factory implementation | 241 |
| registry.py | Registry implementation | 306 |
| themes.py | Theme definitions | 402 |
| var_plots.py | VAR visualizations | 1,164 |
| spatial_plots.py | Spatial visualizations | 866 |

## Dependencies

Required:
- plotly
- matplotlib
- numpy
- scipy

Optional:
- kaleido (for image export)
- seaborn (matplotlib styling)

## Example: Complete Workflow

```python
from panelbox.visualization import (
    create_validation_charts,
    create_residual_diagnostics,
    export_charts_multiple_formats,
    ACADEMIC_THEME
)

# Create validation charts
validation_charts = create_validation_charts(
    validation_results,
    theme='academic'
)

# Create residual diagnostics
residual_charts = create_residual_diagnostics(
    model_results,
    theme='academic'
)

# Combine and export
all_charts = {**validation_charts, **residual_charts}

paths = export_charts_multiple_formats(
    all_charts,
    output_dir='reports/charts/',
    formats=['png', 'pdf'],
    width=1200,
    height=800
)
```

## Integration with PanelBox Models

All PanelBox model results work seamlessly with visualization:

```python
from panelbox import FixedEffects, RandomEffects

# Fit models
fe = FixedEffects("y ~ x1 + x2", data, "firm", "year")
fe_results = fe.fit()

# Create visualizations directly
from panelbox.visualization import create_residual_diagnostics
charts = create_residual_diagnostics(fe_results)
```

## Resources

- Full documentation: `/home/guhaase/projetos/panelbox/VISUALIZATION_SUMMARY.md`
- Directory tree: `/home/guhaase/projetos/panelbox/VISUALIZATION_DIRECTORY_TREE.txt`
- Module location: `/home/guhaase/projetos/panelbox/panelbox/visualization/`
