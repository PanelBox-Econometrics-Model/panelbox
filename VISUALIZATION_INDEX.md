# PanelBox Visualization Module - Complete Documentation Index

This document serves as an index to the comprehensive documentation of the PanelBox visualization module.

## Documentation Files

Three detailed documentation files have been created in the project root:

### 1. [VISUALIZATION_SUMMARY.md](VISUALIZATION_SUMMARY.md) (630 lines, 20 KB)
**Comprehensive technical overview of the visualization module**

Contents:
- Top-level module structure and directories
- Detailed description of all visualization modules
- Complete API reference with all classes and functions
- All 35 registered chart types (organized by category)
- Visualization examples and demos
- Design patterns and architecture explanation
- Key features and capabilities checklist
- 8 detailed usage examples
- File metrics and code statistics
- Dependencies and import conventions

**Best for:** Understanding the complete architecture and getting reference information.

### 2. [VISUALIZATION_DIRECTORY_TREE.txt](VISUALIZATION_DIRECTORY_TREE.txt) (326 lines, 11 KB)
**Detailed directory structure with file-level documentation**

Contents:
- Complete directory tree with file annotations
- Description of each module's purpose and contents
- Function listings for each major module
- Chart types organized by category with class names
- Module dependencies and import hierarchy
- File metrics and line counts
- Clear visual hierarchy of the project structure

**Best for:** Navigating the codebase and finding specific files/functions.

### 3. [VISUALIZATION_QUICK_REFERENCE.md](VISUALIZATION_QUICK_REFERENCE.md) (279 lines, 8 KB)
**Quick start guide and API reference for common tasks**

Contents:
- Module structure at a glance
- Quick start examples (4 most common use cases)
- Available themes
- Chart types reference table
- High-level API function signatures
- ChartFactory methods
- Export formats supported
- Key classes overview
- Design patterns summary
- Main files reference
- Dependencies
- Complete workflow example
- Integration with PanelBox models

**Best for:** Getting started quickly and finding common API signatures.

---

## Module Location

```
/home/guhaase/projetos/panelbox/panelbox/visualization/
```

## Quick Navigation

### By Task

**I want to create a specific chart type:**
1. See [VISUALIZATION_QUICK_REFERENCE.md](VISUALIZATION_QUICK_REFERENCE.md) Chart Types table
2. Use `ChartFactory.create()` or convenience API function
3. Reference example code in [VISUALIZATION_SUMMARY.md](VISUALIZATION_SUMMARY.md) section 8

**I want to understand the architecture:**
1. Read [VISUALIZATION_SUMMARY.md](VISUALIZATION_SUMMARY.md) sections 1-6
2. See design patterns in section 6
3. Check directory tree in [VISUALIZATION_DIRECTORY_TREE.txt](VISUALIZATION_DIRECTORY_TREE.txt)

**I want to find a specific file or function:**
1. Use [VISUALIZATION_DIRECTORY_TREE.txt](VISUALIZATION_DIRECTORY_TREE.txt)
2. Search by module name or purpose

**I want to export charts:**
1. See [VISUALIZATION_QUICK_REFERENCE.md](VISUALIZATION_QUICK_REFERENCE.md) Export Formats
2. See [VISUALIZATION_SUMMARY.md](VISUALIZATION_SUMMARY.md) section 3 High-level APIs
3. Use `export_chart()`, `export_charts()`, or `export_charts_multiple_formats()`

**I want to customize appearance:**
1. See Themes section in [VISUALIZATION_QUICK_REFERENCE.md](VISUALIZATION_QUICK_REFERENCE.md)
2. Read Theme management in [VISUALIZATION_SUMMARY.md](VISUALIZATION_SUMMARY.md) section 3.D
3. Create custom Theme using dataclass

---

## Key Concepts

### Factory Pattern
- Entry point: `ChartFactory.create()`
- Creates any of 35+ chart types
- Handles theme resolution and configuration

### Registry Pattern
- Central management: `ChartRegistry`
- Decorator: `@register_chart('name')`
- Dynamic chart discovery

### Template Method Pattern
- Base class: `BaseChart`
- Subclasses: `PlotlyChartBase`, `MatplotlibChartBase`
- Consistent creation workflow

### Themes
- 3 pre-built: Professional, Academic, Presentation
- Customizable: Color scheme, fonts, layouts
- Applied automatically during chart creation

### Chart Types (35 Total)
- 7 Residual diagnostics
- 4 Model comparison
- 4 Distribution charts
- 2 Correlation charts
- 3 Time series charts
- 4 Panel-specific charts
- 5 Validation charts
- 4 Econometric test charts
- 2 Basic charts

---

## Code Structure Overview

```
panelbox/visualization/
├── CORE ARCHITECTURE (1,500+ lines)
│   ├── api.py              (1,386 lines) - High-level convenience APIs
│   ├── base.py             (854 lines)   - Abstract base classes
│   ├── factory.py          (241 lines)   - Factory implementation
│   ├── registry.py         (306 lines)   - Registry system
│   ├── themes.py           (402 lines)   - Theme definitions
│   └── exceptions.py       (340 lines)   - Custom exceptions
│
├── CHART IMPLEMENTATIONS (3,000+ lines)
│   ├── plotly/             - 35 interactive charts
│   ├── quantile/           - Quantile regression visualizations
│   ├── var_plots.py        - VAR model charts
│   └── spatial_plots.py    - Spatial model charts
│
└── SUPPORTING (500+ lines)
    ├── config/             - Configuration classes
    ├── transformers/       - Data transformation
    └── utils/              - Utility functions
```

---

## Main APIs at a Glance

### Convenience Functions (api.py)
```python
# Validation
create_validation_charts()

# Residuals
create_residual_diagnostics()

# Comparison
create_comparison_charts()

# Panel Analysis
create_panel_charts()
create_entity_effects_plot()
create_time_effects_plot()
create_between_within_plot()
create_panel_structure_plot()

# Econometric Tests
create_acf_pacf_plot()
create_unit_root_test_plot()
create_cointegration_heatmap()
create_cross_sectional_dependence_plot()

# Export
export_chart()
export_charts()
export_charts_multiple_formats()
```

### Factory (factory.py)
```python
ChartFactory.create(chart_type, data, theme, config)
ChartFactory.create_multiple(chart_specs, common_theme)
ChartFactory.list_available_charts()
ChartFactory.get_chart_info(chart_type)
```

### Base Classes (base.py)
```python
BaseChart              # Abstract base for all charts
PlotlyChartBase        # Interactive Plotly charts
MatplotlibChartBase    # Static Matplotlib charts
```

---

## Example Usage Patterns

### Pattern 1: Simple Chart Creation
```python
from panelbox.visualization import create_residual_diagnostics
charts = create_residual_diagnostics(results, theme='academic')
charts['qq_plot'].to_html()
```

### Pattern 2: Direct Factory Usage
```python
from panelbox.visualization import ChartFactory
chart = ChartFactory.create('residual_qq_plot', data=data, theme='professional')
html = chart.to_html()
```

### Pattern 3: Batch Export
```python
from panelbox.visualization import export_charts_multiple_formats
paths = export_charts_multiple_formats(charts, 'output/', formats=['png', 'pdf'])
```

### Pattern 4: Custom Theme
```python
from panelbox.visualization import Theme, ChartFactory
theme = Theme(name='custom', color_scheme=[...], font_config={...}, ...)
chart = ChartFactory.create('qq_plot', data=data, theme=theme)
```

---

## Dependencies

### Required
- plotly (interactive charts)
- matplotlib (static charts)
- numpy (numerical operations)
- scipy (statistical functions)

### Optional
- kaleido (image export: PNG, SVG, PDF, JPEG)
- seaborn (matplotlib styling)

---

## File Sizes Summary

| Document | Lines | Size |
|----------|-------|------|
| VISUALIZATION_SUMMARY.md | 630 | 20 KB |
| VISUALIZATION_DIRECTORY_TREE.txt | 326 | 11 KB |
| VISUALIZATION_QUICK_REFERENCE.md | 279 | 8 KB |
| **Total** | **1,235** | **39 KB** |

---

## Next Steps

1. **Quick Start**: Read [VISUALIZATION_QUICK_REFERENCE.md](VISUALIZATION_QUICK_REFERENCE.md)
2. **Deep Dive**: Read [VISUALIZATION_SUMMARY.md](VISUALIZATION_SUMMARY.md)
3. **Navigation**: Use [VISUALIZATION_DIRECTORY_TREE.txt](VISUALIZATION_DIRECTORY_TREE.txt)
4. **Start Coding**: Use examples from the documentation

---

## Document Generation Date

February 17, 2025

## Module Version

As of latest commit: **0.6.0**

---

*For the latest source code, see `/home/guhaase/projetos/panelbox/panelbox/visualization/`*
