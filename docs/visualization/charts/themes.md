---
title: "Themes & Customization"
description: "Built-in visual themes and custom theme creation for PanelBox charts"
---

# Themes & Customization

PanelBox provides 3 built-in themes and a simple API for creating custom themes. Every chart in the library supports theming, ensuring visual consistency across your reports and publications.

## Built-in Themes

| Theme | Style | Use Case | Plotly Template | Matplotlib Style |
|-------|-------|----------|-----------------|-----------------|
| `professional` | Clean, modern business | Reports, dashboards | `plotly_white` | `seaborn-v0_8-whitegrid` |
| `academic` | Minimal, publication-ready | Papers, dissertations | `simple_white` | `seaborn-v0_8-paper` |
| `presentation` | Bold, high-contrast | Slides, talks | `plotly_white` | `seaborn-v0_8-talk` |

### Professional Theme (Default)

Modern accessible palette with 10 colors. Clean sans-serif fonts, light gray plot area, unified hover mode.

```python
from panelbox.visualization import PROFESSIONAL_THEME

# Color scheme (10 colors):
# #1f77b4 (blue), #ff7f0e (orange), #2ca02c (green), #d62728 (red),
# #9467bd (purple), #8c564b (brown), #e377c2 (pink), #7f7f7f (gray),
# #bcbd22 (yellow-green), #17becf (cyan)

# Font: Arial, Helvetica, sans-serif, size 12
# Background: white paper, light gray (#F8F9FA) plot area
```

### Academic Theme

Grayscale-first palette for print-friendly output. Serif fonts, pure white background, box borders around plot area.

```python
from panelbox.visualization import ACADEMIC_THEME

# Color scheme (6 colors):
# #000000 (black), #404040 (dark gray), #808080 (medium gray),
# #A0A0A0 (light gray), #1f77b4 (blue accent), #2ca02c (green accent)

# Font: Times New Roman, Georgia, serif, size 11
# Background: pure white, outside ticks, mirrored axes
```

### Presentation Theme

Saturated, high-contrast palette for visibility on projectors. Large bold fonts, prominent gridlines.

```python
from panelbox.visualization import PRESENTATION_THEME

# Color scheme (8 colors):
# #E63946 (red), #1D3557 (navy), #2A9D8F (teal), #E9C46A (gold),
# #F4A261 (orange), #A8DADC (light blue), #457B9D (medium blue),
# #E76F51 (terracotta)

# Font: Helvetica, Arial, sans-serif, size 14
# Background: white paper, light gray plot area, thick borders
```

## Using Themes

### By Name (String)

Pass a theme name string to any chart creation function or the factory:

```python
from panelbox.visualization import create_residual_diagnostics, ChartFactory

# High-level API
charts = create_residual_diagnostics(results, theme="academic")

# Factory
chart = ChartFactory.create("bar_chart", data=data, theme="presentation")
```

### By Theme Object

Import and pass the theme object directly:

```python
from panelbox.visualization import ACADEMIC_THEME, ChartFactory

chart = ChartFactory.create("bar_chart", data=data, theme=ACADEMIC_THEME)
```

### Theme Utilities

```python
from panelbox.visualization.themes import get_theme, list_themes

# List all registered themes
themes = list_themes()
# ['academic', 'presentation', 'professional']

# Get a theme by name
theme = get_theme("academic")
print(theme.name)              # 'academic'
print(theme.plotly_template)   # 'simple_white'
```

## Theme Anatomy

The `Theme` class is a Python dataclass with these attributes:

```python
from panelbox.visualization import Theme

theme = Theme(
    name="my_theme",

    # Color palette — ordered hex colors for data series
    color_scheme=["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"],

    # Font configuration — passed directly to Plotly layout
    font_config={
        "family": "Arial, sans-serif",
        "size": 12,
        "color": "#333333",
    },

    # Layout configuration — Plotly layout defaults
    layout_config={
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#F5F5F5",
        "hovermode": "x unified",
        "showlegend": True,
        "margin": {"l": 60, "r": 40, "t": 80, "b": 60},
        "xaxis": {"gridcolor": "#E0E0E0", "zeroline": True},
        "yaxis": {"gridcolor": "#E0E0E0", "zeroline": True},
    },

    # Backend templates
    plotly_template="plotly_white",              # Plotly base template
    matplotlib_style="seaborn-v0_8-whitegrid",   # Matplotlib style

    # Semantic status colors
    success_color="#28a745",     # Green — tests passed, positive results
    warning_color="#ffc107",     # Amber — marginal significance, caution
    danger_color="#dc3545",      # Red — tests failed, rejection
    info_color="#17a2b8",        # Cyan — informational annotations
)
```

### Color Cycling

Themes automatically cycle through their color scheme when more data series exist than colors:

```python
theme = get_theme("professional")

theme.get_color(0)   # '#1f77b4' (blue)
theme.get_color(1)   # '#ff7f0e' (orange)
theme.get_color(10)  # '#1f77b4' (wraps back to blue)
```

### Semantic Colors

Status colors provide consistent meaning across all charts:

| Attribute | Purpose | Professional | Academic | Presentation |
|-----------|---------|-------------|----------|-------------|
| `success_color` | Tests passed, positive | `#28a745` | `#2ca02c` | `#2A9D8F` |
| `warning_color` | Marginal significance | `#ffc107` | `#ff7f0e` | `#E9C46A` |
| `danger_color` | Tests failed, rejection | `#dc3545` | `#d62728` | `#E63946` |
| `info_color` | Informational | `#17a2b8` | `#1f77b4` | `#457B9D` |

## Creating Custom Themes

### Register a Custom Theme

Create and register a theme so it can be used by name:

```python
from panelbox.visualization import Theme, register_theme

corporate_theme = Theme(
    name="corporate",
    color_scheme=[
        "#003f5c",   # Dark blue — primary
        "#58508d",   # Purple — secondary
        "#bc5090",   # Magenta — accent
        "#ff6361",   # Coral — highlight
        "#ffa600",   # Amber — warning
    ],
    font_config={
        "family": "Segoe UI, Roboto, sans-serif",
        "size": 13,
        "color": "#1a1a2e",
    },
    layout_config={
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#F8F9FA",
        "hovermode": "x unified",
        "showlegend": True,
        "margin": {"l": 70, "r": 50, "t": 90, "b": 70},
        "xaxis": {"gridcolor": "#E8E8E8", "gridwidth": 1},
        "yaxis": {"gridcolor": "#E8E8E8", "gridwidth": 1},
    },
    plotly_template="plotly_white",
    matplotlib_style="seaborn-v0_8-whitegrid",
    success_color="#28a745",
    warning_color="#ffc107",
    danger_color="#dc3545",
    info_color="#17a2b8",
)

# Register so you can use it by name
register_theme(corporate_theme)

# Now use it anywhere
charts = create_residual_diagnostics(results, theme="corporate")
```

### Use Without Registering

You can also pass a `Theme` object directly without registering:

```python
chart = ChartFactory.create("bar_chart", data=data, theme=corporate_theme)
```

### Export Theme Configuration

Inspect a theme's configuration:

```python
theme_dict = corporate_theme.to_dict()
# {
#     'name': 'corporate',
#     'color_scheme': ['#003f5c', ...],
#     'font_config': {...},
#     'layout_config': {...},
#     'plotly_template': 'plotly_white',
#     'matplotlib_style': 'seaborn-v0_8-whitegrid',
# }
```

## Color Scheme Utilities

The `config/color_schemes.py` module provides predefined palettes and utility functions.

### Predefined Palettes

```python
from panelbox.visualization.config.color_schemes import (
    COLORBLIND_FRIENDLY,     # 8 colorblind-safe colors
    MONOCHROME,              # Grayscale palette
    SEQUENTIAL_BLUE,         # 9-color blue sequential
    SEQUENTIAL_GREEN,        # 9-color green sequential
    SEQUENTIAL_RED,          # 9-color red sequential
    DIVERGING_RED_BLUE,      # 11-color diverging
    CATEGORICAL_VIBRANT,     # Vibrant categorical
    CATEGORICAL_MUTED,       # Muted categorical
)
```

### Significance Color Mapping

```python
from panelbox.visualization.config.color_schemes import (
    get_color_for_pvalue,
    SIGNIFICANCE_COLORS,
    STATUS_COLORS,
)

# Automatic color for p-values
color = get_color_for_pvalue(0.003, alpha=0.05)   # Returns danger/red color
color = get_color_for_pvalue(0.42, alpha=0.05)     # Returns success/green color

# Significance color dictionary
SIGNIFICANCE_COLORS["highly_significant"]   # p < 0.01
SIGNIFICANCE_COLORS["significant"]          # p < 0.05
SIGNIFICANCE_COLORS["marginally_significant"]  # p < 0.10
SIGNIFICANCE_COLORS["not_significant"]      # p >= 0.10
```

## Chart Configuration

The `ChartConfig` dataclass provides standardized defaults for all charts:

```python
from panelbox.visualization.config.chart_config import ChartConfig

config = ChartConfig(
    width=800,
    height=600,
    title="My Chart Title",
    title_font_size=16,
    xaxis_title="X Variable",
    yaxis_title="Y Variable",
    show_legend=True,
    legend_position="top right",
    responsive=True,
    hover_mode="x unified",
)

# Use with factory
chart = ChartFactory.create(
    "bar_chart",
    data=data,
    config=config.to_dict(),
    theme="professional",
)

# Merge configurations
base = ChartConfig(width=800, height=600)
override = ChartConfig(title="Custom Title", width=1000)
merged = base.merge(override)
# merged.width = 1000, merged.height = 600, merged.title = "Custom Title"
```

## Export Customization

Control image export quality and dimensions:

```python
from panelbox.visualization import export_chart, export_charts

# Single chart — high resolution for print
export_chart(
    chart,
    "figure1.png",
    width=1200,      # Pixels
    height=800,
    scale=2.0,       # 2x resolution (retina/print)
)

# SVG for publications (vector, infinitely scalable)
export_chart(chart, "figure1.svg", width=800, height=600)

# PDF for LaTeX inclusion
export_chart(chart, "figure1.pdf", width=800, height=600)

# Batch export with consistent sizing
export_charts(
    charts,
    output_dir="./figures",
    format="svg",
    prefix="fig_",
    width=800,
    height=600,
)
```

!!! tip "Publication Recommendations"
    - **Journals**: Use `academic` theme + SVG format at 800x600px
    - **Presentations**: Use `presentation` theme + PNG at 1200x800px with scale=2.0
    - **Dashboards**: Use `professional` theme + HTML with `include_plotlyjs='cdn'`
    - **Print**: Use `academic` theme + PDF at 800x600px

## Comparison with Other Software

| Feature | PanelBox | Stata | R |
|---------|----------|-------|---|
| Built-in themes | 3 + custom | `scheme` system | `ggtheme`, `theme_set()` |
| Theme registration | `register_theme()` | N/A | `theme_set()` |
| Semantic colors | Built-in | Manual | Manual |
| Colorblind palettes | `COLORBLIND_FRIENDLY` | Manual | `viridis`, `RColorBrewer` |
| Configuration dataclass | `ChartConfig` | N/A | `theme()` arguments |

## See Also

- [Visualization Overview](index.md) -- Full chart gallery and architecture
- [Diagnostics Plots](model-diagnostics.md) -- Apply themes to diagnostic charts
- [Comparison Plots](comparison.md) -- Themed model comparisons
- [Specialized Plots](specialized.md) -- Domain-specific themes (quantile, SFA)
