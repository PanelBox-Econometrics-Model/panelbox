# Custom Themes Tutorial

📐 **Complete Guide to Creating and Using Custom Themes in PanelBox**

This tutorial shows you how to create, customize, and use themes for your visualizations.

---

## Table of Contents

1. [Built-in Themes](#built-in-themes)
2. [Creating Custom Themes](#creating-custom-themes)
3. [Loading Themes from Files](#loading-themes-from-files)
4. [Merging Themes](#merging-themes)
5. [Theme Anatomy](#theme-anatomy)
6. [Best Practices](#best-practices)
7. [Examples](#examples)

---

## Built-in Themes

PanelBox comes with 3 professional themes:

### 1. Professional Theme (Default)

**Best for**: Corporate reports, dashboards, general use

```python
from panelbox.visualization import create_validation_charts

charts = create_validation_charts(
    validation_report,
    theme='professional'  # Corporate blue
)
```

**Colors**: Blue, red, green, orange, purple
**Style**: Clean, modern, business-friendly

### 2. Academic Theme

**Best for**: Research papers, academic publications

```python
charts = create_validation_charts(
    validation_report,
    theme='academic'  # Grayscale-friendly
)
```

**Colors**: Colorblind-safe palette
**Style**: Journal-ready, print-friendly

### 3. Presentation Theme

**Best for**: Slides, presentations, projector displays

```python
charts = create_validation_charts(
    validation_report,
    theme='presentation'  # High-contrast
)
```

**Colors**: High-contrast, vibrant
**Style**: Large fonts, clear separation

---

## Creating Custom Themes

### Method 1: Create Theme Template

The easiest way to create a custom theme is to start from a template:

```python
from panelbox.visualization.utils import create_theme_template

# Create a template file
create_theme_template('my_theme.yaml')
```

This creates a YAML file with all available options:

```yaml
# my_theme.yaml
name: My Custom Theme
colors:
  - '#1f77b4'  # Blue
  - '#ff7f0e'  # Orange
  - '#2ca02c'  # Green
  - '#d62728'  # Red
  - '#9467bd'  # Purple
  - '#8c564b'  # Brown
  - '#e377c2'  # Pink
  - '#7f7f7f'  # Gray
  - '#bcbd22'  # Yellow-green
  - '#17becf'  # Cyan

# Typography
font_family: 'Inter, system-ui, -apple-system, sans-serif'
font_size: 12
title_font_size: 20
subtitle_font_size: 14
axis_label_font_size: 11
legend_font_size: 11
annotation_font_size: 10

# Colors
background_color: '#ffffff'
text_color: '#333333'
grid_color: '#e0e0e0'
success_color: '#10b981'
warning_color: '#f59e0b'
danger_color: '#ef4444'
info_color: '#3b82f6'
axis_line_color: '#666666'

# Styling
marker_size: 8
line_width: 2.0
border_width: 1
corner_radius: 4
spacing: 10

# Layout
height: 500
width: 800
margin:
  l: 80
  r: 40
  t: 80
  b: 60
```

### Method 2: Create Theme Programmatically

```python
from panelbox.visualization import Theme

# Create custom theme
dark_theme = Theme(
    name='Dark Mode',
    colors=[
        '#60a5fa',  # Light blue
        '#f87171',  # Light red
        '#34d399',  # Light green
        '#fbbf24',  # Light yellow
        '#a78bfa',  # Light purple
    ],
    font_family='Consolas, Monaco, monospace',
    font_size=11,
    background_color='#1a1a1a',
    text_color='#f5f5f5',
    grid_color='#404040',
    title_font_size=18,
    subtitle_font_size=13
)

# Use immediately
from panelbox.visualization import ChartFactory
chart = ChartFactory.create('bar_chart', data, theme=dark_theme)
```

---

## Loading Themes from Files

### YAML Format

```yaml
# corporate_theme.yaml
name: Corporate Blue
colors:
  - '#003f5c'
  - '#2f4b7c'
  - '#665191'
  - '#a05195'
  - '#d45087'
font_family: 'Helvetica Neue, Arial, sans-serif'
font_size: 12
background_color: '#fafafa'
text_color: '#1a1a1a'
```

```python
from panelbox.visualization.utils import load_theme

# Load theme
theme = load_theme('corporate_theme.yaml')

# Use with charts
from panelbox.visualization import create_residual_diagnostics
diagnostics = create_residual_diagnostics(results, theme=theme)
```

### JSON Format

```json
{
  "name": "Retro Theme",
  "colors": [
    "#ff6b6b",
    "#4ecdc4",
    "#45b7d1",
    "#f9ca24",
    "#6c5ce7"
  ],
  "font_family": "Courier New, monospace",
  "font_size": 11,
  "background_color": "#f7f1e3",
  "text_color": "#2c3e50"
}
```

```python
theme = load_theme('retro_theme.json')
```

---

## Merging Themes

Create variations of existing themes:

```python
from panelbox.visualization import PROFESSIONAL_THEME
from panelbox.visualization.utils import merge_themes

# Start from professional theme, customize specific fields
custom_theme = merge_themes(
    PROFESSIONAL_THEME,
    {
        'name': 'My Custom Professional',
        'background_color': '#f0f4f8',
        'colors': [
            '#0066cc',  # Custom blue
            '#cc0000',  # Custom red
            '#00994d',  # Custom green
        ] + PROFESSIONAL_THEME.colors[3:]  # Keep rest
    }
)

# Use merged theme
chart = ChartFactory.create('bar_chart', data, theme=custom_theme)
```

---

## Theme Anatomy

### Required Fields

These fields are **required** for every theme:

```python
{
    'name': 'Theme Name',           # str
    'colors': ['#hex', ...],        # list of 3+ hex colors
    'font_family': 'Font Name',     # str
    'font_size': 12                 # int
}
```

### Optional Fields

All other fields are optional and will use defaults if not provided:

**Colors**:
- `background_color`: Chart background
- `text_color`: Text color
- `grid_color`: Grid lines
- `success_color`: Success indicators (green)
- `warning_color`: Warning indicators (yellow)
- `danger_color`: Danger indicators (red)
- `info_color`: Info indicators (blue)
- `axis_line_color`: Axis lines

**Typography**:
- `title_font_size`: Chart titles
- `subtitle_font_size`: Subtitles
- `axis_label_font_size`: Axis labels
- `legend_font_size`: Legend text
- `annotation_font_size`: Annotations

**Styling**:
- `marker_size`: Scatter plot markers
- `line_width`: Line thickness
- `border_width`: Border thickness
- `corner_radius`: Corner rounding
- `spacing`: Internal spacing

**Layout**:
- `height`: Default chart height (px)
- `width`: Default chart width (px)
- `margin`: Dict with l, r, t, b keys

---

## Best Practices

### 1. Color Palette Selection

```python
# ✅ Good: 5-10 distinct colors
colors = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
]

# ❌ Bad: Too few colors
colors = ['#0000ff', '#ff0000']  # Only 2 colors

# ❌ Bad: Too similar colors
colors = ['#1f77b4', '#2f87c4', '#3f97d4']  # All shades of blue
```

### 2. Accessibility

```python
# ✅ Good: High contrast
background_color = '#ffffff'
text_color = '#1a1a1a'  # 15.8:1 contrast ratio

# ❌ Bad: Low contrast
background_color = '#e0e0e0'
text_color = '#cccccc'  # Poor contrast
```

Use colorblind-safe palettes:

```python
# Colorblind-safe palette (Paul Tol)
colors = [
    '#332288',  # Indigo
    '#88CCEE',  # Cyan
    '#44AA99',  # Teal
    '#117733',  # Green
    '#999933',  # Olive
    '#DDCC77',  # Sand
    '#CC6677',  # Rose
    '#882255',  # Wine
]
```

### 3. Font Selection

```python
# ✅ Good: System fonts with fallbacks
font_family = 'Inter, system-ui, -apple-system, sans-serif'

# ✅ Good: Web-safe fonts
font_family = 'Arial, Helvetica, sans-serif'

# ❌ Bad: No fallback
font_family = 'CustomFont'  # May not be available
```

### 4. Consistent Sizing

```python
# ✅ Good: Proportional sizes
title_font_size = 20
subtitle_font_size = 14      # 70% of title
axis_label_font_size = 11    # 55% of title
legend_font_size = 11        # Same as axis
annotation_font_size = 10    # Slightly smaller

# ❌ Bad: Inconsistent sizing
title_font_size = 20
subtitle_font_size = 19      # Too similar
axis_label_font_size = 8     # Too small
```

---

## Examples

### Example 1: Dark Mode Theme

```yaml
# dark_mode.yaml
name: Dark Mode
colors:
  - '#60a5fa'
  - '#f87171'
  - '#34d399'
  - '#fbbf24'
  - '#a78bfa'
  - '#f472b6'
  - '#fb923c'

font_family: 'SF Pro Display, -apple-system, sans-serif'
font_size: 12

background_color: '#1e1e1e'
text_color: '#e0e0e0'
grid_color: '#3a3a3a'
axis_line_color: '#666666'

success_color: '#10b981'
warning_color: '#f59e0b'
danger_color: '#ef4444'
info_color: '#3b82f6'

title_font_size: 18
marker_size: 6
line_width: 2.5
```

Usage:
```python
from panelbox.visualization.utils import load_theme

dark = load_theme('dark_mode.yaml')
chart = create_validation_charts(report, theme=dark)
```

### Example 2: Minimalist Theme

```python
from panelbox.visualization import Theme

minimalist = Theme(
    name='Minimalist',
    colors=[
        '#000000',  # Black
        '#666666',  # Dark gray
        '#999999',  # Medium gray
        '#cccccc',  # Light gray
    ],
    font_family='Helvetica Neue, Arial, sans-serif',
    font_size=11,
    background_color='#ffffff',
    text_color='#000000',
    grid_color='#e8e8e8',
    title_font_size=16,
    subtitle_font_size=12,
    marker_size=6,
    line_width=1.5,
    border_width=0,  # No borders
)
```

### Example 3: Vibrant Presentation Theme

```yaml
# vibrant.yaml
name: Vibrant Presentation
colors:
  - '#FF6B6B'  # Coral
  - '#4ECDC4'  # Turquoise
  - '#45B7D1'  # Sky blue
  - '#FFA07A'  # Light salmon
  - '#98D8C8'  # Mint
  - '#F7DC6F'  # Yellow
  - '#BB8FCE'  # Purple

font_family: 'Montserrat, Arial, sans-serif'
font_size: 14

background_color: '#ffffff'
text_color: '#2c3e50'
grid_color: '#ecf0f1'

title_font_size: 24
subtitle_font_size: 18
axis_label_font_size: 14
legend_font_size: 14

marker_size: 10
line_width: 3
height: 600
width: 1000
```

### Example 4: Academic Grayscale

```python
from panelbox.visualization import Theme

academic_gray = Theme(
    name='Academic Grayscale',
    colors=[
        '#1a1a1a',  # Very dark gray
        '#4d4d4d',  # Dark gray
        '#808080',  # Medium gray
        '#b3b3b3',  # Light gray
        '#333333',  # Darker gray
    ],
    font_family='Times New Roman, serif',
    font_size: 10,
    background_color='#ffffff',
    text_color='#000000',
    grid_color='#cccccc',
    title_font_size=12,
    subtitle_font_size=10,
    marker_size=4,
    line_width=1.0,
)
```

---

## Saving and Sharing Themes

### Save Theme to File

```python
from panelbox.visualization import PROFESSIONAL_THEME
from panelbox.visualization.utils import save_theme

# Save built-in theme
save_theme(PROFESSIONAL_THEME, 'professional.yaml')

# Save custom theme
save_theme(my_custom_theme, 'my_theme.json', format='json')
```

### Share Theme

```python
# Export without defaults (cleaner file)
save_theme(
    my_theme,
    'shared_theme.yaml',
    include_defaults=False
)
```

---

## Troubleshooting

### Theme Not Loading

```python
# ❌ Error: Missing required field
theme = {
    'name': 'Incomplete',
    'colors': ['#ff0000']  # Missing font_family, font_size
}

# ✅ Fix: Include all required fields
theme = {
    'name': 'Complete',
    'colors': ['#ff0000', '#00ff00', '#0000ff'],
    'font_family': 'Arial',
    'font_size': 12
}
```

### Colors Not Showing

```python
# ❌ Error: Invalid hex code
colors = ['red', 'blue']  # String names not supported

# ✅ Fix: Use hex codes
colors = ['#ff0000', '#0000ff']
```

### Theme Validation Error

```python
from panelbox.visualization.utils import load_theme

try:
    theme = load_theme('my_theme.yaml')
except ThemeLoadError as e:
    print(f"Error: {e}")
    print(f"Suggestion: {e.suggestion}")
```

---

## Advanced: Color Palette Tools

### Extract Colors from Built-in Themes

```python
from panelbox.visualization.utils import get_theme_colors

# Get color palette
prof_colors = get_theme_colors('professional')
print(f"Professional theme colors: {prof_colors}")

# Use in custom theme
my_colors = prof_colors[:3] + ['#custom1', '#custom2']
```

### List All Built-in Themes

```python
from panelbox.visualization.utils import list_builtin_themes

themes = list_builtin_themes()
print(f"Available themes: {themes}")
# Output: ['professional', 'academic', 'presentation']
```

---

## Summary Checklist

When creating a custom theme:

- [x] Include at least 3 colors
- [x] Use hex color codes (#RRGGBB)
- [x] Specify font_family and font_size
- [x] Ensure good contrast (background vs text)
- [x] Test with colorblind simulation
- [x] Use system fonts or web-safe fonts
- [x] Keep proportional font sizes
- [x] Save and share for reuse

---

**Ready to customize? Start with a template:**

```bash
python -c "from panelbox.visualization.utils import create_theme_template; create_theme_template('my_theme.yaml')"
```

Then edit `my_theme.yaml` and load it:

```python
from panelbox.visualization.utils import load_theme
theme = load_theme('my_theme.yaml')
```

**Happy theming! 🎨**
