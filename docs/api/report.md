---
title: "Report API"
description: "API reference for panelbox.report — ReportManager, HTML/LaTeX/Markdown exporters, templates, and CSS system"
---

# Report API Reference

!!! info "Module"
    **Import**: `from panelbox.report import ...`
    **Source**: `panelbox/report/`

## Overview

The report module generates publication-ready reports from model results and validation diagnostics. It supports three output formats (HTML, LaTeX, Markdown) with a template-based architecture and a 3-layer CSS theming system.

| Component | Class | Purpose |
|-----------|-------|---------|
| **ReportManager** | `ReportManager` | Central orchestrator for report generation |
| **TemplateManager** | `TemplateManager` | Jinja2 template loading and rendering |
| **AssetManager** | `AssetManager` | CSS, JS, and image asset management |
| **CSSManager** | `CSSManager` | 3-layer CSS system (Base, Theme, Component) |
| **Exporters** | `HTMLExporter`, `LaTeXExporter`, `MarkdownExporter` | Format-specific export |
| **Transformer** | `ValidationTransformer` | Convert validation data for templates |

---

## ReportManager

Central class for generating and saving reports.

```python
class ReportManager(
    template_dir: Path | None = None,
    asset_dir: Path | None = None,
    enable_cache: bool = True,
    minify: bool = False,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `template_dir` | `Path \| None` | `None` | Custom template directory (uses built-in if None) |
| `asset_dir` | `Path \| None` | `None` | Custom asset directory |
| `enable_cache` | `bool` | `True` | Cache compiled templates |
| `minify` | `bool` | `False` | Minify CSS/JS output |

**Report Generation Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `generate_report(report_type, template, context, ...)` | `str` | Generate generic report |
| `generate_validation_report(validation_data, interactive=True, title=None, subtitle=None)` | `str` | Validation report HTML |
| `generate_regression_report(regression_data, title=None, subtitle=None)` | `str` | Regression report HTML |
| `generate_gmm_report(gmm_data, title=None, subtitle=None)` | `str` | GMM report HTML |
| `generate_residual_report(residual_data, title=None, subtitle=None, interactive=True)` | `str` | Residual diagnostics HTML |
| `generate_comparison_report(comparison_data, title=None, subtitle=None, interactive=True)` | `str` | Model comparison HTML |

**Utility Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `save_report(html, output_path, overwrite=False)` | `Path` | Save HTML to file |
| `clear_cache()` | — | Clear template cache |
| `get_info()` | `dict` | Module info (template dir, asset dir, etc.) |

**Example:**

```python
from panelbox.report import ReportManager

manager = ReportManager(minify=True)

# Generate validation report
html = manager.generate_validation_report(
    validation_data=validation_dict,
    interactive=True,
    title="Model Diagnostics",
)
manager.save_report(html, "validation_report.html")
```

---

## TemplateManager

Manages Jinja2 templates for report rendering.

```python
class TemplateManager(
    template_dir: Path | None = None,
    enable_cache: bool = True,
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_template(template_path)` | `Template` | Load a Jinja2 template |
| `render_template(template_path, context)` | `str` | Render template with context |
| `render_string(template_string, context)` | `str` | Render a template string |
| `clear_cache()` | — | Clear template cache |
| `list_templates(pattern="*.html")` | `list` | List available templates |
| `template_exists(template_path)` | `bool` | Check if template exists |

**Built-in Jinja2 Filters:**

| Filter | Usage | Description |
|--------|-------|-------------|
| `number_format` | `{{ value \| number_format(3) }}` | Format number with N decimals |
| `pvalue_format` | `{{ pval \| pvalue_format }}` | Smart p-value formatting |
| `percentage` | `{{ value \| percentage(2) }}` | Format as percentage |
| `significance_stars` | `{{ pval \| significance_stars }}` | `***`, `**`, `*`, or empty |
| `round` | `{{ value \| round(2) }}` | Round to N decimals |

---

## AssetManager

Manages CSS, JavaScript, and image assets for reports.

```python
class AssetManager(
    asset_dir: Path | None = None,
    minify: bool = False,
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_css(css_path)` | `str` | Load CSS file content |
| `get_js(js_path)` | `str` | Load JS file content |
| `get_image_base64(image_path)` | `str` | Load image as base64 |
| `collect_css(css_files)` | `str` | Concatenate multiple CSS files |
| `collect_js(js_files)` | `str` | Concatenate multiple JS files |
| `embed_plotly(include_plotly=True)` | `str` | Embed Plotly.js library |
| `clear_cache()` | — | Clear asset cache |
| `list_assets(asset_type="all")` | `dict` | List available assets |

---

## CSSManager

3-layer CSS system for modular styling.

```python
class CSSManager(
    asset_manager: AssetManager | None = None,
    minify: bool = False,
)
```

**CSS Layers:**

| Layer | Priority | Description |
|-------|----------|-------------|
| `BASE` | 0 | Reset, typography, layout foundations |
| `THEME` | 1 | Theme-specific colors and fonts |
| `COMPONENT` | 2 | Component-specific styles |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `add_layer(name, files, priority)` | — | Add a CSS layer |
| `add_css_to_layer(layer_name, css_file)` | — | Add CSS file to existing layer |
| `remove_css_from_layer(layer_name, css_file)` | — | Remove CSS file from layer |
| `add_custom_css(css_file)` | — | Add custom CSS to component layer |
| `add_inline_css(css_content)` | — | Add inline CSS content |
| `compile(force=False)` | `str` | Compile all layers into single CSS |
| `compile_for_report_type(report_type)` | `str` | Compile CSS for specific report type |

### `CSSLayer`

```python
@dataclass
class CSSLayer:
    name: str
    files: list[str]
    priority: int = 0
```

---

## ValidationTransformer

Transforms validation data into the format expected by report templates.

```python
class ValidationTransformer:
    def transform(
        self,
        include_charts: bool = True,
        use_new_visualization: bool = True,
    ) -> dict
```

---

## Exporters

### `HTMLExporter`

Generate self-contained HTML files.

```python
class HTMLExporter(
    minify: bool = False,
    pretty_print: bool = False,
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `export(html_content, output_path, overwrite=False, add_metadata=True)` | `Path` | Export single report |
| `export_multiple(reports, output_dir, overwrite=False)` | `dict[str, Path]` | Export multiple reports |
| `export_with_index(reports, output_dir, index_title="PanelBox Reports", overwrite=False)` | — | Export with index page |

### `LaTeXExporter`

Generate publication-ready LaTeX tables.

```python
class LaTeXExporter(
    table_style: str = "booktabs",
    float_format: str = ".3f",
    escape_special_chars: bool = True,
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `export_validation_tests(tests, caption, label)` | `str` | LaTeX table of validation tests |

**Example:**

```python
from panelbox.report import LaTeXExporter

latex = LaTeXExporter(table_style="booktabs")
table = latex.export_validation_tests(
    tests=test_list,
    caption="Validation Test Results",
    label="tab:validation",
)
print(table)
```

### `MarkdownExporter`

Generate GitHub-flavored Markdown reports.

```python
class MarkdownExporter(
    include_toc: bool = True,
    github_flavor: bool = True,
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `export_validation_report(validation_data, title="Validation Report")` | `str` | Markdown validation report |

---

## Report Themes

PanelBox provides three professional themes for HTML reports:

| Theme | Primary Color | Use Case |
|-------|---------------|----------|
| **Professional** | Blue (#2563eb) | Corporate reports, general analysis |
| **Academic** | Gray (#4b5563) | Research papers, publications |
| **Presentation** | Purple (#7c3aed) | Presentations, slides |

```python
# Theme is passed via report generation context
html = manager.generate_validation_report(
    validation_data=data,
    title="Report",
)
# Or via PanelExperiment
experiment.save_master_report("report.html", theme="academic")
```

---

## See Also

- [Visualization API](visualization.md) — chart creation and themes
- [Experiment API](experiment.md) — integrated model fitting and report generation
- [Tutorials: Production](../tutorials/production.md) — report generation workflows
