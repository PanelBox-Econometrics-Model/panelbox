# PanelBox Report Generation System

Professional report generation for panel data analysis with support for HTML, LaTeX, and Markdown formats.

## 📋 Overview

The PanelBox report system provides comprehensive tools for generating publication-quality reports from your panel data analysis. It supports:

- **Interactive HTML reports** with Plotly visualizations
- **Static HTML reports** with Matplotlib charts (suitable for PDFs)
- **LaTeX tables** for academic papers
- **Markdown reports** for GitHub documentation

## 🚀 Quick Start

### Basic Usage

```python
from panelbox.report import ReportManager
from panelbox.report.validation_transformer import ValidationTransformer

# Initialize report manager
report_mgr = ReportManager()

# Transform your ValidationReport
transformer = ValidationTransformer(validation_report)
validation_data = transformer.transform(include_charts=True)

# Generate interactive HTML report
html = report_mgr.generate_validation_report(
    validation_data=validation_data,
    interactive=True,
    title='Panel Data Validation Report'
)

# Save report
report_mgr.save_report(html, 'validation_report.html')
```

## 📊 Report Types

### 1. Interactive HTML Reports

Generate interactive reports with Plotly charts:

```python
html = report_mgr.generate_validation_report(
    validation_data=validation_data,
    interactive=True,
    title='Interactive Validation Report',
    subtitle='Fixed Effects Model'
)
```

**Features:**
- Tab-based navigation
- Interactive Plotly charts (zoom, pan, hover)
- Self-contained HTML (all assets embedded)
- Responsive design
- Print-friendly CSS
- Export to CSV functionality

### 2. Static HTML Reports

Generate static reports with Matplotlib charts:

```python
from panelbox.report.renderers import StaticValidationRenderer

# Render static charts
renderer = StaticValidationRenderer(dpi=300)
charts = renderer.render_validation_charts(validation_data)

# Add charts to validation_data
validation_data['static_charts'] = charts

# Generate static report
html = report_mgr.generate_validation_report(
    validation_data=validation_data,
    interactive=False,
    title='Static Validation Report'
)
```

**Use cases:**
- PDF generation
- Printed reports
- Email attachments
- Low-bandwidth environments

### 3. LaTeX Tables

Export tables for academic papers:

```python
from panelbox.report.exporters import LaTeXExporter

exporter = LaTeXExporter(table_style='booktabs')

# Validation tests table
latex = exporter.export_validation_tests(
    tests=validation_data['tests'],
    caption="Panel Data Validation Test Results",
    label="tab:validation"
)

# Save
exporter.save(latex, 'validation_table.tex')

# With preamble (for standalone compilation)
exporter.save(latex, 'validation_table.tex', add_preamble=True)
```

**Supported table styles:**
- `booktabs` (recommended for publications)
- `standard` (basic LaTeX tables)
- `threeparttable` (tables with footnotes)

### 4. Markdown Reports

Export for GitHub and documentation:

```python
from panelbox.report.exporters import MarkdownExporter

exporter = MarkdownExporter(
    include_toc=True,
    github_flavor=True
)

markdown = exporter.export_validation_report(
    validation_data,
    title="Panel Data Validation Report"
)

exporter.save(markdown, 'VALIDATION_REPORT.md')
```

**Features:**
- GitHub-flavored Markdown
- Automatic table of contents
- Emoji indicators (✅/❌)
- Properly formatted tables

## 🎨 Customization

### CSS Customization

The report system uses a 3-layer CSS architecture:

```python
from panelbox.report import ReportManager

report_mgr = ReportManager()

# Add custom CSS
report_mgr.css_manager.add_custom_css('my-custom-styles.css')

# Or add inline CSS
report_mgr.css_manager.add_inline_css("""
.my-custom-class {
    color: #ff0000;
}
""")

# Generate report with custom styles
html = report_mgr.generate_report(
    report_type='validation',
    template='validation/interactive/index.html',
    context=validation_data,
    custom_css=['my-custom-styles.css']
)
```

### Template Customization

Create custom templates using Jinja2:

```python
# Use custom template directory
from pathlib import Path

report_mgr = ReportManager(
    template_dir=Path('my_templates'),
    asset_dir=Path('my_assets')
)

# Render custom template
html = report_mgr.template_manager.render_template(
    'my_custom_report.html',
    context={'data': my_data}
)
```

## 📦 Components

### ReportManager

Main orchestrator for report generation:

```python
from panelbox.report import ReportManager

report_mgr = ReportManager(
    template_dir=None,  # Use default templates
    asset_dir=None,     # Use default assets
    enable_cache=True,  # Enable template/asset caching
    minify=False        # Minify CSS/JS
)

# Generate reports
html = report_mgr.generate_validation_report(...)
html = report_mgr.generate_regression_report(...)
html = report_mgr.generate_gmm_report(...)

# Save reports
report_mgr.save_report(html, 'report.html', overwrite=True)

# Get info
info = report_mgr.get_info()

# Clear caches
report_mgr.clear_cache()
```

### ValidationTransformer

Transforms ValidationReport into template-ready data:

```python
from panelbox.report.validation_transformer import ValidationTransformer

transformer = ValidationTransformer(validation_report)

# Transform with charts
data = transformer.transform(include_charts=True)

# Access components
model_info = data['model_info']
tests = data['tests']
summary = data['summary']
recommendations = data['recommendations']
charts = data['charts']
```

### Exporters

#### HTMLExporter

```python
from panelbox.report.exporters import HTMLExporter

exporter = HTMLExporter(minify=False, pretty_print=False)

# Export single report
path = exporter.export(html, 'report.html', overwrite=True)

# Export multiple reports with index
reports = {
    'Validation Report': validation_html,
    'Regression Results': regression_html
}
paths = exporter.export_with_index(
    reports,
    output_dir='reports/',
    index_title='PanelBox Analysis Reports'
)

# Get file size
sizes = exporter.get_file_size(html)
print(f"Report size: {sizes['kb']:.1f} KB")
```

#### LaTeXExporter

```python
from panelbox.report.exporters import LaTeXExporter

exporter = LaTeXExporter(
    table_style='booktabs',
    float_format='.3f',
    escape_special_chars=True
)

# Export validation tests
latex = exporter.export_validation_tests(tests, caption="...", label="...")

# Export regression table
latex = exporter.export_regression_table(coefs, model_info, caption="...")

# Export summary statistics
latex = exporter.export_summary_stats(stats, caption="...")

# Save
exporter.save(latex, 'table.tex', add_preamble=True)
```

#### MarkdownExporter

```python
from panelbox.report.exporters import MarkdownExporter

exporter = MarkdownExporter(include_toc=True, github_flavor=True)

# Export full report
markdown = exporter.export_validation_report(validation_data, title="...")

# Export individual components
tests_md = exporter.export_validation_tests(tests)
regression_md = exporter.export_regression_table(coefs, model_info)
stats_md = exporter.export_summary_stats(stats)

# Save
exporter.save(markdown, 'REPORT.md')
```

### Renderers

#### StaticValidationRenderer

Generates static charts with Matplotlib:

```python
from panelbox.report.renderers import StaticValidationRenderer

renderer = StaticValidationRenderer(
    figure_size=(10, 6),
    dpi=300,  # High DPI for publications
    style='seaborn-v0_8-darkgrid'
)

# Render all charts
charts = renderer.render_validation_charts(validation_data)

# Charts are returned as base64-encoded PNG data URIs
# Use directly in HTML: <img src="{{charts.test_overview}}">

# Render individual charts
summary_chart = renderer.render_summary_chart(summary)
```

## 🔧 Advanced Usage

### Multiple Reports with Index

```python
from panelbox.report.exporters import HTMLExporter

exporter = HTMLExporter()

# Create multiple reports
reports = {}
reports['Validation Report'] = validation_html
reports['Fixed Effects Results'] = fe_html
reports['Random Effects Results'] = re_html
reports['GMM Results'] = gmm_html

# Export with index page
paths = exporter.export_with_index(
    reports,
    output_dir='analysis_reports/',
    index_title='Complete Panel Data Analysis',
    overwrite=True
)

# Access individual paths
index_path = paths['_index']
validation_path = paths['Validation Report']
```

### Custom CSS Layers

```python
from panelbox.report import CSSManager, CSSLayer

css_mgr = CSSManager()

# Add custom layer
css_mgr.add_layer(
    name='theme',
    files=['dark-theme.css'],
    priority=15  # Between base (0) and custom (20)
)

# Add CSS to existing layer
css_mgr.add_css_to_layer('custom', 'my-overrides.css')

# Compile
compiled_css = css_mgr.compile()

# Get layer info
layers = css_mgr.get_layer_info()

# Validate
missing = css_mgr.validate_layers()
```

### Asset Management

```python
from panelbox.report import AssetManager

asset_mgr = AssetManager(minify=True)

# Load CSS
css = asset_mgr.get_css('base_styles.css')

# Load JavaScript
js = asset_mgr.get_js('tab-navigation.js')

# Load and encode image
img_data_uri = asset_mgr.get_image_base64('images/logo.png')

# Collect multiple files
all_css = asset_mgr.collect_css([
    'base_styles.css',
    'report_components.css',
    'custom.css'
])

# List available assets
assets = asset_mgr.list_assets()
print(assets['css'])
print(assets['js'])
print(assets['images'])

# Clear cache
asset_mgr.clear_cache()
```

## 📝 Complete Example

See [`examples/report_generation_example.py`](../../examples/report_generation_example.py) for a complete end-to-end example demonstrating:

1. Creating sample panel data
2. Estimating a Fixed Effects model
3. Running validation tests
4. Generating reports in all formats (HTML, LaTeX, Markdown)
5. Saving and organizing output files

Run the example:

```bash
python examples/report_generation_example.py
```

## 🎯 Best Practices

### 1. Use Self-Contained HTML

Always enable asset embedding for portable reports:

```python
html = report_mgr.generate_validation_report(
    validation_data=validation_data,
    embed_assets=True  # Default
)
```

### 2. Cache When Possible

Enable caching for better performance:

```python
report_mgr = ReportManager(enable_cache=True)
```

### 3. Choose Appropriate DPI

- **Screen viewing:** 150 DPI (default)
- **Printing:** 300 DPI
- **Publications:** 600 DPI

```python
renderer = StaticValidationRenderer(dpi=300)
```

### 4. Organize Output Files

```python
from pathlib import Path

output_dir = Path('output/reports')
output_dir.mkdir(parents=True, exist_ok=True)

html_dir = output_dir / 'html'
latex_dir = output_dir / 'latex'
md_dir = output_dir / 'markdown'

for dir in [html_dir, latex_dir, md_dir]:
    dir.mkdir(exist_ok=True)
```

### 5. Version Your Reports

```python
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'validation_report_{timestamp}.html'

report_mgr.save_report(html, output_dir / filename)
```

## 🐛 Troubleshooting

### Templates Not Found

```python
# Check template directory
print(report_mgr.template_manager.template_dir)

# List available templates
templates = report_mgr.template_manager.list_templates()
print(templates)

# Check if template exists
exists = report_mgr.template_manager.template_exists('validation/interactive/index.html')
```

### Assets Not Loading

```python
# Check asset directory
print(report_mgr.asset_manager.asset_dir)

# List available assets
assets = report_mgr.asset_manager.list_assets()

# Validate CSS files
missing = report_mgr.css_manager.validate_layers()
if any(missing.values()):
    print(f"Missing CSS files: {missing}")
```

### Large File Sizes

```python
# Enable minification
report_mgr = ReportManager(minify=True)

# Check file size
from panelbox.report.exporters import HTMLExporter
exporter = HTMLExporter()
sizes = exporter.get_file_size(html)
print(f"Size: {sizes['mb']:.2f} MB")

# For very large reports, consider:
# 1. Using static charts instead of interactive
# 2. Splitting into multiple reports
# 3. Reducing chart resolution
```

## 📚 API Reference

See docstrings in each module for complete API documentation:

- `panelbox.report.report_manager`
- `panelbox.report.template_manager`
- `panelbox.report.asset_manager`
- `panelbox.report.css_manager`
- `panelbox.report.validation_transformer`
- `panelbox.report.exporters`
- `panelbox.report.renderers`

## 🤝 Contributing

To add new report types or customize existing templates:

1. Create template in `panelbox/templates/report_types/`
2. Create transformer in `panelbox/report/transformers/`
3. Add convenience method to `ReportManager`
4. Add tests in `tests/report/`
5. Update documentation

## 📄 License

Part of the PanelBox project. See main LICENSE file.
