# PanelBox Reports - Quick Start Guide

## 🚀 Installation

### Prerequisites

1. **Create virtual environment** (if not exists):
```bash
cd /path/to/panelbox
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies**:

**Option A: Automated install (recommended)**
```bash
./install_reports.sh
```

**Option B: Manual install**
```bash
source venv/bin/activate
pip install jinja2>=3.0.0
pip install matplotlib>=3.5.0  # Optional, for static charts
```

**Option C: Using requirements file**
```bash
source venv/bin/activate
pip install -r requirements_reports.txt
```

### Verify Installation

```bash
python examples/minimal_report_example.py
```

If you see "All exporters working correctly!", you're ready! ✅

## 📊 Running Examples

### Minimal Example (Tests exporters only)

```bash
python3 examples/minimal_report_example.py
```

**Output:**
- `output/test_reports/test.html`
- `output/test_reports/test.tex`
- `output/test_reports/test.md`

### Complete Example (Full validation report workflow)

```bash
python3 examples/simple_report_example.py
```

**Output:**
- `output/reports/validation_report.html` (82 KB, interactive with Plotly)
- `output/reports/validation_tests.tex` (LaTeX table)
- `output/reports/VALIDATION_REPORT.md` (Markdown for GitHub)

## 🎯 Basic Usage

### 1. Generate HTML Report

```python
from panelbox.report import ReportManager
from panelbox.report.validation_transformer import ValidationTransformer

# Transform validation data
transformer = ValidationTransformer(validation_report)
data = transformer.transform(include_charts=True)

# Generate HTML
report_mgr = ReportManager()
html = report_mgr.generate_validation_report(
    validation_data=data,
    interactive=True,
    title='My Validation Report'
)

# Save
report_mgr.save_report(html, 'report.html')
```

### 2. Export LaTeX Table

```python
from panelbox.report.exporters import LaTeXExporter

exporter = LaTeXExporter(table_style='booktabs')
latex = exporter.export_validation_tests(
    data['tests'],
    caption="Panel Data Validation Tests",
    label="tab:validation"
)
exporter.save(latex, 'table.tex')
```

### 3. Export Markdown

```python
from panelbox.report.exporters import MarkdownExporter

exporter = MarkdownExporter()
markdown = exporter.export_validation_report(
    data,
    title="Validation Report"
)
exporter.save(markdown, 'REPORT.md')
```

## 📁 Generated Files

### HTML Report Features
- ✅ Interactive Plotly charts
- ✅ Tab navigation
- ✅ Self-contained (all assets embedded)
- ✅ Responsive design
- ✅ Print-friendly
- ✅ ~80-300 KB file size

### LaTeX Table Features
- ✅ Booktabs style (publication-ready)
- ✅ Automatic character escaping
- ✅ Significance stars
- ✅ Professional formatting

### Markdown Report Features
- ✅ GitHub-flavored Markdown
- ✅ Table of contents
- ✅ Emoji indicators (✅/❌)
- ✅ Properly formatted tables

## 🔧 Dependencies

Required:
- `jinja2` - Template engine
- `python >= 3.9`

Optional:
- `matplotlib` - For static charts
- `plotly` - For interactive charts (CDN used by default)

## 📚 Documentation

Complete documentation: `panelbox/report/README.md`

## ⚠️ Current Status

**Phase 3: COMPLETE** ✅

Implemented:
- ✅ HTML Report Generation (interactive & static)
- ✅ LaTeX Export
- ✅ Markdown Export
- ✅ Validation Reports
- ✅ 3-layer CSS Architecture
- ✅ Asset Management
- ✅ Template System

Pending (Future phases):
- ⏳ Regression Reports (detailed)
- ⏳ GMM Reports
- ⏳ PDF Generation
- ⏳ Multi-model comparison

## 🐛 Known Issues

1. **Warning: CSS file not found: validation_report.css**
   - This is optional CSS for customization
   - Safe to ignore
   - Add custom CSS if needed using `CSSManager`

2. **Module not found errors**
   - Ensure `PYTHONPATH` is set, or
   - Install in development mode with `pip install -e .`

## 💡 Tips

### Reduce File Size

```python
# Enable minification
report_mgr = ReportManager(minify=True)
```

### High-DPI Charts

```python
from panelbox.report.renderers import StaticValidationRenderer

renderer = StaticValidationRenderer(dpi=300)  # High DPI
```

### Custom CSS

```python
report_mgr.css_manager.add_inline_css("""
.custom-class {
    color: #ff0000;
}
""")
```

### Multiple Reports

```python
from panelbox.report.exporters import HTMLExporter

exporter = HTMLExporter()
reports = {
    'Validation': validation_html,
    'Regression': regression_html
}
paths = exporter.export_with_index(reports, 'reports/')
```

## 🎉 Success!

If you see output like this, everything is working:

```
✓ HTML report generated: output/reports/validation_report.html
  - File size: 82.5 KB

✓ LaTeX table generated: output/reports/validation_tests.tex

✓ Markdown report generated: output/reports/VALIDATION_REPORT.md
```

Open the HTML file in your browser to see the interactive report!

## 🤝 Need Help?

- Check `panelbox/report/README.md` for detailed documentation
- Run examples to see working code
- Review test files in `tests/report/` for more examples
