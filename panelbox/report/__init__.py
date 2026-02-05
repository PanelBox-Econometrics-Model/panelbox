"""
PanelBox Report Generation Module.

Provides comprehensive report generation capabilities for panel data analysis.

Main Components
---------------
- ReportManager: Main orchestrator for report generation
- TemplateManager: Jinja2 template management
- AssetManager: CSS, JS, and image asset management
- CSSManager: 3-layer CSS compilation system

Examples
--------
Generate a validation report:

>>> from panelbox.report import ReportManager
>>> report_mgr = ReportManager()
>>> html = report_mgr.generate_validation_report(
...     validation_data={'tests': [...], 'model_info': {...}},
...     title='Panel Validation Report'
... )
>>> report_mgr.save_report(html, 'validation_report.html')

Custom report generation:

>>> context = {
...     'report_title': 'Custom Analysis',
...     'data': {...}
... }
>>> html = report_mgr.generate_report(
...     report_type='custom',
...     template='custom/report.html',
...     context=context
... )
"""

from .asset_manager import AssetManager
from .css_manager import CSSLayer, CSSManager

# Exporters
from .exporters import HTMLExporter, LaTeXExporter, MarkdownExporter
from .report_manager import ReportManager
from .template_manager import TemplateManager
from .validation_transformer import ValidationTransformer

__all__ = [
    "ReportManager",
    "TemplateManager",
    "AssetManager",
    "CSSManager",
    "CSSLayer",
    "ValidationTransformer",
    "HTMLExporter",
    "LaTeXExporter",
    "MarkdownExporter",
]
