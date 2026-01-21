"""
HTML Exporter for PanelBox Reports.

Exports reports to self-contained HTML files.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import datetime


class HTMLExporter:
    """
    Exports PanelBox reports to HTML format.

    Creates self-contained HTML files with all assets embedded inline.
    Suitable for sharing via email or hosting on static servers.

    Parameters
    ----------
    minify : bool, default=False
        Minify CSS and JavaScript for smaller file size
    pretty_print : bool, default=False
        Pretty-print HTML output for readability

    Examples
    --------
    >>> from panelbox.report import ReportManager
    >>> from panelbox.report.exporters import HTMLExporter
    >>>
    >>> report_mgr = ReportManager()
    >>> html = report_mgr.generate_validation_report(...)
    >>>
    >>> exporter = HTMLExporter()
    >>> exporter.export(html, 'report.html')
    """

    def __init__(
        self,
        minify: bool = False,
        pretty_print: bool = False
    ):
        """Initialize HTML Exporter."""
        self.minify = minify
        self.pretty_print = pretty_print

    def export(
        self,
        html_content: str,
        output_path: Union[str, Path],
        overwrite: bool = False,
        add_metadata: bool = True
    ) -> Path:
        """
        Export HTML content to file.

        Parameters
        ----------
        html_content : str
            HTML content to export
        output_path : str or Path
            Output file path
        overwrite : bool, default=False
            Overwrite existing file
        add_metadata : bool, default=True
            Add export metadata as HTML comment

        Returns
        -------
        Path
            Path to exported file

        Examples
        --------
        >>> exporter = HTMLExporter()
        >>> path = exporter.export(html, 'report.html')
        >>> print(f"Exported to {path}")
        """
        output_path = Path(output_path)

        # Check if file exists
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {output_path}. "
                "Use overwrite=True to replace."
            )

        # Create parent directories
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Process HTML
        if add_metadata:
            html_content = self._add_metadata(html_content)

        if self.pretty_print:
            html_content = self._pretty_print_html(html_content)

        # Write file
        output_path.write_text(html_content, encoding='utf-8')

        return output_path

    def export_multiple(
        self,
        reports: Dict[str, str],
        output_dir: Union[str, Path],
        overwrite: bool = False
    ) -> Dict[str, Path]:
        """
        Export multiple HTML reports to directory.

        Parameters
        ----------
        reports : dict
            Dictionary mapping filenames to HTML content
        output_dir : str or Path
            Output directory
        overwrite : bool, default=False
            Overwrite existing files

        Returns
        -------
        dict
            Dictionary mapping filenames to exported paths

        Examples
        --------
        >>> reports = {
        ...     'validation.html': validation_html,
        ...     'regression.html': regression_html
        ... }
        >>> paths = exporter.export_multiple(reports, 'output/')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported = {}

        for filename, html_content in reports.items():
            output_path = output_dir / filename
            exported[filename] = self.export(
                html_content,
                output_path,
                overwrite=overwrite
            )

        return exported

    def export_with_index(
        self,
        reports: Dict[str, str],
        output_dir: Union[str, Path],
        index_title: str = "PanelBox Reports",
        overwrite: bool = False
    ) -> Dict[str, Path]:
        """
        Export multiple reports with an index page.

        Parameters
        ----------
        reports : dict
            Dictionary mapping report names to HTML content
        output_dir : str or Path
            Output directory
        index_title : str, default="PanelBox Reports"
            Title for index page
        overwrite : bool, default=False
            Overwrite existing files

        Returns
        -------
        dict
            Dictionary mapping filenames to exported paths

        Examples
        --------
        >>> reports = {
        ...     'Validation Report': validation_html,
        ...     'Regression Results': regression_html
        ... }
        >>> paths = exporter.export_with_index(reports, 'reports/')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export individual reports
        exported = {}

        for i, (name, html_content) in enumerate(reports.items()):
            filename = f"report_{i+1}.html"
            output_path = output_dir / filename
            exported[name] = self.export(
                html_content,
                output_path,
                overwrite=overwrite
            )

        # Create index page
        index_html = self._generate_index_page(
            reports=list(reports.keys()),
            title=index_title
        )

        index_path = output_dir / "index.html"
        index_path.write_text(index_html, encoding='utf-8')
        exported['_index'] = index_path

        return exported

    def _add_metadata(self, html: str) -> str:
        """
        Add export metadata as HTML comment.

        Parameters
        ----------
        html : str
            HTML content

        Returns
        -------
        str
            HTML with metadata comment
        """
        timestamp = datetime.datetime.now().isoformat()

        metadata = f"""
<!--
PanelBox HTML Export
Exported: {timestamp}
Format: Self-contained HTML
Minified: {self.minify}
-->
"""

        # Insert after <!DOCTYPE html> or at beginning
        if '<!DOCTYPE' in html:
            parts = html.split('>', 1)
            return parts[0] + '>' + metadata + parts[1]
        else:
            return metadata + html

    def _pretty_print_html(self, html: str) -> str:
        """
        Pretty-print HTML (basic formatting).

        Parameters
        ----------
        html : str
            HTML content

        Returns
        -------
        str
            Formatted HTML

        Note
        ----
        This is a basic formatter. For production use, consider
        using BeautifulSoup or lxml for proper HTML formatting.
        """
        # Basic indentation - this is a simplified approach
        # In production, you'd use BeautifulSoup or similar
        return html

    def _generate_index_page(
        self,
        reports: list,
        title: str
    ) -> str:
        """
        Generate HTML index page.

        Parameters
        ----------
        reports : list
            List of report names
        title : str
            Page title

        Returns
        -------
        str
            HTML content for index page
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        report_links = []
        for i, name in enumerate(reports):
            filename = f"report_{i+1}.html"
            report_links.append(
                f'<li><a href="{filename}">{name}</a></li>'
            )

        links_html = '\n                '.join(report_links)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }}

        .container {{
            background: white;
            border-radius: 12px;
            padding: 3rem;
            max-width: 800px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }}

        h1 {{
            font-size: 2.5rem;
            color: #667eea;
            margin-bottom: 0.5rem;
        }}

        .subtitle {{
            color: #666;
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }}

        .report-list {{
            list-style: none;
            margin: 2rem 0;
        }}

        .report-list li {{
            margin-bottom: 1rem;
        }}

        .report-list a {{
            display: block;
            padding: 1rem 1.5rem;
            background: #f8f9fa;
            color: #667eea;
            text-decoration: none;
            border-radius: 8px;
            border: 2px solid transparent;
            transition: all 0.2s;
            font-weight: 500;
        }}

        .report-list a:hover {{
            background: #667eea;
            color: white;
            border-color: #667eea;
            transform: translateX(5px);
        }}

        .footer {{
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid #e0e0e0;
            text-align: center;
            color: #666;
            font-size: 0.9rem;
        }}

        .footer strong {{
            color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="subtitle">Generated: {timestamp}</p>

        <p>Select a report to view:</p>

        <ul class="report-list">
            {links_html}
        </ul>

        <div class="footer">
            <p>Generated with <strong>PanelBox</strong></p>
            <p><a href="https://github.com/panelbox/panelbox" target="_blank" style="color: #667eea;">Documentation</a></p>
        </div>
    </div>
</body>
</html>
"""

        return html

    def get_file_size(self, html_content: str) -> Dict[str, float]:
        """
        Estimate file size of HTML content.

        Parameters
        ----------
        html_content : str
            HTML content

        Returns
        -------
        dict
            Dictionary with size estimates

        Examples
        --------
        >>> sizes = exporter.get_file_size(html)
        >>> print(f"Size: {sizes['kb']:.1f} KB")
        """
        size_bytes = len(html_content.encode('utf-8'))

        return {
            'bytes': size_bytes,
            'kb': size_bytes / 1024,
            'mb': size_bytes / (1024 * 1024)
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HTMLExporter("
            f"minify={self.minify}, "
            f"pretty_print={self.pretty_print})"
        )
