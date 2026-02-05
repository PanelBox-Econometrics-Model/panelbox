"""
Template Manager for PanelBox Reports.

Manages loading, caching, and rendering of Jinja2 templates.
"""

import datetime
import os
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape


class TemplateManager:
    """
    Manages Jinja2 templates for report generation.

    Provides template loading, caching, and custom filters/functions.

    Parameters
    ----------
    template_dir : str or Path, optional
        Directory containing templates. If None, uses package templates.
    enable_cache : bool, default=True
        Enable template caching for better performance.

    Attributes
    ----------
    env : jinja2.Environment
        Jinja2 environment instance
    template_cache : dict
        Cache for loaded templates

    Examples
    --------
    >>> manager = TemplateManager()
    >>> template = manager.get_template('validation/interactive/index.html')
    >>> html = template.render(context)
    """

    def __init__(self, template_dir: Optional[Path] = None, enable_cache: bool = True):
        """Initialize Template Manager."""
        # Determine template directory
        if template_dir is None:
            # Use package templates
            package_dir = Path(__file__).parent.parent
            template_dir = package_dir / "templates"
        else:
            template_dir = Path(template_dir)

        if not template_dir.exists():
            raise ValueError(f"Template directory does not exist: {template_dir}")

        self.template_dir = template_dir
        self.enable_cache = enable_cache
        self.template_cache: Dict[str, Template] = {}

        # Create Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
            enable_async=False,
        )

        # Register custom filters
        self._register_filters()

        # Register custom globals
        self._register_globals()

    def _register_filters(self) -> None:
        """Register custom Jinja2 filters."""
        self.env.filters["number_format"] = self._filter_number_format
        self.env.filters["pvalue_format"] = self._filter_pvalue_format
        self.env.filters["percentage"] = self._filter_percentage
        self.env.filters["significance_stars"] = self._filter_significance_stars
        self.env.filters["round"] = self._filter_round

    def _register_globals(self) -> None:
        """Register custom Jinja2 global functions."""
        self.env.globals["now"] = datetime.datetime.now
        self.env.globals["range"] = range
        self.env.globals["len"] = len
        self.env.globals["enumerate"] = enumerate
        self.env.globals["zip"] = zip

    def get_template(self, template_path: str) -> Template:
        """
        Load a template by path.

        Parameters
        ----------
        template_path : str
            Relative path to template from template directory.
            Example: 'validation/interactive/index.html'

        Returns
        -------
        Template
            Loaded Jinja2 template

        Examples
        --------
        >>> template = manager.get_template('validation/interactive/index.html')
        >>> html = template.render({'title': 'My Report'})
        """
        # Check cache first
        if self.enable_cache and template_path in self.template_cache:
            return self.template_cache[template_path]

        # Load template
        template = self.env.get_template(template_path)

        # Cache if enabled
        if self.enable_cache:
            self.template_cache[template_path] = template

        return template

    def render_template(self, template_path: str, context: Dict[str, Any]) -> str:
        """
        Load and render a template with context.

        Parameters
        ----------
        template_path : str
            Relative path to template
        context : dict
            Template context variables

        Returns
        -------
        str
            Rendered HTML

        Examples
        --------
        >>> html = manager.render_template(
        ...     'validation/interactive/index.html',
        ...     {'title': 'Report', 'data': {...}}
        ... )
        """
        template = self.get_template(template_path)
        return template.render(**context)

    def render_string(self, template_string: str, context: Dict[str, Any]) -> str:
        """
        Render a template from string.

        Parameters
        ----------
        template_string : str
            Template content as string
        context : dict
            Template context variables

        Returns
        -------
        str
            Rendered HTML

        Examples
        --------
        >>> html = manager.render_string(
        ...     '<h1>{{ title }}</h1>',
        ...     {'title': 'Hello'}
        ... )
        """
        template = self.env.from_string(template_string)
        return template.render(**context)

    def clear_cache(self) -> None:
        """Clear template cache."""
        self.template_cache.clear()

    # Custom Filters

    @staticmethod
    def _filter_number_format(value, decimals: int = 3) -> str:
        """
        Format a number with specified decimals.

        Examples
        --------
        {{ value|number_format }}        -> "123.456"
        {{ value|number_format(2) }}     -> "123.46"
        """
        if value is None or (isinstance(value, float) and value != value):  # NaN check
            return "N/A"
        try:
            return f"{float(value):,.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def _filter_pvalue_format(value) -> str:
        """
        Format a p-value with scientific notation if small.

        Examples
        --------
        {{ 0.0432|pvalue_format }}       -> "0.0432"
        {{ 0.00001|pvalue_format }}      -> "1.00e-05"
        """
        if value is None or (isinstance(value, float) and value != value):
            return "N/A"
        try:
            value = float(value)
            if value < 0.001:
                return f"{value:.2e}"
            return f"{value:.4f}"
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def _filter_percentage(value, decimals: int = 2) -> str:
        """
        Format a value as percentage.

        Examples
        --------
        {{ 0.1234|percentage }}          -> "12.34%"
        {{ 0.1234|percentage(1) }}       -> "12.3%"
        """
        if value is None or (isinstance(value, float) and value != value):
            return "N/A"
        try:
            return f"{float(value) * 100:.{decimals}f}%"
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def _filter_significance_stars(pvalue) -> str:
        """
        Add significance stars based on p-value.

        Examples
        --------
        {{ 0.001|significance_stars }}   -> "***"
        {{ 0.02|significance_stars }}    -> "**"
        {{ 0.04|significance_stars }}    -> "*"
        {{ 0.08|significance_stars }}    -> "."
        {{ 0.15|significance_stars }}    -> ""
        """
        try:
            pvalue = float(pvalue)
            if pvalue < 0.001:
                return "***"
            elif pvalue < 0.01:
                return "**"
            elif pvalue < 0.05:
                return "*"
            elif pvalue < 0.1:
                return "."
            return ""
        except (ValueError, TypeError):
            return ""

    @staticmethod
    def _filter_round(value, decimals: int = 0) -> float:
        """
        Round a number to specified decimals.

        Examples
        --------
        {{ 3.14159|round(2) }}           -> 3.14
        """
        try:
            return round(float(value), decimals)
        except (ValueError, TypeError):
            return value

    def list_templates(self, pattern: str = "*.html") -> list:
        """
        List available templates matching pattern.

        Parameters
        ----------
        pattern : str, default="*.html"
            Glob pattern to match templates

        Returns
        -------
        list
            List of template paths relative to template directory

        Examples
        --------
        >>> manager.list_templates("validation/*.html")
        ['validation/interactive/index.html', ...]
        """
        templates = []
        for path in self.template_dir.rglob(pattern):
            rel_path = path.relative_to(self.template_dir)
            templates.append(str(rel_path))
        return sorted(templates)

    def template_exists(self, template_path: str) -> bool:
        """
        Check if a template exists.

        Parameters
        ----------
        template_path : str
            Relative path to template

        Returns
        -------
        bool
            True if template exists

        Examples
        --------
        >>> manager.template_exists('validation/interactive/index.html')
        True
        """
        full_path = self.template_dir / template_path
        return full_path.exists()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TemplateManager(template_dir={self.template_dir}, "
            f"cache_enabled={self.enable_cache}, "
            f"cached_templates={len(self.template_cache)})"
        )
