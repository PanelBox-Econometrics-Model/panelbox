"""
Chart factory for centralized chart creation.

This module implements the Factory pattern for creating charts with
consistent theming and configuration management.
"""

from typing import Any, Dict, Optional, Union

from .base import BaseChart
from .registry import ChartRegistry
from .themes import Theme, get_theme


class ChartFactory:
    """
    Factory for creating charts with configuration management.

    Provides a centralized interface for chart creation with automatic
    theme resolution and configuration handling.

    The factory pattern allows:
    - Centralized chart creation
    - Consistent theme application
    - Configuration management
    - Easy testing and mocking

    Examples
    --------
    Create a chart with default theme:

    >>> chart = ChartFactory.create(
    ...     chart_type='residual_qq_plot',
    ...     data={'residuals': [0.1, -0.2, 0.3, -0.1]}
    ... )

    Create with custom theme:

    >>> chart = ChartFactory.create(
    ...     chart_type='validation_test_overview',
    ...     data=test_data,
    ...     theme='academic'
    ... )

    Create with configuration:

    >>> chart = ChartFactory.create(
    ...     chart_type='residual_vs_fitted',
    ...     data={'fitted': fitted, 'residuals': residuals},
    ...     theme='presentation',
    ...     config={'title': 'Residual Diagnostics', 'width': 1000}
    ... )
    """

    @staticmethod
    def create(
        chart_type: str,
        data: Optional[Dict[str, Any]] = None,
        theme: Union[str, Theme, None] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> BaseChart:
        """
        Create a chart instance.

        This is the main factory method for creating charts. It handles:
        1. Chart class retrieval from registry
        2. Theme resolution
        3. Chart instantiation
        4. Optional data creation

        Parameters
        ----------
        chart_type : str
            Registered chart type name (e.g., 'qq_plot', 'test_overview')
        data : dict, optional
            Data to pass to chart.create(). If None, returns un-created chart
        theme : str or Theme, optional
            Theme name or Theme object. If None, uses PROFESSIONAL_THEME
        config : dict, optional
            Chart configuration options
        **kwargs
            Additional arguments passed to chart.create()

        Returns
        -------
        BaseChart
            Chart instance (created if data is provided, otherwise just instantiated)

        Raises
        ------
        ValueError
            If chart_type is not registered or theme is invalid

        Examples
        --------
        Create and immediately render:

        >>> chart = ChartFactory.create(
        ...     'validation_test_overview',
        ...     data={'categories': [...], 'passed': [...], 'failed': [...]},
        ...     theme='professional'
        ... )
        >>> html = chart.to_html()

        Create without data (for later use):

        >>> chart = ChartFactory.create('qq_plot', theme='academic')
        >>> # ... later ...
        >>> chart.create(data={'residuals': residuals})
        """
        # Step 1: Get chart class from registry
        chart_class = ChartRegistry.get(chart_type)

        # Step 2: Resolve theme
        if theme is None:
            resolved_theme = None
        elif isinstance(theme, Theme):
            resolved_theme = theme
        else:
            resolved_theme = get_theme(theme)

        # Step 3: Instantiate chart
        chart_instance = chart_class(theme=resolved_theme, config=config)

        # Step 4: Create chart with data (if provided)
        if data is not None:
            chart_instance.create(data, **kwargs)

        return chart_instance

    @staticmethod
    def create_multiple(
        chart_specs: list[Dict[str, Any]], common_theme: Union[str, Theme, None] = None
    ) -> Dict[str, BaseChart]:
        """
        Create multiple charts at once.

        Useful for generating a set of charts for a dashboard or report.

        Parameters
        ----------
        chart_specs : list of dict
            List of chart specifications. Each dict should have:
            - 'type': chart type name (required)
            - 'data': chart data (required)
            - 'name': key for returned dict (optional, defaults to type)
            - 'theme': chart-specific theme (optional)
            - 'config': chart config (optional)
        common_theme : str or Theme, optional
            Default theme for all charts (can be overridden per chart)

        Returns
        -------
        dict
            Dictionary mapping names to created charts

        Examples
        --------
        >>> specs = [
        ...     {
        ...         'type': 'qq_plot',
        ...         'name': 'qq',
        ...         'data': {'residuals': residuals}
        ...     },
        ...     {
        ...         'type': 'residual_vs_fitted',
        ...         'name': 'rvf',
        ...         'data': {'fitted': fitted, 'residuals': residuals},
        ...         'config': {'title': 'Residuals vs Fitted'}
        ...     }
        ... ]
        >>> charts = ChartFactory.create_multiple(specs, common_theme='academic')
        >>> qq_chart = charts['qq']
        >>> rvf_chart = charts['rvf']
        """
        charts = {}

        for spec in chart_specs:
            # Extract spec components
            chart_type = spec.get("type")
            if not chart_type:
                raise ValueError("Each chart spec must have a 'type' field")

            data = spec.get("data")
            if data is None:
                raise ValueError(f"Chart spec for type '{chart_type}' must have 'data'")

            name = spec.get("name", chart_type)
            theme = spec.get("theme", common_theme)
            config = spec.get("config")

            # Create chart
            chart = ChartFactory.create(chart_type=chart_type, data=data, theme=theme, config=config)

            charts[name] = chart

        return charts

    @staticmethod
    def list_available_charts() -> list[str]:
        """
        List all available chart types.

        Returns
        -------
        list of str
            Sorted list of registered chart types

        Examples
        --------
        >>> charts = ChartFactory.list_available_charts()
        >>> print(f"Available: {', '.join(charts)}")
        """
        return ChartRegistry.list_charts()

    @staticmethod
    def get_chart_info(chart_type: str) -> Dict[str, str]:
        """
        Get information about a chart type.

        Parameters
        ----------
        chart_type : str
            Chart type name

        Returns
        -------
        dict
            Chart information (name, class, description)

        Examples
        --------
        >>> info = ChartFactory.get_chart_info('qq_plot')
        >>> print(info['description'])
        Q-Q plot for normality testing
        """
        return ChartRegistry.get_chart_info(chart_type)
