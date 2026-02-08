"""
Chart registry system.

This module implements the Registry pattern for chart management, allowing
decorator-based registration and centralized retrieval of chart classes.

The registry pattern decouples chart creation from chart usage, making it
easy to add new chart types without modifying existing code.
"""

from typing import Dict, List, Type

from .base import BaseChart


class ChartRegistry:
    """
    Central registry for chart types.

    Provides a centralized location to register and retrieve chart classes.
    Uses decorator-based registration for clean, declarative code.

    The registry is a class-level dictionary, shared across all instances.

    Examples
    --------
    Register a chart using the decorator:

    >>> from panelbox.visualization import register_chart, PlotlyChartBase
    >>>
    >>> @register_chart('my_custom_chart')
    ... class MyCustomChart(PlotlyChartBase):
    ...     def _create_figure(self, data, **kwargs):
    ...         # Implementation
    ...         pass

    Retrieve and use a registered chart:

    >>> chart_class = ChartRegistry.get('my_custom_chart')
    >>> chart = chart_class(theme='professional')
    >>> chart.create(data={'x': [1, 2, 3], 'y': [4, 5, 6]})

    List all registered charts:

    >>> all_charts = ChartRegistry.list_charts()
    >>> print(all_charts)
    ['my_custom_chart', 'validation_test_overview', ...]
    """

    # Class-level registry dictionary
    _registry: Dict[str, Type[BaseChart]] = {}

    @classmethod
    def register(cls, name: str, chart_class: Type[BaseChart]) -> None:
        """
        Register a chart class.

        Parameters
        ----------
        name : str
            Unique identifier for the chart type (e.g., 'qq_plot')
        chart_class : Type[BaseChart]
            Chart class to register (must inherit from BaseChart)

        Raises
        ------
        ValueError
            If chart is already registered or doesn't inherit from BaseChart
        TypeError
            If chart_class is not a class

        Examples
        --------
        >>> ChartRegistry.register('my_chart', MyChartClass)
        """
        # Validation
        if not isinstance(chart_class, type):
            raise TypeError(f"chart_class must be a class, got {type(chart_class)}")

        if not issubclass(chart_class, BaseChart):
            raise ValueError(
                f"Chart class must inherit from BaseChart, got {chart_class.__bases__}"
            )

        if name in cls._registry:
            existing_class = cls._registry[name]
            if existing_class is not chart_class:
                raise ValueError(
                    f"Chart '{name}' is already registered with class {existing_class}. "
                    f"Cannot register {chart_class}."
                )
            # Same class re-registered - this is okay (e.g., module reload)
            return

        # Register
        cls._registry[name] = chart_class

    @classmethod
    def get(cls, name: str) -> Type[BaseChart]:
        """
        Get a registered chart class.

        Parameters
        ----------
        name : str
            Chart type identifier

        Returns
        -------
        Type[BaseChart]
            Registered chart class

        Raises
        ------
        ValueError
            If chart type is not registered

        Examples
        --------
        >>> chart_class = ChartRegistry.get('qq_plot')
        >>> chart = chart_class()
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Chart type '{name}' is not registered. "
                f"Available charts: {available or 'none'}"
            )

        return cls._registry[name]

    @classmethod
    def list_charts(cls) -> List[str]:
        """
        List all registered chart types.

        Returns
        -------
        list of str
            Sorted list of chart type names

        Examples
        --------
        >>> charts = ChartRegistry.list_charts()
        >>> print(f"Available: {', '.join(charts)}")
        """
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a chart type is registered.

        Parameters
        ----------
        name : str
            Chart type to check

        Returns
        -------
        bool
            True if registered, False otherwise

        Examples
        --------
        >>> if ChartRegistry.is_registered('qq_plot'):
        ...     chart = ChartRegistry.get('qq_plot')
        """
        return name in cls._registry

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a chart type.

        Useful for testing or dynamic chart management.

        Parameters
        ----------
        name : str
            Chart type to unregister

        Raises
        ------
        ValueError
            If chart type is not registered

        Examples
        --------
        >>> ChartRegistry.unregister('my_test_chart')
        """
        if name not in cls._registry:
            raise ValueError(f"Chart type '{name}' is not registered, cannot unregister")

        del cls._registry[name]

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered charts.

        WARNING: This removes all registrations. Mainly useful for testing.

        Examples
        --------
        >>> # In test teardown
        >>> ChartRegistry.clear()
        """
        cls._registry.clear()

    @classmethod
    def get_chart_info(cls, name: str) -> Dict[str, str]:
        """
        Get information about a registered chart.

        Parameters
        ----------
        name : str
            Chart type identifier

        Returns
        -------
        dict
            Information about the chart (name, class, docstring)

        Raises
        ------
        ValueError
            If chart type is not registered

        Examples
        --------
        >>> info = ChartRegistry.get_chart_info('qq_plot')
        >>> print(info['description'])
        """
        chart_class = cls.get(name)

        return {
            "name": name,
            "class": chart_class.__name__,
            "module": chart_class.__module__,
            "description": (
                chart_class.__doc__.split("\n")[0] if chart_class.__doc__ else "No description"
            ),
        }


def register_chart(name: str):
    """
    Decorator for registering chart classes.

    This decorator provides a clean, declarative way to register charts
    at class definition time.

    Parameters
    ----------
    name : str
        Unique identifier for the chart type

    Returns
    -------
    callable
        Decorator function

    Examples
    --------
    Register a chart:

    >>> from panelbox.visualization import register_chart, PlotlyChartBase
    >>>
    >>> @register_chart('my_scatter_plot')
    ... class ScatterPlot(PlotlyChartBase):
    ...     '''Interactive scatter plot.'''
    ...
    ...     def _create_figure(self, data, **kwargs):
    ...         import plotly.graph_objects as go
    ...         fig = go.Figure()
    ...         fig.add_trace(go.Scatter(
    ...             x=data['x'],
    ...             y=data['y'],
    ...             mode='markers'
    ...         ))
    ...         return fig

    Use the registered chart:

    >>> from panelbox.visualization import ChartRegistry
    >>> ScatterPlot = ChartRegistry.get('my_scatter_plot')
    >>> chart = ScatterPlot()
    >>> chart.create(data={'x': [1, 2, 3], 'y': [4, 5, 6]})

    Or use the factory:

    >>> from panelbox.visualization import ChartFactory
    >>> chart = ChartFactory.create('my_scatter_plot', data={'x': [1, 2, 3], 'y': [4, 5, 6]})
    """

    def decorator(chart_class: Type[BaseChart]) -> Type[BaseChart]:
        """Register the chart class."""
        ChartRegistry.register(name, chart_class)
        # Add registry name as class attribute for introspection
        chart_class._registry_name = name
        return chart_class

    return decorator
