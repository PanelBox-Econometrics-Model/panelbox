"""
Base classes for all chart types.

This module provides the abstract base classes that all charts must inherit from.
It implements the Template Method pattern to ensure consistent chart creation.
"""

from __future__ import annotations

import base64
import io
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .themes import Theme

import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class BaseChart(ABC):
    """
    Abstract base class for all chart types.

    This class implements the Template Method pattern, defining the skeleton
    of chart creation while allowing subclasses to provide specific implementations.

    All charts must inherit from this class and implement the abstract methods.

    Parameters
    ----------
    theme : Theme, optional
        Visual theme configuration. If None, uses PROFESSIONAL_THEME.
    config : Dict, optional
        Chart-specific configuration options.

    Attributes
    ----------
    theme : Theme
        The visual theme being used
    config : Dict
        Chart configuration options
    figure : Any
        The underlying figure object (Plotly or Matplotlib)

    Methods
    -------
    create(data, **kwargs)
        Main method to create the chart (Template Method)
    to_json()
        Export chart as JSON string
    to_html()
        Export chart as standalone HTML
    to_dict()
        Export chart as dictionary

    Notes
    -----
    The create() method follows the Template Method pattern:
    1. Validate input data
    2. Preprocess data
    3. Create figure (subclass-specific)
    4. Apply theme
    5. Finalize and return

    Examples
    --------
    Subclasses must implement abstract methods:

    >>> class MyChart(BaseChart):
    ...     def _create_figure(self, data, **kwargs):
    ...         # Create chart-specific figure
    ...         return figure
    ...
    ...     def to_json(self):
    ...         # Convert to JSON
    ...         return json.dumps(self.figure)
    ...
    ...     def to_html(self):
    ...         # Convert to HTML
    ...         return f"<div>{self.figure}</div>"
    """

    def __init__(self, theme: Optional[Theme] = None, config: Optional[Dict] = None):
        """
        Initialize base chart.

        Parameters
        ----------
        theme : Theme, optional
            Visual theme. If None, will use PROFESSIONAL_THEME
        config : Dict, optional
            Chart configuration options
        """
        # Avoid circular import
        if theme is None:
            from .themes import PROFESSIONAL_THEME

            theme = PROFESSIONAL_THEME

        self.theme = theme
        self.config = config or {}
        self.figure = None
        self._data = None

    def create(self, data: Dict[str, Any], **kwargs) -> BaseChart:
        """
        Create chart from data (Template Method).

        This method orchestrates the chart creation process:
        1. Validate data
        2. Preprocess data
        3. Create figure
        4. Apply theme
        5. Finalize

        Parameters
        ----------
        data : dict
            Input data for the chart. Structure depends on chart type.
        **kwargs
            Additional chart-specific options

        Returns
        -------
        BaseChart
            Self, for method chaining

        Raises
        ------
        ValueError
            If data validation fails
        """
        # Store data
        self._data = data

        # Step 1: Validate data
        self._validate_data(data)

        # Step 2: Preprocess data
        processed_data = self._preprocess_data(data)

        # Step 3: Create figure (subclass-specific)
        self.figure = self._create_figure(processed_data, **kwargs)

        # Step 4: Apply theme
        self.figure = self._apply_theme(self.figure)

        # Step 5: Finalize
        self._finalize()

        return self

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """
        Validate input data.

        Override this method to add custom validation logic.

        Parameters
        ----------
        data : dict
            Input data to validate

        Raises
        ------
        ValueError
            If validation fails
        """
        if not isinstance(data, dict):
            raise ValueError(f"Data must be a dictionary, got {type(data)}")

    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess data before chart creation.

        Override this method to add preprocessing logic (e.g., cleaning,
        transforming, computing derived values).

        Parameters
        ----------
        data : dict
            Raw input data

        Returns
        -------
        dict
            Processed data ready for chart creation
        """
        # Default: return data as-is
        return data

    @abstractmethod
    def _create_figure(self, data: Dict[str, Any], **kwargs) -> Any:
        """
        Create the chart figure.

        This is the core method that subclasses must implement.
        It should create and return the underlying figure object
        (Plotly Figure, Matplotlib Figure, etc.).

        Parameters
        ----------
        data : dict
            Preprocessed data
        **kwargs
            Chart-specific options

        Returns
        -------
        Any
            Chart figure object
        """
        pass

    def _apply_theme(self, figure: Any) -> Any:
        """
        Apply theme to figure.

        Override this method to customize how themes are applied.

        Parameters
        ----------
        figure : Any
            Figure object to theme

        Returns
        -------
        Any
            Themed figure
        """
        # Default: return figure as-is
        # Subclasses should implement theme application
        return figure

    def _finalize(self) -> None:
        """
        Finalize chart creation.

        Override this method for any cleanup or final adjustments.
        """
        pass

    @abstractmethod
    def to_json(self) -> str:
        """
        Export chart as JSON string.

        Returns
        -------
        str
            JSON representation of the chart
        """
        pass

    @abstractmethod
    def to_html(self, **kwargs) -> str:
        """
        Export chart as standalone HTML.

        Parameters
        ----------
        **kwargs
            Export options (e.g., include_plotlyjs)

        Returns
        -------
        str
            HTML representation of the chart
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Export chart as dictionary.

        Returns
        -------
        dict
            Dictionary representation of the chart
        """
        # Default implementation
        return {"type": self.__class__.__name__, "config": self.config}

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(theme={self.theme.name})"


class PlotlyChartBase(BaseChart):
    """
    Base class for Plotly charts.

    Provides common Plotly functionality including:
    - JSON serialization with numpy type handling
    - HTML generation
    - Theme application using Plotly templates
    - Configuration management

    Attributes
    ----------
    figure : plotly.graph_objects.Figure
        Plotly figure object

    Examples
    --------
    >>> class MyPlotlyChart(PlotlyChartBase):
    ...     def _create_figure(self, data, **kwargs):
    ...         fig = go.Figure()
    ...         fig.add_trace(go.Scatter(x=data['x'], y=data['y']))
    ...         return fig
    """

    def __init__(self, theme: Optional[Theme] = None, config: Optional[Dict] = None):
        """Initialize Plotly chart."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for PlotlyChartBase. Install with: pip install plotly")

        super().__init__(theme, config)

    def _apply_theme(self, figure: go.Figure) -> go.Figure:
        """
        Apply Plotly theme to figure.

        Parameters
        ----------
        figure : plotly.graph_objects.Figure
            Figure to theme

        Returns
        -------
        plotly.graph_objects.Figure
            Themed figure
        """
        # Get base layout from theme
        base_layout = self._create_base_layout()

        # Update figure layout
        figure.update_layout(**base_layout)

        # Update color scheme
        if hasattr(figure, "data") and self.theme.color_scheme:
            for i, trace in enumerate(figure.data):
                # Only apply to traces with markers (not Heatmap, etc.)
                if hasattr(trace, 'marker') and hasattr(trace.marker, 'color'):
                    # Check if color is empty/None (handle both scalar and array cases)
                    color = trace.marker.color
                    is_empty = (color is None or
                              (isinstance(color, str) and not color) or
                              (hasattr(color, '__len__') and len(color) == 0))

                    if is_empty:
                        color_idx = i % len(self.theme.color_scheme)
                        trace.marker.color = self.theme.color_scheme[color_idx]

        return figure

    def _create_base_layout(self) -> Dict:
        """
        Create base Plotly layout from theme.

        Returns
        -------
        dict
            Plotly layout configuration
        """
        layout = {
            "template": self.theme.plotly_template,
            "font": self.theme.font_config,
            "colorway": self.theme.color_scheme,
            **self.theme.layout_config,
        }

        # Add config-specific overrides
        if "title" in self.config:
            layout["title"] = self.config["title"]
        if "width" in self.config:
            layout["width"] = self.config["width"]
        if "height" in self.config:
            layout["height"] = self.config["height"]

        return layout

    def to_json(self) -> str:
        """
        Convert Plotly figure to JSON.

        Handles numpy types and ensures proper serialization.

        Returns
        -------
        str
            JSON string representation
        """
        if self.figure is None:
            raise ValueError("Chart has not been created yet. Call create() first.")

        # Convert to dict and handle numpy types
        fig_dict = self.figure.to_dict()
        return json.dumps(fig_dict, cls=NumpyEncoder)

    def to_html(
        self, include_plotlyjs: str = "cdn", config: Optional[Dict] = None, **kwargs
    ) -> str:
        """
        Convert to standalone HTML.

        Parameters
        ----------
        include_plotlyjs : str, default='cdn'
            How to include plotly.js ('cdn', True, False, or path)
        config : dict, optional
            Plotly configuration options
        **kwargs
            Additional options passed to plotly.io.to_html

        Returns
        -------
        str
            HTML string
        """
        if self.figure is None:
            raise ValueError("Chart has not been created yet. Call create() first.")

        # Default config
        plotly_config = config or {
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        }

        return pio.to_html(self.figure, include_plotlyjs=include_plotlyjs, config=plotly_config, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export as dictionary.

        Returns
        -------
        dict
            Dictionary with chart metadata and figure
        """
        base_dict = super().to_dict()
        if self.figure is not None:
            base_dict["figure"] = self.figure.to_dict()
        return base_dict

    def to_image(
        self,
        format: str = "png",
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: float = 1.0,
        **kwargs
    ) -> bytes:
        """
        Export chart as image bytes.

        Requires kaleido to be installed: pip install kaleido

        Parameters
        ----------
        format : str, default='png'
            Image format: 'png', 'svg', 'jpeg', 'webp'
        width : int, optional
            Image width in pixels. If None, uses figure width.
        height : int, optional
            Image height in pixels. If None, uses figure height.
        scale : float, default=1.0
            Scale factor for image resolution (e.g., 2.0 for retina displays)
        **kwargs
            Additional options passed to kaleido

        Returns
        -------
        bytes
            Image data as bytes

        Raises
        ------
        ValueError
            If chart hasn't been created yet
        ImportError
            If kaleido is not installed

        Examples
        --------
        >>> chart = MyChart()
        >>> chart.create(data)
        >>> # Export as PNG
        >>> png_bytes = chart.to_image('png', width=800, height=600)
        >>> with open('chart.png', 'wb') as f:
        ...     f.write(png_bytes)
        >>> # Export as high-res PNG for retina displays
        >>> png_bytes = chart.to_image('png', scale=2.0)
        >>> # Export as SVG
        >>> svg_bytes = chart.to_image('svg')
        """
        if self.figure is None:
            raise ValueError("Chart has not been created yet. Call create() first.")

        try:
            image_bytes = pio.to_image(
                self.figure,
                format=format,
                width=width,
                height=height,
                scale=scale,
                **kwargs
            )
            return image_bytes
        except ValueError as e:
            if "kaleido" in str(e).lower():
                raise ImportError(
                    "kaleido is required for image export. "
                    "Install with: pip install kaleido"
                ) from e
            raise

    def save_image(
        self,
        file_path: str,
        format: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: float = 1.0,
        **kwargs
    ) -> None:
        """
        Save chart as image file.

        Requires kaleido to be installed: pip install kaleido

        Parameters
        ----------
        file_path : str
            Output file path (e.g., 'chart.png', 'chart.svg')
        format : str, optional
            Image format. If None, inferred from file_path extension.
            Options: 'png', 'svg', 'jpeg', 'pdf', 'webp'
        width : int, optional
            Image width in pixels. If None, uses figure width.
        height : int, optional
            Image height in pixels. If None, uses figure height.
        scale : float, default=1.0
            Scale factor for image resolution (e.g., 2.0 for retina displays)
        **kwargs
            Additional options passed to kaleido

        Raises
        ------
        ValueError
            If chart hasn't been created yet or invalid format
        ImportError
            If kaleido is not installed

        Examples
        --------
        >>> chart = MyChart()
        >>> chart.create(data)
        >>> # Save as PNG
        >>> chart.save_image('output/chart.png')
        >>> # Save as high-res PNG
        >>> chart.save_image('output/chart_2x.png', scale=2.0)
        >>> # Save as SVG with custom size
        >>> chart.save_image('output/chart.svg', width=1200, height=800)
        >>> # Save as PDF
        >>> chart.save_image('output/chart.pdf')
        """
        if self.figure is None:
            raise ValueError("Chart has not been created yet. Call create() first.")

        # Infer format from file extension if not provided
        if format is None:
            import os
            _, ext = os.path.splitext(file_path)
            format = ext.lstrip('.').lower()
            if not format:
                raise ValueError(
                    "Cannot infer format from file_path. "
                    "Please provide format parameter or use file extension."
                )

        # Validate format
        valid_formats = ['png', 'svg', 'jpeg', 'jpg', 'pdf', 'webp']
        if format not in valid_formats:
            raise ValueError(
                f"Invalid format '{format}'. "
                f"Must be one of: {', '.join(valid_formats)}"
            )

        # Get image bytes
        image_bytes = self.to_image(
            format=format,
            width=width,
            height=height,
            scale=scale,
            **kwargs
        )

        # Write to file
        from pathlib import Path
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(image_bytes)

    def to_png(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: float = 1.0
    ) -> bytes:
        """
        Export chart as PNG image bytes.

        Convenience method for PNG export.

        Parameters
        ----------
        width : int, optional
            Image width in pixels
        height : int, optional
            Image height in pixels
        scale : float, default=1.0
            Scale factor for resolution

        Returns
        -------
        bytes
            PNG image data

        Examples
        --------
        >>> chart.create(data)
        >>> png_bytes = chart.to_png(width=800, height=600)
        >>> with open('chart.png', 'wb') as f:
        ...     f.write(png_bytes)
        """
        return self.to_image(format='png', width=width, height=height, scale=scale)

    def to_svg(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> bytes:
        """
        Export chart as SVG image bytes.

        Convenience method for SVG export.

        Parameters
        ----------
        width : int, optional
            Image width in pixels
        height : int, optional
            Image height in pixels

        Returns
        -------
        bytes
            SVG image data

        Examples
        --------
        >>> chart.create(data)
        >>> svg_bytes = chart.to_svg(width=800, height=600)
        >>> with open('chart.svg', 'wb') as f:
        ...     f.write(svg_bytes)
        """
        return self.to_image(format='svg', width=width, height=height)

    def to_pdf(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> bytes:
        """
        Export chart as PDF bytes.

        Convenience method for PDF export.

        Parameters
        ----------
        width : int, optional
            Image width in pixels
        height : int, optional
            Image height in pixels

        Returns
        -------
        bytes
            PDF data

        Examples
        --------
        >>> chart.create(data)
        >>> pdf_bytes = chart.to_pdf(width=800, height=600)
        >>> with open('chart.pdf', 'wb') as f:
        ...     f.write(pdf_bytes)
        """
        return self.to_image(format='pdf', width=width, height=height)


class MatplotlibChartBase(BaseChart):
    """
    Base class for Matplotlib charts.

    Provides common Matplotlib functionality including:
    - Base64 image encoding
    - Figure and axes management
    - Theme application via matplotlib styles
    - Image export in multiple formats

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib figure object
    ax : matplotlib.axes.Axes or array of Axes
        Axes object(s)

    Examples
    --------
    >>> class MyMatplotlibChart(MatplotlibChartBase):
    ...     def _create_figure(self, data, **kwargs):
    ...         self.fig, self.ax = plt.subplots()
    ...         self.ax.plot(data['x'], data['y'])
    ...         return self.fig
    """

    def __init__(self, theme: Optional[Theme] = None, config: Optional[Dict] = None):
        """Initialize Matplotlib chart."""
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "Matplotlib is required for MatplotlibChartBase. " "Install with: pip install matplotlib"
            )

        super().__init__(theme, config)
        self.fig = None
        self.ax = None

    def _apply_theme(self, figure):
        """
        Apply Matplotlib theme.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            Figure to theme

        Returns
        -------
        matplotlib.figure.Figure
            Themed figure
        """
        # Apply matplotlib style
        if self.theme.matplotlib_style:
            plt.style.use(self.theme.matplotlib_style)

        return figure

    def to_base64(self, format: str = "png", dpi: int = 150, **kwargs) -> str:
        """
        Convert figure to base64-encoded image.

        Parameters
        ----------
        format : str, default='png'
            Image format ('png', 'jpg', 'svg', etc.)
        dpi : int, default=150
            Resolution (dots per inch)
        **kwargs
            Additional options passed to savefig

        Returns
        -------
        str
            Base64-encoded image as data URI
        """
        if self.fig is None:
            raise ValueError("Chart has not been created yet. Call create() first.")

        buffer = io.BytesIO()
        self.fig.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight", **kwargs)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close(self.fig)

        return f"data:image/{format};base64,{img_base64}"

    def to_json(self) -> str:
        """
        Convert to JSON with embedded base64 image.

        Returns
        -------
        str
            JSON with base64 image
        """
        return json.dumps({"type": "matplotlib", "image": self.to_base64()})

    def to_html(self, **kwargs) -> str:
        """
        Convert to HTML with embedded image.

        Returns
        -------
        str
            HTML img tag with base64 image
        """
        img_data = self.to_base64(**kwargs)
        return f'<img src="{img_data}" alt="Chart" />'


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles numpy types.

    Converts numpy arrays and scalars to native Python types.
    Handles NaN and Inf values by converting to None.
    """

    def default(self, obj):
        """Convert numpy types to JSON-serializable types."""
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle numpy scalars
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float_)):
            # Handle NaN and Inf
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)

        # Default behavior
        return super().default(obj)
