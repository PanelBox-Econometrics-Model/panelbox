"""
Chart configuration dataclass.

Provides a structured way to configure individual charts.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ChartConfig:
    """
    Configuration for individual charts.

    This class encapsulates all configurable aspects of a chart that
    can be customized independently of the theme.

    Parameters
    ----------
    width : int, default=800
        Chart width in pixels
    height : int, default=600
        Chart height in pixels
    title : str, optional
        Chart title
    title_font_size : int, default=16
        Title font size in points
    xaxis_title : str, optional
        X-axis label
    yaxis_title : str, optional
        Y-axis label
    show_legend : bool, default=True
        Whether to display legend
    legend_position : str, default='top right'
        Legend position ('top right', 'top left', 'bottom right', 'bottom left')
    responsive : bool, default=True
        Whether chart is responsive to container size
    static_plot : bool, default=False
        If True, disable all interactivity (useful for exports)
    hover_mode : str, default='x unified'
        Plotly hover mode ('x unified', 'closest', 'x', 'y', False)
    plotly_config : dict, optional
        Additional Plotly-specific configuration
    margin : dict, optional
        Custom margins {'l': left, 'r': right, 't': top, 'b': bottom}

    Examples
    --------
    Create basic config:

    >>> config = ChartConfig(
    ...     title="Residuals vs Fitted Values",
    ...     width=1000,
    ...     height=700
    ... )

    Config for static export:

    >>> config = ChartConfig(
    ...     width=1200,
    ...     height=900,
    ...     static_plot=True,
    ...     title="Q-Q Plot for Normality"
    ... )

    Config with custom Plotly settings:

    >>> config = ChartConfig(
    ...     title="Interactive Dashboard",
    ...     plotly_config={
    ...         'displayModeBar': True,
    ...         'displaylogo': False,
    ...         'toImageButtonOptions': {
    ...             'format': 'png',
    ...             'filename': 'chart',
    ...             'height': 1080,
    ...             'width': 1920,
    ...             'scale': 2
    ...         }
    ...     }
    ... )
    """

    # Dimensions
    width: int = 800
    height: int = 600

    # Titles and labels
    title: Optional[str] = None
    title_font_size: int = 16
    xaxis_title: Optional[str] = None
    yaxis_title: Optional[str] = None

    # Legend
    show_legend: bool = True
    legend_position: str = "top right"  # 'top right', 'top left', 'bottom right', 'bottom left'

    # Interactivity
    responsive: bool = True
    static_plot: bool = False
    hover_mode: str = "x unified"  # 'x unified', 'closest', 'x', 'y', False

    # Plotly-specific configuration
    plotly_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "chart",
                "height": 600,
                "width": 800,
                "scale": 2,  # Higher quality
            },
        }
    )

    # Layout
    margin: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns
        -------
        dict
            Configuration as dictionary
        """
        return {
            "width": self.width,
            "height": self.height,
            "title": self.title,
            "title_font_size": self.title_font_size,
            "xaxis_title": self.xaxis_title,
            "yaxis_title": self.yaxis_title,
            "show_legend": self.show_legend,
            "legend_position": self.legend_position,
            "responsive": self.responsive,
            "static_plot": self.static_plot,
            "hover_mode": self.hover_mode,
            "plotly_config": self.plotly_config,
            "margin": self.margin,
        }

    def get_legend_config(self) -> Dict[str, Any]:
        """
        Get Plotly legend configuration based on position.

        Returns
        -------
        dict
            Plotly legend config
        """
        position_map = {
            "top right": {"x": 1, "y": 1, "xanchor": "right", "yanchor": "top"},
            "top left": {"x": 0, "y": 1, "xanchor": "left", "yanchor": "top"},
            "bottom right": {"x": 1, "y": 0, "xanchor": "right", "yanchor": "bottom"},
            "bottom left": {"x": 0, "y": 0, "xanchor": "left", "yanchor": "bottom"},
        }

        legend_config = position_map.get(self.legend_position, position_map["top right"])
        legend_config["orientation"] = "v"  # Vertical by default

        return legend_config

    def merge(self, other: "ChartConfig") -> "ChartConfig":
        """
        Merge with another config (other takes precedence).

        Parameters
        ----------
        other : ChartConfig
            Config to merge with

        Returns
        -------
        ChartConfig
            New merged config
        """
        merged = ChartConfig()

        # For each field, use other's value if set, otherwise use self's
        for field_name in self.__dataclass_fields__:
            other_value = getattr(other, field_name)
            self_value = getattr(self, field_name)

            # Use default value from class as reference
            default_value = self.__dataclass_fields__[field_name].default

            # If other has non-default value, use it; otherwise use self's
            if other_value != default_value:
                setattr(merged, field_name, other_value)
            else:
                setattr(merged, field_name, self_value)

        return merged
