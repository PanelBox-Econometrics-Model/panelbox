"""
Quantile Regression Visualization Suite.

This module provides professional visualization tools for quantile regression,
including coefficient paths, surface plots, fan charts, and more.
"""

try:
    from .process_plots import qq_plot, quantile_process_plot, residual_plot
except ImportError:
    pass

# New advanced visualization components - optional dependencies (matplotlib/plotly)
try:
    from .advanced_plots import QuantileVisualizer
except ImportError:
    pass

try:
    from .interactive import InteractivePlotter
except ImportError:
    pass

try:
    from .surface_plots import SurfacePlotter
except ImportError:
    pass

try:
    from .themes import PublicationTheme
except ImportError:
    pass

__all__ = [
    "QuantileVisualizer",
    "PublicationTheme",
    "SurfacePlotter",
    "InteractivePlotter",
    "quantile_process_plot",
    "residual_plot",
    "qq_plot",
]
