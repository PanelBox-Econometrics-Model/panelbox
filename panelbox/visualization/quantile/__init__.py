"""
Quantile Regression Visualization Suite.

This module provides professional visualization tools for quantile regression,
including coefficient paths, surface plots, fan charts, and more.
"""

try:
    from .process_plots import qq_plot, quantile_process_plot, residual_plot
except ImportError:
    pass

# New advanced visualization components
from .advanced_plots import QuantileVisualizer
from .interactive import InteractivePlotter
from .surface_plots import SurfacePlotter
from .themes import PublicationTheme

__all__ = [
    "QuantileVisualizer",
    "PublicationTheme",
    "SurfacePlotter",
    "InteractivePlotter",
    "quantile_process_plot",
    "residual_plot",
    "qq_plot",
]
