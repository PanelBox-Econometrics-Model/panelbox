"""
Quantile Regression Visualization Suite.

This module provides professional visualization tools for quantile regression,
including coefficient paths, surface plots, fan charts, and more.
"""

from __future__ import annotations

import contextlib

with contextlib.suppress(ImportError):
    from .process_plots import qq_plot, quantile_process_plot, residual_plot

# New advanced visualization components - optional dependencies (matplotlib/plotly)
with contextlib.suppress(ImportError):
    from .advanced_plots import QuantileVisualizer

with contextlib.suppress(ImportError):
    from .interactive import InteractivePlotter

with contextlib.suppress(ImportError):
    from .surface_plots import SurfacePlotter

with contextlib.suppress(ImportError):
    from .themes import PublicationTheme

__all__ = [
    "InteractivePlotter",
    "PublicationTheme",
    "QuantileVisualizer",
    "SurfacePlotter",
    "qq_plot",
    "quantile_process_plot",
    "residual_plot",
]
