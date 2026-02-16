"""
Visualization module for stochastic frontier analysis.

This module provides plotting functions for SFA results:
- Efficiency distribution and rankings
- Temporal evolution of efficiency
- Estimated frontier visualization
- Diagnostic plots

All plots support both Plotly (interactive) and Matplotlib (static) backends.

Report generation:
- LaTeX tables for academic papers
- HTML reports with interactive plots
- Markdown documentation
- Model comparison tables
"""

from .efficiency_plots import (
    plot_efficiency_boxplot,
    plot_efficiency_distribution,
    plot_efficiency_ranking,
)
from .evolution_plots import (
    plot_efficiency_fanchart,
    plot_efficiency_heatmap,
    plot_efficiency_spaghetti,
    plot_efficiency_timeseries,
)
from .frontier_plots import (
    plot_frontier_2d,
    plot_frontier_3d,
    plot_frontier_contour,
    plot_frontier_partial,
)
from .reports import compare_models, efficiency_table, to_html, to_latex, to_markdown

__all__ = [
    # Efficiency plots
    "plot_efficiency_distribution",
    "plot_efficiency_ranking",
    "plot_efficiency_boxplot",
    # Evolution plots
    "plot_efficiency_timeseries",
    "plot_efficiency_spaghetti",
    "plot_efficiency_heatmap",
    "plot_efficiency_fanchart",
    # Frontier plots
    "plot_frontier_2d",
    "plot_frontier_3d",
    "plot_frontier_contour",
    "plot_frontier_partial",
    # Reports
    "to_latex",
    "to_html",
    "to_markdown",
    "compare_models",
    "efficiency_table",
]
