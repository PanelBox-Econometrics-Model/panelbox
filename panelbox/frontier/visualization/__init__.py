"""Visualization module for stochastic frontier analysis.

This module provides plotting and visualization functions for SFA models,
including efficiency distributions, frontier plots, and evolution analysis.
"""

from .four_component_plots import (
    plot_comprehensive_summary,
    plot_efficiency_distributions,
    plot_efficiency_evolution,
    plot_efficiency_scatter,
    plot_entity_decomposition,
    plot_variance_decomposition,
)

__all__ = [
    "plot_efficiency_distributions",
    "plot_efficiency_scatter",
    "plot_efficiency_evolution",
    "plot_entity_decomposition",
    "plot_variance_decomposition",
    "plot_comprehensive_summary",
]
