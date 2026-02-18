"""
Discrete Choice Models - Utilities

Part of the PanelBox tutorial series on discrete choice econometrics.

This module provides shared utility functions for the discrete choice tutorials:

Modules:
    data_generators        : Functions to generate synthetic discrete choice data
    visualization_helpers  : Standardized plotting functions for discrete models

Example:
    >>> from discrete.utils.data_generators import generate_labor_data
    >>> from discrete.utils.visualization_helpers import plot_link_functions
    >>>
    >>> data = generate_labor_data(n_individuals=1000, n_periods=5)
    >>> plot_link_functions(compare=['logit', 'probit', 'lpm'])
"""

__all__ = ["data_generators", "visualization_helpers"]
