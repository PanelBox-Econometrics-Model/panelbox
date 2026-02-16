"""
Utility functions for stochastic frontier analysis.

This module provides helper functions for computing marginal effects,
transformations, and other utilities.
"""

from .marginal_effects import compute_delta_method_se, marginal_effects_wang_2002

__all__ = [
    "marginal_effects_wang_2002",
    "compute_delta_method_se",
]
