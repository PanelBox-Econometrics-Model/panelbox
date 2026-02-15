"""
Effects decomposition for econometric models.

This module provides tools for decomposing effects in various econometric models,
particularly spatial models where effects can be separated into direct, indirect,
and total components.
"""

from panelbox.effects.spatial_effects import (
    SpatialEffectsResult,
    compute_spatial_effects,
    spatial_impact_matrix,
)

__all__ = ["compute_spatial_effects", "SpatialEffectsResult", "spatial_impact_matrix"]
