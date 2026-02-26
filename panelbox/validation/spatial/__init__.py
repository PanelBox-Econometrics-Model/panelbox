"""
Spatial diagnostics and validation tests for panel data models.

This module provides a comprehensive suite of spatial diagnostics including:
- Moran's I (global and local)
- LM tests for spatial dependence
- Spatial Hausman tests
- Integration with PanelBox validation framework
"""

from __future__ import annotations

from .lm_tests import LMErrorTest, LMLagTest, RobustLMErrorTest, RobustLMLagTest, run_lm_tests
from .moran_i import MoranIByPeriodResult, MoranIPanelTest
from .spatial_hausman import SpatialHausmanTest
from .utils import standardize_spatial_weights, validate_spatial_weights

__all__ = [
    "LMErrorTest",
    # LM Tests
    "LMLagTest",
    "MoranIByPeriodResult",
    # Moran's I
    "MoranIPanelTest",
    "RobustLMErrorTest",
    "RobustLMLagTest",
    # Hausman Test
    "SpatialHausmanTest",
    "run_lm_tests",
    "standardize_spatial_weights",
    # Utilities
    "validate_spatial_weights",
]
