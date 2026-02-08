"""
Configuration system for charts.

This module provides configuration classes and utilities for customizing
chart appearance and behavior.
"""

from .chart_config import ChartConfig
from .color_schemes import (
    COLORBLIND_FRIENDLY,
    MONOCHROME,
    SEQUENTIAL_BLUE,
    SEQUENTIAL_GREEN,
    SEQUENTIAL_RED,
)

__all__ = [
    "ChartConfig",
    "COLORBLIND_FRIENDLY",
    "MONOCHROME",
    "SEQUENTIAL_BLUE",
    "SEQUENTIAL_GREEN",
    "SEQUENTIAL_RED",
]
