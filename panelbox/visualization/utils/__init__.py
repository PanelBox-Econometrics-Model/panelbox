"""
Visualization Utilities Module.

This module provides helper utilities for the PanelBox visualization system,
including chart selection assistance, theme management, and performance tools.
"""

from __future__ import annotations

from .chart_selector import (
    CHART_RECOMMENDATIONS,
    ChartRecommendation,
    get_categories,
    list_all_charts,
    suggest_chart,
)
from .theme_loader import (
    create_theme_template,
    get_theme_colors,
    list_builtin_themes,
    load_theme,
    merge_themes,
    save_theme,
)

__all__ = [
    "CHART_RECOMMENDATIONS",
    # Chart Selection
    "ChartRecommendation",
    "create_theme_template",
    "get_categories",
    "get_theme_colors",
    "list_all_charts",
    "list_builtin_themes",
    # Theme Management
    "load_theme",
    "merge_themes",
    "save_theme",
    "suggest_chart",
]
