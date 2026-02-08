"""
Visualization Utilities Module.

This module provides helper utilities for the PanelBox visualization system,
including chart selection assistance, theme management, and performance tools.
"""

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
    # Chart Selection
    "ChartRecommendation",
    "suggest_chart",
    "list_all_charts",
    "get_categories",
    "CHART_RECOMMENDATIONS",
    # Theme Management
    "load_theme",
    "save_theme",
    "merge_themes",
    "create_theme_template",
    "list_builtin_themes",
    "get_theme_colors",
]
