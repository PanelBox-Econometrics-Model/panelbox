"""
Visualization Utilities Module.

This module provides helper utilities for the PanelBox visualization system,
including chart selection assistance, theme management, and performance tools.
"""

from .chart_selector import (
    ChartRecommendation,
    suggest_chart,
    list_all_charts,
    get_categories,
    CHART_RECOMMENDATIONS,
)

from .theme_loader import (
    load_theme,
    save_theme,
    merge_themes,
    create_theme_template,
    list_builtin_themes,
    get_theme_colors,
)

__all__ = [
    # Chart Selection
    'ChartRecommendation',
    'suggest_chart',
    'list_all_charts',
    'get_categories',
    'CHART_RECOMMENDATIONS',
    # Theme Management
    'load_theme',
    'save_theme',
    'merge_themes',
    'create_theme_template',
    'list_builtin_themes',
    'get_theme_colors',
]
