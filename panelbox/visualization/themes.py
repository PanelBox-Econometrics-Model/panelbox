"""
Visual theming system for charts.

This module provides professional themes for consistent chart styling across
all visualization types. Themes follow modern design system principles with
design tokens, color palettes, and layout configurations.

Available Themes
----------------
- PROFESSIONAL_THEME: Default theme for business reports
- ACADEMIC_THEME: Publication-ready theme for academic papers
- PRESENTATION_THEME: Bold, high-contrast theme for presentations
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List


@dataclass
class Theme:
    """
    Visual theme configuration for charts.

    A theme encapsulates all visual styling for charts including colors,
    fonts, layouts, and backend-specific templates.

    Parameters
    ----------
    name : str
        Theme identifier (e.g., 'professional', 'academic')
    color_scheme : list of str
        Ordered list of hex colors for data series
    font_config : dict
        Font configuration (family, size, color)
    layout_config : dict
        Layout defaults (margins, background colors, etc.)
    plotly_template : str, default='plotly_white'
        Plotly template name
    matplotlib_style : str, default='seaborn-v0_8-whitegrid'
        Matplotlib style name

    Examples
    --------
    Create a custom theme:

    >>> custom_theme = Theme(
    ...     name='corporate',
    ...     color_scheme=['#003366', '#FF6600', '#00CC99'],
    ...     font_config={'family': 'Arial', 'size': 12, 'color': '#333333'},
    ...     layout_config={'paper_bgcolor': '#FFFFFF'}
    ... )

    Use with a chart:

    >>> chart = ChartFactory.create('qq_plot', data=residuals, theme=custom_theme)
    """

    name: str
    color_scheme: List[str]
    font_config: Dict[str, any]
    layout_config: Dict[str, any]
    plotly_template: str = "plotly_white"
    matplotlib_style: str = "seaborn-v0_8-whitegrid"

    # Additional color mappings
    success_color: str = field(default="#28a745")
    warning_color: str = field(default="#ffc107")
    danger_color: str = field(default="#dc3545")
    info_color: str = field(default="#17a2b8")

    def get_color(self, index: int) -> str:
        """
        Get color from scheme by index (with wrapping).

        Parameters
        ----------
        index : int
            Color index

        Returns
        -------
        str
            Hex color code
        """
        return self.color_scheme[index % len(self.color_scheme)]

    def to_dict(self) -> Dict:
        """
        Export theme as dictionary.

        Returns
        -------
        dict
            Theme configuration
        """
        return {
            "name": self.name,
            "color_scheme": self.color_scheme,
            "font_config": self.font_config,
            "layout_config": self.layout_config,
            "plotly_template": self.plotly_template,
            "matplotlib_style": self.matplotlib_style,
        }


# ============================================================================
# Professional Theme (Default)
# ============================================================================

PROFESSIONAL_THEME = Theme(
    name="professional",
    # Color scheme: Modern, accessible palette
    color_scheme=[
        "#1f77b4",  # Blue - primary
        "#ff7f0e",  # Orange - secondary
        "#2ca02c",  # Green - success
        "#d62728",  # Red - danger
        "#9467bd",  # Purple - accent
        "#8c564b",  # Brown - neutral
        "#e377c2",  # Pink - highlight
        "#7f7f7f",  # Gray - muted
        "#bcbd22",  # Yellow-green
        "#17becf",  # Cyan
    ],
    # Font configuration
    font_config={
        "family": "Arial, Helvetica, sans-serif",
        "size": 12,
        "color": "#2c3e50",  # Dark blue-gray for readability
    },
    # Layout configuration
    layout_config={
        "paper_bgcolor": "#FFFFFF",  # White background
        "plot_bgcolor": "#F8F9FA",  # Light gray plot area
        "hovermode": "x unified",  # Unified hover for better UX
        "showlegend": True,
        "legend": {
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "#dee2e6",
            "borderwidth": 1,
        },
        "margin": {"l": 60, "r": 40, "t": 80, "b": 60},
        # Grid styling
        "xaxis": {
            "gridcolor": "#dee2e6",
            "gridwidth": 1,
            "zeroline": True,
            "zerolinecolor": "#495057",
            "zerolinewidth": 1.5,
        },
        "yaxis": {
            "gridcolor": "#dee2e6",
            "gridwidth": 1,
            "zeroline": True,
            "zerolinecolor": "#495057",
            "zerolinewidth": 1.5,
        },
    },
    plotly_template="plotly_white",
    matplotlib_style="seaborn-v0_8-whitegrid",
    # Status colors
    success_color="#28a745",  # Green
    warning_color="#ffc107",  # Amber
    danger_color="#dc3545",  # Red
    info_color="#17a2b8",  # Cyan
)


# ============================================================================
# Academic Theme (Publication-Ready)
# ============================================================================

ACADEMIC_THEME = Theme(
    name="academic",
    # Grayscale + accent colors for print-friendly output
    color_scheme=[
        "#000000",  # Black - primary
        "#404040",  # Dark gray
        "#808080",  # Medium gray
        "#A0A0A0",  # Light gray
        "#1f77b4",  # Blue accent (for color figures)
        "#2ca02c",  # Green accent
    ],
    # Conservative font for publications
    font_config={
        "family": "Times New Roman, Georgia, serif",
        "size": 11,
        "color": "#000000",
    },
    # Minimal layout for publication
    layout_config={
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",  # Pure white for print
        "hovermode": "closest",
        "showlegend": True,
        "legend": {
            "bgcolor": "rgba(255, 255, 255, 1)",
            "bordercolor": "#000000",
            "borderwidth": 1,
        },
        "margin": {"l": 70, "r": 50, "t": 70, "b": 60},
        # Subtle grid
        "xaxis": {
            "gridcolor": "#d0d0d0",
            "gridwidth": 0.5,
            "zeroline": True,
            "zerolinecolor": "#000000",
            "zerolinewidth": 1,
            "mirror": True,  # Box around plot
            "ticks": "outside",
            "linecolor": "#000000",
            "linewidth": 1,
        },
        "yaxis": {
            "gridcolor": "#d0d0d0",
            "gridwidth": 0.5,
            "zeroline": True,
            "zerolinecolor": "#000000",
            "zerolinewidth": 1,
            "mirror": True,
            "ticks": "outside",
            "linecolor": "#000000",
            "linewidth": 1,
        },
    },
    plotly_template="simple_white",
    matplotlib_style="seaborn-v0_8-paper",
    # Muted status colors for academic context
    success_color="#2ca02c",
    warning_color="#ff7f0e",
    danger_color="#d62728",
    info_color="#1f77b4",
)


# ============================================================================
# Presentation Theme (High Contrast)
# ============================================================================

PRESENTATION_THEME = Theme(
    name="presentation",
    # Bold, saturated colors for visibility
    color_scheme=[
        "#E63946",  # Vibrant red
        "#1D3557",  # Navy blue
        "#2A9D8F",  # Teal
        "#E9C46A",  # Golden yellow
        "#F4A261",  # Orange
        "#A8DADC",  # Light blue
        "#457B9D",  # Medium blue
        "#E76F51",  # Terracotta
    ],
    # Large, bold font for readability
    font_config={
        "family": "Helvetica, Arial, sans-serif",
        "size": 14,
        "color": "#1a1a1a",
    },
    # High contrast layout
    layout_config={
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#F8F9FA",
        "hovermode": "x unified",
        "showlegend": True,
        "legend": {
            "bgcolor": "rgba(255, 255, 255, 0.95)",
            "bordercolor": "#1a1a1a",
            "borderwidth": 2,
            "font": {"size": 13},
        },
        "margin": {"l": 80, "r": 60, "t": 100, "b": 80},
        # Prominent grid
        "xaxis": {
            "gridcolor": "#dee2e6",
            "gridwidth": 1.5,
            "zeroline": True,
            "zerolinecolor": "#1a1a1a",
            "zerolinewidth": 2,
            "linecolor": "#1a1a1a",
            "linewidth": 2,
        },
        "yaxis": {
            "gridcolor": "#dee2e6",
            "gridwidth": 1.5,
            "zeroline": True,
            "zerolinecolor": "#1a1a1a",
            "zerolinewidth": 2,
            "linecolor": "#1a1a1a",
            "linewidth": 2,
        },
    },
    plotly_template="plotly_white",
    matplotlib_style="seaborn-v0_8-talk",
    # Vibrant status colors
    success_color="#2A9D8F",
    warning_color="#E9C46A",
    danger_color="#E63946",
    info_color="#457B9D",
)


# ============================================================================
# Theme Registry and Utilities
# ============================================================================

# Registry of available themes
_THEME_REGISTRY: Dict[str, Theme] = {
    "professional": PROFESSIONAL_THEME,
    "academic": ACADEMIC_THEME,
    "presentation": PRESENTATION_THEME,
}


def get_theme(theme: str | Theme) -> Theme:
    """
    Get theme by name or return theme object.

    Provides flexible theme specification with caching for performance.

    Parameters
    ----------
    theme : str or Theme
        Theme name ('professional', 'academic', 'presentation') or Theme object

    Returns
    -------
    Theme
        Theme configuration

    Raises
    ------
    ValueError
        If theme name is not registered

    Examples
    --------
    Get by name:

    >>> theme = get_theme('professional')

    Pass through Theme object:

    >>> custom = Theme(name='custom', ...)
    >>> theme = get_theme(custom)  # Returns custom unchanged
    """
    # If already a Theme object, return it (can't cache because Theme isn't hashable)
    if isinstance(theme, Theme):
        return theme

    # Look up by name (cached)
    return _get_theme_by_name(theme)


@lru_cache(maxsize=32)
def _get_theme_by_name(theme_name: str) -> Theme:
    """Internal cached theme lookup by name."""
    if not isinstance(theme_name, str):
        raise TypeError(f"theme must be str or Theme, got {type(theme_name)}")

    theme_lower = theme_name.lower()
    if theme_lower not in _THEME_REGISTRY:
        available = ", ".join(_THEME_REGISTRY.keys())
        raise ValueError(
            f"Theme '{theme_name}' not found. Available themes: {available}. "
            "Or pass a custom Theme object."
        )
    return _THEME_REGISTRY[theme_lower]


def register_theme(theme: Theme) -> None:
    """
    Register a custom theme.

    Allows users to register their own themes for use throughout the library.

    Parameters
    ----------
    theme : Theme
        Theme to register

    Examples
    --------
    >>> corporate_theme = Theme(
    ...     name='corporate',
    ...     color_scheme=['#003366', '#FF6600'],
    ...     font_config={'family': 'Arial', 'size': 12, 'color': '#333'},
    ...     layout_config={'paper_bgcolor': '#FFF'}
    ... )
    >>> register_theme(corporate_theme)
    >>> chart = ChartFactory.create('qq_plot', theme='corporate', ...)
    """
    if not isinstance(theme, Theme):
        raise TypeError(f"theme must be a Theme object, got {type(theme)}")

    _THEME_REGISTRY[theme.name.lower()] = theme
    # Clear cache to pick up new theme
    _get_theme_by_name.cache_clear()


def list_themes() -> List[str]:
    """
    List all registered theme names.

    Returns
    -------
    list of str
        Sorted list of theme names

    Examples
    --------
    >>> themes = list_themes()
    >>> print(f"Available themes: {', '.join(themes)}")
    Available themes: academic, presentation, professional
    """
    return sorted(_THEME_REGISTRY.keys())
