"""
Color schemes and palettes for charts.

Provides curated color palettes for different use cases:
- Colorblind-friendly palettes
- Monochrome palettes
- Sequential color scales
- Diverging color scales
"""

from typing import List

# ============================================================================
# Colorblind-Friendly Palettes
# ============================================================================

COLORBLIND_FRIENDLY: List[str] = [
    "#0173B2",  # Blue
    "#DE8F05",  # Orange
    "#029E73",  # Green
    "#CC78BC",  # Pink
    "#CA9161",  # Tan
    "#949494",  # Gray
    "#ECE133",  # Yellow
    "#56B4E9",  # Sky blue
]
"""
Colorblind-friendly palette based on research by Wong (2011).

Safe for most types of color blindness (protanopia, deuteranopia, tritanopia).

References
----------
Wong, B. (2011). Color blindness. Nature Methods, 8(6), 441.
"""

# ============================================================================
# Monochrome Palettes
# ============================================================================

MONOCHROME: List[str] = [
    "#000000",  # Black
    "#2D2D2D",  # Very dark gray
    "#5A5A5A",  # Dark gray
    "#787878",  # Medium gray
    "#A0A0A0",  # Light gray
    "#C8C8C8",  # Very light gray
]
"""Monochrome grayscale palette for print-friendly charts."""

# ============================================================================
# Sequential Color Scales
# ============================================================================

SEQUENTIAL_BLUE: List[str] = [
    "#F7FBFF",  # Very light blue
    "#DEEBF7",  # Light blue
    "#C6DBEF",  # Light-medium blue
    "#9ECAE1",  # Medium blue
    "#6BAED6",  # Medium-dark blue
    "#4292C6",  # Dark blue
    "#2171B5",  # Darker blue
    "#08519C",  # Very dark blue
    "#08306B",  # Darkest blue
]
"""Sequential blue color scale for continuous data."""

SEQUENTIAL_GREEN: List[str] = [
    "#F7FCF5",  # Very light green
    "#E5F5E0",  # Light green
    "#C7E9C0",  # Light-medium green
    "#A1D99B",  # Medium green
    "#74C476",  # Medium-dark green
    "#41AB5D",  # Dark green
    "#238B45",  # Darker green
    "#006D2C",  # Very dark green
    "#00441B",  # Darkest green
]
"""Sequential green color scale for continuous data."""

SEQUENTIAL_RED: List[str] = [
    "#FFF5F0",  # Very light red
    "#FEE0D2",  # Light red
    "#FCBBA1",  # Light-medium red
    "#FC9272",  # Medium red
    "#FB6A4A",  # Medium-dark red
    "#EF3B2C",  # Dark red
    "#CB181D",  # Darker red
    "#A50F15",  # Very dark red
    "#67000D",  # Darkest red
]
"""Sequential red color scale for continuous data (warnings/errors)."""

# ============================================================================
# Diverging Color Scales
# ============================================================================

DIVERGING_RED_BLUE: List[str] = [
    "#67001F",  # Dark red
    "#B2182B",  # Red
    "#D6604D",  # Light red
    "#F4A582",  # Very light red
    "#FDDBC7",  # Pale red
    "#F7F7F7",  # White (neutral)
    "#D1E5F0",  # Pale blue
    "#92C5DE",  # Very light blue
    "#4393C3",  # Light blue
    "#2166AC",  # Blue
    "#053061",  # Dark blue
]
"""Diverging red-blue color scale (for positive/negative values)."""

DIVERGING_BROWN_TEAL: List[str] = [
    "#8C510A",  # Dark brown
    "#BF812D",  # Brown
    "#DFC27D",  # Light brown
    "#F6E8C3",  # Very light brown
    "#F5F5F5",  # White (neutral)
    "#C7EAE5",  # Very light teal
    "#80CDC1",  # Light teal
    "#35978F",  # Teal
    "#01665E",  # Dark teal
]
"""Diverging brown-teal color scale (earth tones)."""

# ============================================================================
# Categorical Palettes
# ============================================================================

CATEGORICAL_VIBRANT: List[str] = [
    "#EE7733",  # Orange
    "#0077BB",  # Blue
    "#33BBEE",  # Cyan
    "#EE3377",  # Magenta
    "#CC3311",  # Red
    "#009988",  # Teal
    "#BBBBBB",  # Gray
]
"""Vibrant categorical palette (Paul Tol)."""

CATEGORICAL_MUTED: List[str] = [
    "#CC6677",  # Rose
    "#332288",  # Indigo
    "#DDCC77",  # Sand
    "#117733",  # Green
    "#88CCEE",  # Cyan
    "#882255",  # Wine
    "#44AA99",  # Teal
    "#999933",  # Olive
    "#AA4499",  # Purple
]
"""Muted categorical palette (Paul Tol)."""

# ============================================================================
# Status/Semantic Colors
# ============================================================================

STATUS_COLORS = {
    "success": "#28a745",  # Green
    "warning": "#ffc107",  # Amber/Yellow
    "danger": "#dc3545",  # Red
    "info": "#17a2b8",  # Cyan
    "primary": "#007bff",  # Blue
    "secondary": "#6c757d",  # Gray
}
"""Standard status colors for UI elements."""

SIGNIFICANCE_COLORS = {
    "highly_significant": "#d62728",  # Red (p < 0.01)
    "significant": "#ff7f0e",  # Orange (0.01 <= p < 0.05)
    "marginally_significant": "#ffbb78",  # Light orange (0.05 <= p < 0.10)
    "not_significant": "#2ca02c",  # Green (p >= 0.10)
}
"""Colors for statistical significance levels."""


def get_color_for_pvalue(pvalue: float, alpha: float = 0.05) -> str:
    """
    Get color based on p-value and significance threshold.

    Parameters
    ----------
    pvalue : float
        P-value to color-code
    alpha : float, default=0.05
        Significance threshold

    Returns
    -------
    str
        Hex color code

    Examples
    --------
    >>> get_color_for_pvalue(0.001)
    '#d62728'  # Red (highly significant)
    >>> get_color_for_pvalue(0.03)
    '#ff7f0e'  # Orange (significant)
    >>> get_color_for_pvalue(0.15)
    '#2ca02c'  # Green (not significant)
    """
    if pvalue < alpha / 10:  # Very strong evidence
        return SIGNIFICANCE_COLORS["highly_significant"]
    elif pvalue < alpha:  # Strong evidence
        return SIGNIFICANCE_COLORS["significant"]
    elif pvalue < alpha * 2:  # Marginal evidence
        return SIGNIFICANCE_COLORS["marginally_significant"]
    else:  # No evidence
        return SIGNIFICANCE_COLORS["not_significant"]


def interpolate_color_scale(colors: List[str], n_colors: int) -> List[str]:
    """
    Interpolate a color scale to a specific number of colors.

    Parameters
    ----------
    colors : list of str
        Base colors to interpolate
    n_colors : int
        Desired number of colors

    Returns
    -------
    list of str
        Interpolated color scale

    Examples
    --------
    >>> scale = interpolate_color_scale(SEQUENTIAL_BLUE, n_colors=5)
    >>> len(scale)
    5
    """
    if n_colors <= len(colors):
        # Subsample
        indices = [int(i * (len(colors) - 1) / (n_colors - 1)) for i in range(n_colors)]
        return [colors[i] for i in indices]
    else:
        # Simple repetition for now (can implement proper interpolation later)
        repeated = colors * ((n_colors // len(colors)) + 1)
        return repeated[:n_colors]
