"""Shared utilities for report chart builders.

Provides color palettes, layout defaults, and formatting helpers
for generating Plotly charts embeddable in HTML reports.
"""

from __future__ import annotations

import math

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ---------------------------------------------------------------------------
# Color palette aligned with branding.css
# ---------------------------------------------------------------------------

PANELBOX_COLORS: dict[str, str] = {
    "primary": "#1B2A4A",  # --brand-dark-blue
    "secondary": "#2E5090",  # --brand-secondary-blue
    "accent": "#20B2AA",  # --brand-teal
    "success": "#28a745",
    "danger": "#dc3545",
    "warning": "#ffc107",
    "info": "#17a2b8",
    "muted": "#6c757d",
    "light": "#f8f9fa",
}

# ---------------------------------------------------------------------------
# Default Plotly layout settings
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT_DEFAULTS: dict = {
    "font": {
        "family": "Arial, Helvetica, sans-serif",
        "size": 12,
        "color": "#1B2A4A",
    },
    "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "template": "plotly_white",
    "colorway": [
        PANELBOX_COLORS["secondary"],
        PANELBOX_COLORS["accent"],
        PANELBOX_COLORS["danger"],
        PANELBOX_COLORS["warning"],
        PANELBOX_COLORS["info"],
        PANELBOX_COLORS["success"],
        PANELBOX_COLORS["muted"],
        PANELBOX_COLORS["primary"],
    ],
}

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_pvalue(pval: float | None) -> str:
    """Format a p-value for display.

    Parameters
    ----------
    pval : float or None
        The p-value to format.

    Returns
    -------
    str
        Formatted string such as "<0.001", "0.045", or "N/A".
    """
    if pval is None or (isinstance(pval, float) and math.isnan(pval)):
        return "N/A"
    if pval < 0.001:
        return "<0.001"
    return f"{pval:.3f}"


def significance_color(pval: float | None) -> str:
    """Return a color based on statistical significance level.

    Parameters
    ----------
    pval : float or None
        The p-value.

    Returns
    -------
    str
        Hex color string: green for significant, red for not, muted for N/A.
    """
    if pval is None or (isinstance(pval, float) and math.isnan(pval)):
        return PANELBOX_COLORS["muted"]
    if pval < 0.01:
        return PANELBOX_COLORS["success"]
    if pval < 0.05:
        return PANELBOX_COLORS["info"]
    if pval < 0.10:
        return PANELBOX_COLORS["warning"]
    return PANELBOX_COLORS["danger"]


def fig_to_html(fig: go.Figure) -> str:
    """Convert a Plotly figure to an embeddable HTML string.

    Uses ``full_html=False`` and ``include_plotlyjs=False`` so the
    resulting ``<div>`` can be placed in a page that already loads
    Plotly.js once via the base template.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to convert.

    Returns
    -------
    str
        HTML ``<div>`` string containing the chart.
    """
    if not HAS_PLOTLY:
        return "<div>Plotly is not installed. Charts are unavailable.</div>"
    return fig.to_html(full_html=False, include_plotlyjs=False)
