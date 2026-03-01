"""Base class for report chart builders.

All report-specific chart builders (GMM, Regression, etc.) inherit from
:class:`BaseReportChartBuilder` which provides shared layout helpers and
the ``fig_to_html`` conversion.
"""

from __future__ import annotations

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from panelbox.report.charts._utils import (
    PANELBOX_COLORS,
    PLOTLY_LAYOUT_DEFAULTS,
    fig_to_html,
)


class BaseReportChartBuilder:
    """Base class for generating Plotly charts embeddable in HTML reports.

    Parameters
    ----------
    data : dict
        Transformer output dict containing the data needed for chart
        generation.
    """

    def __init__(self, data: dict) -> None:
        self.data = data
        self.colors = PANELBOX_COLORS
        self.layout_defaults = PLOTLY_LAYOUT_DEFAULTS

    def build_all(self) -> dict[str, str]:
        """Build all charts and return them as a dict of HTML strings.

        Returns
        -------
        dict[str, str]
            Mapping of chart names to HTML ``<div>`` strings.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError

    def _apply_layout(
        self,
        fig: go.Figure,
        title: str = "",
        height: int = 400,
    ) -> go.Figure:
        """Apply the standard PanelBox layout to a Plotly figure.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            The figure to update.
        title : str
            Chart title.
        height : int
            Chart height in pixels.

        Returns
        -------
        plotly.graph_objects.Figure
            The updated figure.
        """
        fig.update_layout(
            title=title,
            height=height,
            **self.layout_defaults,
        )
        return fig

    def _fig_to_html(self, fig: go.Figure) -> str:
        """Convert a Plotly figure to an embeddable HTML string.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            The figure to convert.

        Returns
        -------
        str
            HTML ``<div>`` string.
        """
        return fig_to_html(fig)
