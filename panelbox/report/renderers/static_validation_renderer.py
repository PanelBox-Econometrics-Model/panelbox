"""
Static Validation Renderer.

Generates static charts for validation reports using Matplotlib.
"""

import base64
import warnings
from io import BytesIO
from typing import Any, Dict, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn(
        "Matplotlib not available. Static charts will not be generated. "
        "Install with: pip install matplotlib"
    )


class StaticValidationRenderer:
    """
    Renders static validation charts using Matplotlib.

    Creates publication-quality static charts suitable for PDFs,
    printed reports, and non-interactive viewing.

    Parameters
    ----------
    figure_size : tuple, default=(10, 6)
        Default figure size (width, height) in inches
    dpi : int, default=150
        Resolution in dots per inch
    style : str, default='seaborn-v0_8-darkgrid'
        Matplotlib style to use

    Examples
    --------
    >>> from panelbox.report.renderers import StaticValidationRenderer
    >>>
    >>> renderer = StaticValidationRenderer(dpi=300)
    >>> charts = renderer.render_validation_charts(validation_data)
    >>> # charts contains base64-encoded PNG images
    """

    def __init__(
        self,
        figure_size: Tuple[int, int] = (10, 6),
        dpi: int = 150,
        style: str = "seaborn-v0_8-darkgrid",
    ):
        """Initialize Static Validation Renderer."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "Matplotlib is required for static chart rendering. "
                "Install with: pip install matplotlib"
            )

        self.figure_size = figure_size
        self.dpi = dpi
        self.style = style

        # Set style
        try:
            plt.style.use(style)
        except (OSError, KeyError):
            # Fallback to default if style not available
            plt.style.use("default")

    def render_validation_charts(self, validation_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Render all validation charts as base64 PNG images.

        Parameters
        ----------
        validation_data : dict
            Validation data from ValidationTransformer

        Returns
        -------
        dict
            Dictionary mapping chart names to base64 image data URIs

        Examples
        --------
        >>> charts = renderer.render_validation_charts(validation_data)
        >>> overview_img = charts['test_overview']
        >>> # Use in HTML: <img src="{{overview_img}}">
        """
        charts = {}

        chart_data = validation_data.get("charts", {})

        if "test_overview" in chart_data:
            charts["test_overview"] = self._render_test_overview(chart_data["test_overview"])

        if "pvalue_distribution" in chart_data:
            charts["pvalue_distribution"] = self._render_pvalue_distribution(
                chart_data["pvalue_distribution"]
            )

        if "test_statistics" in chart_data:
            charts["test_statistics"] = self._render_test_statistics(chart_data["test_statistics"])

        return charts

    def _render_test_overview(self, data: Dict[str, Any]) -> str:
        """
        Render test overview bar chart.

        Parameters
        ----------
        data : dict
            Chart data with categories, passed, and failed counts

        Returns
        -------
        str
            Base64-encoded PNG image data URI
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        categories = data["categories"]
        passed = data["passed"]
        failed = data["failed"]

        x = np.arange(len(categories))
        width = 0.35

        # Stacked bar chart
        ax.bar(x, passed, width, label="Passed", color="#10b981")
        ax.bar(x, failed, width, bottom=passed, label="Failed", color="#ef4444")

        ax.set_xlabel("Test Category", fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of Tests", fontsize=12, fontweight="bold")
        ax.set_title("Test Results by Category", fontsize=14, fontweight="bold", pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _render_pvalue_distribution(self, data: Dict[str, Any]) -> str:
        """
        Render p-value distribution chart.

        Parameters
        ----------
        data : dict
            Chart data with test names and p-values

        Returns
        -------
        str
            Base64-encoded PNG image data URI
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        test_names = data["test_names"]
        pvalues = data["pvalues"]

        x = np.arange(len(test_names))

        # Color-code by significance
        colors = []
        for pval in pvalues:
            if pval < 0.01:
                colors.append("#ef4444")  # Red - highly significant
            elif pval < 0.05:
                colors.append("#f59e0b")  # Orange - significant
            elif pval < 0.1:
                colors.append("#eab308")  # Yellow - marginally significant
            else:
                colors.append("#10b981")  # Green - not significant

        ax.bar(x, pvalues, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

        # Add significance threshold lines
        ax.axhline(y=0.05, color="red", linestyle="--", linewidth=2, label="α = 0.05")
        ax.axhline(y=0.01, color="darkred", linestyle=":", linewidth=1.5, label="α = 0.01")

        ax.set_xlabel("Test Name", fontsize=12, fontweight="bold")
        ax.set_ylabel("P-value", fontsize=12, fontweight="bold")
        ax.set_title("P-values by Test", fontsize=14, fontweight="bold", pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha="right", fontsize=9)
        ax.set_yscale("log")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3, which="both")

        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _render_test_statistics(self, data: Dict[str, Any]) -> str:
        """
        Render test statistics scatter plot.

        Parameters
        ----------
        data : dict
            Chart data with test names and statistics

        Returns
        -------
        str
            Base64-encoded PNG image data URI
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        test_names = data["test_names"]
        statistics = data["statistics"]

        x = np.arange(len(test_names))

        ax.scatter(
            x,
            statistics,
            s=100,
            color="#2563eb",
            alpha=0.7,
            edgecolors="#1e40af",
            linewidth=2,
            zorder=3,
        )

        ax.set_xlabel("Test Name", fontsize=12, fontweight="bold")
        ax.set_ylabel("Test Statistic", fontsize=12, fontweight="bold")
        ax.set_title("Test Statistics", fontsize=14, fontweight="bold", pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        return self._fig_to_base64(fig)

    def render_summary_chart(self, summary: Dict[str, Any]) -> str:
        """
        Render summary pie chart.

        Parameters
        ----------
        summary : dict
            Summary data with total_passed and total_failed

        Returns
        -------
        str
            Base64-encoded PNG image data URI

        Examples
        --------
        >>> summary_chart = renderer.render_summary_chart(summary)
        """
        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)

        passed = summary["total_passed"]
        failed = summary["total_failed"]

        sizes = [passed, failed]
        labels = [f"Passed ({passed})", f"Failed ({failed})"]
        colors = ["#10b981", "#ef4444"]
        explode = (0.05, 0.05)

        wedges, texts, autotexts = ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
        )

        # Enhance text
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight("bold")

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(12)
            autotext.set_fontweight("bold")

        ax.set_title(
            f'Test Results Summary\n({summary["total_tests"]} total tests)',
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        """
        Convert matplotlib figure to base64 data URI.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to convert

        Returns
        -------
        str
            Base64-encoded PNG data URI
        """
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        buf.seek(0)

        # Encode as base64
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        # Close figure to free memory
        plt.close(fig)

        # Create data URI
        return f"data:image/png;base64,{img_base64}"

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StaticValidationRenderer("
            f"figure_size={self.figure_size}, "
            f"dpi={self.dpi}, "
            f"style='{self.style}')"
        )
