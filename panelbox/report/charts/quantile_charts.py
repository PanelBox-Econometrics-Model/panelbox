"""Chart builder for Quantile Diagnostics reports.

Generates Plotly charts for quantile regression diagnostic results:
health score gauge indicator and test results p-value chart.
"""

from __future__ import annotations

import plotly.graph_objects as go

from panelbox.report.charts._base import BaseReportChartBuilder
from panelbox.report.charts._utils import PANELBOX_COLORS

STATUS_COLORS = {
    "pass": PANELBOX_COLORS["success"],
    "warning": PANELBOX_COLORS["warning"],
    "fail": PANELBOX_COLORS["danger"],
}


class QuantileChartBuilder(BaseReportChartBuilder):
    """Build interactive Plotly charts for Quantile Diagnostics report pages.

    Parameters
    ----------
    data : dict
        Transformer output from ``QuantileTransformer``.  Expected keys:
        ``health``, ``tests``, ``recommendations``.
    """

    def build_all(self) -> dict[str, str]:
        """Build all Quantile Diagnostics charts and return as HTML strings.

        Returns
        -------
        dict[str, str]
            Mapping of chart names to HTML ``<div>`` strings.
            Keys: ``health_gauge``, ``test_results_chart``.
        """
        charts = {}
        charts["health_gauge"] = self._build_health_gauge()
        charts["test_results_chart"] = self._build_test_results_chart()
        return {k: v for k, v in charts.items() if v}

    def _build_health_gauge(self) -> str | None:
        """Gauge indicator showing the model health score.

        The score is provided as a float in [0, 1] and is multiplied by 100
        for display.  The gauge arc is colored: 0-40 red, 40-70 yellow,
        70-100 green.
        """
        health = self.data.get("health")
        if health is None:
            return None

        score = health.get("score")
        if score is None:
            return None

        score_pct = score * 100

        fig = go.Figure()

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=score_pct,
                number={"suffix": "%", "valueformat": ".0f"},
                title={
                    "text": "Model Health Score",
                    "font": {"size": 16, "color": PANELBOX_COLORS["primary"]},
                },
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": PANELBOX_COLORS["primary"]},
                    "steps": [
                        {"range": [0, 40], "color": "#f8d7da"},
                        {"range": [40, 70], "color": "#fff3cd"},
                        {"range": [70, 100], "color": "#d4edda"},
                    ],
                    "threshold": {
                        "line": {"color": PANELBOX_COLORS["danger"], "width": 2},
                        "thickness": 0.8,
                        "value": score_pct,
                    },
                },
            )
        )

        self._apply_layout(fig, title="", height=300)
        fig.update_layout(margin={"l": 30, "r": 30, "t": 50, "b": 30})

        return self._fig_to_html(fig)

    def _build_test_results_chart(self) -> str | None:
        """Horizontal bar chart of test p-values sorted by value.

        Each bar is colored according to its status (pass/warning/fail).
        A vertical reference line at p=0.05 marks the significance threshold.
        """
        tests = self.data.get("tests", [])
        if not tests:
            return None

        # Sort tests by p-value ascending
        sorted_tests = sorted(
            tests,
            key=lambda t: t.get("pvalue", 1.0) if t.get("pvalue") is not None else 1.0,
        )

        names = []
        pvalues = []
        bar_colors = []

        for t in sorted_tests:
            pval = t.get("pvalue")
            if pval is None:
                continue
            names.append(t.get("name", ""))
            pvalues.append(pval)
            status = t.get("status", "fail")
            bar_colors.append(STATUS_COLORS.get(status, PANELBOX_COLORS["muted"]))

        if not names:
            return None

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=pvalues,
                y=names,
                orientation="h",
                marker={"color": bar_colors},
                text=[f"{p:.4f}" for p in pvalues],
                textposition="outside",
                hovertemplate="%{y}: p = %{x:.4f}<extra></extra>",
                showlegend=False,
            )
        )

        # Significance threshold line at p=0.05
        fig.add_vline(
            x=0.05,
            line_dash="dash",
            line_color=PANELBOX_COLORS["danger"],
            line_width=1.5,
            annotation_text="p=0.05",
            annotation_position="top",
        )

        self._apply_layout(
            fig, title="Test Results (p-values)", height=max(300, len(names) * 50 + 100)
        )
        fig.update_layout(
            xaxis_title="p-value",
            xaxis={"range": [0, max(max(pvalues) * 1.3, 0.15)]},
            yaxis_title="",
            margin={"l": 120, "r": 80, "t": 50, "b": 50},
        )

        return self._fig_to_html(fig)
