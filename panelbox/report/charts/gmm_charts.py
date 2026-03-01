"""Chart builder for GMM reports.

Generates Plotly charts for Arellano-Bond and Blundell-Bond GMM
estimation results: coefficient forest plot, specification tests,
and instrument summary.
"""

from __future__ import annotations

import plotly.graph_objects as go

from panelbox.report.charts._base import BaseReportChartBuilder
from panelbox.report.charts._utils import PANELBOX_COLORS


class GMMChartBuilder(BaseReportChartBuilder):
    """Build interactive Plotly charts for GMM report pages.

    Parameters
    ----------
    data : dict
        Transformer output from ``GMMTransformer``.  Expected keys:
        ``coefficients``, ``diagnostics``, ``model_info``.
    """

    def build_all(self) -> dict[str, str]:
        """Build all GMM charts and return as HTML strings.

        Returns
        -------
        dict[str, str]
            Mapping of chart names to HTML ``<div>`` strings.
            Keys: ``coefficient_plot``, ``diagnostic_chart``,
            ``instrument_chart``.
        """
        charts = {}
        charts["coefficient_plot"] = self._build_coefficient_plot()
        charts["diagnostic_chart"] = self._build_diagnostic_chart()
        charts["instrument_chart"] = self._build_instrument_chart()
        return {k: v for k, v in charts.items() if v}

    def _build_coefficient_plot(self) -> str | None:
        """Horizontal forest plot with coefficient +/- 1.96*SE (95% CI).

        Significant coefficients are colored in dark blue; non-significant
        ones appear in grey.  A vertical line at x=0 is drawn for reference.
        """
        coefficients = self.data.get("coefficients", [])
        if not coefficients:
            return None

        names = []
        coefs = []
        ci_lower = []
        ci_upper = []
        colors = []
        labels = []

        for c in reversed(coefficients):
            name = c.get("name", "")
            coef = c.get("coef", 0)
            se = c.get("se", 0)
            pval = c.get("pvalue")
            stars = c.get("stars", "")

            names.append(f"{name} {stars}".strip())
            coefs.append(coef)
            ci_lower.append(coef - 1.96 * se)
            ci_upper.append(coef + 1.96 * se)

            is_sig = pval is not None and pval < 0.05
            colors.append(PANELBOX_COLORS["primary"] if is_sig else PANELBOX_COLORS["muted"])
            labels.append(f"{coef:.4f} ({stars})" if stars else f"{coef:.4f}")

        fig = go.Figure()

        # Error bars (CI whiskers)
        fig.add_trace(
            go.Scatter(
                x=coefs,
                y=names,
                mode="markers",
                marker={"color": colors, "size": 8, "symbol": "diamond"},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": [u - c for c, u in zip(coefs, ci_upper)],
                    "arrayminus": [c - lo for c, lo in zip(coefs, ci_lower)],
                    "color": PANELBOX_COLORS["muted"],
                    "thickness": 1.5,
                    "width": 4,
                },
                text=labels,
                hovertemplate="%{y}: %{x:.4f}<br>95% CI: [%{error_x.arrayminus:.4f}, +%{error_x.array:.4f}]<extra></extra>",
                showlegend=False,
            )
        )

        # Vertical reference line at x=0
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color=PANELBOX_COLORS["danger"],
            line_width=1,
        )

        self._apply_layout(
            fig,
            title="Coefficient Estimates (95% CI)",
            height=max(300, len(coefficients) * 50 + 100),
        )
        fig.update_layout(
            xaxis_title="Coefficient",
            yaxis_title="",
            margin={"l": 120, "r": 30, "t": 50, "b": 50},
        )

        return self._fig_to_html(fig)

    def _build_diagnostic_chart(self) -> str | None:
        """Horizontal bar chart of specification test p-values.

        Draws p-values for Hansen J, AR(1), and AR(2) tests with a
        vertical threshold line at p=0.05.  Bars are green for PASS
        (p >= 0.05) and red for FAIL (p < 0.05).  AR(1) is expected
        to reject, so its color logic is inverted.
        """
        diagnostics = self.data.get("diagnostics", {})
        if not diagnostics:
            return None

        test_names = []
        pvalues = []
        bar_colors = []
        status_labels = []

        # Hansen J test
        hansen = diagnostics.get("hansen", {})
        if hansen and hansen.get("pvalue") is not None:
            pval = hansen["pvalue"]
            test_names.append("Hansen J")
            pvalues.append(pval)
            passed = pval >= 0.05
            bar_colors.append(PANELBOX_COLORS["success"] if passed else PANELBOX_COLORS["danger"])
            status_labels.append("PASS" if passed else "FAIL")

        # AR(1) test - rejection (p<0.05) is expected/desired
        ar1 = diagnostics.get("ar1", {})
        if ar1 and ar1.get("pvalue") is not None:
            pval = ar1["pvalue"]
            test_names.append("AR(1)")
            pvalues.append(pval)
            # For AR(1), we expect rejection (p < 0.05 is good)
            passed = pval < 0.05
            bar_colors.append(PANELBOX_COLORS["success"] if passed else PANELBOX_COLORS["warning"])
            status_labels.append("PASS" if passed else "WARN")

        # AR(2) test
        ar2 = diagnostics.get("ar2", {})
        if ar2 and ar2.get("pvalue") is not None:
            pval = ar2["pvalue"]
            test_names.append("AR(2)")
            pvalues.append(pval)
            passed = pval >= 0.05
            bar_colors.append(PANELBOX_COLORS["success"] if passed else PANELBOX_COLORS["danger"])
            status_labels.append("PASS" if passed else "FAIL")

        if not test_names:
            return None

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=pvalues,
                y=test_names,
                orientation="h",
                marker={"color": bar_colors},
                text=[f"{p:.3f} ({s})" for p, s in zip(pvalues, status_labels)],
                textposition="outside",
                hovertemplate="%{y}: p = %{x:.4f}<extra></extra>",
                showlegend=False,
            )
        )

        # Threshold line at p=0.05
        fig.add_vline(
            x=0.05,
            line_dash="dash",
            line_color=PANELBOX_COLORS["danger"],
            line_width=1.5,
            annotation_text="p=0.05",
            annotation_position="top",
        )

        self._apply_layout(fig, title="Specification Tests (p-values)", height=300)
        fig.update_layout(
            xaxis_title="p-value",
            xaxis={"range": [0, max(max(pvalues) * 1.3, 0.15)]},
            yaxis_title="",
            margin={"l": 80, "r": 80, "t": 50, "b": 50},
        )

        return self._fig_to_html(fig)

    def _build_instrument_chart(self) -> str | None:
        """Gauge indicator showing instrument-to-group ratio.

        Green for ratio < 1, yellow for 1-2, red for > 2.
        """
        model_info = self.data.get("model_info", {})
        n_instruments = model_info.get("n_instruments")
        n_groups = model_info.get("n_groups")

        if n_instruments is None or n_groups is None:
            return None
        if n_groups == 0:
            return None

        ratio = n_instruments / n_groups

        fig = go.Figure()

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=ratio,
                number={"suffix": "", "valueformat": ".2f"},
                title={
                    "text": (
                        f"Instrument Ratio<br>"
                        f"<span style='font-size:0.7em;color:{PANELBOX_COLORS['muted']}'>"
                        f"{n_instruments} instruments / {n_groups} groups</span>"
                    ),
                },
                gauge={
                    "axis": {"range": [0, max(3, ratio * 1.2)]},
                    "bar": {"color": PANELBOX_COLORS["primary"]},
                    "steps": [
                        {"range": [0, 1], "color": "#d4edda"},  # green zone
                        {"range": [1, 2], "color": "#fff3cd"},  # yellow zone
                        {"range": [2, max(3, ratio * 1.2)], "color": "#f8d7da"},  # red zone
                    ],
                    "threshold": {
                        "line": {"color": PANELBOX_COLORS["danger"], "width": 2},
                        "thickness": 0.8,
                        "value": 1,
                    },
                },
            )
        )

        self._apply_layout(fig, title="", height=300)
        fig.update_layout(margin={"l": 30, "r": 30, "t": 30, "b": 30})

        return self._fig_to_html(fig)
