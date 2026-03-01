"""Chart builder for Regression reports.

Generates Plotly charts for panel regression results: coefficient
forest plot, fit statistics dashboard, and p-value distribution.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from panelbox.report.charts._base import BaseReportChartBuilder
from panelbox.report.charts._utils import PANELBOX_COLORS


class RegressionChartBuilder(BaseReportChartBuilder):
    """Build interactive Plotly charts for Regression report pages.

    Parameters
    ----------
    data : dict
        Transformer output from ``RegressionTransformer``.  Expected keys:
        ``coefficients``, ``fit_statistics``, ``model_info``.
    """

    def build_all(self) -> dict[str, str]:
        """Build all Regression charts and return as HTML strings.

        Returns
        -------
        dict[str, str]
            Mapping of chart names to HTML ``<div>`` strings.
            Keys: ``coefficient_plot``, ``fit_chart``, ``pvalue_chart``.
        """
        charts = {}
        charts["coefficient_plot"] = self._build_coefficient_plot()
        charts["fit_chart"] = self._build_fit_chart()
        charts["pvalue_chart"] = self._build_pvalue_chart()
        return {k: v for k, v in charts.items() if v}

    def _build_coefficient_plot(self) -> str | None:
        """Horizontal forest plot with exact confidence intervals.

        Uses ``ci_lower`` and ``ci_upper`` from the transformer when
        available, falling back to coef +/- 1.96*SE otherwise.
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

            # Prefer exact CI if available
            lo = c.get("ci_lower")
            hi = c.get("ci_upper")
            if lo is None or hi is None:
                lo = coef - 1.96 * se
                hi = coef + 1.96 * se
            ci_lower.append(lo)
            ci_upper.append(hi)

            is_sig = pval is not None and pval < 0.05
            colors.append(PANELBOX_COLORS["primary"] if is_sig else PANELBOX_COLORS["muted"])
            labels.append(f"{coef:.4f} ({stars})" if stars else f"{coef:.4f}")

        fig = go.Figure()

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

    def _build_fit_chart(self) -> str | None:
        """Number indicators for R-squared, Adj. R-squared, and F-statistic."""
        fit = self.data.get("fit_statistics", {})
        if not fit:
            return None

        r2 = fit.get("r_squared")
        adj_r2 = fit.get("adj_r_squared")
        f_stat = fit.get("f_statistic")

        # Need at least one statistic
        indicators = []
        if r2 is not None:
            indicators.append(("R-squared", r2, ".4f", self._r2_color(r2)))
        if adj_r2 is not None:
            indicators.append(("Adj. R-squared", adj_r2, ".4f", self._r2_color(adj_r2)))
        if f_stat is not None:
            indicators.append(("F-statistic", f_stat, ".2f", PANELBOX_COLORS["secondary"]))

        if not indicators:
            return None

        n_indicators = len(indicators)
        fig = make_subplots(
            rows=1,
            cols=n_indicators,
            specs=[[{"type": "indicator"}] * n_indicators],
        )

        for i, (title, value, fmt, color) in enumerate(indicators, start=1):
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=value,
                    number={
                        "valueformat": fmt,
                        "font": {"size": 36, "color": color},
                    },
                    title={
                        "text": title,
                        "font": {"size": 14, "color": PANELBOX_COLORS["primary"]},
                    },
                ),
                row=1,
                col=i,
            )

        self._apply_layout(fig, title="Model Fit Statistics", height=250)
        fig.update_layout(margin={"l": 30, "r": 30, "t": 60, "b": 30})

        return self._fig_to_html(fig)

    def _build_pvalue_chart(self) -> str | None:
        """Bar chart of per-coefficient p-values with significance thresholds."""
        coefficients = self.data.get("coefficients", [])
        if not coefficients:
            return None

        names = [c.get("name", "") for c in coefficients]
        pvalues = [c.get("pvalue", 1.0) for c in coefficients]
        bar_colors = [self._pvalue_bar_color(p) for p in pvalues]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=names,
                y=pvalues,
                marker={"color": bar_colors},
                text=[f"{p:.4f}" for p in pvalues],
                textposition="outside",
                hovertemplate="%{x}: p = %{y:.4f}<extra></extra>",
                showlegend=False,
            )
        )

        # Significance threshold lines
        thresholds = [
            (0.01, "p=0.01", PANELBOX_COLORS["success"]),
            (0.05, "p=0.05", PANELBOX_COLORS["info"]),
            (0.10, "p=0.10", PANELBOX_COLORS["warning"]),
        ]
        for val, label, color in thresholds:
            fig.add_hline(
                y=val,
                line_dash="dash",
                line_color=color,
                line_width=1,
                annotation_text=label,
                annotation_position="right",
            )

        self._apply_layout(fig, title="P-values by Coefficient", height=350)
        fig.update_layout(
            xaxis_title="Coefficient",
            yaxis_title="p-value",
            yaxis={"range": [0, min(max(pvalues) * 1.3, 1.05)]},
            margin={"l": 60, "r": 60, "t": 50, "b": 80},
        )

        return self._fig_to_html(fig)

    @staticmethod
    def _r2_color(r2: float) -> str:
        """Return color based on R-squared quality."""
        if r2 >= 0.7:
            return PANELBOX_COLORS["success"]
        if r2 >= 0.4:
            return PANELBOX_COLORS["warning"]
        return PANELBOX_COLORS["danger"]

    @staticmethod
    def _pvalue_bar_color(pval: float) -> str:
        """Return bar color based on significance level."""
        if pval < 0.01:
            return PANELBOX_COLORS["success"]
        if pval < 0.05:
            return PANELBOX_COLORS["info"]
        if pval < 0.10:
            return PANELBOX_COLORS["warning"]
        return PANELBOX_COLORS["danger"]
