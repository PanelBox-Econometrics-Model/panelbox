"""Chart builder for Discrete/MLE reports.

Generates Plotly charts for discrete choice model results (Logit, Probit,
Ordered Probit, Multinomial Logit): coefficient forest plot, classification
metrics, and information criteria comparison.
"""

from __future__ import annotations

import plotly.graph_objects as go

from panelbox.report.charts._base import BaseReportChartBuilder
from panelbox.report.charts._utils import PANELBOX_COLORS


class DiscreteChartBuilder(BaseReportChartBuilder):
    """Build interactive Plotly charts for Discrete/MLE report pages.

    Parameters
    ----------
    data : dict
        Transformer output from ``DiscreteTransformer``.  Expected keys:
        ``coefficients``, ``fit_statistics``, ``classification`` (optional),
        ``model_info``.
    """

    def build_all(self) -> dict[str, str]:
        """Build all Discrete/MLE charts and return as HTML strings.

        Returns
        -------
        dict[str, str]
            Mapping of chart names to HTML ``<div>`` strings.
            Keys: ``coefficient_plot``, ``classification_chart`` (conditional),
            ``ic_chart``.
        """
        charts = {}
        charts["coefficient_plot"] = self._build_coefficient_plot()
        charts["classification_chart"] = self._build_classification_chart()
        charts["ic_chart"] = self._build_ic_chart()
        return {k: v for k, v in charts.items() if v}

    def _build_coefficient_plot(self) -> str | None:
        """Horizontal forest plot with coefficient estimates and 95% CIs.

        Uses z-statistics for discrete choice models.  Significant
        coefficients (p < 0.05) are colored in dark blue; non-significant
        ones appear in grey.  A vertical reference line at x=0 is drawn.
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
            if lo is None or hi is None or lo == "" or hi == "":
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
                hovertemplate=(
                    "%{y}: %{x:.4f}<br>"
                    "95% CI: [%{error_x.arrayminus:.4f}, +%{error_x.array:.4f}]"
                    "<extra></extra>"
                ),
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

    def _build_classification_chart(self) -> str | None:
        """Bar chart of classification metrics (accuracy, precision, recall, F1).

        Only generated when ``classification`` data is present (binary models).
        Non-binary models (Ordered Probit, Multinomial Logit) may not have
        classification metrics and this method returns None.
        """
        classification = self.data.get("classification")
        if classification is None:
            return None

        metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
        metric_keys = ["accuracy", "precision", "recall", "f1_score"]
        metric_colors = [
            PANELBOX_COLORS["primary"],
            PANELBOX_COLORS["secondary"],
            PANELBOX_COLORS["accent"],
            PANELBOX_COLORS["info"],
        ]

        values = []
        names = []
        colors = []
        for name, key, color in zip(metric_names, metric_keys, metric_colors):
            val = classification.get(key)
            if val is not None:
                names.append(name)
                values.append(val)
                colors.append(color)

        if not values:
            return None

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=names,
                y=values,
                marker={"color": colors},
                text=[f"{v:.2%}" for v in values],
                textposition="outside",
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
                showlegend=False,
            )
        )

        # Random baseline reference line at 0.5
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color=PANELBOX_COLORS["danger"],
            line_width=1,
            annotation_text="Random baseline (0.5)",
            annotation_position="right",
        )

        self._apply_layout(fig, title="Classification Metrics", height=350)
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Score",
            yaxis={"range": [0, 1.1]},
            margin={"l": 60, "r": 80, "t": 50, "b": 50},
        )

        return self._fig_to_html(fig)

    def _build_ic_chart(self) -> str | None:
        """Bar chart comparing AIC and BIC information criteria."""
        fit = self.data.get("fit_statistics", {})
        if not fit:
            return None

        aic = fit.get("aic")
        bic = fit.get("bic")

        names = []
        values = []
        colors = []

        if aic is not None:
            names.append("AIC")
            values.append(aic)
            colors.append(PANELBOX_COLORS["secondary"])
        if bic is not None:
            names.append("BIC")
            values.append(bic)
            colors.append(PANELBOX_COLORS["accent"])

        if not values:
            return None

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=names,
                y=values,
                marker={"color": colors},
                text=[f"{v:.2f}" for v in values],
                textposition="outside",
                hovertemplate="%{x}: %{y:.2f}<extra></extra>",
                showlegend=False,
            )
        )

        self._apply_layout(fig, title="Information Criteria", height=350)
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Value",
            margin={"l": 60, "r": 30, "t": 50, "b": 50},
        )

        return self._fig_to_html(fig)
