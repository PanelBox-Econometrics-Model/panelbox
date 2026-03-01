"""Chart builder for SFA (Stochastic Frontier Analysis) reports.

Generates Plotly charts for SFA estimation results: efficiency distribution
summary, efficiency indicator grid, variance decomposition (sigma_v vs
sigma_u), and coefficient forest plot.
"""

from __future__ import annotations

import plotly.graph_objects as go

from panelbox.report.charts._base import BaseReportChartBuilder
from panelbox.report.charts._utils import PANELBOX_COLORS


class SFAChartBuilder(BaseReportChartBuilder):
    """Build interactive Plotly charts for SFA report pages.

    Parameters
    ----------
    data : dict
        Transformer output from ``SFATransformer``.  Expected keys:
        ``coefficients``, ``variance_components``, ``efficiency`` (may be
        None), ``fit_statistics``, ``model_info``.
    """

    def build_all(self) -> dict[str, str]:
        """Build all SFA charts and return as HTML strings.

        Returns
        -------
        dict[str, str]
            Mapping of chart names to HTML ``<div>`` strings.
            Keys: ``efficiency_distribution``, ``efficiency_summary``,
            ``variance_chart``, ``coefficient_plot``.
        """
        charts = {}
        charts["efficiency_distribution"] = self._build_efficiency_distribution()
        charts["efficiency_summary"] = self._build_efficiency_summary()
        charts["variance_chart"] = self._build_variance_chart()
        charts["coefficient_plot"] = self._build_coefficient_plot()
        return {k: v for k, v in charts.items() if v}

    def _build_efficiency_distribution(self) -> str | None:
        """Summary bar chart showing efficiency score distribution range.

        Displays a horizontal bar spanning from min to max efficiency, with
        vertical markers for mean and median.  Since the transformer only
        provides summary statistics (not individual scores), this is a
        visual representation of the distribution range.
        """
        efficiency = self.data.get("efficiency")
        if efficiency is None:
            return None

        eff_mean = efficiency.get("mean")
        eff_median = efficiency.get("median")
        eff_min = efficiency.get("min")
        eff_max = efficiency.get("max")
        eff_std = efficiency.get("std")

        if eff_mean is None:
            return None

        fig = go.Figure()

        # Range bar from min to max
        if eff_min is not None and eff_max is not None:
            fig.add_trace(
                go.Bar(
                    x=[eff_max - eff_min],
                    y=["Efficiency"],
                    base=[eff_min],
                    orientation="h",
                    marker={
                        "color": PANELBOX_COLORS["secondary"],
                        "opacity": 0.3,
                    },
                    name="Range (Min-Max)",
                    hovertemplate=f"Range: [{eff_min:.4f}, {eff_max:.4f}]<extra></extra>",
                    showlegend=True,
                )
            )

        # +/- 1 std dev band around mean
        if eff_std is not None:
            std_lo = max(0, eff_mean - eff_std)
            std_hi = min(1, eff_mean + eff_std)
            fig.add_trace(
                go.Bar(
                    x=[std_hi - std_lo],
                    y=["Efficiency"],
                    base=[std_lo],
                    orientation="h",
                    marker={
                        "color": PANELBOX_COLORS["secondary"],
                        "opacity": 0.5,
                    },
                    name="Mean +/- 1 SD",
                    hovertemplate=f"Mean +/- 1 SD: [{std_lo:.4f}, {std_hi:.4f}]<extra></extra>",
                    showlegend=True,
                )
            )

        # Mean marker
        fig.add_trace(
            go.Scatter(
                x=[eff_mean],
                y=["Efficiency"],
                mode="markers",
                marker={
                    "color": PANELBOX_COLORS["primary"],
                    "size": 14,
                    "symbol": "diamond",
                    "line": {"width": 2, "color": "white"},
                },
                name=f"Mean ({eff_mean:.4f})",
                hovertemplate=f"Mean: {eff_mean:.4f}<extra></extra>",
                showlegend=True,
            )
        )

        # Median marker
        if eff_median is not None:
            fig.add_trace(
                go.Scatter(
                    x=[eff_median],
                    y=["Efficiency"],
                    mode="markers",
                    marker={
                        "color": PANELBOX_COLORS["accent"],
                        "size": 14,
                        "symbol": "circle",
                        "line": {"width": 2, "color": "white"},
                    },
                    name=f"Median ({eff_median:.4f})",
                    hovertemplate=f"Median: {eff_median:.4f}<extra></extra>",
                    showlegend=True,
                )
            )

        fig.update_layout(
            title="Efficiency Score Distribution",
            height=350,
            xaxis_title="Efficiency Score",
            xaxis={"range": [0, 1.05], "automargin": False},
            yaxis={"automargin": False},
            barmode="overlay",
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.15,
                "xanchor": "center",
                "x": 0.5,
            },
            margin={"l": 80, "r": 30, "t": 50, "b": 100},
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=self.layout_defaults.get("font"),
        )

        return self._fig_to_html(fig)

    def _build_efficiency_summary(self) -> str | None:
        """Grid of numeric indicators for efficiency summary statistics.

        Shows Mean, Median, Min, Max, and Std as large-number text
        annotations.  Colors are based on the efficiency value (green for
        high, red for low efficiency).

        Uses annotations instead of go.Indicator + make_subplots to avoid
        the Plotly 'Too many auto-margin redraws' bug.
        """
        efficiency = self.data.get("efficiency")
        if efficiency is None:
            return None

        metrics = [
            ("Mean", efficiency.get("mean")),
            ("Median", efficiency.get("median")),
            ("Std Dev", efficiency.get("std")),
            ("Min", efficiency.get("min")),
            ("Max", efficiency.get("max")),
        ]

        # Filter out None values
        metrics = [(name, val) for name, val in metrics if val is not None]
        if not metrics:
            return None

        n = len(metrics)
        fig = go.Figure()

        # Add invisible scatter to create the figure area
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker={"opacity": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )

        for i, (name, val) in enumerate(metrics):
            x_pos = (i + 0.5) / n

            # Color based on value
            if name == "Std Dev":
                color = PANELBOX_COLORS["secondary"]
            elif val >= 0.7:
                color = PANELBOX_COLORS["success"]
            elif val >= 0.4:
                color = PANELBOX_COLORS["warning"]
            else:
                color = PANELBOX_COLORS["danger"]

            # Value text (large number)
            fig.add_annotation(
                text=f"<b>{val:.4f}</b>",
                x=x_pos,
                y=0.45,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 28, "color": color},
                xanchor="center",
                yanchor="middle",
            )

            # Label text (smaller, above)
            fig.add_annotation(
                text=name,
                x=x_pos,
                y=0.85,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 14, "color": PANELBOX_COLORS["primary"]},
                xanchor="center",
                yanchor="middle",
            )

        fig.update_layout(
            title="Efficiency Summary",
            height=180,
            margin={"l": 20, "r": 20, "t": 50, "b": 10},
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=self.layout_defaults.get("font"),
            xaxis={"visible": False, "range": [0, 1]},
            yaxis={"visible": False, "range": [0, 1]},
        )

        return self._fig_to_html(fig)

    def _build_variance_chart(self) -> str | None:
        """Horizontal bar chart showing variance decomposition.

        Shows the proportion of noise (sigma_v^2) vs inefficiency
        (sigma_u^2) in the total variance, with gamma annotation.

        Uses a stacked bar instead of a Pie chart to avoid the Plotly
        'Too many auto-margin redraws' bug caused by
        ``textposition="outside"`` on Pie traces.
        """
        vc = self.data.get("variance_components")
        if vc is None:
            return None

        sigma_v = vc.get("sigma_v")
        sigma_u = vc.get("sigma_u")
        gamma = vc.get("gamma")

        if sigma_v is None or sigma_u is None:
            return None

        sigma_v_sq = sigma_v**2
        sigma_u_sq = sigma_u**2
        total = sigma_v_sq + sigma_u_sq
        noise_pct = sigma_v_sq / total * 100 if total > 0 else 0
        ineff_pct = sigma_u_sq / total * 100 if total > 0 else 0

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=[noise_pct],
                y=["Variance"],
                orientation="h",
                name=f"Noise \u03c3\u00b2v ({noise_pct:.1f}%)",
                marker={"color": PANELBOX_COLORS["secondary"]},
                text=[f"\u03c3\u00b2v = {sigma_v_sq:.4f}"],
                textposition="inside",
                textfont={"color": "white", "size": 13},
                hovertemplate=(
                    "Noise (\u03c3\u00b2<sub>v</sub>)<br>"
                    f"Value: {sigma_v_sq:.4f}<br>"
                    f"Share: {noise_pct:.1f}%<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Bar(
                x=[ineff_pct],
                y=["Variance"],
                orientation="h",
                name=f"Inefficiency \u03c3\u00b2u ({ineff_pct:.1f}%)",
                marker={"color": PANELBOX_COLORS["danger"]},
                text=[f"\u03c3\u00b2u = {sigma_u_sq:.4f}"],
                textposition="inside",
                textfont={"color": "white", "size": 13},
                hovertemplate=(
                    "Inefficiency (\u03c3\u00b2<sub>u</sub>)<br>"
                    f"Value: {sigma_u_sq:.4f}<br>"
                    f"Share: {ineff_pct:.1f}%<extra></extra>"
                ),
            )
        )

        gamma_text = f"\u03b3 = {gamma:.4f}" if gamma is not None else ""

        fig.update_layout(
            title="Variance Decomposition",
            height=250,
            barmode="stack",
            xaxis={
                "title": "Share (%)",
                "range": [0, 100],
                "automargin": False,
            },
            yaxis={"automargin": False},
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.25,
                "xanchor": "center",
                "x": 0.5,
            },
            margin={"l": 80, "r": 30, "t": 50, "b": 80},
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=self.layout_defaults.get("font"),
        )

        if gamma_text:
            fig.add_annotation(
                text=f"<b>{gamma_text}</b>  (share of variance due to inefficiency)",
                x=50,
                y=-0.05,
                xref="x",
                yref="paper",
                showarrow=False,
                font={"size": 14, "color": PANELBOX_COLORS["primary"]},
                xanchor="center",
            )

        return self._fig_to_html(fig)

    def _build_coefficient_plot(self) -> str | None:
        """Horizontal forest plot with coefficient estimates and 95% CIs.

        Significant coefficients (p < 0.05) are colored in dark blue;
        non-significant ones appear in grey.  A vertical reference line
        at x=0 is drawn.
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
                hovertemplate=(
                    "%{y}: %{x:.4f}<br>"
                    "95% CI: [%{error_x.arrayminus:.4f}, +%{error_x.array:.4f}]"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

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
