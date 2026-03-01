"""Chart builder for VAR (Panel Vector Autoregression) reports.

Generates Plotly charts for VAR estimation results: stability indicator,
information criteria comparison, and coefficient heatmap across equations.
"""

from __future__ import annotations

import plotly.graph_objects as go

from panelbox.report.charts._base import BaseReportChartBuilder
from panelbox.report.charts._utils import PANELBOX_COLORS


class VARChartBuilder(BaseReportChartBuilder):
    """Build interactive Plotly charts for VAR report pages.

    Parameters
    ----------
    data : dict
        Transformer output from ``VARTransformer``.  Expected keys:
        ``equations``, ``diagnostics``, ``stability``, ``model_info``.
    """

    def build_all(self) -> dict[str, str]:
        """Build all VAR charts and return as HTML strings.

        Returns
        -------
        dict[str, str]
            Mapping of chart names to HTML ``<div>`` strings.
            Keys: ``stability_chart``, ``ic_chart``, ``coefficient_heatmap``.
        """
        charts = {}
        charts["stability_chart"] = self._build_stability_chart()
        charts["ic_chart"] = self._build_ic_chart()
        charts["coefficient_heatmap"] = self._build_coefficient_heatmap()
        return {k: v for k, v in charts.items() if v}

    def _build_stability_chart(self) -> str | None:
        """Gauge indicator showing maximum eigenvalue modulus vs 1.0.

        The VAR system is stable when all eigenvalues lie inside the unit
        circle, i.e. max eigenvalue modulus < 1.  The gauge shows this
        value with green/yellow/red zones.
        """
        stability = self.data.get("stability")
        if stability is None:
            return None

        max_mod = stability.get("max_eigenvalue_modulus")
        is_stable = stability.get("is_stable")
        margin_val = stability.get("stability_margin")

        if max_mod is None:
            return None

        # Determine status label
        if is_stable is not None:
            status = "STABLE" if is_stable else "UNSTABLE"
            status_color = PANELBOX_COLORS["success"] if is_stable else PANELBOX_COLORS["danger"]
        else:
            status = "STABLE" if max_mod < 1.0 else "UNSTABLE"
            status_color = (
                PANELBOX_COLORS["success"] if max_mod < 1.0 else PANELBOX_COLORS["danger"]
            )

        margin_text = ""
        if margin_val is not None:
            margin_text = f"<br>Margin: {margin_val:.4f}"

        fig = go.Figure()

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=max_mod,
                number={"valueformat": ".4f"},
                title={
                    "text": (
                        f"Max Eigenvalue Modulus<br>"
                        f"<span style='font-size:0.8em;color:{status_color}'>"
                        f"{status}</span>"
                        f"<span style='font-size:0.7em;color:{PANELBOX_COLORS['muted']}'>"
                        f"{margin_text}</span>"
                    ),
                },
                gauge={
                    "axis": {"range": [0, max(1.5, max_mod * 1.2)]},
                    "bar": {"color": PANELBOX_COLORS["primary"]},
                    "steps": [
                        {"range": [0, 0.8], "color": "#d4edda"},
                        {"range": [0.8, 1.0], "color": "#fff3cd"},
                        {
                            "range": [1.0, max(1.5, max_mod * 1.2)],
                            "color": "#f8d7da",
                        },
                    ],
                    "threshold": {
                        "line": {"color": PANELBOX_COLORS["danger"], "width": 3},
                        "thickness": 0.8,
                        "value": 1.0,
                    },
                },
            )
        )

        self._apply_layout(fig, title="", height=300)
        fig.update_layout(margin={"l": 30, "r": 30, "t": 30, "b": 30})

        return self._fig_to_html(fig)

    def _build_ic_chart(self) -> str | None:
        """Bar chart comparing AIC, BIC, and HQIC information criteria."""
        diagnostics = self.data.get("diagnostics", {})
        if not diagnostics:
            return None

        ic_keys = [
            ("AIC", "aic", PANELBOX_COLORS["secondary"]),
            ("BIC", "bic", PANELBOX_COLORS["accent"]),
            ("HQIC", "hqic", PANELBOX_COLORS["info"]),
        ]

        names = []
        values = []
        colors = []

        for label, key, color in ic_keys:
            val = diagnostics.get(key)
            if val is not None:
                names.append(label)
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

    def _build_coefficient_heatmap(self) -> str | None:
        """Heatmap of coefficients across equations.

        Rows represent regressors (union of all coefficient names), columns
        represent equations.  Values are coefficient estimates with a
        diverging blue-white-red colorscale.  Cell annotations show the
        coefficient value and significance stars.
        """
        equations = self.data.get("equations", [])
        if not equations:
            return None

        # Build the coefficient matrix
        eq_names = []
        eq_coef_maps: list[dict[str, dict]] = []

        for eq in equations:
            eq_name = eq.get("name", "")
            eq_names.append(eq_name)
            coef_map = {}
            for c in eq.get("coefficients", []):
                coef_map[c.get("name", "")] = c
            eq_coef_maps.append(coef_map)

        if not eq_names:
            return None

        # Collect all regressor names in order of first appearance
        regressor_names: list[str] = []
        seen: set[str] = set()
        for coef_map in eq_coef_maps:
            for name in coef_map:
                if name not in seen:
                    regressor_names.append(name)
                    seen.add(name)

        if not regressor_names:
            return None

        # Build matrix (rows=regressors, cols=equations)
        z_values: list[list[float | None]] = []
        annotations: list[list[str]] = []

        for reg_name in regressor_names:
            row_vals: list[float | None] = []
            row_annots: list[str] = []
            for coef_map in eq_coef_maps:
                c = coef_map.get(reg_name)
                if c is not None:
                    coef_val = c.get("coef", 0)
                    stars = c.get("stars", "")
                    row_vals.append(coef_val)
                    text = f"{coef_val:.3f}"
                    if stars:
                        text += f" {stars}"
                    row_annots.append(text)
                else:
                    row_vals.append(None)
                    row_annots.append("")
            z_values.append(row_vals)
            annotations.append(row_annots)

        # Find the absolute max for symmetric colorscale
        flat_vals = [v for row in z_values for v in row if v is not None]
        if not flat_vals:
            return None
        abs_max = max(abs(v) for v in flat_vals)
        if abs_max == 0:
            abs_max = 1.0

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                z=z_values,
                x=eq_names,
                y=regressor_names,
                colorscale=[
                    [0, PANELBOX_COLORS["secondary"]],
                    [0.5, "white"],
                    [1, PANELBOX_COLORS["danger"]],
                ],
                zmin=-abs_max,
                zmax=abs_max,
                text=annotations,
                texttemplate="%{text}",
                hovertemplate=(
                    "Regressor: %{y}<br>Equation: %{x}<br>Coefficient: %{z:.4f}<extra></extra>"
                ),
                showscale=True,
                colorbar={"title": "Coef"},
            )
        )

        self._apply_layout(
            fig,
            title="Coefficient Heatmap",
            height=max(350, len(regressor_names) * 40 + 150),
        )
        fig.update_layout(
            xaxis_title="Equation",
            yaxis_title="Regressor",
            yaxis={"autorange": "reversed"},
            margin={"l": 120, "r": 80, "t": 50, "b": 50},
        )

        return self._fig_to_html(fig)
