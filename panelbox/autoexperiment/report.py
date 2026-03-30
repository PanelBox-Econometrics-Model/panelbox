"""AutoExperiment HTML report generator.

Produces a self-contained, interactive HTML report by leveraging
PanelBox's existing ReportManager infrastructure (Jinja2 templates,
CSS layers, embedded assets).
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# SE type → human-readable justification
_SE_JUSTIFICATION = {
    "robust": "Heteroscedasticity detected (Modified Wald / Breusch-Pagan); using White robust SE.",
    "clustered": "Cross-sectional dependence detected (Pesaran CD); using cluster-robust SE.",
    "driscoll_kraay": "Cross-sectional dependence and serial correlation detected; using Driscoll-Kraay SE.",
    "newey_west": "Serial correlation detected (Wooldridge); using Newey-West HAC SE.",
    "nonrobust": "No heteroscedasticity or dependence detected; using conventional SE.",
}


class AutoExperimentReportGenerator:
    """Generate a professional HTML report for :class:`AutoExperimentResults`.

    Parameters
    ----------
    results : AutoExperimentResults
        Output container from :meth:`AutoExperiment.run`.
    """

    _ESTIMATOR_LABELS = {
        "pooled_ols": "Pooled OLS",
        "fe": "Fixed Effects",
        "re": "Random Effects",
        "fd": "First Difference",
    }

    def __init__(self, results: Any):
        self.results = results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, filepath: str | None = None) -> str:
        """Generate a complete HTML report.

        Parameters
        ----------
        filepath : str, optional
            If provided the report is saved to this path.
        theme : str, default ``'professional'``
            Reserved for future theme support.

        Returns
        -------
        str
            Self-contained HTML string.
        """
        from panelbox.report import ReportManager

        report_mgr = ReportManager()
        context = self._build_template_data()

        html = report_mgr.generate_report(
            report_type="autoexperiment",
            template="autoexperiment/index.html",
            context=context,
            include_plotly=False,
            custom_css=["autoexperiment_report.css"],
        )

        if filepath:
            report_mgr.save_report(html, filepath, overwrite=True)

        return html

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_template_data(self) -> dict[str, Any]:
        """Prepare the full Jinja2 template context."""
        res = self.results

        # Best model info
        best_classification = self._best_classification()
        best_model_info = self._best_model_metrics()

        return {
            # Header / meta
            "report_title": "AutoExperiment Report \u2014 PanelBox",
            "report_subtitle": f"Best: {self._estimator_label()} | {res.best_formula}",
            "model_type": "AutoExperiment",
            "formula": res.best_formula,
            "show_export_buttons": True,
            # Executive summary
            "best_estimator_label": self._estimator_label(),
            "best_formula": res.best_formula,
            "best_bic": self._fmt(best_model_info.get("bic"), 2),
            "best_rsq_adj": self._fmt(best_model_info.get("rsq_adj"), 4),
            "best_classification": best_classification,
            "best_cov_type": res.best_cov_type,
            "n_models_tested": len(res.all_results),
            "n_combinations_tested": res.n_combinations_tested,
            "datamining_warning": res.datamining_warning,
            # Ranking
            "ranking_rows": self._ranking_rows(),
            # Best model detail
            "best_model_info": best_model_info,
            "coefficients": self._coefficients(),
            "se_justification": _SE_JUSTIFICATION.get(res.best_cov_type, ""),
            "best_diagnostics": self._best_diagnostics(),
            # Tests
            "test_matrix": self._test_matrix(),
            "test_names": self._all_test_names(),
            # Variable selection
            "variable_selection": res.variable_selection,
            "bic_chart_b64": self._render_bic_chart(),
            # Transformations
            "transformations": self._transformations(),
        }

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _ranking_rows(self) -> list[dict]:
        """Return top-10 ranking rows as list of dicts."""
        df = self.results.ranking
        if df.empty:
            return []

        rows = []
        for idx, row in df.head(10).iterrows():
            rsq_adj = row.get("rsq_adj")
            if rsq_adj is not None and isinstance(rsq_adj, float) and np.isnan(rsq_adj):
                rsq_adj = None

            rows.append(
                {
                    "rank": idx + 1,
                    "model_type": row.get("model_type", ""),
                    "formula": row.get("formula", ""),
                    "bic": row.get("bic", 0),
                    "aic": row.get("aic", 0),
                    "rsq_adj": rsq_adj,
                    "classification": row.get("classification", "N/A"),
                    "n_vars": row.get("n_vars", 0),
                    "score": row.get("score", 0),
                }
            )
        return rows

    def _coefficients(self) -> list[dict]:
        """Extract coefficient table from best model."""
        model = self.results.best_model
        if model is None:
            return []

        try:
            params = model.params
            std_errors = model.std_errors
            tstats = model.tstats
            pvalues = model.pvalues

            coefs = []
            for name in params.index:
                coefs.append(
                    {
                        "name": name,
                        "estimate": float(params[name]),
                        "std_error": float(std_errors[name]),
                        "tstat": float(tstats[name]),
                        "pvalue": float(pvalues[name]),
                    }
                )
            return coefs
        except Exception:
            logger.debug("Could not extract coefficients from best model.", exc_info=True)
            return []

    def _test_matrix(self) -> dict:
        """Build models x tests matrix for the heatmap."""
        test_results = self.results.test_results
        if not test_results:
            return {}

        matrix: dict[str, dict[str, str]] = {}
        for model_name, info in test_results.items():
            passed = set(info.get("passed", []))
            failed = set(info.get("failed", []))
            classification = info.get("classification", "N/A")

            cell: dict[str, str] = {}
            for t in passed:
                cell[t] = "pass"
            for t in failed:
                cell[t] = "fail"
            cell["_classification"] = classification
            matrix[model_name] = cell

        return matrix

    def _all_test_names(self) -> list[str]:
        """Collect all unique test names across models, sorted."""
        names: set[str] = set()
        for info in self.results.test_results.values():
            names.update(info.get("passed", []))
            names.update(info.get("failed", []))
        return sorted(names)

    def _render_bic_chart(self) -> str:
        """Render BIC path chart as base64-encoded PNG (matplotlib)."""
        variable_selection = self.results.variable_selection
        if not variable_selection:
            return ""

        # Check if any model has a bic_path
        has_path = any(len(sel.get("bic_path", [])) > 1 for sel in variable_selection.values())
        if not has_path:
            return ""

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.debug("matplotlib not available; skipping BIC chart.")
            return ""

        fig, ax = plt.subplots(figsize=(8, 4))

        colors = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]
        for i, (model_type, sel) in enumerate(variable_selection.items()):
            bic_path = sel.get("bic_path", [])
            if len(bic_path) > 1:
                steps = list(range(1, len(bic_path) + 1))
                color = colors[i % len(colors)]
                ax.plot(
                    steps,
                    bic_path,
                    marker="o",
                    linewidth=2,
                    label=model_type,
                    color=color,
                )

        ax.set_xlabel("Forward Step")
        ax.set_ylabel("BIC")
        ax.set_title("BIC Path — Forward Stepwise Selection")
        ax.legend(frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _transformations(self) -> list[dict]:
        """Format transformations for the template."""
        items = []
        for name, (source, ttype) in self.results.transformations_used.items():
            items.append({"name": name, "source": source, "type": ttype})
        return sorted(items, key=lambda x: (x["source"], x["type"]))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimator_label(self) -> str:
        label = self._ESTIMATOR_LABELS.get(self.results.best_estimator)
        return label if label is not None else self.results.best_estimator

    def _best_classification(self) -> str:
        """Look up classification of the best model."""
        for info in self.results.all_results.values():
            if info.get("results") is self.results.best_model:
                cls = info.get("classification", "N/A")
                return cls if isinstance(cls, str) else "N/A"
        return "N/A"

    def _best_model_metrics(self) -> dict:
        """Return metrics dict of the best model."""
        for info in self.results.all_results.values():
            if info.get("results") is self.results.best_model:
                metrics = info.get("metrics", {})
                return {
                    k: self._fmt(v, 4 if "rsq" in k else 2)
                    for k, v in metrics.items()
                    if not isinstance(v, (int,))
                } | {k: v for k, v in metrics.items() if isinstance(v, (int,))}
        return {}

    def _best_diagnostics(self) -> dict:
        """Return diagnostic test results for the best model."""
        for name, info in self.results.all_results.items():
            if info.get("results") is self.results.best_model:
                tests = self.results.test_results.get(name, {})
                diags = {}
                classification = tests.get("classification", "N/A")
                diags["Classification"] = classification
                passed = tests.get("passed", [])
                failed = tests.get("failed", [])
                if passed:
                    diags["Tests Passed"] = ", ".join(passed)
                if failed:
                    diags["Tests Failed"] = ", ".join(failed)
                return diags
        return {}

    @staticmethod
    def _fmt(value: Any, decimals: int = 2) -> str:
        """Format a numeric value, returning 'N/A' for None/NaN."""
        if value is None:
            return "N/A"
        try:
            fval = float(value)
            if np.isnan(fval):
                return "N/A"
            return f"{fval:.{decimals}f}"
        except (TypeError, ValueError):
            return str(value)
