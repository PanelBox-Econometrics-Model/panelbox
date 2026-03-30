"""AutoExperimentResults: container for AutoExperiment pipeline outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class AutoExperimentResults:
    """Container for all results produced by :class:`AutoExperiment`.

    Parameters
    ----------
    best_model : PanelResults or None
        Fitted result object for the best model.
    best_formula : str
        Formula of the best model.
    best_estimator : str
        Canonical model type of the best model (``'fe'``, ``'re'``, etc.).
    best_cov_type : str
        Standard-error type auto-selected for the best model.
    ranking : pd.DataFrame
        Model ranking table sorted by composite score.
    all_results : dict
        ``{model_name: {'results': PanelResults, 'metrics': dict, ...}}``.
    transformations_used : dict
        ``{transformed_var: (source_var, transformation_type)}``.
    variable_selection : dict
        ``{model_type: {'selected_vars': [...], 'rejected_vars': {...}, 'bic_path': [...]}}``.
    test_results : dict
        ``{model_name: {'classification': str, 'passed': [...], 'failed': [...]}}``.
    n_combinations_tested : int
        Total number of model/variable combinations evaluated.
    datamining_warning : bool
        ``True`` if > 100 combinations were tested.
    """

    _ESTIMATOR_LABELS = {
        "pooled_ols": "Pooled OLS",
        "fe": "Fixed Effects",
        "re": "Random Effects",
        "fd": "First Difference",
    }

    def __init__(
        self,
        best_model: Any | None,
        best_formula: str,
        best_estimator: str,
        best_cov_type: str,
        ranking: pd.DataFrame,
        all_results: dict,
        transformations_used: dict,
        variable_selection: dict,
        test_results: dict,
        n_combinations_tested: int,
        datamining_warning: bool,
    ):
        self.best_model = best_model
        self.best_formula = best_formula
        self.best_estimator = best_estimator
        self.best_cov_type = best_cov_type
        self.ranking = ranking
        self.all_results = all_results
        self.transformations_used = transformations_used
        self.variable_selection = variable_selection
        self.test_results = test_results
        self.n_combinations_tested = n_combinations_tested
        self.datamining_warning = datamining_warning

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the experiment results."""
        sep_heavy = "\u2550" * 50
        sep_light = "\u2500" * 50

        label = self._ESTIMATOR_LABELS.get(self.best_estimator, self.best_estimator)
        classification = "N/A"
        if self.best_model is not None:
            # Find the best model's classification
            for _name, info in self.all_results.items():
                if info.get("results") is self.best_model:
                    classification = info.get("classification", "N/A")
                    break

        # Best model metrics
        best_bic = ""
        best_rsq = ""
        if not self.ranking.empty:
            valid = self.ranking[self.ranking["classification"].isin(["VALID", "WARNING"])]
            if not valid.empty:
                row = valid.iloc[0]
                best_bic = f"{row['bic']:.1f}"
                rsq_val = row.get("rsq")
                if rsq_val is not None and not (isinstance(rsq_val, float) and np.isnan(rsq_val)):
                    best_rsq = f"{rsq_val:.3f}"
                else:
                    best_rsq = "N/A"

        # Count unique transformation types
        transform_types = set()
        for _, (_, ttype) in self.transformations_used.items():
            transform_types.add(ttype.rstrip("0123456789"))

        # Count variables tested (all candidates across selectors)
        n_vars_tested = 0
        for _, sel in self.variable_selection.items():
            n_selected = len(sel.get("selected_vars", []))
            n_rejected = len(sel.get("rejected_vars", {}))
            n_vars_tested = max(n_vars_tested, n_selected + n_rejected)

        # Status
        if classification == "VALID":
            status = "VALID (all tests passed)"
        elif classification == "WARNING":
            status = "WARNING (some tests failed)"
        elif classification == "INVALID":
            status = "INVALID (critical test failures)"
        else:
            status = classification

        lines = [
            sep_heavy,
            "AutoExperiment Results",
            sep_heavy,
            f"Best model: {label}",
            f"Formula:    {self.best_formula}",
            f"BIC:        {best_bic}",
            f"R-squared:  {best_rsq}",
            f"Cov type:   {self.best_cov_type}",
            f"Status:     {status}",
            sep_light,
            f"Models tested:     {len(self.all_results)}",
            f"Combinations:      {self.n_combinations_tested}",
            f"Variables tested:  {n_vars_tested}",
            f"Transformations:   {len(transform_types)} types",
        ]

        if self.datamining_warning:
            lines.append(f"  Data mining warning: {self.n_combinations_tested} combinations tested")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def compare_top(self, n: int = 5) -> pd.DataFrame:
        """Return the top *n* models with key metrics side by side.

        Parameters
        ----------
        n : int
            Number of top models to include.

        Returns
        -------
        pd.DataFrame
            Columns: model_type, formula, bic, aic, rsq, rsq_adj,
            classification, n_vars, cov_type, score.
        """
        if self.ranking.empty:
            return pd.DataFrame()
        cols = [
            "model_type",
            "formula",
            "bic",
            "aic",
            "rsq",
            "rsq_adj",
            "classification",
            "n_vars",
            "cov_type",
            "score",
        ]
        available = [c for c in cols if c in self.ranking.columns]
        return self.ranking[available].head(n).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_bic_comparison(self):
        """Barplot of BIC by model.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting.")
            return None

        if self.ranking.empty:
            return None

        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [f"{row['model_type']}" for _, row in self.ranking.iterrows()]
        bics = self.ranking["bic"].values
        colors = [
            "#2ecc71" if c == "VALID" else "#f39c12" if c == "WARNING" else "#e74c3c"
            for c in self.ranking["classification"]
        ]

        ax.barh(labels, bics, color=colors)
        ax.set_xlabel("BIC")
        ax.set_title("BIC Comparison by Model")
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    def plot_variable_importance(self):
        """Barplot of how many times each variable was selected across models.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting.")
            return None

        counts: dict[str, int] = {}
        for _, sel in self.variable_selection.items():
            for var in sel.get("selected_vars", []):
                counts[var] = counts.get(var, 0) + 1

        if not counts:
            return None

        sorted_vars = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        names = [v for v, _ in sorted_vars]
        vals = [c for _, c in sorted_vars]

        fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.4)))
        ax.barh(names, vals, color="#3498db")
        ax.set_xlabel("Times selected")
        ax.set_title("Variable Importance (selection frequency)")
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # HTML Report
    # ------------------------------------------------------------------

    def report(self, filepath: str | None = None) -> str:
        """Generate a comprehensive HTML report.

        Parameters
        ----------
        filepath : str, optional
            If provided, save the report to this path.

        Returns
        -------
        str
            HTML content.
        """
        html_parts = [
            "<!DOCTYPE html>",
            '<html lang="en"><head>',
            '<meta charset="utf-8">',
            "<title>AutoExperiment Report</title>",
            "<style>",
            "body { font-family: 'Segoe UI', Tahoma, sans-serif; margin: 40px; "
            "background: #f5f6fa; color: #2c3e50; }",
            "h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }",
            "h2 { color: #34495e; margin-top: 30px; }",
            "table { border-collapse: collapse; width: 100%; margin: 15px 0; "
            "background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
            "th, td { padding: 10px 14px; text-align: left; border: 1px solid #ddd; }",
            "th { background: #3498db; color: white; }",
            "tr:nth-child(even) { background: #f8f9fa; }",
            ".valid { color: #27ae60; font-weight: bold; }",
            ".warning { color: #f39c12; font-weight: bold; }",
            ".invalid { color: #e74c3c; font-weight: bold; }",
            ".summary-box { background: white; padding: 20px; border-radius: 8px; "
            "box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }",
            ".metric { font-size: 1.2em; margin: 5px 0; }",
            ".datamining { background: #fff3cd; border: 1px solid #ffc107; "
            "padding: 15px; border-radius: 5px; margin: 15px 0; }",
            "</style>",
            "</head><body>",
            "<h1>AutoExperiment Report</h1>",
        ]

        # Summary box
        label = self._ESTIMATOR_LABELS.get(self.best_estimator, self.best_estimator)
        html_parts.append('<div class="summary-box">')
        html_parts.append(f'<p class="metric"><strong>Best Model:</strong> {label}</p>')
        html_parts.append(
            f'<p class="metric"><strong>Formula:</strong> <code>{self.best_formula}</code></p>'
        )
        html_parts.append(f'<p class="metric"><strong>Cov Type:</strong> {self.best_cov_type}</p>')
        html_parts.append(
            f'<p class="metric"><strong>Combinations Tested:</strong> '
            f"{self.n_combinations_tested}</p>"
        )
        html_parts.append("</div>")

        # Data mining warning
        if self.datamining_warning:
            html_parts.append('<div class="datamining">')
            html_parts.append(
                f"<strong>Data Mining Warning:</strong> "
                f"{self.n_combinations_tested} combinations were tested. "
                f"Results should be validated on out-of-sample data."
            )
            html_parts.append("</div>")

        # Ranking table
        if not self.ranking.empty:
            html_parts.append("<h2>Model Ranking</h2>")
            html_parts.append("<table>")
            html_parts.append(
                "<tr><th>#</th><th>Model</th><th>Formula</th>"
                "<th>BIC</th><th>AIC</th><th>R&sup2;</th>"
                "<th>Status</th><th>Cov Type</th><th>Score</th></tr>"
            )
            for i, row in self.ranking.iterrows():
                cls_class = row["classification"].lower()
                rsq = row["rsq"]
                rsq_str = (
                    f"{rsq:.3f}"
                    if rsq is not None and not (isinstance(rsq, float) and np.isnan(rsq))
                    else "N/A"
                )
                html_parts.append(
                    f"<tr><td>{i + 1}</td>"
                    f"<td>{row['model_type']}</td>"
                    f"<td><code>{row['formula']}</code></td>"
                    f"<td>{row['bic']:.2f}</td>"
                    f"<td>{row['aic']:.2f}</td>"
                    f"<td>{rsq_str}</td>"
                    f'<td class="{cls_class}">{row["classification"]}</td>'
                    f"<td>{row['cov_type']}</td>"
                    f"<td>{row['score']:.2f}</td></tr>"
                )
            html_parts.append("</table>")

        # Variable selection details
        html_parts.append("<h2>Variable Selection</h2>")
        for model_type, sel in self.variable_selection.items():
            selected = sel.get("selected_vars", [])
            html_parts.append(f"<h3>{model_type}</h3>")
            html_parts.append(
                f"<p>Selected: <code>{', '.join(selected) if selected else 'None'}</code></p>"
            )

        # Test results
        html_parts.append("<h2>Diagnostic Tests</h2>")
        for name, tests in self.test_results.items():
            cls_class = tests["classification"].lower()
            html_parts.append(
                f"<p><strong>{name}</strong>: "
                f'<span class="{cls_class}">{tests["classification"]}</span>'
            )
            if tests["failed"]:
                html_parts.append(f" — Failed: {', '.join(tests['failed'])}")
            html_parts.append("</p>")

        html_parts.append("</body></html>")
        html = "\n".join(html_parts)

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)

        return html
