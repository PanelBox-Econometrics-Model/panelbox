"""
Markdown Exporter for PanelBox Reports.

Exports validation and regression results to Markdown format.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MarkdownExporter:
    """
    Exports PanelBox reports to Markdown format.

    Creates GitHub-flavored Markdown reports suitable for documentation,
    README files, and issue tracking.

    Parameters
    ----------
    include_toc : bool, default=True
        Include table of contents
    github_flavor : bool, default=True
        Use GitHub-flavored Markdown extensions

    Examples
    --------
    >>> from panelbox.report.exporters import MarkdownExporter
    >>>
    >>> exporter = MarkdownExporter()
    >>> md = exporter.export_validation_report(validation_data)
    >>> exporter.save(md, "validation_report.md")
    """

    def __init__(self, include_toc: bool = True, github_flavor: bool = True):
        """Initialize Markdown Exporter."""
        self.include_toc = include_toc
        self.github_flavor = github_flavor

    _SEVERITY_EMOJI = {"CRITICAL": "\U0001f534", "HIGH": "\U0001f7e0", "MEDIUM": "\U0001f7e1"}

    def _format_model_info(self, model_info: dict[str, Any]) -> list[str]:
        """Format model information section."""
        lines = [
            "## Model Information",
            "",
            f"- **Model Type:** {model_info.get('model_type', 'Unknown')}",
        ]
        if "formula" in model_info:
            lines.append(f"- **Formula:** `{model_info['formula']}`")
        lines.append(
            f"- **Observations:** {model_info.get('nobs_formatted', model_info.get('nobs', 'N/A'))}"
        )
        if "n_entities" in model_info:
            lines.append(
                f"- **Entities:** "
                f"{model_info.get('n_entities_formatted', model_info.get('n_entities'))}"
            )
        if "n_periods" in model_info:
            lines.append(
                f"- **Time Periods:** "
                f"{model_info.get('n_periods_formatted', model_info.get('n_periods'))}"
            )
        lines.append("")
        return lines

    def _format_test_results(self, tests: list[dict[str, Any]]) -> list[str]:
        """Format test results section grouped by category."""
        lines = ["## Test Results", ""]
        categories: dict[str, list] = {}
        for test in tests:
            categories.setdefault(test["category"], []).append(test)

        for category, cat_tests in categories.items():
            lines.extend([f"### {category}", ""])
            lines.append("| Test | Statistic | P-value | Result |")
            lines.append("|------|-----------|---------|--------|")
            for test in cat_tests:
                result_emoji = "\u274c REJECT" if test["result"] == "REJECT" else "\u2705 ACCEPT"
                sig = test.get("significance", "")
                lines.append(
                    f"| {test['name']} | {test['statistic_formatted']} "
                    f"| {test['pvalue_formatted']}{sig} | {result_emoji} |"
                )
            lines.append("")
        return lines

    def _format_recommendation(self, index: int, rec: dict[str, Any]) -> list[str]:
        """Format a single recommendation entry."""
        severity = rec["severity"].upper()
        emoji = self._SEVERITY_EMOJI.get(severity, "\U0001f535")
        lines = [
            f"### {index}. {emoji} {rec['category']} ({severity})",
            "",
            f"**Issue:** {rec['issue']}",
            "",
        ]
        if rec.get("tests"):
            lines.append("**Failed Tests:**")
            lines.extend(f"- {t}" for t in rec["tests"])
            lines.append("")
        if rec.get("suggestions"):
            lines.append("**Suggested Actions:**")
            lines.extend(f"1. {s}" for s in rec["suggestions"])
            lines.append("")
        return lines

    def export_validation_report(
        self, validation_data: dict[str, Any], title: str = "Validation Report"
    ) -> str:
        """
        Export complete validation report to Markdown.

        Parameters
        ----------
        validation_data : dict
            Validation data from ValidationTransformer
        title : str, default="Validation Report"
            Report title

        Returns
        -------
        str
            Markdown content

        Examples
        --------
        >>> md = exporter.export_validation_report(validation_data, title="Panel Data Validation")
        """
        lines = [f"# {title}", ""]

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.extend([f"**Generated:** {timestamp}", ""])

        # Summary
        summary = validation_data.get("summary", {})
        lines.extend(
            [
                "## Summary",
                "",
                f"- **Total Tests:** {summary.get('total_tests', 0)}",
                f"- **Passed:** {summary.get('total_passed', 0)} \u2705",
                f"- **Failed:** {summary.get('total_failed', 0)} \u274c",
                f"- **Pass Rate:** {summary.get('pass_rate_formatted', '0%')}",
                "",
            ]
        )

        if summary.get("has_issues", False):
            lines.append(f"> \u26a0\ufe0f **{summary.get('status_message', 'Issues detected')}**")
        else:
            lines.append(f"> \u2705 **{summary.get('status_message', 'All tests passed')}**")
        lines.append("")

        # TOC
        if self.include_toc:
            lines.extend(
                [
                    "## Table of Contents",
                    "",
                    "- [Model Information](#model-information)",
                    "- [Test Results](#test-results)",
                ]
            )
            if validation_data.get("recommendations"):
                lines.append("- [Recommendations](#recommendations)")
            lines.append("")

        # Model Information
        lines.extend(self._format_model_info(validation_data.get("model_info", {})))

        # Test Results
        tests = validation_data.get("tests", [])
        if tests:
            lines.extend(self._format_test_results(tests))

        # Recommendations
        recommendations = validation_data.get("recommendations", [])
        if recommendations:
            lines.extend(["## Recommendations", ""])
            for i, rec in enumerate(recommendations, 1):
                lines.extend(self._format_recommendation(i, rec))

        # Footer
        lines.extend(
            [
                "---",
                "",
                "*Generated with [PanelBox](https://github.com/panelbox/panelbox)*",
                "",
            ]
        )

        return "\n".join(lines)

    def export_validation_tests(self, tests: list[dict[str, Any]]) -> str:
        """
        Export validation tests as Markdown table.

        Parameters
        ----------
        tests : list of dict
            List of test results

        Returns
        -------
        str
            Markdown table

        Examples
        --------
        >>> md_table = exporter.export_validation_tests(tests)
        """
        lines = []

        # Table header
        lines.append("| Category | Test | Statistic | P-value | DF | Result |")
        lines.append("|----------|------|-----------|---------|----|----|")

        # Group by category
        categories: dict[str, list] = {}
        for test in tests:
            cat = test["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(test)

        # Table rows
        for category, cat_tests in categories.items():
            for i, test in enumerate(cat_tests):
                # Show category only for first test in group
                cat_display = category if i == 0 else ""

                name = test["name"]
                stat = test["statistic_formatted"]
                pval = test["pvalue_formatted"]
                sig = test.get("significance", "")
                df = test.get("df", "N/A")
                result = test["result"]

                # Result emoji
                result_emoji = "❌" if result == "REJECT" else "✅"

                lines.append(
                    f"| {cat_display} | {name} | {stat} | {pval}{sig} | {df} | {result_emoji} {result} |"
                )

        return "\n".join(lines)

    def export_regression_table(
        self,
        coefficients: list[dict[str, Any]],
        model_info: dict[str, Any],
        title: str = "Regression Results",
    ) -> str:
        """
        Export regression results as Markdown.

        Parameters
        ----------
        coefficients : list of dict
            Coefficient results
        model_info : dict
            Model information
        title : str, default="Regression Results"
            Table title

        Returns
        -------
        str
            Markdown table

        Examples
        --------
        >>> md = exporter.export_regression_table(coefs, info)
        """
        lines = []

        lines.append(f"## {title}")
        lines.append("")

        # Table header
        lines.append("| Variable | Coefficient | Std. Error | t-statistic | P-value |")
        lines.append("|----------|-------------|------------|-------------|---------|")

        # Coefficient rows
        for coef in coefficients:
            var = coef["variable"]
            beta = f"{coef['coefficient']:.4f}"
            se = f"{coef['std_error']:.4f}"
            tstat = f"{coef['t_statistic']:.3f}"
            pval = f"{coef['pvalue']:.4f}"

            # Significance stars
            if coef["pvalue"] < 0.001:
                stars = "***"
            elif coef["pvalue"] < 0.01:
                stars = "**"
            elif coef["pvalue"] < 0.05:
                stars = "*"
            else:
                stars = ""

            lines.append(f"| {var} | {beta}{stars} | ({se}) | {tstat} | {pval} |")

        lines.append("")

        # Model statistics
        lines.append("**Model Statistics:**")
        lines.append("")
        if "r_squared" in model_info:
            lines.append(f"- R²: {model_info['r_squared']:.4f}")
        if "nobs" in model_info:
            lines.append(f"- Observations: {model_info['nobs']}")
        if "n_entities" in model_info:
            lines.append(f"- Entities: {model_info['n_entities']}")
        lines.append("")

        # Note
        lines.append(
            "*Note:* Standard errors in parentheses. Significance: *** p<0.001, ** p<0.01, * p<0.05"
        )
        lines.append("")

        return "\n".join(lines)

    def export_summary_stats(
        self, stats: list[dict[str, Any]], title: str = "Summary Statistics"
    ) -> str:
        """
        Export summary statistics as Markdown.

        Parameters
        ----------
        stats : list of dict
            Variable statistics
        title : str, default="Summary Statistics"
            Table title

        Returns
        -------
        str
            Markdown table

        Examples
        --------
        >>> md = exporter.export_summary_stats(stats)
        """
        lines = []

        lines.append(f"## {title}")
        lines.append("")

        # Table header
        lines.append("| Variable | N | Mean | Std. Dev. | Min | Max |")
        lines.append("|----------|---|------|-----------|-----|-----|")

        # Data rows
        for stat in stats:
            var = stat["variable"]
            n = stat["count"]
            mean = f"{stat['mean']:.3f}"
            std = f"{stat['std']:.3f}"
            min_val = f"{stat['min']:.3f}"
            max_val = f"{stat['max']:.3f}"

            lines.append(f"| {var} | {n} | {mean} | {std} | {min_val} | {max_val} |")

        lines.append("")

        return "\n".join(lines)

    def save(
        self, markdown_content: str, output_path: str | Path, overwrite: bool = False
    ) -> Path:
        """
        Save Markdown content to file.

        Parameters
        ----------
        markdown_content : str
            Markdown content
        output_path : str or Path
            Output file path
        overwrite : bool, default=False
            Overwrite existing file

        Returns
        -------
        Path
            Path to saved file

        Examples
        --------
        >>> exporter.save(md, "report.md")
        """
        output_path = Path(output_path)

        # Check if file exists
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {output_path}. Use overwrite=True to replace."
            )

        # Create parent directories
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        output_path.write_text(markdown_content, encoding="utf-8")

        return output_path

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MarkdownExporter(include_toc={self.include_toc}, github_flavor={self.github_flavor})"
        )
