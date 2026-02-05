"""
LaTeX Exporter for PanelBox Reports.

Exports validation and regression results to LaTeX format for academic papers.
"""

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class LaTeXExporter:
    """
    Exports PanelBox reports to LaTeX format.

    Creates publication-ready LaTeX tables for academic papers.
    Supports validation test results, regression tables, and summary statistics.

    Parameters
    ----------
    table_style : str, default='booktabs'
        LaTeX table style: 'booktabs', 'standard', or 'threeparttable'
    float_format : str, default='.3f'
        Float formatting string
    escape_special_chars : bool, default=True
        Escape LaTeX special characters in strings

    Examples
    --------
    >>> from panelbox.report.exporters import LaTeXExporter
    >>>
    >>> exporter = LaTeXExporter(table_style='booktabs')
    >>> latex = exporter.export_validation_tests(tests)
    >>> exporter.save(latex, 'validation_table.tex')
    """

    def __init__(
        self,
        table_style: str = "booktabs",
        float_format: str = ".3f",
        escape_special_chars: bool = True,
    ):
        """Initialize LaTeX Exporter."""
        if table_style not in ("booktabs", "standard", "threeparttable"):
            raise ValueError(
                f"Invalid table_style: {table_style}. "
                "Must be 'booktabs', 'standard', or 'threeparttable'."
            )

        self.table_style = table_style
        self.float_format = float_format
        self.escape_special_chars = escape_special_chars

    def export_validation_tests(
        self,
        tests: List[Dict[str, Any]],
        caption: str = "Validation Test Results",
        label: str = "tab:validation",
    ) -> str:
        """
        Export validation tests to LaTeX table.

        Parameters
        ----------
        tests : list of dict
            List of test results (from ValidationTransformer)
        caption : str, default="Validation Test Results"
            Table caption
        label : str, default="tab:validation"
            LaTeX label for cross-referencing

        Returns
        -------
        str
            LaTeX table code

        Examples
        --------
        >>> latex = exporter.export_validation_tests(
        ...     tests,
        ...     caption="Panel Data Validation Tests",
        ...     label="tab:validation"
        ... )
        """
        lines = []

        # Begin table
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"    \centering")
        lines.append(f"    \\caption{{{caption}}}")
        lines.append(f"    \\label{{{label}}}")

        if self.table_style == "booktabs":
            lines.append(r"    \begin{tabular}{lcccc}")
            lines.append(r"        \toprule")
        else:
            lines.append(r"    \begin{tabular}{|l|c|c|c|c|}")
            lines.append(r"        \hline")

        # Header
        lines.append(r"        Test & Statistic & P-value & DF & Result \\")

        if self.table_style == "booktabs":
            lines.append(r"        \midrule")
        else:
            lines.append(r"        \hline")

        # Group by category
        categories = {}
        for test in tests:
            cat = test["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(test)

        # Add rows by category
        for category, cat_tests in categories.items():
            # Category header
            if self.table_style == "booktabs":
                lines.append(f"        \\multicolumn{{5}}{{l}}{{\\textit{{{category}}}}} \\\\")
            else:
                lines.append(f"        \\multicolumn{{5}}{{|l|}}{{\\textbf{{{category}}}}} \\\\")
                lines.append(r"        \hline")

            # Test rows
            for test in cat_tests:
                name = self._escape(test["name"])
                stat = self._format_float(test["statistic"])
                pval = self._format_pvalue(test["pvalue"])
                df = test["df"] if test["df"] else "--"
                result = test["result"]

                # Add significance stars
                stars = test.get("significance", "")

                lines.append(f"        {name} & {stat} & {pval}{stars} & {df} & {result} \\\\")

            if self.table_style != "booktabs":
                lines.append(r"        \hline")

        # End table
        if self.table_style == "booktabs":
            lines.append(r"        \bottomrule")
        else:
            lines.append(r"        \hline")

        lines.append(r"    \end{tabular}")

        # Add notes
        if self.table_style == "booktabs":
            lines.append(r"    \medskip")
            lines.append(r"    \begin{minipage}{\textwidth}")
            lines.append(r"        \small")
            lines.append(
                r"        \textit{Note:} Significance levels: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1."
            )
            lines.append(r"    \end{minipage}")

        lines.append(r"\end{table}")

        return "\n".join(lines)

    def export_regression_table(
        self,
        coefficients: List[Dict[str, Any]],
        model_info: Dict[str, Any],
        caption: str = "Regression Results",
        label: str = "tab:regression",
    ) -> str:
        """
        Export regression results to LaTeX table.

        Parameters
        ----------
        coefficients : list of dict
            List of coefficient results
        model_info : dict
            Model information (R², N, etc.)
        caption : str, default="Regression Results"
            Table caption
        label : str, default="tab:regression"
            LaTeX label

        Returns
        -------
        str
            LaTeX table code

        Examples
        --------
        >>> latex = exporter.export_regression_table(
        ...     coefficients=coefs,
        ...     model_info=info,
        ...     caption="Fixed Effects Regression"
        ... )
        """
        lines = []

        # Begin table
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"    \centering")
        lines.append(f"    \\caption{{{caption}}}")
        lines.append(f"    \\label{{{label}}}")

        if self.table_style == "booktabs":
            lines.append(r"    \begin{tabular}{lcccc}")
            lines.append(r"        \toprule")
        else:
            lines.append(r"    \begin{tabular}{|l|c|c|c|c|}")
            lines.append(r"        \hline")

        # Header
        lines.append(r"        Variable & Coefficient & Std. Error & t-statistic & P-value \\")

        if self.table_style == "booktabs":
            lines.append(r"        \midrule")
        else:
            lines.append(r"        \hline")

        # Coefficient rows
        for coef in coefficients:
            var = self._escape(coef["variable"])
            beta = self._format_float(coef["coefficient"])
            se = self._format_float(coef["std_error"])
            tstat = self._format_float(coef["t_statistic"])
            pval = self._format_pvalue(coef["pvalue"])

            # Add significance stars
            stars = self._get_stars(coef["pvalue"])

            lines.append(f"        {var} & {beta}{stars} & ({se}) & {tstat} & {pval} \\\\")

        if self.table_style == "booktabs":
            lines.append(r"        \midrule")
        else:
            lines.append(r"        \hline")

        # Model statistics
        lines.append(r"        \multicolumn{5}{l}{\textit{Model Statistics}} \\")

        if "r_squared" in model_info:
            r2 = self._format_float(model_info["r_squared"])
            lines.append(f"        R² & \\multicolumn{{4}}{{c}}{{{r2}}} \\\\")

        if "nobs" in model_info:
            nobs = model_info["nobs"]
            lines.append(f"        Observations & \\multicolumn{{4}}{{c}}{{{nobs}}} \\\\")

        if "n_entities" in model_info:
            n_ent = model_info["n_entities"]
            lines.append(f"        Entities & \\multicolumn{{4}}{{c}}{{{n_ent}}} \\\\")

        # End table
        if self.table_style == "booktabs":
            lines.append(r"        \bottomrule")
        else:
            lines.append(r"        \hline")

        lines.append(r"    \end{tabular}")

        # Add notes
        if self.table_style == "booktabs":
            lines.append(r"    \medskip")
            lines.append(r"    \begin{minipage}{\textwidth}")
            lines.append(r"        \small")
            lines.append(r"        \textit{Note:} Standard errors in parentheses. ")
            lines.append(r"        Significance levels: *** p<0.001, ** p<0.01, * p<0.05.")
            lines.append(r"    \end{minipage}")

        lines.append(r"\end{table}")

        return "\n".join(lines)

    def export_summary_stats(
        self,
        stats: List[Dict[str, Any]],
        caption: str = "Summary Statistics",
        label: str = "tab:summary",
    ) -> str:
        """
        Export summary statistics to LaTeX table.

        Parameters
        ----------
        stats : list of dict
            List of variable statistics
        caption : str, default="Summary Statistics"
            Table caption
        label : str, default="tab:summary"
            LaTeX label

        Returns
        -------
        str
            LaTeX table code

        Examples
        --------
        >>> latex = exporter.export_summary_stats(
        ...     stats,
        ...     caption="Descriptive Statistics"
        ... )
        """
        lines = []

        # Begin table
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"    \centering")
        lines.append(f"    \\caption{{{caption}}}")
        lines.append(f"    \\label{{{label}}}")

        if self.table_style == "booktabs":
            lines.append(r"    \begin{tabular}{lcccccc}")
            lines.append(r"        \toprule")
        else:
            lines.append(r"    \begin{tabular}{|l|c|c|c|c|c|c|}")
            lines.append(r"        \hline")

        # Header
        lines.append(r"        Variable & N & Mean & Std. Dev. & Min & Max & Median \\")

        if self.table_style == "booktabs":
            lines.append(r"        \midrule")
        else:
            lines.append(r"        \hline")

        # Data rows
        for stat in stats:
            var = self._escape(stat["variable"])
            n = stat["count"]
            mean = self._format_float(stat["mean"])
            std = self._format_float(stat["std"])
            min_val = self._format_float(stat["min"])
            max_val = self._format_float(stat["max"])
            median = self._format_float(stat.get("median", stat.get("50%", 0)))

            lines.append(
                f"        {var} & {n} & {mean} & {std} & {min_val} & {max_val} & {median} \\\\"
            )

        # End table
        if self.table_style == "booktabs":
            lines.append(r"        \bottomrule")
        else:
            lines.append(r"        \hline")

        lines.append(r"    \end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def save(
        self,
        latex_content: str,
        output_path: Union[str, Path],
        overwrite: bool = False,
        add_preamble: bool = False,
    ) -> Path:
        """
        Save LaTeX content to file.

        Parameters
        ----------
        latex_content : str
            LaTeX content
        output_path : str or Path
            Output file path
        overwrite : bool, default=False
            Overwrite existing file
        add_preamble : bool, default=False
            Add complete LaTeX document preamble

        Returns
        -------
        Path
            Path to saved file

        Examples
        --------
        >>> exporter.save(latex, 'table.tex')
        >>> # With preamble for standalone compilation
        >>> exporter.save(latex, 'table.tex', add_preamble=True)
        """
        output_path = Path(output_path)

        # Check if file exists
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {output_path}. " "Use overwrite=True to replace."
            )

        # Create parent directories
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add preamble if requested
        if add_preamble:
            latex_content = self._add_preamble(latex_content)

        # Write file
        output_path.write_text(latex_content, encoding="utf-8")

        return output_path

    def _add_preamble(self, content: str) -> str:
        """
        Add LaTeX preamble for standalone document.

        Parameters
        ----------
        content : str
            LaTeX table content

        Returns
        -------
        str
            Complete LaTeX document
        """
        preamble = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{margin=1in}

\begin{document}

"""

        postamble = r"""

\end{document}
"""

        return preamble + content + postamble

    def _escape(self, text: str) -> str:
        """
        Escape LaTeX special characters.

        Parameters
        ----------
        text : str
            Text to escape

        Returns
        -------
        str
            Escaped text
        """
        if not self.escape_special_chars:
            return text

        # LaTeX special characters
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
            "\\": r"\textbackslash{}",
        }

        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        return text

    def _format_float(self, value: float) -> str:
        """Format float value."""
        try:
            return f"{value:{self.float_format}}"
        except (ValueError, TypeError):
            return str(value)

    def _format_pvalue(self, pvalue: float) -> str:
        """Format p-value."""
        try:
            if pvalue < 0.001:
                return f"{pvalue:.2e}"
            return f"{pvalue:.4f}"
        except (ValueError, TypeError):
            return str(pvalue)

    def _get_stars(self, pvalue: float) -> str:
        """Get significance stars."""
        try:
            if pvalue < 0.001:
                return r"^{***}"
            elif pvalue < 0.01:
                return r"^{**}"
            elif pvalue < 0.05:
                return r"^{*}"
            return ""
        except (ValueError, TypeError):
            return ""

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LaTeXExporter("
            f"table_style='{self.table_style}', "
            f"float_format='{self.float_format}')"
        )
