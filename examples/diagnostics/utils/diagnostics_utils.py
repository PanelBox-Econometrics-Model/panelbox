"""
General diagnostic utilities for formatting and export.

Provides helper functions to standardise the presentation of panel data
diagnostic test results, produce comparison tables, and export to LaTeX
or HTML report formats.

Functions:
- format_test_results: Standardise test output to formatted string
- create_results_table: Generate comparison table from dict of results
- export_to_latex: Export DataFrame to LaTeX table
- save_diagnostic_report: Save comprehensive HTML report
"""

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def format_test_results(
    test_name: str,
    statistic: float,
    pvalue: float,
    reject: bool,
    alpha: float = 0.05,
) -> str:
    """
    Standardise a single test result into a human-readable formatted string.

    Parameters
    ----------
    test_name : str
        Name of the diagnostic test (e.g. ``'LLC'``, ``'IPS'``).
    statistic : float
        Value of the test statistic.
    pvalue : float
        P-value associated with the test.
    reject : bool
        Whether the null hypothesis is rejected at the given ``alpha``.
    alpha : float, default 0.05
        Significance level used for the decision.

    Returns
    -------
    str
        Multi-line formatted string summarising the test outcome.

    Examples
    --------
    >>> print(format_test_results("LLC", -3.21, 0.001, True))
    ============================================================
    LLC Test
    ============================================================
    Test Statistic:  -3.2100
    P-value:          0.0010
    Significance:     0.05
    Decision:         Reject H0
    ============================================================
    """
    decision = "Reject H0" if reject else "Fail to Reject H0"

    # Significance stars for quick visual scanning
    if np.isfinite(pvalue):
        if pvalue < 0.01:
            stars = "***"
        elif pvalue < 0.05:
            stars = "**"
        elif pvalue < 0.10:
            stars = "*"
        else:
            stars = ""
    else:
        stars = ""

    separator = "=" * 60
    lines = [
        separator,
        f"{test_name} Test",
        separator,
        f"Test Statistic:  {statistic:>10.4f}",
        f"P-value:         {pvalue:>10.4f}  {stars}",
        f"Significance:    {alpha:>10.2f}",
        f"Decision:         {decision}",
        separator,
    ]
    return "\n".join(lines)


def create_results_table(
    results_dict: dict[str, dict[str, Any]],
    format: str = "dataframe",
) -> pd.DataFrame:
    """
    Generate a comparison table from a dictionary of test results.

    The returned DataFrame has columns **Test**, **H0**, **Statistic**,
    **P-value**, and **Decision**, which makes it straightforward to
    display in a Jupyter notebook or export to LaTeX/HTML.

    Parameters
    ----------
    results_dict : dict of dict
        Mapping ``{test_name: info_dict}`` where each *info_dict* should
        contain at least ``'statistic'`` and ``'pvalue'``.  Optional keys:

        * ``'H0'`` -- null-hypothesis description (str).
        * ``'reject'`` -- pre-computed boolean decision.
        * ``'alpha'`` -- significance level (float, default 0.05).

        If ``'reject'`` is absent, the decision is derived automatically
        from ``pvalue < alpha``.
    format : str, default 'dataframe'
        Output format.  Currently only ``'dataframe'`` is supported
        (returns a ``pd.DataFrame``).

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per test.

    Examples
    --------
    >>> results = {
    ...     "LLC": {"statistic": -3.21, "pvalue": 0.001, "H0": "Common unit root"},
    ...     "IPS": {"statistic": -2.80, "pvalue": 0.003, "H0": "Individual unit root"},
    ... }
    >>> df = create_results_table(results)
    >>> print(df.to_string(index=False))
    """
    rows: list[dict[str, Any]] = []

    for test_name, info in results_dict.items():
        statistic = info.get("statistic", np.nan)
        pvalue = info.get("pvalue", np.nan)
        h0 = info.get("H0", "Not specified")
        test_alpha = info.get("alpha", 0.05)

        # Determine reject decision
        if "reject" in info:
            reject = bool(info["reject"])
        else:
            reject = bool(np.isfinite(pvalue) and pvalue < test_alpha)

        # Significance stars
        if np.isfinite(pvalue):
            if pvalue < 0.01:
                stars = "***"
            elif pvalue < 0.05:
                stars = "**"
            elif pvalue < 0.10:
                stars = "*"
            else:
                stars = ""
        else:
            stars = ""

        decision = "Reject H0" if reject else "Fail to Reject H0"

        rows.append(
            {
                "Test": test_name,
                "H0": h0,
                "Statistic": statistic,
                "P-value": pvalue,
                "Stars": stars,
                "Decision": decision,
            }
        )

    df = pd.DataFrame(rows)

    # Format numeric columns for display while keeping the original types
    # (formatting is deferred to export functions / notebook rendering)
    return df


def export_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    output_path: str,
) -> str:
    r"""
    Export a DataFrame to a LaTeX table with proper formatting.

    Wraps the table in a ``table`` float environment with ``\\centering``,
    caption, and label.  Numeric columns are formatted to four decimal
    places, and significance stars are preserved.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export (e.g. from :func:`create_results_table`).
    caption : str
        LaTeX caption for the table.
    label : str
        LaTeX label for cross-referencing (e.g. ``'tab:unit_root'``).
    output_path : str
        File path where the ``.tex`` file is written.

    Returns
    -------
    str
        The complete LaTeX table as a string (also written to *output_path*).

    Examples
    --------
    >>> latex_str = export_to_latex(
    ...     df,
    ...     caption="Unit Root Tests",
    ...     label="tab:unit_root",
    ...     output_path="results/unit_root_table.tex",
    ... )
    """
    # Determine column alignment: 'l' for strings, 'c' for numbers
    alignments = []
    for col in df.columns:
        if df[col].dtype == object:
            alignments.append("l")
        else:
            alignments.append("c")

    col_spec = "".join(alignments)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline\hline",
    ]

    # Header row
    header = " & ".join(col.replace("_", r"\_") for col in df.columns)
    lines.append(f"{header} \\\\")
    lines.append(r"\hline")

    # Data rows
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                if np.isfinite(val):
                    cells.append(f"{val:.4f}")
                else:
                    cells.append("--")
            else:
                # Escape underscores in string values for LaTeX
                cells.append(str(val).replace("_", r"\_"))
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(
        [
            r"\hline\hline",
            r"\multicolumn{"
            + str(len(df.columns))
            + r"}{l}{\small *** $p<0.01$, ** $p<0.05$, * $p<0.10$} \\",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    latex_str = "\n".join(lines)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    return latex_str


def save_diagnostic_report(
    results: dict[str, dict[str, Any]],
    output_path: str,
    title: str = "Diagnostic Report",
) -> str:
    """
    Save a comprehensive HTML report with all test results and figures.

    The report includes:

    * Executive summary with overall pass/fail counts.
    * Detailed results table (test name, null hypothesis, statistic,
      p-value, decision).
    * Interpretation section with colour-coded decisions.
    * Footer with generation timestamp.

    Parameters
    ----------
    results : dict of dict
        Same structure as :func:`create_results_table` expects.  Each
        value dict should contain ``'statistic'``, ``'pvalue'``, and
        optionally ``'H0'``, ``'reject'``, ``'alpha'``, and
        ``'details'`` (a free-text string appended under the test row).
    output_path : str
        File path for the generated ``.html`` file.
    title : str, default 'Diagnostic Report'
        Title shown at the top of the report.

    Returns
    -------
    str
        Absolute path to the generated report file.

    Examples
    --------
    >>> results = {
    ...     "LLC": {"statistic": -3.21, "pvalue": 0.001, "H0": "Common unit root", "reject": True},
    ...     "IPS": {
    ...         "statistic": -2.80,
    ...         "pvalue": 0.003,
    ...         "H0": "Individual unit root",
    ...         "reject": True,
    ...     },
    ... }
    >>> path = save_diagnostic_report(results, "report.html")
    """
    # Build results table DataFrame for rendering
    table_df = create_results_table(results)

    n_total = len(table_df)
    n_reject = (table_df["Decision"] == "Reject H0").sum()
    n_fail = n_total - n_reject

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- HTML construction ----
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        f"  <title>{title}</title>",
        "  <meta charset='UTF-8'>",
        "  <style>",
        "    body {",
        "      font-family: 'Segoe UI', Arial, sans-serif;",
        "      margin: 40px;",
        "      color: #333;",
        "      line-height: 1.6;",
        "    }",
        "    h1 {",
        "      color: #2c3e50;",
        "      border-bottom: 3px solid #3498db;",
        "      padding-bottom: 10px;",
        "    }",
        "    h2 {",
        "      color: #2980b9;",
        "      margin-top: 30px;",
        "    }",
        "    table {",
        "      border-collapse: collapse;",
        "      width: 100%;",
        "      margin: 15px 0;",
        "    }",
        "    th, td {",
        "      border: 1px solid #ddd;",
        "      padding: 10px 14px;",
        "      text-align: left;",
        "    }",
        "    th {",
        "      background-color: #3498db;",
        "      color: white;",
        "      font-weight: 600;",
        "    }",
        "    tr:nth-child(even) { background-color: #f2f2f2; }",
        "    tr:hover { background-color: #e8f4f8; }",
        "    .reject { color: #27ae60; font-weight: bold; }",
        "    .fail   { color: #e74c3c; font-weight: bold; }",
        "    .summary-box {",
        "      background-color: #ecf0f1;",
        "      padding: 20px;",
        "      border-radius: 8px;",
        "      margin: 15px 0;",
        "      display: flex;",
        "      gap: 30px;",
        "    }",
        "    .summary-item {",
        "      text-align: center;",
        "      flex: 1;",
        "    }",
        "    .summary-item .number {",
        "      font-size: 28px;",
        "      font-weight: bold;",
        "    }",
        "    .summary-item .label {",
        "      font-size: 14px;",
        "      color: #666;",
        "    }",
        "    .details-text {",
        "      background-color: #f8f9fa;",
        "      padding: 12px;",
        "      border-left: 3px solid #3498db;",
        "      margin: 5px 0 15px 0;",
        "      font-size: 13px;",
        "    }",
        "    .footer {",
        "      margin-top: 40px;",
        "      padding-top: 10px;",
        "      border-top: 1px solid #ddd;",
        "      color: #999;",
        "      font-size: 12px;",
        "    }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{title}</h1>",
        f"  <p>Generated: {timestamp}</p>",
        "",
        "  <h2>1. Executive Summary</h2>",
        "  <div class='summary-box'>",
        "    <div class='summary-item'>",
        f"      <div class='number'>{n_total}</div>",
        "      <div class='label'>Total Tests</div>",
        "    </div>",
        "    <div class='summary-item'>",
        f"      <div class='number' style='color:#27ae60;'>{n_reject}</div>",
        "      <div class='label'>Reject H0</div>",
        "    </div>",
        "    <div class='summary-item'>",
        f"      <div class='number' style='color:#e74c3c;'>{n_fail}</div>",
        "      <div class='label'>Fail to Reject H0</div>",
        "    </div>",
        "  </div>",
        "",
        "  <h2>2. Detailed Results</h2>",
        "  <table>",
        "    <thead>",
        "      <tr>",
        "        <th>Test</th>",
        "        <th>Null Hypothesis (H0)</th>",
        "        <th>Statistic</th>",
        "        <th>P-value</th>",
        "        <th>Significance</th>",
        "        <th>Decision</th>",
        "      </tr>",
        "    </thead>",
        "    <tbody>",
    ]

    for _, row in table_df.iterrows():
        test_name = row["Test"]
        h0 = row["H0"]
        stat = row["Statistic"]
        pval = row["P-value"]
        stars = row["Stars"]
        decision = row["Decision"]

        stat_str = f"{stat:.4f}" if np.isfinite(stat) else "--"
        pval_str = f"{pval:.4f}" if np.isfinite(pval) else "--"
        decision_class = "reject" if decision == "Reject H0" else "fail"

        html_parts.append("      <tr>")
        html_parts.append(f"        <td><strong>{test_name}</strong></td>")
        html_parts.append(f"        <td>{h0}</td>")
        html_parts.append(f"        <td>{stat_str}</td>")
        html_parts.append(f"        <td>{pval_str}</td>")
        html_parts.append(f"        <td>{stars}</td>")
        html_parts.append(f"        <td class='{decision_class}'>{decision}</td>")
        html_parts.append("      </tr>")

        # Optional details row
        details = results[test_name].get("details", "")
        if details:
            html_parts.append("      <tr>")
            html_parts.append(f"        <td colspan='6' class='details-text'>{details}</td>")
            html_parts.append("      </tr>")

    html_parts.extend(
        [
            "    </tbody>",
            "  </table>",
            "",
            "  <h2>3. Interpretation Guide</h2>",
            "  <ul>",
            "    <li><strong>Reject H0</strong>: Sufficient statistical evidence "
            "against the null hypothesis at the chosen significance level.</li>",
            "    <li><strong>Fail to Reject H0</strong>: Insufficient evidence "
            "to reject the null hypothesis.  This does <em>not</em> prove H0 "
            "is true.</li>",
            "    <li>Significance stars: *** p&lt;0.01, ** p&lt;0.05, * p&lt;0.10</li>",
            "  </ul>",
            "",
            "  <div class='footer'>",
            "    <p>Report generated using PanelBox Diagnostics module.</p>",
            f"    <p>Timestamp: {timestamp}</p>",
            "  </div>",
            "</body>",
            "</html>",
        ]
    )

    html_content = "\n".join(html_parts)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "create_results_table",
    "export_to_latex",
    "format_test_results",
    "save_diagnostic_report",
]


if __name__ == "__main__":
    print("Diagnostics utilities module loaded successfully!")
    print("Functions available:")
    print("  - format_test_results")
    print("  - create_results_table")
    print("  - export_to_latex")
    print("  - save_diagnostic_report")
