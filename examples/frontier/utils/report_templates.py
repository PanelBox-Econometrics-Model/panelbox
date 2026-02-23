"""
Report generation templates for SFA analysis.

Functions:
- generate_efficiency_report: Comprehensive HTML report from SFA results
- estimation_table_latex: LaTeX table of estimation results
- model_comparison_table: Comparison table across SFA specifications
- efficiency_ranking_table: Ranked efficiency table for top/bottom entities
"""

from datetime import datetime
from typing import Optional

import pandas as pd


def generate_efficiency_report(
    result,
    output_path: str,
    title: str = "Efficiency Analysis Report",
    include_plots: bool = True,
) -> str:
    """
    Generate comprehensive HTML report from SFA results.

    Includes estimation summary, efficiency distribution, rankings, and diagnostics.

    Parameters
    ----------
    result : SFResult or PanelSFResult
        Fitted SFA result object.
    output_path : str
        Path to save the HTML report.
    title : str
        Report title.
    include_plots : bool
        Whether to include embedded plots.

    Returns
    -------
    str
        Path to the generated report.
    """
    # Get summary
    try:
        summary_text = result.summary()
    except Exception:
        summary_text = "Summary not available."

    # Get efficiency
    try:
        eff_df = result.efficiency(estimator="bc")
        eff_col = "efficiency"
        if eff_col not in eff_df.columns:
            eff_col = next(c for c in eff_df.columns if "effic" in c.lower())
        eff_stats = eff_df[eff_col].describe()
    except Exception:
        eff_stats = pd.Series({"mean": "N/A", "std": "N/A"})
        eff_df = pd.DataFrame()

    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px;
               overflow-x: auto; font-size: 12px; }}
        .stats-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px;
                      margin: 10px 0; }}
        .footer {{ margin-top: 40px; padding-top: 10px; border-top: 1px solid #ddd;
                   color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

    <h2>1. Estimation Summary</h2>
    <pre>{summary_text}</pre>

    <h2>2. Efficiency Statistics</h2>
    <div class="stats-box">
        <table>
            <tr><th>Statistic</th><th>Value</th></tr>
"""
    for stat_name, stat_val in eff_stats.items():
        try:
            html += f"            <tr><td>{stat_name}</td><td>{stat_val:.4f}</td></tr>\n"
        except (ValueError, TypeError):
            html += f"            <tr><td>{stat_name}</td><td>{stat_val}</td></tr>\n"

    html += """        </table>
    </div>
"""

    # Model information
    html += """
    <h2>3. Model Information</h2>
    <table>
        <tr><th>Attribute</th><th>Value</th></tr>
"""
    try:
        html += f"        <tr><td>Log-Likelihood</td><td>{result.loglik:.4f}</td></tr>\n"
        html += f"        <tr><td>AIC</td><td>{result.aic:.4f}</td></tr>\n"
        html += f"        <tr><td>BIC</td><td>{result.bic:.4f}</td></tr>\n"
        html += f"        <tr><td>N observations</td><td>{result.nobs}</td></tr>\n"
        html += f"        <tr><td>Converged</td><td>{result.converged}</td></tr>\n"
    except Exception:
        html += "        <tr><td>Info</td><td>Not available</td></tr>\n"

    html += """    </table>

    <div class="footer">
        <p>Report generated using PanelBox Stochastic Frontier Analysis module.</p>
    </div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def estimation_table_latex(
    result,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """
    Generate LaTeX table of estimation results.

    Parameters
    ----------
    result : SFResult or PanelSFResult
        Fitted SFA result object.
    caption : str, optional
        Table caption.
    label : str, optional
        LaTeX label for cross-referencing.

    Returns
    -------
    str
        LaTeX table string.
    """
    try:
        params = result.params
        se = result.se
        tvals = result.tvalues
        pvals = result.pvalues
    except Exception:
        return "% LaTeX table could not be generated: result object missing attributes."

    caption = caption or "Stochastic Frontier Estimation Results"
    label = label or "tab:sfa_results"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\hline\hline",
        r"Variable & Coefficient & Std. Error & $t$-stat & $p$-value \\",
        r"\hline",
    ]

    for var in params.index:
        coef = params[var]
        std_err = se[var] if var in se.index else float("nan")
        t_val = tvals[var] if var in tvals.index else float("nan")
        p_val = pvals[var] if var in pvals.index else float("nan")

        # Significance stars
        stars = ""
        if p_val < 0.01:
            stars = "***"
        elif p_val < 0.05:
            stars = "**"
        elif p_val < 0.10:
            stars = "*"

        var_clean = var.replace("_", r"\_")
        lines.append(
            f"{var_clean} & {coef:.4f}{stars} & ({std_err:.4f}) & {t_val:.3f} & {p_val:.4f} \\\\"
        )

    lines.extend(
        [
            r"\hline",
        ]
    )

    # Add model statistics
    try:
        lines.append(f"Log-Likelihood & \\multicolumn{{4}}{{c}}{{{result.loglik:.4f}}} \\\\")
        lines.append(f"AIC & \\multicolumn{{4}}{{c}}{{{result.aic:.4f}}} \\\\")
        lines.append(f"BIC & \\multicolumn{{4}}{{c}}{{{result.bic:.4f}}} \\\\")
        lines.append(f"N & \\multicolumn{{4}}{{c}}{{{result.nobs}}} \\\\")
    except Exception:
        pass

    lines.extend(
        [
            r"\hline\hline",
            r"\multicolumn{5}{l}{\small *** $p<0.01$, ** $p<0.05$, * $p<0.10$} \\",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def model_comparison_table(
    results_dict: dict[str, object],
    format: str = "latex",
) -> str:
    """
    Generate comparison table across multiple SFA specifications.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to fitted result objects.
    format : str
        Output format: 'latex' or 'html'.

    Returns
    -------
    str
        Formatted comparison table.
    """
    rows = []
    for name, result in results_dict.items():
        row = {"Model": name}
        try:
            row["Log-Lik"] = result.loglik
            row["AIC"] = result.aic
            row["BIC"] = result.bic
            row["sigma_v"] = result.sigma_v
            row["sigma_u"] = result.sigma_u
            row["gamma"] = result.gamma
            row["Mean TE"] = result.mean_efficiency
            row["N"] = result.nobs
        except Exception:
            pass
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")

    if format == "html":
        return df.to_html(float_format="%.4f")
    else:
        # LaTeX
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Model Comparison}",
            r"\label{tab:model_comparison}",
            r"\small",
        ]
        lines.append(df.to_latex(float_format="%.4f"))
        lines.extend(
            [
                r"\end{table}",
            ]
        )
        return "\n".join(lines)


def efficiency_ranking_table(
    efficiency_df: pd.DataFrame,
    efficiency_col: str = "efficiency",
    entity_col: str = "entity",
    top_n: int = 20,
    format: str = "latex",
) -> str:
    """
    Generate ranked efficiency table for top/bottom entities.

    Parameters
    ----------
    efficiency_df : pd.DataFrame
        DataFrame with entity and efficiency columns.
    efficiency_col : str
        Column name for efficiency scores.
    entity_col : str
        Column name for entity identifiers.
    top_n : int
        Number of top and bottom entities to include.
    format : str
        Output format: 'latex' or 'html'.

    Returns
    -------
    str
        Formatted ranking table.
    """
    avg_eff = efficiency_df.groupby(entity_col)[efficiency_col].mean().sort_values(ascending=False)

    top = avg_eff.head(top_n).reset_index()
    top.insert(0, "Rank", range(1, len(top) + 1))
    top.columns = ["Rank", entity_col, efficiency_col]

    bottom = avg_eff.tail(top_n).sort_values().reset_index()
    bottom.insert(0, "Rank", range(len(avg_eff), len(avg_eff) - len(bottom), -1))
    bottom.columns = ["Rank", entity_col, efficiency_col]

    if format == "html":
        html = "<h3>Top Performers</h3>\n"
        html += top.to_html(index=False, float_format="%.4f")
        html += "\n<h3>Bottom Performers</h3>\n"
        html += bottom.to_html(index=False, float_format="%.4f")
        return html
    else:
        latex = "% Top performers\n"
        latex += top.to_latex(index=False, float_format="%.4f")
        latex += "\n% Bottom performers\n"
        latex += bottom.to_latex(index=False, float_format="%.4f")
        return latex
