"""
Report generation for SFA results.

This module provides functions to generate professional reports in various formats:
- HTML with interactive plots
- LaTeX for academic papers
- Markdown for documentation
- Model comparison tables
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def to_latex(
    result,
    include_stats: Optional[List[str]] = None,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    float_format: str = "%.4f",
) -> str:
    """Generate LaTeX table of SFA results.

    Parameters:
        result: SFResult object
        include_stats: Statistics to include ('coef', 'se', 'tval', 'pval')
                      Default: ['coef', 'se', 'pval']
        caption: Table caption
        label: Table label for referencing
        float_format: Format string for floating point numbers

    Returns:
        LaTeX table string

    Example:
        >>> result = sf.fit()
        >>> latex = result.to_latex(
        ...     caption='SFA Results for Banking Efficiency',
        ...     label='tab:sfa_results'
        ... )
        >>> print(latex)
    """
    if include_stats is None:
        include_stats = ["coef", "se", "pval"]

    # Build parameter table
    param_names = result.params.index.tolist()

    # Filter out variance parameters (shown separately)
    frontier_params = [
        name for name in param_names if "sigma" not in name.lower() and "ln_" not in name.lower()
    ]

    # Start LaTeX table
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")

    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")

    # Table header
    n_cols = 1 + len(include_stats)
    lines.append(f"\\begin{{tabular}}{{l{'r' * len(include_stats)}}}")
    lines.append("\\toprule")

    header_parts = ["Variable"]
    if "coef" in include_stats:
        header_parts.append("Coefficient")
    if "se" in include_stats:
        header_parts.append("Std. Error")
    if "tval" in include_stats:
        header_parts.append("t-value")
    if "pval" in include_stats:
        header_parts.append("p-value")

    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    # Parameter rows
    for param in frontier_params:
        row_parts = [param.replace("_", "\\_")]

        if "coef" in include_stats:
            coef = result.params[param]
            row_parts.append(float_format % coef)

        if "se" in include_stats:
            se = result.se[param]
            row_parts.append(float_format % se)

        if "tval" in include_stats:
            tval = result.tvalues[param]
            row_parts.append(float_format % tval)

        if "pval" in include_stats:
            pval = result.pvalues[param]
            # Add significance stars
            stars = ""
            if pval < 0.01:
                stars = "***"
            elif pval < 0.05:
                stars = "**"
            elif pval < 0.1:
                stars = "*"
            row_parts.append((float_format % pval) + stars)

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\midrule")

    # Variance components (fill empty columns based on include_stats length)
    empty_cols = " &" * len(include_stats)
    lines.append(f"$\\sigma_v$ & {float_format % result.sigma_v}{empty_cols} \\\\")
    lines.append(f"$\\sigma_u$ & {float_format % result.sigma_u}{empty_cols} \\\\")
    lines.append(f"$\\lambda$ & {float_format % result.lambda_param}{empty_cols} \\\\")
    lines.append(f"$\\gamma$ & {float_format % result.gamma}{empty_cols} \\\\")

    lines.append("\\midrule")

    # Model statistics
    lines.append(f"Log-Likelihood & {float_format % result.loglik}{empty_cols} \\\\")
    lines.append(f"AIC & {float_format % result.aic}{empty_cols} \\\\")
    lines.append(f"BIC & {float_format % result.bic}{empty_cols} \\\\")
    lines.append(f"N & {result.nobs}{empty_cols} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    # Add notes
    lines.append("\\\\[0.5em]")
    lines.append("\\begin{minipage}{\\textwidth}")
    lines.append("\\small")
    lines.append("\\textit{Notes:} Significance levels: * p<0.1, ** p<0.05, *** p<0.01.")
    lines.append(f"Frontier: {result.model.frontier_type.value}. ")
    lines.append(f"Distribution: {result.model.dist.value}.")
    lines.append("\\end{minipage}")

    lines.append("\\end{table}")

    return "\n".join(lines)


def to_html(
    result,
    filename: Optional[str] = None,
    include_plots: bool = True,
    theme: str = "academic",
    **kwargs,
) -> str:
    """Generate HTML report with interactive plots.

    Parameters:
        result: SFResult object
        filename: If provided, save HTML to this file
        include_plots: Include interactive Plotly plots
        theme: Report theme ('academic', 'professional', 'presentation')
        **kwargs: Additional plot configuration

    Returns:
        HTML string

    Example:
        >>> result = sf.fit()
        >>> result.to_html(
        ...     filename='sfa_report.html',
        ...     include_plots=True,
        ...     theme='academic'
        ... )
    """
    # HTML template
    html_parts = []

    # Header
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html>")
    html_parts.append("<head>")
    html_parts.append("<meta charset='utf-8'>")
    html_parts.append("<title>SFA Results Report</title>")

    # CSS styling
    css = get_html_css(theme)
    html_parts.append(f"<style>{css}</style>")

    # Plotly if needed
    if include_plots:
        html_parts.append("<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>")

    html_parts.append("</head>")
    html_parts.append("<body>")

    # Title
    html_parts.append("<div class='container'>")
    html_parts.append("<h1>Stochastic Frontier Analysis Results</h1>")

    # Model Information
    html_parts.append("<section class='section'>")
    html_parts.append("<h2>Model Information</h2>")
    html_parts.append("<table class='info-table'>")
    html_parts.append(f"<tr><td>Frontier Type</td><td>{result.model.frontier_type.value}</td></tr>")
    html_parts.append(f"<tr><td>Distribution</td><td>{result.model.dist.value}</td></tr>")
    html_parts.append(f"<tr><td>Observations</td><td>{result.nobs}</td></tr>")

    if result.model.is_panel:
        html_parts.append(f"<tr><td>Entities</td><td>{result.model.n_entities}</td></tr>")
        html_parts.append(f"<tr><td>Time Periods</td><td>{result.model.n_periods}</td></tr>")

    html_parts.append(f"<tr><td>Log-Likelihood</td><td>{result.loglik:.4f}</td></tr>")
    html_parts.append(f"<tr><td>AIC</td><td>{result.aic:.4f}</td></tr>")
    html_parts.append(f"<tr><td>BIC</td><td>{result.bic:.4f}</td></tr>")
    html_parts.append(f"<tr><td>Converged</td><td>{'Yes' if result.converged else 'No'}</td></tr>")
    html_parts.append("</table>")
    html_parts.append("</section>")

    # Variance Components
    html_parts.append("<section class='section'>")
    html_parts.append("<h2>Variance Components</h2>")
    html_parts.append("<table class='info-table'>")
    html_parts.append(f"<tr><td>σ<sub>v</sub> (noise)</td><td>{result.sigma_v:.6f}</td></tr>")
    html_parts.append(
        f"<tr><td>σ<sub>u</sub> (inefficiency)</td><td>{result.sigma_u:.6f}</td></tr>"
    )
    html_parts.append(f"<tr><td>σ (composite)</td><td>{result.sigma:.6f}</td></tr>")
    html_parts.append(
        f"<tr><td>λ = σ<sub>u</sub>/σ<sub>v</sub></td><td>{result.lambda_param:.6f}</td></tr>"
    )
    html_parts.append(f"<tr><td>γ = σ²<sub>u</sub>/σ²</td><td>{result.gamma:.6f}</td></tr>")
    html_parts.append("</table>")
    html_parts.append("</section>")

    # Parameter Estimates
    html_parts.append("<section class='section'>")
    html_parts.append("<h2>Parameter Estimates</h2>")
    html_parts.append("<table class='params-table'>")
    html_parts.append("<thead>")
    html_parts.append(
        "<tr><th>Variable</th><th>Coefficient</th><th>Std. Error</th><th>t-value</th><th>p-value</th></tr>"
    )
    html_parts.append("</thead>")
    html_parts.append("<tbody>")

    param_names = result.params.index.tolist()
    frontier_params = [
        name for name in param_names if "sigma" not in name.lower() and "ln_" not in name.lower()
    ]

    for param in frontier_params:
        coef = result.params[param]
        se = result.se[param]
        tval = result.tvalues[param]
        pval = result.pvalues[param]

        # Significance stars
        stars = ""
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.1:
            stars = "*"

        html_parts.append(f"<tr>")
        html_parts.append(f"<td>{param}</td>")
        html_parts.append(f"<td>{coef:.6f}{stars}</td>")
        html_parts.append(f"<td>{se:.6f}</td>")
        html_parts.append(f"<td>{tval:.4f}</td>")
        html_parts.append(f"<td>{pval:.4f}</td>")
        html_parts.append(f"</tr>")

    html_parts.append("</tbody>")
    html_parts.append("</table>")
    html_parts.append("<p class='note'>* p<0.1, ** p<0.05, *** p<0.01</p>")
    html_parts.append("</section>")

    # Efficiency Summary
    html_parts.append("<section class='section'>")
    html_parts.append("<h2>Efficiency Summary</h2>")
    eff_df = result.efficiency(estimator="bc")
    eff_values = eff_df["efficiency"].values

    html_parts.append("<table class='info-table'>")
    html_parts.append(f"<tr><td>Mean Efficiency</td><td>{np.mean(eff_values):.4f}</td></tr>")
    html_parts.append(f"<tr><td>Median Efficiency</td><td>{np.median(eff_values):.4f}</td></tr>")
    html_parts.append(f"<tr><td>Std. Dev.</td><td>{np.std(eff_values):.4f}</td></tr>")
    html_parts.append(f"<tr><td>Minimum</td><td>{np.min(eff_values):.4f}</td></tr>")
    html_parts.append(f"<tr><td>Maximum</td><td>{np.max(eff_values):.4f}</td></tr>")
    html_parts.append("</table>")
    html_parts.append("</section>")

    # Plots
    if include_plots:
        try:
            # Efficiency distribution
            html_parts.append("<section class='section'>")
            html_parts.append("<h2>Efficiency Distribution</h2>")

            fig = result.plot_efficiency(kind="histogram", backend="plotly")
            plot_html = fig.to_html(full_html=False, include_plotlyjs=False)
            html_parts.append(plot_html)

            html_parts.append("</section>")

            # Efficiency ranking
            html_parts.append("<section class='section'>")
            html_parts.append("<h2>Efficiency Rankings</h2>")

            fig = result.plot_efficiency(kind="ranking", backend="plotly", top_n=10, bottom_n=10)
            plot_html = fig.to_html(full_html=False, include_plotlyjs=False)
            html_parts.append(plot_html)

            html_parts.append("</section>")

        except Exception as e:
            html_parts.append(f"<p class='error'>Could not generate plots: {e}</p>")

    # Footer
    html_parts.append("<footer>")
    html_parts.append("<p>Generated with PanelBox SFA module</p>")
    html_parts.append("</footer>")

    html_parts.append("</div>")  # container
    html_parts.append("</body>")
    html_parts.append("</html>")

    html_content = "\n".join(html_parts)

    # Save to file if requested
    if filename is not None:
        with open(filename, "w") as f:
            f.write(html_content)

    return html_content


def get_html_css(theme: str = "academic") -> str:
    """Get CSS styling for HTML reports."""
    base_css = """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }
        .section {
            margin: 30px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .info-table td {
            padding: 8px 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        .info-table td:first-child {
            font-weight: bold;
            width: 40%;
        }
        .params-table {
            font-size: 0.9em;
        }
        .params-table th {
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
        }
        .params-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        .params-table tbody tr:hover {
            background-color: #f8f9fa;
        }
        .note {
            font-size: 0.85em;
            color: #7f8c8d;
            font-style: italic;
        }
        .error {
            color: #e74c3c;
            background-color: #fadbd8;
            padding: 10px;
            border-radius: 4px;
        }
        footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            text-align: center;
            color: #95a5a6;
            font-size: 0.9em;
        }
    """

    if theme == "professional":
        base_css += """
        body { background-color: #e8eaf6; }
        h1 { color: #1a237e; border-bottom-color: #3f51b5; }
        .params-table th { background-color: #3f51b5; }
        """
    elif theme == "presentation":
        base_css += """
        body { background-color: #fff; }
        h1 { font-size: 2.5em; color: #d32f2f; border-bottom-color: #f44336; }
        h2 { font-size: 1.8em; }
        .params-table th { background-color: #d32f2f; }
        """

    return base_css


def to_markdown(result) -> str:
    """Generate Markdown report of SFA results.

    Parameters:
        result: SFResult object

    Returns:
        Markdown string

    Example:
        >>> result = sf.fit()
        >>> md = result.to_markdown()
        >>> with open('report.md', 'w') as f:
        ...     f.write(md)
    """
    lines = []

    # Title
    lines.append("# Stochastic Frontier Analysis Results")
    lines.append("")

    # Model Information
    lines.append("## Model Information")
    lines.append("")
    lines.append(f"- **Frontier Type**: {result.model.frontier_type.value}")
    lines.append(f"- **Distribution**: {result.model.dist.value}")
    lines.append(f"- **Observations**: {result.nobs}")

    if result.model.is_panel:
        lines.append(f"- **Entities**: {result.model.n_entities}")
        lines.append(f"- **Time Periods**: {result.model.n_periods}")

    lines.append(f"- **Log-Likelihood**: {result.loglik:.4f}")
    lines.append(f"- **AIC**: {result.aic:.4f}")
    lines.append(f"- **BIC**: {result.bic:.4f}")
    lines.append(f"- **Converged**: {'Yes' if result.converged else 'No'}")
    lines.append("")

    # Variance Components
    lines.append("## Variance Components")
    lines.append("")
    lines.append(f"- **σ_v (noise)**: {result.sigma_v:.6f}")
    lines.append(f"- **σ_u (inefficiency)**: {result.sigma_u:.6f}")
    lines.append(f"- **σ (composite)**: {result.sigma:.6f}")
    lines.append(f"- **λ = σ_u/σ_v**: {result.lambda_param:.6f}")
    lines.append(f"- **γ = σ²_u/σ²**: {result.gamma:.6f}")
    lines.append("")

    # Parameter Estimates
    lines.append("## Parameter Estimates")
    lines.append("")
    lines.append("| Variable | Coefficient | Std. Error | t-value | p-value |")
    lines.append("|----------|-------------|------------|---------|---------|")

    param_names = result.params.index.tolist()
    frontier_params = [
        name for name in param_names if "sigma" not in name.lower() and "ln_" not in name.lower()
    ]

    for param in frontier_params:
        coef = result.params[param]
        se = result.se[param]
        tval = result.tvalues[param]
        pval = result.pvalues[param]

        # Significance stars
        stars = ""
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.1:
            stars = "*"

        lines.append(f"| {param} | {coef:.6f}{stars} | {se:.6f} | {tval:.4f} | {pval:.4f} |")

    lines.append("")
    lines.append("*Note: \\* p<0.1, \\*\\* p<0.05, \\*\\*\\* p<0.01*")
    lines.append("")

    # Efficiency Summary
    lines.append("## Efficiency Summary")
    lines.append("")

    eff_df = result.efficiency(estimator="bc")
    eff_values = eff_df["efficiency"].values

    lines.append(f"- **Mean Efficiency**: {np.mean(eff_values):.4f}")
    lines.append(f"- **Median Efficiency**: {np.median(eff_values):.4f}")
    lines.append(f"- **Std. Dev.**: {np.std(eff_values):.4f}")
    lines.append(f"- **Minimum**: {np.min(eff_values):.4f}")
    lines.append(f"- **Maximum**: {np.max(eff_values):.4f}")
    lines.append("")

    return "\n".join(lines)


def compare_models(
    models: Dict[str, Any], output_format: str = "dataframe"
) -> Union[pd.DataFrame, str]:
    """Compare multiple SFA models side by side.

    Parameters:
        models: Dictionary mapping model names to SFResult objects
        output_format: Output format ('dataframe', 'latex', 'markdown')

    Returns:
        DataFrame or formatted string

    Example:
        >>> result_hn = sf_hn.fit()
        >>> result_tn = sf_tn.fit()
        >>> result_bc95 = sf_bc95.fit()
        >>> comparison = compare_models({
        ...     'Half-Normal': result_hn,
        ...     'Truncated Normal': result_tn,
        ...     'BC95': result_bc95
        ... }, output_format='latex')
    """
    # Collect statistics from all models
    comparison_data = {}

    for name, result in models.items():
        comparison_data[name] = {
            "Log-Likelihood": result.loglik,
            "AIC": result.aic,
            "BIC": result.bic,
            "σ_v": result.sigma_v,
            "σ_u": result.sigma_u,
            "λ": result.lambda_param,
            "γ": result.gamma,
            "Mean Efficiency": result.mean_efficiency,
            "Converged": result.converged,
        }

    df = pd.DataFrame(comparison_data).T

    # Add best model indicators
    df["Best AIC"] = df["AIC"] == df["AIC"].min()
    df["Best BIC"] = df["BIC"] == df["BIC"].min()

    if output_format == "dataframe":
        return df
    elif output_format == "latex":
        return df.to_latex(float_format="%.4f", escape=False)
    elif output_format == "markdown":
        return df.to_markdown(floatfmt=".4f")
    else:
        raise ValueError(f"Unknown output_format: {output_format}")


def efficiency_table(
    result,
    sort_by: str = "te",
    ascending: bool = False,
    top_n: Optional[int] = None,
    estimator: str = "bc",
) -> pd.DataFrame:
    """Create formatted efficiency rankings table.

    Parameters:
        result: SFResult object
        sort_by: Column to sort by ('te', 'entity', 'time')
        ascending: Sort order
        top_n: Limit to top N rows
        estimator: Efficiency estimator

    Returns:
        Formatted DataFrame

    Example:
        >>> result = sf.fit()
        >>> eff_table = result.efficiency_table(
        ...     sort_by='te',
        ...     ascending=False,
        ...     top_n=20
        ... )
        >>> eff_table.to_excel('efficiency_rankings.xlsx')
    """
    eff_df = result.efficiency(estimator=estimator)

    # Rename efficiency column
    eff_df = eff_df.rename(columns={"efficiency": "te"})

    # Sort
    if sort_by in eff_df.columns:
        eff_df = eff_df.sort_values(sort_by, ascending=ascending)

    # Limit rows
    if top_n is not None:
        eff_df = eff_df.head(top_n)

    # Add rank
    eff_df.insert(0, "rank", range(1, len(eff_df) + 1))

    return eff_df
