"""
Result Table Generation Utilities

This module provides functions to generate formatted result tables.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def create_regression_table(
    result: object,
    title: Optional[str] = None,
    variable_labels: Optional[Dict[str, str]] = None,
    include_diagnostics: bool = True,
) -> pd.DataFrame:
    """
    Create formatted regression table from single model result.

    Parameters
    ----------
    result : PanelResults
        Fitted model result object
    title : str, optional
        Table title
    variable_labels : dict, optional
        Dictionary mapping variable names to display labels
    include_diagnostics : bool, optional
        Include diagnostic statistics (default: True)

    Returns
    -------
    pd.DataFrame
        Formatted regression table

    Examples
    --------
    >>> table = create_regression_table(
    ...     fe_result,
    ...     title='Fixed Effects Model',
    ...     variable_labels={'log_capital': 'Log Capital Stock'}
    ... )
    """
    rows = []

    if title:
        rows.append(["", title])
        rows.append(["", ""])

    # Coefficients
    for var in result.params.index:
        var_label = variable_labels.get(var, var) if variable_labels else var
        coef = result.params[var]
        se = result.std_errors[var]
        pval = result.pvalues.get(var, np.nan) if hasattr(result, "pvalues") else np.nan

        # Significance stars
        stars = ""
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.1:
            stars = "*"

        rows.append([var_label, f"{coef:.4f}{stars}"])
        rows.append(["", f"({se:.4f})"])

    if include_diagnostics:
        rows.append(["", ""])
        rows.append(["R²", f"{result.r_squared:.4f}"])

        if hasattr(result, "r_squared_adj"):
            rows.append(["Adjusted R²", f"{result.r_squared_adj:.4f}"])

        rows.append(["N", str(result.nobs)])

        if hasattr(result, "f_statistic"):
            rows.append(["F-statistic", f"{result.f_statistic:.2f}"])

    df = pd.DataFrame(rows, columns=["Variable", "Coefficient"])
    return df


def summary_statistics_table(
    data: pd.DataFrame,
    variables: List[str],
    by_entity: bool = False,
    entity_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create summary statistics table.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    variables : list of str
        Variables to summarize
    by_entity : bool, optional
        Compute within-entity statistics (default: False)
    entity_col : str, optional
        Entity column name (required if by_entity=True)

    Returns
    -------
    pd.DataFrame
        Summary statistics table

    Examples
    --------
    >>> summary_statistics_table(data, ['invest', 'value', 'capital'])
    """
    stats_list = []

    for var in variables:
        if by_entity and entity_col:
            # Within-entity statistics
            entity_means = data.groupby(entity_col)[var].mean()
            stats = {
                "Variable": var,
                "Mean": entity_means.mean(),
                "Std Dev": entity_means.std(),
                "Min": entity_means.min(),
                "Max": entity_means.max(),
                "Between Entities": entity_means.std(),
            }

            # Within-entity variation
            within_var = data.groupby(entity_col)[var].apply(lambda x: x.std()).mean()
            stats["Within Entities"] = within_var
        else:
            # Overall statistics
            stats = {
                "Variable": var,
                "Mean": data[var].mean(),
                "Std Dev": data[var].std(),
                "Min": data[var].min(),
                "Max": data[var].max(),
                "N": data[var].count(),
            }

        stats_list.append(stats)

    df = pd.DataFrame(stats_list)
    return df.round(4)


def correlation_table(
    data: pd.DataFrame, variables: List[str], method: str = "pearson"
) -> pd.DataFrame:
    """
    Create correlation table.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    variables : list of str
        Variables for correlation matrix
    method : str, optional
        Correlation method: 'pearson', 'spearman', 'kendall' (default: 'pearson')

    Returns
    -------
    pd.DataFrame
        Correlation matrix

    Examples
    --------
    >>> correlation_table(data, ['invest', 'value', 'capital'])
    """
    corr_matrix = data[variables].corr(method=method)
    return corr_matrix.round(3)


def format_table_for_publication(
    df: pd.DataFrame,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    format: str = "latex",
) -> str:
    """
    Format table for publication (LaTeX or HTML).

    Parameters
    ----------
    df : pd.DataFrame
        Table to format
    caption : str, optional
        Table caption
    label : str, optional
        Table label (for LaTeX references)
    format : str, optional
        Output format: 'latex' or 'html' (default: 'latex')

    Returns
    -------
    str
        Formatted table string

    Examples
    --------
    >>> latex_table = format_table_for_publication(
    ...     comparison_df,
    ...     caption='Model Comparison',
    ...     label='tab:models',
    ...     format='latex'
    ... )
    """
    if format == "latex":
        latex_str = df.to_latex(
            index=False,
            escape=False,
            caption=caption,
            label=label,
            column_format="l" + "c" * (len(df.columns) - 1),
        )

        # Add note about significance stars
        note = (
            "\\begin{tablenotes}\n"
            "\\small\n"
            "\\item Notes: *** p$<$0.01, ** p$<$0.05, * p$<$0.1. "
            "Standard errors in parentheses.\n"
            "\\end{tablenotes}"
        )

        # Insert note before \end{table}
        latex_str = latex_str.replace("\\end{table}", f"{note}\n\\end{{table}}")

        return latex_str

    elif format == "html":
        html_str = df.to_html(index=False, escape=False, classes="table table-striped")

        if caption:
            html_str = f"<caption>{caption}</caption>\n" + html_str

        # Add note
        note = (
            "<p><small>Notes: *** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1. "
            "Standard errors in parentheses.</small></p>"
        )
        html_str += note

        return html_str

    else:
        raise ValueError(f"Unsupported format: {format}")
