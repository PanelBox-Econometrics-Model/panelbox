"""
Model Comparison Utilities

This module provides functions to compare multiple panel model results.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def compare_models(
    results_dict: Dict[str, object],
    variables: Optional[List[str]] = None,
    include_se: bool = True,
    include_pvalues: bool = True,
    star_levels: tuple = (0.1, 0.05, 0.01),
) -> pd.DataFrame:
    """
    Create comparison table of model results.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to PanelResults objects
    variables : list of str, optional
        Variables to include. If None, uses all common variables
    include_se : bool, optional
        Include standard errors in parentheses (default: True)
    include_pvalues : bool, optional
        Include significance stars (default: True)
    star_levels : tuple, optional
        P-value thresholds for stars (default: (0.1, 0.05, 0.01))

    Returns
    -------
    pd.DataFrame
        Comparison table with coefficients, SEs, and stars

    Examples
    --------
    >>> results = {
    ...     'Pooled OLS': pooled_result,
    ...     'Fixed Effects': fe_result,
    ...     'Random Effects': re_result
    ... }
    >>> comparison_table = compare_models(results)
    >>> print(comparison_table)
    """
    if variables is None:
        # Find common variables across all models
        all_vars = [set(res.params.index) for res in results_dict.values()]
        variables = sorted(list(set.intersection(*all_vars)))

    rows = []

    for var in variables:
        row_coef = [var]
        row_se = [""]

        for model_name, result in results_dict.items():
            if var in result.params.index:
                coef = result.params[var]
                se = result.std_errors[var]
                pval = result.pvalues.get(var, 1.0) if hasattr(result, "pvalues") else 1.0

                # Add significance stars
                stars = ""
                if include_pvalues:
                    if pval < star_levels[2]:
                        stars = "***"
                    elif pval < star_levels[1]:
                        stars = "**"
                    elif pval < star_levels[0]:
                        stars = "*"

                row_coef.append(f"{coef:.4f}{stars}")
                row_se.append(f"({se:.4f})")
            else:
                row_coef.append("—")
                row_se.append("")

        rows.append(row_coef)
        if include_se:
            rows.append(row_se)

    # Add summary statistics
    rows.append([""] * (len(results_dict) + 1))  # Blank row
    rows.append(["R²"] + [f"{res.r_squared:.4f}" for res in results_dict.values()])

    if all(hasattr(res, "r_squared_adj") for res in results_dict.values()):
        rows.append(["Adjusted R²"] + [f"{res.r_squared_adj:.4f}" for res in results_dict.values()])

    rows.append(["N"] + [str(res.nobs) for res in results_dict.values()])

    # Create DataFrame
    columns = ["Variable"] + list(results_dict.keys())
    df = pd.DataFrame(rows, columns=columns)

    return df


def aic_bic_comparison(results_dict: Dict[str, object]) -> pd.DataFrame:
    """
    Compare AIC and BIC across models.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to PanelResults objects

    Returns
    -------
    pd.DataFrame
        Table with AIC, BIC, and delta values

    Examples
    --------
    >>> aic_bic_comparison({'Pooled': pooled, 'FE': fe, 'RE': re})
    """
    aic_values = {}
    bic_values = {}

    for model_name, result in results_dict.items():
        aic_values[model_name] = getattr(result, "aic", np.nan)
        bic_values[model_name] = getattr(result, "bic", np.nan)

    # Compute deltas (difference from minimum)
    min_aic = min(aic_values.values())
    min_bic = min(bic_values.values())

    comparison = pd.DataFrame(
        {
            "AIC": aic_values,
            "Δ AIC": {k: v - min_aic for k, v in aic_values.items()},
            "BIC": bic_values,
            "Δ BIC": {k: v - min_bic for k, v in bic_values.items()},
        }
    )

    return comparison.round(2)


def hausman_test_summary(fe_result: object, re_result: object) -> Dict[str, Any]:
    """
    Perform and summarize Hausman test.

    Parameters
    ----------
    fe_result : PanelResults
        Fixed effects model result
    re_result : PanelResults
        Random effects model result

    Returns
    -------
    dict
        Dictionary with test statistic, p-value, and interpretation

    Examples
    --------
    >>> hausman = hausman_test_summary(fe_result, re_result)
    >>> print(hausman['interpretation'])
    """
    # This is a placeholder - actual implementation depends on PanelBox API
    # for Hausman test functionality

    summary = {
        "test_statistic": np.nan,
        "p_value": np.nan,
        "degrees_of_freedom": np.nan,
        "interpretation": "Hausman test not yet implemented in helper function",
    }

    # If PanelBox provides Hausman test:
    # from panelbox.tests import HausmanTest
    # hausman = HausmanTest(fe_result, re_result)
    # summary['test_statistic'] = hausman.statistic
    # summary['p_value'] = hausman.pvalue
    # ...

    return summary


def export_comparison_table(
    comparison_df: pd.DataFrame, output_path: str, format: str = "latex"
) -> None:
    """
    Export comparison table to file.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison table from compare_models()
    output_path : str
        Path for output file
    format : str, optional
        Output format: 'latex', 'html', 'csv', 'excel' (default: 'latex')

    Examples
    --------
    >>> table = compare_models(results)
    >>> export_comparison_table(table, 'results.tex', format='latex')
    """
    if format == "latex":
        latex_str = comparison_df.to_latex(index=False, escape=False)
        with open(output_path, "w") as f:
            f.write(latex_str)
    elif format == "html":
        html_str = comparison_df.to_html(index=False, escape=False)
        with open(output_path, "w") as f:
            f.write(html_str)
    elif format == "csv":
        comparison_df.to_csv(output_path, index=False)
    elif format == "excel":
        comparison_df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Table exported to {output_path}")
