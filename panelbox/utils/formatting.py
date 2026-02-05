"""
Formatting utilities for output display.

This module provides functions for formatting statistical output
in a readable, publication-quality format.
"""

from typing import Union


def format_pvalue(pvalue: float, digits: int = 4) -> str:
    """
    Format p-value for display.

    Parameters
    ----------
    pvalue : float
        p-value
    digits : int, default=4
        Number of decimal places

    Returns
    -------
    str
        Formatted p-value string

    Examples
    --------
    >>> format_pvalue(0.0001)
    '<0.0001'
    >>> format_pvalue(0.1234)
    '0.1234'
    """
    threshold = 10 ** (-digits)

    if pvalue < threshold:
        return f"<{threshold:.{digits}f}"
    else:
        return f"{pvalue:.{digits}f}"


def format_number(value: Union[int, float], decimals: int = 4, width: int = 10) -> str:
    """
    Format number for tabular display.

    Parameters
    ----------
    value : int or float
        Number to format
    decimals : int, default=4
        Number of decimal places
    width : int, default=10
        Total width of formatted string

    Returns
    -------
    str
        Formatted number string
    """
    if isinstance(value, int):
        return f"{value:>{width},}"
    else:
        return f"{value:>{width}.{decimals}f}"


def significance_stars(pvalue: float) -> str:
    """
    Return significance stars based on p-value.

    Parameters
    ----------
    pvalue : float
        p-value

    Returns
    -------
    str
        Significance stars

    Examples
    --------
    >>> significance_stars(0.0001)
    '***'
    >>> significance_stars(0.01)
    '**'
    >>> significance_stars(0.04)
    '*'
    >>> significance_stars(0.08)
    '.'
    >>> significance_stars(0.20)
    ''
    """
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.05:
        return "*"
    elif pvalue < 0.10:
        return "."
    else:
        return ""


def format_coefficient_table(params, std_errors, tvalues, pvalues, conf_int=None) -> str:
    """
    Format coefficient table for display.

    Parameters
    ----------
    params : pd.Series
        Coefficients
    std_errors : pd.Series
        Standard errors
    tvalues : pd.Series
        t-statistics
    pvalues : pd.Series
        p-values
    conf_int : pd.DataFrame, optional
        Confidence intervals

    Returns
    -------
    str
        Formatted table
    """
    lines = []

    # Header
    if conf_int is not None:
        lines.append(
            f"{'Variable':<15} {'Coef.':<12} {'Std.Err.':<12} {'t':<8} "
            f"{'P>|t|':<8} {'[0.025':<10} {'0.975]':<10}"
        )
    else:
        lines.append(f"{'Variable':<15} {'Coef.':<12} {'Std.Err.':<12} {'t':<8} {'P>|t|':<8}")

    lines.append("-" * 78)

    # Rows
    for var in params.index:
        coef = params[var]
        se = std_errors[var]
        t = tvalues[var]
        p = pvalues[var]
        stars = significance_stars(p)

        if conf_int is not None:
            ci_lower = conf_int.loc[var, "lower"]
            ci_upper = conf_int.loc[var, "upper"]
            lines.append(
                f"{var:<15} {coef:>11.4f} {se:>11.4f} {t:>7.3f} "
                f"{p:>7.4f} {ci_lower:>9.4f} {ci_upper:>9.4f} {stars}"
            )
        else:
            lines.append(f"{var:<15} {coef:>11.4f} {se:>11.4f} {t:>7.3f} {p:>7.4f} {stars}")

    return "\n".join(lines)
