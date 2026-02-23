"""
Helper functions for panel unit root testing tutorials.

Provides comparison tables, interpretation, plotting, and transformation
recommendations across multiple unit root tests available in PanelBox.

Functions:
- compare_unit_root_tests: Run all available tests, return comparison DataFrame.
- interpret_results: Generate textual interpretation from test results dict.
- plot_levels_vs_differences: Side-by-side panel of levels and first differences.
- recommend_transformation: Suggest 'levels', 'first_difference', or 'borderline'.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Compare all unit root tests
# ---------------------------------------------------------------------------


def compare_unit_root_tests(
    data: pd.DataFrame,
    variable: str,
    entity_col: str,
    time_col: str,
    trend: str = "c",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run all available panel unit root tests and return a comparison table.

    Attempts to run LLC, IPS, Hadri, and Breitung tests.  Each test is
    wrapped in its own try/except block so that unavailable or failing
    tests are reported gracefully rather than aborting the comparison.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    variable : str
        Name of the variable to test.
    entity_col : str
        Name of the entity (cross-section) identifier column.
    time_col : str
        Name of the time identifier column.
    trend : str, default 'c'
        Deterministic specification passed to each test.
        Typical values: 'c' (constant only), 'ct' (constant + trend).
    save_path : str, optional
        If provided, save the resulting DataFrame as a CSV file at this path.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Test, H0, Statistic, P-value, Reject_5pct,
        and Status.  One row per test attempted.

    Notes
    -----
    The returned DataFrame always has the same column schema regardless of
    which tests succeed.  Failed tests have Status='error' and NaN values
    for Statistic / P-value / Reject_5pct.

    Examples
    --------
    >>> from examples.diagnostics.utils.data_generators import generate_penn_world_table
    >>> from examples.diagnostics.utils.unit_root_helpers import compare_unit_root_tests
    >>> pwt = generate_penn_world_table(n_countries=10, n_years=30)
    >>> df = compare_unit_root_tests(pwt, "rgdpna", "countrycode", "year")
    >>> print(df.to_string(index=False))
    """
    rows = []

    # --- LLC test (H0: unit root) ---
    try:
        from panelbox.validation.unit_root import LLCTest

        llc = LLCTest(data, variable, entity_col, time_col, trend=trend)
        llc_result = llc.run()
        rows.append(
            {
                "Test": "LLC (Levin-Lin-Chu)",
                "H0": "Unit root (common)",
                "Statistic": round(llc_result.statistic, 4),
                "P-value": round(llc_result.pvalue, 4),
                "Reject_5pct": llc_result.pvalue < 0.05,
                "Status": "ok",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "Test": "LLC (Levin-Lin-Chu)",
                "H0": "Unit root (common)",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Reject_5pct": np.nan,
                "Status": f"error: {exc}",
            }
        )

    # --- IPS test (H0: unit root) ---
    try:
        from panelbox.validation.unit_root import IPSTest

        ips = IPSTest(data, variable, entity_col, time_col, trend=trend)
        ips_result = ips.run()
        rows.append(
            {
                "Test": "IPS (Im-Pesaran-Shin)",
                "H0": "Unit root (heterogeneous)",
                "Statistic": round(ips_result.statistic, 4),
                "P-value": round(ips_result.pvalue, 4),
                "Reject_5pct": ips_result.pvalue < 0.05,
                "Status": "ok",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "Test": "IPS (Im-Pesaran-Shin)",
                "H0": "Unit root (heterogeneous)",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Reject_5pct": np.nan,
                "Status": f"error: {exc}",
            }
        )

    # --- Hadri test (H0: stationarity) ---
    try:
        from panelbox.diagnostics.unit_root import hadri_test

        hadri_result = hadri_test(data, variable, entity_col, time_col, trend=trend, robust=True)
        rows.append(
            {
                "Test": "Hadri (LM)",
                "H0": "Stationarity",
                "Statistic": round(hadri_result.statistic, 4),
                "P-value": round(hadri_result.pvalue, 4),
                "Reject_5pct": hadri_result.reject,
                "Status": "ok",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "Test": "Hadri (LM)",
                "H0": "Stationarity",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Reject_5pct": np.nan,
                "Status": f"error: {exc}",
            }
        )

    # --- Breitung test (H0: unit root) ---
    try:
        from panelbox.diagnostics.unit_root import breitung_test

        breitung_result = breitung_test(data, variable, entity_col, time_col, trend=trend)
        rows.append(
            {
                "Test": "Breitung",
                "H0": "Unit root",
                "Statistic": round(breitung_result.statistic, 4),
                "P-value": round(breitung_result.pvalue, 4),
                "Reject_5pct": breitung_result.reject,
                "Status": "ok",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "Test": "Breitung",
                "H0": "Unit root",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Reject_5pct": np.nan,
                "Status": f"error: {exc}",
            }
        )

    result_df = pd.DataFrame(rows)

    if save_path is not None:
        result_df.to_csv(save_path, index=False)

    return result_df


# ---------------------------------------------------------------------------
# 2. Interpret results
# ---------------------------------------------------------------------------


def interpret_results(results_dict: dict[str, dict], alpha: float = 0.05) -> str:
    """
    Generate a textual interpretation from a dictionary of unit root test results.

    The function classifies each test result as evidence for stationarity or
    for a unit root, taking into account that the Hadri test has the opposite
    null hypothesis (H0: stationarity) compared to LLC, IPS, and Breitung
    (H0: unit root).

    Parameters
    ----------
    results_dict : dict
        Mapping from test name (str) to a dict with at least the keys
        'pvalue' (float) and 'h0' (str, one of 'unit_root' or 'stationarity').
        Example::

            {
                "LLC": {"pvalue": 0.03, "h0": "unit_root"},
                "IPS": {"pvalue": 0.12, "h0": "unit_root"},
                "Hadri": {"pvalue": 0.001, "h0": "stationarity"},
                "Breitung": {"pvalue": 0.07, "h0": "unit_root"},
            }

    alpha : float, default 0.05
        Significance level for rejection decisions.

    Returns
    -------
    str
        Multi-line text summarising the evidence and overall conclusion.

    Notes
    -----
    When tests point in conflicting directions the interpretation
    explicitly flags this and recommends additional investigation.

    Examples
    --------
    >>> res = {
    ...     "LLC": {"pvalue": 0.03, "h0": "unit_root"},
    ...     "Hadri": {"pvalue": 0.001, "h0": "stationarity"},
    ... }
    >>> print(interpret_results(res))
    """
    evidence_stationary = 0
    evidence_unit_root = 0
    n_tests = 0
    details = []

    for test_name, info in results_dict.items():
        pval = info.get("pvalue", np.nan)
        h0 = info.get("h0", "unit_root").lower()

        if np.isnan(pval):
            details.append(f"  - {test_name}: p-value not available (skipped)")
            continue

        n_tests += 1
        reject = pval < alpha

        if h0 == "stationarity":
            # Hadri-type: rejecting H0 means evidence of unit root
            if reject:
                evidence_unit_root += 1
                details.append(
                    f"  - {test_name} (H0: stationarity): REJECT H0 "
                    f"(p={pval:.4f}) -> evidence of unit root"
                )
            else:
                evidence_stationary += 1
                details.append(
                    f"  - {test_name} (H0: stationarity): Fail to reject H0 "
                    f"(p={pval:.4f}) -> consistent with stationarity"
                )
        else:
            # LLC / IPS / Breitung: rejecting H0 means evidence of stationarity
            if reject:
                evidence_stationary += 1
                details.append(
                    f"  - {test_name} (H0: unit root): REJECT H0 "
                    f"(p={pval:.4f}) -> evidence of stationarity"
                )
            else:
                evidence_unit_root += 1
                details.append(
                    f"  - {test_name} (H0: unit root): Fail to reject H0 "
                    f"(p={pval:.4f}) -> evidence of unit root"
                )

    # Build interpretation text
    lines = [
        "=" * 70,
        "Unit Root Test Interpretation",
        "=" * 70,
        f"Significance level: {alpha}",
        f"Tests evaluated: {n_tests}",
        "",
        "Individual test results:",
    ]
    lines.extend(details)
    lines.append("")

    # Overall verdict
    lines.append("Overall assessment:")

    if n_tests == 0:
        lines.append("  No valid test results available.")
    elif evidence_stationary > evidence_unit_root:
        lines.append(f"  {evidence_stationary} of {n_tests} test(s) point toward STATIONARITY.")
        lines.append(
            "  Recommendation: The series is likely stationary. "
            "Using levels in regression is appropriate."
        )
    elif evidence_unit_root > evidence_stationary:
        lines.append(f"  {evidence_unit_root} of {n_tests} test(s) point toward a UNIT ROOT.")
        lines.append(
            "  Recommendation: Consider first-differencing or "
            "testing for cointegration before using in levels."
        )
    else:
        lines.append(
            f"  Tests are evenly split ({evidence_stationary} stationary vs "
            f"{evidence_unit_root} unit root)."
        )
        lines.append(
            "  Recommendation: Results are INCONCLUSIVE. "
            "Consider economic theory, longer time series, "
            "or alternative test specifications."
        )

    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Plot levels vs. first differences
# ---------------------------------------------------------------------------


def plot_levels_vs_differences(
    data: pd.DataFrame,
    variable: str,
    entity_col: str,
    time_col: str,
    n_entities: int = 4,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side plot of a variable in levels and first differences.

    Selects up to ``n_entities`` cross-sectional units and draws two
    columns: the left column shows the raw series (levels) and the right
    column shows first differences (delta y_t = y_t - y_{t-1}).

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    variable : str
        Name of the variable to plot.
    entity_col : str
        Name of the entity (cross-section) identifier column.
    time_col : str
        Name of the time identifier column.
    n_entities : int, default 4
        Number of entities to display. The first ``n_entities`` unique
        values in ``entity_col`` are selected.
    save_path : str, optional
        If provided, save the figure to this file path.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.

    Examples
    --------
    >>> from examples.diagnostics.utils.data_generators import generate_penn_world_table
    >>> from examples.diagnostics.utils.unit_root_helpers import plot_levels_vs_differences
    >>> pwt = generate_penn_world_table(n_countries=10, n_years=30)
    >>> fig = plot_levels_vs_differences(pwt, "rgdpna", "countrycode", "year")
    >>> plt.show()
    """
    entities = data[entity_col].unique()[:n_entities]
    n_show = len(entities)

    fig, axes = plt.subplots(n_show, 2, figsize=(14, 3.5 * n_show), squeeze=False)

    for i, entity in enumerate(entities):
        subset = data[data[entity_col] == entity].sort_values(time_col)
        t = subset[time_col].values
        y = subset[variable].values

        # Left: levels
        ax_level = axes[i, 0]
        ax_level.plot(t, y, linewidth=1.5, color="#1f77b4")
        ax_level.set_ylabel(variable)
        ax_level.set_title(f"{entity} -- Levels", fontweight="bold")
        ax_level.grid(True, alpha=0.3)

        # Right: first differences
        ax_diff = axes[i, 1]
        dy = np.diff(y)
        t_diff = t[1:]
        ax_diff.plot(t_diff, dy, linewidth=1.5, color="#d62728")
        ax_diff.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax_diff.set_ylabel(f"d({variable})")
        ax_diff.set_title(f"{entity} -- First Differences", fontweight="bold")
        ax_diff.grid(True, alpha=0.3)

    # Common x-label on bottom row only
    axes[-1, 0].set_xlabel(time_col)
    axes[-1, 1].set_xlabel(time_col)

    fig.suptitle(
        f"Levels vs. First Differences: {variable}",
        fontweight="bold",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 4. Recommend transformation
# ---------------------------------------------------------------------------


def recommend_transformation(
    results_dict: dict[str, dict],
    alpha: float = 0.05,
) -> str:
    """
    Recommend a data transformation based on unit root test results.

    Uses a simple majority rule across all tests.  Each test is classified
    as providing evidence for stationarity or for a unit root, respecting
    the direction of its null hypothesis.

    Parameters
    ----------
    results_dict : dict
        Same format as for :func:`interpret_results`.  Mapping from test
        name to a dict with keys ``'pvalue'`` and ``'h0'`` (one of
        ``'unit_root'`` or ``'stationarity'``).
    alpha : float, default 0.05
        Significance level for rejection decisions.

    Returns
    -------
    str
        One of:

        * ``'levels'`` -- majority of tests indicate stationarity.
        * ``'first_difference'`` -- majority of tests indicate a unit root.
        * ``'borderline'`` -- tests are evenly split or no valid results.

    Examples
    --------
    >>> res = {
    ...     "LLC": {"pvalue": 0.03, "h0": "unit_root"},
    ...     "IPS": {"pvalue": 0.01, "h0": "unit_root"},
    ...     "Hadri": {"pvalue": 0.40, "h0": "stationarity"},
    ... }
    >>> recommend_transformation(res)
    'levels'

    >>> res2 = {
    ...     "LLC": {"pvalue": 0.50, "h0": "unit_root"},
    ...     "IPS": {"pvalue": 0.70, "h0": "unit_root"},
    ...     "Hadri": {"pvalue": 0.001, "h0": "stationarity"},
    ... }
    >>> recommend_transformation(res2)
    'first_difference'
    """
    evidence_stationary = 0
    evidence_unit_root = 0

    for _test_name, info in results_dict.items():
        pval = info.get("pvalue", np.nan)
        h0 = info.get("h0", "unit_root").lower()

        if np.isnan(pval):
            continue

        reject = pval < alpha

        if h0 == "stationarity":
            if reject:
                evidence_unit_root += 1
            else:
                evidence_stationary += 1
        else:
            if reject:
                evidence_stationary += 1
            else:
                evidence_unit_root += 1

    if evidence_stationary > evidence_unit_root:
        return "levels"
    elif evidence_unit_root > evidence_stationary:
        return "first_difference"
    else:
        return "borderline"
