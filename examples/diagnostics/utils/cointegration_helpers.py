"""
Cointegration analysis helpers for Diagnostics tutorial series.

Utility functions for running, comparing, and visualizing panel
cointegration tests (Pedroni, Westerlund, Kao) from panelbox.

Functions:
- compare_cointegration_methods: Run all three tests and build comparison table
- plot_cointegration_residuals: Plot residuals per entity (stationarity check)
- extract_cointegration_vectors: Extract entity-specific cointegrating coefficients
- compute_half_lives: Convert error-correction speeds to half-lives
"""

import warnings
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Compare all three cointegration test methods
# ---------------------------------------------------------------------------


def compare_cointegration_methods(
    data: pd.DataFrame,
    y_var: str,
    x_vars: Union[str, list[str]],
    entity_col: str,
    time_col: str,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run Pedroni, Westerlund, and Kao cointegration tests and return a
    unified comparison table.

    Each test is executed inside a try/except block so that failures in
    one method do not prevent the others from running.  Statistics that
    are returned as dicts (multiple sub-tests) are expanded into
    separate rows; scalar statistics produce a single row.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y_var : str
        Name of the dependent variable column.
    x_vars : str or list of str
        Name(s) of independent variable column(s).
    entity_col : str
        Name of the entity identifier column.
    time_col : str
        Name of the time identifier column.
    save_path : str, optional
        If provided, save the comparison table as a CSV file at this path.

    Returns
    -------
    pd.DataFrame
        Comparison table with columns: Method, Test, Statistic, P-value,
        Reject_5pct.

    Notes
    -----
    Westerlund bootstrap is run with ``n_bootstrap=500`` and
    ``use_bootstrap=False`` by default for speed.  To obtain
    bootstrap p-values, call ``westerlund_test`` directly.
    """
    if isinstance(x_vars, str):
        x_vars = [x_vars]

    rows: list[dict] = []

    # ---- Pedroni ----
    try:
        from panelbox.diagnostics.cointegration import pedroni_test

        pedroni_result = pedroni_test(
            data,
            entity_col=entity_col,
            time_col=time_col,
            y_var=y_var,
            x_vars=x_vars,
            method="all",
        )

        stat = pedroni_result.statistic
        pval = pedroni_result.pvalue

        if isinstance(stat, dict):
            for test_name in stat:
                s = stat[test_name]
                p = pval.get(test_name, np.nan) if isinstance(pval, dict) else pval
                reject = (
                    (p < 0.05)
                    if not (p is None or (isinstance(p, float) and np.isnan(p)))
                    else None
                )
                rows.append(
                    {
                        "Method": "Pedroni",
                        "Test": test_name,
                        "Statistic": s,
                        "P-value": p,
                        "Reject_5pct": reject,
                    }
                )
        else:
            p = pval if not isinstance(pval, dict) else np.nan
            reject = (
                (p < 0.05) if not (p is None or (isinstance(p, float) and np.isnan(p))) else None
            )
            rows.append(
                {
                    "Method": "Pedroni",
                    "Test": "pedroni",
                    "Statistic": stat,
                    "P-value": p,
                    "Reject_5pct": reject,
                }
            )
    except Exception as exc:
        warnings.warn(f"Pedroni test failed: {exc}", stacklevel=2)
        rows.append(
            {
                "Method": "Pedroni",
                "Test": "ERROR",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Reject_5pct": None,
            }
        )

    # ---- Westerlund ----
    try:
        from panelbox.diagnostics.cointegration import westerlund_test

        westerlund_result = westerlund_test(
            data,
            entity_col=entity_col,
            time_col=time_col,
            y_var=y_var,
            x_vars=x_vars,
            method="all",
            use_bootstrap=False,
        )

        stat = westerlund_result.statistic
        pval = westerlund_result.pvalue

        if isinstance(stat, dict):
            for test_name in stat:
                s = stat[test_name]
                p = pval.get(test_name, np.nan) if isinstance(pval, dict) else pval
                reject = (
                    (p < 0.05)
                    if not (p is None or (isinstance(p, float) and np.isnan(p)))
                    else None
                )
                rows.append(
                    {
                        "Method": "Westerlund",
                        "Test": test_name,
                        "Statistic": s,
                        "P-value": p,
                        "Reject_5pct": reject,
                    }
                )
        else:
            p = pval if not isinstance(pval, dict) else np.nan
            reject = (
                (p < 0.05) if not (p is None or (isinstance(p, float) and np.isnan(p))) else None
            )
            rows.append(
                {
                    "Method": "Westerlund",
                    "Test": "westerlund",
                    "Statistic": stat,
                    "P-value": p,
                    "Reject_5pct": reject,
                }
            )
    except Exception as exc:
        warnings.warn(f"Westerlund test failed: {exc}", stacklevel=2)
        rows.append(
            {
                "Method": "Westerlund",
                "Test": "ERROR",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Reject_5pct": None,
            }
        )

    # ---- Kao ----
    try:
        from panelbox.diagnostics.cointegration import kao_test

        kao_result = kao_test(
            data,
            entity_col=entity_col,
            time_col=time_col,
            y_var=y_var,
            x_vars=x_vars,
            method="all",
        )

        stat = kao_result.statistic
        pval = kao_result.pvalue

        if isinstance(stat, dict):
            for test_name in stat:
                s = stat[test_name]
                p = pval.get(test_name, np.nan) if isinstance(pval, dict) else pval
                reject = (
                    (p < 0.05)
                    if not (p is None or (isinstance(p, float) and np.isnan(p)))
                    else None
                )
                rows.append(
                    {
                        "Method": "Kao",
                        "Test": test_name,
                        "Statistic": s,
                        "P-value": p,
                        "Reject_5pct": reject,
                    }
                )
        else:
            p = pval if not isinstance(pval, dict) else np.nan
            reject = (
                (p < 0.05) if not (p is None or (isinstance(p, float) and np.isnan(p))) else None
            )
            rows.append(
                {
                    "Method": "Kao",
                    "Test": "kao",
                    "Statistic": stat,
                    "P-value": p,
                    "Reject_5pct": reject,
                }
            )
    except Exception as exc:
        warnings.warn(f"Kao test failed: {exc}", stacklevel=2)
        rows.append(
            {
                "Method": "Kao",
                "Test": "ERROR",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Reject_5pct": None,
            }
        )

    df = pd.DataFrame(rows)

    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df


# ---------------------------------------------------------------------------
# 2. Plot cointegration regression residuals
# ---------------------------------------------------------------------------


def plot_cointegration_residuals(
    data: pd.DataFrame,
    residuals: pd.Series,
    entity_col: str,
    time_col: str,
    n_entities: int = 6,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot cointegration regression residuals for a subset of entities.

    Stationary-looking residuals are visual evidence supporting the
    presence of a cointegrating relationship.  Each subplot shows one
    entity's residual series with a horizontal zero-line.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format (used for entity and time columns).
    residuals : pd.Series
        Residual values aligned with *data* (same length / index).
    entity_col : str
        Name of the entity identifier column.
    time_col : str
        Name of the time identifier column.
    n_entities : int, default 6
        Number of entities to display.  Entities are chosen evenly
        across the sorted unique entity list.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the residual subplots.
    """
    # Attach residuals to the data frame for easy grouping
    plot_df = data[[entity_col, time_col]].copy()
    plot_df["residual"] = np.asarray(residuals)

    entities = sorted(plot_df[entity_col].unique())
    n_available = len(entities)
    n_show = min(n_entities, n_available)

    # Pick evenly spaced entities
    indices = np.linspace(0, n_available - 1, n_show, dtype=int)
    selected = [entities[i] for i in indices]

    n_cols = min(3, n_show)
    n_rows = int(np.ceil(n_show / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)

    for idx, entity in enumerate(selected):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx, col_idx]

        entity_data = plot_df[plot_df[entity_col] == entity].sort_values(time_col)

        ax.plot(entity_data[time_col], entity_data["residual"], linewidth=1.0)
        ax.axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"{entity}", fontsize=10)
        ax.set_xlabel(time_col)
        ax.set_ylabel("Residual")
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for idx in range(n_show, n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx, col_idx].set_visible(False)

    fig.suptitle(
        "Cointegration Regression Residuals by Entity",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


# ---------------------------------------------------------------------------
# 3. Extract entity-specific cointegration vectors
# ---------------------------------------------------------------------------


def extract_cointegration_vectors(
    result: object,
    entity_col: str,
) -> pd.DataFrame:
    """
    Extract entity-specific cointegrating coefficients from a test result.

    For Westerlund-type results that store per-entity ECM estimates, the
    function returns one row per entity with the error-correction speed
    (alpha) and its standard error.  For Pedroni-type results, entity-
    level statistics are returned when available.

    If the result object does not expose per-entity information (e.g.
    Kao, which pools across entities), the function falls back to
    reporting whatever summary statistics are available.

    Parameters
    ----------
    result : PedroniResult, WesterlundResult, KaoResult, or similar
        A cointegration test result object.  The function inspects
        common attributes such as ``statistic``, ``pvalue``,
        ``n_entities``, and any entity-level arrays that may be stored
        on the object.
    entity_col : str
        Name to use for the entity identifier column in the returned
        DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with entity-level cointegrating information.  The
        exact columns depend on the result type but will always include
        the *entity_col* column.
    """
    rows: list[dict] = []

    # --- Attempt to read entity-level attributes ---
    # Westerlund results may expose per-entity alphas/se_alphas if the
    # implementation stores them.  Check for common attribute names.
    alphas = getattr(result, "alphas", None)
    se_alphas = getattr(result, "se_alphas", None)
    entities = getattr(result, "entities", None)

    if alphas is not None:
        alphas = np.asarray(alphas)
        n = len(alphas)
        if entities is None:
            entities = [f"Entity_{i + 1}" for i in range(n)]

        for i, ent in enumerate(entities):
            row = {entity_col: ent, "alpha": alphas[i]}
            if se_alphas is not None:
                se_arr = np.asarray(se_alphas)
                row["se_alpha"] = se_arr[i] if i < len(se_arr) else np.nan
                if se_arr[i] != 0 and not np.isnan(se_arr[i]):
                    row["t_stat"] = alphas[i] / se_arr[i]
                else:
                    row["t_stat"] = np.nan
            rows.append(row)

        return pd.DataFrame(rows)

    # --- Fallback: aggregate-level summary from statistic dict ---
    stat = getattr(result, "statistic", None)
    pval = getattr(result, "pvalue", None)

    if stat is not None:
        if isinstance(stat, dict):
            for test_name, value in stat.items():
                p = np.nan
                if isinstance(pval, dict):
                    p = pval.get(test_name, np.nan)
                rows.append(
                    {
                        entity_col: "Pooled",
                        "test": test_name,
                        "statistic": value,
                        "pvalue": p,
                    }
                )
        else:
            rows.append(
                {
                    entity_col: "Pooled",
                    "test": getattr(result, "method", "unknown"),
                    "statistic": stat,
                    "pvalue": pval if not isinstance(pval, dict) else np.nan,
                }
            )

        return pd.DataFrame(rows)

    # Last resort: empty frame
    return pd.DataFrame(columns=[entity_col])


# ---------------------------------------------------------------------------
# 4. Convert error-correction speeds to half-lives
# ---------------------------------------------------------------------------


def compute_half_lives(
    gamma_values: Union[pd.Series, np.ndarray, list[float]],
) -> pd.DataFrame:
    """
    Convert error-correction speeds (gamma / alpha) to half-lives.

    The half-life measures how many periods it takes for a deviation from
    the long-run equilibrium to shrink by 50%.  For an AR(1) error-
    correction coefficient gamma (where |1 + gamma| < 1 for stability):

        half_life = -ln(2) / ln(1 + gamma)

    A more negative gamma implies faster adjustment (shorter half-life).
    Values of gamma outside (-2, 0) indicate either no adjustment
    (gamma >= 0) or explosive dynamics (gamma <= -2); these are flagged
    in the output.

    Parameters
    ----------
    gamma_values : array-like
        Error-correction speeds.  Typically these are negative values
        estimated from an ECM (e.g. the alpha_i from Westerlund tests).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gamma, half_life, interpretation.

    Examples
    --------
    >>> import numpy as np
    >>> from examples.diagnostics.utils.cointegration_helpers import compute_half_lives
    >>> gammas = np.array([-0.10, -0.25, -0.50, -0.80, 0.05])
    >>> compute_half_lives(gammas)
    """
    gamma_arr = np.asarray(gamma_values, dtype=float)

    half_lives = np.full_like(gamma_arr, np.nan)
    interpretations: list[str] = []

    for i, g in enumerate(gamma_arr):
        if np.isnan(g):
            interpretations.append("missing")
            continue

        if g >= 0:
            # No error correction (non-negative gamma)
            half_lives[i] = np.inf
            interpretations.append("no adjustment (gamma >= 0)")
        elif g <= -2:
            # Explosive
            half_lives[i] = np.nan
            interpretations.append("explosive (gamma <= -2)")
        else:
            # Valid error-correction range: -2 < gamma < 0
            # AR(1) coefficient is (1 + gamma)
            ar_coeff = 1.0 + g
            if ar_coeff <= 0:
                # Oscillatory convergence; half-life from |ar_coeff|
                half_lives[i] = -np.log(2) / np.log(abs(ar_coeff))
                interpretations.append("oscillatory convergence")
            else:
                half_lives[i] = -np.log(2) / np.log(ar_coeff)
                if half_lives[i] < 1:
                    interpretations.append("very fast adjustment")
                elif half_lives[i] < 5:
                    interpretations.append("fast adjustment")
                elif half_lives[i] < 15:
                    interpretations.append("moderate adjustment")
                else:
                    interpretations.append("slow adjustment")

    df = pd.DataFrame(
        {
            "gamma": gamma_arr,
            "half_life": half_lives,
            "interpretation": interpretations,
        }
    )

    return df
