"""
me_helpers.py
-------------
Shared plotting and table-formatting utilities for the Marginal Effects
Tutorial Series.

Functions
---------
plot_forest(me_result, title, figsize, color, zero_line)
    Forest plot showing point estimates ± confidence intervals.

format_me_table(me_result, decimals, stars)
    Return a tidy DataFrame from a MarginalEffectsResult (or plain dict).

plot_me_by_x(me_values, x_values, var_name, xlabel, ylabel, figsize)
    Line plot of marginal effects as a continuous variable changes.

plot_interaction_heatmap(interaction_df, x_var, z_var, me_var, figsize)
    Heatmap of interaction marginal effects over a grid of two variables.
"""

import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------


def plot_forest(
    me_result,
    title: str = "Marginal Effects",
    figsize: tuple = (8, 5),
    color: str = "#2196F3",
    zero_line: bool = True,
) -> plt.Figure:
    """Forest plot for a MarginalEffectsResult object (or compatible dict).

    Parameters
    ----------
    me_result : MarginalEffectsResult or dict-like
        Must expose ``.summary()`` returning a DataFrame with columns
        ``['dy/dx', 'Std. Err.', 'z', 'P>|z|', '[0.025', '0.975]']``,
        **or** be a plain ``dict`` with keys ``'effect'``, ``'se'``,
        ``'ci_low'``, ``'ci_high'``, and ``'variable'``.
    title : str
        Plot title.
    figsize : tuple
        Figure dimensions ``(width, height)`` in inches.
    color : str
        Colour for point estimates and CI bars.
    zero_line : bool
        Draw a vertical dashed line at x = 0.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = _extract_me_df(me_result)

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(df))

    ax.errorbar(
        x=df["effect"],
        y=y_pos,
        xerr=[df["effect"] - df["ci_low"], df["ci_high"] - df["effect"]],
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=1.5,
        capsize=4,
        markersize=6,
        zorder=3,
    )

    if zero_line:
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)

    # Significance stars on the right
    for i, row in df.iterrows():
        stars = _stars(row.get("pval", np.nan))
        ax.text(
            df["ci_high"].max() * 1.02 if df["ci_high"].max() > 0 else 0.02,
            y_pos[list(df.index).index(i)],
            stars,
            va="center",
            fontsize=10,
            color="red" if stars else "gray",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["variable"])
    ax.set_xlabel("Marginal Effect (dy/dx)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Table formatter
# ---------------------------------------------------------------------------


def format_me_table(me_result, decimals: int = 4, stars: bool = True) -> pd.DataFrame:
    """Return a formatted DataFrame from a MarginalEffectsResult.

    Parameters
    ----------
    me_result : MarginalEffectsResult or dict-like
        Same interface as ``plot_forest``.
    decimals : int
        Number of decimal places to display.
    stars : bool
        Append significance stars to the effect column.

    Returns
    -------
    pd.DataFrame
        Columns: ``Variable``, ``dy/dx``, ``Std. Err.``, ``z``, ``P>|z|``,
        ``[95% CI]``, and optionally ``Sig.``.
    """
    df = _extract_me_df(me_result)

    fmt = f"{{:.{decimals}f}}"

    out = pd.DataFrame()
    out["Variable"] = df["variable"]
    out["dy/dx"] = df["effect"].map(lambda v: fmt.format(v))
    out["Std. Err."] = df["se"].map(lambda v: fmt.format(v) if not np.isnan(v) else "—")

    if "z" in df.columns:
        out["z"] = df["z"].map(lambda v: fmt.format(v) if not np.isnan(v) else "—")
    if "pval" in df.columns:
        out["P>|z|"] = df["pval"].map(lambda v: f"{v:.4f}" if not np.isnan(v) else "—")

    out["[95% CI]"] = df.apply(
        lambda r: f"[{r['ci_low']:{'.'+str(decimals)+'f'}}, "
        f"{r['ci_high']:{'.'+str(decimals)+'f'}}]",
        axis=1,
    )

    if stars:
        out["Sig."] = df["pval"].map(_stars) if "pval" in df.columns else ""

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# ME vs. continuous variable
# ---------------------------------------------------------------------------


def plot_me_by_x(
    me_values: np.ndarray,
    x_values: np.ndarray,
    var_name: str = "x",
    xlabel: str | None = None,
    ylabel: str = "Marginal Effect",
    figsize: tuple = (8, 4),
    ci_low: np.ndarray | None = None,
    ci_high: np.ndarray | None = None,
    color: str = "#E91E63",
) -> plt.Figure:
    """Line plot of marginal effects as a continuous variable changes.

    Parameters
    ----------
    me_values : array-like
        Point estimates of the marginal effect at each value in ``x_values``.
    x_values : array-like
        Grid of values for the conditioning variable.
    var_name : str
        Name of the variable being varied.
    xlabel : str, optional
        X-axis label (defaults to ``var_name``).
    ylabel : str
        Y-axis label.
    figsize : tuple
        Figure size in inches.
    ci_low, ci_high : array-like, optional
        Lower and upper confidence-interval bounds. When provided a shaded
        band is drawn.
    color : str
        Line colour.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x_values, me_values, color=color, linewidth=2, label="ME")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    if ci_low is not None and ci_high is not None:
        ax.fill_between(x_values, ci_low, ci_high, alpha=0.2, color=color, label="95% CI")

    ax.set_xlabel(xlabel if xlabel is not None else var_name)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Marginal Effect of {var_name} by {var_name}")
    ax.legend()
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interaction heatmap
# ---------------------------------------------------------------------------


def plot_interaction_heatmap(
    interaction_df: pd.DataFrame,
    x_var: str,
    z_var: str,
    me_var: str,
    figsize: tuple = (8, 6),
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """Heatmap of interaction marginal effects over a grid of two variables.

    Parameters
    ----------
    interaction_df : pd.DataFrame
        Must contain columns ``x_var``, ``z_var``, and ``me_var``.
    x_var, z_var : str
        Names of the two variables forming the grid.
    me_var : str
        Name of the column holding the marginal effect values.
    figsize : tuple
        Figure size in inches.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pivot = interaction_df.pivot(index=z_var, columns=x_var, values=me_var)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, origin="lower")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2g}" for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2g}" for v in pivot.index])
    ax.set_xlabel(x_var)
    ax.set_ylabel(z_var)
    ax.set_title(f"Interaction Effect: {me_var}")

    plt.colorbar(im, ax=ax, label=me_var)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_me_df(me_result) -> pd.DataFrame:
    """Normalise various result types into a common DataFrame."""
    # Case 1: dict with 'effect', 'se', 'ci_low', 'ci_high', 'variable'
    if isinstance(me_result, dict):
        required = {"effect", "se", "ci_low", "ci_high", "variable"}
        if required.issubset(me_result.keys()):
            df = pd.DataFrame(me_result)
            return df

    # Case 2: object with .summary() that returns a DataFrame
    if hasattr(me_result, "summary"):
        try:
            summary = me_result.summary()
            if isinstance(summary, pd.DataFrame):
                return _normalise_summary_df(summary)
        except Exception:
            pass

    # Case 3: already a DataFrame
    if isinstance(me_result, pd.DataFrame):
        return _normalise_summary_df(me_result)

    raise TypeError(
        "me_result must be a MarginalEffectsResult, a compatible DataFrame, "
        "or a dict with keys 'effect', 'se', 'ci_low', 'ci_high', 'variable'."
    )


def _normalise_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column aliases to a canonical schema."""
    col_map = {
        # effect
        "dy/dx": "effect",
        "Marginal Effect": "effect",
        "me": "effect",
        "effect": "effect",
        # se
        "Std. Err.": "se",
        "std_err": "se",
        "se": "se",
        # z-stat
        "z": "z",
        "t": "z",
        # p-value
        "P>|z|": "pval",
        "P>|t|": "pval",
        "pval": "pval",
        "p_value": "pval",
        # CI bounds
        "[0.025": "ci_low",
        "ci_low": "ci_low",
        "0.975]": "ci_high",
        "ci_high": "ci_high",
    }
    df = df.rename(columns=col_map)

    # variable column
    if "variable" not in df.columns:
        df = df.reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": "variable"})
        else:
            df["variable"] = [f"x{i}" for i in range(len(df))]

    # Fill missing columns with NaN
    for col in ("effect", "se", "z", "pval", "ci_low", "ci_high"):
        if col not in df.columns:
            df[col] = np.nan

    return df[["variable", "effect", "se", "z", "pval", "ci_low", "ci_high"]]


def _stars(pval: float) -> str:
    """Return significance star string for a p-value."""
    if np.isnan(pval):
        return ""
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""
