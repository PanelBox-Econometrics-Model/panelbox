"""
plot_helpers.py — Reusable plotting utilities for the validation tutorial series.

All functions return a ``matplotlib.figure.Figure`` so callers can save or
display them with ``fig.savefig(...)`` / ``plt.show()``.

Dependencies
------------
matplotlib >= 3.5 (required)
scipy >= 1.7   (for ACF computation)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from collections.abc import Sequence

# Use non-interactive backend if running headless
matplotlib.rcParams.update(
    {
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


# ---------------------------------------------------------------------------
# 1. plot_residuals_by_entity
# ---------------------------------------------------------------------------


def plot_residuals_by_entity(
    resid: pd.Series,
    entity_col: pd.Series,
    *,
    max_entities: int = 20,
    title: str = "Residuals by Entity",
    figsize: tuple[int, int] = (14, 5),
) -> Figure:
    """
    Boxplot of residuals grouped by entity.

    Parameters
    ----------
    resid : pd.Series
        Model residuals.
    entity_col : pd.Series
        Entity identifier aligned with *resid*.
    max_entities : int
        Maximum number of entities to display (sorted by median residual).
    title : str
        Plot title.
    figsize : tuple[int, int]
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = pd.DataFrame({"residual": resid.values, "entity": entity_col.values})
    medians = df.groupby("entity")["residual"].median().sort_values()
    top_entities = medians.index[:max_entities].tolist()
    df_sub = df[df["entity"].isin(top_entities)]
    groups = [df_sub.loc[df_sub["entity"] == e, "residual"].values for e in top_entities]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(
        groups,
        labels=[str(e) for e in top_entities],
        patch_artist=True,
        boxprops={"facecolor": "steelblue", "alpha": 0.6},
    )
    ax.axhline(0, color="red", linewidth=1, linestyle="--", label="Zero line")
    ax.set_xlabel("Entity")
    ax.set_ylabel("Residual")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. plot_acf_panel
# ---------------------------------------------------------------------------


def plot_acf_panel(
    resid: pd.Series,
    entity_col: pd.Series,
    lags: int = 10,
    *,
    n_sample: int = 6,
    title: str = "ACF of Residuals — Sample Entities",
    figsize: tuple[int, int] = (14, 8),
    random_state: int = 42,
) -> Figure:
    """
    ACF plots for a random sample of entities.

    Parameters
    ----------
    resid : pd.Series
        Model residuals.
    entity_col : pd.Series
        Entity identifier aligned with *resid*.
    lags : int
        Number of lags to display.
    n_sample : int
        Number of entities to sample.
    """
    from scipy import signal as sp_signal

    rng = np.random.default_rng(random_state)
    df = pd.DataFrame({"residual": resid.values, "entity": entity_col.values})
    entities = df["entity"].unique()
    sample = rng.choice(entities, size=min(n_sample, len(entities)), replace=False)

    ncols = 3
    nrows = int(np.ceil(len(sample) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes = np.array(axes).flatten()

    conf_level = 1.96 / np.sqrt(lags)  # approximate CI

    for i, entity in enumerate(sample):
        ax = axes[i]
        series = df.loc[df["entity"] == entity, "residual"].values
        n = len(series)
        acf_vals = [1.0]
        for lag in range(1, lags + 1):
            if lag < n:
                acf_vals.append(np.corrcoef(series[lag:], series[:-lag])[0, 1])
            else:
                acf_vals.append(np.nan)
        lag_arr = np.arange(lags + 1)
        ax.bar(lag_arr, acf_vals, color="steelblue", alpha=0.7)
        ax.axhline(conf_level, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(-conf_level, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"Entity {entity}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_ylim(-1.1, 1.1)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. plot_correlation_heatmap
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    resid: pd.Series,
    entity_col: pd.Series,
    *,
    max_entities: int = 20,
    title: str = "Cross-Entity Residual Correlations",
    figsize: tuple[int, int] = (10, 8),
) -> Figure:
    """
    Heatmap of cross-entity residual correlation matrix.

    Parameters
    ----------
    resid : pd.Series
        Model residuals.
    entity_col : pd.Series
        Entity identifier aligned with *resid*.
    max_entities : int
        Cap on the number of entities shown (select largest by obs count).
    """
    df = pd.DataFrame({"residual": resid.values, "entity": entity_col.values})
    counts = df.groupby("entity").size().sort_values(ascending=False)
    top = counts.index[:max_entities]
    wide = df[df["entity"].isin(top)].pivot_table(
        index=df.index, columns="entity", values="residual"
    )
    corr = wide.corr()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = [str(e) for e in corr.columns]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. plot_bootstrap_distribution
# ---------------------------------------------------------------------------


def plot_bootstrap_distribution(
    boot_estimates: np.ndarray,
    param_name: str = "Parameter",
    ci: float = 0.95,
    *,
    point_estimate: float | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (8, 5),
) -> Figure:
    """
    Histogram of bootstrap estimates with confidence interval overlay.

    Parameters
    ----------
    boot_estimates : np.ndarray
        1-D array of bootstrap replications.
    param_name : str
        Label for the x-axis.
    ci : float
        Coverage probability for the CI (default 0.95).
    point_estimate : float, optional
        Original (non-bootstrap) estimate to mark with a vertical line.
    """
    alpha = 1 - ci
    lo = np.percentile(boot_estimates, 100 * alpha / 2)
    hi = np.percentile(boot_estimates, 100 * (1 - alpha / 2))

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(boot_estimates, bins=40, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(
        lo,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"{int(ci * 100)}% CI: [{lo:.3f}, {hi:.3f}]",
    )
    ax.axvline(hi, color="red", linestyle="--", linewidth=1.5)
    if point_estimate is not None:
        ax.axvline(
            point_estimate, color="black", linewidth=2, label=f"Estimate: {point_estimate:.3f}"
        )
    ax.set_xlabel(param_name)
    ax.set_ylabel("Count")
    ax.set_title(title or f"Bootstrap Distribution — {param_name}")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. plot_cv_predictions
# ---------------------------------------------------------------------------


def plot_cv_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    folds: np.ndarray | None = None,
    *,
    title: str = "Cross-Validation: Actual vs. Predicted",
    figsize: tuple[int, int] = (8, 6),
) -> Figure:
    """
    Scatter plot of actual vs. predicted values, coloured by CV fold.

    Parameters
    ----------
    actual : np.ndarray
        Ground-truth values.
    predicted : np.ndarray
        Model predictions.
    folds : np.ndarray, optional
        Integer fold labels for each observation (same length as actual).
    """
    fig, ax = plt.subplots(figsize=figsize)
    if folds is not None:
        unique_folds = np.unique(folds)
        cmap = plt.cm.get_cmap("tab10", len(unique_folds))
        for k, fold in enumerate(unique_folds):
            mask = folds == fold
            ax.scatter(
                actual[mask], predicted[mask], s=20, alpha=0.6, color=cmap(k), label=f"Fold {fold}"
            )
        ax.legend(title="Fold", markerscale=1.5, fontsize=8)
    else:
        ax.scatter(actual, predicted, s=20, alpha=0.6, color="steelblue")

    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="45° line")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    ax.text(0.05, 0.93, f"RMSE = {rmse:.4f}", transform=ax.transAxes, fontsize=9, va="top")
    if folds is None:
        ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. plot_influence_index
# ---------------------------------------------------------------------------


def plot_influence_index(
    cooks_d: np.ndarray,
    threshold: float | None = None,
    *,
    labels: Sequence[str] | None = None,
    title: str = "Cook's Distance — Influence Diagnostics",
    figsize: tuple[int, int] = (12, 4),
) -> Figure:
    """
    Index plot of Cook's distance with a threshold line.

    Parameters
    ----------
    cooks_d : np.ndarray
        Cook's D values, one per observation.
    threshold : float, optional
        Influence threshold (default: 4 / n).
    labels : sequence of str, optional
        Labels for flagged observations.
    """
    n = len(cooks_d)
    if threshold is None:
        threshold = 4.0 / n

    indices = np.arange(n)
    flagged = cooks_d > threshold

    fig, ax = plt.subplots(figsize=figsize)
    ax.vlines(indices[~flagged], 0, cooks_d[~flagged], colors="steelblue", linewidth=0.8, alpha=0.7)
    ax.vlines(
        indices[flagged], 0, cooks_d[flagged], colors="crimson", linewidth=1.2, label="Influential"
    )
    ax.scatter(indices[flagged], cooks_d[flagged], color="crimson", s=30, zorder=5)
    ax.axhline(
        threshold,
        color="darkorange",
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold = {threshold:.4f}",
    )

    if labels is not None:
        for i in np.where(flagged)[0]:
            ax.annotate(
                labels[i], (i, cooks_d[i]), textcoords="offset points", xytext=(0, 4), fontsize=7
            )

    ax.set_xlabel("Observation index")
    ax.set_ylabel("Cook's D")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. plot_forest_plot
# ---------------------------------------------------------------------------


def plot_forest_plot(
    estimates: np.ndarray,
    se: np.ndarray,
    labels: Sequence[str],
    *,
    ci_level: float = 0.95,
    reference: float = 0.0,
    title: str = "Coefficient Forest Plot",
    figsize: tuple[int, int] = (8, 5),
) -> Figure:
    """
    Forest plot showing coefficient estimates with confidence intervals.

    Parameters
    ----------
    estimates : np.ndarray
        Point estimates (one per specification/model).
    se : np.ndarray
        Standard errors aligned with *estimates*.
    labels : sequence of str
        Labels for each specification/model.
    ci_level : float
        Coverage probability (default 0.95 → z = 1.96).
    reference : float
        Reference vertical line (default 0).
    """
    from scipy import stats as sp_stats

    z = sp_stats.norm.ppf(0.5 + ci_level / 2)
    lo = np.asarray(estimates) - z * np.asarray(se)
    hi = np.asarray(estimates) + z * np.asarray(se)
    n = len(estimates)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hlines(y_pos, lo, hi, colors="steelblue", linewidth=2, alpha=0.8)
    ax.scatter(estimates, y_pos, color="steelblue", s=60, zorder=5)
    ax.axvline(
        reference, color="red", linestyle="--", linewidth=1, label=f"Reference = {reference}"
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Coefficient")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
