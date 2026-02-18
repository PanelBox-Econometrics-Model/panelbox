"""
Visualization helpers for discrete choice tutorials.

This module provides standardized plotting functions for discrete choice
models, including link function comparisons, classification diagnostics,
marginal effects, and model comparisons.

Functions:
    plot_link_functions         : Compare logit, probit, and LPM links
    plot_predicted_probabilities: Plot predicted vs actual probabilities
    plot_confusion_matrix       : Confusion matrix heatmap
    plot_roc_curve              : ROC curve with AUC
    plot_marginal_effects       : Marginal effects with confidence intervals
    plot_choice_probabilities   : Predicted probabilities for multinomial models

Author: PanelBox Contributors
Date: 2026-02-16
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import auc, confusion_matrix, roc_curve


def plot_link_functions(
    compare: List[str] = ["logit", "probit", "lpm"],
    x_range: Tuple[float, float] = (-4, 4),
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot and compare different link functions for binary choice models.

    Parameters
    ----------
    compare : list of str, default=['logit', 'probit', 'lpm']
        Link functions to compare. Options: 'logit', 'probit', 'lpm', 'cloglog'.
    x_range : tuple of float, default=(-4, 4)
        Range of x-values (xb) to plot.
    figsize : tuple of int, default=(10, 6)
        Figure size in inches.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.

    Examples
    --------
    >>> fig = plot_link_functions(compare=['logit', 'probit'])
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.linspace(x_range[0], x_range[1], 500)

    link_functions = {
        "logit": lambda z: 1 / (1 + np.exp(-z)),
        "probit": lambda z: stats.norm.cdf(z),
        "lpm": lambda z: np.clip(z, 0, 1),
        "cloglog": lambda z: 1 - np.exp(-np.exp(z)),
    }

    link_labels = {
        "logit": "Logit: $\\Lambda(x\\beta) = 1/(1+e^{-x\\beta})$",
        "probit": "Probit: $\\Phi(x\\beta)$",
        "lpm": "Linear Probability: $x\\beta$",
        "cloglog": "Complementary log-log",
    }

    colors = {"logit": "blue", "probit": "red", "lpm": "green", "cloglog": "orange"}
    linestyles = {"logit": "-", "probit": "--", "lpm": "-.", "cloglog": ":"}

    for link in compare:
        if link not in link_functions:
            raise ValueError(f"Unknown link function: {link}")

        y = link_functions[link](x)
        ax.plot(
            x,
            y,
            label=link_labels[link],
            color=colors[link],
            linestyle=linestyles[link],
            linewidth=2,
        )

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="P=0.5")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("$x\\beta$ (Index)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("Comparison of Link Functions", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_predicted_probabilities(
    results,
    actual_y: np.ndarray,
    bins: int = 20,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot predicted probabilities vs actual outcomes.

    Creates a two-panel figure: (1) histogram of predicted probabilities
    by actual outcome, (2) calibration plot comparing predicted vs observed.

    Parameters
    ----------
    results : PanelBoxResults
        Fitted model results with predict_proba() method.
    actual_y : np.ndarray
        Actual binary outcomes (0/1).
    bins : int, default=20
        Number of bins for calibration plot.
    figsize : tuple of int, default=(12, 5)
        Figure size in inches.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Get predicted probabilities
    try:
        pred_probs = results.predict_proba()
    except AttributeError:
        pred_probs = results.predict()

    # Panel 1: Distribution of predicted probabilities
    ax1.hist(
        pred_probs[actual_y == 0],
        bins=30,
        alpha=0.5,
        label="Actual = 0",
        color="blue",
        density=True,
    )
    ax1.hist(
        pred_probs[actual_y == 1], bins=30, alpha=0.5, label="Actual = 1", color="red", density=True
    )
    ax1.set_xlabel("Predicted Probability", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Distribution of Predicted Probabilities", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Calibration plot
    df = pd.DataFrame({"pred": pred_probs, "actual": actual_y})
    df["bin"] = pd.cut(df["pred"], bins=bins, include_lowest=True)
    calibration = (
        df.groupby("bin", observed=True)
        .agg(mean_pred=("pred", "mean"), mean_actual=("actual", "mean"), count=("actual", "size"))
        .reset_index()
    )

    ax2.scatter(
        calibration["mean_pred"],
        calibration["mean_actual"],
        s=calibration["count"],
        alpha=0.6,
        color="purple",
    )
    ax2.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)
    ax2.set_xlabel("Predicted Probability", fontsize=11)
    ax2.set_ylabel("Observed Frequency", fontsize=11)
    ax2.set_title("Calibration Plot", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    results,
    actual_y: np.ndarray,
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix for binary classification.

    Parameters
    ----------
    results : PanelBoxResults
        Fitted model results with predict_proba() method.
    actual_y : np.ndarray
        Actual binary outcomes (0/1).
    threshold : float, default=0.5
        Probability threshold for classification.
    figsize : tuple of int, default=(8, 6)
        Figure size in inches.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get predictions
    try:
        pred_probs = results.predict_proba()
    except AttributeError:
        pred_probs = results.predict()

    pred_class = (pred_probs >= threshold).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(actual_y, pred_class)

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"label": "Count"})

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix (threshold = {threshold})", fontsize=14, fontweight="bold")
    ax.set_xticklabels(["Negative (0)", "Positive (1)"])
    ax.set_yticklabels(["Negative (0)", "Positive (1)"], rotation=0)

    # Add accuracy metrics
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0

    metrics_text = f"Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}"
    ax.text(
        2.3,
        0.5,
        metrics_text,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curve(
    results,
    actual_y: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curve with AUC for binary classification.

    Parameters
    ----------
    results : PanelBoxResults
        Fitted model results with predict_proba() method.
    actual_y : np.ndarray
        Actual binary outcomes (0/1).
    figsize : tuple of int, default=(8, 6)
        Figure size in inches.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get predicted probabilities
    try:
        pred_probs = results.predict_proba()
    except AttributeError:
        pred_probs = results.predict()

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(actual_y, pred_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_marginal_effects(
    results,
    variable: str,
    at_values: Optional[Union[str, dict]] = "mean",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot marginal effects with confidence intervals.

    Parameters
    ----------
    results : PanelBoxResults
        Fitted model results with marginal_effects() method.
    variable : str
        Variable name for which to plot marginal effects.
    at_values : str or dict, default='mean'
        Values at which to evaluate marginal effects.
        - 'mean': Evaluate at sample means
        - dict: Specific values for covariates
    figsize : tuple of int, default=(10, 6)
        Figure size in inches.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get marginal effects
    try:
        me = results.marginal_effects(at=at_values)
    except AttributeError:
        raise AttributeError("Results object does not have marginal_effects() method")

    # Extract marginal effect for specified variable
    if variable not in me.index:
        raise ValueError(f"Variable '{variable}' not found in marginal effects")

    me_value = me.loc[variable, "marginal_effect"]
    se = me.loc[variable, "std_err"]
    ci_lower = me.loc[variable, "ci_lower"]
    ci_upper = me.loc[variable, "ci_upper"]

    # Create bar plot
    ax.bar(0, me_value, color="steelblue", alpha=0.7, label="Marginal Effect")
    ax.errorbar(
        0,
        me_value,
        yerr=[[me_value - ci_lower], [ci_upper - me_value]],
        fmt="none",
        color="black",
        capsize=10,
        capthick=2,
        label="95% Confidence Interval",
    )

    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_ylabel("Marginal Effect on Pr(Y=1)", fontsize=12)
    ax.set_title(f"Marginal Effect of {variable}", fontsize=14, fontweight="bold")
    ax.set_xticks([0])
    ax.set_xticklabels([variable])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add text with exact values
    text = f"ME = {me_value:.4f}\nSE = {se:.4f}\n95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]"
    ax.text(
        0.5,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_choice_probabilities(
    results,
    data: pd.DataFrame,
    individual_id: Optional[int] = None,
    n_individuals: int = 10,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot predicted choice probabilities for multinomial models.

    Parameters
    ----------
    results : PanelBoxResults
        Fitted multinomial model results.
    data : pd.DataFrame
        Data used for prediction.
    individual_id : int, optional
        Specific individual ID to plot. If None, plots first n_individuals.
    n_individuals : int, default=10
        Number of individuals to plot if individual_id is None.
    figsize : tuple of int, default=(12, 6)
        Figure size in inches.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get predicted probabilities
    try:
        pred_probs = results.predict_proba()
    except AttributeError:
        raise AttributeError("Results object does not have predict_proba() method")

    # Handle individual selection
    if individual_id is not None:
        mask = data["id"] == individual_id
        plot_data = pred_probs[mask]
        title = f"Predicted Choice Probabilities (Individual {individual_id})"
    else:
        unique_ids = data["id"].unique()[:n_individuals]
        mask = data["id"].isin(unique_ids)
        plot_data = pred_probs[mask]
        title = f"Predicted Choice Probabilities (First {n_individuals} individuals)"

    # Create stacked bar chart
    n_alternatives = plot_data.shape[1] if len(plot_data.shape) > 1 else 1
    x = np.arange(len(plot_data))

    bottom = np.zeros(len(plot_data))
    colors = plt.cm.Set3(np.linspace(0, 1, n_alternatives))

    for alt in range(n_alternatives):
        probs = plot_data[:, alt] if len(plot_data.shape) > 1 else plot_data
        ax.bar(x, probs, bottom=bottom, label=f"Alternative {alt}", color=colors[alt], alpha=0.8)
        bottom += probs

    ax.set_xlabel("Observation", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    # Example usage
    print("Visualization helpers for discrete choice models")
    print("Example: plot_link_functions()")

    fig = plot_link_functions(compare=["logit", "probit", "lpm"])
    plt.show()
