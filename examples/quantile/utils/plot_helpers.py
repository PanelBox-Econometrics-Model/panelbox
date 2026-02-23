"""
Visualization helpers for Quantile Regression tutorials.

Provides publication-quality plotting functions for coefficient paths,
fan charts, diagnostic plots, and model comparisons.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_quantile_style():
    """Set matplotlib style for publication-quality quantile regression figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (12, 7)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12


def plot_coefficient_path(
    results_dict: dict[float, object],
    variable: str,
    ols_result: Optional[object] = None,
    ci: bool = True,
    alpha: float = 0.05,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot QR coefficient path beta(tau) for a single variable.

    Parameters
    ----------
    results_dict : dict
        Mapping from tau -> result object with .params and .std_errors attributes.
        If result has params as 1D array, variable should be an integer index.
        If result has param_names, variable can be a string.
    variable : str or int
        Variable name or index to plot.
    ols_result : object, optional
        OLS result for reference line. Must have .params attribute.
    ci : bool
        Whether to show confidence intervals.
    alpha : float
        Significance level for CI.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.Figure
    """
    from scipy import stats as sp_stats

    taus = sorted(results_dict.keys())
    coefs = []
    ses = []

    for tau in taus:
        res = results_dict[tau]
        if isinstance(variable, str) and hasattr(res, "param_names"):
            idx = res.param_names.index(variable)
        elif isinstance(variable, int):
            idx = variable
        else:
            idx = int(variable)

        if hasattr(res, "params"):
            p = res.params
            if p.ndim == 2:
                coefs.append(p[idx, 0])
            else:
                coefs.append(p[idx])
        else:
            coefs.append(res[idx])

        if hasattr(res, "std_errors"):
            se = res.std_errors
            if hasattr(se, "ndim") and se.ndim == 2:
                ses.append(se[idx, 0])
            else:
                ses.append(se[idx])
        elif hasattr(res, "bse"):
            ses.append(res.bse[idx])
        else:
            ses.append(0)

    coefs = np.array(coefs)
    ses = np.array(ses)

    fig, ax = plt.subplots(figsize=(10, 6))

    # QR coefficients
    ax.plot(taus, coefs, "o-", color="#1f77b4", linewidth=2, markersize=6, label="QR")

    # Confidence intervals
    if ci and np.any(ses > 0):
        z = sp_stats.norm.ppf(1 - alpha / 2)
        lower = coefs - z * ses
        upper = coefs + z * ses
        ax.fill_between(taus, lower, upper, alpha=0.2, color="#1f77b4")

    # OLS reference
    if ols_result is not None:
        if isinstance(variable, str) and hasattr(ols_result, "param_names"):
            ols_idx = ols_result.param_names.index(variable)
        elif isinstance(variable, int):
            ols_idx = variable
        else:
            ols_idx = int(variable)

        if hasattr(ols_result, "params"):
            ols_coef = ols_result.params[ols_idx]
        else:
            ols_coef = ols_result[ols_idx]

        ax.axhline(ols_coef, color="#d62728", linestyle="--", linewidth=1.5, label="OLS")

        if hasattr(ols_result, "bse"):
            ols_se = ols_result.bse[ols_idx]
            z = sp_stats.norm.ppf(1 - alpha / 2)
            ax.axhspan(
                ols_coef - z * ols_se,
                ols_coef + z * ols_se,
                alpha=0.1,
                color="#d62728",
            )

    ax.set_xlabel("Quantile (τ)", fontweight="bold")
    ax.set_ylabel("Coefficient", fontweight="bold")
    ax.set_title(
        title or f"Quantile Regression Coefficient Path: {variable}",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_coefficient_grid(
    results_dict: dict[float, object],
    variables: list,
    var_labels: Optional[dict] = None,
    ols_result: Optional[object] = None,
    ncols: int = 2,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grid of coefficient paths for multiple variables.

    Parameters
    ----------
    results_dict : dict
        Mapping from tau -> result object.
    variables : list
        List of variable names or indices.
    var_labels : dict, optional
        Mapping from variable to display label.
    ols_result : object, optional
        OLS result for reference lines.
    ncols : int
        Number of columns in grid.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.Figure
    """
    from scipy import stats as sp_stats

    n_vars = len(variables)
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    taus = sorted(results_dict.keys())

    for i, var in enumerate(variables):
        ax = axes[i]
        coefs = []
        ses = []

        for tau in taus:
            res = results_dict[tau]
            if isinstance(var, str) and hasattr(res, "param_names"):
                idx = res.param_names.index(var)
            elif isinstance(var, int):
                idx = var
            else:
                idx = int(var)

            if hasattr(res, "params"):
                p = res.params
                coefs.append(p[idx] if p.ndim == 1 else p[idx, 0])
            else:
                coefs.append(res[idx])

            if hasattr(res, "std_errors"):
                se = res.std_errors
                ses.append(se[idx] if se.ndim == 1 else se[idx, 0])
            elif hasattr(res, "bse"):
                ses.append(res.bse[idx])
            else:
                ses.append(0)

        coefs = np.array(coefs)
        ses = np.array(ses)

        ax.plot(taus, coefs, "o-", color="#1f77b4", linewidth=2, markersize=5)

        if np.any(ses > 0):
            z = sp_stats.norm.ppf(0.975)
            ax.fill_between(taus, coefs - z * ses, coefs + z * ses, alpha=0.2, color="#1f77b4")

        if ols_result is not None:
            if isinstance(var, str) and hasattr(ols_result, "param_names"):
                ols_idx = ols_result.param_names.index(var)
            elif isinstance(var, int):
                ols_idx = var
            else:
                ols_idx = int(var)
            ols_coef = ols_result.params[ols_idx]
            ax.axhline(ols_coef, color="#d62728", linestyle="--", linewidth=1.5)

        label = var_labels.get(var, str(var)) if var_labels else str(var)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("τ")
        ax.set_ylabel("β(τ)")
        ax.grid(True, alpha=0.3)

    # Hide extra axes
    for j in range(n_vars, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_quantile_fan_chart(
    predictions: dict[float, np.ndarray],
    x_values: np.ndarray,
    x_label: str = "X",
    y_label: str = "Y",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fan chart showing predicted quantiles as shaded bands.

    Darker = closer to median.

    Parameters
    ----------
    predictions : dict
        Mapping from tau -> predicted values (sorted by x_values).
    x_values : array
        X-axis values.
    x_label : str
        X-axis label.
    y_label : str
        Y-axis label.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    taus = sorted(predictions.keys())

    # Create symmetric pairs around median
    cmap = plt.get_cmap("Blues")
    pairs = []
    for i in range(len(taus) // 2):
        pairs.append((taus[i], taus[-(i + 1)]))

    sort_idx = np.argsort(x_values)
    x_sorted = x_values[sort_idx]

    for i, (tau_low, tau_high) in enumerate(pairs):
        pred_low = predictions[tau_low][sort_idx]
        pred_high = predictions[tau_high][sort_idx]
        intensity = 0.3 + 0.5 * (i + 1) / len(pairs)
        ax.fill_between(
            x_sorted,
            pred_low,
            pred_high,
            alpha=intensity,
            color=cmap(0.3 + 0.6 * (i + 1) / len(pairs)),
            label=f"τ=[{tau_low:.2f}, {tau_high:.2f}]",
        )

    # Plot median if available
    if 0.5 in predictions:
        ax.plot(
            x_sorted,
            predictions[0.5][sort_idx],
            color="darkblue",
            linewidth=2,
            label="Median (τ=0.50)",
        )

    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title("Quantile Fan Chart", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_check_loss(
    tau_list: list[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize check loss function rho_tau(u) for different tau values.

    Parameters
    ----------
    tau_list : list of float
        Quantile levels to plot.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.Figure
    """
    if tau_list is None:
        tau_list = [0.1, 0.5, 0.9]
    fig, ax = plt.subplots(figsize=(10, 6))

    u = np.linspace(-3, 3, 500)
    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.8, len(tau_list)))

    for tau, color in zip(tau_list, colors):
        rho = np.where(u >= 0, tau * u, (tau - 1) * u)
        ax.plot(u, rho, linewidth=2.5, color=color, label=f"τ = {tau:.1f}")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("u (residual)", fontweight="bold")
    ax.set_ylabel("ρ_τ(u)", fontweight="bold")
    ax.set_title("Check Loss Function ρ_τ(u) = u·(τ - 𝟙(u < 0))", fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_residual_diagnostics(
    result: object,
    tau: float = 0.5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    4-panel diagnostic plot: residuals vs fitted, Q-Q, histogram, scale-location.

    Parameters
    ----------
    result : object
        Fitted QR result with .predict() and model.endog.
    tau : float
        Quantile level.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.Figure
    """
    from scipy import stats as sp_stats

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Get fitted values and residuals
    if hasattr(result, "predict"):
        fitted = result.predict()
    elif hasattr(result, "fittedvalues"):
        fitted = result.fittedvalues
    else:
        fitted = result.model.exog @ (
            result.params if result.params.ndim == 1 else result.params[:, 0]
        )

    if hasattr(result, "model") and hasattr(result.model, "endog"):
        y = result.model.endog
    else:
        y = np.zeros_like(fitted)

    residuals = y - fitted

    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(fitted, residuals, alpha=0.3, s=10, color="#1f77b4")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")

    # 2. Q-Q Plot
    ax = axes[0, 1]
    sp_stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Normal Q-Q Plot")

    # 3. Histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=40, density=True, alpha=0.7, color="#1f77b4", edgecolor="white")
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(
        x_range, sp_stats.norm.pdf(x_range, residuals.mean(), residuals.std()), "r-", linewidth=2
    )
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")

    # 4. Scale-Location
    ax = axes[1, 1]
    standardized = np.abs(residuals) / (residuals.std() + 1e-10)
    ax.scatter(fitted, np.sqrt(standardized), alpha=0.3, s=10, color="#1f77b4")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("√|Standardized Residuals|")
    ax.set_title("Scale-Location")

    plt.suptitle(f"Diagnostic Plots (τ = {tau})", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_bootstrap_distribution(
    boot_params: np.ndarray,
    original_params: np.ndarray,
    var_names: list[str],
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grid of bootstrap parameter distributions with CI lines.

    Parameters
    ----------
    boot_params : array (n_boot, n_vars)
        Bootstrap parameter samples.
    original_params : array (n_vars,)
        Original point estimates.
    var_names : list of str
        Variable names.
    ci_lower : array (n_vars,)
        Lower CI bounds.
    ci_upper : array (n_vars,)
        Upper CI bounds.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.Figure
    """
    n_vars = len(var_names)
    ncols = min(3, n_vars)
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (name, ax) in enumerate(zip(var_names, axes)):
        ax.hist(
            boot_params[:, i],
            bins=40,
            density=True,
            alpha=0.7,
            color="#1f77b4",
            edgecolor="white",
        )
        ax.axvline(original_params[i], color="red", linewidth=2, label="Original")
        ax.axvline(ci_lower[i], color="green", linewidth=1.5, linestyle="--", label="CI")
        ax.axvline(ci_upper[i], color="green", linewidth=1.5, linestyle="--")
        ax.set_title(name, fontweight="bold")
        ax.legend(fontsize=8)

    for j in range(n_vars, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Bootstrap Parameter Distributions", fontweight="bold", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_crossing_detection(
    predictions: dict[float, np.ndarray],
    tau_grid: np.ndarray,
    obs_idx: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot predicted quantiles for a single observation to detect crossing.

    Parameters
    ----------
    predictions : dict
        Mapping from tau -> predicted values array.
    tau_grid : array
        Quantile levels.
    obs_idx : int
        Index of observation to examine.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    pred_values = [predictions[tau][obs_idx] for tau in tau_grid]

    ax.plot(tau_grid, pred_values, "o-", color="#1f77b4", linewidth=2, markersize=6)

    # Detect and mark crossings
    for i in range(1, len(pred_values)):
        if pred_values[i] < pred_values[i - 1]:
            ax.plot(
                [tau_grid[i - 1], tau_grid[i]],
                [pred_values[i - 1], pred_values[i]],
                "r-",
                linewidth=3,
            )
            ax.plot(tau_grid[i], pred_values[i], "ro", markersize=10, zorder=5)

    ax.set_xlabel("Quantile (τ)", fontweight="bold")
    ax.set_ylabel("Predicted Value", fontweight="bold")
    ax.set_title(
        f"Predicted Quantiles for Observation {obs_idx} (Crossing Detection)",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_qte_comparison(
    methods_results: dict[str, dict[float, float]],
    tau_grid: np.ndarray,
    ate: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare QTE estimates from different methods (CQTE, UQTE, DiD, CiC).

    Parameters
    ----------
    methods_results : dict
        Mapping from method name -> dict of tau -> QTE estimate.
    tau_grid : array
        Quantile levels.
    ate : float, optional
        Average treatment effect for reference line.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.get_cmap("Set1")(np.linspace(0, 0.8, len(methods_results)))
    markers = ["o", "s", "D", "^", "v", "<", ">"]

    for i, (method, results) in enumerate(methods_results.items()):
        taus = sorted(results.keys())
        qtes = [results[tau] for tau in taus]
        ax.plot(
            taus,
            qtes,
            marker=markers[i % len(markers)],
            color=colors[i],
            linewidth=2,
            markersize=7,
            label=method,
        )

    if ate is not None:
        ax.axhline(ate, color="black", linestyle="--", linewidth=1.5, label=f"ATE = {ate:.1f}")

    ax.set_xlabel("Quantile (τ)", fontweight="bold")
    ax.set_ylabel("Treatment Effect", fontweight="bold")
    ax.set_title("Quantile Treatment Effects Comparison", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
