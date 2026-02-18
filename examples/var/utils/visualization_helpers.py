"""
Visualization helpers for VAR tutorial notebooks.

Provides consistent, publication-quality plotting functions for:
- IRF grids and comparisons
- FEVD stacked area charts
- Coefficient heatmaps
- Stability diagrams (unit circle)
- Forecast fan charts

Functions:
- plot_irf_grid: K*K grid of IRF subplots
- plot_irf_comparison: Side-by-side Cholesky vs Generalized
- plot_fevd_stacked: Stacked area chart for FEVD
- plot_coefficient_heatmap: Heatmap of VAR coefficient matrix
- plot_stability_diagram: Eigenvalues on unit circle
- plot_forecast_fan: Fan chart with confidence intervals
- set_academic_style: Configure matplotlib for publication quality
"""

from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_academic_style():
    """
    Set matplotlib style for publication-quality figures.

    Configures font sizes, figure defaults, color palette, and grid
    style suitable for academic papers and presentations.
    """
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("ggplot")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12


def plot_irf_grid(irf_result, variables=None, figsize=None, save_path=None):
    """
    Plot IRF grid (K*K subplots) from IRFResult.

    Each subplot shows the impulse response of one variable to a shock in
    another variable, with optional confidence bands.

    Parameters
    ----------
    irf_result : IRFResult
        From results.irf(). Has attributes: irf_matrix (periods+1, K, K),
        var_names (list), ci_lower/ci_upper (optional), periods (int),
        method (str).
    variables : list of str, optional
        Subset of variables to plot. If None, plots all variables.
    figsize : tuple, optional
        Figure size. Auto-calculated from number of variables if None.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Get variable names and determine subset
    var_names = list(irf_result.var_names)
    irf_matrix = irf_result.irf_matrix
    periods = irf_result.periods
    horizons = np.arange(periods + 1)

    # Filter variables if requested
    if variables is not None:
        var_indices = [var_names.index(v) for v in variables]
        plot_names = variables
    else:
        var_indices = list(range(len(var_names)))
        plot_names = var_names

    K = len(var_indices)

    # Auto-calculate figure size
    if figsize is None:
        figsize = (4 * K, 3.5 * K)

    fig, axes = plt.subplots(K, K, figsize=figsize, squeeze=False)

    # Determine method label for the title
    try:
        method_label = irf_result.method.capitalize()
    except AttributeError:
        method_label = "IRF"

    for row_pos, i in enumerate(var_indices):
        for col_pos, j in enumerate(var_indices):
            ax = axes[row_pos, col_pos]

            # Plot the IRF line
            irf_line = irf_matrix[:, i, j]
            ax.plot(horizons, irf_line, color="#2166ac", linewidth=1.8)

            # Plot confidence bands if available
            try:
                if irf_result.ci_lower is not None and irf_result.ci_upper is not None:
                    ci_lo = irf_result.ci_lower[:, i, j]
                    ci_hi = irf_result.ci_upper[:, i, j]
                    ax.fill_between(horizons, ci_lo, ci_hi, alpha=0.2, color="#2166ac", label="CI")
            except (AttributeError, IndexError):
                pass

            # Reference line at zero
            ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

            # Title: Impulse -> Response
            ax.set_title(f"{var_names[j]} -> {var_names[i]}", fontsize=10, fontweight="bold")

            # Axis labels on edges only
            if row_pos == K - 1:
                ax.set_xlabel("Horizon", fontsize=9)
            if col_pos == 0:
                ax.set_ylabel("Response", fontsize=9)

            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Impulse Response Functions ({method_label})", fontsize=14, fontweight="bold", y=1.02
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_irf_comparison(irf_cholesky, irf_generalized, impulse, response, save_path=None):
    """
    Side-by-side comparison of Cholesky vs Generalized IRFs.

    Plots both identification strategies for the same impulse-response pair,
    using a common y-axis for direct visual comparison.

    Parameters
    ----------
    irf_cholesky : IRFResult
        Cholesky-identified IRF.
    irf_generalized : IRFResult
        Generalized IRF.
    impulse : str
        Impulse variable name.
    response : str
        Response variable name.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    results = [
        (irf_cholesky, "Cholesky", "#2166ac"),
        (irf_generalized, "Generalized", "#b2182b"),
    ]

    # Collect all y-values to set common limits
    y_min, y_max = np.inf, -np.inf

    for ax, (irf_result, label, color) in zip(axes, results):
        var_names = list(irf_result.var_names)
        imp_idx = var_names.index(impulse)
        resp_idx = var_names.index(response)

        periods = irf_result.periods
        horizons = np.arange(periods + 1)
        irf_line = irf_result.irf_matrix[:, resp_idx, imp_idx]

        ax.plot(horizons, irf_line, color=color, linewidth=2, label=label)

        # Track y range
        y_min = min(y_min, np.min(irf_line))
        y_max = max(y_max, np.max(irf_line))

        # Confidence intervals
        try:
            if irf_result.ci_lower is not None and irf_result.ci_upper is not None:
                ci_lo = irf_result.ci_lower[:, resp_idx, imp_idx]
                ci_hi = irf_result.ci_upper[:, resp_idx, imp_idx]
                ax.fill_between(horizons, ci_lo, ci_hi, alpha=0.2, color=color)
                y_min = min(y_min, np.min(ci_lo))
                y_max = max(y_max, np.max(ci_hi))
        except (AttributeError, IndexError):
            pass

        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Horizon", fontsize=12)
        ax.set_ylabel("Response", fontsize=12)
        ax.set_title(f"{label} IRF", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Apply common y-axis limits with padding
    padding = (y_max - y_min) * 0.1
    if padding == 0:
        padding = 0.05
    for ax in axes:
        ax.set_ylim(y_min - padding, y_max + padding)

    fig.suptitle(f"IRF Comparison: {impulse} -> {response}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_fevd_stacked(fevd_result, variable, save_path=None):
    """
    Stacked area chart of FEVD decomposition for a single variable.

    Displays how the forecast error variance of the chosen variable is
    decomposed into contributions from shocks to all variables over the
    forecast horizon.

    Parameters
    ----------
    fevd_result : FEVDResult
        From results.fevd(). Has attributes: decomposition (periods+1, K, K),
        var_names (list), periods (int), method (str).
    variable : str
        Variable to plot decomposition for.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    var_names = list(fevd_result.var_names)
    var_idx = var_names.index(variable)

    periods = fevd_result.periods
    horizons = np.arange(periods + 1)

    # Extract decomposition for the chosen variable: (periods+1, K)
    # decomposition[h, var_idx, j] = share explained by shock j at horizon h
    decomp = fevd_result.decomposition[:, var_idx, :]
    K = decomp.shape[1]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use a visually distinct color palette
    colors = sns.color_palette("husl", n_colors=K)

    # Stack plot
    ax.stackplot(
        horizons, *[decomp[:, j] for j in range(K)], labels=var_names, colors=colors, alpha=0.85
    )

    ax.set_xlabel("Horizon", fontsize=12)
    ax.set_ylabel("Share of Variance", fontsize=12)
    ax.set_xlim(horizons[0], horizons[-1])
    ax.set_ylim(0, 1.0)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    try:
        method_label = fevd_result.method.capitalize()
    except AttributeError:
        method_label = "FEVD"

    ax.set_title(
        f"Forecast Error Variance Decomposition: {variable} ({method_label})",
        fontsize=14,
        fontweight="bold",
    )

    ax.legend(
        loc="upper right", title="Shock Source", fontsize=10, title_fontsize=11, framealpha=0.9
    )
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_coefficient_heatmap(results, lag=1, save_path=None):
    """
    Heatmap of VAR coefficient matrix A_l with annotated values.

    Visualizes the strength and direction of inter-variable relationships
    at a specific lag using a diverging colormap centered at zero.

    Parameters
    ----------
    results : PanelVARResult
        Fitted VAR result. Has A_matrices (list of K*K ndarrays),
        endog_names (list of str), K (int), p (int).
    lag : int, default 1
        Which lag's coefficient matrix to display (1-indexed).
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Validate lag
    try:
        p = results.p
    except AttributeError:
        p = len(results.A_matrices)

    if lag < 1 or lag > p:
        raise ValueError(f"lag must be between 1 and {p}, got {lag}")

    # Get coefficient matrix
    A_l = results.A_matrices[lag - 1]

    # Get variable names
    try:
        var_names = list(results.endog_names)
    except AttributeError:
        var_names = [f"y{k}" for k in range(A_l.shape[0])]

    # Create DataFrame for seaborn
    coef_df = pd.DataFrame(A_l, index=var_names, columns=var_names)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine symmetric color limits
    vmax = np.max(np.abs(A_l))
    if vmax < 1e-10:
        vmax = 1.0  # Avoid degenerate colorbar

    sns.heatmap(
        coef_df,
        annot=True,
        fmt=".4f",
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"label": "Coefficient Value", "shrink": 0.8},
        ax=ax,
    )

    ax.set_xlabel("Cause (column variable at t-{})".format(lag), fontsize=12)
    ax.set_ylabel("Effect (row equation)", fontsize=12)
    ax.set_title(f"VAR Coefficient Matrix A$_{lag}$ (Lag {lag})", fontsize=14, fontweight="bold")

    # Rotate tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_stability_diagram(results, save_path=None):
    """
    Unit circle plot showing eigenvalues of companion matrix.

    Plots the eigenvalues of the VAR companion matrix on the complex plane
    alongside the unit circle. All eigenvalues must lie strictly inside the
    unit circle for the VAR system to be stable.

    Parameters
    ----------
    results : PanelVARResult
        Has eigenvalues property returning a complex array, and
        companion_matrix() method.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Obtain eigenvalues
    try:
        eigenvalues = results.eigenvalues
    except AttributeError:
        companion = results.companion_matrix()
        eigenvalues = np.linalg.eigvals(companion)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "b--", linewidth=1.5, alpha=0.6, label="Unit Circle")

    # Compute moduli for coloring
    moduli = np.abs(eigenvalues)
    max_modulus = np.max(moduli)

    # Separate stable and unstable eigenvalues
    stable_mask = moduli < 1.0
    unstable_mask = ~stable_mask

    # Plot stable eigenvalues (inside circle)
    if np.any(stable_mask):
        ax.scatter(
            eigenvalues[stable_mask].real,
            eigenvalues[stable_mask].imag,
            c="#2ca02c",
            s=80,
            zorder=5,
            edgecolors="darkgreen",
            linewidth=1.2,
            label=f"Stable (|z| < 1)",
            marker="o",
        )

    # Plot unstable eigenvalues (outside or on circle)
    if np.any(unstable_mask):
        ax.scatter(
            eigenvalues[unstable_mask].real,
            eigenvalues[unstable_mask].imag,
            c="#d62728",
            s=100,
            zorder=5,
            edgecolors="darkred",
            linewidth=1.5,
            label=f"Unstable (|z| >= 1)",
            marker="X",
        )

    # Annotate maximum modulus
    max_idx = np.argmax(moduli)
    max_eig = eigenvalues[max_idx]
    ax.annotate(
        f"max |z| = {max_modulus:.4f}",
        xy=(max_eig.real, max_eig.imag),
        xytext=(max_eig.real + 0.15, max_eig.imag + 0.15),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.9),
    )

    # Axes styling
    ax.set_xlabel("Real Part", fontsize=12)
    ax.set_ylabel("Imaginary Part", fontsize=12)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Determine axis limits
    limit = max(1.3, max_modulus + 0.3)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    # Stability verdict
    is_stable = max_modulus < 1.0
    verdict = "STABLE" if is_stable else "UNSTABLE"
    verdict_color = "#2ca02c" if is_stable else "#d62728"

    ax.set_title(
        f"Eigenvalues of Companion Matrix ({verdict})",
        fontsize=14,
        fontweight="bold",
        color=verdict_color,
    )

    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_forecast_fan(forecast_result, entity, variable, actual=None, save_path=None):
    """
    Fan chart of forecast with confidence intervals.

    Displays point forecasts as a line with shaded confidence bands,
    and optionally overlays actual observed values for evaluation.

    Parameters
    ----------
    forecast_result : ForecastResult
        Has forecasts (steps, N, K), ci_lower, ci_upper,
        endog_names, entity_names.
    entity : str or int
        Entity name or index.
    variable : str
        Variable name to plot.
    actual : pd.Series or np.ndarray, optional
        Actual values for comparison. Should have length >= forecast horizon.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Resolve entity index
    if isinstance(entity, str):
        try:
            entity_idx = list(forecast_result.entity_names).index(entity)
        except (ValueError, AttributeError):
            raise ValueError(f"Entity '{entity}' not found in entity_names.")
        entity_label = entity
    else:
        entity_idx = entity
        try:
            entity_label = forecast_result.entity_names[entity_idx]
        except (AttributeError, IndexError):
            entity_label = f"Entity {entity_idx}"

    # Resolve variable index
    try:
        var_names = list(forecast_result.endog_names)
    except AttributeError:
        var_names = [f"y{k}" for k in range(forecast_result.forecasts.shape[-1])]
    var_idx = var_names.index(variable)

    # Extract forecast values
    forecasts = forecast_result.forecasts
    # Handle both (steps, N, K) and (steps, K) shapes
    if forecasts.ndim == 2:
        fcst = forecasts[:, var_idx]
    else:
        fcst = forecasts[:, entity_idx, var_idx]

    steps = len(fcst)
    horizons = np.arange(1, steps + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot point forecast
    ax.plot(
        horizons,
        fcst,
        color="#2166ac",
        linewidth=2,
        marker="o",
        markersize=5,
        label="Forecast",
        zorder=4,
    )

    # Confidence intervals
    has_ci = False
    try:
        if forecast_result.ci_lower is not None and forecast_result.ci_upper is not None:
            ci_lower = forecast_result.ci_lower
            ci_upper = forecast_result.ci_upper

            if ci_lower.ndim == 2:
                ci_lo = ci_lower[:, var_idx]
                ci_hi = ci_upper[:, var_idx]
            else:
                ci_lo = ci_lower[:, entity_idx, var_idx]
                ci_hi = ci_upper[:, entity_idx, var_idx]

            # Primary band (darker)
            ax.fill_between(
                horizons,
                ci_lo,
                ci_hi,
                alpha=0.25,
                color="#2166ac",
                label=f"{_get_ci_level(forecast_result):.0%} CI",
            )

            # Boundary lines
            ax.plot(horizons, ci_lo, color="#2166ac", linewidth=0.8, linestyle=":", alpha=0.6)
            ax.plot(horizons, ci_hi, color="#2166ac", linewidth=0.8, linestyle=":", alpha=0.6)
            has_ci = True
    except (AttributeError, IndexError):
        pass

    # Overlay actual values if provided
    if actual is not None:
        if isinstance(actual, pd.Series):
            actual_values = actual.values
        else:
            actual_values = np.asarray(actual)

        # Trim to forecast horizon
        actual_plot = actual_values[:steps]
        ax.plot(
            horizons[: len(actual_plot)],
            actual_plot,
            color="#d62728",
            linewidth=2,
            marker="s",
            markersize=5,
            label="Actual",
            zorder=5,
        )

    ax.set_xlabel("Forecast Horizon", fontsize=12)
    ax.set_ylabel(variable, fontsize=12)
    ax.set_title(f"Forecast: {variable} ({entity_label})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Integer x-axis ticks
    ax.set_xticks(horizons)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _get_ci_level(forecast_result):
    """
    Safely extract confidence level from a ForecastResult.

    Parameters
    ----------
    forecast_result : ForecastResult
        Forecast result object.

    Returns
    -------
    float
        Confidence level (e.g. 0.95). Defaults to 0.95 if not available.
    """
    try:
        level = forecast_result.ci_level
        if level is not None:
            return level
    except AttributeError:
        pass
    return 0.95
