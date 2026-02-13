"""
Visualization functions for Panel VAR models.

This module provides plotting functions for Panel VAR diagnostics and results.
"""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def plot_stability(
    eigenvalues: np.ndarray,
    title: str = "VAR Stability (Roots of Companion Matrix)",
    backend: str = "matplotlib",
    figsize: Tuple[int, int] = (8, 8),
    show: bool = True,
) -> Optional[object]:
    """
    Plot eigenvalues of the companion matrix with unit circle.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of complex eigenvalues from companion matrix
    title : str, default="VAR Stability (Roots of Companion Matrix)"
        Plot title
    backend : str, default="matplotlib"
        Plotting backend: "matplotlib" or "plotly"
    figsize : tuple, default=(8, 8)
        Figure size for matplotlib (width, height in inches)
    show : bool, default=True
        Whether to display the plot immediately

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        Figure object (if show=False)

    Notes
    -----
    A VAR system is stable if all eigenvalues lie within the unit circle
    (modulus < 1). The plot shows:
    - Unit circle in blue (dashed line)
    - Eigenvalues as points:
      - Green: stable (modulus < 1)
      - Red: unstable (modulus >= 1)

    Examples
    --------
    >>> from panelbox.var import PanelVAR, PanelVARData
    >>> # ... fit model ...
    >>> results.plot_stability()
    >>> # Or get figure for customization
    >>> fig = results.plot_stability(show=False)
    >>> plt.savefig('stability.png')
    """
    if backend == "matplotlib":
        return _plot_stability_matplotlib(eigenvalues, title, figsize, show)
    elif backend == "plotly":
        if not HAS_PLOTLY:
            raise ImportError(
                "Plotly is required for backend='plotly'. Install with: pip install plotly"
            )
        return _plot_stability_plotly(eigenvalues, title, show)
    else:
        raise ValueError(f"backend must be 'matplotlib' or 'plotly', got '{backend}'")


def _plot_stability_matplotlib(
    eigenvalues: np.ndarray, title: str, figsize: Tuple[int, int], show: bool
) -> Optional[plt.Figure]:
    """Plot stability using matplotlib."""
    fig, ax = plt.subplots(figsize=figsize)

    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.plot(circle_x, circle_y, "b--", linewidth=2, label="Unit Circle", alpha=0.7)

    # Plot eigenvalues
    eig_real = eigenvalues.real
    eig_imag = eigenvalues.imag
    moduli = np.abs(eigenvalues)

    # Separate stable and unstable eigenvalues
    stable_mask = moduli < 1.0
    unstable_mask = ~stable_mask

    if np.any(stable_mask):
        ax.scatter(
            eig_real[stable_mask],
            eig_imag[stable_mask],
            c="green",
            s=100,
            marker="o",
            label="Stable (|λ| < 1)",
            alpha=0.7,
            edgecolors="black",
            linewidths=1.5,
        )

    if np.any(unstable_mask):
        ax.scatter(
            eig_real[unstable_mask],
            eig_imag[unstable_mask],
            c="red",
            s=100,
            marker="X",
            label="Unstable (|λ| ≥ 1)",
            alpha=0.7,
            edgecolors="black",
            linewidths=1.5,
        )

    # Add labels with modulus for each eigenvalue
    for i, (re, im, mod) in enumerate(zip(eig_real, eig_imag, moduli)):
        ax.annotate(
            f"|λ|={mod:.3f}",
            xy=(re, im),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
        )

    # Set equal aspect ratio
    ax.set_aspect("equal", adjustable="box")

    # Set axis labels and title
    ax.set_xlabel("Real Part", fontsize=12)
    ax.set_ylabel("Imaginary Part", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle=":")

    # Add legend
    ax.legend(loc="upper right", fontsize=10)

    # Add horizontal and vertical lines at origin
    ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.5)

    # Set axis limits with some padding
    max_val = max(1.2, np.max(np.abs(eigenvalues)) * 1.1)
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    plt.tight_layout()

    if show:
        plt.show()
        return None
    else:
        return fig


def _plot_stability_plotly(eigenvalues: np.ndarray, title: str, show: bool) -> Optional[go.Figure]:
    """Plot stability using plotly."""
    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    # Eigenvalue properties
    eig_real = eigenvalues.real
    eig_imag = eigenvalues.imag
    moduli = np.abs(eigenvalues)
    stable_mask = moduli < 1.0

    # Create figure
    fig = go.Figure()

    # Add unit circle
    fig.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            name="Unit Circle",
            line=dict(color="blue", width=2, dash="dash"),
            hoverinfo="name",
        )
    )

    # Add stable eigenvalues
    if np.any(stable_mask):
        stable_hover = [
            f"λ = {re:.4f} + {im:.4f}i<br>|λ| = {mod:.4f}"
            for re, im, mod in zip(
                eig_real[stable_mask], eig_imag[stable_mask], moduli[stable_mask]
            )
        ]
        fig.add_trace(
            go.Scatter(
                x=eig_real[stable_mask],
                y=eig_imag[stable_mask],
                mode="markers",
                name="Stable (|λ| < 1)",
                marker=dict(
                    color="green", size=12, symbol="circle", line=dict(color="black", width=1)
                ),
                text=stable_hover,
                hoverinfo="text",
            )
        )

    # Add unstable eigenvalues
    if np.any(~stable_mask):
        unstable_hover = [
            f"λ = {re:.4f} + {im:.4f}i<br>|λ| = {mod:.4f}"
            for re, im, mod in zip(
                eig_real[~stable_mask], eig_imag[~stable_mask], moduli[~stable_mask]
            )
        ]
        fig.add_trace(
            go.Scatter(
                x=eig_real[~stable_mask],
                y=eig_imag[~stable_mask],
                mode="markers",
                name="Unstable (|λ| ≥ 1)",
                marker=dict(color="red", size=12, symbol="x", line=dict(color="black", width=1)),
                text=unstable_hover,
                hoverinfo="text",
            )
        )

    # Set layout
    max_val = max(1.2, np.max(np.abs(eigenvalues)) * 1.1)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, weight="bold")),
        xaxis=dict(
            title="Real Part",
            range=[-max_val, max_val],
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            showgrid=True,
            gridcolor="lightgray",
        ),
        yaxis=dict(
            title="Imaginary Part",
            range=[-max_val, max_val],
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            showgrid=True,
            gridcolor="lightgray",
            scaleanchor="x",
            scaleratio=1,
        ),
        hovermode="closest",
        showlegend=True,
        plot_bgcolor="white",
        width=700,
        height=700,
    )

    if show:
        fig.show()
        return None
    else:
        return fig


def plot_instrument_sensitivity(
    sensitivity_results: Dict,
    title: str = "Instrument Sensitivity Analysis",
    backend: str = "matplotlib",
    figsize: Tuple[int, int] = (12, 6),
    show: bool = True,
    max_coefs_to_plot: int = 6,
) -> Optional[object]:
    """
    Plot coefficient stability across different instrument counts.

    Visualizes how coefficients change as the number of instruments varies.
    Stable coefficients (small changes) indicate valid instruments.
    Large changes suggest instrument proliferation or weak instruments.

    Parameters
    ----------
    sensitivity_results : dict
        Results from instrument_sensitivity_analysis()
    title : str, default="Instrument Sensitivity Analysis"
        Plot title
    backend : str, default="matplotlib"
        Plotting backend: "matplotlib" or "plotly"
    figsize : tuple, default=(12, 6)
        Figure size for matplotlib (width, height in inches)
    show : bool, default=True
        Whether to display the plot immediately
    max_coefs_to_plot : int, default=6
        Maximum number of coefficients to plot (to avoid clutter)

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        Figure object (if show=False)

    Notes
    -----
    Following Roodman (2009), coefficients should be stable as instruments increase.
    This plot helps diagnose:
    - Instrument proliferation (large changes)
    - Weak instruments (erratic patterns)
    - Optimal instrument count (where coefficients stabilize)

    Examples
    --------
    >>> from panelbox.var.diagnostics import instrument_sensitivity_analysis
    >>> sensitivity = instrument_sensitivity_analysis(
    ...     model_func=lambda **kw: estimate_panelvar_gmm(data, **kw),
    ...     max_instruments_list=[6, 12, 24, 48]
    ... )
    >>> plot_instrument_sensitivity(sensitivity)
    """
    if backend == "matplotlib":
        return _plot_sensitivity_matplotlib(
            sensitivity_results, title, figsize, show, max_coefs_to_plot
        )
    elif backend == "plotly":
        if not HAS_PLOTLY:
            raise ImportError(
                "Plotly is required for backend='plotly'. Install with: pip install plotly"
            )
        return _plot_sensitivity_plotly(sensitivity_results, title, show, max_coefs_to_plot)
    else:
        raise ValueError(f"backend must be 'matplotlib' or 'plotly', got '{backend}'")


def _plot_sensitivity_matplotlib(
    sensitivity_results: Dict,
    title: str,
    figsize: Tuple[int, int],
    show: bool,
    max_coefs: int,
) -> Optional[plt.Figure]:
    """Plot sensitivity analysis using matplotlib."""
    fig, ax = plt.subplots(figsize=figsize)

    coefficients = sensitivity_results["coefficients"]
    n_instruments = sensitivity_results["n_instruments_actual"]

    # Select coefficients to plot (avoid clutter)
    coef_names = list(coefficients.keys())
    if len(coef_names) > max_coefs:
        # Select coefficients with largest changes
        changes = sensitivity_results.get("coefficient_changes", {})
        if changes:
            sorted_names = sorted(changes.keys(), key=lambda k: changes[k], reverse=True)
            coef_names = sorted_names[:max_coefs]
        else:
            coef_names = coef_names[:max_coefs]

    # Plot each coefficient
    for coef_name in coef_names:
        values = coefficients[coef_name]
        ax.plot(n_instruments, values, marker="o", linewidth=2, label=coef_name, alpha=0.8)

    # Formatting
    ax.set_xlabel("Number of Instruments", fontsize=12)
    ax.set_ylabel("Coefficient Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="best", fontsize=10, ncol=2)

    # Add interpretation text
    interpretation = sensitivity_results.get("interpretation", "")
    if interpretation:
        fig.text(
            0.5,
            0.02,
            interpretation,
            ha="center",
            fontsize=10,
            style="italic",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

    plt.tight_layout()

    if show:
        plt.show()
        return None
    else:
        return fig


def _plot_sensitivity_plotly(
    sensitivity_results: Dict,
    title: str,
    show: bool,
    max_coefs: int,
) -> Optional[go.Figure]:
    """Plot sensitivity analysis using plotly."""
    coefficients = sensitivity_results["coefficients"]
    n_instruments = sensitivity_results["n_instruments_actual"]

    # Select coefficients to plot
    coef_names = list(coefficients.keys())
    if len(coef_names) > max_coefs:
        changes = sensitivity_results.get("coefficient_changes", {})
        if changes:
            sorted_names = sorted(changes.keys(), key=lambda k: changes[k], reverse=True)
            coef_names = sorted_names[:max_coefs]
        else:
            coef_names = coef_names[:max_coefs]

    # Create figure
    fig = go.Figure()

    # Add traces for each coefficient
    for coef_name in coef_names:
        values = coefficients[coef_name]
        change_pct = sensitivity_results.get("coefficient_changes", {}).get(coef_name, 0.0)

        hover_text = [
            f"{coef_name}<br>Value: {val:.4f}<br>N instr: {n_instr}<br>Change: {change_pct:.2f}%"
            for val, n_instr in zip(values, n_instruments)
        ]

        fig.add_trace(
            go.Scatter(
                x=n_instruments,
                y=values,
                mode="lines+markers",
                name=f"{coef_name} ({change_pct:.1f}%)",
                text=hover_text,
                hoverinfo="text",
                line=dict(width=2),
                marker=dict(size=8),
            )
        )

    # Layout
    interpretation = sensitivity_results.get("interpretation", "")
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Number of Instruments", showgrid=True),
        yaxis=dict(title="Coefficient Value", showgrid=True),
        hovermode="closest",
        showlegend=True,
        plot_bgcolor="white",
        width=1000,
        height=600,
    )

    # Add interpretation annotation
    if interpretation:
        fig.add_annotation(
            text=interpretation,
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.15,
            showarrow=False,
            font=dict(
                size=12, color="darkgreen" if "stable" in interpretation.lower() else "darkred"
            ),
            bgcolor="lightyellow",
            bordercolor="gray",
            borderwidth=1,
        )

    if show:
        fig.show()
        return None
    else:
        return fig


def plot_irf(
    irf_result: "IRFResult",
    impulse: Optional[str] = None,
    response: Optional[str] = None,
    variables: Optional[list] = None,
    ci: bool = True,
    backend: str = "matplotlib",
    figsize: Optional[Tuple[int, int]] = None,
    theme: str = "academic",
    show: bool = True,
) -> Optional[object]:
    """
    Plot Impulse Response Functions in a grid layout.

    Parameters
    ----------
    irf_result : IRFResult
        IRF result object containing IRF estimates
    impulse : str, optional
        If specified, plot only responses to this impulse variable (K subplots)
    response : str, optional
        If specified, plot only how this variable responds (K subplots)
    variables : list of str, optional
        If specified, plot only these variables (subset grid)
    ci : bool, default=True
        Show confidence intervals (if available)
    backend : str, default='matplotlib'
        Plotting backend: 'matplotlib' or 'plotly'
    figsize : tuple, optional
        Figure size (width, height). If None, auto-sized based on grid
    theme : str, default='academic'
        Visual theme: 'academic', 'professional', or 'presentation'
    show : bool, default=True
        Whether to display the plot immediately

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure or None
        Figure object if show=False, otherwise None

    Examples
    --------
    >>> result = model.fit()
    >>> irf = result.irf(periods=20, ci_method='bootstrap')
    >>> # Plot all IRFs
    >>> irf.plot()
    >>> # Plot only responses to GDP shock
    >>> irf.plot(impulse='gdp')
    >>> # Plot only how inflation responds
    >>> irf.plot(response='inflation')
    """
    if backend == "matplotlib":
        return _plot_irf_matplotlib(
            irf_result, impulse, response, variables, ci, figsize, theme, show
        )
    elif backend == "plotly":
        if not HAS_PLOTLY:
            raise ImportError(
                "Plotly is required for backend='plotly'. Install with: pip install plotly"
            )
        return _plot_irf_plotly(irf_result, impulse, response, variables, ci, theme, show)
    else:
        raise ValueError(f"backend must be 'matplotlib' or 'plotly', got '{backend}'")


def _plot_irf_matplotlib(
    irf_result: "IRFResult",
    impulse: Optional[str],
    response: Optional[str],
    variables: Optional[list],
    ci: bool,
    figsize: Optional[Tuple[int, int]],
    theme: str,
    show: bool,
) -> Optional[plt.Figure]:
    """Plot IRFs using matplotlib."""
    # Determine which variables to plot
    if variables is not None:
        var_names = variables
    else:
        var_names = irf_result.var_names

    # Determine grid size
    if impulse is not None and response is not None:
        # Single subplot
        n_rows, n_cols = 1, 1
    elif impulse is not None:
        # Column: responses to one impulse
        n_rows, n_cols = len(var_names), 1
    elif response is not None:
        # Row: one response to all impulses
        n_rows, n_cols = 1, len(var_names)
    else:
        # Full grid
        K = len(var_names)
        n_rows, n_cols = K, K

    # Auto-size figure if not specified
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    # Apply theme
    theme_config = _get_theme_config(theme)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    horizons = np.arange(irf_result.periods + 1)

    # Determine pairs to plot
    if impulse is not None and response is not None:
        pairs = [(response, impulse)]
    elif impulse is not None:
        pairs = [(resp, impulse) for resp in var_names]
    elif response is not None:
        pairs = [(response, imp) for imp in var_names]
    else:
        pairs = [(resp, imp) for resp in var_names for imp in var_names]

    # Plot each pair
    for idx, (resp, imp) in enumerate(pairs):
        if impulse is not None and response is not None:
            ax = axes[0, 0]
        elif impulse is not None:
            ax = axes[idx, 0]
        elif response is not None:
            ax = axes[0, idx]
        else:
            resp_idx = var_names.index(resp)
            imp_idx = var_names.index(imp)
            ax = axes[resp_idx, imp_idx]

        # Get IRF data
        irf_values = irf_result[resp, imp]

        # Plot IRF line
        ax.plot(
            horizons,
            irf_values,
            color=theme_config["line_color"],
            linewidth=theme_config["linewidth"],
            label="IRF",
        )

        # Plot confidence intervals
        if ci and irf_result.ci_lower is not None:
            resp_idx_full = irf_result.var_names.index(resp)
            imp_idx_full = irf_result.var_names.index(imp)
            ci_lower = irf_result.ci_lower[:, resp_idx_full, imp_idx_full]
            ci_upper = irf_result.ci_upper[:, resp_idx_full, imp_idx_full]

            ax.fill_between(
                horizons,
                ci_lower,
                ci_upper,
                alpha=0.3,
                color=theme_config["ci_color"],
                label=f"{int(irf_result.ci_level*100)}% CI",
            )

        # Zero line
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # Title and labels
        ax.set_title(f"{resp} ← {imp}", fontsize=theme_config["fontsize"])
        ax.set_xlabel("Horizon", fontsize=theme_config["fontsize"] - 2)
        ax.set_ylabel("Response", fontsize=theme_config["fontsize"] - 2)

        if theme_config["grid"]:
            ax.grid(True, alpha=0.3)

        # Legend only on first subplot
        if idx == 0:
            ax.legend(fontsize=theme_config["fontsize"] - 2)

    plt.tight_layout()

    if show:
        plt.show()
        return None
    else:
        return fig


def _plot_irf_plotly(
    irf_result: "IRFResult",
    impulse: Optional[str],
    response: Optional[str],
    variables: Optional[list],
    ci: bool,
    theme: str,
    show: bool,
) -> Optional[go.Figure]:
    """Plot IRFs using Plotly."""
    try:
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("Plotly is required. Install with: pip install plotly")

    # Determine which variables to plot
    if variables is not None:
        var_names = variables
    else:
        var_names = irf_result.var_names

    # Determine grid size
    if impulse is not None and response is not None:
        # Single subplot
        n_rows, n_cols = 1, 1
        pairs = [(response, impulse)]
    elif impulse is not None:
        # Column: responses to one impulse
        n_rows, n_cols = len(var_names), 1
        pairs = [(resp, impulse) for resp in var_names]
    elif response is not None:
        # Row: one response to all impulses
        n_rows, n_cols = 1, len(var_names)
        pairs = [(response, imp) for imp in var_names]
    else:
        # Full grid
        K = len(var_names)
        n_rows, n_cols = K, K
        pairs = [(resp, imp) for resp in var_names for imp in var_names]

    # Create subplots
    subplot_titles = [f"{resp} ← {imp}" for resp, imp in pairs]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    horizons = np.arange(irf_result.periods + 1)
    theme_config = _get_theme_config(theme)

    # Plot each pair
    for idx, (resp, imp) in enumerate(pairs):
        if impulse is not None and response is not None:
            row, col = 1, 1
        elif impulse is not None:
            row, col = idx + 1, 1
        elif response is not None:
            row, col = 1, idx + 1
        else:
            resp_idx = var_names.index(resp)
            imp_idx = var_names.index(imp)
            row, col = resp_idx + 1, imp_idx + 1

        # Get IRF data
        irf_values = irf_result[resp, imp]

        # Plot IRF line
        fig.add_trace(
            go.Scatter(
                x=horizons,
                y=irf_values,
                mode="lines",
                name="IRF",
                line=dict(color=theme_config["line_color"], width=theme_config["linewidth"]),
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )

        # Plot confidence intervals
        if ci and irf_result.ci_lower is not None:
            resp_idx_full = irf_result.var_names.index(resp)
            imp_idx_full = irf_result.var_names.index(imp)
            ci_lower = irf_result.ci_lower[:, resp_idx_full, imp_idx_full]
            ci_upper = irf_result.ci_upper[:, resp_idx_full, imp_idx_full]

            fig.add_trace(
                go.Scatter(
                    x=horizons,
                    y=ci_upper,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=horizons,
                    y=ci_lower,
                    mode="lines",
                    line=dict(width=0),
                    fillcolor=theme_config["ci_color"],
                    fill="tonexty",
                    name=f"{int(irf_result.ci_level*100)}% CI",
                    showlegend=(idx == 0),
                ),
                row=row,
                col=col,
            )

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)

        # Update axes
        fig.update_xaxes(title_text="Horizon", row=row, col=col)
        fig.update_yaxes(title_text="Response", row=row, col=col)

    # Update layout
    fig.update_layout(
        height=300 * n_rows,
        width=400 * n_cols,
        showlegend=True,
        title_text=f"Impulse Response Functions ({irf_result.method.upper()})",
    )

    if show:
        fig.show()
        return None
    else:
        return fig


def _get_theme_config(theme: str) -> Dict:
    """Get theme configuration for plots."""
    themes = {
        "academic": {
            "fontsize": 10,
            "linewidth": 2,
            "line_color": "black",
            "ci_color": "blue",
            "grid": True,
        },
        "professional": {
            "fontsize": 11,
            "linewidth": 2.5,
            "line_color": "#2E86AB",
            "ci_color": "#A23B72",
            "grid": False,
        },
        "presentation": {
            "fontsize": 14,
            "linewidth": 3,
            "line_color": "#1f77b4",
            "ci_color": "#ff7f0e",
            "grid": True,
        },
    }

    if theme not in themes:
        raise ValueError(
            f"Unknown theme '{theme}'. Use 'academic', 'professional', or 'presentation'."
        )

    return themes[theme]


def plot_fevd(
    fevd_result: "FEVDResult",
    kind: str = "area",
    variables: Optional[list] = None,
    horizons: Optional[list] = None,
    backend: str = "matplotlib",
    figsize: Optional[Tuple[int, int]] = None,
    theme: str = "academic",
    show: bool = True,
) -> Optional[object]:
    """
    Plot Forecast Error Variance Decomposition.

    Parameters
    ----------
    fevd_result : FEVDResult
        FEVD result object
    kind : str, default='area'
        Type of plot: 'area' (stacked area) or 'bar' (stacked bar)
    variables : list of str, optional
        If specified, plot only these variables
    horizons : list of int, optional
        For 'bar' plot: horizons to display. If None, uses [1, 5, 10, periods]
    backend : str, default='matplotlib'
        Plotting backend: 'matplotlib' or 'plotly'
    figsize : tuple, optional
        Figure size (width, height)
    theme : str, default='academic'
        Visual theme
    show : bool, default=True
        Whether to display the plot immediately

    Returns
    -------
    fig : Figure or None
        Figure object if show=False, otherwise None

    Examples
    --------
    >>> fevd = result.fevd(periods=20)
    >>> fevd.plot()
    >>> fevd.plot(kind='bar', horizons=[1, 5, 10, 20])
    """
    if backend == "matplotlib":
        return _plot_fevd_matplotlib(fevd_result, kind, variables, horizons, figsize, theme, show)
    elif backend == "plotly":
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required. Install with: pip install plotly")
        return _plot_fevd_plotly(fevd_result, kind, variables, horizons, theme, show)
    else:
        raise ValueError(f"backend must be 'matplotlib' or 'plotly', got '{backend}'")


def _plot_fevd_matplotlib(
    fevd_result: "FEVDResult",
    kind: str,
    variables: Optional[list],
    horizons: Optional[list],
    figsize: Optional[Tuple[int, int]],
    theme: str,
    show: bool,
) -> Optional[plt.Figure]:
    """Plot FEVD using matplotlib."""
    # Determine which variables to plot
    if variables is not None:
        var_names = variables
    else:
        var_names = fevd_result.var_names

    K_plot = len(var_names)

    # Auto-size figure
    if figsize is None:
        if kind == "area":
            figsize = (10, 3 * K_plot)
        else:
            figsize = (12, 3 * K_plot)

    theme_config = _get_theme_config(theme)

    if kind == "area":
        # Stacked area chart
        fig, axes = plt.subplots(K_plot, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        horizons_plot = np.arange(fevd_result.periods + 1)

        # Color scheme
        colors = plt.cm.tab10(np.linspace(0, 1, fevd_result.K))

        for idx, var_name in enumerate(var_names):
            ax = axes[idx]
            var_idx = fevd_result.var_names.index(var_name)

            # Get FEVD data for this variable
            data = fevd_result.decomposition[:, var_idx, :]  # (periods+1, K)

            # Stack plot
            ax.stackplot(
                horizons_plot,
                *[data[:, j] * 100 for j in range(fevd_result.K)],
                labels=fevd_result.var_names,
                colors=colors,
                alpha=0.8,
            )

            ax.set_title(f"FEVD of {var_name}", fontsize=theme_config["fontsize"])
            ax.set_xlabel("Horizon", fontsize=theme_config["fontsize"] - 2)
            ax.set_ylabel("Variance Share (%)", fontsize=theme_config["fontsize"] - 2)
            ax.set_ylim([0, 100])

            if theme_config["grid"]:
                ax.grid(True, alpha=0.3, axis="y")

            ax.legend(
                loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=theme_config["fontsize"] - 2
            )

        plt.tight_layout()

    elif kind == "bar":
        # Stacked bar chart
        if horizons is None:
            horizons = [1, 5, 10, fevd_result.periods]
            horizons = [h for h in horizons if h <= fevd_result.periods]

        fig, axes = plt.subplots(K_plot, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Color scheme
        colors = plt.cm.tab10(np.linspace(0, 1, fevd_result.K))

        bar_width = 0.6
        x_pos = np.arange(len(horizons))

        for idx, var_name in enumerate(var_names):
            ax = axes[idx]
            var_idx = fevd_result.var_names.index(var_name)

            # Get FEVD data for selected horizons
            data = fevd_result.decomposition[horizons, var_idx, :]  # (len(horizons), K)

            # Stacked bars
            bottom = np.zeros(len(horizons))
            for j in range(fevd_result.K):
                ax.bar(
                    x_pos,
                    data[:, j] * 100,
                    bar_width,
                    bottom=bottom,
                    label=fevd_result.var_names[j],
                    color=colors[j],
                    alpha=0.8,
                )
                bottom += data[:, j] * 100

            ax.set_title(f"FEVD of {var_name}", fontsize=theme_config["fontsize"])
            ax.set_xlabel("Horizon", fontsize=theme_config["fontsize"] - 2)
            ax.set_ylabel("Variance Share (%)", fontsize=theme_config["fontsize"] - 2)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(horizons)
            ax.set_ylim([0, 100])

            if theme_config["grid"]:
                ax.grid(True, alpha=0.3, axis="y")

            ax.legend(
                loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=theme_config["fontsize"] - 2
            )

        plt.tight_layout()

    else:
        raise ValueError(f"kind must be 'area' or 'bar', got '{kind}'")

    if show:
        plt.show()
        return None
    else:
        return fig


def _plot_fevd_plotly(
    fevd_result: "FEVDResult",
    kind: str,
    variables: Optional[list],
    horizons: Optional[list],
    theme: str,
    show: bool,
) -> Optional[go.Figure]:
    """Plot FEVD using Plotly."""
    try:
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("Plotly is required. Install with: pip install plotly")

    # Determine which variables to plot
    if variables is not None:
        var_names = variables
    else:
        var_names = fevd_result.var_names

    K_plot = len(var_names)

    # Create subplots
    fig = make_subplots(
        rows=K_plot,
        cols=1,
        subplot_titles=[f"FEVD of {var}" for var in var_names],
        vertical_spacing=0.1,
    )

    # Color scheme
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    if kind == "area":
        horizons_plot = np.arange(fevd_result.periods + 1)

        for idx, var_name in enumerate(var_names):
            var_idx = fevd_result.var_names.index(var_name)
            data = fevd_result.decomposition[:, var_idx, :]  # (periods+1, K)

            # Add traces for each shock (in reverse order for proper stacking)
            for j in reversed(range(fevd_result.K)):
                fig.add_trace(
                    go.Scatter(
                        x=horizons_plot,
                        y=data[:, j] * 100,
                        mode="lines",
                        name=fevd_result.var_names[j],
                        stackgroup="one",
                        line=dict(width=0.5, color=colors[j % len(colors)]),
                        fillcolor=colors[j % len(colors)],
                        showlegend=(idx == 0),
                    ),
                    row=idx + 1,
                    col=1,
                )

            # Update axes
            fig.update_xaxes(title_text="Horizon", row=idx + 1, col=1)
            fig.update_yaxes(title_text="Variance Share (%)", range=[0, 100], row=idx + 1, col=1)

    elif kind == "bar":
        if horizons is None:
            horizons = [1, 5, 10, fevd_result.periods]
            horizons = [h for h in horizons if h <= fevd_result.periods]

        for idx, var_name in enumerate(var_names):
            var_idx = fevd_result.var_names.index(var_name)
            data = fevd_result.decomposition[horizons, var_idx, :]  # (len(horizons), K)

            # Add traces for each shock
            for j in range(fevd_result.K):
                fig.add_trace(
                    go.Bar(
                        x=horizons,
                        y=data[:, j] * 100,
                        name=fevd_result.var_names[j],
                        marker=dict(color=colors[j % len(colors)]),
                        showlegend=(idx == 0),
                    ),
                    row=idx + 1,
                    col=1,
                )

            # Update axes
            fig.update_xaxes(title_text="Horizon", row=idx + 1, col=1)
            fig.update_yaxes(title_text="Variance Share (%)", range=[0, 100], row=idx + 1, col=1)

        # Update layout for stacked bars
        fig.update_layout(barmode="stack")

    else:
        raise ValueError(f"kind must be 'area' or 'bar', got '{kind}'")

    # Update layout
    fig.update_layout(
        height=400 * K_plot,
        showlegend=True,
        title_text=f"Forecast Error Variance Decomposition ({fevd_result.method.upper()})",
    )

    if show:
        fig.show()
        return None
    else:
        return fig
