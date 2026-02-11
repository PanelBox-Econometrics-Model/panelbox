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
