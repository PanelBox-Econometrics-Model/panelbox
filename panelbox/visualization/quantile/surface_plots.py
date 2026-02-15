"""
3D and surface plot visualizations for quantile regression.

This module provides 3D surface plots and contour visualizations for
exploring how quantile regression coefficients vary across quantiles
and covariate space.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, interp1d


class SurfacePlotter:
    """
    3D surface and contour plotting for quantile regression.

    Creates interactive and static 3D visualizations showing how
    predictions or coefficients vary across quantile and covariate space.

    Parameters
    ----------
    figsize : tuple
        Default figure size for matplotlib plots
    colormap : str
        Default colormap for surface plots

    Examples
    --------
    >>> plotter = SurfacePlotter()
    >>> fig = plotter.plot_surface(result, ['education', 'experience'])
    >>> fig_interactive = plotter.plot_interactive(result)
    """

    def __init__(self, figsize: Tuple[float, float] = (12, 8), colormap: str = "viridis"):
        """Initialize surface plotter with default settings."""
        self.figsize = figsize
        self.colormap = colormap

    def plot_surface(
        self,
        result,
        var_names: List[str],
        tau_grid: Optional[np.ndarray] = None,
        X_grid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        projection: str = "3d",
        figsize: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        """
        Create 3D surface plot showing β(τ, X).

        Shows how coefficients or predictions vary across quantile and
        covariate space.

        Parameters
        ----------
        result : QuantilePanelResult
            Fitted result
        var_names : list
            Two variables for X and Y axes
        tau_grid : array, optional
            Quantile grid for surface
        X_grid : tuple, optional
            (X1_range, X2_range) for surface evaluation
        projection : str
            '3d' for 3D surface or 'contour' for 2D contours
        figsize : tuple, optional
            Figure size

        Returns
        -------
        fig : matplotlib.Figure
            The figure object
        """
        if tau_grid is None:
            tau_grid = np.linspace(0.1, 0.9, 20)

        if len(var_names) != 2:
            raise ValueError("Exactly 2 variables needed for surface plot")

        # Get variable indices (assuming var_names are indices or we use first two)
        if isinstance(var_names[0], int):
            idx1, idx2 = var_names[0], var_names[1]
        else:
            # For now use first two variables
            idx1, idx2 = 0, 1

        # Create meshgrid for X values
        if X_grid is None:
            # Generate default ranges
            X1_range = np.linspace(-2, 2, 25)
            X2_range = np.linspace(-2, 2, 25)
        else:
            X1_range, X2_range = X_grid

        X1_mesh, X2_mesh = np.meshgrid(X1_range, X2_range)

        # Compute predictions for each (tau, X1, X2)
        Z = np.zeros((len(tau_grid), X1_mesh.shape[0], X1_mesh.shape[1]))

        # Get number of parameters from first result
        first_tau = list(result.results.keys())[0]
        first_result = result.results[first_tau]
        if hasattr(first_result, "params"):
            n_params = len(first_result.params)
        else:
            n_params = len(first_result) if hasattr(first_result, "__len__") else 1

        for i, tau in enumerate(tau_grid):
            if tau in result.results:
                res_tau = result.results[tau]
            else:
                # Interpolate coefficients for this tau
                tau_available = sorted(result.results.keys())
                coef_available = []
                for t in tau_available:
                    if hasattr(result.results[t], "params"):
                        coef_available.append(result.results[t].params)
                    else:
                        coef_available.append(result.results[t])

                coef_available = np.array(coef_available)

                # Interpolate each coefficient
                coef_interp = []
                for j in range(n_params):
                    f = interp1d(
                        tau_available,
                        coef_available[:, j],
                        kind="linear",
                        fill_value="extrapolate",
                        bounds_error=False,
                    )
                    coef_interp.append(f(tau))

                class InterpolatedResult:
                    def __init__(self, params):
                        self.params = params

                res_tau = InterpolatedResult(np.array(coef_interp))

            # Create prediction matrix
            X_pred = np.zeros((X1_mesh.size, n_params))
            if n_params > 0:
                X_pred[:, 0] = 1  # Intercept
            if n_params > idx1:
                X_pred[:, idx1] = X1_mesh.flatten()
            if n_params > idx2:
                X_pred[:, idx2] = X2_mesh.flatten()

            # Predict
            if hasattr(res_tau, "params"):
                predictions = X_pred @ res_tau.params
            else:
                predictions = X_pred @ res_tau

            Z[i] = predictions.reshape(X1_mesh.shape)

        # Create plot based on projection type
        figsize = figsize or self.figsize

        if projection == "3d":
            fig = self._plot_3d_surface(X1_mesh, X2_mesh, Z, tau_grid, var_names, figsize)

        elif projection == "contour":
            fig = self._plot_contours(X1_mesh, X2_mesh, Z, tau_grid, var_names, figsize)

        else:
            raise ValueError(f"Unknown projection: {projection}")

        return fig

    def _plot_3d_surface(
        self,
        X1_mesh: np.ndarray,
        X2_mesh: np.ndarray,
        Z: np.ndarray,
        tau_grid: np.ndarray,
        var_names: List[str],
        figsize: Tuple[float, float],
    ) -> plt.Figure:
        """Create matplotlib 3D surface plot."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Create surface for median
        median_idx = np.argmin(np.abs(tau_grid - 0.5))

        # Plot main surface
        surf = ax.plot_surface(
            X1_mesh,
            X2_mesh,
            Z[median_idx],
            cmap=self.colormap,
            alpha=0.8,
            linewidth=0,
            antialiased=True,
            label="τ=0.5",
        )

        # Add wireframes for other quantiles
        quantile_indices = [0, len(tau_grid) // 4, 3 * len(tau_grid) // 4, -1]
        for i in quantile_indices:
            if i != median_idx and 0 <= i < len(tau_grid):
                ax.plot_wireframe(
                    X1_mesh,
                    X2_mesh,
                    Z[i],
                    color="gray",
                    alpha=0.3,
                    linewidth=0.5,
                    label=f"τ={tau_grid[i]:.2f}",
                )

        # Labels and title
        ax.set_xlabel(var_names[0], fontweight="bold", labelpad=10)
        ax.set_ylabel(var_names[1], fontweight="bold", labelpad=10)
        ax.set_zlabel("Predicted Value", fontweight="bold", labelpad=10)
        ax.set_title("Quantile Regression Surface", fontweight="bold", pad=20)

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)

        # Improve viewing angle
        ax.view_init(elev=20, azim=45)

        # Add legend
        ax.legend(loc="upper left")

        return fig

    def _plot_contours(
        self,
        X1_mesh: np.ndarray,
        X2_mesh: np.ndarray,
        Z: np.ndarray,
        tau_grid: np.ndarray,
        var_names: List[str],
        figsize: Tuple[float, float],
    ) -> plt.Figure:
        """Create contour plots for selected quantiles."""
        # Select representative quantiles
        tau_select = [0.1, 0.25, 0.5, 0.75]
        tau_indices = []
        for tau in tau_select:
            idx = np.argmin(np.abs(tau_grid - tau))
            tau_indices.append(idx)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        for idx, (tau_idx, tau) in enumerate(zip(tau_indices, tau_select)):
            ax = axes[idx]

            # Create filled contour plot
            contour_filled = ax.contourf(
                X1_mesh, X2_mesh, Z[tau_idx], levels=15, cmap=self.colormap, alpha=0.8
            )

            # Add contour lines
            contour_lines = ax.contour(
                X1_mesh, X2_mesh, Z[tau_idx], levels=15, colors="black", alpha=0.3, linewidths=0.5
            )

            # Add contour labels
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.1f")

            # Labels and title
            ax.set_xlabel(var_names[0])
            ax.set_ylabel(var_names[1])
            ax.set_title(f"τ = {tau:.2f}", fontweight="bold")

            # Add colorbar
            plt.colorbar(contour_filled, ax=ax, fraction=0.046, pad=0.04)

            # Add grid
            ax.grid(True, alpha=0.3, linestyle="--")

        # Overall title
        fig.suptitle(
            "Quantile Surfaces at Different Levels", fontweight="bold", fontsize=14, y=1.02
        )

        plt.tight_layout()
        return fig

    def plot_interactive(
        self,
        result,
        var_names: Optional[List[str]] = None,
        tau_list: Optional[List[float]] = None,
        X_grid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> go.Figure:
        """
        Create interactive 3D surface plot using Plotly.

        Parameters
        ----------
        result : QuantilePanelResult
            Fitted result
        var_names : list, optional
            Variable names for axes
        tau_list : list, optional
            Quantiles to include in plot
        X_grid : tuple, optional
            Grid for X values

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive plotly figure
        """
        if tau_list is None:
            tau_list = [0.1, 0.25, 0.5, 0.75, 0.9]

        if var_names is None:
            var_names = ["X1", "X2"]

        # Create grid for X values
        if X_grid is None:
            X1_range = np.linspace(-2, 2, 30)
            X2_range = np.linspace(-2, 2, 30)
        else:
            X1_range, X2_range = X_grid

        X1_mesh, X2_mesh = np.meshgrid(X1_range, X2_range)

        # Get number of parameters
        first_tau = list(result.results.keys())[0]
        first_result = result.results[first_tau]
        if hasattr(first_result, "params"):
            n_params = len(first_result.params)
        else:
            n_params = len(first_result) if hasattr(first_result, "__len__") else 1

        # Create figure
        fig = go.Figure()

        # Add surfaces for each quantile
        for tau in tau_list:
            if tau in result.results:
                res_tau = result.results[tau]
            else:
                # Interpolate
                tau_available = sorted(result.results.keys())
                coef_available = []
                for t in tau_available:
                    if hasattr(result.results[t], "params"):
                        coef_available.append(result.results[t].params)
                    else:
                        coef_available.append(result.results[t])

                coef_available = np.array(coef_available)

                # Interpolate each coefficient
                coef_interp = []
                for j in range(n_params):
                    f = interp1d(
                        tau_available,
                        coef_available[:, j],
                        kind="linear",
                        fill_value="extrapolate",
                        bounds_error=False,
                    )
                    coef_interp.append(f(tau))

                class InterpolatedResult:
                    def __init__(self, params):
                        self.params = params

                res_tau = InterpolatedResult(np.array(coef_interp))

            # Create prediction matrix
            X_pred = np.zeros((X1_mesh.size, n_params))
            if n_params > 0:
                X_pred[:, 0] = 1  # Intercept
            if n_params > 1:
                X_pred[:, 1] = X1_mesh.flatten()
            if n_params > 2:
                X_pred[:, 2] = X2_mesh.flatten()

            # Predict
            if hasattr(res_tau, "params"):
                predictions = X_pred @ res_tau.params
            else:
                predictions = X_pred @ res_tau

            Z = predictions.reshape(X1_mesh.shape)

            # Add surface trace
            fig.add_trace(
                go.Surface(
                    x=X1_range,
                    y=X2_range,
                    z=Z,
                    name=f"τ={tau:.2f}",
                    opacity=0.7,
                    colorscale="Viridis",
                    showscale=(tau == tau_list[0]),  # Only show scale for first surface
                    hovertemplate=f"τ={tau:.2f}<br>"
                    + var_names[0]
                    + ": %{x:.2f}<br>"
                    + var_names[1]
                    + ": %{y:.2f}<br>"
                    + "Value: %{z:.2f}<extra></extra>",
                )
            )

        # Update layout
        fig.update_layout(
            title={
                "text": "Interactive Quantile Regression Surfaces",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 16, "family": "Arial Black"},
            },
            scene=dict(
                xaxis=dict(
                    title=var_names[0],
                    titlefont=dict(size=12),
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="rgba(230, 230, 230, 0.5)",
                ),
                yaxis=dict(
                    title=var_names[1],
                    titlefont=dict(size=12),
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="rgba(230, 230, 230, 0.5)",
                ),
                zaxis=dict(
                    title="Predicted Value",
                    titlefont=dict(size=12),
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="rgba(230, 230, 230, 0.5)",
                ),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            width=900,
            height=700,
            margin=dict(l=65, r=50, b=65, t=90),
        )

        # Add dropdown menu for different viewing angles
        camera_views = {
            "Default": dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            "Top View": dict(eye=dict(x=0, y=0, z=2.5)),
            "Side View": dict(eye=dict(x=2.5, y=0, z=0)),
            "Front View": dict(eye=dict(x=0, y=2.5, z=0)),
        }

        updatemenus = [
            dict(
                buttons=[
                    dict(args=[{"scene.camera": camera}], label=name, method="relayout")
                    for name, camera in camera_views.items()
                ],
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
            )
        ]

        fig.update_layout(updatemenus=updatemenus)

        return fig

    def coefficient_heatmap(
        self,
        result,
        var_names: Optional[List[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        """
        Create heatmap showing coefficient values across quantiles.

        Parameters
        ----------
        result : QuantilePanelResult
            Fitted result
        var_names : list, optional
            Variable names
        figsize : tuple, optional
            Figure size

        Returns
        -------
        fig : matplotlib.Figure
            The figure object
        """
        import seaborn as sns

        # Extract coefficients
        tau_list = sorted(result.results.keys())

        # Get coefficients matrix
        first_result = result.results[tau_list[0]]
        if hasattr(first_result, "params"):
            n_vars = len(first_result.params)
        else:
            n_vars = len(first_result) if hasattr(first_result, "__len__") else 1

        if var_names is None:
            var_names = [f"X{i}" for i in range(n_vars)]

        coef_matrix = np.zeros((len(tau_list), n_vars))

        for i, tau in enumerate(tau_list):
            res = result.results[tau]
            if hasattr(res, "params"):
                coef_matrix[i, :] = res.params
            else:
                if hasattr(res, "__len__"):
                    coef_matrix[i, :] = res
                else:
                    coef_matrix[i, 0] = res

        # Create heatmap
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap with diverging colormap
        sns.heatmap(
            coef_matrix.T,
            xticklabels=[f"{tau:.2f}" for tau in tau_list],
            yticklabels=var_names,
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "Coefficient Value"},
            ax=ax,
        )

        # Labels and title
        ax.set_xlabel("Quantile (τ)", fontweight="bold")
        ax.set_ylabel("Variable", fontweight="bold")
        ax.set_title("Coefficient Heatmap Across Quantiles", fontweight="bold", fontsize=14)

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        return fig
