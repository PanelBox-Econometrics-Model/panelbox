"""
Frontier estimation plots for SFA results.

This module provides visualization functions for:
- 2D frontier plots (1 input)
- 3D frontier surfaces (2 inputs)
- Contour plots
- Partial frontier plots (fixing other inputs)
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def _get_X_df(result):
    """Get X dataframe from result, creating if necessary."""
    if hasattr(result.model, "X_df"):
        return result.model.X_df
    else:
        return result.model.data[result.model.exog]


def plot_frontier_2d(
    result,
    input_var: str,
    backend: str = "plotly",
    show_distance: bool = False,
    n_observations: Optional[int] = None,
    title: Optional[str] = None,
    **kwargs,
) -> Any:
    """Plot 2D frontier with one input variable.

    Parameters:
        result: SFResult object with fitted model
        input_var: Name of input variable to plot on x-axis
        backend: 'plotly' for interactive or 'matplotlib' for static
        show_distance: Show vertical lines from observations to frontier
        n_observations: Sample size for clarity (if None, plot all)
        title: Custom plot title
        **kwargs: Additional arguments passed to plotting backend

    Returns:
        Plotly Figure or Matplotlib Figure object

    Example:
        >>> result = sf.fit()
        >>> fig = plot_frontier_2d(
        ...     result,
        ...     input_var='log_labor',
        ...     show_distance=True,
        ...     n_observations=100
        ... )
        >>> fig.show()
    """
    # Get data
    X_df = _get_X_df(result)
    y = result.model.y

    if input_var not in X_df.columns:
        raise ValueError(f"Input variable '{input_var}' not found in model data")

    # Get frontier parameters (exclude variance parameters)
    param_names = result.params.index.tolist()
    beta_names = [
        name for name in param_names if "sigma" not in name.lower() and "ln_" not in name.lower()
    ]
    beta = result.params[beta_names].values

    # Sample observations if requested
    if n_observations is not None and len(y) > n_observations:
        sample_idx = np.random.choice(len(y), n_observations, replace=False)
        X_sample = X_df.iloc[sample_idx]
        y_sample = y[sample_idx]
    else:
        X_sample = X_df
        y_sample = y

    # Get x values (input variable)
    x_values = X_sample[input_var].values

    # Compute fitted frontier (deterministic part: X'β)
    X_matrix = result.model.X[sample_idx] if n_observations is not None else result.model.X
    y_frontier = X_matrix @ beta

    # Compute efficiency to color points
    eff_df = result.efficiency(estimator="bc")
    if n_observations is not None:
        eff_sample = eff_df.iloc[sample_idx]["efficiency"].values
    else:
        eff_sample = eff_df["efficiency"].values

    # Sort for smooth frontier line
    sort_idx = np.argsort(x_values)
    x_sorted = x_values[sort_idx]
    y_frontier_sorted = y_frontier[sort_idx]
    y_obs_sorted = y_sample[sort_idx]
    eff_sorted = eff_sample[sort_idx]

    if title is None:
        frontier_type = "Production" if result.model.frontier_type.value == "production" else "Cost"
        title = f"{frontier_type} Frontier: {input_var}"

    if backend == "plotly":
        import plotly.express as px
        import plotly.graph_objects as go

        fig = go.Figure()

        # Frontier line
        fig.add_trace(
            go.Scatter(
                x=x_sorted,
                y=y_frontier_sorted,
                mode="lines",
                line=dict(color="red", width=3),
                name="Frontier",
                hovertemplate="<b>Frontier</b><br>%{x:.4f}<br>Output: %{y:.4f}<extra></extra>",
            )
        )

        # Observations (colored by efficiency)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_sample,
                mode="markers",
                marker=dict(
                    size=8,
                    color=eff_sample,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Efficiency"),
                    line=dict(color="black", width=1),
                ),
                name="Observations",
                hovertemplate="<b>Observation</b><br>%{x:.4f}<br>Output: %{y:.4f}<br>Efficiency: %{marker.color:.4f}<extra></extra>",
            )
        )

        # Distance lines
        if show_distance:
            for i in range(len(x_values)):
                fig.add_trace(
                    go.Scatter(
                        x=[x_values[i], x_values[i]],
                        y=[y_sample[i], y_frontier[i]],
                        mode="lines",
                        line=dict(color="gray", width=1, dash="dot"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        fig.update_layout(
            title=title,
            xaxis_title=input_var,
            yaxis_title="Output",
            template="plotly_white",
            hovermode="closest",
            **kwargs,
        )

        return fig

    elif backend == "matplotlib":
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))

        # Color mapping for efficiency
        norm = Normalize(vmin=eff_sample.min(), vmax=eff_sample.max())
        cmap = cm.get_cmap("RdYlGn")
        colors = [cmap(norm(e)) for e in eff_sample]

        # Observations
        scatter = ax.scatter(
            x_values,
            y_sample,
            c=eff_sample,
            cmap="RdYlGn",
            s=80,
            edgecolors="black",
            linewidths=1,
            alpha=0.7,
            label="Observations",
        )

        # Frontier line
        ax.plot(x_sorted, y_frontier_sorted, "r-", linewidth=3, label="Frontier")

        # Distance lines
        if show_distance:
            for i in range(len(x_values)):
                ax.plot(
                    [x_values[i], x_values[i]],
                    [y_sample[i], y_frontier[i]],
                    "gray",
                    linestyle=":",
                    linewidth=1,
                )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Efficiency", rotation=270, labelpad=20)

        ax.set_xlabel(input_var, fontsize=12)
        ax.set_ylabel("Output", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'plotly' or 'matplotlib'.")


def plot_frontier_3d(
    result,
    input_vars: List[str],
    backend: str = "plotly",
    n_grid: int = 30,
    title: Optional[str] = None,
    **kwargs,
) -> Any:
    """Plot 3D frontier surface with two input variables.

    Parameters:
        result: SFResult object with fitted model
        input_vars: List of two input variable names for x and y axes
        backend: 'plotly' for interactive or 'matplotlib' for static
        n_grid: Number of grid points for surface (default: 30)
        title: Custom plot title
        **kwargs: Additional arguments passed to plotting backend

    Returns:
        Plotly Figure or Matplotlib Figure object

    Example:
        >>> result = sf.fit()
        >>> fig = plot_frontier_3d(
        ...     result,
        ...     input_vars=['log_labor', 'log_capital'],
        ...     n_grid=30
        ... )
        >>> fig.show()
    """
    if len(input_vars) != 2:
        raise ValueError("Must provide exactly 2 input variables for 3D plot")

    X_df = _get_X_df(result)
    y = result.model.y

    for var in input_vars:
        if var not in X_df.columns:
            raise ValueError(f"Input variable '{var}' not found in model data")

    # Get frontier parameters
    param_names = result.params.index.tolist()
    beta_names = [
        name for name in param_names if "sigma" not in name.lower() and "ln_" not in name.lower()
    ]
    beta = result.params[beta_names].values

    # Create grid for surface
    x1_range = np.linspace(X_df[input_vars[0]].min(), X_df[input_vars[0]].max(), n_grid)
    x2_range = np.linspace(X_df[input_vars[1]].min(), X_df[input_vars[1]].max(), n_grid)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

    # Build design matrix for grid
    # Assume intercept + 2 inputs (may need to handle more complex cases)
    n_points = n_grid * n_grid
    X_grid = np.ones((n_points, len(beta)))

    # Find indices of input variables in design matrix
    var1_idx = X_df.columns.get_loc(input_vars[0])
    var2_idx = X_df.columns.get_loc(input_vars[1])

    X_grid[:, var1_idx + 1] = X1_grid.flatten()  # +1 for intercept
    X_grid[:, var2_idx + 1] = X2_grid.flatten()

    # Compute frontier surface
    Y_frontier = X_grid @ beta
    Y_grid = Y_frontier.reshape(n_grid, n_grid)

    # Get observations
    x1_obs = X_df[input_vars[0]].values
    x2_obs = X_df[input_vars[1]].values
    y_obs = y

    # Compute efficiency for coloring
    eff_df = result.efficiency(estimator="bc")
    eff_values = eff_df["efficiency"].values

    if title is None:
        frontier_type = "Production" if result.model.frontier_type.value == "production" else "Cost"
        title = f"{frontier_type} Frontier Surface"

    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure()

        # Surface
        fig.add_trace(
            go.Surface(
                x=x1_range,
                y=x2_range,
                z=Y_grid,
                colorscale="Reds",
                opacity=0.7,
                name="Frontier",
                showscale=False,
            )
        )

        # Observations
        fig.add_trace(
            go.Scatter3d(
                x=x1_obs,
                y=x2_obs,
                z=y_obs,
                mode="markers",
                marker=dict(
                    size=5,
                    color=eff_values,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Efficiency"),
                    line=dict(color="black", width=0.5),
                ),
                name="Observations",
                hovertemplate=f"<b>Observation</b><br>{input_vars[0]}: %{{x:.4f}}<br>{input_vars[1]}: %{{y:.4f}}<br>Output: %{{z:.4f}}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            scene=dict(xaxis_title=input_vars[0], yaxis_title=input_vars[1], zaxis_title="Output"),
            **kwargs,
        )

        return fig

    elif backend == "matplotlib":
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=kwargs.get("figsize", (12, 8)))
        ax = fig.add_subplot(111, projection="3d")

        # Surface
        ax.plot_surface(X1_grid, X2_grid, Y_grid, cmap="Reds", alpha=0.6, edgecolor="none")

        # Observations
        scatter = ax.scatter(
            x1_obs,
            x2_obs,
            y_obs,
            c=eff_values,
            cmap="RdYlGn",
            s=50,
            edgecolors="black",
            linewidths=0.5,
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label("Efficiency", rotation=270, labelpad=20)

        ax.set_xlabel(input_vars[0], fontsize=12)
        ax.set_ylabel(input_vars[1], fontsize=12)
        ax.set_zlabel("Output", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()
        return fig

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'plotly' or 'matplotlib'.")


def plot_frontier_contour(
    result,
    input_vars: List[str],
    backend: str = "plotly",
    n_grid: int = 50,
    levels: int = 20,
    title: Optional[str] = None,
    **kwargs,
) -> Any:
    """Plot contour plot of frontier with two inputs.

    Parameters:
        result: SFResult object with fitted model
        input_vars: List of two input variable names
        backend: 'plotly' for interactive or 'matplotlib' for static
        n_grid: Number of grid points (default: 50)
        levels: Number of contour levels (default: 20)
        title: Custom plot title
        **kwargs: Additional arguments passed to plotting backend

    Returns:
        Plotly Figure or Matplotlib Figure object

    Example:
        >>> result = sf.fit()
        >>> fig = plot_frontier_contour(
        ...     result,
        ...     input_vars=['log_labor', 'log_capital'],
        ...     levels=20
        ... )
        >>> fig.show()
    """
    if len(input_vars) != 2:
        raise ValueError("Must provide exactly 2 input variables for contour plot")

    X_df = _get_X_df(result)
    y = result.model.y

    for var in input_vars:
        if var not in X_df.columns:
            raise ValueError(f"Input variable '{var}' not found in model data")

    # Get frontier parameters
    param_names = result.params.index.tolist()
    beta_names = [
        name for name in param_names if "sigma" not in name.lower() and "ln_" not in name.lower()
    ]
    beta = result.params[beta_names].values

    # Create grid
    x1_range = np.linspace(X_df[input_vars[0]].min(), X_df[input_vars[0]].max(), n_grid)
    x2_range = np.linspace(X_df[input_vars[1]].min(), X_df[input_vars[1]].max(), n_grid)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

    # Build design matrix
    n_points = n_grid * n_grid
    X_grid = np.ones((n_points, len(beta)))

    var1_idx = X_df.columns.get_loc(input_vars[0])
    var2_idx = X_df.columns.get_loc(input_vars[1])

    X_grid[:, var1_idx + 1] = X1_grid.flatten()
    X_grid[:, var2_idx + 1] = X2_grid.flatten()

    # Compute frontier
    Y_frontier = X_grid @ beta
    Y_grid = Y_frontier.reshape(n_grid, n_grid)

    # Get observations
    x1_obs = X_df[input_vars[0]].values
    x2_obs = X_df[input_vars[1]].values

    # Efficiency for coloring
    eff_df = result.efficiency(estimator="bc")
    eff_values = eff_df["efficiency"].values

    if title is None:
        title = "Frontier Contour Plot"

    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure()

        # Contour
        fig.add_trace(
            go.Contour(
                x=x1_range,
                y=x2_range,
                z=Y_grid,
                colorscale="Reds",
                contours=dict(
                    start=Y_grid.min(),
                    end=Y_grid.max(),
                    size=(Y_grid.max() - Y_grid.min()) / levels,
                ),
                colorbar=dict(title="Output"),
                name="Frontier",
            )
        )

        # Observations
        fig.add_trace(
            go.Scatter(
                x=x1_obs,
                y=x2_obs,
                mode="markers",
                marker=dict(
                    size=8,
                    color=eff_values,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Efficiency", x=1.15),
                    line=dict(color="black", width=1),
                ),
                name="Observations",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=input_vars[0],
            yaxis_title=input_vars[1],
            template="plotly_white",
            **kwargs,
        )

        return fig

    elif backend == "matplotlib":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 8)))

        # Contour
        contour = ax.contour(X1_grid, X2_grid, Y_grid, levels=levels, cmap="Reds")
        ax.clabel(contour, inline=True, fontsize=8)

        # Observations
        scatter = ax.scatter(
            x1_obs,
            x2_obs,
            c=eff_values,
            cmap="RdYlGn",
            s=80,
            edgecolors="black",
            linewidths=1,
            alpha=0.7,
        )

        # Colorbars
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Efficiency", rotation=270, labelpad=20)

        ax.set_xlabel(input_vars[0], fontsize=12)
        ax.set_ylabel(input_vars[1], fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'plotly' or 'matplotlib'.")


def plot_frontier_partial(
    result,
    input_var: str,
    fix_others_at: Union[str, Dict[str, float]] = "mean",
    backend: str = "plotly",
    title: Optional[str] = None,
    **kwargs,
) -> Any:
    """Plot partial frontier fixing other inputs at specified values.

    Parameters:
        result: SFResult object with fitted model
        input_var: Name of input variable to vary
        fix_others_at: How to fix other inputs
                       'mean' - fix at mean values
                       'median' - fix at median values
                       dict - specify value for each variable
        backend: 'plotly' for interactive or 'matplotlib' for static
        title: Custom plot title
        **kwargs: Additional arguments passed to plotting backend

    Returns:
        Plotly Figure or Matplotlib Figure object

    Example:
        >>> result = sf.fit()
        >>> # Fix other inputs at mean
        >>> fig = plot_frontier_partial(
        ...     result,
        ...     input_var='log_labor',
        ...     fix_others_at='mean'
        ... )
        >>> # Specify fixed values
        >>> fig = plot_frontier_partial(
        ...     result,
        ...     input_var='log_labor',
        ...     fix_others_at={'log_capital': 5.0, 'log_materials': 3.5}
        ... )
        >>> fig.show()
    """
    X_df = _get_X_df(result)

    if input_var not in X_df.columns:
        raise ValueError(f"Input variable '{input_var}' not found in model data")

    # Get frontier parameters
    param_names = result.params.index.tolist()
    beta_names = [
        name for name in param_names if "sigma" not in name.lower() and "ln_" not in name.lower()
    ]
    beta = result.params[beta_names].values

    # Determine fixed values for other inputs
    other_vars = [col for col in X_df.columns if col != input_var]

    if isinstance(fix_others_at, str):
        if fix_others_at == "mean":
            fixed_values = X_df[other_vars].mean().to_dict()
        elif fix_others_at == "median":
            fixed_values = X_df[other_vars].median().to_dict()
        else:
            raise ValueError(
                f"Unknown fix_others_at: {fix_others_at}. Use 'mean', 'median', or dict."
            )
    elif isinstance(fix_others_at, dict):
        fixed_values = fix_others_at
    else:
        raise ValueError("fix_others_at must be 'mean', 'median', or dict")

    # Create range for varying input
    x_range = np.linspace(X_df[input_var].min(), X_df[input_var].max(), 100)

    # Build design matrix
    n_points = len(x_range)
    X_grid = np.ones((n_points, len(beta)))

    # Set varying input
    var_idx = X_df.columns.get_loc(input_var)
    X_grid[:, var_idx + 1] = x_range

    # Set fixed inputs
    for var, value in fixed_values.items():
        if var in X_df.columns:
            var_idx = X_df.columns.get_loc(var)
            X_grid[:, var_idx + 1] = value

    # Compute frontier
    y_frontier = X_grid @ beta

    # Get observed values for this input
    x_obs = X_df[input_var].values
    y_obs = result.model.y

    if title is None:
        title = f"Partial Frontier: {input_var} (other inputs fixed)"

    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure()

        # Frontier line
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_frontier,
                mode="lines",
                line=dict(color="red", width=3),
                name="Partial Frontier",
            )
        )

        # Observations
        fig.add_trace(
            go.Scatter(
                x=x_obs,
                y=y_obs,
                mode="markers",
                marker=dict(size=6, color="blue", opacity=0.5),
                name="Observations",
            )
        )

        # Add annotation with fixed values
        fixed_text = "<b>Fixed at:</b><br>" + "<br>".join(
            [f"{k}: {v:.4f}" for k, v in fixed_values.items()]
        )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.02,
            xanchor="right",
            yanchor="bottom",
            text=fixed_text,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            bgcolor="white",
            opacity=0.8,
        )

        fig.update_layout(
            title=title,
            xaxis_title=input_var,
            yaxis_title="Output",
            template="plotly_white",
            **kwargs,
        )

        return fig

    elif backend == "matplotlib":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))

        # Frontier
        ax.plot(x_range, y_frontier, "r-", linewidth=3, label="Partial Frontier")

        # Observations
        ax.scatter(x_obs, y_obs, c="blue", s=40, alpha=0.5, label="Observations")

        # Annotation with fixed values
        fixed_text = "Fixed at:\n" + "\n".join([f"{k}: {v:.4f}" for k, v in fixed_values.items()])
        ax.text(
            0.98,
            0.02,
            fixed_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black"),
        )

        ax.set_xlabel(input_var, fontsize=12)
        ax.set_ylabel("Output", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'plotly' or 'matplotlib'.")
