"""
Interactive visualizations for quantile regression using Plotly.

This module provides interactive plots that allow users to explore
quantile regression results dynamically through web-based interfaces.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class InteractivePlotter:
    """
    Interactive visualization tools for quantile regression.

    Creates dynamic, web-based visualizations using Plotly that allow
    users to explore results interactively.

    Examples
    --------
    >>> plotter = InteractivePlotter()
    >>> fig = plotter.coefficient_dashboard(result)
    >>> fig.show()
    """

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize interactive plotter.

        Parameters
        ----------
        theme : str
            Plotly theme to use
        """
        self.theme = theme

    def coefficient_dashboard(self, result, var_names: Optional[List[str]] = None) -> go.Figure:
        """
        Create interactive dashboard for coefficient exploration.

        Parameters
        ----------
        result : QuantilePanelResult
            Fitted quantile regression result
        var_names : list, optional
            Variable names

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive dashboard
        """
        # Get data
        tau_list = sorted(result.results.keys())

        # Determine variables
        first_result = result.results[tau_list[0]]
        if hasattr(first_result, "params"):
            n_vars = len(first_result.params)
            coef_matrix = np.array([result.results[tau].params for tau in tau_list])
        else:
            n_vars = len(first_result) if hasattr(first_result, "__len__") else 1
            coef_matrix = np.array([result.results[tau] for tau in tau_list])

        if var_names is None:
            var_names = [f"Variable {i+1}" for i in range(n_vars)]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Coefficient Paths",
                "Coefficient Heatmap",
                "Coefficient Distribution",
                "Significance Plot",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "heatmap"}],
                [{"type": "violin"}, {"type": "scatter"}],
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        # 1. Coefficient paths with dropdown
        for i, var_name in enumerate(var_names):
            visible = i == 0  # Only first variable visible initially

            # Add coefficient line
            fig.add_trace(
                go.Scatter(
                    x=tau_list,
                    y=coef_matrix[:, i],
                    mode="lines+markers",
                    name=var_name,
                    visible=visible,
                    line=dict(width=3),
                    marker=dict(size=8),
                    hovertemplate="τ: %{x:.2f}<br>Coefficient: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Add confidence bands if available
            if hasattr(first_result, "bse"):
                lower = coef_matrix[:, i] - 1.96 * np.array(
                    [result.results[tau].bse[i] for tau in tau_list]
                )
                upper = coef_matrix[:, i] + 1.96 * np.array(
                    [result.results[tau].bse[i] for tau in tau_list]
                )

                # Upper band
                fig.add_trace(
                    go.Scatter(
                        x=tau_list,
                        y=upper,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        visible=visible,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )

                # Lower band
                fig.add_trace(
                    go.Scatter(
                        x=tau_list,
                        y=lower,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(68, 68, 68, 0.2)",
                        showlegend=False,
                        visible=visible,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )

        # 2. Heatmap
        fig.add_trace(
            go.Heatmap(
                x=[f"τ={tau:.2f}" for tau in tau_list],
                y=var_names,
                z=coef_matrix.T,
                colorscale="RdBu",
                zmid=0,
                text=coef_matrix.T.round(3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="Variable: %{y}<br>Quantile: %{x}<br>Coefficient: %{z:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # 3. Violin plot for coefficient distribution
        violin_data = []
        for i, var_name in enumerate(var_names):
            for j, tau in enumerate(tau_list):
                violin_data.append(
                    {"Variable": var_name, "Coefficient": coef_matrix[j, i], "Quantile": tau}
                )

        import pandas as pd

        df_violin = pd.DataFrame(violin_data)

        for var_name in var_names:
            df_var = df_violin[df_violin["Variable"] == var_name]
            fig.add_trace(
                go.Violin(
                    y=df_var["Coefficient"], name=var_name, box_visible=True, meanline_visible=True
                ),
                row=2,
                col=1,
            )

        # 4. Significance plot (p-values if available)
        if hasattr(first_result, "pvalues"):
            p_matrix = np.array([result.results[tau].pvalues for tau in tau_list])

            for i, var_name in enumerate(var_names):
                fig.add_trace(
                    go.Scatter(
                        x=tau_list,
                        y=-np.log10(p_matrix[:, i]),
                        mode="lines+markers",
                        name=var_name,
                        hovertemplate="τ: %{x:.2f}<br>-log10(p): %{y:.2f}<extra></extra>",
                    ),
                    row=2,
                    col=2,
                )

            # Add significance threshold line
            fig.add_hline(
                y=-np.log10(0.05),
                row=2,
                col=2,
                line_dash="dash",
                line_color="red",
                annotation_text="p=0.05",
            )

        # Create dropdown menu for variable selection
        n_traces_per_var = 3 if hasattr(first_result, "bse") else 1
        dropdown_buttons = []

        for i, var_name in enumerate(var_names):
            # Create visibility list
            visible_list = [False] * (n_vars * n_traces_per_var)
            for j in range(n_traces_per_var):
                visible_list[i * n_traces_per_var + j] = True

            # Keep other plots visible
            visible_list.extend([True] * (len(fig.data) - n_vars * n_traces_per_var))

            dropdown_buttons.append(
                dict(
                    label=var_name,
                    method="update",
                    args=[
                        {"visible": visible_list},
                        {"title.text": f"Quantile Regression Analysis - {var_name}"},
                    ],
                )
            )

        # Update layout
        fig.update_layout(
            title="Quantile Regression Interactive Dashboard",
            template=self.theme,
            showlegend=True,
            height=800,
            updatemenus=[
                dict(
                    active=0,
                    buttons=dropdown_buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                )
            ],
        )

        # Update axes
        fig.update_xaxes(title_text="Quantile (τ)", row=1, col=1)
        fig.update_yaxes(title_text="Coefficient", row=1, col=1)
        fig.update_yaxes(title_text="Coefficient", row=2, col=1)
        fig.update_xaxes(title_text="Quantile (τ)", row=2, col=2)
        fig.update_yaxes(title_text="-log10(p-value)", row=2, col=2)

        return fig

    def animated_coefficient_path(
        self, result, var_idx: int = 0, var_name: Optional[str] = None
    ) -> go.Figure:
        """
        Create animated coefficient path showing evolution across quantiles.

        Parameters
        ----------
        result : QuantilePanelResult
            Fitted result
        var_idx : int
            Variable index to animate
        var_name : str, optional
            Variable name for display

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Animated figure
        """
        tau_list = sorted(result.results.keys())

        if var_name is None:
            var_name = f"Variable {var_idx + 1}"

        # Extract coefficients
        coefs = []
        for tau in tau_list:
            res = result.results[tau]
            if hasattr(res, "params"):
                coefs.append(res.params[var_idx])
            else:
                coefs.append(res[var_idx] if hasattr(res, "__getitem__") else res)

        # Create frames for animation
        frames = []
        for i in range(1, len(tau_list) + 1):
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=tau_list[:i],
                        y=coefs[:i],
                        mode="lines+markers",
                        line=dict(color="blue", width=3),
                        marker=dict(size=8, color="blue"),
                    )
                ],
                name=str(i),
            )
            frames.append(frame)

        # Create initial figure
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=[tau_list[0]],
                    y=[coefs[0]],
                    mode="lines+markers",
                    line=dict(color="blue", width=3),
                    marker=dict(size=8, color="blue"),
                )
            ],
            frames=frames,
        )

        # Add play button and slider
        fig.update_layout(
            title=f"Animated Coefficient Path - {var_name}",
            template=self.theme,
            xaxis=dict(range=[0, 1], title="Quantile (τ)"),
            yaxis=dict(range=[min(coefs) * 1.1, max(coefs) * 1.1], title="Coefficient"),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                            ],
                        ),
                    ],
                    direction="left",
                    pad={"r": 10, "t": 70},
                    x=0.1,
                    xanchor="right",
                    y=0,
                    yanchor="top",
                )
            ],
            sliders=[
                dict(
                    active=0,
                    steps=[
                        dict(
                            args=[
                                [frame.name],
                                {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"},
                            ],
                            label=f"τ={tau_list[i]:.2f}",
                            method="animate",
                        )
                        for i, frame in enumerate(frames)
                    ],
                    x=0.1,
                    xanchor="left",
                    y=0,
                    yanchor="top",
                    pad={"b": 10, "t": 50},
                    len=0.9,
                    currentvalue={
                        "font": {"size": 16},
                        "prefix": "Quantile: ",
                        "visible": True,
                        "xanchor": "right",
                    },
                )
            ],
        )

        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        return fig

    def quantile_surface_interactive(
        self, result, X_grid: Optional[np.ndarray] = None
    ) -> go.Figure:
        """
        Create fully interactive 3D surface with controls.

        Parameters
        ----------
        result : QuantilePanelResult
            Fitted result
        X_grid : array, optional
            Grid for X values

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive 3D figure
        """
        tau_list = sorted(result.results.keys())

        # Create grid
        if X_grid is None:
            X1 = np.linspace(-2, 2, 30)
            X2 = np.linspace(-2, 2, 30)
        else:
            X1 = X_grid[0] if len(X_grid) > 0 else np.linspace(-2, 2, 30)
            X2 = X_grid[1] if len(X_grid) > 1 else np.linspace(-2, 2, 30)

        X1_mesh, X2_mesh = np.meshgrid(X1, X2)

        # Get number of parameters
        first_result = result.results[tau_list[0]]
        if hasattr(first_result, "params"):
            n_params = len(first_result.params)
        else:
            n_params = len(first_result) if hasattr(first_result, "__len__") else 1

        # Create figure with multiple surfaces
        fig = go.Figure()

        # Add surface for each quantile
        for tau in tau_list:
            res = result.results[tau]

            # Create predictions
            X_pred = np.zeros((X1_mesh.size, n_params))
            if n_params > 0:
                X_pred[:, 0] = 1  # Intercept
            if n_params > 1:
                X_pred[:, 1] = X1_mesh.flatten()
            if n_params > 2:
                X_pred[:, 2] = X2_mesh.flatten()

            if hasattr(res, "params"):
                Z = (X_pred @ res.params).reshape(X1_mesh.shape)
            else:
                params = res if hasattr(res, "__len__") else [res]
                Z = (X_pred[:, : len(params)] @ params).reshape(X1_mesh.shape)

            # Add surface
            fig.add_trace(
                go.Surface(
                    x=X1,
                    y=X2,
                    z=Z,
                    name=f"τ={tau:.2f}",
                    visible=(tau == 0.5),  # Only median visible initially
                    opacity=0.8,
                    colorscale="Viridis",
                    showscale=True,
                )
            )

        # Create slider for quantile selection
        steps = []
        for i, tau in enumerate(tau_list):
            step = dict(
                method="update",
                args=[
                    {"visible": [j == i for j in range(len(tau_list))]},
                    {"title": f"Quantile Surface: τ = {tau:.2f}"},
                ],
                label=f"{tau:.2f}",
            )
            steps.append(step)

        sliders = [
            dict(
                active=len(tau_list) // 2,  # Start at median
                currentvalue={"prefix": "Quantile: "},
                pad={"t": 50},
                steps=steps,
            )
        ]

        # Update layout
        fig.update_layout(
            title="Interactive Quantile Surface",
            template=self.theme,
            scene=dict(
                xaxis_title="X1",
                yaxis_title="X2",
                zaxis_title="Predicted Value",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            sliders=sliders,
            height=700,
            showlegend=True,
        )

        return fig

    def parallel_coordinates(self, result, n_samples: int = 100) -> go.Figure:
        """
        Create parallel coordinates plot for multivariate coefficient analysis.

        Parameters
        ----------
        result : QuantilePanelResult
            Fitted result
        n_samples : int
            Number of quantiles to sample

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Parallel coordinates plot
        """
        tau_list = sorted(result.results.keys())

        # Sample quantiles if too many
        if len(tau_list) > n_samples:
            import random

            tau_sample = sorted(random.sample(tau_list, n_samples))
        else:
            tau_sample = tau_list

        # Extract coefficients
        first_result = result.results[tau_sample[0]]
        if hasattr(first_result, "params"):
            n_vars = len(first_result.params)
            var_names = [f"Var{i+1}" for i in range(n_vars)]
        else:
            n_vars = 1
            var_names = ["Var1"]

        # Create data for parallel coordinates
        data = []
        for tau in tau_sample:
            res = result.results[tau]
            if hasattr(res, "params"):
                coefs = res.params
            else:
                coefs = [res] if not hasattr(res, "__len__") else res

            row = {"tau": tau}
            for i, var_name in enumerate(var_names):
                if i < len(coefs):
                    row[var_name] = coefs[i]
                else:
                    row[var_name] = 0
            data.append(row)

        import pandas as pd

        df = pd.DataFrame(data)

        # Create parallel coordinates plot
        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=df["tau"],
                    colorscale="Viridis",
                    showscale=True,
                    cmin=0,
                    cmax=1,
                    colorbar=dict(title="Quantile"),
                ),
                dimensions=[
                    dict(range=[df[col].min(), df[col].max()], label=col, values=df[col])
                    for col in var_names
                ],
            )
        )

        fig.update_layout(
            title="Parallel Coordinates: Coefficient Evolution", template=self.theme, height=600
        )

        return fig
