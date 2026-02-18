"""
Advanced visualization tools for quantile regression.

This module provides professional-quality plots for quantile regression results,
including coefficient paths, fan charts, spaghetti plots, and more.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from scipy.interpolate import griddata, interp1d


class QuantileVisualizer:
    """
    Professional visualization suite for quantile regression.

    Provides publication-ready plots with customizable themes for presenting
    quantile regression results in academic papers and presentations.

    Parameters
    ----------
    style : str, default='academic'
        Plot style: 'academic', 'presentation', 'minimal'
    dpi : int, default=300
        Resolution for saved figures
    figsize : tuple, optional
        Default figure size (width, height)

    Examples
    --------
    >>> viz = QuantileVisualizer(style='academic')
    >>> fig = viz.coefficient_path(result, var_names=['education', 'experience'])
    >>> viz.save_all(result, output_dir='figures/')
    """

    def __init__(
        self, style: str = "academic", dpi: int = 300, figsize: Optional[Tuple[float, float]] = None
    ):
        """Initialize visualizer with specified style and settings."""
        self.style = style
        self.dpi = dpi
        self.figsize = figsize or (10, 6)
        self._setup_theme()

    def _setup_theme(self):
        """Configure matplotlib theme for publication."""
        # Use available styles
        available_styles = plt.style.available

        if self.style == "academic":
            if "seaborn-v0_8-paper" in available_styles:
                plt.style.use("seaborn-v0_8-paper")
            elif "seaborn-paper" in available_styles:
                plt.style.use("seaborn-paper")
            else:
                plt.style.use("seaborn" if "seaborn" in available_styles else "default")

            plt.rcParams.update(
                {
                    "font.size": 11,
                    "axes.labelsize": 12,
                    "axes.titlesize": 13,
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "legend.fontsize": 10,
                    "figure.dpi": self.dpi,
                    "font.family": "serif",
                    "font.serif": ["Times New Roman", "DejaVu Serif"],
                    "text.usetex": False,
                    "axes.grid": True,
                    "grid.alpha": 0.3,
                    "grid.linestyle": "--",
                }
            )

        elif self.style == "presentation":
            if "seaborn-v0_8-talk" in available_styles:
                plt.style.use("seaborn-v0_8-talk")
            elif "seaborn-talk" in available_styles:
                plt.style.use("seaborn-talk")
            else:
                plt.style.use("seaborn" if "seaborn" in available_styles else "default")
            plt.rcParams["figure.dpi"] = self.dpi

        elif self.style == "minimal":
            if "seaborn-v0_8-white" in available_styles:
                plt.style.use("seaborn-v0_8-white")
            elif "seaborn-white" in available_styles:
                plt.style.use("seaborn-white")
            else:
                plt.style.use("default")
            plt.rcParams["figure.dpi"] = self.dpi

    def coefficient_path(
        self,
        result,
        var_names: Optional[List[str]] = None,
        uniform_bands: bool = True,
        comparison: Optional[Dict] = None,
        colors: Optional[List] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        """
        Create coefficient path plot across quantiles.

        Shows how coefficients evolve across the quantile range with confidence bands.

        Parameters
        ----------
        result : QuantilePanelResult
            Fitted quantile regression result
        var_names : list, optional
            Variable names to plot
        uniform_bands : bool, default=True
            Include uniform confidence bands
        comparison : dict, optional
            Additional results to compare (e.g., {'OLS': ols_result})
        colors : list, optional
            Custom color palette
        figsize : tuple, optional
            Figure size (width, height)

        Returns
        -------
        fig : matplotlib.Figure
            The figure object

        Examples
        --------
        >>> fig = viz.coefficient_path(result, var_names=['education', 'age'])
        >>> plt.show()
        """
        # Get variable names and coefficients
        if not hasattr(result, "results"):
            raise ValueError("Result object must have 'results' attribute with quantile estimates")

        tau_list = sorted(result.results.keys())

        # Determine variable names and count
        first_result = result.results[tau_list[0]]
        n_vars = (
            len(first_result.params) if hasattr(first_result, "params") else first_result.shape[0]
        )

        if var_names is None:
            var_names = [f"X{i}" for i in range(n_vars)]
        else:
            n_vars = len(var_names)

        if colors is None:
            colors = sns.color_palette("husl", n_vars)

        figsize = figsize or self.figsize

        # Setup subplots
        n_cols = 2
        n_rows = (n_vars + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / 2), constrained_layout=True
        )

        if n_vars == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else np.array([axes])

        for idx in range(n_vars):
            ax = axes[idx] if n_vars > 1 else axes[0]
            color = colors[idx]
            var_name = var_names[idx]

            # Extract coefficients
            coefs = []
            lower_ci = []
            upper_ci = []

            for tau in tau_list:
                res_tau = result.results[tau]

                # Handle different result formats
                if hasattr(res_tau, "params"):
                    coef = res_tau.params[idx]

                    # Get confidence intervals
                    if hasattr(res_tau, "conf_int"):
                        ci = res_tau.conf_int()
                        lower_ci.append(ci[idx, 0])
                        upper_ci.append(ci[idx, 1])
                    elif hasattr(res_tau, "bse"):
                        # Use standard errors to compute CI
                        se = res_tau.bse[idx]
                        lower_ci.append(coef - 1.96 * se)
                        upper_ci.append(coef + 1.96 * se)
                    else:
                        lower_ci.append(coef)
                        upper_ci.append(coef)
                else:
                    # Direct coefficient array
                    coef = res_tau[idx] if hasattr(res_tau, "__getitem__") else res_tau
                    coefs.append(coef)
                    lower_ci.append(coef)
                    upper_ci.append(coef)

                coefs.append(coef)

            # Plot coefficient path
            ax.plot(tau_list, coefs, color=color, linewidth=2.5, label="QR Estimate", zorder=3)

            # Add confidence bands
            if len(lower_ci) == len(tau_list) and len(upper_ci) == len(tau_list):
                if uniform_bands:
                    # Compute uniform bands if requested
                    lower_band, upper_band = self._compute_uniform_bands(result, idx, tau_list)
                    ax.fill_between(
                        tau_list,
                        lower_band,
                        upper_band,
                        color=color,
                        alpha=0.2,
                        label="95% Uniform CI",
                    )
                else:
                    ax.fill_between(
                        tau_list,
                        lower_ci,
                        upper_ci,
                        color=color,
                        alpha=0.2,
                        label="95% Pointwise CI",
                    )

            # Add comparison (e.g., OLS)
            if comparison:
                for name, comp_result in comparison.items():
                    if hasattr(comp_result, "params"):
                        comp_coef = comp_result.params[idx]
                        comp_se = comp_result.bse[idx] if hasattr(comp_result, "bse") else 0
                        ax.axhline(
                            comp_coef,
                            color="red",
                            linestyle="--",
                            linewidth=1.5,
                            label=name,
                            zorder=2,
                        )
                        if comp_se > 0:
                            ax.fill_between(
                                [0, 1],
                                [comp_coef - 1.96 * comp_se] * 2,
                                [comp_coef + 1.96 * comp_se] * 2,
                                color="red",
                                alpha=0.1,
                            )

            # Formatting
            ax.set_xlabel("Quantile (τ)", fontweight="bold")
            ax.set_ylabel(f"Coefficient: {var_name}", fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(loc="best", frameon=True, shadow=True)

            # Add zero line
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

            # Highlight specific quantiles
            for tau_special in [0.25, 0.5, 0.75]:
                if tau_special in tau_list:
                    idx_special = tau_list.index(tau_special)
                    ax.plot(
                        tau_special, coefs[idx_special], "o", color="black", markersize=6, zorder=4
                    )

        # Hide unused subplots
        for j in range(n_vars, len(axes)):
            axes[j].set_visible(False)

        # Main title
        fig.suptitle(
            "Quantile Process: Coefficient Evolution", fontsize=14, fontweight="bold", y=1.02
        )

        return fig

    def fan_chart(
        self,
        result,
        X_forecast: np.ndarray,
        time_index: Optional[np.ndarray] = None,
        tau_list: Optional[List[float]] = None,
        colors: Optional[Union[str, List]] = None,
        alpha_gradient: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        """
        Create fan chart for quantile predictions.

        Perfect for time series forecasting visualization showing prediction
        intervals at multiple quantile levels.

        Parameters
        ----------
        result : QuantilePanelResult
            Fitted model
        X_forecast : array
            Covariate values for forecast
        time_index : array, optional
            Time index for x-axis
        tau_list : list, optional
            Quantiles to show (symmetric around median)
        colors : str or list
            Color scheme
        alpha_gradient : bool, default=True
            Use gradient transparency for outer quantiles
        figsize : tuple, optional
            Figure size

        Returns
        -------
        fig : matplotlib.Figure
            The figure object
        """
        if tau_list is None:
            tau_list = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        if time_index is None:
            time_index = np.arange(len(X_forecast))

        # Ensure median is included
        if 0.5 not in tau_list:
            tau_list = sorted(tau_list + [0.5])
        else:
            tau_list = sorted(tau_list)

        # Compute predictions
        predictions = {}
        for tau in tau_list:
            if tau in result.results:
                res = result.results[tau]
            else:
                # Need to fit for this quantile
                warnings.warn(f"Quantile {tau} not found in results, skipping")
                continue

            # Get parameters
            if hasattr(res, "params"):
                params = res.params
            else:
                params = res

            # Make predictions
            if X_forecast.ndim == 1:
                X_forecast = X_forecast.reshape(-1, 1)

            predictions[tau] = (
                X_forecast @ params
                if len(params) == X_forecast.shape[1]
                else X_forecast[:, : len(params)] @ params
            )

        # Create figure
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)

        # Color setup
        if colors is None:
            colors = "Blues"

        if isinstance(colors, str):
            cmap = plt.get_cmap(colors)
        else:
            cmap = colors

        # Find pairs of quantiles (symmetric around median)
        median_idx = tau_list.index(0.5) if 0.5 in tau_list else len(tau_list) // 2
        lower_taus = tau_list[:median_idx]
        upper_taus = tau_list[median_idx + 1 :][::-1]

        # Plot fan
        for i, (tau_lower, tau_upper) in enumerate(zip(lower_taus, upper_taus)):
            if tau_lower not in predictions or tau_upper not in predictions:
                continue

            if alpha_gradient:
                alpha = 0.2 + 0.5 * (i / max(len(lower_taus), 1))
            else:
                alpha = 0.3

            if isinstance(cmap, str):
                color = plt.get_cmap(cmap)(0.5 + 0.4 * (i / max(len(lower_taus), 1)))
            else:
                color = cmap(0.5 + 0.4 * (i / max(len(lower_taus), 1)))

            ax.fill_between(
                time_index,
                predictions[tau_lower],
                predictions[tau_upper],
                alpha=alpha,
                color=color,
                label=f"{int(tau_lower*100)}-{int(tau_upper*100)}%",
            )

        # Plot median
        if 0.5 in predictions:
            ax.plot(
                time_index, predictions[0.5], color="red", linewidth=2.5, label="Median", zorder=3
            )

        # Formatting
        ax.set_xlabel("Time", fontweight="bold")
        ax.set_ylabel("Predicted Value", fontweight="bold")
        ax.set_title("Quantile Fan Chart", fontweight="bold", fontsize=14)
        ax.legend(loc="upper left", frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Add percentage labels at the end
        for tau in [0.05, 0.95]:
            if tau in predictions:
                y_pos = predictions[tau][-1]
                ax.text(
                    time_index[-1], y_pos, f"{int(tau*100)}%", fontsize=8, ha="left", va="center"
                )

        return fig

    def conditional_density(
        self,
        result,
        X_values: Union[np.ndarray, Dict],
        y_grid: Optional[np.ndarray] = None,
        method: str = "kernel",
        bandwidth: Union[str, float] = "silverman",
        figsize: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        """
        Estimate and plot conditional density from quantiles.

        Parameters
        ----------
        result : QuantilePanelResult
            Must have many quantiles estimated
        X_values : array or dict
            Covariate values. If dict, keys are scenario labels
        y_grid : array, optional
            Grid for density evaluation
        method : str, default='kernel'
            'kernel' or 'interpolation'
        bandwidth : float or str
            Bandwidth for kernel density
        figsize : tuple, optional
            Figure size

        Returns
        -------
        fig : matplotlib.Figure
            The figure object
        """
        # Ensure dict format
        if not isinstance(X_values, dict):
            X_values = {"Scenario": X_values}

        # Need many quantiles for good density estimation
        tau_dense = np.linspace(0.01, 0.99, 99)

        # Estimate/interpolate for dense tau grid
        quantile_functions = {}

        for label, X in X_values.items():
            quantiles = []

            # Get available quantiles
            tau_available = sorted(result.results.keys())

            # Get coefficients for available quantiles
            coef_available = []
            for t in tau_available:
                res = result.results[t]
                if hasattr(res, "params"):
                    coef_available.append(res.params)
                else:
                    coef_available.append(res)

            coef_available = np.array(coef_available)

            # Interpolate for dense grid
            for tau in tau_dense:
                if tau in result.results:
                    res = result.results[tau]
                    if hasattr(res, "params"):
                        params = res.params
                    else:
                        params = res
                else:
                    # Interpolate each coefficient
                    coef_interp = []
                    for j in range(coef_available.shape[1]):
                        f = interp1d(
                            tau_available,
                            coef_available[:, j],
                            kind="linear",
                            fill_value="extrapolate",
                            bounds_error=False,
                        )
                        coef_interp.append(f(tau))
                    params = np.array(coef_interp)

                # Compute prediction
                if isinstance(X, np.ndarray):
                    if X.ndim == 1:
                        X_use = X
                    else:
                        X_use = X[0] if X.shape[0] > 0 else X
                else:
                    X_use = np.array(X)

                pred = (
                    np.dot(X_use, params)
                    if len(params) == len(X_use)
                    else X_use[: len(params)] @ params
                )
                quantiles.append(pred)

            quantile_functions[label] = np.array(quantiles).flatten()

        # Set y_grid based on quantile range
        if y_grid is None:
            y_min = min(qf.min() for qf in quantile_functions.values())
            y_max = max(qf.max() for qf in quantile_functions.values())
            y_range = y_max - y_min
            y_grid = np.linspace(y_min - 0.1 * y_range, y_max + 0.1 * y_range, 200)

        # Create figure
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        colors = sns.color_palette("husl", len(X_values))

        for (label, quantiles), color in zip(quantile_functions.items(), colors):

            if method == "kernel":
                # Kernel density from quantile function
                try:
                    from scipy.stats import gaussian_kde

                    # Create pseudo-sample from quantiles
                    if bandwidth == "silverman":
                        kde = gaussian_kde(quantiles.reshape(1, -1))
                    else:
                        kde = gaussian_kde(quantiles.reshape(1, -1), bw_method=bandwidth)

                    density = kde(y_grid)[0]
                except Exception as e:
                    warnings.warn(f"KDE failed: {e}, using interpolation instead")
                    # Fallback to interpolation
                    density = np.gradient(tau_dense, np.interp(y_grid, quantiles, tau_dense))
                    density = np.abs(density)
                    if density.sum() > 0:
                        density = density / np.trapz(density, y_grid)

            else:  # interpolation
                # Numerical derivative of quantile function
                density = np.gradient(tau_dense, np.interp(y_grid, quantiles, tau_dense))
                density = np.abs(density)
                if density.sum() > 0:
                    density = density / np.trapz(density, y_grid)

            ax.plot(y_grid, density, color=color, linewidth=2.5, label=label)
            ax.fill_between(y_grid, 0, density, color=color, alpha=0.2)

        # Formatting
        ax.set_xlabel("Value", fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.set_title("Conditional Density Estimation", fontweight="bold")
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle="--")

        return fig

    def spaghetti_plot(
        self,
        result,
        sample_size: int = 100,
        highlight_quantiles: List[float] = None,
        individual_alpha: float = 0.1,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        """
        Spaghetti plot showing individual quantile curves.

        Parameters
        ----------
        result : QuantilePanelResult
            Fitted model
        sample_size : int
            Number of individuals to show
        highlight_quantiles : list
            Quantiles to highlight
        individual_alpha : float
            Transparency for individual curves
        figsize : tuple, optional
            Figure size

        Returns
        -------
        fig : matplotlib.Figure
            The figure object
        """
        if highlight_quantiles is None:
            highlight_quantiles = [0.25, 0.5, 0.75]

        tau_grid = np.linspace(0.05, 0.95, 50)

        # Get model information
        if hasattr(result, "model") and hasattr(result.model, "n_entities"):
            n_entities = result.model.n_entities
            entity_ids = (
                result.model.entity_ids
                if hasattr(result.model, "entity_ids")
                else range(n_entities)
            )
            X_data = result.model.X if hasattr(result.model, "X") else None
        else:
            # Generate synthetic data for demonstration
            n_entities = min(sample_size, 100)
            entity_ids = range(n_entities)
            X_data = None

        # Sample individuals
        if sample_size < n_entities:
            sample_ids = np.random.choice(n_entities, sample_size, replace=False)
        else:
            sample_ids = range(n_entities)

        # Create figure
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)

        # Plot individual curves
        for entity_id in sample_ids:
            predictions = []

            # Get entity data or use random for demonstration
            if X_data is not None and hasattr(result.model, "entity_ids"):
                entity_mask = result.model.entity_ids == entity_id
                X_entity = X_data[entity_mask].mean(axis=0) if entity_mask.any() else X_data[0]
            else:
                # Use random variation for demonstration
                first_result = result.results[list(result.results.keys())[0]]
                n_params = (
                    len(first_result.params)
                    if hasattr(first_result, "params")
                    else len(first_result)
                )
                X_entity = np.random.randn(n_params) * 0.5 + 1.0

            for tau in tau_grid:
                if tau in result.results:
                    res = result.results[tau]
                    if hasattr(res, "params"):
                        pred = (
                            X_entity @ res.params
                            if len(res.params) == len(X_entity)
                            else X_entity[: len(res.params)] @ res.params
                        )
                    else:
                        pred = (
                            X_entity @ res
                            if len(res) == len(X_entity)
                            else X_entity[: len(res)] @ res
                        )
                else:
                    # Interpolate
                    tau_available = sorted(result.results.keys())
                    pred_available = []
                    for t in tau_available:
                        res = result.results[t]
                        if hasattr(res, "params"):
                            p = (
                                X_entity @ res.params
                                if len(res.params) == len(X_entity)
                                else X_entity[: len(res.params)] @ res.params
                            )
                        else:
                            p = (
                                X_entity @ res
                                if len(res) == len(X_entity)
                                else X_entity[: len(res)] @ res
                            )
                        pred_available.append(p)

                    f = interp1d(
                        tau_available,
                        pred_available,
                        kind="linear",
                        fill_value="extrapolate",
                        bounds_error=False,
                    )
                    pred = f(tau)

                predictions.append(pred)

            ax.plot(tau_grid, predictions, color="gray", alpha=individual_alpha, linewidth=0.5)

        # Highlight specific quantiles
        colors = ["blue", "red", "green"]
        for tau_h, color in zip(highlight_quantiles[: len(colors)], colors):
            predictions_h = []

            for entity_id in sample_ids:
                if X_data is not None and hasattr(result.model, "entity_ids"):
                    entity_mask = result.model.entity_ids == entity_id
                    X_entity = X_data[entity_mask].mean(axis=0) if entity_mask.any() else X_data[0]
                else:
                    first_result = result.results[list(result.results.keys())[0]]
                    n_params = (
                        len(first_result.params)
                        if hasattr(first_result, "params")
                        else len(first_result)
                    )
                    X_entity = np.random.randn(n_params) * 0.5 + 1.0

                if tau_h in result.results:
                    res = result.results[tau_h]
                    if hasattr(res, "params"):
                        pred = (
                            X_entity @ res.params
                            if len(res.params) == len(X_entity)
                            else X_entity[: len(res.params)] @ res.params
                        )
                    else:
                        pred = (
                            X_entity @ res
                            if len(res) == len(X_entity)
                            else X_entity[: len(res)] @ res
                        )
                else:
                    # Interpolate
                    tau_available = sorted(result.results.keys())
                    pred_available = []
                    for t in tau_available:
                        res = result.results[t]
                        if hasattr(res, "params"):
                            p = (
                                X_entity @ res.params
                                if len(res.params) == len(X_entity)
                                else X_entity[: len(res.params)] @ res.params
                            )
                        else:
                            p = (
                                X_entity @ res
                                if len(res) == len(X_entity)
                                else X_entity[: len(res)] @ res
                            )
                        pred_available.append(p)

                    f = interp1d(
                        tau_available,
                        pred_available,
                        kind="linear",
                        fill_value="extrapolate",
                        bounds_error=False,
                    )
                    pred = f(tau_h)

                predictions_h.append(pred)

            ax.scatter(
                [tau_h] * len(predictions_h),
                predictions_h,
                color=color,
                s=10,
                alpha=0.5,
                label=f"τ={tau_h}",
                zorder=3,
            )

        # Formatting
        ax.set_xlabel("Quantile (τ)", fontweight="bold")
        ax.set_ylabel("Predicted Value", fontweight="bold")
        ax.set_title("Individual Quantile Curves (Spaghetti Plot)", fontweight="bold")
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle="--")

        return fig

    def _compute_uniform_bands(
        self, result, var_idx: int, tau_list: List[float], alpha: float = 0.05
    ) -> Tuple[List[float], List[float]]:
        """
        Compute uniform confidence bands.

        Uses Bonferroni adjustment for conservative bands.
        """
        if hasattr(result, "uniform_bands"):
            return result.uniform_bands[var_idx]

        # Conservative Bonferroni adjustment
        m = len(tau_list)
        alpha_adj = alpha / m

        lower = []
        upper = []

        for tau in tau_list:
            res = result.results[tau]

            if hasattr(res, "conf_int"):
                ci = res.conf_int(alpha=alpha_adj)
                lower.append(ci[var_idx, 0])
                upper.append(ci[var_idx, 1])
            elif hasattr(res, "bse"):
                # Use standard errors
                coef = res.params[var_idx]
                se = res.bse[var_idx]
                z_val = 2.576  # Roughly for alpha_adj with Bonferroni
                lower.append(coef - z_val * se)
                upper.append(coef + z_val * se)
            else:
                # No uncertainty available
                if hasattr(res, "params"):
                    coef = res.params[var_idx]
                else:
                    coef = res[var_idx] if hasattr(res, "__getitem__") else res
                lower.append(coef)
                upper.append(coef)

        return lower, upper

    def save_all(self, result, output_dir: str, formats: List[str] = None, **kwargs) -> None:
        """
        Generate and save all standard plots.

        Parameters
        ----------
        result : QuantilePanelResult
            Fitted model
        output_dir : str
            Output directory
        formats : list
            File formats to save (default: ['png', 'pdf'])
        **kwargs
            Additional arguments for plots
        """
        import os

        if formats is None:
            formats = ["png", "pdf"]

        os.makedirs(output_dir, exist_ok=True)

        try:
            # Coefficient paths
            fig = self.coefficient_path(result, **kwargs)
            for fmt in formats:
                fig.savefig(
                    os.path.join(output_dir, f"coefficient_paths.{fmt}"),
                    dpi=self.dpi,
                    bbox_inches="tight",
                )
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"Could not generate coefficient path: {e}")

        # Fan chart if we have forecast data
        try:
            if hasattr(result, "model") and hasattr(result.model, "X"):
                # Use last observations as forecast base
                X_forecast = (
                    result.model.X[-20:] if result.model.X.shape[0] >= 20 else result.model.X
                )
                fig = self.fan_chart(result, X_forecast, **kwargs)
                for fmt in formats:
                    fig.savefig(
                        os.path.join(output_dir, f"fan_chart.{fmt}"),
                        dpi=self.dpi,
                        bbox_inches="tight",
                    )
                plt.close(fig)
        except Exception as e:
            warnings.warn(f"Could not generate fan chart: {e}")

        # Spaghetti plot
        try:
            fig = self.spaghetti_plot(result, **kwargs)
            for fmt in formats:
                fig.savefig(
                    os.path.join(output_dir, f"spaghetti_plot.{fmt}"),
                    dpi=self.dpi,
                    bbox_inches="tight",
                )
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"Could not generate spaghetti plot: {e}")

        plt.close("all")
        print(f"All plots saved to {output_dir}")
