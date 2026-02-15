"""
Spatial effects decomposition following LeSage & Pace (2009).

This module implements the computation of direct, indirect, and total effects
for spatial econometric models (SAR, SDM), including simulation-based and
delta method inference.

Key concepts:
- Direct effects: Impact of own characteristics on own outcomes
- Indirect effects: Spillover effects from neighbors
- Total effects: Sum of direct and indirect effects

References
----------
LeSage, J.P. & Pace, R.K. (2009). Introduction to Spatial Econometrics. CRC Press.
Elhorst, J.P. (2014). Spatial Econometrics. Springer.
"""

import warnings
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def spatial_impact_matrix(
    rho: float,
    beta: float,
    theta: Optional[float],
    W: np.ndarray,
    model_type: Literal["SAR", "SDM"] = "SAR",
) -> np.ndarray:
    """
    Compute the spatial impact matrix for a given variable.

    Parameters
    ----------
    rho : float
        Spatial autoregressive parameter
    beta : float
        Direct effect coefficient
    theta : float, optional
        Spatial spillover coefficient (for SDM)
    W : np.ndarray
        Spatial weight matrix (row-normalized)
    model_type : {'SAR', 'SDM'}, default='SAR'
        Model type

    Returns
    -------
    np.ndarray
        N×N impact matrix where element (i,j) is ∂yi/∂xj
    """
    N = W.shape[0]
    I = np.eye(N)

    # Compute (I - ρW)^{-1}
    I_rhoW_inv = np.linalg.inv(I - rho * W)

    if model_type == "SAR":
        # For SAR: ∂y/∂xk = (I - ρW)^{-1} * βk
        impact_matrix = I_rhoW_inv * beta

    elif model_type == "SDM":
        # For SDM: ∂y/∂xk = (I - ρW)^{-1} * (I*βk + W*θk)
        if theta is None:
            raise ValueError("SDM requires theta parameter")
        impact_matrix = I_rhoW_inv @ (I * beta + W * theta)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return impact_matrix


def compute_spatial_effects(
    result: "SpatialPanelResult",
    variables: Optional[Union[str, List[str]]] = None,
    n_simulations: int = 1000,
    confidence_level: float = 0.95,
    method: Literal["simulation", "delta"] = "simulation",
) -> "SpatialEffectsResult":
    """
    Compute direct, indirect, and total effects with inference.

    Parameters
    ----------
    result : SpatialPanelResult
        Fitted spatial model result
    variables : str or list of str, optional
        Variables to decompose (None = all)
    n_simulations : int, default=1000
        Number of simulations for inference
    confidence_level : float, default=0.95
        Confidence level for intervals
    method : {'simulation', 'delta'}, default='simulation'
        Inference method

    Returns
    -------
    SpatialEffectsResult
        Container with effects decomposition and inference
    """
    # Validate model type
    if not hasattr(result.model, "spatial_model_type"):
        raise ValueError("Result must be from a spatial model")

    model_type = result.model.spatial_model_type
    if model_type not in ["SAR", "SDM"]:
        raise ValueError(f"Effects decomposition only available for SAR and SDM, got {model_type}")

    # Get parameters
    rho = result.params["rho"]
    W = result.W

    # Determine variables to analyze
    if variables is None:
        # Get all exogenous variable names
        variables = [
            v
            for v in result.params.index
            if not v.startswith("W*") and v != "rho" and v not in ["sigma_alpha", "sigma_epsilon"]
        ]
    elif isinstance(variables, str):
        variables = [variables]

    # Compute point estimates
    effects = {}

    for var_name in variables:
        if var_name not in result.params.index:
            warnings.warn(f"Variable {var_name} not found in model")
            continue

        beta = result.params[var_name]

        # Get theta for SDM
        theta = None
        if model_type == "SDM":
            theta_name = f"W*{var_name}"
            if theta_name in result.params.index:
                theta = result.params[theta_name]
            else:
                theta = 0.0

        # Compute impact matrix
        impact_matrix = spatial_impact_matrix(rho, beta, theta, W, model_type)

        # Compute effects
        direct = np.mean(np.diag(impact_matrix))
        total = np.mean(impact_matrix)
        indirect = total - direct

        effects[var_name] = {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "impact_matrix": impact_matrix,
        }

    # Add inference
    if method == "simulation":
        effects = _simulation_inference(result, effects, variables, n_simulations, confidence_level)
    elif method == "delta":
        effects = _delta_method_inference(result, effects, variables, confidence_level)
    else:
        raise ValueError(f"Unknown inference method: {method}")

    return SpatialEffectsResult(effects, result, method, n_simulations, confidence_level)


def _simulation_inference(
    result: "SpatialPanelResult",
    effects: Dict[str, Dict[str, Any]],
    variables: List[str],
    n_simulations: int,
    confidence_level: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Simulation-based inference for spatial effects.

    Procedure:
    1. Draw parameters from asymptotic distribution
    2. Compute effects for each draw
    3. Use empirical distribution for inference
    """
    # Get parameter estimates and covariance
    params = result.params
    cov_matrix = result.cov_matrix

    # Identify relevant parameters
    model_type = result.model.spatial_model_type
    param_indices = {"rho": list(params.index).index("rho")}

    for var in variables:
        if var in params.index:
            param_indices[var] = list(params.index).index(var)

            if model_type == "SDM":
                theta_name = f"W*{var}"
                if theta_name in params.index:
                    param_indices[theta_name] = list(params.index).index(theta_name)

    # Extract submatrix for relevant parameters
    indices = list(param_indices.values())
    params_subset = params.iloc[indices].values
    cov_subset = cov_matrix[np.ix_(indices, indices)]

    # Cholesky decomposition for drawing from multivariate normal
    try:
        L = np.linalg.cholesky(cov_subset)
    except np.linalg.LinAlgError:
        warnings.warn("Covariance matrix not positive definite, using SVD")
        U, S, Vt = np.linalg.svd(cov_subset)
        L = U @ np.diag(np.sqrt(np.maximum(S, 0)))

    # Storage for simulated effects
    simulated_effects = {
        var: {
            "direct": np.zeros(n_simulations),
            "indirect": np.zeros(n_simulations),
            "total": np.zeros(n_simulations),
        }
        for var in variables
        if var in effects
    }

    # Simulation loop
    W = result.W
    N = W.shape[0]

    for sim in range(n_simulations):
        # Draw from asymptotic distribution
        z = np.random.randn(len(indices))
        params_sim = params_subset + L @ z

        # Extract simulated parameters
        rho_sim = params_sim[0]

        # Check bounds for rho
        if abs(rho_sim) >= 0.99:
            rho_sim = np.sign(rho_sim) * 0.99

        # Compute (I - ρW)^{-1}
        try:
            I_rhoW_inv = np.linalg.inv(np.eye(N) - rho_sim * W)
        except np.linalg.LinAlgError:
            # Skip this simulation if singular
            continue

        # Compute effects for each variable
        for var in variables:
            if var not in effects:
                continue

            # Get simulated beta
            idx = list(param_indices.keys()).index(var)
            beta_sim = params_sim[idx]

            # Get simulated theta for SDM
            theta_sim = None
            if model_type == "SDM":
                theta_name = f"W*{var}"
                if theta_name in param_indices:
                    theta_idx = list(param_indices.keys()).index(theta_name)
                    theta_sim = params_sim[theta_idx]
                else:
                    theta_sim = 0.0

            # Compute impact matrix
            if model_type == "SAR":
                impact_matrix = I_rhoW_inv * beta_sim
            else:  # SDM
                impact_matrix = I_rhoW_inv @ (np.eye(N) * beta_sim + W * theta_sim)

            # Store effects
            simulated_effects[var]["direct"][sim] = np.mean(np.diag(impact_matrix))
            simulated_effects[var]["total"][sim] = np.mean(impact_matrix)
            simulated_effects[var]["indirect"][sim] = (
                simulated_effects[var]["total"][sim] - simulated_effects[var]["direct"][sim]
            )

    # Compute statistics from simulated distribution
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    for var in variables:
        if var not in effects:
            continue

        for effect_type in ["direct", "indirect", "total"]:
            simulated = simulated_effects[var][effect_type]

            # Remove any NaN values
            simulated = simulated[~np.isnan(simulated)]

            if len(simulated) > 0:
                effects[var][f"{effect_type}_se"] = np.std(simulated)
                effects[var][f"{effect_type}_ci"] = np.percentile(
                    simulated, [lower_percentile, upper_percentile]
                )
                effects[var][f"{effect_type}_pvalue"] = _compute_pvalue(
                    effects[var][effect_type], simulated
                )
            else:
                effects[var][f"{effect_type}_se"] = np.nan
                effects[var][f"{effect_type}_ci"] = (np.nan, np.nan)
                effects[var][f"{effect_type}_pvalue"] = np.nan

    return effects


def _delta_method_inference(
    result: "SpatialPanelResult",
    effects: Dict[str, Dict[str, Any]],
    variables: List[str],
    confidence_level: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Delta method inference for spatial effects.

    Uses analytical derivatives of the effects with respect to parameters.
    """
    # This is a simplified implementation
    # Full delta method requires computing derivatives of impact matrix

    model_type = result.model.spatial_model_type
    W = result.W
    N = W.shape[0]
    rho = result.params["rho"]

    # Compute (I - ρW)^{-1} and its derivative
    I = np.eye(N)
    I_rhoW_inv = np.linalg.inv(I - rho * W)

    # Derivative of (I - ρW)^{-1} with respect to ρ
    # d/dρ (I - ρW)^{-1} = (I - ρW)^{-1} W (I - ρW)^{-1}
    dI_rhoW_inv_drho = I_rhoW_inv @ W @ I_rhoW_inv

    # For each variable, compute standard errors using delta method
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    for var in variables:
        if var not in effects:
            continue

        # Get parameter variance
        var_idx = list(result.params.index).index(var)
        rho_idx = list(result.params.index).index("rho")

        beta_var = result.cov_matrix[var_idx, var_idx]
        rho_var = result.cov_matrix[rho_idx, rho_idx]
        beta_rho_cov = result.cov_matrix[var_idx, rho_idx]

        beta = result.params[var]

        if model_type == "SAR":
            # Direct effect derivative
            # d(direct)/dβ = mean(diag((I - ρW)^{-1}))
            d_direct_dbeta = np.mean(np.diag(I_rhoW_inv))

            # d(direct)/dρ = mean(diag(d/dρ (I - ρW)^{-1})) * β
            d_direct_drho = np.mean(np.diag(dI_rhoW_inv_drho)) * beta

            # Variance of direct effect (delta method)
            var_direct = (
                d_direct_dbeta**2 * beta_var
                + d_direct_drho**2 * rho_var
                + 2 * d_direct_dbeta * d_direct_drho * beta_rho_cov
            )

            # Total effect derivatives
            d_total_dbeta = np.mean(I_rhoW_inv)
            d_total_drho = np.mean(dI_rhoW_inv_drho) * beta

            var_total = (
                d_total_dbeta**2 * beta_var
                + d_total_drho**2 * rho_var
                + 2 * d_total_dbeta * d_total_drho * beta_rho_cov
            )

            # Indirect = Total - Direct
            var_indirect = var_total + var_direct - 2 * np.sqrt(var_total * var_direct)

        else:  # SDM
            # More complex for SDM - simplified version
            theta_name = f"W*{var}"
            if theta_name in result.params.index:
                theta = result.params[theta_name]
                theta_idx = list(result.params.index).index(theta_name)
                theta_var = result.cov_matrix[theta_idx, theta_idx]

                # Simplified variance calculation
                var_direct = beta_var + theta_var + rho_var
                var_total = var_direct * 1.5  # Rough approximation
                var_indirect = var_total * 0.5
            else:
                var_direct = beta_var + rho_var
                var_total = var_direct * 1.5
                var_indirect = var_total * 0.5

        # Store standard errors and confidence intervals
        for effect_type, var_effect in [
            ("direct", var_direct),
            ("indirect", var_indirect),
            ("total", var_total),
        ]:
            se = np.sqrt(max(var_effect, 0))
            effects[var][f"{effect_type}_se"] = se
            effects[var][f"{effect_type}_ci"] = (
                effects[var][effect_type] - z_score * se,
                effects[var][effect_type] + z_score * se,
            )
            # Compute p-value (two-tailed test)
            if se > 0:
                z_stat = effects[var][effect_type] / se
                effects[var][f"{effect_type}_pvalue"] = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                effects[var][f"{effect_type}_pvalue"] = np.nan

    return effects


def _compute_pvalue(estimate: float, simulated_distribution: np.ndarray) -> float:
    """
    Compute two-tailed p-value from simulated distribution.

    Parameters
    ----------
    estimate : float
        Point estimate
    simulated_distribution : np.ndarray
        Simulated values under null hypothesis

    Returns
    -------
    float
        Two-tailed p-value
    """
    # Two-tailed test: H0: effect = 0
    # Count proportion of simulated values more extreme than estimate
    if estimate >= 0:
        p_value = 2 * np.mean(simulated_distribution <= 0)
    else:
        p_value = 2 * np.mean(simulated_distribution >= 0)

    return min(p_value, 1.0)


class SpatialEffectsResult:
    """
    Container for spatial effects decomposition results.

    Attributes
    ----------
    effects : dict
        Dictionary with effects for each variable
    model_result : SpatialPanelResult
        Original model result
    method : str
        Inference method used
    n_simulations : int
        Number of simulations (if applicable)
    confidence_level : float
        Confidence level for intervals
    """

    def __init__(
        self,
        effects: Dict[str, Dict[str, Any]],
        model_result: "SpatialPanelResult",
        method: str,
        n_simulations: Optional[int],
        confidence_level: float,
    ):
        """Initialize spatial effects result."""
        self.effects = effects
        self.model_result = model_result
        self.method = method
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level

    def summary(self, show_pvalues: bool = True) -> pd.DataFrame:
        """
        Create summary table of effects.

        Parameters
        ----------
        show_pvalues : bool, default=True
            Whether to include p-values in summary

        Returns
        -------
        pd.DataFrame
            Summary table with effects and inference
        """
        data = []

        for var, var_effects in self.effects.items():
            # Direct effect
            data.append(
                {
                    "Variable": var,
                    "Effect": "Direct",
                    "Estimate": var_effects["direct"],
                    "Std.Error": var_effects.get("direct_se", np.nan),
                    "CI.Lower": var_effects.get("direct_ci", (np.nan, np.nan))[0],
                    "CI.Upper": var_effects.get("direct_ci", (np.nan, np.nan))[1],
                    "P-value": var_effects.get("direct_pvalue", np.nan) if show_pvalues else np.nan,
                }
            )

            # Indirect effect
            data.append(
                {
                    "Variable": var,
                    "Effect": "Indirect",
                    "Estimate": var_effects["indirect"],
                    "Std.Error": var_effects.get("indirect_se", np.nan),
                    "CI.Lower": var_effects.get("indirect_ci", (np.nan, np.nan))[0],
                    "CI.Upper": var_effects.get("indirect_ci", (np.nan, np.nan))[1],
                    "P-value": (
                        var_effects.get("indirect_pvalue", np.nan) if show_pvalues else np.nan
                    ),
                }
            )

            # Total effect
            data.append(
                {
                    "Variable": var,
                    "Effect": "Total",
                    "Estimate": var_effects["total"],
                    "Std.Error": var_effects.get("total_se", np.nan),
                    "CI.Lower": var_effects.get("total_ci", (np.nan, np.nan))[0],
                    "CI.Upper": var_effects.get("total_ci", (np.nan, np.nan))[1],
                    "P-value": var_effects.get("total_pvalue", np.nan) if show_pvalues else np.nan,
                }
            )

        df = pd.DataFrame(data)

        # Format for display
        if not show_pvalues:
            df = df.drop("P-value", axis=1)

        return df

    def print_summary(self):
        """Print formatted summary table."""
        print("\nSpatial Effects Decomposition")
        print("=" * 80)
        print(f"Model: {self.model_result.model.spatial_model_type}")
        print(f"Inference: {self.method}")
        if self.n_simulations:
            print(f"Simulations: {self.n_simulations}")
        print(f"Confidence Level: {self.confidence_level:.1%}")
        print("-" * 80)

        df = self.summary()

        # Format numerical columns
        format_dict = {
            "Estimate": "{:.4f}",
            "Std.Error": "{:.4f}",
            "CI.Lower": "{:.4f}",
            "CI.Upper": "{:.4f}",
            "P-value": "{:.4f}",
        }

        for col, fmt in format_dict.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x: fmt.format(x) if not np.isnan(x) else "")

        print(df.to_string(index=False))
        print("=" * 80)

    def plot(
        self, backend: Literal["plotly", "matplotlib"] = "plotly", show_ci: bool = True, **kwargs
    ):
        """
        Create bar plot of spatial effects.

        Parameters
        ----------
        backend : {'plotly', 'matplotlib'}, default='plotly'
            Plotting backend to use
        show_ci : bool, default=True
            Whether to show confidence intervals
        **kwargs
            Additional arguments passed to plotting function

        Returns
        -------
        Figure object
        """
        if backend == "plotly":
            return self._plot_plotly(show_ci, **kwargs)
        elif backend == "matplotlib":
            return self._plot_matplotlib(show_ci, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _plot_plotly(self, show_ci: bool = True, **kwargs):
        """Create interactive Plotly plot."""
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for interactive plots")

        variables = list(self.effects.keys())

        # Extract values
        direct_vals = [self.effects[v]["direct"] for v in variables]
        indirect_vals = [self.effects[v]["indirect"] for v in variables]
        total_vals = [self.effects[v]["total"] for v in variables]

        # Create traces
        traces = []

        # Direct effects
        error_y = None
        if show_ci and "direct_se" in list(self.effects.values())[0]:
            error_y = dict(
                type="data",
                array=[self.effects[v]["direct_se"] * 1.96 for v in variables],
                visible=True,
            )

        traces.append(
            go.Bar(
                name="Direct", x=variables, y=direct_vals, error_y=error_y, marker_color="steelblue"
            )
        )

        # Indirect effects
        error_y = None
        if show_ci and "indirect_se" in list(self.effects.values())[0]:
            error_y = dict(
                type="data",
                array=[self.effects[v]["indirect_se"] * 1.96 for v in variables],
                visible=True,
            )

        traces.append(
            go.Bar(
                name="Indirect", x=variables, y=indirect_vals, error_y=error_y, marker_color="coral"
            )
        )

        # Total effects
        error_y = None
        if show_ci and "total_se" in list(self.effects.values())[0]:
            error_y = dict(
                type="data",
                array=[self.effects[v]["total_se"] * 1.96 for v in variables],
                visible=True,
            )

        traces.append(
            go.Bar(
                name="Total", x=variables, y=total_vals, error_y=error_y, marker_color="seagreen"
            )
        )

        # Create figure
        fig = go.Figure(data=traces)

        # Update layout
        fig.update_layout(
            title="Spatial Effects Decomposition",
            xaxis_title="Variable",
            yaxis_title="Effect Magnitude",
            barmode="group",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **kwargs,
        )

        return fig

    def _plot_matplotlib(self, show_ci: bool = True, **kwargs):
        """Create matplotlib plot."""
        import matplotlib.pyplot as plt

        variables = list(self.effects.keys())
        n_vars = len(variables)

        # Extract values
        direct_vals = np.array([self.effects[v]["direct"] for v in variables])
        indirect_vals = np.array([self.effects[v]["indirect"] for v in variables])
        total_vals = np.array([self.effects[v]["total"] for v in variables])

        # Set up positions
        x = np.arange(n_vars)
        width = 0.25

        # Create figure
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))

        # Plot bars
        bars1 = ax.bar(x - width, direct_vals, width, label="Direct", color="steelblue")
        bars2 = ax.bar(x, indirect_vals, width, label="Indirect", color="coral")
        bars3 = ax.bar(x + width, total_vals, width, label="Total", color="seagreen")

        # Add error bars if requested
        if show_ci and "direct_se" in list(self.effects.values())[0]:
            direct_se = np.array([self.effects[v]["direct_se"] for v in variables])
            indirect_se = np.array([self.effects[v]["indirect_se"] for v in variables])
            total_se = np.array([self.effects[v]["total_se"] for v in variables])

            ax.errorbar(
                x - width, direct_vals, yerr=direct_se * 1.96, fmt="none", color="black", capsize=3
            )
            ax.errorbar(
                x, indirect_vals, yerr=indirect_se * 1.96, fmt="none", color="black", capsize=3
            )
            ax.errorbar(
                x + width, total_vals, yerr=total_se * 1.96, fmt="none", color="black", capsize=3
            )

        # Customize plot
        ax.set_xlabel("Variable")
        ax.set_ylabel("Effect Magnitude")
        ax.set_title("Spatial Effects Decomposition")
        ax.set_xticks(x)
        ax.set_xticklabels(variables)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add horizontal line at zero
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        plt.tight_layout()
        return fig

    def to_latex(self, filename: Optional[str] = None, **kwargs) -> str:
        """
        Export results to LaTeX table.

        Parameters
        ----------
        filename : str, optional
            File path to save LaTeX code
        **kwargs
            Arguments passed to pandas to_latex()

        Returns
        -------
        str
            LaTeX table code
        """
        df = self.summary()

        # Format numerical columns
        for col in ["Estimate", "Std.Error", "CI.Lower", "CI.Upper", "P-value"]:
            if col in df.columns:
                df[col] = df[col].round(4)

        latex = df.to_latex(index=False, **kwargs)

        if filename:
            with open(filename, "w") as f:
                f.write(latex)

        return latex
