"""
Forecasting functionality for Panel VAR models.

This module provides classes and functions for generating h-step-ahead forecasts
from Panel VAR models, including confidence intervals via bootstrap or analytical methods.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ForecastResult:
    """
    Container for Panel VAR forecast results.

    This class stores forecasts, confidence intervals, and provides methods
    for visualization and evaluation.

    Parameters
    ----------
    forecasts : np.ndarray
        Forecast values, shape (steps, N, K) or (steps, K) for single entity
    ci_lower : np.ndarray, optional
        Lower confidence interval bounds, same shape as forecasts
    ci_upper : np.ndarray, optional
        Upper confidence interval bounds, same shape as forecasts
    endog_names : List[str]
        Names of endogenous variables
    entity_names : List[str], optional
        Names of entities
    horizon : int
        Forecast horizon (number of steps)
    ci_level : float
        Confidence level for intervals (e.g., 0.95)
    method : str
        Forecast method ('iterative', 'direct')
    ci_method : str, optional
        CI method ('bootstrap', 'analytical', None)

    Attributes
    ----------
    forecasts : np.ndarray
        Point forecasts
    ci_lower : np.ndarray
        Lower CI bounds
    ci_upper : np.ndarray
        Upper CI bounds
    K : int
        Number of variables
    N : int
        Number of entities
    horizon : int
        Forecast horizon
    """

    def __init__(
        self,
        forecasts: np.ndarray,
        ci_lower: Optional[np.ndarray] = None,
        ci_upper: Optional[np.ndarray] = None,
        endog_names: Optional[List[str]] = None,
        entity_names: Optional[List[str]] = None,
        horizon: Optional[int] = None,
        ci_level: float = 0.95,
        method: str = "iterative",
        ci_method: Optional[str] = None,
    ):
        self.forecasts = forecasts
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.ci_level = ci_level
        self.method = method
        self.ci_method = ci_method

        # Determine dimensions
        if forecasts.ndim == 2:
            # (steps, K) - single entity
            self.horizon, self.K = forecasts.shape
            self.N = 1
            self.forecasts = forecasts[:, np.newaxis, :]  # (steps, 1, K)
            if ci_lower is not None:
                self.ci_lower = ci_lower[:, np.newaxis, :]
            if ci_upper is not None:
                self.ci_upper = ci_upper[:, np.newaxis, :]
        else:
            # (steps, N, K)
            self.horizon, self.N, self.K = forecasts.shape

        if horizon is not None and horizon != self.horizon:
            raise ValueError(
                f"horizon={horizon} inconsistent with forecasts.shape[0]={self.horizon}"
            )

        # Names
        self.endog_names = endog_names or [f"y{k}" for k in range(self.K)]
        self.entity_names = entity_names or [f"entity_{i}" for i in range(self.N)]

        if len(self.endog_names) != self.K:
            raise ValueError(f"len(endog_names)={len(self.endog_names)} != K={self.K}")
        if len(self.entity_names) != self.N:
            raise ValueError(f"len(entity_names)={len(self.entity_names)} != N={self.N}")

    def to_dataframe(
        self, entity: Optional[Union[int, str]] = None, variable: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Convert forecasts to DataFrame.

        Parameters
        ----------
        entity : int or str, optional
            Entity index or name. If None, returns all entities (MultiIndex).
        variable : str, optional
            Variable name. If None, returns all variables.

        Returns
        -------
        pd.DataFrame
            Forecasts as DataFrame with columns for point forecast and CIs.
        """
        # Determine entity index
        if entity is None:
            entity_indices = list(range(self.N))
        elif isinstance(entity, str):
            entity_indices = [self.entity_names.index(entity)]
        else:
            entity_indices = [entity]

        # Determine variable index
        if variable is None:
            var_indices = list(range(self.K))
        else:
            var_indices = [self.endog_names.index(variable)]

        # If single entity, return wide format with variables as columns
        if len(entity_indices) == 1 and variable is None:
            # Wide format: horizon as index, variables as columns
            data = {}
            entity_idx = entity_indices[0]

            # Add forecast columns for each variable
            for k in var_indices:
                var_name = self.endog_names[k]
                data[var_name] = self.forecasts[:, entity_idx, k]

                # Add CI columns if available
                if self.ci_lower is not None:
                    data[f"{var_name}_ci_lower"] = self.ci_lower[:, entity_idx, k]
                if self.ci_upper is not None:
                    data[f"{var_name}_ci_upper"] = self.ci_upper[:, entity_idx, k]

            df = pd.DataFrame(data, index=np.arange(1, self.horizon + 1))
            df.index.name = "horizon"
            return df

        # Otherwise, use long format
        data = []
        for h in range(self.horizon):
            for i in entity_indices:
                for k in var_indices:
                    row = {
                        "horizon": h + 1,
                        "entity": self.entity_names[i],
                        "variable": self.endog_names[k],
                        "forecast": self.forecasts[h, i, k],
                    }
                    if self.ci_lower is not None:
                        row["ci_lower"] = self.ci_lower[h, i, k]
                    if self.ci_upper is not None:
                        row["ci_upper"] = self.ci_upper[h, i, k]
                    data.append(row)

        df = pd.DataFrame(data)

        # Set index
        if len(entity_indices) > 1 or len(var_indices) > 1:
            index_cols = []
            if len(entity_indices) > 1:
                index_cols.append("entity")
            if len(var_indices) > 1:
                index_cols.append("variable")
            index_cols.append("horizon")
            df = df.set_index(index_cols)

        return df

    def plot(
        self,
        entity: Union[int, str],
        variable: str,
        actual: Optional[np.ndarray] = None,
        backend: str = "plotly",
        show: bool = True,
        **kwargs,
    ):
        """
        Plot forecast with confidence intervals.

        Parameters
        ----------
        entity : int or str
            Entity index or name to plot
        variable : str
            Variable name to plot
        actual : np.ndarray, optional
            Actual values for comparison (for out-of-sample evaluation).
            Shape: (n_periods,) where n_periods >= horizon
        backend : str, default='plotly'
            Plotting backend: 'plotly' or 'matplotlib'
        show : bool, default=True
            Whether to display the plot
        **kwargs
            Additional arguments passed to plotting function

        Returns
        -------
        fig
            Plotly or Matplotlib figure object
        """
        # Get entity and variable indices
        if isinstance(entity, str):
            entity_idx = self.entity_names.index(entity)
            entity_name = entity
        else:
            entity_idx = entity
            entity_name = self.entity_names[entity_idx]

        var_idx = self.endog_names.index(variable)

        # Extract forecasts
        fcst = self.forecasts[:, entity_idx, var_idx]

        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError(
                    "plotly is required for plotly backend. Install with: pip install plotly"
                )
            return self._plot_plotly(
                fcst, entity_name, variable, var_idx, entity_idx, actual, show, **kwargs
            )
        else:
            if not HAS_MATPLOTLIB:
                raise ImportError(
                    "matplotlib is required for matplotlib backend. Install with: pip install matplotlib"
                )
            return self._plot_matplotlib(
                fcst, entity_name, variable, var_idx, entity_idx, actual, show, **kwargs
            )

    def _plot_plotly(
        self,
        fcst: np.ndarray,
        entity_name: str,
        variable: str,
        var_idx: int,
        entity_idx: int,
        actual: Optional[np.ndarray],
        show: bool,
        **kwargs,
    ):
        """Plot using Plotly."""
        fig = go.Figure()

        horizons = np.arange(1, self.horizon + 1)

        # Forecast line
        fig.add_trace(
            go.Scatter(
                x=horizons,
                y=fcst,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="blue", dash="dash"),
            )
        )

        # Confidence intervals
        if self.ci_lower is not None and self.ci_upper is not None:
            ci_lower = self.ci_lower[:, entity_idx, var_idx]
            ci_upper = self.ci_upper[:, entity_idx, var_idx]

            # Upper bound
            fig.add_trace(
                go.Scatter(
                    x=horizons,
                    y=ci_upper,
                    mode="lines",
                    name=f"{int(self.ci_level*100)}% CI Upper",
                    line=dict(color="lightblue", width=1),
                    showlegend=True,
                )
            )

            # Lower bound with fill
            fig.add_trace(
                go.Scatter(
                    x=horizons,
                    y=ci_lower,
                    mode="lines",
                    name=f"{int(self.ci_level*100)}% CI Lower",
                    line=dict(color="lightblue", width=1),
                    fill="tonexty",
                    fillcolor="rgba(173, 216, 230, 0.3)",
                    showlegend=True,
                )
            )

        # Actual values if provided
        if actual is not None:
            actual_subset = actual[: self.horizon]
            fig.add_trace(
                go.Scatter(
                    x=horizons[: len(actual_subset)],
                    y=actual_subset,
                    mode="lines+markers",
                    name="Actual",
                    line=dict(color="black"),
                )
            )

        fig.update_layout(
            title=f"Forecast: {variable} for {entity_name}",
            xaxis_title="Horizon",
            yaxis_title=variable,
            hovermode="x unified",
            **kwargs,
        )

        if show:
            fig.show()

        return fig

    def _plot_matplotlib(
        self,
        fcst: np.ndarray,
        entity_name: str,
        variable: str,
        var_idx: int,
        entity_idx: int,
        actual: Optional[np.ndarray],
        show: bool,
        figsize: Tuple[int, int] = (10, 6),
        **kwargs,
    ):
        """Plot using Matplotlib."""
        fig, ax = plt.subplots(figsize=figsize)

        horizons = np.arange(1, self.horizon + 1)

        # Forecast line
        ax.plot(horizons, fcst, "b--", marker="o", label="Forecast")

        # Confidence intervals
        if self.ci_lower is not None and self.ci_upper is not None:
            ci_lower = self.ci_lower[:, entity_idx, var_idx]
            ci_upper = self.ci_upper[:, entity_idx, var_idx]
            ax.fill_between(
                horizons,
                ci_lower,
                ci_upper,
                alpha=0.3,
                color="lightblue",
                label=f"{int(self.ci_level*100)}% CI",
            )

        # Actual values if provided
        if actual is not None:
            actual_subset = actual[: self.horizon]
            ax.plot(
                horizons[: len(actual_subset)],
                actual_subset,
                "k-",
                marker="s",
                label="Actual",
            )

        ax.set_xlabel("Horizon")
        ax.set_ylabel(variable)
        ax.set_title(f"Forecast: {variable} for {entity_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()

        return fig

    def evaluate(
        self, actual: Union[np.ndarray, pd.DataFrame], entity: Optional[Union[int, str]] = None
    ) -> pd.DataFrame:
        """
        Evaluate forecast accuracy against actual values.

        Parameters
        ----------
        actual : np.ndarray or pd.DataFrame
            Actual values. Can be:
            - np.ndarray with shape (horizon, N, K) or (horizon, K) for single entity
            - pd.DataFrame with endog variables as columns
        entity : int or str, optional
            Specific entity to evaluate. If None, evaluates all entities.

        Returns
        -------
        pd.DataFrame
            Forecast accuracy metrics (RMSE, MAE, MAPE) by variable and entity
        """
        # Convert DataFrame to numpy array if needed
        if isinstance(actual, pd.DataFrame):
            # Extract just the endogenous variables
            actual_values = actual[self.endog_names].values  # (n_rows, K)
            # Reshape to (horizon, N, K)
            # Assume data is sorted by entity then time
            n_entities = len(actual.groupby(actual.columns[0]))  # first column is entity
            actual = actual_values.reshape((-1, n_entities, self.K))

        # Ensure actual has correct shape
        if actual.ndim == 2:
            actual = actual[:, np.newaxis, :]  # (horizon, 1, K)

        if actual.shape[0] < self.horizon:
            raise ValueError(f"actual has {actual.shape[0]} periods, need at least {self.horizon}")

        # Truncate to horizon
        actual = actual[: self.horizon, :, :]

        # Determine entities to evaluate
        if entity is None:
            entity_indices = list(range(self.N))
        elif isinstance(entity, str):
            entity_indices = [self.entity_names.index(entity)]
        else:
            entity_indices = [entity]

        # Compute metrics
        results = []
        for i in entity_indices:
            for k in range(self.K):
                fcst = self.forecasts[:, i, k]
                act = actual[:, i, k]

                # Remove any NaN values
                valid_mask = ~(np.isnan(fcst) | np.isnan(act))
                if not valid_mask.any():
                    continue

                fcst_valid = fcst[valid_mask]
                act_valid = act[valid_mask]

                errors = fcst_valid - act_valid
                abs_errors = np.abs(errors)
                sq_errors = errors**2

                rmse = np.sqrt(np.mean(sq_errors))
                mae = np.mean(abs_errors)

                # MAPE (handle division by zero)
                with np.errstate(divide="ignore", invalid="ignore"):
                    pct_errors = abs_errors / np.abs(act_valid)
                    pct_errors = pct_errors[np.isfinite(pct_errors)]
                    mape = np.mean(pct_errors) * 100 if len(pct_errors) > 0 else np.nan

                results.append(
                    {
                        "entity": self.entity_names[i],
                        "variable": self.endog_names[k],
                        "RMSE": rmse,
                        "MAE": mae,
                        "MAPE": mape,
                    }
                )

        df_results = pd.DataFrame(results)

        # If single entity and no entity filter, aggregate across all entities/variables
        if entity is None and len(results) > 1:
            # Return average metrics across all entities and variables
            return df_results[["RMSE", "MAE", "MAPE"]].mean()

        return df_results

    def summary(self) -> str:
        """
        Generate summary of forecast results.

        Returns
        -------
        str
            Formatted summary string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("Panel VAR Forecast Results")
        lines.append("=" * 70)
        lines.append(f"Number of entities: {self.N}")
        lines.append(f"Number of variables: {self.K}")
        lines.append(f"Forecast horizon: {self.horizon}")
        lines.append(f"Forecast method: {self.method}")

        if self.ci_lower is not None:
            lines.append(f"Confidence level: {self.ci_level:.1%}")
            lines.append(f"CI method: {self.ci_method}")

        lines.append("")
        lines.append("Variables: " + ", ".join(self.endog_names))
        if self.N <= 10:
            lines.append("Entities: " + ", ".join(self.entity_names))
        else:
            lines.append(f"Entities: {self.entity_names[0]}, ..., {self.entity_names[-1]}")

        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ForecastResult(horizon={self.horizon}, N={self.N}, K={self.K}, "
            f"method='{self.method}', has_ci={self.ci_lower is not None})"
        )
