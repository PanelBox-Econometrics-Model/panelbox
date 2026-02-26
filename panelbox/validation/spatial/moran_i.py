"""Moran's I spatial autocorrelation tests for panel data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from panelbox.core.spatial_weights import SpatialWeights

from ..base import ValidationTest, ValidationTestResult
from .utils import standardize_spatial_weights, validate_spatial_weights

logger = logging.getLogger(__name__)


@dataclass
class MoranIByPeriodResult:
    """Results container for Moran's I by period."""

    results_by_period: dict[Any, dict[str, float]]

    def summary(self) -> pd.DataFrame:
        """Create summary DataFrame."""
        return pd.DataFrame.from_dict(self.results_by_period, orient="index")

    def __str__(self) -> str:
        df = self.summary()
        return f"Moran's I by Period Results:\n{df}"


class MoranIPanelTest(ValidationTest):
    """
    Moran's I test for spatial autocorrelation in panel data.

    Tests for global spatial autocorrelation in panel data residuals or variables.
    Can compute pooled Moran's I (across all periods) or period-specific values.

    Parameters
    ----------
    residuals : array-like
        Residuals or variable to test (NT×1)
    W : np.ndarray or SpatialWeights
        Spatial weight matrix (N×N)
    entity_index : array-like
        Entity identifiers (NT×1)
    time_index : array-like
        Time identifiers (NT×1)
    method : str
        'pooled' (default) or 'period'

    Attributes
    ----------
    N : int
        Number of spatial units
    T : int
        Number of time periods
    entities : array
        Unique entity identifiers
    times : array
        Unique time identifiers
    """

    def __init__(
        self,
        residuals: np.ndarray,
        W: np.ndarray | SpatialWeights,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        method: Literal["pooled", "period"] = "pooled",
    ):
        """Initialize Moran's I test."""
        self.residuals = np.asarray(residuals).flatten()
        self.W = validate_spatial_weights(W)
        self.entity_index = np.asarray(entity_index)
        self.time_index = np.asarray(time_index)
        self.method = method

        # Extract unique entities and times
        self.entities = np.unique(entity_index)
        self.times = np.unique(time_index)
        self.N = len(self.entities)
        self.T = len(self.times)

        # Validate dimensions
        if len(self.residuals) != self.N * self.T:
            raise ValueError(f"Residuals length {len(self.residuals)} != N*T ({self.N}*{self.T})")

        if self.W.shape[0] != self.N or self.W.shape[1] != self.N:
            raise ValueError(f"W shape {self.W.shape} incompatible with N={self.N}")

    def run(
        self,
        alpha: float = 0.05,
        n_permutations: int = 999,
        inference: Literal["normal", "permutation"] = "normal",
    ) -> ValidationTestResult | MoranIByPeriodResult:
        """
        Run Moran's I test.

        Parameters
        ----------
        alpha : float
            Significance level
        n_permutations : int
            Number of permutations (if inference='permutation')
        inference : str
            'normal' (asymptotic) or 'permutation'

        Returns
        -------
        ValidationTestResult or MoranIByPeriodResult
            Test results
        """
        if self.method == "pooled":
            return self._run_pooled(alpha, n_permutations, inference)
        else:
            return self._run_by_period(alpha, inference)

    def _run_pooled(
        self, alpha: float, n_permutations: int, inference: str
    ) -> ValidationTestResult:
        """Run pooled Moran's I test."""
        I, EI, VI = self._compute_pooled_morans_i()
        z_stat = (I - EI) / np.sqrt(VI) if VI > 0 else 0

        if inference == "normal":
            pvalue = 2 * stats.norm.cdf(-np.abs(z_stat))
        else:
            pvalue = self._permutation_inference(n_permutations)

        # Interpret results
        if pvalue < alpha:
            if I > EI:
                conclusion = "Reject H0: Positive spatial autocorrelation detected"
            else:
                conclusion = "Reject H0: Negative spatial autocorrelation detected"
        else:
            conclusion = "Fail to reject H0: No significant spatial autocorrelation"

        return ValidationTestResult(
            test_name="Moran's I (Pooled)",
            statistic=I,
            pvalue=pvalue,
            null_hypothesis="No spatial autocorrelation",
            alternative_hypothesis="Spatial autocorrelation present",
            alpha=alpha,
            metadata={
                "expected_value": EI,
                "variance": VI,
                "z_statistic": z_stat,
                "method": "pooled",
                "inference": inference,
                "N": self.N,
                "T": self.T,
                "conclusion": conclusion,
            },
        )

    def _run_by_period(self, alpha: float, inference: str) -> MoranIByPeriodResult:
        """Run Moran's I test by period."""
        results_by_period = {}

        for t in self.times:
            # Extract residuals for period t
            mask = self.time_index == t

            # Create entity mapping for this period
            entities_t = self.entity_index[mask]
            residuals_t = self.residuals[mask]

            # Sort by entity to ensure correct ordering
            df_t = pd.DataFrame({"entity": entities_t, "residual": residuals_t}).sort_values(
                "entity"
            )

            # Check if we have all entities
            if len(df_t) != self.N:
                # Handle unbalanced panel - fill missing with NaN
                full_df = pd.DataFrame({"entity": self.entities})
                df_t = full_df.merge(df_t, on="entity", how="left")

            r_t = df_t["residual"].values

            # Skip if too many missing values
            if np.sum(~np.isnan(r_t)) < 3:
                continue

            I_t, EI_t, VI_t = self._compute_morans_i_period(r_t)
            z_t = (I_t - EI_t) / np.sqrt(VI_t) if VI_t > 0 else 0

            pvalue_t = 2 * stats.norm.cdf(-np.abs(z_t))

            results_by_period[t] = {
                "statistic": I_t,
                "expected_value": EI_t,
                "variance": VI_t,
                "z_statistic": z_t,
                "pvalue": pvalue_t,
            }

        return MoranIByPeriodResult(results_by_period)

    def _compute_pooled_morans_i(self) -> tuple:
        """
        Compute Moran's I for pooled panel data.

        Returns
        -------
        tuple
            (I, EI, VI) - statistic, expected value, variance
        """
        # Reshape residuals to (T, N) format
        df = pd.DataFrame(
            {"entity": self.entity_index, "time": self.time_index, "residual": self.residuals}
        )

        # Sort by time first, then entity
        df = df.sort_values(["time", "entity"])
        r = df["residual"].values
        r_mean = np.nanmean(r)

        # Compute numerator: Σt Σi Σj wij (rit - r̄)(rjt - r̄)
        numerator = 0
        denominator = 0

        for _t_idx, t in enumerate(self.times):
            # Extract residuals for period t
            mask = df["time"] == t
            r_t = df.loc[mask, "residual"].values

            if len(r_t) != self.N:
                # Handle unbalanced panel
                entity_order = df.loc[mask, "entity"].values
                r_t_full = np.full(self.N, np.nan)
                for i, entity in enumerate(entity_order):
                    entity_idx = np.where(self.entities == entity)[0][0]
                    r_t_full[entity_idx] = r_t[i]
                r_t = r_t_full

            # Center residuals
            r_t_centered = r_t - r_mean

            # Handle NaN values
            valid_mask = ~np.isnan(r_t_centered)
            if np.sum(valid_mask) < 2:
                continue

            # Compute contribution for this period
            for i in range(self.N):
                if not valid_mask[i]:
                    continue
                for j in range(self.N):
                    if not valid_mask[j]:
                        continue
                    if self.W[i, j] != 0:
                        numerator += self.W[i, j] * r_t_centered[i] * r_t_centered[j]

            denominator += np.nansum(r_t_centered**2)

        # S0: sum of all weights (considering all periods)
        S0 = self.T * self.W.sum()

        # Moran's I
        NT = self.N * self.T
        I = (NT / S0) * (numerator / denominator) if denominator > 0 else 0

        # Expected value under randomization
        EI = -1 / (NT - 1)

        # Variance (simplified formula)
        W_row_sum = self.W.sum(axis=1)
        W_col_sum = self.W.sum(axis=0)

        S1 = 0.5 * np.sum((self.W + self.W.T) ** 2)
        S2 = np.sum((W_row_sum + W_col_sum) ** 2)

        b2 = NT * np.nansum((r - r_mean) ** 4) / (np.nansum((r - r_mean) ** 2) ** 2)

        VI = self._compute_variance(NT, S0, S1, S2, b2)

        return I, EI, VI

    def _compute_morans_i_period(self, r_t: np.ndarray) -> tuple:
        """
        Compute Moran's I for a single period.

        Parameters
        ----------
        r_t : array (N,)
            Residuals for period t

        Returns
        -------
        tuple
            (I, EI, VI) - statistic, expected value, variance
        """
        # Handle missing values
        valid_mask = ~np.isnan(r_t)
        n_valid = np.sum(valid_mask)

        if n_valid < 3:
            return np.nan, np.nan, np.nan

        r_mean = np.nanmean(r_t)
        r_centered = np.where(valid_mask, r_t - r_mean, 0)

        # Compute numerator and denominator
        numerator = 0
        denominator = np.nansum(r_centered**2)

        for i in range(self.N):
            if not valid_mask[i]:
                continue
            for j in range(self.N):
                if not valid_mask[j]:
                    continue
                if self.W[i, j] != 0:
                    numerator += self.W[i, j] * r_centered[i] * r_centered[j]

        # S0 (adjusted for valid observations)
        S0 = 0
        for i in range(self.N):
            if not valid_mask[i]:
                continue
            for j in range(self.N):
                if not valid_mask[j]:
                    continue
                S0 += self.W[i, j]

        # Moran's I
        I = (n_valid / S0) * (numerator / denominator) if S0 > 0 and denominator > 0 else 0

        # Expected value
        EI = -1 / (n_valid - 1) if n_valid > 1 else 0

        # Variance (simplified)
        if n_valid > 2:
            W_valid = self.W[np.ix_(valid_mask, valid_mask)]
            S1 = 0.5 * np.sum((W_valid + W_valid.T) ** 2)
            S2 = np.sum((W_valid.sum(axis=0) + W_valid.sum(axis=1)) ** 2)

            b2 = n_valid * np.nansum(r_centered**4) / (denominator**2) if denominator > 0 else 0

            VI = self._compute_variance(n_valid, S0, S1, S2, b2)
        else:
            VI = 1.0

        return I, EI, VI

    def _compute_variance(self, n: int, S0: float, S1: float, S2: float, b2: float) -> float:
        """
        Compute variance of Moran's I under randomization.

        Uses the formula from Cliff and Ord (1981).
        """
        if n <= 1 or S0 <= 0:
            return 1.0

        n_sq = n * n
        n_1 = n - 1
        n_2 = n - 2
        n_3 = n - 3

        if n <= 3:
            return 1.0

        # First term
        term1 = n * ((n_sq - 3 * n + 3) * S1 - n * S2 + 3 * S0 * S0)

        # Second term
        term2 = b2 * ((n_sq - n) * S1 - 2 * n * S2 + 6 * S0 * S0)

        # Denominator
        denom = n_1 * n_2 * n_3 * S0 * S0

        if denom == 0:
            return 1.0

        # Variance
        VI = (term1 - term2) / denom - (1 / (n_1)) ** 2

        return max(VI, 1e-10)  # Ensure positive variance

    def _permutation_inference(self, n_permutations: int = 999) -> float:
        """
        Permutation-based p-value.

        Randomly permutes residuals across spatial units within time periods.
        """
        observed_I, _, _ = self._compute_pooled_morans_i()

        permuted_I = []
        for _ in range(n_permutations):
            # Permute residuals within each time period
            df = pd.DataFrame(
                {"entity": self.entity_index, "time": self.time_index, "residual": self.residuals}
            )

            # Permute within each time period
            df["residual_perm"] = df.groupby("time")["residual"].transform(
                lambda x: np.random.permutation(x)
            )

            # Create temporary test object with permuted data
            test_perm = MoranIPanelTest(
                df["residual_perm"].values,
                self.W,
                df["entity"].values,
                df["time"].values,
                method="pooled",
            )
            I_perm, _, _ = test_perm._compute_pooled_morans_i()
            permuted_I.append(I_perm)

        # Two-sided p-value
        permuted_I = np.array(permuted_I)
        pvalue = np.mean(np.abs(permuted_I) >= np.abs(observed_I))

        return max(pvalue, 1 / n_permutations)  # Minimum p-value

    def plot(self, backend: str = "plotly", **kwargs):
        """
        Plot Moran's I results.

        For method='period', creates time series plot.
        For method='pooled', creates Moran scatterplot.

        Parameters
        ----------
        backend : str
            'plotly' or 'matplotlib'
        **kwargs
            Additional plotting arguments
        """
        if self.method == "period":
            return self._plot_by_period(backend, **kwargs)
        else:
            return self._plot_moran_scatterplot(backend, **kwargs)

    def _plot_by_period(self, backend: str, **kwargs):
        """Plot Moran's I by period."""
        # Run test if not already run
        results = self.run()

        if not isinstance(results, MoranIByPeriodResult):
            raise ValueError("No period results available")

        # Extract statistics by period
        periods = list(results.results_by_period.keys())
        I_values = [results.results_by_period[t]["statistic"] for t in periods]
        pvalues = [results.results_by_period[t]["pvalue"] for t in periods]
        expected = [results.results_by_period[t]["expected_value"] for t in periods]

        if backend == "plotly":
            import plotly.graph_objects as go

            fig = go.Figure()

            # Moran's I line
            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=I_values,
                    mode="lines+markers",
                    name="Moran's I",
                    line={"color": "blue", "width": 2},
                    marker={"size": 8},
                )
            )

            # Expected value line
            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=expected,
                    mode="lines",
                    name="E[I] (no autocorrelation)",
                    line={"color": "gray", "width": 1, "dash": "dash"},
                )
            )

            # Highlight significant periods
            significant = [p < 0.05 for p in pvalues]
            if any(significant):
                fig.add_trace(
                    go.Scatter(
                        x=[periods[i] for i in range(len(periods)) if significant[i]],
                        y=[I_values[i] for i in range(len(periods)) if significant[i]],
                        mode="markers",
                        name="Significant (p<0.05)",
                        marker={"color": "red", "size": 12, "symbol": "star"},
                    )
                )

            fig.update_layout(
                title="Moran's I by Period",
                xaxis_title="Time Period",
                yaxis_title="Moran's I",
                hovermode="x unified",
                showlegend=True,
                template="plotly_white",
            )

            return fig

        else:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            # Moran's I line
            ax.plot(periods, I_values, "b-o", label="Moran's I", linewidth=2)

            # Expected value line
            ax.axhline(
                y=expected[0], color="gray", linestyle="--", label="E[I] (no autocorrelation)"
            )

            # Highlight significant periods
            significant = [p < 0.05 for p in pvalues]
            if any(significant):
                sig_periods = [periods[i] for i in range(len(periods)) if significant[i]]
                sig_values = [I_values[i] for i in range(len(periods)) if significant[i]]
                ax.scatter(
                    sig_periods,
                    sig_values,
                    color="red",
                    s=100,
                    marker="*",
                    label="Significant (p<0.05)",
                    zorder=5,
                )

            ax.set_xlabel("Time Period")
            ax.set_ylabel("Moran's I")
            ax.set_title("Moran's I by Period")
            ax.legend()
            ax.grid(True, alpha=0.3)

            return fig

    def _plot_moran_scatterplot(self, backend: str, **kwargs):
        """Create Moran scatterplot for pooled analysis."""
        # Compute spatial lag of residuals
        r = self.residuals
        r_mean = np.nanmean(r)
        r_std = np.nanstd(r)

        # Standardize
        z = (r - r_mean) / r_std if r_std > 0 else r - r_mean

        # Reshape to (T, N) and compute average spatial lag
        df = pd.DataFrame({"entity": self.entity_index, "time": self.time_index, "z": z})

        # Compute spatial lag for each period and average
        Wz_list = []
        for t in self.times:
            mask = df["time"] == t
            z_t = df.loc[mask, "z"].values

            if len(z_t) == self.N:
                Wz_t = standardize_spatial_weights(self.W, "row") @ z_t
                Wz_list.extend(Wz_t)
            else:
                # Handle unbalanced panel
                Wz_list.extend(np.full(len(z_t), np.nan))

        Wz = np.array(Wz_list)

        # Remove NaN values for plotting
        valid_mask = ~(np.isnan(z) | np.isnan(Wz))
        z_valid = z[valid_mask]
        Wz_valid = Wz[valid_mask]

        if backend == "plotly":
            import plotly.graph_objects as go

            fig = go.Figure()

            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=z_valid,
                    y=Wz_valid,
                    mode="markers",
                    name="Observations",
                    marker={"color": "blue", "size": 5, "opacity": 0.5},
                )
            )

            # Regression line
            if len(z_valid) > 1:
                coef = np.polyfit(z_valid, Wz_valid, 1)
                x_line = np.array([z_valid.min(), z_valid.max()])
                y_line = coef[0] * x_line + coef[1]

                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name=f"Slope = {coef[0]:.3f}",
                        line={"color": "red", "width": 2},
                    )
                )

            # Add quadrant lines
            fig.add_hline(y=0, line_color="gray", line_width=0.5)
            fig.add_vline(x=0, line_color="gray", line_width=0.5)

            fig.update_layout(
                title="Moran Scatterplot",
                xaxis_title="Standardized Values",
                yaxis_title="Spatial Lag of Standardized Values",
                showlegend=True,
                template="plotly_white",
            )

            return fig

        else:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))

            # Scatter plot
            ax.scatter(z_valid, Wz_valid, alpha=0.5, s=20)

            # Regression line
            if len(z_valid) > 1:
                coef = np.polyfit(z_valid, Wz_valid, 1)
                x_line = np.array([z_valid.min(), z_valid.max()])
                y_line = coef[0] * x_line + coef[1]
                ax.plot(x_line, y_line, "r-", linewidth=2, label=f"Slope = {coef[0]:.3f}")

            # Add quadrant lines
            ax.axhline(y=0, color="gray", linewidth=0.5)
            ax.axvline(x=0, color="gray", linewidth=0.5)

            ax.set_xlabel("Standardized Values")
            ax.set_ylabel("Spatial Lag of Standardized Values")
            ax.set_title("Moran Scatterplot")
            ax.legend()
            ax.grid(True, alpha=0.3)

            return fig
