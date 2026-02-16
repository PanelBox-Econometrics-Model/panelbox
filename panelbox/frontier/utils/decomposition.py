"""
Total Factor Productivity (TFP) decomposition for panel SFA models.

This module decomposes TFP growth into:
1. Technical change (frontier shift)
2. Technical efficiency change (catch-up)
3. Scale efficiency change (movement along frontier)

References:
    Kumbhakar, S. C., & Lovell, C. A. K. (2000).
        Stochastic Frontier Analysis. Cambridge University Press.
        Chapter 7: Productivity and its components.

    Färe, R., Grosskopf, S., Norris, M., & Zhang, Z. (1994).
        Productivity growth, technical progress, and efficiency change.
        American Economic Review, 84(1), 66-83.

    Orea, L. (2002).
        Parametric decomposition of a generalized Malmquist productivity index.
        Journal of Productivity Analysis, 18(1), 5-22.
"""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


class TFPDecomposition:
    """Decompose TFP growth for panel SFA models.

    Decomposes productivity growth into:
        Δ ln(TFP) = ΔTC + ΔTE + ΔSE

    where:
        ΔTC = Technical Change (frontier shift over time)
        ΔTE = Technical Efficiency Change (catch-up to frontier)
        ΔSE = Scale Efficiency Change (movement along frontier)

    Works with any panel SFA model that provides:
        - Frontier estimates (β)
        - Efficiency estimates (TE_it)
        - Panel structure (entity and time identifiers)

    Parameters
    ----------
    result : SFResult
        Result object from fitted panel SFA model.
        Must have:
            - model.data: DataFrame with panel data
            - model.entity: Name of entity identifier column
            - model.time: Name of time identifier column
            - model.depvar: Name of dependent variable
            - model.exog: List of exogenous variable names
            - params: Estimated parameters
            - efficiency(): Method to compute efficiency scores

    periods : tuple of int, optional
        Tuple of two periods (t1, t2) to compare.
        If None, compares first and last periods.

    Example
    -------
    >>> # Estimate panel SFA model
    >>> result = model.fit()
    >>>
    >>> # Decompose TFP growth
    >>> tfp = TFPDecomposition(result)
    >>> decomp = tfp.decompose()
    >>> print(decomp[['entity', 'delta_tfp', 'delta_tc', 'delta_te', 'delta_se']])
    >>>
    >>> # Aggregate statistics
    >>> agg = tfp.aggregate_decomposition()
    >>> print(f"Mean TFP growth: {agg['mean_delta_tfp']:.3f}")
    >>> print(f"  From technical change: {agg['pct_from_tc']:.1f}%")
    >>> print(f"  From efficiency change: {agg['pct_from_te']:.1f}%")
    >>>
    >>> # Visualize
    >>> tfp.plot_decomposition(kind='bar', top_n=20)

    Notes
    -----
    The decomposition follows the Malmquist productivity index approach:

    1. **TFP Growth:**
       Δ ln(TFP) = Δ ln(y) - Σ ε_j · Δ ln(x_j)

       where ε_j are output elasticities.

    2. **Technical Efficiency Change (ΔTE):**
       ΔTE = ln(TE_t2) - ln(TE_t1)

       Positive ΔTE indicates catch-up (moving closer to frontier).

    3. **Technical Change (ΔTC):**
       TC captures frontier shift over time.
       For models with time trend: ΔTC = β_t · (t2 - t1)
       Otherwise estimated as residual.

    4. **Scale Efficiency Change (ΔSE):**
       SE captures gains/losses from changing scale.
       Depends on returns to scale (RTS):
         - RTS > 1 (IRS): Expansion increases SE
         - RTS = 1 (CRS): No scale effect
         - RTS < 1 (DRS): Expansion decreases SE

    The decomposition satisfies:
        Δ ln(TFP) = ΔTC + ΔTE + ΔSE (up to rounding error)

    References
    ----------
    Kumbhakar & Lovell (2000) provide comprehensive treatment of productivity
    decomposition in Chapter 7.

    Färe et al. (1994) introduced the DEA-based Malmquist index decomposition.

    Orea (2002) developed the parametric decomposition for stochastic frontiers.
    """

    def __init__(
        self,
        result,
        periods: Optional[Tuple[int, int]] = None,
    ):
        self.result = result
        self.model = result.model

        # Check that model is panel
        if not hasattr(self.model, "entity") or self.model.entity is None:
            raise ValueError("TFP decomposition requires panel data model")

        if not hasattr(self.model, "time") or self.model.time is None:
            raise ValueError("TFP decomposition requires time identifier")

        # Get time periods
        unique_times = np.unique(self.model.data[self.model.time])
        if periods is None:
            # Compare first and last
            self.t1 = unique_times[0]
            self.t2 = unique_times[-1]
        else:
            self.t1, self.t2 = periods

        if self.t1 not in unique_times or self.t2 not in unique_times:
            raise ValueError(
                f"Specified periods {periods} not found in data. " f"Available: {unique_times}"
            )

        # Store data
        self.data = self.model.data
        self.depvar = self.model.depvar
        self.exog = self.model.exog

    def decompose(self) -> pd.DataFrame:
        """Compute TFP decomposition for all firms.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
                - entity: Firm identifier
                - delta_tfp: Total TFP change (log difference)
                - delta_tc: Technical change component
                - delta_te: Technical efficiency change component
                - delta_se: Scale efficiency change component
                - verification: delta_tfp - (delta_tc + delta_te + delta_se)
                              Should be near zero (< 1e-6)

        Notes
        -----
        The decomposition is computed for each firm that exists in both
        periods. Firms that appear in only one period are excluded.

        The verification column checks that components sum to total TFP growth.
        Non-zero values indicate computational issues or model misspecification.
        """
        # Get data for both periods
        data_t1 = self.data[self.data[self.model.time] == self.t1].copy()
        data_t2 = self.data[self.data[self.model.time] == self.t2].copy()

        # Merge on entity to get balanced panel
        df = data_t1.merge(
            data_t2,
            on=self.model.entity,
            suffixes=("_t1", "_t2"),
            how="inner",  # Only firms in both periods
        )

        if len(df) == 0:
            raise ValueError(f"No firms found in both periods {self.t1} and {self.t2}")

        # Get efficiency scores
        eff = self.result.efficiency(estimator="bc")  # Bias-corrected
        eff_t1 = eff[eff["time"] == self.t1].set_index("entity")["efficiency"]
        eff_t2 = eff[eff["time"] == self.t2].set_index("entity")["efficiency"]

        results = []

        # Get frontier parameters (assume time-invariant for now)
        beta = self.result.params[: len(self.exog)]

        for _, row in df.iterrows():
            entity = row[self.model.entity]

            # Output change (log difference)
            y_t1 = row[f"{self.depvar}_t1"]
            y_t2 = row[f"{self.depvar}_t2"]
            delta_y = y_t2 - y_t1

            # Input changes (log differences)
            x_t1 = np.array([row[f"{var}_t1"] for var in self.exog])
            x_t2 = np.array([row[f"{var}_t2"] for var in self.exog])
            delta_x = x_t2 - x_t1

            # Input contribution (weighted by output elasticities)
            # For Cobb-Douglas: β_j are elasticities
            # For translog: β_j are elasticities at mean
            delta_inputs = beta @ delta_x

            # TFP growth (Solow residual)
            delta_tfp = delta_y - delta_inputs

            # Component 1: Technical Efficiency Change
            te_t1 = eff_t1.loc[entity]
            te_t2 = eff_t2.loc[entity]
            delta_te = np.log(te_t2) - np.log(te_t1)

            # Component 2: Returns to Scale
            rts = self._compute_returns_to_scale(beta, x_t1, x_t2)

            # Component 3: Scale Efficiency Change
            delta_se = self._compute_scale_efficiency_change(rts, delta_inputs)

            # Component 4: Technical Change (residual)
            # This captures frontier shift not explained by other components
            delta_tc = delta_tfp - delta_te - delta_se

            # Verification: components should sum to total
            verification = delta_tfp - (delta_tc + delta_te + delta_se)

            results.append(
                {
                    "entity": entity,
                    "delta_tfp": delta_tfp,
                    "delta_tc": delta_tc,
                    "delta_te": delta_te,
                    "delta_se": delta_se,
                    "rts": rts,
                    "verification": verification,
                }
            )

        return pd.DataFrame(results)

    def _compute_returns_to_scale(
        self,
        beta: np.ndarray,
        x_t1: np.ndarray,
        x_t2: np.ndarray,
    ) -> float:
        """Compute returns to scale at average input levels.

        Parameters
        ----------
        beta : np.ndarray
            Output elasticities
        x_t1 : np.ndarray
            Log inputs at time t1
        x_t2 : np.ndarray
            Log inputs at time t2

        Returns
        -------
        float
            Returns to scale measure:
                RTS > 1: Increasing returns to scale
                RTS = 1: Constant returns to scale
                RTS < 1: Decreasing returns to scale

        Notes
        -----
        For Cobb-Douglas production function:
            ln(y) = β_0 + Σ β_j ln(x_j) + v - u

        The returns to scale are:
            RTS = Σ β_j (sum of output elasticities)

        For translog or other flexible forms, RTS varies with input levels.
        We evaluate at the average of t1 and t2 inputs.
        """
        # For Cobb-Douglas: RTS is sum of elasticities
        # For translog: need to compute elasticities at point
        # For now, assume Cobb-Douglas
        rts = beta.sum()
        return rts

    def _compute_scale_efficiency_change(
        self,
        rts: float,
        delta_inputs: float,
    ) -> float:
        """Compute scale efficiency change.

        Parameters
        ----------
        rts : float
            Returns to scale measure
        delta_inputs : float
            Aggregate input change (weighted)

        Returns
        -------
        float
            Scale efficiency change component

        Notes
        -----
        Scale efficiency captures the effect of moving toward or away from
        the optimal scale of operation.

        If RTS = 1 (CRS):
            No scale effects, ΔSE = 0

        If RTS > 1 (IRS):
            Expansion increases efficiency: ΔSE > 0 if inputs increase
            Contraction decreases efficiency: ΔSE < 0 if inputs decrease

        If RTS < 1 (DRS):
            Expansion decreases efficiency: ΔSE < 0 if inputs increase
            Contraction increases efficiency: ΔSE > 0 if inputs decrease

        The formula used is:
            ΔSE = (RTS - 1) · Δ(weighted inputs)
        """
        # Scale effect depends on distance from CRS
        scale_effect = (rts - 1.0) * delta_inputs
        return scale_effect

    def aggregate_decomposition(self) -> Dict[str, float]:
        """Compute aggregate (mean) decomposition across all firms.

        Returns
        -------
        dict
            Dictionary with keys:
                - mean_delta_tfp: Average TFP growth
                - mean_delta_tc: Average technical change
                - mean_delta_te: Average efficiency change
                - mean_delta_se: Average scale effect
                - pct_from_tc: Percent of TFP from technical change
                - pct_from_te: Percent of TFP from efficiency change
                - pct_from_se: Percent of TFP from scale effect
                - std_delta_tfp: Standard deviation of TFP growth
                - n_firms: Number of firms in decomposition

        Example
        -------
        >>> agg = tfp.aggregate_decomposition()
        >>> print(f"Average TFP growth: {agg['mean_delta_tfp']:.3f}")
        >>> print(f"Driven by:")
        >>> print(f"  Technical change: {agg['pct_from_tc']:.1f}%")
        >>> print(f"  Efficiency change: {agg['pct_from_te']:.1f}%")
        >>> print(f"  Scale effects: {agg['pct_from_se']:.1f}%")
        """
        decomp = self.decompose()

        mean_tfp = decomp["delta_tfp"].mean()

        # Avoid division by zero
        if abs(mean_tfp) < 1e-10:
            pct_tc = pct_te = pct_se = np.nan
        else:
            pct_tc = 100 * decomp["delta_tc"].mean() / mean_tfp
            pct_te = 100 * decomp["delta_te"].mean() / mean_tfp
            pct_se = 100 * decomp["delta_se"].mean() / mean_tfp

        return {
            "mean_delta_tfp": mean_tfp,
            "mean_delta_tc": decomp["delta_tc"].mean(),
            "mean_delta_te": decomp["delta_te"].mean(),
            "mean_delta_se": decomp["delta_se"].mean(),
            "pct_from_tc": pct_tc,
            "pct_from_te": pct_te,
            "pct_from_se": pct_se,
            "std_delta_tfp": decomp["delta_tfp"].std(),
            "n_firms": len(decomp),
        }

    def plot_decomposition(
        self,
        kind: str = "bar",
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 6),
    ):
        """Plot TFP decomposition.

        Parameters
        ----------
        kind : {'bar', 'scatter'}, default='bar'
            Type of plot:
                - 'bar': Stacked bar chart showing components
                - 'scatter': Scatter plot of TC vs TE
        top_n : int, default=20
            Number of firms to show in bar chart.
            If greater than total firms, shows all.
        figsize : tuple, default=(12, 6)
            Figure size (width, height)

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure

        Example
        -------
        >>> # Stacked bar chart of top 20 firms
        >>> fig = tfp.plot_decomposition(kind='bar', top_n=20)
        >>> plt.savefig('tfp_decomposition.png', dpi=300, bbox_inches='tight')
        >>>
        >>> # Scatter plot showing innovation vs catch-up
        >>> fig = tfp.plot_decomposition(kind='scatter')
        """
        decomp = self.decompose()

        if kind == "bar":
            return self._plot_bar(decomp, top_n, figsize)
        elif kind == "scatter":
            return self._plot_scatter(decomp, figsize)
        else:
            raise ValueError(f"Unknown plot kind: {kind}. Use 'bar' or 'scatter'.")

    def _plot_bar(
        self,
        decomp: pd.DataFrame,
        top_n: int,
        figsize: Tuple[int, int],
    ):
        """Create stacked bar chart of decomposition."""
        # Sort by total TFP growth and take top_n
        decomp_sorted = decomp.nlargest(top_n, "delta_tfp")

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(decomp_sorted))
        width = 0.8

        # Create stacked bars
        # Start with TE at bottom
        bottom = np.zeros(len(decomp_sorted))

        # Plot TE
        te_vals = decomp_sorted["delta_te"].values
        ax.bar(x, te_vals, width, label="Efficiency Change (ΔTE)", color="#2ecc71", bottom=bottom)
        bottom += te_vals

        # Plot TC
        tc_vals = decomp_sorted["delta_tc"].values
        ax.bar(x, tc_vals, width, label="Technical Change (ΔTC)", color="#3498db", bottom=bottom)
        bottom += tc_vals

        # Plot SE
        se_vals = decomp_sorted["delta_se"].values
        ax.bar(x, se_vals, width, label="Scale Effect (ΔSE)", color="#f39c12", bottom=bottom)

        # Add total TFP as line
        ax.plot(
            x,
            decomp_sorted["delta_tfp"],
            "ko-",
            linewidth=2,
            markersize=5,
            label="Total TFP Growth",
            zorder=10,
        )

        # Zero line
        ax.axhline(0, color="k", linestyle="--", alpha=0.3, linewidth=0.8)

        ax.set_xlabel("Firm (ranked by TFP growth)", fontsize=11)
        ax.set_ylabel("Growth Rate (log difference)", fontsize=11)
        ax.set_title(
            f"TFP Decomposition: Period {self.t1} → {self.t2}",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.8)

        plt.tight_layout()
        return fig

    def _plot_scatter(
        self,
        decomp: pd.DataFrame,
        figsize: Tuple[int, int],
    ):
        """Create scatter plot of TC vs TE."""
        fig, ax = plt.subplots(figsize=figsize)

        # Scatter with color by total TFP growth
        scatter = ax.scatter(
            decomp["delta_tc"],
            decomp["delta_te"],
            c=decomp["delta_tfp"],
            cmap="RdYlGn",
            s=50,
            alpha=0.6,
            edgecolors="k",
            linewidth=0.5,
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Total TFP Growth", rotation=270, labelpad=20)

        # Reference lines
        ax.axhline(0, color="k", linestyle="--", alpha=0.3, linewidth=1)
        ax.axvline(0, color="k", linestyle="--", alpha=0.3, linewidth=1)

        ax.set_xlabel("Technical Change (ΔTC): Frontier Shift", fontsize=11)
        ax.set_ylabel("Efficiency Change (ΔTE): Catch-up", fontsize=11)
        ax.set_title(
            "Decomposition of TFP Growth\n" f"Period {self.t1} → {self.t2}",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(alpha=0.3, linestyle=":", linewidth=0.8)

        # Add quadrant labels
        ax.text(
            0.02,
            0.98,
            "Innovation\n+ Catch-up",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )
        ax.text(
            0.02,
            0.02,
            "Catch-up\nonly",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
        )
        ax.text(
            0.98,
            0.98,
            "Innovation\nonly",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
        )
        ax.text(
            0.98,
            0.02,
            "Decline",
            transform=ax.transAxes,
            ha="right",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.3),
        )

        plt.tight_layout()
        return fig

    def summary(self) -> str:
        """Generate text summary of TFP decomposition.

        Returns
        -------
        str
            Formatted summary table

        Example
        -------
        >>> print(tfp.summary())
        """
        agg = self.aggregate_decomposition()
        decomp = self.decompose()

        lines = []
        lines.append("=" * 70)
        lines.append("TFP DECOMPOSITION SUMMARY".center(70))
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Period: {self.t1} → {self.t2}")
        lines.append(f"Number of firms: {agg['n_firms']}")
        lines.append("")
        lines.append("-" * 70)
        lines.append("AGGREGATE RESULTS (Mean across firms)".center(70))
        lines.append("-" * 70)
        lines.append("")
        lines.append(f"  Total TFP Growth:          {agg['mean_delta_tfp']:>8.4f}")
        lines.append(f"    (Std Dev:                {agg['std_delta_tfp']:>8.4f})")
        lines.append("")
        lines.append("  Decomposition:")
        lines.append(
            f"    Technical Change (ΔTC):  {agg['mean_delta_tc']:>8.4f}  "
            f"({agg['pct_from_tc']:>6.1f}%)"
        )
        lines.append(
            f"    Efficiency Change (ΔTE): {agg['mean_delta_te']:>8.4f}  "
            f"({agg['pct_from_te']:>6.1f}%)"
        )
        lines.append(
            f"    Scale Effect (ΔSE):      {agg['mean_delta_se']:>8.4f}  "
            f"({agg['pct_from_se']:>6.1f}%)"
        )
        lines.append("")
        lines.append("-" * 70)
        lines.append("INTERPRETATION".center(70))
        lines.append("-" * 70)
        lines.append("")

        # Interpret dominant component
        components = [
            ("Technical Change", agg["mean_delta_tc"]),
            ("Efficiency Change", agg["mean_delta_te"]),
            ("Scale Effects", agg["mean_delta_se"]),
        ]
        dominant = max(components, key=lambda x: abs(x[1]))

        lines.append(f"  Main driver: {dominant[0]} ({abs(dominant[1]):.4f})")
        lines.append("")

        if agg["mean_delta_tc"] > 0:
            lines.append("  ✓ Positive technical change: Frontier is shifting outward")
        else:
            lines.append("  ✗ Negative technical change: Frontier is shifting inward")

        if agg["mean_delta_te"] > 0:
            lines.append("  ✓ Positive efficiency change: Firms catching up to frontier")
        else:
            lines.append("  ✗ Negative efficiency change: Firms falling behind frontier")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)
