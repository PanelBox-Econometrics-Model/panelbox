"""
Influence diagnostics for panel data models.

This module provides comprehensive influence diagnostics including
Cook's distance, DFFITS, and DFBETAS for panel data regression models.

References
----------
Cook, R. D. (1977). Detection of influential observation in linear regression.
    Technometrics, 19(1), 15-18.
Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). Regression Diagnostics:
    Identifying Influential Data and Sources of Collinearity.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from panelbox.core.results import PanelResults


@dataclass
class InfluenceResults:
    """
    Container for influence diagnostics results.

    Attributes
    ----------
    cooks_d : pd.Series
        Cook's distance for each observation
    dffits : pd.Series
        DFFITS for each observation
    dfbetas : pd.DataFrame
        DFBETAS for each observation and parameter
    leverage : pd.Series
        Leverage values (hat values)
    standardized_residuals : pd.Series
        Standardized residuals
    """

    cooks_d: pd.Series
    dffits: pd.Series
    dfbetas: pd.DataFrame
    leverage: pd.Series
    standardized_residuals: pd.Series

    def summary(self, n_top: int = 10) -> str:
        """Generate summary of influence diagnostics."""
        lines = []
        lines.append("Influence Diagnostics Summary")
        lines.append("=" * 70)

        # Cook's D summary
        cooksd_threshold = 4 / len(self.cooks_d)
        n_influential_cooksd = (self.cooks_d > cooksd_threshold).sum()
        lines.append(
            f"Cook's Distance: {n_influential_cooksd} influential obs (threshold: {cooksd_threshold:.4f})"
        )

        # DFFITS summary
        dffits_threshold = 2 * np.sqrt(self.dfbetas.shape[1] / len(self.dffits))
        n_influential_dffits = (np.abs(self.dffits) > dffits_threshold).sum()
        lines.append(
            f"DFFITS: {n_influential_dffits} influential obs (threshold: {dffits_threshold:.4f})"
        )

        lines.append("")
        lines.append(f"Top {n_top} observations by Cook's Distance:")
        lines.append("-" * 70)
        top_cooksd = self.cooks_d.nlargest(n_top)
        for idx, val in top_cooksd.items():
            lines.append(f"  Obs {idx}: {val:.6f}")

        return "\n".join(lines)


class InfluenceDiagnostics:
    """
    Influence diagnostics for panel data models.

    Computes various influence measures to identify observations that
    have disproportionate impact on the regression results.

    Parameters
    ----------
    results : PanelResults
        Fitted model results to analyze
    verbose : bool, default=True
        Whether to print progress information

    Examples
    --------
    >>> import panelbox as pb
    >>> fe = pb.FixedEffects("y ~ x1 + x2", data, "entity", "time")
    >>> results = fe.fit()
    >>>
    >>> influence = pb.InfluenceDiagnostics(results)
    >>> infl_results = influence.compute()
    >>> print(influence.summary())
    >>>
    >>> # Get influential observations
    >>> influential = influence.influential_observations()
    """

    def __init__(self, results: PanelResults, verbose: bool = True):
        self.results = results
        self.verbose = verbose
        self.model = results._model
        self.data = self.model.data.data
        self.entity_col = self.model.data.entity_col
        self.time_col = self.model.data.time_col

        self.influence_results_: Optional[InfluenceResults] = None

    def compute(self) -> InfluenceResults:
        """
        Compute all influence diagnostics.

        Returns
        -------
        influence_results : InfluenceResults
            Complete influence diagnostics
        """
        if self.verbose:
            print("Computing influence diagnostics...")

        # Get residuals and fitted values
        residuals = self.results.resid
        n = len(residuals)
        k = len(self.results.params)

        # MSE
        mse = np.sum(residuals**2) / self.results.df_resid

        # Approximate leverage (exact would need full hat matrix)
        leverage = self._approximate_leverage()

        # Standardized residuals
        std_residuals = residuals / np.sqrt(mse * (1 - leverage))

        # Cook's Distance: (r_i^2 / (k * MSE)) * (h_i / (1 - h_i)^2)
        cooks_d = (residuals**2 / (k * mse)) * (leverage / (1 - leverage) ** 2)

        # DFFITS: r_i * sqrt(h_i / (1 - h_i))
        dffits = std_residuals * np.sqrt(leverage / (1 - leverage))

        # DFBETAS: approximate using jackknife-like calculation
        dfbetas = self._compute_dfbetas()

        # Create series with proper index
        index = pd.RangeIndex(n)

        self.influence_results_ = InfluenceResults(
            cooks_d=pd.Series(cooks_d, index=index),
            dffits=pd.Series(dffits, index=index),
            dfbetas=dfbetas,
            leverage=pd.Series(leverage, index=index),
            standardized_residuals=pd.Series(std_residuals, index=index),
        )

        if self.verbose:
            print("Influence diagnostics computed successfully")

        return self.influence_results_

    def _approximate_leverage(self) -> np.ndarray:
        """Compute approximate leverage values."""
        from patsy import dmatrix

        formula_rhs = self.results.formula.split("~")[1].strip()
        X = dmatrix(formula_rhs, self.data, return_type="dataframe")

        n = len(X)
        mean = X.mean().values

        try:
            cov = np.cov(X.values.T)
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            warnings.warn("Using pseudo-inverse for leverage")
            cov_inv = np.linalg.pinv(np.cov(X.values.T))

        diff = X.values - mean
        mahal_dist_sq = np.sum(diff @ cov_inv * diff, axis=1)
        leverage = mahal_dist_sq / (n - 1) + 1 / n

        # Ensure leverage is in valid range [0, 1]
        leverage = np.clip(leverage, 1e-10, 0.999)

        return leverage

    def _compute_dfbetas(self) -> pd.DataFrame:
        """
        Compute approximate DFBETAS.

        DFBETAS measures the change in each parameter estimate when
        observation i is deleted, scaled by the standard error.
        """
        # For computational efficiency with large panels,
        # we use an approximation based on influence
        n = len(self.results.resid)
        k = len(self.results.params)

        # Approximate DFBETAS using leverage and residuals
        leverage = self._approximate_leverage()
        residuals = self.results.resid
        mse = np.sum(residuals**2) / self.results.df_resid

        # Simplified DFBETAS approximation
        dfbetas_values = np.zeros((n, k))

        for j in range(k):
            # Approximate influence on parameter j
            dfbetas_values[:, j] = (residuals * leverage) / np.sqrt(mse)

        dfbetas_df = pd.DataFrame(dfbetas_values, columns=self.results.params.index)

        return dfbetas_df

    def influential_observations(
        self, method: str = "cooks_d", threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Identify influential observations.

        Parameters
        ----------
        method : {'cooks_d', 'dffits', 'dfbetas'}, default='cooks_d'
            Method to use for identification
        threshold : float, optional
            Custom threshold. If None, uses standard threshold for method

        Returns
        -------
        influential : pd.DataFrame
            DataFrame of influential observations
        """
        if self.influence_results_ is None:
            self.compute()

        if method == "cooks_d":
            if threshold is None:
                threshold = 4 / len(self.influence_results_.cooks_d)

            mask = self.influence_results_.cooks_d > threshold
            influential = pd.DataFrame(
                {
                    "observation": mask[mask].index,
                    "cooks_d": self.influence_results_.cooks_d[mask].values,
                    "threshold": threshold,
                }
            )

        elif method == "dffits":
            k = len(self.results.params)
            n = len(self.influence_results_.dffits)

            if threshold is None:
                threshold = 2 * np.sqrt(k / n)

            mask = np.abs(self.influence_results_.dffits) > threshold
            influential = pd.DataFrame(
                {
                    "observation": mask[mask].index,
                    "dffits": self.influence_results_.dffits[mask].values,
                    "threshold": threshold,
                }
            )

        elif method == "dfbetas":
            if threshold is None:
                threshold = 2 / np.sqrt(len(self.influence_results_.dfbetas))

            # Find observations exceeding threshold for any parameter
            mask = (np.abs(self.influence_results_.dfbetas) > threshold).any(axis=1)
            influential = pd.DataFrame(
                {
                    "observation": mask[mask].index,
                    "max_dfbetas": np.abs(self.influence_results_.dfbetas[mask]).max(axis=1).values,
                    "threshold": threshold,
                }
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        return influential

    def summary(self) -> str:
        """Generate summary of influence diagnostics."""
        if self.influence_results_ is None:
            self.compute()

        return self.influence_results_.summary()

    def plot_influence(self, save_path: Optional[str] = None):
        """
        Plot influence diagnostics.

        Parameters
        ----------
        save_path : str, optional
            Path to save plot
        """
        if self.influence_results_ is None:
            self.compute()

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Cook's Distance
        ax1 = axes[0, 0]
        ax1.stem(self.influence_results_.cooks_d.values, basefmt=" ")
        threshold = 4 / len(self.influence_results_.cooks_d)
        ax1.axhline(y=threshold, color="r", linestyle="--", label=f"Threshold ({threshold:.4f})")
        ax1.set_xlabel("Observation")
        ax1.set_ylabel("Cook's Distance")
        ax1.set_title("Cook's Distance")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: DFFITS
        ax2 = axes[0, 1]
        ax2.stem(self.influence_results_.dffits.values, basefmt=" ")
        k = len(self.results.params)
        n = len(self.influence_results_.dffits)
        threshold = 2 * np.sqrt(k / n)
        ax2.axhline(y=threshold, color="r", linestyle="--", label=f"Threshold ({threshold:.4f})")
        ax2.axhline(y=-threshold, color="r", linestyle="--")
        ax2.set_xlabel("Observation")
        ax2.set_ylabel("DFFITS")
        ax2.set_title("DFFITS")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Leverage vs Residuals
        ax3 = axes[1, 0]
        ax3.scatter(
            self.influence_results_.leverage,
            self.influence_results_.standardized_residuals,
            alpha=0.5,
            s=20,
        )
        ax3.set_xlabel("Leverage")
        ax3.set_ylabel("Standardized Residuals")
        ax3.set_title("Leverage vs Residuals")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Index plot of leverage
        ax4 = axes[1, 1]
        ax4.stem(self.influence_results_.leverage.values, basefmt=" ")
        avg_leverage = self.influence_results_.leverage.mean()
        ax4.axhline(
            y=2 * avg_leverage, color="r", linestyle="--", label=f"2 × mean ({2*avg_leverage:.4f})"
        )
        ax4.set_xlabel("Observation")
        ax4.set_ylabel("Leverage")
        ax4.set_title("Leverage Values")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.verbose:
                print(f"Plot saved to {save_path}")
        else:
            plt.show()
