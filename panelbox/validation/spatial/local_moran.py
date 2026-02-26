"""Local Moran's I (LISA) for spatial cluster detection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd

    from panelbox.core.spatial_weights import SpatialWeights

from .utils import standardize_spatial_weights, validate_spatial_weights

logger = logging.getLogger(__name__)


class LocalMoranI:
    """
    Local Indicators of Spatial Association (LISA).

    Computes local Moran's I statistics to identify spatial clusters
    and outliers. Each location gets its own statistic measuring
    local spatial autocorrelation.

    Parameters
    ----------
    variable : array-like
        Variable to analyze (N×1 for cross-section, NT×1 for panel)
    W : np.ndarray or SpatialWeights
        Spatial weight matrix (N×N)
    entity_index : array-like, optional
        Entity identifiers (for panel data)
    time_index : array-like, optional
        Time identifiers (for panel data, uses last period by default)
    threshold : float
        Significance threshold for cluster classification

    Attributes
    ----------
    N : int
        Number of spatial units
    cluster_types : list
        Available cluster classifications
    """

    def __init__(
        self,
        variable: np.ndarray,
        W: np.ndarray | SpatialWeights,
        entity_index: np.ndarray | None = None,
        time_index: np.ndarray | None = None,
        threshold: float = 0.05,
    ):
        """Initialize Local Moran's I."""
        self.variable = np.asarray(variable).flatten()
        self.W = validate_spatial_weights(W)
        self.threshold = threshold

        # Handle panel data
        if entity_index is not None and time_index is not None:
            self.entity_index = np.asarray(entity_index)
            self.time_index = np.asarray(time_index)
            self.is_panel = True

            # By default, use last period for LISA
            self.entities = np.unique(entity_index)
            self.times = np.unique(time_index)
            self.N = len(self.entities)

            # Extract last period data
            last_time = self.times[-1]
            mask = self.time_index == last_time

            df = pd.DataFrame(
                {"entity": self.entity_index[mask], "value": self.variable[mask]}
            ).sort_values("entity")

            # Ensure all entities present
            if len(df) != self.N:
                full_df = pd.DataFrame({"entity": self.entities})
                df = full_df.merge(df, on="entity", how="left")

            self.variable_cross = df["value"].values
            self.entity_labels = df["entity"].values
        else:
            self.is_panel = False
            self.N = len(self.variable)
            self.variable_cross = self.variable
            self.entity_labels = np.arange(self.N)

            if self.W.shape[0] != self.N:
                raise ValueError(f"W shape {self.W.shape} incompatible with N={self.N}")

        # Cluster type definitions
        self.cluster_types = ["HH", "LL", "HL", "LH", "Not significant"]

    def run(self, n_permutations: int = 999, seed: int | None = None) -> pd.DataFrame:
        """
        Compute local Moran's I for each spatial unit.

        Parameters
        ----------
        n_permutations : int
            Number of permutations for inference
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        pd.DataFrame
            Results with columns:
            - entity: entity identifier
            - value: original value
            - Ii: local Moran's I statistic
            - EIi: expected value of Ii
            - VIi: variance of Ii
            - z_score: standardized Ii
            - pvalue: p-value from conditional permutation
            - cluster_type: 'HH', 'LL', 'HL', 'LH', or 'Not significant'
        """
        if seed is not None:
            np.random.seed(seed)

        # Standardize variable
        v = self.variable_cross
        valid_mask = ~np.isnan(v)
        n_valid = np.sum(valid_mask)

        if n_valid < 3:
            # Not enough data
            return pd.DataFrame(
                {
                    "entity": self.entity_labels,
                    "value": v,
                    "Ii": np.nan,
                    "EIi": np.nan,
                    "VIi": np.nan,
                    "z_score": np.nan,
                    "pvalue": 1.0,
                    "cluster_type": "Not significant",
                }
            )

        v_mean = np.nanmean(v)
        v_std = np.nanstd(v)

        if v_std > 0:
            z = np.where(valid_mask, (v - v_mean) / v_std, 0)
        else:
            z = np.where(valid_mask, v - v_mean, 0)

        # Row-standardize W
        W_std = standardize_spatial_weights(self.W, "row")

        # Compute local Moran's I for each unit
        Ii = np.zeros(self.N)
        Wz = W_std @ z

        for i in range(self.N):
            if valid_mask[i]:
                Ii[i] = z[i] * Wz[i]
            else:
                Ii[i] = np.nan

        # Expected values and variance under randomization
        EIi = self._compute_expected_values(valid_mask)
        VIi = self._compute_variance(z, W_std, valid_mask)

        # Z-scores
        z_scores = np.where(VIi > 0, (Ii - EIi) / np.sqrt(VIi), 0)

        # Conditional permutation inference
        pvalues = self._conditional_permutation(z, W_std, Ii, n_permutations, valid_mask)

        # Classify clusters
        cluster_types = self._classify_clusters(v, v_mean, Ii, pvalues, valid_mask)

        # Create results DataFrame
        results = pd.DataFrame(
            {
                "entity": self.entity_labels,
                "value": v,
                "Ii": Ii,
                "EIi": EIi,
                "VIi": VIi,
                "z_score": z_scores,
                "pvalue": pvalues,
                "cluster_type": cluster_types,
            }
        )

        return results

    def _compute_expected_values(self, valid_mask: np.ndarray) -> np.ndarray:
        """Compute expected values of local Moran's I."""
        n_valid = np.sum(valid_mask)

        if n_valid > 1:
            # Under randomization
            EIi = -1 / (n_valid - 1)
            return np.where(valid_mask, EIi, np.nan)
        else:
            return np.full(self.N, np.nan)

    def _compute_variance(self, z: np.ndarray, W: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Compute variance of local Moran's I."""
        n_valid = np.sum(valid_mask)

        if n_valid <= 2:
            return np.full(self.N, np.nan)

        # Moments
        b2 = np.nansum(z**4) / n_valid
        s2 = np.nansum(z**2) / n_valid

        VIi = np.zeros(self.N)

        for i in range(self.N):
            if not valid_mask[i]:
                VIi[i] = np.nan
                continue

            # Wi squared sum
            wi2 = np.sum(W[i, valid_mask] ** 2)

            # Variance formula (simplified)
            term1 = (n_valid - b2 / s2**2) * wi2
            term2 = 2 * b2 / s2**2 - (n_valid - 1)
            term3 = wi2**2

            VIi[i] = term1 + term2 * term3

            # Adjust for expected value
            VIi[i] = VIi[i] / ((n_valid - 1) ** 2)

            # Ensure positive
            VIi[i] = max(VIi[i], 1e-10)

        return VIi

    def _conditional_permutation(
        self,
        z: np.ndarray,
        W: np.ndarray,
        observed_Ii: np.ndarray,
        n_permutations: int,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Conditional permutation test.

        For each location i, hold i fixed and permute its neighbors.
        """
        pvalues = np.ones(self.N)

        for i in range(self.N):
            if not valid_mask[i]:
                continue

            # Find neighbors of i
            neighbors = np.where((W[i] > 0) & valid_mask)[0]

            if len(neighbors) == 0:
                pvalues[i] = 1.0
                continue

            # Permutation test
            permuted_Ii = []

            for _ in range(n_permutations):
                # Permute neighbor values
                z_perm = z.copy()
                neighbor_values = z[neighbors].copy()
                np.random.shuffle(neighbor_values)
                z_perm[neighbors] = neighbor_values

                # Compute local Moran for i with permuted neighbors
                Wz_perm_i = W[i] @ z_perm
                Ii_perm = z[i] * Wz_perm_i
                permuted_Ii.append(Ii_perm)

            # Two-sided p-value
            permuted_Ii = np.array(permuted_Ii)

            # Handle pseudo p-value
            pvalues[i] = (np.sum(np.abs(permuted_Ii) >= np.abs(observed_Ii[i])) + 1) / (
                n_permutations + 1
            )

        return pvalues

    def _classify_clusters(
        self,
        values: np.ndarray,
        mean_value: float,
        Ii: np.ndarray,
        pvalues: np.ndarray,
        valid_mask: np.ndarray,
    ) -> list:
        """
        Classify spatial clusters.

        HH: High-High (hot spots)
        LL: Low-Low (cold spots)
        HL: High-Low (high outlier)
        LH: Low-High (low outlier)
        """
        cluster_types = []

        for i in range(self.N):
            if not valid_mask[i]:
                cluster_types.append("Not significant")
            elif pvalues[i] < self.threshold:
                # Significant local autocorrelation
                if Ii[i] > 0:
                    # Positive local autocorrelation
                    if values[i] > mean_value:
                        cluster_types.append("HH")  # High-High cluster
                    else:
                        cluster_types.append("LL")  # Low-Low cluster
                else:
                    # Negative local autocorrelation
                    if values[i] > mean_value:
                        cluster_types.append("HL")  # High-Low outlier
                    else:
                        cluster_types.append("LH")  # Low-High outlier
            else:
                cluster_types.append("Not significant")

        return cluster_types

    def plot_clusters(
        self,
        results: pd.DataFrame,
        gdf: gpd.GeoDataFrame | None = None,
        backend: str = "plotly",
        **kwargs,
    ):
        """
        Create LISA cluster map.

        Parameters
        ----------
        results : pd.DataFrame
            Results from run()
        gdf : GeoDataFrame, optional
            Geometries for spatial units
        backend : str
            'plotly' or 'matplotlib'
        **kwargs
            Additional plotting arguments

        Returns
        -------
        figure
            Plot object
        """
        # Color mapping
        color_map = {
            "HH": "#d7191c",  # Red (hot spots)
            "LL": "#2c7bb6",  # Blue (cold spots)
            "HL": "#fdae61",  # Orange (high outlier)
            "LH": "#abd9e9",  # Light blue (low outlier)
            "Not significant": "#ffffbf",  # Light yellow
        }

        if backend == "plotly":
            import plotly.graph_objects as go

            if gdf is not None:
                # Choropleth map if geometries available
                # This would require geopandas and proper geometry handling
                pass
            else:
                # Create scatter plot with cluster colors
                fig = go.Figure()

                for cluster_type in self.cluster_types:
                    mask = results["cluster_type"] == cluster_type
                    if not mask.any():
                        continue

                    fig.add_trace(
                        go.Scatter(
                            x=results.loc[mask, "entity"],
                            y=results.loc[mask, "Ii"],
                            mode="markers",
                            name=cluster_type,
                            marker={
                                "color": color_map[cluster_type],
                                "size": 10,
                                "line": {"width": 1, "color": "black"},
                            },
                        )
                    )

                fig.update_layout(
                    title="LISA Cluster Map",
                    xaxis_title="Entity",
                    yaxis_title="Local Moran I",
                    showlegend=True,
                    template="plotly_white",
                )

                return fig

        else:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            # Bar plot colored by cluster type
            entities = results["entity"].values
            Ii_values = results["Ii"].values
            colors = [color_map[ct] for ct in results["cluster_type"]]

            ax.bar(range(len(entities)), Ii_values, color=colors)

            # Add legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor=color_map[ct], label=ct)
                for ct in self.cluster_types
                if ct in results["cluster_type"].values
            ]
            ax.legend(handles=legend_elements)

            ax.set_xlabel("Entity")
            ax.set_ylabel("Local Moran I")
            ax.set_title("LISA Cluster Map")
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.grid(True, alpha=0.3)

            # Set x-axis labels if not too many
            if len(entities) <= 30:
                ax.set_xticks(range(len(entities)))
                ax.set_xticklabels(entities, rotation=45, ha="right")

            return fig

    def summary(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary statistics by cluster type.

        Parameters
        ----------
        results : pd.DataFrame
            Results from run()

        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        summary = results.groupby("cluster_type").agg(
            {"entity": "count", "Ii": ["mean", "std"], "pvalue": "mean"}
        )

        summary.columns = ["Count", "Mean_Ii", "Std_Ii", "Mean_pvalue"]
        summary["Percentage"] = summary["Count"] / len(results) * 100

        return summary.round(3)
