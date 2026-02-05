"""
Cluster-robust standard errors for panel data.

This module implements one-way and two-way cluster-robust covariance
estimators commonly used in panel data applications.
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .utils import (
    compute_bread,
    compute_clustered_meat,
    compute_twoway_clustered_meat,
    sandwich_covariance,
    clustered_covariance,
    twoway_clustered_covariance
)


@dataclass
class ClusteredCovarianceResult:
    """
    Result of cluster-robust covariance estimation.

    Attributes
    ----------
    cov_matrix : np.ndarray
        Cluster-robust covariance matrix (k x k)
    std_errors : np.ndarray
        Cluster-robust standard errors (k,)
    n_clusters : int or tuple
        Number of clusters (or tuple for two-way)
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters
    cluster_dims : int
        Number of clustering dimensions (1 or 2)
    df_correction : bool
        Whether finite-sample correction was applied
    """
    cov_matrix: np.ndarray
    std_errors: np.ndarray
    n_clusters: Union[int, tuple]
    n_obs: int
    n_params: int
    cluster_dims: int
    df_correction: bool


class ClusteredStandardErrors:
    """
    Cluster-robust standard errors for panel data.

    Implements one-way and two-way clustering with finite-sample corrections.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray or tuple of np.ndarray
        Cluster identifiers. Can be:
        - 1D array for one-way clustering
        - Tuple of two 1D arrays for two-way clustering
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Attributes
    ----------
    X : np.ndarray
        Design matrix
    resid : np.ndarray
        Residuals
    n_obs : int
        Number of observations
    n_params : int
        Number of parameters

    Examples
    --------
    >>> # One-way clustering by entity
    >>> clustered = ClusteredStandardErrors(X, resid, entity_ids)
    >>> result = clustered.compute()
    >>> print(result.std_errors)

    >>> # Two-way clustering by entity and time
    >>> clustered = ClusteredStandardErrors(X, resid, (entity_ids, time_ids))
    >>> result = clustered.compute()
    >>> print(result.std_errors)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.

    Petersen, M. A. (2009). Estimating standard errors in finance panel
        data sets: Comparing approaches. Review of Financial Studies,
        22(1), 435-480.
    """

    def __init__(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: Union[np.ndarray, tuple],
        df_correction: bool = True
    ):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = X.shape
        self.df_correction = df_correction

        # Handle one-way vs two-way clustering
        if isinstance(clusters, tuple):
            if len(clusters) != 2:
                raise ValueError("Two-way clustering requires exactly 2 cluster dimensions")
            self.clusters1 = np.asarray(clusters[0])
            self.clusters2 = np.asarray(clusters[1])
            self.cluster_dims = 2
        else:
            self.clusters = np.asarray(clusters)
            self.cluster_dims = 1

        # Validate dimensions
        if self.cluster_dims == 1:
            if len(self.clusters) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, "
                    f"got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread = None

    @property
    def bread(self) -> np.ndarray:
        """Compute and cache bread matrix."""
        if self._bread is None:
            self._bread = compute_bread(self.X)
        return self._bread

    @property
    def n_clusters(self) -> Union[int, tuple]:
        """Number of clusters."""
        if self.cluster_dims == 1:
            return len(np.unique(self.clusters))
        else:
            n_clusters1 = len(np.unique(self.clusters1))
            n_clusters2 = len(np.unique(self.clusters2))
            return (n_clusters1, n_clusters2)

    def compute(self) -> ClusteredCovarianceResult:
        """
        Compute cluster-robust covariance matrix.

        Returns
        -------
        result : ClusteredCovarianceResult
            Cluster-robust covariance and standard errors

        Notes
        -----
        For one-way clustering:
            V = (X'X)^{-1} [Σ_g (X_g'ε_g)(X_g'ε_g)'] (X'X)^{-1}

        For two-way clustering (Cameron, Gelbach, Miller 2011):
            V = V_1 + V_2 - V_12

        where V_1 and V_2 are one-way clustered, and V_12 is clustered
        by the intersection.
        """
        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(
                self.X,
                self.resid,
                self.clusters,
                self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X,
                self.resid,
                self.clusters1,
                self.clusters2,
                self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (
                len(np.unique(self.clusters1)),
                len(np.unique(self.clusters2))
            )

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction
        )

    def diagnostic_summary(self) -> str:
        """
        Generate diagnostic summary for clustering.

        Returns
        -------
        summary : str
            Diagnostic information about clustering

        Notes
        -----
        Provides information about:
        - Number of clusters
        - Cluster sizes (min, max, mean)
        - Warnings if few clusters
        """
        lines = []
        lines.append("Cluster-Robust Standard Errors Diagnostics")
        lines.append("=" * 50)

        if self.cluster_dims == 1:
            unique_clusters = np.unique(self.clusters)
            n_clust = len(unique_clusters)
            cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]

            lines.append(f"Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs / n_clust:.1f}")
            lines.append(f"Cluster size - min: {min(cluster_sizes)}")
            lines.append(f"Cluster size - max: {max(cluster_sizes)}")
            lines.append(f"Cluster size - mean: {np.mean(cluster_sizes):.1f}")

            # Warnings
            if n_clust < 20:
                lines.append("")
                lines.append("⚠ WARNING: Few clusters detected (<20)")
                lines.append("  Cluster-robust SEs may be unreliable with few clusters")
            if n_clust < 10:
                lines.append("⚠ CRITICAL: Very few clusters (<10)")
                lines.append("  Consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append(f"Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) < 20:
                lines.append("")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)


def cluster_by_entity(
    X: np.ndarray,
    resid: np.ndarray,
    entity_ids: np.ndarray,
    df_correction: bool = True
) -> ClusteredCovarianceResult:
    """
    Convenience function for clustering by entity.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    entity_ids : np.ndarray
        Entity identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    result : ClusteredCovarianceResult
        Cluster-robust covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import cluster_by_entity
    >>> result = cluster_by_entity(X, resid, entity_ids)
    >>> print(result.std_errors)
    """
    clustered = ClusteredStandardErrors(X, resid, entity_ids, df_correction)
    return clustered.compute()


def cluster_by_time(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    df_correction: bool = True
) -> ClusteredCovarianceResult:
    """
    Convenience function for clustering by time.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    time_ids : np.ndarray
        Time identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    result : ClusteredCovarianceResult
        Cluster-robust covariance and standard errors
    """
    clustered = ClusteredStandardErrors(X, resid, time_ids, df_correction)
    return clustered.compute()


def twoway_cluster(
    X: np.ndarray,
    resid: np.ndarray,
    cluster1: np.ndarray,
    cluster2: np.ndarray,
    df_correction: bool = True
) -> ClusteredCovarianceResult:
    """
    Convenience function for two-way clustering.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    cluster1 : np.ndarray
        First clustering dimension (e.g., entity_ids)
    cluster2 : np.ndarray
        Second clustering dimension (e.g., time_ids)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    result : ClusteredCovarianceResult
        Two-way cluster-robust covariance and standard errors

    Examples
    --------
    >>> from panelbox.standard_errors import twoway_cluster
    >>> result = twoway_cluster(X, resid, entity_ids, time_ids)
    >>> print(result.std_errors)
    """
    clustered = ClusteredStandardErrors(
        X, resid, (cluster1, cluster2), df_correction
    )
    return clustered.compute()
