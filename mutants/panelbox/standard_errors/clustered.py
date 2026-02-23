"""
Cluster-robust standard errors for panel data.

This module implements one-way and two-way cluster-robust covariance
estimators commonly used in panel data applications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .utils import (
    compute_bread,
    compute_clustered_meat,
    compute_twoway_clustered_meat,
    sandwich_covariance,
)

logger = logging.getLogger(__name__)
from typing import Annotated, Callable, ClassVar

MutantDict = Annotated[dict[str, Callable], "Mutant"]  # type: ignore


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg=None):  # type: ignore
    """Forward call to original or mutated function, depending on the environment."""
    import os  # type: ignore

    mutant_under_test = os.environ["MUTANT_UNDER_TEST"]  # type: ignore
    if mutant_under_test == "fail":  # type: ignore
        from mutmut.__main__ import MutmutProgrammaticFailException  # type: ignore

        raise MutmutProgrammaticFailException("Failed programmatically")  # type: ignore
    elif mutant_under_test == "stats":  # type: ignore
        from mutmut.__main__ import record_trampoline_hit  # type: ignore

        record_trampoline_hit(orig.__module__ + "." + orig.__name__)  # type: ignore
        # (for class methods, orig is bound and thus does not need the explicit self argument)
        result = orig(*call_args, **call_kwargs)  # type: ignore
        return result  # type: ignore
    prefix = orig.__module__ + "." + orig.__name__ + "__mutmut_"  # type: ignore
    if not mutant_under_test.startswith(prefix):  # type: ignore
        result = orig(*call_args, **call_kwargs)  # type: ignore
        return result  # type: ignore
    mutant_name = mutant_under_test.rpartition(".")[-1]  # type: ignore
    if self_arg is not None:  # type: ignore
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs)  # type: ignore
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs)  # type: ignore
    return result  # type: ignore


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
    n_clusters: int | tuple
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
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
    ):
        args = [X, resid, clusters, df_correction]  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁClusteredStandardErrorsǁ__init____mutmut_orig"),
            object.__getattribute__(self, "xǁClusteredStandardErrorsǁ__init____mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁClusteredStandardErrorsǁ__init____mutmut_orig(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_1(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = False,
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_2(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
    ):
        self.X = None
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_3(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
    ):
        self.X = X
        self.resid = None
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_4(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
    ):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = None
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_5(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
    ):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = X.shape
        self.df_correction = None

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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_6(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
    ):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = X.shape
        self.df_correction = df_correction

        # Handle one-way vs two-way clustering
        if isinstance(clusters, tuple):
            if len(clusters) == 2:
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_7(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
    ):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = X.shape
        self.df_correction = df_correction

        # Handle one-way vs two-way clustering
        if isinstance(clusters, tuple):
            if len(clusters) != 3:
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_8(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
    ):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = X.shape
        self.df_correction = df_correction

        # Handle one-way vs two-way clustering
        if isinstance(clusters, tuple):
            if len(clusters) != 2:
                raise ValueError(None)
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_9(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
    ):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = X.shape
        self.df_correction = df_correction

        # Handle one-way vs two-way clustering
        if isinstance(clusters, tuple):
            if len(clusters) != 2:
                raise ValueError("XXTwo-way clustering requires exactly 2 cluster dimensionsXX")
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_10(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
    ):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = X.shape
        self.df_correction = df_correction

        # Handle one-way vs two-way clustering
        if isinstance(clusters, tuple):
            if len(clusters) != 2:
                raise ValueError("two-way clustering requires exactly 2 cluster dimensions")
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_11(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
    ):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = X.shape
        self.df_correction = df_correction

        # Handle one-way vs two-way clustering
        if isinstance(clusters, tuple):
            if len(clusters) != 2:
                raise ValueError("TWO-WAY CLUSTERING REQUIRES EXACTLY 2 CLUSTER DIMENSIONS")
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_12(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
    ):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = X.shape
        self.df_correction = df_correction

        # Handle one-way vs two-way clustering
        if isinstance(clusters, tuple):
            if len(clusters) != 2:
                raise ValueError("Two-way clustering requires exactly 2 cluster dimensions")
            self.clusters1 = None
            self.clusters2 = np.asarray(clusters[1])
            self.cluster_dims = 2
        else:
            self.clusters = np.asarray(clusters)
            self.cluster_dims = 1

        # Validate dimensions
        if self.cluster_dims == 1:
            if len(self.clusters) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_13(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
    ):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = X.shape
        self.df_correction = df_correction

        # Handle one-way vs two-way clustering
        if isinstance(clusters, tuple):
            if len(clusters) != 2:
                raise ValueError("Two-way clustering requires exactly 2 cluster dimensions")
            self.clusters1 = np.asarray(None)
            self.clusters2 = np.asarray(clusters[1])
            self.cluster_dims = 2
        else:
            self.clusters = np.asarray(clusters)
            self.cluster_dims = 1

        # Validate dimensions
        if self.cluster_dims == 1:
            if len(self.clusters) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_14(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
    ):
        self.X = X
        self.resid = resid
        self.n_obs, self.n_params = X.shape
        self.df_correction = df_correction

        # Handle one-way vs two-way clustering
        if isinstance(clusters, tuple):
            if len(clusters) != 2:
                raise ValueError("Two-way clustering requires exactly 2 cluster dimensions")
            self.clusters1 = np.asarray(clusters[1])
            self.clusters2 = np.asarray(clusters[1])
            self.cluster_dims = 2
        else:
            self.clusters = np.asarray(clusters)
            self.cluster_dims = 1

        # Validate dimensions
        if self.cluster_dims == 1:
            if len(self.clusters) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_15(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
            self.clusters2 = None
            self.cluster_dims = 2
        else:
            self.clusters = np.asarray(clusters)
            self.cluster_dims = 1

        # Validate dimensions
        if self.cluster_dims == 1:
            if len(self.clusters) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_16(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
            self.clusters2 = np.asarray(None)
            self.cluster_dims = 2
        else:
            self.clusters = np.asarray(clusters)
            self.cluster_dims = 1

        # Validate dimensions
        if self.cluster_dims == 1:
            if len(self.clusters) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_17(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
            self.clusters2 = np.asarray(clusters[2])
            self.cluster_dims = 2
        else:
            self.clusters = np.asarray(clusters)
            self.cluster_dims = 1

        # Validate dimensions
        if self.cluster_dims == 1:
            if len(self.clusters) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_18(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
            self.cluster_dims = None
        else:
            self.clusters = np.asarray(clusters)
            self.cluster_dims = 1

        # Validate dimensions
        if self.cluster_dims == 1:
            if len(self.clusters) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_19(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
            self.cluster_dims = 3
        else:
            self.clusters = np.asarray(clusters)
            self.cluster_dims = 1

        # Validate dimensions
        if self.cluster_dims == 1:
            if len(self.clusters) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_20(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
            self.clusters = None
            self.cluster_dims = 1

        # Validate dimensions
        if self.cluster_dims == 1:
            if len(self.clusters) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_21(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
            self.clusters = np.asarray(None)
            self.cluster_dims = 1

        # Validate dimensions
        if self.cluster_dims == 1:
            if len(self.clusters) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_22(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
            self.cluster_dims = None

        # Validate dimensions
        if self.cluster_dims == 1:
            if len(self.clusters) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_23(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
            self.cluster_dims = 2

        # Validate dimensions
        if self.cluster_dims == 1:
            if len(self.clusters) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_24(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
        if self.cluster_dims != 1:
            if len(self.clusters) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_25(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
        if self.cluster_dims == 2:
            if len(self.clusters) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_26(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
            if len(self.clusters) == self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_27(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
                raise ValueError(None)
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_28(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs and len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_29(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) == self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_30(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) == self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_31(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(None)

        # Cache
        self._bread: np.ndarray | None = None

    def xǁClusteredStandardErrorsǁ__init____mutmut_32(
        self,
        X: np.ndarray,
        resid: np.ndarray,
        clusters: np.ndarray | tuple,
        df_correction: bool = True,
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
                    f"Cluster dimension mismatch: expected {self.n_obs}, got {len(self.clusters)}"
                )
        else:
            if len(self.clusters1) != self.n_obs or len(self.clusters2) != self.n_obs:
                raise ValueError(
                    f"Cluster dimension mismatch: expected {self.n_obs} for each dimension"
                )

        # Cache
        self._bread: np.ndarray | None = ""

    xǁClusteredStandardErrorsǁ__init____mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁClusteredStandardErrorsǁ__init____mutmut_1": xǁClusteredStandardErrorsǁ__init____mutmut_1,
        "xǁClusteredStandardErrorsǁ__init____mutmut_2": xǁClusteredStandardErrorsǁ__init____mutmut_2,
        "xǁClusteredStandardErrorsǁ__init____mutmut_3": xǁClusteredStandardErrorsǁ__init____mutmut_3,
        "xǁClusteredStandardErrorsǁ__init____mutmut_4": xǁClusteredStandardErrorsǁ__init____mutmut_4,
        "xǁClusteredStandardErrorsǁ__init____mutmut_5": xǁClusteredStandardErrorsǁ__init____mutmut_5,
        "xǁClusteredStandardErrorsǁ__init____mutmut_6": xǁClusteredStandardErrorsǁ__init____mutmut_6,
        "xǁClusteredStandardErrorsǁ__init____mutmut_7": xǁClusteredStandardErrorsǁ__init____mutmut_7,
        "xǁClusteredStandardErrorsǁ__init____mutmut_8": xǁClusteredStandardErrorsǁ__init____mutmut_8,
        "xǁClusteredStandardErrorsǁ__init____mutmut_9": xǁClusteredStandardErrorsǁ__init____mutmut_9,
        "xǁClusteredStandardErrorsǁ__init____mutmut_10": xǁClusteredStandardErrorsǁ__init____mutmut_10,
        "xǁClusteredStandardErrorsǁ__init____mutmut_11": xǁClusteredStandardErrorsǁ__init____mutmut_11,
        "xǁClusteredStandardErrorsǁ__init____mutmut_12": xǁClusteredStandardErrorsǁ__init____mutmut_12,
        "xǁClusteredStandardErrorsǁ__init____mutmut_13": xǁClusteredStandardErrorsǁ__init____mutmut_13,
        "xǁClusteredStandardErrorsǁ__init____mutmut_14": xǁClusteredStandardErrorsǁ__init____mutmut_14,
        "xǁClusteredStandardErrorsǁ__init____mutmut_15": xǁClusteredStandardErrorsǁ__init____mutmut_15,
        "xǁClusteredStandardErrorsǁ__init____mutmut_16": xǁClusteredStandardErrorsǁ__init____mutmut_16,
        "xǁClusteredStandardErrorsǁ__init____mutmut_17": xǁClusteredStandardErrorsǁ__init____mutmut_17,
        "xǁClusteredStandardErrorsǁ__init____mutmut_18": xǁClusteredStandardErrorsǁ__init____mutmut_18,
        "xǁClusteredStandardErrorsǁ__init____mutmut_19": xǁClusteredStandardErrorsǁ__init____mutmut_19,
        "xǁClusteredStandardErrorsǁ__init____mutmut_20": xǁClusteredStandardErrorsǁ__init____mutmut_20,
        "xǁClusteredStandardErrorsǁ__init____mutmut_21": xǁClusteredStandardErrorsǁ__init____mutmut_21,
        "xǁClusteredStandardErrorsǁ__init____mutmut_22": xǁClusteredStandardErrorsǁ__init____mutmut_22,
        "xǁClusteredStandardErrorsǁ__init____mutmut_23": xǁClusteredStandardErrorsǁ__init____mutmut_23,
        "xǁClusteredStandardErrorsǁ__init____mutmut_24": xǁClusteredStandardErrorsǁ__init____mutmut_24,
        "xǁClusteredStandardErrorsǁ__init____mutmut_25": xǁClusteredStandardErrorsǁ__init____mutmut_25,
        "xǁClusteredStandardErrorsǁ__init____mutmut_26": xǁClusteredStandardErrorsǁ__init____mutmut_26,
        "xǁClusteredStandardErrorsǁ__init____mutmut_27": xǁClusteredStandardErrorsǁ__init____mutmut_27,
        "xǁClusteredStandardErrorsǁ__init____mutmut_28": xǁClusteredStandardErrorsǁ__init____mutmut_28,
        "xǁClusteredStandardErrorsǁ__init____mutmut_29": xǁClusteredStandardErrorsǁ__init____mutmut_29,
        "xǁClusteredStandardErrorsǁ__init____mutmut_30": xǁClusteredStandardErrorsǁ__init____mutmut_30,
        "xǁClusteredStandardErrorsǁ__init____mutmut_31": xǁClusteredStandardErrorsǁ__init____mutmut_31,
        "xǁClusteredStandardErrorsǁ__init____mutmut_32": xǁClusteredStandardErrorsǁ__init____mutmut_32,
    }
    xǁClusteredStandardErrorsǁ__init____mutmut_orig.__name__ = "xǁClusteredStandardErrorsǁ__init__"

    @property
    def bread(self) -> np.ndarray:
        """Compute and cache bread matrix."""
        if self._bread is None:
            self._bread = compute_bread(self.X)
        return self._bread

    @property
    def n_clusters(self) -> int | tuple:
        """Number of clusters."""
        if self.cluster_dims == 1:
            return len(np.unique(self.clusters))
        else:
            n_clusters1 = len(np.unique(self.clusters1))
            n_clusters2 = len(np.unique(self.clusters2))
            return (n_clusters1, n_clusters2)

    def compute(self) -> ClusteredCovarianceResult:
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(self, "xǁClusteredStandardErrorsǁcompute__mutmut_orig"),
            object.__getattribute__(self, "xǁClusteredStandardErrorsǁcompute__mutmut_mutants"),
            args,
            kwargs,
            self,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_orig(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_1(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims != 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_2(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 2:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_3(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = None
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_4(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(None, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_5(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, None, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_6(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, None, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_7(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, None)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_8(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_9(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_10(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_11(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(
                self.X,
                self.resid,
                self.clusters,
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_12(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = None
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_13(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(None, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_14(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, None)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_15(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_16(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(
                self.bread,
            )
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_17(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = None

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_18(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = None
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_19(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                None, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_20(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, None, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_21(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, None, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_22(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, None, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_23(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, None
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_24(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_25(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_26(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_27(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_28(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X,
                self.resid,
                self.clusters1,
                self.clusters2,
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_29(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = None
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_30(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(None, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_31(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, None)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_32(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_33(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(
                self.bread,
            )
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_34(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = None

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_35(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = None

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_36(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(None)

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_37(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(None))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_38(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=None,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_39(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=None,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_40(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=None,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_41(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=None,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_42(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=None,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_43(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=None,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_44(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=None,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_45(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_46(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_47(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_48(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_49(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            cluster_dims=self.cluster_dims,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_50(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            df_correction=self.df_correction,
        )

    def xǁClusteredStandardErrorsǁcompute__mutmut_51(self) -> ClusteredCovarianceResult:
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
        n_clust: int | tuple[int, int]

        if self.cluster_dims == 1:
            # One-way clustering
            meat = compute_clustered_meat(self.X, self.resid, self.clusters, self.df_correction)
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = len(np.unique(self.clusters))

        else:
            # Two-way clustering
            meat = compute_twoway_clustered_meat(
                self.X, self.resid, self.clusters1, self.clusters2, self.df_correction
            )
            cov_matrix = sandwich_covariance(self.bread, meat)
            n_clust = (len(np.unique(self.clusters1)), len(np.unique(self.clusters2)))

        std_errors = np.sqrt(np.diag(cov_matrix))

        return ClusteredCovarianceResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            n_clusters=n_clust,
            n_obs=self.n_obs,
            n_params=self.n_params,
            cluster_dims=self.cluster_dims,
        )

    xǁClusteredStandardErrorsǁcompute__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁClusteredStandardErrorsǁcompute__mutmut_1": xǁClusteredStandardErrorsǁcompute__mutmut_1,
        "xǁClusteredStandardErrorsǁcompute__mutmut_2": xǁClusteredStandardErrorsǁcompute__mutmut_2,
        "xǁClusteredStandardErrorsǁcompute__mutmut_3": xǁClusteredStandardErrorsǁcompute__mutmut_3,
        "xǁClusteredStandardErrorsǁcompute__mutmut_4": xǁClusteredStandardErrorsǁcompute__mutmut_4,
        "xǁClusteredStandardErrorsǁcompute__mutmut_5": xǁClusteredStandardErrorsǁcompute__mutmut_5,
        "xǁClusteredStandardErrorsǁcompute__mutmut_6": xǁClusteredStandardErrorsǁcompute__mutmut_6,
        "xǁClusteredStandardErrorsǁcompute__mutmut_7": xǁClusteredStandardErrorsǁcompute__mutmut_7,
        "xǁClusteredStandardErrorsǁcompute__mutmut_8": xǁClusteredStandardErrorsǁcompute__mutmut_8,
        "xǁClusteredStandardErrorsǁcompute__mutmut_9": xǁClusteredStandardErrorsǁcompute__mutmut_9,
        "xǁClusteredStandardErrorsǁcompute__mutmut_10": xǁClusteredStandardErrorsǁcompute__mutmut_10,
        "xǁClusteredStandardErrorsǁcompute__mutmut_11": xǁClusteredStandardErrorsǁcompute__mutmut_11,
        "xǁClusteredStandardErrorsǁcompute__mutmut_12": xǁClusteredStandardErrorsǁcompute__mutmut_12,
        "xǁClusteredStandardErrorsǁcompute__mutmut_13": xǁClusteredStandardErrorsǁcompute__mutmut_13,
        "xǁClusteredStandardErrorsǁcompute__mutmut_14": xǁClusteredStandardErrorsǁcompute__mutmut_14,
        "xǁClusteredStandardErrorsǁcompute__mutmut_15": xǁClusteredStandardErrorsǁcompute__mutmut_15,
        "xǁClusteredStandardErrorsǁcompute__mutmut_16": xǁClusteredStandardErrorsǁcompute__mutmut_16,
        "xǁClusteredStandardErrorsǁcompute__mutmut_17": xǁClusteredStandardErrorsǁcompute__mutmut_17,
        "xǁClusteredStandardErrorsǁcompute__mutmut_18": xǁClusteredStandardErrorsǁcompute__mutmut_18,
        "xǁClusteredStandardErrorsǁcompute__mutmut_19": xǁClusteredStandardErrorsǁcompute__mutmut_19,
        "xǁClusteredStandardErrorsǁcompute__mutmut_20": xǁClusteredStandardErrorsǁcompute__mutmut_20,
        "xǁClusteredStandardErrorsǁcompute__mutmut_21": xǁClusteredStandardErrorsǁcompute__mutmut_21,
        "xǁClusteredStandardErrorsǁcompute__mutmut_22": xǁClusteredStandardErrorsǁcompute__mutmut_22,
        "xǁClusteredStandardErrorsǁcompute__mutmut_23": xǁClusteredStandardErrorsǁcompute__mutmut_23,
        "xǁClusteredStandardErrorsǁcompute__mutmut_24": xǁClusteredStandardErrorsǁcompute__mutmut_24,
        "xǁClusteredStandardErrorsǁcompute__mutmut_25": xǁClusteredStandardErrorsǁcompute__mutmut_25,
        "xǁClusteredStandardErrorsǁcompute__mutmut_26": xǁClusteredStandardErrorsǁcompute__mutmut_26,
        "xǁClusteredStandardErrorsǁcompute__mutmut_27": xǁClusteredStandardErrorsǁcompute__mutmut_27,
        "xǁClusteredStandardErrorsǁcompute__mutmut_28": xǁClusteredStandardErrorsǁcompute__mutmut_28,
        "xǁClusteredStandardErrorsǁcompute__mutmut_29": xǁClusteredStandardErrorsǁcompute__mutmut_29,
        "xǁClusteredStandardErrorsǁcompute__mutmut_30": xǁClusteredStandardErrorsǁcompute__mutmut_30,
        "xǁClusteredStandardErrorsǁcompute__mutmut_31": xǁClusteredStandardErrorsǁcompute__mutmut_31,
        "xǁClusteredStandardErrorsǁcompute__mutmut_32": xǁClusteredStandardErrorsǁcompute__mutmut_32,
        "xǁClusteredStandardErrorsǁcompute__mutmut_33": xǁClusteredStandardErrorsǁcompute__mutmut_33,
        "xǁClusteredStandardErrorsǁcompute__mutmut_34": xǁClusteredStandardErrorsǁcompute__mutmut_34,
        "xǁClusteredStandardErrorsǁcompute__mutmut_35": xǁClusteredStandardErrorsǁcompute__mutmut_35,
        "xǁClusteredStandardErrorsǁcompute__mutmut_36": xǁClusteredStandardErrorsǁcompute__mutmut_36,
        "xǁClusteredStandardErrorsǁcompute__mutmut_37": xǁClusteredStandardErrorsǁcompute__mutmut_37,
        "xǁClusteredStandardErrorsǁcompute__mutmut_38": xǁClusteredStandardErrorsǁcompute__mutmut_38,
        "xǁClusteredStandardErrorsǁcompute__mutmut_39": xǁClusteredStandardErrorsǁcompute__mutmut_39,
        "xǁClusteredStandardErrorsǁcompute__mutmut_40": xǁClusteredStandardErrorsǁcompute__mutmut_40,
        "xǁClusteredStandardErrorsǁcompute__mutmut_41": xǁClusteredStandardErrorsǁcompute__mutmut_41,
        "xǁClusteredStandardErrorsǁcompute__mutmut_42": xǁClusteredStandardErrorsǁcompute__mutmut_42,
        "xǁClusteredStandardErrorsǁcompute__mutmut_43": xǁClusteredStandardErrorsǁcompute__mutmut_43,
        "xǁClusteredStandardErrorsǁcompute__mutmut_44": xǁClusteredStandardErrorsǁcompute__mutmut_44,
        "xǁClusteredStandardErrorsǁcompute__mutmut_45": xǁClusteredStandardErrorsǁcompute__mutmut_45,
        "xǁClusteredStandardErrorsǁcompute__mutmut_46": xǁClusteredStandardErrorsǁcompute__mutmut_46,
        "xǁClusteredStandardErrorsǁcompute__mutmut_47": xǁClusteredStandardErrorsǁcompute__mutmut_47,
        "xǁClusteredStandardErrorsǁcompute__mutmut_48": xǁClusteredStandardErrorsǁcompute__mutmut_48,
        "xǁClusteredStandardErrorsǁcompute__mutmut_49": xǁClusteredStandardErrorsǁcompute__mutmut_49,
        "xǁClusteredStandardErrorsǁcompute__mutmut_50": xǁClusteredStandardErrorsǁcompute__mutmut_50,
        "xǁClusteredStandardErrorsǁcompute__mutmut_51": xǁClusteredStandardErrorsǁcompute__mutmut_51,
    }
    xǁClusteredStandardErrorsǁcompute__mutmut_orig.__name__ = "xǁClusteredStandardErrorsǁcompute"

    def diagnostic_summary(self) -> str:
        args = []  # type: ignore
        kwargs = {}  # type: ignore
        return _mutmut_trampoline(
            object.__getattribute__(
                self, "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_orig"
            ),
            object.__getattribute__(
                self, "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_mutants"
            ),
            args,
            kwargs,
            self,
        )

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_orig(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_1(self) -> str:
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
        lines = None
        lines.append("Cluster-Robust Standard Errors Diagnostics")
        lines.append("=" * 50)

        if self.cluster_dims == 1:
            unique_clusters = np.unique(self.clusters)
            n_clust = len(unique_clusters)
            cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_2(self) -> str:
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
        lines.append(None)
        lines.append("=" * 50)

        if self.cluster_dims == 1:
            unique_clusters = np.unique(self.clusters)
            n_clust = len(unique_clusters)
            cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_3(self) -> str:
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
        lines.append("XXCluster-Robust Standard Errors DiagnosticsXX")
        lines.append("=" * 50)

        if self.cluster_dims == 1:
            unique_clusters = np.unique(self.clusters)
            n_clust = len(unique_clusters)
            cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_4(self) -> str:
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
        lines.append("cluster-robust standard errors diagnostics")
        lines.append("=" * 50)

        if self.cluster_dims == 1:
            unique_clusters = np.unique(self.clusters)
            n_clust = len(unique_clusters)
            cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_5(self) -> str:
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
        lines.append("CLUSTER-ROBUST STANDARD ERRORS DIAGNOSTICS")
        lines.append("=" * 50)

        if self.cluster_dims == 1:
            unique_clusters = np.unique(self.clusters)
            n_clust = len(unique_clusters)
            cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_6(self) -> str:
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
        lines.append(None)

        if self.cluster_dims == 1:
            unique_clusters = np.unique(self.clusters)
            n_clust = len(unique_clusters)
            cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_7(self) -> str:
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
        lines.append("=" / 50)

        if self.cluster_dims == 1:
            unique_clusters = np.unique(self.clusters)
            n_clust = len(unique_clusters)
            cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_8(self) -> str:
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
        lines.append("XX=XX" * 50)

        if self.cluster_dims == 1:
            unique_clusters = np.unique(self.clusters)
            n_clust = len(unique_clusters)
            cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_9(self) -> str:
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
        lines.append("=" * 51)

        if self.cluster_dims == 1:
            unique_clusters = np.unique(self.clusters)
            n_clust = len(unique_clusters)
            cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_10(self) -> str:
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

        if self.cluster_dims != 1:
            unique_clusters = np.unique(self.clusters)
            n_clust = len(unique_clusters)
            cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_11(self) -> str:
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

        if self.cluster_dims == 2:
            unique_clusters = np.unique(self.clusters)
            n_clust = len(unique_clusters)
            cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_12(self) -> str:
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
            unique_clusters = None
            n_clust = len(unique_clusters)
            cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_13(self) -> str:
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
            unique_clusters = np.unique(None)
            n_clust = len(unique_clusters)
            cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_14(self) -> str:
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
            n_clust = None
            cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_15(self) -> str:
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
            cluster_sizes = None

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_16(self) -> str:
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
            cluster_sizes = [np.sum(None) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_17(self) -> str:
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
            cluster_sizes = [np.sum(self.clusters != c) for c in unique_clusters]

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_18(self) -> str:
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

            lines.append(None)
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_19(self) -> str:
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

            lines.append("XXClustering dimension: 1XX")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_20(self) -> str:
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

            lines.append("clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_21(self) -> str:
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

            lines.append("CLUSTERING DIMENSION: 1")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_22(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(None)
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_23(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(None)
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_24(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(None)
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_25(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs * n_clust:.1f}")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_26(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs / n_clust:.1f}")
            lines.append(None)
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_27(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs / n_clust:.1f}")
            lines.append(f"Cluster size - min: {min(None)}")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_28(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs / n_clust:.1f}")
            lines.append(f"Cluster size - min: {min(cluster_sizes)}")
            lines.append(None)
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_29(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs / n_clust:.1f}")
            lines.append(f"Cluster size - min: {min(cluster_sizes)}")
            lines.append(f"Cluster size - max: {max(None)}")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_30(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs / n_clust:.1f}")
            lines.append(f"Cluster size - min: {min(cluster_sizes)}")
            lines.append(f"Cluster size - max: {max(cluster_sizes)}")
            lines.append(None)

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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_31(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs / n_clust:.1f}")
            lines.append(f"Cluster size - min: {min(cluster_sizes)}")
            lines.append(f"Cluster size - max: {max(cluster_sizes)}")
            lines.append(f"Cluster size - mean: {np.mean(None):.1f}")

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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_32(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs / n_clust:.1f}")
            lines.append(f"Cluster size - min: {min(cluster_sizes)}")
            lines.append(f"Cluster size - max: {max(cluster_sizes)}")
            lines.append(f"Cluster size - mean: {np.mean(cluster_sizes):.1f}")

            # Warnings
            if n_clust <= 20:
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_33(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs / n_clust:.1f}")
            lines.append(f"Cluster size - min: {min(cluster_sizes)}")
            lines.append(f"Cluster size - max: {max(cluster_sizes)}")
            lines.append(f"Cluster size - mean: {np.mean(cluster_sizes):.1f}")

            # Warnings
            if n_clust < 21:
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_34(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs / n_clust:.1f}")
            lines.append(f"Cluster size - min: {min(cluster_sizes)}")
            lines.append(f"Cluster size - max: {max(cluster_sizes)}")
            lines.append(f"Cluster size - mean: {np.mean(cluster_sizes):.1f}")

            # Warnings
            if n_clust < 20:
                lines.append(None)
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_35(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs / n_clust:.1f}")
            lines.append(f"Cluster size - min: {min(cluster_sizes)}")
            lines.append(f"Cluster size - max: {max(cluster_sizes)}")
            lines.append(f"Cluster size - mean: {np.mean(cluster_sizes):.1f}")

            # Warnings
            if n_clust < 20:
                lines.append("XXXX")
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

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_36(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs / n_clust:.1f}")
            lines.append(f"Cluster size - min: {min(cluster_sizes)}")
            lines.append(f"Cluster size - max: {max(cluster_sizes)}")
            lines.append(f"Cluster size - mean: {np.mean(cluster_sizes):.1f}")

            # Warnings
            if n_clust < 20:
                lines.append("")
                lines.append(None)
                lines.append("  Cluster-robust SEs may be unreliable with few clusters")
            if n_clust < 10:
                lines.append("⚠ CRITICAL: Very few clusters (<10)")
                lines.append("  Consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_37(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs / n_clust:.1f}")
            lines.append(f"Cluster size - min: {min(cluster_sizes)}")
            lines.append(f"Cluster size - max: {max(cluster_sizes)}")
            lines.append(f"Cluster size - mean: {np.mean(cluster_sizes):.1f}")

            # Warnings
            if n_clust < 20:
                lines.append("")
                lines.append("XX⚠ WARNING: Few clusters detected (<20)XX")
                lines.append("  Cluster-robust SEs may be unreliable with few clusters")
            if n_clust < 10:
                lines.append("⚠ CRITICAL: Very few clusters (<10)")
                lines.append("  Consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_38(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs / n_clust:.1f}")
            lines.append(f"Cluster size - min: {min(cluster_sizes)}")
            lines.append(f"Cluster size - max: {max(cluster_sizes)}")
            lines.append(f"Cluster size - mean: {np.mean(cluster_sizes):.1f}")

            # Warnings
            if n_clust < 20:
                lines.append("")
                lines.append("⚠ warning: few clusters detected (<20)")
                lines.append("  Cluster-robust SEs may be unreliable with few clusters")
            if n_clust < 10:
                lines.append("⚠ CRITICAL: Very few clusters (<10)")
                lines.append("  Consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_39(self) -> str:
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

            lines.append("Clustering dimension: 1")
            lines.append(f"Number of clusters: {n_clust}")
            lines.append(f"Observations: {self.n_obs}")
            lines.append(f"Avg obs per cluster: {self.n_obs / n_clust:.1f}")
            lines.append(f"Cluster size - min: {min(cluster_sizes)}")
            lines.append(f"Cluster size - max: {max(cluster_sizes)}")
            lines.append(f"Cluster size - mean: {np.mean(cluster_sizes):.1f}")

            # Warnings
            if n_clust < 20:
                lines.append("")
                lines.append("⚠ WARNING: FEW CLUSTERS DETECTED (<20)")
                lines.append("  Cluster-robust SEs may be unreliable with few clusters")
            if n_clust < 10:
                lines.append("⚠ CRITICAL: Very few clusters (<10)")
                lines.append("  Consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_40(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
                lines.append(None)
            if n_clust < 10:
                lines.append("⚠ CRITICAL: Very few clusters (<10)")
                lines.append("  Consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_41(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
                lines.append("XX  Cluster-robust SEs may be unreliable with few clustersXX")
            if n_clust < 10:
                lines.append("⚠ CRITICAL: Very few clusters (<10)")
                lines.append("  Consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_42(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
                lines.append("  cluster-robust ses may be unreliable with few clusters")
            if n_clust < 10:
                lines.append("⚠ CRITICAL: Very few clusters (<10)")
                lines.append("  Consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_43(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
                lines.append("  CLUSTER-ROBUST SES MAY BE UNRELIABLE WITH FEW CLUSTERS")
            if n_clust < 10:
                lines.append("⚠ CRITICAL: Very few clusters (<10)")
                lines.append("  Consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_44(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
            if n_clust <= 10:
                lines.append("⚠ CRITICAL: Very few clusters (<10)")
                lines.append("  Consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_45(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
            if n_clust < 11:
                lines.append("⚠ CRITICAL: Very few clusters (<10)")
                lines.append("  Consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_46(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
                lines.append(None)
                lines.append("  Consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_47(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
                lines.append("XX⚠ CRITICAL: Very few clusters (<10)XX")
                lines.append("  Consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_48(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
                lines.append("⚠ critical: very few clusters (<10)")
                lines.append("  Consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_49(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
                lines.append("⚠ CRITICAL: VERY FEW CLUSTERS (<10)")
                lines.append("  Consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_50(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
                lines.append(None)

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_51(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
                lines.append("XX  Consider using alternative inference methodsXX")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_52(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
                lines.append("  consider using alternative inference methods")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_53(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
                lines.append("  CONSIDER USING ALTERNATIVE INFERENCE METHODS")

        else:
            unique_clusters1 = np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_54(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
            unique_clusters1 = None
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_55(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
            unique_clusters1 = np.unique(None)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_56(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
            unique_clusters2 = None
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_57(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
            unique_clusters2 = np.unique(None)
            n_clust1 = len(unique_clusters1)
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_58(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
            np.unique(self.clusters1)
            unique_clusters2 = np.unique(self.clusters2)
            n_clust1 = None
            n_clust2 = len(unique_clusters2)

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_59(self) -> str:
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

            lines.append("Clustering dimension: 1")
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
            np.unique(self.clusters2)
            n_clust1 = len(unique_clusters1)
            n_clust2 = None

            lines.append("Clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_60(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append(None)
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_61(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("XXClustering dimensions: 2XX")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_62(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("clustering dimensions: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_63(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("CLUSTERING DIMENSIONS: 2")
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

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_64(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(None)
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) < 20:
                lines.append("")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_65(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(None)
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) < 20:
                lines.append("")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_66(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(None)

            # Warnings
            if min(n_clust1, n_clust2) < 20:
                lines.append("")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_67(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(None, n_clust2) < 20:
                lines.append("")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_68(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, None) < 20:
                lines.append("")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_69(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust2) < 20:
                lines.append("")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_70(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if (
                min(
                    n_clust1,
                )
                < 20
            ):
                lines.append("")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_71(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) <= 20:
                lines.append("")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_72(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) < 21:
                lines.append("")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_73(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) < 20:
                lines.append(None)
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_74(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) < 20:
                lines.append("XXXX")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_75(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) < 20:
                lines.append("")
                lines.append(None)

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_76(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) < 20:
                lines.append("")
                lines.append("XX⚠ WARNING: Few clusters in at least one dimension (<20)XX")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_77(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) < 20:
                lines.append("")
                lines.append("⚠ warning: few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_78(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) < 20:
                lines.append("")
                lines.append("⚠ WARNING: FEW CLUSTERS IN AT LEAST ONE DIMENSION (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_79(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) < 20:
                lines.append("")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append(None)
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_80(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) < 20:
                lines.append("")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("XXXX")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_81(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) < 20:
                lines.append("")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(None)

        return "\n".join(lines)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_82(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) < 20:
                lines.append("")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "\n".join(None)

    def xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_83(self) -> str:
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

            lines.append("Clustering dimension: 1")
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

            lines.append("Clustering dimensions: 2")
            lines.append(f"Number of clusters (dim 1): {n_clust1}")
            lines.append(f"Number of clusters (dim 2): {n_clust2}")
            lines.append(f"Observations: {self.n_obs}")

            # Warnings
            if min(n_clust1, n_clust2) < 20:
                lines.append("")
                lines.append("⚠ WARNING: Few clusters in at least one dimension (<20)")

        lines.append("")
        lines.append(f"Finite-sample correction: {self.df_correction}")

        return "XX\nXX".join(lines)

    xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_1": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_1,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_2": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_2,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_3": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_3,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_4": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_4,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_5": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_5,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_6": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_6,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_7": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_7,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_8": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_8,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_9": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_9,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_10": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_10,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_11": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_11,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_12": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_12,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_13": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_13,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_14": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_14,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_15": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_15,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_16": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_16,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_17": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_17,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_18": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_18,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_19": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_19,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_20": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_20,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_21": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_21,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_22": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_22,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_23": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_23,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_24": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_24,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_25": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_25,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_26": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_26,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_27": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_27,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_28": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_28,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_29": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_29,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_30": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_30,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_31": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_31,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_32": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_32,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_33": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_33,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_34": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_34,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_35": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_35,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_36": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_36,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_37": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_37,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_38": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_38,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_39": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_39,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_40": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_40,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_41": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_41,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_42": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_42,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_43": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_43,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_44": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_44,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_45": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_45,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_46": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_46,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_47": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_47,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_48": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_48,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_49": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_49,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_50": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_50,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_51": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_51,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_52": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_52,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_53": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_53,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_54": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_54,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_55": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_55,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_56": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_56,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_57": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_57,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_58": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_58,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_59": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_59,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_60": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_60,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_61": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_61,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_62": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_62,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_63": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_63,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_64": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_64,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_65": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_65,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_66": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_66,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_67": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_67,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_68": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_68,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_69": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_69,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_70": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_70,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_71": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_71,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_72": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_72,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_73": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_73,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_74": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_74,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_75": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_75,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_76": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_76,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_77": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_77,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_78": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_78,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_79": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_79,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_80": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_80,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_81": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_81,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_82": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_82,
        "xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_83": xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_83,
    }
    xǁClusteredStandardErrorsǁdiagnostic_summary__mutmut_orig.__name__ = (
        "xǁClusteredStandardErrorsǁdiagnostic_summary"
    )


def cluster_by_entity(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, df_correction: bool = True
) -> ClusteredCovarianceResult:
    args = [X, resid, entity_ids, df_correction]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_cluster_by_entity__mutmut_orig, x_cluster_by_entity__mutmut_mutants, args, kwargs, None
    )


def x_cluster_by_entity__mutmut_orig(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, df_correction: bool = True
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


def x_cluster_by_entity__mutmut_1(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, df_correction: bool = False
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


def x_cluster_by_entity__mutmut_2(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, df_correction: bool = True
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
    clustered = None
    return clustered.compute()


def x_cluster_by_entity__mutmut_3(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(None, resid, entity_ids, df_correction)
    return clustered.compute()


def x_cluster_by_entity__mutmut_4(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(X, None, entity_ids, df_correction)
    return clustered.compute()


def x_cluster_by_entity__mutmut_5(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(X, resid, None, df_correction)
    return clustered.compute()


def x_cluster_by_entity__mutmut_6(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(X, resid, entity_ids, None)
    return clustered.compute()


def x_cluster_by_entity__mutmut_7(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(resid, entity_ids, df_correction)
    return clustered.compute()


def x_cluster_by_entity__mutmut_8(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(X, entity_ids, df_correction)
    return clustered.compute()


def x_cluster_by_entity__mutmut_9(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(X, resid, df_correction)
    return clustered.compute()


def x_cluster_by_entity__mutmut_10(
    X: np.ndarray, resid: np.ndarray, entity_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(
        X,
        resid,
        entity_ids,
    )
    return clustered.compute()


x_cluster_by_entity__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_cluster_by_entity__mutmut_1": x_cluster_by_entity__mutmut_1,
    "x_cluster_by_entity__mutmut_2": x_cluster_by_entity__mutmut_2,
    "x_cluster_by_entity__mutmut_3": x_cluster_by_entity__mutmut_3,
    "x_cluster_by_entity__mutmut_4": x_cluster_by_entity__mutmut_4,
    "x_cluster_by_entity__mutmut_5": x_cluster_by_entity__mutmut_5,
    "x_cluster_by_entity__mutmut_6": x_cluster_by_entity__mutmut_6,
    "x_cluster_by_entity__mutmut_7": x_cluster_by_entity__mutmut_7,
    "x_cluster_by_entity__mutmut_8": x_cluster_by_entity__mutmut_8,
    "x_cluster_by_entity__mutmut_9": x_cluster_by_entity__mutmut_9,
    "x_cluster_by_entity__mutmut_10": x_cluster_by_entity__mutmut_10,
}
x_cluster_by_entity__mutmut_orig.__name__ = "x_cluster_by_entity"


def cluster_by_time(
    X: np.ndarray, resid: np.ndarray, time_ids: np.ndarray, df_correction: bool = True
) -> ClusteredCovarianceResult:
    args = [X, resid, time_ids, df_correction]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_cluster_by_time__mutmut_orig, x_cluster_by_time__mutmut_mutants, args, kwargs, None
    )


def x_cluster_by_time__mutmut_orig(
    X: np.ndarray, resid: np.ndarray, time_ids: np.ndarray, df_correction: bool = True
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


def x_cluster_by_time__mutmut_1(
    X: np.ndarray, resid: np.ndarray, time_ids: np.ndarray, df_correction: bool = False
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


def x_cluster_by_time__mutmut_2(
    X: np.ndarray, resid: np.ndarray, time_ids: np.ndarray, df_correction: bool = True
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
    clustered = None
    return clustered.compute()


def x_cluster_by_time__mutmut_3(
    X: np.ndarray, resid: np.ndarray, time_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(None, resid, time_ids, df_correction)
    return clustered.compute()


def x_cluster_by_time__mutmut_4(
    X: np.ndarray, resid: np.ndarray, time_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(X, None, time_ids, df_correction)
    return clustered.compute()


def x_cluster_by_time__mutmut_5(
    X: np.ndarray, resid: np.ndarray, time_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(X, resid, None, df_correction)
    return clustered.compute()


def x_cluster_by_time__mutmut_6(
    X: np.ndarray, resid: np.ndarray, time_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(X, resid, time_ids, None)
    return clustered.compute()


def x_cluster_by_time__mutmut_7(
    X: np.ndarray, resid: np.ndarray, time_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(resid, time_ids, df_correction)
    return clustered.compute()


def x_cluster_by_time__mutmut_8(
    X: np.ndarray, resid: np.ndarray, time_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(X, time_ids, df_correction)
    return clustered.compute()


def x_cluster_by_time__mutmut_9(
    X: np.ndarray, resid: np.ndarray, time_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(X, resid, df_correction)
    return clustered.compute()


def x_cluster_by_time__mutmut_10(
    X: np.ndarray, resid: np.ndarray, time_ids: np.ndarray, df_correction: bool = True
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
    clustered = ClusteredStandardErrors(
        X,
        resid,
        time_ids,
    )
    return clustered.compute()


x_cluster_by_time__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_cluster_by_time__mutmut_1": x_cluster_by_time__mutmut_1,
    "x_cluster_by_time__mutmut_2": x_cluster_by_time__mutmut_2,
    "x_cluster_by_time__mutmut_3": x_cluster_by_time__mutmut_3,
    "x_cluster_by_time__mutmut_4": x_cluster_by_time__mutmut_4,
    "x_cluster_by_time__mutmut_5": x_cluster_by_time__mutmut_5,
    "x_cluster_by_time__mutmut_6": x_cluster_by_time__mutmut_6,
    "x_cluster_by_time__mutmut_7": x_cluster_by_time__mutmut_7,
    "x_cluster_by_time__mutmut_8": x_cluster_by_time__mutmut_8,
    "x_cluster_by_time__mutmut_9": x_cluster_by_time__mutmut_9,
    "x_cluster_by_time__mutmut_10": x_cluster_by_time__mutmut_10,
}
x_cluster_by_time__mutmut_orig.__name__ = "x_cluster_by_time"


def twoway_cluster(
    X: np.ndarray,
    resid: np.ndarray,
    cluster1: np.ndarray,
    cluster2: np.ndarray,
    df_correction: bool = True,
) -> ClusteredCovarianceResult:
    args = [X, resid, cluster1, cluster2, df_correction]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_twoway_cluster__mutmut_orig, x_twoway_cluster__mutmut_mutants, args, kwargs, None
    )


def x_twoway_cluster__mutmut_orig(
    X: np.ndarray,
    resid: np.ndarray,
    cluster1: np.ndarray,
    cluster2: np.ndarray,
    df_correction: bool = True,
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
    clustered = ClusteredStandardErrors(X, resid, (cluster1, cluster2), df_correction)
    return clustered.compute()


def x_twoway_cluster__mutmut_1(
    X: np.ndarray,
    resid: np.ndarray,
    cluster1: np.ndarray,
    cluster2: np.ndarray,
    df_correction: bool = False,
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
    clustered = ClusteredStandardErrors(X, resid, (cluster1, cluster2), df_correction)
    return clustered.compute()


def x_twoway_cluster__mutmut_2(
    X: np.ndarray,
    resid: np.ndarray,
    cluster1: np.ndarray,
    cluster2: np.ndarray,
    df_correction: bool = True,
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
    clustered = None
    return clustered.compute()


def x_twoway_cluster__mutmut_3(
    X: np.ndarray,
    resid: np.ndarray,
    cluster1: np.ndarray,
    cluster2: np.ndarray,
    df_correction: bool = True,
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
    clustered = ClusteredStandardErrors(None, resid, (cluster1, cluster2), df_correction)
    return clustered.compute()


def x_twoway_cluster__mutmut_4(
    X: np.ndarray,
    resid: np.ndarray,
    cluster1: np.ndarray,
    cluster2: np.ndarray,
    df_correction: bool = True,
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
    clustered = ClusteredStandardErrors(X, None, (cluster1, cluster2), df_correction)
    return clustered.compute()


def x_twoway_cluster__mutmut_5(
    X: np.ndarray,
    resid: np.ndarray,
    cluster1: np.ndarray,
    cluster2: np.ndarray,
    df_correction: bool = True,
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
    clustered = ClusteredStandardErrors(X, resid, None, df_correction)
    return clustered.compute()


def x_twoway_cluster__mutmut_6(
    X: np.ndarray,
    resid: np.ndarray,
    cluster1: np.ndarray,
    cluster2: np.ndarray,
    df_correction: bool = True,
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
    clustered = ClusteredStandardErrors(X, resid, (cluster1, cluster2), None)
    return clustered.compute()


def x_twoway_cluster__mutmut_7(
    X: np.ndarray,
    resid: np.ndarray,
    cluster1: np.ndarray,
    cluster2: np.ndarray,
    df_correction: bool = True,
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
    clustered = ClusteredStandardErrors(resid, (cluster1, cluster2), df_correction)
    return clustered.compute()


def x_twoway_cluster__mutmut_8(
    X: np.ndarray,
    resid: np.ndarray,
    cluster1: np.ndarray,
    cluster2: np.ndarray,
    df_correction: bool = True,
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
    clustered = ClusteredStandardErrors(X, (cluster1, cluster2), df_correction)
    return clustered.compute()


def x_twoway_cluster__mutmut_9(
    X: np.ndarray,
    resid: np.ndarray,
    cluster1: np.ndarray,
    cluster2: np.ndarray,
    df_correction: bool = True,
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
    clustered = ClusteredStandardErrors(X, resid, df_correction)
    return clustered.compute()


def x_twoway_cluster__mutmut_10(
    X: np.ndarray,
    resid: np.ndarray,
    cluster1: np.ndarray,
    cluster2: np.ndarray,
    df_correction: bool = True,
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
        X,
        resid,
        (cluster1, cluster2),
    )
    return clustered.compute()


x_twoway_cluster__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_twoway_cluster__mutmut_1": x_twoway_cluster__mutmut_1,
    "x_twoway_cluster__mutmut_2": x_twoway_cluster__mutmut_2,
    "x_twoway_cluster__mutmut_3": x_twoway_cluster__mutmut_3,
    "x_twoway_cluster__mutmut_4": x_twoway_cluster__mutmut_4,
    "x_twoway_cluster__mutmut_5": x_twoway_cluster__mutmut_5,
    "x_twoway_cluster__mutmut_6": x_twoway_cluster__mutmut_6,
    "x_twoway_cluster__mutmut_7": x_twoway_cluster__mutmut_7,
    "x_twoway_cluster__mutmut_8": x_twoway_cluster__mutmut_8,
    "x_twoway_cluster__mutmut_9": x_twoway_cluster__mutmut_9,
    "x_twoway_cluster__mutmut_10": x_twoway_cluster__mutmut_10,
}
x_twoway_cluster__mutmut_orig.__name__ = "x_twoway_cluster"
