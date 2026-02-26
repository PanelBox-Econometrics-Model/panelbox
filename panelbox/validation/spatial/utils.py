"""Utility functions for spatial validation tests."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from panelbox.core.spatial_weights import SpatialWeights

logger = logging.getLogger(__name__)


def validate_spatial_weights(W: np.ndarray | SpatialWeights) -> np.ndarray:
    """
    Validate and convert spatial weights to numpy array.

    Parameters
    ----------
    W : np.ndarray or SpatialWeights
        Spatial weight matrix

    Returns
    -------
    np.ndarray
        Validated spatial weight matrix

    Raises
    ------
    ValueError
        If W is not square or has invalid values
    """
    # Convert SpatialWeights object if necessary
    if hasattr(W, "to_dense"):
        W_array = W.to_dense()
    elif hasattr(W, "todense"):
        W_array = W.todense()
    elif hasattr(W, "toarray"):
        W_array = W.toarray()
    else:
        W_array = np.asarray(W)

    # Check square
    if W_array.ndim != 2:
        raise ValueError("Spatial weight matrix must be 2-dimensional")

    n_rows, n_cols = W_array.shape
    if n_rows != n_cols:
        raise ValueError(f"Spatial weight matrix must be square, got {n_rows}x{n_cols}")

    # Check diagonal is zero
    if np.any(np.diag(W_array) != 0):
        raise ValueError("Diagonal of spatial weight matrix must be zero (no self-neighbors)")

    # Check for NaN or Inf
    if np.any(~np.isfinite(W_array)):
        raise ValueError("Spatial weight matrix contains NaN or Inf values")

    return W_array


def standardize_spatial_weights(W: np.ndarray, style: str = "row") -> np.ndarray:
    """
    Standardize spatial weight matrix.

    Parameters
    ----------
    W : np.ndarray
        Spatial weight matrix
    style : str
        Standardization style:
        - 'row': Row standardization (default)
        - 'spectral': Spectral normalization (divide by largest eigenvalue)
        - 'none': No standardization

    Returns
    -------
    np.ndarray
        Standardized spatial weight matrix
    """
    if style == "none":
        return W

    elif style == "row":
        # Row standardization
        row_sums = W.sum(axis=1)
        # Avoid division by zero for isolates
        row_sums[row_sums == 0] = 1.0
        W_std = W / row_sums[:, np.newaxis]
        return W_std

    elif style == "spectral":
        # Spectral normalization
        eigenvalues = np.linalg.eigvalsh(W)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            return W / max_eigenvalue
        else:
            return W

    else:
        raise ValueError(f"Unknown standardization style: {style}")


def compute_spatial_lag(W: np.ndarray, y: np.ndarray, standardize: bool = True) -> np.ndarray:
    """
    Compute spatial lag Wy.

    Parameters
    ----------
    W : np.ndarray
        Spatial weight matrix
    y : np.ndarray
        Variable to lag
    standardize : bool
        Whether to row-standardize W first

    Returns
    -------
    np.ndarray
        Spatial lag Wy
    """
    if standardize:
        W = standardize_spatial_weights(W, style="row")

    return W @ y


def permutation_inference(
    statistic_func, data: np.ndarray, n_permutations: int = 999, seed: int = None
) -> float:
    """
    Compute p-value using permutation inference.

    Parameters
    ----------
    statistic_func : callable
        Function that computes test statistic from data
    data : np.ndarray
        Original data
    n_permutations : int
        Number of permutations
    seed : int
        Random seed for reproducibility

    Returns
    -------
    float
        P-value from permutation test
    """
    if seed is not None:
        np.random.seed(seed)

    # Observed statistic
    observed = statistic_func(data)

    # Permutation statistics
    permuted_stats = []
    for _ in range(n_permutations):
        data_perm = np.random.permutation(data)
        stat_perm = statistic_func(data_perm)
        permuted_stats.append(stat_perm)

    # Two-sided p-value
    permuted_stats = np.array(permuted_stats)
    pvalue = np.mean(np.abs(permuted_stats) >= np.abs(observed))

    return pvalue
