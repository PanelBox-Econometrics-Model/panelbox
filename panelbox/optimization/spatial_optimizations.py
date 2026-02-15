"""
Performance optimizations for spatial panel models.

Includes caching, sparse operations, and JIT compilation for faster estimation.
"""

import warnings
from functools import lru_cache
from typing import Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse import linalg as sp_linalg

try:
    from numba import jit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn(
        "Numba not installed. Some performance optimizations will be unavailable. "
        "Install with: pip install numba",
        ImportWarning,
    )


class EigenvalueCache:
    """Cache for eigenvalues of spatial weight matrices."""

    def __init__(self):
        """Initialize the eigenvalue cache."""
        self._cache = {}

    def get_eigenvalues(self, W: np.ndarray, force_recalc: bool = False) -> np.ndarray:
        """
        Get eigenvalues of W, using cache if available.

        Parameters
        ----------
        W : np.ndarray
            Spatial weight matrix
        force_recalc : bool, default False
            If True, recalculate even if cached

        Returns
        -------
        eigenvalues : np.ndarray
            Eigenvalues of W
        """
        # Create a hash key for the matrix
        W_key = self._matrix_hash(W)

        if not force_recalc and W_key in self._cache:
            return self._cache[W_key]

        # Calculate eigenvalues
        if issparse(W):
            # For sparse matrices, compute only real eigenvalues
            eigenvalues = sp_linalg.eigs(
                W,
                k=min(W.shape[0] - 2, 100),  # Get top 100 eigenvalues
                return_eigenvectors=False,
                which="LR",  # Largest real part
            ).real
        else:
            eigenvalues = np.linalg.eigvals(W).real

        # Cache the result
        self._cache[W_key] = eigenvalues

        return eigenvalues

    def _matrix_hash(self, W: np.ndarray) -> int:
        """Create a hash key for a matrix."""
        # Use the matrix data buffer for hashing
        if issparse(W):
            # For sparse matrices, hash the data and indices
            return hash((W.data.tobytes(), W.indices.tobytes()))
        else:
            return hash(W.tobytes())

    def clear(self):
        """Clear the eigenvalue cache."""
        self._cache.clear()


# Global cache instance
_eigenvalue_cache = EigenvalueCache()


def get_eigenvalues_cached(W: np.ndarray) -> np.ndarray:
    """
    Get eigenvalues of W using global cache.

    Parameters
    ----------
    W : np.ndarray
        Spatial weight matrix

    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues of W
    """
    return _eigenvalue_cache.get_eigenvalues(W)


class SparseOperations:
    """Optimized operations for sparse spatial weight matrices."""

    @staticmethod
    def is_sparse_efficient(W: np.ndarray, threshold: float = 0.1) -> bool:
        """
        Check if sparse operations would be more efficient.

        Parameters
        ----------
        W : np.ndarray
            Spatial weight matrix
        threshold : float, default 0.1
            Sparsity threshold (fraction of non-zero elements)

        Returns
        -------
        bool
            True if sparse operations recommended
        """
        if issparse(W):
            return True

        nnz = np.count_nonzero(W)
        total = W.size
        sparsity = nnz / total

        return sparsity < threshold

    @staticmethod
    def to_sparse(W: np.ndarray) -> csr_matrix:
        """
        Convert to sparse format if beneficial.

        Parameters
        ----------
        W : np.ndarray
            Spatial weight matrix

        Returns
        -------
        W_sparse : csr_matrix or np.ndarray
            Sparse matrix if beneficial, otherwise original
        """
        if issparse(W):
            return W

        if SparseOperations.is_sparse_efficient(W):
            return csr_matrix(W)

        return W

    @staticmethod
    def spatial_lag(W: Union[np.ndarray, csr_matrix], y: np.ndarray) -> np.ndarray:
        """
        Compute spatial lag Wy efficiently.

        Parameters
        ----------
        W : np.ndarray or csr_matrix
            Spatial weight matrix
        y : np.ndarray
            Variable to lag

        Returns
        -------
        Wy : np.ndarray
            Spatial lag
        """
        if issparse(W):
            return W.dot(y)
        else:
            return W @ y

    @staticmethod
    def log_determinant(I_rho_W: Union[np.ndarray, csr_matrix], method: str = "eigen") -> float:
        """
        Compute log determinant efficiently.

        Parameters
        ----------
        I_rho_W : np.ndarray or csr_matrix
            I - rho*W matrix
        method : str, default 'eigen'
            Method: 'eigen', 'lu', or 'cholesky'

        Returns
        -------
        log_det : float
            Log determinant
        """
        if issparse(I_rho_W):
            if method == "lu":
                # Use sparse LU decomposition
                from scipy.sparse.linalg import splu

                lu = splu(I_rho_W)
                diagL = lu.L.diagonal()
                diagU = lu.U.diagonal()
                return np.log(np.abs(diagL)).sum() + np.log(np.abs(diagU)).sum()
            else:
                # Convert to dense for other methods
                I_rho_W = I_rho_W.toarray()

        if method == "eigen":
            eigenvalues = np.linalg.eigvals(I_rho_W)
            return np.log(np.abs(eigenvalues)).sum()
        elif method == "cholesky":
            try:
                L = np.linalg.cholesky(I_rho_W @ I_rho_W.T)
                return np.log(np.diagonal(L)).sum()
            except np.linalg.LinAlgError:
                # Fall back to eigenvalues
                return SparseOperations.log_determinant(I_rho_W, "eigen")
        else:
            # Default LU decomposition
            sign, log_det = np.linalg.slogdet(I_rho_W)
            return log_det


# Numba JIT compiled functions (if available)
if HAS_NUMBA:

    @jit(nopython=True, parallel=True, fastmath=True)
    def compute_spatial_lag_fast(
        W_data: np.ndarray, W_indices: np.ndarray, W_indptr: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Fast spatial lag computation using Numba JIT.

        For CSR sparse matrices.
        """
        n = len(y)
        result = np.zeros(n)

        for i in prange(n):
            row_start = W_indptr[i]
            row_end = W_indptr[i + 1]

            for j in range(row_start, row_end):
                col = W_indices[j]
                val = W_data[j]
                result[i] += val * y[col]

        return result

    @jit(nopython=True, fastmath=True)
    def compute_quadratic_form(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> float:
        """
        Compute y'X'WXy efficiently.

        Used in likelihood calculations.
        """
        n, k = X.shape
        WX = np.zeros((n, k))

        # Compute WX
        for i in range(n):
            for j in range(k):
                for l in range(n):
                    WX[i, j] += W[i, l] * X[l, j]

        # Compute X'WX
        XWX = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                for l in range(n):
                    XWX[i, j] += X[l, i] * WX[l, j]

        # Compute y'X'WXy
        result = 0.0
        temp = np.zeros(k)

        for i in range(k):
            for j in range(n):
                temp[i] += X[j, i] * y[j]

        for i in range(k):
            for j in range(k):
                result += temp[i] * XWX[i, j] * temp[j]

        return result

    @jit(nopython=True, parallel=True)
    def row_standardize_fast(W: np.ndarray) -> np.ndarray:
        """
        Fast row standardization using Numba.

        Parameters
        ----------
        W : np.ndarray
            Weight matrix to standardize

        Returns
        -------
        W_std : np.ndarray
            Row-standardized matrix
        """
        n = W.shape[0]
        W_std = W.copy()

        for i in prange(n):
            row_sum = 0.0
            for j in range(n):
                row_sum += W[i, j]

            if row_sum > 0:
                for j in range(n):
                    W_std[i, j] /= row_sum

        return W_std

else:
    # Fallback implementations without Numba
    def compute_spatial_lag_fast(W_data, W_indices, W_indptr, y):
        """Fallback spatial lag computation."""
        W = csr_matrix((W_data, W_indices, W_indptr))
        return W.dot(y)

    def compute_quadratic_form(X, W, y):
        """Fallback quadratic form computation."""
        WX = W @ X
        return y.T @ X.T @ WX @ X @ y

    def row_standardize_fast(W):
        """Fallback row standardization."""
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return W / row_sums


class ChebyshevApproximation:
    """
    Chebyshev approximation for log-determinant calculation.

    For very large N, exact eigenvalue calculation becomes prohibitive.
    """

    def __init__(self, order: int = 50):
        """
        Initialize Chebyshev approximation.

        Parameters
        ----------
        order : int, default 50
            Order of Chebyshev polynomial approximation
        """
        self.order = order

    def log_determinant(
        self, rho: float, W: Union[np.ndarray, csr_matrix], eigenvalues: Optional[np.ndarray] = None
    ) -> float:
        """
        Approximate log|I - rho*W| using Chebyshev polynomials.

        Parameters
        ----------
        rho : float
            Spatial autoregressive parameter
        W : np.ndarray or csr_matrix
            Spatial weight matrix
        eigenvalues : np.ndarray, optional
            Pre-computed eigenvalues of W

        Returns
        -------
        log_det : float
            Approximated log determinant
        """
        n = W.shape[0]

        if eigenvalues is None:
            # Sample eigenvalues using Lanczos iteration
            if issparse(W):
                k = min(self.order * 2, n - 2)
                eigenvalues = sp_linalg.eigsh(
                    W, k=k, return_eigenvectors=False, which="BE"  # Both ends
                )
            else:
                eigenvalues = np.linalg.eigvals(W).real

        # Compute log-det using eigenvalue approximation
        log_det = np.sum(np.log(1 - rho * eigenvalues))

        return log_det

    def trace_powers(self, W: Union[np.ndarray, csr_matrix], max_power: int = 10) -> np.ndarray:
        """
        Compute trace of powers of W efficiently.

        Used in some approximation schemes.

        Parameters
        ----------
        W : np.ndarray or csr_matrix
            Spatial weight matrix
        max_power : int, default 10
            Maximum power to compute

        Returns
        -------
        traces : np.ndarray
            Array of tr(W), tr(W^2), ..., tr(W^max_power)
        """
        traces = np.zeros(max_power)
        W_power = W.copy()

        for p in range(max_power):
            if issparse(W_power):
                traces[p] = W_power.diagonal().sum()
                W_power = W_power @ W
            else:
                traces[p] = np.trace(W_power)
                W_power = W_power @ W

        return traces


def optimize_spatial_model(model_class):
    """
    Decorator to add performance optimizations to spatial models.

    Usage:
        @optimize_spatial_model
        class SpatialLagModel:
            ...
    """
    # Add optimization methods to the model class
    original_init = model_class.__init__

    def optimized_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        # Add optimization attributes
        self._eigenvalue_cache = None
        self._use_sparse = False
        self._chebyshev = None

        # Check if sparse operations would help
        if hasattr(self, "W"):
            self._use_sparse = SparseOperations.is_sparse_efficient(self.W)
            if self._use_sparse:
                self.W = SparseOperations.to_sparse(self.W)

            # For large N, prepare Chebyshev approximation
            if self.W.shape[0] > 5000:
                self._chebyshev = ChebyshevApproximation()

    # Replace methods
    model_class.__init__ = optimized_init

    # Add optimized methods
    if not hasattr(model_class, "get_eigenvalues"):

        def get_eigenvalues(self):
            """Get cached eigenvalues."""
            if self._eigenvalue_cache is None:
                self._eigenvalue_cache = get_eigenvalues_cached(self.W)
            return self._eigenvalue_cache

        model_class.get_eigenvalues = get_eigenvalues

    return model_class
