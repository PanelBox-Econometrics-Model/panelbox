"""
Spatial weight matrix infrastructure integrated with PySAL.
"""

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse


class SpatialWeights:
    """
    Spatial weight matrix wrapper integrated with PySAL.

    This class provides methods for creating and manipulating spatial weight
    matrices for use in spatial econometric models.

    Attributes
    ----------
    matrix : np.ndarray or sparse matrix
        N×N spatial weight matrix
    n : int
        Number of spatial units (observations)
    normalized : bool
        Whether matrix is row-normalized
    sparse : bool
        Whether using sparse matrix representation
    """

    def __init__(
        self, matrix: Union[np.ndarray, csr_matrix], normalized: bool = False, validate: bool = True
    ):
        """
        Initialize spatial weight matrix.

        Parameters
        ----------
        matrix : array-like
            N×N spatial weight matrix
        normalized : bool
            Whether matrix is already row-normalized
        validate : bool
            Whether to validate matrix properties
        """
        # Convert to appropriate format
        if issparse(matrix):
            self.matrix = matrix
            self.sparse = True
        else:
            self.matrix = np.asarray(matrix)
            self.sparse = False

        self.n = self.matrix.shape[0]
        self.normalized = normalized

        # Cache for eigenvalues
        self._eigenvalues = None

        if validate:
            self._validate()

    def _validate(self):
        """Validate W matrix properties."""
        # Check square
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("W must be square")

        # Check diagonal (should be zero)
        if self.sparse:
            diag = self.matrix.diagonal()
        else:
            diag = np.diag(self.matrix)

        if not np.allclose(diag, 0):
            warnings.warn("W has non-zero diagonal; setting to zero")
            if self.sparse:
                self.matrix.setdiag(0)
            else:
                np.fill_diagonal(self.matrix, 0)

        # Check non-negative
        if self.sparse:
            if np.any(self.matrix.data < 0):
                raise ValueError("W contains negative values")
        else:
            if np.any(self.matrix < 0):
                raise ValueError("W contains negative values")

    @classmethod
    def from_contiguity(cls, gdf, criterion: str = "queen", **kwargs):
        """
        Create W from spatial contiguity.

        Parameters
        ----------
        gdf : GeoDataFrame
            Spatial units with geometry
        criterion : str
            'queen' (default) or 'rook' contiguity

        Returns
        -------
        SpatialWeights
            Weight matrix based on contiguity
        """
        try:
            from libpysal.weights import Queen, Rook
        except ImportError:
            raise ImportError(
                "libpysal required for contiguity weights. " "Install with: pip install libpysal"
            )

        if criterion.lower() == "queen":
            w = Queen.from_dataframe(gdf)
        elif criterion.lower() == "rook":
            w = Rook.from_dataframe(gdf)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        # Convert to matrix
        matrix = w.full()[0]
        return cls(matrix, **kwargs)

    @classmethod
    def from_distance(
        cls, coords: np.ndarray, threshold: float, p: float = 2.0, binary: bool = True, **kwargs
    ):
        """
        Create W from distance threshold.

        Parameters
        ----------
        coords : np.ndarray
            N×2 array of coordinates
        threshold : float
            Distance threshold for neighbors
        p : float
            Minkowski p-norm distance (1=Manhattan, 2=Euclidean)
        binary : bool
            If True, use binary weights; if False, use inverse distance

        Returns
        -------
        SpatialWeights
            Weight matrix based on distance
        """
        try:
            from libpysal.weights import DistanceBand
        except ImportError:
            raise ImportError("libpysal required for distance weights")

        w = DistanceBand(coords, threshold=threshold, p=p, binary=binary)
        matrix = w.full()[0]
        return cls(matrix, **kwargs)

    @classmethod
    def from_knn(cls, coords: np.ndarray, k: int = 5, **kwargs):
        """
        Create W from k-nearest neighbors.

        Parameters
        ----------
        coords : np.ndarray
            N×2 array of coordinates
        k : int
            Number of nearest neighbors

        Returns
        -------
        SpatialWeights
            Weight matrix based on k-NN
        """
        try:
            from libpysal.weights import KNN
        except ImportError:
            raise ImportError("libpysal required for k-NN weights")

        w = KNN.from_array(coords, k=k)
        matrix = w.full()[0]
        return cls(matrix, **kwargs)

    @classmethod
    def from_matrix(cls, array: Union[np.ndarray, list]):
        """
        Create W from numpy array or list.

        Parameters
        ----------
        array : array-like
            N×N weight matrix

        Returns
        -------
        SpatialWeights
        """
        return cls(np.asarray(array))

    def standardize(self, method: str = "row") -> "SpatialWeights":
        """
        Normalize spatial weight matrix.

        Parameters
        ----------
        method : str
            'row': row-standardization (each row sums to 1)
            'spectral': spectral normalization (max eigenvalue = 1)

        Returns
        -------
        SpatialWeights
            Self for method chaining
        """
        if method == "row":
            if self.sparse:
                # Row-standardize sparse matrix
                row_sums = np.asarray(self.matrix.sum(axis=1)).flatten()
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                diag = 1.0 / row_sums
                self.matrix = csr_matrix(np.diag(diag)) @ self.matrix
            else:
                # Row-standardize dense matrix
                row_sums = self.matrix.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                self.matrix = self.matrix / row_sums
            self.normalized = True

        elif method == "spectral":
            # Compute max eigenvalue
            if self._eigenvalues is None:
                self._compute_eigenvalues()
            max_eigenvalue = np.max(np.abs(self._eigenvalues.real))

            # Normalize
            self.matrix = self.matrix / max_eigenvalue
            self.normalized = True

        else:
            raise ValueError(f"Unknown method: {method}")

        # Clear eigenvalue cache after modification
        self._eigenvalues = None

        return self

    def _compute_eigenvalues(self):
        """Compute and cache eigenvalues."""
        if self.sparse:
            # Convert to dense for eigenvalue computation
            # (only for small matrices)
            if self.n > 1000:
                warnings.warn("Computing eigenvalues for large sparse matrix")
            self._eigenvalues = np.linalg.eigvals(self.to_dense())
        else:
            self._eigenvalues = np.linalg.eigvals(self.matrix)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Get eigenvalues of W."""
        if self._eigenvalues is None:
            self._compute_eigenvalues()
        return self._eigenvalues

    @property
    def s0(self) -> float:
        """Sum of all weights."""
        if self.sparse:
            return self.matrix.sum()
        return self.matrix.sum()

    @property
    def s1(self) -> float:
        """Sum of squared row + column sums (for Moran's I)."""
        if self.sparse:
            row_sums = np.asarray(self.matrix.sum(axis=1)).flatten()
            col_sums = np.asarray(self.matrix.sum(axis=0)).flatten()
        else:
            row_sums = self.matrix.sum(axis=1)
            col_sums = self.matrix.sum(axis=0)
        return 0.5 * ((row_sums + col_sums) ** 2).sum()

    @property
    def s2(self) -> float:
        """Sum of squared weights (for Moran's I)."""
        if self.sparse:
            return (self.matrix.data**2).sum()
        return (self.matrix**2).sum()

    def to_sparse(self) -> csr_matrix:
        """Convert to scipy sparse matrix."""
        if self.sparse:
            return self.matrix
        return csr_matrix(self.matrix)

    def to_dense(self) -> np.ndarray:
        """Convert to dense numpy array."""
        if self.sparse:
            return self.matrix.toarray()
        return self.matrix

    def spatial_lag(self, x: np.ndarray) -> np.ndarray:
        """
        Compute spatial lag Wx.

        Parameters
        ----------
        x : np.ndarray
            Vector or matrix to lag

        Returns
        -------
        np.ndarray
            Spatial lag of x
        """
        return self.matrix @ x

    def get_bounds(self) -> Tuple[float, float]:
        """
        Get bounds for spatial coefficient.

        For row-normalized W, bounds are typically close to (-1, 1).
        For general W, bounds are (1/λ_min, 1/λ_max).

        Returns
        -------
        tuple
            (lower_bound, upper_bound) for spatial parameter
        """
        eigenvalues = self.eigenvalues.real
        lambda_min = np.min(eigenvalues)
        lambda_max = np.max(eigenvalues)

        # Compute bounds
        if lambda_min < 0:
            rho_min = 1.0 / lambda_min
        else:
            rho_min = -0.99

        if lambda_max > 0:
            rho_max = 1.0 / lambda_max
        else:
            rho_max = 0.99

        # Clip for stability
        rho_min = max(rho_min, -0.99)
        rho_max = min(rho_max, 0.99)

        return (rho_min, rho_max)

    def plot(
        self,
        gdf=None,
        figsize: Tuple[float, float] = (10, 8),
        backend: str = "matplotlib",
        **kwargs,
    ):
        """
        Plot spatial connections.

        Parameters
        ----------
        gdf : GeoDataFrame, optional
            Spatial units to plot
        figsize : tuple
            Figure size
        backend : str
            'matplotlib' or 'plotly'
        """
        if backend == "matplotlib":
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=figsize)

            if gdf is not None:
                gdf.plot(ax=ax, edgecolor="black", facecolor="lightgray")

                # Plot connections
                centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry])

                # Draw edges
                for i in range(self.n):
                    for j in range(self.n):
                        if self.sparse:
                            weight = self.matrix[i, j]
                        else:
                            weight = self.matrix[i, j]

                        if weight > 0:
                            ax.plot(
                                [centroids[i, 0], centroids[j, 0]],
                                [centroids[i, 1], centroids[j, 1]],
                                "r-",
                                alpha=0.3,
                                linewidth=0.5,
                            )

            ax.set_title("Spatial Weight Matrix Connections")
            plt.show()

        elif backend == "plotly":
            import plotly.graph_objects as go

            # Implementation for plotly
            raise NotImplementedError("Plotly backend not yet implemented")

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def __repr__(self):
        """String representation."""
        return (
            f"SpatialWeights(n={self.n}, "
            f"normalized={self.normalized}, "
            f"sparse={self.sparse}, "
            f"s0={self.s0:.2f})"
        )

    def summary(self):
        """Print summary statistics."""
        print(f"Spatial Weight Matrix Summary")
        print(f"{'='*40}")
        print(f"Number of units:     {self.n}")
        print(f"Normalized:         {self.normalized}")
        print(f"Sparse:             {self.sparse}")
        print(f"Sum of weights (s0): {self.s0:.4f}")
        print(f"Sum (row+col)^2 (s1): {self.s1:.4f}")
        print(f"Sum of w^2 (s2):     {self.s2:.4f}")

        if self.normalized:
            print(f"Row sums:           All = 1.0")
        else:
            if self.sparse:
                row_sums = np.asarray(self.matrix.sum(axis=1)).flatten()
            else:
                row_sums = self.matrix.sum(axis=1)
            print(f"Row sums:           [{row_sums.min():.2f}, {row_sums.max():.2f}]")

        # Connectivity
        if self.sparse:
            n_edges = self.matrix.nnz
        else:
            n_edges = np.count_nonzero(self.matrix)
        print(f"Number of edges:     {n_edges}")
        print(f"Density:            {n_edges / (self.n * self.n):.4f}")
