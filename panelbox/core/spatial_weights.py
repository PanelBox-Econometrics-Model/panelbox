"""
Spatial Weight Matrix Infrastructure

This module provides the foundation for spatial econometric models through
spatial weight matrices that define neighborhood relationships between spatial units.

Classes:
    SpatialWeights: Main class for handling spatial weight matrices
"""

import warnings
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import cdist


class SpatialWeights:
    """
    Spatial weight matrix wrapper integrado com PySAL.

    This class provides a flexible interface for creating and manipulating
    spatial weight matrices used in spatial econometric models.

    Attributes
    ----------
    matrix : np.ndarray or sparse matrix
        N×N spatial weight matrix
    n : int
        Number of observations (spatial units)
    s0 : float
        Sum of all weights
    normalized : bool
        Whether matrix is row-normalized
    eigenvalues : np.ndarray or None
        Cached eigenvalues of the weight matrix

    Examples
    --------
    >>> # Create from contiguity
    >>> import geopandas as gpd
    >>> gdf = gpd.read_file("regions.shp")
    >>> W = SpatialWeights.from_contiguity(gdf, criterion='queen')
    >>> W.standardize('row')

    >>> # Create from distance
    >>> coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    >>> W = SpatialWeights.from_distance(coords, threshold=1.5)

    >>> # Create from k-nearest neighbors
    >>> W = SpatialWeights.from_knn(coords, k=2)
    """

    def __init__(self, matrix: Union[np.ndarray, csr_matrix], normalized: bool = False):
        """
        Initialize spatial weight matrix.

        Parameters
        ----------
        matrix : np.ndarray or sparse matrix
            Square spatial weight matrix
        normalized : bool, default=False
            Whether the matrix is already normalized
        """
        self.matrix = matrix
        self.n = matrix.shape[0]
        self.normalized = normalized
        self._eigenvalues = None
        self._validate()

    def _validate(self):
        """Validate W matrix properties."""
        # Check square
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("W must be square")

        # Check diagonal (should be zero)
        if issparse(self.matrix):
            diag = self.matrix.diagonal()
        else:
            diag = np.diag(self.matrix)

        if not np.allclose(diag, 0):
            warnings.warn("W has non-zero diagonal; setting to zero")
            if issparse(self.matrix):
                self.matrix.setdiag(0)
            else:
                np.fill_diagonal(self.matrix, 0)

        # Check non-negative
        if issparse(self.matrix):
            if (self.matrix.data < 0).any():
                raise ValueError("W contains negative values")
        else:
            if np.any(self.matrix < 0):
                raise ValueError("W contains negative values")

    @classmethod
    def from_contiguity(cls, gdf, criterion: Literal["queen", "rook"] = "queen"):
        """
        Create W from spatial contiguity.

        Parameters
        ----------
        gdf : GeoDataFrame
            Spatial units with geometry
        criterion : {'queen', 'rook'}, default='queen'
            Contiguity criterion
            - 'queen': shared edges or vertices
            - 'rook': shared edges only

        Returns
        -------
        SpatialWeights
            Weight matrix based on contiguity
        """
        try:
            from libpysal.weights import Queen, Rook
        except ImportError:
            raise ImportError(
                "libpysal is required for contiguity weights. " "Install with: pip install libpysal"
            )

        if criterion == "queen":
            w = Queen.from_dataframe(gdf)
        elif criterion == "rook":
            w = Rook.from_dataframe(gdf)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        # Convert to matrix
        matrix = w.full()[0]
        return cls(matrix)

    @classmethod
    def from_distance(cls, coords: np.ndarray, threshold: float, p: float = 2, binary: bool = True):
        """
        Create W from distance threshold.

        Parameters
        ----------
        coords : np.ndarray
            N×2 array of coordinates
        threshold : float
            Distance threshold for neighbors
        p : float, default=2
            Minkowski p-norm (2 for Euclidean)
        binary : bool, default=True
            If True, use binary weights; if False, use inverse distance

        Returns
        -------
        SpatialWeights
            Weight matrix based on distance
        """
        n = len(coords)

        # Compute pairwise distances
        distances = cdist(coords, coords, metric="minkowski", p=p)

        # Create weight matrix
        if binary:
            # Binary weights
            matrix = (distances <= threshold).astype(float)
        else:
            # Inverse distance weights
            matrix = np.zeros_like(distances)
            mask = (distances > 0) & (distances <= threshold)
            matrix[mask] = 1.0 / distances[mask]

        # Set diagonal to zero
        np.fill_diagonal(matrix, 0)

        return cls(matrix)

    @classmethod
    def from_knn(cls, coords: np.ndarray, k: int = 5):
        """
        Create W from k-nearest neighbors.

        Parameters
        ----------
        coords : np.ndarray
            N×2 array of coordinates
        k : int, default=5
            Number of nearest neighbors

        Returns
        -------
        SpatialWeights
            Weight matrix based on k-nearest neighbors
        """
        try:
            from libpysal.weights import KNN
        except ImportError:
            # Fallback implementation
            n = len(coords)
            distances = cdist(coords, coords)
            matrix = np.zeros((n, n))

            for i in range(n):
                # Get k nearest neighbors (excluding self)
                neighbors = np.argsort(distances[i])[1 : k + 1]
                matrix[i, neighbors] = 1.0

            return cls(matrix)

        # Use libpysal if available
        w = KNN.from_array(coords, k=k)
        matrix = w.full()[0]
        return cls(matrix)

    @classmethod
    def from_matrix(cls, array: Union[np.ndarray, list]):
        """
        Create W from numpy array or list.

        Parameters
        ----------
        array : np.ndarray or list
            Square weight matrix

        Returns
        -------
        SpatialWeights
            Weight matrix object
        """
        return cls(np.asarray(array))

    @classmethod
    def from_sparse(cls, sparse_matrix: csr_matrix):
        """
        Create W from scipy sparse matrix.

        Parameters
        ----------
        sparse_matrix : scipy.sparse matrix
            Sparse weight matrix

        Returns
        -------
        SpatialWeights
            Weight matrix object
        """
        return cls(sparse_matrix)

    def standardize(self, method: Literal["row", "spectral"] = "row"):
        """
        Normalize spatial weight matrix.

        Parameters
        ----------
        method : {'row', 'spectral'}, default='row'
            Normalization method
            - 'row': row-standardization (each row sums to 1)
            - 'spectral': spectral normalization (max eigenvalue = 1)

        Returns
        -------
        self
            Modified in place for chaining
        """
        if method == "row":
            # Row standardization
            if issparse(self.matrix):
                row_sums = np.asarray(self.matrix.sum(axis=1)).flatten()
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                # Create diagonal matrix for division
                from scipy.sparse import diags

                D = diags(1.0 / row_sums)
                self.matrix = D @ self.matrix
            else:
                row_sums = self.matrix.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                self.matrix = self.matrix / row_sums

            self.normalized = True

        elif method == "spectral":
            # Spectral normalization
            if issparse(self.matrix):
                from scipy.sparse.linalg import eigs

                # Get largest eigenvalue
                eigenvalues, _ = eigs(self.matrix, k=1, which="LM")
                max_eigenvalue = np.abs(eigenvalues[0])
            else:
                eigenvalues = np.linalg.eigvals(self.matrix)
                max_eigenvalue = np.max(np.abs(eigenvalues))

            self.matrix = self.matrix / max_eigenvalue
            self.normalized = True
            self._eigenvalues = None  # Reset cached eigenvalues

        else:
            raise ValueError(f"Unknown method: {method}")

        return self

    @property
    def s0(self) -> float:
        """Sum of all weights."""
        if issparse(self.matrix):
            return self.matrix.sum()
        return self.matrix.sum()

    @property
    def s1(self) -> float:
        """Sum of squared row + column sums (for Moran's I)."""
        if issparse(self.matrix):
            row_sums = np.asarray(self.matrix.sum(axis=1)).flatten()
            col_sums = np.asarray(self.matrix.sum(axis=0)).flatten()
        else:
            row_sums = self.matrix.sum(axis=1)
            col_sums = self.matrix.sum(axis=0)

        return 0.5 * ((row_sums + col_sums) ** 2).sum()

    @property
    def s2(self) -> float:
        """Sum of squared weights (for Moran's I)."""
        if issparse(self.matrix):
            return (self.matrix.data**2).sum()
        return (self.matrix**2).sum()

    @property
    def eigenvalues(self) -> np.ndarray:
        """
        Get eigenvalues of the weight matrix.

        Returns
        -------
        np.ndarray
            Eigenvalues of W
        """
        if self._eigenvalues is None:
            if issparse(self.matrix):
                # For sparse matrices, compute only a subset of eigenvalues
                from scipy.sparse.linalg import eigs

                # Get extremal eigenvalues
                k = min(6, self.n - 2)
                eigenvalues, _ = eigs(self.matrix, k=k, which="BE")
                self._eigenvalues = eigenvalues.real
            else:
                self._eigenvalues = np.linalg.eigvals(self.matrix).real

        return self._eigenvalues

    def to_sparse(self) -> csr_matrix:
        """
        Convert to scipy sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse representation of weight matrix
        """
        if issparse(self.matrix):
            return self.matrix
        return csr_matrix(self.matrix)

    def to_dense(self) -> np.ndarray:
        """
        Convert to dense numpy array.

        Returns
        -------
        np.ndarray
            Dense representation of weight matrix
        """
        if issparse(self.matrix):
            return self.matrix.toarray()
        return self.matrix

    def get_neighbors(self, i: int) -> np.ndarray:
        """
        Get neighbors of unit i.

        Parameters
        ----------
        i : int
            Index of spatial unit

        Returns
        -------
        np.ndarray
            Indices of neighbors
        """
        if issparse(self.matrix):
            row = self.matrix.getrow(i).toarray().flatten()
        else:
            row = self.matrix[i]

        return np.where(row > 0)[0]

    def plot(
        self,
        gdf=None,
        coords=None,
        figsize: Tuple[int, int] = (10, 8),
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        show_edges: bool = True,
        edge_alpha: float = 0.5,
    ):
        """
        Plot spatial connections.

        Parameters
        ----------
        gdf : GeoDataFrame, optional
            Spatial units to plot with geometry
        coords : np.ndarray, optional
            N×2 array of coordinates (if gdf not provided)
        figsize : tuple, default=(10, 8)
            Figure size for matplotlib
        backend : {'matplotlib', 'plotly'}, default='matplotlib'
            Plotting backend
        show_edges : bool, default=True
            Whether to show edges between neighbors
        edge_alpha : float, default=0.5
            Transparency of edges

        Returns
        -------
        Figure object or None
            Depends on backend
        """
        if backend == "matplotlib":
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=figsize)

            # Plot spatial units
            if gdf is not None:
                gdf.plot(ax=ax, edgecolor="black", facecolor="lightblue")

                # Get centroids for connections
                centroids = gdf.geometry.centroid
                coords = np.array([[p.x, p.y] for p in centroids])

            elif coords is not None:
                ax.scatter(
                    coords[:, 0], coords[:, 1], s=50, c="lightblue", edgecolor="black", zorder=3
                )
            else:
                raise ValueError("Either gdf or coords must be provided")

            # Plot connections
            if show_edges:
                for i in range(self.n):
                    neighbors = self.get_neighbors(i)
                    for j in neighbors:
                        if i < j:  # Avoid duplicate edges
                            ax.plot(
                                [coords[i, 0], coords[j, 0]],
                                [coords[i, 1], coords[j, 1]],
                                "k-",
                                alpha=edge_alpha,
                                linewidth=0.5,
                                zorder=1,
                            )

            ax.set_title(f"Spatial Connections (N={self.n}, edges={int(self.s0/2)})")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")

            plt.tight_layout()
            return fig

        elif backend == "plotly":
            import plotly.graph_objects as go

            if coords is None:
                if gdf is not None:
                    centroids = gdf.geometry.centroid
                    coords = np.array([[p.x, p.y] for p in centroids])
                else:
                    raise ValueError("Either gdf or coords must be provided")

            # Create edge trace
            edge_trace = go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(width=0.5, color="gray"),
                hoverinfo="none",
                showlegend=False,
            )

            for i in range(self.n):
                neighbors = self.get_neighbors(i)
                for j in neighbors:
                    if i < j:
                        edge_trace["x"] += (coords[i, 0], coords[j, 0], None)
                        edge_trace["y"] += (coords[i, 1], coords[j, 1], None)

            # Create node trace
            node_trace = go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker=dict(size=10, color="lightblue", line=dict(width=1, color="black")),
                text=[f"Unit {i}" for i in range(self.n)],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )

            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace])

            fig.update_layout(
                title=f"Spatial Connections (N={self.n}, edges={int(self.s0/2)})",
                xaxis=dict(title="X Coordinate"),
                yaxis=dict(title="Y Coordinate"),
                hovermode="closest",
                height=600,
                showlegend=False,
            )

            return fig

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def summary(self) -> pd.DataFrame:
        """
        Generate summary statistics for the weight matrix.

        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        # Calculate statistics
        if issparse(self.matrix):
            row_sums = np.asarray(self.matrix.sum(axis=1)).flatten()
            nonzero = self.matrix.nnz
        else:
            row_sums = self.matrix.sum(axis=1)
            nonzero = np.count_nonzero(self.matrix)

        # Create summary
        summary = {
            "N (spatial units)": self.n,
            "Non-zero weights": nonzero,
            "Density (%)": 100 * nonzero / (self.n * self.n),
            "Row-normalized": self.normalized,
            "Sum of weights (S0)": self.s0,
            "Min row sum": row_sums.min(),
            "Mean row sum": row_sums.mean(),
            "Max row sum": row_sums.max(),
            "Min neighbors": np.sum(self.matrix > 0, axis=1).min(),
            "Mean neighbors": np.mean(np.sum(self.matrix > 0, axis=1)),
            "Max neighbors": np.sum(self.matrix > 0, axis=1).max(),
        }

        return pd.DataFrame([summary]).T.rename(columns={0: "Value"})

    def __repr__(self) -> str:
        """String representation."""
        sparse_str = " (sparse)" if issparse(self.matrix) else ""
        norm_str = " [row-normalized]" if self.normalized else ""
        return (
            f"SpatialWeights(n={self.n}, edges={int(self.s0)}, "
            f"density={100*self.s0/(self.n*self.n):.1f}%"
            f"{sparse_str}{norm_str})"
        )

    def __str__(self) -> str:
        """Detailed string representation."""
        return self.summary().to_string()
