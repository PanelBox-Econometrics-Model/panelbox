"""
Base class for spatial panel models.

This module provides the foundation for spatial econometric models
with panel data, including utilities for spatial weight matrices,
log-determinant computation, and spatial transformations.
"""

import warnings
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import eigs, splu

from panelbox.core.base_model import PanelModel
from panelbox.core.spatial_weights import SpatialWeights


class SpatialPanelModel(PanelModel):
    """
    Base class for spatial panel models.

    This class extends PanelModel with spatial econometric capabilities,
    including spatial weight matrix handling, log-determinant computation,
    and spatial transformations.

    Parameters
    ----------
    formula : str
        Patsy formula for the model
    data : pd.DataFrame
        Panel data with MultiIndex (entity, time)
    entity_col : str
        Name of entity identifier column
    time_col : str
        Name of time identifier column
    W : np.ndarray or SpatialWeights
        Spatial weight matrix (N×N)
    weights : np.ndarray, optional
        Observation weights for weighted estimation

    Attributes
    ----------
    W : np.ndarray
        Raw spatial weight matrix
    W_normalized : np.ndarray
        Row-normalized spatial weight matrix
    n_entities : int
        Number of spatial units
    _W_eigenvalues : np.ndarray or None
        Cached eigenvalues of W for log-det computation
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        W: Union[np.ndarray, SpatialWeights],
        weights: Optional[np.ndarray] = None,
    ):
        """Initialize spatial panel model."""
        super().__init__(formula, data, entity_col, time_col, weights)

        # Add compatibility properties
        if hasattr(self, "data"):
            # Store column names
            self.entity_col = entity_col
            self.time_col = time_col
            # New PanelModel interface
            self.entity_ids = self.data.data[entity_col].values
            self.entities = self.data.entities
            self.time_ids = self.data.data[time_col].values
            self.T = self.data.n_periods
            self.n_periods = self.data.n_periods
            self.n_obs = self.data.n_obs

            # Build design matrices for compatibility
            y, X = self.formula_parser.build_design_matrices(
                self.data.data, return_type="dataframe"
            )
            self.endog = y
            self.exog = X

        # Validate and store W
        self.W = self._validate_weight_matrix(W)
        self.W_normalized = self._normalize_weights(self.W)
        if hasattr(self, "entities"):
            self.n_entities = len(self.entities)
        else:
            self.n_entities = self.data.n_entities

        # Cache eigenvalues for log-det computation
        self._W_eigenvalues = None
        self._sparse_W = None

    def _validate_weight_matrix(self, W: Union[np.ndarray, SpatialWeights]) -> np.ndarray:
        """
        Validate spatial weight matrix.

        Parameters
        ----------
        W : np.ndarray or SpatialWeights
            Spatial weight matrix

        Returns
        -------
        np.ndarray
            Validated weight matrix

        Raises
        ------
        ValueError
            If W dimensions don't match number of entities
            If W contains negative values
        """
        if isinstance(W, SpatialWeights):
            W_matrix = W.to_dense()
        else:
            W_matrix = np.asarray(W)

        # Check dimensions
        N = len(self.entities)
        if W_matrix.shape != (N, N):
            raise ValueError(f"W must be {N}×{N}, got {W_matrix.shape}")

        # Check diagonal
        if not np.allclose(np.diag(W_matrix), 0):
            warnings.warn("W has non-zero diagonal; setting to zero")
            np.fill_diagonal(W_matrix, 0)

        # Check non-negative
        if np.any(W_matrix < 0):
            raise ValueError("W contains negative values")

        return W_matrix

    def _normalize_weights(
        self, W: np.ndarray, method: Literal["row", "none"] = "row"
    ) -> np.ndarray:
        """
        Normalize spatial weight matrix.

        Parameters
        ----------
        W : np.ndarray
            Weight matrix to normalize
        method : {'row', 'none'}, default='row'
            Normalization method

        Returns
        -------
        np.ndarray
            Normalized weight matrix
        """
        if method == "row":
            row_sums = W.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            return W / row_sums
        else:
            return W.copy()

    def _within_transformation(
        self, X: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> np.ndarray:
        """
        Apply within transformation (entity demeaning).

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame, optional
            Data to transform (uses self.exog if None)

        Returns
        -------
        np.ndarray
            Within-transformed data
        """
        if X is None:
            X = self.exog

        # Normalise: convert Series and ndarrays to DataFrame
        if isinstance(X, pd.Series):
            # 1-D Series → 2-D DataFrame with one column
            X = X.to_frame()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Now X is a DataFrame (possibly from the original exog DataFrame)
        X_df = X.copy()
        X_df[self.entity_col] = self.entity_ids

        # Demean by entity
        cols = [c for c in X_df.columns if c != self.entity_col]
        X_demeaned = X_df.groupby(self.entity_col)[cols].transform(lambda x: x - x.mean())

        # Return 1-D array if input was effectively 1-D (single column)
        if X_demeaned.shape[1] == 1:
            return X_demeaned.values.flatten()
        return X_demeaned.values

    def _spatial_lag(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute spatial lag WX.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Variable to lag spatially

        Returns
        -------
        np.ndarray
            Spatial lag of X
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Handle panel structure
        if X.ndim == 1:
            # Single cross-section
            if len(X) == self.n_entities:
                return self.W_normalized @ X

            # Panel data: need to apply W to each time period
            T = self.T
            N = self.n_entities

            # Reshape to (T, N)
            X_reshaped = X.reshape(T, N)
            WX = np.zeros_like(X_reshaped)
            for t in range(T):
                WX[t] = self.W_normalized @ X_reshaped[t]
            return WX.flatten()

        # Multiple variables
        T = self.T
        N = self.n_entities
        K = X.shape[1] if X.ndim == 2 else 1

        if len(X) == N * T:
            X_reshaped = X.reshape(T, N, K)
            WX = np.zeros_like(X_reshaped)
            for t in range(T):
                for k in range(K):
                    WX[t, :, k] = self.W_normalized @ X_reshaped[t, :, k]
            return WX.reshape(N * T, K)

        return self.W_normalized @ X

    def _log_det_jacobian(
        self,
        rho: float,
        W: Optional[np.ndarray] = None,
        method: Literal["auto", "eigenvalue", "sparse_lu", "chebyshev"] = "auto",
    ) -> float:
        """
        Compute log|I - ρW| efficiently.

        This is the log-determinant of the Jacobian term that appears
        in spatial models' likelihood functions.

        Parameters
        ----------
        rho : float
            Spatial parameter
        W : np.ndarray, optional
            Weight matrix (uses self.W_normalized if None)
        method : str, default='auto'
            Computation method:
            - 'auto': Select based on matrix size
            - 'eigenvalue': Use eigenvalues (N < 1000)
            - 'sparse_lu': Use sparse LU decomposition (1000 ≤ N < 10000)
            - 'chebyshev': Use Chebyshev approximation (N ≥ 10000)

        Returns
        -------
        float
            Log-determinant value

        Notes
        -----
        For SAR model: y = ρWy + Xβ + ε
        The log-likelihood includes the term log|I - ρW|
        """
        if W is None:
            W = self.W_normalized

        N = W.shape[0]

        # Auto-select method
        if method == "auto":
            if N < 1000:
                method = "eigenvalue"
            elif N < 10000:
                method = "sparse_lu"
            else:
                method = "chebyshev"

        if method == "eigenvalue":
            # Cache eigenvalues
            if self._W_eigenvalues is None:
                self._W_eigenvalues = np.linalg.eigvals(W)

            return np.sum(np.log(1 - rho * self._W_eigenvalues.real))

        elif method == "sparse_lu":
            # Convert to sparse if needed
            if self._sparse_W is None:
                self._sparse_W = csc_matrix(W)

            I_rhoW = csc_matrix(np.eye(N)) - rho * self._sparse_W
            lu = splu(I_rhoW)
            return np.sum(np.log(np.abs(lu.U.diagonal())))

        elif method == "chebyshev":
            # Chebyshev approximation (Pace & Barry 1997)
            # Simplified implementation
            if self._W_eigenvalues is None:
                # Get extremal eigenvalues for bounds
                if issparse(W):
                    eigenvalues, _ = eigs(W, k=2, which="BE")
                    self._W_eigenvalues = eigenvalues.real
                else:
                    eigenvalues = np.linalg.eigvals(W)
                    self._W_eigenvalues = eigenvalues.real

            # Approximate using Taylor expansion
            # log|I - ρW| ≈ -tr(ρW) - tr((ρW)²)/2 - ...
            trace1 = np.trace(W)  # Should be 0 for normalized W
            W2 = W @ W
            trace2 = np.trace(W2)

            log_det = -rho * trace1 - (rho**2 * trace2) / 2

            # Add higher order terms for better approximation
            if abs(rho) < 0.5:
                W3 = W2 @ W
                trace3 = np.trace(W3)
                log_det -= (rho**3 * trace3) / 3

            return log_det

        else:
            raise ValueError(f"Unknown method: {method}")

    def _spatial_coefficient_bounds(self, W: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Compute bounds for spatial parameter.

        For stability, the spatial parameter ρ must satisfy:
        1/λ_min < ρ < 1/λ_max

        Parameters
        ----------
        W : np.ndarray, optional
            Weight matrix (uses self.W_normalized if None)

        Returns
        -------
        tuple of float
            (rho_min, rho_max) bounds for spatial parameter
        """
        if W is None:
            W = self.W_normalized

        # Cache eigenvalues
        if self._W_eigenvalues is None:
            if W.shape[0] < 1000:
                self._W_eigenvalues = np.linalg.eigvals(W).real
            else:
                # For large matrices, get extremal eigenvalues only
                from scipy.sparse.linalg import eigs

                W_sparse = csc_matrix(W) if not issparse(W) else W
                eigenvalues, _ = eigs(W_sparse, k=2, which="BE")
                self._W_eigenvalues = eigenvalues.real

        eigenvalues = self._W_eigenvalues
        lambda_min = np.min(eigenvalues)
        lambda_max = np.max(eigenvalues)

        # Compute bounds
        if lambda_min < 0:
            rho_min = 1 / lambda_min
        else:
            rho_min = -0.99

        if lambda_max > 0:
            rho_max = 1 / lambda_max
        else:
            rho_max = 0.99

        # Clip for stability
        rho_min = max(rho_min, -0.99)
        rho_max = min(rho_max, 0.99)

        return (rho_min, rho_max)

    def _compute_spatial_instruments(self, X: np.ndarray, n_lags: int = 2) -> np.ndarray:
        """
        Compute spatial instruments [X, WX, W²X, ...].

        Parameters
        ----------
        X : np.ndarray
            Exogenous variables
        n_lags : int, default=2
            Number of spatial lags to include

        Returns
        -------
        np.ndarray
            Matrix of instruments
        """
        instruments = [X]
        WkX = X.copy()

        for _ in range(n_lags):
            WkX = self._spatial_lag(WkX)
            instruments.append(WkX)

        return np.hstack(instruments)

    def _validate_panel_structure(self):
        """
        Validate that panel is balanced for spatial models.

        Raises
        ------
        ValueError
            If panel is unbalanced
        """
        # Check if panel is balanced
        # self.data is a PanelData object; its underlying DataFrame is self.data.data
        raw_data = self.data.data if hasattr(self.data, "data") else self.data
        entity_counts = raw_data.groupby(self.entity_col).size()
        if not all(entity_counts == self.T):
            raise ValueError("Spatial models require balanced panels")

        # Check that W dimensions match entities
        if self.W.shape[0] != self.n_entities:
            raise ValueError(
                f"W dimensions ({self.W.shape[0]}) don't match "
                f"number of entities ({self.n_entities})"
            )

    def summary(self) -> str:
        """
        Generate model summary.

        Returns
        -------
        str
            Summary string
        """
        summary = super().summary()

        # Add spatial-specific information
        spatial_info = f"""
Spatial Information:
--------------------
Number of spatial units: {self.n_entities}
Spatial weight matrix density: {100 * np.count_nonzero(self.W) / (self.n_entities**2):.2f}%
Row-normalized: Yes
"""
        return summary + spatial_info

    def plot_spatial_connections(self, **kwargs):
        """
        Plot spatial weight matrix connections.

        Parameters
        ----------
        **kwargs
            Arguments passed to SpatialWeights.plot()

        Returns
        -------
        Figure object
        """
        W_obj = SpatialWeights.from_matrix(self.W_normalized)
        return W_obj.plot(**kwargs)
