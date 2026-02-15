"""
Spatial HAC covariance matrix estimator (Conley 1999).

This module implements the Spatial Heteroskedasticity and Autocorrelation
Consistent (HAC) covariance matrix estimator following Conley (1999).

The estimator is robust to:
- Spatial autocorrelation up to a specified distance
- Temporal autocorrelation up to a specified lag
- Heteroskedasticity

References
----------
Conley, T.G. (1999). "GMM Estimation with Cross Sectional Dependence."
    Journal of Econometrics, 92(1), 1-45.
Hsiang, S.M. (2010). "Temperatures and cyclones strongly associated with
    economic production in the Caribbean and Central America."
    PNAS, 107(35), 15367-15372.
"""

import warnings
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


class SpatialHAC:
    """
    Spatial HAC covariance matrix estimator (Conley 1999).

    This class implements heteroskedasticity and autocorrelation consistent
    standard errors for panel data with spatial and temporal correlation.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Geographic distances between entities (N×N)
    spatial_cutoff : float
        Maximum distance for spatial correlation (in same units as distance_matrix)
    temporal_cutoff : int
        Maximum lag for temporal correlation (default: 0)
    spatial_kernel : {'bartlett', 'uniform', 'triangular', 'epanechnikov'}
        Spatial kernel function
    temporal_kernel : {'bartlett', 'uniform', 'parzen', 'quadratic_spectral'}
        Temporal kernel function

    Attributes
    ----------
    distance_matrix : np.ndarray
        Stored distance matrix
    spatial_cutoff : float
        Spatial correlation cutoff
    temporal_cutoff : int
        Temporal correlation cutoff
    spatial_kernel : str
        Selected spatial kernel
    temporal_kernel : str
        Selected temporal kernel
    """

    def __init__(
        self,
        distance_matrix: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        spatial_kernel: str = "bartlett",
        temporal_kernel: str = "bartlett",
    ):
        """Initialize Spatial HAC estimator."""

        self.distance_matrix = distance_matrix
        self.spatial_cutoff = spatial_cutoff
        self.temporal_cutoff = temporal_cutoff
        self.spatial_kernel = spatial_kernel.lower()
        self.temporal_kernel = temporal_kernel.lower()

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input parameters."""
        # Check distance matrix
        if self.distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        # Check cutoffs
        if self.spatial_cutoff <= 0:
            raise ValueError("spatial_cutoff must be positive")

        if self.temporal_cutoff < 0:
            raise ValueError("temporal_cutoff must be non-negative")

        # Check kernels
        valid_spatial = ["bartlett", "uniform", "triangular", "epanechnikov"]
        if self.spatial_kernel not in valid_spatial:
            raise ValueError(f"spatial_kernel must be one of {valid_spatial}")

        valid_temporal = ["bartlett", "uniform", "parzen", "quadratic_spectral"]
        if self.temporal_kernel not in valid_temporal:
            raise ValueError(f"temporal_kernel must be one of {valid_temporal}")

    def _spatial_kernel_weight(self, distance: np.ndarray) -> np.ndarray:
        """
        Compute spatial kernel weights.

        Parameters
        ----------
        distance : np.ndarray
            Distance values

        Returns
        -------
        np.ndarray
            Kernel weights
        """
        # Normalize distance by cutoff
        u = distance / self.spatial_cutoff

        if self.spatial_kernel == "bartlett":
            # Bartlett kernel: K(u) = 1 - u if u <= 1, 0 otherwise
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "uniform":
            # Uniform kernel: K(u) = 1 if u <= 1, 0 otherwise
            weight = (u <= 1).astype(float)

        elif self.spatial_kernel == "triangular":
            # Triangular kernel (same as Bartlett)
            weight = np.maximum(1 - u, 0)

        elif self.spatial_kernel == "epanechnikov":
            # Epanechnikov kernel: K(u) = 0.75(1 - u²) if u <= 1
            weight = np.where(u <= 1, 0.75 * (1 - u**2), 0)

        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

        return weight

    def _temporal_kernel_weight(self, lag: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute temporal kernel weights.

        Parameters
        ----------
        lag : int or np.ndarray
            Time lag(s)

        Returns
        -------
        float or np.ndarray
            Kernel weight(s)
        """
        if self.temporal_cutoff == 0:
            # No temporal correlation
            return (lag == 0).astype(float) if isinstance(lag, np.ndarray) else float(lag == 0)

        # Normalize lag by cutoff
        u = np.abs(lag) / (self.temporal_cutoff + 1)

        if self.temporal_kernel == "bartlett":
            # Bartlett kernel
            weight = np.maximum(1 - u, 0)

        elif self.temporal_kernel == "uniform":
            # Uniform kernel
            weight = (np.abs(lag) <= self.temporal_cutoff).astype(float)

        elif self.temporal_kernel == "parzen":
            # Parzen kernel
            u_abs = np.abs(u)
            weight = np.where(
                u_abs <= 0.5,
                1 - 6 * u_abs**2 + 6 * u_abs**3,
                np.where(u_abs <= 1, 2 * (1 - u_abs) ** 3, 0),
            )

        elif self.temporal_kernel == "quadratic_spectral":
            # Quadratic spectral kernel
            x = 6 * np.pi * u / 5
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 3 / x**2 * (np.sin(x) / x - np.cos(x))
                weight = np.where(u == 0, 1, weight)  # Handle u=0 case

        else:
            raise ValueError(f"Unknown temporal kernel: {self.temporal_kernel}")

        return weight

    def compute(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        entity_index: np.ndarray,
        time_index: np.ndarray,
        small_sample_correction: bool = True,
    ) -> np.ndarray:
        """
        Compute Spatial HAC covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (NT × K)
        residuals : np.ndarray
            Residuals (NT × 1)
        entity_index : np.ndarray
            Entity identifiers for each observation
        time_index : np.ndarray
            Time identifiers for each observation
        small_sample_correction : bool
            Apply small sample correction

        Returns
        -------
        np.ndarray
            HAC covariance matrix (K × K)
        """
        # Get dimensions
        n_obs, k_vars = X.shape
        entities = np.unique(entity_index)
        times = np.unique(time_index)
        N = len(entities)
        T = len(times)

        if n_obs != N * T:
            warnings.warn(f"Unbalanced panel detected: {n_obs} obs != {N} × {T}")

        # Create mapping from entity/time to observation index
        obs_map = {}
        for idx, (e, t) in enumerate(zip(entity_index, time_index)):
            obs_map[(e, t)] = idx

        # Initialize meat matrix
        Omega = np.zeros((k_vars, k_vars))

        # Double loop over observations
        for t1_idx, t1 in enumerate(times):
            for t2_idx, t2 in enumerate(times):
                # Temporal weight
                lag = abs(t1_idx - t2_idx)
                if lag > self.temporal_cutoff:
                    continue
                w_temporal = self._temporal_kernel_weight(lag)

                if w_temporal == 0:
                    continue

                # Loop over entities
                for i_idx, entity_i in enumerate(entities):
                    # Check if observation exists
                    if (entity_i, t1) not in obs_map:
                        continue
                    obs_i_t1 = obs_map[(entity_i, t1)]

                    for j_idx, entity_j in enumerate(entities):
                        # Check if observation exists
                        if (entity_j, t2) not in obs_map:
                            continue
                        obs_j_t2 = obs_map[(entity_j, t2)]

                        # Spatial weight
                        distance = self.distance_matrix[i_idx, j_idx]
                        if distance > self.spatial_cutoff:
                            continue
                        w_spatial = self._spatial_kernel_weight(distance)

                        if w_spatial == 0:
                            continue

                        # Combined weight
                        weight = w_spatial * w_temporal

                        # Contribution to Omega
                        Omega += weight * (
                            residuals[obs_i_t1]
                            * residuals[obs_j_t2]
                            * np.outer(X[obs_i_t1], X[obs_j_t2])
                        )

        # Sandwich formula: V = (X'X)⁻¹ Ω (X'X)⁻¹
        XtX = X.T @ X

        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("X'X is singular, using pseudo-inverse")
            XtX_inv = np.linalg.pinv(XtX)

        # Compute sandwich
        V_hac = XtX_inv @ Omega @ XtX_inv

        # Small sample correction
        if small_sample_correction:
            correction_factor = n_obs / (n_obs - k_vars)
            V_hac *= correction_factor

        return V_hac

    @classmethod
    def from_coordinates(
        cls,
        coords: np.ndarray,
        spatial_cutoff: float,
        temporal_cutoff: int = 0,
        distance_metric: str = "haversine",
        **kwargs,
    ):
        """
        Create SpatialHAC from geographic coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Geographic coordinates (N × 2), typically (latitude, longitude)
        spatial_cutoff : float
            Maximum distance for spatial correlation
        temporal_cutoff : int
            Maximum lag for temporal correlation
        distance_metric : {'haversine', 'euclidean', 'manhattan'}
            Distance calculation method
        **kwargs
            Additional arguments passed to SpatialHAC constructor

        Returns
        -------
        SpatialHAC
            Initialized SpatialHAC object
        """
        if coords.shape[1] != 2:
            raise ValueError("coords must have shape (N, 2)")

        if distance_metric == "haversine":
            distance_matrix = cls._haversine_distance_matrix(coords)
        elif distance_metric in ["euclidean", "manhattan", "cityblock"]:
            distance_matrix = cdist(coords, coords, metric=distance_metric)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        return cls(distance_matrix, spatial_cutoff, temporal_cutoff, **kwargs)

    @staticmethod
    def _haversine_distance_matrix(coords: np.ndarray) -> np.ndarray:
        """
        Compute great circle distances between coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates in degrees (N × 2): [latitude, longitude]

        Returns
        -------
        np.ndarray
            Distance matrix in kilometers (N × N)
        """
        # Convert to radians
        coords_rad = np.radians(coords)
        lat = coords_rad[:, 0]
        lon = coords_rad[:, 1]

        # Haversine formula
        N = len(coords)
        distance_matrix = np.zeros((N, N))

        for i in range(N):
            # Vectorized computation for all j
            dlat = lat - lat[i]
            dlon = lon - lon[i]

            a = np.sin(dlat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))

            # Earth radius in kilometers
            R = 6371.0
            distance_matrix[i] = R * c

        return distance_matrix

    def compare_with_standard_errors(
        self, X: np.ndarray, residuals: np.ndarray, entity_index: np.ndarray, time_index: np.ndarray
    ) -> dict:
        """
        Compare Spatial HAC with standard OLS and clustered standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            Residuals
        entity_index : np.ndarray
            Entity identifiers
        time_index : np.ndarray
            Time identifiers

        Returns
        -------
        dict
            Dictionary with different variance estimates
        """
        n_obs, k_vars = X.shape

        # OLS standard errors (homoskedastic)
        sigma2 = np.sum(residuals**2) / (n_obs - k_vars)
        XtX_inv = np.linalg.inv(X.T @ X)
        V_ols = sigma2 * XtX_inv

        # White robust standard errors (heteroskedastic)
        Omega_white = X.T @ np.diag(residuals.flatten() ** 2) @ X
        V_white = XtX_inv @ Omega_white @ XtX_inv

        # Spatial HAC
        V_hac = self.compute(X, residuals, entity_index, time_index)

        # Extract standard errors
        se_ols = np.sqrt(np.diag(V_ols))
        se_white = np.sqrt(np.diag(V_white))
        se_hac = np.sqrt(np.diag(V_hac))

        return {
            "V_ols": V_ols,
            "V_white": V_white,
            "V_hac": V_hac,
            "se_ols": se_ols,
            "se_white": se_white,
            "se_hac": se_hac,
            "se_ratio_hac_ols": se_hac / se_ols,
            "se_ratio_hac_white": se_hac / se_white,
        }


class DriscollKraayComparison:
    """
    Compare Spatial HAC with Driscoll-Kraay standard errors.

    Both methods are robust to spatial correlation, but:
    - Spatial HAC uses explicit geographic distance
    - Driscoll-Kraay assumes uniform spatial correlation
    """

    @staticmethod
    def compare(
        spatial_hac_se: np.ndarray,
        driscoll_kraay_se: np.ndarray,
        param_names: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Compare Spatial HAC and Driscoll-Kraay standard errors.

        Parameters
        ----------
        spatial_hac_se : np.ndarray
            Standard errors from Spatial HAC
        driscoll_kraay_se : np.ndarray
            Standard errors from Driscoll-Kraay
        param_names : list, optional
            Parameter names for the DataFrame

        Returns
        -------
        pd.DataFrame
            Comparison of standard errors
        """
        if param_names is None:
            param_names = [f"param_{i}" for i in range(len(spatial_hac_se))]

        comparison_df = pd.DataFrame(
            {
                "parameter": param_names,
                "Spatial_HAC": spatial_hac_se,
                "Driscoll_Kraay": driscoll_kraay_se,
                "ratio": spatial_hac_se / driscoll_kraay_se,
                "pct_diff": 100 * (spatial_hac_se - driscoll_kraay_se) / driscoll_kraay_se,
            }
        )

        # Add summary statistics
        comparison_df.loc["mean"] = comparison_df.mean(numeric_only=True)
        comparison_df.loc["median"] = comparison_df.median(numeric_only=True)
        comparison_df.loc["std"] = comparison_df.std(numeric_only=True)

        return comparison_df
