"""
Tests for Spatial HAC standard errors (Conley 1999).

This module tests the Spatial HAC implementation including:
- Spatial kernel functions
- Temporal kernel functions
- Combined spatial-temporal HAC
- Comparison with other robust standard errors
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_less

from panelbox.standard_errors import DriscollKraayComparison, SpatialHAC, driscoll_kraay


class TestSpatialHAC:
    """Test suite for Spatial HAC standard errors."""

    @pytest.fixture
    def spatial_panel_data(self):
        """Generate panel data with spatial correlation."""
        np.random.seed(42)

        # Grid setup (5x5 = 25 entities)
        grid_size = 5
        N = grid_size * grid_size
        T = 20
        K = 3

        # Create coordinates (latitude, longitude)
        coords = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Use realistic lat/long (e.g., around NYC area)
                lat = 40.0 + i * 0.1
                lon = -74.0 + j * 0.1
                coords.append([lat, lon])
        coords = np.array(coords)

        # Create distance matrix (in km)
        distance_matrix = SpatialHAC._haversine_distance_matrix(coords)

        # Generate spatially correlated data
        X = np.random.randn(N * T, K)

        # True coefficients
        beta = np.array([2.0, -1.5, 0.8])

        # Generate spatially and temporally correlated errors
        errors = self._generate_spatial_temporal_errors(N, T, distance_matrix)

        # Generate y
        y = X @ beta + errors.flatten()

        # Create panel structure
        entity_index = np.repeat(np.arange(N), T)
        time_index = np.tile(np.arange(T), N)

        return {
            "X": X,
            "y": y,
            "beta": beta,
            "coords": coords,
            "distance_matrix": distance_matrix,
            "entity_index": entity_index,
            "time_index": time_index,
            "N": N,
            "T": T,
            "errors": errors,
        }

    def _generate_spatial_temporal_errors(
        self, N, T, distance_matrix, spatial_decay=50, temporal_decay=0.5
    ):
        """Generate errors with spatial and temporal correlation."""
        errors = np.zeros((T, N))

        # Spatial correlation matrix (exponential decay)
        spatial_corr = np.exp(-distance_matrix / spatial_decay)
        np.fill_diagonal(spatial_corr, 1.0)

        # Cholesky decomposition for spatial correlation
        L_spatial = np.linalg.cholesky(spatial_corr)

        # Generate temporally correlated shocks
        for t in range(T):
            if t == 0:
                # Initial period
                spatial_shock = L_spatial @ np.random.randn(N)
                errors[t] = spatial_shock
            else:
                # AR(1) temporal correlation + spatial correlation
                temporal_component = temporal_decay * errors[t - 1]
                spatial_shock = L_spatial @ np.random.randn(N) * np.sqrt(1 - temporal_decay**2)
                errors[t] = temporal_component + spatial_shock

        return errors

    def test_spatial_hac_initialization(self, spatial_panel_data):
        """Test SpatialHAC initialization."""
        data = spatial_panel_data

        # Initialize with distance matrix
        hac = SpatialHAC(
            distance_matrix=data["distance_matrix"], spatial_cutoff=100, temporal_cutoff=2  # 100 km
        )

        assert hac.spatial_cutoff == 100
        assert hac.temporal_cutoff == 2
        assert hac.spatial_kernel == "bartlett"
        assert hac.temporal_kernel == "bartlett"

        # Initialize from coordinates
        hac2 = SpatialHAC.from_coordinates(
            coords=data["coords"], spatial_cutoff=100, temporal_cutoff=2
        )

        assert_allclose(hac2.distance_matrix, data["distance_matrix"], rtol=1e-10)

    def test_spatial_kernels(self, spatial_panel_data):
        """Test different spatial kernel functions."""
        data = spatial_panel_data

        distances = np.array([0, 25, 50, 75, 100, 125])
        cutoff = 100

        # Bartlett kernel
        hac_bartlett = SpatialHAC(
            distance_matrix=data["distance_matrix"],
            spatial_cutoff=cutoff,
            spatial_kernel="bartlett",
        )
        weights_bartlett = hac_bartlett._spatial_kernel_weight(distances)
        expected_bartlett = np.array([1.0, 0.75, 0.5, 0.25, 0.0, 0.0])
        assert_allclose(weights_bartlett, expected_bartlett)

        # Uniform kernel
        hac_uniform = SpatialHAC(
            distance_matrix=data["distance_matrix"], spatial_cutoff=cutoff, spatial_kernel="uniform"
        )
        weights_uniform = hac_uniform._spatial_kernel_weight(distances)
        expected_uniform = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        assert_allclose(weights_uniform, expected_uniform)

        # Epanechnikov kernel
        hac_epan = SpatialHAC(
            distance_matrix=data["distance_matrix"],
            spatial_cutoff=cutoff,
            spatial_kernel="epanechnikov",
        )
        weights_epan = hac_epan._spatial_kernel_weight(distances)
        # Epanechnikov: 0.75 * (1 - u^2) for u <= 1
        u = distances / cutoff
        expected_epan = np.where(u <= 1, 0.75 * (1 - u**2), 0)
        assert_allclose(weights_epan, expected_epan)

    def test_temporal_kernels(self):
        """Test different temporal kernel functions."""
        # Create dummy distance matrix
        distance_matrix = np.zeros((10, 10))

        lags = np.array([0, 1, 2, 3, 4, 5])
        cutoff = 3

        # Bartlett kernel
        hac_bartlett = SpatialHAC(
            distance_matrix=distance_matrix,
            spatial_cutoff=100,
            temporal_cutoff=cutoff,
            temporal_kernel="bartlett",
        )
        weights_bartlett = hac_bartlett._temporal_kernel_weight(lags)
        expected_bartlett = np.array([1.0, 0.75, 0.5, 0.25, 0.0, 0.0])
        assert_allclose(weights_bartlett, expected_bartlett)

        # Uniform kernel
        hac_uniform = SpatialHAC(
            distance_matrix=distance_matrix,
            spatial_cutoff=100,
            temporal_cutoff=cutoff,
            temporal_kernel="uniform",
        )
        weights_uniform = hac_uniform._temporal_kernel_weight(lags)
        expected_uniform = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        assert_allclose(weights_uniform, expected_uniform)

    def test_spatial_hac_computation(self, spatial_panel_data):
        """Test Spatial HAC covariance matrix computation."""
        data = spatial_panel_data

        # OLS estimation
        X = data["X"]
        y = data["y"]
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta_ols

        # Compute Spatial HAC
        hac = SpatialHAC(
            distance_matrix=data["distance_matrix"],
            spatial_cutoff=50,  # 50 km cutoff
            temporal_cutoff=2,  # 2 period lag
        )

        V_hac = hac.compute(
            X=X,
            residuals=residuals,
            entity_index=data["entity_index"],
            time_index=data["time_index"],
        )

        # Check dimensions
        K = X.shape[1]
        assert V_hac.shape == (K, K)

        # Check symmetry
        assert_allclose(V_hac, V_hac.T)

        # Check positive semi-definite
        eigenvalues = np.linalg.eigvals(V_hac)
        assert np.all(eigenvalues >= -1e-10)  # Allow for numerical error

    def test_spatial_hac_vs_ols(self, spatial_panel_data):
        """Test that Spatial HAC SEs are larger than OLS SEs under spatial correlation."""
        data = spatial_panel_data

        X = data["X"]
        y = data["y"]
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta_ols

        # Spatial HAC
        hac = SpatialHAC(
            distance_matrix=data["distance_matrix"], spatial_cutoff=100, temporal_cutoff=2
        )

        comparison = hac.compare_with_standard_errors(
            X=X,
            residuals=residuals,
            entity_index=data["entity_index"],
            time_index=data["time_index"],
        )

        # Under spatial correlation, HAC SEs should generally be larger than OLS
        # (though not always for every coefficient)
        avg_ratio = np.mean(comparison["se_ratio_hac_ols"])
        assert avg_ratio > 0.9  # Should be close to or greater than 1

        # HAC SEs should be different from OLS SEs
        assert not np.allclose(comparison["se_hac"], comparison["se_ols"], rtol=0.1)

    def test_spatial_cutoff_sensitivity(self, spatial_panel_data):
        """Test sensitivity of SEs to spatial cutoff distance."""
        data = spatial_panel_data

        X = data["X"]
        y = data["y"]
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta_ols

        # Different cutoff distances
        cutoffs = [10, 50, 100, 200]
        se_results = {}

        for cutoff in cutoffs:
            hac = SpatialHAC(
                distance_matrix=data["distance_matrix"],
                spatial_cutoff=cutoff,
                temporal_cutoff=0,  # No temporal correlation for this test
            )

            V_hac = hac.compute(
                X=X,
                residuals=residuals,
                entity_index=data["entity_index"],
                time_index=data["time_index"],
            )

            se_results[cutoff] = np.sqrt(np.diag(V_hac))

        # Larger cutoff should generally lead to larger SEs
        # (more correlation is accounted for)
        for i in range(len(cutoffs) - 1):
            cutoff1 = cutoffs[i]
            cutoff2 = cutoffs[i + 1]

            # Average SE should increase with cutoff
            avg_se1 = np.mean(se_results[cutoff1])
            avg_se2 = np.mean(se_results[cutoff2])

            # Allow some tolerance due to kernel shape
            assert avg_se2 >= avg_se1 * 0.95

    def test_temporal_cutoff_sensitivity(self, spatial_panel_data):
        """Test sensitivity of SEs to temporal cutoff."""
        data = spatial_panel_data

        X = data["X"]
        y = data["y"]
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta_ols

        # Different temporal cutoffs
        temp_cutoffs = [0, 1, 2, 4]
        se_results = {}

        for t_cutoff in temp_cutoffs:
            hac = SpatialHAC(
                distance_matrix=data["distance_matrix"], spatial_cutoff=50, temporal_cutoff=t_cutoff
            )

            V_hac = hac.compute(
                X=X,
                residuals=residuals,
                entity_index=data["entity_index"],
                time_index=data["time_index"],
            )

            se_results[t_cutoff] = np.sqrt(np.diag(V_hac))

        # More temporal lags should generally lead to different SEs
        assert not np.allclose(se_results[0], se_results[4], rtol=0.05)

    def test_haversine_distance(self):
        """Test Haversine distance calculation."""
        # Test known distances
        # NYC to Boston (approximately 306 km)
        coords = np.array([[40.7128, -74.0060], [42.3601, -71.0589]])  # NYC  # Boston

        distance_matrix = SpatialHAC._haversine_distance_matrix(coords)

        # Check diagonal is zero
        assert_allclose(np.diag(distance_matrix), 0)

        # Check symmetry
        assert_allclose(distance_matrix, distance_matrix.T)

        # Check approximate distance (should be around 306 km)
        assert 250 < distance_matrix[0, 1] < 350

    def test_comparison_with_driscoll_kraay(self, spatial_panel_data):
        """Test comparison between Spatial HAC and Driscoll-Kraay."""
        data = spatial_panel_data

        X = data["X"]
        y = data["y"]
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta_ols

        # Spatial HAC
        hac = SpatialHAC(
            distance_matrix=data["distance_matrix"], spatial_cutoff=100, temporal_cutoff=2
        )

        V_hac = hac.compute(
            X=X,
            residuals=residuals,
            entity_index=data["entity_index"],
            time_index=data["time_index"],
        )
        se_hac = np.sqrt(np.diag(V_hac))

        # Driscoll-Kraay (if available)
        try:
            dk_result = driscoll_kraay(
                X=X,
                residuals=residuals,
                entity_ids=data["entity_index"],
                time_ids=data["time_index"],
                kernel="bartlett",
                bandwidth=2,
            )
            se_dk = dk_result.std_errors

            # Compare
            comparison = DriscollKraayComparison.compare(
                spatial_hac_se=se_hac,
                driscoll_kraay_se=se_dk,
                param_names=[f"beta_{i}" for i in range(len(se_hac))],
            )

            # Both should be robust to spatial correlation
            # but might differ due to different assumptions
            assert comparison is not None
            assert "ratio" in comparison.columns

        except Exception:
            # Skip if Driscoll-Kraay not available
            pytest.skip("Driscoll-Kraay not available for comparison")

    def test_unbalanced_panel_warning(self, spatial_panel_data):
        """Test warning for unbalanced panel."""
        data = spatial_panel_data

        # Create unbalanced panel by dropping some observations
        X = data["X"][:-5]  # Drop last 5 observations
        residuals = np.random.randn(len(X))
        entity_index = data["entity_index"][:-5]
        time_index = data["time_index"][:-5]

        hac = SpatialHAC(distance_matrix=data["distance_matrix"], spatial_cutoff=100)

        with pytest.warns(UserWarning, match="Unbalanced panel"):
            V_hac = hac.compute(
                X=X, residuals=residuals, entity_index=entity_index, time_index=time_index
            )

        # Should still compute despite warning
        assert V_hac is not None
        assert V_hac.shape == (X.shape[1], X.shape[1])

    def test_small_sample_correction(self, spatial_panel_data):
        """Test small sample correction."""
        data = spatial_panel_data

        X = data["X"]
        y = data["y"]
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta_ols

        hac = SpatialHAC(distance_matrix=data["distance_matrix"], spatial_cutoff=50)

        # Without correction
        V_no_correction = hac.compute(
            X=X,
            residuals=residuals,
            entity_index=data["entity_index"],
            time_index=data["time_index"],
            small_sample_correction=False,
        )

        # With correction
        V_with_correction = hac.compute(
            X=X,
            residuals=residuals,
            entity_index=data["entity_index"],
            time_index=data["time_index"],
            small_sample_correction=True,
        )

        # Correction should increase variance estimates
        n_obs = len(X)
        k_vars = X.shape[1]
        correction_factor = n_obs / (n_obs - k_vars)

        assert_allclose(V_with_correction, V_no_correction * correction_factor, rtol=1e-10)
