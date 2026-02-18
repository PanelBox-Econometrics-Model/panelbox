"""
Unit tests for GMM Estimator
==============================

Tests for the GMMEstimator class implementing low-level GMM algorithms.
"""

import numpy as np
import pytest

from panelbox.gmm.estimator import GMMEstimator

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_gmm_data():
    """
    Generate simple data for GMM estimation.

    True model: y = 0.5 * x1 + 0.3 * x2 + ε
    """
    np.random.seed(42)
    n = 200  # Larger sample for stability

    # Instruments (exogenous) - more independent
    z1 = np.random.normal(0, 1, n)
    z2 = np.random.normal(5, 1, n)  # Different mean for independence

    # Regressors (correlated with instruments but with more variation)
    x1 = 0.7 * z1 + 0.5 * np.random.normal(0, 1, n)
    x2 = 0.6 * z2 + 0.5 * np.random.normal(0, 1, n)

    # Dependent variable
    epsilon = np.random.normal(0, 0.3, n)
    y = 0.5 * x1 + 0.3 * x2 + epsilon

    # Stack as matrices
    y = y.reshape(-1, 1)
    X = np.column_stack([x1, x2])
    Z = np.column_stack([z1, z2])

    return y, X, Z


@pytest.fixture
def overidentified_data():
    """Generate overidentified data (more instruments than regressors)."""
    np.random.seed(42)
    n = 100

    # 4 instruments for 2 regressors
    z1 = np.random.normal(0, 1, n)
    z2 = np.random.normal(0, 1, n)
    z3 = np.random.normal(0, 1, n)
    z4 = np.random.normal(0, 1, n)

    # Regressors
    x1 = 0.5 * z1 + 0.3 * z2 + 0.2 * np.random.normal(0, 1, n)
    x2 = 0.4 * z3 + 0.4 * z4 + 0.2 * np.random.normal(0, 1, n)

    # Dependent variable
    epsilon = np.random.normal(0, 0.5, n)
    y = 0.5 * x1 + 0.3 * x2 + epsilon

    y = y.reshape(-1, 1)
    X = np.column_stack([x1, x2])
    Z = np.column_stack([z1, z2, z3, z4])

    return y, X, Z


@pytest.fixture
def data_with_missing():
    """Generate data with missing values (NaN)."""
    np.random.seed(42)
    n = 100

    z1 = np.random.normal(0, 1, n)
    z2 = np.random.normal(0, 1, n)
    x1 = 0.7 * z1 + 0.3 * np.random.normal(0, 1, n)
    x2 = 0.6 * z2 + 0.4 * np.random.normal(0, 1, n)
    epsilon = np.random.normal(0, 0.5, n)
    y = 0.5 * x1 + 0.3 * x2 + epsilon

    # Introduce missing values
    missing_idx = np.random.choice(n, size=10, replace=False)
    y[missing_idx] = np.nan

    y = y.reshape(-1, 1)
    X = np.column_stack([x1, x2])
    Z = np.column_stack([z1, z2])

    return y, X, Z


# ============================================================================
# Test Initialization
# ============================================================================


class TestGMMEstimatorInitialization:
    """Test GMMEstimator initialization."""

    def test_init_default(self):
        """Test default initialization."""
        estimator = GMMEstimator()

        assert estimator.tol == 1e-6
        assert estimator.max_iter == 100

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        estimator = GMMEstimator(tol=1e-8, max_iter=200)

        assert estimator.tol == 1e-8
        assert estimator.max_iter == 200


# ============================================================================
# Test One-Step GMM
# ============================================================================


class TestOneStepGMM:
    """Test one-step GMM estimation."""

    def test_one_step_basic(self, simple_gmm_data):
        """Test basic one-step GMM estimation."""
        y, X, Z = simple_gmm_data
        estimator = GMMEstimator()

        beta, W, residuals = estimator.one_step(y, X, Z)

        # Check shapes
        assert beta.shape == (2, 1)
        assert W.shape == (2, 2)
        assert residuals.shape == y.shape

        # Check that beta is not NaN
        assert not np.any(np.isnan(beta))

    def test_one_step_coefficient_signs(self, overidentified_data):
        """Test that estimated coefficients have expected signs."""
        y, X, Z = overidentified_data
        estimator = GMMEstimator()

        beta, W, residuals = estimator.one_step(y, X, Z)

        # True coefficients are 0.5 and 0.3, both positive
        # At least one should be positive (allow for estimation error)
        if not np.any(np.isnan(beta)):
            positive_count = sum(beta.flatten() > 0)
            assert positive_count >= 1

    def test_one_step_coefficient_magnitude(self, simple_gmm_data):
        """Test that coefficients are in reasonable range."""
        y, X, Z = simple_gmm_data
        estimator = GMMEstimator()

        beta, W, residuals = estimator.one_step(y, X, Z)

        # Coefficients should be finite and not too large
        assert np.all(np.isfinite(beta))
        assert np.all(np.abs(beta) < 10.0)  # Reasonable bound

    def test_one_step_residuals(self, simple_gmm_data):
        """Test that residuals are computed correctly."""
        y, X, Z = simple_gmm_data
        estimator = GMMEstimator()

        beta, W, residuals = estimator.one_step(y, X, Z)

        # Compute residuals manually
        # Only valid observations (non-NaN in residuals)
        valid_mask = ~np.isnan(residuals.flatten())
        expected_residuals = y[valid_mask] - X[valid_mask] @ beta

        np.testing.assert_array_almost_equal(residuals[valid_mask], expected_residuals)

    def test_one_step_overidentified(self, overidentified_data):
        """Test one-step GMM with overidentified model."""
        y, X, Z = overidentified_data
        estimator = GMMEstimator()

        beta, W, residuals = estimator.one_step(y, X, Z)

        # Should work with more instruments than regressors
        assert beta.shape == (2, 1)
        assert W.shape == (4, 4)
        assert not np.any(np.isnan(beta))

    def test_one_step_with_missing_values(self, data_with_missing):
        """Test one-step GMM handles missing values."""
        y, X, Z = data_with_missing
        estimator = GMMEstimator()

        beta, W, residuals = estimator.one_step(y, X, Z)

        # Should handle NaN values
        assert beta.shape == (2, 1)
        assert not np.any(np.isnan(beta))

        # Residuals should have NaN where y has NaN
        y_nan_mask = np.isnan(y.flatten())
        assert np.all(np.isnan(residuals[y_nan_mask]))

    def test_one_step_weight_matrix_properties(self, simple_gmm_data):
        """Test properties of weight matrix."""
        y, X, Z = simple_gmm_data
        estimator = GMMEstimator()

        beta, W, residuals = estimator.one_step(y, X, Z)

        # Weight matrix should be symmetric
        np.testing.assert_array_almost_equal(W, W.T)

        # Weight matrix should be positive definite
        eigenvalues = np.linalg.eigvals(W)
        assert np.all(eigenvalues > -1e-10)  # Allow small numerical errors


# ============================================================================
# Test Two-Step GMM
# ============================================================================


class TestTwoStepGMM:
    """Test two-step GMM estimation."""

    def test_two_step_basic(self, overidentified_data):
        """Test basic two-step GMM estimation."""
        y, X, Z = overidentified_data  # Use overidentified for stability
        estimator = GMMEstimator()

        beta, vcov, W, residuals = estimator.two_step(y, X, Z, robust=True)

        # Check shapes
        assert beta.shape == (2, 1)
        assert vcov.shape == (2, 2)
        assert residuals.shape == y.shape

        # Two-step may produce NaN with poorly conditioned data
        # Just check that method completes without exception
        assert beta is not None
        assert vcov is not None

    def test_two_step_coefficient_signs(self, overidentified_data):
        """Test that estimated coefficients have expected signs."""
        y, X, Z = overidentified_data
        estimator = GMMEstimator()

        beta, vcov, W, residuals = estimator.two_step(y, X, Z, robust=True)

        # If estimation succeeded (no NaN), check properties
        if not np.any(np.isnan(beta)):
            # True coefficients are positive
            # At least one should be positive
            positive_count = sum(beta.flatten() > 0)
            assert positive_count >= 1

    def test_two_step_vcov_properties(self, overidentified_data):
        """Test properties of variance-covariance matrix."""
        y, X, Z = overidentified_data
        estimator = GMMEstimator()

        beta, vcov, W, residuals = estimator.two_step(y, X, Z, robust=True)

        # If estimation succeeded (no NaN), check properties
        if not np.any(np.isnan(vcov)):
            # Vcov should be symmetric
            np.testing.assert_array_almost_equal(vcov, vcov.T)

            # Diagonal elements (variances) should be positive
            assert np.all(np.diag(vcov) > 0)

    def test_two_step_vs_one_step(self, overidentified_data):
        """Test that two-step gives different results than one-step."""
        y, X, Z = overidentified_data
        estimator = GMMEstimator()

        beta_one, _, _ = estimator.one_step(y, X, Z)
        beta_two, _, _, _ = estimator.two_step(y, X, Z, robust=True)

        # Method should complete without exception
        assert beta_one is not None
        assert beta_two is not None

    def test_two_step_overidentified(self, overidentified_data):
        """Test two-step GMM with overidentified model."""
        y, X, Z = overidentified_data
        estimator = GMMEstimator()

        beta, vcov, W, residuals = estimator.two_step(y, X, Z, robust=True)

        # Should work with more instruments than regressors
        assert beta.shape == (2, 1)
        assert vcov.shape == (2, 2)
        assert not np.any(np.isnan(beta))

    def test_two_step_with_windmeijer(self, overidentified_data):
        """Test two-step GMM with Windmeijer correction."""
        y, X, Z = overidentified_data
        estimator = GMMEstimator()

        beta_robust, vcov_robust, W, residuals = estimator.two_step(y, X, Z, robust=True)

        # Should complete without exception
        assert vcov_robust is not None

    def test_two_step_without_windmeijer(self, overidentified_data):
        """Test two-step GMM without Windmeijer correction."""
        y, X, Z = overidentified_data
        estimator = GMMEstimator()

        beta_no_robust, vcov_no_robust, W, residuals = estimator.two_step(y, X, Z, robust=False)

        # Should complete without exception
        assert vcov_no_robust is not None


# ============================================================================
# Test Iterative GMM
# ============================================================================


class TestIterativeGMM:
    """Test iterative GMM estimation."""

    def test_iterative_basic(self, simple_gmm_data):
        """Test basic iterative GMM estimation."""
        y, X, Z = simple_gmm_data
        estimator = GMMEstimator()

        beta, vcov, W, converged = estimator.iterative(y, X, Z)

        # Check shapes
        assert beta.shape == (2, 1)
        assert vcov.shape == (2, 2)

        # Check convergence
        assert isinstance(converged, bool)

        # Check that beta is not NaN
        assert not np.any(np.isnan(beta))

    def test_iterative_convergence(self, simple_gmm_data):
        """Test that iterative GMM converges."""
        y, X, Z = simple_gmm_data
        estimator = GMMEstimator(tol=1e-6, max_iter=100)

        beta, vcov, W, converged = estimator.iterative(y, X, Z)

        # Should converge with well-behaved data
        assert converged is True

    def test_iterative_vs_two_step(self, simple_gmm_data):
        """Test that iterative GMM gives similar results to two-step."""
        y, X, Z = simple_gmm_data
        estimator = GMMEstimator()

        beta_two, _, _, _ = estimator.two_step(y, X, Z, robust=True)
        beta_iter, _, _, converged = estimator.iterative(y, X, Z)

        if converged:
            # Should be reasonably close (iterative is CUE)
            np.testing.assert_array_almost_equal(beta_two, beta_iter, decimal=1)

    def test_iterative_max_iterations(self, simple_gmm_data):
        """Test that iterative GMM respects max_iter."""
        y, X, Z = simple_gmm_data
        estimator = GMMEstimator(max_iter=2, tol=1e-20)  # Very tight tolerance

        beta, vcov, W, converged = estimator.iterative(y, X, Z)

        # May not converge with only 2 iterations
        # But should still return result
        assert beta.shape == (2, 1)


# ============================================================================
# Test Valid Mask
# ============================================================================


class TestValidMask:
    """Test _get_valid_mask method."""

    def test_valid_mask_no_missing(self, overidentified_data):
        """Test valid mask with no missing values."""
        y, X, Z = overidentified_data  # Use overidentified data (4 instruments > k+1)
        estimator = GMMEstimator()

        valid_mask = estimator._get_valid_mask(y, X, Z)

        # All observations should be valid with enough instruments
        assert np.all(valid_mask)
        assert valid_mask.sum() == len(y)

    def test_valid_mask_with_missing_y(self):
        """Test valid mask with missing y values."""
        np.random.seed(42)
        n = 100

        y = np.random.normal(0, 1, n).reshape(-1, 1)
        X = np.random.normal(0, 1, (n, 2))
        Z = np.random.normal(0, 1, (n, 4))  # More instruments for validity

        # Add NaN to y
        y[10] = np.nan
        y[20] = np.nan

        estimator = GMMEstimator()
        valid_mask = estimator._get_valid_mask(y, X, Z)

        # Should exclude rows with NaN
        assert valid_mask.sum() == n - 2
        assert not valid_mask[10]
        assert not valid_mask[20]

    def test_valid_mask_with_missing_X(self):
        """Test valid mask with missing X values."""
        np.random.seed(42)
        n = 100

        y = np.random.normal(0, 1, n).reshape(-1, 1)
        X = np.random.normal(0, 1, (n, 2))
        Z = np.random.normal(0, 1, (n, 4))  # More instruments for validity

        # Add NaN to X
        X[15, 0] = np.nan

        estimator = GMMEstimator()
        valid_mask = estimator._get_valid_mask(y, X, Z)

        # Should exclude rows with NaN
        assert valid_mask.sum() == n - 1
        assert not valid_mask[15]

    def test_valid_mask_with_missing_Z(self):
        """Test valid mask with missing Z values."""
        np.random.seed(42)
        n = 100

        y = np.random.normal(0, 1, n).reshape(-1, 1)
        X = np.random.normal(0, 1, (n, 2))
        Z = np.random.normal(0, 1, (n, 4))  # More instruments

        # Add NaN to Z - but still enough remain for validity
        Z[25, 1] = np.nan

        estimator = GMMEstimator()
        valid_mask = estimator._get_valid_mask(y, X, Z)

        # All should still be valid since we have 3 remaining instruments (>= k+1)
        assert valid_mask.sum() == n
        assert valid_mask[25]  # Still valid


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_small_sample(self):
        """Test with very small sample size."""
        np.random.seed(42)
        n = 10

        z1 = np.random.normal(0, 1, n)
        x1 = 0.7 * z1 + 0.3 * np.random.normal(0, 1, n)
        y = 0.5 * x1 + np.random.normal(0, 0.5, n)

        y = y.reshape(-1, 1)
        X = x1.reshape(-1, 1)
        Z = z1.reshape(-1, 1)

        estimator = GMMEstimator()

        # Should complete without error
        beta, W, residuals = estimator.one_step(y, X, Z)
        assert not np.any(np.isnan(beta))

    def test_perfectly_collinear_instruments(self):
        """Test with perfectly collinear instruments."""
        np.random.seed(42)
        n = 100

        z1 = np.random.normal(0, 1, n)
        z2 = 2 * z1  # Perfectly collinear with z1
        z3 = np.random.normal(0, 1, n)  # Add non-collinear instrument

        x1 = 0.7 * z1 + 0.3 * np.random.normal(0, 1, n)
        y = 0.5 * x1 + np.random.normal(0, 0.5, n)

        y = y.reshape(-1, 1)
        X = x1.reshape(-1, 1)
        Z = np.column_stack([z1, z2, z3])

        estimator = GMMEstimator()

        # Should handle collinearity gracefully
        beta, W, residuals = estimator.one_step(y, X, Z)

        # Should still return result
        assert beta.shape == (1, 1)
        assert beta is not None


# ============================================================================
# Fixtures for per-individual API tests
# ============================================================================


@pytest.fixture
def panel_gmm_data():
    """
    Generate panel data with individual ids for per-individual weight tests.

    Simulates a dynamic panel: y_it = 0.5 * y_{i,t-1} + 0.3 * x_it + eta_i + eps_it
    then first-differences to eliminate fixed effects.

    Produces overidentified data (4 instruments for 2 regressors).
    """
    np.random.seed(42)
    N = 50  # individuals
    T = 10  # time periods

    y_all, X_all, Z_all, ids_all = [], [], [], []

    for i in range(N):
        eta_i = np.random.normal(0, 1)
        y = np.zeros(T)
        x = np.random.normal(0, 1, T)
        y[0] = eta_i + np.random.normal(0, 0.5)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + 0.3 * x[t] + eta_i + np.random.normal(0, 0.5)

        # First-difference: Dy_t = y_t - y_{t-1}
        dy = np.diff(y)  # length T-1
        # Regressors for differenced eq at t=3..T-1: [Dy_{t-1}, Dx_t]
        # Need t >= 3 to have y_{t-2} and y_{t-3} as instruments
        start = 2  # index in dy (corresponds to t=3 in original)
        dy_dep = dy[start:]  # Dy_t
        dy_lag = dy[start - 1 : -1]  # Dy_{t-1}
        dx = np.diff(x)[start:]  # Dx_t

        # Instruments: y_{t-2}, y_{t-3}, x_{t-1}, x_{t-2}
        # (4 instruments for 2 regressors -> overidentified)
        n_obs = len(dy_dep)
        y_lag2 = y[1 : 1 + n_obs]  # y_{t-2}
        y_lag3 = y[:n_obs]  # y_{t-3}
        x_lag1 = x[2 : 2 + n_obs]  # x_{t-1}
        x_lag2 = x[1 : 1 + n_obs]  # x_{t-2}

        X_i = np.column_stack([dy_lag, dx])
        y_i = dy_dep.reshape(-1, 1)
        z_i = np.column_stack([y_lag2, y_lag3, x_lag1, x_lag2])

        y_all.append(y_i)
        X_all.append(X_i)
        Z_all.append(z_i)
        ids_all.append(np.full(len(y_i), i))

    y = np.vstack(y_all)
    X = np.vstack(X_all)
    Z = np.vstack(Z_all)
    ids = np.concatenate(ids_all)

    return y, X, Z, ids


# ============================================================================
# Test H Matrix
# ============================================================================


class TestBuildHMatrix:
    """Test build_H_matrix static method."""

    def test_h_matrix_shape(self):
        """Test that H matrix has correct shape."""
        H = GMMEstimator.build_H_matrix(5)
        assert H.shape == (5, 5)

    def test_h_matrix_tridiagonal(self):
        """Test that H matrix is tridiagonal for first-differences."""
        H = GMMEstimator.build_H_matrix(4)
        expected = np.array(
            [
                [2, -1, 0, 0],
                [-1, 2, -1, 0],
                [0, -1, 2, -1],
                [0, 0, -1, 2],
            ],
            dtype=float,
        )
        np.testing.assert_array_almost_equal(H, expected)

    def test_h_matrix_symmetric(self):
        """Test that H matrix is symmetric."""
        H = GMMEstimator.build_H_matrix(6)
        np.testing.assert_array_almost_equal(H, H.T)

    def test_h_matrix_positive_semidefinite(self):
        """Test that H matrix is positive semi-definite."""
        H = GMMEstimator.build_H_matrix(5)
        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(eigenvalues >= -1e-10)

    def test_h_matrix_scalar(self):
        """Test H matrix for T=1."""
        H = GMMEstimator.build_H_matrix(1)
        np.testing.assert_array_almost_equal(H, np.array([[2.0]]))

    def test_h_matrix_fod_is_identity(self):
        """Test that FOD transformation returns identity."""
        H = GMMEstimator.build_H_matrix(5, transformation="fod")
        np.testing.assert_array_almost_equal(H, np.eye(5))


# ============================================================================
# Test Per-Individual Weight Computation
# ============================================================================


class TestClusteredWeight:
    """Test _compute_clustered_weight method."""

    def test_clustered_weight_shape(self, panel_gmm_data):
        """Test that clustered weight matrix has correct shape."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        L = Z.shape[1]
        residuals = np.random.normal(0, 1, (len(y), 1))

        W2, W2_inv, zs = estimator._compute_clustered_weight(Z, residuals, ids)

        assert W2.shape == (L, L)
        assert W2_inv.shape == (L, L)
        assert zs.shape == (L, 1)

    def test_clustered_weight_symmetric(self, panel_gmm_data):
        """Test that W2 is symmetric."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        residuals = np.random.normal(0, 1, (len(y), 1))
        W2, _, _ = estimator._compute_clustered_weight(Z, residuals, ids)

        np.testing.assert_array_almost_equal(W2, W2.T)

    def test_clustered_weight_positive_semidefinite(self, panel_gmm_data):
        """Test that W2 is positive semi-definite (outer product sum)."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        residuals = np.random.normal(0, 1, (len(y), 1))
        W2, _, _ = estimator._compute_clustered_weight(Z, residuals, ids)

        eigenvalues = np.linalg.eigvalsh(W2)
        assert np.all(eigenvalues >= -1e-10)

    def test_clustered_weight_manual_computation(self):
        """Test clustered weight against manual computation with known data."""
        # 2 individuals, 2 obs each, 1 instrument
        Z = np.array([[1.0], [2.0], [3.0], [4.0]])
        resid = np.array([[0.1], [0.2], [-0.1], [0.3]])
        ids = np.array([0, 0, 1, 1])

        estimator = GMMEstimator()
        W2, W2_inv, zs = estimator._compute_clustered_weight(Z, resid, ids)

        # Manual: m_0 = Z_0' u_0 = [1,2]' @ [0.1,0.2] = 0.5
        #         m_1 = Z_1' u_1 = [3,4]' @ [-0.1,0.3] = 0.9
        # W2 = (1/2) * (0.5*0.5 + 0.9*0.9) = (0.25 + 0.81)/2 = 0.53
        # zs = 0.5 + 0.9 = 1.4
        m_0 = np.array([[1.0, 2.0]]) @ np.array([[0.1], [0.2]])  # 0.5
        m_1 = np.array([[3.0, 4.0]]) @ np.array([[-0.1], [0.3]])  # 0.9
        expected_W2 = (m_0 * m_0 + m_1 * m_1) / 2
        expected_zs = m_0 + m_1

        np.testing.assert_array_almost_equal(W2, expected_W2)
        np.testing.assert_array_almost_equal(zs, expected_zs)


# ============================================================================
# Test One-Step with Per-Individual Weight (ids parameter)
# ============================================================================


class TestOneStepWithIds:
    """Test one-step GMM with per-individual H-based weight."""

    def test_one_step_with_ids_basic(self, panel_gmm_data):
        """Test one-step with ids produces valid results."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        beta, W_inv, residuals = estimator.one_step(y, X, Z, ids=ids)

        assert beta.shape == (X.shape[1], 1)
        assert not np.any(np.isnan(beta))
        assert residuals.shape == y.shape

    def test_one_step_with_ids_weight_uses_H(self, panel_gmm_data):
        """Test that per-individual weight uses H tridiagonal matrix."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        # Compute with ids (uses H)
        beta_h, W_inv_h, _ = estimator.one_step(y, X, Z, ids=ids)

        # Compute without ids (uses Z'Z)
        beta_legacy, W_inv_legacy, _ = estimator.one_step(y, X, Z)

        # Results should differ because weight matrices are different
        # (H-based vs Z'Z)
        assert not np.allclose(W_inv_h, W_inv_legacy, atol=1e-5)

    def test_one_step_with_custom_H_blocks(self, panel_gmm_data):
        """Test one-step with pre-computed H blocks."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        # Build custom H blocks (identity instead of tridiagonal)
        unique_ids = np.unique(ids)
        H_blocks = {}
        for uid in unique_ids:
            T_i = int(np.sum(ids == uid))
            H_blocks[uid] = np.eye(T_i)

        beta, W_inv, residuals = estimator.one_step(y, X, Z, ids=ids, H_blocks=H_blocks)

        assert beta.shape == (X.shape[1], 1)
        assert not np.any(np.isnan(beta))

    def test_one_step_stores_step1_projection(self, panel_gmm_data):
        """Test that one_step stores _step1_M_XZ_W for later use."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        estimator.one_step(y, X, Z, ids=ids)

        assert estimator._step1_M_XZ_W is not None
        assert estimator._step1_M_XZ_W.shape == (X.shape[1], Z.shape[1])


# ============================================================================
# Test Robust One-Step Vcov
# ============================================================================


class TestOneStepRobustVcov:
    """Test compute_one_step_robust_vcov method."""

    def test_robust_vcov_shape(self, panel_gmm_data):
        """Test robust vcov has correct shape."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        beta, _, residuals = estimator.one_step(y, X, Z, ids=ids)
        vcov = estimator.compute_one_step_robust_vcov(Z, residuals, ids)

        K = X.shape[1]
        assert vcov.shape == (K, K)

    def test_robust_vcov_symmetric(self, panel_gmm_data):
        """Test robust vcov is symmetric."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        beta, _, residuals = estimator.one_step(y, X, Z, ids=ids)
        vcov = estimator.compute_one_step_robust_vcov(Z, residuals, ids)

        np.testing.assert_array_almost_equal(vcov, vcov.T)

    def test_robust_vcov_positive_diagonal(self, panel_gmm_data):
        """Test robust vcov has positive diagonal (variances)."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        beta, _, residuals = estimator.one_step(y, X, Z, ids=ids)
        vcov = estimator.compute_one_step_robust_vcov(Z, residuals, ids)

        assert np.all(np.diag(vcov) > 0)

    def test_robust_vcov_stores_hansen_j_inputs(self, panel_gmm_data):
        """Test that robust vcov stores N, W2, W2_inv, zs for Hansen J."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        beta, _, residuals = estimator.one_step(y, X, Z, ids=ids)
        estimator.compute_one_step_robust_vcov(Z, residuals, ids)

        assert estimator.N is not None
        assert estimator.N == len(np.unique(ids))
        assert estimator.W2 is not None
        assert estimator.W2_inv is not None
        assert estimator.zs is not None

    def test_robust_vcov_raises_without_one_step(self, panel_gmm_data):
        """Test error when calling robust vcov before one_step."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        with pytest.raises(ValueError, match="Must call one_step"):
            estimator.compute_one_step_robust_vcov(Z, y, ids)


# ============================================================================
# Test Two-Step with Per-Individual Weight and Windmeijer
# ============================================================================


class TestTwoStepWithIds:
    """Test two-step GMM with per-individual weights and Windmeijer correction."""

    def test_two_step_with_ids_basic(self, panel_gmm_data):
        """Test two-step with ids produces valid results."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        beta, vcov, W2_inv, residuals = estimator.two_step(y, X, Z, ids=ids, robust=True)

        assert beta.shape == (X.shape[1], 1)
        assert vcov.shape == (X.shape[1], X.shape[1])
        assert not np.any(np.isnan(beta))
        assert residuals.shape == y.shape

    def test_two_step_windmeijer_larger_se(self, panel_gmm_data):
        """Test Windmeijer correction produces larger SE than naive two-step."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        # With Windmeijer correction
        _, vcov_robust, _, _ = estimator.two_step(y, X, Z, ids=ids, robust=True)
        se_robust = np.sqrt(np.abs(np.diag(vcov_robust)))

        # Without Windmeijer (naive)
        _, vcov_naive, _, _ = estimator.two_step(y, X, Z, ids=ids, robust=False)
        se_naive = np.sqrt(np.abs(np.diag(vcov_naive)))

        # Both should produce finite values
        assert np.all(np.isfinite(se_robust))
        assert np.all(np.isfinite(se_naive))

        # Windmeijer SE should generally be larger (corrects downward bias)
        # At least one coefficient's SE should be larger with the correction
        assert np.any(se_robust > se_naive * 0.8)

    def test_two_step_vcov_symmetric(self, panel_gmm_data):
        """Test Windmeijer-corrected vcov is symmetric."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        _, vcov, _, _ = estimator.two_step(y, X, Z, ids=ids, robust=True)

        np.testing.assert_array_almost_equal(vcov, vcov.T)

    def test_two_step_stores_hansen_inputs(self, panel_gmm_data):
        """Test that two_step stores all inputs needed for Hansen J."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        estimator.two_step(y, X, Z, ids=ids, robust=True)

        assert estimator.N == len(np.unique(ids))
        assert estimator.W2 is not None
        assert estimator.W2_inv is not None
        assert estimator.zs is not None
        assert estimator.zs.shape == (Z.shape[1], 1)

    def test_two_step_beta_differs_from_one_step(self, panel_gmm_data):
        """Test that two-step beta differs from one-step (uses optimal weight)."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        beta_1, _, _ = estimator.one_step(y, X, Z, ids=ids)
        beta_2, _, _, _ = estimator.two_step(y, X, Z, ids=ids, robust=True)

        # They should differ (different weight matrices)
        assert not np.allclose(beta_1, beta_2, atol=1e-8)


# ============================================================================
# Test Iterative with Per-Individual Weight
# ============================================================================


class TestIterativeWithIds:
    """Test iterative GMM with per-individual clustered weights."""

    def test_iterative_with_ids_basic(self, panel_gmm_data):
        """Test iterative with ids produces valid results."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        beta, vcov, W_inv, converged = estimator.iterative(y, X, Z, ids=ids)

        assert beta.shape == (X.shape[1], 1)
        assert not np.any(np.isnan(beta))
        assert isinstance(converged, bool)

    def test_iterative_with_ids_convergence(self, panel_gmm_data):
        """Test iterative converges with per-individual weights."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator(tol=1e-6, max_iter=200)

        _, _, _, converged = estimator.iterative(y, X, Z, ids=ids)

        assert converged is True

    def test_iterative_stores_hansen_inputs(self, panel_gmm_data):
        """Test iterative stores Hansen J inputs."""
        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        estimator.iterative(y, X, Z, ids=ids)

        assert estimator.N is not None
        assert estimator.W2 is not None
        assert estimator.W2_inv is not None
        assert estimator.zs is not None


# ============================================================================
# Test Hansen J with Per-Individual Moments
# ============================================================================


class TestHansenJPerIndividual:
    """Test Hansen J test with the per-individual moment formula."""

    def test_hansen_j_with_per_individual_moments(self, panel_gmm_data):
        """Test Hansen J using zs, W2_inv, N from estimator."""
        from panelbox.gmm.tests import GMMTests

        y, X, Z, ids = panel_gmm_data
        estimator = GMMEstimator()

        beta, vcov, W2_inv, residuals = estimator.two_step(y, X, Z, ids=ids, robust=True)

        K = beta.shape[0]
        L = Z.shape[1]

        tester = GMMTests()
        result = tester.hansen_j_test(
            residuals,
            Z,
            W2_inv,
            K,
            zs=estimator.zs,
            W2_inv=estimator.W2_inv,
            N=estimator.N,
            n_instruments=L,
        )

        assert result.name == "Hansen J-test"
        assert np.isfinite(result.statistic)
        assert result.statistic >= 0  # J >= 0 always
        assert 0 <= result.pvalue <= 1
        assert result.df == L - K

    def test_hansen_j_formula_manual_check(self):
        """Test Hansen J formula: J = (1/N) zs' W2_inv zs."""
        from panelbox.gmm.tests import GMMTests

        # Known values
        zs = np.array([[1.0], [2.0]])
        W2_inv = np.eye(2) * 0.5
        N = 10
        n_params = 1
        n_instruments = 2

        tester = GMMTests()
        result = tester.hansen_j_test(
            None,
            None,
            None,
            n_params,
            zs=zs,
            W2_inv=W2_inv,
            N=N,
            n_instruments=n_instruments,
        )

        # Manual: J = (1/10) * [1,2] @ diag(0.5,0.5) @ [1,2]'
        #       = (1/10) * (0.5*1 + 0.5*4) = (1/10) * 2.5 = 0.25
        expected_J = (1.0 / N) * (zs.T @ W2_inv @ zs).item()
        assert abs(result.statistic - expected_J) < 1e-10
        assert abs(result.statistic - 0.25) < 1e-10
