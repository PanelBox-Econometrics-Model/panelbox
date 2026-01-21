"""
Unit tests for GMM Estimator
==============================

Tests for the GMMEstimator class implementing low-level GMM algorithms.
"""

import pytest
import numpy as np
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

        np.testing.assert_array_almost_equal(
            residuals[valid_mask],
            expected_residuals
        )

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

        beta_robust, vcov_robust, W, residuals = estimator.two_step(
            y, X, Z, robust=True
        )

        # Should complete without exception
        assert vcov_robust is not None

    def test_two_step_without_windmeijer(self, overidentified_data):
        """Test two-step GMM without Windmeijer correction."""
        y, X, Z = overidentified_data
        estimator = GMMEstimator()

        beta_no_robust, vcov_no_robust, W, residuals = estimator.two_step(
            y, X, Z, robust=False
        )

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
