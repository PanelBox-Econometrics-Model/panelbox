"""Tests for matrix operations utilities."""

import numpy as np
import pytest

from panelbox.utils.matrix_ops import add_intercept, compute_ols, demean_matrix


class TestAddIntercept:
    """Test add_intercept function."""

    def test_add_intercept_basic(self):
        """Test adding intercept to simple matrix."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_with_intercept = add_intercept(X)

        # Check shape
        assert X_with_intercept.shape == (3, 3)

        # Check first column is ones
        assert np.all(X_with_intercept[:, 0] == 1)

        # Check remaining columns are same as original
        assert np.allclose(X_with_intercept[:, 1:], X)

    def test_add_intercept_single_column(self):
        """Test adding intercept to single column."""
        X = np.array([[1], [2], [3]])
        X_with_intercept = add_intercept(X)

        assert X_with_intercept.shape == (3, 2)
        assert np.all(X_with_intercept[:, 0] == 1)

    def test_add_intercept_large_matrix(self):
        """Test adding intercept to larger matrix."""
        X = np.random.randn(100, 5)
        X_with_intercept = add_intercept(X)

        assert X_with_intercept.shape == (100, 6)
        assert np.all(X_with_intercept[:, 0] == 1)
        assert np.allclose(X_with_intercept[:, 1:], X)


class TestDemeanMatrix:
    """Test demean_matrix function."""

    def test_demean_matrix_basic(self):
        """Test basic demeaning by groups."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        groups = np.array([1, 1, 2, 2])

        X_demeaned = demean_matrix(X, groups)

        # Group 1 mean: [2, 3]
        # Group 2 mean: [6, 7]
        expected = np.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]])

        assert np.allclose(X_demeaned, expected)

    def test_demean_matrix_single_group(self):
        """Test demeaning with single group."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        groups = np.array([1, 1, 1])

        X_demeaned = demean_matrix(X, groups)

        # All in same group, mean is [3, 4]
        expected = np.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])

        assert np.allclose(X_demeaned, expected)

    def test_demean_matrix_preserves_original(self):
        """Test that original matrix is not modified."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        groups = np.array([1, 2])
        X_copy = X.copy()

        demean_matrix(X, groups)

        assert np.allclose(X, X_copy)


class TestComputeOLS:
    """Test compute_ols function."""

    def test_compute_ols_basic(self):
        """Test basic OLS computation."""
        # Simple regression: y = 2 + 3*x
        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
        y = np.array([5, 8, 11, 14, 17])

        beta, resid, fitted = compute_ols(y, X)

        # Should recover true coefficients (approximately)
        assert beta.shape == (2, 1)
        assert np.allclose(beta, [[2], [3]], atol=1e-10)

        # Residuals should be near zero
        assert np.allclose(resid, 0, atol=1e-10)

        # Fitted values should match y
        assert np.allclose(fitted.ravel(), y, atol=1e-10)

    def test_compute_ols_with_noise(self):
        """Test OLS with noisy data."""
        np.random.seed(42)
        X = np.column_stack([np.ones(100), np.random.randn(100)])
        true_beta = np.array([[2.0], [3.0]])
        y = X @ true_beta + np.random.randn(100, 1) * 0.1

        beta, resid, fitted = compute_ols(y.ravel(), X)

        # Should be close to true coefficients
        assert np.allclose(beta, true_beta, atol=0.5)

        # Check shapes (when y is 1D, resid and fitted are also 1D)
        assert beta.shape == (2, 1)
        assert resid.shape == (100,)
        assert fitted.shape == (100,)

    def test_compute_ols_with_weights(self):
        """Test weighted least squares."""
        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        y = np.array([1, 2, 3, 4])
        weights = np.array([1.0, 1.0, 1.0, 1.0])

        beta_weighted, resid_weighted, fitted_weighted = compute_ols(y, X, weights)

        # With equal weights, should be same as unweighted
        beta_unweighted, _, _ = compute_ols(y, X)

        assert np.allclose(beta_weighted, beta_unweighted)

    def test_compute_ols_unequal_weights(self):
        """Test weighted OLS with unequal weights."""
        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        y = np.array([1, 2, 3, 10])  # Last observation is outlier
        weights = np.array([1.0, 1.0, 1.0, 0.1])  # Down-weight outlier

        beta_weighted, _, _ = compute_ols(y, X, weights)
        beta_unweighted, _, _ = compute_ols(y, X)

        # Weighted should be less influenced by outlier
        assert not np.allclose(beta_weighted, beta_unweighted)

    def test_compute_ols_1d_y(self):
        """Test that 1D y is handled correctly."""
        X = np.array([[1, 1], [1, 2], [1, 3]])
        y_1d = np.array([1, 2, 3])

        beta, resid, fitted = compute_ols(y_1d, X)

        # Should work and return correct shapes (when y is 1D, resid and fitted preserve that)
        assert beta.shape == (2, 1)
        assert resid.shape == (3,)
        assert fitted.shape == (3,)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_intercept_and_ols(self):
        """Test adding intercept then running OLS."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([3, 5, 7, 9, 11])

        X_with_intercept = add_intercept(X)
        beta, resid, fitted = compute_ols(y, X_with_intercept)

        # Should recover y = 1 + 2*x
        assert np.allclose(beta, [[1], [2]], atol=1e-10)

    def test_demean_and_ols(self):
        """Test demeaning then OLS (within estimator)."""
        X = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 3.0], [2.0, 4.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        groups = np.array([1, 1, 2, 2])

        # Demean X and y
        X_demeaned = demean_matrix(X, groups)
        y_demeaned = demean_matrix(y.reshape(-1, 1), groups).ravel()

        # Run OLS on demeaned data
        beta, _, _ = compute_ols(y_demeaned, X_demeaned)

        # Check that it runs without error
        assert beta.shape[0] == 2
