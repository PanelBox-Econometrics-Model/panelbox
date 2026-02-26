"""
Tests for Spatial Weight Matrices

This module tests the SpatialWeights class functionality including
creation methods, normalization, and properties.

Tests two separate classes:
- panelbox.core.spatial_weights.SpatialWeights (original tests)
- panelbox.models.spatial.spatial_weights.SpatialWeights (new tests below)
"""

from unittest.mock import patch

import matplotlib
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from scipy.sparse import csr_matrix, issparse

from panelbox.core.spatial_weights import SpatialWeights
from panelbox.models.spatial.spatial_weights import SpatialWeights as ModelSpatialWeights

matplotlib.use("Agg")


class TestSpatialWeightsCreation:
    """Test creation methods for spatial weights."""

    def test_from_matrix(self):
        """Test creating weights from numpy array."""
        # Simple 3×3 matrix
        matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        W = SpatialWeights.from_matrix(matrix)

        assert W.n == 3
        assert W.s0 == 6  # Sum of all weights
        assert not W.normalized
        assert_array_almost_equal(W.matrix, matrix)

    def test_from_matrix_with_diagonal(self):
        """Test that diagonal is set to zero with warning."""
        # Matrix with non-zero diagonal
        matrix = np.array([[1, 1, 0], [1, 2, 1], [0, 1, 3]])

        with pytest.warns(UserWarning, match="non-zero diagonal"):
            W = SpatialWeights.from_matrix(matrix)

        # Check diagonal is zero
        assert_array_almost_equal(np.diag(W.matrix), [0, 0, 0])

    def test_from_matrix_negative_values(self):
        """Test that negative values raise error."""
        matrix = np.array([[0, 1, -1], [1, 0, 1], [-1, 1, 0]])

        with pytest.raises(ValueError, match="negative values"):
            SpatialWeights.from_matrix(matrix)

    def test_from_matrix_not_square(self):
        """Test that non-square matrix raises error."""
        matrix = np.array([[0, 1, 1], [1, 0, 1]])

        with pytest.raises(ValueError, match="must be square"):
            SpatialWeights.from_matrix(matrix)

    def test_from_sparse(self):
        """Test creating weights from sparse matrix."""
        dense = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        sparse = csr_matrix(dense)

        W = SpatialWeights.from_sparse(sparse)

        assert W.n == 4
        assert W.s0 == 8
        assert_array_almost_equal(W.to_dense(), dense)

    def test_from_distance_binary(self):
        """Test distance-based weights with binary values."""
        # 2×2 grid coordinates
        coords = np.array(
            [[0, 0], [1, 0], [0, 1], [1, 1]]  # Bottom-left  # Bottom-right  # Top-left  # Top-right
        )

        # Threshold = 1.1 (connects direct neighbors only, excludes diagonals at √2≈1.414)
        W = SpatialWeights.from_distance(coords, threshold=1.1, binary=True)

        # Expected: each point connects to its direct neighbors
        expected = np.array(
            [
                [0, 1, 1, 0],  # 0 connects to 1, 2
                [1, 0, 0, 1],  # 1 connects to 0, 3
                [1, 0, 0, 1],  # 2 connects to 0, 3
                [0, 1, 1, 0],  # 3 connects to 1, 2
            ]
        )

        assert_array_almost_equal(W.matrix, expected)

    def test_from_distance_inverse(self):
        """Test distance-based weights with inverse distance."""
        # Simple 1D coordinates
        coords = np.array([[0, 0], [1, 0], [3, 0]])

        W = SpatialWeights.from_distance(coords, threshold=2.5, binary=False)

        # Check that weights are inverse distance
        assert W.matrix[0, 1] == pytest.approx(1.0)  # Distance = 1
        assert W.matrix[0, 2] == 0  # Distance = 3 > threshold
        assert W.matrix[1, 2] == pytest.approx(0.5)  # Distance = 2

    def test_from_knn_fallback(self):
        """Test k-NN weights creation with fallback implementation."""
        # Triangle coordinates
        coords = np.array([[0, 0], [1, 0], [0.5, 0.866]])  # Equilateral triangle

        W = SpatialWeights.from_knn(coords, k=2)

        # Each point should connect to the other 2
        expected = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        assert_array_almost_equal(W.matrix, expected)


class TestSpatialWeightsNormalization:
    """Test normalization methods."""

    def test_row_standardization_dense(self):
        """Test row standardization for dense matrix."""
        matrix = np.array([[0, 1, 1], [1, 0, 0], [1, 2, 0]])

        W = SpatialWeights.from_matrix(matrix)
        W.standardize("row")

        # Each row should sum to 1
        row_sums = W.matrix.sum(axis=1)
        assert_allclose(row_sums, [1, 1, 1])
        assert W.normalized

    def test_row_standardization_sparse(self):
        """Test row standardization for sparse matrix."""
        dense = np.array([[0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 2], [0, 1, 2, 0]])
        sparse = csr_matrix(dense)

        W = SpatialWeights.from_sparse(sparse)
        W.standardize("row")

        # Each row should sum to 1
        row_sums = np.asarray(W.matrix.sum(axis=1)).flatten()
        assert_allclose(row_sums, np.ones(4))
        assert W.normalized


class TestSpatialWeightsProperties:
    """Test weight matrix properties and statistics."""

    def test_s0_property(self):
        """Test sum of all weights."""
        matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])

        W = SpatialWeights.from_matrix(matrix)
        assert W.s0 == 12

    def test_s1_property(self):
        """Test sum of squared row + column sums."""
        matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        W = SpatialWeights.from_matrix(matrix)

        # Row sums: [1, 2, 1]
        # Col sums: [1, 2, 1]
        # (row + col)^2: [4, 16, 4]
        # s1 = 0.5 * sum = 12
        assert W.s1 == 12

    def test_s2_property(self):
        """Test sum of squared weights."""
        matrix = np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]])

        W = SpatialWeights.from_matrix(matrix)
        # Squared weights: 4 + 4 + 1 + 1 = 10
        assert W.s2 == 10


# ===========================================================================
# Tests for panelbox.models.spatial.spatial_weights.SpatialWeights
# ===========================================================================


class TestModelSpatialWeightsCreation:
    """Test creation methods for models.spatial SpatialWeights."""

    def test_from_matrix_dense(self):
        """Test creating weights from a dense numpy array via from_matrix."""
        matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        W = ModelSpatialWeights.from_matrix(matrix)

        assert W.n == 3
        assert not W.normalized
        assert not W.sparse
        assert_array_almost_equal(W.matrix, matrix)

    def test_from_matrix_list(self):
        """Test creating weights from a Python list via from_matrix."""
        data = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        W = ModelSpatialWeights.from_matrix(data)

        assert W.n == 3
        assert isinstance(W.matrix, np.ndarray)
        assert_array_almost_equal(W.matrix, np.array(data, dtype=float))

    def test_init_dense_validated(self):
        """Test direct __init__ with dense array and validation enabled."""
        matrix = np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float)
        W = ModelSpatialWeights(matrix, normalized=False, validate=True)

        assert W.n == 3
        assert not W.sparse
        assert not W.normalized
        assert W._eigenvalues is None

    def test_init_sparse(self):
        """Test direct __init__ with a sparse CSR matrix."""
        dense = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        sparse = csr_matrix(dense)
        W = ModelSpatialWeights(sparse)

        assert W.n == 3
        assert W.sparse
        assert issparse(W.matrix)

    def test_validation_square(self):
        """Test that non-square matrix raises ValueError."""
        matrix = np.array([[0, 1, 1], [1, 0, 1]], dtype=float)
        with pytest.raises(ValueError, match="must be square"):
            ModelSpatialWeights(matrix)

    def test_validation_diagonal_warning(self):
        """Test that non-zero diagonal triggers warning and is zeroed out."""
        matrix = np.array([[5, 1, 0], [1, 3, 1], [0, 1, 7]], dtype=float)
        with pytest.warns(UserWarning, match="non-zero diagonal"):
            W = ModelSpatialWeights(matrix)

        # Diagonal should be forced to zero
        assert_array_almost_equal(np.diag(W.matrix), [0, 0, 0])
        # Off-diagonal should be preserved
        assert W.matrix[0, 1] == 1.0
        assert W.matrix[2, 1] == 1.0

    def test_validation_diagonal_warning_sparse(self):
        """Test that non-zero diagonal on sparse matrix triggers warning."""
        dense = np.array([[2, 1, 0], [1, 0, 1], [0, 1, 3]], dtype=float)
        sparse = csr_matrix(dense)
        with pytest.warns(UserWarning, match="non-zero diagonal"):
            W = ModelSpatialWeights(sparse)

        dense_result = W.to_dense()
        assert_array_almost_equal(np.diag(dense_result), [0, 0, 0])

    def test_validation_negative_values(self):
        """Test that negative values raise ValueError."""
        matrix = np.array([[0, -1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        with pytest.raises(ValueError, match="negative values"):
            ModelSpatialWeights(matrix)

    def test_validation_negative_values_sparse(self):
        """Test that negative values in sparse matrix raise ValueError."""
        dense = np.array([[0, -1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        sparse = csr_matrix(dense)
        with pytest.raises(ValueError, match="negative values"):
            ModelSpatialWeights(sparse)

    def test_skip_validation(self):
        """Test that validate=False skips all validation checks."""
        # Non-square matrix -- normally invalid, but validation is skipped
        matrix = np.array([[0, 1, -1], [1, 0, 1], [-1, 1, 5]], dtype=float)
        W = ModelSpatialWeights(matrix, validate=False)
        assert W.n == 3
        # The matrix is stored as-is, no changes
        assert_array_almost_equal(W.matrix, matrix)

    def test_from_contiguity_import_error(self):
        """Test from_contiguity raises ImportError without libpysal."""
        # Mock the import inside the method to simulate missing libpysal
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def _mock_import(name, *args, **kwargs):
            if name == "libpysal.weights" or name.startswith("libpysal"):
                raise ImportError("No module named 'libpysal'")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=_mock_import),
            pytest.raises(ImportError, match="libpysal"),
        ):
            ModelSpatialWeights.from_contiguity(None, criterion="queen")

    def test_from_distance_import_error(self):
        """Test from_distance raises ImportError without libpysal."""
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def _mock_import(name, *args, **kwargs):
            if name == "libpysal.weights" or name.startswith("libpysal"):
                raise ImportError("No module named 'libpysal'")
            return original_import(name, *args, **kwargs)

        coords = np.array([[0, 0], [1, 1]])
        with (
            patch("builtins.__import__", side_effect=_mock_import),
            pytest.raises(ImportError, match="libpysal"),
        ):
            ModelSpatialWeights.from_distance(coords, threshold=2.0)

    def test_from_knn_import_error(self):
        """Test from_knn raises ImportError without libpysal."""
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def _mock_import(name, *args, **kwargs):
            if name == "libpysal.weights" or name.startswith("libpysal"):
                raise ImportError("No module named 'libpysal'")
            return original_import(name, *args, **kwargs)

        coords = np.array([[0, 0], [1, 1], [2, 2]])
        with (
            patch("builtins.__import__", side_effect=_mock_import),
            pytest.raises(ImportError, match="libpysal"),
        ):
            ModelSpatialWeights.from_knn(coords, k=2)

    def test_from_contiguity_invalid_criterion(self):
        """Test from_contiguity with invalid criterion.

        This test is skipped if libpysal is not installed, since the
        ImportError is raised before the criterion check.
        """
        pytest.importorskip("libpysal")
        with pytest.raises(ValueError, match="Unknown criterion"):
            ModelSpatialWeights.from_contiguity(None, criterion="invalid")


class TestModelSpatialWeightsNormalization:
    """Test normalization methods for models.spatial SpatialWeights."""

    def test_row_standardize_dense(self):
        """Test row standardization on a dense matrix."""
        matrix = np.array([[0, 1, 3], [2, 0, 0], [1, 2, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)
        result = W.standardize("row")

        # Each row should sum to 1
        row_sums = W.matrix.sum(axis=1)
        assert_allclose(row_sums, [1.0, 1.0, 1.0])
        assert W.normalized
        # Check method chaining
        assert result is W

    def test_row_standardize_sparse(self):
        """Test row standardization on a sparse matrix."""
        dense = np.array([[0, 2, 1], [2, 0, 0], [1, 0, 0]], dtype=float)
        sparse = csr_matrix(dense)
        W = ModelSpatialWeights(sparse)
        W.standardize("row")

        row_sums = np.asarray(W.matrix.sum(axis=1)).flatten()
        assert_allclose(row_sums, [1.0, 1.0, 1.0])
        assert W.normalized

    def test_spectral_standardize(self):
        """Test spectral normalization (max eigenvalue = 1 after)."""
        matrix = np.array([[0, 2, 0], [2, 0, 2], [0, 2, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)
        W.standardize("spectral")

        assert W.normalized
        # After spectral normalization, max |eigenvalue| should be ~1
        eigenvalues = np.linalg.eigvals(W.matrix)
        max_abs_eig = np.max(np.abs(eigenvalues.real))
        assert max_abs_eig == pytest.approx(1.0, abs=1e-10)

    def test_standardize_invalid_method(self):
        """Test that unknown normalization method raises ValueError."""
        matrix = np.array([[0, 1], [1, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)
        with pytest.raises(ValueError, match="Unknown method"):
            W.standardize("invalid")

    def test_standardize_clears_eigenvalue_cache(self):
        """Test that standardize clears the eigenvalue cache."""
        matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)

        # Force eigenvalue computation to populate cache
        _ = W.eigenvalues
        assert W._eigenvalues is not None

        # Standardize should clear the cache
        W.standardize("row")
        assert W._eigenvalues is None


class TestModelSpatialWeightsProperties:
    """Test eigenvalue, s0, s1, s2 properties for models.spatial SpatialWeights."""

    def test_eigenvalues_dense(self):
        """Test eigenvalue computation for a dense matrix."""
        matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)

        eigs = W.eigenvalues
        # Symmetric matrix -> real eigenvalues
        expected_eigs = np.sort(np.linalg.eigvals(matrix).real)
        actual_sorted = np.sort(eigs.real)
        assert_allclose(actual_sorted, expected_eigs, atol=1e-10)

    def test_eigenvalues_sparse(self):
        """Test eigenvalue computation for a sparse matrix."""
        dense = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        sparse = csr_matrix(dense)
        W = ModelSpatialWeights(sparse)

        eigs = W.eigenvalues
        expected = np.sort(np.linalg.eigvals(dense).real)
        actual_sorted = np.sort(eigs.real)
        assert_allclose(actual_sorted, expected, atol=1e-10)

    def test_eigenvalue_caching(self):
        """Test that eigenvalues are computed once and then cached."""
        matrix = np.array([[0, 1], [1, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)

        assert W._eigenvalues is None

        eigs1 = W.eigenvalues
        assert W._eigenvalues is not None

        # Second access should return the same cached object
        eigs2 = W.eigenvalues
        assert eigs1 is eigs2

    def test_s0_dense(self):
        """Test s0 (sum of all weights) for dense matrix."""
        matrix = np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)
        # s0 = 1 + 2 + 3 + 4 + 5 + 6 = 21
        assert W.s0 == pytest.approx(21.0)

    def test_s0_sparse(self):
        """Test s0 (sum of all weights) for sparse matrix."""
        dense = np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0]], dtype=float)
        sparse = csr_matrix(dense)
        W = ModelSpatialWeights(sparse)
        assert W.s0 == pytest.approx(21.0)

    def test_s1_dense(self):
        """Test s1 for dense matrix.

        s1 = 0.5 * sum_i (row_sum_i + col_sum_i)^2
        """
        matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)

        # Row sums: [1, 2, 1], Col sums: [1, 2, 1]
        # (row+col)^2 = [4, 16, 4] => s1 = 0.5 * 24 = 12
        assert W.s1 == pytest.approx(12.0)

    def test_s1_sparse(self):
        """Test s1 for sparse matrix."""
        dense = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        sparse = csr_matrix(dense)
        W = ModelSpatialWeights(sparse)
        assert W.s1 == pytest.approx(12.0)

    def test_s2_dense(self):
        """Test s2 (sum of squared weights) for dense matrix."""
        matrix = np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)
        # s2 = 4 + 4 + 9 + 9 = 26
        assert W.s2 == pytest.approx(26.0)

    def test_s2_sparse(self):
        """Test s2 (sum of squared weights) for sparse matrix."""
        dense = np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float)
        sparse = csr_matrix(dense)
        W = ModelSpatialWeights(sparse)
        assert W.s2 == pytest.approx(26.0)


class TestModelSpatialWeightsConversion:
    """Test conversion methods for models.spatial SpatialWeights."""

    def test_to_sparse(self):
        """Test converting a dense weight matrix to sparse."""
        matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)

        sparse_result = W.to_sparse()
        assert issparse(sparse_result)
        assert_array_almost_equal(sparse_result.toarray(), matrix)

    def test_to_sparse_already_sparse(self):
        """Test to_sparse when matrix is already sparse returns same object."""
        dense = np.array([[0, 1], [1, 0]], dtype=float)
        sparse = csr_matrix(dense)
        W = ModelSpatialWeights(sparse)

        result = W.to_sparse()
        assert result is W.matrix  # Should be the same object

    def test_to_dense(self):
        """Test converting a sparse weight matrix to dense."""
        dense = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        sparse = csr_matrix(dense)
        W = ModelSpatialWeights(sparse)

        dense_result = W.to_dense()
        assert isinstance(dense_result, np.ndarray)
        assert_array_almost_equal(dense_result, dense)

    def test_to_dense_already_dense(self):
        """Test to_dense when matrix is already dense returns same object."""
        matrix = np.array([[0, 1], [1, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)

        result = W.to_dense()
        assert result is W.matrix  # Should be the same object

    def test_spatial_lag(self):
        """Test spatial lag computation Wx."""
        matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)

        x = np.array([1.0, 2.0, 3.0])
        lag = W.spatial_lag(x)

        # Wx = [[0,1,0],[1,0,1],[0,1,0]] @ [1,2,3] = [2, 4, 2]
        expected = np.array([2.0, 4.0, 2.0])
        assert_allclose(lag, expected)

    def test_spatial_lag_2d(self):
        """Test spatial lag computation with 2D input."""
        matrix = np.array([[0, 1], [1, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)

        x = np.array([[1.0, 10.0], [2.0, 20.0]])
        lag = W.spatial_lag(x)

        expected = np.array([[2.0, 20.0], [1.0, 10.0]])
        assert_allclose(lag, expected)

    def test_spatial_lag_sparse(self):
        """Test spatial lag with sparse weight matrix."""
        dense = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        sparse = csr_matrix(dense)
        W = ModelSpatialWeights(sparse)

        x = np.array([1.0, 2.0, 3.0])
        lag = W.spatial_lag(x)

        expected = np.array([2.0, 4.0, 2.0])
        assert_allclose(lag, expected)


class TestModelSpatialWeightsBounds:
    """Test get_bounds for models.spatial SpatialWeights."""

    def test_get_bounds(self):
        """Test spatial coefficient bounds computation."""
        # Symmetric matrix with known eigenvalues
        matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)

        lower, upper = W.get_bounds()

        # Eigenvalues of this matrix: -sqrt(2), 0, sqrt(2)
        # lower = 1/(-sqrt(2)) = -1/sqrt(2) ~ -0.707
        # upper = 1/sqrt(2) ~ 0.707
        assert lower == pytest.approx(1.0 / (-np.sqrt(2)), abs=1e-6)
        assert upper == pytest.approx(1.0 / np.sqrt(2), abs=1e-6)
        assert lower < upper

    def test_get_bounds_clipping(self):
        """Test that bounds are clipped to (-0.99, 0.99)."""
        # Identity-like off-diagonal -- small eigenvalues produce large bounds
        # that should be clipped
        matrix = np.array([[0, 0.01], [0.01, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)

        lower, upper = W.get_bounds()
        assert lower >= -0.99
        assert upper <= 0.99


class TestModelSpatialWeightsDisplay:
    """Test display/repr/summary for models.spatial SpatialWeights."""

    def test_repr(self):
        """Test string representation."""
        matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)

        repr_str = repr(W)
        assert "SpatialWeights" in repr_str
        assert "n=3" in repr_str
        assert "normalized=False" in repr_str
        assert "sparse=False" in repr_str
        assert "s0=" in repr_str

    def test_summary_normalized(self, capsys):
        """Test summary output for a normalized weight matrix."""
        matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)
        W.standardize("row")
        W.summary()

        captured = capsys.readouterr()
        assert "Spatial Weight Matrix Summary" in captured.out
        assert "Number of units:     3" in captured.out
        assert "Normalized:         True" in captured.out
        assert "All = 1.0" in captured.out

    def test_summary_not_normalized(self, capsys):
        """Test summary output for a non-normalized weight matrix."""
        matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)
        W.summary()

        captured = capsys.readouterr()
        assert "Spatial Weight Matrix Summary" in captured.out
        assert "Normalized:         False" in captured.out
        assert "Sparse:             False" in captured.out
        assert "Row sums:" in captured.out
        assert "Number of edges:" in captured.out
        assert "Density:" in captured.out

    def test_summary_sparse(self, capsys):
        """Test summary output for a sparse weight matrix."""
        dense = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        sparse = csr_matrix(dense)
        W = ModelSpatialWeights(sparse)
        W.summary()

        captured = capsys.readouterr()
        assert "Sparse:             True" in captured.out
        assert "Number of edges:" in captured.out

    def test_plot_matplotlib(self):
        """Test matplotlib plot (Agg backend, no display)."""
        import matplotlib.pyplot as plt

        matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)

        # Plot without geodataframe -- should succeed without error
        W.plot(backend="matplotlib")
        plt.close("all")

    def test_plot_plotly_raises(self):
        """Test that plotly backend raises NotImplementedError."""
        matrix = np.array([[0, 1], [1, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)

        with pytest.raises(NotImplementedError, match="Plotly backend"):
            W.plot(backend="plotly")

    def test_plot_invalid_backend(self):
        """Test that invalid backend raises ValueError."""
        matrix = np.array([[0, 1], [1, 0]], dtype=float)
        W = ModelSpatialWeights(matrix)

        with pytest.raises(ValueError, match="Unknown backend"):
            W.plot(backend="invalid_backend")
