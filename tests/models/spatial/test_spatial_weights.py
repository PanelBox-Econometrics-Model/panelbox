"""
Tests for Spatial Weight Matrices

This module tests the SpatialWeights class functionality including
creation methods, normalization, and properties.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from scipy.sparse import csr_matrix

from panelbox.core.spatial_weights import SpatialWeights


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
            W = SpatialWeights.from_matrix(matrix)

    def test_from_matrix_not_square(self):
        """Test that non-square matrix raises error."""
        matrix = np.array([[0, 1, 1], [1, 0, 1]])

        with pytest.raises(ValueError, match="must be square"):
            W = SpatialWeights.from_matrix(matrix)

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

        # Threshold = 1.5 (connects direct neighbors only)
        W = SpatialWeights.from_distance(coords, threshold=1.5, binary=True)

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
        row_sums = W.matrix.sum(axis=1)
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
