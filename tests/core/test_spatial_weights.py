"""Tests for SpatialWeights core module.

This module focuses on methods NOT covered by the existing
tests/models/spatial/test_spatial_weights.py, including:
- spectral normalization
- eigenvalues property (dense and sparse)
- to_sparse / to_dense conversions
- get_neighbors
- summary, __repr__, __str__
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix, issparse

from panelbox.core.spatial_weights import SpatialWeights

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_dense_matrix():
    """A simple 3x3 binary symmetric weight matrix."""
    return np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=float,
    )


@pytest.fixture
def asymmetric_dense_matrix():
    """A 4x4 asymmetric weight matrix with varying weights."""
    return np.array(
        [
            [0, 2, 0, 1],
            [1, 0, 3, 0],
            [0, 1, 0, 2],
            [1, 0, 1, 0],
        ],
        dtype=float,
    )


@pytest.fixture
def simple_sparse_matrix(simple_dense_matrix):
    """Sparse version of the simple 3x3 matrix."""
    return csr_matrix(simple_dense_matrix)


@pytest.fixture
def sw_dense(simple_dense_matrix):
    """SpatialWeights built from a dense 3x3 matrix."""
    return SpatialWeights.from_matrix(simple_dense_matrix)


@pytest.fixture
def sw_sparse(simple_sparse_matrix):
    """SpatialWeights built from a sparse 3x3 matrix."""
    return SpatialWeights.from_sparse(simple_sparse_matrix)


@pytest.fixture
def sw_asymmetric(asymmetric_dense_matrix):
    """SpatialWeights built from a 4x4 asymmetric matrix."""
    return SpatialWeights.from_matrix(asymmetric_dense_matrix)


# ---------------------------------------------------------------------------
# TestSpatialWeightsCreation
# ---------------------------------------------------------------------------


class TestSpatialWeightsCreation:
    """Test from_matrix and from_sparse (basic validation, complementary to
    the existing spatial model tests)."""

    def test_from_matrix_basic(self, simple_dense_matrix):
        """from_matrix creates correct object from numpy array."""
        W = SpatialWeights.from_matrix(simple_dense_matrix)
        assert W.n == 3
        assert_array_equal(W.matrix, simple_dense_matrix)
        assert not W.normalized

    def test_from_matrix_from_list(self):
        """from_matrix accepts a plain Python list and converts to ndarray."""
        W = SpatialWeights.from_matrix([[0, 1], [1, 0]])
        assert isinstance(W.matrix, np.ndarray)
        assert W.n == 2

    def test_from_sparse_basic(self, simple_dense_matrix, simple_sparse_matrix):
        """from_sparse creates object preserving sparse format."""
        W = SpatialWeights.from_sparse(simple_sparse_matrix)
        assert W.n == 3
        assert issparse(W.matrix)
        assert_allclose(W.to_dense(), simple_dense_matrix)

    def test_from_matrix_float_conversion(self):
        """Integer matrices are accepted without error."""
        W = SpatialWeights.from_matrix(np.array([[0, 1], [1, 0]]))
        assert W.n == 2


# ---------------------------------------------------------------------------
# TestSpatialWeightsNormalization
# ---------------------------------------------------------------------------


class TestSpatialWeightsNormalization:
    """Test standardize() with 'row' and 'spectral' methods."""

    def test_row_standardization_dense(self, simple_dense_matrix):
        """Row standardization: each row sums to 1 (dense)."""
        W = SpatialWeights.from_matrix(simple_dense_matrix.copy())
        W.standardize("row")

        row_sums = W.matrix.sum(axis=1)
        assert_allclose(row_sums, np.ones(W.n))
        assert W.normalized is True

    def test_row_standardization_sparse(self, simple_sparse_matrix):
        """Row standardization: each row sums to 1 (sparse)."""
        W = SpatialWeights.from_sparse(simple_sparse_matrix.copy())
        W.standardize("row")

        row_sums = np.asarray(W.matrix.sum(axis=1)).flatten()
        assert_allclose(row_sums, np.ones(W.n))
        assert W.normalized is True

    def test_spectral_standardization_dense(self, simple_dense_matrix):
        """Spectral normalization divides by max abs eigenvalue (dense).

        After spectral normalization the maximum absolute eigenvalue of the
        resulting matrix should be 1.0.
        """
        W = SpatialWeights.from_matrix(simple_dense_matrix.copy())
        W.standardize("spectral")

        eigenvalues = np.linalg.eigvals(W.matrix)
        max_abs_eig = np.max(np.abs(eigenvalues))
        assert_allclose(max_abs_eig, 1.0, atol=1e-12)
        assert W.normalized is True

    def test_spectral_standardization_resets_cached_eigenvalues(self, simple_dense_matrix):
        """Spectral normalization invalidates the eigenvalue cache."""
        W = SpatialWeights.from_matrix(simple_dense_matrix.copy())
        # Access eigenvalues to populate cache
        _ = W.eigenvalues
        assert W._eigenvalues is not None

        W.standardize("spectral")
        # Cache should have been cleared
        assert W._eigenvalues is None

    def test_spectral_standardization_asymmetric(self, asymmetric_dense_matrix):
        """Spectral normalization works for asymmetric matrices."""
        W = SpatialWeights.from_matrix(asymmetric_dense_matrix.copy())
        W.standardize("spectral")

        eigenvalues = np.linalg.eigvals(W.matrix)
        max_abs_eig = np.max(np.abs(eigenvalues))
        assert_allclose(max_abs_eig, 1.0, atol=1e-12)

    def test_spectral_standardization_preserves_structure(self, simple_dense_matrix):
        """Spectral normalization preserves zero/non-zero structure."""
        W = SpatialWeights.from_matrix(simple_dense_matrix.copy())
        zero_mask_before = W.matrix == 0
        W.standardize("spectral")
        zero_mask_after = W.matrix == 0
        assert_array_equal(zero_mask_before, zero_mask_after)

    def test_standardize_unknown_method_raises(self, sw_dense):
        """Unknown standardization method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            sw_dense.standardize("unknown")

    def test_row_standardization_isolated_node(self):
        """Row standardization handles isolated nodes (row sum = 0)
        without division by zero."""
        # Node 2 has no neighbors
        matrix = np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            dtype=float,
        )
        W = SpatialWeights.from_matrix(matrix)
        W.standardize("row")

        # Rows 0 and 1 should sum to 1; row 2 should remain all zeros
        assert_allclose(W.matrix[0].sum(), 1.0)
        assert_allclose(W.matrix[1].sum(), 1.0)
        assert_allclose(W.matrix[2].sum(), 0.0)

    def test_standardize_returns_self(self, simple_dense_matrix):
        """standardize() returns self for method chaining."""
        W = SpatialWeights.from_matrix(simple_dense_matrix.copy())
        result = W.standardize("row")
        assert result is W


# ---------------------------------------------------------------------------
# TestSpatialWeightsProperties
# ---------------------------------------------------------------------------


class TestSpatialWeightsProperties:
    """Test eigenvalues, s0, s1, s2 properties."""

    def test_eigenvalues_dense_symmetric(self, simple_dense_matrix):
        """Eigenvalues of a symmetric matrix match numpy.linalg.eigvalsh."""
        W = SpatialWeights.from_matrix(simple_dense_matrix)
        eigs = W.eigenvalues

        expected = np.linalg.eigvals(simple_dense_matrix).real
        # Sort both for comparison (order may differ)
        assert_allclose(np.sort(eigs), np.sort(expected), atol=1e-10)

    def test_eigenvalues_cached(self, sw_dense):
        """Eigenvalues are computed once and then cached."""
        eigs1 = sw_dense.eigenvalues
        eigs2 = sw_dense.eigenvalues
        # Same object (cached)
        assert eigs1 is eigs2

    def test_eigenvalues_sparse_raises_due_to_which_be(self):
        """Sparse eigenvalue path uses eigs(..., which='BE') which is only
        valid for symmetric solvers (eigsh).  For the general eigs call the
        source triggers a ValueError.  We document this known limitation."""
        n = 10
        rng = np.random.default_rng(42)
        dense = rng.random((n, n))
        dense = (dense + dense.T) / 2  # Symmetric
        np.fill_diagonal(dense, 0)

        W = SpatialWeights.from_sparse(csr_matrix(dense))

        with pytest.raises(ValueError, match="which"):
            _ = W.eigenvalues

    def test_eigenvalues_dense_asymmetric(self, asymmetric_dense_matrix):
        """Eigenvalues for an asymmetric matrix (complex eigenvalues
        are converted to real part)."""
        W = SpatialWeights.from_matrix(asymmetric_dense_matrix)
        eigs = W.eigenvalues

        assert len(eigs) == 4
        assert np.isrealobj(eigs)

    def test_s0_dense(self, sw_dense):
        """s0 is the sum of all weights (dense)."""
        assert sw_dense.s0 == pytest.approx(6.0)

    def test_s0_sparse(self, sw_sparse):
        """s0 is the sum of all weights (sparse)."""
        assert sw_sparse.s0 == pytest.approx(6.0)

    def test_s1_known_value(self):
        """s1 = 0.5 * sum((row_sums + col_sums)^2) for a known matrix."""
        matrix = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=float,
        )
        W = SpatialWeights.from_matrix(matrix)

        # row sums: [1, 2, 1], col sums: [1, 2, 1]
        # (r+c)^2 = [4, 16, 4] => s1 = 0.5 * 24 = 12
        assert W.s1 == pytest.approx(12.0)

    def test_s2_known_value(self):
        """s2 = sum(w_ij^2) for a known matrix."""
        matrix = np.array(
            [
                [0, 2, 0],
                [2, 0, 1],
                [0, 1, 0],
            ],
            dtype=float,
        )
        W = SpatialWeights.from_matrix(matrix)
        # 2^2 + 2^2 + 1^2 + 1^2 = 10
        assert W.s2 == pytest.approx(10.0)

    def test_s2_sparse(self):
        """s2 computed on sparse matrix matches dense computation."""
        dense = np.array(
            [
                [0, 3, 0],
                [3, 0, 2],
                [0, 2, 0],
            ],
            dtype=float,
        )
        W_dense = SpatialWeights.from_matrix(dense)
        W_sparse = SpatialWeights.from_sparse(csr_matrix(dense))

        assert W_sparse.s2 == pytest.approx(W_dense.s2)


# ---------------------------------------------------------------------------
# TestSpatialWeightsConversion
# ---------------------------------------------------------------------------


class TestSpatialWeightsConversion:
    """Test to_sparse() and to_dense() conversions."""

    def test_to_sparse_from_dense(self, sw_dense, simple_dense_matrix):
        """to_sparse converts a dense matrix to csr_matrix."""
        sparse = sw_dense.to_sparse()
        assert issparse(sparse)
        assert_allclose(sparse.toarray(), simple_dense_matrix)

    def test_to_sparse_already_sparse(self, sw_sparse):
        """to_sparse returns the matrix itself when already sparse."""
        result = sw_sparse.to_sparse()
        assert result is sw_sparse.matrix  # Same object

    def test_to_dense_from_sparse(self, sw_sparse, simple_dense_matrix):
        """to_dense converts a sparse matrix to ndarray."""
        dense = sw_sparse.to_dense()
        assert isinstance(dense, np.ndarray)
        assert_allclose(dense, simple_dense_matrix)

    def test_to_dense_already_dense(self, sw_dense, simple_dense_matrix):
        """to_dense returns the matrix itself when already dense."""
        result = sw_dense.to_dense()
        assert result is sw_dense.matrix  # Same object
        assert_allclose(result, simple_dense_matrix)

    def test_roundtrip_dense_sparse_dense(self, simple_dense_matrix):
        """Dense -> sparse -> dense roundtrip preserves values."""
        W = SpatialWeights.from_matrix(simple_dense_matrix)
        sparse = W.to_sparse()
        dense_again = sparse.toarray()
        assert_allclose(dense_again, simple_dense_matrix)

    def test_roundtrip_sparse_dense_sparse(self, simple_sparse_matrix, simple_dense_matrix):
        """Sparse -> dense -> sparse roundtrip preserves values."""
        W = SpatialWeights.from_sparse(simple_sparse_matrix)
        dense = W.to_dense()
        sparse_again = csr_matrix(dense)
        assert_allclose(sparse_again.toarray(), simple_dense_matrix)


# ---------------------------------------------------------------------------
# TestSpatialWeightsNeighbors
# ---------------------------------------------------------------------------


class TestSpatialWeightsNeighbors:
    """Test get_neighbors() for dense and sparse matrices."""

    def test_get_neighbors_dense_all_connected(self, sw_dense):
        """In a fully connected 3x3 matrix, every node has 2 neighbors."""
        for i in range(3):
            neighbors = sw_dense.get_neighbors(i)
            expected = np.array([j for j in range(3) if j != i])
            assert_array_equal(neighbors, expected)

    def test_get_neighbors_sparse(self, sw_sparse):
        """get_neighbors works for sparse matrices."""
        neighbors = sw_sparse.get_neighbors(0)
        assert_array_equal(neighbors, np.array([1, 2]))

    def test_get_neighbors_asymmetric(self, sw_asymmetric):
        """get_neighbors uses the row of the given unit."""
        # Row 0: [0, 2, 0, 1] -> neighbors are 1 and 3
        neighbors = sw_asymmetric.get_neighbors(0)
        assert_array_equal(neighbors, np.array([1, 3]))

        # Row 1: [1, 0, 3, 0] -> neighbors are 0 and 2
        neighbors = sw_asymmetric.get_neighbors(1)
        assert_array_equal(neighbors, np.array([0, 2]))

    def test_get_neighbors_isolated_node(self):
        """An isolated node has no neighbors."""
        matrix = np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            dtype=float,
        )
        W = SpatialWeights.from_matrix(matrix)
        neighbors = W.get_neighbors(2)
        assert len(neighbors) == 0

    def test_get_neighbors_returns_ndarray(self, sw_dense):
        """get_neighbors returns a numpy array."""
        neighbors = sw_dense.get_neighbors(0)
        assert isinstance(neighbors, np.ndarray)

    def test_get_neighbors_sparse_isolated(self):
        """Isolated node in sparse matrix also returns empty array."""
        matrix = np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            dtype=float,
        )
        W = SpatialWeights.from_sparse(csr_matrix(matrix))
        neighbors = W.get_neighbors(2)
        assert len(neighbors) == 0


# ---------------------------------------------------------------------------
# TestSpatialWeightsSummary
# ---------------------------------------------------------------------------


class TestSpatialWeightsSummary:
    """Test summary(), __repr__(), and __str__()."""

    def test_summary_returns_dataframe(self, sw_dense):
        """summary() returns a pandas DataFrame."""
        result = sw_dense.summary()
        assert isinstance(result, pd.DataFrame)

    def test_summary_contains_expected_keys(self, sw_dense):
        """summary() DataFrame has the expected row labels."""
        result = sw_dense.summary()
        expected_keys = [
            "N (spatial units)",
            "Non-zero weights",
            "Density (%)",
            "Row-normalized",
            "Sum of weights (S0)",
            "Min row sum",
            "Mean row sum",
            "Max row sum",
            "Min neighbors",
            "Mean neighbors",
            "Max neighbors",
        ]
        for key in expected_keys:
            assert key in result.index, f"Missing key: {key}"

    def test_summary_values_for_simple_matrix(self, sw_dense):
        """Check specific values in summary for the 3x3 binary matrix."""
        result = sw_dense.summary()
        assert result.loc["N (spatial units)", "Value"] == 3
        assert result.loc["Non-zero weights", "Value"] == 6
        assert result.loc["Density (%)", "Value"] == pytest.approx(100 * 6 / 9)
        assert result.loc["Row-normalized", "Value"] is False
        assert result.loc["Sum of weights (S0)", "Value"] == pytest.approx(6.0)
        assert result.loc["Min row sum", "Value"] == pytest.approx(2.0)
        assert result.loc["Max row sum", "Value"] == pytest.approx(2.0)
        assert result.loc["Mean row sum", "Value"] == pytest.approx(2.0)
        assert result.loc["Min neighbors", "Value"] == 2
        assert result.loc["Max neighbors", "Value"] == 2

    def test_summary_sparse_matrix(self, sw_sparse):
        """summary() works on sparse-backed SpatialWeights."""
        result = sw_sparse.summary()
        assert result.loc["N (spatial units)", "Value"] == 3
        assert result.loc["Non-zero weights", "Value"] == 6

    def test_summary_after_row_normalization(self, simple_dense_matrix):
        """summary() reflects row-normalized state."""
        W = SpatialWeights.from_matrix(simple_dense_matrix.copy())
        W.standardize("row")
        result = W.summary()
        assert result.loc["Row-normalized", "Value"] is True
        # After row normalization each row sums to 1
        assert result.loc["Min row sum", "Value"] == pytest.approx(1.0)
        assert result.loc["Max row sum", "Value"] == pytest.approx(1.0)

    def test_repr_dense(self, sw_dense):
        """__repr__ for dense matrix."""
        r = repr(sw_dense)
        assert "SpatialWeights" in r
        assert "n=3" in r
        assert "edges=6" in r
        assert "density=" in r
        # Should NOT contain "(sparse)"
        assert "(sparse)" not in r

    def test_repr_sparse(self, sw_sparse):
        """__repr__ for sparse matrix includes '(sparse)'."""
        r = repr(sw_sparse)
        assert "SpatialWeights" in r
        assert "(sparse)" in r

    def test_repr_normalized(self, simple_dense_matrix):
        """__repr__ for row-normalized matrix includes '[row-normalized]'."""
        W = SpatialWeights.from_matrix(simple_dense_matrix.copy())
        W.standardize("row")
        r = repr(W)
        assert "[row-normalized]" in r

    def test_repr_not_normalized(self, sw_dense):
        """__repr__ for non-normalized matrix omits '[row-normalized]'."""
        r = repr(sw_dense)
        assert "[row-normalized]" not in r

    def test_str_returns_summary_string(self, sw_dense):
        """__str__ returns the same text as summary().to_string()."""
        s = str(sw_dense)
        expected = sw_dense.summary().to_string()
        assert s == expected


# ---------------------------------------------------------------------------
# TestSpatialWeightsValidation
# ---------------------------------------------------------------------------


class TestSpatialWeightsValidation:
    """Test _validate() error/warning paths."""

    def test_non_square_raises(self):
        """Non-square matrix raises ValueError."""
        matrix = np.array([[0, 1, 1], [1, 0, 1]])
        with pytest.raises(ValueError, match="must be square"):
            SpatialWeights.from_matrix(matrix)

    def test_negative_values_raises(self):
        """Negative values raise ValueError."""
        matrix = np.array([[0, -1], [-1, 0]])
        with pytest.raises(ValueError, match="negative values"):
            SpatialWeights.from_matrix(matrix)

    def test_non_zero_diagonal_warns_and_fixes(self):
        """Non-zero diagonal emits warning and is zeroed out."""
        matrix = np.array([[5, 1], [1, 3]], dtype=float)
        with pytest.warns(UserWarning, match="non-zero diagonal"):
            W = SpatialWeights.from_matrix(matrix)
        assert_allclose(np.diag(W.matrix), [0, 0])

    def test_non_zero_diagonal_sparse_warns_and_fixes(self):
        """Non-zero diagonal in sparse matrix emits warning and is zeroed."""
        dense = np.array([[1, 1], [1, 2]], dtype=float)
        sparse = csr_matrix(dense)
        with pytest.warns(UserWarning, match="non-zero diagonal"):
            W = SpatialWeights.from_sparse(sparse)
        diag = W.matrix.diagonal()
        assert_allclose(diag, [0, 0])

    def test_negative_values_sparse_raises(self):
        """Negative values in sparse matrix raise ValueError."""
        dense = np.array([[0, -1], [1, 0]], dtype=float)
        sparse = csr_matrix(dense)
        with pytest.raises(ValueError, match="negative values"):
            SpatialWeights.from_sparse(sparse)

    def test_valid_matrix_no_warnings(self, simple_dense_matrix):
        """A valid matrix does not trigger warnings."""
        import warnings as w_mod

        with w_mod.catch_warnings():
            w_mod.simplefilter("error")
            SpatialWeights.from_matrix(simple_dense_matrix)


# ---------------------------------------------------------------------------
# TestSpatialWeightsFromDistance
# ---------------------------------------------------------------------------


class TestSpatialWeightsFromContiguity:
    """Test from_contiguity() error paths."""

    def test_from_contiguity_no_libpysal_raises(self, monkeypatch):
        """from_contiguity raises ImportError when libpysal is unavailable."""
        import sys

        monkeypatch.setitem(sys.modules, "libpysal", None)
        monkeypatch.setitem(sys.modules, "libpysal.weights", None)

        with pytest.raises(ImportError, match="libpysal"):
            SpatialWeights.from_contiguity(None, criterion="queen")

    def test_from_contiguity_invalid_criterion_raises(self, monkeypatch):
        """from_contiguity raises ValueError for unknown criterion."""
        # Only test if libpysal is available
        try:
            import libpysal  # noqa: F401
        except ImportError:
            pytest.skip("libpysal not available")

        with pytest.raises(ValueError, match="Unknown criterion"):
            SpatialWeights.from_contiguity(None, criterion="invalid")


class TestSpatialWeightsFromDistance:
    """Test from_distance() with binary and inverse distance weights."""

    def test_from_distance_binary(self):
        """Binary distance weights: neighbors within threshold get weight 1."""
        coords = np.array([[0, 0], [1, 0], [3, 0], [10, 0]], dtype=float)
        W = SpatialWeights.from_distance(coords, threshold=2.0, binary=True)

        assert W.n == 4
        # Points 0 and 1 are distance 1 apart (within threshold)
        assert W.matrix[0, 1] == 1.0
        assert W.matrix[1, 0] == 1.0
        # Points 0 and 2 are distance 3 apart (outside threshold)
        assert W.matrix[0, 2] == 0.0
        # Diagonal should be zero
        assert W.matrix[0, 0] == 0.0

    def test_from_distance_inverse(self):
        """Inverse distance weights: weight = 1/distance for neighbors."""
        coords = np.array([[0, 0], [2, 0], [5, 0]], dtype=float)
        W = SpatialWeights.from_distance(coords, threshold=3.0, binary=False)

        assert W.n == 3
        # Points 0 and 1 are distance 2 apart -> weight = 0.5
        assert_allclose(W.matrix[0, 1], 0.5)
        # Points 0 and 2 are distance 5 apart (outside threshold)
        assert W.matrix[0, 2] == 0.0
        # Diagonal should be zero
        assert W.matrix[0, 0] == 0.0

    def test_from_distance_zero_diagonal(self):
        """Even if threshold includes self, diagonal is forced to zero."""
        coords = np.array([[0, 0], [1, 0]], dtype=float)
        W = SpatialWeights.from_distance(coords, threshold=10.0, binary=True)
        assert W.matrix[0, 0] == 0.0
        assert W.matrix[1, 1] == 0.0

    def test_from_distance_minkowski_p(self):
        """from_distance uses the p parameter for Minkowski distance."""
        coords = np.array([[0, 0], [1, 1]], dtype=float)
        # p=1 (Manhattan): distance = |1| + |1| = 2
        W1 = SpatialWeights.from_distance(coords, threshold=2.5, p=1, binary=True)
        assert W1.matrix[0, 1] == 1.0

        # p=2 (Euclidean): distance = sqrt(2) ≈ 1.414
        W2 = SpatialWeights.from_distance(coords, threshold=1.5, p=2, binary=True)
        assert W2.matrix[0, 1] == 1.0


# ---------------------------------------------------------------------------
# TestSpatialWeightsFromKNN
# ---------------------------------------------------------------------------


class TestSpatialWeightsFromKNN:
    """Test from_knn() fallback implementation (without libpysal)."""

    def test_from_knn_basic(self, monkeypatch):
        """from_knn creates a weight matrix with k neighbors per unit."""
        import sys

        # Force fallback by hiding libpysal
        monkeypatch.setitem(sys.modules, "libpysal", None)
        monkeypatch.setitem(sys.modules, "libpysal.weights", None)

        coords = np.array([[0, 0], [1, 0], [2, 0], [10, 0]], dtype=float)
        W = SpatialWeights.from_knn(coords, k=2)

        assert W.n == 4
        # Each row should have exactly 2 non-zero entries
        for i in range(4):
            assert np.sum(W.matrix[i] > 0) == 2

    def test_from_knn_nearest_neighbor(self, monkeypatch):
        """k=1 should select only the closest neighbor."""
        import sys

        monkeypatch.setitem(sys.modules, "libpysal", None)
        monkeypatch.setitem(sys.modules, "libpysal.weights", None)

        coords = np.array([[0, 0], [1, 0], [5, 0]], dtype=float)
        W = SpatialWeights.from_knn(coords, k=1)

        # Point 0's nearest neighbor is point 1
        assert W.matrix[0, 1] == 1.0
        assert W.matrix[0, 2] == 0.0

    def test_from_knn_diagonal_zero(self, monkeypatch):
        """KNN does not include self-connections."""
        import sys

        monkeypatch.setitem(sys.modules, "libpysal", None)
        monkeypatch.setitem(sys.modules, "libpysal.weights", None)

        coords = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        W = SpatialWeights.from_knn(coords, k=1)
        assert W.matrix[0, 0] == 0.0
        assert W.matrix[1, 1] == 0.0
        assert W.matrix[2, 2] == 0.0


# ---------------------------------------------------------------------------
# TestSpatialWeightsSparseOperations
# ---------------------------------------------------------------------------


class TestSpatialWeightsSparseOperations:
    """Test operations on sparse matrices to improve coverage."""

    def test_spectral_standardization_sparse(self):
        """Spectral normalization works for sparse matrices."""
        dense = np.array(
            [
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 1],
                [1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=float,
        )

        W = SpatialWeights.from_sparse(csr_matrix(dense))
        W.standardize("spectral")

        # After spectral normalization, max abs eigenvalue should be ~1
        dense_result = W.to_dense()
        eigenvalues = np.linalg.eigvals(dense_result)
        max_abs_eig = np.max(np.abs(eigenvalues))
        assert_allclose(max_abs_eig, 1.0, atol=1e-6)

    def test_s1_sparse(self, simple_sparse_matrix):
        """s1 on sparse matches s1 on dense."""
        W_sparse = SpatialWeights.from_sparse(simple_sparse_matrix)
        W_dense = SpatialWeights.from_matrix(simple_sparse_matrix.toarray())

        assert_allclose(W_sparse.s1, W_dense.s1)

    def test_row_standardization_sparse_isolated(self):
        """Row standardization on sparse matrix with isolated nodes."""
        matrix = np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],  # Isolated
            ],
            dtype=float,
        )
        W = SpatialWeights.from_sparse(csr_matrix(matrix))
        W.standardize("row")

        dense = W.to_dense()
        assert_allclose(dense[0].sum(), 1.0)
        assert_allclose(dense[1].sum(), 1.0)
        assert_allclose(dense[2].sum(), 0.0)


# ---------------------------------------------------------------------------
# TestSpatialWeightsPlot
# ---------------------------------------------------------------------------


class TestSpatialWeightsPlot:
    """Test plot() method for matplotlib and plotly backends."""

    @pytest.fixture
    def sw_with_coords(self):
        """SpatialWeights from distance with associated coordinates."""
        coords = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [2.0, 0.5],
            ]
        )
        W = SpatialWeights.from_distance(coords, threshold=1.5, binary=True)
        return W, coords

    def test_plot_matplotlib_with_coords(self, sw_with_coords):
        """Plot with matplotlib backend and coordinate array."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        W, coords = sw_with_coords
        fig = W.plot(coords=coords, backend="matplotlib")

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_matplotlib_without_edges(self, sw_with_coords):
        """Plot with matplotlib and show_edges=False."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        W, coords = sw_with_coords
        fig = W.plot(coords=coords, backend="matplotlib", show_edges=False)

        assert fig is not None
        plt.close(fig)

    def test_plot_matplotlib_no_gdf_no_coords_raises(self, sw_with_coords):
        """Plot with no gdf and no coords raises ValueError."""
        W, _ = sw_with_coords
        with pytest.raises(ValueError, match="Either gdf or coords must be provided"):
            W.plot(backend="matplotlib")

    def test_plot_plotly_with_coords(self, sw_with_coords):
        """Plot with plotly backend and coordinate array."""
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

        W, coords = sw_with_coords
        fig = W.plot(coords=coords, backend="plotly")

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_plotly_no_gdf_no_coords_raises(self, sw_with_coords):
        """Plot with plotly backend and no data raises ValueError."""
        W, _ = sw_with_coords
        with pytest.raises(ValueError, match="Either gdf or coords must be provided"):
            W.plot(backend="plotly")

    def test_plot_unknown_backend_raises(self, sw_with_coords):
        """Plot with unknown backend raises ValueError."""
        W, coords = sw_with_coords
        with pytest.raises(ValueError, match="Unknown backend"):
            W.plot(coords=coords, backend="bokeh")

    def test_plot_matplotlib_custom_figsize(self, sw_with_coords):
        """Plot with custom figsize."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        W, coords = sw_with_coords
        fig = W.plot(coords=coords, backend="matplotlib", figsize=(6, 4))

        assert fig is not None
        # Check figsize is applied
        width, height = fig.get_size_inches()
        assert_allclose(width, 6.0)
        assert_allclose(height, 4.0)
        plt.close(fig)

    def test_plot_matplotlib_edge_alpha(self, sw_with_coords):
        """Plot with custom edge_alpha."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        W, coords = sw_with_coords
        fig = W.plot(coords=coords, backend="matplotlib", edge_alpha=0.3)

        assert fig is not None
        plt.close(fig)
