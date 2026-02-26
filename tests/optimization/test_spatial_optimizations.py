"""Tests for spatial model performance optimizations.

Tests panelbox/optimization/spatial_optimizations.py covering:
- EigenvalueCache: caching, hashing, force_recalc, clear
- SparseOperations: is_sparse_efficient, to_sparse, spatial_lag, log_determinant
- ChebyshevApproximation: log_determinant, trace_powers
- Numba JIT functions (or fallbacks): compute_spatial_lag_fast, compute_quadratic_form, row_standardize_fast
- optimize_spatial_model decorator
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix, issparse

from panelbox.optimization.spatial_optimizations import (
    ChebyshevApproximation,
    EigenvalueCache,
    SparseOperations,
    compute_quadratic_form,
    compute_spatial_lag_fast,
    get_eigenvalues_cached,
    optimize_spatial_model,
    row_standardize_fast,
)


@pytest.fixture
def small_W():
    """Small 5x5 row-standardized spatial weight matrix (rook contiguity)."""
    np.random.seed(42)
    W = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=float,
    )
    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums
    return W


@pytest.fixture
def sparse_W(small_W):
    """Sparse version of the small weight matrix."""
    return csr_matrix(small_W)


class TestEigenvalueCache:
    """Tests for EigenvalueCache class."""

    def test_basic_caching(self, small_W):
        """Cache should return same eigenvalues on second call."""
        cache = EigenvalueCache()
        eigs1 = cache.get_eigenvalues(small_W)
        eigs2 = cache.get_eigenvalues(small_W)

        assert_allclose(eigs1, eigs2)

    def test_force_recalc(self, small_W):
        """force_recalc should bypass cache."""
        cache = EigenvalueCache()
        eigs1 = cache.get_eigenvalues(small_W)
        eigs2 = cache.get_eigenvalues(small_W, force_recalc=True)

        assert_allclose(eigs1, eigs2)

    def test_different_matrices(self):
        """Different matrices should have different eigenvalues."""
        cache = EigenvalueCache()
        W1 = np.eye(3)
        W2 = np.ones((3, 3)) / 3

        eigs1 = cache.get_eigenvalues(W1)
        eigs2 = cache.get_eigenvalues(W2)

        assert not np.allclose(eigs1, eigs2)

    def test_clear(self, small_W):
        """clear should remove cached results."""
        cache = EigenvalueCache()
        cache.get_eigenvalues(small_W)
        assert len(cache._cache) > 0

        cache.clear()
        assert len(cache._cache) == 0

    def test_eigenvalues_correct(self, small_W):
        """Eigenvalues should match numpy computation."""
        cache = EigenvalueCache()
        eigs_cached = np.sort(cache.get_eigenvalues(small_W))
        eigs_direct = np.sort(np.linalg.eigvals(small_W).real)

        assert_allclose(eigs_cached, eigs_direct, atol=1e-10)

    def test_sparse_eigenvalues(self, sparse_W):
        """Should handle sparse matrices (uses eigs with k < n-2)."""
        cache = EigenvalueCache()
        # sparse_W is 5x5, k = min(3, 100) = 3
        eigs = cache.get_eigenvalues(sparse_W)
        assert len(eigs) > 0
        assert np.all(np.isfinite(eigs))

    def test_matrix_hash_dense(self):
        """Hash should be different for different dense matrices."""
        cache = EigenvalueCache()
        W1 = np.eye(3)
        W2 = np.zeros((3, 3))

        hash1 = cache._matrix_hash(W1)
        hash2 = cache._matrix_hash(W2)

        assert hash1 != hash2

    def test_matrix_hash_sparse(self):
        """Hash should work for sparse matrices."""
        cache = EigenvalueCache()
        W = csr_matrix(np.eye(3))
        h = cache._matrix_hash(W)
        assert isinstance(h, int)


class TestGetEigenvaluesCached:
    """Test module-level cached eigenvalue function."""

    def test_basic_usage(self, small_W):
        """Should return eigenvalues using global cache."""
        eigs = get_eigenvalues_cached(small_W)
        assert len(eigs) == small_W.shape[0]
        assert np.all(np.isfinite(eigs))


class TestSparseOperations:
    """Tests for SparseOperations class."""

    def test_is_sparse_efficient_dense(self, small_W):
        """Dense matrix with many nonzeros should not be sparse efficient."""
        # small_W has ~8/25 = 32% nonzeros, not sparse efficient at 10% threshold
        result = SparseOperations.is_sparse_efficient(small_W, threshold=0.1)
        assert not result  # 32% > 10% threshold

    def test_is_sparse_efficient_sparse_input(self, sparse_W):
        """Sparse matrix should always be efficient."""
        assert SparseOperations.is_sparse_efficient(sparse_W)

    def test_is_sparse_efficient_very_sparse(self):
        """Very sparse matrix should be detected as efficient."""
        W = np.zeros((100, 100))
        W[0, 1] = 1.0
        W[1, 0] = 1.0
        assert SparseOperations.is_sparse_efficient(W, threshold=0.1)

    def test_to_sparse_already_sparse(self, sparse_W):
        """Should return sparse matrix as-is."""
        result = SparseOperations.to_sparse(sparse_W)
        assert issparse(result)

    def test_to_sparse_dense_not_efficient(self):
        """Dense matrix that's not sparse-efficient should stay dense."""
        W = np.ones((3, 3))
        result = SparseOperations.to_sparse(W)
        assert not issparse(result)

    def test_to_sparse_dense_efficient(self):
        """Dense matrix that's sparse-efficient should be converted."""
        W = np.zeros((100, 100))
        W[0, 1] = 1.0
        result = SparseOperations.to_sparse(W)
        assert issparse(result)

    def test_spatial_lag_dense(self, small_W):
        """Spatial lag Wy for dense W."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Wy = SparseOperations.spatial_lag(small_W, y)

        expected = small_W @ y
        assert_allclose(Wy, expected)

    def test_spatial_lag_sparse(self, sparse_W):
        """Spatial lag Wy for sparse W."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Wy = SparseOperations.spatial_lag(sparse_W, y)

        expected = sparse_W.toarray() @ y
        assert_allclose(Wy, expected)

    def test_log_determinant_eigen(self, small_W):
        """Log-determinant via eigenvalue method."""
        rho = 0.3
        I_rho_W = np.eye(5) - rho * small_W
        log_det = SparseOperations.log_determinant(I_rho_W, method="eigen")

        expected = np.log(np.abs(np.linalg.det(I_rho_W)))
        assert_allclose(log_det, expected, atol=1e-10)

    def test_log_determinant_cholesky(self, small_W):
        """Log-determinant via Cholesky method."""
        rho = 0.1
        I_rho_W = np.eye(5) - rho * small_W
        log_det = SparseOperations.log_determinant(I_rho_W, method="cholesky")

        expected = np.log(np.abs(np.linalg.det(I_rho_W)))
        assert_allclose(log_det, expected, atol=1e-6)

    def test_log_determinant_default(self, small_W):
        """Log-determinant via default slogdet method."""
        rho = 0.3
        I_rho_W = np.eye(5) - rho * small_W
        log_det = SparseOperations.log_determinant(I_rho_W, method="slogdet")

        expected = np.log(np.abs(np.linalg.det(I_rho_W)))
        assert_allclose(log_det, expected, atol=1e-10)

    def test_log_determinant_sparse_lu(self):
        """Log-determinant via sparse LU decomposition."""
        W = np.array(
            [
                [0, 0.5, 0.5, 0, 0],
                [0.5, 0, 0, 0.5, 0],
                [0.5, 0, 0, 0, 0.5],
                [0, 0.5, 0, 0, 0.5],
                [0, 0, 0.5, 0.5, 0],
            ]
        )
        rho = 0.3
        I_rho_W = np.eye(5) - rho * W
        I_rho_W_sparse = csr_matrix(I_rho_W)

        log_det_sparse = SparseOperations.log_determinant(I_rho_W_sparse, method="lu")
        log_det_dense = np.log(np.abs(np.linalg.det(I_rho_W)))

        assert_allclose(log_det_sparse, log_det_dense, atol=1e-10)

    def test_log_determinant_sparse_nonlu(self, sparse_W):
        """Sparse matrix with non-LU method converts to dense."""
        rho = 0.3
        I_rho_W_dense = np.eye(5) - rho * sparse_W.toarray()
        I_rho_W_sparse = csr_matrix(I_rho_W_dense)

        log_det = SparseOperations.log_determinant(I_rho_W_sparse, method="eigen")
        expected = np.log(np.abs(np.linalg.det(I_rho_W_dense)))

        assert_allclose(log_det, expected, atol=1e-10)

    def test_log_determinant_cholesky_fallback(self):
        """Cholesky should fallback to eigen for non-PD matrix."""
        # Create a matrix where I_rho_W @ I_rho_W.T might fail cholesky
        W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        W = W / W.sum(axis=1, keepdims=True)
        rho = 0.9  # Large rho might cause issues
        I_rho_W = np.eye(3) - rho * W

        # Should still compute (either via cholesky or fallback to eigen)
        log_det = SparseOperations.log_determinant(I_rho_W, method="cholesky")
        assert np.isfinite(log_det)


class TestNumericalFunctions:
    """Tests for Numba JIT functions (or fallbacks)."""

    def test_compute_spatial_lag_fast(self, sparse_W):
        """Fast spatial lag should match dense computation."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        W_dense = sparse_W.toarray()

        result = compute_spatial_lag_fast(sparse_W.data, sparse_W.indices, sparse_W.indptr, y)
        expected = W_dense @ y

        assert_allclose(result, expected, atol=1e-10)

    def test_compute_quadratic_form(self, small_W):
        """Quadratic form should produce a finite scalar result."""
        n = 5
        k = 2
        np.random.seed(42)
        X = np.random.randn(n, k)
        y = np.random.randn(n)

        result = compute_quadratic_form(X, small_W, y)

        # This should be a scalar value
        assert np.isfinite(result)

    def test_row_standardize_fast(self):
        """Row standardization should make each row sum to 1."""
        W = np.array(
            [[0, 1, 2], [3, 0, 1], [0, 0, 0]],
            dtype=float,
        )

        W_std = row_standardize_fast(W)

        # Non-zero rows should sum to 1
        assert_allclose(W_std[0].sum(), 1.0, atol=1e-10)
        assert_allclose(W_std[1].sum(), 1.0, atol=1e-10)

        # Zero row should remain zero
        assert_allclose(W_std[2].sum(), 0.0, atol=1e-10)

    def test_row_standardize_preserves_zeros(self):
        """Row standardization should preserve structural zeros."""
        W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        W_std = row_standardize_fast(W)

        assert W_std[0, 0] == 0.0
        assert W_std[0, 2] == 0.0
        assert W_std[2, 0] == 0.0
        assert W_std[2, 2] == 0.0


class TestChebyshevApproximation:
    """Tests for ChebyshevApproximation class."""

    def test_log_determinant_with_eigenvalues(self, small_W):
        """Log-det with provided eigenvalues."""
        cheb = ChebyshevApproximation(order=50)
        eigenvalues = np.linalg.eigvals(small_W).real
        rho = 0.3

        log_det = cheb.log_determinant(rho, small_W, eigenvalues=eigenvalues)
        expected = np.sum(np.log(1 - rho * eigenvalues))

        assert_allclose(log_det, expected, atol=1e-10)

    def test_log_determinant_without_eigenvalues(self, small_W):
        """Log-det without eigenvalues should compute them."""
        cheb = ChebyshevApproximation(order=50)
        rho = 0.3

        log_det = cheb.log_determinant(rho, small_W)
        expected = np.log(np.abs(np.linalg.det(np.eye(5) - rho * small_W)))

        assert_allclose(log_det, expected, atol=1e-6)

    def test_log_determinant_sparse(self):
        """Log-det with sparse matrix without eigenvalues."""
        W = np.array(
            [
                [0, 0.5, 0.5, 0, 0],
                [0.5, 0, 0, 0.5, 0],
                [0.5, 0, 0, 0, 0.5],
                [0, 0.5, 0, 0, 0.5],
                [0, 0, 0.5, 0.5, 0],
            ]
        )
        W_sparse = csr_matrix(W)
        cheb = ChebyshevApproximation(order=50)
        rho = 0.3

        log_det = cheb.log_determinant(rho, W_sparse)

        # Sparse approximation uses subset of eigenvalues so check sign & finiteness
        assert np.isfinite(log_det)

    def test_trace_powers_dense(self, small_W):
        """Trace of powers for dense matrix."""
        cheb = ChebyshevApproximation()
        traces = cheb.trace_powers(small_W, max_power=5)

        # Verify first few traces
        assert_allclose(traces[0], np.trace(small_W), atol=1e-10)
        assert_allclose(traces[1], np.trace(small_W @ small_W), atol=1e-10)
        assert_allclose(traces[2], np.trace(small_W @ small_W @ small_W), atol=1e-10)

    def test_trace_powers_sparse(self):
        """Trace of powers for sparse matrix."""
        W = csr_matrix(np.eye(3) * 0.5)
        cheb = ChebyshevApproximation()
        traces = cheb.trace_powers(W, max_power=3)

        # tr(0.5*I) = 1.5, tr((0.5*I)^2) = 0.75, etc.
        assert_allclose(traces[0], 1.5, atol=1e-10)
        assert_allclose(traces[1], 0.75, atol=1e-10)
        assert_allclose(traces[2], 0.375, atol=1e-10)


class TestOptimizeSpatialModelDecorator:
    """Tests for the optimize_spatial_model decorator."""

    def test_decorator_basic(self, small_W):
        """Decorator should add optimization attributes."""

        @optimize_spatial_model
        class MockSpatialModel:
            def __init__(self, W):
                self.W = W

        model = MockSpatialModel(W=small_W)

        assert hasattr(model, "_eigenvalue_cache")
        assert hasattr(model, "_use_sparse")
        assert hasattr(model, "_chebyshev")

    def test_decorator_adds_get_eigenvalues(self, small_W):
        """Decorator should add get_eigenvalues method."""

        @optimize_spatial_model
        class MockSpatialModel:
            def __init__(self, W):
                self.W = W

        model = MockSpatialModel(W=small_W)

        assert hasattr(model, "get_eigenvalues")
        eigs = model.get_eigenvalues()
        assert len(eigs) == 5
        assert np.all(np.isfinite(eigs))

    def test_decorator_sparse_conversion(self):
        """Decorator should convert sparse-efficient matrices."""
        # Very sparse 100x100 matrix
        W = np.zeros((100, 100))
        W[0, 1] = 0.5
        W[0, 2] = 0.5
        W[1, 0] = 1.0
        W[2, 0] = 1.0

        @optimize_spatial_model
        class MockSpatialModel:
            def __init__(self, W):
                self.W = W

        model = MockSpatialModel(W=W)
        assert model._use_sparse
        assert issparse(model.W)

    def test_decorator_no_W(self):
        """Decorator should handle models without W attribute."""

        @optimize_spatial_model
        class MockModel:
            def __init__(self):
                pass

        model = MockModel()
        assert hasattr(model, "_eigenvalue_cache")
        assert not model._use_sparse

    def test_decorator_existing_get_eigenvalues(self, small_W):
        """Should not override existing get_eigenvalues."""

        @optimize_spatial_model
        class MockSpatialModel:
            def __init__(self, W):
                self.W = W

            def get_eigenvalues(self):
                return np.array([42.0])

        model = MockSpatialModel(W=small_W)
        # Should use the existing method, not the added one
        eigs = model.get_eigenvalues()
        assert_allclose(eigs, [42.0])

    def test_eigenvalue_cache_reuse(self, small_W):
        """Cached eigenvalues should be reused on second call."""

        @optimize_spatial_model
        class MockSpatialModel:
            def __init__(self, W):
                self.W = W

        model = MockSpatialModel(W=small_W)

        eigs1 = model.get_eigenvalues()
        eigs2 = model.get_eigenvalues()

        assert_allclose(eigs1, eigs2)
        assert model._eigenvalue_cache is not None
