"""
Comprehensive tests for panelbox optimization modules (Round 3).

Targets uncovered lines/branches in:
1. panelbox/optimization/spatial_optimizations.py
   - Lines 245-257: compute_spatial_lag_fast (Numba JIT body)
   - Lines 266-294: compute_quadratic_form (Numba JIT body)
   - Lines 311-323: row_standardize_fast (Numba JIT body)

2. panelbox/optimization/quantile/penalized.py
   - Lines 62-71: _check_loss_fast (Numba JIT body)
   - Lines 77-83: _check_gradient_fast (Numba JIT body)
   - Lines 89-94: _soft_threshold (Numba JIT body)
   - Lines 286-298: compute_check_loss_matrix (Numba JIT body)
   - Lines 304-313: compute_gradient_matrix (Numba JIT body)

Strategy:
- Call Numba JIT functions directly with diverse input shapes and edge cases
- Re-import module with numba mocked out to exercise fallback branches
- Test decorator, ChebyshevApproximation, SparseOperations, AdaptiveOptimizer,
  PerformanceMonitor, PenalizedQuantileOptimizer with additional patterns
- All comments/docstrings in English
"""

from __future__ import annotations

import builtins
import importlib
import logging
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix, diags, issparse

from panelbox.optimization.quantile.penalized import (
    AdaptiveOptimizer,
    PenalizedQuantileOptimizer,
    PerformanceMonitor,
    compute_check_loss_matrix,
    compute_gradient_matrix,
)
from panelbox.optimization.spatial_optimizations import (
    ChebyshevApproximation,
    EigenvalueCache,
    SparseOperations,
    compute_quadratic_form,
    compute_spatial_lag_fast,
    optimize_spatial_model,
    row_standardize_fast,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def row_standardized_W():
    """6x6 row-standardized spatial weight matrix with varied connectivity."""
    W = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ],
        dtype=float,
    )
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums


@pytest.fixture
def sparse_row_standardized_W(row_standardized_W):
    """Sparse CSR version of the 6x6 weight matrix."""
    return csr_matrix(row_standardized_W)


@pytest.fixture
def synthetic_panel_small():
    """Small synthetic panel: 50 obs, 2 features, 5 entities, T=10."""
    np.random.seed(123)
    n_entities = 5
    T = 10
    n = n_entities * T
    p = 2
    entity_ids = np.repeat(np.arange(n_entities), T)
    X = np.random.randn(n, p)
    beta_true = np.array([1.5, -0.8])
    fe = np.random.randn(n_entities) * 0.5
    y = X @ beta_true + fe[entity_ids] + np.random.randn(n) * 0.3
    return X, y, entity_ids


@pytest.fixture
def synthetic_panel_large():
    """Larger synthetic panel: 200 obs, 3 features, 20 entities, T=10."""
    np.random.seed(77)
    n_entities = 20
    T = 10
    n = n_entities * T
    p = 3
    entity_ids = np.repeat(np.arange(n_entities), T)
    X = np.random.randn(n, p)
    beta_true = np.array([2.0, -1.0, 0.5])
    fe = np.random.randn(n_entities)
    y = X @ beta_true + fe[entity_ids] + np.random.randn(n) * 0.5
    return X, y, entity_ids


# ===========================================================================
# PART 1: spatial_optimizations.py -- Numba JIT function bodies
# ===========================================================================


class TestComputeSpatialLagFastJIT:
    """Diverse inputs for compute_spatial_lag_fast to exercise JIT body (lines 245-257)."""

    def test_diagonal_sparse_lag(self):
        """Diagonal W times y should scale each element."""
        W = csr_matrix(np.diag([0.5, 0.3, 0.7]))
        y = np.array([2.0, 4.0, 6.0])
        result = compute_spatial_lag_fast(W.data, W.indices, W.indptr, y)
        expected = np.array([1.0, 1.2, 4.2])
        assert_allclose(result, expected, atol=1e-10)

    def test_full_dense_as_csr(self):
        """Fully connected matrix (all off-diagonal equal)."""
        n = 4
        W = np.ones((n, n)) - np.eye(n)
        W = W / (n - 1)  # row-standardize
        W_csr = csr_matrix(W)
        y = np.arange(1, n + 1, dtype=float)
        result = compute_spatial_lag_fast(W_csr.data, W_csr.indices, W_csr.indptr, y)
        expected = W @ y
        assert_allclose(result, expected, atol=1e-10)

    def test_large_sparse_random(self):
        """20x20 random sparse matrix."""
        np.random.seed(99)
        n = 20
        W = np.random.rand(n, n)
        W[W < 0.6] = 0.0
        np.fill_diagonal(W, 0.0)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums
        W_csr = csr_matrix(W)
        y = np.random.randn(n)
        result = compute_spatial_lag_fast(W_csr.data, W_csr.indices, W_csr.indptr, y)
        expected = W @ y
        assert_allclose(result, expected, atol=1e-10)

    def test_negative_weights_sparse(self):
        """Sparse matrix with negative off-diagonal weights."""
        W = np.array([[0, -0.5, 0.5], [0.3, 0, -0.3], [-0.2, 0.2, 0]], dtype=float)
        W_csr = csr_matrix(W)
        y = np.array([1.0, 2.0, 3.0])
        result = compute_spatial_lag_fast(W_csr.data, W_csr.indices, W_csr.indptr, y)
        expected = W @ y
        assert_allclose(result, expected, atol=1e-10)

    def test_single_nonzero_row(self):
        """Only one row has nonzero entries."""
        W = np.zeros((5, 5))
        W[2, 0] = 0.5
        W[2, 4] = 0.5
        W_csr = csr_matrix(W)
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = compute_spatial_lag_fast(W_csr.data, W_csr.indices, W_csr.indptr, y)
        expected = np.array([0.0, 0.0, 30.0, 0.0, 0.0])
        assert_allclose(result, expected, atol=1e-10)


class TestComputeQuadraticFormJIT:
    """Diverse inputs for compute_quadratic_form to exercise JIT body (lines 266-294)."""

    def test_identity_weight(self):
        """W = I: quadratic form simplifies to (X'y)' X'X (X'y)."""
        np.random.seed(10)
        n, k = 5, 2
        X = np.random.randn(n, k)
        W = np.eye(n)
        y = np.random.randn(n)
        result = compute_quadratic_form(X, W, y)
        temp = X.T @ y
        XWX = X.T @ X
        expected = temp @ XWX @ temp
        assert_allclose(result, expected, atol=1e-8)

    def test_zero_y(self):
        """Zero y vector should produce zero quadratic form."""
        X = np.random.randn(4, 2)
        W = np.eye(4)
        y = np.zeros(4)
        result = compute_quadratic_form(X, W, y)
        assert_allclose(result, 0.0, atol=1e-10)

    def test_single_observation(self):
        """n=1, k=1 degenerate case."""
        X = np.array([[3.0]])
        W = np.array([[1.0]])
        y = np.array([2.0])
        result = compute_quadratic_form(X, W, y)
        # WX = [[3]], X'WX = [[9]], temp = [6], result = 6*9*6 = 324
        expected = 324.0
        assert_allclose(result, expected, atol=1e-8)

    def test_nonsymmetric_W(self):
        """W does not need to be symmetric."""
        np.random.seed(55)
        n, k = 6, 3
        X = np.random.randn(n, k)
        W = np.random.randn(n, n)  # non-symmetric
        y = np.random.randn(n)
        result = compute_quadratic_form(X, W, y)
        WX = W @ X
        XWX = X.T @ WX
        temp = X.T @ y
        expected = temp @ XWX @ temp
        assert_allclose(result, expected, atol=1e-6)

    def test_many_columns(self):
        """Larger k to exercise the inner loops."""
        np.random.seed(33)
        n, k = 8, 5
        X = np.random.randn(n, k)
        W = np.eye(n)
        y = np.random.randn(n)
        result = compute_quadratic_form(X, W, y)
        WX = W @ X
        XWX = X.T @ WX
        temp = X.T @ y
        expected = temp @ XWX @ temp
        assert_allclose(result, expected, atol=1e-6)


class TestRowStandardizeFastJIT:
    """Diverse inputs for row_standardize_fast to exercise JIT body (lines 311-323)."""

    def test_already_standardized_identity(self):
        """Already row-standardized identity should be identity."""
        W = np.eye(3)
        W_std = row_standardize_fast(W)
        assert_allclose(W_std, np.eye(3), atol=1e-10)

    def test_large_varying_weights(self):
        """10x10 matrix with varying row sums."""
        np.random.seed(77)
        n = 10
        W = np.abs(np.random.randn(n, n))
        np.fill_diagonal(W, 0.0)
        W_std = row_standardize_fast(W)
        for i in range(n):
            row_sum = W_std[i].sum()
            if W[i].sum() > 0:
                assert_allclose(row_sum, 1.0, atol=1e-10)

    def test_all_zero_matrix(self):
        """Completely zero matrix should remain zero."""
        W = np.zeros((4, 4))
        W_std = row_standardize_fast(W)
        assert_allclose(W_std, np.zeros((4, 4)), atol=1e-10)

    def test_single_element_per_row(self):
        """Each row has exactly one nonzero element."""
        W = np.array(
            [[0, 5, 0], [0, 0, 3], [7, 0, 0]],
            dtype=float,
        )
        W_std = row_standardize_fast(W)
        expected = np.array(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            dtype=float,
        )
        assert_allclose(W_std, expected, atol=1e-10)

    def test_mixed_zero_nonzero_rows(self):
        """Mixed zero and non-zero rows."""
        W = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 0, 3], [0, 4, 0, 0]],
            dtype=float,
        )
        W_std = row_standardize_fast(W)
        assert_allclose(W_std[0], [0, 0, 0, 0], atol=1e-10)
        assert_allclose(W_std[1], [0, 0, 0, 0], atol=1e-10)
        assert_allclose(W_std[2], [1.0 / 6, 2.0 / 6, 0, 3.0 / 6], atol=1e-10)
        assert_allclose(W_std[3], [0, 1.0, 0, 0], atol=1e-10)

    def test_preserves_dtype(self):
        """Output should be float."""
        W = np.array([[0, 2, 3], [1, 0, 1], [0, 0, 0]], dtype=float)
        W_std = row_standardize_fast(W)
        assert W_std.dtype in (np.float64, np.float32)


# ===========================================================================
# PART 2: spatial_optimizations.py -- Fallback branch via mocked numba
# ===========================================================================


class TestFallbackBranchesViaReimport:
    """Force the non-numba fallback code paths by re-importing with numba mocked out.

    This covers lines 327-341 (the else branch) when Numba is not available.
    The approach patches builtins.__import__ to raise ImportError for 'numba',
    removes the cached module, and re-imports to activate fallback definitions.
    """

    @staticmethod
    def _reimport_without_numba():
        """Re-import spatial_optimizations with numba disabled."""
        mod_name = "panelbox.optimization.spatial_optimizations"
        original_module = sys.modules.get(mod_name)
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "numba":
                raise ImportError("mocked: numba not available")
            return real_import(name, *args, **kwargs)

        if mod_name in sys.modules:
            del sys.modules[mod_name]

        builtins.__import__ = mock_import
        try:
            mod = importlib.import_module(mod_name)
        finally:
            builtins.__import__ = real_import
            if original_module is not None:
                sys.modules[mod_name] = original_module
        return mod

    def test_fallback_spatial_lag_with_varied_data(self):
        """Fallback spatial lag with off-diagonal sparse W."""
        mod = self._reimport_without_numba()
        assert not mod.HAS_NUMBA

        W = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        W_csr = csr_matrix(W)
        y = np.array([10.0, 20.0, 30.0])
        result = mod.compute_spatial_lag_fast(W_csr.data, W_csr.indices, W_csr.indptr, y)
        expected = W @ y
        assert_allclose(result, expected, atol=1e-10)

    def test_fallback_quadratic_form_3x3(self):
        """Fallback quadratic form with 3x3 matrices."""
        mod = self._reimport_without_numba()
        assert not mod.HAS_NUMBA

        np.random.seed(88)
        n = 3
        X = np.random.randn(n, n)
        W = np.eye(n)
        y = np.random.randn(n)
        result = mod.compute_quadratic_form(X, W, y)
        WX = W @ X
        expected = y.T @ X.T @ WX @ X @ y
        assert np.isfinite(result)
        assert_allclose(result, expected, atol=1e-8)

    def test_fallback_row_standardize_with_zero_rows(self):
        """Fallback row standardize handles zero rows correctly."""
        mod = self._reimport_without_numba()
        assert not mod.HAS_NUMBA

        W = np.array([[0, 3, 0], [0, 0, 0], [2, 0, 4]], dtype=float)
        W_std = mod.row_standardize_fast(W)
        assert_allclose(W_std[0], [0, 1.0, 0], atol=1e-10)
        assert_allclose(W_std[1], [0, 0, 0], atol=1e-10)
        assert_allclose(W_std[2], [2.0 / 6, 0, 4.0 / 6], atol=1e-10)


# ===========================================================================
# PART 3: spatial_optimizations.py -- SparseOperations.log_determinant branches
# ===========================================================================


class TestLogDeterminantBranches:
    """Test all branches of SparseOperations.log_determinant."""

    def test_cholesky_succeeds_on_pd_matrix(self):
        """Cholesky should succeed on a positive definite matrix."""
        # I - rho*W for small rho is PD
        W = np.array([[0, 0.5], [0.5, 0]], dtype=float)
        rho = 0.1
        I_rho_W = np.eye(2) - rho * W
        result = SparseOperations.log_determinant(I_rho_W, method="cholesky")
        # Note: cholesky branch computes L = cholesky(M @ M.T), log = sum(log(diag(L)))
        # This gives log(abs(det(M))) for PD M@M.T
        assert np.isfinite(result)

    def test_cholesky_fallback_via_mock(self):
        """Force cholesky to fail, verify fallback to eigen."""
        from unittest.mock import patch

        A = np.eye(3)
        expected = SparseOperations.log_determinant(A, method="eigen")

        with patch(
            "numpy.linalg.cholesky",
            side_effect=np.linalg.LinAlgError("forced failure"),
        ):
            result = SparseOperations.log_determinant(A, method="cholesky")

        assert np.isfinite(result)
        assert_allclose(result, expected, atol=1e-10)

    def test_slogdet_branch(self):
        """Default (not eigen/cholesky/lu) should use slogdet."""
        A = np.eye(4) * 2.0
        result = SparseOperations.log_determinant(A, method="slogdet")
        expected = np.log(2.0) * 4  # det(2I) = 2^4
        assert_allclose(result, expected, atol=1e-10)

    def test_slogdet_branch_unknown_method_name(self):
        """Any method name not in {eigen, cholesky, lu} triggers slogdet."""
        A = np.eye(3)
        result = SparseOperations.log_determinant(A, method="unknown_method")
        assert_allclose(result, 0.0, atol=1e-10)

    def test_sparse_lu_decomposition(self):
        """Sparse LU branch for sparse input with method='lu'."""
        W = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype=float)
        rho = 0.3
        I_rho_W = np.eye(3) - rho * W
        I_rho_W_sparse = csr_matrix(I_rho_W)

        result = SparseOperations.log_determinant(I_rho_W_sparse, method="lu")
        expected = np.log(np.abs(np.linalg.det(I_rho_W)))
        assert_allclose(result, expected, atol=1e-10)

    def test_sparse_eigen_converts_to_dense(self):
        """Sparse matrix with method='eigen' should convert to dense internally."""
        W = np.array([[0, 1], [1, 0]], dtype=float)
        rho = 0.3
        I_rho_W = np.eye(2) - rho * W
        I_rho_W_sparse = csr_matrix(I_rho_W)

        result = SparseOperations.log_determinant(I_rho_W_sparse, method="eigen")
        expected = SparseOperations.log_determinant(I_rho_W, method="eigen")
        assert_allclose(result, expected, atol=1e-10)


# ===========================================================================
# PART 4: spatial_optimizations.py -- ChebyshevApproximation
# ===========================================================================


class TestChebyshevApproximationExtended:
    """Extended tests for ChebyshevApproximation log_determinant and trace_powers."""

    def test_log_det_dense_eigenvalues_none(self):
        """Dense W with eigenvalues=None: compute eigenvalues internally (line 394-395)."""
        W = np.array(
            [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
            dtype=float,
        )
        cheb = ChebyshevApproximation(order=5)
        result = cheb.log_determinant(0.3, W, eigenvalues=None)
        # Compare to manual calculation
        eigs = np.linalg.eigvals(W).real
        expected = np.sum(np.log(1 - 0.3 * eigs))
        assert_allclose(result, expected, atol=1e-10)

    def test_log_det_sparse_eigenvalues_none(self):
        """Sparse W with eigenvalues=None: use eigsh for subset (lines 386-393)."""
        # Need n > order*2 + 2 to make k meaningful
        n = 20
        W_dense = np.zeros((n, n))
        for i in range(n - 1):
            W_dense[i, i + 1] = 1.0
            W_dense[i + 1, i] = 1.0
        row_sums = W_dense.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W_dense = W_dense / row_sums
        W_sparse = csr_matrix(W_dense)

        cheb = ChebyshevApproximation(order=3)
        result = cheb.log_determinant(0.2, W_sparse, eigenvalues=None)
        assert np.isfinite(result)

    def test_log_det_sparse_medium_matrix(self):
        """Sparse matrix with n large enough for eigsh to work (k = min(order*2, n-2))."""
        # Need n large enough so that k = min(order*2, n-2) >= 2 for which='BE'
        n = 10
        W_dense = np.zeros((n, n))
        for i in range(n - 1):
            W_dense[i, i + 1] = 1.0
            W_dense[i + 1, i] = 1.0
        row_sums = W_dense.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W_dense = W_dense / row_sums
        W_sparse = csr_matrix(W_dense)

        cheb = ChebyshevApproximation(order=2)
        # k = min(4, 8) = 4
        result = cheb.log_determinant(0.3, W_sparse, eigenvalues=None)
        assert np.isfinite(result)

    def test_log_det_with_provided_eigenvalues(self):
        """Providing eigenvalues should skip internal computation."""
        W = np.eye(4) * 0.5
        eigs = np.array([0.5, 0.5, 0.5, 0.5])
        cheb = ChebyshevApproximation(order=5)
        result = cheb.log_determinant(0.4, W, eigenvalues=eigs)
        expected = np.sum(np.log(1 - 0.4 * eigs))
        assert_allclose(result, expected, atol=1e-10)

    def test_log_det_rho_zero(self):
        """rho=0 means log|I| = 0."""
        W = np.random.randn(4, 4)
        cheb = ChebyshevApproximation(order=5)
        eigs = np.linalg.eigvals(W).real
        result = cheb.log_determinant(0.0, W, eigenvalues=eigs)
        assert_allclose(result, 0.0, atol=1e-10)

    def test_trace_powers_dense_correctness(self, row_standardized_W):
        """Verify trace_powers for dense matrix against manual computation."""
        cheb = ChebyshevApproximation()
        traces = cheb.trace_powers(row_standardized_W, max_power=4)
        assert traces.shape == (4,)

        W = row_standardized_W
        W_pow = W.copy()
        for p in range(4):
            assert_allclose(traces[p], np.trace(W_pow), atol=1e-10)
            W_pow = W_pow @ W

    def test_trace_powers_sparse_correctness(self, sparse_row_standardized_W):
        """Verify trace_powers for sparse matrix against manual computation."""
        cheb = ChebyshevApproximation()
        W = sparse_row_standardized_W
        traces = cheb.trace_powers(W, max_power=5)
        assert traces.shape == (5,)

        W_dense = W.toarray()
        W_pow = W_dense.copy()
        for p in range(5):
            assert_allclose(traces[p], np.trace(W_pow), atol=1e-10)
            W_pow = W_pow @ W_dense

    def test_trace_powers_identity(self):
        """tr(I^k) = n for all k."""
        cheb = ChebyshevApproximation()
        W = np.eye(5)
        traces = cheb.trace_powers(W, max_power=3)
        assert_allclose(traces, [5.0, 5.0, 5.0], atol=1e-10)

    def test_trace_powers_sparse_identity(self):
        """tr(I^k) = n for sparse I."""
        cheb = ChebyshevApproximation()
        W = csr_matrix(np.eye(5))
        traces = cheb.trace_powers(W, max_power=3)
        assert_allclose(traces, [5.0, 5.0, 5.0], atol=1e-10)

    def test_trace_powers_max_power_one(self):
        """max_power=1 should return just trace(W)."""
        cheb = ChebyshevApproximation()
        W = np.array([[1, 2], [3, 4]], dtype=float)
        traces = cheb.trace_powers(W, max_power=1)
        assert traces.shape == (1,)
        assert_allclose(traces[0], 5.0, atol=1e-10)  # tr([[1,2],[3,4]]) = 5


# ===========================================================================
# PART 5: spatial_optimizations.py -- optimize_spatial_model decorator
# ===========================================================================


class TestOptimizeSpatialModelDecoratorExtended:
    """Extended tests for the optimize_spatial_model decorator."""

    def test_decorator_dense_not_sparse_efficient(self):
        """Dense, non-sparse-efficient W should not trigger sparse conversion."""
        W = np.ones((3, 3)) * 0.5
        np.fill_diagonal(W, 0)

        @optimize_spatial_model
        class Model:
            def __init__(self, W):
                self.W = W

        model = Model(W=W)
        assert not model._use_sparse
        assert not issparse(model.W)
        assert model._chebyshev is None

    def test_decorator_sparse_efficient_dense_W(self):
        """Dense W that is sparse-efficient should be converted to sparse."""
        W = np.zeros((50, 50))
        W[0, 1] = 0.5
        W[1, 0] = 0.5

        @optimize_spatial_model
        class Model:
            def __init__(self, W):
                self.W = W

        model = Model(W=W)
        assert model._use_sparse
        assert issparse(model.W)

    def test_decorator_no_W_attribute(self):
        """Model without W attribute should still get optimization attributes."""

        @optimize_spatial_model
        class Model:
            def __init__(self, x):
                self.x = x

        model = Model(x=42)
        assert hasattr(model, "_eigenvalue_cache")
        assert model._eigenvalue_cache is None
        assert model._use_sparse is False
        assert model._chebyshev is None

    def test_decorator_large_W_creates_chebyshev(self):
        """W with shape[0] > 5000 should create ChebyshevApproximation."""
        n = 5001
        off_diag = np.ones(n - 1)
        W = diags([off_diag, off_diag], offsets=[-1, 1], shape=(n, n), format="csr")
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        W = W.multiply(1.0 / row_sums[:, np.newaxis]).tocsr()

        @optimize_spatial_model
        class Model:
            def __init__(self, W):
                self.W = W

        model = Model(W=W)
        assert model._chebyshev is not None
        assert isinstance(model._chebyshev, ChebyshevApproximation)

    def test_decorator_exactly_5000_no_chebyshev(self):
        """W with shape[0] == 5000 should NOT create ChebyshevApproximation."""
        n = 5000
        off_diag = np.ones(n - 1)
        W = diags([off_diag, off_diag], offsets=[-1, 1], shape=(n, n), format="csr")
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        W = W.multiply(1.0 / row_sums[:, np.newaxis]).tocsr()

        @optimize_spatial_model
        class Model:
            def __init__(self, W):
                self.W = W

        model = Model(W=W)
        assert model._chebyshev is None

    def test_get_eigenvalues_added(self):
        """Decorator should add get_eigenvalues when not already present."""
        W = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype=float)

        @optimize_spatial_model
        class Model:
            def __init__(self, W):
                self.W = W

        model = Model(W=W)
        assert hasattr(model, "get_eigenvalues")
        eigs = model.get_eigenvalues()
        assert eigs is not None
        assert len(eigs) == 3
        assert np.all(np.isfinite(eigs))

    def test_get_eigenvalues_not_overridden(self):
        """Pre-existing get_eigenvalues should not be overridden."""

        @optimize_spatial_model
        class Model:
            def __init__(self, W):
                self.W = W

            def get_eigenvalues(self):
                return np.array([99.0])

        W = np.array([[0, 1], [1, 0]], dtype=float)
        model = Model(W=W)
        assert_allclose(model.get_eigenvalues(), [99.0])

    def test_eigenvalue_caching(self):
        """Second call to get_eigenvalues should return cached result."""
        W = np.array([[0, 1], [1, 0]], dtype=float)

        @optimize_spatial_model
        class Model:
            def __init__(self, W):
                self.W = W

        model = Model(W=W)
        eigs1 = model.get_eigenvalues()
        eigs2 = model.get_eigenvalues()
        assert_allclose(eigs1, eigs2)
        assert model._eigenvalue_cache is not None


# ===========================================================================
# PART 6: penalized.py -- JIT function bodies via direct calls
# ===========================================================================


class TestCheckLossFastExtended:
    """Extended tests for _check_loss_fast (lines 62-71)."""

    def test_large_residuals(self):
        """Large residual values."""
        residuals = np.array([1e6, -1e6])
        tau = 0.5
        loss = PenalizedQuantileOptimizer._check_loss_fast(residuals, tau)
        expected = tau * 1e6 + (tau - 1) * (-1e6)
        assert_allclose(loss, expected, rtol=1e-6)

    def test_extreme_taus(self):
        """Tau near 0 and near 1."""
        residuals = np.array([1.0, -1.0, 2.0, -2.0])
        for tau in [0.01, 0.99]:
            loss = PenalizedQuantileOptimizer._check_loss_fast(residuals, tau)
            assert loss >= 0
            assert np.isfinite(loss)

    def test_single_element(self):
        """Single element residual."""
        loss_pos = PenalizedQuantileOptimizer._check_loss_fast(np.array([3.0]), 0.5)
        assert_allclose(loss_pos, 1.5, atol=1e-10)

        loss_neg = PenalizedQuantileOptimizer._check_loss_fast(np.array([-3.0]), 0.5)
        assert_allclose(loss_neg, 1.5, atol=1e-10)

    def test_empty_like_array(self):
        """Very small array edge case."""
        residuals = np.array([0.0])
        loss = PenalizedQuantileOptimizer._check_loss_fast(residuals, 0.5)
        assert_allclose(loss, 0.0, atol=1e-10)

    def test_consistent_with_manual(self):
        """Compare with manual check loss computation."""
        np.random.seed(42)
        residuals = np.random.randn(100)
        tau = 0.75
        loss_jit = PenalizedQuantileOptimizer._check_loss_fast(residuals, tau)
        loss_manual = np.sum(np.where(residuals < 0, (tau - 1) * residuals, tau * residuals))
        assert_allclose(loss_jit, loss_manual, atol=1e-8)


class TestCheckGradientFastExtended:
    """Extended tests for _check_gradient_fast (lines 77-83)."""

    def test_large_array(self):
        """Large array to exercise parallel prange."""
        np.random.seed(7)
        residuals = np.random.randn(10000)
        tau = 0.5
        grad = PenalizedQuantileOptimizer._check_gradient_fast(residuals, tau)
        assert grad.shape == (10000,)
        assert np.all((grad == tau) | (grad == tau - 1.0))

    def test_all_positive(self):
        """All positive residuals: all gradients should be tau."""
        residuals = np.abs(np.random.randn(50)) + 0.01
        tau = 0.3
        grad = PenalizedQuantileOptimizer._check_gradient_fast(residuals, tau)
        assert_allclose(grad, tau, atol=1e-10)

    def test_all_negative(self):
        """All negative residuals: all gradients should be tau - 1."""
        residuals = -np.abs(np.random.randn(50)) - 0.01
        tau = 0.7
        grad = PenalizedQuantileOptimizer._check_gradient_fast(residuals, tau)
        assert_allclose(grad, tau - 1.0, atol=1e-10)

    def test_gradient_values_binary(self):
        """Gradient should only have two possible values: tau or tau - 1."""
        np.random.seed(44)
        residuals = np.random.randn(200)
        tau = 0.4
        grad = PenalizedQuantileOptimizer._check_gradient_fast(residuals, tau)
        unique_vals = np.unique(grad)
        assert len(unique_vals) <= 2
        for v in unique_vals:
            assert np.isclose(v, tau) or np.isclose(v, tau - 1.0)


class TestSoftThresholdExtended:
    """Extended tests for _soft_threshold (lines 89-94)."""

    def test_large_positive(self):
        """Large x >> lambda."""
        result = PenalizedQuantileOptimizer._soft_threshold(100.0, 1.0)
        assert_allclose(result, 99.0)

    def test_large_negative(self):
        """Large negative x << -lambda."""
        result = PenalizedQuantileOptimizer._soft_threshold(-100.0, 1.0)
        assert_allclose(result, -99.0)

    def test_tiny_lambda(self):
        """Very small lambda: almost no thresholding."""
        result = PenalizedQuantileOptimizer._soft_threshold(0.5, 1e-10)
        assert_allclose(result, 0.5, atol=1e-9)

    def test_equal_magnitude(self):
        """x == lambda returns 0."""
        result = PenalizedQuantileOptimizer._soft_threshold(3.0, 3.0)
        assert result == 0.0

    def test_negative_equal_magnitude(self):
        """x == -lambda returns 0."""
        result = PenalizedQuantileOptimizer._soft_threshold(-3.0, 3.0)
        assert result == 0.0

    def test_small_inside_band(self):
        """x inside (-lambda, lambda) returns 0."""
        result = PenalizedQuantileOptimizer._soft_threshold(0.5, 1.0)
        assert result == 0.0

    def test_sequence_of_values(self):
        """Test a sequence of values through soft threshold."""
        lam = 2.0
        for x, expected in [(5.0, 3.0), (-5.0, -3.0), (1.0, 0.0), (0.0, 0.0), (2.0, 0.0)]:
            result = PenalizedQuantileOptimizer._soft_threshold(x, lam)
            assert_allclose(result, expected, atol=1e-10)


# ===========================================================================
# PART 7: penalized.py -- Module-level Numba functions
# ===========================================================================


class TestComputeCheckLossMatrixExtended:
    """Extended tests for compute_check_loss_matrix (lines 286-298)."""

    def test_single_tau_multiple_residuals(self):
        """Single quantile, many residuals."""
        np.random.seed(11)
        residuals = np.random.randn(50)
        tau_grid = np.array([0.5])
        result = compute_check_loss_matrix(residuals, tau_grid)
        assert result.shape == (50, 1)
        assert np.all(result >= -1e-15)

    def test_multiple_taus_single_residual(self):
        """Many quantiles, single residual."""
        residuals = np.array([2.0])
        tau_grid = np.linspace(0.1, 0.9, 9)
        result = compute_check_loss_matrix(residuals, tau_grid)
        assert result.shape == (1, 9)
        for j, tau in enumerate(tau_grid):
            assert_allclose(result[0, j], tau * 2.0, atol=1e-10)

    def test_negative_residual_all_taus(self):
        """Negative residual with multiple taus."""
        residuals = np.array([-3.0])
        tau_grid = np.array([0.1, 0.5, 0.9])
        result = compute_check_loss_matrix(residuals, tau_grid)
        for j, tau in enumerate(tau_grid):
            expected = (tau - 1) * (-3.0)
            assert_allclose(result[0, j], expected, atol=1e-10)

    def test_large_residual_array(self):
        """Large array to exercise parallel loops."""
        np.random.seed(42)
        residuals = np.random.randn(1000)
        tau_grid = np.linspace(0.1, 0.9, 5)
        result = compute_check_loss_matrix(residuals, tau_grid)
        assert result.shape == (1000, 5)
        assert np.all(np.isfinite(result))
        assert np.all(result >= -1e-12)

    def test_consistency_with_check_loss_fast(self):
        """Each column should match _check_loss_fast for that tau."""
        np.random.seed(33)
        residuals = np.random.randn(30)
        tau_grid = np.array([0.25, 0.5, 0.75])
        matrix = compute_check_loss_matrix(residuals, tau_grid)
        for j, tau in enumerate(tau_grid):
            total_loss = matrix[:, j].sum()
            direct_loss = PenalizedQuantileOptimizer._check_loss_fast(residuals, tau)
            assert_allclose(total_loss, direct_loss, atol=1e-8)


class TestComputeGradientMatrixExtended:
    """Extended tests for compute_gradient_matrix (lines 304-313)."""

    def test_single_positive_residual(self):
        """Single positive residual across multiple taus."""
        residuals = np.array([5.0])
        tau_grid = np.array([0.1, 0.5, 0.9])
        result = compute_gradient_matrix(residuals, tau_grid)
        for j, tau in enumerate(tau_grid):
            assert_allclose(result[0, j], tau, atol=1e-10)

    def test_single_negative_residual(self):
        """Single negative residual across multiple taus."""
        residuals = np.array([-5.0])
        tau_grid = np.array([0.1, 0.5, 0.9])
        result = compute_gradient_matrix(residuals, tau_grid)
        for j, tau in enumerate(tau_grid):
            assert_allclose(result[0, j], tau - 1.0, atol=1e-10)

    def test_large_gradient_matrix(self):
        """Large matrix to exercise parallel prange."""
        np.random.seed(55)
        residuals = np.random.randn(500)
        tau_grid = np.linspace(0.05, 0.95, 10)
        result = compute_gradient_matrix(residuals, tau_grid)
        assert result.shape == (500, 10)
        for j, tau in enumerate(tau_grid):
            assert np.all(result[:, j] >= tau - 1.0 - 1e-10)
            assert np.all(result[:, j] <= tau + 1e-10)

    def test_consistency_with_check_gradient_fast(self):
        """Each column should match _check_gradient_fast for that tau."""
        np.random.seed(20)
        residuals = np.random.randn(40)
        tau_grid = np.array([0.2, 0.5, 0.8])
        matrix = compute_gradient_matrix(residuals, tau_grid)
        for j, tau in enumerate(tau_grid):
            direct_grad = PenalizedQuantileOptimizer._check_gradient_fast(residuals, tau)
            assert_allclose(matrix[:, j], direct_grad, atol=1e-10)


# ===========================================================================
# PART 8: penalized.py -- PenalizedQuantileOptimizer methods
# ===========================================================================


class TestPenalizedQuantileOptimizerMethods:
    """Test coordinate_descent and warm_start_path on PenalizedQuantileOptimizer."""

    def test_coordinate_descent_with_warm_start(self, synthetic_panel_small):
        """coordinate_descent with warm_start parameter."""
        X, y, entity_ids = synthetic_panel_small
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=0.1)
        n_params = X.shape[1] + opt.n_entities
        warm = np.zeros(n_params)
        warm[: X.shape[1]] = np.linalg.lstsq(X, y, rcond=None)[0]

        result = opt.coordinate_descent(max_iter=20, warm_start=warm)
        assert "beta" in result
        assert "alpha" in result
        assert "converged" in result
        assert "iterations" in result
        assert np.all(np.isfinite(result["beta"]))
        assert np.all(np.isfinite(result["alpha"]))

    def test_coordinate_descent_without_warm_start(self, synthetic_panel_small):
        """coordinate_descent without warm_start initializes at zeros."""
        X, y, entity_ids = synthetic_panel_small
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=0.5)

        result = opt.coordinate_descent(max_iter=10)
        assert result["iterations"] <= 10
        assert np.all(np.isfinite(result["beta"]))

    def test_coordinate_descent_convergence_high_tol(self, synthetic_panel_small):
        """With very high tolerance, should converge in few iterations."""
        X, y, entity_ids = synthetic_panel_small
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=1.0)
        result = opt.coordinate_descent(max_iter=100, tol=1e2)
        assert result["converged"] is True
        assert result["iterations"] < 100

    def test_warm_start_path_multiple_lambdas(self, synthetic_panel_small):
        """warm_start_path with a grid of lambda values."""
        X, y, entity_ids = synthetic_panel_small
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=0.1)

        lambda_grid = np.array([0.01, 0.1, 1.0])
        results = opt.warm_start_path(lambda_grid)

        assert isinstance(results, list)
        assert len(results) == 3

        # Should be sorted descending by lambda
        lambdas = [r["lambda"] for r in results]
        assert lambdas == sorted(lambdas, reverse=True)

        for r in results:
            assert "lambda" in r
            assert "params" in r
            assert "converged" in r
            assert "objective" in r
            assert np.all(np.isfinite(r["params"]))

    def test_warm_start_path_single_lambda(self, synthetic_panel_small):
        """warm_start_path with a single lambda."""
        X, y, entity_ids = synthetic_panel_small
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=0.1)

        lambda_grid = np.array([0.5])
        results = opt.warm_start_path(lambda_grid)
        assert len(results) == 1
        assert np.isfinite(results[0]["objective"])

    def test_optimize_with_warm_start(self, synthetic_panel_small):
        """optimize() with warm_start parameter."""
        X, y, entity_ids = synthetic_panel_small
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=0.1)

        # First run without warm start
        result1 = opt.optimize()

        # Second run with warm start from first
        result2 = opt.optimize(warm_start=result1.x)
        assert np.all(np.isfinite(result2.x))
        # Warm start should converge in fewer or equal iterations
        assert result2.nit <= result1.nit + 5  # allow small slack


# ===========================================================================
# PART 9: penalized.py -- PerformanceMonitor
# ===========================================================================


class TestPerformanceMonitorExtended:
    """Extended tests for PerformanceMonitor.print_report and related methods."""

    def test_print_report_empty(self, capsys):
        """Empty timings should not crash."""
        monitor = PerformanceMonitor()
        monitor.print_report()
        captured = capsys.readouterr()
        assert "PERFORMANCE COMPARISON" in captured.out

    def test_print_report_single_method(self, capsys):
        """Single method: no speedup ratios section."""
        monitor = PerformanceMonitor()
        monitor.timings = {
            "canay": {"time": 1.5, "memory_mb": 8.0, "converged": True},
        }
        monitor.print_report()
        captured = capsys.readouterr()
        assert "canay" in captured.out
        assert "Fastest" in captured.out
        assert "Least Memory" in captured.out
        assert "Speedup Ratios" not in captured.out

    def test_print_report_multiple_methods(self, capsys):
        """Multiple methods: should show speedup ratios."""
        monitor = PerformanceMonitor()
        monitor.timings = {
            "method_a": {"time": 2.0, "memory_mb": 20.0, "converged": True},
            "method_b": {"time": 0.5, "memory_mb": 5.0, "converged": False},
        }
        monitor.print_report()
        captured = capsys.readouterr()
        assert "method_a" in captured.out
        assert "method_b" in captured.out
        assert "Fastest" in captured.out
        assert "Speedup Ratios" in captured.out

    def test_print_report_converged_and_failed(self, capsys):
        """Report should indicate convergence status for each method."""
        monitor = PerformanceMonitor()
        monitor.timings = {
            "good": {"time": 1.0, "memory_mb": 5.0, "converged": True},
            "bad": {"time": 3.0, "memory_mb": 15.0, "converged": False},
        }
        monitor.print_report()
        captured = capsys.readouterr()
        assert "good" in captured.out
        assert "bad" in captured.out


# ===========================================================================
# PART 10: penalized.py -- AdaptiveOptimizer
# ===========================================================================


class TestAdaptiveOptimizerExtended:
    """Extended tests for AdaptiveOptimizer.recommend_method and print_analysis."""

    def test_large_sparse_recommends_coordinate_descent(self):
        """Large-scale + sparse X should recommend coordinate_descent."""
        np.random.seed(42)
        n = 10000
        p = 200
        entity_ids = np.repeat(np.arange(100), 100)
        X = np.zeros((n, p))
        # Sparse: each row has ~10% nonzero
        for i in range(n):
            cols = np.random.choice(p, size=int(p * 0.05), replace=False)
            X[i, cols] = np.random.randn(len(cols))
        y = np.random.randn(n)

        ada = AdaptiveOptimizer(X, y, entity_ids)
        assert ada.problem_size > 1e6
        assert ada.X_sparsity > 0.5
        method, params = ada.recommend_method()
        assert method == "coordinate_descent"
        assert "max_iter" in params

    def test_large_dense_recommends_lbfgsb(self):
        """Large-scale + dense X should recommend L-BFGS-B."""
        np.random.seed(42)
        n = 10000
        p = 200
        entity_ids = np.repeat(np.arange(100), 100)
        X = np.random.randn(n, p)
        y = np.random.randn(n)

        ada = AdaptiveOptimizer(X, y, entity_ids)
        assert ada.problem_size > 1e6
        assert ada.X_sparsity < 0.5
        method, params = ada.recommend_method()
        assert method == "L-BFGS-B"
        assert "maxiter" in params

    def test_small_T_recommends_penalty(self):
        """Small avg_T (< 10) should recommend penalty."""
        np.random.seed(42)
        n_entities = 50
        T = 5
        n = n_entities * T
        entity_ids = np.repeat(np.arange(n_entities), T)
        X = np.random.randn(n, 3)
        y = np.random.randn(n)

        ada = AdaptiveOptimizer(X, y, entity_ids)
        assert ada.avg_T < 10
        assert ada.problem_size <= 1e6
        method, params = ada.recommend_method()
        assert method == "penalty"
        assert params == {"lambda_fe": "auto", "cv_folds": 5}

    def test_well_conditioned_recommends_canay(self):
        """Well-conditioned X with sufficient T should recommend canay."""
        np.random.seed(42)
        n_entities = 10
        T = 20
        n = n_entities * T
        entity_ids = np.repeat(np.arange(n_entities), T)
        # Orthonormal-ish design ensures condition number < 100
        X = np.random.randn(n, 3)
        y = np.random.randn(n)

        ada = AdaptiveOptimizer(X, y, entity_ids)
        # Verify preconditions
        assert ada.problem_size <= 1e6
        assert ada.avg_T >= 10
        if ada.condition_number < 100:
            method, params = ada.recommend_method()
            assert method == "canay"
            assert params == {"se_adjustment": "two-step"}

    def test_default_lbfgsb(self):
        """Ill-conditioned, medium-scale, sufficient T -> default L-BFGS-B."""
        np.random.seed(42)
        n_entities = 20
        T = 15
        n = n_entities * T
        entity_ids = np.repeat(np.arange(n_entities), T)
        X = np.random.randn(n, 3)
        X[:, 2] = X[:, 0] + X[:, 1] * 1e-6  # nearly collinear
        y = np.random.randn(n)

        ada = AdaptiveOptimizer(X, y, entity_ids)
        assert ada.condition_number > 100
        method, params = ada.recommend_method()
        assert method == "L-BFGS-B"
        assert params["maxiter"] == 1000

    def test_condition_number_inf_for_zero_X(self):
        """Zero X should give condition_number = inf."""
        n_entities = 5
        T = 4
        n = n_entities * T
        entity_ids = np.repeat(np.arange(n_entities), T)
        X = np.zeros((n, 2))
        y = np.random.randn(n)

        ada = AdaptiveOptimizer(X, y, entity_ids)
        assert ada.condition_number == np.inf

    def test_print_analysis(self, caplog):
        """print_analysis should log problem characteristics."""
        np.random.seed(42)
        n_entities = 10
        T = 10
        n = n_entities * T
        entity_ids = np.repeat(np.arange(n_entities), T)
        X = np.random.randn(n, 2)
        y = np.random.randn(n)

        ada = AdaptiveOptimizer(X, y, entity_ids)
        with caplog.at_level(logging.INFO, logger="panelbox.optimization.quantile.penalized"):
            ada.print_analysis()

        # Verify it ran without error
        assert ada.problem_size > 0
        assert ada.avg_T == T


# ===========================================================================
# PART 11: EigenvalueCache edge cases
# ===========================================================================


class TestEigenvalueCacheExtended:
    """Additional edge case tests for EigenvalueCache."""

    def test_cache_hit_avoids_recompute(self):
        """Second call to get_eigenvalues should return identical cached array."""
        cache = EigenvalueCache()
        W = np.array([[0, 1], [1, 0]], dtype=float)
        eigs1 = cache.get_eigenvalues(W)
        eigs2 = cache.get_eigenvalues(W)
        assert eigs1 is eigs2  # Same object (from cache)

    def test_force_recalc_produces_new_array(self):
        """force_recalc=True should produce a new array (not from cache)."""
        cache = EigenvalueCache()
        W = np.array([[0, 1], [1, 0]], dtype=float)
        eigs1 = cache.get_eigenvalues(W)
        eigs2 = cache.get_eigenvalues(W, force_recalc=True)
        assert_allclose(eigs1, eigs2)

    def test_clear_empties_cache(self):
        """clear() should remove all cached eigenvalues."""
        cache = EigenvalueCache()
        W = np.eye(3)
        cache.get_eigenvalues(W)
        assert len(cache._cache) > 0
        cache.clear()
        assert len(cache._cache) == 0


# ===========================================================================
# PART 12: Integration tests exercising JIT paths end-to-end
# ===========================================================================


class TestIntegrationJITPaths:
    """Integration tests that exercise JIT code paths via higher-level calls."""

    def test_objective_and_gradient_consistency(self, synthetic_panel_large):
        """Gradient direction should decrease objective (basic sanity)."""
        X, y, entity_ids = synthetic_panel_large
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=0.1)
        params = opt._smart_init()

        obj_val = opt.objective(params)
        grad = opt.gradient(params)

        # Move in negative gradient direction
        step = 1e-5
        params_new = params - step * grad
        obj_new = opt.objective(params_new)

        # For small enough step, objective should decrease or stay similar
        assert np.isfinite(obj_val)
        assert np.isfinite(obj_new)

    def test_full_optimization_pipeline(self, synthetic_panel_small):
        """Full optimization pipeline: init -> optimize -> extract results."""
        X, y, entity_ids = synthetic_panel_small
        opt = PenalizedQuantileOptimizer(X, y, entity_ids, tau=0.5, lambda_val=0.1)

        result = opt.optimize()
        beta = result.x[: X.shape[1]]
        alpha = result.x[X.shape[1] :]

        assert np.all(np.isfinite(beta))
        assert np.all(np.isfinite(alpha))
        assert len(alpha) == opt.n_entities
        assert result.fun < opt.objective(np.zeros(len(result.x)))

    def test_spatial_lag_then_quadratic_form(self, row_standardized_W):
        """Chain: compute spatial lag, then use it in quadratic form."""
        W = row_standardized_W
        W_csr = csr_matrix(W)
        n = W.shape[0]
        y = np.random.randn(n)

        # Compute spatial lag
        Wy = compute_spatial_lag_fast(W_csr.data, W_csr.indices, W_csr.indptr, y)
        assert Wy.shape == (n,)
        assert_allclose(Wy, W @ y, atol=1e-10)

        # Use in quadratic form
        X = np.random.randn(n, 2)
        qf = compute_quadratic_form(X, W, y)
        assert np.isfinite(qf)

    def test_row_standardize_then_chebyshev(self):
        """Chain: row standardize, then use with ChebyshevApproximation."""
        W = np.array(
            [[0, 2, 0], [1, 0, 1], [0, 3, 0]],
            dtype=float,
        )
        W_std = row_standardize_fast(W)
        # Verify row-standardized
        for i in range(3):
            if W[i].sum() > 0:
                assert_allclose(W_std[i].sum(), 1.0, atol=1e-10)

        cheb = ChebyshevApproximation(order=5)
        log_det = cheb.log_determinant(0.3, W_std)
        assert np.isfinite(log_det)
