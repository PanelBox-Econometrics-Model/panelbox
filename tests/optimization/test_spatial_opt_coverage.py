"""Coverage-focused tests for panelbox.optimization.spatial_optimizations.

Targets uncovered lines from the coverage report:
- Lines 20-22: HAS_NUMBA=False import fallback path
- Lines 224-226: SparseOperations.log_determinant cholesky LinAlgError fallback
- Lines 245-257: Numba JIT compute_spatial_lag_fast body
- Lines 266-294: Numba JIT compute_quadratic_form body
- Lines 311-341: Numba JIT row_standardize_fast body + fallback implementations
- Line 463: optimize_spatial_model W.shape[0] > 5000 branch (Chebyshev init)
"""

from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from panelbox.optimization.spatial_optimizations import (
    ChebyshevApproximation,
    SparseOperations,
    compute_quadratic_form,
    compute_spatial_lag_fast,
    optimize_spatial_model,
    row_standardize_fast,
)


@pytest.fixture
def small_W():
    """5x5 row-standardized spatial weight matrix."""
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
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return W / row_sums


@pytest.fixture
def sparse_W(small_W):
    """CSR sparse version of the 5x5 weight matrix."""
    return csr_matrix(small_W)


# ---------------------------------------------------------------------------
# Tests for the non-numba fallback implementations (lines 327-341)
# ---------------------------------------------------------------------------


class TestFallbackImplementations:
    """Test the non-numba fallback code paths by mocking HAS_NUMBA=False.

    When numba IS installed, the module defines JIT versions of
    compute_spatial_lag_fast, compute_quadratic_form, and row_standardize_fast.
    The fallback (else) branch at lines 325-341 is never executed.

    We re-import the module with HAS_NUMBA patched to False so that the
    fallback definitions become the active ones, covering lines 327-341.
    """

    def _reimport_with_no_numba(self):
        """Re-import the module with numba disabled to activate fallbacks."""
        import importlib
        import sys

        mod_name = "panelbox.optimization.spatial_optimizations"
        # Temporarily make 'numba' unimportable
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        original_module = sys.modules.get(mod_name)

        def mock_import(name, *args, **kwargs):
            if name == "numba":
                raise ImportError("mocked: numba not available")
            return real_import(name, *args, **kwargs)

        # Remove cached module so importlib reloads it
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        import builtins

        original_import = builtins.__import__
        builtins.__import__ = mock_import
        try:
            import importlib

            mod = importlib.import_module(mod_name)
        finally:
            builtins.__import__ = original_import
            # Restore the original module so other tests are not affected
            if original_module is not None:
                sys.modules[mod_name] = original_module

        return mod

    def test_fallback_compute_spatial_lag_fast(self, sparse_W):
        """Fallback compute_spatial_lag_fast reconstructs CSR and uses .dot (line 327-330)."""
        mod = self._reimport_with_no_numba()
        assert not mod.HAS_NUMBA

        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mod.compute_spatial_lag_fast(sparse_W.data, sparse_W.indices, sparse_W.indptr, y)
        expected = sparse_W.toarray() @ y
        assert_allclose(result, expected, atol=1e-10)

    def test_fallback_compute_quadratic_form(self):
        """Fallback compute_quadratic_form uses matrix multiply chain (lines 332-335).

        Note: the fallback computes y.T @ X.T @ WX @ X @ y where WX = W @ X.
        This requires n == k for the matrix dimensions to be compatible.
        We use square X (n=k=3) to exercise the code path.
        """
        mod = self._reimport_with_no_numba()
        assert not mod.HAS_NUMBA

        np.random.seed(42)
        n = 3
        X = np.random.randn(n, n)
        W = np.array(
            [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
            dtype=float,
        )
        y = np.random.randn(n)

        result = mod.compute_quadratic_form(X, W, y)
        # Reproduce the fallback formula: y.T @ X.T @ (W @ X) @ X @ y
        WX = W @ X
        expected = y.T @ X.T @ WX @ X @ y
        assert np.isfinite(result)
        assert_allclose(result, expected, atol=1e-10)

    def test_fallback_row_standardize_fast(self):
        """Fallback row_standardize_fast uses numpy division (lines 337-341)."""
        mod = self._reimport_with_no_numba()
        assert not mod.HAS_NUMBA

        W = np.array(
            [[0, 1, 2], [3, 0, 1], [0, 0, 0]],
            dtype=float,
        )
        W_std = mod.row_standardize_fast(W)

        # Non-zero rows should sum to 1
        assert_allclose(W_std[0].sum(), 1.0, atol=1e-10)
        assert_allclose(W_std[1].sum(), 1.0, atol=1e-10)
        # Zero row stays zero
        assert_allclose(W_std[2].sum(), 0.0, atol=1e-10)

    def test_fallback_import_warning(self):
        """Importing without numba should emit an ImportWarning (lines 20-22)."""
        import builtins
        import importlib
        import sys

        mod_name = "panelbox.optimization.spatial_optimizations"
        original_module = sys.modules.get(mod_name)

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "numba":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        if mod_name in sys.modules:
            del sys.modules[mod_name]

        builtins.__import__ = mock_import
        try:
            with pytest.warns(ImportWarning, match="Numba not installed"):
                importlib.import_module(mod_name)
        finally:
            builtins.__import__ = real_import
            if original_module is not None:
                sys.modules[mod_name] = original_module


# ---------------------------------------------------------------------------
# Tests for Numba JIT function bodies (lines 245-257, 266-294, 311-323)
# ---------------------------------------------------------------------------


class TestJITFunctionBodies:
    """Exercise the JIT-compiled function bodies with various inputs.

    Coverage tools may not trace into numba @jit code, but calling these
    functions with different data shapes and values ensures correctness
    and may help with some coverage configurations.
    """

    def test_spatial_lag_fast_single_element(self):
        """Spatial lag on a 1x1 sparse matrix (lines 245-257 edge case)."""
        W = csr_matrix(np.array([[0.0]]))
        y = np.array([5.0])
        result = compute_spatial_lag_fast(W.data, W.indices, W.indptr, y)
        assert_allclose(result, [0.0], atol=1e-10)

    def test_spatial_lag_fast_identity_like(self):
        """Spatial lag with off-diagonal identity-like sparse matrix."""
        W_dense = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        W = csr_matrix(W_dense)
        y = np.array([10.0, 20.0, 30.0])
        result = compute_spatial_lag_fast(W.data, W.indices, W.indptr, y)
        expected = W_dense @ y
        assert_allclose(result, expected, atol=1e-10)

    def test_spatial_lag_fast_all_zeros(self):
        """Spatial lag with a zero sparse matrix."""
        W = csr_matrix(np.zeros((4, 4)))
        y = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_spatial_lag_fast(W.data, W.indices, W.indptr, y)
        assert_allclose(result, np.zeros(4), atol=1e-10)

    def test_spatial_lag_fast_larger_matrix(self):
        """Spatial lag on a 10x10 sparse matrix."""
        np.random.seed(42)
        W_dense = np.random.rand(10, 10)
        W_dense[W_dense < 0.7] = 0.0
        row_sums = W_dense.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W_dense = W_dense / row_sums
        np.fill_diagonal(W_dense, 0)
        W = csr_matrix(W_dense)
        y = np.random.randn(10)

        result = compute_spatial_lag_fast(W.data, W.indices, W.indptr, y)
        expected = W_dense @ y
        assert_allclose(result, expected, atol=1e-10)

    def test_quadratic_form_known_value(self):
        """Quadratic form y'X'WXy with known matrices (lines 266-294)."""
        np.random.seed(42)
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        W = np.array(
            [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
            dtype=float,
        )
        y = np.array([1.0, 2.0, 3.0])

        result = compute_quadratic_form(X, W, y)

        # Verify manually: WX, X'WX, temp = X'y, result = temp' X'WX temp
        WX = W @ X
        XWX = X.T @ WX
        temp = X.T @ y
        expected = temp @ XWX @ temp
        assert_allclose(result, expected, atol=1e-10)

    def test_quadratic_form_single_column(self):
        """Quadratic form with k=1 (single regressor)."""
        X = np.array([[1.0], [2.0], [3.0]])
        W = np.eye(3)
        y = np.array([1.0, 1.0, 1.0])

        result = compute_quadratic_form(X, W, y)
        # WX = IX = X, X'WX = X'X = [[14]], temp = X'y = [[6]]
        # result = 6 * 14 * 6 = 504
        expected = 504.0
        assert_allclose(result, expected, atol=1e-10)

    def test_quadratic_form_larger(self):
        """Quadratic form on 10x3 problem."""
        np.random.seed(42)
        n, k = 10, 3
        X = np.random.randn(n, k)
        W_dense = np.random.rand(n, n)
        row_sums = W_dense.sum(axis=1, keepdims=True)
        W_dense = W_dense / row_sums
        y = np.random.randn(n)

        result = compute_quadratic_form(X, W_dense, y)

        WX = W_dense @ X
        XWX = X.T @ WX
        temp = X.T @ y
        expected = temp @ XWX @ temp
        assert_allclose(result, expected, atol=1e-8)

    def test_row_standardize_fast_all_equal(self):
        """Row standardize with equal weights (lines 311-323)."""
        W = np.ones((4, 4), dtype=float)
        np.fill_diagonal(W, 0)
        W_std = row_standardize_fast(W)

        # Each row has 3 nonzero entries, so each should be 1/3
        for i in range(4):
            assert_allclose(W_std[i, i], 0.0, atol=1e-10)
            nonzero_vals = W_std[i, W_std[i] != 0]
            assert_allclose(nonzero_vals, 1.0 / 3.0, atol=1e-10)

    def test_row_standardize_fast_already_standardized(self, small_W):
        """Row standardizing an already-standardized matrix should be idempotent."""
        W_std = row_standardize_fast(small_W)
        # Row sums should all be 1 (except zero rows)
        for i in range(small_W.shape[0]):
            row_sum = W_std[i].sum()
            if small_W[i].sum() > 0:
                assert_allclose(row_sum, 1.0, atol=1e-10)

    def test_row_standardize_fast_mixed_rows(self):
        """Row standardize with a mix of zero and non-zero rows."""
        W = np.array(
            [
                [0, 5, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [2, 0, 0, 3, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0],
            ],
            dtype=float,
        )
        W_std = row_standardize_fast(W)

        assert_allclose(W_std[0], [0, 1.0, 0, 0, 0], atol=1e-10)
        assert_allclose(W_std[1], [0, 0, 0, 0, 0], atol=1e-10)
        assert_allclose(W_std[2], [0.4, 0, 0, 0.6, 0], atol=1e-10)
        assert_allclose(W_std[3], [0, 0, 0, 0, 0], atol=1e-10)
        assert_allclose(W_std[4], [0.25, 0.25, 0.25, 0.25, 0], atol=1e-10)


# ---------------------------------------------------------------------------
# Test for Cholesky LinAlgError fallback (lines 224-226)
# ---------------------------------------------------------------------------


class TestCholeskyFallback:
    """Test that cholesky method falls back to eigen on LinAlgError."""

    def test_cholesky_fallback_on_non_pd_matrix(self):
        """Force cholesky to fail by providing a matrix with negative eigenvalues.

        The log_determinant cholesky path computes L = cholesky(M @ M.T).
        We mock np.linalg.cholesky to raise LinAlgError to guarantee we
        exercise the except branch at lines 224-226.
        """
        W = np.array(
            [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
            dtype=float,
        )
        rho = 0.3
        I_rho_W = np.eye(3) - rho * W

        # Compute expected via eigen method
        expected = SparseOperations.log_determinant(I_rho_W, method="eigen")

        # Mock cholesky to raise LinAlgError so fallback is triggered
        with patch(
            "numpy.linalg.cholesky",
            side_effect=np.linalg.LinAlgError("not positive definite"),
        ):
            result = SparseOperations.log_determinant(I_rho_W, method="cholesky")

        assert np.isfinite(result)
        assert_allclose(result, expected, atol=1e-10)

    def test_cholesky_fallback_singular_matrix(self):
        """Singular I_rho_W triggers cholesky failure and fallback."""
        # Create a singular-ish matrix (rows are linearly dependent)
        I_rho_W = np.array(
            [[1.0, -0.5, -0.5], [-0.5, 1.0, -0.5], [-0.5, -0.5, 1.0]],
            dtype=float,
        )

        with patch(
            "numpy.linalg.cholesky",
            side_effect=np.linalg.LinAlgError("singular"),
        ):
            result = SparseOperations.log_determinant(I_rho_W, method="cholesky")

        expected = SparseOperations.log_determinant(I_rho_W, method="eigen")
        assert np.isfinite(result)
        assert_allclose(result, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Test for optimize_spatial_model W.shape[0] > 5000 branch (line 463)
# ---------------------------------------------------------------------------


class TestOptimizeSpatialModelLargeW:
    """Test the Chebyshev initialization branch for large W matrices."""

    @staticmethod
    def _make_large_sparse_W(n=5001):
        """Build a sparse n x n chain-topology weight matrix (CSR).

        Returns a CSR matrix directly so we can pass it to the decorator
        without creating an enormous dense array.
        """
        from scipy.sparse import diags

        # Chain: each node i connected to i-1 and i+1
        off_diag = np.ones(n - 1)
        W = diags([off_diag, off_diag], offsets=[-1, 1], shape=(n, n), format="csr")
        # Row-standardize: interior rows sum to 2, boundary rows sum to 1
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        W = W.multiply(1.0 / row_sums[:, np.newaxis])
        return W.tocsr()

    def test_large_w_triggers_chebyshev(self):
        """When W.shape[0] > 5000, decorator should initialize ChebyshevApproximation."""
        W_sparse = self._make_large_sparse_W(5001)

        @optimize_spatial_model
        class LargeSpatialModel:
            def __init__(self, W):
                self.W = W

        model = LargeSpatialModel(W=W_sparse)

        # Sparse matrix input stays sparse
        assert model._use_sparse
        # Since W.shape[0] > 5000, _chebyshev should be initialized
        assert model._chebyshev is not None
        assert isinstance(model._chebyshev, ChebyshevApproximation)

    def test_small_w_no_chebyshev(self, small_W):
        """When W.shape[0] <= 5000, no ChebyshevApproximation is created."""

        @optimize_spatial_model
        class SmallSpatialModel:
            def __init__(self, W):
                self.W = W

        model = SmallSpatialModel(W=small_W)
        assert model._chebyshev is None

    def test_large_w_boundary_not_triggered(self):
        """Exactly 5000 should NOT trigger Chebyshev (> 5000 required)."""
        W_sparse = self._make_large_sparse_W(5000)

        @optimize_spatial_model
        class BoundarySpatialModel:
            def __init__(self, W):
                self.W = W

        model = BoundarySpatialModel(W=W_sparse)
        assert model._chebyshev is None
