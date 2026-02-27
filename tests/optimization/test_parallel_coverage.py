"""
Tests for parallel inference and spatial optimizations - Coverage improvement.

Targets:
- panelbox/optimization/parallel_inference.py
- panelbox/optimization/spatial_optimizations.py

Covers uncovered branches:
- ParallelPermutationTest._worker_permutations with panel entity/time permutation
- ParallelBootstrap._worker_bootstrap with wild, block, residual types
- ParallelBootstrap estimation failure handling
- ParallelSpatialHAC._compute_chunk with different kernels
- _evaluate_params with different scoring
- EigenvalueCache, SparseOperations, ChebyshevApproximation edge cases
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from panelbox.optimization.parallel_inference import (
    ParallelBootstrap,
    ParallelPermutationTest,
    ParallelSpatialHAC,
    _evaluate_params,
)
from panelbox.optimization.spatial_optimizations import (
    ChebyshevApproximation,
    EigenvalueCache,
    SparseOperations,
    compute_quadratic_form,
    compute_spatial_lag_fast,
    get_eigenvalues_cached,
    row_standardize_fast,
)


def _simple_estimator(data, W, formula, entity_col, time_col, **kwargs):
    """Simple OLS estimator for testing (module-level for pickling)."""
    x = data["x"].values
    y = data["y"].values
    denom = np.sum(x**2)
    beta = np.sum(x * y) / denom if denom > 0 else 0.0
    return {"beta": beta}


def _failing_estimator(data, W, formula, entity_col, time_col, **kwargs):
    """Estimator that always fails (module-level for pickling)."""
    raise RuntimeError("Estimation failed")


def _moran_stat(data, W, **kwargs):
    """Simple Moran's I for testing."""
    n = len(data)
    z = data - data.mean()
    denom = W.sum() * (z @ z)
    if denom == 0:
        return 0.0
    return n * z @ W @ z / denom


# -----------------------------------------------------------------------
# Parallel bootstrap tests
# -----------------------------------------------------------------------


class TestParallelBootstrapCoverage:
    """Additional tests for ParallelBootstrap covering uncovered branches."""

    @pytest.fixture
    def panel_data(self):
        """Panel data for bootstrap."""
        np.random.seed(42)
        n_entities, n_time = 10, 5
        n = n_entities * n_time
        entities = np.repeat(np.arange(n_entities), n_time)
        times = np.tile(np.arange(n_time), n_entities)
        x = np.random.randn(n)
        y = 0.5 * x + np.random.randn(n) * 0.5
        return pd.DataFrame({"entity": entities, "time": times, "x": x, "y": y})

    @pytest.fixture
    def small_W(self):
        """Small weight matrix."""
        return np.eye(10)

    def test_worker_bootstrap_wild(self, panel_data):
        """Test _worker_bootstrap with wild bootstrap type directly."""
        work_package = (
            5,
            panel_data,
            np.eye(10),
            "y ~ x",
            "entity",
            "time",
            {},
            "wild",
            42,
        )
        result = ParallelBootstrap._worker_bootstrap(_simple_estimator, work_package)
        assert result.shape[0] == 5
        assert np.all(np.isfinite(result))

    def test_worker_bootstrap_block(self, panel_data):
        """Test _worker_bootstrap with block bootstrap type directly."""
        work_package = (
            5,
            panel_data,
            np.eye(10),
            "y ~ x",
            "entity",
            "time",
            {},
            "block",
            42,
        )
        result = ParallelBootstrap._worker_bootstrap(_simple_estimator, work_package)
        assert result.shape[0] == 5
        assert np.all(np.isfinite(result))

    def test_worker_bootstrap_residual(self, panel_data):
        """Test _worker_bootstrap with residual bootstrap type directly."""
        work_package = (
            5,
            panel_data,
            np.eye(10),
            "y ~ x",
            "entity",
            "time",
            {},
            "residual",
            42,
        )
        result = ParallelBootstrap._worker_bootstrap(_simple_estimator, work_package)
        assert result.shape[0] == 5
        assert np.all(np.isfinite(result))

    def test_worker_bootstrap_no_seed(self, panel_data):
        """Test _worker_bootstrap without seed (None)."""
        work_package = (
            3,
            panel_data,
            np.eye(10),
            "y ~ x",
            "entity",
            "time",
            {},
            "pairs",
            None,
        )
        result = ParallelBootstrap._worker_bootstrap(_simple_estimator, work_package)
        assert result.shape[0] == 3


# -----------------------------------------------------------------------
# Parallel permutation tests
# -----------------------------------------------------------------------


class TestParallelPermutationCoverage:
    """Additional permutation test coverage."""

    def test_worker_permutations_panel(self):
        """Test _worker_permutations with panel data (entity_ids + time_ids)."""
        np.random.seed(42)
        n = 20
        data = np.random.randn(n)
        W = np.zeros((n, n))
        for i in range(n - 1):
            W[i, i + 1] = 1.0
            W[i + 1, i] = 1.0
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

        entity_ids = np.repeat(np.arange(4), 5)
        time_ids = np.tile(np.arange(5), 4)

        work_package = (
            0,
            10,
            data,
            W,
            {"entity_ids": entity_ids, "time_ids": time_ids},
            42,
        )
        result = ParallelPermutationTest._worker_permutations(_moran_stat, work_package)
        assert result.shape == (10,)
        assert np.all(np.isfinite(result))

    def test_worker_permutations_cross_sectional(self):
        """Test _worker_permutations without entity_ids (cross-sectional)."""
        np.random.seed(42)
        n = 15
        data = np.random.randn(n)
        W = np.eye(n)

        work_package = (0, 8, data, W, {}, 42)
        result = ParallelPermutationTest._worker_permutations(_moran_stat, work_package)
        assert result.shape == (8,)

    def test_worker_permutations_no_seed(self):
        """Test _worker_permutations without seed."""
        np.random.seed(42)
        n = 10
        data = np.random.randn(n)
        W = np.eye(n)

        work_package = (0, 5, data, W, {}, None)
        result = ParallelPermutationTest._worker_permutations(_moran_stat, work_package)
        assert result.shape == (5,)


# -----------------------------------------------------------------------
# Spatial HAC tests
# -----------------------------------------------------------------------


class TestSpatialHACCoverage:
    """Additional ParallelSpatialHAC coverage."""

    def test_compute_chunk_default_kernel(self):
        """Test _compute_chunk with unknown kernel falls back to weight=1.0."""
        coords = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        chunk_info = (0, 3, coords, 5.0, "unknown_kernel")
        result = ParallelSpatialHAC._compute_chunk(chunk_info)
        assert result.shape == (3, 3)
        # All within cutoff should get weight 1.0 (default)
        assert result[0, 1] == 1.0
        assert result[0, 2] == 1.0

    def test_compute_chunk_epanechnikov(self):
        """Test _compute_chunk with epanechnikov kernel."""
        coords = np.array([[0, 0], [1, 0], [3, 0]], dtype=float)
        chunk_info = (0, 3, coords, 4.0, "epanechnikov")
        result = ParallelSpatialHAC._compute_chunk(chunk_info)
        assert result.shape == (3, 3)
        # distance=1, u=0.25: weight = 0.75*(1-0.0625) = 0.703125
        expected = 0.75 * (1 - (1.0 / 4.0) ** 2)
        assert_allclose(result[0, 1], expected, atol=1e-10)


# -----------------------------------------------------------------------
# _evaluate_params edge cases
# -----------------------------------------------------------------------


class TestEvaluateParamsCoverage:
    """Additional _evaluate_params coverage."""

    def test_evaluate_log_likelihood_failure(self):
        """Test _evaluate_params with log_likelihood scoring and failure."""

        class FailModel:
            def __init__(self, data, W):
                raise RuntimeError("fail")

            def fit(self):
                pass

        work_package = (
            FailModel,
            {},
            pd.DataFrame({"x": [1]}),
            np.eye(1),
            "log_likelihood",
            {},
        )
        result = _evaluate_params(work_package)
        assert result["score"] == -np.inf
        assert result["converged"] is False

    def test_evaluate_bic_success(self):
        """Test _evaluate_params with BIC scoring."""

        class MockResult:
            aic = 100.0
            bic = 95.0
            log_likelihood = -47.0
            converged = True

        class MockModel:
            def __init__(self, data, W):
                pass

            def fit(self):
                return MockResult()

        work_package = (
            MockModel,
            {},
            pd.DataFrame({"x": [1]}),
            np.eye(1),
            "bic",
            {},
        )
        result = _evaluate_params(work_package)
        assert result["score"] == 95.0
        assert result["converged"] is True


# -----------------------------------------------------------------------
# Spatial optimizations coverage
# -----------------------------------------------------------------------


class TestSpatialOptCoverage:
    """Additional spatial optimizations coverage."""

    def test_eigenvalue_cache_sparse(self):
        """Test EigenvalueCache with sparse matrix."""
        cache = EigenvalueCache()
        W = csr_matrix(np.eye(5) * 0.5)
        eigs = cache.get_eigenvalues(W)
        assert len(eigs) > 0
        assert np.all(np.isfinite(eigs))

    def test_sparse_operations_spatial_lag_sparse(self):
        """Test SparseOperations.spatial_lag with sparse input."""
        W = csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
        y = np.array([3.0, 7.0])
        result = SparseOperations.spatial_lag(W, y)
        assert_allclose(result, [7.0, 3.0])

    def test_sparse_operations_log_determinant_lu_sparse(self):
        """Test log_determinant with sparse LU method."""
        W = np.array([[0, 0.5], [0.5, 0]], dtype=float)
        rho = 0.3
        I_rho_W = np.eye(2) - rho * W
        sparse_mat = csr_matrix(I_rho_W)
        log_det = SparseOperations.log_determinant(sparse_mat, method="lu")
        expected = np.log(np.abs(np.linalg.det(I_rho_W)))
        assert_allclose(log_det, expected, atol=1e-10)

    def test_chebyshev_trace_powers_sparse(self):
        """Test ChebyshevApproximation trace_powers with sparse matrix."""
        W = csr_matrix(np.diag([0.5, 0.5, 0.5]))
        cheb = ChebyshevApproximation(order=10)
        traces = cheb.trace_powers(W, max_power=3)
        assert_allclose(traces[0], 1.5, atol=1e-10)
        assert_allclose(traces[1], 0.75, atol=1e-10)

    def test_get_eigenvalues_cached_module_level(self):
        """Test module-level get_eigenvalues_cached function."""
        W = np.array([[0, 1], [1, 0]], dtype=float)
        eigs = get_eigenvalues_cached(W)
        assert len(eigs) == 2
        assert np.all(np.isfinite(eigs))

    def test_compute_spatial_lag_fast_basic(self):
        """Test compute_spatial_lag_fast with simple CSR data."""
        W = csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
        y = np.array([2.0, 5.0])
        result = compute_spatial_lag_fast(W.data, W.indices, W.indptr, y)
        assert_allclose(result, [5.0, 2.0])

    def test_compute_quadratic_form_basic(self):
        """Test compute_quadratic_form returns finite scalar."""
        np.random.seed(42)
        n, k = 5, 2
        X = np.random.randn(n, k)
        W = np.eye(n)
        y = np.random.randn(n)
        result = compute_quadratic_form(X, W, y)
        assert np.isfinite(result)

    def test_row_standardize_fast_basic(self):
        """Test row_standardize_fast normalizes rows."""
        W = np.array([[0, 2, 4], [1, 0, 3], [0, 0, 0]], dtype=float)
        W_std = row_standardize_fast(W)
        # Non-zero rows sum to 1
        assert_allclose(W_std[0].sum(), 1.0, atol=1e-10)
        assert_allclose(W_std[1].sum(), 1.0, atol=1e-10)
        # Zero row stays zero
        assert_allclose(W_std[2].sum(), 0.0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
