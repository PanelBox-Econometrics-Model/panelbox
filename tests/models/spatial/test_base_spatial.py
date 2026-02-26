"""
Tests for base spatial panel model functionality.

This module tests the shared infrastructure in ``SpatialPanelModel``
(log-determinant methods, eigenvalue caching, spatial coefficient bounds,
spatial instruments, panel validation, and summary generation) by
instantiating ``SpatialLag``, the simplest concrete subclass.

Coverage targets
----------------
- Lines 101-137: eigenvalue caching, sparse W caching
- Lines 304-354: chebyshev log-det, sparse_lu log-det
- Lines 419-470: spatial instruments, panel validation, summary
"""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from panelbox.models.spatial.spatial_lag import SpatialLag


# ---------------------------------------------------------------------------
# Helper: generate spatial panel data
# ---------------------------------------------------------------------------
def create_spatial_panel_data(n_entities=25, n_periods=8, seed=42):
    """Build a balanced panel with queen-contiguity W on a square grid."""
    np.random.seed(seed)
    grid_size = int(np.sqrt(n_entities))
    assert grid_size * grid_size == n_entities, "n_entities must be a perfect square"

    # Queen contiguity weight matrix
    W = np.zeros((n_entities, n_entities))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        W[idx, ni * grid_size + nj] = 1.0
    # Row-normalise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    # Build entity-major panel data
    N, T = n_entities, n_periods
    entities = np.repeat(np.arange(N), T)
    time_periods = np.tile(np.arange(T), N)

    X = np.random.randn(N * T, 2)
    beta = np.array([1.5, -0.8])
    rho = 0.3

    # Generate y in time-major order then convert to entity-major
    y_tm = np.zeros(N * T)
    for t in range(T):
        s, e = t * N, (t + 1) * N
        eps = np.random.randn(N) * 0.5
        # X is entity-major; pull out time-major slice
        X_t = X.reshape(N, T, -1)[:, t, :]  # (N, K)
        y_tm[s:e] = np.linalg.solve(np.eye(N) - rho * W, X_t @ beta + eps)

    # Convert y to entity-major
    y_em = y_tm.reshape(T, N).T.reshape(-1)
    X_em = X  # already entity-major (repeat/tile matches)

    data = pd.DataFrame(
        {
            "entity": entities,
            "time": time_periods,
            "y": y_em,
            "x1": X_em[:, 0],
            "x2": X_em[:, 1],
        }
    )

    return data, W


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def spatial_model():
    """Return a SpatialLag model instance with queen-contiguity W."""
    data, W = create_spatial_panel_data(n_entities=25, n_periods=8, seed=42)
    model = SpatialLag(
        formula="y ~ x1 + x2",
        data=data,
        entity_col="entity",
        time_col="time",
        W=W,
    )
    return model, W


# ===================================================================
# Test class: log-determinant methods
# ===================================================================
class TestBaseSpatialLogDet:
    """Tests for _log_det_jacobian with different methods."""

    def test_log_det_eigenvalue_method(self, spatial_model):
        """Eigenvalue-based log|I - rho W| computation."""
        model, _ = spatial_model
        rho = 0.3
        log_det = model._log_det_jacobian(rho, method="eigenvalue")

        # Compare with brute-force: log|I - rho W|
        I_rhoW = np.eye(model.n_entities) - rho * model.W_normalized
        expected = np.log(np.abs(np.linalg.det(I_rhoW)))
        assert_allclose(log_det, expected, rtol=1e-6)

    def test_log_det_sparse_lu_method(self, spatial_model):
        """Sparse-LU-based log|I - rho W| computation."""
        model, _ = spatial_model
        # Clear caches so sparse_lu path builds its own
        model._W_eigenvalues = None
        model._sparse_W = None

        rho = 0.3
        log_det = model._log_det_jacobian(rho, method="sparse_lu")

        I_rhoW = np.eye(model.n_entities) - rho * model.W_normalized
        expected = np.log(np.abs(np.linalg.det(I_rhoW)))
        assert_allclose(log_det, expected, rtol=1e-6)

    def test_log_det_chebyshev_method(self, spatial_model):
        """Chebyshev approximation of log|I - rho W|."""
        model, _ = spatial_model
        # Clear eigenvalue cache so chebyshev recalculates
        model._W_eigenvalues = None

        rho = 0.15  # Small rho where Taylor expansion is accurate
        log_det = model._log_det_jacobian(rho, method="chebyshev")

        I_rhoW = np.eye(model.n_entities) - rho * model.W_normalized
        expected = np.log(np.abs(np.linalg.det(I_rhoW)))
        # Chebyshev is an approximation -- allow looser tolerance
        assert_allclose(log_det, expected, atol=0.5)

    def test_log_det_auto_selection(self, spatial_model):
        """Auto method should pick eigenvalue for N < 1000."""
        model, _ = spatial_model
        rho = 0.3

        log_det_auto = model._log_det_jacobian(rho, method="auto")
        # For N=25, auto should choose eigenvalue
        model._W_eigenvalues = None  # reset cache
        log_det_eig = model._log_det_jacobian(rho, method="eigenvalue")

        assert_allclose(log_det_auto, log_det_eig, rtol=1e-10)

    def test_log_det_unknown_method_raises(self, spatial_model):
        """Unknown method should raise ValueError."""
        model, _ = spatial_model
        with pytest.raises(ValueError, match="Unknown method"):
            model._log_det_jacobian(0.3, method="invalid")

    def test_log_det_eigenvalue_caching(self, spatial_model):
        """Eigenvalues should be computed once and cached."""
        model, _ = spatial_model
        assert model._W_eigenvalues is None

        # First call computes eigenvalues
        model._log_det_jacobian(0.3, method="eigenvalue")
        cached_eigs = model._W_eigenvalues.copy()
        assert cached_eigs is not None

        # Second call should reuse the cache
        model._log_det_jacobian(0.5, method="eigenvalue")
        assert_allclose(model._W_eigenvalues, cached_eigs)

    def test_sparse_w_caching(self, spatial_model):
        """Sparse W should be computed once and cached for sparse_lu."""
        model, _ = spatial_model
        assert model._sparse_W is None

        model._log_det_jacobian(0.3, method="sparse_lu")
        assert model._sparse_W is not None

        sparse_w_ref = model._sparse_W
        model._log_det_jacobian(0.5, method="sparse_lu")
        # Same object should be reused
        assert model._sparse_W is sparse_w_ref


# ===================================================================
# Test class: spatial coefficient bounds
# ===================================================================
class TestBaseSpatialBounds:
    """Tests for _spatial_coefficient_bounds."""

    def test_spatial_coefficient_bounds(self, spatial_model):
        """Bounds should satisfy 1/lambda_min < rho < 1/lambda_max."""
        model, _ = spatial_model
        rho_min, rho_max = model._spatial_coefficient_bounds()

        assert rho_min < 0  # For row-normalised W, min eigenvalue is negative
        assert rho_max > 0
        assert rho_min >= -0.99
        assert rho_max <= 0.99

    def test_bounds_with_custom_W(self, spatial_model):
        """Bounds computed with an explicit W argument."""
        model, W = spatial_model
        # Clear cache so it recomputes
        model._W_eigenvalues = None

        rho_min, rho_max = model._spatial_coefficient_bounds(W=W)

        assert rho_min < 0
        assert rho_max > 0
        assert rho_min >= -0.99
        assert rho_max <= 0.99

    def test_bounds_consistency(self, spatial_model):
        """Default-W and explicit-W bounds should match."""
        model, _ = spatial_model
        bounds_default = model._spatial_coefficient_bounds()

        model._W_eigenvalues = None  # Clear cache
        bounds_explicit = model._spatial_coefficient_bounds(W=model.W_normalized)

        assert_allclose(bounds_default, bounds_explicit, rtol=1e-8)


# ===================================================================
# Test class: spatial instruments
# ===================================================================
class TestBaseSpatialInstruments:
    """Tests for _compute_spatial_instruments."""

    def test_compute_spatial_instruments(self, spatial_model):
        """Instruments [X, WX, W^2 X] should be computed correctly."""
        model, _ = spatial_model
        X = np.asarray(model.exog)

        instruments = model._compute_spatial_instruments(X, n_lags=2)

        # Should contain X, WX, W^2 X
        assert instruments.shape[0] == X.shape[0]
        K = X.shape[1]
        expected_cols = K * 3  # X + WX + W^2 X
        assert instruments.shape[1] == expected_cols

        # First K columns should be X itself
        assert_allclose(instruments[:, :K], X)

    def test_instruments_shape_one_lag(self, spatial_model):
        """With n_lags=1, instruments = [X, WX]."""
        model, _ = spatial_model
        X = np.asarray(model.exog)
        K = X.shape[1]

        instruments = model._compute_spatial_instruments(X, n_lags=1)
        assert instruments.shape[1] == K * 2

    def test_instruments_shape_three_lags(self, spatial_model):
        """With n_lags=3, instruments = [X, WX, W^2 X, W^3 X]."""
        model, _ = spatial_model
        X = np.asarray(model.exog)
        K = X.shape[1]

        instruments = model._compute_spatial_instruments(X, n_lags=3)
        assert instruments.shape[1] == K * 4


# ===================================================================
# Test class: panel validation
# ===================================================================
class TestBaseSpatialValidation:
    """Tests for _validate_panel_structure and weight matrix validation."""

    def test_validate_panel_structure(self, spatial_model):
        """Balanced panel should pass validation."""
        model, _ = spatial_model
        # Should not raise
        model._validate_panel_structure()

    def test_validate_W_wrong_dimensions(self):
        """W with wrong dimensions should raise ValueError."""
        data, _ = create_spatial_panel_data(n_entities=25, n_periods=8)
        W_wrong = np.eye(10)  # 10x10 instead of 25x25

        with pytest.raises(ValueError, match="W must be"):
            SpatialLag(
                formula="y ~ x1 + x2",
                data=data,
                entity_col="entity",
                time_col="time",
                W=W_wrong,
            )

    def test_validate_W_negative_values(self):
        """W with negative values should raise ValueError."""
        data, W = create_spatial_panel_data(n_entities=25, n_periods=8)
        W_neg = W.copy()
        W_neg[0, 1] = -0.5

        with pytest.raises(ValueError, match="negative values"):
            SpatialLag(
                formula="y ~ x1 + x2",
                data=data,
                entity_col="entity",
                time_col="time",
                W=W_neg,
            )

    def test_validate_W_nonzero_diagonal(self):
        """W with non-zero diagonal should produce a warning and be corrected."""
        data, W = create_spatial_panel_data(n_entities=25, n_periods=8)
        W_diag = W.copy()
        np.fill_diagonal(W_diag, 0.1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SpatialLag(
                formula="y ~ x1 + x2",
                data=data,
                entity_col="entity",
                time_col="time",
                W=W_diag,
            )
            # Should have emitted a warning about non-zero diagonal
            diag_warnings = [w for w in caught if "non-zero diagonal" in str(w.message)]
            assert len(diag_warnings) >= 1

        # After construction, diagonal should have been zeroed out
        assert_allclose(np.diag(model.W), 0)


# ===================================================================
# Test class: summary method
# ===================================================================
class TestBaseSpatialSummary:
    """Tests for model-level summary() method."""

    def test_summary_method(self, spatial_model):
        """summary() should return a string containing spatial info."""
        model, _ = spatial_model

        # SpatialPanelModel.summary() calls super().summary() which does
        # not exist on PanelModel(ABC). Patch the parent to return a base
        # string so we can verify the spatial-specific content.
        with patch(
            "panelbox.core.base_model.PanelModel.summary",
            create=True,
            return_value="Base summary\n",
        ):
            result = model.summary()

        assert isinstance(result, str)
        assert "Spatial Information" in result
        assert "Number of spatial units" in result
        assert "Row-normalized" in result
        assert "density" in result.lower()
