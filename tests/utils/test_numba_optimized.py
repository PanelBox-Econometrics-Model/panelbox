"""Tests for Numba-optimized functions."""

import numpy as np
import pytest

from panelbox.utils.numba_optimized import (
    NUMBA_AVAILABLE,
    compute_gmm_weight_matrix_numba,
    demean_within_1d_numba,
    demean_within_numba,
    fill_gmm_style_instruments_numba,
    fill_iv_instruments_numba,
    get_numba_status,
    suggest_optimization_targets,
    validate_numba_optimization,
)


class TestNumbaAvailability:
    """Test Numba availability and status."""

    def test_get_numba_status_returns_dict(self):
        """Test get_numba_status returns proper dictionary."""
        status = get_numba_status()

        assert isinstance(status, dict)
        assert "available" in status
        assert "version" in status
        assert "parallel_available" in status
        assert "cache_enabled" in status

    def test_get_numba_status_available_field(self):
        """Test available field is boolean."""
        status = get_numba_status()
        assert isinstance(status["available"], bool)

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_get_numba_status_with_numba(self):
        """Test status when Numba is available."""
        status = get_numba_status()

        assert status["available"] is True
        assert status["version"] is not None
        assert isinstance(status["parallel_available"], bool)

    def test_numba_available_constant(self):
        """Test NUMBA_AVAILABLE constant is boolean."""
        assert isinstance(NUMBA_AVAILABLE, bool)


class TestOptimizationSuggestions:
    """Test optimization suggestion helpers."""

    def test_suggest_optimization_targets_pooled(self):
        """Test suggestions for pooled OLS."""
        suggestions = suggest_optimization_targets("pooled")
        assert isinstance(suggestions, list)
        assert len(suggestions) == 0  # Pooled doesn't need Numba

    def test_suggest_optimization_targets_fe(self):
        """Test suggestions for fixed effects."""
        suggestions = suggest_optimization_targets("fe")
        assert isinstance(suggestions, list)
        assert "demean_within_numba" in suggestions
        assert "demean_within_1d_numba" in suggestions

    def test_suggest_optimization_targets_re(self):
        """Test suggestions for random effects."""
        suggestions = suggest_optimization_targets("re")
        assert isinstance(suggestions, list)
        assert "demean_within_numba" in suggestions

    def test_suggest_optimization_targets_diff_gmm(self):
        """Test suggestions for difference GMM."""
        suggestions = suggest_optimization_targets("diff_gmm")
        assert isinstance(suggestions, list)
        assert "fill_iv_instruments_numba" in suggestions
        assert "compute_gmm_weight_matrix_numba" in suggestions

    def test_suggest_optimization_targets_sys_gmm(self):
        """Test suggestions for system GMM."""
        suggestions = suggest_optimization_targets("sys_gmm")
        assert isinstance(suggestions, list)
        assert "fill_iv_instruments_numba" in suggestions
        assert "fill_gmm_style_instruments_numba" in suggestions
        assert "compute_gmm_weight_matrix_numba" in suggestions

    def test_suggest_optimization_targets_unknown(self):
        """Test suggestions for unknown model type."""
        suggestions = suggest_optimization_targets("unknown_model")
        assert suggestions == []


class TestDemeanWithin:
    """Test within-transformation (demeaning) functions."""

    @pytest.fixture
    def sample_data_2d(self):
        """Create sample 2D data for demeaning."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        entity_ids = np.array([1, 1, 2, 2, 2])
        return X, entity_ids

    @pytest.fixture
    def sample_data_1d(self):
        """Create sample 1D data for demeaning."""
        x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        entity_ids = np.array([1, 1, 2, 2, 2])
        return x, entity_ids

    def test_demean_within_numba_shape(self, sample_data_2d):
        """Test demeaning preserves shape."""
        X, entity_ids = sample_data_2d
        X_demeaned = demean_within_numba(X, entity_ids)

        assert X_demeaned.shape == X.shape

    def test_demean_within_numba_correctness(self, sample_data_2d):
        """Test demeaning produces correct results."""
        X, entity_ids = sample_data_2d
        X_demeaned = demean_within_numba(X, entity_ids)

        # Entity 1: rows 0-1, mean = [2, 3]
        assert np.allclose(X_demeaned[0], [-1.0, -1.0])
        assert np.allclose(X_demeaned[1], [1.0, 1.0])

        # Entity 2: rows 2-4, mean = [7, 8]
        assert np.allclose(X_demeaned[2], [-2.0, -2.0])
        assert np.allclose(X_demeaned[3], [0.0, 0.0])
        assert np.allclose(X_demeaned[4], [2.0, 2.0])

    def test_demean_within_numba_zero_mean(self, sample_data_2d):
        """Test demeaned data has zero mean per entity."""
        X, entity_ids = sample_data_2d
        X_demeaned = demean_within_numba(X, entity_ids)

        # Entity 1
        entity1_mean = X_demeaned[entity_ids == 1].mean(axis=0)
        assert np.allclose(entity1_mean, 0, atol=1e-10)

        # Entity 2
        entity2_mean = X_demeaned[entity_ids == 2].mean(axis=0)
        assert np.allclose(entity2_mean, 0, atol=1e-10)

    def test_demean_within_1d_numba_shape(self, sample_data_1d):
        """Test 1D demeaning preserves shape."""
        x, entity_ids = sample_data_1d
        x_demeaned = demean_within_1d_numba(x, entity_ids)

        assert x_demeaned.shape == x.shape

    def test_demean_within_1d_numba_correctness(self, sample_data_1d):
        """Test 1D demeaning produces correct results."""
        x, entity_ids = sample_data_1d
        x_demeaned = demean_within_1d_numba(x, entity_ids)

        # Entity 1: mean = 2
        assert np.allclose(x_demeaned[0], -1.0)
        assert np.allclose(x_demeaned[1], 1.0)

        # Entity 2: mean = 7
        assert np.allclose(x_demeaned[2], -2.0)
        assert np.allclose(x_demeaned[3], 0.0)
        assert np.allclose(x_demeaned[4], 2.0)

    def test_demean_single_entity(self):
        """Test demeaning with single entity."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        entity_ids = np.array([1, 1, 1])

        X_demeaned = demean_within_numba(X, entity_ids)

        # All same entity, should center at zero
        assert np.allclose(X_demeaned.mean(axis=0), 0, atol=1e-10)

    def test_demean_single_obs_per_entity(self):
        """Test demeaning with single observation per entity."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        entity_ids = np.array([1, 2, 3])

        X_demeaned = demean_within_numba(X, entity_ids)

        # Single obs per entity means demeaned = 0
        assert np.allclose(X_demeaned, 0, atol=1e-10)


class TestFillIVInstruments:
    """Test IV-style instrument filling."""

    @pytest.fixture
    def sample_iv_data(self):
        """Create sample data for IV instruments."""
        n_obs = 6
        Z = np.zeros((n_obs, 3))  # 3 instruments (lags 1-3)
        var_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ids = np.array([1, 1, 1, 2, 2, 2])
        times = np.array([1, 2, 3, 1, 2, 3])
        min_lag = 1
        max_lag = 3
        return Z, var_data, ids, times, min_lag, max_lag

    def test_fill_iv_instruments_numba_shape(self, sample_iv_data):
        """Test instrument matrix retains shape."""
        Z, var_data, ids, times, min_lag, max_lag = sample_iv_data
        Z_filled = fill_iv_instruments_numba(
            Z.copy(), var_data, ids, times, min_lag, max_lag, "diff"
        )

        assert Z_filled.shape == Z.shape

    def test_fill_iv_instruments_numba_diff_equation(self, sample_iv_data):
        """Test filling for difference equation."""
        Z, var_data, ids, times, min_lag, max_lag = sample_iv_data
        Z_filled = fill_iv_instruments_numba(
            Z.copy(), var_data, ids, times, min_lag, max_lag, "diff"
        )

        # For difference equation, instruments are lagged levels
        # t=2 for entity 1: lag 1 should be var_data[0] = 1.0
        assert Z_filled[1, 0] == 1.0  # Row 1 (t=2), lag 1

        # t=3 for entity 1: lag 1 should be 2.0, lag 2 should be 1.0
        assert Z_filled[2, 0] == 2.0  # lag 1
        assert Z_filled[2, 1] == 1.0  # lag 2

    def test_fill_iv_instruments_numba_level_equation(self, sample_iv_data):
        """Test filling for level equation."""
        Z, var_data, ids, times, min_lag, max_lag = sample_iv_data
        Z_filled = fill_iv_instruments_numba(
            Z.copy(), var_data, ids, times, min_lag, max_lag, "level"
        )

        # For level equation, instruments are lagged differences
        # t=2 for entity 1: lag 1 difference should be var_data[0] - var_data[-1]
        # But var_data[-1] doesn't exist for t=0, so should be 0

        # Non-zero values should exist where lags are available
        assert Z_filled.shape == Z.shape

    def test_fill_iv_instruments_numba_missing_lags(self):
        """Test handling of missing lags."""
        Z = np.zeros((3, 2))
        var_data = np.array([1.0, 2.0, 3.0])
        ids = np.array([1, 1, 1])
        times = np.array([1, 2, 3])

        Z_filled = fill_iv_instruments_numba(Z, var_data, ids, times, 1, 2, "diff")

        # t=1: no lags available
        assert np.allclose(Z_filled[0], 0)

        # t=2: only lag 1 available
        assert Z_filled[1, 0] == 1.0
        assert Z_filled[1, 1] == 0.0

        # t=3: lags 1 and 2 available
        assert Z_filled[2, 0] == 2.0
        assert Z_filled[2, 1] == 1.0


class TestFillGMMStyleInstruments:
    """Test GMM-style instrument filling."""

    def test_fill_gmm_style_instruments_numba_basic(self):
        """Test basic GMM-style instrument filling."""
        var_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ids = np.array([1, 1, 1, 2, 2, 2])
        times = np.array([1, 2, 3, 1, 2, 3])
        unique_times = np.array([1, 2, 3])

        # Create list of instrument matrices (one per time)
        Z_list = [np.zeros((6, 2)) for _ in range(3)]

        Z_list_filled = fill_gmm_style_instruments_numba(
            Z_list, var_data, ids, times, unique_times, 1, 2
        )

        assert len(Z_list_filled) == 3
        for Z in Z_list_filled:
            assert Z.shape == (6, 2)

    def test_fill_gmm_style_instruments_numba_correctness(self):
        """Test correctness of GMM-style filling."""
        var_data = np.array([1.0, 2.0, 3.0, 4.0])
        ids = np.array([1, 1, 1, 1])
        times = np.array([1, 2, 3, 4])
        unique_times = np.array([1, 2, 3, 4])

        Z_list = [np.zeros((4, 2)) for _ in range(4)]

        Z_list_filled = fill_gmm_style_instruments_numba(
            Z_list, var_data, ids, times, unique_times, 1, 2
        )

        # t=2 (idx=1): lag 1 = var_data[0] = 1.0
        assert Z_list_filled[1][1, 0] == 1.0

        # t=3 (idx=2): lag 1 = 2.0, lag 2 = 1.0
        assert Z_list_filled[2][2, 0] == 2.0
        assert Z_list_filled[2][2, 1] == 1.0


class TestComputeGMMWeightMatrix:
    """Test GMM weight matrix computation."""

    @pytest.fixture
    def sample_gmm_data(self):
        """Create sample data for weight matrix."""
        residuals = np.array([0.1, -0.2, 0.15, -0.1, 0.05, -0.15])
        Z = np.random.randn(6, 4)
        entity_ids = np.array([1, 1, 1, 2, 2, 2])
        return residuals, Z, entity_ids

    def test_compute_gmm_weight_matrix_numba_shape(self, sample_gmm_data):
        """Test weight matrix has correct shape."""
        residuals, Z, entity_ids = sample_gmm_data
        W = compute_gmm_weight_matrix_numba(residuals, Z, entity_ids, robust=True)

        n_instruments = Z.shape[1]
        assert W.shape == (n_instruments, n_instruments)

    def test_compute_gmm_weight_matrix_numba_symmetric(self, sample_gmm_data):
        """Test weight matrix is symmetric."""
        residuals, Z, entity_ids = sample_gmm_data
        W = compute_gmm_weight_matrix_numba(residuals, Z, entity_ids, robust=True)

        assert np.allclose(W, W.T)

    def test_compute_gmm_weight_matrix_numba_robust(self, sample_gmm_data):
        """Test robust weight matrix computation."""
        residuals, Z, entity_ids = sample_gmm_data
        W_robust = compute_gmm_weight_matrix_numba(residuals, Z, entity_ids, robust=True)

        # Should be positive values on diagonal
        assert np.all(np.diag(W_robust) >= 0)

    def test_compute_gmm_weight_matrix_numba_nonrobust(self, sample_gmm_data):
        """Test non-robust weight matrix computation."""
        residuals, Z, entity_ids = sample_gmm_data
        W_nonrobust = compute_gmm_weight_matrix_numba(residuals, Z, entity_ids, robust=False)

        # Should be positive values on diagonal
        assert np.all(np.diag(W_nonrobust) >= 0)

    def test_compute_gmm_weight_matrix_robust_vs_nonrobust(self, sample_gmm_data):
        """Test robust and non-robust produce different results."""
        residuals, Z, entity_ids = sample_gmm_data
        W_robust = compute_gmm_weight_matrix_numba(residuals, Z, entity_ids, robust=True)
        W_nonrobust = compute_gmm_weight_matrix_numba(residuals, Z, entity_ids, robust=False)

        # They should generally differ
        assert not np.allclose(W_robust, W_nonrobust)

    def test_compute_gmm_weight_matrix_zero_residuals(self):
        """Test weight matrix with zero residuals."""
        residuals = np.zeros(6)
        Z = np.random.randn(6, 3)
        entity_ids = np.array([1, 1, 1, 2, 2, 2])

        W = compute_gmm_weight_matrix_numba(residuals, Z, entity_ids, robust=True)

        # With zero residuals, weight matrix should be zero
        assert np.allclose(W, 0)


class TestValidateNumbaOptimization:
    """Test validation helper function."""

    def test_validate_numba_optimization_match(self):
        """Test validation when functions match."""

        def func_original(x):
            return x * 2

        def func_numba(x):
            return x * 2

        result = validate_numba_optimization(func_original, func_numba, 5.0)

        assert result["match"] is True
        assert result["max_diff"] == 0.0

    def test_validate_numba_optimization_mismatch(self):
        """Test validation when functions don't match."""

        def func_original(x):
            return x * 2

        def func_numba(x):
            return x * 3

        result = validate_numba_optimization(func_original, func_numba, 5.0)

        assert result["match"] is False
        assert result["max_diff"] > 0

    def test_validate_numba_optimization_arrays(self):
        """Test validation with numpy arrays."""

        def func_original(x):
            return x * 2

        def func_numba(x):
            return x * 2

        x = np.array([1.0, 2.0, 3.0])
        result = validate_numba_optimization(func_original, func_numba, x)

        assert result["match"]
        assert result["max_diff"] == 0.0

    def test_validate_numba_optimization_small_diff(self):
        """Test validation with very small differences."""

        def func_original(x):
            return x

        def func_numba(x):
            return x + 1e-15

        result = validate_numba_optimization(func_original, func_numba, 5.0)

        # Should match with 1e-10 tolerance
        assert result["match"] is True


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_demean_empty_array(self):
        """Test demeaning empty array."""
        X = np.array([]).reshape(0, 2)
        entity_ids = np.array([])

        X_demeaned = demean_within_numba(X, entity_ids)
        assert X_demeaned.shape == (0, 2)

    def test_demean_single_column(self):
        """Test demeaning single column."""
        X = np.array([[1.0], [2.0], [3.0]])
        entity_ids = np.array([1, 1, 1])

        X_demeaned = demean_within_numba(X, entity_ids)
        assert X_demeaned.shape == (3, 1)
        assert np.allclose(X_demeaned.mean(), 0, atol=1e-10)

    def test_fill_iv_instruments_zero_lags(self):
        """Test IV instruments with zero lag range."""
        Z = np.zeros((3, 0))  # No instruments
        var_data = np.array([1.0, 2.0, 3.0])
        ids = np.array([1, 1, 1])
        times = np.array([1, 2, 3])

        # min_lag > max_lag means no lags
        Z_filled = fill_iv_instruments_numba(Z, var_data, ids, times, 2, 1, "diff")
        assert Z_filled.shape[1] == 0

    def test_weight_matrix_single_instrument(self):
        """Test weight matrix with single instrument."""
        residuals = np.array([0.1, -0.2, 0.15])
        Z = np.random.randn(3, 1)
        entity_ids = np.array([1, 1, 1])

        W = compute_gmm_weight_matrix_numba(residuals, Z, entity_ids, robust=True)

        assert W.shape == (1, 1)
        assert W[0, 0] >= 0

    def test_demean_large_number_entities(self):
        """Test demeaning with many entities."""
        n_entities = 100
        n_obs_per_entity = 5
        n_obs = n_entities * n_obs_per_entity

        X = np.random.randn(n_obs, 3)
        entity_ids = np.repeat(np.arange(n_entities), n_obs_per_entity)

        X_demeaned = demean_within_numba(X, entity_ids)

        # Verify each entity has zero mean
        for entity in range(n_entities):
            entity_mask = entity_ids == entity
            entity_mean = X_demeaned[entity_mask].mean(axis=0)
            assert np.allclose(entity_mean, 0, atol=1e-10)


class TestFallbackBehavior:
    """Test fallback behavior when Numba is not available."""

    def test_functions_work_without_numba(self):
        """Test that functions work even without Numba (fallback)."""
        # These tests will run regardless of Numba availability
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        entity_ids = np.array([1, 1])

        # Should not raise error
        result = demean_within_numba(X, entity_ids)
        assert result.shape == X.shape

    def test_jit_decorator_fallback(self):
        """Test JIT decorator fallback when Numba not available."""
        if not NUMBA_AVAILABLE:
            # Import the fallback jit decorator
            from panelbox.utils.numba_optimized import jit

            # Should work as no-op decorator
            @jit
            def test_func(x):
                return x * 2

            assert test_func(5) == 10

            # Should also work with parentheses
            @jit()
            def test_func2(x):
                return x * 3

            assert test_func2(5) == 15
