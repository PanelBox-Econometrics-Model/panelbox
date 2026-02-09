"""
Tests for Driscoll-Kraay Standard Errors.
"""

import numpy as np
import pytest

from panelbox.standard_errors.driscoll_kraay import (
    DriscollKraayResult,
    DriscollKraayStandardErrors,
    driscoll_kraay,
)


@pytest.fixture
def panel_data():
    """Create panel data for testing."""
    # 10 entities, 20 time periods (good for D-K which needs large T)
    n_entities = 10
    n_periods = 20
    n_params = 3

    np.random.seed(42)

    # Create time IDs (repeated for each entity)
    time_ids = np.tile(np.arange(n_periods), n_entities)

    # Create design matrix
    X = np.random.randn(n_entities * n_periods, n_params)

    # Create residuals with some autocorrelation
    resid = np.random.randn(n_entities * n_periods) * 0.5

    return X, resid, time_ids


@pytest.fixture
def small_panel_data():
    """Create smaller panel data."""
    n_entities = 5
    n_periods = 10
    n_params = 2

    np.random.seed(123)
    time_ids = np.tile(np.arange(n_periods), n_entities)
    X = np.random.randn(n_entities * n_periods, n_params)
    resid = np.random.randn(n_entities * n_periods)

    return X, resid, time_ids


class TestDriscollKraayResult:
    """Test DriscollKraayResult dataclass."""

    def test_result_creation(self):
        """Test creating DriscollKraayResult."""
        cov_matrix = np.array([[1.0, 0.1, 0.05], [0.1, 2.0, 0.15], [0.05, 0.15, 1.5]])
        std_errors = np.array([1.0, 1.414, 1.225])

        result = DriscollKraayResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=3,
            kernel="bartlett",
            n_obs=200,
            n_params=3,
            n_periods=20,
            bandwidth=None,
        )

        assert result.n_obs == 200
        assert result.n_params == 3
        assert result.n_periods == 20
        assert result.max_lags == 3
        assert result.kernel == "bartlett"
        assert result.cov_matrix.shape == (3, 3)
        assert result.std_errors.shape == (3,)
        assert result.bandwidth is None


class TestDriscollKraayStandardErrors:
    """Test DriscollKraayStandardErrors class."""

    def test_initialization(self, panel_data):
        """Test D-K initialization."""
        X, resid, time_ids = panel_data

        dk = DriscollKraayStandardErrors(X, resid, time_ids)

        assert dk.n_obs == 200
        assert dk.n_params == 3
        assert dk.n_periods == 20
        assert dk.kernel == "bartlett"  # default
        assert dk.X.shape == (200, 3)
        assert dk.resid.shape == (200,)

    def test_dimension_validation(self, panel_data):
        """Test that time_ids dimension mismatch raises error."""
        X, resid, time_ids = panel_data

        with pytest.raises(ValueError, match="time_ids dimension mismatch"):
            DriscollKraayStandardErrors(X, resid, time_ids[:-5])

    def test_automatic_lag_selection(self, panel_data):
        """Test automatic lag selection using Newey-West rule."""
        X, resid, time_ids = panel_data

        dk = DriscollKraayStandardErrors(X, resid, time_ids)

        # For T=20: floor(4*(20/100)^(2/9)) = floor(4*0.688) = floor(2.75) = 2
        expected_lags = int(np.floor(4 * (20 / 100) ** (2 / 9)))
        assert dk.max_lags == expected_lags

    def test_manual_lag_specification(self, panel_data):
        """Test manual lag specification."""
        X, resid, time_ids = panel_data

        dk = DriscollKraayStandardErrors(X, resid, time_ids, max_lags=5)

        assert dk.max_lags == 5

    def test_max_lags_capped_at_t_minus_1(self):
        """Test that max_lags is capped at T-1."""
        n_periods = 10
        time_ids = np.tile(np.arange(n_periods), 5)
        X = np.random.randn(50, 2)
        resid = np.random.randn(50)

        # Try to set max_lags > T
        dk = DriscollKraayStandardErrors(X, resid, time_ids, max_lags=15)

        # Should be capped at T-1
        assert dk.max_lags == n_periods - 1

    def test_bartlett_kernel_weights(self, panel_data):
        """Test Bartlett kernel weights."""
        X, resid, time_ids = panel_data

        dk = DriscollKraayStandardErrors(X, resid, time_ids, max_lags=4, kernel="bartlett")

        # Bartlett: w(l) = 1 - l/(L+1)
        assert dk._kernel_weight(0) == pytest.approx(1.0)
        assert dk._kernel_weight(1) == pytest.approx(1 - 1 / 5)  # 0.8
        assert dk._kernel_weight(2) == pytest.approx(1 - 2 / 5)  # 0.6
        assert dk._kernel_weight(4) == pytest.approx(1 - 4 / 5)  # 0.2
        assert dk._kernel_weight(5) == 0.0  # Beyond max_lags

    def test_parzen_kernel_weights(self, panel_data):
        """Test Parzen kernel weights."""
        X, resid, time_ids = panel_data

        dk = DriscollKraayStandardErrors(X, resid, time_ids, max_lags=6, kernel="parzen")

        # For lag=0, z=0, weight = 1 - 0 + 0 = 1
        assert dk._kernel_weight(0) == pytest.approx(1.0)

        # For small z (z <= 0.5), w = 1 - 6z^2 + 6z^3
        # Beyond max_lags should be 0
        assert dk._kernel_weight(10) == 0.0

    def test_quadratic_spectral_kernel_weights(self, panel_data):
        """Test Quadratic Spectral kernel weights."""
        X, resid, time_ids = panel_data

        dk = DriscollKraayStandardErrors(
            X, resid, time_ids, max_lags=4, kernel="quadratic_spectral"
        )

        # For lag=0, should be 1.0
        assert dk._kernel_weight(0) == pytest.approx(1.0)

        # For lag > max_lags, should be 0
        assert dk._kernel_weight(10) == 0.0

    def test_invalid_kernel_raises_error(self, panel_data):
        """Test that invalid kernel raises error."""
        X, resid, time_ids = panel_data

        dk = DriscollKraayStandardErrors(X, resid, time_ids, kernel="bartlett")
        dk.kernel = "invalid_kernel"

        with pytest.raises(ValueError, match="Unknown kernel"):
            dk._kernel_weight(1)

    def test_time_sorting(self, panel_data):
        """Test time sorting functionality."""
        X, resid, time_ids = panel_data

        dk = DriscollKraayStandardErrors(X, resid, time_ids)
        sorted_data = dk._sort_by_time()

        assert "X" in sorted_data
        assert "resid" in sorted_data
        assert "time_ids" in sorted_data
        assert "unique_times" in sorted_data
        assert len(sorted_data["unique_times"]) == 20

    def test_bread_matrix_caching(self, panel_data):
        """Test that bread matrix is cached."""
        X, resid, time_ids = panel_data

        dk = DriscollKraayStandardErrors(X, resid, time_ids)

        # First access computes
        bread1 = dk.bread
        # Second access uses cache
        bread2 = dk.bread

        assert bread1 is bread2  # Same object
        assert bread1.shape == (3, 3)

    def test_compute_basic(self, panel_data):
        """Test basic D-K computation."""
        X, resid, time_ids = panel_data

        dk = DriscollKraayStandardErrors(X, resid, time_ids)
        result = dk.compute()

        assert isinstance(result, DriscollKraayResult)
        assert result.cov_matrix.shape == (3, 3)
        assert result.std_errors.shape == (3,)
        assert np.all(result.std_errors > 0)
        assert result.n_periods == 20

    def test_compute_with_different_kernels(self, small_panel_data):
        """Test computation with different kernel types."""
        X, resid, time_ids = small_panel_data

        # Bartlett
        dk_bartlett = DriscollKraayStandardErrors(X, resid, time_ids, kernel="bartlett")
        result_bartlett = dk_bartlett.compute()

        # Parzen
        dk_parzen = DriscollKraayStandardErrors(X, resid, time_ids, kernel="parzen")
        result_parzen = dk_parzen.compute()

        # Quadratic Spectral
        dk_qs = DriscollKraayStandardErrors(X, resid, time_ids, kernel="quadratic_spectral")
        result_qs = dk_qs.compute()

        # All should produce valid results
        assert result_bartlett.kernel == "bartlett"
        assert result_parzen.kernel == "parzen"
        assert result_qs.kernel == "quadratic_spectral"

        # All should have positive standard errors
        assert np.all(result_bartlett.std_errors > 0)
        assert np.all(result_parzen.std_errors > 0)
        assert np.all(result_qs.std_errors > 0)

    def test_covariance_matrix_symmetry(self, panel_data):
        """Test that covariance matrix is symmetric."""
        X, resid, time_ids = panel_data

        dk = DriscollKraayStandardErrors(X, resid, time_ids)
        result = dk.compute()

        assert np.allclose(result.cov_matrix, result.cov_matrix.T)

    def test_std_errors_from_diagonal(self, panel_data):
        """Test that standard errors are sqrt of diagonal."""
        X, resid, time_ids = panel_data

        dk = DriscollKraayStandardErrors(X, resid, time_ids)
        result = dk.compute()

        expected_se = np.sqrt(np.diag(result.cov_matrix))
        assert np.allclose(result.std_errors, expected_se)

    def test_diagnostic_summary(self, panel_data):
        """Test diagnostic summary generation."""
        X, resid, time_ids = panel_data

        dk = DriscollKraayStandardErrors(X, resid, time_ids)
        summary = dk.diagnostic_summary()

        assert "Driscoll-Kraay Standard Errors Diagnostics" in summary
        assert "Number of observations: 200" in summary
        assert "Number of time periods: 20" in summary
        assert "Maximum lags:" in summary
        assert "Kernel function: bartlett" in summary

    def test_diagnostic_warning_few_periods(self):
        """Test diagnostic warning for few time periods."""
        n_periods = 15  # < 20
        time_ids = np.tile(np.arange(n_periods), 5)
        X = np.random.randn(75, 2)
        resid = np.random.randn(75)

        dk = DriscollKraayStandardErrors(X, resid, time_ids)
        summary = dk.diagnostic_summary()

        assert "WARNING: Few time periods (<20)" in summary

    def test_diagnostic_warning_large_lags(self):
        """Test diagnostic warning for large max_lags."""
        n_periods = 10
        time_ids = np.tile(np.arange(n_periods), 5)
        X = np.random.randn(50, 2)
        resid = np.random.randn(50)

        # Set max_lags > T/4
        dk = DriscollKraayStandardErrors(X, resid, time_ids, max_lags=4)
        summary = dk.diagnostic_summary()

        assert "WARNING: Large max_lags relative to T" in summary

    def test_different_lag_specifications(self, panel_data):
        """Test with different lag specifications."""
        X, resid, time_ids = panel_data

        # Zero lags (only contemporaneous)
        dk0 = DriscollKraayStandardErrors(X, resid, time_ids, max_lags=0)
        result0 = dk0.compute()
        assert result0.max_lags == 0

        # Few lags
        dk2 = DriscollKraayStandardErrors(X, resid, time_ids, max_lags=2)
        result2 = dk2.compute()
        assert result2.max_lags == 2

        # Many lags
        dk10 = DriscollKraayStandardErrors(X, resid, time_ids, max_lags=10)
        result10 = dk10.compute()
        assert result10.max_lags == 10

        # All should produce valid results
        assert np.all(result0.std_errors > 0)
        assert np.all(result2.std_errors > 0)
        assert np.all(result10.std_errors > 0)


class TestConvenienceFunction:
    """Test the driscoll_kraay() convenience function."""

    def test_convenience_function(self, panel_data):
        """Test driscoll_kraay() convenience function."""
        X, resid, time_ids = panel_data

        result = driscoll_kraay(X, resid, time_ids)

        assert isinstance(result, DriscollKraayResult)
        assert result.cov_matrix.shape == (3, 3)
        assert result.std_errors.shape == (3,)

    def test_convenience_with_parameters(self, panel_data):
        """Test convenience function with parameters."""
        X, resid, time_ids = panel_data

        result = driscoll_kraay(X, resid, time_ids, max_lags=5, kernel="parzen")

        assert result.max_lags == 5
        assert result.kernel == "parzen"

    def test_convenience_equals_class_method(self, panel_data):
        """Test that convenience function equals class method."""
        X, resid, time_ids = panel_data

        # Using convenience function
        result1 = driscoll_kraay(X, resid, time_ids, max_lags=3)

        # Using class method
        dk = DriscollKraayStandardErrors(X, resid, time_ids, max_lags=3)
        result2 = dk.compute()

        assert np.allclose(result1.std_errors, result2.std_errors)
        assert np.allclose(result1.cov_matrix, result2.cov_matrix)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_parameter(self):
        """Test with single parameter (k=1)."""
        n_periods = 20
        time_ids = np.tile(np.arange(n_periods), 10)
        X = np.random.randn(200, 1)
        resid = np.random.randn(200)

        result = driscoll_kraay(X, resid, time_ids)

        assert result.cov_matrix.shape == (1, 1)
        assert result.std_errors.shape == (1,)

    def test_many_parameters(self):
        """Test with many parameters."""
        n_periods = 30
        n_params = 8
        time_ids = np.tile(np.arange(n_periods), 10)
        X = np.random.randn(300, n_params)
        resid = np.random.randn(300)

        result = driscoll_kraay(X, resid, time_ids)

        assert result.cov_matrix.shape == (n_params, n_params)
        assert result.std_errors.shape == (n_params,)

    def test_unbalanced_panel(self):
        """Test with unbalanced panel (varying obs per period)."""
        # Create unbalanced panel
        time_ids = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3])
        X = np.random.randn(12, 2)
        resid = np.random.randn(12)

        dk = DriscollKraayStandardErrors(X, resid, time_ids)
        result = dk.compute()

        # Should still work
        assert isinstance(result, DriscollKraayResult)
        assert np.all(result.std_errors > 0)

    def test_string_time_ids(self):
        """Test with string time IDs."""
        time_ids = np.tile(["2020", "2021", "2022", "2023", "2024"], 10)
        X = np.random.randn(50, 2)
        resid = np.random.randn(50)

        dk = DriscollKraayStandardErrors(X, resid, time_ids)
        result = dk.compute()

        assert isinstance(result, DriscollKraayResult)
        assert result.n_periods == 5
