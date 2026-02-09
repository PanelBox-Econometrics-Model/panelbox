"""
Tests for Newey-West HAC Standard Errors.
"""

import numpy as np
import pytest

from panelbox.standard_errors.newey_west import NeweyWestResult, NeweyWestStandardErrors, newey_west


@pytest.fixture
def time_series_data():
    """Create time-series data for testing."""
    # 100 observations (good for N-W which needs reasonable sample size)
    n_obs = 100
    n_params = 3

    np.random.seed(42)

    # Create design matrix
    X = np.random.randn(n_obs, n_params)

    # Create residuals with some autocorrelation
    resid = np.random.randn(n_obs) * 0.5
    # Add AR(1) structure
    for t in range(1, n_obs):
        resid[t] += 0.3 * resid[t - 1]

    return X, resid


@pytest.fixture
def small_sample_data():
    """Create smaller sample data."""
    n_obs = 30
    n_params = 2

    np.random.seed(123)
    X = np.random.randn(n_obs, n_params)
    resid = np.random.randn(n_obs)

    return X, resid


class TestNeweyWestResult:
    """Test NeweyWestResult dataclass."""

    def test_result_creation(self):
        """Test creating NeweyWestResult."""
        cov_matrix = np.array([[1.0, 0.1, 0.05], [0.1, 2.0, 0.15], [0.05, 0.15, 1.5]])
        std_errors = np.array([1.0, 1.414, 1.225])

        result = NeweyWestResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            max_lags=4,
            kernel="bartlett",
            n_obs=100,
            n_params=3,
            prewhitening=False,
        )

        assert result.n_obs == 100
        assert result.n_params == 3
        assert result.max_lags == 4
        assert result.kernel == "bartlett"
        assert not result.prewhitening
        assert result.cov_matrix.shape == (3, 3)
        assert result.std_errors.shape == (3,)


class TestNeweyWestStandardErrors:
    """Test NeweyWestStandardErrors class."""

    def test_initialization(self, time_series_data):
        """Test Newey-West initialization."""
        X, resid = time_series_data

        nw = NeweyWestStandardErrors(X, resid)

        assert nw.n_obs == 100
        assert nw.n_params == 3
        assert nw.kernel == "bartlett"  # default
        assert not nw.prewhitening  # default
        assert nw.X.shape == (100, 3)
        assert nw.resid.shape == (100,)

    def test_automatic_lag_selection(self, time_series_data):
        """Test automatic lag selection using Newey-West rule."""
        X, resid = time_series_data

        nw = NeweyWestStandardErrors(X, resid)

        # For T=100: floor(4*(100/100)^(2/9)) = floor(4*1.0) = 4
        expected_lags = int(np.floor(4 * (100 / 100) ** (2 / 9)))
        assert nw.max_lags == expected_lags

    def test_manual_lag_specification(self, time_series_data):
        """Test manual lag specification."""
        X, resid = time_series_data

        nw = NeweyWestStandardErrors(X, resid, max_lags=6)

        assert nw.max_lags == 6

    def test_max_lags_capped_at_n_minus_1(self):
        """Test that max_lags is capped at n-1."""
        n_obs = 20
        X = np.random.randn(n_obs, 2)
        resid = np.random.randn(n_obs)

        # Try to set max_lags > n
        nw = NeweyWestStandardErrors(X, resid, max_lags=25)

        # Should be capped at n-1
        assert nw.max_lags == n_obs - 1

    def test_bartlett_kernel_weights(self, time_series_data):
        """Test Bartlett kernel weights."""
        X, resid = time_series_data

        nw = NeweyWestStandardErrors(X, resid, max_lags=4, kernel="bartlett")

        # Bartlett: w(l) = 1 - l/(L+1)
        assert nw._kernel_weight(0) == pytest.approx(1.0)
        assert nw._kernel_weight(1) == pytest.approx(1 - 1 / 5)  # 0.8
        assert nw._kernel_weight(2) == pytest.approx(1 - 2 / 5)  # 0.6
        assert nw._kernel_weight(4) == pytest.approx(1 - 4 / 5)  # 0.2
        assert nw._kernel_weight(5) == 0.0  # Beyond max_lags

    def test_parzen_kernel_weights(self, time_series_data):
        """Test Parzen kernel weights."""
        X, resid = time_series_data

        nw = NeweyWestStandardErrors(X, resid, max_lags=6, kernel="parzen")

        # For lag=0, z=0, weight = 1 - 0 + 0 = 1
        assert nw._kernel_weight(0) == pytest.approx(1.0)

        # Beyond max_lags should be 0
        assert nw._kernel_weight(10) == 0.0

    def test_quadratic_spectral_kernel_weights(self, time_series_data):
        """Test Quadratic Spectral kernel weights."""
        X, resid = time_series_data

        nw = NeweyWestStandardErrors(X, resid, max_lags=4, kernel="quadratic_spectral")

        # For lag=0, should be 1.0
        assert nw._kernel_weight(0) == pytest.approx(1.0)

        # For lag > max_lags, should be 0
        assert nw._kernel_weight(10) == 0.0

    def test_invalid_kernel_raises_error(self, time_series_data):
        """Test that invalid kernel raises error."""
        X, resid = time_series_data

        nw = NeweyWestStandardErrors(X, resid, kernel="bartlett")
        nw.kernel = "invalid_kernel"

        with pytest.raises(ValueError, match="Unknown kernel"):
            nw._kernel_weight(1)

    def test_bread_matrix_caching(self, time_series_data):
        """Test that bread matrix is cached."""
        X, resid = time_series_data

        nw = NeweyWestStandardErrors(X, resid)

        # First access computes
        bread1 = nw.bread
        # Second access uses cache
        bread2 = nw.bread

        assert bread1 is bread2  # Same object
        assert bread1.shape == (3, 3)

    def test_compute_gamma_lag0(self, time_series_data):
        """Test computing lag-0 autocovariance (heteroskedasticity component)."""
        X, resid = time_series_data

        nw = NeweyWestStandardErrors(X, resid)
        gamma_0 = nw._compute_gamma(0)

        assert gamma_0.shape == (3, 3)
        # Should be symmetric
        assert np.allclose(gamma_0, gamma_0.T)

    def test_compute_gamma_lag_positive(self, time_series_data):
        """Test computing positive lag autocovariance."""
        X, resid = time_series_data

        nw = NeweyWestStandardErrors(X, resid)
        gamma_1 = nw._compute_gamma(1)
        gamma_2 = nw._compute_gamma(2)

        assert gamma_1.shape == (3, 3)
        assert gamma_2.shape == (3, 3)

    def test_compute_basic(self, time_series_data):
        """Test basic Newey-West computation."""
        X, resid = time_series_data

        nw = NeweyWestStandardErrors(X, resid)
        result = nw.compute()

        assert isinstance(result, NeweyWestResult)
        assert result.cov_matrix.shape == (3, 3)
        assert result.std_errors.shape == (3,)
        assert np.all(result.std_errors > 0)
        assert result.n_obs == 100
        assert not result.prewhitening

    def test_compute_with_different_kernels(self, small_sample_data):
        """Test computation with different kernel types."""
        X, resid = small_sample_data

        # Bartlett
        nw_bartlett = NeweyWestStandardErrors(X, resid, kernel="bartlett")
        result_bartlett = nw_bartlett.compute()

        # Parzen
        nw_parzen = NeweyWestStandardErrors(X, resid, kernel="parzen")
        result_parzen = nw_parzen.compute()

        # Quadratic Spectral
        nw_qs = NeweyWestStandardErrors(X, resid, kernel="quadratic_spectral")
        result_qs = nw_qs.compute()

        # All should produce valid results
        assert result_bartlett.kernel == "bartlett"
        assert result_parzen.kernel == "parzen"
        assert result_qs.kernel == "quadratic_spectral"

        # All should have positive standard errors
        assert np.all(result_bartlett.std_errors > 0)
        assert np.all(result_parzen.std_errors > 0)
        assert np.all(result_qs.std_errors > 0)

    def test_covariance_matrix_symmetry(self, time_series_data):
        """Test that covariance matrix is symmetric."""
        X, resid = time_series_data

        nw = NeweyWestStandardErrors(X, resid)
        result = nw.compute()

        assert np.allclose(result.cov_matrix, result.cov_matrix.T)

    def test_std_errors_from_diagonal(self, time_series_data):
        """Test that standard errors are sqrt of diagonal."""
        X, resid = time_series_data

        nw = NeweyWestStandardErrors(X, resid)
        result = nw.compute()

        expected_se = np.sqrt(np.diag(result.cov_matrix))
        assert np.allclose(result.std_errors, expected_se)

    def test_different_lag_specifications(self, time_series_data):
        """Test with different lag specifications."""
        X, resid = time_series_data

        # Zero lags (only heteroskedasticity, no autocorrelation)
        nw0 = NeweyWestStandardErrors(X, resid, max_lags=0)
        result0 = nw0.compute()
        assert result0.max_lags == 0

        # Few lags
        nw2 = NeweyWestStandardErrors(X, resid, max_lags=2)
        result2 = nw2.compute()
        assert result2.max_lags == 2

        # Many lags
        nw10 = NeweyWestStandardErrors(X, resid, max_lags=10)
        result10 = nw10.compute()
        assert result10.max_lags == 10

        # All should produce valid results
        assert np.all(result0.std_errors > 0)
        assert np.all(result2.std_errors > 0)
        assert np.all(result10.std_errors > 0)

    def test_diagnostic_summary(self, time_series_data):
        """Test diagnostic summary generation."""
        X, resid = time_series_data

        nw = NeweyWestStandardErrors(X, resid)
        summary = nw.diagnostic_summary()

        assert "Newey-West HAC Standard Errors Diagnostics" in summary
        assert "Number of observations: 100" in summary
        assert "Number of parameters: 3" in summary
        assert "Maximum lags:" in summary
        assert "Kernel function: bartlett" in summary
        assert "Prewhitening: False" in summary

    def test_diagnostic_warning_small_sample(self):
        """Test diagnostic warning for small sample size."""
        n_obs = 40  # < 50
        X = np.random.randn(n_obs, 2)
        resid = np.random.randn(n_obs)

        nw = NeweyWestStandardErrors(X, resid)
        summary = nw.diagnostic_summary()

        assert "WARNING: Small sample size (<50)" in summary

    def test_diagnostic_warning_large_lags(self):
        """Test diagnostic warning for large max_lags."""
        n_obs = 30
        X = np.random.randn(n_obs, 2)
        resid = np.random.randn(n_obs)

        # Set max_lags > n/3
        nw = NeweyWestStandardErrors(X, resid, max_lags=12)
        summary = nw.diagnostic_summary()

        assert "WARNING: Large max_lags relative to sample size" in summary

    def test_prewhitening_flag(self, time_series_data):
        """Test prewhitening flag is stored correctly."""
        X, resid = time_series_data

        nw = NeweyWestStandardErrors(X, resid, prewhitening=True)
        result = nw.compute()

        assert result.prewhitening


class TestConvenienceFunction:
    """Test the newey_west() convenience function."""

    def test_convenience_function(self, time_series_data):
        """Test newey_west() convenience function."""
        X, resid = time_series_data

        result = newey_west(X, resid)

        assert isinstance(result, NeweyWestResult)
        assert result.cov_matrix.shape == (3, 3)
        assert result.std_errors.shape == (3,)

    def test_convenience_with_parameters(self, time_series_data):
        """Test convenience function with parameters."""
        X, resid = time_series_data

        result = newey_west(X, resid, max_lags=5, kernel="parzen", prewhitening=True)

        assert result.max_lags == 5
        assert result.kernel == "parzen"
        assert result.prewhitening

    def test_convenience_equals_class_method(self, time_series_data):
        """Test that convenience function equals class method."""
        X, resid = time_series_data

        # Using convenience function
        result1 = newey_west(X, resid, max_lags=3)

        # Using class method
        nw = NeweyWestStandardErrors(X, resid, max_lags=3)
        result2 = nw.compute()

        assert np.allclose(result1.std_errors, result2.std_errors)
        assert np.allclose(result1.cov_matrix, result2.cov_matrix)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_parameter(self):
        """Test with single parameter (k=1)."""
        n_obs = 100
        X = np.random.randn(n_obs, 1)
        resid = np.random.randn(n_obs)

        result = newey_west(X, resid)

        assert result.cov_matrix.shape == (1, 1)
        assert result.std_errors.shape == (1,)

    def test_many_parameters(self):
        """Test with many parameters."""
        n_obs = 150
        n_params = 8
        X = np.random.randn(n_obs, n_params)
        resid = np.random.randn(n_obs)

        result = newey_west(X, resid)

        assert result.cov_matrix.shape == (n_params, n_params)
        assert result.std_errors.shape == (n_params,)

    def test_no_autocorrelation(self):
        """Test with iid residuals (no autocorrelation)."""
        n_obs = 100
        X = np.random.randn(n_obs, 2)
        resid = np.random.randn(n_obs)  # Pure white noise

        # With zero lags, should be similar to White's robust SE
        nw = NeweyWestStandardErrors(X, resid, max_lags=0)
        result = nw.compute()

        assert isinstance(result, NeweyWestResult)
        assert np.all(result.std_errors > 0)

    def test_strong_autocorrelation(self):
        """Test with strongly autocorrelated residuals."""
        n_obs = 100
        X = np.random.randn(n_obs, 2)

        # Create highly autocorrelated residuals
        resid = np.zeros(n_obs)
        resid[0] = np.random.randn()
        for t in range(1, n_obs):
            resid[t] = 0.8 * resid[t - 1] + 0.2 * np.random.randn()

        result = newey_west(X, resid, max_lags=10)

        assert isinstance(result, NeweyWestResult)
        assert np.all(result.std_errors > 0)

    def test_small_residuals(self):
        """Test with very small residuals."""
        n_obs = 100
        X = np.random.randn(n_obs, 2)
        resid = np.random.randn(n_obs) * 1e-10

        result = newey_west(X, resid)

        # Should still work with very small residuals
        assert np.all(result.std_errors > 0)
        assert np.all(result.std_errors < 1e-5)  # Should be small
