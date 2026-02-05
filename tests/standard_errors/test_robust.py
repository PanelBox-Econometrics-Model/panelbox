"""
Unit tests for heteroskedasticity-robust standard errors (HC0, HC1, HC2, HC3).

Tests cover:
- HC0, HC1, HC2, HC3 covariance computation
- Sandwich formula verification
- Leverage calculations
- Finite-sample corrections
- Comparison with statsmodels (when available)
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less, assert_equal

from panelbox.standard_errors import (
    RobustStandardErrors,
    RobustCovarianceResult,
    robust_covariance
)
from panelbox.standard_errors.utils import (
    compute_leverage,
    compute_bread,
    compute_meat_hc,
    sandwich_covariance
)


class TestLeverageComputation:
    """Test leverage (hat values) computation."""

    def test_leverage_range(self):
        """Leverage values should be between 0 and 1."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        leverage = compute_leverage(X)

        assert_array_less(-0.0001, leverage)  # >= 0
        assert_array_less(leverage, 1.0001)   # <= 1

    def test_leverage_sum(self):
        """Sum of leverage values should equal number of parameters."""
        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        leverage = compute_leverage(X)

        assert_allclose(np.sum(leverage), k, rtol=1e-10)

    def test_leverage_simple_case(self):
        """Test leverage for simple X = [1, x] case."""
        # For simple regression y = α + βx, leverage is known
        x = np.array([1, 2, 3, 4, 5])
        X = np.column_stack([np.ones(5), x])
        leverage = compute_leverage(X)

        # Expected leverage for this case
        x_centered = x - x.mean()
        expected = 1/5 + (x_centered**2) / np.sum(x_centered**2)

        assert_allclose(leverage, expected, rtol=1e-10)

    def test_leverage_orthogonal(self):
        """For orthogonal X, leverage should be uniform."""
        # Create orthogonal design
        n, k = 100, 3
        np.random.seed(42)
        X, _ = np.linalg.qr(np.random.randn(n, k))
        X = X * np.sqrt(n)  # Scale to X'X = I

        leverage = compute_leverage(X)

        # For orthonormal X, leverage should be k/n for all observations
        expected = k / n
        assert_allclose(leverage, expected, rtol=1e-6, atol=1e-10)


class TestBreadComputation:
    """Test bread matrix computation."""

    def test_bread_shape(self):
        """Bread should be k x k matrix."""
        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        bread = compute_bread(X)

        assert bread.shape == (k, k)

    def test_bread_symmetric(self):
        """Bread should be symmetric."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        bread = compute_bread(X)

        assert_allclose(bread, bread.T, rtol=1e-10)

    def test_bread_formula(self):
        """Verify bread = (X'X)^{-1}."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        bread = compute_bread(X)

        XTX = X.T @ X
        expected = np.linalg.inv(XTX)

        assert_allclose(bread, expected, rtol=1e-10)


class TestMeatComputation:
    """Test meat matrix computation for HC variants."""

    @pytest.fixture
    def setup_data(self):
        """Create test data."""
        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        resid = np.random.randn(n)
        return X, resid

    def test_meat_shape(self, setup_data):
        """Meat should be k x k matrix."""
        X, resid = setup_data
        k = X.shape[1]

        for method in ['HC0', 'HC1', 'HC2', 'HC3']:
            meat = compute_meat_hc(X, resid, method=method)
            assert meat.shape == (k, k)

    def test_meat_symmetric(self, setup_data):
        """Meat should be symmetric."""
        X, resid = setup_data

        for method in ['HC0', 'HC1', 'HC2', 'HC3']:
            meat = compute_meat_hc(X, resid, method=method)
            assert_allclose(meat, meat.T, rtol=1e-10)

    def test_hc0_formula(self, setup_data):
        """Verify HC0: X'ΩX where Ω = diag(ε²)."""
        X, resid = setup_data
        meat = compute_meat_hc(X, resid, method='HC0')

        # Manual computation
        omega = np.diag(resid ** 2)
        expected = X.T @ omega @ X

        assert_allclose(meat, expected, rtol=1e-10)

    def test_hc1_scaling(self, setup_data):
        """Verify HC1 = [n/(n-k)] × HC0."""
        X, resid = setup_data
        n, k = X.shape

        meat_hc0 = compute_meat_hc(X, resid, method='HC0')
        meat_hc1 = compute_meat_hc(X, resid, method='HC1')

        expected = (n / (n - k)) * meat_hc0
        assert_allclose(meat_hc1, expected, rtol=1e-10)

    def test_hc2_leverage(self, setup_data):
        """Verify HC2 uses leverage adjustment."""
        X, resid = setup_data
        leverage = compute_leverage(X)

        meat_hc2 = compute_meat_hc(X, resid, method='HC2', leverage=leverage)

        # Manual computation
        weights = (resid ** 2) / (1 - leverage)
        X_weighted = X * np.sqrt(weights)[:, np.newaxis]
        expected = X_weighted.T @ X_weighted

        assert_allclose(meat_hc2, expected, rtol=1e-10)

    def test_hc3_leverage(self, setup_data):
        """Verify HC3 uses squared leverage adjustment."""
        X, resid = setup_data
        leverage = compute_leverage(X)

        meat_hc3 = compute_meat_hc(X, resid, method='HC3', leverage=leverage)

        # Manual computation
        weights = (resid ** 2) / ((1 - leverage) ** 2)
        X_weighted = X * np.sqrt(weights)[:, np.newaxis]
        expected = X_weighted.T @ X_weighted

        assert_allclose(meat_hc3, expected, rtol=1e-10)

    def test_hc_ordering(self, setup_data):
        """Generally: HC0 ≤ HC1 ≤ HC2 ≤ HC3 (in terms of SEs)."""
        X, resid = setup_data

        # Compute covariances
        bread = compute_bread(X)
        cov_hc0 = sandwich_covariance(bread, compute_meat_hc(X, resid, 'HC0'))
        cov_hc1 = sandwich_covariance(bread, compute_meat_hc(X, resid, 'HC1'))
        cov_hc2 = sandwich_covariance(bread, compute_meat_hc(X, resid, 'HC2'))
        cov_hc3 = sandwich_covariance(bread, compute_meat_hc(X, resid, 'HC3'))

        # Extract standard errors
        se_hc0 = np.sqrt(np.diag(cov_hc0))
        se_hc1 = np.sqrt(np.diag(cov_hc1))
        se_hc2 = np.sqrt(np.diag(cov_hc2))
        se_hc3 = np.sqrt(np.diag(cov_hc3))

        # HC1 should be larger than HC0
        assert np.all(se_hc1 >= se_hc0 - 1e-10)


class TestSandwichCovariance:
    """Test sandwich covariance computation."""

    def test_sandwich_formula(self):
        """Verify V = Bread @ Meat @ Bread."""
        np.random.seed(42)
        k = 5
        bread = np.random.randn(k, k)
        bread = bread @ bread.T  # Make symmetric
        meat = np.random.randn(k, k)
        meat = meat @ meat.T  # Make symmetric

        cov = sandwich_covariance(bread, meat)
        expected = bread @ meat @ bread

        assert_allclose(cov, expected, rtol=1e-10)

    def test_sandwich_symmetric(self):
        """Sandwich should be symmetric."""
        np.random.seed(42)
        k = 5
        bread = np.random.randn(k, k)
        bread = bread @ bread.T
        meat = np.random.randn(k, k)
        meat = meat @ meat.T

        cov = sandwich_covariance(bread, meat)
        assert_allclose(cov, cov.T, rtol=1e-10)


class TestRobustStandardErrors:
    """Test RobustStandardErrors class."""

    @pytest.fixture
    def setup_regression(self):
        """Create simple regression setup."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n, 4)])
        beta = np.array([1.0, 2.0, -1.5, 0.5, 1.0])

        # Create heteroskedastic errors
        epsilon = np.random.randn(n) * (1 + 0.5 * np.abs(X[:, 1]))
        y = X @ beta + epsilon

        # OLS estimation
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta_hat

        return X, resid, beta_hat

    def test_initialization(self, setup_regression):
        """Test RobustStandardErrors initialization."""
        X, resid, _ = setup_regression
        robust = RobustStandardErrors(X, resid)

        assert robust.n_obs == 100
        assert robust.n_params == 5
        assert robust.X.shape == (100, 5)
        assert robust.resid.shape == (100,)

    def test_hc0_result(self, setup_regression):
        """Test HC0 computation."""
        X, resid, _ = setup_regression
        robust = RobustStandardErrors(X, resid)
        result = robust.hc0()

        assert isinstance(result, RobustCovarianceResult)
        assert result.method == 'HC0'
        assert result.cov_matrix.shape == (5, 5)
        assert result.std_errors.shape == (5,)
        assert result.n_obs == 100
        assert result.n_params == 5
        assert result.leverage is None

    def test_hc1_result(self, setup_regression):
        """Test HC1 computation."""
        X, resid, _ = setup_regression
        robust = RobustStandardErrors(X, resid)
        result = robust.hc1()

        assert result.method == 'HC1'
        assert result.cov_matrix.shape == (5, 5)
        assert result.std_errors.shape == (5,)

    def test_hc2_result(self, setup_regression):
        """Test HC2 computation."""
        X, resid, _ = setup_regression
        robust = RobustStandardErrors(X, resid)
        result = robust.hc2()

        assert result.method == 'HC2'
        assert result.leverage is not None
        assert result.leverage.shape == (100,)

    def test_hc3_result(self, setup_regression):
        """Test HC3 computation."""
        X, resid, _ = setup_regression
        robust = RobustStandardErrors(X, resid)
        result = robust.hc3()

        assert result.method == 'HC3'
        assert result.leverage is not None

    def test_compute_method(self, setup_regression):
        """Test compute() with method selection."""
        X, resid, _ = setup_regression
        robust = RobustStandardErrors(X, resid)

        for method in ['HC0', 'HC1', 'HC2', 'HC3']:
            result = robust.compute(method)
            assert result.method == method

    def test_compute_case_insensitive(self, setup_regression):
        """Test that method names are case-insensitive."""
        X, resid, _ = setup_regression
        robust = RobustStandardErrors(X, resid)

        result_lower = robust.compute('hc1')
        result_upper = robust.compute('HC1')

        assert_allclose(result_lower.std_errors, result_upper.std_errors)

    def test_compute_invalid_method(self, setup_regression):
        """Test that invalid method raises error."""
        X, resid, _ = setup_regression
        robust = RobustStandardErrors(X, resid)

        with pytest.raises(ValueError, match="Unknown HC method"):
            robust.compute('HC4')

    def test_leverage_caching(self, setup_regression):
        """Test that leverage is cached."""
        X, resid, _ = setup_regression
        robust = RobustStandardErrors(X, resid)

        assert robust._leverage is None

        # Access leverage
        leverage1 = robust.leverage
        assert robust._leverage is not None

        # Access again - should be same object
        leverage2 = robust.leverage
        assert leverage1 is leverage2

    def test_bread_caching(self, setup_regression):
        """Test that bread is cached."""
        X, resid, _ = setup_regression
        robust = RobustStandardErrors(X, resid)

        assert robust._bread is None

        # Access bread
        bread1 = robust.bread
        assert robust._bread is not None

        # Access again - should be same object
        bread2 = robust.bread
        assert bread1 is bread2

    def test_positive_standard_errors(self, setup_regression):
        """All standard errors should be positive."""
        X, resid, _ = setup_regression
        robust = RobustStandardErrors(X, resid)

        for method in ['HC0', 'HC1', 'HC2', 'HC3']:
            result = robust.compute(method)
            assert np.all(result.std_errors > 0)

    def test_symmetric_covariance(self, setup_regression):
        """Covariance matrices should be symmetric."""
        X, resid, _ = setup_regression
        robust = RobustStandardErrors(X, resid)

        for method in ['HC0', 'HC1', 'HC2', 'HC3']:
            result = robust.compute(method)
            assert_allclose(result.cov_matrix, result.cov_matrix.T, rtol=1e-10)


class TestRobustCovarianceFunction:
    """Test convenience function robust_covariance()."""

    def test_convenience_function(self):
        """Test that convenience function works."""
        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        resid = np.random.randn(n)

        result = robust_covariance(X, resid, method='HC1')

        assert isinstance(result, RobustCovarianceResult)
        assert result.method == 'HC1'
        assert result.cov_matrix.shape == (k, k)

    def test_convenience_matches_class(self):
        """Test that convenience function matches class method."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        resid = np.random.randn(100)

        # Using class
        robust = RobustStandardErrors(X, resid)
        result1 = robust.hc1()

        # Using function
        result2 = robust_covariance(X, resid, method='HC1')

        assert_allclose(result1.std_errors, result2.std_errors)
        assert_allclose(result1.cov_matrix, result2.cov_matrix)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_perfect_fit(self):
        """Test when residuals are zero (perfect fit)."""
        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        resid = np.zeros(n)

        robust = RobustStandardErrors(X, resid)
        result = robust.hc1()

        # With zero residuals, covariance should be zero
        assert_allclose(result.cov_matrix, 0, atol=1e-10)
        assert_allclose(result.std_errors, 0, atol=1e-10)

    def test_single_regressor(self):
        """Test with single regressor (k=1)."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 1)
        resid = np.random.randn(n)

        robust = RobustStandardErrors(X, resid)
        result = robust.hc1()

        assert result.cov_matrix.shape == (1, 1)
        assert result.std_errors.shape == (1,)

    def test_constant_only(self):
        """Test with constant only (intercept)."""
        np.random.seed(42)
        n = 100
        X = np.ones((n, 1))
        resid = np.random.randn(n)

        robust = RobustStandardErrors(X, resid)
        result = robust.hc1()

        # Standard error should be sqrt(var(resid)/n) × correction
        expected_se = np.sqrt(np.var(resid, ddof=0) / n)

        # HC1 correction: n/(n-k) where k=1
        correction = np.sqrt(n / (n - 1))
        expected_se *= correction

        assert_allclose(result.std_errors[0], expected_se, rtol=1e-6)


class TestNumericalStability:
    """Test numerical stability with ill-conditioned data."""

    def test_high_correlation(self):
        """Test with highly correlated regressors."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + 0.001 * np.random.randn(n)  # Highly correlated
        X = np.column_stack([np.ones(n), x1, x2])
        resid = np.random.randn(n)

        robust = RobustStandardErrors(X, resid)

        # Should not raise errors
        result = robust.hc1()
        assert result.cov_matrix.shape == (3, 3)
        assert np.all(np.isfinite(result.std_errors))

    def test_large_residuals(self):
        """Test with very large residuals."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        resid = 1e10 * np.random.randn(100)

        robust = RobustStandardErrors(X, resid)
        result = robust.hc1()

        assert np.all(np.isfinite(result.std_errors))
        assert np.all(result.std_errors > 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
