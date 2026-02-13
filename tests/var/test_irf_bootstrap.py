"""
Tests for bootstrap confidence intervals for IRFs.
"""

import numpy as np
import pytest

from panelbox.var.irf import bootstrap_irf, compute_irf_cholesky


class TestBootstrapIRF:
    """Test suite for bootstrap IRF confidence intervals."""

    def setup_method(self):
        """Set up test fixtures."""
        # Simple VAR(1) with K=2
        self.K = 2
        self.p = 1
        self.n_obs = 100
        self.periods = 10

        # Coefficient matrix (stable VAR)
        self.A1 = np.array([[0.5, 0.1], [0.2, 0.4]])
        self.A_matrices = [self.A1]

        # Residual covariance
        self.Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

        # Generate synthetic residuals
        np.random.seed(42)
        self.residuals = np.random.multivariate_normal(
            mean=np.zeros(self.K), cov=self.Sigma, size=self.n_obs
        )

    def test_bootstrap_percentile(self):
        """Test standard percentile bootstrap."""
        ci_lower, ci_upper, bootstrap_dist = bootstrap_irf(
            A_matrices=self.A_matrices,
            Sigma=self.Sigma,
            residuals=self.residuals,
            periods=self.periods,
            method="cholesky",
            n_bootstrap=100,  # Small number for speed
            ci_level=0.95,
            ci_method="percentile",
            n_jobs=1,  # Single core for reproducibility
            seed=123,
            verbose=False,
        )

        # Check shapes
        assert ci_lower.shape == (self.periods + 1, self.K, self.K)
        assert ci_upper.shape == (self.periods + 1, self.K, self.K)
        assert bootstrap_dist.shape == (100, self.periods + 1, self.K, self.K)

        # Lower < upper
        assert np.all(ci_lower <= ci_upper)

        # Original IRF should be near the center of CI
        irf_original = compute_irf_cholesky(self.A_matrices, self.Sigma, self.periods)

        # Check that original IRF is mostly within CI
        within_ci = np.logical_and(irf_original >= ci_lower, irf_original <= ci_upper)
        coverage = np.mean(within_ci)

        # Should have high coverage (though not guaranteed for small n_bootstrap)
        assert coverage > 0.7

    def test_bootstrap_bias_corrected(self):
        """Test bias-corrected bootstrap."""
        ci_lower, ci_upper, bootstrap_dist = bootstrap_irf(
            A_matrices=self.A_matrices,
            Sigma=self.Sigma,
            residuals=self.residuals,
            periods=self.periods,
            method="cholesky",
            n_bootstrap=100,
            ci_level=0.95,
            ci_method="bias_corrected",
            n_jobs=1,
            seed=123,
            verbose=False,
        )

        # Check shapes
        assert ci_lower.shape == (self.periods + 1, self.K, self.K)
        assert ci_upper.shape == (self.periods + 1, self.K, self.K)

        # Lower < upper
        assert np.all(ci_lower <= ci_upper)

    def test_bootstrap_cumulative(self):
        """Test bootstrap for cumulative IRFs."""
        ci_lower, ci_upper, bootstrap_dist = bootstrap_irf(
            A_matrices=self.A_matrices,
            Sigma=self.Sigma,
            residuals=self.residuals,
            periods=self.periods,
            method="cholesky",
            n_bootstrap=50,
            cumulative=True,
            n_jobs=1,
            seed=123,
            verbose=False,
        )

        # Check shapes
        assert ci_lower.shape == (self.periods + 1, self.K, self.K)
        assert ci_upper.shape == (self.periods + 1, self.K, self.K)

        # Lower < upper
        assert np.all(ci_lower <= ci_upper)

        # Cumulative IRF should be increasing (or at least non-decreasing in absolute value)
        # This is a weak test but checks basic sanity

    def test_bootstrap_generalized(self):
        """Test bootstrap for generalized IRFs."""
        ci_lower, ci_upper, bootstrap_dist = bootstrap_irf(
            A_matrices=self.A_matrices,
            Sigma=self.Sigma,
            residuals=self.residuals,
            periods=self.periods,
            method="generalized",
            n_bootstrap=50,
            n_jobs=1,
            seed=123,
            verbose=False,
        )

        # Check shapes
        assert ci_lower.shape == (self.periods + 1, self.K, self.K)
        assert ci_upper.shape == (self.periods + 1, self.K, self.K)

        # Lower < upper
        assert np.all(ci_lower <= ci_upper)

    def test_bootstrap_reproducibility(self):
        """Test that bootstrap is reproducible with same seed."""
        ci_lower1, ci_upper1, _ = bootstrap_irf(
            A_matrices=self.A_matrices,
            Sigma=self.Sigma,
            residuals=self.residuals,
            periods=self.periods,
            n_bootstrap=50,
            n_jobs=1,
            seed=999,
            verbose=False,
        )

        ci_lower2, ci_upper2, _ = bootstrap_irf(
            A_matrices=self.A_matrices,
            Sigma=self.Sigma,
            residuals=self.residuals,
            periods=self.periods,
            n_bootstrap=50,
            n_jobs=1,
            seed=999,
            verbose=False,
        )

        # Should be identical
        np.testing.assert_allclose(ci_lower1, ci_lower2, rtol=1e-10)
        np.testing.assert_allclose(ci_upper1, ci_upper2, rtol=1e-10)

    def test_bootstrap_ci_level(self):
        """Test different confidence levels."""
        # 90% CI
        ci_lower_90, ci_upper_90, _ = bootstrap_irf(
            A_matrices=self.A_matrices,
            Sigma=self.Sigma,
            residuals=self.residuals,
            periods=self.periods,
            n_bootstrap=50,
            ci_level=0.90,
            n_jobs=1,
            seed=123,
            verbose=False,
        )

        # 95% CI
        ci_lower_95, ci_upper_95, _ = bootstrap_irf(
            A_matrices=self.A_matrices,
            Sigma=self.Sigma,
            residuals=self.residuals,
            periods=self.periods,
            n_bootstrap=50,
            ci_level=0.95,
            n_jobs=1,
            seed=123,
            verbose=False,
        )

        # 95% CI should be wider than 90% CI
        width_90 = ci_upper_90 - ci_lower_90
        width_95 = ci_upper_95 - ci_lower_95

        assert np.all(width_95 >= width_90 - 1e-10)  # Small tolerance for numerical errors
