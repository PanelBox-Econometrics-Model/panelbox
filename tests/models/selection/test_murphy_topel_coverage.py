"""
Tests for panelbox.models.selection.murphy_topel module.

Targets: murphy_topel_variance, compute_cross_derivative_heckman,
         heckman_two_step_variance, bootstrap_two_step_variance.
Goal: 13.3% -> 60%+ coverage.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# murphy_topel_variance
# ---------------------------------------------------------------------------


class TestMurphyTopelVariance:
    """Tests for the murphy_topel_variance function."""

    def test_basic_correction(self):
        """Compute Murphy-Topel corrected variance with valid inputs."""
        from panelbox.models.selection.murphy_topel import murphy_topel_variance

        rng = np.random.RandomState(42)
        k1, k2 = 3, 5

        vcov_step1 = np.eye(k1) * 0.01
        vcov_step2 = np.eye(k2) * 0.04
        cross_deriv = rng.randn(k2, k1) * 0.1

        result = murphy_topel_variance(vcov_step1, vcov_step2, cross_deriv)

        assert result.shape == (k2, k2)
        # Corrected variance should be >= uncorrected (adds positive semidefinite term)
        # Check symmetry
        np.testing.assert_allclose(result, result.T)

    def test_zero_cross_derivative(self):
        """With zero cross-derivative, corrected == uncorrected."""
        from panelbox.models.selection.murphy_topel import murphy_topel_variance

        k1, k2 = 3, 4
        vcov_step1 = np.eye(k1) * 0.01
        vcov_step2 = np.eye(k2) * 0.04
        cross_deriv = np.zeros((k2, k1))

        result = murphy_topel_variance(vcov_step1, vcov_step2, cross_deriv)
        np.testing.assert_allclose(result, vcov_step2)

    def test_invalid_vcov_1d_raises(self):
        """Raise ValueError for 1D covariance matrices."""
        from panelbox.models.selection.murphy_topel import murphy_topel_variance

        with pytest.raises(ValueError, match="2D arrays"):
            murphy_topel_variance(
                np.array([1.0, 2.0]),
                np.eye(3),
                np.ones((3, 2)),
            )

    def test_invalid_vcov2_1d_raises(self):
        """Raise ValueError for 1D vcov_step2."""
        from panelbox.models.selection.murphy_topel import murphy_topel_variance

        with pytest.raises(ValueError, match="2D arrays"):
            murphy_topel_variance(
                np.eye(2),
                np.array([1.0, 2.0, 3.0]),
                np.ones((3, 2)),
            )

    def test_invalid_cross_deriv_1d_raises(self):
        """Raise ValueError for 1D cross-derivative."""
        from panelbox.models.selection.murphy_topel import murphy_topel_variance

        with pytest.raises(ValueError, match="2D array"):
            murphy_topel_variance(
                np.eye(2),
                np.eye(3),
                np.array([1.0, 2.0]),
            )

    def test_shape_mismatch_raises(self):
        """Raise ValueError for incompatible cross-derivative shape."""
        from panelbox.models.selection.murphy_topel import murphy_topel_variance

        with pytest.raises(ValueError, match="incompatible"):
            murphy_topel_variance(
                np.eye(3),
                np.eye(4),
                np.ones((3, 4)),  # Should be (4, 3)
            )

    def test_symmetry_enforcement(self):
        """Correction enforces symmetry even with small numerical errors."""
        from panelbox.models.selection.murphy_topel import murphy_topel_variance

        rng = np.random.RandomState(10)
        k1, k2 = 2, 3
        vcov_step1 = np.eye(k1) * 0.01
        vcov_step2 = np.eye(k2) * 0.04
        cross_deriv = rng.randn(k2, k1) * 0.1

        result = murphy_topel_variance(vcov_step1, vcov_step2, cross_deriv)
        np.testing.assert_allclose(result, result.T, atol=1e-15)


# ---------------------------------------------------------------------------
# compute_cross_derivative_heckman
# ---------------------------------------------------------------------------


class TestComputeCrossDerivativeHeckman:
    """Tests for the compute_cross_derivative_heckman function."""

    def test_basic_cross_derivative(self):
        """Compute cross-derivative with valid Heckman inputs."""
        from panelbox.models.selection.murphy_topel import (
            compute_cross_derivative_heckman,
        )

        rng = np.random.RandomState(42)
        n = 20
        k_outcome = 3
        k_selection = 4

        X = rng.randn(n, k_outcome)
        W = rng.randn(n, k_selection)
        imr = rng.rand(n)
        imr_deriv = -rng.rand(n)  # Typically negative
        beta = rng.randn(k_outcome)
        theta = 0.5
        selected = np.ones(n)

        result = compute_cross_derivative_heckman(X, W, imr, imr_deriv, beta, theta, selected)

        assert result.shape == (k_outcome + 1, k_selection)
        assert np.all(np.isfinite(result))

    def test_zero_theta(self):
        """With theta=0, cross-derivative should be zero."""
        from panelbox.models.selection.murphy_topel import (
            compute_cross_derivative_heckman,
        )

        n = 10
        k_outcome = 2
        k_selection = 3
        X = np.ones((n, k_outcome))
        W = np.ones((n, k_selection))
        imr = np.ones(n)
        imr_deriv = np.ones(n)
        beta = np.ones(k_outcome)
        theta = 0.0
        selected = np.ones(n)

        result = compute_cross_derivative_heckman(X, W, imr, imr_deriv, beta, theta, selected)
        np.testing.assert_allclose(result, 0.0)


# ---------------------------------------------------------------------------
# heckman_two_step_variance
# ---------------------------------------------------------------------------


class TestHeckmanTwoStepVariance:
    """Tests for the heckman_two_step_variance convenience function."""

    def test_full_pipeline(self):
        """Run the full Heckman two-step variance correction."""
        from panelbox.models.selection.murphy_topel import (
            heckman_two_step_variance,
        )

        rng = np.random.RandomState(42)
        n = 100
        k_outcome = 2
        k_selection = 3

        # Create synthetic data
        X = np.column_stack([np.ones(n), rng.randn(n)])
        W = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])

        gamma = np.array([0.5, 0.3, 0.4])
        beta = np.array([1.0, 0.5])
        theta = 0.3
        sigma = 1.0

        # Generate selection
        linear_pred = W @ gamma
        selection = (linear_pred + rng.randn(n) > 0).astype(float)

        # Generate outcome
        y = X @ beta + rng.randn(n) * sigma
        y[selection == 0] = np.nan

        # Probit variance (mock, from first step)
        vcov_probit = np.eye(k_selection) * 0.01

        vcov_corrected, se_corrected = heckman_two_step_variance(
            X, W, y, beta, gamma, theta, sigma, selection, vcov_probit
        )

        assert vcov_corrected.shape == (k_outcome + 1, k_outcome + 1)
        assert len(se_corrected) == k_outcome + 1
        assert np.all(se_corrected > 0)
        # Check symmetry
        np.testing.assert_allclose(vcov_corrected, vcov_corrected.T)


# ---------------------------------------------------------------------------
# bootstrap_two_step_variance
# ---------------------------------------------------------------------------


class TestBootstrapTwoStepVariance:
    """Tests for the bootstrap_two_step_variance function."""

    def test_not_implemented(self):
        """Bootstrap should raise NotImplementedError."""
        from panelbox.models.selection.murphy_topel import (
            bootstrap_two_step_variance,
        )

        def dummy_estimator(data):
            return np.array([1.0, 2.0])

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            bootstrap_two_step_variance(dummy_estimator, (None,))

    def test_not_implemented_with_seed(self):
        """Bootstrap with seed still raises NotImplementedError."""
        from panelbox.models.selection.murphy_topel import (
            bootstrap_two_step_variance,
        )

        def dummy_estimator(data):
            return np.array([1.0, 2.0])

        with pytest.raises(NotImplementedError):
            bootstrap_two_step_variance(dummy_estimator, (None,), seed=42)
