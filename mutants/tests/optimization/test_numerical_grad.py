"""
Unit tests for numerical gradient and Hessian computation.

Tests cover:
- Forward and central difference methods
- Automatic step size selection
- Comparison with analytical derivatives
- Hessian symmetry
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from panelbox.optimization.numerical_grad import approx_gradient, approx_hessian


class TestNumericalGradient:
    """Tests for numerical gradient computation."""

    def test_gradient_quadratic(self):
        """Test gradient of quadratic function."""
        # f(x) = x'Ax + b'x
        A = np.array([[2, 1], [1, 3]])
        b = np.array([1, -1])

        def f(x):
            return x @ A @ x + b @ x

        # Test point
        x = np.array([1.0, 2.0])

        # Analytical gradient: 2*A*x + b
        grad_true = 2 * A @ x + b

        # Numerical gradients
        grad_central = approx_gradient(f, x, method="central")
        grad_forward = approx_gradient(f, x, method="forward")

        # Central should be more accurate
        assert_allclose(grad_central, grad_true, atol=1e-8)
        assert_allclose(grad_forward, grad_true, atol=1e-5)

    def test_gradient_exponential(self):
        """Test gradient of exponential function."""

        def f(x):
            return np.exp(x[0]) + x[1] ** 2

        x = np.array([0.5, 1.0])

        # Analytical gradient: [exp(x0), 2*x1]
        grad_true = np.array([np.exp(x[0]), 2 * x[1]])

        # Numerical gradient
        grad_num = approx_gradient(f, x, method="central")

        assert_allclose(grad_num, grad_true, rtol=1e-7)

    def test_gradient_trigonometric(self):
        """Test gradient of trigonometric function."""

        def f(x):
            return np.sin(x[0]) * np.cos(x[1])

        x = np.array([np.pi / 4, np.pi / 3])

        # Analytical gradient
        grad_true = np.array([np.cos(x[0]) * np.cos(x[1]), -np.sin(x[0]) * np.sin(x[1])])

        # Numerical gradient
        grad_num = approx_gradient(f, x, method="central")

        assert_allclose(grad_num, grad_true, rtol=1e-7)

    def test_gradient_auto_epsilon(self):
        """Test automatic step size selection."""

        def f(x):
            return x[0] ** 2 + x[1] ** 2

        # Test at different scales
        x_small = np.array([1e-5, 1e-5])
        x_large = np.array([1e5, 1e5])

        grad_small = approx_gradient(f, x_small, epsilon="auto")
        grad_large = approx_gradient(f, x_large, epsilon="auto")

        # Should still get accurate gradients
        assert_allclose(grad_small, 2 * x_small, rtol=1e-5)
        assert_allclose(grad_large, 2 * x_large, rtol=1e-5)

    def test_gradient_manual_epsilon(self):
        """Test manual step size specification."""

        def f(x):
            return x[0] ** 3

        x = np.array([1.0])

        # Analytical gradient: 3*x^2
        grad_true = np.array([3 * x[0] ** 2])

        # Try different step sizes
        epsilons = [1e-3, 1e-5, 1e-7]
        errors = []

        for eps in epsilons:
            grad_num = approx_gradient(f, x, epsilon=eps)
            error = np.abs(grad_num - grad_true)[0]
            errors.append(error)

        # Smaller epsilon should give better accuracy (up to rounding error)
        assert errors[1] < errors[0]
        assert errors[2] < errors[0]

    def test_gradient_dimension(self):
        """Test gradient computation for different dimensions."""
        for n_dim in [1, 5, 10, 20]:
            # Simple quadratic
            def f(x):
                return np.sum(x**2)

            x = np.random.randn(n_dim)
            grad_true = 2 * x
            grad_num = approx_gradient(f, x)

            assert grad_num.shape == (n_dim,)
            assert_allclose(grad_num, grad_true, rtol=1e-7)


class TestNumericalHessian:
    """Tests for numerical Hessian computation."""

    def test_hessian_quadratic(self):
        """Test Hessian of quadratic function."""
        # f(x) = x'Ax
        A = np.array([[3, 1], [1, 2]])

        def f(x):
            return x @ A @ x

        x = np.array([1.0, 1.0])

        # Analytical Hessian: 2*A
        hess_true = 2 * A

        # Numerical Hessian
        hess_num = approx_hessian(f, x)

        assert_allclose(hess_num, hess_true, atol=1e-6)

    def test_hessian_symmetry(self):
        """Test that numerical Hessian is symmetric."""

        def f(x):
            return x[0] ** 3 + x[1] ** 2 + x[0] * x[1]

        x = np.array([0.5, 0.5])

        hess_num = approx_hessian(f, x)

        # Check symmetry
        assert_allclose(hess_num, hess_num.T, atol=1e-10)

    def test_hessian_exponential(self):
        """Test Hessian of exponential function."""

        def f(x):
            return np.exp(x[0] + x[1])

        x = np.array([0.5, 0.5])

        # Analytical Hessian
        val = np.exp(x[0] + x[1])
        hess_true = np.array([[val, val], [val, val]])

        # Numerical Hessian
        hess_num = approx_hessian(f, x)

        assert_allclose(hess_num, hess_true, rtol=1e-6)

    def test_hessian_mixed_derivatives(self):
        """Test Hessian with mixed partial derivatives."""

        def f(x):
            return x[0] ** 2 * x[1] + x[0] * x[1] ** 2

        x = np.array([1.0, 2.0])

        # Analytical Hessian
        # f_xx = 2*y, f_yy = 2*x, f_xy = 2*x + 2*y
        hess_true = np.array([[2 * x[1], 2 * x[0] + 2 * x[1]], [2 * x[0] + 2 * x[1], 2 * x[0]]])

        # Numerical Hessian
        hess_num = approx_hessian(f, x)

        assert_allclose(hess_num, hess_true, rtol=1e-5)

    def test_hessian_dimension(self):
        """Test Hessian computation for different dimensions."""
        for n_dim in [2, 3, 5]:
            # Simple quadratic with identity Hessian
            def f(x):
                return np.sum(x**2)

            x = np.random.randn(n_dim)

            # Analytical Hessian: 2*I
            hess_true = 2 * np.eye(n_dim)

            # Numerical Hessian
            hess_num = approx_hessian(f, x)

            assert hess_num.shape == (n_dim, n_dim)
            assert_allclose(hess_num, hess_true, atol=1e-6)

    def test_hessian_singular(self):
        """Test Hessian computation for function with singular Hessian."""

        def f(x):
            # Function with zero curvature in one direction
            return (x[0] + x[1]) ** 2

        x = np.array([1.0, 1.0])

        # Analytical Hessian (rank 1)
        hess_true = np.array([[2, 2], [2, 2]])

        # Numerical Hessian
        hess_num = approx_hessian(f, x)

        assert_allclose(hess_num, hess_true, atol=1e-6)

        # Check that it's singular (low rank)
        eigenvals = np.linalg.eigvalsh(hess_num)
        assert np.abs(eigenvals[0]) < 1e-6  # One zero eigenvalue


class TestComparison:
    """Test comparison with other numerical differentiation libraries."""

    @pytest.mark.skipif(
        not pytest.importorskip("numdifftools"), reason="numdifftools not installed"
    )
    def test_compare_with_numdifftools(self):
        """Compare with numdifftools package."""
        import numdifftools as nd

        def f(x):
            return x[0] ** 3 + np.sin(x[1]) + x[0] * x[1] ** 2

        x = np.array([0.5, 1.0])

        # Our implementation
        grad_ours = approx_gradient(f, x, method="central")
        hess_ours = approx_hessian(f, x)

        # numdifftools
        grad_nd = nd.Gradient(f)(x)
        hess_nd = nd.Hessian(f)(x)

        # Should be very close
        assert_allclose(grad_ours, grad_nd, rtol=1e-6)
        assert_allclose(hess_ours, hess_nd, rtol=1e-5)


class TestLogitGradients:
    """Test numerical gradients for Logit model."""

    def test_logit_gradient_vs_analytic(self):
        """Compare numerical vs analytical gradient for Logit."""
        np.random.seed(42)
        n = 100
        k = 3

        # Generate data
        X = np.random.randn(n, k)
        beta_true = np.array([0.5, -0.3, 0.2])
        z = X @ beta_true
        p = 1 / (1 + np.exp(-z))
        y = np.random.binomial(1, p)

        # Log-likelihood function
        def log_likelihood(beta):
            z = X @ beta
            p = 1 / (1 + np.exp(-z))
            # Avoid log(0)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        # Analytical gradient (score)
        def analytical_gradient(beta):
            z = X @ beta
            p = 1 / (1 + np.exp(-z))
            return X.T @ (y - p)

        # Test point
        beta_test = np.array([0.3, -0.2, 0.1])

        # Compare
        grad_analytical = analytical_gradient(beta_test)
        grad_numerical = approx_gradient(log_likelihood, beta_test, method="central")

        # Should be very close
        assert_allclose(grad_numerical, grad_analytical, rtol=1e-5)

    def test_logit_hessian_vs_analytic(self):
        """Compare numerical vs analytical Hessian for Logit."""
        np.random.seed(42)
        n = 50
        k = 2

        # Generate data
        X = np.random.randn(n, k)
        beta_true = np.array([0.5, -0.3])
        z = X @ beta_true
        p = 1 / (1 + np.exp(-z))
        y = np.random.binomial(1, p)

        # Log-likelihood function
        def log_likelihood(beta):
            z = X @ beta
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        # Analytical Hessian
        def analytical_hessian(beta):
            z = X @ beta
            p = 1 / (1 + np.exp(-z))
            W = np.diag(p * (1 - p))
            return -X.T @ W @ X

        # Test point
        beta_test = np.array([0.2, -0.1])

        # Compare
        hess_analytical = analytical_hessian(beta_test)
        hess_numerical = approx_hessian(log_likelihood, beta_test)

        # Should be close (Hessian is harder to approximate)
        assert_allclose(hess_numerical, hess_analytical, rtol=1e-3)


class TestKnownFunctions:
    """Test gradients and Hessians for known mathematical functions."""

    def test_polynomial_derivatives(self):
        """Test derivatives of polynomial functions."""

        # f(x, y) = x^3 + 2*x^2*y + y^3
        def f(x):
            return x[0] ** 3 + 2 * x[0] ** 2 * x[1] + x[1] ** 3

        # Gradient: [3*x^2 + 4*x*y, 2*x^2 + 3*y^2]
        def true_gradient(x):
            return np.array([3 * x[0] ** 2 + 4 * x[0] * x[1], 2 * x[0] ** 2 + 3 * x[1] ** 2])

        # Hessian: [[6*x + 4*y, 4*x], [4*x, 6*y]]
        def true_hessian(x):
            return np.array([[6 * x[0] + 4 * x[1], 4 * x[0]], [4 * x[0], 6 * x[1]]])

        # Test at multiple points
        test_points = [np.array([1.0, 2.0]), np.array([-0.5, 0.5]), np.array([2.0, -1.0])]

        for x in test_points:
            grad_num = approx_gradient(f, x, method="central")
            grad_true = true_gradient(x)
            assert_allclose(grad_num, grad_true, rtol=1e-7)

            hess_num = approx_hessian(f, x)
            hess_true = true_hessian(x)
            assert_allclose(hess_num, hess_true, rtol=1e-5)

    def test_logarithmic_derivatives(self):
        """Test derivatives of logarithmic functions."""

        # f(x, y) = log(x^2 + y^2 + 1)
        def f(x):
            return np.log(x[0] ** 2 + x[1] ** 2 + 1)

        # Gradient: [2*x/(x^2 + y^2 + 1), 2*y/(x^2 + y^2 + 1)]
        def true_gradient(x):
            denom = x[0] ** 2 + x[1] ** 2 + 1
            return np.array([2 * x[0] / denom, 2 * x[1] / denom])

        # Test points
        test_points = [np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([2.0, -1.5])]

        for x in test_points:
            grad_num = approx_gradient(f, x, method="central")
            grad_true = true_gradient(x)
            assert_allclose(grad_num, grad_true, rtol=1e-7)

    def test_rosenbrock_function(self):
        """Test derivatives of Rosenbrock function (common optimization test)."""

        # f(x, y) = (a - x)^2 + b*(y - x^2)^2 with a=1, b=100
        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        # Gradient
        def rosenbrock_gradient(x):
            return np.array(
                [-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)]
            )

        # Test points including minimum
        test_points = [np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([0.5, 0.5])]  # Minimum

        for x in test_points:
            grad_num = approx_gradient(rosenbrock, x, method="central")
            grad_true = rosenbrock_gradient(x)
            assert_allclose(grad_num, grad_true, rtol=1e-6)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_gradient_zero_point(self):
        """Test gradient at zero."""

        def f(x):
            return x[0] ** 2 + x[1] ** 2

        x = np.array([0.0, 0.0])
        grad_num = approx_gradient(f, x)
        grad_true = np.array([0.0, 0.0])

        assert_allclose(grad_num, grad_true, atol=1e-8)

    def test_gradient_large_values(self):
        """Test gradient for large parameter values."""

        def f(x):
            return x[0] ** 2 + x[1] ** 2

        x = np.array([1e10, 1e10])
        grad_num = approx_gradient(f, x, epsilon="auto")
        grad_true = 2 * x

        # Should still work with auto epsilon
        assert_allclose(grad_num, grad_true, rtol=1e-5)

    def test_hessian_discontinuous(self):
        """Test Hessian near discontinuity (should still run)."""

        def f(x):
            # Continuous but not smooth at x=0
            return np.abs(x[0]) + x[1] ** 2

        x = np.array([0.1, 0.5])

        # Should not raise error
        hess_num = approx_hessian(f, x)

        # Check shape and finiteness
        assert hess_num.shape == (2, 2)
        assert np.all(np.isfinite(hess_num))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
