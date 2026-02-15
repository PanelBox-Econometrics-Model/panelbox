"""
Tests for Gauss-Hermite quadrature implementation.

Tests numerical integration over normal distributions using
Gauss-Hermite quadrature.
"""

import numpy as np
import pytest
from scipy import special, stats

from panelbox.optimization.quadrature import (
    GaussHermiteQuadrature,
    adaptive_gauss_hermite,
    gauss_hermite_2d,
    gauss_hermite_quadrature,
    integrate_normal,
    integrate_product_normal,
)


class TestGaussHermiteQuadrature:
    """Test basic Gauss-Hermite quadrature."""

    def test_quadrature_nodes_weights(self):
        """Test that nodes and weights are computed correctly."""
        # Test different numbers of points
        for n_points in [2, 5, 10, 20]:
            nodes, weights = gauss_hermite_quadrature(n_points)

            # Check dimensions
            assert len(nodes) == n_points
            assert len(weights) == n_points

            # Weights should sum to 1 (normalized for standard normal)
            assert np.allclose(np.sum(weights), 1.0, rtol=1e-10)

            # Nodes should be symmetric around 0
            if n_points % 2 == 0:
                # Even number of points - check symmetry
                mid = n_points // 2
                assert np.allclose(nodes[:mid], -nodes[n_points - 1 : mid - 1 : -1])

            # Weights should be positive
            assert np.all(weights > 0)

    def test_invalid_n_points(self):
        """Test error handling for invalid n_points."""
        with pytest.raises(ValueError):
            gauss_hermite_quadrature(1)  # Too few

        with pytest.raises(ValueError):
            gauss_hermite_quadrature(51)  # Too many

    def test_compare_scipy_roots(self):
        """Compare with scipy.special.roots_hermite."""
        n_points = 10

        nodes, weights = gauss_hermite_quadrature(n_points)

        # Get scipy's version
        scipy_nodes, scipy_weights = special.roots_hermite(n_points)

        # Our normalization is different (for standard normal)
        # scipy gives weights for exp(-x²), we want exp(-x²/2)/√(2π)
        scipy_weights_normalized = scipy_weights / np.sqrt(np.pi)

        # Should match
        assert np.allclose(nodes, scipy_nodes)
        assert np.allclose(weights, scipy_weights_normalized)


class TestIntegrateNormal:
    """Test integration over normal distributions."""

    def test_integrate_constant(self):
        """Test integrating a constant function."""
        # ∫ c * φ(x) dx = c
        c = 3.5
        result = integrate_normal(lambda x: c, n_points=10)
        assert np.allclose(result, c, rtol=1e-10)

    def test_integrate_identity(self):
        """Test integrating identity function."""
        # E[X] for N(μ, σ²) = μ
        mu, sigma = 2.0, 3.0
        result = integrate_normal(lambda x: x, n_points=12, mu=mu, sigma=sigma)
        assert np.allclose(result, mu, rtol=1e-10)

    def test_integrate_square(self):
        """Test integrating x²."""
        # E[X²] for N(μ, σ²) = μ² + σ²
        mu, sigma = 1.5, 2.0
        result = integrate_normal(lambda x: x**2, n_points=12, mu=mu, sigma=sigma)
        expected = mu**2 + sigma**2
        assert np.allclose(result, expected, rtol=1e-10)

    def test_integrate_exp(self):
        """Test integrating exponential function."""
        # E[exp(X)] for N(μ, σ²) = exp(μ + σ²/2)
        mu, sigma = 0.5, 0.8
        result = integrate_normal(np.exp, n_points=15, mu=mu, sigma=sigma)
        expected = np.exp(mu + sigma**2 / 2)
        assert np.allclose(result, expected, rtol=1e-6)

    def test_integrate_normal_cdf(self):
        """Test integrating normal CDF (probit-like)."""
        # E[Φ(a + bX)] for X ~ N(μ, σ²)
        a, b = 0.5, 1.2
        mu, sigma = 0, 1

        def integrand(x):
            return stats.norm.cdf(a + b * x)

        result = integrate_normal(integrand, n_points=20, mu=mu, sigma=sigma)

        # Analytical result: Φ((a + b*μ) / √(1 + b²σ²))
        expected = stats.norm.cdf((a + b * mu) / np.sqrt(1 + b**2 * sigma**2))
        assert np.allclose(result, expected, rtol=1e-5)

    def test_accuracy_increases_with_points(self):
        """Test that accuracy improves with more quadrature points."""

        # Complex function
        def f(x):
            return np.exp(-0.5 * x**2) * np.sin(2 * x)

        results = []
        for n_points in [5, 10, 20, 40]:
            result = integrate_normal(f, n_points=n_points)
            results.append(result)

        # Compute differences between consecutive results
        diffs = [abs(results[i + 1] - results[i]) for i in range(len(results) - 1)]

        # Differences should decrease (convergence)
        assert diffs[1] < diffs[0]  # 20 vs 10 is better than 10 vs 5
        assert diffs[2] < diffs[1]  # 40 vs 20 is better than 20 vs 10


class TestAdaptiveQuadrature:
    """Test adaptive Gauss-Hermite quadrature."""

    def test_adaptive_convergence(self):
        """Test that adaptive quadrature converges."""

        # Function that needs higher order quadrature
        def f(x):
            return x**4 * np.exp(-(x**2) / 4)

        result, n_used = adaptive_gauss_hermite(f, n_points_list=[4, 8, 12, 16], tol=1e-8)

        # Should converge before using all points
        assert n_used <= 16

        # Result should be accurate
        # For this function, can compute analytically
        # (involves Gamma function)
        assert np.isfinite(result)

    def test_adaptive_simple_function(self):
        """Test adaptive quadrature on simple function."""

        # Linear function should converge quickly
        def f(x):
            return 2 * x + 1

        mu, sigma = 1.0, 2.0
        result, n_used = adaptive_gauss_hermite(
            f, n_points_list=[2, 4, 8], mu=mu, sigma=sigma, tol=1e-10
        )

        # Should converge with minimal points
        assert n_used <= 4

        # Check result: E[2X + 1] = 2μ + 1
        expected = 2 * mu + 1
        assert np.allclose(result, expected, rtol=1e-10)

    def test_adaptive_warning(self):
        """Test warning when adaptive quadrature doesn't converge."""

        # Highly oscillatory function that won't converge easily
        def f(x):
            return np.cos(10 * x) * np.exp(-(x**2))

        with pytest.warns(UserWarning, match="did not converge"):
            result, n_used = adaptive_gauss_hermite(
                f, n_points_list=[2, 4], tol=1e-12  # Too few points  # Very tight tolerance
            )

        # Should return last attempt
        assert n_used == 4


class TestProductIntegration:
    """Test integration of products of functions."""

    def test_product_integration(self):
        """Test integrating product of functions."""
        # E[X * X²] = E[X³] for N(0, 1) = 0
        funcs = [lambda x: x, lambda x: x**2]
        result = integrate_product_normal(funcs, n_points=15)
        assert np.allclose(result, 0, atol=1e-10)

        # E[X² * X²] = E[X⁴] for N(0, 1) = 3
        funcs = [lambda x: x**2, lambda x: x**2]
        result = integrate_product_normal(funcs, n_points=15)
        assert np.allclose(result, 3, rtol=1e-8)

    def test_product_with_mean_shift(self):
        """Test product integration with non-zero mean."""
        # E[X * (X-μ)] for N(μ, σ²)
        mu, sigma = 2.0, 1.5

        funcs = [lambda x: x, lambda x: x - mu]

        result = integrate_product_normal(funcs, n_points=12, mu=mu, sigma=sigma)

        # E[X * (X-μ)] = E[X²] - μE[X] = (μ² + σ²) - μ² = σ²
        expected = sigma**2
        assert np.allclose(result, expected, rtol=1e-8)


class Test2DQuadrature:
    """Test 2D Gauss-Hermite quadrature."""

    def test_2d_nodes_weights(self):
        """Test 2D quadrature nodes and weights."""
        n_points = 5
        nodes_2d, weights_2d = gauss_hermite_2d(n_points)

        # Check dimensions
        assert nodes_2d.shape == (n_points**2, 2)
        assert weights_2d.shape == (n_points**2,)

        # Weights should sum to 1
        assert np.allclose(np.sum(weights_2d), 1.0, rtol=1e-10)

        # All weights positive
        assert np.all(weights_2d > 0)

    def test_2d_integration(self):
        """Test 2D integration of simple functions."""
        n_points = 7
        nodes_2d, weights_2d = gauss_hermite_2d(n_points)

        # Integrate constant: ∫∫ c * φ(x) * φ(y) dx dy = c
        c = 2.5
        values = np.full(len(nodes_2d), c)
        result = np.sum(weights_2d * values)
        assert np.allclose(result, c, rtol=1e-10)

        # Integrate x + y (should be 0 for standard bivariate normal)
        values = nodes_2d[:, 0] * np.sqrt(2) + nodes_2d[:, 1] * np.sqrt(2)
        result = np.sum(weights_2d * values)
        assert np.allclose(result, 0, atol=1e-10)


class TestQuadratureClass:
    """Test GaussHermiteQuadrature class interface."""

    def test_class_initialization(self):
        """Test class initialization."""
        quad = GaussHermiteQuadrature(n_points=10)

        assert quad.n_points == 10
        assert len(quad.nodes) == 10
        assert len(quad.weights) == 10

    def test_class_integration(self):
        """Test integration using class interface."""
        quad = GaussHermiteQuadrature(n_points=12)

        # Integrate x²
        mu, sigma = 1.0, 2.0
        result = quad.integrate(lambda x: x**2, mu=mu, sigma=sigma)
        expected = mu**2 + sigma**2
        assert np.allclose(result, expected, rtol=1e-10)

    def test_class_vectorized_integration(self):
        """Test vectorized integration."""
        quad = GaussHermiteQuadrature(n_points=10)

        # Vectorized function
        def f_vec(x):
            return x**2 + 2 * x + 1

        mu, sigma = 0.5, 1.5
        result = quad.integrate_vectorized(f_vec, mu=mu, sigma=sigma)

        # E[X² + 2X + 1] = (μ² + σ²) + 2μ + 1
        expected = (mu**2 + sigma**2) + 2 * mu + 1
        assert np.allclose(result, expected, rtol=1e-10)

    def test_class_caching(self):
        """Test that nodes/weights are cached."""
        quad = GaussHermiteQuadrature(n_points=8, cache=True)

        # Access nodes twice
        nodes1 = quad.nodes
        nodes2 = quad.nodes

        # Should be same object (cached)
        assert nodes1 is nodes2

        # Same for weights
        weights1 = quad.weights
        weights2 = quad.weights
        assert weights1 is weights2


class TestQuadratureApplications:
    """Test quadrature in panel model contexts."""

    def test_random_effects_integration(self):
        """Test integration pattern used in Random Effects models."""
        # Simulate RE Probit likelihood contribution
        n_time = 5
        quadrature_points = 12

        # Parameters
        beta = np.array([0.5, -0.3])
        sigma_alpha = 0.8

        # Data for one entity
        X_i = np.random.randn(n_time, 2)
        y_i = np.random.binomial(1, 0.5, n_time)

        # Quadrature
        nodes, weights = gauss_hermite_quadrature(quadrature_points)

        # Integrate likelihood contribution
        integral = 0.0
        for node, weight in zip(nodes, weights):
            alpha = np.sqrt(2) * sigma_alpha * node

            # Product of probabilities over time
            prob_product = 1.0
            for t in range(n_time):
                linear_pred = X_i[t] @ beta + alpha
                if y_i[t] == 1:
                    prob_product *= stats.norm.cdf(linear_pred)
                else:
                    prob_product *= 1 - stats.norm.cdf(linear_pred)

            integral += weight * prob_product

        # Should be between 0 and 1 (it's a likelihood)
        assert 0 <= integral <= 1

    def test_marginal_probability_computation(self):
        """Test computing marginal probabilities with RE."""
        # P(Y=1|X) = ∫ Φ(X'β + α) φ(α/σ) dα

        beta = np.array([0.3])
        sigma_alpha = 1.2
        x = np.array([1.5])

        quad = GaussHermiteQuadrature(n_points=20)

        # Compute marginal probability
        def integrand(alpha):
            return stats.norm.cdf(x @ beta + alpha)

        marginal_prob = quad.integrate(integrand, mu=0, sigma=sigma_alpha)

        # Should be valid probability
        assert 0 <= marginal_prob <= 1

        # Compare with crude Monte Carlo
        np.random.seed(42)
        mc_samples = np.random.normal(0, sigma_alpha, 10000)
        mc_probs = stats.norm.cdf(x @ beta + mc_samples)
        mc_estimate = np.mean(mc_probs)

        # Quadrature should be more accurate than crude MC
        assert np.abs(marginal_prob - mc_estimate) < 0.01
