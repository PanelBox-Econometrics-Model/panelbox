"""
Quadrature methods for numerical integration.

This module implements Gauss-Hermite quadrature for integrating functions
over normal distributions, which is essential for Random Effects models.
"""

from typing import Callable, Optional, Tuple

import numpy as np
from scipy import special


def gauss_hermite_quadrature(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Gauss-Hermite quadrature nodes and weights.

    The quadrature approximates integrals of the form:
    ∫_{-∞}^{∞} f(x) exp(-x²) dx ≈ Σᵢ wᵢ f(xᵢ)

    For standard normal integration:
    ∫ f(x) φ(x) dx ≈ Σᵢ wᵢ f(√2 xᵢ)

    Parameters
    ----------
    n_points : int
        Number of quadrature points (2 to 50)

    Returns
    -------
    nodes : np.ndarray (n_points,)
        Quadrature nodes (roots of Hermite polynomial)
    weights : np.ndarray (n_points,)
        Quadrature weights (normalized for standard normal)

    Raises
    ------
    ValueError
        If n_points is not between 2 and 50

    Examples
    --------
    >>> nodes, weights = gauss_hermite_quadrature(5)
    >>> # Integrate x² over standard normal
    >>> integral = np.sum(weights * (np.sqrt(2) * nodes)**2)
    >>> np.allclose(integral, 1.0)  # E[X²] for N(0,1)
    True
    """
    if not 2 <= n_points <= 50:
        raise ValueError(f"n_points must be between 2 and 50, got {n_points}")

    # Get Hermite polynomial roots and weights
    nodes, weights = special.roots_hermite(n_points)

    # Normalize weights for standard normal integration
    # Original weights are for exp(-x²), we need exp(-x²/2)/√(2π)
    # The adjustment factor is 1/√π
    weights = weights / np.sqrt(np.pi)

    return nodes, weights


def integrate_normal(
    func: Callable[[float], float], n_points: int = 12, mu: float = 0.0, sigma: float = 1.0
) -> float:
    """
    Integrate a function over a normal distribution using Gauss-Hermite quadrature.

    Computes: ∫ func(x) φ(x; μ, σ²) dx

    Parameters
    ----------
    func : callable
        Function to integrate. Should accept scalar and return scalar.
    n_points : int, default=12
        Number of quadrature points
    mu : float, default=0.0
        Mean of normal distribution
    sigma : float, default=1.0
        Standard deviation of normal distribution

    Returns
    -------
    float
        Approximate integral value

    Examples
    --------
    >>> # E[X] for N(2, 3²)
    >>> integral = integrate_normal(lambda x: x, n_points=12, mu=2, sigma=3)
    >>> np.allclose(integral, 2.0)
    True
    """
    nodes, weights = gauss_hermite_quadrature(n_points)

    # Transform nodes from standard to N(μ, σ²)
    # For standard normal: x = μ + √2 σ ξ
    transformed_nodes = mu + np.sqrt(2) * sigma * nodes

    # Evaluate function at transformed nodes
    values = np.array([func(x) for x in transformed_nodes])

    # Weighted sum
    integral = np.sum(weights * values)

    return integral


def adaptive_gauss_hermite(
    func: Callable[[float], float],
    n_points_list: list = [8, 12, 16, 20],
    mu: float = 0.0,
    sigma: float = 1.0,
    tol: float = 1e-8,
) -> Tuple[float, int]:
    """
    Adaptive Gauss-Hermite quadrature with automatic selection of points.

    Increases the number of quadrature points until convergence.

    Parameters
    ----------
    func : callable
        Function to integrate
    n_points_list : list, default=[8, 12, 16, 20]
        List of quadrature points to try (in order)
    mu : float, default=0.0
        Mean of normal distribution
    sigma : float, default=1.0
        Standard deviation
    tol : float, default=1e-8
        Convergence tolerance

    Returns
    -------
    integral : float
        Converged integral value
    n_points_used : int
        Number of quadrature points used

    Examples
    --------
    >>> # Integrate exp(x) over N(0,1)
    >>> integral, n_pts = adaptive_gauss_hermite(np.exp)
    >>> np.allclose(integral, np.exp(0.5))  # E[exp(X)] = exp(μ + σ²/2)
    True
    """
    if len(n_points_list) < 2:
        raise ValueError("Need at least 2 different n_points for adaptation")

    prev_integral = None

    for n_points in n_points_list:
        integral = integrate_normal(func, n_points, mu, sigma)

        if prev_integral is not None:
            # Check convergence
            if np.abs(integral - prev_integral) < tol:
                return integral, n_points

        prev_integral = integral

    # If we get here, we haven't converged
    # Return the last computed value and warn
    import warnings

    warnings.warn(
        f"Adaptive quadrature did not converge to tolerance {tol}. "
        f"Final difference: {np.abs(integral - prev_integral):.2e}"
    )

    return integral, n_points_list[-1]


def integrate_product_normal(
    func_list: list, n_points: int = 12, mu: float = 0.0, sigma: float = 1.0
) -> float:
    """
    Integrate a product of functions over a normal distribution.

    Useful for panel data where we have Πₜ f(x, t).

    Parameters
    ----------
    func_list : list of callables
        List of functions, each taking x and returning scalar
    n_points : int, default=12
        Number of quadrature points
    mu : float, default=0.0
        Mean of normal distribution
    sigma : float, default=1.0
        Standard deviation

    Returns
    -------
    float
        Integral of product

    Examples
    --------
    >>> # Integrate x * x² over N(0,1)
    >>> funcs = [lambda x: x, lambda x: x**2]
    >>> integral = integrate_product_normal(funcs)
    >>> np.allclose(integral, 0.0)  # E[X³] for N(0,1)
    True
    """

    def product_func(x):
        result = 1.0
        for f in func_list:
            result *= f(x)
        return result

    return integrate_normal(product_func, n_points, mu, sigma)


def gauss_hermite_2d(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D Gauss-Hermite quadrature for bivariate normal integration.

    Parameters
    ----------
    n_points : int
        Number of points in each dimension (total points = n_points²)

    Returns
    -------
    nodes : np.ndarray (n_points², 2)
        2D quadrature nodes
    weights : np.ndarray (n_points²,)
        Quadrature weights

    Notes
    -----
    This uses a tensor product approach for 2D integration.
    """
    nodes_1d, weights_1d = gauss_hermite_quadrature(n_points)

    # Create 2D grid
    xx, yy = np.meshgrid(nodes_1d, nodes_1d)
    nodes_2d = np.column_stack([xx.ravel(), yy.ravel()])

    # Weights are products
    wx, wy = np.meshgrid(weights_1d, weights_1d)
    weights_2d = (wx * wy).ravel()

    return nodes_2d, weights_2d


class GaussHermiteQuadrature:
    """
    Class-based interface for Gauss-Hermite quadrature.

    This provides a convenient interface for repeated quadrature
    with the same settings.

    Parameters
    ----------
    n_points : int, default=12
        Number of quadrature points
    cache : bool, default=True
        Whether to cache nodes and weights

    Attributes
    ----------
    nodes : np.ndarray
        Quadrature nodes
    weights : np.ndarray
        Quadrature weights

    Examples
    --------
    >>> quad = GaussHermiteQuadrature(n_points=10)
    >>> # Integrate multiple functions
    >>> integral1 = quad.integrate(lambda x: x**2)
    >>> integral2 = quad.integrate(lambda x: np.exp(x), mu=1, sigma=2)
    """

    def __init__(self, n_points: int = 12, cache: bool = True):
        """Initialize quadrature."""
        self.n_points = n_points
        self.cache = cache
        self._nodes = None
        self._weights = None

        if cache:
            self._compute_nodes_weights()

    def _compute_nodes_weights(self):
        """Compute and cache nodes and weights."""
        self._nodes, self._weights = gauss_hermite_quadrature(self.n_points)

    @property
    def nodes(self):
        """Get quadrature nodes."""
        if self._nodes is None:
            self._compute_nodes_weights()
        return self._nodes

    @property
    def weights(self):
        """Get quadrature weights."""
        if self._weights is None:
            self._compute_nodes_weights()
        return self._weights

    def integrate(
        self, func: Callable[[float], float], mu: float = 0.0, sigma: float = 1.0
    ) -> float:
        """
        Integrate function over normal distribution.

        Parameters
        ----------
        func : callable
            Function to integrate
        mu : float, default=0.0
            Mean of normal distribution
        sigma : float, default=1.0
            Standard deviation

        Returns
        -------
        float
            Integral value
        """
        # Transform nodes
        transformed_nodes = mu + np.sqrt(2) * sigma * self.nodes

        # Evaluate function
        values = np.array([func(x) for x in transformed_nodes])

        # Weighted sum
        return np.sum(self.weights * values)

    def integrate_vectorized(
        self, func: Callable[[np.ndarray], np.ndarray], mu: float = 0.0, sigma: float = 1.0
    ) -> float:
        """
        Integrate using vectorized function evaluation.

        Parameters
        ----------
        func : callable
            Vectorized function accepting array and returning array
        mu : float, default=0.0
            Mean of normal distribution
        sigma : float, default=1.0
            Standard deviation

        Returns
        -------
        float
            Integral value
        """
        # Transform all nodes at once
        transformed_nodes = mu + np.sqrt(2) * sigma * self.nodes

        # Vectorized evaluation
        values = func(transformed_nodes)

        # Weighted sum
        return np.sum(self.weights * values)
