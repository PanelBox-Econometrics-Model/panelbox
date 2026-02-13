"""
Numerical differentiation for gradients and Hessians.

This module provides robust numerical approximations of derivatives
using finite differences. These are used when analytical derivatives
are not available or for validation purposes.

The module implements:
- Forward and central difference gradients
- Second-order Hessian approximations
- Automatic step size selection
- Symmetrization of Hessian matrices

References
----------
.. [1] Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.).
       Springer.
.. [2] Dennis, J. E., & Schnabel, R. B. (1996). Numerical Methods for
       Unconstrained Optimization and Nonlinear Equations. SIAM.
"""

from typing import Callable, Literal, Union

import numpy as np


def approx_gradient(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    method: Literal["central", "forward"] = "central",
    epsilon: Union[float, str] = "auto",
) -> np.ndarray:
    """
    Approximate gradient using finite differences.

    Parameters
    ----------
    func : callable
        Scalar-valued function f: R^n -> R
    x : np.ndarray
        Point at which to evaluate gradient (shape: (n,))
    method : {'central', 'forward'}, default='central'
        Finite difference method:
        - 'central': (f(x+h) - f(x-h))/(2h) - more accurate, O(h^2)
        - 'forward': (f(x+h) - f(x))/h - less accurate, O(h)
    epsilon : float or 'auto', default='auto'
        Step size. If 'auto', uses sqrt(machine_eps) * max(1, |x|)

    Returns
    -------
    np.ndarray
        Gradient vector (shape: (n,))

    Notes
    -----
    **Central Difference:**

    .. math::
        \\frac{\\partial f}{\\partial x_i} \\approx \\frac{f(x + h e_i) - f(x - h e_i)}{2h}

    where e_i is the i-th unit vector.

    **Forward Difference:**

    .. math::
        \\frac{\\partial f}{\\partial x_i} \\approx \\frac{f(x + h e_i) - f(x)}{h}

    **Automatic Step Size:**

    When epsilon='auto', the step size is chosen as:

    .. math::
        h_i = \\sqrt{\\epsilon_{\\text{mach}}} \\times \\max(1, |x_i|)

    where epsilon_mach is machine precision (np.finfo(float).eps).

    This balances truncation error (which decreases with h) and
    rounding error (which increases as h -> 0).

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.optimization.numerical_grad import approx_gradient
    >>>
    >>> # Simple quadratic function f(x) = x'Ax
    >>> A = np.array([[2, 1], [1, 3]])
    >>> f = lambda x: x @ A @ x
    >>> x = np.array([1.0, 2.0])
    >>>
    >>> # Numerical gradient
    >>> grad_num = approx_gradient(f, x, method='central')
    >>>
    >>> # Analytical gradient: 2*A*x
    >>> grad_true = 2 * A @ x
    >>>
    >>> # Compare
    >>> np.allclose(grad_num, grad_true, atol=1e-5)
    True

    >>> # Exponential function
    >>> f_exp = lambda x: np.exp(x.sum())
    >>> x = np.array([1.0, 2.0])
    >>> grad_num = approx_gradient(f_exp, x)
    >>> grad_true = np.exp(x.sum()) * np.ones_like(x)
    >>> np.allclose(grad_num, grad_true, atol=1e-6)
    True

    See Also
    --------
    approx_hessian : Approximate Hessian matrix
    """
    x = np.atleast_1d(x).astype(float)
    n = len(x)
    grad = np.zeros(n)

    # Determine step size
    if epsilon == "auto":
        # Use sqrt of machine epsilon scaled by magnitude of x
        eps_mach = np.finfo(float).eps
        h = np.sqrt(eps_mach) * np.maximum(1.0, np.abs(x))
    else:
        h = np.full(n, float(epsilon))

    # Compute gradient using finite differences
    if method == "central":
        # Central difference: O(h^2) accuracy
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h[i]
            x_minus[i] -= h[i]

            f_plus = func(x_plus)
            f_minus = func(x_minus)

            grad[i] = (f_plus - f_minus) / (2 * h[i])

    elif method == "forward":
        # Forward difference: O(h) accuracy
        f_x = func(x)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += h[i]
            f_plus = func(x_plus)
            grad[i] = (f_plus - f_x) / h[i]

    else:
        raise ValueError(f"method must be 'central' or 'forward', got '{method}'")

    return grad


def approx_hessian(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    method: Literal["central", "forward"] = "central",
    epsilon: Union[float, str] = "auto",
) -> np.ndarray:
    """
    Approximate Hessian matrix using finite differences.

    Parameters
    ----------
    func : callable
        Scalar-valued function f: R^n -> R
    x : np.ndarray
        Point at which to evaluate Hessian (shape: (n,))
    method : {'central', 'forward'}, default='central'
        Finite difference method (central is more accurate)
    epsilon : float or 'auto', default='auto'
        Step size. If 'auto', uses eps^(1/3) * max(1, |x|)
        for second derivatives

    Returns
    -------
    np.ndarray
        Symmetric Hessian matrix (shape: (n, n))

    Notes
    -----
    **Central Difference (Second Order):**

    For diagonal elements:

    .. math::
        \\frac{\\partial^2 f}{\\partial x_i^2} \\approx
        \\frac{f(x + 2h e_i) - 2f(x) + f(x - 2h e_i)}{4h^2}

    For off-diagonal elements:

    .. math::
        \\frac{\\partial^2 f}{\\partial x_i \\partial x_j} \\approx
        \\frac{f(x+h_i e_i+h_j e_j) - f(x+h_i e_i-h_j e_j) -
               f(x-h_i e_i+h_j e_j) + f(x-h_i e_i-h_j e_j)}{4 h_i h_j}

    **Automatic Step Size:**

    For second derivatives, the optimal step size scales as eps^(1/3)
    rather than eps^(1/2):

    .. math::
        h_i = \\epsilon_{\\text{mach}}^{1/3} \\times \\max(1, |x_i|)

    **Symmetrization:**

    The Hessian is forced to be symmetric by averaging:

    .. math::
        H_{\\text{sym}} = \\frac{H + H^T}{2}

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.optimization.numerical_grad import approx_hessian
    >>>
    >>> # Quadratic function f(x) = x'Ax
    >>> A = np.array([[2, 1], [1, 3]])
    >>> f = lambda x: x @ A @ x
    >>> x = np.array([1.0, 2.0])
    >>>
    >>> # Numerical Hessian
    >>> H_num = approx_hessian(f, x)
    >>>
    >>> # Analytical Hessian: 2*A (for quadratic form)
    >>> H_true = 2 * A
    >>>
    >>> # Compare
    >>> np.allclose(H_num, H_true, atol=1e-4)
    True

    >>> # Check symmetry
    >>> np.allclose(H_num, H_num.T)
    True

    See Also
    --------
    approx_gradient : Approximate gradient vector
    """
    x = np.atleast_1d(x).astype(float)
    n = len(x)
    H = np.zeros((n, n))

    # Determine step size (cube root for second derivatives)
    if epsilon == "auto":
        eps_mach = np.finfo(float).eps
        # For second derivatives, optimal h ~ eps^(1/3)
        h = (eps_mach ** (1 / 3)) * np.maximum(1.0, np.abs(x))
    else:
        h = np.full(n, float(epsilon))

    if method == "central":
        # Diagonal elements: ∂²f/∂x_i²
        f_x = func(x)
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h[i]
            x_minus[i] -= h[i]

            f_plus = func(x_plus)
            f_minus = func(x_minus)

            H[i, i] = (f_plus - 2 * f_x + f_minus) / (h[i] ** 2)

        # Off-diagonal elements: ∂²f/∂x_i∂x_j
        for i in range(n):
            for j in range(i + 1, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()

                x_pp[i] += h[i]
                x_pp[j] += h[j]

                x_pm[i] += h[i]
                x_pm[j] -= h[j]

                x_mp[i] -= h[i]
                x_mp[j] += h[j]

                x_mm[i] -= h[i]
                x_mm[j] -= h[j]

                f_pp = func(x_pp)
                f_pm = func(x_pm)
                f_mp = func(x_mp)
                f_mm = func(x_mm)

                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h[i] * h[j])
                H[j, i] = H[i, j]  # Symmetry

    elif method == "forward":
        # Forward difference for Hessian (less accurate, not recommended)
        # Use gradient-based approach: H ≈ (∇f(x+h) - ∇f(x)) / h
        grad_x = approx_gradient(func, x, method="forward", epsilon=epsilon)

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += h[i]
            grad_plus = approx_gradient(func, x_plus, method="forward", epsilon=epsilon)
            H[:, i] = (grad_plus - grad_x) / h[i]

        # Symmetrize
        H = (H + H.T) / 2

    else:
        raise ValueError(f"method must be 'central' or 'forward', got '{method}'")

    # Force exact symmetry
    H = (H + H.T) / 2

    return H
