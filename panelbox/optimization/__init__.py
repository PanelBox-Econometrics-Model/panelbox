"""
Optimization utilities for panel econometric models.

This package provides numerical optimization tools including:
- Numerical gradients and Hessians
- Multiple starting values
- Convergence diagnostics
- Constrained optimization
"""

from panelbox.optimization.numerical_grad import approx_gradient, approx_hessian

__all__ = [
    "approx_gradient",
    "approx_hessian",
]
