"""
Optimization algorithms for quantile regression.

This module contains implementations of efficient optimization methods
for solving quantile regression problems.
"""

from __future__ import annotations

from .interior_point import frisch_newton_qr
from .smooth_qr import smooth_check_gradient, smooth_check_loss, smooth_qr


def optimize_quantile(
    objective, gradient, X, y, tau, method="interior-point", n_params=None, **kwargs
):
    """
    Main optimization dispatcher for quantile regression.

    Parameters
    ----------
    objective : callable
        Objective function
    gradient : callable
        Gradient function
    X : array-like
        Design matrix
    y : array-like
        Response vector
    tau : float
        Quantile level
    method : str
        Optimization method
    n_params : int
        Number of parameters
    **kwargs
        Additional arguments

    Returns
    -------
    dict
        Optimization result
    """
    if method == "interior-point":
        beta, info = frisch_newton_qr(X, y, tau, **kwargs)
        return {
            "params": beta,
            "converged": info["converged"],
            "iterations": info.get("iterations", 0),
            "info": info,
        }
    elif method == "smooth":
        beta, result = smooth_qr(X, y, tau, **kwargs)
        return {
            "params": beta,
            "converged": result.success,
            "iterations": result.nit,
            "info": result,
        }
    else:
        raise ValueError(f"Unknown optimization method: {method}")


__all__ = [
    "frisch_newton_qr",
    "optimize_quantile",
    "smooth_check_gradient",
    "smooth_check_loss",
    "smooth_qr",
]
