"""
Smooth approximation methods for quantile regression.

This module implements smoothed quantile regression methods that replace the
non-differentiable check loss with a smooth approximation, enabling the use
of standard gradient-based optimization methods.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def smooth_check_loss(u, tau, epsilon=1e-6):
    """
    Smooth approximation of check loss using Huber-type smoothing.

    For small epsilon, approximates the check loss but is differentiable.

    Parameters
    ----------
    u : ndarray
        Residuals
    tau : float
        Quantile level
    epsilon : float
        Smoothing parameter (smaller = closer to check loss)

    Returns
    -------
    loss : float
        Smoothed check loss
    """
    # Smooth approximation
    loss = np.where(
        np.abs(u) <= epsilon,
        u**2 / (2 * epsilon) * (tau - (u < 0)),
        np.abs(u) * (tau - (u < 0)) - epsilon * (tau - (u < 0)) ** 2 / 2,
    )

    return np.sum(loss)


def smooth_check_gradient(u, tau, epsilon=1e-6):
    """
    Gradient of smoothed check loss.

    Parameters
    ----------
    u : ndarray
        Residuals
    tau : float
        Quantile level
    epsilon : float
        Smoothing parameter

    Returns
    -------
    grad : ndarray
        Gradient values
    """
    grad = np.where(
        np.abs(u) <= epsilon, u / epsilon * (tau - (u < 0)), tau - (u < 0).astype(float)
    )

    return grad


def smooth_qr(X, y, tau, epsilon=1e-6, method="L-BFGS-B", **kwargs):
    """
    Quantile regression via smooth approximation.

    Uses standard optimization methods on smoothed objective.

    Parameters
    ----------
    X : ndarray (n, p)
        Design matrix
    y : ndarray (n,)
        Response vector
    tau : float
        Quantile level
    epsilon : float
        Smoothing parameter
    method : str
        Optimization method for scipy.optimize.minimize
    **kwargs
        Additional arguments to optimizer

    Returns
    -------
    beta : ndarray
        Estimated coefficients
    result : OptimizeResult
        Full optimization result
    """
    _n, _p = X.shape

    def objective(beta):
        """Compute the smoothed quantile regression objective."""
        u = y - X @ beta
        return smooth_check_loss(u, tau, epsilon)

    def gradient(beta):
        """Compute the gradient of the smoothed objective."""
        u = y - X @ beta
        grad_u = smooth_check_gradient(u, tau, epsilon)
        return -X.T @ grad_u

    # Initial value from OLS
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0]

    # Optimize
    result = minimize(fun=objective, x0=beta0, method=method, jac=gradient, **kwargs)

    return result.x, result
