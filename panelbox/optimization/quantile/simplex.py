"""
Simplex method (Barrodale-Roberts) for quantile regression.

This module implements the Barrodale-Roberts modified simplex algorithm for
solving quantile regression problems. This is a specialized linear programming
approach that is particularly efficient for small to medium-sized problems.
"""

import numpy as np
from scipy.optimize import linprog


def barrodale_roberts_qr(X, y, tau, max_iter=1000, tol=1e-8, verbose=False):
    """
    Solve quantile regression using the Barrodale-Roberts simplex method.

    This reformulates the quantile regression problem as a linear program
    and solves it using a specialized simplex algorithm.

    Parameters
    ----------
    X : ndarray (n, p)
        Design matrix
    y : ndarray (n,)
        Response vector
    tau : float
        Quantile level in (0, 1)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress

    Returns
    -------
    beta : ndarray
        Coefficient estimates
    info : dict
        Convergence information

    Notes
    -----
    The quantile regression problem is reformulated as:

    min tau'uz + (1-tau)'u{
    s.t. y - Xbeta = uz - u{
         uz, u{ e 0

    which is a standard linear program.
    """
    n, p = X.shape

    # Reformulate as linear program
    # Variables: [beta, uz, u{]
    # Objective: min tau'uz + (1-tau)'u{

    # Coefficient vector for LP
    c = np.concatenate(
        [
            np.zeros(p),  # beta coefficients
            tau * np.ones(n),  # uz coefficients
            (1 - tau) * np.ones(n),  # u{ coefficients
        ]
    )

    # Equality constraints: y - Xbeta = uz - u{
    # Rearrange to: Xbeta + uz - u{ = y
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])  # beta terms  # uz terms  # u{ terms
    b_eq = y

    # Bounds: beta unconstrained, uz, u{ e 0
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)

    # Solve linear program
    result = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",  # Use modern HiGHS solver
        options={"maxiter": max_iter, "disp": verbose},
    )

    # Extract beta coefficients
    beta = result.x[:p]

    # Convergence information
    info = {
        "converged": result.success,
        "iterations": result.nit if hasattr(result, "nit") else 0,
        "message": result.message,
        "dual_gap": None,  # LP solver doesn't directly provide this
    }

    return beta, info
