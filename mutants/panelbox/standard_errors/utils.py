"""
Utility functions for covariance matrix estimation.

This module provides common functions for computing sandwich covariance
matrices and their components (bread and meat).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)
from typing import Annotated, Callable, ClassVar

MutantDict = Annotated[dict[str, Callable], "Mutant"]  # type: ignore


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg=None):  # type: ignore
    """Forward call to original or mutated function, depending on the environment."""
    import os  # type: ignore

    mutant_under_test = os.environ["MUTANT_UNDER_TEST"]  # type: ignore
    if mutant_under_test == "fail":  # type: ignore
        from mutmut.__main__ import MutmutProgrammaticFailException  # type: ignore

        raise MutmutProgrammaticFailException("Failed programmatically")  # type: ignore
    elif mutant_under_test == "stats":  # type: ignore
        from mutmut.__main__ import record_trampoline_hit  # type: ignore

        record_trampoline_hit(orig.__module__ + "." + orig.__name__)  # type: ignore
        # (for class methods, orig is bound and thus does not need the explicit self argument)
        result = orig(*call_args, **call_kwargs)  # type: ignore
        return result  # type: ignore
    prefix = orig.__module__ + "." + orig.__name__ + "__mutmut_"  # type: ignore
    if not mutant_under_test.startswith(prefix):  # type: ignore
        result = orig(*call_args, **call_kwargs)  # type: ignore
        return result  # type: ignore
    mutant_name = mutant_under_test.rpartition(".")[-1]  # type: ignore
    if self_arg is not None:  # type: ignore
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs)  # type: ignore
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs)  # type: ignore
    return result  # type: ignore


def compute_leverage(X: np.ndarray) -> np.ndarray:
    args = [X]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_compute_leverage__mutmut_orig, x_compute_leverage__mutmut_mutants, args, kwargs, None
    )


def x_compute_leverage__mutmut_orig(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 0, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_1(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = None

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 0, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_2(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = None

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 0, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_3(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(None)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 0, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_4(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = None

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 0, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_5(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum(None, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 0, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_6(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=None)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 0, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_7(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum(axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 0, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_8(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum(
        (X @ XTX_inv) * X,
    )

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 0, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_9(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) / X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 0, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_10(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=2)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 0, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_11(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = None

    return np.asarray(leverage)


def x_compute_leverage__mutmut_12(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(None, 0, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_13(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, None, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_14(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 0, None)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_15(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(0, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_16(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_17(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(
        leverage,
        0,
    )

    return np.asarray(leverage)


def x_compute_leverage__mutmut_18(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 1, 1)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_19(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 0, 2)

    return np.asarray(leverage)


def x_compute_leverage__mutmut_20(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.

    The leverage h_i is the diagonal element of the hat matrix:
    H = X(X'X)^{-1}X'

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    leverage : np.ndarray
        Leverage values (n,)

    Notes
    -----
    Leverage values satisfy:
    - 0 <= h_i <= 1
    - sum(h_i) = k (number of parameters)
    - Average leverage = k/n

    High leverage points (h_i > 2k/n or 3k/n) may be influential.
    """
    _n, _k = X.shape

    # Compute hat values
    # h_i = X_i (X'X)^{-1} X_i'
    XTX_inv = np.linalg.inv(X.T @ X)

    # Efficient computation: diag(X @ XTX_inv @ X.T)
    leverage = np.sum((X @ XTX_inv) * X, axis=1)

    # Ensure leverage is between 0 and 1 (numerical stability)
    leverage = np.clip(leverage, 0, 1)

    return np.asarray(None)


x_compute_leverage__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_compute_leverage__mutmut_1": x_compute_leverage__mutmut_1,
    "x_compute_leverage__mutmut_2": x_compute_leverage__mutmut_2,
    "x_compute_leverage__mutmut_3": x_compute_leverage__mutmut_3,
    "x_compute_leverage__mutmut_4": x_compute_leverage__mutmut_4,
    "x_compute_leverage__mutmut_5": x_compute_leverage__mutmut_5,
    "x_compute_leverage__mutmut_6": x_compute_leverage__mutmut_6,
    "x_compute_leverage__mutmut_7": x_compute_leverage__mutmut_7,
    "x_compute_leverage__mutmut_8": x_compute_leverage__mutmut_8,
    "x_compute_leverage__mutmut_9": x_compute_leverage__mutmut_9,
    "x_compute_leverage__mutmut_10": x_compute_leverage__mutmut_10,
    "x_compute_leverage__mutmut_11": x_compute_leverage__mutmut_11,
    "x_compute_leverage__mutmut_12": x_compute_leverage__mutmut_12,
    "x_compute_leverage__mutmut_13": x_compute_leverage__mutmut_13,
    "x_compute_leverage__mutmut_14": x_compute_leverage__mutmut_14,
    "x_compute_leverage__mutmut_15": x_compute_leverage__mutmut_15,
    "x_compute_leverage__mutmut_16": x_compute_leverage__mutmut_16,
    "x_compute_leverage__mutmut_17": x_compute_leverage__mutmut_17,
    "x_compute_leverage__mutmut_18": x_compute_leverage__mutmut_18,
    "x_compute_leverage__mutmut_19": x_compute_leverage__mutmut_19,
    "x_compute_leverage__mutmut_20": x_compute_leverage__mutmut_20,
}
x_compute_leverage__mutmut_orig.__name__ = "x_compute_leverage"


def compute_bread(X: np.ndarray) -> np.ndarray:
    args = [X]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_compute_bread__mutmut_orig, x_compute_bread__mutmut_mutants, args, kwargs, None
    )


def x_compute_bread__mutmut_orig(X: np.ndarray) -> np.ndarray:
    """
    Compute the "bread" of the sandwich covariance estimator.

    Bread = (X'X)^{-1}

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    bread : np.ndarray
        Bread matrix (k x k)

    Notes
    -----
    The sandwich covariance estimator is:
    V = Bread @ Meat @ Bread

    where Meat depends on the specific robust estimator (HC, cluster, etc.)
    """
    XTX = X.T @ X
    bread = np.linalg.inv(XTX)
    return bread


def x_compute_bread__mutmut_1(X: np.ndarray) -> np.ndarray:
    """
    Compute the "bread" of the sandwich covariance estimator.

    Bread = (X'X)^{-1}

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    bread : np.ndarray
        Bread matrix (k x k)

    Notes
    -----
    The sandwich covariance estimator is:
    V = Bread @ Meat @ Bread

    where Meat depends on the specific robust estimator (HC, cluster, etc.)
    """
    XTX = None
    bread = np.linalg.inv(XTX)
    return bread


def x_compute_bread__mutmut_2(X: np.ndarray) -> np.ndarray:
    """
    Compute the "bread" of the sandwich covariance estimator.

    Bread = (X'X)^{-1}

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    bread : np.ndarray
        Bread matrix (k x k)

    Notes
    -----
    The sandwich covariance estimator is:
    V = Bread @ Meat @ Bread

    where Meat depends on the specific robust estimator (HC, cluster, etc.)
    """
    X.T @ X
    bread = None
    return bread


def x_compute_bread__mutmut_3(X: np.ndarray) -> np.ndarray:
    """
    Compute the "bread" of the sandwich covariance estimator.

    Bread = (X'X)^{-1}

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)

    Returns
    -------
    bread : np.ndarray
        Bread matrix (k x k)

    Notes
    -----
    The sandwich covariance estimator is:
    V = Bread @ Meat @ Bread

    where Meat depends on the specific robust estimator (HC, cluster, etc.)
    """
    X.T @ X
    bread = np.linalg.inv(None)
    return bread


x_compute_bread__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_compute_bread__mutmut_1": x_compute_bread__mutmut_1,
    "x_compute_bread__mutmut_2": x_compute_bread__mutmut_2,
    "x_compute_bread__mutmut_3": x_compute_bread__mutmut_3,
}
x_compute_bread__mutmut_orig.__name__ = "x_compute_bread"


def compute_meat_hc(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    args = [X, resid, method, leverage]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_compute_meat_hc__mutmut_orig, x_compute_meat_hc__mutmut_mutants, args, kwargs, None
    )


def x_compute_meat_hc__mutmut_orig(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_1(
    X: np.ndarray, resid: np.ndarray, method: str = "XXHC1XX", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_2(
    X: np.ndarray, resid: np.ndarray, method: str = "hc1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_3(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = None

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_4(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method != "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_5(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "XXHC0XX":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_6(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "hc0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_7(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = None

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_8(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid * 2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_9(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**3

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_10(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method != "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_11(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "XXHC1XX":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_12(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "hc1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_13(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    _n, _k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = None

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_14(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) / (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_15(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n * (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_16(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n + k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_17(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid * 2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_18(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**3)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_19(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method != "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_20(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "XXHC2XX":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_21(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "hc2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_22(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is not None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_23(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = None
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_24(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(None)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_25(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = None

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_26(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) * (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_27(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid * 2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_28(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**3) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_29(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 + leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_30(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (2 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_31(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method != "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_32(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "XXHC3XX":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_33(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "hc3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_34(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is not None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_35(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = None
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_36(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(None)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_37(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = None

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_38(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) * ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_39(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid * 2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_40(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**3) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_41(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) * 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_42(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 + leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_43(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((2 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_44(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 3)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_45(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(None)

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_46(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = None
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_47(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X / np.sqrt(weights)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_48(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(None)[:, np.newaxis]
    meat = X_weighted.T @ X_weighted

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_49(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X * np.sqrt(weights)[:, np.newaxis]
    meat = None

    return np.asarray(meat)


def x_compute_meat_hc__mutmut_50(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1", leverage: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the "meat" of the sandwich for heteroskedasticity-robust SEs.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        Type of HC adjustment:
        - 'HC0': White (1980)
        - 'HC1': Degrees of freedom correction
        - 'HC2': Leverage adjustment
        - 'HC3': MacKinnon-White (1985)
    leverage : np.ndarray, optional
        Pre-computed leverage values (for efficiency)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent
        covariance matrix estimators with improved finite sample properties.
        Journal of Econometrics, 29(3), 305-325.
    """
    n, k = X.shape

    # Compute adjustment factors
    if method == "HC0":
        # No adjustment
        weights = resid**2

    elif method == "HC1":
        # Degrees of freedom correction
        weights = (n / (n - k)) * (resid**2)

    elif method == "HC2":
        # Leverage adjustment: ε²/(1-h)
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / (1 - leverage)

    elif method == "HC3":
        # Leverage adjustment: ε²/(1-h)²
        if leverage is None:
            leverage = compute_leverage(X)
        weights = (resid**2) / ((1 - leverage) ** 2)

    else:
        raise ValueError(f"Unknown HC method: {method}")

    # Compute meat: X'ΩX where Ω = diag(weights)
    # Efficient computation: X.T @ diag(weights) @ X
    X_weighted = X * np.sqrt(weights)[:, np.newaxis]
    X_weighted.T @ X_weighted

    return np.asarray(None)


x_compute_meat_hc__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_compute_meat_hc__mutmut_1": x_compute_meat_hc__mutmut_1,
    "x_compute_meat_hc__mutmut_2": x_compute_meat_hc__mutmut_2,
    "x_compute_meat_hc__mutmut_3": x_compute_meat_hc__mutmut_3,
    "x_compute_meat_hc__mutmut_4": x_compute_meat_hc__mutmut_4,
    "x_compute_meat_hc__mutmut_5": x_compute_meat_hc__mutmut_5,
    "x_compute_meat_hc__mutmut_6": x_compute_meat_hc__mutmut_6,
    "x_compute_meat_hc__mutmut_7": x_compute_meat_hc__mutmut_7,
    "x_compute_meat_hc__mutmut_8": x_compute_meat_hc__mutmut_8,
    "x_compute_meat_hc__mutmut_9": x_compute_meat_hc__mutmut_9,
    "x_compute_meat_hc__mutmut_10": x_compute_meat_hc__mutmut_10,
    "x_compute_meat_hc__mutmut_11": x_compute_meat_hc__mutmut_11,
    "x_compute_meat_hc__mutmut_12": x_compute_meat_hc__mutmut_12,
    "x_compute_meat_hc__mutmut_13": x_compute_meat_hc__mutmut_13,
    "x_compute_meat_hc__mutmut_14": x_compute_meat_hc__mutmut_14,
    "x_compute_meat_hc__mutmut_15": x_compute_meat_hc__mutmut_15,
    "x_compute_meat_hc__mutmut_16": x_compute_meat_hc__mutmut_16,
    "x_compute_meat_hc__mutmut_17": x_compute_meat_hc__mutmut_17,
    "x_compute_meat_hc__mutmut_18": x_compute_meat_hc__mutmut_18,
    "x_compute_meat_hc__mutmut_19": x_compute_meat_hc__mutmut_19,
    "x_compute_meat_hc__mutmut_20": x_compute_meat_hc__mutmut_20,
    "x_compute_meat_hc__mutmut_21": x_compute_meat_hc__mutmut_21,
    "x_compute_meat_hc__mutmut_22": x_compute_meat_hc__mutmut_22,
    "x_compute_meat_hc__mutmut_23": x_compute_meat_hc__mutmut_23,
    "x_compute_meat_hc__mutmut_24": x_compute_meat_hc__mutmut_24,
    "x_compute_meat_hc__mutmut_25": x_compute_meat_hc__mutmut_25,
    "x_compute_meat_hc__mutmut_26": x_compute_meat_hc__mutmut_26,
    "x_compute_meat_hc__mutmut_27": x_compute_meat_hc__mutmut_27,
    "x_compute_meat_hc__mutmut_28": x_compute_meat_hc__mutmut_28,
    "x_compute_meat_hc__mutmut_29": x_compute_meat_hc__mutmut_29,
    "x_compute_meat_hc__mutmut_30": x_compute_meat_hc__mutmut_30,
    "x_compute_meat_hc__mutmut_31": x_compute_meat_hc__mutmut_31,
    "x_compute_meat_hc__mutmut_32": x_compute_meat_hc__mutmut_32,
    "x_compute_meat_hc__mutmut_33": x_compute_meat_hc__mutmut_33,
    "x_compute_meat_hc__mutmut_34": x_compute_meat_hc__mutmut_34,
    "x_compute_meat_hc__mutmut_35": x_compute_meat_hc__mutmut_35,
    "x_compute_meat_hc__mutmut_36": x_compute_meat_hc__mutmut_36,
    "x_compute_meat_hc__mutmut_37": x_compute_meat_hc__mutmut_37,
    "x_compute_meat_hc__mutmut_38": x_compute_meat_hc__mutmut_38,
    "x_compute_meat_hc__mutmut_39": x_compute_meat_hc__mutmut_39,
    "x_compute_meat_hc__mutmut_40": x_compute_meat_hc__mutmut_40,
    "x_compute_meat_hc__mutmut_41": x_compute_meat_hc__mutmut_41,
    "x_compute_meat_hc__mutmut_42": x_compute_meat_hc__mutmut_42,
    "x_compute_meat_hc__mutmut_43": x_compute_meat_hc__mutmut_43,
    "x_compute_meat_hc__mutmut_44": x_compute_meat_hc__mutmut_44,
    "x_compute_meat_hc__mutmut_45": x_compute_meat_hc__mutmut_45,
    "x_compute_meat_hc__mutmut_46": x_compute_meat_hc__mutmut_46,
    "x_compute_meat_hc__mutmut_47": x_compute_meat_hc__mutmut_47,
    "x_compute_meat_hc__mutmut_48": x_compute_meat_hc__mutmut_48,
    "x_compute_meat_hc__mutmut_49": x_compute_meat_hc__mutmut_49,
    "x_compute_meat_hc__mutmut_50": x_compute_meat_hc__mutmut_50,
}
x_compute_meat_hc__mutmut_orig.__name__ = "x_compute_meat_hc"


def sandwich_covariance(bread: np.ndarray, meat: np.ndarray) -> np.ndarray:
    args = [bread, meat]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_sandwich_covariance__mutmut_orig,
        x_sandwich_covariance__mutmut_mutants,
        args,
        kwargs,
        None,
    )


def x_sandwich_covariance__mutmut_orig(bread: np.ndarray, meat: np.ndarray) -> np.ndarray:
    """
    Compute sandwich covariance matrix.

    V = Bread @ Meat @ Bread

    Parameters
    ----------
    bread : np.ndarray
        Bread matrix (k x k)
    meat : np.ndarray
        Meat matrix (k x k)

    Returns
    -------
    cov : np.ndarray
        Covariance matrix (k x k)
    """
    return np.asarray(bread @ meat @ bread)


def x_sandwich_covariance__mutmut_1(bread: np.ndarray, meat: np.ndarray) -> np.ndarray:
    """
    Compute sandwich covariance matrix.

    V = Bread @ Meat @ Bread

    Parameters
    ----------
    bread : np.ndarray
        Bread matrix (k x k)
    meat : np.ndarray
        Meat matrix (k x k)

    Returns
    -------
    cov : np.ndarray
        Covariance matrix (k x k)
    """
    return np.asarray(None)


x_sandwich_covariance__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_sandwich_covariance__mutmut_1": x_sandwich_covariance__mutmut_1
}
x_sandwich_covariance__mutmut_orig.__name__ = "x_sandwich_covariance"


def compute_clustered_meat(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    args = [X, resid, clusters, df_correction]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_compute_clustered_meat__mutmut_orig,
        x_compute_clustered_meat__mutmut_mutants,
        args,
        kwargs,
        None,
    )


def x_compute_clustered_meat__mutmut_orig(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_1(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = False
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_2(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = None
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_3(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = None
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_4(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(None)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_5(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = None

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_6(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = None

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_7(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros(None)

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_8(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for _cluster_id in unique_clusters:
        cluster_mask = None
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_9(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters != cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_10(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = None
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_11(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = None

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_12(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X[cluster_mask]
        resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = None
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_13(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat = np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_14(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat -= np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_15(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(None, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_16(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, None)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_17(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_18(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(
            score_c,
        )

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_19(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction or n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_20(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters >= 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_21(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 2:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_22(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    _n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = None
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_23(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) / ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_24(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters * (n_clusters - 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_25(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters + 1)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_26(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 2)) * ((n - 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_27(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) * (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_28(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n + 1) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_29(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 2) / (n - k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_30(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n + k))
        meat *= correction

    return meat


def x_compute_clustered_meat__mutmut_31(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat = correction

    return meat


def x_compute_clustered_meat__mutmut_32(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute meat matrix for cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction: G/(G-1) × (N-1)/(N-K)

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    Notes
    -----
    The cluster-robust meat is:
    Meat = Σ_g (X_g' ε_g)(ε_g' X_g)

    where g indexes clusters.

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Initialize meat
    meat = np.zeros((k, k))

    # Sum over clusters
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_c = X[cluster_mask]
        resid_c = resid[cluster_mask]

        # Compute outer product for this cluster
        # (X_c' ε_c)(ε_c' X_c) = (X_c' ε_c)(X_c' ε_c)'
        score_c = X_c.T @ resid_c
        meat += np.outer(score_c, score_c)

    # Apply finite-sample correction
    if df_correction and n_clusters > 1:
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        meat /= correction

    return meat


x_compute_clustered_meat__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_compute_clustered_meat__mutmut_1": x_compute_clustered_meat__mutmut_1,
    "x_compute_clustered_meat__mutmut_2": x_compute_clustered_meat__mutmut_2,
    "x_compute_clustered_meat__mutmut_3": x_compute_clustered_meat__mutmut_3,
    "x_compute_clustered_meat__mutmut_4": x_compute_clustered_meat__mutmut_4,
    "x_compute_clustered_meat__mutmut_5": x_compute_clustered_meat__mutmut_5,
    "x_compute_clustered_meat__mutmut_6": x_compute_clustered_meat__mutmut_6,
    "x_compute_clustered_meat__mutmut_7": x_compute_clustered_meat__mutmut_7,
    "x_compute_clustered_meat__mutmut_8": x_compute_clustered_meat__mutmut_8,
    "x_compute_clustered_meat__mutmut_9": x_compute_clustered_meat__mutmut_9,
    "x_compute_clustered_meat__mutmut_10": x_compute_clustered_meat__mutmut_10,
    "x_compute_clustered_meat__mutmut_11": x_compute_clustered_meat__mutmut_11,
    "x_compute_clustered_meat__mutmut_12": x_compute_clustered_meat__mutmut_12,
    "x_compute_clustered_meat__mutmut_13": x_compute_clustered_meat__mutmut_13,
    "x_compute_clustered_meat__mutmut_14": x_compute_clustered_meat__mutmut_14,
    "x_compute_clustered_meat__mutmut_15": x_compute_clustered_meat__mutmut_15,
    "x_compute_clustered_meat__mutmut_16": x_compute_clustered_meat__mutmut_16,
    "x_compute_clustered_meat__mutmut_17": x_compute_clustered_meat__mutmut_17,
    "x_compute_clustered_meat__mutmut_18": x_compute_clustered_meat__mutmut_18,
    "x_compute_clustered_meat__mutmut_19": x_compute_clustered_meat__mutmut_19,
    "x_compute_clustered_meat__mutmut_20": x_compute_clustered_meat__mutmut_20,
    "x_compute_clustered_meat__mutmut_21": x_compute_clustered_meat__mutmut_21,
    "x_compute_clustered_meat__mutmut_22": x_compute_clustered_meat__mutmut_22,
    "x_compute_clustered_meat__mutmut_23": x_compute_clustered_meat__mutmut_23,
    "x_compute_clustered_meat__mutmut_24": x_compute_clustered_meat__mutmut_24,
    "x_compute_clustered_meat__mutmut_25": x_compute_clustered_meat__mutmut_25,
    "x_compute_clustered_meat__mutmut_26": x_compute_clustered_meat__mutmut_26,
    "x_compute_clustered_meat__mutmut_27": x_compute_clustered_meat__mutmut_27,
    "x_compute_clustered_meat__mutmut_28": x_compute_clustered_meat__mutmut_28,
    "x_compute_clustered_meat__mutmut_29": x_compute_clustered_meat__mutmut_29,
    "x_compute_clustered_meat__mutmut_30": x_compute_clustered_meat__mutmut_30,
    "x_compute_clustered_meat__mutmut_31": x_compute_clustered_meat__mutmut_31,
    "x_compute_clustered_meat__mutmut_32": x_compute_clustered_meat__mutmut_32,
}
x_compute_clustered_meat__mutmut_orig.__name__ = "x_compute_clustered_meat"


def compute_twoway_clustered_meat(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    args = [X, resid, clusters1, clusters2, df_correction]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_compute_twoway_clustered_meat__mutmut_orig,
        x_compute_twoway_clustered_meat__mutmut_mutants,
        args,
        kwargs,
        None,
    )


def x_compute_twoway_clustered_meat__mutmut_orig(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_1(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = False,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_2(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = None
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_3(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(None, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_4(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, None, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_5(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, None, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_6(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, None)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_7(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_8(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_9(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_10(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(
        X,
        resid,
        clusters1,
    )
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_11(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = None

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_12(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(None, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_13(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, None, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_14(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, None, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_15(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, None)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_16(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_17(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_18(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_19(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(
        X,
        resid,
        clusters2,
    )

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_20(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = None
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_21(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array(None)
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_22(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(None, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_23(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, None)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_24(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_25(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array(
        [
            f"{c1}_{c2}"
            for c1, c2 in zip(
                clusters1,
            )
        ]
    )
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_26(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = None

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_27(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(None, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_28(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, None, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_29(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, None, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_30(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, None)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_31(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_32(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_33(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_34(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(
        X,
        resid,
        clusters_12,
    )

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_35(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    compute_clustered_meat(X, resid, clusters1, df_correction)
    compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = None

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_36(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 + meat2 + meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_37(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat = meat1 - meat2 - meat12

    return np.asarray(meat)


def x_compute_twoway_clustered_meat__mutmut_38(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute meat matrix for two-way cluster-robust standard errors.

    Uses the formula:
    V = V_1 + V_2 - V_12

    where V_1 is clustered by dimension 1, V_2 by dimension 2,
    and V_12 by the intersection.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    meat : np.ndarray
        Meat matrix (k x k)

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
        Robust inference with multiway clustering.
        Journal of Business & Economic Statistics, 29(2), 238-249.
    """
    # Compute meat for each clustering dimension
    meat1 = compute_clustered_meat(X, resid, clusters1, df_correction)
    meat2 = compute_clustered_meat(X, resid, clusters2, df_correction)

    # Create intersection clusters
    # Combine cluster IDs as tuples
    clusters_12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(clusters1, clusters2)])
    meat12 = compute_clustered_meat(X, resid, clusters_12, df_correction)

    # Two-way clustering: V_1 + V_2 - V_12
    meat1 + meat2 - meat12

    return np.asarray(None)


x_compute_twoway_clustered_meat__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_compute_twoway_clustered_meat__mutmut_1": x_compute_twoway_clustered_meat__mutmut_1,
    "x_compute_twoway_clustered_meat__mutmut_2": x_compute_twoway_clustered_meat__mutmut_2,
    "x_compute_twoway_clustered_meat__mutmut_3": x_compute_twoway_clustered_meat__mutmut_3,
    "x_compute_twoway_clustered_meat__mutmut_4": x_compute_twoway_clustered_meat__mutmut_4,
    "x_compute_twoway_clustered_meat__mutmut_5": x_compute_twoway_clustered_meat__mutmut_5,
    "x_compute_twoway_clustered_meat__mutmut_6": x_compute_twoway_clustered_meat__mutmut_6,
    "x_compute_twoway_clustered_meat__mutmut_7": x_compute_twoway_clustered_meat__mutmut_7,
    "x_compute_twoway_clustered_meat__mutmut_8": x_compute_twoway_clustered_meat__mutmut_8,
    "x_compute_twoway_clustered_meat__mutmut_9": x_compute_twoway_clustered_meat__mutmut_9,
    "x_compute_twoway_clustered_meat__mutmut_10": x_compute_twoway_clustered_meat__mutmut_10,
    "x_compute_twoway_clustered_meat__mutmut_11": x_compute_twoway_clustered_meat__mutmut_11,
    "x_compute_twoway_clustered_meat__mutmut_12": x_compute_twoway_clustered_meat__mutmut_12,
    "x_compute_twoway_clustered_meat__mutmut_13": x_compute_twoway_clustered_meat__mutmut_13,
    "x_compute_twoway_clustered_meat__mutmut_14": x_compute_twoway_clustered_meat__mutmut_14,
    "x_compute_twoway_clustered_meat__mutmut_15": x_compute_twoway_clustered_meat__mutmut_15,
    "x_compute_twoway_clustered_meat__mutmut_16": x_compute_twoway_clustered_meat__mutmut_16,
    "x_compute_twoway_clustered_meat__mutmut_17": x_compute_twoway_clustered_meat__mutmut_17,
    "x_compute_twoway_clustered_meat__mutmut_18": x_compute_twoway_clustered_meat__mutmut_18,
    "x_compute_twoway_clustered_meat__mutmut_19": x_compute_twoway_clustered_meat__mutmut_19,
    "x_compute_twoway_clustered_meat__mutmut_20": x_compute_twoway_clustered_meat__mutmut_20,
    "x_compute_twoway_clustered_meat__mutmut_21": x_compute_twoway_clustered_meat__mutmut_21,
    "x_compute_twoway_clustered_meat__mutmut_22": x_compute_twoway_clustered_meat__mutmut_22,
    "x_compute_twoway_clustered_meat__mutmut_23": x_compute_twoway_clustered_meat__mutmut_23,
    "x_compute_twoway_clustered_meat__mutmut_24": x_compute_twoway_clustered_meat__mutmut_24,
    "x_compute_twoway_clustered_meat__mutmut_25": x_compute_twoway_clustered_meat__mutmut_25,
    "x_compute_twoway_clustered_meat__mutmut_26": x_compute_twoway_clustered_meat__mutmut_26,
    "x_compute_twoway_clustered_meat__mutmut_27": x_compute_twoway_clustered_meat__mutmut_27,
    "x_compute_twoway_clustered_meat__mutmut_28": x_compute_twoway_clustered_meat__mutmut_28,
    "x_compute_twoway_clustered_meat__mutmut_29": x_compute_twoway_clustered_meat__mutmut_29,
    "x_compute_twoway_clustered_meat__mutmut_30": x_compute_twoway_clustered_meat__mutmut_30,
    "x_compute_twoway_clustered_meat__mutmut_31": x_compute_twoway_clustered_meat__mutmut_31,
    "x_compute_twoway_clustered_meat__mutmut_32": x_compute_twoway_clustered_meat__mutmut_32,
    "x_compute_twoway_clustered_meat__mutmut_33": x_compute_twoway_clustered_meat__mutmut_33,
    "x_compute_twoway_clustered_meat__mutmut_34": x_compute_twoway_clustered_meat__mutmut_34,
    "x_compute_twoway_clustered_meat__mutmut_35": x_compute_twoway_clustered_meat__mutmut_35,
    "x_compute_twoway_clustered_meat__mutmut_36": x_compute_twoway_clustered_meat__mutmut_36,
    "x_compute_twoway_clustered_meat__mutmut_37": x_compute_twoway_clustered_meat__mutmut_37,
    "x_compute_twoway_clustered_meat__mutmut_38": x_compute_twoway_clustered_meat__mutmut_38,
}
x_compute_twoway_clustered_meat__mutmut_orig.__name__ = "x_compute_twoway_clustered_meat"


def hc_covariance(X: np.ndarray, resid: np.ndarray, method: str = "HC1") -> np.ndarray:
    args = [X, resid, method]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_hc_covariance__mutmut_orig, x_hc_covariance__mutmut_mutants, args, kwargs, None
    )


def x_hc_covariance__mutmut_orig(
    X: np.ndarray, resid: np.ndarray, method: str = "HC1"
) -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_meat_hc(X, resid, method)
    return sandwich_covariance(bread, meat)


def x_hc_covariance__mutmut_1(
    X: np.ndarray, resid: np.ndarray, method: str = "XXHC1XX"
) -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_meat_hc(X, resid, method)
    return sandwich_covariance(bread, meat)


def x_hc_covariance__mutmut_2(X: np.ndarray, resid: np.ndarray, method: str = "hc1") -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_meat_hc(X, resid, method)
    return sandwich_covariance(bread, meat)


def x_hc_covariance__mutmut_3(X: np.ndarray, resid: np.ndarray, method: str = "HC1") -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    bread = None
    meat = compute_meat_hc(X, resid, method)
    return sandwich_covariance(bread, meat)


def x_hc_covariance__mutmut_4(X: np.ndarray, resid: np.ndarray, method: str = "HC1") -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    bread = compute_bread(None)
    meat = compute_meat_hc(X, resid, method)
    return sandwich_covariance(bread, meat)


def x_hc_covariance__mutmut_5(X: np.ndarray, resid: np.ndarray, method: str = "HC1") -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = None
    return sandwich_covariance(bread, meat)


def x_hc_covariance__mutmut_6(X: np.ndarray, resid: np.ndarray, method: str = "HC1") -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_meat_hc(None, resid, method)
    return sandwich_covariance(bread, meat)


def x_hc_covariance__mutmut_7(X: np.ndarray, resid: np.ndarray, method: str = "HC1") -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_meat_hc(X, None, method)
    return sandwich_covariance(bread, meat)


def x_hc_covariance__mutmut_8(X: np.ndarray, resid: np.ndarray, method: str = "HC1") -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_meat_hc(X, resid, None)
    return sandwich_covariance(bread, meat)


def x_hc_covariance__mutmut_9(X: np.ndarray, resid: np.ndarray, method: str = "HC1") -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_meat_hc(resid, method)
    return sandwich_covariance(bread, meat)


def x_hc_covariance__mutmut_10(X: np.ndarray, resid: np.ndarray, method: str = "HC1") -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_meat_hc(X, method)
    return sandwich_covariance(bread, meat)


def x_hc_covariance__mutmut_11(X: np.ndarray, resid: np.ndarray, method: str = "HC1") -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_meat_hc(
        X,
        resid,
    )
    return sandwich_covariance(bread, meat)


def x_hc_covariance__mutmut_12(X: np.ndarray, resid: np.ndarray, method: str = "HC1") -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    compute_bread(X)
    meat = compute_meat_hc(X, resid, method)
    return sandwich_covariance(None, meat)


def x_hc_covariance__mutmut_13(X: np.ndarray, resid: np.ndarray, method: str = "HC1") -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    compute_meat_hc(X, resid, method)
    return sandwich_covariance(bread, None)


def x_hc_covariance__mutmut_14(X: np.ndarray, resid: np.ndarray, method: str = "HC1") -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    compute_bread(X)
    meat = compute_meat_hc(X, resid, method)
    return sandwich_covariance(meat)


def x_hc_covariance__mutmut_15(X: np.ndarray, resid: np.ndarray, method: str = "HC1") -> np.ndarray:
    """
    Compute heteroskedasticity-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    method : str, default='HC1'
        HC method: 'HC0', 'HC1', 'HC2', or 'HC3'

    Returns
    -------
    cov : np.ndarray
        Robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    compute_meat_hc(X, resid, method)
    return sandwich_covariance(
        bread,
    )


x_hc_covariance__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_hc_covariance__mutmut_1": x_hc_covariance__mutmut_1,
    "x_hc_covariance__mutmut_2": x_hc_covariance__mutmut_2,
    "x_hc_covariance__mutmut_3": x_hc_covariance__mutmut_3,
    "x_hc_covariance__mutmut_4": x_hc_covariance__mutmut_4,
    "x_hc_covariance__mutmut_5": x_hc_covariance__mutmut_5,
    "x_hc_covariance__mutmut_6": x_hc_covariance__mutmut_6,
    "x_hc_covariance__mutmut_7": x_hc_covariance__mutmut_7,
    "x_hc_covariance__mutmut_8": x_hc_covariance__mutmut_8,
    "x_hc_covariance__mutmut_9": x_hc_covariance__mutmut_9,
    "x_hc_covariance__mutmut_10": x_hc_covariance__mutmut_10,
    "x_hc_covariance__mutmut_11": x_hc_covariance__mutmut_11,
    "x_hc_covariance__mutmut_12": x_hc_covariance__mutmut_12,
    "x_hc_covariance__mutmut_13": x_hc_covariance__mutmut_13,
    "x_hc_covariance__mutmut_14": x_hc_covariance__mutmut_14,
    "x_hc_covariance__mutmut_15": x_hc_covariance__mutmut_15,
}
x_hc_covariance__mutmut_orig.__name__ = "x_hc_covariance"


def clustered_covariance(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    args = [X, resid, clusters, df_correction]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_clustered_covariance__mutmut_orig,
        x_clustered_covariance__mutmut_mutants,
        args,
        kwargs,
        None,
    )


def x_clustered_covariance__mutmut_orig(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_clustered_meat(X, resid, clusters, df_correction)
    return sandwich_covariance(bread, meat)


def x_clustered_covariance__mutmut_1(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = False
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_clustered_meat(X, resid, clusters, df_correction)
    return sandwich_covariance(bread, meat)


def x_clustered_covariance__mutmut_2(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = None
    meat = compute_clustered_meat(X, resid, clusters, df_correction)
    return sandwich_covariance(bread, meat)


def x_clustered_covariance__mutmut_3(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(None)
    meat = compute_clustered_meat(X, resid, clusters, df_correction)
    return sandwich_covariance(bread, meat)


def x_clustered_covariance__mutmut_4(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = None
    return sandwich_covariance(bread, meat)


def x_clustered_covariance__mutmut_5(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_clustered_meat(None, resid, clusters, df_correction)
    return sandwich_covariance(bread, meat)


def x_clustered_covariance__mutmut_6(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_clustered_meat(X, None, clusters, df_correction)
    return sandwich_covariance(bread, meat)


def x_clustered_covariance__mutmut_7(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_clustered_meat(X, resid, None, df_correction)
    return sandwich_covariance(bread, meat)


def x_clustered_covariance__mutmut_8(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_clustered_meat(X, resid, clusters, None)
    return sandwich_covariance(bread, meat)


def x_clustered_covariance__mutmut_9(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_clustered_meat(resid, clusters, df_correction)
    return sandwich_covariance(bread, meat)


def x_clustered_covariance__mutmut_10(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_clustered_meat(X, clusters, df_correction)
    return sandwich_covariance(bread, meat)


def x_clustered_covariance__mutmut_11(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_clustered_meat(X, resid, df_correction)
    return sandwich_covariance(bread, meat)


def x_clustered_covariance__mutmut_12(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_clustered_meat(
        X,
        resid,
        clusters,
    )
    return sandwich_covariance(bread, meat)


def x_clustered_covariance__mutmut_13(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    compute_bread(X)
    meat = compute_clustered_meat(X, resid, clusters, df_correction)
    return sandwich_covariance(None, meat)


def x_clustered_covariance__mutmut_14(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    compute_clustered_meat(X, resid, clusters, df_correction)
    return sandwich_covariance(bread, None)


def x_clustered_covariance__mutmut_15(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    compute_bread(X)
    meat = compute_clustered_meat(X, resid, clusters, df_correction)
    return sandwich_covariance(meat)


def x_clustered_covariance__mutmut_16(
    X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, df_correction: bool = True
) -> np.ndarray:
    """
    Compute cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters : np.ndarray
        Cluster identifiers (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    compute_clustered_meat(X, resid, clusters, df_correction)
    return sandwich_covariance(
        bread,
    )


x_clustered_covariance__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_clustered_covariance__mutmut_1": x_clustered_covariance__mutmut_1,
    "x_clustered_covariance__mutmut_2": x_clustered_covariance__mutmut_2,
    "x_clustered_covariance__mutmut_3": x_clustered_covariance__mutmut_3,
    "x_clustered_covariance__mutmut_4": x_clustered_covariance__mutmut_4,
    "x_clustered_covariance__mutmut_5": x_clustered_covariance__mutmut_5,
    "x_clustered_covariance__mutmut_6": x_clustered_covariance__mutmut_6,
    "x_clustered_covariance__mutmut_7": x_clustered_covariance__mutmut_7,
    "x_clustered_covariance__mutmut_8": x_clustered_covariance__mutmut_8,
    "x_clustered_covariance__mutmut_9": x_clustered_covariance__mutmut_9,
    "x_clustered_covariance__mutmut_10": x_clustered_covariance__mutmut_10,
    "x_clustered_covariance__mutmut_11": x_clustered_covariance__mutmut_11,
    "x_clustered_covariance__mutmut_12": x_clustered_covariance__mutmut_12,
    "x_clustered_covariance__mutmut_13": x_clustered_covariance__mutmut_13,
    "x_clustered_covariance__mutmut_14": x_clustered_covariance__mutmut_14,
    "x_clustered_covariance__mutmut_15": x_clustered_covariance__mutmut_15,
    "x_clustered_covariance__mutmut_16": x_clustered_covariance__mutmut_16,
}
x_clustered_covariance__mutmut_orig.__name__ = "x_clustered_covariance"


def twoway_clustered_covariance(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    args = [X, resid, clusters1, clusters2, df_correction]  # type: ignore
    kwargs = {}  # type: ignore
    return _mutmut_trampoline(
        x_twoway_clustered_covariance__mutmut_orig,
        x_twoway_clustered_covariance__mutmut_mutants,
        args,
        kwargs,
        None,
    )


def x_twoway_clustered_covariance__mutmut_orig(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_twoway_clustered_meat(X, resid, clusters1, clusters2, df_correction)
    return sandwich_covariance(bread, meat)


def x_twoway_clustered_covariance__mutmut_1(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = False,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_twoway_clustered_meat(X, resid, clusters1, clusters2, df_correction)
    return sandwich_covariance(bread, meat)


def x_twoway_clustered_covariance__mutmut_2(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = None
    meat = compute_twoway_clustered_meat(X, resid, clusters1, clusters2, df_correction)
    return sandwich_covariance(bread, meat)


def x_twoway_clustered_covariance__mutmut_3(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(None)
    meat = compute_twoway_clustered_meat(X, resid, clusters1, clusters2, df_correction)
    return sandwich_covariance(bread, meat)


def x_twoway_clustered_covariance__mutmut_4(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = None
    return sandwich_covariance(bread, meat)


def x_twoway_clustered_covariance__mutmut_5(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_twoway_clustered_meat(None, resid, clusters1, clusters2, df_correction)
    return sandwich_covariance(bread, meat)


def x_twoway_clustered_covariance__mutmut_6(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_twoway_clustered_meat(X, None, clusters1, clusters2, df_correction)
    return sandwich_covariance(bread, meat)


def x_twoway_clustered_covariance__mutmut_7(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_twoway_clustered_meat(X, resid, None, clusters2, df_correction)
    return sandwich_covariance(bread, meat)


def x_twoway_clustered_covariance__mutmut_8(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_twoway_clustered_meat(X, resid, clusters1, None, df_correction)
    return sandwich_covariance(bread, meat)


def x_twoway_clustered_covariance__mutmut_9(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_twoway_clustered_meat(X, resid, clusters1, clusters2, None)
    return sandwich_covariance(bread, meat)


def x_twoway_clustered_covariance__mutmut_10(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_twoway_clustered_meat(resid, clusters1, clusters2, df_correction)
    return sandwich_covariance(bread, meat)


def x_twoway_clustered_covariance__mutmut_11(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_twoway_clustered_meat(X, clusters1, clusters2, df_correction)
    return sandwich_covariance(bread, meat)


def x_twoway_clustered_covariance__mutmut_12(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_twoway_clustered_meat(X, resid, clusters2, df_correction)
    return sandwich_covariance(bread, meat)


def x_twoway_clustered_covariance__mutmut_13(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_twoway_clustered_meat(X, resid, clusters1, df_correction)
    return sandwich_covariance(bread, meat)


def x_twoway_clustered_covariance__mutmut_14(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    meat = compute_twoway_clustered_meat(
        X,
        resid,
        clusters1,
        clusters2,
    )
    return sandwich_covariance(bread, meat)


def x_twoway_clustered_covariance__mutmut_15(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    compute_bread(X)
    meat = compute_twoway_clustered_meat(X, resid, clusters1, clusters2, df_correction)
    return sandwich_covariance(None, meat)


def x_twoway_clustered_covariance__mutmut_16(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    compute_twoway_clustered_meat(X, resid, clusters1, clusters2, df_correction)
    return sandwich_covariance(bread, None)


def x_twoway_clustered_covariance__mutmut_17(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    compute_bread(X)
    meat = compute_twoway_clustered_meat(X, resid, clusters1, clusters2, df_correction)
    return sandwich_covariance(meat)


def x_twoway_clustered_covariance__mutmut_18(
    X: np.ndarray,
    resid: np.ndarray,
    clusters1: np.ndarray,
    clusters2: np.ndarray,
    df_correction: bool = True,
) -> np.ndarray:
    """
    Compute two-way cluster-robust covariance matrix.

    Convenience function that combines bread and meat computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    clusters1 : np.ndarray
        First cluster dimension (n,)
    clusters2 : np.ndarray
        Second cluster dimension (n,)
    df_correction : bool, default=True
        Apply finite-sample correction

    Returns
    -------
    cov : np.ndarray
        Two-way cluster-robust covariance matrix (k x k)
    """
    bread = compute_bread(X)
    compute_twoway_clustered_meat(X, resid, clusters1, clusters2, df_correction)
    return sandwich_covariance(
        bread,
    )


x_twoway_clustered_covariance__mutmut_mutants: ClassVar[MutantDict] = {  # type: ignore
    "x_twoway_clustered_covariance__mutmut_1": x_twoway_clustered_covariance__mutmut_1,
    "x_twoway_clustered_covariance__mutmut_2": x_twoway_clustered_covariance__mutmut_2,
    "x_twoway_clustered_covariance__mutmut_3": x_twoway_clustered_covariance__mutmut_3,
    "x_twoway_clustered_covariance__mutmut_4": x_twoway_clustered_covariance__mutmut_4,
    "x_twoway_clustered_covariance__mutmut_5": x_twoway_clustered_covariance__mutmut_5,
    "x_twoway_clustered_covariance__mutmut_6": x_twoway_clustered_covariance__mutmut_6,
    "x_twoway_clustered_covariance__mutmut_7": x_twoway_clustered_covariance__mutmut_7,
    "x_twoway_clustered_covariance__mutmut_8": x_twoway_clustered_covariance__mutmut_8,
    "x_twoway_clustered_covariance__mutmut_9": x_twoway_clustered_covariance__mutmut_9,
    "x_twoway_clustered_covariance__mutmut_10": x_twoway_clustered_covariance__mutmut_10,
    "x_twoway_clustered_covariance__mutmut_11": x_twoway_clustered_covariance__mutmut_11,
    "x_twoway_clustered_covariance__mutmut_12": x_twoway_clustered_covariance__mutmut_12,
    "x_twoway_clustered_covariance__mutmut_13": x_twoway_clustered_covariance__mutmut_13,
    "x_twoway_clustered_covariance__mutmut_14": x_twoway_clustered_covariance__mutmut_14,
    "x_twoway_clustered_covariance__mutmut_15": x_twoway_clustered_covariance__mutmut_15,
    "x_twoway_clustered_covariance__mutmut_16": x_twoway_clustered_covariance__mutmut_16,
    "x_twoway_clustered_covariance__mutmut_17": x_twoway_clustered_covariance__mutmut_17,
    "x_twoway_clustered_covariance__mutmut_18": x_twoway_clustered_covariance__mutmut_18,
}
x_twoway_clustered_covariance__mutmut_orig.__name__ = "x_twoway_clustered_covariance"
